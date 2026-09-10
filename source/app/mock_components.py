"""Mock components for sandbox-verifiable end-to-end demos (CONTRACT.md).

Every mock satisfies the real component interface (duck-typed), so
:class:`app.pipeline.VoicePipeline` accepts them directly. The mocks are
deterministic and dependency-light (numpy only); timing is simulated with
tiny sleeps so async scheduling, streaming and the barge-in path are
exercised realistically:

* :class:`MockAsrEngine` — preset transcript queue; ``flush()`` pops one.
* :class:`MockLlmClient` — scripted replies, streamed word by word; records
  the messages of every ``chat_stream`` call (assert memory integration).
* :class:`MockTtsEngine` — 0.1 s of silence per chunk; records
  ``(text, language, length_scale)`` tuples; ``stop()`` aborts the
  in-flight synthesis.
* :class:`MockVoiceMemBridge` — scripted memory contexts per turn; records
  ``commit_reply``/``store_fact`` calls and ``cancel_pending`` count.
* :class:`MockVad` — ``prob()`` from a scripted probability timeline.
* :class:`MockEmotionAnalyzer` — scripted prosody results per turn (M2);
  records the analyzed audio sizes; supports a failure mode to exercise
  graceful degradation.
* :class:`MockSpeakerRecognizer` — scripted speaker identifications per
  turn (M3); records the analyzed audio sizes and supports a failure mode
  plus an "unknown speaker" mode (None ids) to exercise the memory-routing
  fallback.
* :class:`MockSpeaker` — EXTRA (not in the CONTRACT mock list): a speaker
  stand-in whose ``play_async`` takes the real audio duration, so barge-in
  demos and tests get a truthful playback timeline. ``stop()`` interrupts.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Optional

import numpy as np

from app.voicemem_bridge import TurnContext
from app.emotion import EmotionResult
from app.speaker import SpeakerIdentification

#: Sample rate of the silence produced by MockTtsEngine / MockSpeaker.
MOCK_SAMPLE_RATE = 22050

#: Reply reused when a MockLlmClient is constructed without scripts.
_DEFAULT_REPLY = (
    "Ez egy hosszabb magyar mondat a mock oltozonytol. "
    "Koszonem szepen a kerdest, igy folytatom a valaszt. "
    "Ez a harmadik mondat mar tenyleg nem tartalmaz semmi meglepetest."
)


class MockAsrEngine:
    """Preset transcript queue: ``flush()`` pops the next scripted transcript.

    ``feed`` counts the samples it saw (the pipeline feeds the whole
    utterance). ``queue`` is a plain settable attribute (CONTRACT).
    """

    def __init__(self, queue: Optional[Sequence[str]] = None) -> None:
        self.queue: list[str] = list(queue) if queue else []
        self.fed_samples = 0

    def is_available(self) -> bool:
        return True

    def feed(self, samples: "np.ndarray") -> str:
        self.fed_samples += int(np.asarray(samples).size)
        return ""

    def flush(self) -> str:
        return self.queue.pop(0) if self.queue else ""

    def reset(self) -> None:
        self.fed_samples = 0

    @property
    def last_partial(self) -> str:
        return ""


class MockLlmClient:
    """Scripted streaming LLM: word-by-word deltas with tiny sleeps.

    ``calls`` records the messages of every streamed request (the system
    prompt with the memory context is ``calls[-1][0]`` — useful to assert
    that VoiceMem context actually reached the LLM).
    """

    def __init__(self, replies: Optional[Sequence[str]] = None, word_delay_s: float = 0.001) -> None:
        self.replies: list[str] = list(replies) if replies else [_DEFAULT_REPLY]
        self.word_delay_s = word_delay_s
        self.calls: list[list[dict]] = []
        self.stream_count = 0

    async def health_check(self) -> bool:
        return True

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        self.calls.append([dict(m) for m in messages])
        text = self.replies[self.stream_count % len(self.replies)]
        self.stream_count += 1
        for word in text.split(" "):
            await asyncio.sleep(self.word_delay_s)
            yield word + " "

    async def chat_json(
        self, messages: list[dict], temperature: Optional[float] = None
    ) -> dict:
        return {"ok": True}

    async def aclose(self) -> None:
        return None


class MockTtsEngine:
    """Deterministic TTS: 0.1 s of int16 silence per synthesized chunk.

    Records ``(text, language, length_scale, voice)`` in ``synthesized``;
    ``stop()`` aborts the in-flight synthesis (the next call resets the flag,
    mirroring the real subprocess wrapper). v0.4.2: accepts and records the
    resolved Piper voice id (4th positional arg) the same way the real
    :class:`app.tts.TtsEngine` does.
    """

    def __init__(self, sample_rate: int = MOCK_SAMPLE_RATE, delay_s: float = 0.002) -> None:
        self.sample_rate = sample_rate
        self.delay_s = delay_s
        self.synthesized: list[tuple[str, str, Optional[float], Optional[str]]] = []
        self.stop_count = 0
        self._stopped = False

    def is_available(self) -> bool:
        return True

    def synthesize(
        self,
        text: str,
        language: str,
        length_scale: Optional[float] = None,
        voice: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        self._stopped = False  # a new request supersedes a previous stop()
        if not text or not text.strip():
            return None
        time.sleep(self.delay_s)  # simulates the piper subprocess latency
        if self._stopped:
            return None
        self.synthesized.append((text, language, length_scale, voice))
        return np.zeros(int(self.sample_rate * 0.1), dtype=np.int16)

    def stop(self) -> None:
        self.stop_count += 1
        self._stopped = True

    def list_voices(self) -> list[str]:
        return ["hu_HU-anna-medium", "en_US-lessac-medium"]


class MockVoiceMemBridge:
    """Scripted memory contexts per turn; records commits and cancellations."""

    def __init__(self, contexts: Optional[Sequence[str]] = None) -> None:
        self.contexts: list[str] = list(contexts) if contexts else []
        self.committed: list[str] = []
        self.facts: list[str] = []
        #: M3 routing records: the speaker_id of every commit/fact (same
        #: order as ``committed``/``facts`` - contamination assertions).
        self.commit_speakers: list[str] = []
        self.fact_speakers: list[str] = []
        self.turn_speakers: list[str] = []
        self.cancel_count = 0
        self.turn_count = 0

    def is_available(self) -> bool:
        return True

    async def process_turn(
        self, transcript: str, speaker_id: str = "voice_user"
    ) -> TurnContext:
        context = self.contexts[self.turn_count] if self.turn_count < len(self.contexts) else ""
        self.turn_count += 1
        self.turn_speakers.append(speaker_id)
        return TurnContext(
            transcript=transcript, memory_context=context, speaker_id=speaker_id
        )

    async def commit_reply(self, reply_text: str, speaker_id: str = "voice_user") -> None:
        self.committed.append(reply_text)
        self.commit_speakers.append(speaker_id)

    async def store_fact(self, text: str, speaker_id: str = "voice_user") -> None:
        """M2 emotion-fact hook (M3: routed per speaker)."""
        self.facts.append(text)
        self.fact_speakers.append(speaker_id)

    def cancel_pending(self) -> None:
        self.cancel_count += 1


class MockVad:
    """``prob(frame)`` from a scripted probability timeline (cycled)."""

    def __init__(self, timeline: Optional[Sequence[float]] = None) -> None:
        self.timeline: list[float] = (
            list(timeline) if timeline else [0.05, 0.9, 0.1, 0.95, 0.02]
        )
        self.calls = 0

    def prob(self, frame: Any) -> float:
        value = self.timeline[self.calls % len(self.timeline)]
        self.calls += 1
        return float(value)

    def reset(self) -> None:
        pass  # stateless mock — the timeline keeps cycling


class MockEmotionAnalyzer:
    """Scripted M2 prosody analyzer: pops one :class:`EmotionResult` per turn.

    ``analyze(audio)`` records the audio size and returns the next scripted
    result (cycled when the queue runs out; ``None`` entries simulate the
    degraded analyzer). ``fail=True`` makes every call raise, exercising the
    pipeline's graceful-degradation path (modularity 8.6 item 5).
    """

    def __init__(
        self,
        results: Optional[Sequence[Optional[EmotionResult]]] = None,
        fail: bool = False,
        delay_s: float = 0.001,
    ) -> None:
        self.scripted: list[Optional[EmotionResult]] = (
            list(results) if results is not None else []
        )
        self.fail = fail
        self.delay_s = delay_s
        self.calls = 0
        self.analyzed_sizes: list[int] = []

    def is_available(self) -> bool:
        return not self.fail

    def analyze(self, audio: Any) -> Optional[EmotionResult]:
        self.calls += 1
        self.analyzed_sizes.append(int(np.asarray(audio).size) if audio is not None else 0)
        time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("mock emotion analyzer failure")
        if not self.scripted:
            return None
        index = (self.calls - 1) % len(self.scripted)
        return self.scripted[index]


class MockSpeakerRecognizer:
    """Scripted M3 speaker recognizer: pops one identification per turn.

    ``identify(audio)`` records the audio size and returns the next
    scripted :class:`SpeakerIdentification` (cycled when the queue runs
    out; entries with ``id=None`` simulate an UNKNOWN speaker - the
    pipeline falls back to the default user id). ``fail=True`` makes every
    call raise, exercising graceful degradation (modularity 9.5 items
    5/6). ``dead=True`` returns ``None`` (analyzer unavailable).
    """

    def __init__(
        self,
        scripted: Optional[Sequence[Optional[SpeakerIdentification]]] = None,
        fail: bool = False,
        dead: bool = False,
        delay_s: float = 0.001,
    ) -> None:
        self.scripted: list[Optional[SpeakerIdentification]] = (
            list(scripted) if scripted is not None else []
        )
        self.fail = fail
        self.dead = dead
        self.delay_s = delay_s
        self.calls = 0
        self.analyzed_sizes: list[int] = []

    def is_available(self) -> bool:
        return not self.fail and not self.dead

    def warm_up(self) -> bool:
        return self.is_available()

    def identify(self, audio: Any) -> Optional[SpeakerIdentification]:
        self.calls += 1
        self.analyzed_sizes.append(int(np.asarray(audio).size) if audio is not None else 0)
        time.sleep(self.delay_s)
        if self.dead:
            return None
        if self.fail:
            raise RuntimeError("mock speaker recognizer failure")
        if not self.scripted:
            return None
        index = (self.calls - 1) % len(self.scripted)
        return self.scripted[index]

    def register(self, speaker_id: str, audio_chunks: Any) -> tuple[bool, float]:
        """Registration mock: always succeeds with the scripted seconds."""
        return True, 10.0


class MockSpeaker:
    """Timing-faithful speaker stand-in: playback takes the audio duration.

    ``play_async`` sleeps in 10 ms slices so ``stop()`` interrupts promptly;
    ``played`` records the sample counts fully played back.
    """

    def __init__(self, sample_rate: int = MOCK_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.played: list[int] = []
        self.stop_count = 0
        self._stopped = False
        self._playing = False

    @property
    def playing(self) -> bool:
        return self._playing

    def play(self, pcm: "np.ndarray", sample_rate: Optional[int] = None) -> None:
        self._stopped = False  # a new request supersedes a previous stop()
        rate = int(sample_rate if sample_rate is not None else self.sample_rate)
        data = np.asarray(pcm)
        n = int(data.size)
        if n == 0:
            return
        self._playing = True
        try:
            remaining = n / float(rate)
            while remaining > 0 and not self._stopped:
                time.sleep(min(0.01, remaining))
                remaining -= 0.01
        finally:
            self._playing = False
        if not self._stopped:
            self.played.append(n)

    async def play_async(self, pcm: "np.ndarray", sample_rate: Optional[int] = None) -> None:
        await asyncio.to_thread(self.play, pcm, sample_rate)

    def stop(self) -> None:
        self.stop_count += 1
        self._stopped = True
        self._playing = False
