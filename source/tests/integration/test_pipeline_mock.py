"""End-to-end pipeline tests over mock components (CONTRACT.md, Task 8).

Runs entirely in the development sandbox: numpy + stdlib only (the mocks
simulate timing with sleeps). Covers the golden paths of the M1 pipeline:

* full turn flow (ASR -> memory -> LLM -> chunked TTS -> commit)
* memory context actually reaching the LLM system prompt
* per-chunk language detection (HU/EN voice switching)
* barge-in: cancellation of TTS/LLM/memory + no commit of partial replies
* external cancel(), empty transcripts, TTS failures, LLM outages,
  audio_out=None (offline audit mode), timing semantics
* the CLI demos as subprocesses (mock demo, barge-in demo, --check, ...)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests/<kind>/x.py -> 3 levels up)

from app.config import AgentConfig
from app.emotion import EmotionMemory, EmotionResult
from app.llm import LlmUnavailableError
from app.mock_components import (
    MockAsrEngine,
    MockEmotionAnalyzer,
    MockLlmClient,
    MockSpeaker,
    MockSpeakerRecognizer,
    MockTtsEngine,
    MockVad,
    MockVoiceMemBridge,
)
from app.speaker import SpeakerIdentification
from app.pipeline import TurnResult, TurnTimings, VoicePipeline
from app.text_utils import LANG_EN, LANG_HU

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Environment variables that AgentConfig.apply_env() consumes — neutralized
#: so the outer environment cannot leak into these tests.
_ENV_KEYS = (
    "VOICEMEM_HOME", "VOICEMEM_MEMORY_ROOT", "VOICEMEM_EMBED_DIM",
    "VOICEMEM_LOG_LEVEL", "VOICEMEM_ENABLE_EMOTION", "VOICEMEM_ENABLE_SPEAKER",
    "PIPER_VOICES_PATH", "PIPER_EXECUTABLE", "TTS_HU_VOICE", "TTS_EN_VOICE",
    "SILERO_VAD_PATH", "QWEN3_ASR_MODEL_PATH", "LLAMA_MODEL_PATH",
    "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT", "LLAMA_CONTEXT_SIZE",
    "LLAMA_N_GPU_LAYERS", "LLAMA_CACHE_TYPE_K", "LLAMA_CACHE_TYPE_V",
    "OPENAI_BASE_URL", "OPENAI_MODEL",
)

EN_REPLY = (
    "This is a longer English sentence about coffee. "
    "Here comes another English sentence for you."
)
HU_REPLY = (
    "Ez egy hosszabb magyar mondat a kávékülönbségekről. "
    "Ez pedig egy második magyar mondat, ami elég hosszú."
)
MIXED_REPLY = EN_REPLY + " " + HU_REPLY
LONG_HU_REPLY = (
    "Röviden összefoglalom a különbséget. "
    "A flat white kisebb, kevesebb tejjel készül. "
    "Vékony mikrohabréteget használ, ami sima felületet ad. "
    "A tejeskávé nagyobb arányban tartalmaz forró tejet. "
    "Vastagabb habrétege krémesebb italt eredményez. "
    "Ezért tompítja az espresso keserűségét is. "
    "A latté tehát ennél még lágyabb ízű ital lesz."
)

_MEMORY_CONTEXT = (
    "- The user repeatedly says 'why is I have went' instead of "
    "'why have I gone wrong'."
)


def _make_config() -> AgentConfig:
    """Fresh config; env leakage is neutralized by the test classes."""
    return AgentConfig()


def _build_pipeline(
    transcripts: Optional[list[str]] = None,
    replies: Optional[list[str]] = None,
    contexts: Optional[list[str]] = None,
    audio_out: Any = None,
    llm: Optional[Any] = None,
    tts: Optional[Any] = None,
    voicemem: Optional[Any] = None,
    word_delay_s: float = 0.001,
    emotion: Any = None,
    emotion_memory: Any = None,
    speaker: Any = None,
    enable_emotion: bool = True,
    enable_speaker: bool = True,
) -> tuple[VoicePipeline, dict[str, Any]]:
    """Assemble a mock pipeline and return (pipeline, components-dict)."""
    cfg = _make_config()
    cfg.enable_emotion = enable_emotion
    cfg.enable_speaker = enable_speaker
    components: dict[str, Any] = {
        "asr": MockAsrEngine(transcripts if transcripts is not None else ["Szia!"]),
        "llm": llm or MockLlmClient(replies or [HU_REPLY], word_delay_s=word_delay_s),
        "tts": tts or MockTtsEngine(),
        "voicemem": voicemem or MockVoiceMemBridge(contexts or [""]),
        "vad": MockVad(),
        "audio_out": audio_out if audio_out is not None else MockSpeaker(),
        "emotion": emotion,
        "emotion_memory": emotion_memory,
        "speaker": speaker,
    }
    pipeline = VoicePipeline(
        cfg,
        asr=components["asr"],
        llm=components["llm"],
        tts=components["tts"],
        voicemem=components["voicemem"],
        vad=components["vad"],
        audio_out=components["audio_out"],
        emotion=components["emotion"],
        emotion_memory=components["emotion_memory"],
        speaker=components["speaker"],
    )
    # v0.4.2 voice selection: the constructor leniently loads the SHARED
    # config/voice_settings.json (a persisted UI selection could flip a
    # forced-language mode and break language assertions). Tests stay
    # hermetic with the default in-memory settings; the dedicated voice
    # tests below inject their own.
    pipeline._voice_settings = None
    return pipeline, components


def _audio(cfg: AgentConfig, seconds: float = 1.0) -> np.ndarray:
    """Silent utterance buffer (the pipeline does not listen to content)."""
    return np.zeros(int(cfg.sample_rate * seconds), dtype=np.float32)


def _high_after(start_after: int = 0) -> Callable[[], float]:
    """Scripted probe: sustained speech from the Nth call on (barge-in)."""
    state = {"calls": 0}

    def prob() -> float:
        state["calls"] += 1
        return 0.95 if state["calls"] > start_after else 0.02

    return prob


class _PipelineTestBase(unittest.IsolatedAsyncioTestCase):
    """Base: neutralize environment variables that leak into AgentConfig."""

    def setUp(self) -> None:
        env_patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        env_patcher.start()
        self.addCleanup(env_patcher.stop)


class PipelineFlowTests(_PipelineTestBase):
    """The golden path: ASR -> memory -> LLM -> chunked TTS -> commit."""

    async def test_full_mock_turn_flow(self) -> None:
        pipeline, parts = _build_pipeline(
            transcripts=["Mi a különbség a flat white és a tejeskávé között?"],
            replies=[LONG_HU_REPLY],
            contexts=[_MEMORY_CONTEXT],
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertIsInstance(result, TurnResult)
        self.assertEqual(
            result.transcript, "Mi a különbség a flat white és a tejeskávé között?"
        )
        self.assertEqual(result.reply.strip(), LONG_HU_REPLY.strip())
        self.assertEqual(result.language, LANG_HU)
        self.assertFalse(result.cancelled)
        self.assertFalse(result.barge_in)
        # LLM saw exactly one streamed request
        self.assertEqual(len(parts["llm"].calls), 1)
        # every sentence became a TTS chunk and was fully played
        self.assertGreaterEqual(len(parts["tts"].synthesized), 5)
        self.assertEqual(len(parts["tts"].synthesized), len(parts["audio_out"].played))
        # the complete reply was committed to long-term memory
        self.assertEqual(parts["voicemem"].committed, [result.reply])

    async def test_memory_context_reaches_llm_system_prompt(self) -> None:
        pipeline, parts = _build_pipeline(
            transcripts=["Why is I have went wrong?"],
            contexts=[_MEMORY_CONTEXT],
        )
        await pipeline.handle_utterance(_audio(pipeline._config))
        messages = parts["llm"].calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("why have I gone wrong", messages[0]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "Why is I have went wrong?"})

    async def test_per_chunk_language_detection(self) -> None:
        pipeline, parts = _build_pipeline(transcripts=["Mixed, please"], replies=[MIXED_REPLY])
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        languages = [lang for _text, lang, _scale, _voice in parts["tts"].synthesized]
        self.assertIn(LANG_EN, languages)
        self.assertIn(LANG_HU, languages)
        self.assertEqual(languages[0], LANG_EN)  # first sentence is English
        self.assertIn(result.language, (LANG_HU, LANG_EN))

    async def test_timings_chain_and_to_dict(self) -> None:
        pipeline, parts = _build_pipeline(transcripts=["timing check"])
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        t = result.timings
        self.assertIsInstance(t, TurnTimings)
        self.assertIsNotNone(t.speech_end_s)
        self.assertIsNotNone(t.asr_final_s)
        self.assertIsNotNone(t.memory_s)
        self.assertIsNotNone(t.first_token_s)
        self.assertIsNotNone(t.first_audio_s)
        self.assertIsNotNone(t.full_response_s)
        self.assertGreaterEqual(t.asr_final_s, t.speech_end_s)
        self.assertGreaterEqual(t.first_token_s, t.asr_final_s)
        self.assertGreaterEqual(t.first_audio_s, t.first_token_s)
        self.assertGreaterEqual(t.full_response_s, t.first_audio_s)
        self.assertGreaterEqual(t.memory_s, 0.0)
        self.assertIsNone(t.emotion_s)  # no analyzer injected -> M1 timings
        self.assertIsNone(t.speaker_s)  # no recognizer injected -> M1/M2 timings
        keys = set(t.to_dict().keys())
        self.assertEqual(
            keys,
            {
                "speech_end_s", "asr_final_s", "memory_s", "emotion_s",
                "speaker_s", "first_token_s", "first_audio_s", "full_response_s",
            },
        )
        self.assertIn("timings", result.to_dict())
        self.assertIn("emotion", result.to_dict())
        self.assertIsNone(result.to_dict()["emotion"])

    async def test_empty_transcript_skips_everything(self) -> None:
        pipeline, parts = _build_pipeline(transcripts=[])
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(result.transcript, "")
        self.assertEqual(result.reply, "")
        self.assertFalse(result.cancelled)
        self.assertEqual(parts["llm"].calls, [])  # chat_stream never iterated
        self.assertEqual(parts["tts"].synthesized, [])
        self.assertEqual(parts["voicemem"].turn_count, 0)
        self.assertEqual(parts["voicemem"].committed, [])
        self.assertIsNotNone(result.timings.full_response_s)
        self.assertIsNone(result.timings.first_token_s)


class PipelineBargeInTests(_PipelineTestBase):
    """Barge-in: interrupting the reply cancels TTS/LLM/memory, no commit."""

    async def test_barge_in_cancels_turn(self) -> None:
        pipeline, parts = _build_pipeline(
            transcripts=["Meséld el még egyszer."], replies=[LONG_HU_REPLY]
        )
        result = await pipeline.handle_utterance(
            _audio(pipeline._config), vad_prob_fn=_high_after()
        )
        self.assertTrue(result.barge_in)
        self.assertTrue(result.cancelled)
        # partial reply: something was spoken, the full script was not
        self.assertGreater(len(result.reply), 0)
        self.assertLess(len(result.reply), len(LONG_HU_REPLY))
        # cancellation reached every component
        self.assertGreaterEqual(parts["tts"].stop_count, 1)
        self.assertGreaterEqual(parts["audio_out"].stop_count, 1)
        self.assertGreaterEqual(parts["voicemem"].cancel_count, 1)
        # partial replies must NOT enter long-term memory
        self.assertEqual(parts["voicemem"].committed, [])
        # some chunks were synthesized before the interruption
        self.assertGreaterEqual(len(parts["tts"].synthesized), 1)

    async def test_barge_in_detector_resets_between_turns(self) -> None:
        pipeline, parts = _build_pipeline(
            transcripts=["első", "második"],
            replies=[LONG_HU_REPLY, LONG_HU_REPLY],
        )
        first = await pipeline.handle_utterance(
            _audio(pipeline._config), vad_prob_fn=_high_after()
        )
        self.assertTrue(first.barge_in)
        # the gate must not leak: a clean second turn completes normally
        second = await pipeline.handle_utterance(
            _audio(pipeline._config), vad_prob_fn=lambda: 0.0
        )
        self.assertFalse(second.barge_in)
        self.assertFalse(second.cancelled)
        self.assertEqual(second.reply.strip(), LONG_HU_REPLY.strip())
        self.assertEqual(parts["voicemem"].committed, [second.reply])
        self.assertFalse(pipeline.echo_gate.tts_playing)

    async def test_stale_cancel_does_not_poison_next_turn(self) -> None:
        # A cancel() landing during turn N must be fully consumed by it: the
        # next turn starts with a clean slate (handle_utterance resets the
        # cancellation flags at entry).
        pipeline, parts = _build_pipeline(
            transcripts=["első", "második"], replies=[LONG_HU_REPLY, LONG_HU_REPLY]
        )

        async def canceller() -> None:
            await asyncio.sleep(0.02)
            pipeline.cancel()

        task = asyncio.create_task(canceller())
        first = await pipeline.handle_utterance(_audio(pipeline._config))
        await asyncio.gather(task)
        self.assertTrue(first.cancelled)
        self.assertGreaterEqual(parts["tts"].stop_count, 1)
        self.assertGreaterEqual(parts["voicemem"].cancel_count, 1)

        second = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertFalse(second.cancelled)
        self.assertFalse(second.barge_in)
        self.assertEqual(second.reply.strip(), LONG_HU_REPLY.strip())
        self.assertEqual(parts["voicemem"].committed, [second.reply])

    async def test_external_cancel_mid_stream(self) -> None:
        pipeline, parts = _build_pipeline(
            transcripts=["cancel mid stream"], replies=[LONG_HU_REPLY], word_delay_s=0.005
        )

        async def canceller() -> None:
            await asyncio.sleep(0.02)
            pipeline.cancel()

        task = asyncio.create_task(canceller())
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        await asyncio.gather(task)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.barge_in)
        self.assertLess(len(result.reply), len(LONG_HU_REPLY))
        self.assertEqual(parts["voicemem"].committed, [])


class PipelineRobustnessTests(_PipelineTestBase):
    """Failure modes: TTS outages, LLM outages, audio_out=None."""

    async def test_tts_failure_skips_chunks_but_commits_reply(self) -> None:
        class _FailingTts(MockTtsEngine):
            def synthesize(
                self,
                text: str,
                language: str,
                length_scale: Optional[float] = None,
                voice: Optional[str] = None,
            ) -> Optional[np.ndarray]:
                self.stop_count += 0  # keep the attribute alive for assertions
                return None

        tts = _FailingTts()
        pipeline, parts = _build_pipeline(
            transcripts=["tts outage"], replies=[LONG_HU_REPLY], tts=tts
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(result.reply.strip(), LONG_HU_REPLY.strip())
        self.assertFalse(result.cancelled)
        self.assertEqual(parts["audio_out"].played, [])  # nothing was audible
        self.assertIsNone(result.timings.first_audio_s)  # no audio ever
        self.assertEqual(parts["voicemem"].committed, [result.reply])

    async def test_audio_out_none_completes(self) -> None:
        pipeline, parts = _build_pipeline(
            transcripts=["no speaker attached"], replies=[HU_REPLY], audio_out=None
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(result.reply.strip(), HU_REPLY.strip())
        self.assertFalse(result.cancelled)
        self.assertIsNotNone(result.timings.first_audio_s)  # stamped after synthesis
        self.assertEqual(parts["voicemem"].committed, [result.reply])
        self.assertFalse(pipeline.echo_gate.tts_playing)

    async def test_llm_unavailable_yields_empty_reply(self) -> None:
        class _UnavailableLlm(MockLlmClient):
            async def chat_stream(self, messages, temperature=None, max_tokens=None):
                yield ""  # async-generator marker; body raises below
                raise LlmUnavailableError("llama-server is down")

        llm = _UnavailableLlm()
        pipeline, parts = _build_pipeline(
            transcripts=["llm down"], llm=llm
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(result.reply, "")
        self.assertFalse(result.cancelled)  # an outage is not a barge-in
        self.assertIsNone(result.timings.first_token_s)
        self.assertEqual(parts["voicemem"].committed, [])  # nothing to commit
        self.assertIsNotNone(result.timings.full_response_s)

    async def test_echo_gate_closed_during_playback_reopened_after(self) -> None:
        pipeline, parts = _build_pipeline(transcripts=["gate check"], replies=[HU_REPLY])
        observed: list[bool] = []
        inner_tts = parts["tts"]

        class _GateProbeTts(MockTtsEngine):
            def synthesize(
                self,
                text: str,
                language: str,
                length_scale: Optional[float] = None,
                voice: Optional[str] = None,
            ) -> Optional[np.ndarray]:
                observed.append(pipeline.echo_gate.tts_playing)
                return inner_tts.synthesize(text, language, length_scale, voice)

        parts["tts"] = _GateProbeTts()
        pipeline._tts = parts["tts"]
        await pipeline.handle_utterance(_audio(pipeline._config))
        # synthesis happens BEFORE playback: gate open while synthesizing
        self.assertIn(False, observed)
        # after the turn the gate is always reopened
        self.assertFalse(pipeline.echo_gate.tts_playing)


class PipelineEmotionTests(_PipelineTestBase):
    """M2: prosody analysis parallel with memory, prompt fusion, TTS pacing."""

    @staticmethod
    def _frustrated_prosody() -> EmotionResult:
        return EmotionResult(
            raw_label="angry", label="frustrated",
            valence=-0.8, arousal=0.8, confidence=0.9,
        )

    @staticmethod
    def _neutral_prosody() -> EmotionResult:
        return EmotionResult(
            raw_label="neutral", label="neutral",
            valence=0.0, arousal=0.0, confidence=0.7,
        )

    def _memory_in_tmp(self) -> EmotionMemory:
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return EmotionMemory(Path(tmp.name) / "emotion_log.jsonl")

    async def test_emotion_block_reaches_system_prompt(self) -> None:
        analyzer = MockEmotionAnalyzer([self._frustrated_prosody()])
        pipeline, parts = _build_pipeline(
            transcripts=["I do not understand this"], emotion=analyzer
        )
        await pipeline.handle_utterance(_audio(pipeline._config))
        system_prompt = parts["llm"].calls[0][0]["content"]
        self.assertIn("frustrated", system_prompt)
        self.assertIn("valence=-0.68", system_prompt)
        self.assertIn("arousal=0.64", system_prompt)
        self.assertIn("Verify this assessment", system_prompt)

    async def test_frustrated_turn_slows_tts(self) -> None:
        analyzer = MockEmotionAnalyzer([self._frustrated_prosody()])
        pipeline, parts = _build_pipeline(
            transcripts=["I do not understand this"], emotion=analyzer
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        scales = [scale for _t, _l, scale, _v in parts["tts"].synthesized]
        self.assertTrue(scales)
        self.assertTrue(
            all(
                scale is not None and abs(scale - 1.1) < 1e-9 for scale in scales
            ),
            scales,
        )
        self.assertEqual(result.emotion["label"], "frustrated")

    async def test_neutral_turn_keeps_default_tts_rate(self) -> None:
        analyzer = MockEmotionAnalyzer([self._neutral_prosody()])
        pipeline, parts = _build_pipeline(
            transcripts=["ez egy sima mondat"], emotion=analyzer
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        scales = [scale for _t, _l, scale, _v in parts["tts"].synthesized]
        self.assertTrue(scales)
        self.assertTrue(all(scale is None for scale in scales), scales)
        self.assertEqual(result.emotion["label"], "neutral")

    async def test_disabled_flag_runs_m1_identically(self) -> None:
        analyzer = MockEmotionAnalyzer([self._frustrated_prosody()])
        pipeline, parts = _build_pipeline(
            transcripts=["I do not understand this"],
            emotion=analyzer,
            enable_emotion=False,
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(analyzer.calls, 0)  # analyzer never consulted
        system_prompt = parts["llm"].calls[0][0]["content"]
        self.assertNotIn("frustrated", system_prompt)
        self.assertNotIn("prosody", system_prompt)
        scales = [scale for _t, _l, scale, _v in parts["tts"].synthesized]
        self.assertTrue(all(scale is None for scale in scales))
        self.assertIsNone(result.emotion)
        self.assertIsNone(result.timings.emotion_s)

    async def test_failing_analyzer_degrades_gracefully(self) -> None:
        analyzer = MockEmotionAnalyzer(fail=True)
        pipeline, parts = _build_pipeline(
            transcripts=["I do not understand this"], emotion=analyzer
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(analyzer.calls, 1)
        self.assertIsNotNone(result.reply)  # the turn survived
        self.assertIsNone(result.emotion)
        self.assertIsNotNone(result.timings.emotion_s)  # the attempt was timed
        system_prompt = parts["llm"].calls[0][0]["content"]
        self.assertNotIn("frustrated", system_prompt)

    async def test_emotion_runs_parallel_with_memory(self) -> None:
        class _SlowBridge(MockVoiceMemBridge):
            async def process_turn(self, transcript, speaker_id="voice_user"):
                await asyncio.sleep(0.25)
                return await super().process_turn(transcript, speaker_id=speaker_id)

        analyzer = MockEmotionAnalyzer(
            [self._frustrated_prosody()], delay_s=0.3
        )
        bridge = _SlowBridge([""])
        pipeline, parts = _build_pipeline(
            transcripts=["parhuzamos futás teszt"],
            voicemem=bridge,
            emotion=analyzer,
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        # Parallel proof: the gap from ASR end to the first LLM token is ~max
        # (0.30 s) when memory and emotion overlapped; SERIAL execution would
        # need ~sum (0.55 s). Overlap = total - gap must be clearly positive.
        gap = result.timings.first_token_s - result.timings.asr_final_s
        total = result.timings.emotion_s + result.timings.memory_s
        self.assertGreater(result.timings.memory_s, 0.2)
        self.assertGreater(result.timings.emotion_s, 0.25)
        self.assertLess(gap, 0.5, "memory+emotion look serial (gap >= 0.5 s)")
        self.assertGreater(
            total - gap, 0.1, "memory and emotion did not overlap (serial run)"
        )

    async def test_strong_emotion_logged_and_stored_as_fact(self) -> None:
        analyzer = MockEmotionAnalyzer([self._frustrated_prosody()])
        memory = self._memory_in_tmp()
        pipeline, parts = _build_pipeline(
            transcripts=["I do not understand this"],
            emotion=analyzer,
            emotion_memory=memory,
        )
        await pipeline.handle_utterance(_audio(pipeline._config))
        lines = memory.path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["label"], "frustrated")
        self.assertTrue(entry["stored_as_fact"])
        self.assertEqual(len(parts["voicemem"].facts), 1)
        self.assertIn("frustrated", parts["voicemem"].facts[0])

    async def test_weak_emotion_logged_but_not_stored(self) -> None:
        weak = EmotionResult(
            raw_label="neutral", label="neutral",
            valence=-0.2, arousal=0.1, confidence=0.6,
        )
        analyzer = MockEmotionAnalyzer([weak])
        memory = self._memory_in_tmp()
        pipeline, parts = _build_pipeline(
            transcripts=["ejnye"], emotion=analyzer, emotion_memory=memory
        )
        await pipeline.handle_utterance(_audio(pipeline._config))
        entry = json.loads(
            memory.path.read_text(encoding="utf-8").strip().splitlines()[0]
        )
        self.assertFalse(entry["stored_as_fact"])
        self.assertEqual(parts["voicemem"].facts, [])

    async def test_cancelled_turn_persists_nothing(self) -> None:
        analyzer = MockEmotionAnalyzer([self._frustrated_prosody()])
        memory = self._memory_in_tmp()
        pipeline, parts = _build_pipeline(
            transcripts=["I do not understand this"],
            replies=[LONG_HU_REPLY],
            emotion=analyzer,
            emotion_memory=memory,
        )
        result = await pipeline.handle_utterance(
            _audio(pipeline._config), vad_prob_fn=_high_after(1)
        )
        self.assertTrue(result.cancelled)
        self.assertFalse(memory.path.exists())
        self.assertEqual(parts["voicemem"].facts, [])
        self.assertEqual(parts["voicemem"].committed, [])


class PipelineSpeakerTests(_PipelineTestBase):
    """M3: speaker identification routes the memory space (spec 9.3/9.5)."""

    def _recognizer(self, ids: list[Optional[str]]) -> MockSpeakerRecognizer:
        scripted = [
            SpeakerIdentification(
                id=spk,
                similarity=0.83 if spk else 0.27,
                best_id=spk or "thomas",
                registered=2,
                latency_ms=35.0,
            )
            for spk in ids
        ]
        return MockSpeakerRecognizer(scripted)

    async def test_identified_speaker_routes_memory_and_commit(self) -> None:
        """The turn's user_id follows the identified speaker (9.5.2)."""
        recognizer = self._recognizer(["thomas"])
        voicemem = MockVoiceMemBridge([""])
        pipeline, parts = _build_pipeline(
            transcripts=["Why is I have went wrong?"],
            voicemem=voicemem,
            speaker=recognizer,
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(result.speaker_id, "thomas")
        self.assertEqual(voicemem.turn_speakers, ["thomas"])
        self.assertEqual(voicemem.commit_speakers, ["thomas"])
        self.assertEqual(result.speaker["id"], "thomas")
        self.assertTrue(result.speaker["known"])
        self.assertIsNotNone(result.timings.speaker_s)

    async def test_unknown_speaker_falls_back_to_default_user(self) -> None:
        """Unknown voice -> 'voice_user' fallback, never a stall (9.5.6)."""
        recognizer = self._recognizer([None])
        voicemem = MockVoiceMemBridge([""])
        pipeline, _parts = _build_pipeline(
            transcripts=["hello"], voicemem=voicemem, speaker=recognizer
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(result.speaker_id, "voice_user")
        self.assertEqual(voicemem.turn_speakers, ["voice_user"])
        self.assertIsNone(result.speaker["id"])
        self.assertFalse(result.speaker["known"])
        self.assertEqual(result.speaker["best_id"], "thomas")  # best candidate kept

    async def test_no_cross_speaker_contamination(self) -> None:
        """thomas and anna alternate: every turn hits ITS OWN memory space."""
        recognizer = self._recognizer(["thomas", "anna", "thomas"])
        contexts = [
            "- thomas made the have-went mistake twice",
            "- anna is practising the present perfect",
            "- thomas asked about the flat white difference",
        ]
        voicemem = MockVoiceMemBridge(contexts)
        pipeline, parts = _build_pipeline(
            transcripts=["a", "b", "c"], voicemem=voicemem, speaker=recognizer
        )
        for i in range(3):
            result = await pipeline.handle_utterance(_audio(pipeline._config))
            expected = ["thomas", "anna", "thomas"][i]
            self.assertEqual(result.speaker_id, expected, f"turn {i + 1}")
            prompt = parts["llm"].calls[i][0]["content"]
            expected_context = contexts[i]
            self.assertIn(expected_context, prompt, f"turn {i + 1} context")
            other_context = contexts[(i + 1) % 3]
            self.assertNotIn(other_context, prompt, f"turn {i + 1} contamination")
        self.assertEqual(voicemem.commit_speakers, ["thomas", "anna", "thomas"])

    async def test_disabled_flag_runs_m1_identically(self) -> None:
        """enable_speaker=False + injected recognizer -> no M3 stage at all."""
        recognizer = self._recognizer(["thomas"])
        voicemem = MockVoiceMemBridge([""])
        pipeline, _parts = _build_pipeline(
            transcripts=["hello"], voicemem=voicemem, speaker=recognizer,
            enable_speaker=False,
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(result.speaker_id, "voice_user")
        self.assertIsNone(result.speaker)
        self.assertIsNone(result.timings.speaker_s)
        self.assertEqual(recognizer.calls, 0)  # recognizer never touched
        self.assertEqual(voicemem.turn_speakers, ["voice_user"])

    async def test_failing_recognizer_degrades_gracefully(self) -> None:
        """A raising recognizer must never kill the turn (9.5.5)."""
        recognizer = MockSpeakerRecognizer([], fail=True)
        voicemem = MockVoiceMemBridge([""])
        pipeline, _parts = _build_pipeline(
            transcripts=["hello"], voicemem=voicemem, speaker=recognizer
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(result.speaker_id, "voice_user")
        self.assertIsNone(result.speaker)
        self.assertTrue(result.reply)  # the reply still happened

    async def test_dead_recognizer_returns_none_not_unknown(self) -> None:
        """embed() -> None (analyzer dead) is NOT an 'unknown speaker' record."""
        recognizer = MockSpeakerRecognizer([], dead=True)
        voicemem = MockVoiceMemBridge([""])
        pipeline, _parts = _build_pipeline(
            transcripts=["hello"], voicemem=voicemem, speaker=recognizer
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertIsNone(result.speaker)  # no identification record at all
        self.assertEqual(result.speaker_id, "voice_user")

    async def test_speaker_parallel_with_emotion_start(self) -> None:
        """Speaker (slow mock) + emotion (slow mock) overlap: the gap from
        ASR end to the first LLM token is ~max(emotion, speaker+memory),
        not their sum (serial would need speaker + emotion + memory)."""
        analyzer = MockEmotionAnalyzer(
            [EmotionResult(raw_label="neutral", label="neutral", valence=0.0, arousal=0.0)],
            delay_s=0.30,
        )
        recognizer = MockSpeakerRecognizer(
            [SpeakerIdentification(id="thomas", similarity=0.9, best_id="thomas", registered=1)],
            delay_s=0.20,
        )
        voicemem = MockVoiceMemBridge([""])
        pipeline, _parts = _build_pipeline(
            transcripts=["hello"],
            voicemem=voicemem,
            emotion=analyzer,
            speaker=recognizer,
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        self.assertEqual(result.speaker_id, "thomas")
        self.assertIsNotNone(result.emotion)  # both M2 + M3 stages ran
        self.assertGreaterEqual(result.timings.speaker_s or 0.0, 0.2)
        self.assertGreaterEqual(result.timings.emotion_s or 0.0, 0.28)
        gap = result.timings.first_token_s - result.timings.asr_final_s
        serial_floor = (result.timings.speaker_s or 0.0) + result.timings.emotion_s
        # The speaker stage STARTED at the same time as the emotion stage;
        # the memory retrieval only waited for the (already finished)
        # speaker result, so the gap stays under the serial sum.
        self.assertLess(
            gap, serial_floor, "speaker+emotion look serial (gap >= sum)"
        )

    async def test_cancelled_turn_persists_no_speaker_memory(self) -> None:
        """Barge-in during the reply -> no commit into ANY speaker space."""
        recognizer = self._recognizer(["thomas"])
        voicemem = MockVoiceMemBridge([""])
        pipeline, _parts = _build_pipeline(
            transcripts=["Meseld el meg egyszer, mi a kulonbseg."],
            replies=[LONG_HU_REPLY],
            voicemem=voicemem,
            speaker=recognizer,
        )
        result = await pipeline.handle_utterance(
            _audio(pipeline._config), vad_prob_fn=_high_after(2)
        )
        self.assertTrue(result.cancelled)
        self.assertEqual(result.speaker_id, "thomas")  # routing still recorded
        self.assertEqual(voicemem.commit_speakers, [])  # but nothing persisted

    async def test_speaker_result_dict_shape(self) -> None:
        recognizer = self._recognizer(["thomas"])
        pipeline, _parts = _build_pipeline(
            transcripts=["hello"], speaker=recognizer
        )
        result = await pipeline.handle_utterance(_audio(pipeline._config))
        payload = result.to_dict()
        self.assertIn("speaker", payload)
        self.assertIn("speaker_id", payload)
        self.assertEqual(
            set(payload["speaker"].keys()),
            {"id", "known", "similarity", "best_id", "registered", "latency_ms"},
        )


class MainCliTests(unittest.TestCase):
    """CLI subprocess smoke tests (python -m app.main ...)."""

    _ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "app.main", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env=self._ENV,
        )

    def test_mock_demo_three_turns(self) -> None:
        proc = self._run("--mock")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-800:])
        self.assertIn('"transcript"', proc.stdout)
        self.assertIn("Do I still make the same mistake?", proc.stdout)
        self.assertIn("commit_reply", proc.stdout)

    def test_mock_custom_text(self) -> None:
        proc = self._run("--mock", "--text", "Egy egyedi próba forduló.")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-800:])
        self.assertIn("Egy egyedi próba forduló.", proc.stdout)

    def test_barge_in_demo_cancels(self) -> None:
        proc = self._run("--mock", "--barge-in-demo")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-800:])
        self.assertIn('"barge_in": true', proc.stdout)
        self.assertIn('"cancelled": true', proc.stdout)

    def test_emotion_demo_adapts_and_exits_zero(self) -> None:
        proc = self._run("--mock", "--emotion-demo")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-800:])
        self.assertIn('"emotion"', proc.stdout)
        self.assertIn('"label": "frustrated"', proc.stdout)
        self.assertIn("emotion-block a promptban: igen", proc.stdout)
        self.assertIn("lassított TTS (length_scale > 1.0): igen", proc.stdout)
        self.assertIn('"emotion_s"', proc.stdout)

    def test_emotion_demo_requires_mock(self) -> None:
        proc = self._run("--emotion-demo")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--emotion-demo", proc.stderr)

    def test_speaker_demo_routes_and_exits_zero(self) -> None:
        proc = self._run("--mock", "--speaker-demo")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-800:])
        self.assertIn("[M3 beszelő: thomas", proc.stdout)
        self.assertIn("[M3 beszelő: other_speaker", proc.stdout)
        self.assertIn("[M3 beszelő: ISMERETLEN", proc.stdout)
        self.assertIn("memória-útválasztás: HELYES", proc.stdout)
        self.assertIn("cross-speaker kontamináció: 0", proc.stdout)
        self.assertIn('"speaker_id": "thomas"', proc.stdout)
        self.assertIn('"speaker_id": "voice_user"', proc.stdout)
        self.assertIn('"speaker_s"', proc.stdout)

    def test_speaker_demo_requires_mock(self) -> None:
        proc = self._run("--speaker-demo")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--speaker-demo", proc.stderr)

    def test_register_speaker_rejects_mock(self) -> None:
        """--register-speaker needs the real microphone (not --mock)."""
        proc = self._run("--mock", "--register-speaker", "thomas")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--register-speaker", proc.stderr)

    def test_mock_demo_without_emotion_stays_m1_clean(self) -> None:
        proc = self._run("--mock")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-800:])
        self.assertIn('"emotion": null', proc.stdout)
        self.assertIn('"emotion_s": null', proc.stdout)

    def test_list_config_outputs_json(self) -> None:
        proc = self._run("--list-config")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-800:])
        data = json.loads(proc.stdout)
        self.assertIn("sample_rate", data)
        self.assertIn("llama_server_url", data)

    def test_check_reports_assets(self) -> None:
        proc = self._run("--check")
        self.assertIn(proc.returncode, (0, 1))
        self.assertIn("root", proc.stdout)

    def test_text_requires_mock(self) -> None:
        proc = self._run("--text", "nope")
        self.assertEqual(proc.returncode, 2)

    def test_barge_in_demo_requires_mock(self) -> None:
        proc = self._run("--barge-in-demo")
        self.assertEqual(proc.returncode, 2)


class PipelineVoiceSettingsTests(_PipelineTestBase):
    """v0.4.2: the CLI pipeline honours the persisted voice selection.

    The web UI persists mode/hu_voice/en_voice in config/voice_settings.json;
    the CLI agent resolves the SAME selection per chunk (auto: the response
    language picks the per-language voice; forced mode: every chunk uses the
    forced voice and language).
    """

    def _settings(self, tmp: Path, **kwargs: str):
        from app.voice_settings import VoiceSettings

        return VoiceSettings(root=tmp, **kwargs)

    async def test_auto_mode_selects_voice_per_language(self) -> None:
        import tempfile

        pipeline, parts = _build_pipeline(transcripts=["Mixed, please"], replies=[MIXED_REPLY])
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._voice_settings = self._settings(
                Path(tmp),
                mode="auto",
                hu_voice="hu_HU-imre-medium",
                en_voice="en_US-lessac-medium",
            )
            await pipeline.handle_utterance(_audio(pipeline._config))
        calls = parts["tts"].synthesized
        self.assertTrue(calls)
        for _text, language, _scale, voice in calls:
            if language == LANG_EN:
                self.assertEqual(voice, "en_US-lessac-medium")
            else:
                self.assertEqual(voice, "hu_HU-imre-medium")

    async def test_forced_mode_speaks_every_chunk_with_the_selected_voice(self) -> None:
        import tempfile

        pipeline, parts = _build_pipeline(transcripts=["Mixed, please"], replies=[MIXED_REPLY])
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._voice_settings = self._settings(
                Path(tmp),
                mode="hu",
                hu_voice="hu_HU-berta-medium",
                en_voice="en_US-lessac-medium",
            )
            await pipeline.handle_utterance(_audio(pipeline._config))
        calls = parts["tts"].synthesized
        self.assertTrue(calls)
        for _text, language, _scale, voice in calls:
            self.assertEqual(language, LANG_HU)
            self.assertEqual(voice, "hu_HU-berta-medium")


if __name__ == "__main__":
    unittest.main()
