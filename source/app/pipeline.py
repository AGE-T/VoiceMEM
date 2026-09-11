"""M1/M2/M3 turn orchestrator: utterance audio -> ASR -> Speaker || Emotion -> VoiceMem -> LLM -> TTS.

One :meth:`VoicePipeline.handle_utterance` call processes ONE completed
utterance (the mic/VAD loop lives in ``app/main.py``):

1. ASR: ``feed`` the utterance audio (600 ms chunk partials), then ``flush``
   for the accurate full-utterance transcript (CONTRACT step 1). Note: the
   partial pass costs extra GPU inference; if the target-machine latency
   benchmark shows it hurting p50, skipping ``feed`` for complete utterances
   is a one-line change documented here.
2. M3 speaker identification (spec 9.3): when a recognizer is injected
   and ``config.enable_speaker``, the utterance tail window is embedded
   (ECAPA, CPU, fast) and matched against the registered speakers; the
   resolved ``speaker_id`` routes the VoiceMem memory space (``user_id``)
   BEFORE retrieval - per-speaker SQLite + Qdrant collections, memories
   never mix (spec 9.3.4). An unknown voice or a dead recognizer falls
   back to the passed ``speaker_id`` (default ``"voice_user"``) - the
   pipeline never stalls (modularity 9.5 item 5/6). The embedding runs on
   a worker thread IN PARALLEL with the M2 prosody analysis start; the
   memory retrieval waits only for the fast speaker stage.
   VoiceMem retrieval AND (M2, when the emotion analyzer is injected and
   ``config.enable_emotion``) prosody analysis run IN PARALLEL on two
   worker threads (spec 8.3: memory + emotion both start from the utterance;
   the parallelism keeps the additive M2 latency near zero). The fused
   emotion (spec 8.3.3: 0.6 prosody + 0.4 semantic) is injected into the
   teacher system prompt; a frustrated user also slows the TTS down
   (Piper ``--length_scale``, spec 8.4). With no analyzer / a failing
   analyzer the turn runs M1-IDENTICALLY (modularity 8.6 item 5).
3. Teacher persona system prompt (emotion block only when a fused emotion
   exists).
4. LLM streaming: llama-server SSE deltas.
5. Speaking: ``SentenceStream`` cuts speakable chunks; each chunk is
   language-detected, synthesized by Piper (worker thread, optional emotion
   ``length_scale``) and played while the barge-in path polls the VAD
   probability (~20 ms cadence during playback; ~frame-rate approximation
   of the 500 ms sustain window).
6. Timings are recorded on a single ``time.perf_counter`` clock.
7. ``commit_reply`` feeds the COMPLETE reply back into the SPEAKER'S
   long-term memory space (M3 routing). Cancelled (barge-in) turns are
   NOT committed — partial replies must not pollute any memory store.
   M2: strong turn emotions are persisted to the JSONL emotion log and
   (when strong enough, spec 8.3.4) as a VoiceMem fact — cancelled turns
   are not persisted either.

Barge-in (milestones 7.4/7.5): the pipeline owns the :class:`EchoGate`
(loopback gating — the mic loop consults ``pipeline.echo_gate`` to decide
whether frames may reach ASR) and the :class:`BargeInDetector`. When the
detector fires (sustained speech during playback), the pipeline stops the
TTS subprocess, silences the speaker, cancels pending VoiceMem work and
aborts the LLM stream.

Timing semantics (locked contract with ``tests/latency_benchmark.py``):
``speech_end_s``/``asr_final_s``/``first_token_s``/``first_audio_s``/
``full_response_s`` are ABSOLUTE ``time.perf_counter()`` timestamps
(speech-end -> first audio = ``first_audio_s - speech_end_s``);
``memory_s`` and ``emotion_s`` are DURATIONS in seconds, not timestamps.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from app.barge_in import BargeInDetector, EchoGate
from app.config import AgentConfig
from app.emotion import EmotionAnalyzer, EmotionMemory, EmotionResult, fuse_emotion
from app.llm import LlmUnavailableError
from app.speaker import SpeakerRecognizer
from app.teacher_persona import (
    build_messages,
    build_system_prompt,
    fit_prompt_budget,
)
from app.text_utils import LANG_HU, SentenceStream, detect_language
from app.voicemem_bridge import (
    COMMIT_COMMITTED,
    COMMIT_FAILED,
    COMMIT_UNAVAILABLE,
    TurnContext,
)

if TYPE_CHECKING:  # pragma: no cover — typing only, never imported at runtime
    import numpy as np

    from app.asr_core import AsrEngineProtocol as AsrEngine
    from app.audio_io import SpeakerOutput
    from app.llm import LlmClient
    from app.speaker import SpeakerIdentification
    from app.tts import TtsEngine
    from app.vad import SileroVad
    from app.voicemem_bridge import VoiceMemBridge

#: Poll cadence while a TTS chunk plays (barge-in responsiveness vs CPU).
_PLAYBACK_POLL_S = 0.02

#: Chunk statuses returned by VoicePipeline._speak_chunk.
_CHUNK_PLAYED = "played"
_CHUNK_FAILED = "failed"  # TTS produced no audio for this chunk (skipped)
_CHUNK_ABORTED = "aborted"  # barge-in or external cancel
_CHUNK_SKIPPED = "skipped"  # non-speakable text

#: v0.4.9 (report issue #1) — memory-commit handling for a completed turn.
_MEMORY_COMMIT_INLINE_RETRIES = 2      # attempts inside the turn (0.5 s apart)
_MEMORY_COMMIT_RETRY_S = 0.5           # inline retry delay (post-reply, cheap)
_MEMORY_COMMIT_BG_INTERVAL_S = 15.0    # background queue flush cadence
_MEMORY_COMMIT_BG_ROUNDS = 5           # background flush rounds before giving up
_MEMORY_COMMIT_QUEUE_MAX = 100         # failed commits kept for retry


@dataclass
class TurnTimings:
    """Per-turn latency measurements on the ``time.perf_counter()`` clock.

    The ``*_s`` fields except ``memory_s``/``emotion_s``/``speaker_s`` are
    absolute perf_counter timestamps; ``memory_s``, ``emotion_s`` and
    ``speaker_s`` are DURATIONS (VoiceMem retrieval / M2 prosody analysis /
    M3 speaker identification seconds - the last one runs parallel with the
    emotion start, additive latency well under the 100 ms M3 budget).
    Latency (spec 19.4) = ``first_audio_s - speech_end_s``.
    """

    speech_end_s: Optional[float] = None  # t0: handle_utterance entry (VAD hangover end)
    asr_final_s: Optional[float] = None
    memory_s: Optional[float] = None  # VoiceMem retrieval duration
    emotion_s: Optional[float] = None  # M2 prosody analysis duration (parallel)
    speaker_s: Optional[float] = None  # M3 speaker identification duration (parallel)
    first_token_s: Optional[float] = None
    first_audio_s: Optional[float] = None
    full_response_s: Optional[float] = None

    def to_dict(self) -> dict[str, Optional[float]]:
        """JSON-friendly dict (all eight fields, in pipeline order)."""
        return {
            "speech_end_s": self.speech_end_s,
            "asr_final_s": self.asr_final_s,
            "memory_s": self.memory_s,
            "emotion_s": self.emotion_s,
            "speaker_s": self.speaker_s,
            "first_token_s": self.first_token_s,
            "first_audio_s": self.first_audio_s,
            "full_response_s": self.full_response_s,
        }


@dataclass
class TurnResult:
    """Everything one conversation turn produced."""

    transcript: str
    reply: str
    language: str  # "hu" | "en" (dominant reply language)
    timings: TurnTimings
    cancelled: bool = False  # barge-in aborted this turn's TTS
    barge_in: bool = False
    emotion: Optional[dict] = None  # M2: fused emotion dict (None = M1 path)
    speaker: Optional[dict] = None  # M3: identification dict (None = no M3 stage)
    speaker_id: str = "voice_user"  # M3: the user_id actually used for routing
    # v0.4.9 (report issue #1): long-term memory outcome of the turn —
    # "committed" (stored), "unavailable" (no voicemem package: degraded
    # mode, not data loss), "failed" (NOT stored even after retries — the
    # reply is queued for background retry and the failure is VISIBLE here
    # instead of the old silent warning), "queued" (stored later by the
    # background flush), "skipped" (no reply / cancelled turn).
    memory_status: str = "skipped"
    memory_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (timings nested, emotion/speaker nested)."""
        return {
            "transcript": self.transcript,
            "reply": self.reply,
            "language": self.language,
            "cancelled": self.cancelled,
            "barge_in": self.barge_in,
            "emotion": self.emotion,
            "speaker": self.speaker,
            "speaker_id": self.speaker_id,
            "memory_status": self.memory_status,
            "memory_error": self.memory_error,
            "timings": self.timings.to_dict(),
        }


class VoicePipeline:
    """Turn orchestrator. Components are injected (real or mock, duck-typed)."""

    def __init__(
        self,
        config: AgentConfig,
        asr: "AsrEngine",
        llm: "LlmClient",
        tts: "TtsEngine",
        voicemem: "VoiceMemBridge",
        vad: "SileroVad",
        audio_out: Optional["SpeakerOutput"],
        logger: Optional[logging.Logger] = None,
        emotion: Optional["EmotionAnalyzer"] = None,
        emotion_memory: Optional["EmotionMemory"] = None,
        speaker: Optional[SpeakerRecognizer] = None,
    ) -> None:
        self._config = config
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._voicemem = voicemem
        self._vad = vad
        self._audio_out = audio_out
        self._log = logger or logging.getLogger(__name__)
        #: M2 emotion analyzer (None -> M1-identical pipeline, 8.6 modularity).
        self._emotion = emotion
        self._emotion_memory = emotion_memory
        #: M3 speaker recognizer (None -> M1/M2-identical pipeline, 9.5 modularity).
        self._speaker = speaker
        #: Loopback gate — the mic loop consults this to block ASR while TTS plays.
        self.echo_gate = EchoGate()
        self._barge_in = BargeInDetector(
            config.barge_in_threshold,
            config.barge_in_min_speech_ms,
            config.vad_frame_ms,
        )
        self._cancelled = False
        self._barge_in_fired = False
        #: v0.4.2 voice selection: the CLI agent honours the SAME persisted
        #: selection as the web UI (config/voice_settings.json) so a voice
        #: chosen in the browser is what speaks here too. Lenient load.
        try:
            from app.voice_settings import VoiceSettings

            root = Path(__file__).resolve().parent.parent
            self._voice_settings = VoiceSettings.load(root, config)
        except Exception:  # noqa: BLE001 - never block the agent on settings
            self._voice_settings = None
        # v0.4.9 (report issue #1): failed memory commits are queued here and
        # retried by a background task instead of vanishing behind a warning
        # log — the user's replies must not silently disappear from long-term
        # memory when the backend hiccups (disk full, Qdrant restart, ...).
        self._pending_commits: "deque[tuple[str, str]]" = deque(
            maxlen=_MEMORY_COMMIT_QUEUE_MAX
        )
        self._commit_flush_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ public

    @property
    def barge_in_detector(self) -> BargeInDetector:
        """The one-shot barge-in detector (reset automatically per turn)."""
        return self._barge_in

    def cancel(self) -> None:
        """Request cancellation of the in-flight turn (safe from any task).

        Stops TTS synthesis and speaker playback immediately; the streaming
        loop observes the flag at its next poll and aborts the turn.
        """
        self._cancelled = True
        self._abort_playback()

    async def handle_utterance(
        self,
        audio: Optional["np.ndarray"],
        speaker_id: str = "voice_user",
        vad_prob_fn: Optional[Callable[[], float]] = None,
    ) -> TurnResult:
        """Process one complete utterance end to end (CONTRACT step list 1-7)."""
        timings = TurnTimings()
        timings.speech_end_s = time.perf_counter()  # t0 of the latency window
        self._cancelled = False
        self._barge_in_fired = False
        self._barge_in.reset()
        self.echo_gate.on_tts_end()  # never leak playback state across turns
        # v0.4.9 (issue #1): a recovered memory backend is used IMMEDIATELY —
        # queued commits from earlier failed turns are retried at the start of
        # the next turn, not only on the background cadence.
        if self._pending_commits:
            try:
                await self.retry_pending_commits()
            except Exception:  # noqa: BLE001 - housekeeping, never blocks a turn
                self._log.debug("pending commit flush failed", exc_info=True)

        # 1) ASR: quasi-streaming partials + accurate full-utterance final.
        transcript = await self._transcribe(audio, timings)
        if not transcript:
            timings.full_response_s = time.perf_counter()
            self._reset_vad()
            self._log.info("Empty transcript — turn skipped")
            return TurnResult(transcript="", reply="", language=LANG_HU, timings=timings)

        # 2) M3 speaker identification FIRST (it routes the memory space).
        #    The ECAPA embedding is fast (CPU) and starts IN PARALLEL with the
        #    M2 prosody analysis; the memory retrieval waits only for the
        #    speaker stage (spec 9.3: user_id is resolved before VoiceMem).
        emotion_enabled = self._emotion is not None and self._config.enable_emotion
        speaker_enabled = self._speaker is not None and self._config.enable_speaker
        emotion_task = (
            asyncio.create_task(self._analyze_emotion(audio, timings))
            if emotion_enabled
            else None
        )
        identification = None
        if speaker_enabled:
            identification = await self._identify_speaker(audio, timings)
            if identification is not None and identification.id:
                speaker_id = identification.id
        # VoiceMem retrieval — in PARALLEL with the M2 prosody analysis
        # (spec 8.3: both start from the same utterance).
        if emotion_task is not None:
            turn_ctx, prosody = await asyncio.gather(
                self._retrieve_memory(transcript, speaker_id, timings),
                emotion_task,
            )
            fused = fuse_emotion(
                prosody, transcript, self._config.emotion_fusion_prosody_weight
            )
        else:
            turn_ctx = await self._retrieve_memory(transcript, speaker_id, timings)
            fused = None

        # 3) Teacher persona prompt (emotion block only when a fused emotion
        #    exists — M1 path builds the identical M1 prompt).
        if fused is not None:
            system_prompt = build_system_prompt(
                turn_ctx.memory_context,
                emotion_label=fused.label,
                emotion_valence=fused.valence,
                emotion_arousal=fused.arousal,
            )
            self._log.info(
                "Emotion (M2): %s valence=%.2f arousal=%.2f (raw=%s, %.1f ms)",
                fused.label,
                fused.valence,
                fused.arousal,
                fused.raw_label,
                fused.latency_ms,
            )
        else:
            system_prompt = build_system_prompt(turn_ctx.memory_context)
        # v0.4.9 (report issue #3): the CLI path had NO context-budget guard
        # (the v0.4.5 fix covered only web_server.py). The teacher prompt with
        # a long memory block could exceed llama-server's fixed 8192-token
        # context and lose the whole reply; the shared 14000-char budget
        # trims oldest history first, then the memory block, never the user
        # text (see app.teacher_persona.fit_prompt_budget).
        messages = fit_prompt_budget(build_messages(transcript, system_prompt))

        # M2 spec 8.4: a frustrated user hears a ~10% slower reply.
        length_scale = (
            self._config.emotion_slow_length_scale
            if fused is not None and fused.label == "frustrated"
            else None
        )

        # 4)+5) LLM streaming + chunked TTS with barge-in polling.
        reply, barge_in_fired, aborted = await self._stream_and_speak(
            messages, timings, vad_prob_fn, length_scale
        )

        timings.full_response_s = time.perf_counter()
        self.echo_gate.on_tts_end()
        self._reset_vad()

        result = TurnResult(
            transcript=transcript,
            reply=reply.strip(),
            language=detect_language(reply) if reply.strip() else LANG_HU,
            timings=timings,
            cancelled=aborted or self._cancelled,
            barge_in=barge_in_fired,
            emotion=fused.to_dict() if fused is not None else None,
            speaker=identification.to_dict() if identification is not None else None,
            speaker_id=speaker_id,
        )

        # 7) Long-term memory consolidation — complete turns only (see module doc).
        if result.reply and not result.cancelled:
            status, error = await self._commit_reply_with_retry(
                result.reply, speaker_id
            )
            result.memory_status = status
            result.memory_error = error
            if status == COMMIT_FAILED:
                self._log.error(
                    "MEMORY COMMIT FAILED after retries: this reply is NOT in "
                    "long-term memory (queued for background retry; %d pending): %s",
                    len(self._pending_commits),
                    error,
                )
            if fused is not None:
                await self._store_emotion(fused, transcript, speaker_id)
        self._log_turn(result)
        return result

    # ----------------------------------------------------------------- stages

    # -- v0.4.9 (report issue #1): memory-commit retry + background queue ------ #

    async def _commit_reply_with_retry(
        self, reply: str, speaker_id: str
    ) -> "tuple[str, str]":
        """Commit the reply with retries; queue it when the backend stays down.

        The v0.4.7 analysis report's issue #1: ``commit_reply`` failures were
        swallowed as warnings and the turn still counted as succeeded — the
        user's replies silently vanished from long-term memory ("the agent has
        amnesia"). Now: up to ``_MEMORY_COMMIT_INLINE_RETRIES`` immediate
        attempts (``_MEMORY_COMMIT_RETRY_S`` apart — the reply was already
        spoken, the delay is post-turn housekeeping); on persistent failure the
        reply is QUEUED and a background task keeps retrying, and the failure
        is returned to the caller (TurnResult.memory_status = "failed") so the
        CLI can show it. Memory failures still never kill a turn.
        """
        status, error = await self._try_commit(reply, speaker_id)
        attempt = 1
        while status == COMMIT_FAILED and attempt < _MEMORY_COMMIT_INLINE_RETRIES:
            await asyncio.sleep(_MEMORY_COMMIT_RETRY_S)
            attempt += 1
            status, error = await self._try_commit(reply, speaker_id)
        if status == COMMIT_FAILED:
            self._pending_commits.append((reply, speaker_id))
            self._schedule_commit_flush()
        return status, error

    async def _try_commit(self, reply: str, speaker_id: str) -> "tuple[str, str]":
        """One commit attempt; maps old-None bridges to COMMIT_COMMITTED."""
        try:
            status = await self._voicemem.commit_reply(reply, speaker_id=speaker_id)
        except Exception as exc:  # noqa: BLE001 - memory must never kill a turn
            return COMMIT_FAILED, str(exc)
        if status is None:
            # Duck-typed bridges on the old (silent) contract: treat as done.
            return COMMIT_COMMITTED, ""
        return str(status), ""

    def _schedule_commit_flush(self) -> None:
        """Start the background queue flusher once (kept referenced, no GC)."""
        if self._commit_flush_task is not None and not self._commit_flush_task.done():
            return
        try:
            self._commit_flush_task = asyncio.get_running_loop().create_task(
                self._commit_flush_loop()
            )
        except RuntimeError:  # no running loop (tests driving methods directly)
            self._commit_flush_task = None

    async def _commit_flush_loop(self) -> None:
        """Retry queued commits every interval, up to N rounds, then stop.

        A round re-attempts every queued commit; committed ones leave the
        queue. When the queue empties, the task ends (it restarts on the next
        failure). Giving up logs each remaining reply at ERROR level — never
        silent.
        """
        for round_no in range(1, _MEMORY_COMMIT_BG_ROUNDS + 1):
            await asyncio.sleep(_MEMORY_COMMIT_BG_INTERVAL_S)
            if not self._pending_commits:
                return
            await self.retry_pending_commits()
            if not self._pending_commits:
                self._log.info(
                    "Memory commit queue drained (round %d): all replies stored",
                    round_no,
                )
                return
        if self._pending_commits:
            self._log.error(
                "Memory commit queue giving up after %d rounds: %d replies are "
                "NOT in long-term memory: %s",
                _MEMORY_COMMIT_BG_ROUNDS,
                len(self._pending_commits),
                "; ".join(
                    f"{reply[:60]!r} (speaker={speaker})"
                    for reply, speaker in list(self._pending_commits)[:5]
                ),
            )

    async def retry_pending_commits(self) -> int:
        """Retry every queued commit once; return how many were stored.

        Public on purpose: the CLI/tests can drive a flush explicitly, and
        the next ``handle_utterance`` calls it so a recovered backend is used
        immediately (not only by the background cadence).
        """
        stored = 0
        for reply, speaker_id in list(self._pending_commits):
            status, _error = await self._try_commit(reply, speaker_id)
            if status != COMMIT_FAILED:
                try:
                    self._pending_commits.remove((reply, speaker_id))
                except ValueError:  # pragma: no cover - concurrent flush
                    pass
                stored += 1
                self._log.info(
                    "Queued memory commit stored (speaker=%s): %r",
                    speaker_id,
                    reply[:60],
                )
        return stored

    async def _transcribe(
        self, audio: Optional["np.ndarray"], timings: TurnTimings
    ) -> str:
        """ASR stage (v0.6.0 engine contract): one authoritative path.

        The completed utterance goes to the selected engine's
        ``transcribe(AudioBuffer)`` and returns the transcript. Engine
        failures are EXPLICIT: the structured AsrError is logged with its
        code/stage/reason and the turn is skipped (no partial-join
        fallback, no engine switch, no empty-string-as-success).
        """
        from app.asr_core import AudioBuffer

        if audio is None or int(getattr(audio, "size", 0) or 0) == 0:
            timings.asr_final_s = time.perf_counter()
            return ""
        try:
            buffer = AudioBuffer.from_float(audio, self._config.sample_rate)
            result = await asyncio.to_thread(self._asr.transcribe, buffer)
        except Exception as exc:  # noqa: BLE001 - engine raised outside the contract
            self._log.error("ASR failed: %s", exc)
            timings.asr_final_s = time.perf_counter()
            return ""
        timings.asr_final_s = time.perf_counter()
        if result.error is not None:
            err = result.error.to_dict()
            self._log.error(
                "ASR failed: %s stage=%s reason=%s — %s",
                err.get("code"),
                err.get("stage"),
                err.get("reason"),
                err.get("detail", ""),
            )
            return ""
        return (result.text or "").strip()

    async def _retrieve_memory(
        self, transcript: str, speaker_id: str, timings: TurnTimings
    ) -> TurnContext:
        """VoiceMem stage: retrieval context for the system prompt."""
        start = time.perf_counter()
        try:
            turn_ctx = await self._voicemem.process_turn(transcript, speaker_id=speaker_id)
        except Exception as exc:  # noqa: BLE001 - degraded mode, never fatal
            self._log.warning("VoiceMem retrieval failed (continuing without memory): %s", exc)
            turn_ctx = TurnContext(transcript=transcript, speaker_id=speaker_id)
        timings.memory_s = time.perf_counter() - start
        if turn_ctx.memory_context:
            self._log.info("VoiceMem context (%d chars) attached", len(turn_ctx.memory_context))
        return turn_ctx

    async def _analyze_emotion(
        self, audio: Optional["np.ndarray"], timings: TurnTimings
    ) -> Optional[EmotionResult]:
        """M2 prosody stage: analyze the utterance tail on a worker thread.

        Any failure (missing model, broken funasr, odd audio) degrades to
        ``None`` — the turn then continues exactly like M1 (8.6 item 5).
        """
        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(self._emotion.analyze, audio)
        except Exception as exc:  # noqa: BLE001 - emotion must never kill a turn
            self._log.warning("Emotion analysis raised (non-fatal): %s", exc)
            result = None
        timings.emotion_s = time.perf_counter() - start
        return result

    async def _identify_speaker(
        self, audio: Optional["np.ndarray"], timings: TurnTimings
    ) -> Optional["SpeakerIdentification"]:
        """M3 speaker stage: embed the utterance tail and match the registry.

        Runs the recognizer on a worker thread. Any failure (missing model,
        broken speechbrain, odd audio) degrades to ``None`` - the turn then
        routes to the passed ``speaker_id`` exactly like M1/M2 (9.5 items
        5/6: graceful degradation, never a stall).
        """
        start = time.perf_counter()
        try:
            identification = await asyncio.to_thread(self._speaker.identify, audio)
        except Exception as exc:  # noqa: BLE001 - speaker must never kill a turn
            self._log.warning("Speaker identification raised (non-fatal): %s", exc)
            identification = None
        timings.speaker_s = time.perf_counter() - start
        if identification is not None:
            self._log.info(
                "Speaker (M3): %s similarity=%.2f registered=%d (%.1f ms)",
                identification.id or "UNKNOWN",
                identification.similarity,
                identification.registered,
                identification.latency_ms,
            )
        return identification

    async def _store_emotion(
        self, fused: EmotionResult, transcript: str, speaker_id: str
    ) -> None:
        """M2 spec 8.3.4: persist the turn emotion.

        Every analyzed turn lands in the JSONL emotion log; a STRONG emotion
        additionally becomes a VoiceMem fact (the "RightBrain" long-term
        aggregation layer). Everything is best-effort and never raises.
        """
        stored_as_fact = False
        threshold = self._config.emotion_store_threshold
        try:
            if self._emotion_memory is not None:
                stored_as_fact = self._emotion_memory.should_store(fused, threshold)
                self._emotion_memory.record_turn(fused, transcript, stored_as_fact)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("Emotion log write failed (non-fatal): %s", exc)
        if not stored_as_fact:
            return
        fact = self._emotion_memory.fact_text(fused, transcript) if self._emotion_memory else None
        if fact:
            try:
                store = getattr(self._voicemem, "store_fact", None)
                if callable(store):
                    await store(fact, speaker_id=speaker_id)
                    self._log.info("Emotion fact stored in VoiceMem: %s", fused.label)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Emotion fact store failed (non-fatal): %s", exc)

    async def _stream_and_speak(
        self,
        messages: list[dict],
        timings: TurnTimings,
        vad_prob_fn: Optional[Callable[[], float]],
        length_scale: Optional[float] = None,
    ) -> tuple[str, bool, bool]:
        """Stages 4+5: stream the reply, speak chunk by chunk, watch barge-in.

        ``length_scale`` (M2): forwarded to every TTS synthesis call (None =
        piper default). Returns ``(reply_text, barge_in_fired, aborted)``.
        """
        stream = SentenceStream(
            first_chunk_chars=self._config.tts_first_chunk_chars,
            chunk_chars=self._config.tts_chunk_chars,
        )
        parts: list[str] = []
        aborted = False
        first_token_done = False

        try:
            async for delta in self._llm.chat_stream(messages):
                if await self._maybe_abort(vad_prob_fn):
                    aborted = True
                    break
                if not delta:
                    continue
                if not first_token_done:
                    timings.first_token_s = time.perf_counter()
                    first_token_done = True
                parts.append(delta)
                for chunk in stream.add_delta(delta):
                    status = await self._speak_chunk(chunk, timings, vad_prob_fn, length_scale)
                    if status == _CHUNK_ABORTED:
                        aborted = True
                        break
                if aborted:
                    break
        except LlmUnavailableError as exc:
            self._log.error("LLM unavailable: %s", exc)
        except Exception as exc:  # noqa: BLE001 - the loop must survive reply errors
            self._log.exception("Unexpected LLM streaming error: %s", exc)

        if not aborted and not self._cancelled:
            for chunk in stream.flush():
                status = await self._speak_chunk(chunk, timings, vad_prob_fn, length_scale)
                if status == _CHUNK_ABORTED:
                    aborted = True
                    break

        reply = "".join(parts)
        return reply, self._barge_in_fired, aborted or self._cancelled

    async def _speak_chunk(
        self,
        text: str,
        timings: TurnTimings,
        vad_prob_fn: Optional[Callable[[], float]],
        length_scale: Optional[float] = None,
    ) -> str:
        """Synthesize and play ONE chunk; returns a ``_CHUNK_*`` status.

        Polls barge-in before synthesis, after synthesis and every
        ``_PLAYBACK_POLL_S`` during playback. ``audio_out=None`` (offline
        audit / latency benchmark) skips playback but still stamps
        ``first_audio_s`` after synthesis. ``length_scale`` (M2) is
        forwarded to the TTS (None = piper default).
        """
        if not text or not text.strip():
            return _CHUNK_SKIPPED
        if await self._maybe_abort(vad_prob_fn):
            return _CHUNK_ABORTED

        language = detect_language(text)
        voice_id: Optional[str] = None
        if self._voice_settings is not None:
            voice_id, language = self._voice_settings.resolve(language)
        try:
            pcm = await asyncio.to_thread(
                self._tts.synthesize, text, language, length_scale, voice_id
            )
        except Exception as exc:  # noqa: BLE001 - TTS failure skips the chunk
            self._log.warning("TTS synthesis raised for chunk %r: %s", text[:40], exc)
            pcm = None
        if pcm is None:
            self._log.warning("TTS produced no audio; chunk skipped: %r", text[:60])
            return _CHUNK_FAILED
        if await self._maybe_abort(vad_prob_fn):
            return _CHUNK_ABORTED

        if self._audio_out is None:
            if timings.first_audio_s is None:
                timings.first_audio_s = time.perf_counter()
            return _CHUNK_PLAYED

        # Playback (gate closes: the mic loop stops feeding ASR).
        self.echo_gate.on_tts_start()
        if timings.first_audio_s is None:
            timings.first_audio_s = time.perf_counter()
        play_task = asyncio.create_task(self._audio_out.play_async(pcm))
        status = _CHUNK_PLAYED
        try:
            while not play_task.done():
                await asyncio.sleep(_PLAYBACK_POLL_S)
                if await self._maybe_abort(vad_prob_fn):
                    status = _CHUNK_ABORTED
                    break
            if status == _CHUNK_ABORTED and not play_task.done():
                # _maybe_abort already called audio_out.stop(); cancel the task
                # as well so teardown does not wait for the audio to finish.
                play_task.cancel()
        finally:
            try:
                await play_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - playback errors are logged, not raised
                self._log.debug("Playback task ended with an error (ignored)", exc_info=True)
        return status

    # -------------------------------------------------------------- internals

    async def _maybe_abort(
        self, vad_prob_fn: Optional[Callable[[], float]]
    ) -> bool:
        """One barge-in poll: True when the turn must abort.

        Feeds the polled probability into the detector (which only
        accumulates while TTS playback is active) and triggers the abort
        actions exactly once when it fires. External ``cancel()`` also
        aborts here.
        """
        if self._cancelled:
            return True
        if vad_prob_fn is None:
            return False
        try:
            prob = float(vad_prob_fn())
        except Exception:  # noqa: BLE001 - a broken probe must not kill the turn
            return False
        if self._barge_in.update(prob, self.echo_gate.tts_playing):
            self._barge_in_fired = True
            self._log.info(
                "Barge-in: sustained speech during playback — cancelling the turn"
            )
            self._abort_playback()
            return True
        return self._cancelled

    def _abort_playback(self) -> None:
        """Stop TTS synthesis, speaker playback and pending memory work now."""
        try:
            self._tts.stop()
        except Exception:  # noqa: BLE001
            self._log.debug("TTS stop failed (ignored)", exc_info=True)
        if self._audio_out is not None:
            try:
                self._audio_out.stop()
            except Exception:  # noqa: BLE001
                self._log.debug("Speaker stop failed (ignored)", exc_info=True)
        try:
            self._voicemem.cancel_pending()
        except Exception:  # noqa: BLE001
            self._log.debug("VoiceMem cancel_pending failed (ignored)", exc_info=True)

    def _reset_vad(self) -> None:
        """Reset the Silero recurrent state between turns (best effort)."""
        vad = self._vad
        reset = getattr(vad, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001
                self._log.debug("VAD reset failed (ignored)", exc_info=True)

    def _log_turn(self, result: TurnResult) -> None:
        """One INFO line summarising the turn (visible with default log level)."""
        t = result.timings
        e2e = (
            f"{(t.full_response_s or t.speech_end_s) - (t.speech_end_s or 0.0):.2f}s"
            if t.speech_end_s is not None
            else "n/a"
        )
        first_audio = (
            f"{t.first_audio_s - t.speech_end_s:.2f}s"
            if t.speech_end_s is not None and t.first_audio_s is not None
            else "n/a"
        )
        emotion = ""
        if result.emotion:
            emotion = f" emotion={result.emotion.get('label')}"
        speaker = ""
        if result.speaker is not None:
            speaker = f" speaker={result.speaker.get('id') or 'UNKNOWN'}"
        self._log.info(
            "Turn done: %s reply=%d chars %s%s%s e2e=%s first_audio=%s cancelled=%s",
            result.language,
            len(result.reply),
            f"[barge-in] " if result.barge_in else "",
            emotion,
            speaker,
            e2e,
            first_audio,
            result.cancelled,
        )
