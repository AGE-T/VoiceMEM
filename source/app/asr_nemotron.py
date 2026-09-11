"""Nemotron 3.5 ASR streaming 0.6B engine adapter (transformers-native).

TASK-A Phase 6A/9. ``nvidia/nemotron-3.5-asr-streaming-0.6b`` run through
the transformers 5.13+ ``AutoModelForRNNT`` + ``AutoProcessor`` classes —
the OFFICIAL NVIDIA integration path (the model card's own examples; no
NeMo toolkit, no community wrapper).

Verified capabilities of this checkpoint (transformers 5.17):

* OFFLINE transcription: ``processor(audio, language=...)`` ->
  ``model.generate(**inputs)``. Language prompt conditioning with 40
  locales (``hu-HU`` prompt index 23, ``auto`` = 101 for LID).
* TRUE cache-aware streaming (``NemotronAsrStreamingGenerationMixin``
  through ``Nemotron3_5AsrGenerationMixin``): ``generate`` accepts a
  GENERATOR of mel chunks with exact cache-aware sizes
  (first chunk ``1 + subsampling * lookahead`` mel frames, subsequent
  ``subsampling * (lookahead + 1)``), maintains encoder past-key cache +
  decoder cache, and emits partial text through a
  ``TextIteratorStreamer``. Supported lookahead values: 0 (80 ms chunks),
  3 (320 ms), 6 (560 ms), 13 (1120 ms).

This adapter exposes BOTH paths through the production contract:

* :meth:`transcribe` — offline one-shot (full utterance).
* :meth:`start` / :meth:`feed` / :meth:`finish` — the REAL streaming
  interface: a queue-backed chunk generator bridges the push-based VAD
  frame cadence (512 samples @ 16 kHz) into the pull-based generate()
  loop, running in a worker thread; partial transcripts surface through
  the streamer. No chunked-batch-as-streaming pretence: partials come
  from the model's cache-aware incremental decode.

Device rule (task contract): cpu and cuda are both valid EXECUTION modes
of the same model. ``ASR_DEVICE=cuda`` without CUDA fails EXPLICITLY
(reason ``cuda_unavailable``) — no silent device fallback.

Thread safety: one module lock serialises model access; a live stream
owns the backend (a concurrent ``transcribe`` would corrupt the RNNT
state — the lock order prevents it).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.asr_core import (
    AsrCapability,
    AsrError,
    AsrErrorCode,
    AsrModelSpec,
    AsrResult,
    AsrResultStatus,
    AudioBuffer,
    REASON_CUDA_UNAVAILABLE,
    REASON_DEPENDENCY_MISSING,
    REASON_DECODE_FAILED,
    REASON_INFERENCE_FAILED,
    REASON_MODEL_UNAVAILABLE,
    REASON_STREAMING_UNSUPPORTED,
)
from app.config import AgentConfig

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import torch  # type: ignore[import-untyped]

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

#: Shared backend cache: ONE loaded model per (model_dir, device, dtype).
_BACKEND_CACHE: "dict[tuple, Any]" = {}
_BACKEND_LOCK = threading.Lock()

#: Default cache-aware streaming lookahead (3 -> 320 ms chunks; the model
#: card's HU accuracy improves with larger chunks, latency tradeoff).
DEFAULT_LOOKAHEAD = 3

#: How long finish() waits for the streaming generate thread (seconds).
#: CPU-only execution of the 0.6B cache-aware streamer is ~RTF 12-16 on
#: the sandbox (no GPU): a 10 s utterance legitimately takes >120 s. CUDA
#: is the production target (80-1120 ms chunk latency per the model card);
#: the generous bound keeps CPU validation honest instead of timing out.
_STREAM_JOIN_TIMEOUT_S = 300.0


def _nemotron_classes() -> tuple[Any, Any]:
    if not _HAS_TORCH:
        raise AsrError(
            code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
            stage="engine_load",
            engine="nemotron",
            reason=REASON_DEPENDENCY_MISSING,
            detail="torch is not installed (pip install torch)",
        )
    try:
        from transformers import AutoModelForRNNT, AutoProcessor  # type: ignore
    except ImportError as exc:
        raise AsrError(
            code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
            stage="engine_load",
            engine="nemotron",
            reason=REASON_DEPENDENCY_MISSING,
            detail=f"transformers with AutoModelForRNNT is required (>=5.13): {exc}",
        ) from exc
    return AutoModelForRNNT, AutoProcessor


def _shared_backend(model_dir: str, device: str, dtype: str) -> Any:
    key = (model_dir, device, dtype)
    with _BACKEND_LOCK:
        cached = _BACKEND_CACHE.get(key)
        if cached is not None:
            return cached
        AutoModelForRNNT, AutoProcessor = _nemotron_classes()
        path = Path(model_dir)
        source: Any = path if path.is_dir() and (path / "config.json").is_file() else model_dir
        t0 = time.perf_counter()
        try:
            processor = AutoProcessor.from_pretrained(source)
            torch_dtype: Any = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(dtype, torch.float32)
            model = AutoModelForRNNT.from_pretrained(source, torch_dtype=torch_dtype)
        except Exception as exc:  # noqa: BLE001
            raise AsrError(
                code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
                stage="engine_load",
                engine="nemotron",
                reason=REASON_MODEL_UNAVAILABLE,
                detail=f"loading {source} failed: {exc}",
                diagnostics={"model_dir": str(model_dir), "device": device},
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise AsrError(
                code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
                stage="engine_load",
                engine="nemotron",
                reason=REASON_CUDA_UNAVAILABLE,
                detail="ASR_DEVICE=cuda but torch.cuda.is_available() is False; "
                "set ASR_DEVICE=cpu or fix the CUDA install",
            )
        try:
            model = model.to(device)
        except Exception as exc:  # noqa: BLE001
            raise AsrError(
                code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
                stage="engine_load",
                engine="nemotron",
                reason=REASON_MODEL_UNAVAILABLE,
                detail=f"moving model to {device} failed: {exc}",
            ) from exc
        model = model.eval()
        backend = {"processor": processor, "model": model, "device": device}
        _BACKEND_CACHE[key] = backend
        logger.info(
            "Nemotron 3.5 ASR backend loaded from %s on %s (%s) in %.0f ms",
            source,
            device,
            dtype,
            (time.perf_counter() - t0) * 1000.0,
        )
        return backend


def _strip_language_tag(text: str) -> tuple[str, str]:
    """Split a leading ``<xx-XX>`` LID tag: (clean_text, language)."""
    import re

    m = re.match(r"^\s*<([a-zA-Z]{2,3}(?:[-_][a-zA-Z]{2,4})?)>\s*", text)
    if not m:
        return text.strip(), ""
    return text[m.end() :].strip(), m.group(1).replace("_", "-")


class NemotronEngine:
    """ASR engine adapter for nvidia/nemotron-3.5-asr-streaming-0.6b.

    Genuine cache-aware streaming (start/feed/finish) AND offline
    transcription, both through the official transformers path.
    """

    engine_id = "nemotron"
    model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    capabilities = AsrCapability(
        streaming=True,
        language_detection=True,
        timestamps=False,
        word_timestamps=False,
        confidence=False,
        multilingual=True,
        hotwords=False,
    )

    def __init__(self, config: AgentConfig, spec: AsrModelSpec) -> None:
        self._config = config
        self._spec = spec
        self._backend: Optional[dict] = None
        self._load_error: Optional[str] = None
        self._last_inference_ms = 0.0
        # Streaming state
        self._stream_q: Optional["queue.Queue[Optional[np.ndarray]]"] = None
        self._partial_text = ""
        self._partial_consumed = 0
        self._gen_thread: Optional[threading.Thread] = None
        self._partial_thread: Optional[threading.Thread] = None
        self._stream_error: Optional[AsrError] = None
        self._stream_started_at = 0.0
        self._lookahead = int(getattr(config, "asr_stream_lookahead", DEFAULT_LOOKAHEAD))

    # ------------------------------------------------------------ lifecycle

    def load(self) -> None:
        if self._backend is not None:
            return
        if self._load_error is not None:
            raise AsrError(
                code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
                stage="engine_load",
                engine=self.engine_id,
                reason=REASON_MODEL_UNAVAILABLE,
                detail=self._load_error,
            )
        model_dir = self._resolve_model_dir()
        try:
            self._backend = _shared_backend(
                model_dir, self._resolve_device(), self._resolve_dtype()
            )
        except AsrError as exc:
            self._load_error = exc.detail
            raise

    def unload(self) -> None:
        self._backend = None
        self._load_error = None

    def is_loaded(self) -> bool:
        return self._backend is not None

    def is_available(self) -> bool:
        return _HAS_TORCH

    def status(self) -> dict:
        return {
            "engine": self.engine_id,
            "model": self.model_id,
            "revision": self._spec.revision,
            "loaded": self.is_loaded(),
            "device": self._resolve_device(),
            "compute_type": self._resolve_dtype(),
            "model_dir": str(self._resolve_model_dir()),
            "model_dir_present": Path(self._resolve_model_dir()).is_dir()
            and (Path(self._resolve_model_dir()) / "config.json").is_file(),
            "capabilities": {
                "streaming": self.capabilities.streaming,
                "language_detection": self.capabilities.language_detection,
                "multilingual": self.capabilities.multilingual,
            },
            "stream_lookahead": self._lookahead,
            "last_inference_ms": round(self._last_inference_ms, 1),
            "last_error": self._load_error or "",
        }

    def warm_up(self) -> bool:
        try:
            self.load()
            result = self.transcribe(
                AudioBuffer(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)
            )
            return result.status == AsrResultStatus.OK
        except Exception as exc:  # noqa: BLE001
            logger.warning("Nemotron warm-up failed: %s", exc)
            self._load_error = str(exc)
            return False

    # ------------------------------------------------------------ language

    def _language_arg(self) -> str:
        """Resolve the processor language argument from config."""
        lang = (self._config.asr_language or "").strip()
        if not lang:
            return "auto"
        # Accept "hu"/"en" ISO codes and map to the processor's locale keys.
        mapping = {"hu": "hu-HU", "en": "en-US", "de": "de-DE", "es": "es-ES", "fr": "fr-FR"}
        return mapping.get(lang.lower(), lang)

    # ------------------------------------------------------------ offline transcribe

    def transcribe(self, audio: AudioBuffer) -> AsrResult:
        """Offline full-utterance transcription -> :class:`AsrResult`."""
        try:
            self.load()
        except AsrError as exc:
            return AsrResult.from_error(exc, self.engine_id, self.model_id)
        try:
            audio.validated(stage="asr_input")
        except AsrError as exc:
            return AsrResult.from_error(
                AsrError(
                    code=exc.code,
                    stage="asr_input",
                    engine=self.engine_id,
                    reason=exc.reason,
                    detail=exc.detail,
                    diagnostics=exc.diagnostics,
                ),
                self.engine_id,
                self.model_id,
            )
        if audio.sample_count == 0:
            return AsrResult(
                text="",
                engine_id=self.engine_id,
                model_id=self.model_id,
                status=AsrResultStatus.OK,
            )
        if self._gen_thread is not None and self._gen_thread.is_alive():
            return AsrResult.from_error(
                AsrError(
                    code=AsrErrorCode.ASR_INPUT_ERROR,
                    stage="asr_transcribe",
                    engine=self.engine_id,
                    reason=REASON_INFERENCE_FAILED,
                    detail="a streaming session is active on this backend; "
                    "transcribe() cannot run concurrently",
                ),
                self.engine_id,
                self.model_id,
            )
        processor = self._backend["processor"]
        model = self._backend["model"]
        device = self._backend["device"]
        lang = self._language_arg()
        t0 = time.perf_counter()
        try:
            with _BACKEND_LOCK:
                inputs = processor(
                    audio=np.asarray(audio.samples, dtype=np.float32),
                    sampling_rate=audio.sample_rate,
                    language=lang,
                    return_tensors="pt",
                )
                inputs = {
                    k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=int(self._config.asr_max_new_tokens))
        except AsrError as exc:
            return AsrResult.from_error(exc, self.engine_id, self.model_id)
        except Exception as exc:  # noqa: BLE001
            return AsrResult.from_error(
                AsrError(
                    code=AsrErrorCode.ASR_INFERENCE_ERROR,
                    stage="asr_inference",
                    engine=self.engine_id,
                    reason=REASON_INFERENCE_FAILED,
                    detail=str(exc)[:500],
                    diagnostics={"audio_s": round(audio.duration_s, 3), "language": lang},
                ),
                self.engine_id,
                self.model_id,
            )
        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
        try:
            sequences = out.sequences if hasattr(out, "sequences") else out
            ids = sequences[0].tolist() if hasattr(sequences, "tolist") else list(sequences[0])
            keep_tag = lang == "auto"
            text = processor.decode(ids, skip_special_tokens=not keep_tag)
        except Exception as exc:  # noqa: BLE001
            return AsrResult.from_error(
                AsrError(
                    code=AsrErrorCode.ASR_DECODE_ERROR,
                    stage="asr_decode",
                    engine=self.engine_id,
                    reason=REASON_DECODE_FAILED,
                    detail=str(exc)[:500],
                ),
                self.engine_id,
                self.model_id,
            )
        language = ""
        if lang == "auto":
            text, language = _strip_language_tag(text)
        else:
            language = lang
        return AsrResult(
            text=text.strip(),
            language=language,
            duration_s=audio.duration_s,
            engine_id=self.engine_id,
            model_id=self.model_id,
            status=AsrResultStatus.OK,
            inference_ms=self._last_inference_ms,
        )

    # ------------------------------------------------------------ streaming (real)

    def start(self) -> None:
        """Begin a cache-aware streaming session (owns the backend)."""
        self.load()
        if self._gen_thread is not None and self._gen_thread.is_alive():
            raise AsrError(
                code=AsrErrorCode.ASR_INPUT_ERROR,
                stage="asr_streaming",
                engine=self.engine_id,
                reason=REASON_STREAMING_UNSUPPORTED,
                detail="a streaming session is already active; call finish() first",
            )
        processor = self._backend["processor"]
        supported = list(
            getattr(processor, "supported_num_lookahead_tokens", [])
            or getattr(
                getattr(processor, "model_config", None), "supported_num_lookahead_tokens", []
            )
            or [0, 3, 6, 13]
        )
        if self._lookahead not in supported:
            raise AsrError(
                code=AsrErrorCode.ASR_INPUT_ERROR,
                stage="asr_streaming",
                engine=self.engine_id,
                reason=REASON_INVALID_AUDIO,
                detail=f"lookahead {self._lookahead} not in supported {supported}",
            )
        processor.set_num_lookahead_tokens(self._lookahead)
        latency_ms = getattr(processor, "streaming_latency_ms", None)
        logger.info(
            "Nemotron streaming session start (lookahead %d -> %s ms chunks, lang %s)",
            self._lookahead,
            latency_ms,
            self._language_arg(),
        )
        self._stream_q = queue.Queue()
        self._partial_text = ""
        self._partial_consumed = 0
        self._stream_error = None
        self._stream_started_at = time.perf_counter()
        # Prompt-side inputs from a silent first chunk (the audio tensor is
        # replaced by the real generator; input_ids carry the language prompt).
        first = processor(
            audio=np.zeros(int(processor.num_samples_first_audio_chunk), dtype=np.float32),
            sampling_rate=16000,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=self._language_arg(),
            return_tensors="pt",
        )
        device = self._backend["device"]
        first = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in first.items()}
        from transformers import TextIteratorStreamer  # type: ignore

        self._streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
        gen = self._chunk_generator()
        self._gen_thread = threading.Thread(
            target=self._generate_worker, args=(first, gen), daemon=True, name="vm-nemotron-gen"
        )
        self._partial_thread = threading.Thread(
            target=self._partial_worker, daemon=True, name="vm-nemotron-partial"
        )
        self._gen_thread.start()
        self._partial_thread.start()

    def feed(self, audio: AudioBuffer) -> Optional[AsrResult]:
        """Push audio into the stream; return the running partial if it grew.

        Returns None while the partial text is unchanged (no new tokens
        decoded). The partial is the model's OWN incremental decode — not a
        re-transcription of a buffer.
        """
        if self._stream_q is None:
            raise AsrError(
                code=AsrErrorCode.ASR_INPUT_ERROR,
                stage="asr_streaming",
                engine=self.engine_id,
                reason=REASON_STREAMING_UNSUPPORTED,
                detail="feed() called without start()",
            )
        samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        if samples.size:
            self._stream_q.put(samples.copy())
        if len(self._partial_text) > self._partial_consumed:
            self._partial_consumed = len(self._partial_text)
            return AsrResult(
                text=self._partial_text,
                engine_id=self.engine_id,
                model_id=self.model_id,
                status=AsrResultStatus.OK,
                partial=True,
            )
        return None

    def finish(self) -> AsrResult:
        """End the stream: flush the final chunk and collect the transcript."""
        if self._stream_q is None:
            return AsrResult.from_error(
                AsrError(
                    code=AsrErrorCode.ASR_INPUT_ERROR,
                    stage="asr_streaming",
                    engine=self.engine_id,
                    reason=REASON_STREAMING_UNSUPPORTED,
                    detail="finish() called without start()",
                ),
                self.engine_id,
                self.model_id,
            )
        self._stream_q.put(None)  # sentinel: pad + flush the final chunk
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=_STREAM_JOIN_TIMEOUT_S)
        if self._partial_thread is not None:
            self._partial_thread.join(timeout=5.0)
        self._stream_q = None
        total_ms = (time.perf_counter() - self._stream_started_at) * 1000.0
        if self._stream_error is not None:
            err = self._stream_error
            self._stream_error = None
            return AsrResult.from_error(err, self.engine_id, self.model_id)
        if self._gen_thread is not None and self._gen_thread.is_alive():
            return AsrResult.from_error(
                AsrError(
                    code=AsrErrorCode.ASR_INFERENCE_ERROR,
                    stage="asr_streaming",
                    engine=self.engine_id,
                    reason=REASON_INFERENCE_FAILED,
                    detail=f"streaming generate did not finish within "
                    f"{_STREAM_JOIN_TIMEOUT_S}s",
                ),
                self.engine_id,
                self.model_id,
            )
        text = self._partial_text.strip()
        self._partial_text = ""
        self._partial_consumed = 0
        return AsrResult(
            text=text,
            language="" if self._language_arg() == "auto" else self._language_arg(),
            engine_id=self.engine_id,
            model_id=self.model_id,
            status=AsrResultStatus.OK,
            inference_ms=total_ms,
        )

    # ------------------------------------------------------------ streaming internals

    def _generate_worker(self, first_inputs: dict, gen: Any) -> None:
        """Run generate() with the queue-backed chunk generator (worker thread)."""
        try:
            kwargs = dict(first_inputs)
            kwargs["input_features"] = gen
            kwargs["streamer"] = self._streamer
            with _BACKEND_LOCK:
                with torch.no_grad():
                    self._backend["model"].generate(**kwargs)
        except AsrError as exc:
            self._stream_error = exc
        except Exception as exc:  # noqa: BLE001
            self._stream_error = AsrError(
                code=AsrErrorCode.ASR_INFERENCE_ERROR,
                stage="asr_streaming",
                engine=self.engine_id,
                reason=REASON_INFERENCE_FAILED,
                detail=str(exc)[:500],
            )

    def _partial_worker(self) -> None:
        """Consume the text streamer: accumulate the running partial text."""
        try:
            for chunk in self._streamer:
                if chunk:
                    self._partial_text += chunk
        except Exception:  # noqa: BLE001 - the generate worker reports the error
            logger.debug("nemotron partial streamer ended", exc_info=True)

    def _chunk_generator(self):
        """Queue-backed mel-chunk generator with EXACT cache-aware sizes.

        Bridges the push-based feed() cadence (512-sample VAD frames) into
        the pull-based generate() loop: raw samples accumulate in a buffer;
        whenever a full cache-aware window is available it is featurised and
        yielded. The final (short) window is ZERO-PADDED to the required
        length, exactly like the model card's streaming example.
        """
        processor = self._backend["processor"]
        hop = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        first_samples = int(processor.num_samples_first_audio_chunk)
        next_samples = int(processor.num_samples_per_audio_chunk)
        first_mel = int(processor.num_mel_frames_first_audio_chunk)
        next_mel = int(processor.num_mel_frames_per_audio_chunk)
        device = self._backend["device"]

        buf = np.zeros(0, dtype=np.float32)
        mel_idx = 0
        is_first = True

        def _window_bounds() -> tuple[int, int, int]:
            """(start, length, mel_frames) of the next cache-aware window."""
            if is_first:
                return 0, first_samples, first_mel
            start = mel_idx * hop - n_fft // 2
            return start, next_samples, next_mel

        def _featurise(window: np.ndarray, first: bool) -> Any:
            inputs = processor(
                audio=window,
                sampling_rate=16000,
                is_streaming=True,
                is_first_audio_chunk=first,
                language=self._language_arg(),
                return_tensors="pt",
            )
            feats = inputs.input_features
            if first:
                # Official streaming example: the FIRST window's features
                # are trimmed to the exact cache-aware frame count (the FE
                # emits one extra edge frame for the lookback context).
                feats = feats[:, :first_mel, :]
            return feats.to(device)

        # NOTE: no history trimming — utterances are seconds long (a few MB
        # of float32) and hop-aligned trimming would drift the window math.
        while True:
            item = self._stream_q.get()
            if item is None:
                break
            buf = np.concatenate((buf, np.asarray(item, dtype=np.float32)))
            while True:
                start, length, mel_frames = _window_bounds()
                end = start + length
                if buf.size < end:
                    break
                window = buf[start:end]
                yield _featurise(window, is_first)
                mel_idx += mel_frames
                if is_first:
                    is_first = False
        # Sentinel: flush a padded final chunk when real content remains.
        start, length, mel_frames = _window_bounds()
        consumed_end = mel_idx * hop
        if buf.size > consumed_end:
            window = buf[start : start + length]
            if window.size < length:
                window = np.concatenate(
                    (window, np.zeros(length - window.size, dtype=np.float32))
                )
            yield _featurise(window, is_first)

    # ------------------------------------------------------------ internals

    def _resolve_model_dir(self) -> str:
        override = (self._config.asr_model_path or "").strip()
        if override:
            return override
        return str(self._spec.local_path(self._config))

    def _resolve_device(self) -> str:
        return (self._config.asr_device or "cpu").strip().lower()

    def _resolve_dtype(self) -> str:
        return (getattr(self._config, "asr_dtype", "") or "float32").strip().lower()
