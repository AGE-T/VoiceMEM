"""Parakeet TDT 0.6B v3 ASR engine adapter (transformers-native, official path).

TASK-A Phase 6B/9. ``nvidia/parakeet-tdt-0.6b-v3`` run through the
transformers 5.x ``ParakeetForTDT`` + ``AutoProcessor`` classes — the
OFFICIAL NVIDIA integration path for this checkpoint (no NeMo, no community
wrapper).

Engine facts (verified on the real corpus, sandbox CPU):

* NON-STREAMING by design: one greedy transducer (TDT) decode over the full
  utterance. Chunked batch inference over a growing buffer is NOT advertised
  as streaming (capability ``streaming=False``); calling start/feed/finish
  raises :class:`AsrError` — a programming error, not a silent no-op.
* Multilingual (25 European languages incl. Hungarian). Measured: real
  Windows field capture 6.2% WER, piper-HU corpus 0-14% WER, English 12%
  (tokenization merge) — see data/asr_bench/ and the benchmark harness.
* No language ID, no timestamps, no confidence in the v3 decode path.
* Devices: cpu and cuda are both valid EXECUTION modes of the same model
  (task device rule). ``ASR_DEVICE=cuda`` without CUDA available fails
  EXPLICITLY (reason ``cuda_unavailable``) — there is no silent device
  fallback.
* Thread safety: the model is loaded ONCE per (model_dir, device) in a
  module-level cache and every inference runs under a module lock —
  generate() is not safe for unsynchronised concurrent calls (v0.4.9
  report issue #2, kept for the new engine layer).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.asr_core import (
    AsrCapability,
    AsrEngineProtocol,
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
)
from app.config import AgentConfig

logger = logging.getLogger(__name__)

try:  # pragma: no cover — sandbox HAS torch; target machine too; CI may not
    import torch  # type: ignore[import-untyped]

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

#: Shared backend cache: ONE loaded model per (model_dir, device, dtype).
_BACKEND_CACHE: "dict[tuple, Any]" = {}
#: Guards the cache AND serialises every generate() call (thread safety).
_BACKEND_LOCK = threading.Lock()


def _parakeet_classes() -> tuple[Any, Any]:
    """Import the transformers classes lazily with an actionable hint."""
    if not _HAS_TORCH:
        raise AsrError(
            code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
            stage="engine_load",
            engine="parakeet",
            reason=REASON_DEPENDENCY_MISSING,
            detail="torch is not installed (pip install torch)",
        )
    try:
        from transformers import AutoProcessor, ParakeetForTDT  # type: ignore
    except ImportError as exc:
        raise AsrError(
            code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
            stage="engine_load",
            engine="parakeet",
            reason=REASON_DEPENDENCY_MISSING,
            detail=f"transformers with ParakeetForTDT is required (transformers>=5.6): {exc}",
        ) from exc
    return AutoProcessor, ParakeetForTDT


def _shared_backend(model_dir: str, device: str, dtype: str) -> Any:
    """Load (or reuse) the ONE model backend for (model_dir, device, dtype)."""
    key = (model_dir, device, dtype)
    with _BACKEND_LOCK:
        cached = _BACKEND_CACHE.get(key)
        if cached is not None:
            return cached
        AutoProcessor, ParakeetForTDT = _parakeet_classes()
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
            model = ParakeetForTDT.from_pretrained(source, torch_dtype=torch_dtype)
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise AsrError(
                code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
                stage="engine_load",
                engine="parakeet",
                reason=REASON_MODEL_UNAVAILABLE,
                detail=f"loading {source} failed: {exc}",
                diagnostics={"model_dir": str(model_dir), "device": device},
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise AsrError(
                code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
                stage="engine_load",
                engine="parakeet",
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
                engine="parakeet",
                reason=REASON_MODEL_UNAVAILABLE,
                detail=f"moving model to {device} failed: {exc}",
            ) from exc
        model = model.eval()
        load_ms = (time.perf_counter() - t0) * 1000.0
        backend = {"processor": processor, "model": model, "device": device}
        _BACKEND_CACHE[key] = backend
        logger.info(
            "Parakeet v3 backend loaded from %s on %s (%s) in %.0f ms",
            source,
            device,
            dtype,
            load_ms,
        )
        return backend


class ParakeetEngine:
    """ASR engine adapter for nvidia/parakeet-tdt-0.6b-v3 (non-streaming)."""

    engine_id = "parakeet"
    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    capabilities = AsrCapability(
        streaming=False,
        language_detection=False,
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
        self._probe_ok = False

    # ------------------------------------------------------------ lifecycle

    def load(self) -> None:
        """Load the model (explicit; also happens lazily on first use)."""
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
        device = self._resolve_device()
        dtype = self._resolve_dtype()
        try:
            self._backend = _shared_backend(model_dir, device, dtype)
        except AsrError as exc:
            self._load_error = exc.detail
            raise

    def unload(self) -> None:
        """Drop this engine's reference (the shared backend stays cached)."""
        self._backend = None
        self._load_error = None
        self._probe_ok = False

    def is_loaded(self) -> bool:
        return self._backend is not None

    def is_available(self) -> bool:
        """True when the dependency stack is importable at all."""
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
                "multilingual": self.capabilities.multilingual,
                "language_detection": self.capabilities.language_detection,
                "timestamps": self.capabilities.timestamps,
            },
            "last_inference_ms": round(self._last_inference_ms, 1),
            "last_error": self._load_error or "",
        }

    def warm_up(self) -> bool:
        """Load + verify one REAL inference (server warm-up contract)."""
        try:
            self.load()
            self._probe()
            return True
        except Exception as exc:  # noqa: BLE001 - warm-up must never kill the server
            logger.warning("Parakeet warm-up failed: %s", exc)
            self._load_error = str(exc)
            return False

    # ------------------------------------------------------------ transcription

    def transcribe(self, audio: AudioBuffer) -> AsrResult:
        """One-shot full-utterance transcription -> :class:`AsrResult`.

        Errors are EXPLICIT (:class:`AsrError` mapped into the result) —
        never an empty-success, never a partial-join fallback.
        """
        try:
            self.load()
        except AsrError as exc:
            return AsrResult.from_error(exc, self.engine_id, self.model_id)
        # Input contract check at the engine boundary.
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
                duration_s=0.0,
                status=AsrResultStatus.OK,
            )
        t0 = time.perf_counter()
        try:
            with _BACKEND_LOCK:
                inputs = self._backend["processor"](
                    audio=np.asarray(audio.samples, dtype=np.float32),
                    sampling_rate=audio.sample_rate,
                    return_tensors="pt",
                )
                inputs = {
                    k: (v.to(self._backend["device"]) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }
                with torch.no_grad():
                    out = self._backend["model"].generate(
                        **inputs, max_new_tokens=int(self._config.asr_max_new_tokens)
                    )
        except AsrError as exc:
            return AsrResult.from_error(exc, self.engine_id, self.model_id)
        except Exception as exc:  # noqa: BLE001 - inference failure mapping
            return AsrResult.from_error(
                AsrError(
                    code=AsrErrorCode.ASR_INFERENCE_ERROR,
                    stage="asr_inference",
                    engine=self.engine_id,
                    reason=REASON_INFERENCE_FAILED,
                    detail=str(exc)[:500],
                    diagnostics={"audio_s": round(audio.duration_s, 3)},
                ),
                self.engine_id,
                self.model_id,
            )
        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
        try:
            sequences = out.sequences if hasattr(out, "sequences") else out
            ids = sequences[0].tolist() if hasattr(sequences, "tolist") else list(sequences[0])
            text = self._backend["processor"].tokenizer.decode(
                ids, skip_special_tokens=True
            ).strip()
        except Exception as exc:  # noqa: BLE001 - decode failure mapping
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
        return AsrResult(
            text=text,
            language="",
            duration_s=audio.duration_s,
            engine_id=self.engine_id,
            model_id=self.model_id,
            status=AsrResultStatus.OK,
            inference_ms=self._last_inference_ms,
        )

    # ------------------------------------------------------------ streaming: NOT supported

    def start(self) -> None:
        """Non-streaming engine: feeding is a contract violation, not a no-op."""
        raise AsrError(
            code=AsrErrorCode.ASR_INPUT_ERROR,
            stage="asr_streaming",
            engine=self.engine_id,
            reason=REASON_DEPENDENCY_MISSING,
            detail="parakeet does not support streaming (capabilities.streaming=False); "
            "use transcribe() on the completed utterance",
        )

    def feed(self, audio: AudioBuffer) -> "AsrResult | None":  # pragma: no cover
        self.start()

    def finish(self) -> AsrResult:  # pragma: no cover
        self.start()
        raise AssertionError("unreachable")

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

    def _probe(self) -> bool:
        """One REAL inference on 1 s of silence proves the whole call path."""
        silence = AudioBuffer(
            samples=np.zeros(16000, dtype=np.float32), sample_rate=16000
        )
        result = self.transcribe(silence)
        if result.status != AsrResultStatus.OK:
            raise AsrError(
                code=AsrErrorCode.ASR_INFERENCE_ERROR,
                stage="engine_load",
                engine=self.engine_id,
                reason=REASON_INFERENCE_FAILED,
                detail=f"warm-up probe failed: {result.error.detail if result.error else 'unknown'}",
            )
        self._probe_ok = True
        return True


# The engine satisfies the protocol structurally (runtime_checkable).
_: AsrEngineProtocol = ParakeetEngine  # type: ignore[assignment]
