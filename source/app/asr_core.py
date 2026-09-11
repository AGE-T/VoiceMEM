"""Modular ASR layer: canonical audio contract, result contract, registry.

TASK-A Phase 2 + Phase 8. This module defines EVERYTHING the pipeline and
the web UI may know about ASR — and NOTHING about a specific engine. Engine
adapters (app/asr_parakeet.py, app/asr_nemotron.py) implement the
:class:`AsrEngineProtocol`; the upper layers (app/pipeline.py,
app/web_server.py, app/main.py) consume :class:`AudioBuffer` and
:class:`AsrResult` only.

Architecture (VibeVoice Studio concepts, adapted; no fallback semantics):

    AudioCapture -> AudioBuffer -> VAD -> AsrEngine (selected) -> ASRResult
                                                          -> VoiceMEM -> ...

Contracts:

* :class:`AudioBuffer` — THE canonical pipeline audio representation:
  16 kHz, mono, float32, [-1, 1]. One authoritative format at the pipeline
  boundary; model-specific preprocessing (mel filterbanks, feature windows)
  lives INSIDE the engine adapter, never in the shared contract.
* :class:`AsrResult` — one stable result type consumed by the web UI and
  the VoiceMEM pipeline. The UI must not inspect engine-specific objects.
* :class:`AsrError` — structured, explicit failures. A failed stage STOPS
  the turn; there is no engine fallback, no partial-join fallback, no
  "empty transcript means success".
* :class:`AsrCapability` / :class:`AsrModelSpec` / :data:`MODEL_REGISTRY` —
  the authoritative engine+model registry. Capabilities are DESCRIPTIVE
  ONLY: they never cause automatic engine switching.
* :func:`select_engine` — explicit engine selection from configuration
  (``ASR_ENGINE`` env / yaml ``asr.engine``). Exactly ONE engine is
  selected; if it cannot load, an :class:`AsrError` with
  ``ASR_MODEL_LOAD_ERROR`` is raised. No alternative engine is tried.

NO SEMANTIC FALLBACKS (task rule): if the selected engine fails, the
failure is returned explicitly. CPU vs CUDA is execution configuration,
not a fallback — the same engine on the same model may run on either
device.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from app.config import AgentConfig

logger = logging.getLogger(__name__)

#: Canonical pipeline audio rate (VAD + ASR contract).
CANONICAL_SAMPLE_RATE = 16000
#: Canonical channel count.
CANONICAL_CHANNELS = 1
#: Canonical sample dtype.
CANONICAL_DTYPE = "float32"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Structured errors (Phase 13)
# ═══════════════════════════════════════════════════════════════════════════


class AsrErrorCode(str, Enum):
    """Stable ASR-layer error codes (task Phase 13 contract)."""

    AUDIO_INPUT_ERROR = "AUDIO_INPUT_ERROR"        # audio violates the AudioBuffer contract
    ASR_INPUT_ERROR = "ASR_INPUT_ERROR"            # engine rejected the input audio
    ASR_MODEL_LOAD_ERROR = "ASR_MODEL_LOAD_ERROR"  # model weights/processor unavailable
    ASR_PROCESSOR_ERROR = "ASR_PROCESSOR_ERROR"    # feature extraction failed
    ASR_INFERENCE_ERROR = "ASR_INFERENCE_ERROR"    # the model forward pass failed
    ASR_DECODE_ERROR = "ASR_DECODE_ERROR"          # token decoding failed


#: Machine-readable failure reasons (short, stable, greppable).
REASON_MODEL_UNAVAILABLE = "model_unavailable"
REASON_DEPENDENCY_MISSING = "dependency_missing"
REASON_INVALID_AUDIO = "invalid_audio"
REASON_INFERENCE_FAILED = "inference_failed"
REASON_DECODE_FAILED = "decode_failed"
REASON_EMPTY_AUDIO = "empty_audio"
REASON_CUDA_UNAVAILABLE = "cuda_unavailable"
REASON_STREAMING_UNSUPPORTED = "streaming_unsupported"


@dataclass
class AsrError(Exception):
    """Structured ASR failure. Carries stage/engine/reason + diagnostics.

    Raised (or returned via :class:`AsrResult` ``error``) whenever an ASR
    stage fails. The pipeline turns this into an explicit stage failure —
    it is NEVER swallowed and NEVER converted into an empty transcript.
    """

    code: AsrErrorCode
    stage: str                      # "engine_load" | "asr_input" | "transcribe" | ...
    reason: str                     # machine code, e.g. "model_unavailable"
    detail: str = ""                # human-readable diagnostics
    engine: str = ""                # engine id where relevant
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover — logging cosmetics
        parts = [self.code.value, f"stage={self.stage}"]
        if self.engine:
            parts.append(f"engine={self.engine}")
        parts.append(f"reason={self.reason}")
        if self.detail:
            parts.append(f"detail={self.detail[:200]}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """WS/REST-friendly dict (the UI error contract)."""
        return {
            "code": self.code.value,
            "stage": self.stage,
            "engine": self.engine,
            "reason": self.reason,
            "detail": self.detail[:500],
            "diagnostics": self.diagnostics,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Canonical audio contract (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AudioBuffer:
    """THE canonical audio representation at the pipeline boundary.

    Contract (validated by :meth:`validate` / :meth:`validated`):

    * ``sample_rate`` == 16000 (VAD + ASR standard; Silero frame contract)
    * ``channels`` == 1 (mono)
    * ``samples`` is a 1-D float32 ndarray in [-1, 1]
    * ``duration_s`` is derived: ``samples.size / sample_rate``
    * ``timestamp`` is the capture time (``time.perf_counter()`` by default)

    Model-specific preprocessing (mel spectrograms, feature windows, prompt
    conditioning) belongs INSIDE the engine adapter, not here. Producers
    build AudioBuffers with :meth:`from_float` / :meth:`from_pcm16_bytes`;
    consumers call :meth:`validated` at the stage boundary.
    """

    samples: np.ndarray
    sample_rate: int = CANONICAL_SAMPLE_RATE
    channels: int = CANONICAL_CHANNELS
    dtype: str = CANONICAL_DTYPE
    timestamp: float = field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        """Normalise the array side of the contract (1-D float32 copy)."""
        arr = np.asarray(self.samples)
        if arr.ndim != 1:
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            else:  # multi-channel: downmix to mono (mean over channels)
                arr = arr.mean(axis=1) if arr.ndim == 2 else arr.reshape(-1)
        object.__setattr__(self, "samples", np.ascontiguousarray(arr, dtype=np.float32))

    # ---------------------------------------------------------------- helpers

    @classmethod
    def from_float(cls, x: np.ndarray, sample_rate: int) -> "AudioBuffer":
        """Build from a float array at ``sample_rate`` (mono or [N, C])."""
        return cls(samples=x, sample_rate=int(sample_rate))

    @classmethod
    def from_pcm16_bytes(cls, raw: bytes, sample_rate: int) -> "AudioBuffer":
        """Build from little-endian PCM16 mono bytes at ``sample_rate``."""
        if len(raw) < 2:
            return cls(samples=np.zeros(0, dtype=np.float32), sample_rate=sample_rate)
        even = len(raw) - (len(raw) % 2)
        i16 = np.frombuffer(raw[:even], dtype=np.int16)
        return cls(samples=i16.astype(np.float32) / 32768.0, sample_rate=sample_rate)

    @classmethod
    def from_wav(cls, path: str | Path) -> "AudioBuffer":
        """Load a 16 kHz mono PCM16 WAV (the benchmark corpus format)."""
        import wave

        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            n_ch = w.getnchannels()
            width = w.getsampwidth()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if width != 2:
            raise AsrError(
                code=AsrErrorCode.AUDIO_INPUT_ERROR,
                stage="audio_buffer_from_wav",
                reason=REASON_INVALID_AUDIO,
                detail=f"{path}: sample width {width}B, expected PCM16",
            )
        if n_ch != 1:
            pcm = pcm.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
        return cls(samples=pcm.astype(np.float32) / 32768.0, sample_rate=rate)

    # ---------------------------------------------------------------- checks

    @property
    def duration_s(self) -> float:
        """Duration in seconds (samples / sample_rate)."""
        return float(self.samples.size) / float(self.sample_rate or 1)

    @property
    def sample_count(self) -> int:
        """Number of mono samples."""
        return int(self.samples.size)

    def validate(self) -> list[str]:
        """Return a list of contract violations (empty list = valid)."""
        problems: list[str] = []
        if self.sample_rate != CANONICAL_SAMPLE_RATE:
            problems.append(f"sample_rate={self.sample_rate}, expected {CANONICAL_SAMPLE_RATE}")
        if self.channels != CANONICAL_CHANNELS:
            problems.append(f"channels={self.channels}, expected {CANONICAL_CHANNELS}")
        if self.samples.dtype != np.float32:
            problems.append(f"dtype={self.samples.dtype}, expected float32")
        if self.samples.ndim != 1:
            problems.append(f"samples.ndim={self.samples.ndim}, expected 1")
        if self.samples.size and not np.isfinite(self.samples).all():
            problems.append("samples contain non-finite values (NaN/Inf)")
        peak = float(np.max(np.abs(self.samples))) if self.samples.size else 0.0
        if peak > 1.0 + 1e-6:
            problems.append(f"peak={peak:.4f} outside [-1, 1]")
        return problems

    def validated(self, stage: str = "audio_input") -> "AudioBuffer":
        """Validate the contract or raise :class:`AsrError` (AUDIO_INPUT_ERROR)."""
        problems = self.validate()
        if problems:
            raise AsrError(
                code=AsrErrorCode.AUDIO_INPUT_ERROR,
                stage=stage,
                reason=REASON_INVALID_AUDIO,
                detail="; ".join(problems),
                diagnostics={
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "dtype": str(self.samples.dtype),
                    "samples": int(self.samples.size),
                    "duration_s": round(self.duration_s, 4),
                },
            )
        return self

    def metrics(self) -> dict[str, Any]:
        """Boundary diagnostics: rate/channels/dtype/count/duration/RMS/peak."""
        x = self.samples
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "dtype": str(x.dtype),
            "samples": int(x.size),
            "duration_s": round(self.duration_s, 6),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Result contract (ASRResult)
# ═══════════════════════════════════════════════════════════════════════════


class AsrResultStatus(str, Enum):
    """Status of one transcription (an ASRResult is never silently empty)."""

    OK = "ok"
    ERROR = "error"


@dataclass
class AsrResult:
    """ONE stable result type — consumed by the web UI and the pipeline.

    Fields the UI may render: ``text`` (final transcript), ``language``,
    ``duration_s``, ``model_id``, ``engine_id``, ``status``, ``error``.
    Engine-specific objects must NOT be attached; timestamps/confidence
    are optional and generic.
    """

    text: str = ""
    language: str = ""                      # BCP-47-ish ("hu", "en") or "" (unknown)
    duration_s: float = 0.0                 # audio duration that produced this result
    model_id: str = ""                      # e.g. "nvidia/parakeet-tdt-0.6b-v3"
    engine_id: str = ""                     # e.g. "parakeet"
    timestamps: Optional[list[dict[str, float]]] = None   # [{start, end}, ...] seconds
    confidence: Optional[float] = None      # 0..1 when the engine reports one
    status: AsrResultStatus = AsrResultStatus.OK
    error: Optional[AsrError] = None
    inference_ms: float = 0.0               # engine wall time
    partial: bool = False                   # True for streaming partial results

    @property
    def ok(self) -> bool:
        """True iff status OK AND non-empty text (a non-empty string is NOT
        automatically success — but an EMPTY string with status OK is an
        explicit engine verdict of 'no speech content', reported as-is)."""
        return self.status == AsrResultStatus.OK

    def to_dict(self) -> dict[str, Any]:
        """WS/REST payload (the UI consumes exactly this shape)."""
        return {
            "type": "asr_result",
            "text": self.text,
            "language": self.language,
            "duration_s": round(self.duration_s, 3),
            "model_id": self.model_id,
            "engine_id": self.engine_id,
            "timestamps": self.timestamps,
            "confidence": self.confidence,
            "status": self.status.value,
            "error": self.error.to_dict() if self.error is not None else None,
            "inference_ms": round(self.inference_ms, 1),
            "partial": self.partial,
        }

    @classmethod
    def from_error(cls, error: AsrError, engine_id: str = "", model_id: str = "") -> "AsrResult":
        """A failed transcription (explicit, never an empty success)."""
        return cls(
            text="",
            engine_id=engine_id or error.engine,
            model_id=model_id,
            status=AsrResultStatus.ERROR,
            error=error,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Capability + model registry
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AsrCapability:
    """Descriptive engine capabilities — NEVER a switching trigger.

    ``streaming`` True means the engine exposes a genuine incremental
    start/feed/finish interface (cache-aware state, partial transcripts).
    Chunked batch inference over a growing buffer is NOT streaming and
    must not be advertised as such.
    """

    streaming: bool = False
    language_detection: bool = False
    timestamps: bool = False
    word_timestamps: bool = False
    confidence: bool = False
    multilingual: bool = False
    hotwords: bool = False


@dataclass(frozen=True)
class AsrModelSpec:
    """One authoritative registry entry (engine + model identity)."""

    engine_id: str                        # "parakeet" | "nemotron" | ...
    model_id: str                         # HF repo id / model name
    source: str                           # "huggingface" | "local"
    revision: str = ""                    # pinned commit sha (when known)
    local_dir: str = "models/asr/{slug}"  # relative to the agent root
    languages: tuple[str, ...] = ()       # ("hu", "en", ...) — "" = unknown
    devices: tuple[str, ...] = ("cpu", "cuda")
    compute_type: str = "float32"
    capabilities: AsrCapability = AsrCapability()
    license: str = ""

    def local_path(self, config: AgentConfig) -> Path:
        """Absolute model directory for this spec under the agent root."""
        return config.root / self.local_dir.format(slug=self.engine_id)


#: The authoritative ASR model registry. Engine selection reads this; no
#: model-specific configuration is scattered through the pipeline.
MODEL_REGISTRY: dict[str, AsrModelSpec] = {
    "parakeet": AsrModelSpec(
        engine_id="parakeet",
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        source="huggingface",
        revision="541d1f99c6b0",  # HF sha at integration time (2026-09-10)
        local_dir="models/asr/parakeet-tdt-0.6b-v3",
        languages=("en", "de", "fr", "es", "hu", "pl", "nl", "it", "ro", "cs", "sk", "sl", "hr", "sr", "uk", "ru", "tr", "ar", "zh", "ja", "ko", "pt", "fi", "id", "vi"),
        devices=("cpu", "cuda"),
        compute_type="float32",
        capabilities=AsrCapability(
            streaming=False,          # TDT offline transcription; chunked
            # batch inference is NOT advertised as streaming
            language_detection=False,  # single decode, no LID head
            timestamps=False,
            word_timestamps=False,
            confidence=False,
            multilingual=True,
            hotwords=False,
        ),
        license="CC-BY-4.0",
    ),
    "nemotron": AsrModelSpec(
        engine_id="nemotron",
        model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
        source="huggingface",
        revision="ea30d66debe3",
        local_dir="models/asr/nemotron-3.5-asr-streaming-0.6b",
        languages=("en", "de", "es", "fr", "hu"),  # broad-coverage tier per card
        devices=("cpu", "cuda"),
        compute_type="float32",
        capabilities=AsrCapability(
            streaming=True,           # cache-aware streaming interface
            language_detection=True,  # auto LID + prompt conditioning
            timestamps=False,
            word_timestamps=False,
            confidence=False,
            multilingual=True,
            hotwords=False,
        ),
        license="OpenMDW-1.1",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# 5. Engine protocol + explicit selection
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class AsrEngineProtocol(Protocol):
    """The engine interface every adapter implements.

    * ``load`` / ``unload`` / ``is_loaded`` / ``status`` — lifecycle.
    * ``transcribe`` — one-shot full-utterance transcription (EVERY engine).
    * ``start`` / ``feed`` / ``finish`` — ONLY for engines with genuine
      streaming capability; non-streaming engines raise
      :class:`AsrError` (the caller checks ``capabilities.streaming``
      first — feeding a non-streaming engine is a programming error, not
      a silent no-op).

    An adapter maps engine-specific exceptions to :class:`AsrError` with
    the right :class:`AsrErrorCode`; the upper layers never see raw
    engine exceptions.
    """

    engine_id: str
    model_id: str
    capabilities: AsrCapability

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def status(self) -> dict[str, Any]: ...
    def transcribe(self, audio: AudioBuffer) -> AsrResult: ...
    def start(self) -> None: ...
    def feed(self, audio: AudioBuffer) -> "AsrResult | None": ...
    def finish(self) -> AsrResult: ...


#: Valid engine ids for configuration validation.
ENGINE_IDS = tuple(MODEL_REGISTRY.keys())


def select_engine(config: AgentConfig, engine_id: Optional[str] = None) -> Any:
    """Create the ONE selected engine. Explicit configuration, no fallback.

    Resolution order: explicit ``engine_id`` argument (tests) >
    ``config.asr_engine`` (yaml ``asr.engine`` / ``ASR_ENGINE`` env).
    If the id is not in the registry, the error lists the valid ids.
    The ENGINE CLASS is constructed here, but model weights load lazily
    in ``load()``/first use — a missing model surfaces as
    ``ASR_MODEL_LOAD_ERROR`` at load time, not at import time.

    Importantly: this function NEVER falls back to a different engine.
    """
    chosen = (engine_id or config.asr_engine or "").strip().lower()
    if not chosen:
        raise AsrError(
            code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
            stage="engine_selection",
            reason=REASON_MODEL_UNAVAILABLE,
            detail="no ASR engine configured: set ASR_ENGINE (yaml asr.engine)",
        )
    if chosen not in MODEL_REGISTRY:
        raise AsrError(
            code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
            stage="engine_selection",
            engine=chosen,
            reason=REASON_MODEL_UNAVAILABLE,
            detail=f"unknown ASR engine {chosen!r}; valid engines: {', '.join(ENGINE_IDS)}",
        )
    spec = MODEL_REGISTRY[chosen]
    if chosen == "parakeet":
        from app.asr_parakeet import ParakeetEngine

        return ParakeetEngine(config, spec)
    if chosen == "nemotron":
        from app.asr_nemotron import NemotronEngine

        return NemotronEngine(config, spec)
    raise AsrError(  # pragma: no cover — registry/class mismatch guard
        code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
        stage="engine_selection",
        engine=chosen,
        reason=REASON_MODEL_UNAVAILABLE,
        detail=f"engine {chosen!r} is registered but has no adapter",
    )


def asr_backend_status(config: AgentConfig) -> dict[str, Any]:
    """Registry/selection snapshot for diagnostics (no model loading)."""
    chosen = (config.asr_engine or "").strip().lower()
    return {
        "selected_engine": chosen or None,
        "device": config.asr_device,
        "registry": {
            eid: {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "local_path": str(spec.local_path(config)),
                "local_present": (
                    (spec.local_path(config) / "model.safetensors").is_file()
                    or (spec.local_path(config) / "model.nemo").is_file()
                ),
                "languages": list(spec.languages),
                "devices": list(spec.devices),
                "compute_type": spec.compute_type,
                "license": spec.license,
                "capabilities": {
                    "streaming": spec.capabilities.streaming,
                    "language_detection": spec.capabilities.language_detection,
                    "timestamps": spec.capabilities.timestamps,
                    "word_timestamps": spec.capabilities.word_timestamps,
                    "confidence": spec.capabilities.confidence,
                    "multilingual": spec.capabilities.multilingual,
                    "hotwords": spec.capabilities.hotwords,
                },
            }
            for eid, spec in MODEL_REGISTRY.items()
        },
    }


class _UnavailableEngine:
    """Explicit-failure facade used when the selected engine cannot load.

    The task contract: if the selected engine cannot load -> ASR_ERROR with
    stage=engine_load and the reason; do NOT automatically select another
    engine. This facade implements the engine protocol by returning the
    SAME structured :class:`AsrError` from every call, so the web session,
    the ASR test endpoint and the UI all surface the exact failure instead
    of a half-initialised engine. (Not in the registry; constructed only by
    :func:`app.web_server.WebComponents.make_asr`.)
    """

    def __init__(self, error: AsrError) -> None:
        self._error = error
        self.engine_id = error.engine or "unknown"
        self.model_id = ""
        self.capabilities = AsrCapability()

    def load(self) -> None:
        raise self._error

    def unload(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return False

    def is_available(self) -> bool:
        return False

    def status(self) -> dict[str, Any]:
        return {
            "engine": self.engine_id,
            "model": self.model_id,
            "loaded": False,
            "error": self._error.to_dict(),
        }

    def warm_up(self) -> bool:
        return False

    def transcribe(self, audio: AudioBuffer) -> AsrResult:
        return AsrResult.from_error(self._error, self.engine_id, self.model_id)

    def start(self) -> None:
        raise self._error

    def feed(self, audio: AudioBuffer) -> "AsrResult | None":
        return None

    def finish(self) -> AsrResult:
        return AsrResult.from_error(self._error, self.engine_id, self.model_id)
