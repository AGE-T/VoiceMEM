"""Supertonic 3 TTS engine (ONNX Runtime, CPU) — v0.7.0 production TTS.

Replaces the Piper subprocess wrapper (v0.4.x–v0.6.x) as the ONE production
TTS engine. Supertonic 3 is the official Supertone open-weight model:

  * model   : supertone-oss-archive/supertonic-3 @ aafc6e3241 (HF archive,
              pinned revision; OpenRAIL-M licence, MIT sample code)
  * runtime: ``supertonic==1.3.1`` PyPI SDK → ONNX Runtime CPUExecutionProvider
              (99M params, 44.1 kHz 16-bit mono output, 31 languages incl. hu/en)
  * assets : models/tts/supertonic-3/ (onnx/*.onnx + voice_styles/*.json,
              downloaded ONCE at setup time by scripts/download_models.ps1
              from the MODELS.lock.json entry; hashes in ASSET_MANIFEST.json)

NO PIPER FALLBACK — explicit error contract (task order v0.7.0):
    Supertonic 3 SUCCESS -> audio
    Supertonic 3 FAILURE -> logged TTS error + False/None return
                             (the web layer reports "TTS: no audio")

Interface parity with the retired Piper wrapper (drop-in for the pipeline /
web layer, both call sites unchanged):
    synthesize_to_file(text, language, out_path, length_scale, voice) -> bool
    synthesize(text, language, length_scale, voice) -> Optional[np.ndarray]
        int16 mono at config.output_sample_rate (44100, engine-native)
    stop()            — barge-in: sets the abandon flag (in-process ONNX
                        cannot be killed mid-call; the result of an
                        abandoned synthesis is discarded, playback stops)
    is_available()    — asset presence (no SDK import, no network)
    list_voices() / voice_exists(voice) / last_voice / last_language

LIFECYCLE: the ONNX sessions are loaded ONCE (lazily, on first synthesis,
under a lock) and kept alive for the process lifetime — never once per
utterance. ``get_voice_style`` objects are cached per preset name.

M2 emotion pacing: Piper's ``--length_scale`` (1.1 = 10% slower) maps to
Supertonic's ``speed`` (0.7–2.0, higher = faster) as ``speed = 1/length_scale``
(clamped to the SDK's valid band). ``length_scale=None`` keeps the configured
default (official recommendation 1.05).

EDGE-SILENCE TRIM (documented boundary conversion): the Supertonic vocoder
pads every utterance with ~0.35–0.9 s of near-silence at both edges
(measured, see data/supertonic_validation/report.json). For chunked
conversational replies the padding would add ~1.5 s of dead air per chunk,
so ``synthesize``/``synthesize_to_file`` trim edges below an RMS floor with
a 100 ms guard when ``supertonic_trim_silence`` is on (default). The
44.1 kHz → 24 kHz resampling for the browser happens at the EXISTING web
boundary (web_server._speak resample_linear), unchanged.
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import AgentConfig
from .text_utils import LANG_EN, LANG_HU

logger = logging.getLogger(__name__)

#: Engine id surfaced in logs/status ("TTS engine = Supertonic 3").
ENGINE_ID = "supertonic3"

#: Supertonic SDK synthesis-parameter band (supertonic/config.py).
SPEED_MIN, SPEED_MAX = 0.7, 2.0
STEPS_MIN, STEPS_MAX = 1, 100

#: App language code -> Supertonic ISO code. Unknown app languages keep the
#: Hungarian behaviour (parity with the Piper "unknown -> HU voice" rule).
_LANG_MAP = {LANG_HU: "hu", LANG_EN: "en"}
_DEFAULT_LANG = "hu"

#: Edge-silence trim: frames (50 ms) with RMS below this are candidates.
_TRIM_RMS_FLOOR = 0.01
#: Guard kept at each edge after trimming (seconds).
_TRIM_GUARD_S = 0.1
#: Guard as frames (computed at the actual sample rate).
_TRIM_FRAME_S = 0.05

#: Asset layout inside the model dir (supertonic loader contract).
_REQUIRED_ONNX = (
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
)


class SupertonicTtsError(RuntimeError):
    """Explicit Supertonic TTS failure (no fallback engine exists)."""


class SupertonicTtsEngine:
    """ONNX-Runtime (CPU) Supertonic 3 engine, Piper-interface compatible.

    Voice per language: preset ids (``F1``–``F5``, ``M1``–``M5``) resolved
    against ``<model_dir>/voice_styles/<id>.json``; the explicit ``voice``
    argument overrides the per-language default, an unknown explicit voice
    logs loudly and falls back to the default (UI selector contract, same
    rule as the Piper wrapper). Every preset speaks both hu and en — the
    per-language fields exist for the UI's HU/EN selectors.
    """

    engine_id = ENGINE_ID

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._tts: Any = None
        self._styles: dict[str, Any] = {}
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()
        self._stop_requested = False
        #: Voice actually used by the last synthesis call (status display).
        self.last_voice: str = ""
        self.last_language: str = ""
        #: Load timing surfaced by the health/status layer.
        self.load_s: Optional[float] = None

    # -- model dir / voice resolution ---------------------------------------- #

    @property
    def model_dir(self) -> Path:
        return self._config.supertonic_model_dir

    def _voice_name(self, language: str, voice: Optional[str] = None) -> str:
        """Resolve the preset id (explicit *voice* wins; unknown -> default)."""
        voice_name = str(voice or "").strip()
        if voice_name:
            if self.voice_exists(voice_name):
                return voice_name
            logger.warning(
                "requested voice %r is not installed (%s) - using the %s default",
                voice_name,
                self.model_dir / "voice_styles" / f"{voice_name}.json",
                language,
            )
        if language == LANG_EN:
            return str(self._config.tts_en_voice).strip() or "F1"
        return str(self._config.tts_hu_voice).strip() or "F1"

    def voice_exists(self, voice: str) -> bool:
        """True when the voice-style JSON for *voice* is installed."""
        name = str(voice or "").strip()
        return bool(name) and (self.model_dir / "voice_styles" / f"{name}.json").is_file()

    def list_voices(self) -> list[str]:
        """Preset ids available in the model dir (``voice_styles/*.json`` stems)."""
        styles_dir = self.model_dir / "voice_styles"
        if not styles_dir.is_dir():
            return []
        return sorted(path.stem for path in styles_dir.glob("*.json"))

    def is_available(self) -> bool:
        """True when all ONNX modules AND both default voice styles exist.

        Pure asset-presence check: no SDK import, no model load, no network.
        """
        model_dir = self.model_dir
        if not all((model_dir / rel).is_file() for rel in _REQUIRED_ONNX):
            return False
        return self.voice_exists(str(self._config.tts_hu_voice).strip()) and self.voice_exists(
            str(self._config.tts_en_voice).strip()
        )

    # -- lifecycle ------------------------------------------------------------ #

    def _ensure_engine(self) -> Any:
        """Load the SDK + ONNX sessions ONCE; reuse forever after.

        Raises SupertonicTtsError on failure (explicit, never falls back).
        """
        if self._tts is not None:
            return self._tts
        with self._lock:
            if self._tts is not None:
                return self._tts
            if self._load_error is not None:
                # A previous load failed hard: report the same explicit error
                # on every call (no silent retry storm, no fallback engine).
                raise SupertonicTtsError(self._load_error)
            if not self.is_available():
                missing = [
                    rel
                    for rel in _REQUIRED_ONNX
                    if not (self.model_dir / rel).is_file()
                ]
                detail = (
                    f"Supertonic 3 model assets incomplete in {self.model_dir}"
                    f" (missing: {', '.join(missing) if missing else 'voice styles'})"
                    "; run scripts/download_models.ps1 (setup-time only)"
                )
                self._load_error = detail
                raise SupertonicTtsError(detail)
            try:
                t0 = time.perf_counter()
                from supertonic import TTS as _SdkTTS  # noqa: PLC0415

                engine = _SdkTTS(
                    model="supertonic-3",
                    model_dir=str(self.model_dir),
                    auto_download=False,
                )
            except Exception as exc:  # noqa: BLE001 - explicit failure, no fallback
                detail = f"Supertonic 3 engine load failed: {exc}"
                self._load_error = detail
                logger.error(detail)
                raise SupertonicTtsError(detail) from exc
            self.load_s = time.perf_counter() - t0
            logger.info(
                "TTS engine = Supertonic 3 (ONNX Runtime CPU, %s, loaded in %.2fs,"
                " %d voices, %d Hz)",
                self.model_dir,
                self.load_s,
                len(engine.voice_style_names),
                engine.sample_rate,
            )
            self._tts = engine
            return engine

    def _style(self, voice_name: str) -> Any:
        """Cached voice-style object for *voice_name*."""
        style = self._styles.get(voice_name)
        if style is None:
            engine = self._ensure_engine()
            style = engine.get_voice_style(voice_name)
            self._styles[voice_name] = style
        return style

    # -- synthesis ------------------------------------------------------------ #

    def _speed_from_length_scale(self, length_scale: Optional[float]) -> float:
        """Configured speed for None; ``1/length_scale`` clamped otherwise."""
        if length_scale is None:
            speed = float(self._config.supertonic_speed)
        else:
            try:
                speed = 1.0 / float(length_scale)
            except (TypeError, ValueError, ZeroDivisionError):
                logger.debug("Ignoring invalid length_scale %r", length_scale)
                speed = float(self._config.supertonic_speed)
        if speed < SPEED_MIN or speed > SPEED_MAX:
            clamped = min(max(speed, SPEED_MIN), SPEED_MAX)
            logger.debug(
                "speed %.3f outside [%s, %s] - clamped to %.3f",
                speed,
                SPEED_MIN,
                SPEED_MAX,
                clamped,
            )
            speed = clamped
        return speed

    def _synthesize_f32(
        self, text: str, language: str, voice: Optional[str], length_scale: Optional[float]
    ) -> np.ndarray:
        """Core synthesis: float32 mono waveform at the engine sample rate."""
        engine = self._ensure_engine()
        voice_name = self._voice_name(language, voice)
        style = self._style(voice_name)
        lang = _LANG_MAP.get(language, _DEFAULT_LANG)
        self.last_voice = voice_name
        self.last_language = language
        wav, _dur = engine.synthesize(
            text,
            voice_style=style,
            total_steps=int(self._config.supertonic_steps),
            speed=self._speed_from_length_scale(length_scale),
            max_chunk_length=int(self._config.supertonic_max_chunk_chars) or None,
            silence_duration=float(self._config.supertonic_silence_duration),
            lang=lang,
        )
        x = np.asarray(wav, dtype=np.float32)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.reshape(-1)
        if x.ndim != 1 or x.size == 0:
            raise SupertonicTtsError(
                f"Supertonic 3 returned unusable audio shape {x.shape}"
            )
        return x

    def synthesize(
        self,
        text: str,
        language: str,
        length_scale: Optional[float] = None,
        voice: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Synthesize *text* -> mono int16 PCM at config.output_sample_rate.

        Returns None on ANY failure (empty text, load error, synthesis
        error) — the caller reports an explicit TTS error; there is NO
        fallback engine. ``length_scale`` (M2 emotion pacing) maps to the
        SDK speed as 1/length_scale; None keeps the configured default.
        Edge silence is trimmed (see module docstring) when configured.
        """
        self._stop_requested = False  # a new request supersedes a previous stop()
        if not text or not text.strip():
            return None
        try:
            x = self._synthesize_f32(text, language, voice, length_scale)
        except SupertonicTtsError as exc:
            logger.error("TTS synthesis failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - explicit failure, no fallback
            logger.error("TTS synthesis raised for %r: %s", text[:60], exc)
            return None
        if self._stop_requested:
            logger.debug("TTS: stop requested during synthesis - result discarded")
            return None
        if self._config.supertonic_trim_silence:
            x = _trim_edge_silence(x)
        return _float32_to_int16(x)

    def synthesize_to_file(
        self,
        text: str,
        language: str,
        out_path: Path,
        length_scale: Optional[float] = None,
        voice: Optional[str] = None,
    ) -> bool:
        """Synthesize *text* into a 44.1 kHz 16-bit mono WAV *out_path*.

        Returns True only when the file exists and is a valid non-empty
        mono 16-bit WAV. False (logged) on any failure — no fallback.
        """
        pcm = self.synthesize(text, language, length_scale, voice)
        if pcm is None or not len(pcm):
            return False
        out_path = Path(out_path)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(self._config.output_sample_rate))
                w.writeframes(pcm.tobytes())
        except OSError as exc:
            logger.warning("TTS cannot write output %s: %s", out_path, exc)
            return False
        return out_path.is_file() and out_path.stat().st_size > 44

    def stop(self) -> None:
        """Barge-in: request abandonment of the in-flight synthesis.

        In-process ONNX inference cannot be killed mid-call (no subprocess
        any more); instead the synthesis result is DISCARDED and playback
        stops (the pipeline also calls SpeakerOutput.stop). The flag is
        reset by the next synthesize call, mirroring the old contract.
        """
        self._stop_requested = True
        logger.debug("TTS stop requested (result of in-flight synthesis discarded)")

    # -- introspection --------------------------------------------------------- #

    def status_detail(self) -> str:
        """One-line runtime status for logs / the diagnostics panel."""
        if self._tts is None:
            if self._load_error:
                return f"Supertonic 3 · LOAD ERROR: {self._load_error}"
            return "Supertonic 3 · not loaded yet (first synthesis loads it)"
        return (
            f"Supertonic 3 · ONNX CPU · {self._config.supertonic_steps} steps · "
            f"speed {self._config.supertonic_speed} · "
            f"{len(self._tts.voice_style_names)} voices · "
            f"{int(self._tts.sample_rate)} Hz"
        )


# ─── helpers (module-level for testability) ─────────────────────────────────


def _trim_edge_silence(x: np.ndarray, rate: int = 44100) -> np.ndarray:
    """Trim leading/trailing near-silence with a 100 ms guard (never raises).

    The Supertonic vocoder pads utterances with 0.35–0.9 s of near-silence
    per edge (measured on M1..F5, data/supertonic_validation/report.json);
    untrimmed this would add ~1.5 s dead air to every conversational chunk.
    """
    try:
        frame = max(1, int(rate * _TRIM_FRAME_S))
        n = len(x)
        m = n // frame
        if m < 4:
            return x
        fr_rms = np.sqrt((x[: m * frame].reshape(m, frame) ** 2).mean(axis=1))
        silent = fr_rms < _TRIM_RMS_FLOOR
        lead = 0
        for s in silent:
            if not s:
                break
            lead += 1
        trail = 0
        for s in silent[::-1]:
            if not s:
                break
            trail += 1
        if lead == 0 and trail == 0:
            return x
        guard = max(1, int(_TRIM_GUARD_S / _TRIM_FRAME_S))
        start = max(0, (max(0, lead - guard)) * frame)
        end = min(n, (m - max(0, trail - guard)) * frame)
        if end - start < frame:
            return x
        return x[start:end]
    except Exception:  # noqa: BLE001 - trimming must never kill synthesis
        return x


def _float32_to_int16(x: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] -> int16 mono (clipped), never raises."""
    try:
        clipped = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)
    except Exception:  # noqa: BLE001
        return np.zeros(0, dtype=np.int16)
