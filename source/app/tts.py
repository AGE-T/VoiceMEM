"""Piper TTS subprocess wrapper.

numpy is allowed at module level; piper is invoked via ``subprocess`` and the
produced WAV is read with the stdlib ``wave`` module. Text is fed on stdin
(one line per synthesis; internal newlines are collapsed to spaces).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .config import AgentConfig
from .text_utils import LANG_EN, LANG_HU

logger = logging.getLogger(__name__)

_SYNTH_TIMEOUT_S = 10.0


class TtsEngine:
    """Piper subprocess wrapper producing mono 16-bit PCM per language.

    Voice per language: ``config.voices_dir / f"{voice_name}.onnx"`` where the
    voice name comes from ``config.tts_hu_voice`` / ``config.tts_en_voice``.
    Unknown languages fall back to the Hungarian voice. ``stop()`` kills the
    in-flight subprocess for barge-in and is reset by the next synthesis call.

    v0.4.2 voice selection: every synthesis method accepts an optional
    ``voice`` (Piper voice id, e.g. ``hu_HU-imre-medium``) that OVERRIDES the
    per-language default — the web UI's Voice section persists the selection
    and the backend resolves it per reply (auto mode: response language).
    ``last_voice`` / ``last_language`` expose what the engine actually used
    (runtime status: "Voice: Imre").
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False
        #: Voice actually used by the last synthesis call (status display).
        self.last_voice: str = ""
        self.last_language: str = ""

    # -- voice resolution ----------------------------------------------------- #

    def _voice_path(self, language: str, voice: Optional[str] = None) -> Path:
        """Resolve the onnx voice model path (explicit *voice* wins; unknown -> HU)."""
        voice_name = str(voice or "").strip()
        if voice_name:
            candidate = self._config.voices_dir / f"{voice_name}.onnx"
            if candidate.is_file():
                return candidate
            # Unknown/missing explicit voice: log loudly (the UI selector
            # reaching Piper is a task contract) and fall through to the
            # per-language default rather than failing the whole reply.
            logger.warning(
                "requested voice %r is not installed (%s) - using the %s default",
                voice_name,
                candidate,
                language,
            )
        if language == LANG_EN:
            voice_name = self._config.tts_en_voice
        else:
            # LANG_HU and any unknown language use the Hungarian voice.
            voice_name = self._config.tts_hu_voice
        return self._config.voices_dir / f"{voice_name}.onnx"

    def voice_exists(self, voice: str) -> bool:
        """True when the Piper .onnx file for *voice* is installed."""
        return bool(voice) and (self._config.voices_dir / f"{str(voice).strip()}.onnx").is_file()

    def list_voices(self) -> list[str]:
        """Names (``.onnx`` stems) of the voices available in voices_dir."""
        voices_dir = self._config.voices_dir
        if not voices_dir.is_dir():
            return []
        return sorted(path.stem for path in voices_dir.glob("*.onnx"))

    def is_available(self) -> bool:
        """True when the piper executable AND both voice onnx files exist."""
        if not self._config.piper_exe_path.is_file():
            return False
        return self._voice_path(LANG_HU).is_file() and self._voice_path(LANG_EN).is_file()

    # -- synthesis ------------------------------------------------------------- #

    def synthesize_to_file(
        self,
        text: str,
        language: str,
        out_path: Path,
        length_scale: Optional[float] = None,
        voice: Optional[str] = None,
    ) -> bool:
        """Synthesize *text* into WAV *out_path* via a piper subprocess.

        Runs ``<piper> --model <voice.onnx> --output_file <wav>`` and feeds the
        text on stdin (utf-8; piper synthesizes per line, so internal newlines
        are collapsed to spaces). Timeout 10 s; stderr is logged on failure.
        Empty/whitespace text returns False without spawning a process.
        Returns True only on returncode 0 AND the file existing.

        ``length_scale`` (M2 emotion adaptation, spec 8.4): when not None it
        adds ``--length_scale <value>`` to the piper command - values > 1.0
        slow the speech down (e.g. 1.1 = 10% slower for a frustrated user).
        None keeps the piper default (1.0) - M1-identical behaviour.

        ``voice`` (v0.4.2): explicit Piper voice id that overrides the
        per-language default (UI Voice section / preview).
        """
        self._stop_requested = False  # a new request supersedes a previous stop()
        if not text or not text.strip():
            return False
        cleaned = " ".join(text.split())
        out_path = Path(out_path)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("TTS cannot create output dir %s: %s", out_path.parent, exc)
            return False

        if self._stop_requested:
            return False

        model_path = self._voice_path(language, voice)
        self.last_voice = model_path.stem
        self.last_language = language
        cmd = [
            str(self._config.piper_exe_path),
            "--model",
            str(model_path),
            "--output_file",
            str(out_path),
        ]
        if length_scale is not None:
            try:
                cmd += ["--length_scale", "%.3f" % float(length_scale)]
            except (TypeError, ValueError):
                logger.debug("Ignoring invalid length_scale %r", length_scale)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.warning("piper executable not found: %s", self._config.piper_exe_path)
            return False
        except OSError as exc:
            logger.warning("TTS subprocess failed to start: %s", exc)
            return False

        self._proc = proc
        if self._stop_requested:
            # v0.4.6: stop() raced with this spawn (it landed between the
            # entry reset and the process registration, so its kill() found
            # nothing to kill). Kill the fresh process so the barge-in is
            # not silently lost - piper would otherwise run to completion
            # (up to 10 s of wasted synthesis).
            logger.debug("TTS: stop raced with spawn - killing fresh piper")
            self._reap_killed(proc)
            return False
        try:
            try:
                _, stderr = proc.communicate(
                    input=cleaned.encode("utf-8"), timeout=_SYNTH_TIMEOUT_S
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "TTS synthesis timed out after %ss: %r", _SYNTH_TIMEOUT_S, cleaned[:80]
                )
                self._reap_killed(proc)
                return False
            if proc.returncode != 0:
                err = (stderr or b"").decode("utf-8", errors="replace").strip()
                logger.warning("piper failed (rc=%s): %s", proc.returncode, err[:400])
                return False
            if not out_path.is_file():
                logger.warning("piper reported success but output missing: %s", out_path)
                return False
            return True
        finally:
            self._proc = None

    def synthesize(
        self,
        text: str,
        language: str,
        length_scale: Optional[float] = None,
        voice: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Synthesize *text* and return mono int16 PCM at output_sample_rate.

        Writes to a temporary file, then reads it with stdlib ``wave``
        (verifying frames, 1 channel, 16-bit samples). Returns None on any
        failure. A sample rate mismatch with ``config.output_sample_rate`` is
        logged as a warning but does not fail the call.

        ``length_scale``: M2 emotion pacing (see :meth:`synthesize_to_file`);
        None keeps the piper default. ``voice``: explicit Piper voice id
        (v0.4.2 UI Voice section) that overrides the per-language default.
        """
        if not text or not text.strip():
            return None
        with tempfile.TemporaryDirectory(prefix="voicemem_tts_") as tmp_dir:
            out_path = Path(tmp_dir) / "tts_chunk.wav"
            if not self.synthesize_to_file(
                text, language, out_path, length_scale, voice=voice
            ):
                return None
            return self._read_wav(out_path)

    def stop(self) -> None:
        """Kill the in-flight piper subprocess (barge-in). Never raises.

        Also sets an internal stop flag that suppresses a subprocess start
        racing with this call; the flag is reset by the next
        ``synthesize_to_file()`` call.
        """
        self._stop_requested = True
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except (OSError, ValueError) as exc:
                logger.debug("TTS stop: kill failed: %s", exc)
        logger.debug("TTS stop requested")

    # -- internals -------------------------------------------------------------- #

    def _read_wav(self, out_path: Path) -> Optional[np.ndarray]:
        """Read a WAV file produced by piper into an int16 numpy array."""
        try:
            with wave.open(str(out_path), "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                if n_frames <= 0:
                    logger.warning("TTS WAV contains no frames: %s", out_path)
                    return None
                if n_channels != 1:
                    logger.warning(
                        "TTS WAV is not mono (channels=%s): %s", n_channels, out_path
                    )
                    return None
                if sample_width != 2:
                    logger.warning(
                        "TTS WAV is not 16-bit (width=%s bytes): %s", sample_width, out_path
                    )
                    return None
                if sample_rate != self._config.output_sample_rate:
                    logger.warning(
                        "TTS WAV sample rate %s differs from config.output_sample_rate %s",
                        sample_rate,
                        self._config.output_sample_rate,
                    )
                frames = wav_file.readframes(n_frames)
        except (wave.Error, EOFError, OSError) as exc:
            logger.warning("TTS WAV read failed (%s): %s", exc, out_path)
            return None
        if not frames:
            return None
        return np.frombuffer(frames, dtype=np.int16)

    @staticmethod
    def _reap_killed(proc: subprocess.Popen) -> None:
        """Kill and reap a timed-out subprocess (best effort, never raises)."""
        try:
            proc.kill()
        except (OSError, ValueError):
            pass
        try:
            proc.communicate(timeout=2.0)
        except Exception:  # noqa: BLE001 - best-effort reaping
            pass
