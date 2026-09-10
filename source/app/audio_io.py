"""Audio I/O: microphone input stream and speaker playback (sounddevice, lazy).

``sounddevice`` (PortAudio) is imported lazily inside the functions that need
it, so this module — and the whole app — stays importable on machines without
audio hardware or drivers (this sandbox included).

* :class:`MicStream` delivers fixed-size mono float32 frames to a callback from
  the PortAudio thread. The callback is fully exception-guarded: an exception
  raised inside it would abort the audio stream, so errors are logged and the
  frame dropped instead.
* :class:`SpeakerOutput` plays int16 mono PCM. ``play`` is blocking, survives
  device changes (one retry, then the chunk is logged and dropped so an audio
  hiccup never crashes the conversation), ``stop`` aborts playback (barge-in).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

import numpy as np

from app.config import AgentConfig

logger = logging.getLogger(__name__)


def _import_sounddevice() -> Any:
    """Import sounddevice lazily with an actionable install hint."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed. Install it with: "
            "pip install sounddevice (the system also needs a PortAudio library)"
        ) from exc
    return sd


def _mic_open_error_hint(sd: Any, exc: Exception) -> str:
    """Turn a mic InputStream failure into an actionable message (v0.4.9).

    Lists the input devices PortAudio actually sees, so 'no mic' vs 'device
    busy' vs 'disconfigured device index' is diagnosable from the log alone.
    """
    lines = [f"Microphone open failed: {exc}"]
    try:
        devices = sd.query_devices()
        inputs = [d for d in devices if int(getattr(d, "max_input_channels", 0) or 0) > 0]
        if inputs:
            names = ", ".join(
                f"[{d['index']}] {d['name']}" for d in inputs[:10]
            )
            lines.append(f"Available input devices: {names}")
        else:
            lines.append(
                "PortAudio sees NO input devices - check that a microphone is "
                "connected and enabled (Windows: Sound settings + mic privacy)."
            )
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        lines.append("(could not list PortAudio devices)")
    lines.append(
        "Set audio_input_device in config/voicemem_config.yaml (device name or "
        "index) or leave it empty for the system default."
    )
    return " ".join(lines)


class MicStream:
    """Mono 16 kHz float32 microphone stream delivering fixed-size frames.

    Wraps a ``sounddevice.InputStream`` with ``blocksize =
    config.vad_frame_samples`` (512 at the default config = 32 ms VAD frames).
    Every block is passed to ``frame_callback`` from the PortAudio callback
    thread — the callback is exception-guarded because an unhandled exception
    there aborts the stream. Used as a context manager (start on enter, stop on
    exit); ``stop()`` is idempotent and never raises.

    v0.4.9 (report issue #4) device fallback: ``config.audio_input_device``
    (device NAME substring or numeric index, "" = system default) is honoured
    when set — and when that device cannot be opened (USB mic unplugged,
    index re-enumerated, device busy) the stream FALLS BACK to the system
    default with a prominent warning instead of leaving the agent unusable.
    The default open itself is retried once after 0.5 s before raising a
    RuntimeError that names the actual input devices PortAudio sees.
    """

    def __init__(self, config: AgentConfig, frame_callback: Callable[["np.ndarray"], None]) -> None:
        """Store config and callback; the stream itself opens in start()/__enter__."""
        self._config = config
        self._frame_callback = frame_callback
        self._stream: Any = None
        self._device_in_use: Any = None

    def __enter__(self) -> "MicStream":
        """Open and start the input stream; return self."""
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Stop and close the input stream (never raises, swallows all exceptions)."""
        self.stop()

    @property
    def device_in_use(self) -> Any:
        """The device the open stream uses (None = system default)."""
        return self._device_in_use

    def start(self) -> None:
        """Open and start the PortAudio input stream (no-op when already running)."""
        if self._stream is not None:
            return
        sd = _import_sounddevice()
        self._stream = self._open_stream(sd)
        self._stream.start()
        logger.info(
            "Mic stream started: %d Hz, %d channel(s), blocksize %d, device %s",
            self._config.sample_rate,
            self._config.channels,
            self._config.vad_frame_samples,
            self._device_in_use if self._device_in_use is not None else "<system default>",
        )

    def _open_stream(self, sd: Any) -> Any:
        """Open the InputStream with the v0.4.9 device fallback chain.

        Order: configured device (when set) -> system default -> one retry of
        the system default (0.5 s later, survives hot-plug re-enumeration) ->
        RuntimeError with the available-device list. Numeric strings are
        device indices (sounddevice would otherwise name-match '3').
        """
        kwargs: dict[str, Any] = {
            "samplerate": self._config.sample_rate,
            "channels": self._config.channels,
            "dtype": "float32",
            "blocksize": self._config.vad_frame_samples,
            "callback": self._on_audio,
        }
        requested = str(self._config.audio_input_device or "").strip()
        if requested:
            device: Any = int(requested) if requested.lstrip("-").isdigit() else requested
            try:
                stream = sd.InputStream(device=device, **kwargs)
                self._device_in_use = device
                return stream
            except Exception as exc:  # noqa: BLE001 - device gone/wrong/busy
                logger.warning(
                    "Configured mic device %r could not be opened (%s); "
                    "falling back to the SYSTEM DEFAULT input device",
                    requested,
                    exc,
                )
        try:
            stream = sd.InputStream(**kwargs)  # system default
            self._device_in_use = None
            return stream
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "System default mic open failed (%s); retrying once in 0.5 s", exc
            )
            time.sleep(0.5)
            try:
                stream = sd.InputStream(**kwargs)
                self._device_in_use = None
                logger.info("Mic stream recovered on the second open attempt")
                return stream
            except Exception as exc2:  # noqa: BLE001
                raise RuntimeError(_mic_open_error_hint(sd, exc2)) from exc2

    def stop(self) -> None:
        """Stop and close the stream (idempotent; errors are logged, never raised)."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
            logger.info("Mic stream stopped")
        except Exception:
            logger.exception("Error while stopping the mic stream (ignored)")

    def _on_audio(self, indata: "np.ndarray", frames: int, time_info: Any, status: Any) -> None:
        """PortAudio callback: forward one mono frame to the user callback.

        Exception-safe by contract — a raise here would abort the PortAudio
        stream, so any error is logged and the frame dropped instead.
        """
        try:
            if status:
                logger.warning("Mic stream status flag: %s", status)
            data = np.asarray(indata, dtype=np.float32)
            if data.ndim > 1:
                data = data[:, 0]  # first channel (mono config)
            self._frame_callback(data.copy())
        except Exception:
            logger.exception(
                "Mic frame callback raised an exception; frame dropped "
                "(callbacks must never propagate into the PortAudio thread)"
            )


class SpeakerOutput:
    """Int16 mono PCM playback through sounddevice.

    ``play`` blocks until playback completes (``sounddevice.play`` + ``sd.wait``)
    and survives device changes: a failed attempt is retried once, and a second
    failure is logged with the dropped sample count so an audio-device hiccup
    never crashes the conversation loop. ``play_async`` wraps ``play`` in
    ``asyncio.to_thread``; ``stop`` aborts in-flight playback (used on barge-in)
    and is a safe no-op when nothing plays or sounddevice is unavailable. The
    default rate is ``config.output_sample_rate`` (22050, Piper native rate);
    pass ``sample_rate`` per call to override.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Store config; sounddevice is imported lazily on first playback."""
        self._config = config

    def play(self, pcm: "np.ndarray", sample_rate: Optional[int] = None) -> None:
        """Play int16 mono PCM, blocking until playback completes.

        Non-int16 input is cast to int16, ``(N, 1)`` arrays flattened. Device
        failures are retried once; after a second failure the chunk is dropped
        with an error log (playback errors never propagate to the caller).
        """
        data = np.asarray(pcm)
        if data.size == 0:
            return
        if data.dtype != np.int16:
            logger.debug("SpeakerOutput: casting %s PCM to int16", data.dtype)
            data = data.astype(np.int16)
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.reshape(-1)
        rate = int(sample_rate if sample_rate is not None else self._config.output_sample_rate)
        sd = _import_sounddevice()
        last_error: Optional[Exception] = None
        for attempt in (1, 2):
            try:
                sd.play(data, samplerate=rate, blocking=False)
                sd.wait()
                return
            except Exception as exc:  # device changes, unplugged devices, etc.
                last_error = exc
                logger.warning("Playback attempt %d failed: %s", attempt, exc)
                try:
                    sd.stop()
                except Exception:
                    logger.debug("sd.stop() after a failed playback raised (ignored)")
        logger.error(
            "Playback failed after retry; dropping %d samples (%s)",
            int(data.shape[0]),
            last_error,
        )

    async def play_async(self, pcm: "np.ndarray", sample_rate: Optional[int] = None) -> None:
        """Non-blocking wrapper around :meth:`play` via ``asyncio.to_thread``."""
        await asyncio.to_thread(self.play, pcm, sample_rate)

    def stop(self) -> None:
        """Abort in-flight playback (barge-in); safe when nothing is playing."""
        try:
            sd = _import_sounddevice()
        except RuntimeError:
            logger.debug("sounddevice unavailable; SpeakerOutput.stop() is a no-op")
            return
        try:
            sd.stop()
        except Exception:
            logger.exception("Error while stopping playback (ignored)")
