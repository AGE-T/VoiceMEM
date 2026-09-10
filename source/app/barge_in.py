"""Echo gate and barge-in detection — PURE Python standard library.

Implements the V1 echo-handling strategy (loopback gating, milestones Section 7.4):

* While TTS is playing, the microphone keeps being captured (VAD must keep
  running) but the frames are NOT forwarded to ASR — :class:`EchoGate` answers
  ``should_process_mic()`` False during playback.
* :class:`BargeInDetector` watches the VAD speech probability DURING playback and
  fires True exactly once when ``prob >= threshold`` is sustained for at least
  ``min_speech_ms`` (frame-count based: ``ceil(min_speech_ms / frame_ms)`` frames),
  signalling the pipeline to cancel TTS/LLM/memory work. The pipeline must call
  ``reset()`` after each turn.
"""

from __future__ import annotations

import math


class EchoGate:
    """Loopback gating: mic input is captured but not fed to ASR while TTS plays.

    VAD keeps running on every frame regardless — barge-in detection needs the
    probabilities. Idempotent: repeated ``on_tts_start`` / ``on_tts_end`` calls
    are harmless.
    """

    def __init__(self) -> None:
        """Start with TTS idle (mic processing enabled)."""
        self._tts_playing = False

    def on_tts_start(self) -> None:
        """Mark the start of TTS playback (mic frames must no longer reach ASR)."""
        self._tts_playing = True

    def on_tts_end(self) -> None:
        """Mark the end of TTS playback (mic processing re-enabled)."""
        self._tts_playing = False

    @property
    def tts_playing(self) -> bool:
        """True while TTS playback is active."""
        return self._tts_playing

    def should_process_mic(self) -> bool:
        """False while TTS is playing (loopback gating), True otherwise."""
        return not self._tts_playing


class BargeInDetector:
    """One-shot sustained-speech detector during TTS playback (barge-in trigger).

    ``update`` fires ``True`` ONCE when the speech probability has stayed at or
    above ``threshold`` for at least ``min_speech_frames`` consecutive frames
    while TTS is playing. The sustain accumulator resets on any below-threshold
    frame or on any frame where TTS is not playing; the fired flag resets only
    via :meth:`reset` (call it after each turn). Outside TTS playback the
    detector never fires.
    """

    def __init__(self, threshold: float, min_speech_ms: int, frame_ms: int) -> None:
        """Configure the detector from raw millisecond units.

        Args:
            threshold: speech probability threshold (inclusive, 0..1).
            min_speech_ms: sustained speech needed to trigger barge-in.
            frame_ms: duration of one VAD frame in milliseconds.
        """
        self._threshold = float(threshold)
        self._frame_ms = max(1, int(frame_ms))
        self._min_speech_frames = max(1, math.ceil(int(min_speech_ms) / self._frame_ms))
        self._sustained_frames = 0
        self._fired = False

    def update(self, prob: float, tts_playing: bool) -> bool:
        """Feed one VAD frame; return True exactly once when barge-in triggers."""
        if self._fired:
            return False
        if not tts_playing or float(prob) < self._threshold:
            # No TTS playback or below-threshold frame: reset the sustain streak.
            self._sustained_frames = 0
            return False
        self._sustained_frames += 1
        if self._sustained_frames >= self._min_speech_frames:
            self._fired = True
            self._sustained_frames = 0
            return True
        return False

    def reset(self) -> None:
        """Clear the fired flag and the sustain streak (call after each turn)."""
        self._sustained_frames = 0
        self._fired = False
