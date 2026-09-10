"""Unit tests for app.barge_in (EchoGate + BargeInDetector) — pure stdlib.

Numbers used throughout: threshold 0.30, min_speech_ms 500, frame_ms 32 —
the sustain requirement is ceil(500/32) = 16 consecutive frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import unittest

from app.barge_in import BargeInDetector, EchoGate


def make_detector() -> BargeInDetector:
    """Fresh detector with the default M1 barge-in parameters."""
    return BargeInDetector(threshold=0.30, min_speech_ms=500, frame_ms=32)


class EchoGateTests(unittest.TestCase):
    """Loopback gating during TTS playback."""

    def test_gate_blocks_mic_processing_during_tts(self) -> None:
        """should_process_mic() is False while TTS is playing."""
        gate = EchoGate()
        self.assertTrue(gate.should_process_mic())
        self.assertFalse(gate.tts_playing)
        gate.on_tts_start()
        self.assertTrue(gate.tts_playing)
        self.assertFalse(gate.should_process_mic())

    def test_gate_re_enables_after_tts(self) -> None:
        """should_process_mic() is True again once TTS playback ends."""
        gate = EchoGate()
        gate.on_tts_start()
        gate.on_tts_end()
        self.assertFalse(gate.tts_playing)
        self.assertTrue(gate.should_process_mic())

    def test_gate_calls_are_idempotent(self) -> None:
        """Repeated start/end calls are harmless."""
        gate = EchoGate()
        gate.on_tts_start()
        gate.on_tts_start()
        self.assertTrue(gate.tts_playing)
        gate.on_tts_end()
        gate.on_tts_end()
        self.assertFalse(gate.tts_playing)


class BargeInDetectorTests(unittest.TestCase):
    """Sustained-speech barge-in detection during TTS playback."""

    def test_fires_after_min_speech_frames_during_tts(self) -> None:
        """Fires exactly on the 16th consecutive above-threshold frame (500 ms)."""
        det = make_detector()
        results = [det.update(0.35, True) for _ in range(15)]
        self.assertTrue(all(r is False for r in results))
        self.assertTrue(det.update(0.35, True))  # 16th consecutive frame

    def test_threshold_is_inclusive(self) -> None:
        """prob == threshold (0.30) counts as sustained speech."""
        det = make_detector()
        results = [det.update(0.30, True) for _ in range(16)]
        self.assertTrue(results[-1])

    def test_does_not_fire_when_tts_not_playing(self) -> None:
        """High probabilities outside TTS playback never trigger barge-in."""
        det = make_detector()
        for _ in range(30):
            self.assertFalse(det.update(0.95, False))

    def test_does_not_fire_when_speech_too_short(self) -> None:
        """13 frames (~400 ms < 500 ms) of speech must NOT trigger barge-in."""
        det = make_detector()
        for _ in range(13):
            self.assertFalse(det.update(0.4, True))
        for _ in range(20):
            self.assertFalse(det.update(0.05, True))

    def test_fires_only_once_until_reset(self) -> None:
        """After firing, further updates return False until reset()."""
        det = make_detector()
        for _ in range(15):
            self.assertFalse(det.update(0.4, True))
        self.assertTrue(det.update(0.4, True))
        for _ in range(10):
            self.assertFalse(det.update(0.4, True))  # already fired
        det.reset()
        for _ in range(15):
            self.assertFalse(det.update(0.4, True))
        self.assertTrue(det.update(0.4, True))  # fires again after reset

    def test_accumulation_resets_on_low_prob_frame(self) -> None:
        """A single below-threshold frame resets the sustain streak."""
        det = make_detector()
        for _ in range(15):
            self.assertFalse(det.update(0.4, True))
        self.assertFalse(det.update(0.2, True))  # streak broken
        for _ in range(15):
            self.assertFalse(det.update(0.4, True))  # 15 since the reset: not enough
        self.assertTrue(det.update(0.4, True))  # 16th consecutive frame since reset

    def test_accumulation_resets_when_tts_stops(self) -> None:
        """Frames with tts_playing=False reset the streak even at high prob."""
        det = make_detector()
        for _ in range(15):
            self.assertFalse(det.update(0.4, True))
        for _ in range(10):
            self.assertFalse(det.update(0.9, False))  # TTS paused: streak resets
        for _ in range(15):
            self.assertFalse(det.update(0.4, True))
        self.assertTrue(det.update(0.4, True))  # 16th consecutive frame after resume

    def test_reset_clears_everything(self) -> None:
        """reset() clears both the fired flag and the accumulated streak."""
        det = make_detector()
        for _ in range(16):
            det.update(0.4, True)
        det.reset()
        for _ in range(15):
            self.assertFalse(det.update(0.4, True))
        self.assertTrue(det.update(0.4, True))


if __name__ == "__main__":
    unittest.main()
