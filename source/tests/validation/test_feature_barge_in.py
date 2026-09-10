"""Barge-in feature validation (Task 12 - M0.1).

Validates the ``barge-in`` feature with the REAL values from
``config/voicemem_config.yaml`` (threshold 0.30, 500 ms sustain, 32 ms frames
-> ceil(500/32) = 16 frames):

* ``app.barge_in.BargeInDetector`` fires exactly ONCE when the speech
  probability stays >= threshold during TTS playback for the sustain window;
  below-threshold frames or no playback never fire it; ``reset()`` re-arms;
* ``app.barge_in.EchoGate`` blocks mic processing while TTS plays
  (loopback gating) and reopens afterwards.

Pure stdlib module - fully runnable in the sandbox.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.barge_in import BargeInDetector, EchoGate
from app.config import AgentConfig
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

_ENV_KEYS = (
    "VOICEMEM_HOME", "QWEN3_ASR_MODEL_PATH", "LLAMA_SERVER_HOST",
    "LLAMA_SERVER_PORT", "OPENAI_BASE_URL", "PIPER_VOICES_PATH",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class BargeInFeatureTest(_EnvNeutralTest):
    """One-shot sustained-speech detector during TTS playback."""

    FEATURE = "barge-in"

    def _make_detector(self) -> BargeInDetector:
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertEqual(cfg.barge_in_threshold, 0.30)
        self.assertEqual(cfg.barge_in_min_speech_ms, 500)
        self.assertEqual(cfg.vad_frame_ms, 32)
        return BargeInDetector(
            threshold=cfg.barge_in_threshold,
            min_speech_ms=cfg.barge_in_min_speech_ms,
            frame_ms=cfg.vad_frame_ms,
        )

    def test_logic_fires_once_after_sustain(self):
        detector = self._make_detector()
        # ceil(500 ms / 32 ms) = 16 consecutive above-threshold frames.
        for _ in range(15):
            self.assertFalse(detector.update(0.50, tts_playing=True))
        self.assertTrue(detector.update(0.50, tts_playing=True), "16th frame must fire")
        # Already fired: further frames never re-fire.
        for _ in range(10):
            self.assertFalse(detector.update(0.50, tts_playing=True))

    def test_logic_below_threshold_or_no_playback_never_fires(self):
        detector = self._make_detector()
        # Below-threshold probabilities during playback: never fires.
        for _ in range(40):
            self.assertFalse(detector.update(0.10, tts_playing=True))
        # Above threshold but TTS not playing: never fires (streak resets).
        for _ in range(40):
            self.assertFalse(detector.update(0.90, tts_playing=False))

    def test_logic_reset_rearms_and_echo_gate_gating(self):
        detector = self._make_detector()
        for _ in range(16):
            detector.update(0.50, tts_playing=True)
        detector.reset()
        # Re-armed: the sustain window starts over and fires again.
        for _ in range(15):
            self.assertFalse(detector.update(0.50, tts_playing=True))
        self.assertTrue(detector.update(0.50, tts_playing=True))

        # Echo gate: mic processing is blocked exactly while TTS plays.
        gate = EchoGate()
        self.assertTrue(gate.should_process_mic())
        gate.on_tts_start()
        self.assertFalse(gate.should_process_mic())
        self.assertTrue(gate.tts_playing)
        gate.on_tts_end()
        self.assertTrue(gate.should_process_mic())


if __name__ == "__main__":
    unittest.main()
