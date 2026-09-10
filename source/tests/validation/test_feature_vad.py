"""VAD feature validation (Task 12 - M0.1).

Validates the ``vad`` feature: ``app.vad.VadStateMachine`` driven with the
REAL values from ``config/voicemem_config.yaml`` (v0.4.11: threshold 0.25,
300 ms hangover, 32 ms frames -> ceil(300/32) = 10 hangover frames) must
emit SPEECH_START once, keep ``in_speech`` through the hangover window,
emit SPEECH_END after the sustained silence and reset cleanly.

Deep check: when onnxruntime and the Silero model file are present, real ONNX
inference runs on a silent and a noise frame and returns probabilities in
[0, 1].
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from app.vad import VadEvent, VadStateMachine
from tests.validation._report import FeatureValidationTest, has_module

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

_ENV_KEYS = (
    "VOICEMEM_HOME", "VOICEMEM_MEMORY_ROOT", "PIPER_VOICES_PATH",
    "PIPER_EXECUTABLE", "SILERO_VAD_PATH", "QWEN3_ASR_MODEL_PATH",
    "LLAMA_MODEL_PATH", "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT",
    "OPENAI_BASE_URL", "OPENAI_MODEL",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class VadFeatureTest(_EnvNeutralTest):
    """Silero VAD state machine with the real repo config values."""

    FEATURE = "vad"

    def _make_machine(self) -> VadStateMachine:
        cfg = AgentConfig.from_yaml(YAML_PATH)
        return VadStateMachine(
            threshold=cfg.vad_threshold,
            hangover_ms=cfg.vad_hangover_ms,
            frame_ms=cfg.vad_frame_ms,
        )

    def test_logic_silence_then_speech_start(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertEqual(cfg.vad_threshold, 0.25)
        self.assertEqual(cfg.vad_hangover_ms, 300)
        self.assertEqual(cfg.vad_frame_ms, 32)
        machine = self._make_machine()
        # Below-threshold silence while out of speech: always NONE.
        for _ in range(5):
            self.assertEqual(machine.update(0.10), VadEvent.NONE)
        self.assertFalse(machine.in_speech)
        # First above-threshold frame -> SPEECH_START exactly once.
        self.assertEqual(machine.update(0.60), VadEvent.SPEECH_START)
        self.assertTrue(machine.in_speech)
        # Sustained speech: no further events.
        for _ in range(5):
            self.assertEqual(machine.update(0.80), VadEvent.NONE)
        self.assertTrue(machine.in_speech)

    def test_logic_hangover_speech_end(self):
        machine = self._make_machine()
        self.assertEqual(machine.update(0.60), VadEvent.SPEECH_START)
        # ceil(300 ms / 32 ms) = 10 consecutive silent frames before the end.
        for _ in range(9):
            self.assertEqual(machine.update(0.10), VadEvent.NONE)
            self.assertTrue(machine.in_speech, "hangover window keeps in_speech")
        self.assertEqual(machine.update(0.10), VadEvent.SPEECH_END)
        self.assertFalse(machine.in_speech)
        # After the end: silence is NONE again.
        self.assertEqual(machine.update(0.10), VadEvent.NONE)
        # A speech frame re-arms a new utterance.
        self.assertEqual(machine.update(0.60), VadEvent.SPEECH_START)

    def test_logic_reset_returns_to_initial_state(self):
        machine = self._make_machine()
        machine.update(0.60)
        self.assertTrue(machine.in_speech)
        machine.reset()
        self.assertFalse(machine.in_speech)
        self.assertEqual(machine.update(0.10), VadEvent.NONE)
        self.assertEqual(machine.update(0.60), VadEvent.SPEECH_START)

    def test_deep_silero_onnx_inference(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        if not has_module("onnxruntime"):
            self.deep_skip("onnxruntime not installed")
        if not cfg.silero_vad_path.is_file():
            self.deep_skip("silero_vad.onnx not downloaded")
        from app.vad import SileroVad

        vad = SileroVad(cfg)
        silent = vad.prob(np.zeros(512, dtype=np.float32))
        self.assertIsInstance(silent, float)
        self.assertGreaterEqual(silent, 0.0)
        self.assertLessEqual(silent, 1.0)
        noise = np.random.default_rng(42).standard_normal(512).astype(np.float32)
        noisy = vad.prob(noise)
        self.assertGreaterEqual(noisy, 0.0)
        self.assertLessEqual(noisy, 1.0)
        self.deep_pass(
            f"silero onnx inference OK (silent={silent:.3f}, noise={noisy:.3f})"
        )


if __name__ == "__main__":
    unittest.main()
