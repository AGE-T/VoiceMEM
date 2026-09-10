"""Audio I/O feature validation (Task 12 - M0.1).

Validates the ``audio`` feature: the audio settings of
``app.config.AgentConfig`` (16 kHz mono mic input, 22.05 kHz output, 512-sample
VAD frames, empty device names = system defaults) and the lazy-import contract
of ``app.audio_io`` - the module (with ``MicStream`` and ``SpeakerOutput``)
must import on a machine WITHOUT sounddevice/PortAudio.

Deep check: when sounddevice is installed, at least one input device must be
visible through PortAudio.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from tests.validation._report import FeatureValidationTest, has_module

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


class AudioFeatureTest(_EnvNeutralTest):
    """Audio config values + lazy sounddevice import contract."""

    FEATURE = "audio"

    def test_logic_audio_config_contract(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertEqual(cfg.output_sample_rate, 22050)
        self.assertEqual(cfg.channels, 1)
        self.assertEqual(cfg.vad_frame_samples, 512)
        self.assertEqual(cfg.audio_input_device, "", "empty = system default mic")
        self.assertEqual(cfg.audio_output_device, "", "empty = system default speaker")

    def test_logic_module_importable_without_sounddevice(self):
        """Lazy-import contract: app.audio_io imports without sounddevice."""
        import app.audio_io

        self.assertTrue(hasattr(app.audio_io, "MicStream"))
        self.assertTrue(hasattr(app.audio_io, "SpeakerOutput"))
        # Constructing the wrappers must not require audio hardware either.
        cfg = AgentConfig.from_yaml(YAML_PATH)
        stream = app.audio_io.MicStream(cfg, lambda frame: None)
        speaker = app.audio_io.SpeakerOutput(cfg)
        self.assertIsNotNone(stream)
        self.assertIsNotNone(speaker)

    def test_deep_portaudio_input_device_available(self):
        if not has_module("sounddevice"):
            self.deep_skip("sounddevice not installed")
        import sounddevice as sd

        devices = sd.query_devices()
        if not devices:
            self.deep_skip("no audio devices")
        inputs = [
            d for d in devices
            if int(d.get("max_input_channels", 0) or 0) >= 1
        ]
        self.assertGreater(
            len(inputs), 0, "sounddevice sees devices but no input channel"
        )
        self.deep_pass(f"{len(devices)} audio devices, input available")


if __name__ == "__main__":
    unittest.main()
