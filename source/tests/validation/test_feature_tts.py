"""TTS feature validation (Task 12 - M0.1).

Validates the ``tts`` feature: the Piper voice/binary path contract of
``app.config.AgentConfig`` (voices under ``models/tts/piper``, the piper
executable under ``bin/``) and the availability logic of
``app.tts.TtsEngine.is_available()`` using a TEMPORARY fake layout (fake piper
executable + fake ``.onnx`` voice files): all present -> True, a missing voice
-> False.

Deep check: with the real piper binary and voices installed, a Hungarian
smoke sentence is synthesized to a WAV file (> 1024 bytes).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from app.tts import TtsEngine
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

_ENV_KEYS = (
    "VOICEMEM_HOME", "PIPER_VOICES_PATH", "PIPER_EXECUTABLE",
    "TTS_HU_VOICE", "TTS_EN_VOICE", "QWEN3_ASR_MODEL_PATH",
    "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT", "OPENAI_BASE_URL",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class TtsFeatureTest(_EnvNeutralTest):
    """Piper voice resolution and availability logic."""

    FEATURE = "tts"

    def test_logic_path_contract(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertTrue(cfg.voices_dir.as_posix().endswith("models/tts/piper"))
        self.assertIn(cfg.piper_exe_path.name, ("piper", "piper.exe"))
        self.assertEqual(cfg.piper_exe_path.parent, cfg.bin_dir)
        self.assertEqual(cfg.tts_hu_voice, "hu_HU-anna-medium")
        self.assertEqual(cfg.tts_en_voice, "en_US-lessac-medium")
        self.assertEqual(cfg.output_sample_rate, 22050)

    def test_logic_is_available_with_fake_layout(self):
        """Fake piper binary + voices in a temp root drive is_available()."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "bin" / "piper.exe"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"fake piper binary")
            voices = root / "models" / "tts" / "piper"
            voices.mkdir(parents=True)
            for voice in ("hu_HU-anna-medium", "en_US-lessac-medium"):
                (voices / f"{voice}.onnx").write_bytes(b"fake voice model")
            cfg = AgentConfig(root=root, piper_executable=str(exe))
            engine = TtsEngine(cfg)
            self.assertTrue(
                engine.is_available(),
                "fake piper + both voices must report availability",
            )
            # Removing one voice file flips availability to False.
            (voices / "en_US-lessac-medium.onnx").unlink()
            self.assertFalse(
                engine.is_available(), "a missing voice must break availability"
            )

    def test_deep_piper_synthesis_smoke(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        voices = cfg.voices_dir
        if not (
            cfg.piper_exe_path.is_file()
            and (voices / f"{cfg.tts_hu_voice}.onnx").is_file()
            and (voices / f"{cfg.tts_en_voice}.onnx").is_file()
        ):
            self.deep_skip("piper binary/voices not installed")
        engine = TtsEngine(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "smoke_hu.wav"
            ok = engine.synthesize_to_file("Jo reggelt!", "hu", out_path)
            self.assertTrue(ok, "piper synthesis returned False")
            self.assertTrue(out_path.is_file(), "piper wrote no output file")
            self.assertGreater(
                out_path.stat().st_size, 1024, "synthesized WAV is suspiciously small"
            )
        self.deep_pass("piper HU synthesis smoke OK")


if __name__ == "__main__":
    unittest.main()
