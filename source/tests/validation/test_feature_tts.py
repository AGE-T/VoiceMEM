"""TTS feature validation (v0.7.0 — Supertonic 3).

Validates the ``tts`` feature: the Supertonic 3 model-asset contract of
``app.config.AgentConfig`` (onnx modules + preset voice styles under
``models/tts/supertonic-3``) and the availability logic of
``app.tts_supertonic.SupertonicTtsEngine.is_available()`` using a TEMPORARY
fake layout (fake onnx files + fake voice-style JSONs): all present -> True,
a missing module -> False.

Deep check: with the real Supertonic 3 assets installed, a Hungarian smoke
sentence is synthesized through the PRODUCTION adapter to a WAV file
(> 1024 bytes, mono, 16-bit, 44100 Hz). No network — the SDK runs with
auto_download=False from the local model dir (Piper retired, no fallback).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from app.tts_supertonic import SupertonicTtsEngine
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

_ENV_KEYS = (
    "VOICEMEM_HOME", "PIPER_VOICES_PATH", "PIPER_EXECUTABLE",
    "SUPERTONIC_MODEL_PATH", "SUPERTONIC_SPEED", "SUPERTONIC_STEPS",
    "SUPERTONIC_TRIM_SILENCE", "TTS_ENGINE", "TTS_HU_VOICE", "TTS_EN_VOICE",
    "QWEN3_ASR_MODEL_PATH", "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT",
    "OPENAI_BASE_URL",
)

_ONNX_MODULES = (
    "tts.json",
    "unicode_indexer.json",
    "duration_predictor.onnx",
    "text_encoder.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class TtsFeatureTest(_EnvNeutralTest):
    """Supertonic 3 asset contract and availability logic."""

    FEATURE = "tts"

    def test_logic_path_contract(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertTrue(
            cfg.supertonic_model_dir.as_posix().endswith("models/tts/supertonic-3")
        )
        self.assertEqual(cfg.tts_engine, "supertonic3")
        self.assertEqual(cfg.tts_hu_voice, "F1")
        self.assertEqual(cfg.tts_en_voice, "F1")
        self.assertEqual(cfg.output_sample_rate, 44100)
        self.assertEqual(cfg.supertonic_steps, 8)
        self.assertAlmostEqual(cfg.supertonic_speed, 1.05)
        self.assertTrue(cfg.supertonic_trim_silence)

    def test_logic_is_available_with_fake_layout(self):
        """Fake onnx modules + voice styles in a temp root drive is_available()."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "tts" / "supertonic-3"
            (model_dir / "onnx").mkdir(parents=True)
            for name in _ONNX_MODULES:
                (model_dir / "onnx" / name).write_bytes(b"fake module")
            styles = model_dir / "voice_styles"
            styles.mkdir()
            for v in ("F1", "M1"):
                (styles / f"{v}.json").write_text("{}")
            cfg = AgentConfig(root=root)
            engine = SupertonicTtsEngine(cfg)
            self.assertTrue(
                engine.is_available(),
                "fake onnx + default voice styles must report availability",
            )
            # Removing one onnx module flips availability to False.
            (model_dir / "onnx" / "vocoder.onnx").unlink()
            self.assertFalse(
                engine.is_available(), "a missing onnx module must break availability"
            )

    def test_deep_supertonic_synthesis_smoke(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        model_dir = cfg.supertonic_model_dir
        if not all((model_dir / "onnx" / n).is_file() for n in _ONNX_MODULES) or not (
            cfg.supertonic_voices_dir / f"{cfg.tts_hu_voice}.json"
        ).is_file():
            self.deep_skip("Supertonic 3 model assets not installed")
        engine = SupertonicTtsEngine(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "smoke_hu.wav"
            ok = engine.synthesize_to_file("Jo reggelt!", "hu", out_path)
            self.assertTrue(ok, "Supertonic 3 synthesis returned False")
            self.assertTrue(out_path.is_file(), "engine wrote no output file")
            self.assertGreater(
                out_path.stat().st_size, 1024, "synthesized WAV is suspiciously small"
            )
            with wave.open(str(out_path), "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 44100)
        self.deep_pass("Supertonic 3 HU synthesis smoke OK (44.1 kHz 16-bit mono)")


if __name__ == "__main__":
    unittest.main()
