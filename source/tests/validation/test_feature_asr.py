"""ASR feature validation (Task 12 - M0.1).

Validates the ``asr`` feature: the Qwen3-ASR-0.6B contract of
``app.config.AgentConfig`` (600 ms quasi-streaming chunks = 9600 samples at
16 kHz, the local model directory name and the HuggingFace model name) and
the lazy-import contract of ``app.asr.AsrEngine`` (``is_available()`` False
without torch, module importable in a dependency-free sandbox).

Deep check: M0 exit criterion 6 - with torch + transformers installed and the
model downloaded, the tokenizer must load OFFLINE from the local directory.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.asr import AsrEngine
from app.config import AgentConfig
from tests.validation._report import FeatureValidationTest, has_module

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

_ENV_KEYS = (
    "VOICEMEM_HOME", "QWEN3_ASR_MODEL_PATH", "LLAMA_MODEL_PATH",
    "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT", "OPENAI_BASE_URL", "OPENAI_MODEL",
    "PIPER_VOICES_PATH", "PIPER_EXECUTABLE", "SILERO_VAD_PATH",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class AsrFeatureTest(_EnvNeutralTest):
    """Qwen3-ASR-0.6B configuration + lazy availability contract."""

    FEATURE = "asr"

    def test_logic_model_and_chunk_contract(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertEqual(cfg.asr_chunk_samples, 9600, "600 ms @ 16 kHz must be 9600")
        self.assertEqual(cfg.asr_model_dir.name, "qwen3-asr-0.6b")
        self.assertEqual(cfg.asr_model_name, "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(cfg.sample_rate, 16000)

    def test_logic_engine_unavailable_without_torch(self):
        """Lazy-import contract: no torch -> is_available() is False (no raise)."""
        cfg = AgentConfig.from_yaml(YAML_PATH)
        engine = AsrEngine(cfg)
        if has_module("torch") and has_module("transformers"):
            # On a machine with the full ASR stack the engine reports availability.
            self.assertTrue(engine.is_available())
        else:
            self.assertFalse(engine.is_available())

    def test_logic_version_failure_hint_is_actionable(self):
        """v0.4.5 (field report #5): the transformers-version failure hint must
        name the START.bat self-repair path, not only the raw pip command -
        the user saw "ASR not green" with no way forward."""
        source = (REPO_ROOT / "app" / "asr.py").read_text(encoding="utf-8")
        self.assertIn("transformers >= 5.0", source)
        self.assertIn("START.bat", source)
        self.assertIn("automatically", source)

    def test_deep_offline_tokenizer_load(self):
        """M0 exit criterion 6: offline tokenizer load from the local dir."""
        cfg = AgentConfig.from_yaml(YAML_PATH)
        if not (has_module("torch") and has_module("transformers")):
            self.deep_skip("torch/transformers not installed")
        if not (cfg.asr_model_dir / "config.json").is_file():
            self.deep_skip("ASR model not downloaded")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(cfg.asr_model_dir), local_files_only=True, trust_remote_code=True
        )
        self.assertIsNotNone(tokenizer, "offline tokenizer load returned None")
        self.deep_pass(f"offline tokenizer loaded from {cfg.asr_model_dir}")


if __name__ == "__main__":
    unittest.main()
