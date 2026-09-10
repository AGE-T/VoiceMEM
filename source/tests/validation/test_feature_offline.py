"""Offline guarantee feature validation (Task 12 - M0.1).

Validates the ``offline`` feature: runtime must be fully local - the config
declares ``offline: true`` with the llama-server endpoint bound to loopback
only, the env templates (``config/.env.example``, ``env.local.ps1``,
``env.local.sh``) switch the HuggingFace stack into offline mode with the
cache isolated under ``models/hf``, and no environment file points at the
real OpenAI endpoint (``api.openai.com`` must never appear).
"""
from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"
ENV_EXAMPLE = REPO_ROOT / "config" / ".env.example"
ENV_PS1 = REPO_ROOT / "config" / "env.local.ps1"
ENV_SH = REPO_ROOT / "config" / "env.local.sh"

_ENV_KEYS = (
    "VOICEMEM_HOME", "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT",
    "OPENAI_BASE_URL", "OPENAI_MODEL", "LLAMA_MODEL_PATH",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class OfflineFeatureTest(_EnvNeutralTest):
    """Fully-local runtime: offline flag + loopback endpoint + env templates."""

    FEATURE = "offline"

    def test_logic_config_offline_and_loopback(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertTrue(cfg.offline, "runtime config must declare offline: true")
        self.assertEqual(cfg.llama_server_host, "127.0.0.1", "loopback only")
        yaml_text = YAML_PATH.read_text(encoding="utf-8")
        self.assertIn("offline: true", yaml_text)
        self.assertIn('llama_server_host: "127.0.0.1"', yaml_text)

    def test_logic_env_example_offline_block(self):
        self.assertTrue(ENV_EXAMPLE.is_file(), "config/.env.example missing")
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("TRANSFORMERS_OFFLINE=1", text)
        self.assertIn("HF_HOME=models/hf", text)
        self.assertIn("OPENAI_BASE_URL=http://127.0.0.1:8080/v1", text)

    def test_logic_env_local_loaders_offline_no_real_openai(self):
        for path in (ENV_PS1, ENV_SH):
            self.assertTrue(path.is_file(), f"{path.name} missing")
            text = path.read_text(encoding="utf-8")
            # HF_HUB_OFFLINE set to 1 in each loader's own syntax.
            self.assertRegex(
                text, re.compile(r"HF_HUB_OFFLINE\s*=\s*[\"']?1"),
                f"{path.name} must set HF_HUB_OFFLINE to 1",
            )
            # No real OpenAI endpoint anywhere.
            self.assertNotIn(
                "api.openai.com", text.lower(),
                f"{path.name} must not reference the real OpenAI endpoint",
            )
        # The .env.example must not reference it either.
        self.assertNotIn("api.openai.com", ENV_EXAMPLE.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
