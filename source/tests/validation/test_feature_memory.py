"""Memory feature validation (Task 12 - M0.1).

Validates the ``memory`` feature: the VoiceMem long-term memory root of
``app.config.AgentConfig`` (repo ``memory/`` with the ``sqlite``/``qdrant``/
``backups`` subdirectories, all writable), and the degraded-mode contract of
``app.voicemem_bridge.VoiceMemBridge`` - without the ``voicemem`` package the
bridge reports ``is_available() == False`` and ``process_turn`` still returns
a ``TurnContext`` with an empty memory context (never fatal).

Deep check: when the ``voicemem`` package is installed, it must import.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from app.voicemem_bridge import TurnContext, VoiceMemBridge
from tests.validation._report import FeatureValidationTest, has_module

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

_ENV_KEYS = (
    "VOICEMEM_HOME", "VOICEMEM_MEMORY_ROOT", "EMBEDDING_MODEL_PATH",
    "VOICEMEM_EMBED_DIM", "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT",
    "OPENAI_BASE_URL", "OPENAI_MODEL",
)

MEMORY_SUBDIRS = ("sqlite", "qdrant", "backups")


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class MemoryFeatureTest(_EnvNeutralTest):
    """Memory root layout, writability and the degraded-mode bridge."""

    FEATURE = "memory"

    def test_logic_memory_root_and_write_probe(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertEqual(cfg.memory_root_path, REPO_ROOT / "memory")
        probes: list[Path] = []
        try:
            for name in MEMORY_SUBDIRS:
                target = cfg.memory_root_path / name
                self.assertTrue(target.is_dir(), f"memory/{name} directory missing")
                probe = target / ".write_probe"
                probes.append(probe)
                probe.write_text("probe", encoding="utf-8")
                self.assertTrue(probe.is_file(), f"memory/{name} is not writable")
        finally:
            for probe in probes:
                try:
                    probe.unlink()
                except FileNotFoundError:
                    pass

    def test_logic_bridge_degraded_mode(self):
        """Without the voicemem package the bridge degrades, never breaks."""
        cfg = AgentConfig.from_yaml(YAML_PATH)
        bridge = VoiceMemBridge(cfg)
        self.assertEqual(bridge.is_available(), has_module("voicemem"))
        if bridge.is_available():
            return  # real package present: the deep test covers it
        context = asyncio.run(bridge.process_turn("hello"))
        self.assertIsInstance(context, TurnContext)
        self.assertEqual(context.transcript, "hello")
        self.assertEqual(context.memory_context, "", "degraded mode: empty context")

    def test_deep_voicemem_importable(self):
        if not has_module("voicemem"):
            self.deep_skip("voicemem package not installed (degraded mode)")
        import voicemem  # noqa: F401 - present but broken -> real FAIL

        self.deep_pass("voicemem importable")


if __name__ == "__main__":
    unittest.main()
