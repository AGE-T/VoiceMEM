"""v0.4.4 field-report regression tests: the E5 offline-resolution fix.

Root cause: the runtime is fully offline by design (HF_HUB_OFFLINE=1 /
TRANSFORMERS_OFFLINE=1 in config/env.local.ps1), but voicemem's
``hf_model("embedding", ...)`` resolver only recognizes a local model when
the files sit DIRECTLY in ``<models>/embedding/`` (flat layout). Our M0
layout is the per-model subdirectory ``models/embedding/multilingual-e5-small``
→ the resolver returned the HF repo id → ``SentenceTransformer`` tried the
hub → offline connection error → memory/embedding ERROR on every turn.

Fix: :func:`app.voicemem_bridge.pin_e5_local_model` exports
``VOICEMEM_E5_MODEL`` (the resolver's highest-priority env override)
pointing at our local copy — invoked from BOTH the bridge constructor and
the web server's RealMemoryLayer; ``config/env.local.ps1`` exports the same
variable for the CLI path.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from app.config import AgentConfig
from app.voicemem_bridge import VoiceMemBridge, pin_e5_local_model


def _clear_env() -> None:
    os.environ.pop("VOICEMEM_E5_MODEL", None)


class PinE5LocalModelTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_env()

    tearDown = setUp

    def _config_with_embedding_dir(self, tmp: Path, with_config: bool = True) -> AgentConfig:
        cfg = AgentConfig()
        model_dir = tmp / "models" / "embedding" / "multilingual-e5-small"
        model_dir.mkdir(parents=True, exist_ok=True)
        if with_config:
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
        cfg.embedding_model_path = str(model_dir)
        return cfg

    def test_pins_local_dir_with_config_json(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self._config_with_embedding_dir(Path(td))
            pinned = pin_e5_local_model(cfg)
            self.assertEqual(pinned, cfg.embedding_model_path)
            self.assertEqual(os.environ["VOICEMEM_E5_MODEL"], cfg.embedding_model_path)

    def test_no_pin_without_config_json(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self._config_with_embedding_dir(Path(td), with_config=False)
            self.assertIsNone(pin_e5_local_model(cfg))
            self.assertNotIn("VOICEMEM_E5_MODEL", os.environ)

    def test_existing_env_is_never_overridden(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self._config_with_embedding_dir(Path(td))
            os.environ["VOICEMEM_E5_MODEL"] = "/custom/already/set"
            self.assertIsNone(pin_e5_local_model(cfg))
            self.assertEqual(os.environ["VOICEMEM_E5_MODEL"], "/custom/already/set")

    def test_bridge_constructor_pins(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self._config_with_embedding_dir(Path(td))
            VoiceMemBridge(cfg)
            self.assertEqual(os.environ["VOICEMEM_E5_MODEL"], cfg.embedding_model_path)

    def test_instance_alias_pins(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self._config_with_embedding_dir(Path(td))
            bridge = VoiceMemBridge(cfg)
            _clear_env()
            bridge._ensure_e5_local_model()
            self.assertEqual(os.environ["VOICEMEM_E5_MODEL"], cfg.embedding_model_path)

    def test_env_local_ps1_exports_the_variable(self):
        """The CLI path is covered by the same override in env.local.ps1."""
        raw = (Path(__file__).resolve().parents[2] / "config" / "env.local.ps1").read_text(
            encoding="ascii"
        )
        self.assertIn('VOICEMEM_E5_MODEL', raw)
        self.assertIn(
            'models\\embedding\\multilingual-e5-small',
            raw.replace("\\r\\n", "\\n"),
        )
        self.assertIn("pin_e5_local_model", raw)


if __name__ == "__main__":
    unittest.main()
