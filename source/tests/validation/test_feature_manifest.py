"""Install manifest feature validation (Task 12 - M0.1).

Validates the ``manifest`` feature: the reproducibility record written by
``scripts/write_install_manifest.py`` (M0 spec Section 23). The committed
example (``INSTALL_MANIFEST.example.json``) must parse as JSON and carry the
exact top-level blocks ``build_manifest()`` produces, and the builder itself
must produce those blocks for the real repo (with the fake GPU block for
sandbox runs) plus a path-sorted models list.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "INSTALL_MANIFEST.example.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "write_install_manifest.py"

#: Top-level blocks produced by build_manifest() - derived from the return
#: statement of scripts/write_install_manifest.py:build_manifest().
MANIFEST_BLOCKS = (
    "generated_at", "project", "os", "python", "gpu", "torch", "voicemem",
    "models", "dependencies", "offline",
)

#: Per-model entry keys (see _collect_models() in the builder script).
MODEL_ENTRY_KEYS = ("component", "path", "size_bytes", "sha256", "hf_repo")


class ManifestFeatureTest(FeatureValidationTest):
    """INSTALL_MANIFEST schema + builder behaviour."""

    FEATURE = "manifest"

    def test_logic_example_json_schema(self):
        self.assertTrue(EXAMPLE_PATH.is_file(), "INSTALL_MANIFEST.example.json missing")
        data = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            set(data.keys()), set(MANIFEST_BLOCKS),
            "example manifest top-level blocks differ from build_manifest()",
        )
        for entry in data["models"]:
            for key in MODEL_ENTRY_KEYS:
                self.assertIn(key, entry, f"example model entry missing '{key}'")

    def test_logic_build_manifest_blocks(self):
        self.assertTrue(SCRIPT_PATH.is_file(), "scripts/write_install_manifest.py missing")
        saved_path = list(sys.path)
        try:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            import write_install_manifest as wim
        finally:
            sys.path[:] = saved_path
        manifest = wim.build_manifest(REPO_ROOT, fake_gpu=True)
        self.assertIsInstance(manifest, dict)
        self.assertEqual(
            set(manifest.keys()), set(MANIFEST_BLOCKS),
            "build_manifest() top-level blocks changed",
        )
        # Fake GPU mode fills the GPU/torch blocks with target-machine data.
        self.assertEqual(manifest["gpu"]["compute_capability"], "12.0")
        self.assertEqual(manifest["torch"]["cuda"], "12.8")
        # The models list is deterministic: path-sorted (empty in a fresh clone
        # where only README.md/.gitkeep repo docs exist).
        models = manifest["models"]
        self.assertIsInstance(models, list)
        if models:
            paths = [entry["path"] for entry in models]
            self.assertEqual(paths, sorted(paths), "models list must be path-sorted")


if __name__ == "__main__":
    unittest.main()
