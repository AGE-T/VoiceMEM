"""Unit tests for scripts/voicemem_vendor_pin.py (controlled-vendor pin).

The pin is the machine-verifiable identity of the source-owned VoiceMem
tree: upstream base (repo/tag/commit), local-fix count, and a content
fingerprint. Tests cover determinism, junk exclusion, tamper detection,
missing-tree/missing-pin failures, and the CLI exit codes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.voicemem_vendor_pin import (  # noqa: E402
    PIN_FILENAME,
    fingerprint,
    file_count,
    load_pin,
    verify,
    write_pin,
)

UPSTREAM = {
    "upstream_repo": "https://github.com/xzf-thu/VoiceMem",
    "upstream_tag": "v0.0.1",
    "upstream_commit": "e8384e087bd2f44eb05fc7ae1a3c525ea8244179",
    "base_taken": "2026-09",
    "local_fixes": 6,
}


def _make_vendor(root: Path) -> Path:
    vendor = root / "vendor" / "voicemem"
    (vendor / "voicemem").mkdir(parents=True)
    (vendor / "pyproject.toml").write_text("[project]\nname='voicemem'\n",
                                           encoding="utf-8")
    (vendor / "voicemem" / "__init__.py").write_text("", encoding="utf-8")
    (vendor / "PROVENANCE.md").write_text("doc", encoding="utf-8")
    return vendor


class PinMechanicsTests(unittest.TestCase):
    def test_fingerprint_deterministic_and_content_sensitive(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            v1 = _make_vendor(Path(td))
            fp1 = fingerprint(v1)
            self.assertEqual(fingerprint(v1), fp1)      # stable
            self.assertGreater(file_count(v1), 0)

            # same content elsewhere -> same fingerprint
            v2_dir = Path(td) / "copy"
            v2_dir.mkdir()
            for f in v1.rglob("*"):
                if f.is_file():
                    dst = v2_dir / f.relative_to(v1)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(f.read_bytes())
            self.assertEqual(fingerprint(v2_dir), fp1)

            # content change -> different fingerprint
            (v1 / "voicemem" / "__init__.py").write_text("x", encoding="utf-8")
            self.assertNotEqual(fingerprint(v1), fp1)

    def test_fingerprint_ignores_runtime_junk_and_itself(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            v = _make_vendor(Path(td))
            fp = fingerprint(v)
            (v / "voicemem" / "__pycache__").mkdir()
            (v / "voicemem" / "__pycache__" / "m.cpython-312.pyc").write_bytes(
                b"junk")
            (v / "voicemem" / "leftover.pyc").write_bytes(b"junk")
            (v / ".clone_ref").write_text("v0.0.1\n", encoding="utf-8")
            (v / PIN_FILENAME).write_text("{}", encoding="utf-8")
            self.assertEqual(fingerprint(v), fp)   # junk invisible

    def test_write_then_verify_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            v = _make_vendor(Path(td))
            pin = write_pin(v, **UPSTREAM)
            self.assertEqual(pin["source"], "controlled-vendor")
            ok, problems, loaded = verify(v)
            self.assertTrue(ok, problems)
            self.assertEqual(loaded["upstream_commit"],
                             UPSTREAM["upstream_commit"])

    def test_tamper_detection(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            v = _make_vendor(Path(td))
            write_pin(v, **UPSTREAM)
            (v / "voicemem" / "__init__.py").write_text(
                "tampered", encoding="utf-8")
            ok, problems, _ = verify(v)
            self.assertFalse(ok)
            self.assertTrue(any("MISMATCH" in p for p in problems))

    def test_partial_extraction_detected(self):
        """A partially copied vendor tree fails verification (the release
        self-check and installer both rely on this)."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            v = _make_vendor(Path(td))
            write_pin(v, **UPSTREAM)
            (v / "PROVENANCE.md").unlink()
            ok, problems, _ = verify(v)
            self.assertFalse(ok)
            self.assertTrue(any("PROVENANCE.md" in p for p in problems))

    def test_missing_pin_reports_clearly(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            v = _make_vendor(Path(td))
            ok, problems, loaded = verify(v)
            self.assertFalse(ok)
            self.assertIsNone(loaded)
            self.assertTrue(any(PIN_FILENAME in p for p in problems))

    def test_missing_tree_fails_without_raising(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            ok, problems, _ = verify(Path(td) / "nope")
            self.assertFalse(ok)
            self.assertTrue(problems)

    def test_load_pin_rejects_broken_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            v = _make_vendor(Path(td))
            (v / PIN_FILENAME).write_text("{broken", encoding="utf-8")
            self.assertIsNone(load_pin(v))


class PinCliTests(unittest.TestCase):
    """CLI contract used by the installer / verify_m1 / release builds."""

    def test_repo_pin_verifies_via_cli(self):
        """The REAL repo vendor tree must verify (exit 0)."""
        proc = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "scripts" / "voicemem_vendor_pin.py")],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("OK", proc.stdout)

    def test_cli_fails_on_missing_root(self):
        proc = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "scripts" / "voicemem_vendor_pin.py"),
             "--root", "/nonexistent/voicemem"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("not found", proc.stdout)

    def test_real_repo_pin_fields(self):
        """The shipped pin carries the documented upstream base identity."""
        pin = load_pin(REPO_ROOT / "vendor" / "voicemem")
        self.assertIsNotNone(pin)
        self.assertEqual(pin["source"], "controlled-vendor")
        self.assertEqual(pin["upstream_tag"], "v0.0.1")
        self.assertEqual(pin["upstream_commit"],
                         "e8384e087bd2f44eb05fc7ae1a3c525ea8244179")
        self.assertEqual(pin["local_fixes"], 7)   # PROVENANCE.md row count
        self.assertTrue(pin["fingerprint"])
        # JSON is deterministic + sorted (diff-friendly)
        raw = (REPO_ROOT / "vendor" / "voicemem" / PIN_FILENAME).read_text(
            encoding="utf-8")
        self.assertEqual(raw, json.dumps(json.loads(raw), indent=2,
                                          sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
