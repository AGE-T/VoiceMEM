"""[v0.8.1] Behavioural tests: the release gate ORDER contract.

Post-implementation-audit P1-2: the v0.8.0 release ran its full test gate
BEFORE the VERSION bump, so the packaged tree (with the PAGE_VERSION='0.7.2'
defect) was never gated. The fix has three mechanical pieces, all tested
here with REAL code paths (no mocks of the logic under test):

  * scripts/run_release_gate.py: result parsing + failure classification
    (a failure outside the pinned sandbox env-gap baseline is a REGRESSION
    → RED; a test-count drop below the floor → RED);
  * scripts/build_release_sandbox.py::_verify_gate_record: the build
    REFUSES unless a GREEN gate record exists whose version matches the
    builder's NEW_VERSION (the gate ran AFTER the version update), whose
    source-tree fingerprint matches the CURRENT tree (nothing changed
    between gate and build), and whose page version is in sync;
  * scripts/sync_page_version.py: the page literal regenerates from the
    VERSION file (byte-level, idempotent);
  * scripts/release_tree.py: the fingerprint is deterministic and reacts
    to a single byte change in any shipped file.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import unittest

import release_tree
import run_release_gate
import sync_page_version
import build_release_sandbox


def _load_module(name: str):
    """(Re-)import one of the script modules fresh (isolated globals)."""
    return importlib.import_module(name)


class _TempTree:
    """A minimal fake repo: VERSION + web/voicemem.html (+ releases/)."""

    def __init__(self, version: str = "9.9.9",
                 page_literal: str | None = None):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="gate_order_")
        self.root = Path(self._tmp.name)
        (self.root / "VERSION").write_text(f"{version}\n", encoding="utf-8")
        (self.root / "web").mkdir()
        lit = page_literal if page_literal is not None else version
        html = (
            "<!doctype html><script>\n"
            f"const PAGE_VERSION='{lit}';\n"
            "rest of the page\n</script>\n"
        )
        (self.root / "web" / "voicemem.html").write_text(html, encoding="utf-8")
        (self.root / "releases").mkdir()

    def cleanup(self):
        self._tmp.cleanup()

    def fingerprint(self) -> str:
        fp, _ = release_tree.tree_fingerprint(self.root)
        return fp

    def write_record(self, **fields) -> Path:
        record = {
            "schema_version": 1,
            "verdict": "GREEN",
            "version": "9.9.9",
            "git_commit": "test",
            "tree_fingerprint": self.fingerprint(),
            "totals": {"tests": 1093, "failures": 0, "errors": 0, "skipped": 28},
            "deep_validation_pass": "release, runtime-deps",
            "deep_validation_skipped": 12,
            "env_gap_failures": {},
            "regressions": [],
        }
        record.update(fields)
        path = self.root / "releases" / "gate_record.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return path


def _verify(tree: _TempTree, new_version: str = "9.9.9", argv=None):
    """Run build_release_sandbox._verify_gate_record against the temp tree.

    Patches the module globals the function reads (REPO/RELEASES/
    NEW_VERSION) and restores them after — the real module stays intact.
    """
    saved = (
        build_release_sandbox.REPO,
        build_release_sandbox.RELEASES,
        build_release_sandbox.NEW_VERSION,
        sys.argv,
    )
    try:
        build_release_sandbox.REPO = tree.root
        build_release_sandbox.RELEASES = tree.root / "releases"
        build_release_sandbox.NEW_VERSION = new_version
        sys.argv = ["build_release_sandbox.py"] + (
            argv if argv is not None else []
        )
        return build_release_sandbox._verify_gate_record()
    finally:
        (
            build_release_sandbox.REPO,
            build_release_sandbox.RELEASES,
            build_release_sandbox.NEW_VERSION,
            sys.argv,
        ) = saved


class GateClassificationTests(unittest.TestCase):
    """run_release_gate.classify_failures — the verdict logic."""

    def test_env_gap_only_is_green(self):
        known = {"tests.unit.fake.EnvGapTests.test_x": "weights absent"}
        res = classify_helper(known, [("tests.unit.fake.EnvGapTests.test_x",
                                        1093)])
        self.assertTrue(res["green"])
        self.assertEqual(res["regressions"], [])
        self.assertEqual(res["env_gap_failures"],
                         ["tests.unit.fake.EnvGapTests.test_x"])

    def test_unknown_failure_is_regression_red(self):
        known = {}
        res = classify_helper(known, [("tests.unit.new.Regression.test_y",
                                        1093)])
        self.assertFalse(res["green"])
        self.assertEqual(res["regressions"], ["tests.unit.new.Regression.test_y"])

    def test_total_below_floor_is_red(self):
        res = classify_helper({}, [], total=10, floor=1090)
        self.assertFalse(res["green"])

    def test_parse_package_result(self):
        text = (
            "FF..\n"
            "======================================================================\n"
            "FAIL: test_one (tests.unit.mod.Cls.test_one)\n"
            "----------------------------------------------------------------------\n"
            "ERROR: test_two (tests.unit.mod.Cls.test_two)\n"
            "----------------------------------------------------------------------\n"
            "Ran 42 tests in 1.234s\n\n"
            "FAILED (failures=1, errors=1, skipped=3)\n"
        )
        parsed = run_release_gate._parse_package_result(text)
        self.assertEqual(parsed["total"], 42)
        self.assertEqual(parsed["failures"], 1)
        self.assertEqual(parsed["errors"], 1)
        self.assertEqual(parsed["skipped"], 3)
        self.assertEqual(
            sorted(parsed["failing_ids"]),
            ["tests.unit.mod.Cls.test_one", "tests.unit.mod.Cls.test_two"],
        )

    def test_parse_package_result_ok_with_skips(self):
        text = "Ran 7 tests in 0.5s\n\nOK (skipped=5)\n"
        parsed = run_release_gate._parse_package_result(text)
        self.assertEqual(parsed["total"], 7)
        self.assertEqual(parsed["failures"], 0)
        self.assertEqual(parsed["errors"], 0)
        self.assertEqual(parsed["skipped"], 5)
        self.assertEqual(parsed["failing_ids"], [])

    def test_parse_package_result_errors_only(self):
        text = (
            "ERROR: setUpClass (tests.integration.mod.Cls)\n"
            "Ran 63 tests in 2.0s\n\nFAILED (errors=5)\n"
        )
        parsed = run_release_gate._parse_package_result(text)
        self.assertEqual(parsed["errors"], 5)
        self.assertEqual(parsed["failing_ids"],
                         ["setUpClass (tests.integration.mod.Cls)"])


def classify_helper(known, failing, total=1093, floor=1090):
    pkg = {
        "tests/unit": {
            "total": total, "failures": len(failing), "errors": 0,
            "skipped": 0,
            "failing_ids": [t for t, _ in failing],
        }
    }
    return run_release_gate.classify_failures(pkg, known, floor)


class BuildGateRecordVerificationTests(unittest.TestCase):
    """build_release_sandbox._verify_gate_record — the build-side refusal.

    THE EXACT TREE THAT IS PACKAGED MUST BE THE TREE THAT PASSED THE GATE.
    """

    def test_missing_record_refused(self):
        tree = _TempTree()
        try:
            self.assertEqual(_verify(tree), {})
        finally:
            tree.cleanup()

    def test_red_record_refused(self):
        tree = _TempTree()
        try:
            tree.write_record(verdict="RED",
                              regressions=["tests.unit.new.Bug.test_x"])
            self.assertEqual(_verify(tree), {})
        finally:
            tree.cleanup()

    def test_version_mismatch_record_refused(self):
        """The v0.8.0 defect shape: the gate ran BEFORE the VERSION bump —
        a record whose version differs from the builder's NEW_VERSION must
        be refused (the gate must see the UPDATED version)."""
        tree = _TempTree()
        try:
            tree.write_record(version="0.8.0")   # gate saw the OLD version
            self.assertEqual(_verify(tree, new_version="9.9.9"), {})
        finally:
            tree.cleanup()

    def test_fingerprint_mismatch_refused(self):
        """A tree change AFTER the gate ran (version bump, 'small fix',
        page edit) invalidates the record — the build refuses."""
        tree = _TempTree()
        try:
            tree.write_record(tree_fingerprint="deadbeef" * 8)
            self.assertEqual(_verify(tree), {})
        finally:
            tree.cleanup()

    def test_tree_version_mismatch_refused(self):
        tree = _TempTree()
        try:
            tree.write_record()            # fingerprint of the 9.9.9 tree
            # but the tree's VERSION file now says something else
            (tree.root / "VERSION").write_text("1.2.3\n", encoding="utf-8")
            # recompute record fingerprint AFTER the change so ONLY the
            # version field can fail:
            tree.write_record(tree_fingerprint=tree.fingerprint(),
                              version="9.9.9")
            self.assertEqual(_verify(tree, new_version="9.9.9"), {})
        finally:
            tree.cleanup()

    def test_stale_page_literal_refused(self):
        """The P1-1 defect class: the page literal must equal NEW_VERSION
        in the tree being packaged."""
        tree = _TempTree(page_literal="0.7.2")
        try:
            tree.write_record(tree_fingerprint=tree.fingerprint())
            self.assertEqual(_verify(tree), {})
        finally:
            tree.cleanup()

    def test_valid_record_accepted(self):
        tree = _TempTree()
        try:
            path = tree.write_record()
            record = _verify(tree)
            self.assertEqual(record.get("verdict"), "GREEN")
            self.assertEqual(record.get("totals", {}).get("tests"), 1093)
            self.assertEqual(
                record.get("deep_validation_pass"), "release, runtime-deps")
        finally:
            tree.cleanup()

    def test_record_path_argument(self):
        tree = _TempTree()
        try:
            path = tree.write_record()
            record = _verify(tree, argv=[str(path)])
            self.assertEqual(record.get("verdict"), "GREEN")
        finally:
            tree.cleanup()


class SyncPageVersionTests(unittest.TestCase):
    """scripts/sync_page_version.py — the page literal generator."""

    def _run(self, html_text: str, version: str):
        import tempfile

        tmp = tempfile.TemporaryDirectory(prefix="sync_page_")
        root = Path(tmp.name)
        (root / "VERSION").write_text(f"{version}\n", encoding="utf-8")
        html = root / "voicemem.html"
        html.write_text(html_text, encoding="utf-8")
        saved = (
            sync_page_version.REPO, sync_page_version.HTML,
            sync_page_version.VERSION,
        )
        try:
            sync_page_version.REPO = root
            sync_page_version.HTML = html
            sync_page_version.VERSION = root / "VERSION"
            rc = sync_page_version.main()
            # bytes, not read_text: universal-newline reading would strip
            # the \r\n this class explicitly tests is preserved.
            return rc, html.read_bytes().decode("utf-8")
        finally:
            (
                sync_page_version.REPO, sync_page_version.HTML,
                sync_page_version.VERSION,
            ) = saved
            tmp.cleanup()

    def test_stale_literal_synced_from_version(self):
        rc, text = self._run("x\nconst PAGE_VERSION='0.7.2';\ny\n", "9.9.9")
        self.assertEqual(rc, 0)
        self.assertIn("const PAGE_VERSION='9.9.9';", text)
        self.assertNotIn("0.7.2", text)
        # byte-level: ONLY the literal changed
        self.assertEqual(
            text.replace("const PAGE_VERSION='9.9.9';",
                         "const PAGE_VERSION='0.7.2';", 1),
            "x\nconst PAGE_VERSION='0.7.2';\ny\n",
        )

    def test_already_synced_is_noop(self):
        original = "a\nconst PAGE_VERSION='4.5.6';\nb\n"
        rc, text = self._run(original, "4.5.6")
        self.assertEqual(rc, 0)
        self.assertEqual(text, original)

    def test_missing_literal_fails_loud(self):
        rc, text = self._run("no literal here", "1.0.0")
        self.assertEqual(rc, 1)
        self.assertEqual(text, "no literal here")

    def test_line_endings_preserved(self):
        original = "a\r\nconst PAGE_VERSION='0.1.0';\r\nb\r\n"
        rc, text = self._run(original, "2.0.0")
        self.assertEqual(rc, 0)
        self.assertIn("\r\n", text)
        # three CRLF-terminated lines in, three out (byte-level surgery:
        # only the ASCII literal is rewritten, never the line endings)
        self.assertEqual(text.count("\r\n"), 3)
        self.assertEqual(
            text,
            "a\r\nconst PAGE_VERSION='2.0.0';\r\nb\r\n",
        )


class TreeFingerprintTests(unittest.TestCase):
    """scripts/release_tree.py — deterministic, change-reactive."""

    def _tree(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory(prefix="fp_tree_")
        root = Path(tmp.name)
        (root / "VERSION").write_text("1.0.0\n", encoding="utf-8")
        (root / "CHANGELOG.md").write_text("ch\n", encoding="utf-8")
        (root / "app").mkdir()
        (root / "app" / "x.py").write_text("print(1)\n", encoding="utf-8")
        (root / "app" / "__pycache__").mkdir()
        (root / "app" / "__pycache__" / "x.pyc").write_bytes(b"junk")
        return tmp, root

    def test_deterministic(self):
        tmp, root = self._tree()
        try:
            fp1, n1 = release_tree.tree_fingerprint(root)
            fp2, n2 = release_tree.tree_fingerprint(root)
            self.assertEqual(fp1, fp2)
            self.assertEqual(n1, n2)
            # __pycache__/*.pyc never ships → not fingerprinted
            self.assertEqual(n1, 3)
        finally:
            tmp.cleanup()

    def test_single_byte_change_detected(self):
        tmp, root = self._tree()
        try:
            fp1, _ = release_tree.tree_fingerprint(root)
            (root / "app" / "x.py").write_text("print(2)\n", encoding="utf-8")
            fp2, _ = release_tree.tree_fingerprint(root)
            self.assertNotEqual(fp1, fp2)
        finally:
            tmp.cleanup()

    def test_releases_ledgers_excluded(self):
        """releases/*.json are build outputs (the builder appends to them
        after the ZIP closes) — they must not invalidate the fingerprint."""
        tmp, root = self._tree()
        try:
            fp1, _ = release_tree.tree_fingerprint(root)
            (root / "releases").mkdir()
            (root / "releases" / "RELEASE_INDEX.json").write_text("{}", encoding="utf-8")
            fp2, _ = release_tree.tree_fingerprint(root)
            self.assertEqual(fp1, fp2)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
