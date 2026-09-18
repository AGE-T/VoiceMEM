"""[2026-09-18] Deterministic Asian-script regression guard (architecture prep).

Implements the validation required by docs/BRITISH_ENGLISH_AND_LOCALISATION.md
section 7: product-owned code (app/, web/, config/, root product files) must
stay free of Asian-script text outside the audited compatibility allowlist;
product-owned tests/scripts/docs must stay within the audited baseline; the
canonical five memory presentation labels must remain present in English.

The scanner itself is tools/asian_script_scan.py (imported here by file path,
the same pattern used for the scripts/ imports in other tests).
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "asian_script_scan", REPO / "tools" / "asian_script_scan.py"
)
scanner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scanner)


class AsianScriptGuardTests(unittest.TestCase):
    def test_hard_fail_zone_clean(self) -> None:
        """Product-owned code/UI/config contains no unapproved Asian text."""
        result = scanner.scan()
        self.assertEqual(
            result["violations"], [],
            "Asian-script regression in product-owned text: "
            + "; ".join(str(v) for v in result["violations"][:10]),
        )

    def test_canonical_english_labels_present(self) -> None:
        """The five canonical English trait-slot labels stay in the display map."""
        web_server = (REPO / "app" / "web_server.py").read_text(encoding="utf-8")
        for label in (
            "likes and dislikes",
            "expression style",
            "thinking style",
            "coping style",
            "emotion",
        ):
            self.assertIn(label, web_server, f"canonical label missing: {label!r}")

    def test_baseline_zone_counts_reported(self) -> None:
        """The scan actually sees the audited baseline-zone files."""
        result = scanner.scan()
        self.assertTrue(
            result["baseline"],
            "baseline zone unexpectedly empty - scanner or baseline file broken",
        )

    def test_scanner_detects_planted_violation(self) -> None:
        """A planted Asian string in a fake app/ tree must be flagged."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app").mkdir()
            (root / "web").mkdir()
            (root / "config").mkdir()
            (root / "app" / "planted.py").write_text(
                'label = "喜好与厌恶"\n', encoding="utf-8"
            )
            result = scanner.scan(root)
            reasons = [v["reason"] for v in result["violations"]]
            self.assertIn(
                "unapproved-asian-text-in-product-code", reasons,
                f"planted violation not detected: {result}",
            )


if __name__ == "__main__":
    unittest.main()
