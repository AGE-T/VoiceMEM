"""Unit tests for scripts/assess_trait_embeddings.py (Phase 14 tooling).

The tool is READ-ONLY by contract: every test drives it against synthetic
sqlite files and asserts (a) correct counting of traits / NULL embeddings /
dimension histograms, (b) the open read-only connection mode, (c) zero
writes anywhere, (d) clean behaviour on an empty memory root.
"""

from __future__ import annotations

import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.assess_trait_embeddings import (  # noqa: E402
    _assess_db,
    _memory_root_from_config,
    _space_dbs,
    main,
)

TOOL = REPO_ROOT / "scripts" / "assess_trait_embeddings.py"

_SCHEMA = """
CREATE TABLE rb_traits (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, slot TEXT NOT NULL,
    claim TEXT NOT NULL, embedding BLOB, confidence REAL NOT NULL DEFAULT 0.9,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE rb_evidence (
    id TEXT PRIMARY KEY, trait_id TEXT NOT NULL, user_id TEXT NOT NULL,
    quote TEXT NOT NULL, emotion TEXT NOT NULL, cause TEXT NOT NULL,
    cause_id TEXT NOT NULL, created_at TEXT NOT NULL);
"""


def _vec(dim: int) -> bytes:
    return struct.pack(f"{dim}f", *([0.01] * dim))


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.execute("INSERT INTO rb_traits VALUES "
                 "('t1','u1','slot','claim A',NULL,0.9,'x','x')")
    conn.execute("INSERT INTO rb_traits VALUES "
                 "('t2','u1','slot','claim B',NULL,0.9,'x','x')")
    conn.execute("INSERT INTO rb_traits VALUES "
                 "('t3','u2','slot','claim C',?,0.9,'x','x')", (_vec(384),))
    conn.execute("INSERT INTO rb_traits VALUES "
                 "('t4','u2','slot','claim D',?,0.9,'x','x')", (_vec(1536),))
    conn.commit()
    conn.close()


class AssessTraitEmbeddingsTests(unittest.TestCase):
    def test_counts_and_histogram(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "space.sqlite"
            _make_db(db)
            report = _assess_db(db)
            self.assertEqual(report["traits_total"], 4)
            self.assertEqual(report["null_embeddings"], 2)
            self.assertEqual(report["dimension_histogram"],
                             {"384": 1, "1536": 1})
            by_user = {u["user_id"]: u for u in report["users"]}
            self.assertEqual(by_user["u1"]["null_embeddings"], 2)
            self.assertEqual(by_user["u2"]["dimensions"], {"384": 1, "1536": 1})

    def test_discovery_finds_only_rb_traits_dbs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_db(root / "space.sqlite")
            # an unrelated sqlite without the table
            other = sqlite3.connect(root / "other.sqlite")
            other.execute("CREATE TABLE foo (id TEXT)")
            other.commit()
            other.close()
            found = _space_dbs(root)
            self.assertEqual([p.name for p in found], ["space.sqlite"])

    def test_read_only_connection_writes_nothing(self):
        """The tool must not modify the assessed database (ro uri mode)."""
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "space.sqlite"
            _make_db(db)
            before = db.read_bytes()
            _assess_db(db)
            self.assertEqual(db.read_bytes(), before,
                             "the assessment modified the database file")

    def test_cli_on_empty_root_is_clean(self):
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [sys.executable, str(TOOL), "--memory-root", td],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("none with rb_traits", proc.stdout)

    def test_cli_json_totals_and_notes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "memory"
            root.mkdir()
            _make_db(root / "memory.sqlite")
            proc = subprocess.run(
                [sys.executable, str(TOOL),
                 "--memory-root", str(root), "--json", "--plan"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0)
            import json
            report = json.loads(proc.stdout)   # --json stdout is pure JSON
            self.assertEqual(report["totals"],
                             {"traits": 4, "null_embeddings": 2})
            self.assertTrue(any("NULL embedding" in n for n in report["notes"]))
            self.assertTrue(any("1536" in n for n in report["notes"]))
            # the plan is printed (stderr) and explicitly non-executing
            self.assertIn("NOT executed", proc.stderr)
            self.assertIn("ROLLBACK", proc.stderr)
            self.assertIn("384", proc.stderr)   # placeholder substituted

    def test_memory_root_from_config_strips_yaml_comment(self):
        """config yaml has an inline comment after memory_root."""
        root = _memory_root_from_config(None)
        self.assertEqual(root, REPO_ROOT / "memory")
        self.assertNotIn("#", str(root))

    def test_tool_never_writes_to_real_memory_root(self):
        """Running against the REPO memory root leaves it untouched."""
        memory = REPO_ROOT / "memory"
        snapshot = {p: p.stat().st_mtime_ns
                    for p in sorted(memory.rglob("*")) if p.is_file()}
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [sys.executable, str(TOOL), "--json"],
                capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0)
        after = {p: p.stat().st_mtime_ns
                 for p in sorted(memory.rglob("*")) if p.is_file()}
        self.assertEqual(snapshot, after)


if __name__ == "__main__":
    unittest.main()
