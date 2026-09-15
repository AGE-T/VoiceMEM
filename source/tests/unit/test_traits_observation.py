"""[VM-LOCAL-013] Behavioural tests: trait observation bookkeeping + confidence
dynamics (external audit v1.0 findings F-D + F-C remainder).

Real SQLite store (tempfile), deterministic fake embedder — no mocks of the
store itself. Covers:
  * first_seen / last_seen / occurrence_count on insert + merge
  * asymptotic confidence reinforcement (never reaches 1)
  * read-time decay (grace period, half-life, legacy rows unpunished)
  * the bounded confidence term in the trait ranking priority (F-D)
  * hit metadata surfacing (confidence / occurrence / first_seen / last_seen)
  * idempotent additive schema migration on a legacy-shaped DB
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
for p in (str(REPO), str(VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import unittest

from voicemem.rightbrain.traits_store import (
    TRAIT_REINFORCE_STEP,
    TraitStore,
    Evidence,
    effective_trait_confidence,
)
from voicemem.rightbrain.brain import (
    TRAIT_CONF_RANK_FLOOR,
    _rb_trait_hits,
)
# NOTE: voicemem imports are MODULE-LEVEL on purpose: unittest discover
# imports every test module first (collection), then runs methods in order —
# by that time other tests (bridge degraded-mode simulations) may have
# removed 'voicemem' from sys.modules, which would break method-level
# imports with ModuleNotFoundError. Binding at collection time is stable.


class _FakeEmbedder:
    """Deterministic: every text maps to one of two orthogonal unit vectors.

    Texts containing 'coffee' (or anything in the same bucket) all map to
    VECTOR_A — cosine 1.0 between them, so merges are exact and predictable.
    Everything else maps to VECTOR_B (cosine 0 vs A).
    """

    DIM = 8

    def _unit(self, idx: int):
        v = [0.0] * self.DIM
        v[idx] = 1.0
        return v

    def __call__(self, text: str):
        if "coffee" in text.lower():
            return self._unit(0)
        return self._unit(1)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
        timespec="seconds")


class TraitObservationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="rb_traits_")
        self.db = os.path.join(self._tmp.name, "space.sqlite")
        self.store = TraitStore(self.db, _FakeEmbedder())

    def tearDown(self):
        self._tmp.cleanup()

    def _add(self, claim="likes coffee", quote="I like coffee"):
        return self.store.add("u1", "喜好与厌恶", claim, Evidence(quote=quote))

    # ── insert: first observation ────────────────────────────────────────────

    def test_first_observation_fields(self):
        tid = self._add()
        traits = self.store.all("u1")
        self.assertEqual(len(traits), 1)
        t = traits[0]
        self.assertEqual(t.id, tid)
        self.assertEqual(t.occurrence_count, 1)
        self.assertTrue(t.first_seen, "first_seen must be set on insert")
        self.assertTrue(t.last_seen)
        self.assertAlmostEqual(t.confidence, 0.9)

    # ── merge: occurrence increments, first_seen preserved ───────────────────

    def test_merge_increments_occurrence_and_evidence(self):
        self._add("likes coffee", quote="first time")
        time.sleep(1.1)   # evidence ordering is created_at DESC (second precision)
        self._add("likes coffee", quote="second time")
        traits = self.store.all("u1")
        self.assertEqual(len(traits), 1, "same-vector claims must merge")
        t = traits[0]
        self.assertEqual(t.occurrence_count, 2)
        self.assertEqual(len(t.evidence), 2, "evidence rows append, never replace")
        self.assertEqual(t.evidence[0].quote, "second time",
                         "most recent evidence first")

    def test_merge_preserves_first_seen(self):
        self._add("likes coffee")
        t0 = self.store.all("u1")[0].first_seen
        time.sleep(1.1)          # ensure a strictly later timestamp
        self._add("likes coffee")
        t1 = self.store.all("u1")[0]
        self.assertEqual(t1.first_seen, t0, "first_seen is immutable on merge")
        self.assertGreaterEqual(t1.last_seen, t0)

    # ── confidence reinforcement: asymptotic, never 1 ────────────────────────

    def test_confidence_asymptotic(self):
        self._add("likes coffee")
        expected = 0.9
        for i in range(2, 8):
            self._add("likes coffee")
            t = self.store.all("u1")[0]
            expected = expected + (1.0 - expected) * TRAIT_REINFORCE_STEP
            self.assertAlmostEqual(t.confidence, round(expected, 4), places=3,
                                   msg=f"after {i} observations")
            self.assertLess(t.confidence, 1.0, "never reaches 1.0")

    def test_distinct_claims_do_not_merge(self):
        self._add("likes coffee")
        self._add("hates mornings")
        self.assertEqual(len(self.store.all("u1")), 2)
        for t in self.store.all("u1"):
            self.assertEqual(t.occurrence_count, 1)

    # ── read-time decay ──────────────────────────────────────────────────────

    def test_decay_within_grace_is_identity(self):
        self.assertAlmostEqual(
            effective_trait_confidence(0.9, _iso_days_ago(30.0)), 0.9)

    def test_decay_after_grace(self):
        conf = effective_trait_confidence(0.9, _iso_days_ago(200.0))
        # 0.9 * 0.5 ** ((200-90)/180) ≈ 0.9 * 0.6536 ≈ 0.588
        self.assertAlmostEqual(conf, 0.9 * 0.5 ** ((200.0 - 90.0) / 180.0),
                               places=4)

    def test_legacy_row_without_last_seen_unpunished(self):
        self.assertEqual(effective_trait_confidence(0.9, ""), 0.9)
        self.assertEqual(effective_trait_confidence(0.9, "not-a-date"), 0.9)

    def test_eff_confidence_property(self):
        tid = self._add()
        with sqlite3.connect(self.db) as c:
            c.execute(
                "UPDATE rb_traits SET last_seen=? WHERE id=?",
                (_iso_days_ago(270.0), tid))
        t = self.store.all("u1")[0]
        # 270 days: 180 past the grace → one full halflife: 0.9 * 0.5
        self.assertAlmostEqual(t.eff_confidence, 0.45, places=4)
        self.assertAlmostEqual(t.confidence, 0.9, places=4,
                                msg="raw confidence stays unchanged in the DB")

    # ── F-D: confidence term in the ranking priority ─────────────────────────

    def test_ranking_blends_confidence(self):
        # two traits: the coffee trait (sim 1.0 to a coffee query, confidence
        # pushed to 0.99) and an orthogonal trait (sim 0.0 — must be filtered
        # by the min-sim gate BEFORE the blend, per the vendor contract).
        a = self._add("likes coffee")
        b = self._add("loves coffee too")
        self.assertEqual(a, b, "same-vector claims merge — need distinct vecs")
        # give the second trait a different vector bucket by direct insert
        with sqlite3.connect(self.db) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT id, embedding FROM rb_traits").fetchall()
            self.assertEqual(len(rows), 1)
            import numpy as np
            other = np.zeros(_FakeEmbedder.DIM, dtype=np.float32)
            other[1] = 1.0
            tid = rows[0]["id"]
            c.execute(
                "INSERT INTO rb_traits (id,user_id,slot,claim,embedding,"
                "confidence,created_at,updated_at,first_seen,last_seen,"
                "occurrence_count) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (tid + "x", "u1", "喜好与厌恶", "hates mornings",
                 other.tobytes(), 0.5, _iso_now(), _iso_now(),
                 _iso_now(), _iso_days_ago(300.0), 1))
            # push the coffee trait's confidence up (as if confirmed often)
            c.execute("UPDATE rb_traits SET confidence=0.99 WHERE id=?", (tid,))

        hits = _rb_trait_hits(self.store, "u1", "coffee?")
        # the 0-sim trait is cut by the min-sim gate (dim 8 -> 0.45) — only
        # the coffee trait remains, blended with its 0.99 confidence.
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].metadata["claim"], "likes coffee")
        self.assertAlmostEqual(
            hits[0].priority,
            round(1.0 * (TRAIT_CONF_RANK_FLOOR
                         + (1 - TRAIT_CONF_RANK_FLOOR) * 0.99), 3),
            places=3)

    def test_ranking_floor_never_zeroes(self):
        a = self._add("likes coffee")
        with sqlite3.connect(self.db) as c:
            c.execute("UPDATE rb_traits SET confidence=0.0, last_seen=? WHERE id=?",
                      (_iso_days_ago(1000.0), a))
        hits = _rb_trait_hits(self.store, "u1", "coffee")
        self.assertEqual(len(hits), 1)
        # sim 1.0 × floor 0.75 → priority exactly 0.75, never 0
        self.assertAlmostEqual(hits[0].priority, 0.75, places=3)

    def test_hit_metadata_surfaced(self):
        self._add("likes coffee")
        self._add("likes coffee")
        hits = _rb_trait_hits(self.store, "u1", "coffee")
        self.assertEqual(len(hits), 1)
        md = hits[0].metadata
        for key in ("confidence", "eff_confidence", "occurrence_count",
                    "first_seen", "last_seen"):
            self.assertIn(key, md)
        self.assertEqual(md["occurrence_count"], 2)

    # ── legacy DB migration ──────────────────────────────────────────────────

    def test_legacy_schema_migrated(self):
        # build an OLD-shaped db (no observation columns), then reopen
        self._tmp2 = tempfile.TemporaryDirectory(prefix="rb_traits_legacy_")
        db2 = os.path.join(self._tmp2.name, "legacy.sqlite")
        with sqlite3.connect(db2) as c:
            c.execute("""CREATE TABLE rb_traits (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, slot TEXT NOT NULL,
                claim TEXT NOT NULL, embedding BLOB,
                confidence REAL NOT NULL DEFAULT 0.9,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            c.execute("""CREATE TABLE rb_evidence (
                id TEXT PRIMARY KEY, trait_id TEXT NOT NULL, user_id TEXT NOT NULL,
                quote TEXT NOT NULL, emotion TEXT NOT NULL DEFAULT '',
                cause TEXT NOT NULL DEFAULT '', cause_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL)""")
            import numpy as np
            v = np.zeros(_FakeEmbedder.DIM, dtype=np.float32)
            v[0] = 1.0
            c.execute(
                "INSERT INTO rb_traits (id,user_id,slot,claim,embedding,"
                "confidence,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                ("legacy1", "u1", "喜好与厌恶", "likes coffee", v.tobytes(),
                 0.85, _iso_now(), _iso_now()))
            # real legacy traits always carry >=1 evidence row (add() appends
            # one on every write) — and vendor all() lists only evidenced traits
            c.execute(
                "INSERT INTO rb_evidence (id,trait_id,user_id,quote,emotion,"
                "cause,cause_id,created_at) VALUES (?,?,?,?,?,?,?,?)",
                ("ev1", "legacy1", "u1", "legacy quote", "", "", "",
                 _iso_now()))
        store2 = TraitStore(db2, _FakeEmbedder())          # migration runs here
        traits = store2.all("u1")
        self.assertEqual(len(traits), 1)
        t = traits[0]
        self.assertEqual(t.occurrence_count, 1, "legacy row defaults to 1")
        self.assertEqual(t.first_seen, "", "legacy row has no first_seen")
        self.assertAlmostEqual(effective_trait_confidence(t.confidence,
                                                           t.last_seen), 0.85)
        # a fresh merge on the legacy row adopts the bookkeeping going forward
        store2.add("u1", "喜好与厌恶", "likes coffee",
                   Evidence(quote="new observation"))
        t2 = store2.all("u1")[0]
        self.assertEqual(t2.occurrence_count, 2)
        self.assertTrue(t2.first_seen, "first_seen backfills on first merge")
        self.assertAlmostEqual(t2.confidence, 0.85 + 0.15 * TRAIT_REINFORCE_STEP,
                               places=3)
        # re-open again: migration is idempotent
        TraitStore(db2, _FakeEmbedder())
        self._tmp2.cleanup()


if __name__ == "__main__":
    unittest.main()
