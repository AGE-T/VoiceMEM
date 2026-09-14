"""[VM-LOCAL-015] Behavioural tests: trait contradiction / supersession
/ validity semantics (v0.9.0 memory semantics, Phases 4-9 + 12 + 13).

Real SQLite store (tempfile), deterministic fake embedder — no mocks of
the store itself. The embedder is DELIBERATELY degenerate: every claim in
the same topic bucket maps to the SAME vector (cosine 1.0). That is the
HARD embedding regime — the measured evidence (docs/SEMANTIC_MATRIX.md)
shows real E5 puts semantic opposites at 0.91-0.98, i.e. mostly ABOVE the
merge threshold; cosine 1.0 is the worst case. Under that regime the
stance gate is the ONLY thing standing between a contradiction and false
reinforcement — exactly what these tests prove.

Phase 12 scenario map (test name → scenario):
  1. duplicate reinforcement ....... test_reinforcement_*
  2. paraphrase .................... test_paraphrase_same_stance_merges
  3. direct contradiction ......... test_direct_contradiction_*
  4. explicit replacement ......... test_explicit_replacement_* (fact side
                                    covered by test_fact_supersession_semantics)
  5. historical state ............. test_past_statement_*, retrieval keeps
                                    the current row first
  6. uncertain contradiction ...... test_uncertainty_*
  7. qualified statement .......... test_qualified_*
  8. Hungarian negation ........... test_hungarian_*
  9. English negation ............. test_english_*
  10. cross language .............. test_cross_language_*
  11. retrieval prefers current ... test_retrieval_*
  12. superseded recoverable ...... test_retrieval_superseded_still_queryable
  13. contradiction never raises
      positive confidence ......... test_contradiction_freezes_old_row
  14. occurrence_count is not a
      false reinforcement count ... test_occurrence_count_semantics
  15. prompt representation ....... test_prompt_suffix_semantics (the
                                    app-side render; vendor metadata here)

Phase 13 failure modes: rapid alternation, restart-between-observations,
duplicate ingest, stale/historical replay, cross-user isolation.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
for p in (str(REPO), str(VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from voicemem.rightbrain.traits_store import (  # noqa: E402
    MERGE_THRESHOLD,
    SUPERSEDE_MIN_SIM,
    SUPERSEDE_MIN_SIM_PRESUPPOSITION,
    TraitStore,
    Evidence,
)
from voicemem.rightbrain.brain import _rb_trait_hits  # noqa: E402

SLOT = "喜好与厌恶"          # the preference slot (package enum)


class _TopicEmbedder:
    """Deterministic degenerate embedder for the hardest regime.

    Every text maps to ONE of three orthogonal unit vectors by topic
    keyword — motorcycle claims, coffee claims, everything else. All
    claims in a topic are cosine 1.0 with each other (the E5 worst case:
    semantic opposites that embed identically). Cross-topic cosine is 0.
    """

    DIM = 8

    def _unit(self, idx: int):
        v = [0.0] * self.DIM
        v[idx] = 1.0
        return v

    def __call__(self, text: str):
        low = (text or "").lower()
        if "motor" in low:            # motorcycles / motorokat / motorcycl...
            return self._unit(0)
        if "coffee" in low or "kávé" in low or "kave" in low:
            return self._unit(1)
        return self._unit(2)


def _iso(days_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(
        timespec="seconds")


class TraitSemanticsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="rb_sem_")
        self.db = os.path.join(self._tmp.name, "space.sqlite")
        self.store = TraitStore(self.db, _TopicEmbedder())

    def tearDown(self):
        self._tmp.cleanup()

    # ── helpers ───────────────────────────────────────────────────────────

    def _rows(self, user="u1"):
        with sqlite3.connect(self.db) as c:
            c.row_factory = sqlite3.Row
            return c.execute(
                "SELECT * FROM rb_traits WHERE user_id=? ORDER BY created_at, id",
                (user,)).fetchall()

    def _active(self, user="u1"):
        return [r for r in self._rows(user) if not (r["superseded_by"] or "")]

    def _add(self, claim, quote=None, at=None, user="u1", slot=SLOT):
        return self.store.add(
            user, slot, claim,
            Evidence(quote=quote if quote is not None else claim, at=at or ""))

    # ── 1. duplicate reinforcement (unchanged v0.8.0 semantics) ──────────

    def test_reinforcement_merges_and_increments(self):
        a = self._add("likes motorcycles")
        b = self._add("likes motorcycles")
        self.assertEqual(a, b, "same-stance duplicate must merge")
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["occurrence_count"], 2)
        self.assertAlmostEqual(rows[0]["confidence"], 0.9 + 0.1 * 0.30, places=3)
        self.assertEqual(rows[0]["stance"], "pos")

    def test_paraphrase_same_stance_merges(self):
        # Scenario 2: paraphrase with the same stance uses the EXISTING
        # similarity semantics (cosine 1.0 here → merge; the measured real
        # E5 value is 0.9764 — also above the threshold).
        a = self._add("likes motorcycles")
        b = self._add("enjoys riding motorcycles")
        self.assertEqual(a, b)
        self.assertEqual(len(self._rows()), 1)
        self.assertEqual(self._rows()[0]["occurrence_count"], 2)

    # ── 3. direct contradiction: supersede, never reinforce ──────────────

    def test_direct_contradiction_supersedes_not_reinforces(self):
        old = self._add("likes motorcycles", quote="I like motorcycles")
        conf_before = 0.9
        occ_before = 1
        time.sleep(1.05)              # distinct superseded_at
        new = self._add("no longer likes motorcycles",
                        quote="I no longer like motorcycles")
        self.assertNotEqual(old, new, "a contradiction must NOT merge")
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(len(rows), 2, "supersede = new current row + marked old row")
        # the old row: frozen bookkeeping + supersession mark, text intact
        self.assertEqual(rows[old]["superseded_by"], new)
        self.assertTrue(rows[old]["superseded_at"])
        self.assertEqual(rows[old]["claim"], "likes motorcycles")
        self.assertEqual(rows[old]["occurrence_count"], occ_before,
                         "contradiction must not increment occurrence_count")
        self.assertAlmostEqual(rows[old]["confidence"], conf_before, places=6,
                               msg="contradiction must not raise confidence")
        # the new row: current, own stance, fresh bookkeeping
        self.assertEqual(rows[new]["stance"], "neg")
        self.assertEqual(rows[new]["occurrence_count"], 1)
        self.assertEqual(rows[new]["supersedes"], old,
                         "the new row records what it replaced")
        self.assertFalse(rows[new]["superseded_by"])

    def test_contradiction_freezes_old_row(self):
        # Scenario 13: confidence never rises from a contradiction.
        self._add("likes motorcycles")
        self._add("likes motorcycles")            # conf → 0.93, occ 2
        mid = self._rows()[0]
        self.assertEqual(mid["occurrence_count"], 2)
        self.assertAlmostEqual(mid["confidence"], 0.93, places=3)
        time.sleep(1.05)
        self._add("no longer likes motorcycles")
        rows = {r["claim"]: r for r in self._rows()}
        self.assertEqual(rows["likes motorcycles"]["occurrence_count"], 2)
        self.assertAlmostEqual(rows["likes motorcycles"]["confidence"], 0.93,
                               places=3)

    def test_occurrence_count_semantics(self):
        # Scenario 14: occurrence_count counts AGREEMENTS only — a
        # contradiction is not a confirmation, and the negative state's
        # own re-observations DO count on the negative row.
        self._add("likes motorcycles")
        self._add("no longer likes motorcycles")
        self._add("no longer likes motorcycles")  # re-confirmed negative
        rows = {r["claim"]: r for r in self._rows()}
        self.assertEqual(rows["likes motorcycles"]["occurrence_count"], 1)
        self.assertEqual(rows["no longer likes motorcycles"]["occurrence_count"], 2,
                         "agreement with the negative state counts on its row")
        self.assertFalse(rows["no longer likes motorcycles"]["superseded_by"])

    # ── 4. explicit replacement (trait form) ──────────────────────────────

    def test_explicit_replacement_gave_up(self):
        old = self._add("likes coffee", quote="I love my coffee")
        time.sleep(1.05)
        new = self._add("gave up coffee", quote="I gave up coffee last month")
        self.assertNotEqual(old, new)
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(rows[old]["superseded_by"], new,
                         "presuppositional negation replaces the positive state")

    # ── 5. historical state: temporal coexistence, not contradiction ─────

    def test_past_statement_is_separate_not_contradiction(self):
        # Scenario E: "previously owned a BMW" vs "owns a Renault" are NOT
        # contradictions; a past statement never flips a current one.
        a = self._add("likes motorcycles", quote="I like motorcycles")
        b = self._add("used to like motorcycles", quote="I used to like motorcycles")
        self.assertNotEqual(a, b)
        rows = {r["claim"]: r for r in self._rows()}
        self.assertFalse(rows["likes motorcycles"]["superseded_by"],
                         "a past statement must not supersede the current one")
        self.assertFalse(rows["used to like motorcycles"]["superseded_by"])
        self.assertEqual(rows["used to like motorcycles"]["stance"], "past")

    def test_past_statement_then_current_positive_stays_separate(self):
        self._add("used to like motorcycles")
        self._add("likes motorcycles")             # current statement
        rows = {r["claim"]: r for r in self._rows()}
        self.assertEqual(len(rows), 2,
                         "current positive must not merge into the past-pattern row")

    # ── 6. uncertain contradiction: deferred, never a hard flip ──────────

    def test_uncertainty_defers_resolution(self):
        # Scenario G: "might no longer like motorcycles" must not become a
        # hard replacement — separate state, both observable.
        a = self._add("likes motorcycles")
        b = self._add("might no longer like motorcycles",
                      quote="Thomas might no longer like motorcycles")
        self.assertNotEqual(a, b)
        rows = {r["claim"]: r for r in self._rows()}
        self.assertFalse(rows["likes motorcycles"]["superseded_by"])
        self.assertEqual(rows["might no longer like motorcycles"]["stance"],
                         "uncertain")

    # ── 7. qualified statement: scoped, not global contradiction ────────

    def test_qualified_statement_is_separate(self):
        # Scenario F: "except during winter" is scoped — no evidence of a
        # global contradiction.
        a = self._add("likes motorcycles")
        b = self._add("likes motorcycles except during winter")
        self.assertNotEqual(a, b)
        rows = {r["claim"]: r for r in self._rows()}
        self.assertFalse(rows["likes motorcycles"]["superseded_by"])
        self.assertEqual(rows["likes motorcycles except during winter"]["stance"],
                         "qualified")

    # ── 8. Hungarian negation ─────────────────────────────────────────────

    def test_hungarian_negation_supersedes(self):
        old = self._add("szereti a motorokat",
                        quote="Thomas szereti a motorokat")
        time.sleep(1.05)
        new = self._add("már nem szereti a motorokat",
                        quote="Thomas már nem szereti a motorokat")
        self.assertNotEqual(old, new)
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(rows[old]["superseded_by"], new)

    def test_hungarian_past_is_separate(self):
        a = self._add("szereti a motorokat")
        b = self._add("régen szerette a motorokat",
                      quote="Thomas régen szerette a motorokat")
        rows = {r["claim"]: r for r in self._rows()}
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows["szereti a motorokat"]["superseded_by"])

    def test_hungarian_uncertainty_defers(self):
        self._add("szereti a motorokat")
        self._add("talán már nem szereti a motorokat")
        rows = {r["claim"]: r for r in self._rows()}
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows["szereti a motorokat"]["superseded_by"])

    def test_hungarian_qualified_defers(self):
        self._add("szereti a motorokat")
        self._add("szereti a motorokat, kivéve télen")
        rows = {r["claim"]: r for r in self._rows()}
        self.assertFalse(rows["szereti a motorokat"]["superseded_by"])

    # ── 9. English negation + antipathy ───────────────────────────────────

    def test_english_negation_supersedes(self):
        old = self._add("likes motorcycles")
        new = self._add("does not like motorcycles")
        self.assertNotEqual(old, new)
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(rows[old]["superseded_by"], new)
        self.assertFalse(rows[new]["superseded_by"])

    def test_antipathy_supersedes_without_negation_word(self):
        # The measured 0.9685 pair — pre-v0.9.0 this MERGED and reinforced
        # the positive trait. Now: flip.
        old = self._add("Thomas szereti a motorokat",
                        quote="Thomas szereti a motorokat")
        new = self._add("Thomas utálja a motorokat",
                        quote="Thomas utálja a motorokat")
        self.assertNotEqual(old, new)
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(rows[old]["superseded_by"], new)

    # ── 10. cross language contradiction ──────────────────────────────────

    def test_cross_language_contradiction_supersedes(self):
        # HU positive then EN negative: the CLAIMS are what the store
        # compares (English patterns in the real pipeline); the quotes
        # carry the language truth. Cosine 1.0 (degenerate) + opposite
        # stance + topic overlap → supersede. Never reinforcement.
        old = self._add("likes motorcycles",
                        quote="Thomas szereti a motorokat")
        time.sleep(1.05)
        new = self._add("no longer likes motorcycles",
                        quote="Thomas no longer likes motorcycles")
        self.assertNotEqual(old, new)
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(rows[old]["superseded_by"], new)
        evs = self._evidence(new)
        self.assertEqual(evs[0].quote, "Thomas no longer likes motorcycles")

    def test_cross_language_hungarian_negative_after_english_positive(self):
        old = self._add("likes motorcycles", quote="Thomas likes motorcycles")
        new = self._add("no longer likes motorcycles",
                        quote="Thomas már nem szereti a motorokat")
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(rows[old]["superseded_by"], new)

    def _evidence(self, trait_id):
        with sqlite3.connect(self.db) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM rb_evidence WHERE trait_id=? ORDER BY created_at",
                (trait_id,)).fetchall()
        return [Evidence(quote=r["quote"], emotion=r["emotion"],
                         cause=r["cause"], cause_id=r["cause_id"],
                         at=r["created_at"]) for r in rows]

    # ── flip-back chains (Phase 13 rapid alternation) ────────────────────

    def test_rapid_alternation_builds_chain_current_is_last(self):
        pos = self._add("likes motorcycles")
        neg = self._add("no longer likes motorcycles")
        pos2 = self._add("likes motorcycles again",
                         quote="Thomas likes motorcycles again")
        neg2 = self._add("no longer likes motorcycles",
                         quote="changed his mind again")
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(len(rows), 4, "every flip appends — history preserved")
        # chain: pos ← neg ← pos2 ← neg2 (current)
        self.assertEqual(rows[pos]["superseded_by"], neg)
        self.assertEqual(rows[neg]["superseded_by"], pos2)
        self.assertEqual(rows[pos2]["superseded_by"], neg2)
        self.assertFalse(rows[neg2]["superseded_by"], "the last observation is current")
        self.assertEqual(rows[neg2]["stance"], "neg")

    def test_flip_back_then_reinforce_counts_on_current(self):
        self._add("likes motorcycles")
        self._add("no longer likes motorcycles")
        again = self._add("likes motorcycles again",
                          quote="likes motorcycles again")
        self._add("likes motorcycles")              # reinforces the CURRENT positive
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(rows[again]["occurrence_count"], 2,
                         "post-flip agreements count on the current row")
        self.assertEqual(rows[again]["stance"], "pos")

    # ── stale replay / historical replay (Phase 13) ──────────────────────

    def test_stale_replay_does_not_flip_current_state(self):
        s = self._add("likes motorcycles", at=_iso(30))      # event 30d ago
        n = self._add("no longer likes motorcycles", at=_iso(20))
        # replay of the OLD positive, event time older than the flip's
        # last observation — must NOT flip the current negative state, and
        # must NOT appear as a new current row: the replayed statement is
        # HISTORICAL evidence and attaches to the historical trait (frozen).
        self._add("likes motorcycles", at=_iso(28))
        rows = {r["id"]: r for r in self._rows()}
        self.assertEqual(len(rows), 2,
                         "the replay attaches evidence, no new current row")
        self.assertFalse(rows[n]["superseded_by"],
                         "the current negative state is untouched")
        self.assertEqual(rows[n]["claim"], "no longer likes motorcycles")
        # the historical row gained a second evidence row, still frozen
        evs = self._evidence(s)
        self.assertEqual(len(evs), 2, "the replay is recorded as evidence")
        self.assertEqual(rows[s]["occurrence_count"], 1,
                         "frozen row bookkeeping untouched by the replay")
        active = [r for r in rows.values() if not r["superseded_by"]]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["claim"], "no longer likes motorcycles")

    def test_untimestamped_replay_flips_documented_failopen(self):
        # Without event times the guard is fail-open on time (the stance
        # gate is the hard safety) — pinned here as DOCUMENTED behaviour.
        self._add("likes motorcycles")
        self._add("no longer likes motorcycles")
        self._add("likes motorcycles")                        # no timestamps
        current = [r for r in self._rows() if not (r["superseded_by"] or "")]
        self.assertEqual(len(current), 1)

    # ── restart between observations (Phase 13) ───────────────────────────

    def test_state_survives_store_restart(self):
        self._add("likes motorcycles")
        n = self._add("no longer likes motorcycles")
        # a NEW store instance over the same sqlite (process restart)
        store2 = TraitStore(self.db, _TopicEmbedder())
        r = store2.add("u1", SLOT, "likes motorcycles",
                       Evidence(quote="likes motorcycles"))
        rows = {r_["id"]: r_ for r_ in self._rows()}
        self.assertFalse(rows[r]["superseded_by"], "restart add is current")
        self.assertEqual(rows[r]["claim"], "likes motorcycles")
        self.assertEqual(rows[n]["superseded_by"], r,
                         "the negative state was flipped on restart (no timestamps → fail-open)")
        active = [x for x in rows.values() if not x["superseded_by"]]
        self.assertEqual(len(active), 1)

    # ── different speakers / users never cross (Phase 13) ────────────────

    def test_users_are_isolated(self):
        self._add("likes motorcycles", user="alice")
        b = self._add("no longer likes motorcycles", user="bob")
        a_rows = self._rows(user="alice")
        self.assertEqual(len(a_rows), 1, "bob's contradiction must not touch alice")
        self.assertFalse(a_rows[0]["superseded_by"])
        self.assertEqual(len(self._rows(user="bob")), 1)

    # ── 11/12. retrieval semantics ────────────────────────────────────────

    def test_retrieval_prefers_current_state(self):
        # Scenario 11: a current-state query must not rank the superseded
        # memory above the current one.
        self._add("likes motorcycles")
        self._add("no longer likes motorcycles")
        scored = self.store.search_scored("u1", "motorcycles")
        self.assertEqual(len(scored), 2, "historical row stays queryable")
        first, second = scored
        self.assertEqual(first[0].claim, "no longer likes motorcycles",
                         "current state ranks first")
        self.assertFalse(first[0].superseded_by)
        self.assertEqual(second[0].claim, "likes motorcycles")
        self.assertTrue(second[0].superseded_by)

    def test_retrieval_superseded_still_queryable(self):
        # Scenario 12: superseded state remains historically recoverable.
        self._add("likes motorcycles")
        self._add("no longer likes motorcycles")
        hits = self.store.search("u1", "likes motorcycles")
        claims = {t.claim for t in hits}
        self.assertIn("likes motorcycles", claims,
                      "the superseded row is still retrievable")

    def test_superseded_row_never_reinforced_after_flip(self):
        self._add("likes motorcycles")
        self._add("no longer likes motorcycles")
        # a third neutral observation matching the topic reinforces the
        # CURRENT row, never the superseded one
        self._add("motorcycles are his thing", quote="motorcycles are his thing")
        rows = {r["claim"]: r for r in self._rows()}
        self.assertEqual(rows["likes motorcycles"]["occurrence_count"], 1,
                         "the superseded row's bookkeeping stays frozen")

    # ── 15. prompt / metadata representation ──────────────────────────────

    def test_hit_metadata_carries_semantic_state(self):
        self._add("likes motorcycles")
        self._add("no longer likes motorcycles")
        hits = _rb_trait_hits(self.store, "u1", "motorcycles")
        self.assertEqual(len(hits), 2)
        by_claim = {h.metadata["claim"]: h for h in hits}
        old = by_claim["likes motorcycles"]
        new = by_claim["no longer likes motorcycles"]
        self.assertTrue(old.metadata["superseded_by"])
        self.assertEqual(old.metadata["stance"], "pos")
        self.assertEqual(new.metadata["stance"], "neg")
        self.assertTrue(new.metadata["supersedes"])
        # demotion: the superseded hit's priority is scaled by 0.75
        self.assertLess(old.priority, new.priority)

    # ── topic guard: the supersede band never crosses topics ─────────────

    def test_topic_guard_blocks_cross_topic_supersede(self):
        # Measured noise pair ("likes motorcycles" ↔ "no longer likes
        # bicycles" = 0.8876, inside the presupposition band): with the
        # degenerate embedder this is cosine 1.0 — ONLY the topic guard
        # blocks the wrongful supersede.
        a = self._add("likes motorcycles")
        b = self._add("no longer likes bicycles", quote="no longer likes bicycles")
        self.assertNotEqual(a, b)
        rows = {r["claim"]: r for r in self._rows()}
        self.assertFalse(rows["likes motorcycles"]["superseded_by"],
                         "different topic must never be superseded")

    # ── band boundaries (measured constants) ──────────────────────────────

    def test_bands_are_the_measured_values(self):
        self.assertEqual(MERGE_THRESHOLD, 0.95)
        self.assertEqual(SUPERSEDE_MIN_SIM, 0.90)
        self.assertEqual(SUPERSEDE_MIN_SIM_PRESUPPOSITION, 0.88)

    def test_below_band_contradiction_is_separate(self):
        # Even opposite stance + overlapping topic: below the supersede
        # band there is no basis to flip — separate (embedder gives 0
        # cross-topic cosine; use the third bucket for a non-matching topic).
        self._add("likes motorcycles")
        self._add("hates long meetings")            # cosine 0 — different bucket
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["superseded_by"])

    # ── legacy migration ──────────────────────────────────────────────────

    def test_legacy_db_gets_semantic_columns_idempotently(self):
        self._tmp2 = tempfile.TemporaryDirectory(prefix="rb_sem_legacy_")
        db2 = os.path.join(self._tmp2.name, "legacy.sqlite")
        with sqlite3.connect(db2) as c:
            c.execute("""CREATE TABLE rb_traits (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, slot TEXT NOT NULL,
                claim TEXT NOT NULL, embedding BLOB,
                confidence REAL NOT NULL DEFAULT 0.9,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                first_seen TEXT NOT NULL DEFAULT '',
                last_seen TEXT NOT NULL DEFAULT '',
                occurrence_count INTEGER NOT NULL DEFAULT 1)""")
            c.execute("""CREATE TABLE rb_evidence (
                id TEXT PRIMARY KEY, trait_id TEXT NOT NULL, user_id TEXT NOT NULL,
                quote TEXT NOT NULL, emotion TEXT NOT NULL DEFAULT '',
                cause TEXT NOT NULL DEFAULT '', cause_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL)""")
            import numpy as np
            v = np.zeros(_TopicEmbedder.DIM, dtype=np.float32)
            v[0] = 1.0
            c.execute(
                "INSERT INTO rb_traits (id,user_id,slot,claim,embedding,"
                "confidence,created_at,updated_at,first_seen,last_seen,"
                "occurrence_count) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("legacy1", "u1", SLOT, "likes motorcycles", v.tobytes(),
                 0.9, _iso(100), _iso(100), _iso(100), _iso(100), 3))
            c.execute(
                "INSERT INTO rb_evidence (id,trait_id,user_id,quote,emotion,"
                "cause,cause_id,created_at) VALUES (?,?,?,?,?,?,?,?)",
                ("ev1", "legacy1", "u1", "legacy quote", "", "", "", _iso(100)))
        store2 = TraitStore(db2, _TopicEmbedder())           # migration runs
        # a negation on a legacy row: stance inferred from the stored claim
        store2.add("u1", SLOT, "no longer likes motorcycles",
                   Evidence(quote="I no longer like motorcycles"))
        with sqlite3.connect(db2) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM rb_traits ORDER BY created_at").fetchall()
        self.assertEqual(len(rows), 2)
        legacy = next(r for r in rows if r["id"] == "legacy1")
        fresh = next(r for r in rows if r["id"] != "legacy1")
        self.assertEqual(legacy["superseded_by"], fresh["id"],
                         "legacy positive row is superseded (stance inferred from claim)")
        self.assertEqual(legacy["occurrence_count"], 3, "legacy bookkeeping frozen")
        self.assertEqual(fresh["stance"], "neg")
        TraitStore(db2, _TopicEmbedder())                     # idempotent re-open
        self._tmp2.cleanup()


if __name__ == "__main__":
    unittest.main()
