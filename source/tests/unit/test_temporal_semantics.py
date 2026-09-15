"""v0.10 PHASE 2/3/6/8 — temporal memory semantics (unit, deterministic).

Covers the deterministic core added in v0.10 (vendor
``voicemem/leftbrain/temporal.py`` + the ``future`` stance class + the
render markers + the ranking-tier legacy equivalence):

* PHASE 2 classes A–G — stance/bounds classification (EN+HU, quote rescue);
* PHASE 3 — observation-time parsing (ISO vs legacy time-of-day) + the
  never-invent-a-date rule;
* PHASE 4 — tier table for A→B / B→A / alternation chains;
* PHASE 6 — render markers (bridge + web copies) + payload keys;
* PHASE 8 — BACKWARD COMPATIBILITY: legacy rows (no temporal keys) keep the
  EXACT v0.9.2 sort order (randomised proof) and derive to current/superseded
  only; zero temporal keys ⇒ zero behaviour change.

The real-stack (E5 + mem0/qdrant) behavioural proofs live in
``tests/integration/test_temporal_memory.py``.
"""
from __future__ import annotations

import random
import sys
import unittest
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
for p in (REPO, VENDOR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from voicemem.leftbrain.temporal import (  # noqa: E402
    FactTemporal,
    classify_fact_temporal,
    fact_status,
    parse_iso_date,
    query_temporal_intent,
    row_status,
    temporal_rank_tier,
)
from voicemem.leftbrain.local_memory_store import MemorySearchHit  # noqa: E402

TODAY = date(2026, 9, 15)


class FactTemporalClassificationTests(unittest.TestCase):
    """PHASE 2 classes A–G (plus quote rescue + boundaries)."""

    def _ft(self, text: str, quote: str = "") -> FactTemporal:
        return classify_fact_temporal(text, quote, today=TODAY)

    # A. current state
    def test_class_a_current_positive(self):
        self.assertEqual(self._ft("Thomas likes coffee").stance, "pos")
        self.assertEqual(self._ft("Tamás szereti a kávét.").stance, "pos")

    # B. past state
    def test_class_b_past(self):
        ft = self._ft("Thomas used to like coffee")
        self.assertEqual(ft.stance, "past")
        self.assertEqual(self._ft("Tamás régen szerette a teát.").stance, "past")

    # C. cessation
    def test_class_c_no_longer(self):
        self.assertEqual(self._ft("Thomas no longer likes coffee").stance, "neg")
        self.assertEqual(self._ft("Tamás már nem szereti a kávét.").stance, "neg")

    # D. uncertain
    def test_class_d_may_not(self):
        self.assertEqual(self._ft("Thomas may not like coffee").stance, "uncertain")
        self.assertEqual(
            self._ft("Talán Tamás már nem szereti a kávét.").stance, "uncertain")

    # E. future
    def test_class_e_will(self):
        ft = self._ft("Thomas will like coffee")
        self.assertEqual(ft.stance, "future")
        self.assertEqual(ft.valid_from, "")  # indefinite future: no date invented
        self.assertEqual(self._ft("Tamás szeretni fogja a bort.").stance, "future")

    def test_class_e_future_with_date(self):
        ft = self._ft("Thomas will like coffee from 2027-01-01")
        self.assertEqual(ft.stance, "future")
        self.assertEqual(ft.valid_from, "2027-01-01")

    # F. bounded past + current negation
    def test_class_f_bounded_interval(self):
        ft = self._ft("Thomas likes coffee until 2026-01-15")
        self.assertEqual(ft.stance, "pos")
        self.assertEqual(ft.valid_until, "2026-01-15")

    # G. repeated then changed — classification side (the chain is proven
    # in the integration suite; here: the confirmation itself stays pos).
    def test_class_g_repeated_confirmation_stays_current(self):
        self.assertEqual(self._ft("Thomas likes coffee").stance, "pos")

    def test_quote_rescues_future_normalised_away_by_extractor(self):
        """The extractor normalises "will like" → "likes"; the QUOTE is the
        ground truth — the future class must survive (stance.py either-side
        rule, v0.10)."""
        self.assertEqual(
            classify_fact_temporal("Tamás szereti a bort.",
                                   "Tamás szeretni fogja a bort.", TODAY).stance,
            "future")

    def test_uncertain_beats_future(self):
        self.assertEqual(
            self._ft("Thomas will probably like coffee").stance, "uncertain")

    def test_neg_beats_future(self):
        self.assertEqual(
            self._ft("Thomas does not like coffee now but will later").stance,
            "neg")

    def test_present_tense_fact_about_future_event_is_current(self):
        """DESIGN BOUNDARY: a flight on a future date is CURRENT knowledge —
        no future cue, no future status (no invented semantics)."""
        ft = self._ft("Thomas has a flight to Paris on 2026-12-01")
        self.assertEqual(ft.stance, "")
        self.assertEqual(ft.valid_from, "")

    def test_hu_future_periphrasis_variants(self):
        for text in ("szeretni fogom", "kedvelni fogja", "jövő hónapban",
                     "meg fogja szeretni"):
            self.assertEqual(self._ft(f"Tamás {text} a kávét.").stance, "future",
                             f"cue failed: {text}")

    def test_legacy_neutral_fact_has_no_temporal_keys(self):
        ft = self._ft("Thomas works at Acme")
        self.assertEqual((ft.stance, ft.valid_from, ft.valid_until),
                         ("", "", ""))


class RowStatusTests(unittest.TestCase):
    """PHASE 2/4 — derived status (structural, no stored enum)."""

    def test_superseded(self):
        self.assertEqual(row_status(superseded_by="x", today=TODAY), "superseded")

    def test_future_by_interval_and_by_stance(self):
        self.assertEqual(
            row_status(valid_from="2027-01-01", today=TODAY), "future")
        self.assertEqual(row_status(stance="future", today=TODAY), "future")

    def test_historical_by_interval_and_by_stance(self):
        self.assertEqual(
            row_status(valid_until="2026-01-01", today=TODAY), "historical")
        self.assertEqual(row_status(stance="past", today=TODAY), "historical")

    def test_uncertain_is_current_facet_not_status(self):
        self.assertEqual(row_status(stance="uncertain", today=TODAY), "current")

    def test_boundaries_inclusive_today(self):
        self.assertEqual(row_status(valid_from=TODAY.isoformat(), today=TODAY),
                         "current")
        self.assertEqual(row_status(valid_until=TODAY.isoformat(), today=TODAY),
                         "current")

    def test_legacy_row_is_current_or_superseded_only(self):
        """PHASE 8: rows stored before v0.10 (no temporal keys) derive to
        EXACTLY the v0.9.2 two-state world."""
        self.assertEqual(row_status(today=TODAY), "current")
        self.assertEqual(fact_status({}, TODAY), "current")
        self.assertEqual(fact_status({"superseded_by": "n"}, TODAY), "superseded")

    def test_garbage_keys_never_invent_dates(self):
        """PHASE 3: unparseable/missing timestamps fail CLOSED to the old
        behaviour — never to an invented date."""
        self.assertEqual(
            row_status(valid_from="15:18:36", today=TODAY), "current")
        self.assertEqual(
            row_status(valid_until="not-a-date", today=TODAY), "current")


class QueryTemporalIntentTests(unittest.TestCase):
    """PHASE 6 — query-side intent (reorder-only, never a filter)."""

    def test_default_current(self):
        for q in ("Mit szeret Tamás?", "What does Thomas like?",
                  "kávé?", ""):
            self.assertEqual(query_temporal_intent(q), "current", q)

    def test_past_intent_en_hu(self):
        for q in ("What did Thomas like before?",
                  "What did Thomas use to like?",
                  "Mit szeretett régen Tamás?",
                  "Tamás korábban mit szeretett?"):
            self.assertEqual(query_temporal_intent(q), "past", q)

    def test_future_intent_en_hu(self):
        for q in ("What will Thomas like?",
                  "What is Thomas going to like?",
                  "Mit fog szeretni Tamás?",
                  "Mi lesz Tamás kedvence jövő hónapban?"):
            self.assertEqual(query_temporal_intent(q), "future", q)


class RankingTierTests(unittest.TestCase):
    """PHASE 4/6 — tier table + the legacy-equivalence proof."""

    def test_tier_table(self):
        # current-intent: current < historical = future < superseded
        self.assertEqual(temporal_rank_tier("current", "current"), 0)
        self.assertEqual(temporal_rank_tier("historical", "current"), 1)
        self.assertEqual(temporal_rank_tier("future", "current"), 1)
        self.assertEqual(temporal_rank_tier("superseded", "current"), 2)
        # past-intent: history first, future last
        self.assertEqual(temporal_rank_tier("historical", "past"), 0)
        self.assertEqual(temporal_rank_tier("superseded", "past"), 0)
        self.assertEqual(temporal_rank_tier("current", "past"), 1)
        self.assertEqual(temporal_rank_tier("future", "past"), 2)
        # future-intent
        self.assertEqual(temporal_rank_tier("future", "future"), 0)
        self.assertEqual(temporal_rank_tier("current", "future"), 1)
        self.assertEqual(temporal_rank_tier("historical", "future"), 2)

    def test_legacy_sort_order_is_byte_identical(self):
        """PHASE 8 (the load-bearing proof): for rows WITHOUT temporal keys
        the v0.10 sort key produces EXACTLY the v0.9.2 order — randomised."""

        def old_key(h):
            return (not h.superseded_by, h.base_score + h.recency_boost)

        def new_key(h, intent="current", today=TODAY):
            tier = temporal_rank_tier(
                row_status(superseded_by=h.superseded_by, stance=h.stance,
                           valid_from=h.valid_from, valid_until=h.valid_until,
                           today=today),
                intent)
            return (-tier, not h.superseded_by,
                    h.base_score + h.recency_boost)

        rng = random.Random(20260915)
        for _ in range(50):
            rows = [
                MemorySearchHit(
                    memory_id=f"m{i}", text=f"t{i}", score=0.0,
                    attributed_to="user", metadata={},
                    base_score=round(rng.uniform(0.1, 0.95), 4),
                    recency_boost=round(rng.uniform(0.0, 0.1), 4),
                    superseded_by=(f"n{i}" if rng.random() < 0.4 else ""),
                )
                for i in range(rng.randint(2, 60))
            ]
            # sort with reverse=True + tuple keys, as production does
            old_order = sorted(rows, key=old_key, reverse=True)
            new_order = sorted(rows, key=new_key, reverse=True)
            self.assertEqual([h.memory_id for h in old_order],
                             [h.memory_id for h in new_order])

    def test_current_intent_demotes_future_and_past(self):
        """PHASE 6 rules 1+4: current state outranks historical; future does
        not appear as present state (demotion, not exclusion)."""
        rows = [
            ("past_row", row_status(stance="past", today=TODAY)),
            ("future_row", row_status(stance="future", today=TODAY)),
            ("cur_row", row_status(today=TODAY)),
        ]
        ranked = sorted(rows, key=lambda r: temporal_rank_tier(r[1], "current"))
        self.assertEqual(ranked[0][0], "cur_row")
        self.assertEqual({r[0] for r in ranked[1:3]},
                         {"past_row", "future_row"})

    def test_past_intent_prefers_history(self):
        rows = [
            ("cur", "current"), ("hist", "historical"), ("sup", "superseded"),
            ("fut", "future"),
        ]
        ranked = [r[0] for r in sorted(
            rows, key=lambda r: temporal_rank_tier(r[1], "past"))]
        self.assertEqual(set(ranked[:2]), {"hist", "sup"})
        self.assertEqual(ranked[3], "fut")


class ObservationTimeParsingTests(unittest.TestCase):
    """PHASE 3 — observation-time helpers."""

    def test_parse_iso_date_strict(self):
        self.assertEqual(parse_iso_date("2026-09-15"), date(2026, 9, 15))
        self.assertEqual(parse_iso_date("2026-09-15T11:45:30"), date(2026, 9, 15))
        self.assertIsNone(parse_iso_date("15:18:36"))  # legacy time-of-day
        self.assertIsNone(parse_iso_date(""))
        self.assertIsNone(parse_iso_date("2026-13-45"))
        self.assertIsNone(parse_iso_date("next tuesday"))

    def test_obs_date_never_invents(self):
        """The ingest helper's contract: legacy time-of-day → None (old
        behaviour), ISO → date (v0.10 anchor fix)."""
        from voicemem.utils.common.voice_input import _obs_date_or_none
        self.assertEqual(_obs_date_or_none("2026-09-15T11:45:30"), "2026-09-15")
        self.assertIsNone(_obs_date_or_none("15:18:36"))
        self.assertIsNone(_obs_date_or_none(""))


class RenderMarkerTests(unittest.TestCase):
    """PHASE 6 — consumer-side markers (both render copies)."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO))  # app/ importable
        from app import voicemem_bridge, web_server  # noqa: PLC0415
        # staticmethod-wrap: plain assignment would bind them as methods
        cls.bridge_suffix = staticmethod(voicemem_bridge.hit_provenance_suffix)
        cls.web_suffix = staticmethod(web_server.hit_provenance_suffix)

    def _hit(self, **kw):
        base = dict(memory_id="m", text="t", score=1.0, attributed_to="user",
                    metadata={})
        base.update(kw)
        return MemorySearchHit(**base)

    def test_both_copies_render_temporal_markers(self):
        for stance, expect in (("past", "past"), ("uncertain", "uncertain"),
                              ("qualified", "qualified"), ("future", "future")):
            for suffix in (self.bridge_suffix, self.web_suffix):
                out = suffix(self._hit(stance=stance))
                self.assertIn(expect, out, f"{stance} missing in {suffix.__module__}")

    def test_future_with_date_renders_from(self):
        out = self.web_suffix(self._hit(stance="future", valid_from="2027-01-01"))
        self.assertIn("future from 2027-01-01", out)

    def test_absent_markers_mean_current(self):
        self.assertNotIn("past", self.web_suffix(self._hit()))
        self.assertNotIn("future", self.web_suffix(self._hit()))
        self.assertNotIn("uncertain", self.web_suffix(self._hit()))
        # pure date/occurrence suffix unchanged for legacy rows
        out = self.web_suffix(self._hit(observed_at="2026-09-01",
                                        occurrence_count=3))
        self.assertEqual(out, " [2026-09-01 | 3x confirmed]")

    def test_superseded_marker_unchanged(self):
        out = self.bridge_suffix(self._hit(superseded_by="n"))
        self.assertIn("superseded", out)

    def test_web_payload_carries_temporal_keys(self):
        """_hits_payload exposes stance/valid_from/valid_until (machine
        consumers); neutral defaults for legacy rows."""
        from app.web_server import WebSession  # noqa: PLC0415
        payload = WebSession._hits_payload(
            type("R", (), {"hits": [self._hit(stance="future",
                                              valid_from="2027-01-01")],
                           "rb_hits": [], "classification": None})())
        lb = payload["left_brain"][0]
        self.assertEqual(lb["stance"], "future")
        self.assertEqual(lb["valid_from"], "2027-01-01")
        self.assertEqual(lb["valid_until"], "")


class UpdateMemoryPassThroughTests(unittest.TestCase):
    """PHASE 4 — update_memory carries valid_until (old-row interval) and
    new_meta (the superseding observation's own classification) into the
    mem0 calls. Fake mem0 client: capture only, no real store."""

    def test_valid_until_and_new_meta_reach_the_calls(self):
        store = object.__new__(
            __import__("voicemem.leftbrain.mem0_backend_store",
                       fromlist=["Mem0BackendStore"]).Mem0BackendStore)

        class FakeMem0:
            def __init__(self):
                self.adds, self.updates, self.rows = [], [], {
                    "old": {"user_id": "u", "role": "user",
                            "memory": "A kedvenc éttermem az Arany Kanna.",
                            "metadata": {}},
                }

            def get(self, mid):
                return self.rows.get(mid)

            def add(self, *, messages, user_id, infer, metadata):
                self.adds.append(metadata)
                return {"results": [{"id": "new"}]}

            def update(self, mid, metadata=None, **kw):
                self.updates.append((mid, dict(metadata or {})))

        fake = FakeMem0()
        store._mem0 = fake
        out = store.update_memory(
            "old", "A kedvenc éttermem mostantól a Bistro Buda.",
            observed_at="2026-09-10", user_id="u",
            valid_until="2026-09-10",
            new_meta={"stance": "neg"})
        self.assertEqual(out, "new")
        # the new row carries supersedes + its own classification + event date
        self.assertEqual(fake.adds[0]["supersedes"], "old")
        self.assertEqual(fake.adds[0]["stance"], "neg")
        self.assertEqual(fake.adds[0]["created_at"], "2026-09-10")
        # the old row is closed: superseded_by + superseded_at + valid_until
        mid, md = fake.updates[0]
        self.assertEqual(mid, "old")
        self.assertEqual(md["superseded_by"], "new")
        self.assertEqual(md["valid_until"], "2026-09-10")

    def test_no_valid_until_keeps_v092_marking(self):
        store = object.__new__(
            __import__("voicemem.leftbrain.mem0_backend_store",
                       fromlist=["Mem0BackendStore"]).Mem0BackendStore)

        class FakeMem0:
            def get(self, mid):
                return {"user_id": "u", "role": "user", "metadata": {}}

            def add(self, *, messages, user_id, infer, metadata):
                return {"results": [{"id": "new"}]}

            def update(self, mid, metadata=None, **kw):
                self.md = dict(metadata or {})

        fake = FakeMem0()
        store._mem0 = fake
        store.update_memory("old", "B", user_id="u")
        self.assertEqual(set(fake.md), {"superseded_by", "superseded_at"})


if __name__ == "__main__":
    unittest.main()
