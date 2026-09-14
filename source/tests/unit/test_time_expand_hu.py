"""v0.6.4 — external audit v0.6.3 findings CD-3 / CD-4 / OCC-1 / F-B (unit).

The external differential audit (VoiceMEM_Audit_v0.6.3, Claude Opus 4.8 for
Boost IT) found the memory layer Chinese-only in three places:

* CD-3 (P1): time_expand.py knew no Hungarian/English relative-time cues —
  "mi lesz jövő héten" / "what did I say yesterday" never expanded to dates,
  so temporal retrieval missed its dates entirely (VM-LOCAL-010).
* CD-4 (P1): observed_at reached the hit objects but never the ranking —
  recency was dead (VM-LOCAL-011).
* OCC-1/F-B: repeated identical confirmations wrote nothing, and the render
  dropped provenance (VM-LOCAL-012 + hit_provenance_suffix).

These tests are pure-unit (no models, no store): regex behaviour, date
parsing, decay math, sort keys and render suffixes. The real-store proof
lives in tests/integration/test_memory_safety.py.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
# sys.path entries are ALSO deferred to setUpModule (see NOTE below).

# NOTE (isolation): the vendor `voicemem` package is imported LAZILY in
# setUpModule and PURGED in tearDownModule — a module-level import would
# make `voicemem` importable for the WHOLE pytest session at collection
# time, which flips two other tests' assumptions (install_manifest's
# pin_verified probe and feature_memory's bridge-degraded-mode check both
# expect the vendor NOT importable in the plain unit battery).

time_expand = None
MemorySearchHit = None
date_overlap_bonus_values = None
parse_date_values = None
query_date_values = None
recency_bonus = None
hit_provenance_suffix = None
_extract_memory_context = None


def setUpModule() -> None:  # noqa: N802
    global time_expand, MemorySearchHit, date_overlap_bonus_values
    global parse_date_values, query_date_values, recency_bonus
    global hit_provenance_suffix, _extract_memory_context
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(VENDOR))
    from voicemem.leftbrain import time_expand as _te
    from voicemem.leftbrain.local_memory_store import (
        MemorySearchHit as _MSH,
        date_overlap_bonus_values as _dobv,
        parse_date_values as _pdv,
        query_date_values as _qdv,
        recency_bonus as _rb,
    )
    from app.voicemem_bridge import (
        _extract_memory_context as _emc,
        hit_provenance_suffix as _hps,
    )
    time_expand = _te
    MemorySearchHit = _MSH
    date_overlap_bonus_values = _dobv
    parse_date_values = _pdv
    query_date_values = _qdv
    recency_bonus = _rb
    hit_provenance_suffix = _hps
    _extract_memory_context = _emc


def tearDownModule() -> None:  # noqa: N802
    for entry in (str(REPO), str(VENDOR)):
        while entry in sys.path:
            sys.path.remove(entry)
    for name in [n for n in sys.modules
                 if n == "voicemem" or n.startswith("voicemem.")]:
        del sys.modules[name]


TODAY = date(2026, 9, 13)  # Sunday


class TimeExpandHungarianTests(unittest.TestCase):
    """CD-3: Hungarian relative-time cues expand to ISO dates."""

    def test_hungarian_days(self):
        out = time_expand.expand_relative_dates("mit mondtam tegnap?", today=TODAY)
        self.assertIn("2026-09-12", out)
        self.assertTrue(out.startswith("mit mondtam tegnap?"))

    def test_hungarian_today_and_tomorrow(self):
        self.assertIn("2026-09-13",
                      time_expand.expand_relative_dates("mi a program ma?", today=TODAY))
        self.assertIn("2026-09-14",
                      time_expand.expand_relative_dates("mi lesz holnap?", today=TODAY))
        self.assertIn("2026-09-11",
                      time_expand.expand_relative_dates("tegnapelőtt", today=TODAY))

    def test_hungarian_next_week_expands_seven_days(self):
        # 2026-09-13 is a Sunday -> this week's Monday is 2026-09-07,
        # next week = 2026-09-14..2026-09-20.
        out = time_expand.expand_relative_dates("mi lesz jövő héten?", today=TODAY)
        for d in range(14, 21):
            self.assertIn(f"2026-09-{d:02d}", out)

    def test_hungarian_one_word_week_fused_forms(self):
        for w in ("jövőhéten", "múlthéten", "ezhéten"):
            out = time_expand.expand_relative_dates(f"program {w} este", today=TODAY)
            self.assertIn("2026-09-", out, f"fused form {w} must expand")

    def test_hungarian_last_week(self):
        out = time_expand.expand_relative_dates("múlt héten mit tanultam?", today=TODAY)
        self.assertIn("2026-08-31", out)  # last week's Monday
        self.assertIn("2026-09-06", out)  # last week's Sunday

    def test_english_days_and_weeks(self):
        self.assertIn("2026-09-12",
                      time_expand.expand_relative_dates("what did I say yesterday?", today=TODAY))
        out = time_expand.expand_relative_dates("what is planned next week?", today=TODAY)
        self.assertIn("2026-09-14", out)
        self.assertIn("2026-09-20", out)
        self.assertIn("2026-09-13",
                      time_expand.expand_relative_dates("my plans today", today=TODAY))

    def test_no_false_positive_inside_words(self):
        # "ma" must NOT match inside "magyar" / "many"; "next week" must not
        # match "next weekly meeting" (word-boundary regex); suffixed forms
        # like "holnaputánig" do not fire (boundary after the stem fails).
        for q in ("a magyar nyelv szép", "many things happened",
                  "the next weekly meeting", "holnaputánig jó"):
            self.assertEqual(time_expand.expand_relative_dates(q, today=TODAY), q)

    def test_no_cues_returns_query_unchanged(self):
        q = "mi a kedvenc színem?"
        self.assertEqual(time_expand.expand_relative_dates(q, today=TODAY), q)

    def test_cjk_behaviour_unchanged(self):
        # The Chinese path keeps its own stamp format and semantics.
        out = time_expand.expand_relative_dates("我下周有什么安排", today=TODAY)
        self.assertIn("2026年9月14日", out)
        self.assertIn("（", out)  # CJK parenthesis, not the Latin variant
        self.assertNotIn("2026-09-", out)

    def test_latin_uses_ascii_parens_and_iso(self):
        out = time_expand.expand_relative_dates("mi lesz holnap?", today=TODAY)
        self.assertIn(" (2026-09-14)", out)
        self.assertNotIn("（", out)


class MultilingualDateParsingTests(unittest.TestCase):
    """CD-3 (parsing half): four date-format families become date values."""

    def test_iso(self):
        self.assertEqual(parse_date_values("találkozó 2026-09-14-kor", TODAY),
                         frozenset({date(2026, 9, 14)}))

    def test_hungarian_long_form(self):
        vals = parse_date_values("szabadság 2026. szeptember 14-től", TODAY)
        self.assertIn(date(2026, 9, 14), vals)

    def test_hungarian_month_day_no_year_defaults_to_this_year(self):
        vals = parse_date_values("augusztus 26-án orvoshoz megyek", TODAY)
        self.assertIn(date(2026, 8, 26), vals)

    def test_english_long_form(self):
        vals = parse_date_values("dinner on August 26, 2026 was lovely", TODAY)
        self.assertIn(date(2026, 8, 26), vals)

    def test_cjk_form(self):
        vals = parse_date_values("2026年8月26日有体检", TODAY)
        self.assertIn(date(2026, 8, 26), vals)

    def test_invalid_dates_are_skipped(self):
        self.assertEqual(parse_date_values("2026-13-45 és 2026-02-31", TODAY),
                         frozenset())

    def test_query_date_values_reads_expanded_stamps(self):
        q = time_expand.expand_relative_dates("mi lesz jövő héten?", today=TODAY)
        vals = query_date_values(q, TODAY)
        self.assertIn(date(2026, 9, 14), vals)
        self.assertIn(date(2026, 9, 20), vals)

    def test_value_overlap_across_format_families(self):
        # The audit's point: "2026-09-14" (query) must reach a memory written
        # as "2026. szeptember 14." — the literal bonus could never do this.
        self.assertGreater(
            date_overlap_bonus_values(frozenset({date(2026, 9, 14)}),
                                      "szabadság 2026. szeptember 14-től", TODAY),
            0.0)
        self.assertEqual(
            date_overlap_bonus_values(frozenset({date(2026, 9, 14)}),
                                      "teljesen más témájú emlék", TODAY),
            0.0)
        self.assertEqual(
            date_overlap_bonus_values(frozenset(), "bármi", TODAY), 0.0)


class RecencyBonusTests(unittest.TestCase):
    """CD-4: event-time decay bonus (VM-LOCAL-011)."""

    def test_today_gets_full_weight(self):
        self.assertAlmostEqual(recency_bonus("2026-09-13", TODAY), 0.10, places=6)

    def test_future_dates_count_as_fresh(self):
        self.assertAlmostEqual(recency_bonus("2026-12-24", TODAY), 0.10, places=6)

    def test_thirty_days_is_half(self):
        self.assertAlmostEqual(recency_bonus("2026-08-14", TODAY), 0.05, places=6)

    def test_empty_and_garbage_get_zero(self):
        for bad in ("", None, "15:18:36", "not-a-date", "2026"):
            self.assertEqual(recency_bonus(bad, TODAY), 0.0)

    def test_sort_key_uses_base_plus_recency(self):
        # The audit's scenario: same cosine, different event dates -> the
        # newer observation ranks first.
        # recency_boost is computed by the store exactly like this (the
        # dataclass keeps it an explicit field — the store wires it):
        old = MemorySearchHit(
            memory_id="old", text="régi", score=0.5,
            attributed_to="user", metadata={},
            base_score=0.5, observed_at="2025-09-13",
            recency_boost=recency_bonus("2025-09-13", TODAY))
        new = MemorySearchHit(
            memory_id="new", text="friss", score=0.5,
            attributed_to="user", metadata={},
            base_score=0.5, observed_at="2026-09-12",
            recency_boost=recency_bonus("2026-09-12", TODAY))
        ranked = sorted([old, new],
                        key=lambda h: (not h.superseded_by, h.base_score + h.recency_boost),
                        reverse=True)
        self.assertEqual(ranked[0].memory_id, "new")

    def test_undated_rows_keep_zero_boost(self):
        h = MemorySearchHit(memory_id="x", text="x", score=0.4,
                            attributed_to="user", metadata={}, base_score=0.4)
        self.assertEqual(h.recency_boost, 0.0)
        self.assertEqual(h.occurrence_count, 0)
        self.assertEqual(h.last_observed_at, "")


class ProvenanceRenderTests(unittest.TestCase):
    """F-B: the render suffix surfaces date / count / supersession / attribution."""

    class _Hit:
        def __init__(self, observed_at="", occurrence_count=0,
                     superseded_by="", attributed_to="user"):
            self.observed_at = observed_at
            self.occurrence_count = occurrence_count
            self.superseded_by = superseded_by
            self.attributed_to = attributed_to

    def test_full_suffix(self):
        s = hit_provenance_suffix(self._Hit(
            observed_at="2026-09-12", occurrence_count=5,
            superseded_by="abc", attributed_to="Anna"))
        self.assertEqual(s, " [2026-09-12 | 5x confirmed | superseded | by Anna]")

    def test_date_and_count_only(self):
        s = hit_provenance_suffix(self._Hit(observed_at="2026-09-12", occurrence_count=3))
        self.assertEqual(s, " [2026-09-12 | 3x confirmed]")

    def test_plain_user_hit_gets_no_suffix(self):
        self.assertEqual(hit_provenance_suffix(self._Hit()), "")

    def test_first_observation_hides_count(self):
        # occurrence_count <= 1 stays silent (1x is not information).
        self.assertEqual(
            hit_provenance_suffix(self._Hit(observed_at="2026-09-12", occurrence_count=1)),
            " [2026-09-12]")

    def test_time_only_observed_at_is_dropped(self):
        # observed_at was normalised to "" upstream when it is a time, not a
        # date — the suffix must not print garbage.
        self.assertEqual(hit_provenance_suffix(self._Hit(observed_at="15:18:36")), "")

    def test_bridged_render_appends_suffix(self):
        class _Result:
            hits = [type("H", (), {
                "text": "A felhasználó allergiás a földimogyoróra.",
                "observed_at": "2026-09-12", "occurrence_count": 4,
                "superseded_by": "", "attributed_to": "user",
                "score": 0.9, "base_score": 0.9, "memory_id": "m1"})]
            rb_hits = []

        ctx = _extract_memory_context(_Result())
        self.assertEqual(
            ctx, "- A felhasználó allergiás a földimogyoróra. [2026-09-12 | 4x confirmed]")


if __name__ == "__main__":
    unittest.main()
