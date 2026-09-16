"""Unit tests for the v0.10.3 upstream-audit selective ports.

Covers (per the upstream audit of xzf-thu/VoiceMem @ 6cacb3c):
  1. TTS speech normalisation (the field-reported „ U+201E crash) — app side.
  2. Query-gated recency (upstream a507978 adaptation) — temporal.py +
     mem0_backend_store sort key.
  3. Right-brain source quotas (upstream 333dbcb + fa537a9 adaptation):
     situation_pattern cap 2, response_experience default 0, and the
     opt-in write gate.
  4. The playback-tail "hearing" window helpers (upstream 7581656
     adaptation) — pure accounting logic, no event loop.

The web_server-integrated behaviours (barge-in during tail, interrupted
history, concurrent TTS pipeline) are covered by the mock-based
integration battery (tests/integration/test_pipeline_mock.py) and the
web e2e; these unit tests pin the deterministic cores.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from app.text_utils import normalize_for_speech  # noqa: E402

# NOTE (isolation): the voicemem imports are MODULE-LEVEL, like
# test_temporal_semantics/test_trait_semantics — the binding is made at
# collection time and survives test_time_expand_hu's deliberate
# sys.path/sys.modules purge (its tearDownModule runs before this module's
# tests under alphabetical discovery, which would break in-method imports).
try:
    from voicemem.leftbrain.temporal import wants_recency  # noqa: E402
    from voicemem.leftbrain.local_memory_store import (  # noqa: E402
        MemorySearchHit,
    )
    from voicemem.rightbrain.brain import (  # noqa: E402
        _apply_source_quota,
        _SOURCE_QUOTA,
    )
    _VENDOR_IMPORTABLE = True
except Exception:  # noqa: BLE001 — degrade to skips, never fail collection
    _VENDOR_IMPORTABLE = False


# --------------------------------------------------------------------------- #
# 1. TTS speech normalisation
# --------------------------------------------------------------------------- #

class NormalizeForSpeechTests(unittest.TestCase):
    def test_field_reported_hungarian_opening_quote(self):
        # The exact field failure: Supertonic rejects U+201E.
        self.assertEqual(normalize_for_speech('Azt mondta „hello” nekem.'),
                         'Azt mondta "hello" nekem.')

    def test_all_typographic_pairs(self):
        src = "„abc“ ‘x’ ‚y‘ «z» ′a′ ″b″"
        out = normalize_for_speech(src)
        for ch in out:
            self.assertTrue(
                ord(ch) < 128 or ch.isalnum(),
                f"non-ASCII punctuation survived: {ch!r} in {out!r}",
            )

    def test_dashes_and_ellipsis(self):
        self.assertEqual(normalize_for_speech("a – b — c ― d − e"), "a - b - c - d - e")
        self.assertEqual(normalize_for_speech("várj…"), "várj...")

    def test_invisible_codepoints_dropped(self):
        out = normalize_for_speech("a\u200Bb\u200Cc\u200Dd\uFEFFe")
        self.assertEqual(out, "abcde")

    def test_nbsp_variants_become_spaces_and_collapse(self):
        out = normalize_for_speech("a\u00A0\u202F\u2007 b")
        self.assertEqual(out, "a b")

    def test_letters_digits_untouched(self):
        src = "Árvíztűrő tükörfúrógép 123 áéíóöőúüű Q&A 50%"
        self.assertEqual(normalize_for_speech(src), src)

    def test_content_preserved_word_count(self):
        # The spoken content must be identical word-for-word. Punctuation
        # attached to a word changes with the mapping („x” → "x") — compare
        # the alphanumerics, which is what TTS actually speaks.
        src = "„Ez egy magyar mondat”, amit „idézőjelbe” tettek — tényleg."
        out = normalize_for_speech(src)
        strip = lambda s: ["".join(c for c in w if c.isalnum()) for w in s.split()]
        self.assertEqual(strip(src), strip(out))

    def test_empty_and_none_safe(self):
        self.assertEqual(normalize_for_speech(""), "")
        self.assertIsNone(normalize_for_speech(None), None)

    def test_plain_ascii_passthrough(self):
        src = "Szia, Imre! Mi ujsag?"
        self.assertEqual(normalize_for_speech(src), src)

    def test_inverted_question_exclamation(self):
        self.assertEqual(normalize_for_speech("¡Hola! ¿Qué tal?"), "!Hola! ?Qué tal?")


# --------------------------------------------------------------------------- #
# 2. Query-gated recency
# --------------------------------------------------------------------------- #

class WantsRecencyTests(unittest.TestCase):
    def _import(self):
        if not _VENDOR_IMPORTABLE:
            self.skipTest("vendor voicemem not importable")
        return wants_recency

    def test_english_recency_cues(self):
        wants_recency = self._import()
        for q in (
            "What have you been up to lately?",
            "What's new?",
            "how are you doing today",
            "anything new this week",
        ):
            self.assertTrue(wants_recency(q), q)

    def test_hungarian_recency_cues(self):
        wants_recency = self._import()
        for q in (
            "Mostan mit csinálsz?",
            "Mi újság?",
            "mit csinaltal ma",
            "mi ujsag legutobb",
            "Ezen a héten voltál színházban?",
        ):
            self.assertTrue(wants_recency(q), q)

    def test_attribute_queries_have_no_recency(self):
        wants_recency = self._import()
        for q in (
            "Where do I study?",
            "Hol tanulok?",
            "what music does he like",
            "milyen zenét szeret",
        ):
            self.assertFalse(wants_recency(q), q)

    def test_empty_query(self):
        wants_recency = self._import()
        self.assertFalse(wants_recency(""))
        self.assertFalse(wants_recency("   "))


class RecencyGatedSortTests(unittest.TestCase):
    """The mem0_backend_store sort key gates recency_boost by the query."""

    def _hit(self, mid: str, base: float, recency: float):
        if not _VENDOR_IMPORTABLE:
            self.skipTest("vendor voicemem not importable")
        return MemorySearchHit(
            memory_id=mid,
            text=f"fact {mid}",
            score=base,
            attributed_to="user",
            metadata={},
            base_score=base,
            recency_boost=recency,
        )

    def test_attribute_query_keeps_similarity_order(self):
        # Old-but-best hit must win when the query is an attribute question
        # (upstream a507978's measured failure case).
        if not _VENDOR_IMPORTABLE:
            self.skipTest("vendor voicemem not importable")
        hits = [
            self._hit("new", 0.773, 0.10),
            self._hit("old-best", 0.810, 0.0),
        ]
        q = "where do i study"  # no recency cue
        self.assertFalse(wants_recency(q))
        gated = sorted(
            hits,
            key=lambda h: h.base_score + (h.recency_boost if wants_recency(q) else 0.0),
            reverse=True,
        )
        self.assertEqual(gated[0].memory_id, "old-best")

    def test_recency_query_prefers_recent(self):
        if not _VENDOR_IMPORTABLE:
            self.skipTest("vendor voicemem not importable")
        hits = [
            self._hit("new", 0.773, 0.10),
            self._hit("old-best", 0.810, 0.0),
        ]
        q = "mostan mit csinalsz"  # recency cue
        self.assertTrue(wants_recency(q))
        gated = sorted(
            hits,
            key=lambda h: h.base_score + (h.recency_boost if wants_recency(q) else 0.0),
            reverse=True,
        )
        self.assertEqual(gated[0].memory_id, "new")


# --------------------------------------------------------------------------- #
# 3. Right-brain source quotas + write gate
# --------------------------------------------------------------------------- #

class SourceQuotaTests(unittest.TestCase):
    def test_defaults_heartnote_2_experience_0(self):
        # Clean env: the v0.10.3 defaults.
        if not _VENDOR_IMPORTABLE:
            self.skipTest("vendor voicemem not importable")
        self.assertEqual(_SOURCE_QUOTA.get("situation_pattern"), 2)
        self.assertEqual(_SOURCE_QUOTA.get("response_experience"), 0)
        self.assertEqual(_SOURCE_QUOTA.get("profile"), 3)

    def test_quota_caps_heartnotes(self):
        if not _VENDOR_IMPORTABLE:
            self.skipTest("vendor voicemem not importable")
        from types import SimpleNamespace

        def hit(i, src):
            return SimpleNamespace(source=src, memory_id=f"m{i}")

        hits = [hit(i, "situation_pattern") for i in range(5)]
        hits += [hit(10, "emotion_trait"), hit(11, "emotion_trait")]
        kept = _apply_source_quota(hits)
        sources = [h.source for h in kept]
        self.assertEqual(sources.count("situation_pattern"), 2)
        self.assertEqual(sources.count("emotion_trait"), 2)

    def test_quota_drops_experience_rows_by_default(self):
        if not _VENDOR_IMPORTABLE:
            self.skipTest("vendor voicemem not importable")
        from types import SimpleNamespace

        hits = [
            SimpleNamespace(source="response_experience", memory_id="m1"),
            SimpleNamespace(source="emotion_trait", memory_id="m2"),
        ]
        kept = _apply_source_quota(hits)
        self.assertEqual(
            [h.memory_id for h in kept], ["m2"],
        )


# --------------------------------------------------------------------------- #
# 4. Playback-tail hearing window (pure accounting)
# --------------------------------------------------------------------------- #

class _TailHarness:
    """Minimal stand-in for the WebSession attributes the helpers use."""

    def __init__(self):
        # Import the real helpers bound to this harness.
        from app.web_server import WebSession

        self._audio_sent_s = 0.0
        self._first_audio_at = None
        for name in ("_playback_tail_active", "_reset_playback_accounting",
                     "_account_sent_audio"):
            fn = getattr(WebSession, name)
            setattr(self, name, fn.__get__(self, type(self)))


class PlaybackTailTests(unittest.TestCase):
    def test_no_audio_means_no_tail(self):
        h = _TailHarness()
        self.assertFalse(h._playback_tail_active())

    def test_tail_active_while_audio_drains(self):
        h = _TailHarness()
        h._account_sent_audio(b"\x00" * 48000)  # 1 s of 24 kHz PCM16
        self.assertAlmostEqual(h._audio_sent_s, 1.0, places=3)
        self.assertTrue(h._playback_tail_active())
        # After 1 s + slack the tail must be over.
        h._first_audio_at -= 1.36 + 0.01
        self.assertFalse(h._playback_tail_active())

    def test_reset_forgets_the_ledger(self):
        h = _TailHarness()
        h._account_sent_audio(b"\x00" * 48000)
        h._reset_playback_accounting()
        self.assertFalse(h._playback_tail_active())
        self.assertEqual(h._audio_sent_s, 0.0)
        self.assertIsNone(h._first_audio_at)

    def test_first_stamp_taken_once(self):
        h = _TailHarness()
        t0 = 1000.0
        h._first_audio_at = t0  # pretend an earlier stamp exists
        h._account_sent_audio(b"\x00" * 4800)
        self.assertEqual(h._first_audio_at, t0)
        self.assertAlmostEqual(h._audio_sent_s, 0.1, places=3)


if __name__ == "__main__":
    unittest.main()
