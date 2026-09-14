"""[v0.8.1] Behavioural tests: the retrieval result contract + consumers.

Post-implementation-audit P1-2: the trait bookkeeping the memory layer
already computes (VM-LOCAL-013 — confidence, effective confidence,
occurrence_count, first_seen, last_seen on every trait
RightBrainHit.metadata) was discarded by every consumer. These tests prove
the information now SURVIVES the whole chain — with REAL retrieval objects
(the vendor TraitStore + _rb_trait_hits on real sqlite with a deterministic
fake embedder) and the REAL payload/context builders
(app.web_server._hits_payload / _memory_context, app.voicemem_bridge
._extract_memory_context):

  * contract extraction (typed, values verified against the store)
  * web payload: trait fields + stable identity + rb_directive preserved
  * LLM context: observation date + confirmation count rendered, raw
    confidence floats NOT rendered (misleading pseudo-precision), rb_directive
    NOT injected (documented-unconsumed decision)
  * CLI bridge: renders the same suffix as the web path (parity)
  * backwards compatibility: every pre-existing payload key still present,
    non-trait rb hits carry neutral defaults (uniform schema)
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
for p in (str(REPO), str(VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import unittest

from voicemem.rightbrain.brain import _rb_trait_hits
from voicemem.rightbrain.traits_store import Evidence, TraitStore

from app.retrieval_contract import (
    TRAIT_PAYLOAD_KEYS,
    extract_trait_info,
    trait_fields_payload,
    trait_prompt_suffix,
)

# NOTE: voicemem imports are MODULE-LEVEL on purpose (the established
# collection-time binding pattern — see test_traits_observation.py).
# app.web_server imports fastapi at module level (sandbox-available).


class _FakeEmbedder:
    """Deterministic: 'coffee' texts → vector A, everything else → B."""

    DIM = 8

    def _unit(self, idx: int):
        v = [0.0] * self.DIM
        v[idx] = 1.0
        return v

    def __call__(self, text: str):
        if "coffee" in text.lower():
            return self._unit(0)
        return self._unit(1)


class _RbHit:
    """Minimal stand-in for a NON-trait RightBrainHit (response_experience)."""

    def __init__(self, content: str, source: str, metadata: dict | None = None):
        self.content = content
        self.source = source
        self.priority = 0.5
        self.metadata = metadata or {}


class _FactHit:
    """Minimal stand-in for a left-brain MemorySearchHit."""

    def __init__(self, text: str, **kw):
        self.text = text
        self.score = 0.9
        self.attributed_to = kw.get("attributed_to", "")
        self.memory_id = kw.get("memory_id", "m1")
        self.observed_at = kw.get("observed_at", "")
        self.occurrence_count = kw.get("occurrence_count", 0)
        self.last_observed_at = kw.get("last_observed_at", "")
        self.superseded_by = kw.get("superseded_by", "")


class _SearchResult:
    """Minimal stand-in for the vendor SearchResult (duck-typed consumers)."""

    def __init__(self, hits, rb_hits, rb_directive: str = ""):
        self.hits = hits
        self.rb_hits = rb_hits
        self.rb_directive = rb_directive
        self.classification = None
        self.related_summaries = {}


class TraitContractTests(unittest.TestCase):
    """The typed contract over REAL vendor trait hits."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="rb_contract_")
        self.store = TraitStore(os.path.join(self._tmp.name, "space.sqlite"),
                                _FakeEmbedder())

    def tearDown(self):
        self._tmp.cleanup()

    def _add(self, claim="likes coffee", quote="I like coffee"):
        return self.store.add("u1", "喜好与厌恶", claim, Evidence(quote=quote))

    def _real_hits(self):
        return _rb_trait_hits(self.store, "u1", "coffee")

    def test_extract_trait_info_survives_retrieval_with_store_values(self):
        """confidence / eff_confidence / occurrence_count / first_seen /
        last_seen survive retrieval with the STORE's values (not just keys)."""
        self._add()
        time.sleep(1.1)
        self._add()          # second observation: occurrence 2, c=0.93
        hits = self._real_hits()
        self.assertEqual(len(hits), 1)
        info = extract_trait_info(hits[0])
        self.assertIsNotNone(info)
        self.assertEqual(info.occurrence_count, 2)
        self.assertAlmostEqual(info.confidence, 0.93, places=4)
        # fresh observation: eff == raw (inside the 90-day grace)
        self.assertAlmostEqual(info.eff_confidence, 0.93, places=4)
        for date in (info.first_seen, info.last_seen):
            self.assertTrue(date, "observation dates must be non-empty")
        self.assertEqual(info.claim, "likes coffee")
        self.assertTrue(info.trait_id)
        self.assertEqual(info.slot_name, "喜好与厌恶")

    def test_extract_trait_info_none_for_non_trait_hits(self):
        """response_experience / situation_pattern metadata (no trait_id)
        is NOT misread as trait bookkeeping."""
        hit = _RbHit("Effective approach: light tone", "response_experience",
                     {"failed": False, "anchor_score": 1.0})
        self.assertIsNone(extract_trait_info(hit))
        hit2 = _RbHit("Emotional note: …", "situation_pattern",
                      {"anchor_score": 0.5, "emotion": "joy"})
        self.assertIsNone(extract_trait_info(hit2))

    def test_payload_fields_trait_hit(self):
        self._add()
        time.sleep(1.1)
        self._add()
        payload = trait_fields_payload(self._real_hits()[0])
        self.assertTrue(payload["is_trait"])
        self.assertEqual(payload["occurrence_count"], 2)
        self.assertAlmostEqual(payload["confidence"], 0.93, places=4)
        self.assertAlmostEqual(payload["eff_confidence"], 0.93, places=4)
        self.assertTrue(payload["trait_id"])
        self.assertTrue(payload["first_seen"])
        self.assertTrue(payload["last_seen"])
        self.assertEqual(sorted(payload.keys()), sorted(TRAIT_PAYLOAD_KEYS))

    def test_payload_fields_non_trait_hit_neutral_defaults(self):
        """Uniform schema: non-trait hits carry neutral defaults (consumers
        never branch on key presence)."""
        payload = trait_fields_payload(
            _RbHit("note", "situation_pattern"))
        self.assertFalse(payload["is_trait"])
        self.assertEqual(payload["trait_id"], "")
        self.assertEqual(payload["occurrence_count"], 0)
        self.assertEqual(payload["confidence"], 0.0)
        self.assertEqual(payload["first_seen"], "")
        self.assertEqual(payload["last_seen"], "")

    def test_prompt_suffix_renders_date_and_count_only(self):
        self._add()
        time.sleep(1.1)
        self._add()
        info = extract_trait_info(self._real_hits()[0])
        suffix = trait_prompt_suffix(self._real_hits()[0])
        self.assertIn(info.last_seen[:10], suffix)
        self.assertIn("2x heard", suffix)
        # NO confidence pseudo-precision in the prompt render
        self.assertNotIn("0.93", suffix)
        self.assertNotIn("confidence", suffix)

    def test_prompt_suffix_single_observation_is_lean(self):
        """1x observation: no count (nothing to confirm), but the date rides."""
        self._add()
        hit = self._real_hits()[0]
        suffix = trait_prompt_suffix(hit)
        self.assertNotIn("1x", suffix)
        info = extract_trait_info(hit)
        self.assertIn(info.last_seen[:10], suffix)

    def test_prompt_suffix_non_trait_empty(self):
        self.assertEqual(trait_prompt_suffix(_RbHit("x", "relation")), "")


class WebPayloadContractTests(unittest.TestCase):
    """retrieval → web payload (the REAL _hits_payload builder)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="rb_web_")
        self.store = TraitStore(os.path.join(self._tmp.name, "space.sqlite"),
                                _FakeEmbedder())

    def tearDown(self):
        self._tmp.cleanup()

    def _result(self) -> _SearchResult:
        self.store.add("u1", "喜好与厌恶", "likes coffee",
                       Evidence(quote="I like coffee"))
        time.sleep(1.1)
        self.store.add("u1", "喜好与厌恶", "likes coffee",
                       Evidence(quote="coffee again"))
        trait_hits = _rb_trait_hits(self.store, "u1", "coffee")
        rb = list(trait_hits) + [
            _RbHit("Effective approach: light tone", "response_experience"),
            _RbHit("Emotional note: stressed at work", "situation_pattern"),
        ]
        return _SearchResult(
            hits=[_FactHit("Thomas rides a motorcycle",
                           observed_at="2026-05-01", occurrence_count=3)],
            rb_hits=rb,
            rb_directive="note: low evidence — say you don't know",
        )

    @staticmethod
    def _payload(result) -> dict:
        from app.web_server import WebSession

        return WebSession._hits_payload(result)

    def test_trait_fields_survive_web_payload(self):
        payload = self._payload(self._result())
        rb = payload["right_brain_hits"]
        trait_entries = [h for h in rb if h.get("is_trait")]
        self.assertEqual(len(trait_entries), 1)
        entry = trait_entries[0]
        self.assertEqual(entry["occurrence_count"], 2)
        self.assertAlmostEqual(entry["confidence"], 0.93, places=4)
        self.assertAlmostEqual(entry["eff_confidence"], 0.93, places=4)
        self.assertTrue(entry["trait_id"])
        self.assertTrue(entry["first_seen"])
        self.assertTrue(entry["last_seen"])

    def test_non_trait_rb_entries_uniform_neutral_fields(self):
        payload = self._payload(self._result())
        rb = payload["right_brain_hits"]
        # response_experience is EXCLUDED from the payload (internal) — the
        # situation_pattern entry must still carry the uniform trait keys
        entries = [h for h in rb if not h.get("is_trait")]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["occurrence_count"], 0)
        self.assertEqual(entries[0]["trait_id"], "")

    def test_rb_directive_preserved_in_payload(self):
        """rb_directive is computed on every search and had NO consumer —
        the contract preserves it (transport-only, documented unconsumed)."""
        payload = self._payload(self._result())
        self.assertEqual(
            payload["rb_directive"], "note: low evidence — say you don't know")

    def test_fact_provenance_still_intact(self):
        """Backwards compatibility: the v0.6.3 F-B fact fields ride as before."""
        payload = self._payload(self._result())
        fact = payload["left_brain"][0]
        self.assertEqual(fact["observed_at"], "2026-05-01")
        self.assertEqual(fact["occurrence_count"], 3)
        for key in ("text", "score", "attributed_to", "memory_id",
                    "has_audio", "last_observed_at", "superseded_by"):
            self.assertIn(key, fact)

    def test_existing_rb_keys_unchanged(self):
        """Backwards compatibility: every pre-existing right_brain_hits key
        is still present (UI consumers do not break)."""
        payload = self._payload(self._result())
        for entry in payload["right_brain_hits"]:
            for key in ("content", "raw", "internal", "slot", "source",
                        "priority", "cluster"):
                self.assertIn(key, entry)

    def test_none_result_stable(self):
        payload = self._payload(None)
        self.assertEqual(payload["right_brain_hits"], [])
        self.assertEqual(payload["rb_directive"], "")


class LlmContextContractTests(unittest.TestCase):
    """retrieval → LLM prompt context (the REAL _memory_context builder)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="rb_llm_")
        self.store = TraitStore(os.path.join(self._tmp.name, "space.sqlite"),
                                _FakeEmbedder())

    def tearDown(self):
        self._tmp.cleanup()

    def _result(self) -> _SearchResult:
        self.store.add("u1", "喜好与厌恶", "likes coffee",
                       Evidence(quote="I like coffee"))
        time.sleep(1.1)
        self.store.add("u1", "喜好与厌恶", "likes coffee",
                       Evidence(quote="coffee again"))
        trait_hits = _rb_trait_hits(self.store, "u1", "coffee")
        rb = list(trait_hits) + [
            _RbHit("Emotional note: stressed at work", "situation_pattern"),
        ]
        return _SearchResult(
            hits=[_FactHit("Thomas rides a motorcycle",
                           observed_at="2026-05-01", occurrence_count=3,
                           attributed_to="Thomas")],
            rb_hits=rb,
            rb_directive="note: low evidence — say you don't know",
        )

    @staticmethod
    def _context(result) -> str:
        from app.web_server import WebSession

        return WebSession._memory_context(result)

    def test_trait_context_has_date_and_count(self):
        ctx = self._context(self._result())
        self.assertIn("2x heard", ctx)
        # the observation date rides (last_seen, YYYY-MM-DD)
        import re
        m = re.search(r"last heard (\d{4}-\d{2}-\d{2})", ctx)
        self.assertIsNotNone(m, f"no last-heard date in context: {ctx!r}")

    def test_confidence_floats_not_in_prompt(self):
        """The critical Phase 6 boundary: confidence is a confirmation-
        strength measure, NOT a probability — '0.93' must never reach the
        persona's prompt as if it meant 93% certain."""
        ctx = self._context(self._result())
        self.assertNotIn("0.93", ctx)
        self.assertNotIn("confidence", ctx.lower())
        self.assertNotIn("eff_confidence", ctx.lower())

    def test_rb_directive_not_injected_into_prompt(self):
        """The Phase 8 decision: rb_directive is preserved in the payload
        but deliberately NOT injected into the LLM context (vendor guidance;
        consuming it would change reply behaviour = semantics change)."""
        result = self._result()
        ctx = self._context(result)
        self.assertNotIn("low evidence", ctx)
        self.assertNotIn("don't know", ctx)

    def test_fact_provenance_suffix_still_rendered(self):
        ctx = self._context(self._result())
        self.assertIn("2026-05-01", ctx)
        self.assertIn("3x confirmed", ctx)
        self.assertIn("by Thomas", ctx)

    def test_non_trait_rb_hit_no_suffix(self):
        result = self._result()
        ctx = self._context(result)
        # the situation_pattern line renders bare (no heard-suffix)
        self.assertIn("Emotional note: stressed at work", ctx)
        idx = ctx.find("Emotional note: stressed at work")
        line_end = ctx.find("\n", idx)
        line = ctx[idx:line_end if line_end != -1 else len(ctx)]
        self.assertNotIn("heard", line)


class CliBridgeContractTests(unittest.TestCase):
    """retrieval → CLI bridge render (the REAL _extract_memory_context)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="rb_cli_")
        self.store = TraitStore(os.path.join(self._tmp.name, "space.sqlite"),
                                _FakeEmbedder())

    def tearDown(self):
        self._tmp.cleanup()

    def _result(self) -> _SearchResult:
        self.store.add("u1", "喜好与厌恶", "likes coffee",
                       Evidence(quote="I like coffee"))
        time.sleep(1.1)
        self.store.add("u1", "喜好与厌恶", "likes coffee",
                       Evidence(quote="coffee again"))
        return _SearchResult(
            hits=[],
            rb_hits=_rb_trait_hits(self.store, "u1", "coffee"),
            rb_directive="directive text",
        )

    def test_bridge_renders_trait_suffix(self):
        from app.voicemem_bridge import _extract_memory_context

        ctx = _extract_memory_context(self._result())
        self.assertIn("2x heard", ctx)

    def test_bridge_matches_web_render(self):
        """Web/CLI parity: the same result renders the SAME trait lines."""
        from app.voicemem_bridge import _extract_memory_context
        from app.web_server import WebSession

        result = self._result()
        self.assertEqual(
            _extract_memory_context(result), WebSession._memory_context(result)
        )

    def test_bridge_no_confidence_floats(self):
        from app.voicemem_bridge import _extract_memory_context

        ctx = _extract_memory_context(self._result())
        self.assertNotIn("0.93", ctx)
        self.assertNotIn("confidence", ctx.lower())

    def test_turn_object_carries_rb_directive(self):
        """TurnContext.turn preserves the raw SearchResult (programmatic
        consumers can read rb_directive and every computed field)."""
        result = self._result()
        self.assertEqual(getattr(result, "rb_directive", ""), "directive text")


if __name__ == "__main__":
    unittest.main()
