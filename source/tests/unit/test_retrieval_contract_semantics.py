"""[v0.9.0 — VM-LOCAL-015] The retrieval-contract extension for trait
semantic state: the typed view, the uniform payload keys, and the prompt
suffix marker (Phase 12 scenario 15: the prompt representation must
distinguish current and historical state).

These extend (never weaken) tests/unit/test_retrieval_contract.py — the
v0.8.1 contract tests still run unchanged; this file covers the NEW keys:

  * ``stance`` / ``supersedes`` / ``superseded_by`` / ``superseded_at``
    ride the typed view and the uniform payload (neutral defaults for
    non-trait hits and for legacy metadata without the keys);
  * ``trait_prompt_suffix`` renders ``superseded`` for a historical trait
    (the fact-side wording — one term, one meaning) and does NOT render it
    for a current one; the LLM-context rule "absence of the marker =
    current" stays intact;
  * the confidence floats are still never rendered (v0.8.1 discipline).
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.retrieval_contract import (  # noqa: E402
    TRAIT_PAYLOAD_KEYS,
    extract_trait_info,
    trait_fields_payload,
    trait_prompt_suffix,
)


@dataclass
class _Hit:
    content: str = "likes motorcycles（喜好与厌恶）"
    source: str = "profile"
    priority: float = 0.5
    metadata: dict = field(default_factory=dict)


def _trait_meta(**over):
    md = {
        "slot_name": "喜好与厌恶", "trait_id": "t1", "claim": "likes motorcycles",
        "confidence": 0.93, "eff_confidence": 0.9, "occurrence_count": 2,
        "first_seen": "2026-06-01T00:00:00", "last_seen": "2026-09-01T00:00:00",
        "stance": "pos", "supersedes": "", "superseded_by": "",
        "superseded_at": "",
    }
    md.update(over)
    return md


class TypedViewTests(unittest.TestCase):
    def test_semantic_fields_extracted(self):
        info = extract_trait_info(_Hit(metadata=_trait_meta(
            stance="neg", supersedes="t0", superseded_by="t2",
            superseded_at="2026-09-14T12:00:00")))
        self.assertIsNotNone(info)
        self.assertEqual(info.stance, "neg")
        self.assertEqual(info.supersedes, "t0")
        self.assertEqual(info.superseded_by, "t2")
        self.assertEqual(info.superseded_at, "2026-09-14T12:00:00")

    def test_legacy_metadata_defaults_to_neutral(self):
        # v0.8.1-era metadata: no semantic keys at all — everything neutral.
        info = extract_trait_info(_Hit(metadata={
            "slot_name": "喜好与厌恶", "trait_id": "t1", "claim": "likes motorcycles",
            "confidence": 0.9, "eff_confidence": 0.9, "occurrence_count": 1,
            "first_seen": "", "last_seen": ""}))
        self.assertEqual(info.stance, "")
        self.assertEqual(info.superseded_by, "")
        self.assertFalse(info.superseded_at)

    def test_non_trait_hit_returns_none(self):
        self.assertIsNone(extract_trait_info(_Hit(metadata={"emotion": "joy"})))


class PayloadTests(unittest.TestCase):
    def test_keys_schema_extended(self):
        for key in ("stance", "supersedes", "superseded_by", "superseded_at"):
            self.assertIn(key, TRAIT_PAYLOAD_KEYS)

    def test_current_trait_payload(self):
        payload = trait_fields_payload(_Hit(metadata=_trait_meta(stance="pos")))
        self.assertTrue(payload["is_trait"])
        self.assertEqual(payload["stance"], "pos")
        self.assertEqual(payload["superseded_by"], "")

    def test_superseded_trait_payload(self):
        payload = trait_fields_payload(_Hit(metadata=_trait_meta(
            stance="pos", superseded_by="t2",
            superseded_at="2026-09-14T12:00:00")))
        self.assertEqual(payload["superseded_by"], "t2")
        self.assertEqual(payload["superseded_at"], "2026-09-14T12:00:00")
        self.assertEqual(payload["stance"], "pos")

    def test_non_trait_neutral_defaults(self):
        payload = trait_fields_payload(_Hit(metadata={}))
        self.assertFalse(payload["is_trait"])
        self.assertEqual(payload["stance"], "")
        self.assertEqual(payload["superseded_by"], "")
        self.assertEqual(payload["supersedes"], "")
        self.assertEqual(payload["superseded_at"], "")
        # the v0.8.1 neutral defaults are unchanged
        self.assertEqual(payload["occurrence_count"], 0)


class PromptSuffixTests(unittest.TestCase):
    def test_current_trait_has_no_currency_marker(self):
        suffix = trait_prompt_suffix(_Hit(metadata=_trait_meta()))
        self.assertNotIn("superseded", suffix)
        # the v0.8.1 provenance render is unchanged for current traits
        self.assertIn("last heard 2026-09-01", suffix)
        self.assertIn("2x heard", suffix)

    def test_superseded_trait_marked(self):
        suffix = trait_prompt_suffix(_Hit(metadata=_trait_meta(
            superseded_by="t2", superseded_at="2026-09-14T12:00:00")))
        self.assertIn("superseded", suffix,
                      "a historical trait must be marked in the prompt")
        self.assertIn("last heard 2026-09-01", suffix,
                      "frozen observation provenance still renders")
        self.assertIn("2x heard", suffix,
                      "the frozen count renders (pre-flip observations only)")

    def test_confidence_floats_never_rendered(self):
        # v0.8.1 discipline: confidence is confirmation strength, not a
        # probability — never printed into the LLM context.
        for md in (_trait_meta(), _trait_meta(superseded_by="t2")):
            suffix = trait_prompt_suffix(_Hit(metadata=md))
            self.assertNotIn("0.9", suffix)
            self.assertNotIn("confidence", suffix)

    def test_non_trait_hit_no_suffix(self):
        self.assertEqual(trait_prompt_suffix(_Hit(metadata={})), "")


class PromptScenarioTest(unittest.TestCase):
    """Phase 12 scenario 15 end-to-end: the rendered prompt lines for a
    current vs a superseded trait differ exactly by the currency marker."""

    def test_prompt_lines_distinguish_current_and_historical(self):
        current = _Hit(metadata=_trait_meta(
            stance="neg", claim="no longer likes motorcycles"))
        historical = _Hit(metadata=_trait_meta(
            stance="pos", claim="likes motorcycles", superseded_by="t2"))
        line_current = f"- {current.content}{trait_prompt_suffix(current)}"
        line_historical = f"- {historical.content}{trait_prompt_suffix(historical)}"
        self.assertIn("superseded", line_historical)
        self.assertNotIn("superseded", line_current)
        # the unmarked line is the current state, the marked one historical:
        # the distinction is visible to the reply model in one word.


if __name__ == "__main__":
    unittest.main()
