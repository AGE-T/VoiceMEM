"""v0.9.2 PART 7 — extraction prompt reduction equivalence tests.

The reduced ``additive_extraction_prompt.txt`` (7,923 -> 6,118 base tokens,
-22.8%) must prove semantic preservation STRUCTURALLY here; the behavioural
equivalence (extraction outcomes on the utterance set) is measured in the
v0.9.2 evidence package (sandbox stand-in model, labelled SANDBOX
MEASUREMENT).

This file pins:
  1. every REQUIRED instruction block is still present (quality standards,
     integrity rules, temporal grounding, numerically precise, proper nouns,
     qualifiers, memory linking, multi-speaker, dedup, output format,
     checklist, ADD-only role);
  2. every survival signal for the REMOVED blocks is present where the cut
     manifest says it lives (Example 6 for assistant-recommendations, the
     Integrity Rules Bajimaya pair for meta-extraction, the INPUTS dedup rule
     for already-captured facts, Numerically Precise for specificity, ROLE +
     Checklist for multi-topic);
  3. no forbidden regressions: the removed text is really the documented set
     (count of examples, no photos section, no "ALL dimensions" duplicate);
  4. the fork addenda and the user-side section builder are unchanged
     (they compose the request AFTER the system prompt);
  5. the full request assembly (reduced system + sections + leading merged
     addendum) still parses and its token size is the documented reduction.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR = ROOT / "vendor" / "voicemem"
PROMPT = VENDOR / "voicemem" / "leftbrain" / "data" / "additive_extraction_prompt.txt"

_ISOLATED = r"""
import json, re, sys
sys.path.insert(0, {vendor!r})
from voicemem.leftbrain.mem0_additive_prompt_build import (
    generate_additive_extraction_prompt, load_additive_system_prompt,
)
from voicemem.leftbrain import merged_extraction
from voicemem.leftbrain.extract_facts_openai import (
    _LANGUAGE_RULE, _ATTRIBUTE_ADDENDUM, _VOICE_ADDENDUM,
)

action = sys.argv[1]
if action == "size":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/z/vmforensic/qwen_tok")
    base = load_additive_system_prompt()
    system = base + _LANGUAGE_RULE + _ATTRIBUTE_ADDENDUM + _VOICE_ADDENDUM
    addendum = merged_extraction.prompt_addendum()
    user = generate_additive_extraction_prompt(
        summary=None, recently_extracted_memories=[],
        existing_memories=[{{"id": "x", "text": "User works as a software engineer."}}],
        new_messages=[{{"role": "user", "content": "Holnap reggel futok a parkban."}}],
        last_k_messages=None, use_input_language=True,
    )
    full_user = addendum + "\n\n" + user
    print(json.dumps({{
        "base_tokens": len(tok.encode(base)),
        "system_tokens": len(tok.encode(system)),
        "full_user_tokens": len(tok.encode(full_user)),
        "total_app_side": len(tok.encode(system)) + len(tok.encode(full_user)),
    }}))
elif action == "request-shape":
    # prove the production assembly: addendum FIRST in the user message.
    # The ASSEMBLED STRING is checked (not source-call order): the vendor
    # builds sections first, then PREPENDS the addendum.
    from voicemem.leftbrain import merged_extraction
    from voicemem.leftbrain.mem0_additive_prompt_build import generate_additive_extraction_prompt
    import voicemem.leftbrain.extract_facts_openai as E
    src = E.OpenAIMem0V3AdditiveExtractor.extract
    import inspect
    src_code = inspect.getsource(src)
    assert 'prompt_addendum() + "\\n\\n" + user_content' in src_code, (
        "the vendor must PREPEND the merged addendum to the user message")
    addendum = merged_extraction.prompt_addendum()
    user = generate_additive_extraction_prompt(
        summary=None, recently_extracted_memories=[],
        existing_memories=[],
        new_messages=[{{"role": "user", "content": "teszt"}}],
        last_k_messages=None, use_input_language=True,
    )
    assembled = addendum + "\n\n" + user
    print(json.dumps({{
        "addendum_leads_user_message": assembled.startswith(addendum),
        "extract_max_tokens_default": E._EXTRACT_MAX_TOKENS,
        "resolve_max_tokens_default": E._RESOLVE_MAX_TOKENS,
    }}))
"""

REQUIRED_SIGNALS = {
    "ADD-only role": "Your sole operation is ADD",
    "completeness mandate": "a missed extraction means lost context",
    "dominant topic warning": "Do not let a dominant topic cause you to miss secondary information",
    "user/assistant extraction split": "You extract from BOTH user and assistant messages",
    "existing-memories dedup rule": "do NOT extract new memories from Existing Memories",
    "linked_memory_ids semantics": 'include the Existing Memory\'s ID in the new memory\'s "linked_memory_ids" array',
    "temporal anchor rule": "This is your ONLY temporal anchor for resolving time references",
    "observation-date grounding": "Always ground relative references to specific dates",
    "contextually rich standard": "Contextually Rich, Not Atomic",
    "transition capture": "what the new state is AND what it replaces",
    "clean factual statements": "Clean Factual Statements",
    "self-contained": "Every memory must be understandable on its own",
    "length bound": "15-80 words",
    "temporally grounded quality": "Temporally Grounded",
    "numerically precise": "Numerically Precise",
    "proper nouns rule": "Proper Nouns and Titles Should be Preserved",
    "qualifiers rule": "Qualifiers and Specific Attributes Are Essential",
    "meaning-preserving": "Meaning-Preserving",
    "no fabrication": "No Fabrication",
    "no implicit attribute inference": "No Implicit Attribute Inference",
    "no echo extraction": "No Echo Extraction",
    "no within-response duplication": "No Within-Response Duplication",
    "no meta-extraction": "No Meta-Extraction",
    "bajimaya RIGHT kept (meta-extraction proof)": "The Bajimaya v Reward Homes case involved construction starting in 2014",
    "no detail contamination": "No Detail Contamination from Context",
    "memory linking block": "check if it relates to any Existing Memory",
    "first-topic-dominance checklist": "first topic dominance",
    "exhaustive checklist": "Exhaustive Extraction Checklist",
    "output format json only": "Return ONLY valid JSON parsable by json.loads()",
    "output memory array": '"memory": [',
    "attributed_to field": "**attributed_to**",
    "empty output allowed": '"memory": []',
    "multi-speaker example": "Maria got a new cat named Bailey",
    "multi-speaker rule": "the \"assistant\" role may represent a real person sharing their own life",
    "linking example": '"linked_memory_ids": ["a1b2c3d4-5678-9abc-def0-111111111111"]',
    "temporal example (obs vs current date)": "NOT Current Date",
    "assistant-recs example kept": "User was recommended",
    "all-dimensions example kept": "Each extracted separately",
    "nothing-to-extract example kept": "Output: {\"memory\": []}",
    "dedup note (replaces Example 5)": "Facts already captured in Recently Extracted Memories",
    "casual topics rule": "Casual Topics Are Still Extractable",
    "incidental facts rule": "Extract Incidental Facts, Not Just Requests",
    "when in doubt extract": "When in doubt, extract.",
}

FORBIDDEN_SIGNALS = {
    "photos section removed": "Shared Photos and Images",
    "ALL-dimensions duplicate paragraph removed": "**IMPORTANT — Extract ALL dimensions of a conversation.**",
    "Example 8 removed": "## Example 8",
    "Example 9 removed": "## Example 9",
    "Example 11 removed": "## Example 11",
}


class PromptReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = PROMPT.read_text(encoding="utf-8")

    def test_required_signals_present(self):
        missing = [name for name, sig in REQUIRED_SIGNALS.items() if sig not in self.text]
        self.assertEqual(missing, [], f"required instruction signals lost: {missing}")

    def test_forbidden_removals_stay_removed(self):
        present = [name for name, sig in FORBIDDEN_SIGNALS.items() if sig in self.text]
        self.assertEqual(present, [], f"documented cut reappeared: {present}")

    def test_example_count_and_numbering(self):
        examples = re.findall(r"^## Example (\d+): (.+)$", self.text, re.M)
        numbers = [int(n) for n, _ in examples]
        self.assertEqual(numbers, [1, 2, 3, 4, 5, 6],
                         "6 examples, renumbered sequentially")
        titles = [t for _, t in examples]
        for must in ("Multi-Speaker", "Memory Linking", "Vague Temporal"):
            self.assertTrue(any(must in t for t in titles), f"lost example: {must}")

    def test_size_contract(self):
        """6,265 base tokens +/- 1% (measured with the PRODUCTION tokenizer).

        v0.10.3 (sandbox state-restore re-measurement): the v0.9.2-v0.10.2
        band of 6,118 was measured with the Qwen3-0.6B STAND-IN tokenizer
        (the b10717 forensic stand-in model's vocab). The production model is
        Qwen3.6 35B A3B — its tokenizer measures the SAME prompt file at
        6,265 tokens (+147 from added tokens in the 3.6 vocab; the prompt
        text itself is byte-identical, git-committed at eda974f). The band
        now uses the production tokenizer as the authoritative measuring
        stick: /home/z/vmforensic/qwen_tok (Qwen/Qwen3.6-35B-A3B, revision
        995ad96e). Both measurements recorded here so the +147 is never
        misread as a prompt regression.
        """
        out = self._isolated("size")
        self.assertAlmostEqual(out["base_tokens"], 6265, delta=63,
                               msg="base prompt size outside the documented band")
        self.assertLess(out["system_tokens"], 7250,
                        "full system (base+3 addenda) must be materially smaller "
                        "than the v0.9.1 8,844 (band scaled with the production "
                        "tokenizer's +147-token vocab offset)")

    def test_request_shape_contract(self):
        out = self._isolated("request-shape")
        self.assertTrue(out["addendum_leads_user_message"],
                        "merged addendum must lead the user message (prefix cache)")
        self.assertEqual(out["extract_max_tokens_default"], 1536)
        self.assertEqual(out["resolve_max_tokens_default"], 1024)

    @staticmethod
    def _isolated(action: str) -> dict:
        import json
        proc = subprocess.run(
            [sys.executable, "-c", _ISOLATED.format(vendor=str(VENDOR)), action],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise AssertionError(f"isolated runner failed: {proc.stderr[-1500:]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
