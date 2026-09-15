"""v0.9.2 PART 6 — extraction/conflict output safety limit regression tests.

Proves (source of truth: vendor/voicemem leftbrain/extract_facts_openai.py):
  1. the pair-ingest extraction request carries an explicit max_tokens
     (llama-server otherwise applies n_predict=-1 = UNBOUNDED decode on the
     shared single slot — the v0.9.1 audit P0-2).
  2. the conflict resolver request carries an explicit max_tokens.
  3. the limits are env-overridable (VOICEMEM_EXTRACT_MAX_TOKENS /
     VOICEMEM_RESOLVE_MAX_TOKENS) and invalid values fall back to defaults.
  4. a truncated response cannot silently create invalid memory operations:
     the parse path fails closed (exception, no partial writes).
  5. conversational max_tokens (app side, 512) is NOT changed by this.

The vendor code under test runs in a SUBPROCESS (see
_isolation_extraction_limits.py) so the parent test process never gains an
importable voicemem package (same discipline as test_voicemem_controlled.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RUNNER = Path(__file__).resolve().parent / "_isolation_extraction_limits.py"


def _run(action: str, env_extra: dict | None = None) -> dict:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(RUNNER), action],
        capture_output=True, text=True, timeout=120, env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(f"isolated runner failed: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class ExtractionMaxTokensTests(unittest.TestCase):
    """PART 6: explicit output limits on the background LLM legs."""

    def test_default_limits_present(self):
        limits = _run("show-limits")
        self.assertGreater(limits["extract"], 0, "extraction limit must be positive")
        self.assertGreater(limits["resolve"], 0, "resolver limit must be positive")
        # documented defaults (schema worst case ~800 tokens; 1536 ~ 2x headroom;
        # resolver NONE entries omitted, typically 0-3 entries)
        self.assertEqual(limits["extract"], 1536)
        self.assertEqual(limits["resolve"], 1024)

    def test_env_override_respected_and_invalid_falls_back(self):
        self.assertEqual(
            _run("show-limits", {"VOICEMEM_EXTRACT_MAX_TOKENS": "2048"})["extract"], 2048
        )
        self.assertEqual(
            _run("show-limits", {"VOICEMEM_RESOLVE_MAX_TOKENS": "777"})["resolve"], 777
        )
        # invalid values fall back to defaults, never to unbounded
        self.assertEqual(
            _run("show-limits", {"VOICEMEM_EXTRACT_MAX_TOKENS": "abc"})["extract"], 1536
        )
        self.assertEqual(
            _run("show-limits", {"VOICEMEM_EXTRACT_MAX_TOKENS": "-5"})["extract"], 1536
        )

    def test_extract_request_carries_max_tokens(self):
        cap = _run("capture-extract")
        self.assertIn("max_tokens", cap, "extraction request must set max_tokens")
        self.assertGreater(cap["max_tokens"], 0)
        # the rest of the request shape is unchanged
        self.assertEqual(cap["temperature"], 0)
        self.assertEqual(cap["response_format"], {"type": "json_object"})
        self.assertEqual(cap["model"], "m")

    def test_resolve_request_carries_max_tokens(self):
        cap = _run("capture-resolve")
        self.assertIn("max_tokens", cap, "conflict request must set max_tokens")
        self.assertGreater(cap["max_tokens"], 0)

    def test_truncated_output_fails_closed(self):
        """A response cut mid-JSON by the limit must NOT produce a partial
        memory write: json parsing raises, extract() propagates."""
        cap = _run("capture-extract", {"VOICEMEM_EXTRACT_MAX_TOKENS": "4"})
        self.assertFalse(cap["parse_ok"], "truncated JSON must fail the parse")
        self.assertIn("JSONDecodeError", cap["err"])

    def test_conversational_max_tokens_unchanged(self):
        """The app-side chat budget stays 512 (config-pinned, not touched)."""
        sys.path.insert(0, str(ROOT))
        from app.llm_config import load_llm_config
        cfg = load_llm_config(root=ROOT)
        self.assertEqual(cfg.max_tokens, 512)
        self.assertEqual(cfg.to_dict()["max_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
