"""Regression tests for the v0.5.1 (TASK 1) bridge adapter fix.

The forensic run proved the break: the bridge's legacy adapter tables target
method names (feed_partial/feed/process_text/handle_text, add_message/
remember) that DO NOT exist on the pinned vendor (e8384e0) — its public API
is ingest/search/classify/reply/stream/flush — so every CLI voice turn ran
with an empty memory_context and the reply was never consolidated.

These tests drive the REAL VoiceMemBridge against a duck-typed facade that
exposes EXACTLY the pinned vendor's public API surface:

* process_turn -> vm.search(transcript) + SearchResult (hits/rb_hits) ->
  memory_context rendered like the proven web path
* commit_reply -> vm.ingest(last_user_text, agent_reply=reply) -> committed
* store_fact -> vm.ingest(fact)
* extraction failure/search failure degrade without crashing

Env-independent (the fake vm is injected into the cache and the
_HAS_VOICEMEM gate is patched) plus a drift guard asserting the REAL pinned
vendor still exposes search/ingest when importable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import app.voicemem_bridge as vb
from app.config import AgentConfig
from app.voicemem_bridge import (
    COMMIT_COMMITTED,
    VoiceMemBridge,
    _extract_memory_context,
    clean_rb_content,
)


class _Hit:
    def __init__(self, text: str, score: float = 0.9) -> None:
        self.text = text
        self.score = score


class _RbHit:
    def __init__(self, content: str) -> None:
        self.content = content


class _SearchResult:
    """Duck type of vendor orchestrator.SearchResult (hits + rb_hits)."""

    def __init__(self, hits: list, rb_hits: list) -> None:
        self.hits = hits
        self.rb_hits = rb_hits


class _PinnedVendorVm:
    """Duck type exposing EXACTLY the pinned vendor's public API surface."""

    def __init__(self) -> None:
        self.search_calls: list[str] = []
        self.ingest_calls: list[tuple[str, dict]] = []
        self.search_result = _SearchResult(
            hits=[_Hit("The user finds the present perfect difficult")],
            rb_hits=[_RbHit("[2026-09-10] 情绪: prefers encouraging feedback (next time: keep it short)")],
        )
        self.search_exc: Exception | None = None
        self.ingest_exc: Exception | None = None

    # -- the pinned vendor's actual public API --------------------------------
    def search(self, query: str, **kw: Any) -> _SearchResult:
        self.search_calls.append(query)
        if self.search_exc:
            raise self.search_exc
        return self.search_result

    def ingest(self, text: str = None, audio: Any = None, **kw: Any) -> dict:
        self.ingest_calls.append((text or "", dict(kw)))
        if self.ingest_exc:
            raise self.ingest_exc
        return {"facts_count": 1, "memory_ids": ["m1"], "affect": None}


class BridgeAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = AgentConfig()
        self.bridge = VoiceMemBridge(self.config)
        self.vm = _PinnedVendorVm()
        # inject the fake facade + open the _HAS_VOICEMEM gate without
        # requiring the real package in this environment
        self.bridge._vm_by_user["voice_user"] = self.vm
        self.bridge._vm_last_access["voice_user"] = __import__("time").monotonic()
        self._gate = patch.object(vb, "_HAS_VOICEMEM", True)
        self._gate.start()

    def tearDown(self) -> None:
        self._gate.stop()

    async def test_process_turn_uses_search_and_renders_context(self) -> None:
        ctx = await self.bridge.process_turn("Nehezen használom a present perfectet.")
        self.assertEqual(self.vm.search_calls, ["Nehezen használom a present perfectet."])
        # the left-brain hit is rendered verbatim...
        self.assertIn("- The user finds the present perfect difficult", ctx.memory_context)
        # ...the right-brain note is cleaned (date/slot prefix + advisory suffix)
        self.assertIn("prefers encouraging feedback", ctx.memory_context)
        self.assertNotIn("2026-09-10", ctx.memory_context)
        self.assertNotIn("next time", ctx.memory_context)
        self.assertEqual(ctx.turn, self.vm.search_result)

    async def test_process_turn_records_transcript_for_commit(self) -> None:
        transcript = "Nehezen használom a present perfectet."
        await self.bridge.process_turn(transcript)
        self.assertEqual(self.bridge._last_user_text.get("voice_user"), transcript)

    async def test_commit_reply_uses_ingest_with_the_pair(self) -> None:
        transcript = "Nehezen használom a present perfectet."
        reply = "Rendben, gyakoroljuk a present perfectet."
        await self.bridge.process_turn(transcript)
        status = await self.bridge.commit_reply(reply)
        self.assertEqual(status, COMMIT_COMMITTED)
        self.assertEqual(len(self.vm.ingest_calls), 1)
        text, kw = self.vm.ingest_calls[0]
        self.assertEqual(text, transcript)
        self.assertEqual(kw.get("agent_reply"), reply)
        self.assertTrue(kw.get("async_facts"))

    async def test_store_fact_uses_ingest(self) -> None:
        await self.bridge.store_fact("Emotional state: the user sounded frustrated.")
        self.assertEqual(len(self.vm.ingest_calls), 1)
        text, kw = self.vm.ingest_calls[0]
        self.assertEqual(text, "Emotional state: the user sounded frustrated.")

    async def test_search_failure_degrades_to_empty_context(self) -> None:
        self.vm.search_exc = RuntimeError("boom")
        ctx = await self.bridge.process_turn("Hello")
        self.assertEqual(ctx.memory_context, "")
        self.assertIsNotNone(ctx)  # the turn survives, memory degrades

    async def test_commit_ingest_failure_reports_failed(self) -> None:
        await self.bridge.process_turn("Hello")
        self.vm.ingest_exc = RuntimeError("boom")
        status = await self.bridge.commit_reply("a reply")
        self.assertEqual(status, vb.COMMIT_FAILED)


class SearchResultRenderingTests(unittest.TestCase):
    def test_hits_and_rb_hits_rendered(self) -> None:
        result = _SearchResult(
            hits=[_Hit("tea, two sugars"), _Hit("mornings in Budapest")],
            rb_hits=[_RbHit("情绪: dislikes cold rooms")],
        )
        ctx = _extract_memory_context(result)
        self.assertIn("- tea, two sugars", ctx)
        self.assertIn("- mornings in Budapest", ctx)
        self.assertIn("dislikes cold rooms", ctx)

    def test_cap_1200(self) -> None:
        result = _SearchResult(hits=[_Hit("x" * 500) for _ in range(10)], rb_hits=[])
        self.assertLessEqual(len(_extract_memory_context(result)), 1200)

    def test_generic_objects_still_work(self) -> None:
        class Legacy:
            memories = ["legacy shape"]

        self.assertIn("legacy shape", _extract_memory_context(Legacy()))

    def test_clean_rb_content(self) -> None:
        cleaned = clean_rb_content("[2026-09-10] 表达风格: concise (下次: shorter)")
        self.assertEqual(cleaned, "concise")


class PinnedVendorDriftGuard(unittest.TestCase):
    """The ACTUAL regression guard for the forensic finding: if the pinned
    vendor ever changes its public API again (or the pin drifts), this fails
    BEFORE the bridge silently degrades in production again."""

    def test_pinned_vendor_exposes_search_and_ingest(self) -> None:
        try:
            import voicemem  # noqa: PLC0415
        except ImportError:
            self.skipTest("voicemem not installed in this environment")
        for method in ("search", "ingest"):
            self.assertTrue(
                callable(getattr(voicemem.VoiceMem, method, None)),
                f"pinned vendor VoiceMem no longer exposes {method}() — the "
                "bridge adapters need re-verification (see TASK 1 forensics)",
            )


if __name__ == "__main__":
    unittest.main()
