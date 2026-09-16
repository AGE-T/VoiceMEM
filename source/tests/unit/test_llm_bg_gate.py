"""v0.10.2 (PART 8/14) — the vendor cooperative-cancel gate unit battery.

Covers ``voicemem/utils/common/llm_bg_gate.py`` (the module the app's
BackgroundMemoryGate drives):

1. set/clear/is_cancelled + the diagnostics counters;
2. ``check_cancel`` raises BEFORE issue and is a ``BaseException`` that
   flies through every ``except Exception`` fallback in the ingest chain
   (the asyncio.CancelledError pattern — a cancellation must never be
   swallowed into an ADD-only fallback);
3. ``bg_chat_create``: streaming assembly (content + final finish_reason
   + usage), kwarg forwarding, mid-stream cancellation closing the
   underlying stream (the llama-server slot release transport, proven
   live on the pinned b10717: see audit/VoiceMEM_llmpriority_v0102);
4. the compat response shape (choices[0].message.content / finish_reason
   / usage) the legs' post-processing reads.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from voicemem.utils.common import llm_bg_gate  # noqa: E402
from voicemem.utils.common.llm_bg_gate import (  # noqa: E402
    BackgroundCancelledError,
    bg_chat_create,
    check_cancel,
)


def _chunk(content=None, finish=None, usage=None):
    chunk = SimpleNamespace(usage=usage)
    chunk.choices = (
        [SimpleNamespace(delta=SimpleNamespace(content=content),
                         finish_reason=finish)]
        if (content is not None or finish is not None) else []
    )
    return chunk


class _FakeStream:
    def __init__(self, chunks):
        self._it = iter(chunks)
        self.closed = False

    def close(self):
        self.closed = True

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class _FakeClient:
    def __init__(self, chunks):
        self.captured_kwargs: dict | None = None
        self._chunks = chunks

    def _create(self, **kwargs):
        self.captured_kwargs = kwargs
        return _FakeStream(self._chunks)


def _client(chunks) -> _FakeClient:
    c = _FakeClient(chunks)
    c.chat = SimpleNamespace(completions=SimpleNamespace(create=c._create))
    return c


class CancelFlagTests(unittest.TestCase):
    def setUp(self):
        llm_bg_gate.BG_CANCEL.clear()
        llm_bg_gate._CANCEL_COUNT = 0
        llm_bg_gate._LAST_LEG = "unset"

    def tearDown(self):
        llm_bg_gate.BG_CANCEL.clear()

    def test_set_and_clear_idempotent(self):
        self.assertFalse(llm_bg_gate.is_cancelled())
        llm_bg_gate.set_cancel("speech")
        self.assertTrue(llm_bg_gate.is_cancelled())
        llm_bg_gate.set_cancel("speech-again")  # idempotent
        self.assertTrue(llm_bg_gate.is_cancelled())
        llm_bg_gate.clear_cancel("idle")
        self.assertFalse(llm_bg_gate.is_cancelled())
        llm_bg_gate.clear_cancel("idle")        # idempotent too
        self.assertFalse(llm_bg_gate.is_cancelled())

    def test_stats_surface(self):
        llm_bg_gate.set_cancel("speech")
        s = llm_bg_gate.stats()
        self.assertTrue(s["cancel_set"])
        self.assertEqual(s["cancel_count"], 1)
        self.assertEqual(s["last_leg"], "speech")
        llm_bg_gate.clear_cancel("idle")
        s = llm_bg_gate.stats()
        self.assertFalse(s["cancel_set"])

    def test_check_cancel_raises_before_issue(self):
        llm_bg_gate.set_cancel("arm")
        with self.assertRaises(BackgroundCancelledError):
            check_cancel("memory-extraction")
        llm_bg_gate.clear_cancel("idle")
        check_cancel("memory-extraction")  # must not raise


class BaseExceptionSemanticsTests(unittest.TestCase):
    """A cancellation must NEVER be swallowed by the chain's fallbacks."""

    def setUp(self):
        llm_bg_gate.BG_CANCEL.clear()

    def tearDown(self):
        llm_bg_gate.BG_CANCEL.clear()

    def test_not_caught_by_except_exception(self):
        llm_bg_gate.set_cancel("arm")

        def fallback_swallowing_leg():
            try:
                check_cancel("conflict-resolution")
            except Exception:  # noqa: BLE001 - the chain's fallback shape
                return "swallowed"
            return "no-cancel"

        # The cancellation MUST propagate OUT of the fallback (the
        # "ConflictResolver failed, falling back to ADD-only" handler in
        # voice_input.py must never swallow it) and reach the gate worker.
        with self.assertRaises(BackgroundCancelledError):
            fallback_swallowing_leg()

    def test_is_base_exception_subclass(self):
        self.assertTrue(issubclass(BackgroundCancelledError, BaseException))
        self.assertFalse(issubclass(BackgroundCancelledError, Exception))


class BgChatCreateTests(unittest.TestCase):
    def setUp(self):
        llm_bg_gate.BG_CANCEL.clear()

    def tearDown(self):
        llm_bg_gate.BG_CANCEL.clear()

    def test_assembles_full_text_with_finish_reason_and_usage(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        client = _client([
            _chunk("Hel"), _chunk("lo "),
            _chunk(None, finish=None),          # empty-choices chunk tolerated
            _chunk("world", finish="stop", usage=usage),
        ])
        resp = bg_chat_create(
            client, leg="memory-extraction",
            messages=[{"role": "user", "content": "x"}],
            model="m", temperature=0, max_tokens=512,
            response_format={"type": "json_object"}, timeout=60.0,
        )
        self.assertEqual(resp.choices[0].message.content, "Hello world")
        self.assertEqual(resp.choices[0].finish_reason, "stop")
        self.assertIs(resp.usage, usage)
        # forwarding: the request itself is streaming + carries the legs' kw
        kw = client.captured_kwargs
        self.assertTrue(kw["stream"])
        self.assertEqual(kw["max_tokens"], 512)
        self.assertEqual(kw["response_format"], {"type": "json_object"})
        self.assertEqual(kw["temperature"], 0)
        self.assertEqual(kw["timeout"], 60.0)
        self.assertEqual(kw["model"], "m")
        self.assertEqual(kw["messages"], [{"role": "user", "content": "x"}])

    def test_empty_content_chunks_ignored(self):
        client = _client([
            _chunk(None, finish=None), _chunk("ok"), _chunk("stop", finish="stop"),
        ])
        resp = bg_chat_create(client, leg="leg", messages=[], model="m")
        self.assertEqual(resp.choices[0].message.content, "okstop")

    def test_midstream_cancel_closes_stream_and_raises(self):
        class _CancelOnSecond(_FakeStream):
            def __init__(self, chunks):
                super().__init__(chunks)
                self._seen = 0

            def __next__(self):
                item = next(self._it)
                self._seen += 1
                if self._seen >= 2:
                    llm_bg_gate.BG_CANCEL.set()
                return item

        client = _FakeClient([])
        stream = _CancelOnSecond([_chunk("partial-"), _chunk("answer")])
        client.chat = SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: stream))
        with self.assertRaises(BackgroundCancelledError):
            bg_chat_create(client, leg="conflict-resolution",
                           messages=[], model="m")
        self.assertTrue(stream.closed,
                        "the HTTP stream must close (server slot release)")

    def test_preissue_cancel_never_touches_the_client(self):
        llm_bg_gate.set_cancel("arm")
        client = _client([_chunk("should never be read")])
        with self.assertRaises(BackgroundCancelledError):
            bg_chat_create(client, leg="memory-extraction",
                           messages=[], model="m")
        self.assertIsNone(client.captured_kwargs,
                          "no request may be issued after cancel")


if __name__ == "__main__":
    unittest.main()
