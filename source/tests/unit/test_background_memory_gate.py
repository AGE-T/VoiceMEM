"""v0.9.2 PART 5 — conversation-first background memory scheduler tests.

Covers app/background_memory.py (BackgroundMemoryGate):
  * background ingest WAITS while a conversation turn is active
  * release() opens the gate only after the grace period
  * a speech frame inside the grace window re-arms (defers again)
  * queued turns are processed sequentially, in order, never dropped
  * exact duplicate (user_text, reply) pairs are coalesced
  * an ingest failure does not kill the worker (next item still runs)
  * the gate calls the ingest with the REAL-completion contract and fires
    the on_begin/on_end/on_fail status callbacks
  * arm() is idempotent and never blocks the caller
  * the worker runs through the session's spawn registry (disconnect cancel)
"""

from __future__ import annotations

import asyncio
import os
import unittest

from app.background_memory import BackgroundMemoryGate


class _Recorder:
    """SYNC ingest stub — the gate runs it via asyncio.to_thread, so it must
    be a plain callable (an async def would silently never run)."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, user_text: str, reply: str) -> None:
        self.calls.append((user_text, reply))

    def done(self) -> list[str]:
        return [u for u, _ in self.calls]


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class GateTests(unittest.TestCase):
    def _make(self, **kw):
        rec = _Recorder()
        gate = BackgroundMemoryGate(rec, grace_s=0.05, **kw)
        return gate, rec

    def test_ingest_waits_while_turn_active(self):
        async def scene():
            gate, rec = self._make()
            gate.arm("speech")
            await gate.submit("hello", "hi there")
            await asyncio.sleep(0.15)  # > grace, but still armed
            self.assertEqual(rec.calls, [], "must not run while armed")
            gate.release()
            await asyncio.sleep(0.2)   # grace 0.05 + margin
            self.assertEqual(rec.done(), ["hello"])
        _run(scene())

    def test_release_opens_only_after_grace(self):
        async def scene():
            gate, rec = self._make()
            gate.arm("speech")
            gate.release()
            await gate.submit("a", "b")
            # immediately after release: still inside grace
            await asyncio.sleep(0.02)
            self.assertEqual(rec.calls, [], "grace must hold the gate")
            await asyncio.sleep(0.2)
            self.assertEqual(rec.done(), ["a"])
        _run(scene())

    def test_speech_rearm_inside_grace_defers(self):
        async def scene():
            gate, rec = self._make()
            gate.arm("speech")
            gate.release()
            await gate.submit("x", "y")
            await asyncio.sleep(0.02)  # inside grace
            gate.arm("speech")         # user speaks again
            await asyncio.sleep(0.2)   # far past original grace
            self.assertEqual(rec.calls, [], "re-arm must cancel the grace opening")
            gate.release()
            await asyncio.sleep(0.2)
            self.assertEqual(rec.done(), ["x"])
        _run(scene())

    def test_queue_sequential_in_order_never_dropped(self):
        async def scene():
            gate, rec = self._make()
            gate.arm("speech")
            for i in range(3):
                await gate.submit(f"u{i}", f"r{i}")
            gate.release()
            await asyncio.sleep(0.6)
            self.assertEqual(rec.done(), ["u0", "u1", "u2"])
        _run(scene())

    def test_exact_duplicates_coalesced(self):
        async def scene():
            gate, rec = self._make()
            gate.arm("speech")
            await gate.submit("same", "same reply")
            await gate.submit("same", "same reply")
            await gate.submit("other", "other reply")
            self.assertEqual(gate.pending, 2, "duplicate pair must be coalesced")
            gate.release()
            await asyncio.sleep(0.3)
            self.assertEqual(rec.done(), ["same", "other"])
            self.assertEqual(gate.stats.coalesced, 1)
        _run(scene())

    def test_ingest_failure_does_not_kill_worker(self):
        async def scene():
            calls: list[str] = []
            fails: list[str] = []
            def bad_ingest(u, r):
                calls.append(u)
                if u == "u0":
                    raise RuntimeError("boom")
            gate = BackgroundMemoryGate(
                bad_ingest, grace_s=0.05, on_fail=fails.append
            )
            gate.arm("speech")
            await gate.submit("u0", "r0")
            await gate.submit("u1", "r1")
            gate.release()
            await asyncio.sleep(0.3)
            self.assertEqual(calls, ["u0", "u1"], "worker must survive the failure")
            self.assertEqual(fails, ["boom"])
            self.assertEqual(gate.stats.failed, 1)
            self.assertEqual(gate.stats.finished, 1, "u1 completed normally")
        _run(scene())

    def test_status_callbacks_fire_around_real_completion(self):
        async def scene():
            events: list[str] = []
            gate = BackgroundMemoryGate(
                lambda u, r: events.append("ingest"),
                grace_s=0.05,
                on_begin=lambda: events.append("begin"),
                on_end=lambda: events.append("end"),
            )
            gate.arm("speech")
            await gate.submit("u", "r")
            gate.release()
            await asyncio.sleep(0.2)
            self.assertEqual(events, ["begin", "ingest", "end"])
        _run(scene())

    def test_arm_is_idempotent_and_fast(self):
        async def scene():
            gate, _ = self._make()
            for _ in range(50):
                gate.arm("speech")
            self.assertTrue(gate.is_armed)
            self.assertEqual(gate.stats.arm_count, 1, "re-arm while armed is not counted")
        _run(scene())

    def test_worker_uses_session_spawn_registry(self):
        async def scene():
            spawned: list = []
            def spawn(coro):
                task = asyncio.get_running_loop().create_task(coro)
                spawned.append(task)
                return task
            gate = BackgroundMemoryGate(lambda u, r: None, grace_s=0.05, spawn=spawn)
            gate.arm("speech")
            await gate.submit("u", "r")
            gate.release()
            await asyncio.sleep(0.2)
            self.assertEqual(len(spawned), 1, "worker must go through spawn()")
            self.assertTrue(spawned[0].done())
        _run(scene())

    def test_grace_env_override(self):
        async def scene():
            os.environ["VOICEMEM_BG_GRACE_S"] = "0.01"
            try:
                gate, rec = self._make()  # constructor re-reads env
                gate.arm("s")
                gate.release()
                await gate.submit("u", "r")
                await asyncio.sleep(0.12)
                self.assertEqual(rec.done(), ["u"])
            finally:
                del os.environ["VOICEMEM_BG_GRACE_S"]
        _run(scene())

    def test_invalid_grace_env_falls_back(self):
        os.environ["VOICEMEM_BG_GRACE_S"] = "not-a-number"
        try:
            gate = BackgroundMemoryGate(lambda u, r: None)
            self.assertEqual(gate._grace_s, 2.0)
        finally:
            del os.environ["VOICEMEM_BG_GRACE_S"]


class RealLayerContractTests(unittest.TestCase):
    """The gate's ingest path must request REAL completion (wait=True)."""

    def test_real_memory_layer_wait_kwarg_forwards_async_facts_false(self):
        """RealMemoryLayer.ingest(wait=True) -> vendor async_facts=False."""
        from app.web_server import RealMemoryLayer

        captured: dict = {}

        class _FakeVm:
            def ingest(self, text, agent_reply=None, async_facts=True):
                captured["text"] = text
                captured["agent_reply"] = agent_reply
                captured["async_facts"] = async_facts

        layer = RealMemoryLayer.__new__(RealMemoryLayer)
        layer._active = "default"
        layer._spaces = {}
        layer._failed = False
        layer._facade = lambda safe: _FakeVm()
        layer.ingest("hello", "hi", wait=True)
        self.assertFalse(captured["async_facts"],
                         "wait=True must run the vendor chain synchronously")
        layer.ingest("hello", "hi")
        self.assertTrue(captured["async_facts"],
                       "default stays fire-and-forget (historical callers)")


if __name__ == "__main__":
    unittest.main()
