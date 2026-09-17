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
import threading
import os
import unittest

from app.background_memory import BackgroundMemoryGate

try:  # the vendor cooperative-cancel surface (present in prod + sandbox)
    from voicemem.utils.common import llm_bg_gate
    from voicemem.utils.common.llm_bg_gate import BackgroundCancelledError
    HAS_VENDOR_GATE = True
except ImportError:  # degraded-mode install: the cancel tests degrade to no-ops
    llm_bg_gate = None
    BackgroundCancelledError = None
    HAS_VENDOR_GATE = False


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
    def setUp(self):
        if HAS_VENDOR_GATE:
            llm_bg_gate.BG_CANCEL.clear()

    def tearDown(self):
        if HAS_VENDOR_GATE:
            llm_bg_gate.BG_CANCEL.clear()

    def _make(self, **kw):
        rec = _Recorder()
        kw.setdefault("idle_s", 0.05)  # v0.10.2 idle policy: fast in tests
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
                bad_ingest, grace_s=0.05, idle_s=0.05, on_fail=fails.append
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
                grace_s=0.05, idle_s=0.05,
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
            gate = BackgroundMemoryGate(lambda u, r: None, grace_s=0.05, idle_s=0.05, spawn=spawn)
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




@unittest.skipUnless(HAS_VENDOR_GATE, "vendored llm_bg_gate required")
class UserPriorityCancellationTests(unittest.TestCase):
    """v0.10.2 (operator PART 8): arm() cancels in-flight work + requeue."""

    def setUp(self):
        llm_bg_gate.BG_CANCEL.clear()

    def tearDown(self):
        llm_bg_gate.BG_CANCEL.clear()

    def _cancelled_once_ingest(self, state, ready):
        """A vendor-chain stand-in: blocks until released, then check_cancel.

        Models the real legs (extract/resolve via bg_chat_create): while
        running, a cancel event makes the next boundary raise
        BackgroundCancelledError. `state["mode"]` switches it to a plain
        success on the retry.
        """
        def ingest(user_text: str, reply: str) -> None:
            ready.wait(timeout=5.0)
            if state["mode"] == "cancel":
                llm_bg_gate.check_cancel("test-leg")
                raise AssertionError("unreachable when cancelled")
            state["completed"].append(user_text)
        return ingest

    def test_arm_cancels_inflight_and_requeues(self):
        state = {"mode": "cancel", "completed": []}
        ready = threading.Event()

        async def scene():
            gate = BackgroundMemoryGate(
                self._cancelled_once_ingest(state, ready),
                grace_s=0.05, idle_s=0.05,
            )
            await gate.submit("hello", "hi")
            await asyncio.sleep(0.1)       # the ingest is now blocked in the thread
            self.assertTrue(gate.is_processing)
            gate.arm("speech")              # the user needs the slot
            self.assertTrue(llm_bg_gate.BG_CANCEL.is_set(),
                            "arm() must set the vendor cancel event")
            ready.set()                     # the blocked leg proceeds -> sees cancel
            await asyncio.sleep(0.2)
            self.assertFalse(gate.is_processing, "the chain must be gone")
            self.assertEqual(state["completed"], [],
                             "a cancelled ingest NEVER counts as done")
            self.assertEqual(gate.stats.cancelled, 1)
            self.assertEqual(gate.stats.requeued, 1)
            self.assertEqual(gate.stats.finished, 0)
            self.assertEqual(gate.pending, 1, "the turn pair is re-queued")
            # memory store untainted: no partial write happened (state empty)
            # now let the user finish + the conversation go idle: the retry runs
            state["mode"] = "ok"
            gate.release()
            await asyncio.sleep(0.25)
            self.assertEqual(state["completed"], ["hello"],
                             "the requeued pair must be retried after idle")
            self.assertEqual(gate.stats.finished, 1)
        _run(scene())

    def test_requeue_coalesces_identical_resubmit(self):
        """A pair submitted again while its cancelled run is being requeued
        must not duplicate in the queue (the v0.9.2 never-double contract)."""
        state = {"mode": "cancel", "completed": []}
        ready = threading.Event()

        async def scene():
            gate = BackgroundMemoryGate(
                self._cancelled_once_ingest(state, ready),
                grace_s=0.05, idle_s=0.05,
            )
            await gate.submit("dup", "reply")
            await asyncio.sleep(0.1)
            gate.arm("speech")
            ready.set()
            await asyncio.sleep(0.15)        # cancelled + requeued
            await gate.submit("dup", "reply")  # exact duplicate arrives again
            self.assertEqual(gate.pending, 1,
                             "identical pair must coalesce, not duplicate")
            self.assertEqual(gate.stats.coalesced, 1)
        _run(scene())

    def test_stale_result_never_reaches_store(self):
        """The mid-request abort discards partial text INSIDE the leg: the
        store only ever sees a completed leg's result (bg_chat_create
        raises before returning anything). Modelled with the real vendor
        helper over a fake streaming client whose SECOND chunk arrival
        sets the cancel event (the user armed mid-generation)."""
        from types import SimpleNamespace

        from voicemem.utils.common.llm_bg_gate import bg_chat_create

        class _Chunk:
            def __init__(self, content=None, finish=None):
                self.choices = [SimpleNamespace(
                    delta=SimpleNamespace(content=content),
                    finish_reason=finish)] if (content or finish) else []
                self.usage = None

        class _Stream:
            """Iterates chunks; entering the SECOND element arms the cancel."""
            def __init__(self, chunks):
                self._it = iter(chunks)
                self._seen = 0
                self.closed = False
            def close(self):
                self.closed = True
            def __iter__(self):
                return self
            def __next__(self):
                item = next(self._it)
                self._seen += 1
                if self._seen >= 2:
                    llm_bg_gate.BG_CANCEL.set()
                return item

        stream = _Stream([_Chunk("partial-"), _Chunk("answer")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: stream)))
        # the user arms the slot mid-generation: the SECOND chunk arrival
        # sets the cancel event, so the helper must abort BEFORE assembling
        # the full answer — nothing is returned, the stream is closed (the
        # server slot is freed), and the store call site never runs.
        with self.assertRaises(BackgroundCancelledError):
            bg_chat_create(client, leg="stale-test",
                           messages=[{"role": "user", "content": "x"}],
                           model="m")
        self.assertTrue(stream.closed, "the stream must be closed (slot freed)")

    def test_arm_idempotent_cancel_and_activity_clock(self):
        async def scene():
            gate = BackgroundMemoryGate(lambda u, r: None,
                                        grace_s=0.05, idle_s=0.05)
            gate.arm("speech")
            t_after_arm = gate._last_activity
            self.assertGreater(t_after_arm, 0.0)
            gate.arm("speech")   # second arm: no crash, clock refreshes
            self.assertGreaterEqual(gate._last_activity, t_after_arm)
        _run(scene())


@unittest.skipUnless(HAS_VENDOR_GATE, "vendored llm_bg_gate required")
class IdlePolicyTests(unittest.TestCase):
    """v0.10.2 (operator PART 9): background starts only in a QUIET window."""

    def setUp(self):
        llm_bg_gate.BG_CANCEL.clear()

    def tearDown(self):
        llm_bg_gate.BG_CANCEL.clear()

    def test_activity_notes_defer_start(self):
        rec = _Recorder()

        async def scene():
            gate = BackgroundMemoryGate(rec, grace_s=0.05, idle_s=0.4)
            gate.arm("turn")
            await gate.submit("u1", "r1")
            gate.release()                    # reply done at t0
            for _ in range(6):                # user keeps making noise
                await asyncio.sleep(0.1)
                gate.note_activity("vad-frame")
            await asyncio.sleep(0.02)
            self.assertTrue(gate.is_armed,
                             "continuous activity must keep the gate closed")
            self.assertEqual(rec.calls, [], "nothing may start")
            self.assertGreaterEqual(gate.stats.idle_waits, 1)
            await asyncio.sleep(0.5)          # quiet for > idle_s now
            self.assertFalse(gate.is_armed,
                             "quiet window reached: the gate must open")
            await asyncio.sleep(0.15)
            self.assertEqual(rec.done(), ["u1"])
        _run(scene())

    def test_idle_window_measured_from_last_activity(self):
        rec = _Recorder()

        async def scene():
            gate = BackgroundMemoryGate(rec, grace_s=0.05, idle_s=0.25)
            gate.arm("turn")
            await gate.submit("u1", "r1")
            gate.release()
            await asyncio.sleep(0.1)          # INSIDE the grace+idle window
            gate.note_activity("llm-delta")   # a late delta resets the clock
            await asyncio.sleep(0.1)
            self.assertTrue(gate.is_armed, "the idle clock restarted")
            await asyncio.sleep(0.35)         # now quiet long enough
            self.assertFalse(gate.is_armed)
            await asyncio.sleep(0.1)
            self.assertEqual(rec.done(), ["u1"])
        _run(scene())

    def test_start_wait_measured_in_stats(self):
        rec = _Recorder()

        async def scene():
            gate = BackgroundMemoryGate(rec, grace_s=0.05, idle_s=0.15)
            gate.arm("turn")
            await gate.submit("u1", "r1")
            gate.release()
            await asyncio.sleep(0.35)
            self.assertEqual(rec.done(), ["u1"])
            self.assertGreaterEqual(gate.stats.last_start_wait_s, 0.15,
                                    "release->start delay must be measured")
            d = gate.stats.as_dict()
            self.assertIn("last_start_wait_s", d)
            self.assertIn("cancelled", d)
            self.assertIn("requeued", d)
            self.assertIn("idle_waits", d)
        _run(scene())


class GateForensicLoggingTests(unittest.TestCase):
    """v0.10.3 forensic validation prep: the gate emits the acquire/release
    timeline lines that scripts/llm_slot_forensic.py S8 parses from
    logs/web-server.log.

    These pin the LOGGING only — the behaviour itself is pinned by the
    other tests in this file (all unchanged; a logging line must never be
    a behavioural change)."""

    def setUp(self):
        if HAS_VENDOR_GATE:
            llm_bg_gate.BG_CANCEL.clear()

    def tearDown(self):
        if HAS_VENDOR_GATE:
            llm_bg_gate.BG_CANCEL.clear()

    def test_armed_released_open_started_lines(self):
        rec = _Recorder()

        async def scene():
            with self.assertLogs("app.background_memory", level="INFO") as cm:
                gate = BackgroundMemoryGate(rec, grace_s=0.05, idle_s=0.05)
                gate.arm("speech")             # open -> armed (acquire)
                await gate.submit("u1", "r1")  # worker defers while armed
                gate.release()                 # release
                await asyncio.sleep(0.25)       # idle window -> open -> started
                self.assertEqual(rec.done(), ["u1"])
            text = "\n".join(cm.output)
            self.assertIn(
                "background memory gate armed (reason=speech)", text)
            self.assertIn(
                "background memory deferred (conversation active; 1 queued)", text)
            self.assertIn(
                "background memory gate released (turn ended; grace=", text)
            self.assertIn(
                "background memory gate open (idle window held; "
                "background may start)", text)
            self.assertIn(
                "background memory ingest started (turn_no=1, 1 queued, "
                "start_wait=", text)
        _run(scene())

    @unittest.skipUnless(HAS_VENDOR_GATE, "vendored llm_bg_gate required")
    def test_cancel_lines_on_arm_while_running(self):
        ready = threading.Event()

        def ingest(user_text: str, reply: str) -> None:
            ready.wait(timeout=5.0)
            llm_bg_gate.check_cancel("test-leg")

        async def scene():
            gate = BackgroundMemoryGate(ingest, grace_s=0.05, idle_s=0.05)
            with self.assertLogs("app.background_memory", level="INFO") as cm:
                await gate.submit("u", "r")
                await asyncio.sleep(0.1)   # the ingest is blocked in its thread
                gate.arm("speech")         # user needs the slot -> CANCELLED
                ready.set()                # the leg now sees the cancel event
                await asyncio.sleep(0.15)  # re-queue logged
            text = "\n".join(cm.output)
            self.assertIn(
                "background memory CANCELLED (user needs the slot: speech)",
                text)
            self.assertIn(
                "background memory cancelled mid-chain, re-queued", text)
        _run(scene())


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
