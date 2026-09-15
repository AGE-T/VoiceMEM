"""Conversation-first scheduling for background memory work (v0.9.2, PART 5).

WHY THIS EXISTS (measured evidence, VoiceMEM v0.9.1 runtime audit):
  A voice turn's reply is followed ~50-100 ms later by the VoiceMem pair-ingest
  extraction — a ~10.4K-token prompt on the SAME llama-server (parallel=1,
  single slot). The measured contention: chat TTFT alone ~1.76 s; chat arriving
  ~2 s after an extraction launch ~163 s (FIFO-queued behind the prefill).
  The accepted 30-50 s tail latency matches this mechanism.

POLICY (operator contract, v0.9.2 PART 5):
  * USER CONVERSATION HAS ABSOLUTE PRIORITY over the single llama-server slot.
  * Background ingests wait while a turn is active — armed from the FIRST VAD
    speech frame (the earliest signal a chat request is coming) or turn start,
    released after the streamed reply ends (plus a small grace).
  * Queued (user_text, reply) pairs are NEVER dropped (memory updates must not
    be lost) except exact duplicates, which are coalesced.
  * The gate defers the START of background work. An in-flight extraction is
    NOT aborted (a mid-request abort would lose that turn's memory update);
    its duration is bounded instead by the extraction max_tokens limit
    (v0.9.2 PART 6) and the reduced prompt (PART 7).
  * Ingest runs with the vendor's async_facts=False inside the gate's worker
    so the whole background LLM chain (extraction -> conflict -> scoring) is
    serialized and VISIBLE — with async_facts=True the vendor hides it on a
    daemon thread no scheduler can see or bound.

DESIGN BOUND (documented, deliberate):
  One background chain at a time, run to completion. No worker framework, no
  thread pool — a single asyncio task with an idle-wait loop.

Run-time knobs:
  VOICEMEM_BG_GRACE_S  (default 2.0) idle seconds after a reply before
                       background work may start; a new speech frame re-arms
                       the gate instantly and cancels the grace timer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _grace_s() -> float:
    raw = os.environ.get("VOICEMEM_BG_GRACE_S", "").strip()
    if not raw:
        return 2.0
    try:
        v = float(raw)
    except ValueError:
        return 2.0
    return v if v >= 0.0 else 2.0


@dataclass
class _TurnRecord:
    """Pending background work: ONE conversational turn to consolidate."""

    user_text: str
    reply: str
    turn_no: int = 0
    coalesce_key: str = ""


@dataclass
class GateStats:
    """Diagnostics surface (never used for control flow)."""

    submitted: int = 0
    coalesced: int = 0
    started: int = 0
    finished: int = 0
    failed: int = 0
    arm_count: int = 0
    waits: int = 0
    current: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "coalesced": self.coalesced,
            "started": self.started,
            "finished": self.finished,
            "failed": self.failed,
            "arm_count": self.arm_count,
            "waits_on_busy": self.waits,
            "pending": None if self.current is None else self.current[:60],
        }


class BackgroundMemoryGate:
    """Gates background memory ingestion behind conversation activity.

    The gate lives on the web session. All public methods are coroutines or
    plain sync setters safe to call from the session's event loop; the ingest
    itself runs in a worker thread via ``asyncio.to_thread``.
    """

    def __init__(
        self,
        ingest: Callable[[str, str], Any],
        *,
        on_begin: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[], None]] = None,
        on_fail: Optional[Callable[[str], None]] = None,
        grace_s: Optional[float] = None,
        spawn: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """``ingest(user_text, reply)`` runs the FULL vendor chain
        synchronously (async_facts=False — see module docstring).
        ``on_begin/on_end/on_fail`` update the UI status chips;
        ``spawn`` creates a tracked task (the web session's ``_spawn``).
        """
        self._ingest = ingest
        self._on_begin = on_begin
        self._on_end = on_end
        self._on_fail = on_fail
        self._grace_s = grace_s if grace_s is not None else _grace_s()
        self._spawn = spawn  # None -> resolve lazily from the running loop

        self._queue: deque[_TurnRecord] = deque()
        # set = background work may start. The session starts IDLE (no turn
        # is active yet — nothing can even be queued before the first reply).
        self._gate_open = asyncio.Event()
        self._gate_open.set()
        self._grace_handle: Optional[asyncio.TimerHandle] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False               # an ingest is executing right now
        self.stats = GateStats()
        self._turn_seq = 0

    # -- turn lifecycle (called by the web session) ------------------------- #

    def arm(self, reason: str = "speech") -> None:
        """A conversation turn needs (or will soon need) the LLM slot."""
        if self._gate_open.is_set():
            self.stats.arm_count += 1
        self._cancel_grace()
        self._gate_open.clear()

    def release(self) -> None:
        """The streamed reply ended: open the gate after the grace period.

        Safe to call from any turn-exit path (success, error, interrupt);
        a speech frame arriving inside the grace window re-arms instantly.
        """
        self._cancel_grace()
        loop = asyncio.get_running_loop()
        self._grace_handle = loop.call_later(self._grace_s, self._open_after_grace)

    # -- submission ---------------------------------------------------------- #

    async def submit(self, user_text: str, reply: str) -> None:
        """Queue one (user_text, reply) pair for background consolidation."""
        if not user_text or not reply:
            return
        self._turn_seq += 1
        key = f"{user_text.strip()}\x00{reply.strip()}"
        if any(item.coalesce_key == key for item in self._queue):
            self.stats.coalesced += 1
            return
        self._queue.append(_TurnRecord(
            user_text=user_text, reply=reply,
            turn_no=self._turn_seq, coalesce_key=key,
        ))
        self.stats.submitted += 1
        self._ensure_worker()

    @property
    def pending(self) -> int:
        return len(self._queue)

    # -- internals ----------------------------------------------------------- #

    def _open_after_grace(self) -> None:
        self._grace_handle = None
        self._gate_open.set()

    def _cancel_grace(self) -> None:
        if self._grace_handle is not None:
            self._grace_handle.cancel()
            self._grace_handle = None

    def _ensure_worker(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        spawn = self._spawn or asyncio.get_running_loop().create_task
        try:
            self._worker_task = spawn(self._worker())
        except RuntimeError:
            # no running loop (e.g. tests constructing directly): the next
            # submit() from a live loop will spawn it.
            self._worker_task = None

    async def _worker(self) -> None:
        """One item at a time, only while the gate is open."""
        while self._queue:
            # wait for the conversation to go idle
            if not self._gate_open.is_set():
                self.stats.waits += 1
                logger.info(
                    "background memory deferred (conversation active; %d queued)",
                    len(self._queue),
                )
            await self._gate_open.wait()
            if not self._queue:  # released after a drain
                break
            item = self._queue.popleft()
            self._running = True
            self.stats.started += 1
            self.stats.current = item.user_text
            if self._on_begin is not None:
                try:
                    self._on_begin()
                except Exception:  # noqa: BLE001 - status is advisory
                    pass
            try:
                # to_thread: the ingest chain is synchronous (async_facts
                # =False on the vendor side) so this awaits REAL completion.
                await asyncio.to_thread(self._ingest, item.user_text, item.reply)
                self.stats.finished += 1
                if self._on_end is not None:
                    try:
                        self._on_end()
                    except Exception:  # noqa: BLE001
                        pass
            except asyncio.CancelledError:
                # session teardown: let the queue die with the task (same
                # contract as the old single _bg_store_task cancellation).
                raise
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                self.stats.failed += 1
                logger.warning("background memory ingest failed: %s", exc, exc_info=True)
                if self._on_fail is not None:
                    try:
                        self._on_fail(str(exc))
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                self._running = False
                self.stats.current = None

    # -- introspection (tests/diagnostics) ----------------------------------- #

    @property
    def is_processing(self) -> bool:
        return self._running

    @property
    def is_armed(self) -> bool:
        return not self._gate_open.is_set()
