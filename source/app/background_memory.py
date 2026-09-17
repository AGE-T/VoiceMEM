"""Conversation-first scheduling for background memory work.

v0.9.2 (PART 5): defer background STARTS behind the conversation.
v0.10.2 (operator PART 8/9): real USER PRIORITY — an in-flight background
chain is CANCELLED (cooperatively, at LLM leg boundaries / between stream
chunks) the moment the user needs the single llama-server slot, and the
turn pair is re-queued for the next idle window.

WHY THIS EXISTS (measured evidence, VoiceMEM v0.9.1 runtime audit + the
v0.10.0 field report):
  A voice turn's reply is followed by the VoiceMem pair-ingest extraction
  — a ~10.4K-token prompt on the SAME llama-server (parallel=1, single
  slot, FIFO queue). The measured contention: chat TTFT alone ~1.76 s; chat
  arriving while background work runs queues behind it (the v0.9.1 audit
  measured ~163 s for a chat landing ~2 s after an extraction launch; the
  v0.10.0 field report: 40-60 s chat TTFT whenever the user speaks during
  the post-reply background window). v0.9.2's gate only deferred STARTS —
  an in-flight chain kept the slot for up to ~111 s (v0.9.2 E4: one full
  warm chain on the production model).

POLICY (operator contract, v0.10.2 PART 8):
  * USER CONVERSATION HAS ABSOLUTE PRIORITY over the single llama-server
    slot. ``arm()`` — fired from the FIRST VAD speech frame (the earliest
    signal a chat request is coming) or a turn start — now ALSO requests
    cancellation of the in-flight background chain: the vendor legs abort
    before issuing their next request or between stream chunks (the HTTP
    stream is closed, which frees the server slot; see
    voicemem/utils/common/llm_bg_gate.py).
  * A cancelled ingest NEVER counts as done: the turn pair is re-queued
    (front of queue, exact duplicates still coalesced) and retried in a
    later idle window — memory updates are not lost, and a partially
    generated LLM answer is discarded by the cancellation itself (the
    vendor writes each leg's results only after the leg completes).
  * Background work may START only when the conversation is truly idle
    (PART 9): no active speech, no active user LLM work, no queued user
    turn, and a sufficient quiet interval (``VOICEMEM_BG_IDLE_S``) since
    the last speech/LLM/turn activity — on top of the v0.9.2 post-reply
    grace (``VOICEMEM_BG_GRACE_S``). This replaces the blind
    "2 s after reply" policy: the gate tracks real activity signals fed
    by the web session (``note_activity`` on every VAD frame and every
    user LLM delta).
  * Queued (user_text, reply) pairs are NEVER dropped (memory updates
    must not be lost) except exact duplicates, which are coalesced. A
    re-queued pair retries without limit (a very talkative user simply
    defers their memory consolidation — observable via
    ``stats.pending``/``requeued``).

DESIGN BOUND (documented, deliberate):
  One background chain at a time, run to completion. No worker framework,
  no thread pool — a single asyncio task with an idle-wait loop. The
  in-flight chain runs in ``asyncio.to_thread`` (the vendor chain is
  synchronous, async_facts=False), so task cancellation cannot stop the
  thread — that is exactly what the cooperative vendor cancel event is
  for.

Run-time knobs:
  VOICEMEM_BG_GRACE_S  (default 2.0) minimum idle seconds after a reply
                       before background work may start (v0.9.2 floor,
                       kept).
  VOICEMEM_BG_IDLE_S   (default 6.0) required quiet seconds — no speech
                       frame, no user LLM delta, no turn — before a
                       background chain may start (v0.10.2 idle policy).
                       A speech frame inside the window re-arms instantly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if v >= minimum else default


def _grace_s() -> float:
    return _env_float("VOICEMEM_BG_GRACE_S", 2.0)


def _idle_s() -> float:
    return _env_float("VOICEMEM_BG_IDLE_S", 6.0)


# -- vendor cooperative-cancel plumbing (guarded: degraded mode without it) -- #

try:  # pragma: no cover - import path depends on the vendored package
    from voicemem.utils.common import llm_bg_gate as _vendor_bg_gate
    from voicemem.utils.common.llm_bg_gate import BackgroundCancelledError
except ImportError:  # degraded mode: no vendor -> nothing to cancel
    _vendor_bg_gate = None  # type: ignore[assignment]

    class BackgroundCancelledError(BaseException):  # type: ignore[no-redef]
        """Degraded-mode stub — never raised (no vendor legs to cancel)."""


def _vendor_set_cancel(leg: str) -> None:
    if _vendor_bg_gate is not None:
        try:
            _vendor_bg_gate.set_cancel(leg)
        except Exception:  # noqa: BLE001 - never break arm() on vendor issues
            logger.exception("vendor bg cancel set failed")


def _vendor_clear_cancel(leg: str) -> None:
    if _vendor_bg_gate is not None:
        try:
            _vendor_bg_gate.clear_cancel(leg)
        except Exception:  # noqa: BLE001
            logger.exception("vendor bg cancel clear failed")


@dataclass
class _TurnRecord:
    """Pending background work: ONE conversational turn to consolidate."""

    user_text: str
    reply: str
    turn_no: int = 0
    coalesce_key: str = ""
    attempts: int = 0


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
    cancelled: int = 0
    requeued: int = 0
    current: Optional[str] = None
    # v0.10.2 PART 9: measure the real delay before a background chain
    # starts (release -> first item started), so the idle policy is
    # tunable with evidence instead of guesses.
    last_release_ts: float = 0.0
    last_start_wait_s: float = 0.0
    idle_waits: int = 0

    def as_dict(self) -> dict[str, Any]:
        out = {
            "submitted": self.submitted,
            "coalesced": self.coalesced,
            "started": self.started,
            "finished": self.finished,
            "failed": self.failed,
            "arm_count": self.arm_count,
            "waits_on_busy": self.waits,
            "cancelled": self.cancelled,
            "requeued": self.requeued,
            "idle_waits": self.idle_waits,
        }
        if self.last_start_wait_s:
            out["last_start_wait_s"] = round(self.last_start_wait_s, 2)
        out["pending"] = None if self.current is None else self.current[:60]
        return out


class BackgroundMemoryGate:
    """Gates background memory ingestion behind conversation activity.

    The gate lives on the web session. All public methods are coroutines or
    plain sync setters safe to call from the session's event loop; the ingest
    itself runs in a worker thread via ``asyncio.to_thread``.

    v0.10.2 surface additions over v0.9.2:
      * ``arm()`` cancels in-flight background LLM work (vendor event);
      * ``note_activity(kind)`` feeds the idle policy (speech frames,
        user LLM deltas, turn events — anything that means "the user is
        about to need / is using the slot");
      * a cancelled ingest is re-queued, never counted done;
      * ``release()`` opens only after the quiet window (grace floor +
        idle interval), not a blind timer.
    """

    def __init__(
        self,
        ingest: Callable[[str, str], Any],
        *,
        on_begin: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[], None]] = None,
        on_fail: Optional[Callable[[str], None]] = None,
        grace_s: Optional[float] = None,
        idle_s: Optional[float] = None,
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
        self._idle_s = idle_s if idle_s is not None else _idle_s()
        self._spawn = spawn  # None -> resolve lazily from the running loop

        self._queue: deque[_TurnRecord] = deque()
        # set = background work may start. The session starts IDLE (no turn
        # is active yet — nothing can even be queued before the first reply).
        self._gate_open = asyncio.Event()
        self._gate_open.set()
        # v0.10.2: distinguish ARMED (a turn needs the slot — never open)
        # from RELEASED-AWAITING-IDLE (the reply ended — open once the
        # quiet window holds). Both states have the event cleared; only
        # the latter may be flipped by _try_open.
        self._awaiting_idle = False
        self._grace_handle: Optional[asyncio.TimerHandle] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False               # an ingest is executing right now
        # v0.10.2 idle policy: loop-time of the last user-visible activity
        # (speech frame / LLM delta / turn event). 0.0 = "no activity yet"
        # (a fresh session — background starts are only possible after a
        # first turn anyway, which itself arms the gate).
        self._last_activity = 0.0
        self.stats = GateStats()
        self._turn_seq = 0

    # -- turn lifecycle (called by the web session) ------------------------- #

    def arm(self, reason: str = "speech") -> None:
        """A conversation turn needs (or will soon need) the LLM slot.

        v0.10.2: this now ALSO cancels the in-flight background chain —
        the vendor legs abort before their next request or between stream
        chunks (the HTTP stream closes, freeing the llama-server slot).
        Idempotent; always refreshes the activity clock.

        v0.10.3 (forensic validation prep): every transition of an OPEN
        gate to ARMED is logged — the ``[gate armed]`` timestamp is the
        "acquire" side of the gate timeline the llm_slot_forensic S8 phase
        reconstructs from logs/web-server.log. Logging only; no control-flow
        change.
        """
        if self._gate_open.is_set():
            self.stats.arm_count += 1
            logger.info("background memory gate armed (reason=%s)", reason)
        self._cancel_grace()
        self._gate_open.clear()
        self._awaiting_idle = False
        self._last_activity = time.monotonic()
        if self._running:
            logger.info(
                "background memory CANCELLED (user needs the slot: %s)",
                reason,
            )
            _vendor_set_cancel(reason)

    def release(self) -> None:
        """The streamed reply ended: open the gate once the conversation
        is truly idle (PART 9: quiet window, not a blind timer).

        Safe to call from any turn-exit path (success, error, interrupt);
        a speech frame arriving inside the window re-arms instantly.

        v0.10.3 (forensic validation prep): the release transition is
        logged — the ``[gate released]`` timestamp is the "release" side of
        the gate timeline the llm_slot_forensic S8 phase reconstructs.
        """
        logger.info(
            "background memory gate released (turn ended; grace=%.1fs idle=%.1fs)",
            self._grace_s, self._idle_s,
        )
        self._cancel_grace()
        self.stats.last_release_ts = time.monotonic()
        self._last_activity = max(self._last_activity, time.monotonic())
        self._awaiting_idle = True
        loop = asyncio.get_running_loop()
        # first attempt after the grace floor; _try_open re-schedules
        # itself until the idle interval is satisfied.
        self._grace_handle = loop.call_later(self._grace_s, self._try_open)

    def note_activity(self, kind: str = "activity") -> None:
        """Feed the idle policy: the user is speaking / the LLM is streaming.

        Called by the web session on every VAD frame with speech energy and
        on every user chat delta — anything that means the slot is (about
        to be) in conversational use. Does NOT cancel in-flight work by
        itself (that is arm()'s job); it only defers background STARTS.
        """
        self._last_activity = time.monotonic()

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

    def _try_open(self) -> None:
        """Timer callback: open the gate only when the quiet window holds."""
        self._grace_handle = None
        if not self._awaiting_idle:
            # release() was superseded by arm() (speech / a new turn):
            # the conversation wins, the gate stays closed.
            return
        elapsed_idle = time.monotonic() - self._last_activity
        if elapsed_idle >= self._idle_s:
            # Idle confirmed: background may use the slot again (clear the
            # vendor cancel FIRST so the legs do not abort on issue).
            self._awaiting_idle = False
            _vendor_clear_cancel("idle")
            self._gate_open.set()
            # v0.10.3 (forensic validation prep): the open transition is
            # logged — from here a background chain MAY start; the S8
            # product-trace phase uses this timestamp to bound "was
            # background work legitimately allowed to be on the slot?".
            logger.info(
                "background memory gate open (idle window held; "
                "background may start)"
            )
            return
        # still busy recently: re-check after the remaining interval
        self.stats.idle_waits += 1
        loop = asyncio.get_running_loop()
        self._grace_handle = loop.call_later(
            max(0.05, self._idle_s - elapsed_idle), self._try_open
        )

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

    def _requeue(self, item: _TurnRecord) -> None:
        """A cancelled ingest goes back to the FRONT (oldest pair first).

        Coalescing still applies: if an identical pair was submitted again
        while this one was running, the queue already holds it — then this
        requeue is a no-op (stats still count the requeue event).
        """
        if not any(rec.coalesce_key == item.coalesce_key for rec in self._queue):
            item.attempts += 1
            self._queue.appendleft(item)
            if item.attempts > 1 and item.attempts % 5 == 0:
                logger.warning(
                    "background memory pair re-queued %d times (user keeps "
                    "needing the slot; %d pairs pending)",
                    item.attempts, len(self._queue),
                )
        self.stats.requeued += 1

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
            # v0.10.2 PART 9: the measured release->start delay (evidence
            # for tuning VOICEMEM_BG_IDLE_S on the target machine).
            if self.stats.last_release_ts:
                self.stats.last_start_wait_s = (
                    time.monotonic() - self.stats.last_release_ts
                )
            self.stats.current = item.user_text
            # v0.10.3 (forensic validation prep): the chain START is logged —
            # this is the timestamp from which an in-flight chain can contend
            # with the next user turn (the S8 outlier attribution checks
            # exactly this window). Pure logging; no behaviour change.
            logger.info(
                "background memory ingest started (turn_no=%d, %d queued, "
                "start_wait=%.1fs)",
                item.turn_no, len(self._queue) + 1,
                self.stats.last_start_wait_s,
            )
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
                # v0.10.2 hygiene: ALSO request the vendor cancel so the
                # still-running ingest thread aborts at its next leg
                # boundary instead of holding the server slot.
                _vendor_set_cancel("session-teardown")
                raise
            except BackgroundCancelledError as exc:
                # v0.10.2 PART 8: the user needed the slot mid-chain —
                # the vendor legs aborted (partial text discarded inside
                # the legs; completed legs wrote only complete results).
                # This turn is NOT done: re-queue for the next idle window.
                self.stats.cancelled += 1
                logger.info(
                    "background memory cancelled mid-chain, re-queued "
                    "(%d queued): %s",
                    len(self._queue), exc,
                )
                self._requeue(item)
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
