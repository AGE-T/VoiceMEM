"""Background-LLM cooperative cancellation gate (v0.10.2, operator PART 8).

WHY THIS EXISTS (measured evidence, v0.9.1 runtime audit + v0.10.0 field
report):
  The voice runtime has ONE llama-server (parallel=1, single slot) and a
  background memory chain (extraction -> conflict resolution -> scoring)
  whose prompts reach ~10.4K tokens. v0.9.2's BackgroundMemoryGate defers
  the START of background work while a turn is active, but an in-flight
  chain is deliberately NOT interrupted — and the v0.9.2 E4 measurement
  puts one full warm chain at ~111 s on the production model. A user who
  starts speaking during that window FIFO-queues behind the running legs:
  the observed 40-60 s chat TTFT regression is exactly this mechanism.

DESIGN (the operator contract, PART 8):
  * ``user speech detected -> background LLM work must yield or cancel
    where safely possible``. The gate (app/background_memory.py) sets
    :data:`BG_CANCEL` the moment a turn needs the LLM slot (VAD speech
    frame or turn start).
  * Every LLM leg of the ingest chain cooperates:
      - BEFORE issuing a request, ``check_cancel()`` aborts immediately
        (no request, no server slot time, nothing written);
      - DURING a request, the two big legs (extraction + conflict
        resolution) run through :func:`bg_chat_create` — a streaming
        request assembled chunk-by-chunk, polling the cancel event
        between chunks. On cancel the stream is CLOSED (the client
        disconnect frees the llama-server slot) and
        :class:`BackgroundCancelledError` propagates up to the gate,
        which re-queues the turn for a later idle window.
  * Memory consistency: cancellation can only strike BETWEEN legs or
    DURING generation — every leg writes its memory updates only after
    its full result is assembled, so a cancelled chain leaves the store
    either without this turn's facts (retried later) or with the fully
    completed legs' facts (complete, valid rows — never a partial text).
    The gate re-queues the turn pair, so the retry sees the same input.

THREAD MODEL:
  The event is a ``threading.Event`` because the vendor chain runs
  synchronously inside a worker THREAD (``asyncio.to_thread`` from the
  app's gate), while set/clear happen on the web session's event loop.
  A ``threading.Event`` is safe from both sides.

NOT a server feature: this module never talks to llama-server itself.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

#: Process-global cancel flag for background (non-user) LLM work. Set by the
#: app's BackgroundMemoryGate on ``arm()``; cleared when the conversation
#: goes idle again. The vendor legs poll it — see :func:`check_cancel`.
BG_CANCEL = threading.Event()

_LAST_LEG = "unset"          # diagnostics: which leg was cancelled last
_CANCEL_COUNT = 0
_CANCEL_LOCK = threading.Lock()


class BackgroundCancelledError(BaseException):
    """A background LLM leg was cancelled because the user needs the slot.

    Deliberately a ``BaseException`` (the ``asyncio.CancelledError``
    pattern): the ingest chain contains several ``except Exception``
    fallbacks that would otherwise SWALLOW the cancellation and keep
    running — e.g. voice_input.py's "ConflictResolver failed, falling back
    to ADD-only" would write every fact the very moment we tried to
    yield. As a BaseException it flies through every broad handler and
    lands only where it is explicitly awaited: the gate worker, which
    re-queues the turn. Nobody else may catch BaseException for real
    failures — genuine LLM/network errors remain ordinary ``Exception``s
    with the old fallback behaviour.

    Carries ``leg`` (which call site) in the message; raised either before
    issuing a request ('before') or mid-request after the stream was
    closed ('during').
    """


def set_cancel(leg: str = "arm") -> None:
    """Request cancellation of in-flight background LLM work (idempotent)."""
    global _LAST_LEG, _CANCEL_COUNT
    if not BG_CANCEL.is_set():
        with _CANCEL_LOCK:
            _CANCEL_COUNT += 1
            _LAST_LEG = leg
        BG_CANCEL.set()
        logger.info(
            "background LLM cancel requested (trigger=%s, #%d)",
            leg, _CANCEL_COUNT,
        )


def clear_cancel(leg: str = "idle") -> None:
    """Conversation is idle again — background LLM work may proceed."""
    if BG_CANCEL.is_set():
        BG_CANCEL.clear()
        logger.info("background LLM cancel cleared (trigger=%s)", leg)


def is_cancelled() -> bool:
    return BG_CANCEL.is_set()


def check_cancel(leg: str) -> None:
    """Raise :class:`BackgroundCancelledError` when cancellation is active.

    Called by every ingest-chain LLM leg BEFORE issuing its request: the
    cheapest possible yield (no request is created, the slot is never
    touched, nothing is written).
    """
    if BG_CANCEL.is_set():
        raise BackgroundCancelledError(
            f"background LLM leg '{leg}' aborted before issue "
            f"(user needs the slot; cancel set by '{_LAST_LEG}')",
        )


def stats() -> dict:
    """Diagnostics surface (never used for control flow)."""
    with _CANCEL_LOCK:
        return {
            "cancel_set": BG_CANCEL.is_set(),
            "cancel_count": _CANCEL_COUNT,
            "last_leg": _LAST_LEG,
        }


def _compat_response(text: str, finish_reason: str, usage) -> Any:
    """A non-streaming-shaped shim over the assembled streaming result.

    The ingest legs' post-processing reads ``resp.choices[0].message.content``
    (the text), ``resp.choices[0].finish_reason`` (truncation warning) and
    ``resp.usage`` (cost log) — this shim provides exactly that surface, so
    switching a leg to streaming cancellation changes only the call line.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def bg_chat_create(
    client,
    *,
    leg: str,
    messages: list,
    model: str,
    timeout: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
    extra_create_kwargs: dict | None = None,
):
    """Issue ONE background LLM request with cooperative cancellation.

    Streaming is used purely as the cancellation transport: llama-server
    frees the single slot when the client disconnects mid-stream, so
    closing the stream on cancel YIELDS the slot to the user request
    immediately (a non-streaming request cannot be interrupted client-side
    — the SDK ``create()`` blocks until the server finishes). The full
    text is assembled from the SSE deltas and wrapped in a
    non-streaming-shaped response (see :func:`_compat_response`), so
    callers behave exactly as with a plain ``create()``.

    Parameters mirror the two big ingest legs' needs:
      ``response_format={"type": "json_object"}`` is forwarded as-is —
      llama.cpp applies the JSON grammar per token, streaming changes
      only the delivery, not the constraint.

    Raises :class:`BackgroundCancelledError` when the cancel event fires
    before issue or between chunks (the stream is closed first, so the
    server slot is released). Any other exception propagates unchanged.
    """
    check_cancel(leg)  # cheapest path: never touch the server at all

    kwargs: dict = {"model": model, "messages": messages, "stream": True}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    if extra_create_kwargs:
        kwargs.update(extra_create_kwargs)

    parts: list[str] = []
    finish_reason = ""
    usage = None
    stream = client.chat.completions.create(**kwargs)
    # close() on the SDK Stream closes the underlying HTTP response — on
    # ANY exit path (normal completion, exception, or the explicit close
    # inside the loop when the user needs the slot).
    try:
        for chunk in stream:
            if BG_CANCEL.is_set():
                try:
                    stream.close()
                except Exception:  # noqa: BLE001 - close is best-effort
                    pass
                raise BackgroundCancelledError(
                    f"background LLM leg '{leg}' aborted mid-request "
                    f"(user needs the slot; {sum(len(p) for p in parts)} "
                    "chars of partial text discarded)",
                )
            # llama.cpp's final SSE chunk carries finish_reason + usage
            # (OpenAI-compatible shape) — keep the last seen of each.
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage = u
            if not getattr(chunk, "choices", None):
                continue
            first = chunk.choices[0]
            fr = getattr(first, "finish_reason", None)
            if fr:
                finish_reason = fr
            delta = getattr(first, "delta", None)
            piece = getattr(delta, "content", None)
            if piece:
                parts.append(piece)
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass
    return _compat_response("".join(parts), finish_reason, usage)
