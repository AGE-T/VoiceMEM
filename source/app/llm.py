"""OpenAI-compatible async client for llama.cpp llama-server.

httpx is allowed at module level (present in both the sandbox and the target
environment). All network-facing methods are async; the ``httpx.AsyncClient``
is created lazily so constructing ``LlmClient`` never touches the network.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, Optional

import httpx

from .config import AgentConfig

logger = logging.getLogger(__name__)


class LlmUnavailableError(RuntimeError):
    """Raised when llama-server cannot be reached or fails during a request."""


def parse_sse_content_delta(line: str) -> str:
    """PURE: map one SSE line to its content delta ('' when not a content chunk).

    Handles: empty lines, ``:`` comments/keep-alives, ``data: [DONE]``
    markers, malformed JSON and non-``data:`` lines (all -> ``''``).
    Returns ``choices[0]["delta"]["content"]`` when present, otherwise
    ``choices[0]["message"]["content"]`` (non-streaming shape), otherwise
    ``''``. Only the first choice is considered. Never raises.

    v0.4.4: this deliberately does NOT read ``reasoning_content`` (the
    thinking channel) — use :func:`parse_sse_reasoning_delta` for that.
    """
    if not isinstance(line, str) or not line:
        return ""
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return ""
    if not stripped.startswith("data:"):
        return ""
    payload = stripped[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return ""
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return ""
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


def sse_lines_to_deltas(lines: Iterable[str]) -> list[str]:
    """PURE test helper: map SSE lines to the non-empty content deltas."""
    deltas: list[str] = []
    for line in lines:
        delta = parse_sse_content_delta(line)
        if delta:
            deltas.append(delta)
    return deltas


def _sse_payload(line: str) -> Optional[dict]:
    """PURE: parse one SSE line into its JSON object payload (None otherwise)."""
    if not isinstance(line, str) or not line:
        return None
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _first_choice(obj: dict) -> Optional[dict]:
    """PURE: ``choices[0]`` as a dict (None for any other shape)."""
    choices = obj.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return None
    first = choices[0]
    return first if isinstance(first, dict) else None


def parse_sse_reasoning_delta(line: str) -> str:
    """PURE: map one SSE line to its REASONING delta ('' when not reasoning).

    v0.4.4: llama.cpp llama-server (b10717+) with the default ``auto``
    reasoning format splits the model's thinking channel into
    ``choices[0]["delta"]["reasoning_content"]`` — the visible answer only
    starts in ``content`` AFTER the model closes its thought channel. A
    hybrid-reasoning model (Qwen3.6) with a small ``max_tokens`` can
    spend the whole budget thinking, leaving ``content`` empty: replies of
    0 characters. This accessor lets the client DETECT that state instead
    of silently yielding nothing.
    """
    obj = _sse_payload(line)
    if obj is None:
        return ""
    first = _first_choice(obj)
    if first is None:
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("reasoning_content"), str):
        return delta["reasoning_content"]
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("reasoning_content"), str):
        return message["reasoning_content"]
    return ""


def sse_lines_to_reasoning(lines: Iterable[str]) -> list[str]:
    """PURE test helper: map SSE lines to the non-empty reasoning deltas."""
    deltas: list[str] = []
    for line in lines:
        delta = parse_sse_reasoning_delta(line)
        if delta:
            deltas.append(delta)
    return deltas


def _extract_message_content(body: Any) -> Optional[str]:
    """Extract ``choices[0].message.content`` from a non-streaming response."""
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return None


def _extract_message_reasoning(body: Any) -> str:
    """Extract ``choices[0].message.reasoning_content`` (thinking channel)."""
    if not isinstance(body, dict):
        return ""
    first = _first_choice(body)
    if first is None:
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("reasoning_content"), str):
        return message["reasoning_content"]
    return ""


def _thinking_control_kwargs(config: "AgentConfig") -> dict[str, Any]:
    """Request-level thinking suppression for hybrid-reasoning models.

    v0.4.4 field report: llama-server's chat handler defaults
    ``enable_thinking`` to TRUE, and a hybrid-reasoning chat template then leaves
    the `` <|channel>thought`` channel OPEN at the generation prompt — the
    model answers INSIDE the thinking channel, the server routes it to
    ``reasoning_content`` and ``content`` stays empty (0-char replies, the
    text-mode outage of v0.4.3). Two independent, request-level switches
    both land on ``enable_thinking = false``:

    * ``chat_template_kwargs.enable_thinking`` — the llama.cpp native
      mechanism (the exact variable the template reads; with it
      false the template pre-closes the thought channel:
      `` <|channel>thought\n<channel|>``);
    * ``reasoning_effort: "none"`` — the OpenAI-standard field, parsed by
      llama-server since b10717 as "disable reasoning".

    Belt and braces: BOTH are sent (they cannot conflict — both just set
    ``enable_thinking = false`` server-side) so a future server that
    deprecates one keeps the other working. Controlled by
    ``config.llm_disable_thinking`` (default True — a local voice assistant
    wants sub-second first tokens, not a 10-60 s silent thought process).
    """
    if getattr(config, "llm_disable_thinking", True):
        return {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
    return {}


class LlmClient:
    """llama.cpp llama-server OpenAI-compatible client.

    Base URL comes from ``config.llama_server_url`` (``http://host:port/v1``).
    The underlying ``httpx.AsyncClient`` is created lazily on first use
    (connect timeout 3 s, read timeout 120 s) and closed by ``aclose()``.
    Assumes a single event loop (no cross-loop locking; M1 pipeline usage).
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    # -- client lifecycle ---------------------------------------------------- #

    def _get_client(self) -> httpx.AsyncClient:
        """Return the lazily created shared async HTTP client."""
        if self._client is None:
            timeout = httpx.Timeout(120.0, connect=3.0)
            self._client = httpx.AsyncClient(
                base_url=self._config.llama_server_url,
                timeout=timeout,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client (idempotent)."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> "LlmClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- API ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        """GET ``{base}/../health`` (i.e. ``http://host:port/health``) with a 2s
        timeout. True on HTTP 200, False on any transport/OS error."""
        try:
            client = self._get_client()
            response = await client.get(
                self._config.llama_server_health_url, timeout=2.0
            )
            return response.status_code == 200
        except (httpx.HTTPError, OSError) as exc:
            logger.debug("llama-server health check failed: %s", exc)
            return False

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """POST ``/v1/chat/completions`` with ``stream=True``; yields content deltas.

        SSE lines are parsed with :func:`parse_sse_content_delta`, tolerating
        comments and keep-alives. Raises :class:`LlmUnavailableError` on
        connection errors (including connection refused) and HTTP status
        errors. The streaming response is always released (async with), even
        when the consumer stops iterating early.
        """
        payload: dict[str, Any] = {
            "model": self._config.llm_model_name,
            "messages": messages,
            "stream": True,
            "temperature": (
                self._config.llm_temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self._config.llm_max_tokens if max_tokens is None else max_tokens
            ),
            **_thinking_control_kwargs(self._config),
        }
        client = self._get_client()
        yielded: list[str] = []
        reasoning_chars = 0
        # v0.4.9 (report issue #6): diagnostics for the empty-reply case. The
        # old code raised ONE generic message for "no tokens" — now the error
        # names WHICH failure mode happened (thinking-only / no data at all /
        # server stalled / keep-alive-only stream), so the operator can tell a
        # network drop from a VRAM-stalled server from a token-budget problem
        # without guessing.
        sse_lines = 0
        first_line_s: Optional[float] = None
        t_open = time.perf_counter()
        try:
            async with client.stream("POST", "chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if first_line_s is None:
                        first_line_s = time.perf_counter() - t_open
                    sse_lines += 1
                    delta = parse_sse_content_delta(line)
                    if delta:
                        yielded.append(delta)
                        yield delta
                        continue
                    reasoning = parse_sse_reasoning_delta(line)
                    if reasoning:
                        reasoning_chars += len(reasoning)
        except httpx.HTTPError as exc:
            raise LlmUnavailableError(f"LLM streaming request failed: {exc}") from exc
        # v0.4.4: a 200-OK stream that yields ZERO content is a hard, visible
        # failure — never a silent 0-char reply ("/health 200 but chat empty"
        # must FAIL, see the smoke-test contract). v0.4.9 (report issue #6):
        # the message now differentiates the four root causes.
        if not yielded:
            if reasoning_chars:
                raise LlmUnavailableError(
                    "LLM reply was empty: the model generated "
                    f"{reasoning_chars} chars of reasoning (thinking) but no visible "
                    "answer — the token budget was consumed by the thought channel "
                    "(llama-server enable_thinking default). Fix: keep "
                    "LLM_DISABLE_THINKING=1 or raise the max_tokens budget."
                )
            if sse_lines == 0:
                raise LlmUnavailableError(
                    "LLM reply was empty: the server returned 200 OK but NO data "
                    "arrived on the stream — network drop or llama-server died "
                    "mid-response. Check the llama-server console/logs "
                    "(start_llama_server writes logs/llama-server.err.log)."
                )
            if first_line_s is not None and first_line_s > 10.0:
                raise LlmUnavailableError(
                    f"LLM reply was empty: the server stalled {first_line_s:.1f} s "
                    "before the first SSE line and then closed without content — "
                    "VRAM full / model swapping? Check GPU status and the "
                    "llama-server console."
                )
            raise LlmUnavailableError(
                f"LLM reply was empty: the stream closed after {sse_lines} SSE "
                "lines without any content delta (keep-alives/comments only) — "
                "llama-server misbehaving; check its console/logs."
            )

    async def chat_json(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
    ) -> dict:
        """Non-streaming POST with ``response_format={"type": "json_object"}``.

        This is the VoiceMem ``_llm_json`` compatibility path: the JSON mode
        is REQUEST-LEVEL on the llama.cpp OpenAI-compatible endpoint (the
        ``response_format`` field of ``POST /v1/chat/completions`` triggers
        llama.cpp constrained generation via its built-in JSON grammar) -
        there is NO server-startup flag for JSON mode (the old
        ``--grammar-json`` CLI argument never existed in llama.cpp builds
        such as the pinned b10717 and made the server exit instantly with
        ``error: invalid argument: --grammar-json``; llama.cpp offers
        ``--json-schema``/``--grammar`` server flags, but those constrain
        EVERY response globally, which would break plain-text streaming).
        Parses ``choices[0]["message"]["content"]`` as JSON and returns the
        dict. Raises :class:`LlmUnavailableError` on transport errors and
        :class:`ValueError` on invalid JSON (or a non-object JSON payload).
        """
        payload: dict[str, Any] = {
            "model": self._config.llm_model_name,
            "messages": messages,
            "stream": False,
            "temperature": (
                self._config.llm_temperature if temperature is None else temperature
            ),
            "response_format": {"type": "json_object"},
            **_thinking_control_kwargs(self._config),
        }
        client = self._get_client()
        try:
            response = await client.post("chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise LlmUnavailableError(f"LLM JSON request failed: {exc}") from exc

        content = _extract_message_content(body)
        if content is None:
            raise ValueError("LLM JSON response has no message content")
        if not content.strip():
            reasoning = _extract_message_reasoning(body)
            if reasoning:
                raise ValueError(
                    "LLM JSON reply was empty: the model spent the whole budget "
                    f"thinking ({len(reasoning)} chars of reasoning_content) — "
                    "thinking is not disabled for this request"
                )
            raise ValueError("LLM JSON response content is empty")
        try:
            parsed = json.loads(content)
        except ValueError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response is not a JSON object")
        return parsed
