"""v0.4.4 field-report regression tests: the Gemma 4 thinking-channel outage.

Root cause (proved live against llama.cpp b10717): the server's chat handler
defaults ``enable_thinking`` to TRUE, the Gemma 4 chat template then leaves
the `` <|channel>thought`` channel OPEN at the generation prompt, the model
answers INSIDE the thinking channel and llama-server routes those tokens to
``reasoning_content`` while ``content`` stays empty — with ``max_tokens``
consumed by the thought process every reply arrived as 0 characters (the
v0.4.3 text-mode outage).

These tests pin the three fix layers:

1. the SSE parser reads ONLY ``content`` deltas but DETECTS reasoning deltas
   (never silently discards the "thinking ate the budget" state);
2. every LlmClient request carries the thinking-suppression kwargs
   (``chat_template_kwargs`` + ``reasoning_effort``) unless disabled;
3. an empty streamed/JSON reply is a HARD ``LlmUnavailableError`` /
   ``ValueError`` — never a silent 0-char answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import asyncio
import json
import unittest

import httpx

from app.config import AgentConfig
from app.llm import (
    LlmClient,
    LlmUnavailableError,
    _thinking_control_kwargs,
    parse_sse_content_delta,
    parse_sse_reasoning_delta,
    sse_lines_to_deltas,
    sse_lines_to_reasoning,
)


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj)


REASONING_LINE = _sse({"choices": [{"delta": {"reasoning_content": "thinking..."}}]})
CONTENT_LINE = _sse({"choices": [{"delta": {"content": "Szia!"}}]})
BOTH_LINE = _sse({"choices": [{"delta": {"content": "answer", "reasoning_content": "th"}}]})
NONSTREAM_REASONING = _sse({"choices": [{"message": {"reasoning_content": "long thoughts", "content": ""}}]})


class ParseSseReasoningDeltaTests(unittest.TestCase):
    """Layer 1: reasoning deltas are extracted from BOTH response shapes."""

    def test_streaming_delta_reasoning(self):
        self.assertEqual(parse_sse_reasoning_delta(REASONING_LINE), "thinking...")

    def test_non_streaming_message_reasoning(self):
        self.assertEqual(parse_sse_reasoning_delta(NONSTREAM_REASONING), "long thoughts")

    def test_content_line_is_not_reasoning(self):
        self.assertEqual(parse_sse_reasoning_delta(CONTENT_LINE), "")

    def test_both_fields_line_yields_reasoning(self):
        self.assertEqual(parse_sse_reasoning_delta(BOTH_LINE), "th")

    def test_reasoning_never_leaks_into_content_parser(self):
        # THE v0.4.3 bug: the old parser saw the reasoning line and returned
        # "" — the reply silently became 0 chars.
        self.assertEqual(parse_sse_content_delta(REASONING_LINE), "")

    def test_content_never_leaks_into_reasoning_parser(self):
        self.assertEqual(parse_sse_reasoning_delta(CONTENT_LINE), "")

    def test_tolerates_garbage(self):
        for line in ("", "   ", ": keep-alive", "data: [DONE]", "data: {not json",
                     "event: message", "data: [1,2,3]", "data: null",
                     'data: {"choices":[]}', 'data: {"choices":[{"delta":{}}]}',
                     'data: {"choices":[{"delta":{"reasoning_content": 7}}]}'):
            self.assertEqual(parse_sse_reasoning_delta(line), "", repr(line))

    def test_sse_lines_to_reasoning_filters(self):
        got = sse_lines_to_reasoning([REASONING_LINE, CONTENT_LINE, REASONING_LINE, "junk"])
        self.assertEqual(got, ["thinking...", "thinking..."])

    def test_sse_lines_to_deltas_ignores_reasoning(self):
        got = sse_lines_to_deltas([REASONING_LINE, CONTENT_LINE])
        self.assertEqual(got, ["Szia!"])


class ThinkingControlKwargsTests(unittest.TestCase):
    """Layer 2: the request payload disables thinking by default."""

    def test_default_config_sends_both_switches(self):
        kwargs = _thinking_control_kwargs(AgentConfig())
        self.assertEqual(kwargs.get("reasoning_effort"), "none")
        self.assertEqual(kwargs.get("chat_template_kwargs"), {"enable_thinking": False})

    def test_disabled_flag_sends_nothing(self):
        cfg = AgentConfig()
        cfg.llm_disable_thinking = False
        self.assertEqual(_thinking_control_kwargs(cfg), {})

    def test_missing_attribute_defaults_to_on(self):
        class Bare:  # config-like object without the field
            pass

        self.assertEqual(_thinking_control_kwargs(Bare()).get("reasoning_effort"), "none")

    def test_config_default_true(self):
        self.assertIs(AgentConfig().llm_disable_thinking, True)

    def test_env_override_off(self):
        cfg = AgentConfig()
        cfg.llm_disable_thinking = True
        import os

        old = os.environ.get("LLM_DISABLE_THINKING")
        os.environ["LLM_DISABLE_THINKING"] = "0"
        try:
            cfg.apply_env()
            self.assertIs(cfg.llm_disable_thinking, False)
        finally:
            if old is None:
                os.environ.pop("LLM_DISABLE_THINKING", None)
            else:
                os.environ["LLM_DISABLE_THINKING"] = old


class _FakeLlm:
    """httpx.MockTransport-backed LlmClient harness (NO network)."""

    def __init__(self, lines: list[str], status: int = 200, config: AgentConfig | None = None):
        self.lines = lines
        self.status = status
        self.requests: list[dict] = []
        cfg = config or AgentConfig()
        self.client = LlmClient(cfg)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content.decode("utf-8")))
        if self.lines and isinstance(self.lines[0], str):
            body = "".join(line + "\n\n" for line in self.lines)
        else:
            body = json.dumps(self.lines[0])
        return httpx.Response(self.status, text=body, headers={"Content-Type": "text/event-stream"})

    async def _install(self) -> None:
        self.client._client = httpx.AsyncClient(
            base_url=self.client._config.llama_server_url,
            transport=httpx.MockTransport(self._handler),
        )

    def run_stream(self):
        async def scenario():
            await self._install()
            out = []
            try:
                async for delta in self.client.chat_stream(
                    [{"role": "user", "content": "Hello?"}]
                ):
                    out.append(delta)
            finally:
                await self.client.aclose()
            return out

        return asyncio.run(scenario())

    def run_stream_error(self):
        async def scenario():
            await self._install()
            try:
                async for _ in self.client.chat_stream(
                    [{"role": "user", "content": "Hello?"}]
                ):
                    pass  # pragma: no cover - expected to raise before/at end
            finally:
                await self.client.aclose()

        return asyncio.run(scenario())

    def run_json(self):
        async def scenario():
            await self._install()
            try:
                result = await self.client.chat_json(
                    [{"role": "user", "content": "json please"}]
                )
            finally:
                await self.client.aclose()
            return result

        return asyncio.run(scenario())


class ChatStreamPayloadTests(unittest.TestCase):
    """Layer 2 on the wire: the exact payload the app sends."""

    def test_stream_payload_carries_thinking_suppression(self):
        fake = _FakeLlm([CONTENT_LINE])
        fake.run_stream()
        payload = fake.requests[0]
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertTrue(payload["stream"])

    def test_stream_payload_respects_disabled_flag(self):
        cfg = AgentConfig()
        cfg.llm_disable_thinking = False
        fake = _FakeLlm([CONTENT_LINE], config=cfg)
        fake.run_stream()
        payload = fake.requests[0]
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("chat_template_kwargs", payload)

    def test_json_payload_carries_thinking_suppression(self):
        fake = _FakeLlm([{"choices": [{"message": {"content": "{\"a\": 1}"}}]}])
        fake.requests = fake.requests  # noqa: PLW0127
        async def scenario():
            await fake._install()
            try:
                result = await fake.client.chat_json([{"role": "user", "content": "x"}])
            finally:
                await fake.client.aclose()
            return result

        result = asyncio.run(scenario())
        payload = fake.requests[0]
        self.assertEqual(result, {"a": 1})
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})


class EmptyReplyIsHardErrorTests(unittest.TestCase):
    """Layer 3: the "/health 200 but chat empty" failure is loud."""

    def test_reasoning_only_stream_raises_with_diagnosis(self):
        fake = _FakeLlm([REASONING_LINE, REASONING_LINE, "data: [DONE]"])
        with self.assertRaises(LlmUnavailableError) as ctx:
            fake.run_stream_error()
        self.assertIn("reasoning", str(ctx.exception).lower())
        self.assertIn("thinking", str(ctx.exception).lower())

    def test_totally_empty_stream_raises(self):
        fake = _FakeLlm(["data: [DONE]"])
        with self.assertRaises(LlmUnavailableError):
            fake.run_stream_error()

    def test_reasoning_then_content_is_fine(self):
        # A model that thinks a little and then answers must NOT raise.
        fake = _FakeLlm([REASONING_LINE, CONTENT_LINE, "data: [DONE]"])
        out = fake.run_stream()
        self.assertEqual("".join(out), "Szia!")

    def test_json_empty_content_with_reasoning_raises(self):
        fake = _FakeLlm(
            [{"choices": [{"message": {"content": "", "reasoning_content": "all thinking"}}]}]
        )

        async def scenario():
            await fake._install()
            try:
                await fake.client.chat_json([{"role": "user", "content": "x"}])
            finally:
                await fake.client.aclose()

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(scenario())
        self.assertIn("thinking", str(ctx.exception).lower())

    def test_json_empty_content_without_reasoning_raises(self):
        fake = _FakeLlm([{"choices": [{"message": {"content": ""}}]}])

        async def scenario():
            await fake._install()
            try:
                await fake.client.chat_json([{"role": "user", "content": "x"}])
            finally:
                await fake.client.aclose()

        with self.assertRaises(ValueError):
            asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
