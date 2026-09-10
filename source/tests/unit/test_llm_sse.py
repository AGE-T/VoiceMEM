"""Unit tests for the pure SSE parser of app.llm (NO network access)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import unittest

from app.config import AgentConfig
from app.llm import (
    LlmClient,
    LlmUnavailableError,
    parse_sse_content_delta,
    sse_lines_to_deltas,
)


class ParseSseContentDeltaTests(unittest.TestCase):
    def test_streaming_delta_content(self):
        line = 'data: {"choices":[{"delta":{"content":"Hell"}}]}'
        self.assertEqual(parse_sse_content_delta(line), "Hell")

    def test_streaming_delta_without_space_after_data(self):
        line = 'data:{"choices":[{"delta":{"content":"o!"}}]}'
        self.assertEqual(parse_sse_content_delta(line), "o!")

    def test_non_streaming_message_content(self):
        line = 'data: {"choices":[{"message":{"role":"assistant","content":"Hello there"}}]}'
        self.assertEqual(parse_sse_content_delta(line), "Hello there")

    def test_done_marker(self):
        self.assertEqual(parse_sse_content_delta("data: [DONE]"), "")
        self.assertEqual(parse_sse_content_delta("data:[DONE]"), "")

    def test_comments_and_keep_alives(self):
        self.assertEqual(parse_sse_content_delta(""), "")
        self.assertEqual(parse_sse_content_delta("   "), "")
        self.assertEqual(parse_sse_content_delta(": keep-alive"), "")
        self.assertEqual(parse_sse_content_delta(":ping"), "")

    def test_non_data_lines(self):
        self.assertEqual(parse_sse_content_delta("event: message"), "")
        self.assertEqual(parse_sse_content_delta("id: 42"), "")
        self.assertEqual(parse_sse_content_delta("retry: 100"), "")
        self.assertEqual(parse_sse_content_delta("random garbage"), "")

    def test_malformed_json_returns_empty(self):
        self.assertEqual(parse_sse_content_delta("data: {not json"), "")
        self.assertEqual(parse_sse_content_delta("data: [1,2,3]"), "")
        self.assertEqual(parse_sse_content_delta("data: null"), "")
        self.assertEqual(parse_sse_content_delta("data: 42"), "")
        self.assertEqual(parse_sse_content_delta("data:"), "")

    def test_delta_without_content(self):
        line = 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
        self.assertEqual(parse_sse_content_delta(line), "")

    def test_finish_reason_chunk(self):
        line = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        self.assertEqual(parse_sse_content_delta(line), "")

    def test_first_choice_only(self):
        line = 'data: {"choices":[{"delta":{"content":"a"}},{"delta":{"content":"b"}}]}'
        self.assertEqual(parse_sse_content_delta(line), "a")

    def test_empty_content_is_not_returned(self):
        line = 'data: {"choices":[{"delta":{"content":""}}]}'
        self.assertEqual(parse_sse_content_delta(line), "")


class SseLinesToDeltasTests(unittest.TestCase):
    def test_filters_non_content_lines(self):
        lines = [
            ": keep-alive",
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
            "event: message",
        ]
        self.assertEqual(sse_lines_to_deltas(lines), ["Hel", "lo"])

    def test_lines_with_trailing_newlines(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"x"}}]}\n',
            'data: {"choices":[{"delta":{"content":"y"}}]}\r\n',
        ]
        self.assertEqual(sse_lines_to_deltas(lines), ["x", "y"])


class LlmUnavailableErrorTests(unittest.TestCase):
    def test_is_runtime_error_subclass(self):
        self.assertTrue(issubclass(LlmUnavailableError, RuntimeError))
        with self.assertRaises(RuntimeError):
            raise LlmUnavailableError("llama-server down")


class LlmClientConstructionTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_is_lazy_and_closes(self):
        config = AgentConfig()
        client = LlmClient(config)
        self.assertIsNone(client._client)  # nothing created at construction
        inner = client._get_client()
        self.assertTrue(str(inner.base_url).startswith(config.llama_server_url))
        await client.aclose()
        self.assertIsNone(client._client)
        # idempotent close
        await client.aclose()

    async def test_context_manager_closes(self):
        async with LlmClient(AgentConfig()) as client:
            self.assertIsNotNone(client)
        self.assertIsNone(client._client)


if __name__ == "__main__":
    unittest.main()
