"""E2E coverage for the app's streaming LLM leg (v0.5.1, TASK 1).

The TASK 1 forensic audit found the app's streaming reply path
(``LlmClient.chat_stream`` — prompt -> HTTP -> SSE -> first token) had NO
end-to-end coverage at all: the only LLM mock in the repo
(tests/e2e_mock_llm.py) is non-streaming, so the "first response token" leg
of the voice chain was literally unproven (and had failed in the field,
v0.4.4/v0.4.9: thinking-only / 0-char replies).

This test drives the REAL LlmClient against the in-process forensic mock
(tests/forensic_llm_mock.py) on an ephemeral port:

* normal mode  -> first token + full streamed reply
* thinking mode -> LlmUnavailableError with the exact root-cause message
  (the field failure mode reproduced and DETECTED, not silent)
* non-streaming (vendor-style) requests -> the routed JSON shape

No GPU, no real llama-server: the wire protocol is the contract under test.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from tests.forensic_llm_mock import make_handler

from app.config import AgentConfig
from app.llm import LlmClient, LlmUnavailableError


class _MockServer:
    """In-process forensic mock on an ephemeral port (no port conflicts)."""

    def __init__(self, mode: str, log: str) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(mode, log))
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_MockServer":
        import time

        self.thread.start()
        # readiness: the accept loop must be live before the first request
        # (starting the thread alone is a race — connection refused)
        import httpx

        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=0.5).status_code == 200:
                    return self
            except Exception:  # noqa: BLE001 - retry until ready
                time.sleep(0.05)
        raise RuntimeError("forensic mock did not become ready")

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


def _config_for(port: int) -> AgentConfig:
    import os

    os.environ["LLAMA_SERVER_HOST"] = "127.0.0.1"
    os.environ["LLAMA_SERVER_PORT"] = str(port)
    # Guard against cross-test env pollution: a previous test's
    # pin_vendor_llm_env(AgentConfig()) may have exported
    # OPENAI_BASE_URL=http://127.0.0.1:8080/v1 (the DEFAULT port), and
    # apply_env's later _apply_base_url(OPENAI_BASE_URL) branch would then
    # SILENTLY reset llama_server_port back to 8080 (config.py gives
    # OPENAI_BASE_URL priority over LLAMA_SERVER_HOST/PORT). Popping it here
    # keeps this test's port resolution deterministic.
    os.environ.pop("OPENAI_BASE_URL", None)
    cfg = AgentConfig()
    cfg.apply_env()
    return cfg


class LlmStreamingE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_stream_first_token_and_full_reply(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            log = fh.name
        with _MockServer("normal", log) as mock:
            client = LlmClient(_config_for(mock.port))
            try:
                deltas = []
                first = None
                import time

                t0 = time.perf_counter()
                async for delta in client.chat_stream(
                    [
                        {
                            "role": "system",
                            "content": "You are an English teacher.",
                        },
                        {"role": "user", "content": "Nehezen használom a present perfectet."},
                    ]
                ):
                    if first is None:
                        first = time.perf_counter() - t0
                    if delta:
                        deltas.append(delta)
            finally:
                await client.aclose()
        self.assertTrue(deltas, "the stream must yield content deltas")
        self.assertGreater(len(deltas), 1, "the reply must arrive in chunks")
        self.assertIsNotNone(first)
        reply = "".join(deltas)
        self.assertIn("present perfect", reply)
        # wiretap: exactly one streaming request left the app
        records = [json.loads(x) for x in open(log, encoding="utf-8") if x.strip()]
        stream_reqs = [r for r in records if r.get("stream")]
        self.assertEqual(len(stream_reqs), 1)
        self.assertEqual(stream_reqs[0]["model"], "qwen3.6-35b-a3b")

    async def test_thinking_only_stream_raises_with_root_cause(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            log = fh.name
        with _MockServer("thinking", log) as mock:
            client = LlmClient(_config_for(mock.port))
            with self.assertRaises(LlmUnavailableError) as raised:
                async for _delta in client.chat_stream(
                    [{"role": "user", "content": "teszt"}]
                ):
                    pass
            await client.aclose()
        self.assertIn("reasoning", str(raised.exception))
        self.assertIn("thinking", str(raised.exception))
        # the v0.4.4/v0.4.9 field failure must FAIL LOUDLY with the fix hint
        self.assertIn("LLM_DISABLE_THINKING", str(raised.exception))

    async def test_nonstream_vendor_style_request_served(self) -> None:
        """The vendor's OpenAI-SDK calls (non-streaming) get the routed JSON
        shape the extraction parser accepts (a "memory" array)."""
        import httpx
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            log = fh.name
        with _MockServer("normal", log) as mock:
            r = httpx.post(
                f"http://127.0.0.1:{mock.port}/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b",
                    "stream": False,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "You are a memory manager."},
                        {"role": "user", "content": "conflict resolution"},
                    ],
                },
                timeout=5.0,
            )
        self.assertEqual(r.status_code, 200)
        content = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        self.assertIn("memory", parsed, "the additive-extraction shape must parse")


if __name__ == "__main__":
    unittest.main()
