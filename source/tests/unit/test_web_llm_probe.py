"""v0.4.4 field-report regression tests: the LLM startup reply probe.

The v0.4.3 outage had llama-server perfectly healthy (/health 200, model
loaded, 60 tok/s generation in the server log) while EVERY chat reply
arrived as 0 characters — the health loop marked the LLM chip READY and
the UI went silent on every turn. ``_verify_llm_reply`` now sends ONE
small real chat turn on the first healthy probe and the chip only says
"reply verified" when non-empty content actually streams back; an empty
reply lands in the ERROR state with the diagnosis (never silently green).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from app.config import AgentConfig  # noqa: E402
from app.llm import LlmUnavailableError  # noqa: E402
from app.web_server import (  # noqa: E402
    WebComponents,
    _verify_llm_reply,
)


def _components(tmpdir: str) -> WebComponents:
    cfg = AgentConfig(root=Path(tmpdir))
    cfg.enable_speaker = False
    return WebComponents(cfg, mock=True).build()


class _ScriptedLlm:
    """Duck-typed LlmClient: scripted stream, records the payload."""

    def __init__(self, chunks: list[str] | None = None, error: Exception | None = None):
        self.chunks = chunks
        self.error = error
        self.calls: list[dict] = []

    async def health_check(self) -> bool:  # pragma: no cover - unused here
        return True

    async def chat_stream(self, messages, temperature=None, max_tokens=None, **_):
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        )
        if self.error is not None:
            raise self.error
        for chunk in self.chunks or []:
            yield chunk


class VerifyLlmReplyTests(unittest.TestCase):
    def _run(self, c: WebComponents) -> bool:
        return asyncio.run(_verify_llm_reply(c))

    def test_nonempty_reply_verifies_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            c = _components(td)
            c.llm = _ScriptedLlm(["rend", "y"])
            self.assertTrue(self._run(c))
            self.assertTrue(c.status.session.get("llm_verified"))
            self.assertNotEqual(c.status.components["llm"].state, "error")
            events = c.status.session.get("events", [])
            self.assertTrue(any("LLM reply verified" in e for e in events), events)

    def test_thinking_only_reply_reports_error_state(self):
        with tempfile.TemporaryDirectory() as td:
            c = _components(td)
            c.llm = _ScriptedLlm(
                error=LlmUnavailableError(
                    "LLM reply was empty: the model generated 500 chars of "
                    "reasoning (thinking) but no visible answer"
                )
            )
            self.assertFalse(self._run(c))
            comp = c.status.components["llm"]
            self.assertEqual(comp.state, "error")
            self.assertIn("reasoning", comp.error)

    def test_transport_error_reports_error_state(self):
        with tempfile.TemporaryDirectory() as td:
            c = _components(td)
            c.llm = _ScriptedLlm(error=LlmUnavailableError("connection refused"))
            self.assertFalse(self._run(c))
            self.assertEqual(c.status.components["llm"].state, "error")

    def test_timeout_reports_error_state(self):
        with tempfile.TemporaryDirectory() as td:
            c = _components(td)

            class _Hanging:
                async def chat_stream(self, messages, **_):
                    await asyncio.sleep(3600)
                    yield ""  # pragma: no cover

            c.llm = _Hanging()

            async def scenario():
                import app.web_server as ws

                real_wait_for = asyncio.wait_for

                async def fast_wait_for(coro, timeout):
                    return await real_wait_for(coro, timeout=0.05)

                saved = ws.asyncio.wait_for
                ws.asyncio.wait_for = fast_wait_for  # type: ignore[method-assign]
                try:
                    return await _verify_llm_reply(c)
                finally:
                    ws.asyncio.wait_for = saved  # type: ignore[method-assign]

            self.assertFalse(asyncio.run(scenario()))
            self.assertEqual(c.status.components["llm"].state, "error")
            self.assertIn("timed out", c.status.components["llm"].error)

    def test_probe_sends_a_small_max_tokens_request(self):
        with tempfile.TemporaryDirectory() as td:
            c = _components(td)
            fake = _ScriptedLlm(["ok"])
            c.llm = fake
            self.assertTrue(self._run(c))
            self.assertLessEqual(fake.calls[0]["max_tokens"], 64)
            contents = [m["content"] for m in fake.calls[0]["messages"]]
            self.assertTrue(any("ready" in c.lower() for c in contents), contents)

    def test_event_ring_records_the_probe(self):
        with tempfile.TemporaryDirectory() as td:
            c = _components(td)
            c.llm = _ScriptedLlm(["ok"])
            self.assertTrue(self._run(c))
            events = c.status.session.get("events", [])
            self.assertTrue(any("LLM reply verified" in e for e in events), events)


class RealMemoryLayerPinsE5Tests(unittest.TestCase):
    def test_layer_construction_pins_local_model(self):
        import os

        os.environ.pop("VOICEMEM_E5_MODEL", None)
        from app.web_server import RealMemoryLayer, PipelineStatus  # noqa: E402

        # The fixture model dir is a DECOY (empty config.json): the test's
        # assertion is about the ENV PIN, not model loading. In a
        # torch-equipped environment SentenceTransformer(decoy_dir) would
        # try to fetch the missing weights FROM THE NETWORK (v0.5.2 gate
        # runs the memory-safety battery, which requires torch) — offline
        # mode makes the load fail fast and deterministically instead, in
        # every environment profile (the layer degrades, the pin still
        # happens). Tests must never touch the network (offline posture).
        _offline = {}
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            _offline[k] = os.environ.get(k)
            os.environ[k] = "1"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                model_dir = root / "models" / "embedding" / "multilingual-e5-small"
                model_dir.mkdir(parents=True)
                (model_dir / "config.json").write_text("{}", encoding="utf-8")
                cfg = AgentConfig(root=root)
                cfg.embedding_model_path = str(model_dir)
                RealMemoryLayer(cfg, PipelineStatus())
                self.assertEqual(os.environ.get("VOICEMEM_E5_MODEL"), str(model_dir))
        finally:
            for k, v in _offline.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
