"""Unit tests for app.voicemem_bridge in degraded mode (voicemem not installed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import unittest

from app.config import AgentConfig
from app.voicemem_bridge import TurnContext, VoiceMemBridge


class _FakeLlm:
    """Duck-typed LlmClient for the reply-hook wiring test (no network)."""

    def __init__(self, scripted: list[str]) -> None:
        self.scripted = scripted
        self.messages: list[dict] = []

    async def chat_stream(self, messages, temperature=None, max_tokens=None):
        self.messages = messages
        for piece in self.scripted:
            yield piece


class VoiceMemBridgeDegradedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = AgentConfig()
        self.bridge = VoiceMemBridge(self.config)

    def test_is_available_false_without_package(self):
        self.assertFalse(self.bridge.is_available())

    async def test_process_turn_degraded_context(self):
        ctx = await self.bridge.process_turn("Szia, hogy vagy?")
        self.assertIsInstance(ctx, TurnContext)
        self.assertEqual(ctx.transcript, "Szia, hogy vagy?")
        self.assertEqual(ctx.memory_context, "")
        self.assertEqual(ctx.speaker_id, "voice_user")
        self.assertIsNone(ctx.turn)

    async def test_process_turn_keeps_speaker_id(self):
        ctx = await self.bridge.process_turn("Hello", speaker_id="voice_user")
        self.assertEqual(ctx.speaker_id, "voice_user")

    async def test_commit_reply_never_raises(self):
        await self.bridge.commit_reply("Rendben, viszlát!")
        await self.bridge.commit_reply("")
        await self.bridge.commit_reply("x", speaker_id="voice_user")

    def test_cancel_pending_never_raises(self):
        self.bridge.cancel_pending()

    async def test_degraded_warning_logged_exactly_once(self):
        with self.assertLogs("app.voicemem_bridge", level="WARNING") as captured:
            await self.bridge.process_turn("first turn")
            await self.bridge.process_turn("second turn")
            await self.bridge.commit_reply("a reply")
        degraded = [rec for rec in captured.records if "degraded" in rec.getMessage()]
        self.assertEqual(len(degraded), 1)

    def test_build_reply_fn_returns_callable(self):
        reply_fn = self.bridge.build_reply_fn()
        self.assertTrue(callable(reply_fn))

    async def test_reply_fn_without_llm_raises_runtime_error(self):
        reply_fn = self.bridge.build_reply_fn()
        with self.assertRaises(RuntimeError):
            async for _delta in reply_fn("hello"):
                pass

    async def test_reply_fn_streams_via_llm_with_teacher_prompt(self):
        fake_llm = _FakeLlm(["Hel", "lo!"])
        bridge = VoiceMemBridge(self.config, llm=fake_llm)
        reply_fn = bridge.build_reply_fn()

        deltas = [d async for d in reply_fn("Hi there", memory_context="likes tea")]

        self.assertEqual(deltas, ["Hel", "lo!"])
        self.assertEqual(len(fake_llm.messages), 2)
        system_message = fake_llm.messages[0]
        self.assertEqual(system_message["role"], "system")
        self.assertIn("Long-term memory about this user:", system_message["content"])
        self.assertIn("likes tea", system_message["content"])
        self.assertIn("voice assistant", system_message["content"].lower())
        self.assertEqual(
            fake_llm.messages[1],
            {"role": "user", "content": "Hi there"},
        )


if __name__ == "__main__":
    unittest.main()
