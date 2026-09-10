"""Unit tests for app.teacher_persona (pure stdlib)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import unittest

from app.teacher_persona import build_messages, build_system_prompt


class BuildSystemPromptTests(unittest.TestCase):
    def test_contains_voice_companion_persona(self):
        """v0.4.14: the Ollama-experiment conversational persona (natural,
        short, spoken Hungarian) replaces the language-teacher scaffold."""
        prompt = build_system_prompt("")
        lowered = prompt.lower()
        self.assertIn("natural, friendly voice assistant", lowered)
        self.assertIn("one or two sentences", lowered)
        self.assertIn("do not over explain", lowered)
        self.assertIn("do not use lists, headings", lowered)
        self.assertIn("do not repeat the user's statement", lowered)
        self.assertIn("spoken hungarian", lowered)
        self.assertIn("conversation is more important than information density", lowered)

    def test_reply_language_rule_present(self):
        """Latest-message language matching (the Qwen HU<->EN switching rule)."""
        lowered = build_system_prompt("").lower()
        self.assertIn("same language as the user's latest message", lowered)
        self.assertIn("explicitly requests another language", lowered)

    def test_no_memory_block_when_context_empty(self):
        prompt = build_system_prompt("")
        self.assertNotIn("Long-term memory", prompt)

    def test_memory_block_present_when_context_given(self):
        prompt = build_system_prompt("The user's name is Anna; she likes tea.")
        self.assertIn("Long-term memory about this user:", prompt)
        self.assertIn("Anna", prompt)

    def test_no_emotion_block_by_default(self):
        prompt = build_system_prompt("")
        self.assertNotIn("valence", prompt)
        self.assertNotIn("arousal", prompt)
        self.assertNotIn("prosody", prompt)

    def test_emotion_block_when_label_given(self):
        prompt = build_system_prompt(
            "", emotion_label="frustrated", emotion_valence=-0.4, emotion_arousal=0.7
        )
        self.assertIn("frustrated", prompt)
        self.assertIn("valence", prompt)
        self.assertIn("arousal", prompt)
        self.assertIn("-0.4", prompt)
        self.assertIn("0.7", prompt)
        self.assertIn("slow down", prompt)
        self.assertIn("normal pace", prompt)

    def test_emotion_block_with_partial_values(self):
        prompt = build_system_prompt("", emotion_label="happy")
        self.assertIn("happy", prompt)
        self.assertIn("valence", prompt)
        self.assertIn("n/a", prompt)

    def test_prompt_stays_reasonably_short(self):
        prompt = build_system_prompt("")
        self.assertLess(len(prompt.split()), 250)

    def test_m2_params_accepted_and_default_equals_explicit_nones(self):
        # M1 promise: the M2 parameters exist and default to "excluded".
        self.assertEqual(build_system_prompt("ctx"), build_system_prompt("ctx", None, None, None))


class BuildMessagesTests(unittest.TestCase):
    def test_structure_system_history_user(self):
        history = [
            {"role": "user", "content": "Szia"},
            {"role": "assistant", "content": "Szia, hogy vagy?"},
        ]
        messages = build_messages("How are you?", "SYSTEM", history)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])
        self.assertEqual(messages[-1], {"role": "user", "content": "How are you?"})

    def test_no_history(self):
        messages = build_messages("hi", "SYS")
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_invalid_history_entries_skipped(self):
        history = [
            {"role": "user"},                       # missing content
            {"content": "no role"},                 # missing role
            "not-a-dict",
            None,
            {"role": 42, "content": "bad role type"},
            {"role": "assistant", "content": 123},  # non-string content
            {"role": "user", "content": "fine"},
        ]
        messages = build_messages("q", "SYS", history)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "fine"},
                {"role": "user", "content": "q"},
            ],
        )

    def test_none_history_treated_as_empty(self):
        self.assertEqual(len(build_messages("x", "SYS", None)), 2)


if __name__ == "__main__":
    unittest.main()
