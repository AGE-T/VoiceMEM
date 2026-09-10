"""Regression tests for pin_vendor_llm_env (v0.5.1, TASK 1).

The forensic run proved the failure modes this pin removes:
  * orchestrator LLM calls: model falls back to gpt-4o-mini, api_key=None
    (OpenAI() constructor raises -> swallowed -> extraction silently "")
  * QuerySlotClassifier: base_url=None -> the OpenAI SDK defaults to
    https://api.openai.com (offline-policy violation)

Env-independent: only the pin function and AgentConfig are exercised; the
operator-override rule ("never clobber explicit values") is asserted too.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from app.config import AgentConfig
from app.voicemem_bridge import pin_vendor_llm_env

_PINNABLE = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_CHAT_MODEL")


class PinVendorLlmEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AgentConfig()

    def test_pins_all_when_unset(self) -> None:
        env = {k: "" for k in _PINNABLE}
        with patch.dict(os.environ, env, clear=False):
            for k in _PINNABLE:
                os.environ.pop(k, None)
            pinned = pin_vendor_llm_env(self.config)
            self.assertEqual(os.environ["OPENAI_BASE_URL"], self.config.llama_server_url)
            self.assertEqual(os.environ["OPENAI_MODEL"], self.config.llm_model_name)
            self.assertEqual(os.environ["OPENAI_CHAT_MODEL"], self.config.llm_model_name)
            self.assertTrue(os.environ["OPENAI_API_KEY"])  # dummy, non-empty
        self.assertEqual(
            sorted(pinned),
            sorted(_PINNABLE),
            "every unset variable must be reported as pinned",
        )

    def test_never_overrides_operator_values(self) -> None:
        operator = {
            "OPENAI_BASE_URL": "http://operator.example:9999/v1",
            "OPENAI_API_KEY": "operator-key",
            "OPENAI_MODEL": "operator-model",
            "OPENAI_CHAT_MODEL": "operator-chat-model",
        }
        with patch.dict(os.environ, operator, clear=False):
            pinned = pin_vendor_llm_env(self.config)
            self.assertEqual(pinned, {}, "an operator-set value must never be touched")
            self.assertEqual(os.environ["OPENAI_MODEL"], "operator-model")

    def test_pins_only_the_missing_subset(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_BASE_URL": "http://operator.example:9999/v1"},
            clear=False,
        ):
            for k in _PINNABLE:
                if k != "OPENAI_BASE_URL":
                    os.environ.pop(k, None)
            pinned = pin_vendor_llm_env(self.config)
            self.assertNotIn("OPENAI_BASE_URL", pinned)
            self.assertIn("OPENAI_MODEL", pinned)
            self.assertEqual(os.environ["OPENAI_MODEL"], self.config.llm_model_name)

    def test_pinned_values_target_the_local_llama_server(self) -> None:
        """The pin must aim every vendor leg at OUR llama-server, never at a
        cloud default, and must never emit the gpt-4o-mini fallback."""
        with patch.dict(os.environ, {}, clear=False):
            for k in _PINNABLE:
                os.environ.pop(k, None)
            pin_vendor_llm_env(self.config)
            for var in ("OPENAI_BASE_URL",):
                self.assertIn("127.0.0.1", os.environ[var])
                self.assertNotIn("api.openai.com", os.environ[var])
            for var in ("OPENAI_MODEL", "OPENAI_CHAT_MODEL"):
                self.assertNotEqual(os.environ[var], "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
