"""Unit tests for the v0.4.5 prompt-budget guard (field report #6).

A long-running web conversation (large memory block + 8 history entries)
once produced a 10459-token prompt that llama-server rejected outright
("request exceeds the available context size") -> the turn lost its reply.
``_cap_memory_context`` and ``_fit_history_budget`` keep every request
inside the fixed 8192-token context (chars-based, conservative estimate),
trimming OLDEST content first. Pure functions - no server, no LLM, no I/O.
"""

from __future__ import annotations

import unittest

from app.web_server import (
    _MEMORY_CONTEXT_MAX_CHARS,
    _PROMPT_CHAR_BUDGET,
    _cap_memory_context,
    _fit_history_budget,
)


def _history(entries: int, chars: int) -> list[dict]:
    return [{"role": "user", "content": "H" * chars} for _ in range(entries)]


class TestCapMemoryContext(unittest.TestCase):
    def test_short_context_untouched(self) -> None:
        self.assertEqual(_cap_memory_context("User likes tea."), "User likes tea.")

    def test_empty_and_blank_pass_through(self) -> None:
        self.assertEqual(_cap_memory_context(""), "")
        self.assertEqual(_cap_memory_context("   "), "   ")

    def test_long_context_capped_with_visible_note(self) -> None:
        big = "\n".join(f"memory line {i}" for i in range(1000))
        capped = _cap_memory_context(big)
        self.assertLessEqual(len(capped), _MEMORY_CONTEXT_MAX_CHARS + 200)
        self.assertTrue(
            capped.endswith("[... older memories truncated to fit the context]")
        )

    def test_cap_cuts_at_line_boundary_when_possible(self) -> None:
        lines = ["first line"] + [f"line {i}" for i in range(2000)]
        capped = _cap_memory_context("\n".join(lines))
        # the head of the block is well inside the cap and must survive
        self.assertIn("first line", capped)
        self.assertIn("line ", capped)
        self.assertNotIn("line 1999", capped)  # the tail is shed
        self.assertTrue(
            capped.endswith("[... older memories truncated to fit the context]")
        )

    def test_cap_strips_outer_whitespace_first(self) -> None:
        text = "  " + "x" * (_MEMORY_CONTEXT_MAX_CHARS - 10) + "  "
        self.assertEqual(_cap_memory_context(text), text.strip())

    def test_truncation_note_is_ascii_safe(self) -> None:
        capped = _cap_memory_context("x" * (_MEMORY_CONTEXT_MAX_CHARS * 2))
        capped.encode("ascii")  # must not raise


class TestFitHistoryBudget(unittest.TestCase):
    def test_small_prompt_keeps_everything(self) -> None:
        history = _history(8, 300)
        out = _fit_history_budget("system", history, "hello")
        self.assertEqual(out, history)

    def test_oversized_prompt_drops_oldest_first(self) -> None:
        system_prompt = "S" * 9000
        history = [
            {"role": "user", "content": f"message-{i}-" + "H" * 1900}
            for i in range(8)
        ]
        out = _fit_history_budget(system_prompt, history, "U" * 100)
        total = len(system_prompt) + 100 + sum(len(e["content"]) for e in out)
        self.assertLessEqual(total, _PROMPT_CHAR_BUDGET)
        self.assertLess(len(out), 8)
        # newest entries survive, oldest are shed
        self.assertEqual(out[-1]["content"], history[-1]["content"])
        self.assertEqual(out[0]["content"], history[8 - len(out)]["content"])

    def test_result_never_larger_than_input(self) -> None:
        history = _history(8, 5000)
        out = _fit_history_budget("S" * 9000, history, "U" * 500)
        self.assertLessEqual(len(out), len(history))

    def test_everything_dropped_when_system_alone_overflows(self) -> None:
        out = _fit_history_budget("S" * 13900, _history(8, 2000), "U" * 100)
        self.assertEqual(out, [])

    def test_malformed_entries_do_not_crash_the_estimator(self) -> None:
        history: list[dict] = [
            {"role": "user"},  # missing content
            {"content": "no role"},
            {"role": "assistant", "content": "fine"},
        ]
        out = _fit_history_budget("S" * 100, history, "U" * 100)
        self.assertIsInstance(out, list)

    def test_budget_keeps_room_for_the_answer(self) -> None:
        """~14k chars ≈ ≤7k estimated tokens + 512 answer < 8192 context."""
        self.assertLessEqual(_PROMPT_CHAR_BUDGET // 2 + 512, 8192)
        self.assertLessEqual(_MEMORY_CONTEXT_MAX_CHARS, _PROMPT_CHAR_BUDGET)


if __name__ == "__main__":
    unittest.main()
