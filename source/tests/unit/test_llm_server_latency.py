"""Unit tests for the pure helpers of tests/benchmark/llm_server_latency.py.

Machine: ANY (sandbox-safe). Only the PURE functions are covered
(percentile, find_first_sentence_end, summarize) plus the fixed probe
prompt constants. No network access, no llama-server, no file writes: the
data/benchmarks JSON output is never created by these tests.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests/<kind>/x.py -> 3 levels up)

try:
    from tests.benchmark.llm_server_latency import (  # package import (repo root on sys.path)
        JSON_MESSAGES,
        JSON_REQUIRED_KEYS,
        STREAM_MESSAGES,
        find_first_sentence_end,
        percentile,
        summarize,
    )
except ImportError:  # direct execution: script dir (tests/benchmark) on sys.path
    from llm_server_latency import (  # type: ignore[no-redef]
        JSON_MESSAGES,
        JSON_REQUIRED_KEYS,
        STREAM_MESSAGES,
        find_first_sentence_end,
        percentile,
        summarize,
    )


class PercentileTests(unittest.TestCase):
    """percentile(): linear interpolation between closest ranks; empty -> 0.0.

    Documented behaviour of the implementation (numpy-style "linear"):
    rank = (n - 1) * pct / 100, interpolated between floor and ceil ranks.
    """

    def test_empty_list_returns_zero(self):
        self.assertEqual(percentile([], 50), 0.0)
        self.assertEqual(percentile([], 95), 0.0)

    def test_single_value(self):
        self.assertEqual(percentile([7.5], 50), 7.5)
        self.assertEqual(percentile([7.5], 95), 7.5)
        self.assertEqual(percentile([7.5], 0), 7.5)

    def test_p50_of_one_to_five(self):
        # rank = 4 * 0.5 = 2.0 -> exact member
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3.0)

    def test_p95_of_one_to_five_interpolates(self):
        # rank = 4 * 0.95 = 3.8 -> 4 + 0.8 * (5 - 4)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 95), 4.8)

    def test_unsorted_input_is_sorted(self):
        self.assertEqual(percentile([5, 1, 4, 2, 3], 50), 3.0)

    def test_p0_and_p100_are_min_and_max(self):
        self.assertEqual(percentile([3, 1, 2], 0), 1.0)
        self.assertEqual(percentile([3, 1, 2], 100), 3.0)

    def test_interpolated_quartile(self):
        # rank = 3 * 0.25 = 0.75 -> 1 + 0.75 * (2 - 1)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 25), 1.75)

    def test_two_values_midpoint(self):
        self.assertEqual(percentile([10.0, 20.0], 50), 15.0)

    def test_result_is_float(self):
        self.assertIsInstance(percentile([1, 2, 3], 50), float)


class FindFirstSentenceEndTests(unittest.TestCase):
    """find_first_sentence_end(): boundary = . ! ? followed by whitespace/end."""

    def test_no_boundary(self):
        self.assertEqual(find_first_sentence_end("hello there"), -1)
        self.assertEqual(find_first_sentence_end(""), -1)
        self.assertEqual(find_first_sentence_end("just words, no end"), -1)

    def test_boundary_at_end_of_string(self):
        self.assertEqual(find_first_sentence_end("Hi."), 3)

    def test_period_followed_by_space(self):
        self.assertEqual(find_first_sentence_end("Hello there. More"), 12)

    def test_exclamation_followed_by_space(self):
        self.assertEqual(find_first_sentence_end("Hello there! More"), 12)

    def test_question_followed_by_space(self):
        # "Really?" has 7 characters; the "?" at index 6 ends the sentence.
        self.assertEqual(find_first_sentence_end("Really? Yes."), 7)

    def test_hungarian_exclamation(self):
        self.assertEqual(find_first_sentence_end("Szia! Hogy vagy?"), 5)

    def test_newline_counts_as_whitespace(self):
        self.assertEqual(find_first_sentence_end("Hi.\nMore text"), 3)

    def test_decimal_point_is_not_a_boundary(self):
        self.assertEqual(find_first_sentence_end("3.14 is pi"), -1)

    def test_trailing_ellipsis_counts_at_last_dot(self):
        self.assertEqual(find_first_sentence_end("Wait..."), 7)

    def test_non_string_input(self):
        self.assertEqual(find_first_sentence_end(None), -1)

    def test_first_boundary_wins(self):
        # "One." ends at index 4; the later "?" must not change the result.
        self.assertEqual(find_first_sentence_end("One. Two? Three!"), 4)

    def test_boundary_without_space_only_at_end(self):
        # "What?!": "?" is followed by "!" (not whitespace), so the trailing
        # "!" at the end of the string is the first boundary.
        self.assertEqual(find_first_sentence_end("What?!"), 6)


class SummarizeTests(unittest.TestCase):
    """summarize(): percentile summary over per-request metric dicts."""

    def test_empty_list_gives_zeroed_summary(self):
        self.assertEqual(
            summarize([]),
            {
                "ttft_p50_s": 0.0,
                "ttft_p95_s": 0.0,
                "first_sentence_p50_s": 0.0,
                "full_p50_s": 0.0,
                "tokens_per_s_p50": 0.0,
                "tokens_per_s_p95": 0.0,
                "n": 0,
            },
        )

    def test_single_sample(self):
        summary = summarize(
            [{"ttft_s": 0.1, "first_sentence_s": 0.5, "full_s": 2.0,
              "tokens": 100, "gen_s": 1.9}]
        )
        self.assertEqual(summary["n"], 1)
        self.assertAlmostEqual(summary["ttft_p50_s"], 0.1)
        self.assertAlmostEqual(summary["ttft_p95_s"], 0.1)
        self.assertAlmostEqual(summary["first_sentence_p50_s"], 0.5)
        self.assertAlmostEqual(summary["full_p50_s"], 2.0)
        self.assertAlmostEqual(summary["tokens_per_s_p50"], 100.0 / 1.9)
        self.assertAlmostEqual(summary["tokens_per_s_p95"], 100.0 / 1.9)

    def test_gen_s_zero_gives_zero_tokens_per_s(self):
        summary = summarize([{"ttft_s": 0.5, "first_sentence_s": 0.5,
                              "full_s": 0.5, "tokens": 10, "gen_s": 0.0}])
        self.assertEqual(summary["tokens_per_s_p50"], 0.0)

    def test_gen_s_negative_gives_zero_tokens_per_s(self):
        summary = summarize([{"ttft_s": 1.0, "full_s": 0.5, "tokens": 5,
                              "gen_s": -0.5}])
        self.assertEqual(summary["tokens_per_s_p50"], 0.0)

    def test_derived_gen_s_when_key_absent(self):
        # gen_s = full_s - ttft_s = 2.0 -> 200 tokens / 2.0 s = 100 tokens/s
        summary = summarize([{"ttft_s": 0.2, "first_sentence_s": 0.8,
                              "full_s": 2.2, "tokens": 200}])
        self.assertAlmostEqual(summary["tokens_per_s_p50"], 100.0)

    def test_derived_gen_s_zero_gives_zero_tokens_per_s(self):
        # full_s == ttft_s -> derived gen_s = 0 -> guard applies
        summary = summarize([{"ttft_s": 1.0, "full_s": 1.0, "tokens": 5}])
        self.assertEqual(summary["tokens_per_s_p50"], 0.0)

    def test_explicit_gen_s_wins_over_derived(self):
        summary = summarize([{"ttft_s": 0.2, "full_s": 2.2, "tokens": 200,
                              "gen_s": 1.0}])
        self.assertAlmostEqual(summary["tokens_per_s_p50"], 200.0)

    def test_percentiles_over_three_samples(self):
        samples = [
            {"ttft_s": 0.1, "first_sentence_s": 0.6, "full_s": 2.1,
             "tokens": 100, "gen_s": 2.0},
            {"ttft_s": 0.2, "first_sentence_s": 0.8, "full_s": 2.2,
             "tokens": 120, "gen_s": 2.0},
            {"ttft_s": 0.3, "first_sentence_s": 1.0, "full_s": 3.3,
             "tokens": 150, "gen_s": 3.0},
        ]
        summary = summarize(samples)
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["ttft_p50_s"], 0.2)
        # rank = 2 * 0.95 = 1.9 -> 0.2 + 0.9 * (0.3 - 0.2)
        self.assertAlmostEqual(summary["ttft_p95_s"], 0.29)
        self.assertAlmostEqual(summary["first_sentence_p50_s"], 0.8)
        self.assertAlmostEqual(summary["full_p50_s"], 2.2)
        # tokens/s per sample: 50.0, 60.0, 50.0 -> p50 50.0, p95 59.0
        self.assertAlmostEqual(summary["tokens_per_s_p50"], 50.0)
        self.assertAlmostEqual(summary["tokens_per_s_p95"], 59.0)

    def test_matches_percentile_directly(self):
        samples = [
            {"ttft_s": 0.4, "full_s": 1.4, "tokens": 10},
            {"ttft_s": 0.6, "full_s": 2.6, "tokens": 30},
        ]
        summary = summarize(samples)
        ttfts = [0.4, 0.6]
        self.assertEqual(summary["ttft_p50_s"], percentile(ttfts, 50))
        self.assertEqual(summary["ttft_p95_s"], percentile(ttfts, 95))

    def test_missing_keys_default_to_zero(self):
        summary = summarize([{"tokens": 5}])
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["ttft_p50_s"], 0.0)
        self.assertEqual(summary["first_sentence_p50_s"], 0.0)
        self.assertEqual(summary["full_p50_s"], 0.0)
        self.assertEqual(summary["tokens_per_s_p50"], 0.0)


class ProbePromptConstantsTests(unittest.TestCase):
    """The fixed prompts keep the probe reproducible; guard against drift."""

    def test_stream_prompt_is_fixed(self):
        self.assertEqual(
            STREAM_MESSAGES,
            [{"role": "user",
              "content": "Hello! Please tell me a few sentences about coffee."}],
        )

    def test_json_prompt_is_fixed(self):
        self.assertEqual(len(JSON_MESSAGES), 1)
        self.assertEqual(JSON_MESSAGES[0]["role"], "user")
        self.assertIn('"name"', JSON_MESSAGES[0]["content"])
        self.assertIn('"language"', JSON_MESSAGES[0]["content"])
        self.assertIn("Thomas", JSON_MESSAGES[0]["content"])
        self.assertIn("Hungarian", JSON_MESSAGES[0]["content"])

    def test_json_required_keys(self):
        self.assertEqual(JSON_REQUIRED_KEYS, ("name", "language"))


if __name__ == "__main__":
    unittest.main()
