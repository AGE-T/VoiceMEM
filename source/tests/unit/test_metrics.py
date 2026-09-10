"""Unit tests for tests/benchmark/metrics.py (pure stdlib, sandbox-safe).

Machine: ANY. These tests run in the sandbox WITHOUT numpy/torch/etc. and on
the target machine alike.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests/<kind>/x.py -> 3 levels up)

try:
    from tests.benchmark.metrics import (  # package import (repo root on sys.path)
        accent_preservation,
        cer,
        levenshtein,
        token_accuracy,
        wer,
    )
except ImportError:  # direct execution: script dir (tests/benchmark) on sys.path
    from metrics import (  # type: ignore[no-redef]
        accent_preservation,
        cer,
        levenshtein,
        token_accuracy,
        wer,
    )

# Short Hungarian reference sample. The VERBATIM M1 ASR test sentence
# (spec Section 19.1 / milestones 7.6.1) lives in tests/benchmark/asr_benchmark_hu.py.
HU_SENTENCE = "A flat white és a tejeskávé egyaránt espresso-alapú italok tejjel."


class TestLevenshtein(unittest.TestCase):
    def test_classic_example(self) -> None:
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_empty_cases(self) -> None:
        self.assertEqual(levenshtein("", ""), 0)
        self.assertEqual(levenshtein("abc", ""), 3)
        self.assertEqual(levenshtein("", "abc"), 3)

    def test_identical(self) -> None:
        self.assertEqual(levenshtein("Tamás", "Tamás"), 0)

    def test_transposition_counts_as_two_edits(self) -> None:
        # Plain Levenshtein has no transposition operation (that would be Damerau).
        self.assertEqual(levenshtein("ab", "ba"), 2)


class TestWer(unittest.TestCase):
    def test_perfect_match_is_zero(self) -> None:
        self.assertEqual(wer("A flat white és a tejeskávé.", "a flat white és a tejeskávé"), 0.0)

    def test_case_and_punctuation_insensitive(self) -> None:
        # Same words, different case and punctuation -> 0 error.
        self.assertEqual(wer("Flat White, espresso!", "flat white espresso"), 0.0)

    def test_single_substitution(self) -> None:
        self.assertAlmostEqual(wer("a b c d", "a x c d"), 0.25)

    def test_single_deletion(self) -> None:
        self.assertAlmostEqual(wer("a b c d", "a b d"), 0.25)

    def test_single_insertion(self) -> None:
        self.assertAlmostEqual(wer("a b", "a b c"), 0.5)

    def test_empty_reference(self) -> None:
        self.assertEqual(wer("", ""), 0.0)
        self.assertEqual(wer("", "hello"), 1.0)

    def test_hungarian_accent_confusion_is_one_substitution(self) -> None:
        # "Tamás" vs "Tamas": one accented char replaced -> 1 of 2 words wrong.
        self.assertAlmostEqual(wer("Hojsz Tamás", "Hojsz Tamas"), 0.5)

    def test_hyphen_and_number_tokenization(self) -> None:
        # "1890-ben" tokenizes into 1890 + ben: a space-separated variant of the
        # same words scores a perfect zero (the hyphen acts as a separator).
        self.assertEqual(wer("1890-ben készítette", "1890 ben készítette"), 0.0)


class TestCer(unittest.TestCase):
    def test_perfect_match_is_zero(self) -> None:
        self.assertEqual(cer("Hojsz Tamás", "Hojsz Tamás"), 0.0)

    def test_case_sensitive(self) -> None:
        # Only the leading "T" differs by case; accented lowercase chars match.
        self.assertAlmostEqual(cer("Tamás", "tamás"), 1.0 / 5.0)
        # Every character differs by case (Á is U+00C1, á is U+00E1).
        self.assertAlmostEqual(cer("TAMÁS", "tamás"), 1.0)

    def test_single_accent_substitution(self) -> None:
        self.assertAlmostEqual(cer("Tamás", "Tamas"), 1.0 / 5.0)

    def test_insertion_of_space(self) -> None:
        self.assertAlmostEqual(cer("ab", "a b"), 1.0 / 2.0)

    def test_empty_reference(self) -> None:
        self.assertEqual(cer("", ""), 0.0)
        self.assertEqual(cer("", "abc"), 1.0)

    def test_longer_reference_missing_word(self) -> None:
        # "flat white" (10 chars) vs "flat" (4): 6 deletions out of 10.
        self.assertAlmostEqual(cer("flat white", "flat"), 0.6)


class TestTokenAccuracy(unittest.TestCase):
    def test_mixed_presence(self) -> None:
        hyp = "A flat white kávé finom volt."
        result = token_accuracy(HU_SENTENCE, hyp, ["flat white", "espresso", "kávé"])
        self.assertTrue(result["flat white"])
        self.assertFalse(result["espresso"])
        self.assertTrue(result["kávé"])

    def test_case_insensitive(self) -> None:
        result = token_accuracy("x", "FLAT WHITE", ["flat white"])
        self.assertTrue(result["flat white"])

    def test_multiword_requires_contiguous(self) -> None:
        # Words present but not adjacent -> miss.
        result = token_accuracy("x", "white kávé flat", ["flat white"])
        self.assertFalse(result["flat white"])

    def test_number_token(self) -> None:
        result = token_accuracy("x", "Első ízlet 1890-ben készítette.", ["1890"])
        self.assertTrue(result["1890"])
        result2 = token_accuracy("x", "ezerkilencszázkilencvenben", ["1890"])
        self.assertFalse(result2["1890"])

    def test_hungarian_name(self) -> None:
        result = token_accuracy("x", "készítette Hojsz Tamás nevű tudós", ["Hojsz Tamás"])
        self.assertTrue(result["Hojsz Tamás"])

    def test_empty_token_list(self) -> None:
        self.assertEqual(token_accuracy("a", "b", []), {})


class TestAccentPreservation(unittest.TestCase):
    def test_all_preserved(self) -> None:
        self.assertEqual(accent_preservation("Tamás készítette", "TAMÁS KÉSZÍTETTE"), 1.0)

    def test_all_lost(self) -> None:
        self.assertEqual(accent_preservation("Tamás", "Tamas"), 0.0)

    def test_partial_loss(self) -> None:
        # Reference accents: á, é (2); hypothesis keeps é only (in "kavé").
        self.assertAlmostEqual(accent_preservation("kávé", "kavé e"), 1.0 / 2.0)

    def test_no_accents_in_reference(self) -> None:
        self.assertEqual(accent_preservation("flat white", "flat white"), 1.0)

    def test_extra_accents_in_hypothesis_do_not_help(self) -> None:
        # Multiset semantics: extra ő in the hypothesis cannot rescue the lost á.
        value = accent_preservation("kávé", "kőőő")
        self.assertAlmostEqual(value, 0.0)

    def test_doubly_accented_letters(self) -> None:
        self.assertEqual(accent_preservation("ő ű", "ő ű"), 1.0)
        self.assertEqual(accent_preservation("ő ű", "o u"), 0.0)


if __name__ == "__main__":
    unittest.main()
