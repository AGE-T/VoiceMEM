"""Unit tests for app.text_utils (pure stdlib, no heavy dependencies)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import unittest

from app.text_utils import (
    LANG_EN,
    LANG_HU,
    SentenceStream,
    detect_language,
    is_speakable,
    split_sentences,
)


def _strip_ws(text: str) -> str:
    """Remove all whitespace (chunk stripping loses inter-chunk spaces)."""
    return "".join(text.split())


class DetectLanguageTests(unittest.TestCase):
    def test_pure_hungarian(self):
        self.assertEqual(detect_language("Ez egy magyar mondat."), LANG_HU)

    def test_pure_english(self):
        self.assertEqual(detect_language("This is an English sentence."), LANG_EN)

    def test_english_without_diacritics_wins(self):
        self.assertEqual(detect_language("I really think this is a good idea"), LANG_EN)

    def test_english_with_hungarian_loanword(self):
        # The diacritic-bearing token weighs 3 and beats the EN stopwords.
        self.assertEqual(detect_language("The tejeskávé was excellent"), LANG_HU)

    def test_code_switched_hungarian_dominant(self):
        self.assertEqual(detect_language("Szeretnék egy flat white-ot kérni"), LANG_HU)

    def test_english_mixed_with_diacritic(self):
        # "is" counts as a HU stopword too, so HU wins even in an EN frame.
        self.assertEqual(detect_language("I think the kávé here is good"), LANG_HU)

    def test_empty_text_is_english(self):
        self.assertEqual(detect_language(""), LANG_EN)
        self.assertEqual(detect_language("   \n\t "), LANG_EN)


class SplitSentencesTests(unittest.TestCase):
    def test_abbreviation_not_split(self):
        text = "Dr. Kovács eljött. Stb."
        self.assertEqual(split_sentences(text), ["Dr. Kovács eljött.", "Stb."])

    def test_decimals_not_split(self):
        text = "A szám 3.14 volt. A másik 2.71."
        self.assertEqual(split_sentences(text), ["A szám 3.14 volt.", "A másik 2.71."])

    def test_multiple_terminators(self):
        text = "What?! Really... Yes."
        self.assertEqual(split_sentences(text), ["What?!", "Really...", "Yes."])

    def test_no_trailing_dot(self):
        text = "Ez egy mondat, amelynek a végén nincs pont\n"
        self.assertEqual(
            split_sentences(text),
            ["Ez egy mondat, amelynek a végén nincs pont"],
        )

    def test_newline_separates_sentences(self):
        text = "First line\nSecond line."
        self.assertEqual(split_sentences(text), ["First line", "Second line."])

    def test_whitespace_only_and_empty(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   \n  "), [])


class SentenceStreamTests(unittest.TestCase):
    def test_first_chunk_emitted_at_first_terminator(self):
        stream = SentenceStream(first_chunk_chars=24, chunk_chars=80)
        text = "Rendben. Ezután jön egy hosszabb mondat, ami már nem fér az első pufferbe."
        chunks: list[str] = []
        for ch in text:
            chunks.extend(stream.add_delta(ch))
        chunks.extend(stream.flush())
        self.assertEqual(chunks[0], "Rendben.")

    def test_long_text_without_terminators_splits_at_boundaries(self):
        stream = SentenceStream(first_chunk_chars=24, chunk_chars=80)
        words = [
            "ez", "egy", "nagyon", "hosszú", "szöveg", "amiben", "semmilyen",
            "mondatvég", "nem", "szerepel", "csak", "vessző,", "és", "szóközök",
            "határolják", "a", "biztonságos", "darabolási", "pontokat",
        ]
        text = " ".join(words)
        chunks: list[str] = []
        for ch in text:
            chunks.extend(stream.add_delta(ch))
        chunks.extend(stream.flush())

        self.assertGreaterEqual(len(chunks), 3)
        self.assertLessEqual(len(chunks[0]), 24)   # aggressive first window
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 80)   # never beyond chunk_chars
        self.assertEqual(_strip_ws("".join(chunks)), _strip_ws(text))

    def test_flush_returns_remainder(self):
        stream = SentenceStream(first_chunk_chars=24, chunk_chars=80)
        # 21 chars: under the first-chunk window, no terminator anywhere,
        # so nothing is emitted during streaming and flush gets it all.
        text = "Ez nem ér véget sehol"
        for ch in text:
            self.assertEqual(stream.add_delta(ch), [])
        remainder = stream.flush()
        self.assertEqual(len(remainder), 1)
        self.assertEqual(remainder[0], text)
        # buffer consumed: nothing left, and nothing re-emitted
        self.assertEqual(stream.flush(), [])

    def test_flush_drops_trivial_remainder(self):
        stream = SentenceStream()
        stream.add_delta("ab")
        self.assertEqual(stream.flush(), [])

    def test_no_duplicate_emissions(self):
        stream = SentenceStream()
        text = "Első. Második mondat. Harmadik!"
        chunks: list[str] = []
        for ch in text:
            chunks.extend(stream.add_delta(ch))
        chunks.extend(stream.flush())
        self.assertEqual(chunks, ["Első.", "Második mondat.", "Harmadik!"])
        self.assertEqual(stream.flush(), [])
        self.assertEqual(stream.flush(), [])
        self.assertEqual(len(set(chunks)), len(chunks))

    def test_streamed_reply_preserves_all_text_exactly_once(self):
        stream = SentenceStream(first_chunk_chars=24, chunk_chars=80)
        reply = (
            "Ez az első mondat. Ez pedig a második, kicsit hosszabb mondat. "
            "És itt van egy harmadik mondat is! Végül egy negyedik?"
        )
        chunks: list[str] = []
        for ch in reply:
            chunks.extend(stream.add_delta(ch))
        chunks.extend(stream.flush())

        self.assertEqual(_strip_ws("".join(chunks)), _strip_ws(reply))
        for chunk in chunks:
            self.assertTrue(chunk)             # stripped, non-empty
            self.assertTrue(is_speakable(chunk))
        # reply ended with a terminator: flush found nothing left
        self.assertEqual(stream.flush(), [])

    def test_decimal_number_not_split_across_deltas(self):
        stream = SentenceStream(first_chunk_chars=24, chunk_chars=80)
        chunks: list[str] = []
        for ch in "A szám 3.14 volt nagy.":
            chunks.extend(stream.add_delta(ch))
        chunks.extend(stream.flush())
        # the first full chunk must contain the intact decimal number
        self.assertIn("3.14", chunks[0])
        self.assertEqual(_strip_ws("".join(chunks)), _strip_ws("A szám 3.14 volt nagy."))

    def test_empty_deltas_are_noops(self):
        stream = SentenceStream()
        self.assertEqual(stream.add_delta(""), [])
        self.assertEqual(stream.add_delta("   "), [])


class IsSpeakableTests(unittest.TestCase):
    def test_true_for_real_text(self):
        self.assertTrue(is_speakable("Szia!"))
        self.assertTrue(is_speakable("  ok  "))
        self.assertTrue(is_speakable("a b"))

    def test_false_for_trivial_or_punctuation(self):
        self.assertFalse(is_speakable("..."))
        self.assertFalse(is_speakable("a"))
        self.assertFalse(is_speakable(""))
        self.assertFalse(is_speakable("   "))


if __name__ == "__main__":
    unittest.main()
