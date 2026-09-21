"""N1/N2 deterministic detector defects — focused regression matrix.

2026-09-20 consolidated-audit defect classes (Workstream 2):

N1 — an English-looking loan carrying orthographic EN evidence ('ee' in
     'meeting') incorrectly absorbs a FOLLOWING Hungarian word into the
     English span ("A 2026-os meeting fontos..." -> en 'meeting fontos').

N2 — an accent-free Hungarian host context incorrectly FLIPS the detected
     host language to English when an English loan is present
     ("Holnap lesz a meeting..." -> host=en, the whole HU sentence on the
     EN voice).

The fix under test (app/text_utils.py):
  * Hungarian vowel-harmony guard on budget absorption — a signal-free
    candidate with >=2 vowels all in one Hungarian harmony class (back
    a/o/u or front e/i/o-umlaut class) is a Hungarian word and is never
    absorbed into an English span (fontos/lesz-class content words);
  * 'holnap' added to HU_PLAIN_WORDS (the F-O accent-free function-word
    category — 'mikor'/'hol' were already there);
  * no-diacritic tie-break prefers HU when positive Hungarian
    function-word evidence exists (the article double-count flip).

Documented residuals (NOT fixed here, out of the two defect classes):
  * the bare Hungarian article 'a' riding into an EN span after a loan
    ("meeting a csapattal" -> en 'meeting a') — ambiguous with the
    English article 'a', a separate decision class;
  * accent-free DISHARMONIC Hungarian content words (fiú/kavics/
    napirend-class) — the harmony guard cannot see them;
  * sentences with NO Hungarian function words at all ("A meeting fontos
    volta meglepett.") still host-flip via article counting — content
    words are not table-able without a lexicon.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from app.text_utils import (  # noqa: E402
    LANG_EN,
    LANG_HU,
    detect_language,
    normalize_for_speech,
    segment_language_spans,
)


def _spans(text: str, host: str) -> list[tuple[str, str]]:
    return segment_language_spans(normalize_for_speech(text), host)


def _langs(spans: list[tuple[str, str]]) -> list[str]:
    return [lang for _text, lang in spans]


def _auto(text: str) -> list[tuple[str, str]]:
    """The production path: host detection then span segmentation."""
    norm = normalize_for_speech(text)
    host = detect_language(norm)
    return segment_language_spans(norm, host)


class N1TrailingHungarianWordTests(unittest.TestCase):
    """N1: a Hungarian word following an EN-evidence loan is never
    swept into the English span."""

    def test_n1_operator_case_meeting_fontos(self):
        # The operator's exact class: 'meeting' (ee) + 'fontos' (HU,
        # o/o harmonic) — 'fontos' must stay with the host.
        text = "A 2026-os meeting fontos lesz mindenkinnek."
        self.assertEqual(detect_language(normalize_for_speech(text)), LANG_HU)
        self.assertEqual(_auto(text), [(normalize_for_speech(text), LANG_HU)])

    def test_n1_loan_followed_by_hungarian_adjective(self):
        # 'fontos' directly after the loan, accented host words around.
        text = "Az új meeting fontos lett."
        self.assertEqual(_spans(text, LANG_HU),
                         [(normalize_for_speech(text), LANG_HU)])

    def test_n1_loan_followed_by_hungarian_verb(self):
        # 'marad' (a/a harmonic) — no diacritic, no stopword, no strong
        # digraph: only the harmony guard can keep it host.
        text = "A meeting marad, ahogy van."
        self.assertEqual(_spans(text, LANG_HU),
                         [(normalize_for_speech(text), LANG_HU)])

    def test_n1_loan_followed_by_hungarian_noun(self):
        # 'asztal'-class harmonic noun after the loan.
        text = "A meeting asztal mellől indult."
        self.assertEqual(_spans(text, LANG_HU),
                         [(normalize_for_speech(text), LANG_HU)])

    def test_n1_equivalent_surrounding_contexts(self):
        # Same loan in other Hungarian frames — never an EN span start.
        for text in (
            "A meeting fontos lesz holnap.",
            "A meeting fontos eleme volt a napnak.",
        ):
            with self.subTest(text=text):
                self.assertEqual(_auto(text),
                                 [(normalize_for_speech(text), LANG_HU)],
                                 text)

    def test_n1_multiple_loans_budget_accounting(self):
        # 'meeting afterparty' is a genuine two-word EN phrase (the
        # trailing budget consumed by the disharmonic 'afterparty');
        # the harmonic 'fontos' after it stays host — budget exhausted.
        text = "A meeting afterparty fontos volt."
        self.assertEqual(_spans(text, LANG_HU),
                         [("A ", LANG_HU),
                          ("meeting afterparty", LANG_EN),
                          (" fontos volt.", LANG_HU)])

    def test_n1_leading_hungarian_word_not_absorbed(self):
        # The leading mirror: a harmonic Hungarian content word before a
        # >=2-evidence run is not pulled into the EN span.
        text = "Fontos see you later, rendben?"
        self.assertEqual(
            _spans(text, LANG_HU),
            [("Fontos ", LANG_HU), ("see you later,", LANG_EN),
             (" rendben?", LANG_HU)],
        )

    def test_n1_differential_english_phrases_still_route(self):
        # The protected differential: the SAME positions, genuine
        # English phrase bodies — unchanged routing.
        self.assertEqual(
            _spans("Holnap touch base veled lesz.", LANG_HU),
            [("Holnap ", LANG_HU), ("touch base", LANG_EN),
             (" veled lesz.", LANG_HU)],
        )
        self.assertEqual(
            _spans("Rendben, catch up után beszélünk.", LANG_HU),
            [("Rendben, ", LANG_HU), ("catch up", LANG_EN),
             (" után beszélünk.", LANG_HU)],
        )


class N2HostLanguageFlipTests(unittest.TestCase):
    """N2: an accent-free Hungarian host never flips to EN on a loan."""

    def test_n2_operator_case(self):
        # The operator's exact sentence: Hungarian host, single loan.
        text = "Holnap lesz a meeting."
        norm = normalize_for_speech(text)
        self.assertEqual(detect_language(norm), LANG_HU)
        # single-word loans never re-route (documented loan policy):
        # the whole sentence stays on the Hungarian voice.
        self.assertEqual(segment_language_spans(norm, LANG_HU),
                         [(norm, LANG_HU)])

    def test_n2_audit_variant_with_trailing_context(self):
        # The consolidated-audit variant: the host must stay Hungarian;
        # the loan routes EN only as a phrase (here: single word -> no
        # re-route; the bare article riding after a loan is a
        # documented pre-existing residual, out of this defect class).
        text = "Holnap lesz a meeting a csapattal."
        norm = normalize_for_speech(text)
        self.assertEqual(detect_language(norm), LANG_HU)

    def test_n2_accent_free_host_with_function_words(self):
        # The general class: accent-free HU sentences whose only EN
        # signal is the article 'a' double-counting — with Hungarian
        # function words present the host stays HU.
        for text in (
            "Reggel lesz a meeting a csapattal.",
            "Holnap lesz a status report a főnökkel.",
            "Ma este lesz a rövid meeting a csapattal.",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    detect_language(normalize_for_speech(text)), LANG_HU,
                    text,
                )

    def test_n2_several_english_loans(self):
        # Several loans cannot outvote the Hungarian function words.
        text = "Holnap lesz a meeting, utána pedig a follow up, végül a catch up."
        self.assertEqual(detect_language(normalize_for_speech(text)), LANG_HU)

    def test_n2_accented_host_control(self):
        # Accented hosts were never broken — must stay HU.
        for text in (
            "Holnap lesz a meeting a csapatával.",
            "A meeting előtt még beszélünk.",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    detect_language(normalize_for_speech(text)), LANG_HU,
                    text,
                )

    def test_n2_touch_base_control(self):
        # The audit's second N2 case: 'Holnap touch base.' — Hungarian
        # host; the phrase routes EN via the span machinery (the
        # sentence-final period rides with the run — the established
        # convention, cf. the mandatory matrix's ("catch up.", EN)).
        text = "Holnap touch base."
        norm = normalize_for_speech(text)
        self.assertEqual(detect_language(norm), LANG_HU)
        self.assertEqual(segment_language_spans(norm, LANG_HU),
                         [("Holnap ", LANG_HU), ("touch base.", LANG_EN)])

    def test_n2_pure_english_stays_english(self):
        # Differential: genuinely English sentences must not flip HU.
        for text in (
            "This only really makes sense in the city.",
            "I will go to the store tomorrow",
            "The quick brown fox jumps over the lazy dog",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    detect_language(normalize_for_speech(text)), LANG_EN,
                    text,
                )

    def test_n2_pure_hungarian_stays_hungarian(self):
        for text in (
            "Szia, hol talalkozunk legkozelebb?",
            "A csapat gyorsan elindult szombaton reggel",
            "Nagyon sok a munka, de rendben van.",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    detect_language(normalize_for_speech(text)), LANG_HU,
                    text,
                )


if __name__ == "__main__":
    unittest.main()
