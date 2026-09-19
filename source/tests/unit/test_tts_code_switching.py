"""TTS code-switching tests (field reports v0.10.4 + v0.10.5 phase 2 + forensic).

Production bug (v0.10.4): language selection happened ONCE PER TTS CHUNK,
so an English phrase quoted inside a Hungarian sentence ("A "touch base" egy
gyakori angol kifejezés.") was synthesized with the Hungarian Supertonic
model and became unintelligible.

Production gap (v0.10.5 phase 2): UNQUOTED embedded English phrases
("Rendben, see you later.", "Szeretnék egy gyors touch base-t veld.")
stayed with the Hungarian model — the LLM rarely sets phrases off with
quotes in conversational turns. Phrase RUNS now route them by evidence;
single words NEVER re-route ("projekt"/"meeting"/"hazai" stay host).

Production bug (v0.10.5 forensic fix, operator case "VOICE MEM — TTS
CODE-SWITCHING FORENSIC + FIX"): multi-word English idioms whose HEAD
word is orthographically signal-free ("**cut** to the chase", "**hit**
the ground running", "**burn** the midnight oil") were AMPUTATED — the
run started at the first evidence word, the head stayed in the host
span and the idiom was spoken half-Hungarian. Leading absorption now
merges ONE preceding signal-free word into a run that carries at least
PHRASE_RUN_LEADING_MIN_EVIDENCE foreign-evidence words. "break a leg"
(zero orthographic evidence) is a documented limitation: it routes by
host language only — fixing it needs a lexicon or loosened
ambiguous-word rules, both rejected by the operator order.

These tests pin the fix at all three levels:

1. ``segment_language_spans`` (pure) — span boundaries and languages for the
   field-reported sentences + the edge battery (contractions, quotes,
   parentheses, commas, hyphens, numbers, URLs, e-mails, unmatched quotes,
   phrase runs, budgets, caps, the run cap and suffix tails) + the
   operator's 10-sentence forensic matrix and the idiom regressions.
2. ``WebSession._synthesize_chunk`` (integration, mock components) — the
   per-span Supertonic language SEQUENCE actually requested by the web
   backend, the returned PCM (audio exists), the text-preservation invariant
   (chunk ordering preserved, nothing lost) and the no-exception contract
   (typographic characters normalized upstream, as in production).
3. ``VoicePipeline._speak_chunk`` (CLI path, mock components) — the same
   assertions for the pipeline caller, including forced voice modes.

Verifications per the field report, for every case:
(1) detected language / span boundaries, (2) selected TTS language,
(3) generated audio exists, (4) no unsupported-character exception,
(5) chunk ordering preserved (text concat invariant + call order).
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from app.config import AgentConfig  # noqa: E402
from app.mock_components import (  # noqa: E402
    MockAsrEngine,
    MockLlmClient,
    MockTtsEngine,
    MockVad,
    MockVoiceMemBridge,
)
from app.text_utils import (  # noqa: E402
    LANG_EN,
    LANG_HU,
    detect_language,
    normalize_for_speech,
    segment_language_spans,
)

#: Every field-reported sentence, in the audit's test matrix order.
HU_PURE = "Szeretnék röviden beszélni veled az új projektről."
HU_PURE_ALT = "Szeretnék röviden beszélni veled az új projekttel kapcsolatban."
EN_PURE = "I'd like to talk to you about the new project."
EN_PURE_TOUCH = "I'd like to touch base with you regarding the new project."
HU_QUOTED_EN = 'A "touch base" egy gyakori angol kifejezés.'
HU_QUOTED_EN_CATCH = 'A "catch up" hasonló jelentésű ebben a mondatban.'
EN_QUOTED_NEUTRAL = 'The Hungarian word "projekt" is used here.'
HU_UNQUOTED_MIX = "Szeretnék röviden touch base-elni veled a projekt miatt."
EN_QUOTED_HU = 'The word "szeretnék" means "I would like" here.'

#: v0.10.5 phase 2 (unquoted phrase runs) — the field-report matrix.
HU_UNQUOTED_TB = "Szeretnék egy gyors touch base-t veled."
HU_UNQUOTED_CU = "Szerintem később catch up-olhatunk."
HU_UNQUOTED_SYL = "Rendben, see you later."
HU_UNQUOTED_SYL_CAPS = "Rendben, SEE YOU LATER!"
HU_LOAN_MEETING = "A meeting után átbeszéljük a projektet."
EN_PURE_CATCH = "Let's catch up later."
EN_PURE_SYL = "See you later."

#: v0.10.5 forensic fix — the operator's four idiom cases (both hosts) and
#: the 10-sentence mandatory matrix, verbatim from the order.
IDIOM_CUT = "Let's cut to the chase."
IDIOM_HIT = "Sometimes you just have to hit the ground running."
IDIOM_BURN = "We had to burn the midnight oil."
IDIOM_BREAK = "Before the show, everyone said break a leg."
HU_IDIOM_CUT = "Most már tényleg cut to the chase, és menjünk tovább."
HU_IDIOM_HIT = "Holnap kell hit the ground running-gyel indulni."
HU_IDIOM_BURN = "Muszáj volt burn the midnight oil, de sikerült."
MANDATORY_MATRIX = [
    ("T1", IDIOM_CUT),
    ("T2", IDIOM_HIT),
    ("T3", IDIOM_BURN),
    ("T4", IDIOM_BREAK),
    ("T5", HU_IDIOM_CUT),
    ("T6", "Ma először checkeljük a resultot, aztán megnézzük a "
           "statuszt, végül pedig beszélünk a next step-ről."),
    ("T7", "A mai test során először checkeljük a resultot, aztán "
           "megnézzük a statuszt, végül pedig beszélünk a next "
           "step-ről."),
    ("T8", "Holnap lesz a meeting, utána pedig catch up."),
    ("T9", "Touch base után megbeszéljük a projektet."),
    ("T10", "Ez a projekt fontos, de most nem akarok meetinget."),
]


def _spans(text: str, host: str) -> list[tuple[str, str]]:
    """Segmentation exactly as the production callers do it (normalized)."""
    return segment_language_spans(normalize_for_speech(text), host)


def _langs(spans: list[tuple[str, str]]) -> list[str]:
    return [lang for _text, lang in spans]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pure segmentation
# ═══════════════════════════════════════════════════════════════════════════


class SegmentPureLanguageTests(unittest.TestCase):
    """Pure HU / pure EN: ONE span, host language — today's behaviour kept."""

    def test_pure_hungarian_single_span(self):
        for text in (HU_PURE, HU_PURE_ALT):
            self.assertEqual(detect_language(text), LANG_HU)
            self.assertEqual(_spans(text, LANG_HU), [(normalize_for_speech(text), LANG_HU)])

    def test_pure_english_single_span(self):
        for text in (EN_PURE, EN_PURE_TOUCH, "Don't worry, we're fine."):
            self.assertEqual(detect_language(text), LANG_EN)
            self.assertEqual(_spans(text, LANG_EN), [(normalize_for_speech(text), LANG_EN)])

    def test_empty_text_single_span(self):
        self.assertEqual(segment_language_spans("", LANG_HU), [("", LANG_HU)])
        self.assertEqual(segment_language_spans("   ", LANG_EN), [("   ", LANG_EN)])


class SegmentQuotedForeignTests(unittest.TestCase):
    """The reported bug: embedded foreign phrases inside a host sentence."""

    def test_quoted_english_inside_hungarian(self):
        spans = _spans(HU_QUOTED_EN, LANG_HU)
        self.assertEqual(
            spans,
            [
                ("A ", LANG_HU),
                ('"touch base"', LANG_EN),
                (" egy gyakori angol kifejezés.", LANG_HU),
            ],
        )

    def test_quoted_english_catch_up_inside_hungarian(self):
        spans = _spans(HU_QUOTED_EN_CATCH, LANG_HU)
        self.assertEqual(
            spans,
            [
                ("A ", LANG_HU),
                ('"catch up"', LANG_EN),
                (" hasonló jelentésű ebben a mondatban.", LANG_HU),
            ],
        )

    def test_quoted_hungarian_inside_english(self):
        spans = _spans(EN_QUOTED_HU, LANG_EN)
        self.assertEqual(
            spans,
            [
                ("The word ", LANG_EN),
                ('"szeretnék"', LANG_HU),
                (' means "I would like" here.', LANG_EN),
            ],
        )

    def test_quoted_neutral_loanword_inside_english_no_split(self):
        # "projekt" carries no Hungarian signal (no diacritics, digraphs or
        # stopwords) — the evidence gate keeps it with the EN host, where
        # the reading approximates the Hungarian pronunciation.
        self.assertEqual(_spans(EN_QUOTED_NEUTRAL, LANG_EN),
                         [(normalize_for_speech(EN_QUOTED_NEUTRAL), LANG_EN)])

    def test_quoted_native_signal_free_word_no_split(self):
        # "hazai" is everyday Hungarian with zero detectable signal — a
        # quoted native word must NOT flip to English (the evidence gate).
        text = 'Az úgynevezett "hazai" megoldás olcsóbb.'
        self.assertEqual(_spans(text, LANG_HU),
                         [(normalize_for_speech(text), LANG_HU)])

    def test_quoted_hungarian_word_with_diacritic(self):
        text = 'He said "oké" and left.'
        spans = _spans(text, LANG_EN)
        self.assertEqual(spans[1], ('"oké"', LANG_HU))

    def test_unquoted_code_switch_routes_by_evidence(self):
        # v0.10.5 phase 2 policy: an UNQUOTED multi-word English phrase with
        # positive evidence routes to EN; the hyphen-attached Hungarian verbal
        # suffix ("-elni") stays with the host model.
        self.assertEqual(
            _spans(HU_UNQUOTED_MIX, LANG_HU),
            [
                ("Szeretnék röviden ", LANG_HU),
                ("touch base", LANG_EN),
                ("-elni veled a projekt miatt.", LANG_HU),
            ],
        )

    def test_quoted_hungarian_loan_with_english_cluster_documented(self):
        # "technika" contains "ch" — the documented false-positive class:
        # quoted Hungarian loans are read with English phonetics. Asserting
        # it here so a future change is a conscious decision.
        text = 'A "technika" szó német eredetű.'
        spans = _spans(text, LANG_HU)
        self.assertEqual(spans[1], ('"technika"', LANG_EN))

    def test_single_quoted_phrase(self):
        text = "A 'touch base' egy gyakori angol kifejezés."
        spans = _spans(text, LANG_HU)
        self.assertEqual(
            spans,
            [
                ("A ", LANG_HU),
                ("'touch base'", LANG_EN),
                (" egy gyakori angol kifejezés.", LANG_HU),
            ],
        )

    def test_parenthesized_phrase(self):
        text = "A meeting (touch base later) volt ma."
        spans = _spans(text, LANG_HU)
        self.assertEqual(
            spans,
            [
                ("A meeting ", LANG_HU),
                ("(touch base later)", LANG_EN),
                (" volt ma.", LANG_HU),
            ],
        )

    def test_two_regions_with_host_between(self):
        text = 'A "touch" és "check" dolog.'
        self.assertEqual(
            _spans(text, LANG_HU),
            [
                ("A ", LANG_HU),
                ('"touch"', LANG_EN),
                (" és ", LANG_HU),
                ('"check"', LANG_EN),
                (" dolog.", LANG_HU),
            ],
        )

    def test_back_to_back_regions_merge(self):
        # No host text between the two regions -> the adjacent same-language
        # spans merge into ONE span.
        spans = _spans('"check""touch" egy dolog.', LANG_HU)
        self.assertEqual(_langs(spans), [LANG_EN, LANG_HU])
        self.assertEqual(spans[0][0], '"check""touch"')

    def test_quoted_native_hungarian_in_hungarian_no_split(self):
        text = 'A "kifejezés" arról szól, hogy mi a helyes.'
        self.assertEqual(_spans(text, LANG_HU),
                         [(normalize_for_speech(text), LANG_HU)])

    def test_quoted_english_in_english_no_split(self):
        text = 'The "catch up" phrase means the same here.'
        self.assertEqual(_spans(text, LANG_EN),
                         [(normalize_for_speech(text), LANG_EN)])


class SegmentPhraseRunTests(unittest.TestCase):
    """v0.10.5 phase 2: UNQUOTED foreign phrase runs (the production gap).

    Policy (audit VoiceMEM_tts_codeswitch_v0105_unquoted): a run needs
    >= 2 words, >= 1 strong foreign-evidence word, zero host evidence,
    starts at the evidence word, may sweep one trailing neutral word (EN;
    HU budget is 0), never across sentence-final punctuation, hyphen
    suffixes stay host, more than PHRASE_RUN_MAX_RUNS runs keep the whole
    chunk host. Single words NEVER re-route.
    """

    def test_required_matrix_unquoted_touch_base(self):
        self.assertEqual(
            _spans(HU_UNQUOTED_TB, LANG_HU),
            [
                ("Szeretnék egy gyors ", LANG_HU),
                ("touch base", LANG_EN),
                ("-t veled.", LANG_HU),
            ],
        )

    def test_required_matrix_unquoted_catch_up(self):
        self.assertEqual(
            _spans(HU_UNQUOTED_CU, LANG_HU),
            [
                ("Szerintem később ", LANG_HU),
                ("catch up", LANG_EN),
                ("-olhatunk.", LANG_HU),
            ],
        )

    def test_required_matrix_unquoted_see_you_later(self):
        self.assertEqual(
            _spans(HU_UNQUOTED_SYL, LANG_HU),
            [("Rendben, ", LANG_HU), ("see you later.", LANG_EN)],
        )

    def test_required_matrix_caps_see_you_later(self):
        # The field-reported rendering: "SEE YOU LATER" in caps.
        self.assertEqual(
            _spans(HU_UNQUOTED_SYL_CAPS, LANG_HU),
            [("Rendben, ", LANG_HU), ("SEE YOU LATER!", LANG_EN)],
        )

    def test_required_matrix_english_sentences_stay_single_span(self):
        # Host=EN, no HU evidence words -> one span (no fragmentation).
        for text in (EN_PURE_TOUCH, EN_PURE_CATCH, EN_PURE_SYL):
            self.assertEqual(detect_language(text), LANG_EN)
            self.assertEqual(_spans(text, LANG_EN),
                             [(normalize_for_speech(text), LANG_EN)], text)

    def test_single_word_loans_stay_host(self):
        # Operator-mandated: ambiguous loanwords keep the host language
        # (single words never re-route).
        for text in (
            "Szeretnék röviden beszélni veled az új projektről.",
            HU_LOAN_MEETING,
            "A hazai megoldás olcsóbb, mint a tea és az autó.",
            "A projekt lezárása holnapra vár.",
        ):
            self.assertEqual(_spans(text, LANG_HU),
                             [(normalize_for_speech(text), LANG_HU)], text)

    def test_hungarian_evidence_word_terminates_run(self):
        # "és" carries Hungarian evidence: the trailing budget never
        # sweeps a host word into the run.
        text = "Szeretnék touch base és utána menni."
        self.assertEqual(
            _spans(text, LANG_HU),
            [
                ("Szeretnék ", LANG_HU),
                ("touch base", LANG_EN),
                (" és utána menni.", LANG_HU),
            ],
        )

    def test_trailing_neutral_budget_is_one_word(self):
        # "veled" is the second signal-free word after "touch" — outside
        # the budget, it stays with the host.
        text = "Holnap touch base veled lesz."
        self.assertEqual(
            _spans(text, LANG_HU),
            [("Holnap ", LANG_HU), ("touch base", LANG_EN),
             (" veled lesz.", LANG_HU)],
        )

    def test_sentence_final_punctuation_blocks_budget(self):
        # "touch." closes a clause — "Base?" is a separate sentence, so
        # no multi-word run forms and the chunk stays host.
        text = "Ez nem touch. Base?"
        self.assertEqual(_spans(text, LANG_HU),
                         [(normalize_for_speech(text), LANG_HU)])

    def test_hu_budget_is_zero_lone_hu_word_in_english(self):
        # foreign=HU runs never sweep a trailing neutral word: the lone
        # Hungarian word "gyors" (digraph evidence, diacritic-free so the
        # chunk detector stays EN) does not drag "place" into a HU span.
        text = "We visited the gyors place today."
        self.assertEqual(detect_language(text), LANG_EN)
        self.assertEqual(_spans(text, LANG_EN),
                         [(normalize_for_speech(text), LANG_EN)])

    def test_consecutive_hu_evidence_words_route_in_english(self):
        # Two consecutive Hungarian-evidence words DO form a run in an
        # English sentence (both words carry their own evidence).
        text = "We had a gyors tempó there."
        spans = _spans(text, LANG_EN)
        self.assertIn(("gyors tempó", LANG_HU), spans)
        joined = "".join(s for s, _ in spans)
        self.assertEqual(joined, normalize_for_speech(text))

    def test_english_hyphen_compound_stays_whole(self):
        # Evidence-bearing heads never trim at the hyphen: "check-in" is
        # an English compound, not a suffixed Hungarian stem.
        text = "Menjünk, let's check-in ott."
        spans = _spans(text, LANG_HU)
        self.assertIn(("let's check-in", LANG_EN), spans)
        self.assertEqual("".join(s for s, _ in spans),
                         normalize_for_speech(text))

    def test_hungarian_case_suffix_variants_stay_host(self):
        for text in (
            "A holnapi touch base-től félek.",
            "A touch base-vel kezdeném.",
        ):
            spans = _spans(text, LANG_HU)
            self.assertEqual(spans[1], ("touch base", LANG_EN), text)
            self.assertTrue(spans[2][0].startswith("-"), text)
            self.assertEqual("".join(s for s, _ in spans),
                             normalize_for_speech(text))

    def test_run_cap_disables_switching_whole_chunk_stays_host(self):
        # 4 detected runs exceed PHRASE_RUN_MAX_RUNS (3): the pathological
        # chunk stays whole — no synthesis-call storm.
        text = ("touch base meg catch up meg see you later meg "
                "touch base meg catch up.")
        self.assertEqual(_spans(text, LANG_HU),
                         [(normalize_for_speech(text), LANG_HU)])

    def test_three_runs_still_switch(self):
        # At the cap exactly: switching still happens (3 runs).
        text = "touch base, aztán catch up, majd see you later."
        spans = _spans(text, LANG_HU)
        self.assertEqual(_langs(spans), [LANG_EN, LANG_HU, LANG_EN,
                                         LANG_HU, LANG_EN])
        self.assertEqual("".join(s for s, _ in spans),
                         normalize_for_speech(text))

    def test_whitespace_only_shards_never_become_spans(self):
        # Runs covering a whole host gap between regions must not leave
        # whitespace-only synthesis calls behind (boundary whitespace is
        # absorbed into the adjacent span — language-neutral material).
        text = 'A "x" touch base "y" és kész.'
        spans = _spans(text, LANG_HU)
        self.assertTrue(all(s.strip() for s, _ in spans), spans)
        self.assertIn(("touch base ", LANG_EN), spans)
        self.assertEqual("".join(s for s, _ in spans),
                         normalize_for_speech(text))

    def test_quoted_region_and_run_in_one_chunk(self):
        # Both levels in one chunk: the quoted region (level 1) and the
        # unquoted run (level 2) coexist without overlap.
        text = 'A "touch base" kifejezés, de ma catch up-olhatunk.'
        spans = _spans(text, LANG_HU)
        self.assertEqual(
            spans,
            [
                ("A ", LANG_HU),
                ('"touch base"', LANG_EN),
                (" kifejezés, de ma ", LANG_HU),
                ("catch up", LANG_EN),
                ("-olhatunk.", LANG_HU),
            ],
        )

    def test_documented_false_positive_neutral_sweep(self):
        # "chat" (ch) is evidence, "ablakot" is signal-free: the budget
        # sweeps it — the documented false-positive class, pinned here so
        # any change is a conscious decision (same class as the quoted
        # "technika" case from v0.10.4).
        text = "Nyisd ki a chat ablakot."
        spans = _spans(text, LANG_HU)
        self.assertIn(("chat ablakot.", LANG_EN), spans)
        self.assertEqual("".join(s for s, _ in spans),
                         normalize_for_speech(text))

    def test_numbers_urls_and_punctuation_only_tokens_are_never_runs(self):
        for text in (
            "A 2026-os team meeting next week lesz.",
            "Küldd el a report a info@voicemem.ai címre.",
        ):
            spans = _spans(text, LANG_HU)
            self.assertEqual("".join(s for s, _ in spans),
                             normalize_for_speech(text))
            self.assertTrue(all(s.strip() for s, _ in spans), text)


class IdiomLeadingAbsorptionTests(unittest.TestCase):
    """v0.10.5 forensic fix: idiom heads join their evidence runs.

    Root cause (audit/VoiceMEM_tts_codeswitch_v0105_forensic): the run
    detector started runs AT the first foreign-evidence word, so English
    idioms with signal-free heads ("cut to the chase", "hit the ground
    running", "burn the midnight oil") were amputated — the head stayed
    in the host span. Leading absorption (PHRASE_RUN_LEADING_BUDGET)
    merges ONE preceding signal-free word when the run carries at least
    PHRASE_RUN_LEADING_MIN_EVIDENCE evidence words. The ambiguous-loanword
    protections (single words, "Holnap"-class neutrals, digits, suffixes)
    are pinned here as guards of exactly that rule.
    """

    def test_idiom_cut_to_the_chase_in_hungarian(self):
        # Operator case T5 — the whole idiom becomes one EN span.
        self.assertEqual(
            _spans(HU_IDIOM_CUT, LANG_HU),
            [
                ("Most már tényleg ", LANG_HU),
                ("cut to the chase,", LANG_EN),
                (" és menjünk tovább.", LANG_HU),
            ],
        )

    def test_idiom_hit_the_ground_running_in_hungarian(self):
        # The head "hit" absorbs; the Hungarian derivational suffix
        # "-gyel" stays with the host span (trailing hyphen rule).
        self.assertEqual(
            _spans(HU_IDIOM_HIT, LANG_HU),
            [
                ("Holnap kell ", LANG_HU),
                ("hit the ground running", LANG_EN),
                ("-gyel indulni.", LANG_HU),
            ],
        )

    def test_idiom_burn_the_midnight_oil_in_hungarian(self):
        self.assertEqual(
            _spans(HU_IDIOM_BURN, LANG_HU),
            [
                ("Muszáj volt ", LANG_HU),
                ("burn the midnight oil,", LANG_EN),
                (" de sikerült.", LANG_HU),
            ],
        )

    def test_idiom_break_a_leg_english_host(self):
        # Operator case T4 — pure-English sentence: one EN span (the
        # chunk-level host detection routes the whole sentence).
        self.assertEqual(detect_language(IDIOM_BREAK), LANG_EN)
        self.assertEqual(_spans(IDIOM_BREAK, LANG_EN),
                         [(normalize_for_speech(IDIOM_BREAK), LANG_EN)])

    def test_idiom_break_a_leg_hungarian_host_is_documented_limitation(self):
        # "break a leg" carries ZERO orthographic English evidence
        # (break/a/leg are all signal-free; "a" is deliberately not an
        # EN word-stopword — it is the Hungarian definite article).
        # Detecting it needs a lexicon or loosened ambiguous-word rules,
        # both rejected by the operator order; the behaviour is pinned so
        # any future change is a conscious decision.
        for text in (
            "Most pedig break a leg, drágám.",
            'A "break a leg" kifejezés bátorságot jelent.',
        ):
            self.assertEqual(_spans(text, LANG_HU),
                             [(normalize_for_speech(text), LANG_HU)], text)

    def test_idiom_head_at_sentence_start_absorbs(self):
        # Sentence-initial idiom heads ("Rendben. Cut to the chase.")
        # join their run: the trailing-rule clause guard is mirrored on
        # the candidate's OWN last character, not the sentence boundary.
        text = "Kész. Cut to the chase, légyszives."
        self.assertEqual(
            _spans(text, LANG_HU),
            [
                ("Kész. ", LANG_HU),
                ("Cut to the chase,", LANG_EN),
                (" légyszives.", LANG_HU),
            ],
        )

    def test_single_evidence_run_never_absorbs_leading_neutral(self):
        # "touch base" carries ONE evidence word — the operator's
        # loanword protection: the signal-free Hungarian "Holnap" must
        # not be pulled into the EN span.
        text = "Holnap touch base veled lesz."
        self.assertEqual(
            _spans(text, LANG_HU),
            [("Holnap ", LANG_HU), ("touch base", LANG_EN),
             (" veled lesz.", LANG_HU)],
        )

    def test_clause_final_candidate_blocks_absorption(self):
        # The candidate itself ends a sentence ("cut.") — the mirror of
        # the trailing rule's guard: "cut. To the chase" are unrelated.
        text = "Láttam a cut. To the chase, légyszives."
        self.assertEqual(
            _spans(text, LANG_HU),
            [
                ("Láttam a cut. ", LANG_HU),
                ("To the chase,", LANG_EN),
                (" légyszives.", LANG_HU),
            ],
        )

    def test_digit_tokens_never_absorbed(self):
        # Numbers never carry language: "2026-os" stays with the host.
        text = "A 2026-os team meeting next week lesz."
        spans = _spans(text, LANG_HU)
        self.assertEqual(spans[0], ("A 2026-os ", LANG_HU))
        self.assertEqual("".join(s for s, _ in spans),
                         normalize_for_speech(text))

    def test_hyphen_trimmed_candidate_never_absorbed(self):
        # Hungarian-suffixed stems ("next step-ről": neutral head +
        # suffix) are never absorbed into a foreign run (white-box check
        # of the trimmed_at_hyphen guard).
        from app.text_utils import _phrase_runs

        runs = _phrase_runs("next step-ről cut to the chase", LANG_EN)
        self.assertEqual(len(runs), 1)
        start, end = runs[0]
        self.assertEqual("next step-ről cut to the chase"[start:end],
                         "cut to the chase")

    def test_documented_false_positive_leading_neutral_sweep(self):
        # The mirror of test_documented_false_positive_neutral_sweep: a
        # signal-free Hungarian word directly before a >=2-evidence run
        # is absorbed ("Holnap see you later." reads "Holnap" with EN
        # phonetics). Pinned as the documented false-positive class —
        # bounded by the budget (one word) and the evidence gate.
        text = "Holnap see you later."
        self.assertEqual(_spans(text, LANG_HU),
                         [(normalize_for_speech(text), LANG_EN)])

    def test_quoted_idiom_with_evidence_routes_english(self):
        # Whole idioms inside quotes (level-1 regions) route by region
        # evidence — already correct before the fix, pinned here.
        for idiom in ("cut to the chase", "hit the ground running",
                      "burn the midnight oil"):
            text = f'A "{idiom}" kifejezés angol idióma.'
            spans = _spans(text, LANG_HU)
            self.assertEqual(spans[1], (f'"{idiom}"', LANG_EN), idiom)

    def test_mandatory_matrix_language_sequences(self):
        # The operator's 10-sentence package, span-level, production host
        # detection per sentence. Expected (see the forensic report):
        #   T1-T4 pure English       -> one EN span (chunk host routing)
        #   T5    HU + clear idiom    -> HU, EN(idiom), HU
        #   T6/T7 ambiguous loanwords -> one HU span (no re-routing)
        #   T8    meeting HU, catch up EN
        #   T9    Touch base EN, rest HU
        #   T10   one HU span (meetinget stays host)
        expected = {
            "T1": [LANG_EN],
            "T2": [LANG_EN],
            "T3": [LANG_EN],
            "T4": [LANG_EN],
            "T5": [LANG_HU, LANG_EN, LANG_HU],
            "T6": [LANG_HU],
            "T7": [LANG_HU],
            "T8": [LANG_HU, LANG_EN],
            "T9": [LANG_EN, LANG_HU],
            "T10": [LANG_HU],
        }
        for case_id, sentence in MANDATORY_MATRIX:
            with self.subTest(case=case_id):
                norm = normalize_for_speech(sentence)
                host = detect_language(norm)
                spans = segment_language_spans(norm, host)
                self.assertEqual(_langs(spans), expected[case_id])
                self.assertEqual("".join(s for s, _ in spans), norm)

    def test_mandatory_matrix_ambiguous_loanwords_stay_host(self):
        # T6/T7/T10 loanword battery: checkeljük/resultot/statuszt/
        # test/next step-ről/meeting(et)/projekt(et) never re-route —
        # whichever span CONTAINS the word must be the host language.
        for case_id in ("T6", "T7", "T10"):
            sentence = dict(MANDATORY_MATRIX)[case_id]
            norm = normalize_for_speech(sentence)
            spans = segment_language_spans(norm, detect_language(norm))
            for word in ("checkeljük", "resultot", "statuszt", "test",
                         "step", "meetinget", "projektet"):
                containing = [(t, l) for t, l in spans if word in t]
                if not containing:
                    continue  # word not in this sentence
                for span_text, lang in containing:
                    self.assertNotEqual(
                        lang, LANG_EN,
                        f"{case_id}: {word!r} must stay host "
                        f"(found in EN span {span_text!r})",
                    )

    def test_mandatory_matrix_clear_phrases_route_english(self):
        # The differential the operator asked for: in the SAME sentence
        # (T8), the clear multi-word phrase routes EN while the ambiguous
        # single-word loanword stays HU.
        spans = _spans(dict(MANDATORY_MATRIX)["T8"], LANG_HU)
        self.assertIn(("catch up.", LANG_EN), spans)
        self.assertIn("meeting", spans[0][0])
        # T9: "Touch base" routes EN, "projektet" stays HU.
        spans = _spans(dict(MANDATORY_MATRIX)["T9"], LANG_HU)
        self.assertEqual(spans[0], ("Touch base", LANG_EN))
        self.assertIn("projektet", spans[1][0])


class SentenceStreamChunkBoundaryTests(unittest.TestCase):
    """The streaming chunker's first-window behaviour (documented).

    The SentenceStream first-chunk window (24 chars, config default)
    cuts at the last space before the limit — independent of language.
    When a foreign phrase STRADDLES that window, its head stays in the
    first (host) chunk: a pre-existing property of the low-TTFB chunker
    that affects every phrase class equally (e.g. "Szeretnék holnap
    touch b|ase-t ..." splits "touch base" too). Fixing it would need
    phrase-aware chunking — rejected by the operator order ("NE
    változtasd meg az egész chunking rendszert"). Pinned as-is so any
    future change is a conscious decision.
    """

    def _chunks(self, sentence: str, delta: int = 3) -> list[str]:
        from app.text_utils import SentenceStream

        stream = SentenceStream(first_chunk_chars=24, chunk_chars=80)
        out: list[str] = []
        for i in range(0, len(sentence), delta):
            out.extend(stream.add_delta(sentence[i : i + delta]))
        out.extend(stream.flush())
        return out

    def test_short_english_sentence_emits_whole(self):
        # 24 chars: never exceeds the first window -> one whole chunk.
        self.assertEqual(self._chunks(IDIOM_CUT), [IDIOM_CUT])

    def test_hungarian_idiom_sentence_chunk_boundaries(self):
        # The T5 sentence splits at the deterministic 24-char boundary:
        # chunk 1 ends inside the idiom ("cut to" stays with the host).
        self.assertEqual(
            self._chunks(HU_IDIOM_CUT),
            ["Most már tényleg cut to", "the chase, és menjünk tovább."],
        )
        # ...and each chunk routes independently (documented residual):
        host = detect_language("Most már tényleg cut to")
        self.assertEqual(
            _spans("Most már tényleg cut to", host),
            [("Most már tényleg cut to", LANG_HU)],
        )
        host = detect_language("the chase, és menjünk tovább.")
        self.assertEqual(host, LANG_HU)
        self.assertEqual(
            _spans("the chase, és menjünk tovább.", host),
            [("the chase,", LANG_EN), (" és menjünk tovább.", LANG_HU)],
        )


class SegmentEdgeCaseTests(unittest.TestCase):
    """Contractions, punctuation, numbers, URLs, e-mails, unmatched quotes."""

    def test_contractions_are_not_delimiters(self):
        # Apostrophes inside words never open/close regions, and the
        # contractions themselves carry English evidence.
        for word in ("I'd", "don't", "I'm", "we're"):
            text = f'Az "{word}" kifejezés angolul gyakori.'
            spans = _spans(text, LANG_HU)
            self.assertEqual(len(spans), 3, word)
            self.assertEqual(spans[1], (f'"{word}"', LANG_EN), word)

    def test_quoted_contraction_phrase_inside_hungarian(self):
        text = 'Az "I\'m ready" azt jelenti, hogy megyek.'
        spans = _spans(text, LANG_HU)
        self.assertEqual(spans[1], ('"I\'m ready"', LANG_EN))

    def test_possessive_after_single_quote_does_not_break_pairing(self):
        # 'touch base'-szerű: the closing quote is BEFORE the hyphen, the
        # hyphenated Hungarian suffix stays host.
        text = "Egy 'touch base'-szerű helyzet volt."
        spans = _spans(text, LANG_HU)
        self.assertEqual(
            spans,
            [
                ("Egy ", LANG_HU),
                ("'touch base'", LANG_EN),
                ("-szerű helyzet volt.", LANG_HU),
            ],
        )

    def test_numbers_and_urls_and_emails_are_neutral(self):
        for quoted in ("42", "3.14", "https://example.com", "info@voicemem.ai", "e-mail"):
            text = f'A "{quoted}" dolog működik.'
            self.assertEqual(
                _spans(text, LANG_HU),
                [(normalize_for_speech(text), LANG_HU)],
                quoted,
            )

    def test_unmatched_quote_degrades_to_host(self):
        # Unclosed quotes drop the REGION (no crash, no lost text) — but the
        # unquoted phrase-run detector still applies to the degraded text:
        # the embedded English phrase keeps its EN routing.
        text = 'A "touch base egy dolog.'
        self.assertEqual(
            _spans(text, LANG_HU),
            [
                ('A "', LANG_HU),
                ("touch base", LANG_EN),
                (" egy dolog.", LANG_HU),
            ],
        )
        text2 = 'Another "unbalanced one'
        self.assertEqual(_spans(text2, LANG_EN),
                         [(normalize_for_speech(text2), LANG_EN)])

    def test_empty_region_no_split(self):
        text = 'A "" dolog.'
        self.assertEqual(_spans(text, LANG_HU), [(normalize_for_speech(text), LANG_HU)])

    def test_typographic_quotes_and_apostrophes(self):
        # The production caller normalizes BEFORE segmentation (v0.10.3
        # choke point); U+201E/U+201D/U+2019 must all survive to spans.
        text = "A „touch base” egy gyakori angol kifejezés."
        spans = _spans(text, LANG_HU)
        self.assertEqual(spans[1], ('"touch base"', LANG_EN))

        text2 = "Az „I’d ready” azt jelenti."
        spans2 = _spans(text2, LANG_HU)
        self.assertEqual(spans2[1], ('"I’d ready"'.replace("’", "'"), LANG_EN))

    def test_text_preservation_invariant_battery(self):
        battery = [
            HU_PURE, EN_PURE, EN_PURE_TOUCH, HU_QUOTED_EN, HU_QUOTED_EN_CATCH,
            EN_QUOTED_NEUTRAL, HU_UNQUOTED_MIX, EN_QUOTED_HU,
            "Vesszővel, pontosvesszővel; és gondolatjellel - vagy így.",
            "Numbers 1, 2.5 and 100% plus 2026-09-18 stay whole.",
            "See https://example.com/a?b=1 or mail info@voicemem.ai now.",
            "A 'touch base'-szerű helyzet, (nem „catch up”), tényleg.",
            "Árvíztűrő tükörfúrógép - don't worry, we're OK.",
            '"Leading quote and trailing quote"',
            " rock 'n' roll music ",
        ]
        for text in battery:
            normalized = normalize_for_speech(text)
            for host in (LANG_HU, LANG_EN):
                spans = segment_language_spans(normalized, host)
                joined = "".join(s for s, _ in spans)
                self.assertEqual(joined, normalized, text)
                self.assertTrue(spans)
                self.assertTrue(all(s for s, _ in spans), text)

    def test_latency_of_segmentation_is_negligible(self):
        import time

        worst = (
            "A „touch base” és a „catch up” és egy (check this out) meg egy "
            "hosszabb magyar szövegrész, hogy legyen mit szegmentálni. "
        ) * 3
        normalized = normalize_for_speech(worst)
        t0 = time.perf_counter()
        for _ in range(100):
            segment_language_spans(normalized, LANG_HU)
        elapsed_ms = (time.perf_counter() - t0) * 10.0  # per call, ms
        self.assertLess(elapsed_ms, 5.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Web integration (WebSession._synthesize_chunk, mock components)
# ═══════════════════════════════════════════════════════════════════════════


class _RecordingSock:
    """Minimal WebSocket stand-in (the chunk synthesizer never sends)."""

    async def send_json(self, payload: dict) -> None:  # pragma: no cover
        pass

    async def send_bytes(self, raw: bytes) -> None:  # pragma: no cover
        pass


def _make_session() -> tuple[Any, Any]:
    """(session, components) with mock engines and auto voice mode.

    The built-in web demo TTS records TRUNCATED (text[:40], language)
    pairs — useless for span assertions. The standard MockTtsEngine records
    the FULL (text, language, length_scale, voice) of every call, which is
    exactly what these tests assert on; the chunk synthesizer only needs
    the engine interface, so it is swapped in after the build.
    """
    from app.web_server import WebComponents, WebSession

    tmp = Path(tempfile.mkdtemp(prefix="vm_tts_codeswitch_"))
    src = Path(__file__).parent / "data"
    if src.is_dir():
        shutil.copytree(src, tmp / "data")
    else:  # pragma: no cover - fixture missing
        (tmp / "data").mkdir(parents=True)
    cfg = AgentConfig(root=tmp)
    components = WebComponents(cfg, mock=True).build()
    components.tts = MockTtsEngine()
    session = WebSession(_RecordingSock(), components)
    return session, components


def _recorded(components: Any) -> list[tuple[str, str]]:
    """(text, language) per TTS call, in call order (the span sequence)."""
    return [(t, lang) for t, lang, _ls, _v in components.tts.synthesized]


class WebSynthesizeChunkTests(unittest.TestCase):
    """The web backend's actual per-span synthesis requests."""

    def _run(self, session: Any, chunk: str) -> Optional[bytes]:
        return asyncio.run(session._synthesize_chunk(chunk, None))

    def test_pure_hungarian_one_call(self):
        session, components = _make_session()
        pcm = self._run(session, HU_PURE)
        self.assertIsNotNone(pcm)
        self.assertGreater(len(pcm), 0)
        self.assertEqual(_recorded(components),
                         [(normalize_for_speech(HU_PURE), LANG_HU)])

    def test_pure_english_one_call(self):
        session, components = _make_session()
        pcm = self._run(session, EN_PURE)
        self.assertIsNotNone(pcm)
        self.assertEqual(_recorded(components),
                         [(normalize_for_speech(EN_PURE), LANG_EN)])

    def test_mixed_quoted_english_three_calls_in_order(self):
        session, components = _make_session()
        pcm = self._run(session, HU_QUOTED_EN)
        # (3) generated audio exists — one concatenated payload, non-empty
        self.assertIsNotNone(pcm)
        self.assertGreater(len(pcm), 0)
        recorded = _recorded(components)
        # (1)+(2) detected span boundaries and selected TTS languages
        self.assertEqual(
            recorded,
            [
                ("A ", LANG_HU),
                ('"touch base"', LANG_EN),
                (" egy gyakori angol kifejezés.", LANG_HU),
            ],
        )
        # (5) chunk ordering preserved: the recorded texts concatenate back
        # to the normalized chunk exactly, in call order.
        joined = "".join(t for t, _ in recorded)
        self.assertEqual(joined, normalize_for_speech(HU_QUOTED_EN))

    def test_mixed_catch_up_three_calls_in_order(self):
        session, components = _make_session()
        pcm = self._run(session, HU_QUOTED_EN_CATCH)
        self.assertIsNotNone(pcm)
        self.assertEqual(
            _recorded(components),
            [
                ("A ", LANG_HU),
                ('"catch up"', LANG_EN),
                (" hasonló jelentésű ebben a mondatban.", LANG_HU),
            ],
        )

    def test_quoted_hungarian_inside_english(self):
        # NOTE: the CHUNK detector is deliberately HU-biased (pinned by
        # test_text_utils: "I think the kávé here is good" -> HU): one
        # "szeretnék" (diacritic 3 + stopword 2 = 5) outweighs the four
        # English stopwords (4), so this sentence's HOST is Hungarian. The
        # v0.10.5 phrase-run pass additionally routes the leading English
        # fragment ("The word") to the EN model — same-lane improvement:
        # the segmentation only ever IMPROVES the mixed rendering; host
        # bias itself is pre-existing, out of scope here.
        session, components = _make_session()
        pcm = self._run(session, EN_QUOTED_HU)
        self.assertIsNotNone(pcm)
        recorded = _recorded(components)
        self.assertEqual(
            recorded,
            [
                ("The word ", LANG_EN),
                ('"szeretnék" means ', LANG_HU),
                ('"I would like"', LANG_EN),
                (" here.", LANG_HU),
            ],
        )
        self.assertEqual("".join(t for t, _ in recorded),
                         normalize_for_speech(EN_QUOTED_HU))

    def test_quoted_hungarian_inside_detected_english_host(self):
        # A sentence whose chunk-level detection really is English ("szia"
        # weighs 2, the EN stopwords 2, tie -> no diacritic -> EN host):
        # the quoted Hungarian word becomes a proper HU span.
        text = 'The word "szia" means hello here.'
        session, components = _make_session()
        pcm = self._run(session, text)
        self.assertIsNotNone(pcm)
        self.assertEqual(
            _recorded(components),
            [
                ("The word ", LANG_EN),
                ('"szia"', LANG_HU),
                (" means hello here.", LANG_EN),
            ],
        )

    def test_neutral_loanword_inside_english_single_call(self):
        session, components = _make_session()
        pcm = self._run(session, EN_QUOTED_NEUTRAL)
        self.assertIsNotNone(pcm)
        self.assertEqual(_recorded(components),
                         [(normalize_for_speech(EN_QUOTED_NEUTRAL), LANG_EN)])

    def test_unquoted_mixed_three_calls_in_order(self):
        # v0.10.5 phase 2: the unquoted phrase becomes its own span —
        # three synthesis calls in text order, same voice, one payload.
        session, components = _make_session()
        pcm = self._run(session, HU_UNQUOTED_MIX)
        self.assertIsNotNone(pcm)
        self.assertGreater(len(pcm), 0)
        recorded = _recorded(components)
        self.assertEqual(
            recorded,
            [
                ("Szeretnék röviden ", LANG_HU),
                ("touch base", LANG_EN),
                ("-elni veled a projekt miatt.", LANG_HU),
            ],
        )
        self.assertEqual("".join(t for t, _ in recorded),
                         normalize_for_speech(HU_UNQUOTED_MIX))

    def test_idiom_three_calls_in_order(self):
        # v0.10.5 forensic fix (operator case T5): the idiom head
        # "cut" absorbs into the EN run — the WHOLE idiom is synthesized
        # with the English model between two Hungarian spans.
        session, components = _make_session()
        pcm = self._run(session, HU_IDIOM_CUT)
        self.assertIsNotNone(pcm)
        self.assertGreater(len(pcm), 0)
        recorded = _recorded(components)
        self.assertEqual(
            recorded,
            [
                ("Most már tényleg ", LANG_HU),
                ("cut to the chase,", LANG_EN),
                (" és menjünk tovább.", LANG_HU),
            ],
        )
        self.assertEqual("".join(t for t, _ in recorded),
                         normalize_for_speech(HU_IDIOM_CUT))
        # Same voice across all three spans (one speaker, not three).
        voices = [v for _t, _l, _ls, v in components.tts.synthesized]
        self.assertEqual(voices, ["F1", "F1", "F1"])

    def test_idiom_hit_and_burn_absorb_heads(self):
        # The other two evidence-bearing idioms route whole; the
        # Hungarian derivational suffix stays host.
        session, components = _make_session()
        pcm = self._run(session, HU_IDIOM_HIT)
        self.assertIsNotNone(pcm)
        recorded = _recorded(components)
        self.assertIn(("hit the ground running", LANG_EN), recorded)
        self.assertIn(("-gyel indulni.", LANG_HU), recorded)
        self.assertEqual("".join(t for t, _ in recorded),
                         normalize_for_speech(HU_IDIOM_HIT))
        session, components = _make_session()
        pcm = self._run(session, HU_IDIOM_BURN)
        self.assertIsNotNone(pcm)
        recorded = _recorded(components)
        self.assertIn(("burn the midnight oil,", LANG_EN), recorded)
        self.assertEqual("".join(t for t, _ in recorded),
                         normalize_for_speech(HU_IDIOM_BURN))

    def test_typographic_quotes_no_unsupported_character_exception(self):
        # (4) the v0.10.3 normalization choke point runs BEFORE segmentation,
        # so „ ” ’ never reach the engine — no unsupported-character error.
        raw = "A „touch base” egy gyakori angol kifejezés."
        session, components = _make_session()
        pcm = self._run(session, raw)
        self.assertIsNotNone(pcm)
        recorded = _recorded(components)
        for text, _lang in recorded:
            self.assertNotIn("„", text)
            self.assertNotIn("”", text)
        self.assertEqual(
            recorded,
            [
                ("A ", LANG_HU),
                ('"touch base"', LANG_EN),
                (" egy gyakori angol kifejezés.", LANG_HU),
            ],
        )

    def test_contraction_sentence_uses_english_and_stays_whole(self):
        session, components = _make_session()
        pcm = self._run(session, EN_PURE_TOUCH)
        self.assertIsNotNone(pcm)
        self.assertEqual(_recorded(components),
                         [(normalize_for_speech(EN_PURE_TOUCH), LANG_EN)])

    def test_forced_hu_mode_keeps_single_language(self):
        session, components = _make_session()
        components.voice.mode = LANG_HU
        pcm = self._run(session, HU_QUOTED_EN)
        self.assertIsNotNone(pcm)
        self.assertEqual(_recorded(components),
                         [(normalize_for_speech(HU_QUOTED_EN), LANG_HU)])

    def test_forced_en_mode_keeps_single_language(self):
        session, components = _make_session()
        components.voice.mode = LANG_EN
        pcm = self._run(session, EN_QUOTED_HU)
        self.assertIsNotNone(pcm)
        self.assertEqual(_recorded(components),
                         [(normalize_for_speech(EN_QUOTED_HU), LANG_EN)])

    def test_voice_continuity_across_spans(self):
        # The EN voice preset differs from the HU preset, yet the embedded
        # EN span is spoken by the SAME (host) voice: one speaker quoting a
        # foreign phrase, not a different person.
        session, components = _make_session()
        components.voice.en_voice = "F2"
        pcm = self._run(session, HU_QUOTED_EN)
        self.assertIsNotNone(pcm)
        voices = [v for _t, _l, _ls, v in components.tts.synthesized]
        self.assertEqual(voices, ["F1", "F1", "F1"])


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLI pipeline path (VoicePipeline._speak_chunk, mock components)
# ═══════════════════════════════════════════════════════════════════════════


def _make_pipeline(voice_settings: Any = None) -> tuple[Any, MockTtsEngine]:
    from app.pipeline import VoicePipeline

    cfg = AgentConfig()
    tts = MockTtsEngine()
    pipeline = VoicePipeline(
        cfg,
        asr=MockAsrEngine(["Szia!"]),
        llm=MockLlmClient(["ok"]),
        tts=tts,
        voicemem=MockVoiceMemBridge([""]),
        vad=MockVad(),
        audio_out=None,  # offline audit mode: synthesis runs, playback skips
    )
    pipeline._voice_settings = voice_settings
    return pipeline, tts


class PipelineSpeakChunkTests(unittest.TestCase):
    """The CLI caller synthesizes the same span sequence."""

    def _speak(self, pipeline: Any, text: str) -> Any:
        from app.pipeline import TurnTimings

        return asyncio.run(pipeline._speak_chunk(text, TurnTimings(), None))

    def test_mixed_quoted_english_span_sequence(self):
        pipeline, tts = _make_pipeline()  # _voice_settings=None -> auto
        status = self._speak(pipeline, HU_QUOTED_EN)
        self.assertEqual(status, "played")
        self.assertEqual(
            [(t, lang) for t, lang, _ls, _v in tts.synthesized],
            [
                ("A ", LANG_HU),
                ('"touch base"', LANG_EN),
                (" egy gyakori angol kifejezés.", LANG_HU),
            ],
        )

    def test_idiom_span_sequence(self):
        # v0.10.5 forensic fix (CLI path mirrors the web backend): the
        # whole idiom routes to the EN model in one span.
        pipeline, tts = _make_pipeline()  # _voice_settings=None -> auto
        status = self._speak(pipeline, HU_IDIOM_CUT)
        self.assertEqual(status, "played")
        self.assertEqual(
            [(t, lang) for t, lang, _ls, _v in tts.synthesized],
            [
                ("Most már tényleg ", LANG_HU),
                ("cut to the chase,", LANG_EN),
                (" és menjünk tovább.", LANG_HU),
            ],
        )

    def test_pure_hungarian_single_call(self):
        pipeline, tts = _make_pipeline()
        status = self._speak(pipeline, HU_PURE)
        self.assertEqual(status, "played")
        self.assertEqual([(t, lang) for t, lang, _ls, _v in tts.synthesized],
                         [(HU_PURE, LANG_HU)])

    def test_forced_hu_voice_settings_disable_splitting(self):
        from app.voice_settings import VoiceSettings

        pipeline, tts = _make_pipeline(voice_settings=VoiceSettings(mode=LANG_HU))
        status = self._speak(pipeline, HU_QUOTED_EN)
        self.assertEqual(status, "played")
        self.assertEqual([(t, lang) for t, lang, _ls, _v in tts.synthesized],
                         [(HU_QUOTED_EN, LANG_HU)])

if __name__ == "__main__":
    unittest.main()
