"""[VM-LOCAL-015] Unit tests: the deterministic stance cue scanner.

Phase 10 (Hungarian) and Phase 3 (structured signals) coverage. The
scanner is the trait contradiction gate's signal layer — pure function,
no I/O, no LLM. These tests pin the REQUIRED semantic cases:

  * "szeretem" / "nem szeretem" / "már nem szeretem" / "régen szerettem" /
    "talán már nem szeretem" / "nem mindig szeretem" / "kivéve télen"
  * English equivalents: likes / does not like / no longer likes /
    used to like / might no longer like / except during winter / hates
  * the antipathy class (opposite affect without a negation word)
  * the presupposition discriminator (wide-band licensing)
  * the topic-overlap guard (supersede must not cross topics)
  * the Hungarian present-tense trap ("szeretem" must NOT classify past)
  * English/Hungarian cross-substring traps (revolt/esteem)

No storage involved — storage-level behaviour is test_trait_semantics.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
for p in (str(REPO), str(VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from voicemem.rightbrain.stance import (  # noqa: E402
    classify_stance,
    is_presuppositional,
    observation_stance,
    topic_overlaps,
)


class HungarianRequiredCases(unittest.TestCase):
    """The exact Phase 10 list — each label pinned to its semantic class."""

    def test_szeretem_is_positive(self):
        self.assertEqual(classify_stance("szeretem"), "pos")

    def test_nem_szeretem_is_neg(self):
        self.assertEqual(classify_stance("nem szeretem"), "neg")

    def test_mar_nem_szeretem_is_neg(self):
        self.assertEqual(classify_stance("már nem szeretem"), "neg")

    def test_regen_szerettem_is_past(self):
        self.assertEqual(classify_stance("régen szerettem"), "past")

    def test_talan_mar_nem_szeretem_is_uncertain(self):
        # Uncertainty must NEVER become a hard replacement (Phase 1 case G).
        self.assertEqual(classify_stance("talán már nem szeretem"), "uncertain")

    def test_nem_mindig_szeretem_is_qualified(self):
        self.assertEqual(classify_stance("nem mindig szeretem"), "qualified")

    def test_kiveve_telen_is_qualified(self):
        self.assertEqual(classify_stance("kivéve télen"), "qualified")

    def test_present_tense_first_person_is_not_past(self):
        # The single-t suffix trap: "szeretem" (I like, PRESENT) ends in
        # "tem" — a naive ending match would misclassify every 1st-person
        # present statement as past and destroy the negation semantics.
        for text in ("szeretem", "kedvelem", "imádom", "nem szeretem"):
            self.assertNotEqual(classify_stance(text), "past", text)

    def test_hungarian_third_person_past_detected(self):
        self.assertEqual(classify_stance("szerette a motorokat"), "past")
        self.assertEqual(classify_stance("lakott Budapesten"), "past")


class EnglishRequiredCases(unittest.TestCase):
    def test_likes_is_positive(self):
        self.assertEqual(classify_stance("Thomas likes motorcycles"), "pos")

    def test_does_not_like_is_neg(self):
        self.assertEqual(classify_stance("Thomas does not like motorcycles"), "neg")

    def test_no_longer_is_neg(self):
        self.assertEqual(classify_stance("Thomas no longer likes motorcycles"), "neg")

    def test_used_to_is_past(self):
        self.assertEqual(classify_stance("Thomas used to like motorcycles"), "past")

    def test_previously_is_past(self):
        self.assertEqual(classify_stance("Thomas previously owned a BMW"), "past")

    def test_might_no_longer_is_uncertain(self):
        self.assertEqual(
            classify_stance("Thomas might no longer like motorcycles"), "uncertain")

    def test_except_during_winter_is_qualified(self):
        self.assertEqual(
            classify_stance("Thomas likes motorcycles except during winter"),
            "qualified")

    def test_hates_is_neg_antipathy(self):
        # Opposite affect WITHOUT a negation word — the pair that measured
        # 0.9685 against "szereti" (inside the merge band!).
        self.assertEqual(classify_stance("Thomas hates motorcycles"), "neg")
        self.assertEqual(classify_stance("Thomas utálja a motorokat"), "neg")

    def test_neutral_claim_has_no_stance(self):
        self.assertEqual(classify_stance("gives examples before conclusions"), "")
        self.assertEqual(classify_stance("weighs every option before deciding"), "")

    def test_dislike_substring_does_not_read_as_positive(self):
        # "dislikes" contains "likes" — antipathy is scanned BEFORE the
        # positive lexicon.
        self.assertEqual(classify_stance("dislikes long meetings"), "neg")
        self.assertEqual(classify_stance("likes long meetings"), "pos")


class CrossLanguageTraps(unittest.TestCase):
    def test_english_words_containing_hungarian_cues(self):
        # "volt" ⊂ "revolt", "este" ⊂ "esteem" — boundary matching must hold.
        self.assertEqual(classify_stance("the revolution changed his approach"), "")
        self.assertEqual(classify_stance("an esteemed colleague"), "")
        self.assertNotEqual(classify_stance("an esteemed colleague"), "qualified")

    def test_cross_language_negation_still_detected(self):
        # Phase 11: the quote is the ground truth — a Hungarian quote with
        # an English claim (or vice versa) still yields the neg stance.
        self.assertEqual(
            observation_stance("likes motorcycles",
                               "Thomas már nem szereti a motorokat"),
            "neg")
        self.assertEqual(
            observation_stance("szereti a motorokat",
                               "Thomas no longer likes motorcycles"),
            "neg")


class PresuppositionDiscriminator(unittest.TestCase):
    def test_presuppositional_negation_detected(self):
        for text in ("no longer likes motorcycles", "gave up coffee",
                     "stopped liking coffee", "már nem szereti",
                     "abbahagyta a kávét"):
            self.assertTrue(is_presuppositional(text), text)

    def test_plain_negation_is_not_presuppositional(self):
        for text in ("does not like motorcycles", "hates motorcycles",
                     "nem szeretem"):
            self.assertFalse(is_presuppositional(text), text)


class ObservationStanceCombine(unittest.TestCase):
    def test_claim_normalisation_loss_recovered_from_quote(self):
        # The extractor may normalise the negation out of the label —
        # the quote still carries it (never miss a negation).
        self.assertEqual(observation_stance("likes motorcycles",
                                            "I don't like them anymore"), "neg")

    def test_uncertainty_in_quote_qualifies_the_claim(self):
        self.assertEqual(observation_stance("no longer likes motorcycles",
                                            "talán már nem szereti"), "uncertain")

    def test_empty_inputs(self):
        self.assertEqual(observation_stance("", ""), "")
        self.assertEqual(observation_stance("likes", ""), "pos")


class TopicOverlapGuard(unittest.TestCase):
    """The supersede band's topic guard — measured noise floor protection."""

    def test_same_topic_overlaps(self):
        self.assertTrue(topic_overlaps("likes motorcycles",
                                       "no longer likes motorcycles"))
        self.assertTrue(topic_overlaps("likes coffee", "gave up coffee"))
        self.assertTrue(topic_overlaps("likes motorcycles",
                                       "már nem szereti a motorokat"))
        self.assertTrue(topic_overlaps("likes motorcycles",
                                       "megint szereti a motorokat"))

    def test_different_topic_does_not_overlap(self):
        # Measured 0.8876 — inside the presupposition band; ONLY the topic
        # guard keeps this pair from superseding each other.
        self.assertFalse(topic_overlaps("likes motorcycles",
                                        "no longer likes bicycles"))
        self.assertFalse(topic_overlaps("likes coffee", "gave up tea"))

    def test_prefix_stem_match(self):
        # motor/motorcycles share a Latin prefix — the cross-language
        # EN claim vs HU claim case.
        self.assertTrue(topic_overlaps("likes motorcycles",
                                       "utálja a motorokat"))

    def test_function_words_are_not_topics(self):
        self.assertFalse(topic_overlaps("likes this", "likes that"))


if __name__ == "__main__":
    unittest.main()
