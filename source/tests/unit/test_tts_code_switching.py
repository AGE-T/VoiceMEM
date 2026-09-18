"""v0.10.4 TTS code-switching tests (field report: embedded language spans).

Production bug: language selection happened ONCE PER TTS CHUNK, so an
English phrase quoted inside a Hungarian sentence ("A "touch base" egy
gyakori angol kifejezés.") was synthesized with the Hungarian Supertonic
model and became unintelligible.

These tests pin the fix at all three levels:

1. ``segment_language_spans`` (pure) — span boundaries and languages for the
   field-reported sentences + the edge battery (contractions, quotes,
   parentheses, commas, hyphens, numbers, URLs, e-mails, unmatched quotes).
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

    def test_unquoted_code_switch_stays_host(self):
        # Documented policy: unquoted code-switching keeps the host language
        # (word-level runs cannot place the phrase boundary without a lexicon).
        self.assertEqual(_spans(HU_UNQUOTED_MIX, LANG_HU),
                         [(normalize_for_speech(HU_UNQUOTED_MIX), LANG_HU)])

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
        text = 'A "touch base egy dolog.'
        self.assertEqual(_spans(text, LANG_HU),
                         [(normalize_for_speech(text), LANG_HU)])
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
        # spans then correctly keep "szeretnék" inside the HU host and
        # render the English quote with the EN model — the segmentation
        # only ever IMPROVES the mixed rendering; host bias itself is
        # pre-existing, out of scope here.
        session, components = _make_session()
        pcm = self._run(session, EN_QUOTED_HU)
        self.assertIsNotNone(pcm)
        recorded = _recorded(components)
        self.assertEqual(
            recorded,
            [
                ('The word "szeretnék" means ', LANG_HU),
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

    def test_unquoted_mixed_single_call(self):
        session, components = _make_session()
        pcm = self._run(session, HU_UNQUOTED_MIX)
        self.assertIsNotNone(pcm)
        self.assertEqual(_recorded(components),
                         [(normalize_for_speech(HU_UNQUOTED_MIX), LANG_HU)])

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
