"""Unit tests for the v0.4.7 ASR output parser and slot localisation.

The parser turns Qwen3-ASR's raw decoded string into the plain transcript:
auto-detect output carries a ``language <LANG><asr_text>`` header, forced
language output is plain text, and a missing tag falls back to the whole
string. ``localise_slot`` maps the vendored VoiceMem's Chinese trait-slot
enum values to English display labels (the v0.4.6 field report showed
"Profile · 应对方式 ×1" cards).

Pure standard library only — no torch/transformers needed (the import in
app.asr is guarded and degrades to None in the sandbox).
"""

from __future__ import annotations

import unittest


class ParseAsrOutputTests(unittest.TestCase):
    def _parse(self, raw: str, forced: bool = False) -> str:
        from app.asr import _parse_asr_output

        return _parse_asr_output(raw, forced)

    def test_auto_detect_with_tag(self) -> None:
        """The canonical auto-detect output: language line + tag + text."""
        self.assertEqual(
            self._parse("language Hungarian<asr_text>Szia, itt vagy?"),
            "Szia, itt vagy?",
        )
        self.assertEqual(
            self._parse("language English<asr_text>Hello there"),
            "Hello there",
        )

    def test_auto_detect_empty_transcript(self) -> None:
        """Silent audio: the model may emit just the language header."""
        self.assertEqual(self._parse("language None<asr_text>"), "")
        self.assertEqual(self._parse(""), "")

    def test_forced_language_passthrough(self) -> None:
        """Forced language prefill: the generation is plain text."""
        self.assertEqual(self._parse("Szia, hogy vagy?", forced=True), "Szia, hogy vagy?")
        # even a stray tag stays verbatim in forced mode (no parsing)
        self.assertEqual(self._parse("<asr_text>x", forced=True), "<asr_text>x")

    def test_no_tag_whole_string(self) -> None:
        """Model omitted the tag: the whole string is the transcript."""
        self.assertEqual(self._parse("plain transcription"), "plain transcription")

    def test_no_tag_bare_language_line(self) -> None:
        """No tag but a bare 'language X' first line: the line is dropped."""
        self.assertEqual(self._parse("language Hungarian\nszia ott"), "szia ott")

    def test_whitespace_normalised(self) -> None:
        self.assertEqual(self._parse("  language Hungarian<asr_text>  hello  "), "hello")

    def test_none_and_non_string(self) -> None:
        self.assertEqual(self._parse(None), "")  # type: ignore[arg-type]
        self.assertEqual(self._parse(123), "123")  # type: ignore[arg-type]


class SlotLocalisationTests(unittest.TestCase):
    def _localise(self, slot: str) -> str:
        from app.web_server import localise_slot

        return localise_slot(slot)

    def test_all_five_vendor_slots(self) -> None:
        """The five Chinese enum values map to English labels."""
        self.assertEqual(self._localise("情绪"), "emotion")
        self.assertEqual(self._localise("应对方式"), "coping style")
        self.assertEqual(self._localise("表达风格"), "expression style")
        self.assertEqual(self._localise("思维模式"), "thinking style")
        self.assertEqual(self._localise("喜好与厌恶"), "likes and dislikes")

    def test_whitespace_stripped(self) -> None:
        self.assertEqual(self._localise("  应对方式  "), "coping style")

    def test_unknown_value_passthrough(self) -> None:
        """English/unknown slots (already localised or from the demo layer)."""
        self.assertEqual(self._localise("emotion"), "emotion")
        self.assertEqual(self._localise("daily_life"), "daily_life")
        self.assertEqual(self._localise(""), "")

    def test_none_tolerated(self) -> None:
        self.assertEqual(self._localise(None), None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
