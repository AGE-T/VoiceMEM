"""[v0.8.1] British English text convention guard (user request, 2026-09-14).

The project's English prose — UI strings, LLM prompts, slot labels, docstrings,
comments, docs — uses BRITISH ENGLISH spelling: unified by the v0.4.6
"British-English release" (user request at the time) and maintained since
("recognised", "Quantisation", "cancelling", "localise_slot"). The v0.8.1
sweep removed the last stragglers ("enrollment" -> "enrolment",
"recognized" -> "recognised", "analyzed" -> "analysed", "normalized" ->
"normalised", …) and this guard pins the four USER-VISIBLE display surfaces
so an American spelling cannot silently regress into the product text:

  * ``web/voicemem.html`` — the whole I18N dictionary (every UI string);
  * ``app/web_server.py`` — the ``_SLOT_EN`` trait-slot display labels
    (the vendor keeps the five slots as Chinese enum identifiers — machine
    facing; the display translation must stay pure-ASCII English);
  * ``app/teacher_persona.py`` — the persona system-prompt constants;
  * ``app/retrieval_contract.py`` — the trait provenance render strings
    ("last heard <date> | Nx heard") that reach the LLM context.

Deliberately NOT scanned (documented exemptions — see CONTRACT.md):

  * code identifiers (``analyze()``, ``summarize()``, ``.normalize``): the
    API surface is not a spelling question and renaming it is a breaking
    change, not a fix;
  * third-party error-matching strings (app/asr.py matches the transformers
    exception text "unrecognized configuration" / "does not recognize this
    architecture" VERBATIM — changing the spelling would break the match);
  * PowerShell parameter values (``-WindowStyle Minimized`` is a parameter
    literal, not prose);
  * historical CHANGELOG entries (the append-only record of past releases).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: American spellings that must not appear in the guarded display text.
#: Every pattern is a word-level match (identifiers stay exempt because the
#: guarded extracts contain no code — only strings and comments).
_AME = re.compile(
    r"\b("
    r"behaviors?|behavioral|behaviorally|"
    r"colors?|"
    r"favorites?|"
    r"recogniz(e|es|ed|ing)|"
    r"organiz(e|es|ed|ing|ation\w*)|"
    r"prioritiz(e|es|ed|ing)|"
    r"summariz(e|es|ed|ing)|"
    r"analyz(e|es|ed|ing)|"
    r"normaliz(e|es|ed|ing)|"
    r"minimiz(e|es|ed|ing)|"
    r"maximiz(e|es|ed|ing)|"
    r"customiz(e|es|ed|ing)|"
    r"centraliz(e|es|ed|ing)|"
    r"initializ(e|es|ed|ing)|"
    r"synchroniz(e|es|ed|ing)|"
    r"stabiliz(e|es|ed|ing)|"
    r"labeled|labeling|"
    r"canceled|canceling|"
    r"modeling|modeled|"
    r"traveled|traveling|"
    r"enrollments?|"
    r"grays?|"
    r"fulfills?|fulfilled|fulfilling|"
    r"toward\b(?!s)"
    r")\b",
    re.IGNORECASE,
)

#: CJK leak detector for display values (the vendored trait slots ARE
#: Chinese identifiers by design — the DISPLAY layer must never show them).
_CJK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def _i18n_block() -> str:
    """The whole I18N dictionary region of web/voicemem.html."""
    html = (REPO / "web" / "voicemem.html").read_text(encoding="utf-8")
    start = html.index("const I18N = {")
    end = html.index("const LANG = 'en';")
    return html[start:end]


def _slot_en_block() -> str:
    """The _SLOT_EN dict in app/web_server.py (display labels only)."""
    src = (REPO / "app" / "web_server.py").read_text(encoding="utf-8")
    start = src.index("_SLOT_EN = {")
    end = src.index("\n}", start)
    return src[start:end]


class BritishEnglishSurfaceTests(unittest.TestCase):
    """No American spelling may enter the guarded display surfaces."""

    def _assert_no_ame(self, surface_name: str, text: str) -> None:
        hits = sorted(set(m.group(0).lower() for m in _AME.finditer(text)))
        self.assertEqual(
            hits, [],
            f"{surface_name}: American spellings found (British English is "
            f"the project convention — see CONTRACT.md): {hits}",
        )

    def test_i18n_dictionary_is_british_english(self) -> None:
        self._assert_no_ame("web/voicemem.html I18N", _i18n_block())

    def test_i18n_dictionary_has_no_cjk_leak(self) -> None:
        """The UI is English-only text: no CJK may appear in the strings."""
        hits = sorted(set(_CJK.findall(_i18n_block())))
        self.assertEqual(
            hits, [],
            "web/voicemem.html I18N: CJK characters leaked into the UI "
            "text (the vendor's Chinese slot enums must be translated via "
            "_SLOT_EN / localise_slot, never shown raw)",
        )

    def test_slot_labels_are_british_english(self) -> None:
        self._assert_no_ame("app/web_server.py _SLOT_EN", _slot_en_block())

    def test_slot_label_values_are_pure_ascii_english(self) -> None:
        """_SLOT_EN KEYS are the vendor's Chinese identifiers (by design);
        the VALUES — what the user sees — must be pure-ASCII English."""
        block = _slot_en_block()
        values = re.findall(r":\s*\"([^\"]+)\"", block)
        self.assertEqual(len(values), 5, f"expected 5 slot labels: {values!r}")
        for v in values:
            self.assertTrue(
                v and all(ord(c) < 128 for c in v),
                f"_SLOT_EN display value not pure ASCII English: {v!r}",
            )

    def test_persona_prompt_is_british_english(self) -> None:
        src = (REPO / "app" / "teacher_persona.py").read_text(encoding="utf-8")
        self._assert_no_ame("app/teacher_persona.py", src)

    def test_retrieval_renders_are_british_english(self) -> None:
        src = (REPO / "app" / "retrieval_contract.py").read_text(encoding="utf-8")
        self._assert_no_ame("app/retrieval_contract.py", src)

    def test_trait_provenance_render_wording(self) -> None:
        """The canonical render strings (Phase 9 wording) stay exact."""
        sys_path_insert = str(REPO)
        import sys

        if sys_path_insert not in sys.path:
            sys.path.insert(0, sys_path_insert)
        from app.retrieval_contract import trait_prompt_suffix

        class _H:  # minimal trait hit shape (metadata dict carrier)
            metadata = {
                "trait_id": "t1", "slot_name": "情绪", "claim": "x",
                "confidence": 0.93, "eff_confidence": 0.93,
                "occurrence_count": 3, "first_seen": "2026-06-01",
                "last_seen": "2026-09-01",
            }

        self.assertEqual(
            trait_prompt_suffix(_H()), " [last heard 2026-09-01 | 3x heard]"
        )


if __name__ == "__main__":
    unittest.main()
