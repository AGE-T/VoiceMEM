"""v0.10.2 (operator PART 3/14) — the ASR language diagnostic mode.

Pinned behaviours:

1. ``_resolve_language_mode`` (PURE): "" / "auto" (any case) -> auto;
   anything else -> a normalised diagnostic hint ("hu").
2. The PRODUCTION parakeet engine now READS ``asr_language`` (previously
   only the legacy engines did — the operator's observation "the adapter
   does not clearly pass that value into the Parakeet inference call"
   was correct); the mode + hint surface in ``status()``. The engine
   cannot force the output language (ParakeetForTDT has no language
   parameter — multilingual auto-detect only), so the hint is recorded
   and the transcript script is verified after the decode instead.
3. Config surface: validation accepts "" / "auto" / simple tags, rejects
   decorated tags; the ASR_LANGUAGE env override works; the shipped yaml
   default stays "" (auto — production is NOT permanently forced).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]


def engine_with_language(value: str):
    """The production parakeet engine, constructed with asr_language=value.

    select_engine is lazy (no model load happens on construction), so this
    is safe without the weights.
    """
    from app.asr_core import select_engine

    cfg = AgentConfig(root=_ROOT)
    cfg.asr_engine = "parakeet"
    cfg.asr_device = "cpu"
    cfg.asr_language = value
    return select_engine(cfg)


class ResolveLanguageModeTests(unittest.TestCase):
    def test_empty_and_auto_normalize_to_auto(self):
        from app.asr_parakeet import _resolve_language_mode
        for raw in ("", "  ", "auto", "AUTO", "Auto", " auto "):
            mode, hint = _resolve_language_mode(raw)
            self.assertEqual((mode, hint), ("auto", ""), f"{raw!r} -> auto")

    def test_language_tag_normalizes_to_hint(self):
        from app.asr_parakeet import _resolve_language_mode
        for raw in ("hu", " Hu ", "HU"):
            mode, hint = _resolve_language_mode(raw)
            self.assertEqual((mode, hint), ("hint", "hu"), f"{raw!r} -> hint hu")
        mode, hint = _resolve_language_mode("de")
        self.assertEqual((mode, hint), ("hint", "de"))

    def test_cyrillic_detector_is_exact(self):
        from app.asr_parakeet import _has_cyrillic
        self.assertTrue(_has_cyrillic("Я им работаю, валь."))
        self.assertFalse(_has_cyrillic("Szia, imre! Mi újság?"))
        self.assertFalse(_has_cyrillic(""))
        self.assertFalse(_has_cyrillic("store school"))


class EngineLanguageModeTests(unittest.TestCase):
    """Engine-level wiring (NO model load — construction only)."""

    def test_engine_records_auto_mode(self):
        eng = engine_with_language("")
        st = eng.status()
        self.assertEqual(st["language_mode"], "auto")
        self.assertEqual(st["language_hint"], "")

    def test_engine_records_hu_hint(self):
        eng = engine_with_language("hu")
        st = eng.status()
        self.assertEqual(st["language_mode"], "hint")
        self.assertEqual(st["language_hint"], "hu")
        self.assertIn("language_mode", st)  # diagnostics surface present

    def test_engine_records_auto_from_literal(self):
        eng = engine_with_language("auto")
        self.assertEqual(eng.status()["language_mode"], "auto")


class ConfigSurfaceTests(unittest.TestCase):
    def test_validation_accepts_auto_empty_and_simple_tags(self):
        for value in ("", "auto", "hu", "HU"):
            cfg = AgentConfig(asr_language=value)
            self.assertEqual(cfg.validate(), [], f"{value!r} must be valid")

    def test_validation_rejects_decorated_tags(self):
        cfg = AgentConfig(asr_language="hu-HU")
        self.assertNotEqual(cfg.validate(), [],
                            "decorated tags must fail loudly (simple tags only)")

    def test_env_override_applies(self):
        old = os.environ.pop("ASR_LANGUAGE", None)
        try:
            os.environ["ASR_LANGUAGE"] = "hu"
            cfg = AgentConfig()
            cfg.apply_env()  # from_yaml calls this; the direct ctor does not
            self.assertEqual(cfg.asr_language, "hu")
            os.environ["ASR_LANGUAGE"] = "auto"
            cfg = AgentConfig()
            cfg.apply_env()
            self.assertEqual(cfg.asr_language, "auto")
        finally:
            if old is None:
                os.environ.pop("ASR_LANGUAGE", None)
            else:
                os.environ["ASR_LANGUAGE"] = old

    def test_shipped_yaml_default_is_auto(self):
        """Production stays on auto (the operator contract: do NOT
        permanently force Hungarian; 'hu' is a diagnostic mode)."""
        import yaml

        data = yaml.safe_load(
            (_ROOT / "config" / "voicemem_config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["app"]["asr_language"], "")


if __name__ == "__main__":
    unittest.main()
