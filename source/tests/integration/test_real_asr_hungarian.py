"""v0.10.1 (P0 ASR forensic) — REAL Parakeet inference regression tests.

WHY THESE TESTS EXIST
====================
The v0.10.0 P0 field report (Hungarian speech -> fluent Russian / unrelated
English on the target machine) revealed a VERIFICATION GAP: the release gate
contained NO test that runs the production Parakeet engine on real audio —
the ASR tests were all contract-level (mocked backends). A real-inference
regression was therefore INVISIBLE to every gate since v0.6.0, on both
versions involved here.

These tests close that gap at the level the sandbox/target can afford:
the known-reference bench corpus (Piper-HU + the real Windows field capture)
through the EXACT production path (AgentConfig.from_yaml -> select_engine ->
ParakeetEngine.transcribe(AudioBuffer.from_wav(...))), on CPU (deterministic;
the on-target CUDA cell is covered by scripts/asr_regression_forensic.py).

Pinned behaviour:

  * status ok, no engine error;
  * NO Cyrillic in any transcript (Hungarian is Latin script; the reported
    failure signature was fluent Russian);
  * character error rate against the reference transcript within the
    release-verified budget (observed: 0.000-0.099 across the corpus on
    transformers 5.9-5.17; budget 0.25 = 2.5x margin).

Env-neutrality: every skip condition is an absent ASSET (torch /
transformers.ParakeetForTDT / model weights / corpus WAV), never a platform
assumption — the same rule the Silero real-speech tests follow.
"""
from __future__ import annotations

import sys
import unicodedata
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.config import AgentConfig  # noqa: E402
from app.asr_core import AudioBuffer, select_engine  # noqa: E402
from app.web_server import _has_cyrillic  # noqa: E402

MODEL_DIR = REPO / "models" / "asr" / "parakeet-tdt-0.6b-v3"
BENCH = REPO / "data" / "asr_bench"

#: (file stem, reference transcript, CER budget)
CORPUS = [
    ("hu_short", "Szia, hogy vagy ma?", 0.15),
    ("hu_names",
     "Kovács Anna és Nagy Béla Szegeden találkozott Szabó Évával a Tisza "
     "partján.", 0.25),
    ("real_windows_capture",
     "Ez egy teszt. Most kipróbáljuk, hogy működik-e az ASR-rendszer, de "
     "szerintem kurvára nem működik úgy.", 0.25),
]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _cer(expected: str, got: str) -> float:
    a, b = _norm(expected), _norm(got)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def _assets_available() -> str:
    """'' when the real-inference tests can run, else the missing asset."""
    try:
        import torch  # noqa: F401
    except Exception:
        return "torch not installed"
    try:
        from transformers import ParakeetForTDT  # noqa: F401
    except Exception:
        return "transformers.ParakeetForTDT unavailable (needs >= 5.9)"
    if not (MODEL_DIR / "config.json").is_file():
        return f"Parakeet weights not present ({MODEL_DIR})"
    for stem, _, _ in CORPUS:
        if not (BENCH / f"{stem}.wav").is_file():
            return f"corpus wav missing ({BENCH / stem}.wav)"
    return ""


_MISSING = _assets_available()


class RealParakeetHungarianTests(unittest.TestCase):
    """Real inference on the known-reference corpus — the v0.10.1 gate gap."""

    @classmethod
    def setUpClass(cls) -> None:
        if _MISSING:
            raise unittest.SkipTest(f"env-gap (asset absent): {_MISSING}")
        cfg = AgentConfig.from_yaml(REPO / "config" / "voicemem_config.yaml")
        cfg.asr_device = "cpu"  # deterministic; cuda leg = on-target forensic
        cls.engine = select_engine(cfg)
        cls.cfg = cfg

    def test_hungarian_transcribes_correctly_and_never_flips_language(self) -> None:
        for stem, expected, budget in CORPUS:
            with self.subTest(file=stem):
                buf = AudioBuffer.from_wav(BENCH / f"{stem}.wav").validated(
                    stage="asr_input"
                )
                res = self.engine.transcribe(buf)
                self.assertEqual(
                    res.status.value, "ok",
                    f"{stem}: engine error: "
                    f"{res.error.detail if res.error else 'none'}",
                )
                text = res.text.strip()
                self.assertTrue(text, f"{stem}: empty transcript")
                self.assertFalse(
                    _has_cyrillic(text),
                    f"{stem}: CYRILLIC transcript (wrong-language flip, the "
                    f"v0.10.0 P0 signature): {text!r}",
                )
                cer = _cer(expected, text)
                self.assertLessEqual(
                    cer, budget,
                    f"{stem}: CER {cer:.3f} over budget {budget} — "
                    f"expected {expected!r}, got {text!r}",
                )

    def test_reported_failure_signature_is_cyrillic_detected(self) -> None:
        """The dumped-evidence trigger for the reported transcripts."""
        self.assertTrue(_has_cyrillic("Я им работаю, валь."))
        self.assertFalse(_has_cyrillic("store school"))
        self.assertFalse(_has_cyrillic("Szia, Imre! Mi újság?"))


if __name__ == "__main__":
    unittest.main()
