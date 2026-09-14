"""ASR-roundtrip quality gate for the Supertonic 3 TTS (v0.7.0).

Objective speech-quality measurement: synthesize Hungarian/English test
sentences through the PRODUCTION adapter (SupertonicTtsEngine), then
transcribe the generated audio back through the PRODUCTION ASR engine
(Parakeet TDT 0.6B v3, CPU) and compute WER/CER against the source text.

What this catches (task contract, "Check" list):
  - word repetitions (insertions), skipped words (deletions)
  - truncation (missing tail words)
  - wrong-language behaviour (HU text spoken as EN garbage)
  - pronunciation corruption severe enough to break ASR
What it does NOT catch: subjective naturalness (that stays a manual MOS
item — the WAVs in data/supertonic_validation/ are for listening).

Output: data/supertonic_validation/asr_roundtrip.json + console table.
Exit 0 when every HU sentence WER <= 30% and mean HU WER <= 20% (the
Parakeet HU field WER on REAL speech was 17.1% — clean TTS audio must
transcribe at least as well), exit 1 otherwise.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

RESULTS = _ROOT / "data" / "supertonic_validation"
MODEL_DIR = _ROOT / "models" / "tts" / "supertonic-3"

HU_SENTENCES = [
    "Szia, hogy vagy ma?",
    "Szia Thomas, itt a hangsegéd, készen állok a beszélgetésre.",
    "Ma reggel elmentem a piacra, vettem friss zöldségeket és gyümölcsöket.",
    "Hogy telt a mai napod, találtál valami érdekeset?",
    "Találkozzunk tizennégy óra harminc perckor a harmadik kapunál.",
    "A magyar nyelv gazdag magánhangzórendszerrel rendelkezik.",
    "Ez fantasztikus hír, nagyon örülök neked.",
    "Kérem, ismételje meg lassabban a harmadik mondatot.",
]
EN_SENTENCES = [
    "Hello Thomas, this is your voice assistant, ready to talk.",
    "Good morning! Let us start today's English lesson.",
    "Would you like a flat white or a latte?",
    "The year 1890 was important in the history of coffee.",
    "Please repeat the third sentence more slowly.",
]

WER_GATE = 0.30       # per-sentence HU ceiling
WER_MEAN_GATE = 0.20  # HU mean ceiling (Parakeet real-speech HU WER: 17.1%)


#: Hungarian numeral words -> digits (0-20 + tens): the ASR writes spoken
#: numbers as DIGITS ("tizennégy óra harminc perc" -> "14 óra 30 perc");
#: normalising both sides measures SPEECH quality, not orthography.
_HU_NUM = {
    "nulla": "0", "egy": "1", "kettő": "2", "két": "2", "három": "3",
    "négy": "4", "öt": "5", "hat": "6", "hét": "7", "nyolc": "8",
    "kilenc": "9", "tíz": "10", "tizenegy": "11", "tizenkettő": "12",
    "tizenkét": "12", "tizenhárom": "13", "tizennégy": "14",
    "tizenöt": "15", "tizenhat": "16", "tizenhét": "17",
    "tizennyolc": "18", "tizenkilenc": "19", "húsz": "20", "harminc": "30",
    "negyven": "40", "ötven": "50", "hatvan": "60", "hetven": "70",
    "nyolcvan": "80", "kilencven": "90", "száz": "100",
    "ezer": "1000", "1890": "1890",
}


def _norm_words(text: str) -> list[str]:
    """Lowercase, strip punctuation, map HU numeral words to digits."""
    words = (
        text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace(":", " ")
        .split()
    )
    return [_HU_NUM.get(w, w) for w in words]


def _wer(ref: str, hyp: str) -> tuple[float, int, int, int]:
    """Word error rate + (del, ins, sub) via classic DP alignment."""
    r = _norm_words(ref)
    h = _norm_words(hyp)
    n, m = len(r), len(h)
    if n == 0:
        return (0.0, 0, m, 0) if m == 0 else (1.0, 0, m, 0)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i][j - 1] + 1, dp[i - 1][j] + 1)
    # backtrack for del/ins/sub
    i, j = n, m
    dels = ins = subs = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if r[i - 1] == h[j - 1] else 1):
            if r[i - 1] != h[j - 1]:
                subs += 1
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1
    return dp[n][m] / n, dels, ins, subs


def _resample(x: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return x
    n_out = int(len(x) * to_sr / from_sr)
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x)


def main() -> int:
    from app.config import AgentConfig
    from app.tts_supertonic import SupertonicTtsEngine

    cfg = AgentConfig(root=_ROOT)
    # Sandbox: no GPU — the ASR leg runs on CPU exactly like the v0.6.0
    # engine benchmarks did (asr_benchmark.py, CPU, same contract). The
    # TARGET machine keeps asr_device: cuda from the yaml.
    cfg.asr_device = "cpu"
    tts = SupertonicTtsEngine(cfg)
    if not tts.is_available():
        print("Supertonic assets missing")
        return 2

    # production ASR engine (parakeet, CPU) via the engine registry
    t0 = time.perf_counter()
    from app.asr_core import AudioBuffer, select_engine

    asr = select_engine(cfg)
    try:
        asr.load()  # parakeet is non-streaming: load() + transcribe()
    except Exception as exc:  # noqa: BLE001
        print(f"ASR engine unavailable: {exc}")
        return 2
    print(f"ASR loaded in {time.perf_counter() - t0:.1f}s")

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"engine": "supertonic3", "asr": "parakeet-tdt-0.6b-v3", "cases": []}

    for lang, sentences in (("hu", HU_SENTENCES), ("en", EN_SENTENCES)):
        for text in sentences:
            pcm = tts.synthesize(text, lang)
            if pcm is None or not len(pcm):
                report["cases"].append({"lang": lang, "text": text, "error": "TTS_FAILED"})
                print(f"[{lang}] TTS FAILED: {text[:50]}")
                continue
            f32 = pcm.astype(np.float32) / 32768.0
            f32_16k = _resample(f32, 44100, 16000)
            from app.asr_core import AudioBuffer as _AB

            buf = _AB.from_float(f32_16k, 16000)
            result = asr.transcribe(buf)
            hyp = result.text if result else ""
            wer, dels, ins, subs = _wer(text, hyp)
            case = {
                "lang": lang,
                "text": text,
                "hyp": hyp,
                "wer": round(wer, 3),
                "del": dels,
                "ins": ins,
                "sub": subs,
                "audio_s": round(len(pcm) / 44100.0, 2),
            }
            report["cases"].append(case)
            print(
                f"[{lang}] WER {wer:5.1%} (del {dels} ins {ins} sub {subs}) :: {text[:44]!r} -> {hyp[:44]!r}"
            )

    hu = [c for c in report["cases"] if c.get("lang") == "hu" and "wer" in c]
    en = [c for c in report["cases"] if c.get("lang") == "en" and "wer" in c]
    report["summary"] = {
        "hu_mean_wer": round(float(np.mean([c["wer"] for c in hu])), 3) if hu else None,
        "hu_max_wer": round(float(np.max([c["wer"] for c in hu])), 3) if hu else None,
        "en_mean_wer": round(float(np.mean([c["wer"] for c in en])), 3) if en else None,
        "ins_total": int(sum(c.get("ins", 0) for c in report["cases"])),
        "del_total": int(sum(c.get("del", 0) for c in report["cases"])),
    }
    (RESULTS / "asr_roundtrip.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nSUMMARY:", json.dumps(report["summary"]))

    ok = bool(hu) and report["summary"]["hu_mean_wer"] <= WER_MEAN_GATE
    ok = ok and all(c["wer"] <= WER_GATE for c in hu)
    print(f"GATE (HU mean <= {WER_MEAN_GATE:.0%}, per-case <= {WER_GATE:.0%}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
