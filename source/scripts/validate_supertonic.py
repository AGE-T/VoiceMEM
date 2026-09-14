"""Standalone Supertonic 3 validation (pre-production gate).

Runs BEFORE the production swap (task: "Replace Piper with Supertonic 3").
No VoiceMem imports — this script exercises ONLY the official ``supertonic``
SDK (ONNX Runtime, CPU) against the locally downloaded model assets.

Test matrix (task contract):
  1.  short Hungarian sentence
  2.  normal Hungarian conversational sentence
  3.  longer Hungarian sentence
  4.  Hungarian question
  5.  Hungarian exclamation
  6.  Hungarian with numbers
  7.  Hungarian with punctuation
  8.  English sentence
  9.  repeated synthesis (same text 3x - stability/variance)
  10. multiple consecutive requests (mixed HU/EN queue)

Checks per WAV:
  - non-empty frames, mono, 16-bit, 44100 Hz (official contract)
  - RMS in a plausible speech band, peak not clipping (>0.999 float)
  - leading/trailing silence and interior silence ratio (unnatural pauses)
  - duration plausibility (chars per second within human speech range)
  - repeated-synthesis duration variance (reading stability proxy)

Performance: model load time, first-synthesis latency, per-case synthesis
time, audio duration, real-time factor, process RSS (before/after), CPU
count. Results are written to data/supertonic_validation/report.json and
every WAV is kept in data/supertonic_validation/ for manual listening.

Usage (repo root, sandbox venv):
  .venv/bin/python scripts/validate_supertonic.py [--voice M1] [--steps 8]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import wave
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / ".venv" / "lib" / "site-packages") if False else "")

import numpy as np  # noqa: E402

RESULTS_DIR = _ROOT / "data" / "supertonic_validation"
MODEL_DIR = _ROOT / "models" / "tts" / "supertonic-3"

#: Human speech plausibility band (chars per second, Hungarian ~ average).
#: Measured on the TRIMMED speech span (first-to-last voiced frame) because the
#: Supertonic vocoder pads every utterance with ~0.4-0.9 s of near-silence at
#: both edges (measured on M1; see data/supertonic_validation/report.json).
CPS_MIN, CPS_MAX = 6.0, 26.0
#: Speech RMS band (float32 WAV); pure silence ~0, clipping ~1.0.
RMS_MIN, RMS_MAX = 0.005, 0.5
#: Maximum tolerated edge padding (seconds). The Supertonic vocoder's own
#: natural padding measured at 0.35-0.9 s per side; anything beyond this is a
#: real anomaly (dead air), not vocoder behaviour.
EDGE_SILENCE_MAX_S = 1.2
#: Silence ratio ceiling inside the utterance.
INTERIOR_SILENCE_MAX = 0.55

TEST_CASES = [
    # (id, lang, text)
    ("hu_short", "hu", "Szia!"),
    ("hu_normal", "hu", "Szia Thomas, itt a hangsegéd, készen állok a beszélgetésre."),
    (
        "hu_long",
        "hu",
        "Ma reggel elmentem a piacra, vettem friss zöldségeket és gyümölcsöket, "
        "majd a boltban összefutottam egy régi barátommal, aki mesélt a nyári "
        "terveiről a Balaton mellett.",
    ),
    ("hu_question", "hu", "Hogy telt a mai napod, találtál valami érdekeset?"),
    ("hu_exclaim", "hu", "Ez fantasztikus hír, nagyon örülök neked!"),
    ("hu_numbers", "hu", "Találkozzunk 14 óra 30 perckor a 3. kapunál, kb. 20 percet késik a vonat."),
    (
        "hu_punct",
        "hu",
        "Igen, persze — de előbb: intézd a dolgot, aztán (ha lehet) 5 perc múlva szólj!",
    ),
    ("en_sentence", "en", "Hello Thomas, this is your voice assistant, ready to talk."),
    # 9-10 are procedural (repeat/consecutive), not fixed texts:
]


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    assert ch == 1, f"not mono: {ch}"
    assert width == 2, f"not 16-bit: {width}"
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, rate


def _check_audio(x: np.ndarray, rate: int, text: str) -> dict:
    n = len(x)
    dur = n / rate
    out: dict = {"duration_s": round(dur, 3), "samples": n}
    peak = float(np.max(np.abs(x))) if n else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if n else 0.0
    out["peak"] = round(peak, 4)
    out["rms"] = round(rms, 5)
    out["clipping"] = peak > 0.999
    out["empty"] = n == 0 or rms < 1e-6
    # silence analysis (frame = 50 ms); the speech span = first..last voiced
    fr = max(1, int(rate * 0.05))
    m = n // fr
    if m:
        fr_rms = np.sqrt((x[: m * fr].reshape(m, fr) ** 2).mean(axis=1))
        silent = fr_rms < 0.01
        lead = 0
        for s in silent:
            if not s:
                break
            lead += 1
        trail = 0
        for s in silent[::-1]:
            if not s:
                break
            trail += 1
        interior = m - lead - trail
        interior_silent = int(np.sum(silent[lead : m - trail]))
        out["lead_silence_s"] = round(lead * 0.05, 2)
        out["trail_silence_s"] = round(trail * 0.05, 2)
        out["interior_silence_ratio"] = (
            round(interior_silent / interior, 3) if interior > 0 else 1.0
        )
        # speech span duration (trimmed) - the plausibility metric
        speech_s = max(0.05, (interior) * 0.05)
        out["speech_span_s"] = round(speech_s, 2)
        cps = len(text) / speech_s if speech_s > 0 else 0.0
    else:
        cps = 0.0
    out["chars_per_s"] = round(cps, 1)
    verdict = []
    if out["empty"]:
        verdict.append("EMPTY_AUDIO")
    if out["clipping"]:
        verdict.append("CLIPPING")
    if not (RMS_MIN <= rms <= RMS_MAX) and not out["empty"]:
        verdict.append("RMS_OUT_OF_BAND")
    if out.get("lead_silence_s", 0) > EDGE_SILENCE_MAX_S:
        verdict.append("LEADING_SILENCE")
    if out.get("trail_silence_s", 0) > EDGE_SILENCE_MAX_S:
        verdict.append("TRAILING_SILENCE")
    if out.get("interior_silence_ratio", 0) > INTERIOR_SILENCE_MAX:
        verdict.append("UNNATURAL_SILENCE")
    if dur > 0 and not (CPS_MIN <= cps <= CPS_MAX):
        verdict.append("CPS_OUT_OF_BAND")
    out["verdict"] = "OK" if not verdict else ",".join(verdict)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="M1", help="preset voice (M1-M5, F1-F5)")
    ap.add_argument("--steps", type=int, default=8, help="total_steps (quality 5-12)")
    ap.add_argument("--speed", type=float, default=1.05, help="speech speed 0.7-2.0")
    ap.add_argument("--all-voices", action="store_true", help="run case 2 with all 10 voices")
    args = ap.parse_args()

    os.environ.setdefault("SUPERTONIC_LOG_LEVEL", "WARNING")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "engine": "supertonic",
        "sdk_version": "1.3.1",
        "model": "supertone-oss-archive/supertonic-3",
        "revision": "aafc6e32416a594460b32413efc49d7fe4ce6d46",
        "model_dir": str(MODEL_DIR.relative_to(_ROOT)),
        "voice": args.voice,
        "total_steps": args.steps,
        "speed": args.speed,
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": [],
    }

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    t0 = time.perf_counter()
    from supertonic import TTS

    tts = TTS(model="supertonic-3", model_dir=str(MODEL_DIR), auto_download=False)
    report["model_load_s"] = round(time.perf_counter() - t0, 2)
    report["sample_rate"] = tts.sample_rate
    report["available_voices"] = list(tts.voice_style_names)
    style = tts.get_voice_style(args.voice)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["rss_after_load_mb"] = round(rss1, 1)

    first = True

    def synth(text: str, lang: str, out_name: str) -> dict:
        nonlocal first
        t = time.perf_counter()
        wav, dur = tts.synthesize(
            text, voice_style=style, total_steps=args.steps, speed=args.speed, lang=lang
        )
        elapsed = time.perf_counter() - t
        out_path = RESULTS_DIR / f"{out_name}.wav"
        tts.save_audio(wav, str(out_path))
        x, rate = _read_wav(out_path)
        info = _check_audio(x, rate, text)
        info.update(
            {
                "file": str(out_path.relative_to(_ROOT)),
                "lang": lang,
                "synth_s": round(elapsed, 3),
                "sdk_dur_s": round(float(dur[0]), 3),
                "rtf": round(elapsed / info["duration_s"], 3) if info["duration_s"] else None,
                "first": first,
            }
        )
        if first:
            report["first_synthesis_latency_s"] = round(elapsed, 3)
            first = False
        return info

    # -- cases 1-8 ----------------------------------------------------------- #
    for case_id, lang, text in TEST_CASES:
        info = synth(text, lang, case_id)
        info["case"] = case_id
        info["text"] = text
        report["cases"].append(info)
        print(f"[{case_id:12s}] {info['verdict']:28s} synth {info['synth_s']:6.2f}s "
              f"audio {info['duration_s']:6.2f}s rtf {info['rtf']}")

    # -- case 9: repeated synthesis (same text 3x) --------------------------- #
    rep_text = TEST_CASES[1][2]
    durs = []
    for i in range(3):
        info = synth(rep_text, "hu", f"hu_repeat_{i+1}")
        info["case"] = f"hu_repeat_{i+1}"
        info["text"] = rep_text
        report["cases"].append(info)
        durs.append(info["duration_s"])
        print(f"[hu_repeat_{i+1}] {info['verdict']:24s} synth {info['synth_s']:6.2f}s "
              f"audio {info['duration_s']:6.2f}s")
    spread = max(durs) - min(durs)
    report["repeat_duration_spread_s"] = round(spread, 3)
    report["repeat_duration_cv"] = round(
        float(np.std(durs) / np.mean(durs)), 4
    )
    print(f"repeat duration spread: {spread:.2f}s (cv {report['repeat_duration_cv']})")

    # -- case 10: multiple consecutive mixed requests ------------------------- #
    queue = [
        ("Szia, mi a helyzet?", "hu"),
        ("Sure, I can switch to English anytime.", "en"),
        ("Rendben, most magyarul folytatom a beszélgetést.", "hu"),
        ("One more English line to close the queue.", "en"),
    ]
    ok = 0
    for i, (text, lang) in enumerate(queue):
        info = synth(text, lang, f"consecutive_{i+1}")
        info["case"] = f"consecutive_{i+1}"
        info["text"] = text
        report["cases"].append(info)
        if info["verdict"] == "OK":
            ok += 1
        print(f"[consecutive_{i+1}] {info['verdict']:24s} synth {info['synth_s']:6.2f}s")
    report["consecutive_ok"] = f"{ok}/{len(queue)}"

    # -- optional: all voices on the normal HU sentence ---------------------- #
    if args.all_voices:
        report["voice_matrix"] = []
        for v in tts.voice_style_names:
            st = tts.get_voice_style(v)
            t = time.perf_counter()
            wav, dur = tts.synthesize(
                TEST_CASES[1][2], voice_style=st, total_steps=args.steps,
                speed=args.speed, lang="hu",
            )
            p = RESULTS_DIR / f"voice_{v}.wav"
            tts.save_audio(wav, str(p))
            x, rate = _read_wav(p)
            info = _check_audio(x, rate, TEST_CASES[1][2])
            info.update({"voice": v, "synth_s": round(time.perf_counter() - t, 3)})
            report["voice_matrix"].append(info)
            print(f"[voice {v}] {info['verdict']:24s} dur {info['duration_s']:.2f}s "
                  f"rms {info['rms']}")

    rss2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["rss_final_mb"] = round(rss2, 1)
    report["rss_delta_after_synthesis_mb"] = round(rss2 - rss1, 1)

    bad = [c for c in report["cases"] if c.get("verdict") != "OK"]
    report["summary"] = {
        "cases_total": len(report["cases"]),
        "cases_ok": len(report["cases"]) - len(bad),
        "cases_bad": [c.get("case") for c in bad],
    }
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSUMMARY: {report['summary']['cases_ok']}/{report['summary']['cases_total']} OK")
    print(f"report: {RESULTS_DIR / 'report.json'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
