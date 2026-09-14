"""v0.7.0 TTS benchmark (Supertonic 3 on the TARGET machine).

Replaces the M1-era Piper benchmark: the production TTS engine is
``app.tts_supertonic.SupertonicTtsEngine`` (ONNX Runtime, CPU — the RTX
5070 stays reserved for Qwen3.6). Importing is safe anywhere:
``app.tts_supertonic`` is imported lazily inside main().

Method:
  1. Synthesize 5 Hungarian + 5 English sentences through the PRODUCTION
     adapter (voice: the configured default, F1).
  2. Measure TTFB (call -> PCM ready, INCLUDING the one-time lazy model
     load on the first sentence) and RTF (synthesis time / synthesized
     audio duration) per sentence with time.perf_counter. The second and
     later sentences measure the warm-engine latency.
  3. Record model load time, warm TTFB, RTF, audio seconds, RSS delta.
  4. Print a table and MOS listening-test instructions (3 native
     listeners, 1-5 scale) — the acceptance criteria include SPEECH
     QUALITY, not only latency.

Exit criteria: every synthesis succeeded (10/10 non-empty int16 PCM).
  -> exit 0 on success, 1 on failures, 2 on missing prerequisites.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

REPO_ROOT = Path(__file__).resolve().parents[2]

# 5 HU + 5 EN benchmark sentences (M1 set preserved — same milestones-
# critical cases: number pronunciation "1890", name "Hojsz Tamás",
# code-switching "flat white" inside Hungarian).
HU_SENTENCES = [
    "Jó reggelt! Ma egy új nyelvleckével kezdjük a napot.",
    "A flat white és a tejeskávé egyaránt espresso-alapú italok tejjel.",
    "1890-ben Hojsz Tamás nevű tudós készítette el az első változatot.",
    "Kérem, ismételje meg lassabban a harmadik mondatot.",
    "A magyar nyelv gazdag magánhangzórendszerrel rendelkezik: ő, ű, á, é, í, ó, ö, ü.",
]

EN_SENTENCES = [
    "Good morning! Let us start today's English lesson.",
    "I have gone to the library every morning this week.",
    "The year 1890 was important in the history of coffee.",
    "Would you like a flat white or a latte?",
    "Please repeat the third sentence more slowly.",
]


def print_mos_instructions(out_dir: Path) -> None:
    """Manual listening test protocol (milestones doc Section 6.6.2)."""
    print("-" * 72)
    print("MOS listening test (manual, after the synthesises):")
    print(f"  1. Play the WAV files in {out_dir} in order (hu_*.wav, then en_*.wav).")
    print("  2. Three native listeners: 2 Hungarian + 1 English (1-5 scale each file).")
    print("  3. Rate overall naturalness (MOS), pronunciation, and explicitly:")
    print("     - the number '1890' (HU: 'ezerkilencszázkilencven', EN: 'eighteen ninety')")
    print("     - the name 'Hojsz Tamás'")
    print("     - code-switching: 'flat white' inside a Hungarian sentence")
    print("     - word repetitions / skipped words / truncation (any occurrence = FAIL)")
    print("  4. Exit metric: MOS >= 3.0 (acceptable), number/name pronunciation 100%.")
    print("  5. Record results in the worklog (per file, per listener).")


def main() -> int:
    parser = argparse.ArgumentParser(description="v0.7.0 TTS benchmark (Supertonic 3)")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "benchmark_results" / "tts"),
        help="output directory for the synthesized WAV files",
    )
    parser.add_argument("--config", help="optional YAML config (config/voicemem_config.yaml)")
    parser.add_argument("--voice", help="optional preset override (F1..F5, M1..M5)")
    args = parser.parse_args()

    from app.config import AgentConfig

    cfg = AgentConfig()
    cfg.apply_env()
    if args.config and Path(args.config).exists():
        try:
            cfg = AgentConfig.from_yaml(Path(args.config))
        except (TypeError, ValueError, RuntimeError) as exc:
            print(f"WARN: YAML config load failed ({exc}); using env + defaults")

    try:
        from app.tts_supertonic import SupertonicTtsEngine
    except ImportError:
        print("app.tts_supertonic is not importable — run from the repo root.")
        return 2

    engine = SupertonicTtsEngine(cfg)
    if not engine.is_available():
        print("SupertonicTtsEngine unavailable: model assets missing.")
        print(f"  model dir: {cfg.supertonic_model_dir}")
        print("  run scripts/download_models.ps1 (setup-time download).")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [("hu", text) for text in HU_SENTENCES] + [("en", text) for text in EN_SENTENCES]
    rows: list[dict[str, Any]] = []
    failures = 0
    import wave

    for index, (lang, text) in enumerate(cases, start=1):
        out_path = out_dir / f"{lang}_{index:02d}.wav"
        t0 = time.perf_counter()
        ok = engine.synthesize_to_file(text, lang, out_path, voice=args.voice)
        elapsed = time.perf_counter() - t0
        duration = 0.0
        if ok:
            with wave.open(str(out_path), "rb") as w:
                duration = w.getnframes() / float(w.getframerate())
        rtf = round(elapsed / duration, 3) if duration > 0 else None
        if not ok:
            failures += 1
        rows.append(
            {
                "file": out_path.name,
                "lang": lang,
                "chars": len(text),
                "ttfb_s": round(elapsed, 3),
                "audio_s": round(duration, 3),
                "rtf": rtf,
                "ok": ok,
                "text": text,
            }
        )

    print("=" * 72)
    print(f"v0.7.0 TTS benchmark (Supertonic 3, voice {args.voice or cfg.tts_hu_voice},"
          f" steps {cfg.supertonic_steps}, speed {cfg.supertonic_speed})")
    print("=" * 72)
    print(f"{'file':12s} {'lang':5s} {'chars':6s} {'TTFB s':8s} {'audio s':8s} {'RTF':7s} status")
    for row in rows:
        status = "OK" if row["ok"] else "FAILED"
        rtf_txt = f"{row['rtf']:.3f}" if row["rtf"] is not None else "n/a"
        print(
            f"{row['file']:12s} {row['lang']:5s} {row['chars']:<6d} "
            f"{row['ttfb_s']:<8.3f} {row['audio_s']:<8.3f} {rtf_txt:7s} {status}"
        )
    print("-" * 72)
    print("Notes:")
    print("  The FIRST row includes the one-time lazy model load (engine stays")
    print("  alive for the process lifetime); later rows are warm-engine calls.")
    print(f"  Model load: {engine.load_s if engine.load_s is not None else 'n/a'} s"
          if engine.load_s else "  Model load: n/a (engine not loaded)")
    print(f"  Model dir: {cfg.supertonic_model_dir}")
    print(f"  Engine: {engine.status_detail()}")

    print_mos_instructions(out_dir)

    print("=" * 72)
    total = len(cases)
    passed = total - failures
    print(f"RESULT: {passed}/{total} syntheses succeeded -> {'PASS' if failures == 0 else 'FAIL'}")
    if failures > 0:
        for row in rows:
            if not row["ok"]:
                print(f"  FAILED: {row['file']} :: {row['text']}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
