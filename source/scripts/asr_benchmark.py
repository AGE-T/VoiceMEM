"""ASR engine benchmark (TASK-A Phase 5/15): reproducible model comparison.

Runs ONE engine over the known-reference corpus (data/asr_bench) and
measures, per file and aggregate:

* WER / CER (Levenshtein, with corpus number-word normalisation)
* model load time (cold), first-result latency, final latency, RTF
* empty-output count, repetition/hallucination indicators, stability
  (two passes -> identical transcripts?)
* engine memory footprint (process RSS delta after load)

The harness is engine-agnostic: it drives the engine through the PRODUCTION
contract (app.asr_core select_engine -> load -> transcribe), so the numbers
reflect the integrated path, not a notebook shortcut. Streaming engines are
additionally exercised through the real start/feed/finish interface
(chunked, cache-aware) with the same corpus — the report marks
``streaming: true`` results separately and never fakes them.

Usage:
    python scripts/asr_benchmark.py --engine parakeet [--json OUT.json]
    python scripts/asr_benchmark.py --engine nemotron --device cpu
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from app.asr_core import AsrResult, AsrResultStatus, AudioBuffer, select_engine  # noqa: E402
from app.config import AgentConfig  # noqa: E402

BENCH_DIR = _ROOT / "data" / "asr_bench"

#: Corpus number words -> digits (both sides of the comparison normalised
#: through this map; the corpus deliberately contains spoken-form numbers).
_NUMBER_WORDS = {
    "száznegyvenkét": "142",
    "száznegyvenkettő": "142",
    "hétszázharminc": "730",
    "tizenkilenc": "19",
    "kettőszáznegyven": "270",
    "kétszázhetven": "270",
}


def _norm_text(text: str) -> list[str]:
    """Normalise for WER: lowercase, strip punctuation, number words->digits."""
    t = text.lower()
    t = t.replace("‑", "-")
    t = re.sub(r"[,.!?;:()\[\]„”\"'-]", " ", t)
    words = [w for w in t.split() if w]
    return [_NUMBER_WORDS.get(w, w) for w in words]


def _norm_chars(text: str) -> list[str]:
    """Normalise for CER: lowercase letters/digits only."""
    t = text.lower()
    return [c for c in t if c.isalnum()]


def wer(ref: str, hyp: str) -> float:
    """Word error rate (Levenshtein over normalised words)."""
    r, h = _norm_text(ref), _norm_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len(h) + 1):
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return dp[len(h)] / float(len(r))


def cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein over normalised chars)."""
    r, h = _norm_chars(ref), _norm_chars(hyp)
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len(h) + 1):
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return dp[len(h)] / float(len(r))


def repetition_score(text: str) -> float:
    """Repetition/hallucination indicator: max token-run ratio.

    0 = no repeated runs; the Qwen failure mode ("让让让让让。") scores ~0.8+
    (5 identical tokens out of 6). Words repeated >= 3 times consecutively
    dominate the score.
    """
    words = _norm_text(text)
    if not words:
        return 0.0
    best_run, cur = 1, 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            cur += 1
            best_run = max(best_run, cur)
        else:
            cur = 1
    chars = _norm_chars(text)
    best_char_run, cc = 1, 1
    for i in range(1, len(chars)):
        if chars[i] == chars[i - 1]:
            cc += 1
            best_char_run = max(best_char_run, cc)
        else:
            cc = 1
    word_score = best_run / float(len(words))
    char_score = best_char_run / float(max(len(chars), 1))
    return max(word_score, char_score * 0.5)


def _rss_mb() -> float:
    try:
        return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        return -1.0


def run_benchmark(engine_id: str, device: str, runs: int = 2) -> dict:
    """Benchmark one engine end-to-end through the production contract."""
    cfg = AgentConfig.from_yaml(_ROOT / "config" / "voicemem_config.yaml")
    cfg.asr_device = device
    cfg.asr_engine = engine_id
    manifest = json.loads((BENCH_DIR / "manifest.json").read_text(encoding="utf-8"))

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    engine = select_engine(cfg)
    engine.load()
    load_s = time.perf_counter() - t0
    rss_after_load = _rss_mb()

    files_report: list[dict] = []
    pass_texts: dict[str, list[str]] = {}
    for run_idx in range(runs):
        for fid, meta in manifest["files"].items():
            buf = AudioBuffer.from_wav(BENCH_DIR / meta["wav"])
            t1 = time.perf_counter()
            result: AsrResult = engine.transcribe(buf)
            dt = time.perf_counter() - t1
            pass_texts.setdefault(fid, []).append(result.text)
            if run_idx == 0:
                files_report.append(
                    {
                        "file": fid,
                        "language": meta["language"],
                        "duration_s": round(buf.duration_s, 3),
                        "transcript": result.text,
                        "reference": meta["text"],
                        "wer": round(wer(meta["text"], result.text), 4),
                        "cer": round(cer(meta["text"], result.text), 4),
                        "empty": not result.text.strip(),
                        "repetition": round(repetition_score(result.text), 3),
                        "latency_s": round(dt, 3),
                        "rtf": round(dt / max(buf.duration_s, 1e-6), 3),
                        "status": result.status.value,
                        "error": result.error.to_dict() if result.error else None,
                    }
                )

    # first-result latency = the FIRST file's wall time (cold inference)
    first_latency = files_report[0]["latency_s"] if files_report else None
    total_audio = sum(f["duration_s"] for f in files_report)
    total_time = sum(f["latency_s"] for f in files_report)
    # streaming exercise (only engines with a genuine streaming interface)
    streaming_report: dict = {"supported": bool(engine.capabilities.streaming)}
    if engine.capabilities.streaming:
        try:
            streaming_report.update(_streaming_smoke(engine, manifest))
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            streaming_report["error"] = f"{type(exc).__name__}: {exc}"

    hu = [f for f in files_report if f["language"] == "hu" and f["status"] == "ok"]
    en = [f for f in files_report if f["language"] == "en" and f["status"] == "ok"]
    return {
        "engine": engine_id,
        "model": engine.model_id,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "load_s": round(load_s, 2),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_load_mb": round(rss_after_load, 1),
        "rss_delta_mb": round(rss_after_load - rss_before, 1),
        "first_result_latency_s": first_latency,
        "total_audio_s": round(total_audio, 2),
        "total_inference_s": round(total_time, 2),
        "aggregate_rtf": round(total_time / max(total_audio, 1e-6), 3),
        "wer_hu_mean": round(float(np.mean([f["wer"] for f in hu])), 4) if hu else None,
        "cer_hu_mean": round(float(np.mean([f["cer"] for f in hu])), 4) if hu else None,
        "wer_en": round(float(np.mean([f["wer"] for f in en])), 4) if en else None,
        "empty_outputs": sum(1 for f in files_report if f["empty"]),
        "repetition_max": max((f["repetition"] for f in files_report), default=0.0),
        "stability_identical": sum(
            1 for fid, texts in pass_texts.items() if len(set(texts)) == 1
        ),
        "stability_files": len(pass_texts),
        "files": files_report,
        "streaming": streaming_report,
    }


def _streaming_smoke(engine, manifest: dict) -> dict:
    """Exercise the REAL start/feed/finish streaming path on one corpus file.

    Chunks are the VAD frame size (512 @ 16 kHz = 32 ms) fed at REAL-TIME
    pacing (the production cadence) — partial transcripts then surface the
    way they do live. The report records partial emission and the final
    transcript vs the offline reference.
    """
    meta = manifest["files"]["hu_with_silence"]  # shortest file: CPU RTF > 1
    buf = AudioBuffer.from_wav(BENCH_DIR / meta["wav"])
    x = buf.samples
    frame = 512
    frame_s = frame / 16000.0
    t0 = time.perf_counter()
    engine.start()
    partials = 0
    first_partial_s = None
    next_deadline = t0
    for i in range(0, x.size, frame):
        # real-time pacing: the VAD loop delivers one frame every 32 ms
        now = time.perf_counter()
        if now < next_deadline:
            time.sleep(next_deadline - now)
        next_deadline += frame_s
        chunk = AudioBuffer.from_float(x[i : i + frame], 16000)
        partial = engine.feed(chunk)
        if partial is not None and partial.text:
            partials += 1
            if first_partial_s is None:
                first_partial_s = time.perf_counter() - t0
    result = engine.finish()
    total_s = time.perf_counter() - t0
    return {
        "file": "hu_with_silence",
        "feed_frame_samples": frame,
        "real_time_paced": True,
        "partials_emitted": partials,
        "first_partial_s": round(first_partial_s, 3) if first_partial_s else None,
        "final_text": result.text,
        "wer_vs_reference": round(wer(meta["text"], result.text), 4),
        "total_s": round(total_s, 3),
        "rtf": round(total_s / max(buf.duration_s, 1e-6), 3),
        "status": result.status.value,
        "error": result.error.to_dict() if result.error else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, help="engine id (parakeet|nemotron)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (execution config)")
    ap.add_argument("--runs", type=int, default=2, help="stability passes")
    ap.add_argument("--json", default="", help="output JSON path")
    args = ap.parse_args()

    report = run_benchmark(args.engine, args.device, runs=args.runs)
    print(f"══ {report['engine']} on {report['device']} ({report['model']}) ══")
    print(
        f"  load {report['load_s']}s · RSS +{report['rss_delta_mb']} MB · "
        f"first result {report['first_result_latency_s']}s · "
        f"aggregate RTF {report['aggregate_rtf']}"
    )
    print(
        f"  HU WER mean {report['wer_hu_mean']} · CER mean {report['cer_hu_mean']} · "
        f"EN WER {report['wer_en']} · empty {report['empty_outputs']} · "
        f"repetition max {report['repetition_max']} · "
        f"stability {report['stability_identical']}/{report['stability_files']}"
    )
    for f in report["files"]:
        print(
            f"    {f['file']:22s} WER {f['wer']*100:5.1f}% CER {f['cer']*100:5.1f}% "
            f"lat {f['latency_s']:5.2f}s RTF {f['rtf']:5.2f} "
            f"{'EMPTY' if f['empty'] else ''}"
        )
        print(f"      HYP: {f['transcript'][:100]}")
    if report["streaming"].get("supported"):
        s = report["streaming"]
        print(
            f"  STREAMING (real interface): partials {s.get('partials_emitted')} "
            f"first {s.get('first_partial_s')}s · WER {s.get('wer_vs_reference')} · "
            f"offline-match {s.get('matches_offline')}"
        )
        if "error" in s:
            print(f"  STREAMING ERROR: {s['error']}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
