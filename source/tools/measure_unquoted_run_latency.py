"""Engine-level latency measurement: unquoted phrase runs, old vs new.

Measures on the REAL Supertonic 3 ONNX engine (CPU), same voice (F1) for
every span, exactly as the production callers drive it:

* OLD (v0.10.5): one synthesis call per chunk with the HOST language —
  the whole unquoted English phrase read by the Hungarian model (the
  production bug; kept here as the latency baseline).
* NEW: per-span synthesis calls from ``segment_language_spans`` (host
  prefix, foreign phrase, host suffix), PCM-concatenated into one
  payload — the shipped v0.10.5 behaviour generalized to unquoted runs.

Records per case: span count, synthesis calls, per-call synthesis
latency, total TTS latency, first-audio latency (first span PCM), total
PCM samples (playback continuity proxy — the concatenated payload is
exactly what the browser receives as ONE chunk) and RMS (non-silence).

Run (repo root, sandbox venv):
  python tools/measure_unquoted_run_latency.py
Writes: audit/VoiceMEM_tts_codeswitch_v0105_unquoted/evidence/
        synthesis_latency_20260918.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from app.config import AgentConfig  # noqa: E402
from app.text_utils import (  # noqa: E402
    LANG_EN,
    LANG_HU,
    detect_language,
    normalize_for_speech,
    segment_language_spans,
)
from app.tts_supertonic import SupertonicTtsEngine  # noqa: E402

OUT = (
    _ROOT / "audit" / "VoiceMEM_tts_codeswitch_v0105_unquoted"
    / "evidence" / "synthesis_latency_20260918.txt"
)

CASES = [
    ("F2 touch base-t", "Szeretnék egy gyors touch base-t veld."),
    ("F3 catch up-olhatunk", "Szerintem később catch up-olhatunk."),
    ("F4 see you later", "Rendben, see you later."),
    ("E1 pure EN (control)", "I'd like to touch base with you."),
    ("D1 pure HU (control)", "Szeretnék röviden beszélni veled az új projektről."),
]


def _rms(pcm: np.ndarray) -> float:
    if not len(pcm):
        return 0.0
    x = pcm.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


def main() -> int:
    lines: list[str] = []
    cfg = AgentConfig(root=_ROOT)
    engine = SupertonicTtsEngine(cfg)
    if not engine.is_available():
        print("supertonic assets unavailable — cannot measure")
        return 1
    lines.append("Unquoted phrase-run synthesis latency (real Supertonic 3, CPU, F1)")
    lines.append("OLD = single host call (v0.10.5 behaviour, the reported bug)")
    lines.append("NEW = per-span calls via segment_language_spans, PCM concat")
    lines.append("")
    hdr = (
        f"{'case':26s} {'mode':4s} {'spans':>5s} {'calls':>5s} "
        f"{'first_audio_ms':>14s} {'total_ms':>9s} {'pcm_ms':>7s} {'rms':>6s}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    # warm the engine once (model load) so per-case numbers are steady-state
    t0 = time.perf_counter()
    engine.synthesize("Bemelegítés.", LANG_HU)
    lines.append(f"engine warmup (load + 1 synth): {time.perf_counter() - t0:.2f} s")
    lines.append("")
    for name, sentence in CASES:
        chunk = normalize_for_speech(sentence)
        host = detect_language(chunk)
        spans = segment_language_spans(chunk, host)
        for mode in ("old", "new"):
            parts: list[np.ndarray] = []
            call_ms: list[float] = []
            first_audio = None
            t_start = time.perf_counter()
            if mode == "old":
                t = time.perf_counter()
                pcm = engine.synthesize(chunk, host, None, "F1")
                call_ms.append((time.perf_counter() - t) * 1000.0)
                first_audio = call_ms[0]
                if pcm is not None and len(pcm):
                    parts.append(pcm)
            else:
                for span_text, span_lang in spans:
                    t = time.perf_counter()
                    pcm = engine.synthesize(span_text, span_lang, None, "F1")
                    ms = (time.perf_counter() - t) * 1000.0
                    call_ms.append(ms)
                    if first_audio is None and pcm is not None and len(pcm):
                        first_audio = ms
                    if pcm is not None and len(pcm):
                        parts.append(pcm)
            total_ms = (time.perf_counter() - t_start) * 1000.0
            joined = parts[0] if len(parts) == 1 else (
                np.concatenate(parts) if parts else np.array([])
            )
            pcm_ms = len(joined) / cfg.output_sample_rate * 1000.0
            n_calls = 1 if mode == "old" else len(spans)
            lines.append(
                f"{name:26s} {mode:4s} {len(spans) if mode == 'new' else 1:>5d} "
                f"{n_calls:>5d} {first_audio if first_audio is not None else -1:>14.1f} "
                f"{total_ms:>9.1f} {pcm_ms:>7.1f} {_rms(joined):>6.3f}"
            )
        lines.append(
            f"    calls[ms]: " + ", ".join(f"{m:.0f}" for m in call_ms)
        )
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
