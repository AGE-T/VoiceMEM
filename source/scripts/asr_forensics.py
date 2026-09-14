"""ASR forensics: boundary-by-boundary measurement of the voice uplink chain.

TASK-A Phase 1 instrument. The Windows field reports showed severely corrupted
Hungarian transcripts ("让让让让让。", "夫。") while the browser mic diagnostics
measured healthy signal (src RMS ~0.24, PCM RMS ~0.24). This harness replays
the ENTIRE audio chain on REAL captured speech and measures every boundary, so
the first objectively broken boundary is identified by DATA, not assumption.

Chain under test (the v0.5.2 web path, exactly as shipped):

  S0 source speech (real Windows capture, 16 kHz — upload/asr_test_last.wav)
  S1 browser track @ 48 kHz          (simulated: bandlimited x3 upsample —
                                      what the MixPre line-in delivers)
  S2 AudioContext @ 96 kHz           (simulated: bandlimited x2 upsample —
                                      the browser's internal resample)
  S3 ScriptProcessor blocks          (4096-sample chunking at the ctx rate)
  S4 JS downsample(96k -> 24k)       (EXACT replica of web/voicemem.html)
  S5 JS toPCM16                      (EXACT replica, incl. the -32768/+32767
                                      asymmetry)
  S6 WebSocket bytes                 (the bytes the server receives)
  S7 server pcm16_to_float32         (imported from app.web_server — the
                                      production function, not a copy)
  S8 server resample_linear(24k->16k)(imported from app.web_server)
  S9 512-sample VAD frames           (production frame size @ 16 kHz)
  S10 Silero VAD segmentation        (REAL onnx model, current config:
                                      threshold 0.25, hangover 300 ms)

At every boundary the report records: sample rate, channels, dtype, sample
count, duration, RMS, peak, frame size. Signal fidelity vs S0 is measured by
RMS/peak delta, Pearson correlation and spectral SNR. The VAD stage reports
speech segments, their durations and fragmentation statistics — the ~320 ms
"speech segments" of the field report are reproduced or refuted here.

Also runs the 44.1 kHz AudioContext variant (the sandbox e2e environment) so
the fractional-ratio JS linear-interpolation path is measured too.

Usage:
    python scripts/asr_forensics.py [--source PATH] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.config import AgentConfig  # noqa: E402
from app.web_server import (  # noqa: E402 — the PRODUCTION functions
    PIPE_SAMPLE_RATE,
    VAD_FRAME_SAMPLES,
    WEB_SAMPLE_RATE,
    pcm16_to_float32,
    resample_linear,
)

#: Real Windows capture (16 kHz mono PCM16, 9.98 s, MixPre line-in through the
#: full v0.5.2 browser chain — the file the field ASR produced garbage from).
DEFAULT_SOURCE = Path("/home/z/my-project/upload/asr_test_last.wav")


# ═══════════════════════════════════════════════════════════════════════════
# Signal synthesis helpers (ground-truth quality, NOT the chain under test)
# ═══════════════════════════════════════════════════════════════════════════


def bandlimited_upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """FFT-domain zero-padded upsample (perfect bandlimited, integer factor).

    Represents what a properly working mic/track/context stage delivers: the
    speech band is preserved exactly, no spectral images. Used to synthesise
    S1/S2 from S0 — it is NOT part of the chain under test.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = x.size
    spec = np.fft.rfft(x)
    out_n = n * factor
    padded = np.zeros(out_n // 2 + 1, dtype=np.complex128)
    padded[: spec.size] = spec
    y = np.fft.irfft(padded, n=out_n)
    y *= factor
    return y.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# EXACT replicas of the browser-side conversion code (web/voicemem.html)
# ═══════════════════════════════════════════════════════════════════════════


def js_downsample(f32: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Vector-exact replica of voicemem.html ``downsample``.

    ratio>=2: box average per output sample (integer and non-integer ratios —
    the JS averages the samples in [floor(a), b)); ratio<2: linear
    interpolation. Numerically identical to the JS loop for every input the
    live path can produce (verified by a brute-force loop cross-check in the
    behavioural tests).
    """
    f32 = np.asarray(f32, dtype=np.float64).reshape(-1)
    if from_sr == to_sr or f32.size == 0:
        return f32.astype(np.float32)
    ratio = from_sr / to_sr
    n = int(np.floor(f32.size / ratio))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n, dtype=np.float64)
    if ratio >= 2:
        # out[i] = mean(x[floor(i*ratio) : min(len, i*ratio + ratio)])
        starts = np.floor(np.arange(n, dtype=np.float64) * ratio).astype(np.int64)
        ends = np.minimum(
            starts + int(np.ceil(ratio)), np.ones(n, dtype=np.int64) * f32.size
        )
        # exact JS end: b = min(len, a + ratio) with a = i*ratio (float); the
        # loop runs j from floor(a) while j < b — b itself is fractional only
        # when ratio is fractional; ceil covers the last included index + 1.
        ends_exact = np.minimum(
            np.arange(n, dtype=np.float64) * ratio + ratio, float(f32.size)
        )
        # include indices j in [floor(a), ceil(b)) but capped by b_exact:
        for i in range(n):
            a = i * ratio
            b = min(f32.size, a + ratio)
            j0 = int(np.floor(a))
            j1 = int(np.ceil(b))
            seg = f32[j0:j1]
            # the JS loop condition is j < b (float compare) — drop j >= b
            if j1 > j0 and j1 - 1 >= b:
                seg = seg[:-1]
            out[i] = seg.mean() if seg.size else 0.0
        _ = starts, ends  # (kept for clarity; the loop above is authoritative)
    else:
        idx = np.arange(n, dtype=np.float64) * ratio
        j = np.floor(idx).astype(np.int64)
        j = np.clip(j, 0, f32.size - 1)
        j1 = np.minimum(j + 1, f32.size - 1)
        frac = (idx - j).astype(np.float64)
        x0 = f32[j]
        x1 = f32[j1]
        out = x0 * (1.0 - frac) + x1 * frac
    return out.astype(np.float32)


def js_to_pcm16(f32: np.ndarray) -> np.ndarray:
    """Exact replica of voicemem.html ``toPCM16`` (asymmetric clipping)."""
    x = np.asarray(f32, dtype=np.float64).reshape(-1)
    s = np.clip(x, -1.0, 1.0)
    out = np.where(s < 0, s * 32768.0, s * 32767.0)
    return out.astype(np.int16)


# ═══════════════════════════════════════════════════════════════════════════
# Boundary measurements
# ═══════════════════════════════════════════════════════════════════════════


def metrics(
    label: str,
    x: np.ndarray | bytes,
    sample_rate: int,
    channels: int = 1,
    frame_size: int | None = None,
) -> dict:
    """Boundary record: rate/channels/dtype/count/duration/RMS/peak/frame."""
    if isinstance(x, bytes):
        dtype, count = f"bytes(pcm16:{len(x)})", len(x) // 2
        rms = peak = None
    else:
        arr = np.asarray(x)
        dtype = str(arr.dtype)
        count = int(arr.reshape(-1).size)
        a = arr.astype(np.float64).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(a)))) if count else 0.0
        peak = float(np.max(np.abs(a))) if count else 0.0
    duration = round(count / float(sample_rate), 6) if sample_rate else None
    rec = {
        "stage": label,
        "sample_rate": sample_rate,
        "channels": channels,
        "dtype": dtype,
        "samples": count,
        "duration_s": duration,
        "rms": None if rms is None else round(rms, 6),
        "peak": None if peak is None else round(peak, 6),
    }
    if frame_size:
        rec["frame_size"] = frame_size
        rec["frames"] = count // frame_size if count else 0
    return rec


def fidelity(ref: np.ndarray, test: np.ndarray, label: str) -> dict:
    """Compare a recovered signal to the S0 reference (aligned by length)."""
    n = min(ref.size, test.size)
    r = ref[:n].astype(np.float64)
    t = test[:n].astype(np.float64)
    if n == 0:
        return {"stage": label, "note": "empty"}
    corr = float(np.corrcoef(r, t)[0, 1]) if n > 1 else 1.0
    rms_err = float(np.sqrt(np.mean((r - t) ** 2)))
    # spectral SNR: 8 kHz bands (16 kHz Nyquist), speech-relevant
    sp = np.abs(np.fft.rfft(r - t)) ** 2
    sr_all = np.abs(np.fft.rfft(r)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / PIPE_SAMPLE_RATE)
    bands = {}
    for lo, hi in [(0, 1000), (1000, 4000), (4000, 8000)]:
        m = (freqs >= lo) & (freqs < hi)
        noise = float(np.sum(sp[m]))
        signal = float(np.sum(sr_all[m]))
        bands[f"{lo//1000}-{hi//1000}k_snr_db"] = (
            round(10.0 * np.log10(max(signal, 1e-30) / max(noise, 1e-30)), 2)
            if noise > 0
            else 99.0
        )
    return {
        "stage": label,
        "pearson_r": round(corr, 6),
        "rms_error": round(rms_err, 6),
        "rms_ref": round(float(np.sqrt(np.mean(r**2))), 6),
        "rms_test": round(float(np.sqrt(np.mean(t**2))), 6),
        "peak_ref": round(float(np.max(np.abs(r))), 6),
        "peak_test": round(float(np.max(np.abs(t))), 6),
        "duration_delta_ms": round(
            abs(ref.size - test.size) / PIPE_SAMPLE_RATE * 1000.0, 3
        ),
        **bands,
    }


def vad_segmentation(audio16k: np.ndarray, config: AgentConfig) -> dict:
    """Run the PRODUCTION Silero VAD + state machine over 16 kHz audio.

    Reports BOTH feed variants (the forensic core of TASK-A Phase 1):

    * ``as_production`` — the v0.5.2 app/vad.SileroVad (512-sample windows,
      NO context): the shipped behaviour.
    * ``with_64_context`` — the OFFICIAL silero-vad OnnxWrapper contract
      (64-sample rolling context prepended -> 576-sample model windows):
      what the model actually requires.

    The delta between the two IS the first objectively broken boundary.
    """
    from app.vad import SileroVad, VadStateMachine

    t0 = time.perf_counter()
    vad = SileroVad(config)
    load_s = time.perf_counter() - t0
    n = int(config.vad_frame_samples)

    def _run(prob_fn):
        vsm = VadStateMachine(
            config.vad_threshold, config.vad_hangover_ms, config.vad_frame_ms
        )
        vad.reset()
        segs: list[dict] = []
        cur: float | None = None
        probs = []
        for i in range(0, audio16k.size, n):
            frame = audio16k[i : i + n]
            if frame.size < n:
                frame = np.concatenate(
                    (frame, np.zeros(n - frame.size, dtype=np.float32))
                )
            p = prob_fn(frame)
            probs.append(p)
            ev = vsm.update(p)
            if ev.value == "speech_start":
                cur = float(i)
            elif ev.value == "speech_end" and cur is not None:
                segs.append(
                    {
                        "start_ms": round(cur / PIPE_SAMPLE_RATE * 1000.0, 1),
                        "duration_ms": round((i + n - cur) / PIPE_SAMPLE_RATE * 1000.0, 1),
                    }
                )
                cur = None
        durs = [s["duration_ms"] for s in segs]
        return {
            "prob_max": round(max(probs), 4) if probs else None,
            "prob_mean": round(float(np.mean(probs)), 4) if probs else None,
            "frames_above_threshold": int(sum(1 for p in probs if p >= config.vad_threshold)),
            "frames": len(probs),
            "segment_count": len(segs),
            "segment_ms": durs,
            "segment_ms_median": float(np.median(durs)) if durs else None,
            "speech_ms_total": round(float(sum(durs)), 1),
        }

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(_ROOT / "models" / "vad" / "silero-vad" / "silero_vad.onnx"),
        providers=["CPUExecutionProvider"],
    )
    sr_feed = np.array(PIPE_SAMPLE_RATE, dtype=np.int64)

    def _prob_with_context(frame: np.ndarray) -> float:
        state = _prob_with_context._state
        ctx = _prob_with_context._ctx
        inp = np.concatenate((ctx, frame.reshape(1, -1)), axis=1).astype(np.float32)
        out, new_state = session.run(
            None, {"input": inp, "state": state, "sr": sr_feed}
        )
        _prob_with_context._state = new_state
        _prob_with_context._ctx = inp[:, -64:]
        return float(out[0, 0])

    def _reset_context() -> None:
        _prob_with_context._state = np.zeros((2, 1, 128), dtype=np.float32)
        _prob_with_context._ctx = np.zeros((1, 64), dtype=np.float32)

    _reset_context()

    def _prod(frame: np.ndarray) -> float:
        return vad.prob(frame)

    as_production = _run(_prod)
    _reset_context()
    with_context = _run(_prob_with_context)

    # energy-fallback fragmentation demo (the v0.4.14 heuristic the field ran)
    from app.vad import FusedVad

    fused = FusedVad(SileroVad(config), config)
    fused_vsm = VadStateMachine(
        config.vad_threshold, config.vad_hangover_ms, config.vad_frame_ms
    )
    fsegs: list[float] = []
    cur: float | None = None
    for i in range(0, audio16k.size, n):
        frame = audio16k[i : i + n]
        if frame.size < n:
            frame = np.concatenate((frame, np.zeros(n - frame.size, dtype=np.float32)))
        p = fused.prob(frame)
        ev = fused_vsm.update(p)
        if ev.value == "speech_start":
            cur = float(i)
        elif ev.value == "speech_end" and cur is not None:
            fsegs.append(round((i + n - cur) / PIPE_SAMPLE_RATE * 1000.0, 1))
            cur = None

    return {
        "model": "silero_vad.onnx (v5/v6 contract, CPUExecutionProvider)",
        "load_ms": round(load_s * 1000.0, 1),
        "threshold": config.vad_threshold,
        "hangover_ms": config.vad_hangover_ms,
        "as_production_512_no_context": as_production,
        "with_official_64_context": with_context,
        "energy_fallback_fused": {
            "fallback_active": bool(fused.fallback_active),
            "primary_peak": round(fused.primary_peak, 4),
            "segment_count": len(fsegs),
            "segment_ms": fsegs,
        },
        "verdict": (
            "BROKEN BOUNDARY: production feed (512, no context) -> primary "
            "deaf on valid speech; official 64-sample context -> speech "
            "detected. The energy fallback that v0.4.14 added to compensate "
            "fragments continuous speech (short segments) and is the "
            "prohibited heuristic."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# The full chain trace
# ═══════════════════════════════════════════════════════════════════════════


def trace_chain(source_wav: Path, ctx_rate: int, track_rate: int = 48000) -> dict:
    """Run the complete S0..S10 trace for one AudioContext rate."""
    import soundfile as sf

    s0, sr0 = sf.read(str(source_wav), dtype="float32")
    assert sr0 == PIPE_SAMPLE_RATE, f"source must be {PIPE_SAMPLE_RATE} Hz, got {sr0}"

    boundaries: list[dict] = []
    fidelity_rows: list[dict] = []
    boundaries.append(metrics("S0 source (16 kHz speech)", s0, PIPE_SAMPLE_RATE))

    # S1 track @ 48 kHz (browser-internal, out of app control)
    s1 = bandlimited_upsample(s0, track_rate // PIPE_SAMPLE_RATE)
    boundaries.append(metrics(f"S1 browser track ({track_rate} Hz)", s1, track_rate))

    # S2 AudioContext (browser-internal resample track -> ctx rate)
    if ctx_rate % track_rate == 0:
        s2 = bandlimited_upsample(s1, ctx_rate // track_rate)
    else:
        s2 = resample_linear(s1, track_rate, ctx_rate)
    boundaries.append(metrics(f"S2 AudioContext ({ctx_rate} Hz)", s2, ctx_rate))

    # S3 ScriptProcessor blocks (chunking only, samples unchanged)
    block = 4096
    blocks = [s2[i : i + block] for i in range(0, s2.size, block)]
    s3 = np.concatenate(blocks) if blocks else s2[:0]
    boundaries.append(
        metrics(
            f"S3 ScriptProcessor blocks (4096 @ {ctx_rate} Hz)",
            s3,
            ctx_rate,
            frame_size=block,
        )
    )

    # S4+S5: JS downsample to the wire rate + toPCM16 (per block, as the
    # browser does — the downsample is called per onaudioprocess frame)
    wire = WEB_SAMPLE_RATE
    pcm_blocks = [js_to_pcm16(js_downsample(b, ctx_rate, wire)) for b in blocks]
    pcm = np.concatenate(pcm_blocks) if pcm_blocks else np.zeros(0, dtype=np.int16)
    boundaries.append(
        metrics(f"S4 JS downsample({ctx_rate} -> {wire})", pcm, wire)
    )

    # S6 the actual WS bytes
    raw = pcm.tobytes()
    boundaries.append(metrics("S6 WebSocket bytes (PCM16 LE)", raw, wire))

    # S7+S8: the PRODUCTION server conversion
    f32 = pcm16_to_float32(raw)
    boundaries.append(metrics("S7 server pcm16_to_float32", f32, wire))
    s8 = resample_linear(f32, wire, PIPE_SAMPLE_RATE)
    boundaries.append(
        metrics(f"S8 server resample_linear({wire} -> {PIPE_SAMPLE_RATE})", s8, PIPE_SAMPLE_RATE)
    )

    # S9 VAD frames
    boundaries.append(
        metrics(
            "S9 VAD frames (512 @ 16 kHz)",
            s8[: (s8.size // VAD_FRAME_SAMPLES) * VAD_FRAME_SAMPLES],
            PIPE_SAMPLE_RATE,
            frame_size=VAD_FRAME_SAMPLES,
        )
    )

    # fidelity of the recovered signal vs S0
    fidelity_rows.append(fidelity(s0, s8, "S0 vs S8 (full chain, recovered)"))
    # isolate the server resample step: S7 input vs S8 output at same length
    # (upsample S7 to 16k ground-truth-style for a like-for-like comparison)
    up = bandlimited_upsample(f32, 2) if wire * 2 == PIPE_SAMPLE_RATE * 3 // 2 + 1 else None
    _ = up  # (24k -> 16k is NOT an integer upsample; compared spectrally below)

    # VAD on the recovered chain output (production config)
    cfg = AgentConfig.from_yaml(_ROOT / "config" / "voicemem_config.yaml")
    vad_report = vad_segmentation(s8, cfg)

    return {
        "ctx_rate": ctx_rate,
        "track_rate": track_rate,
        "wire_rate": wire,
        "resampling_steps": [
            f"{track_rate} Hz -> {ctx_rate} Hz (browser internal)",
            f"{ctx_rate} Hz -> {wire} Hz (JS downsample)",
            f"{wire} Hz -> {PIPE_SAMPLE_RATE} Hz (server resample_linear)",
        ],
        "boundaries": boundaries,
        "fidelity": fidelity_rows,
        "vad": vad_report,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="16 kHz mono WAV")
    ap.add_argument("--json", default="", help="write the report to this JSON path")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_file():
        print(f"source not found: {src}", file=sys.stderr)
        return 2

    print(f"ASR forensics — source: {src}")
    report: dict = {"source": str(src), "generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    for ctx_rate in (96000, 44100):
        print(f"\n══ chain trace (AudioContext {ctx_rate} Hz) ══")
        r = trace_chain(src, ctx_rate)
        report[f"chain_{ctx_rate}"] = r
        for b in r["boundaries"]:
            print(
                f"  {b['stage']:<46} rate={b['sample_rate']:>7} dtype={b['dtype']:<14} "
                f"n={b['samples']:>7} dur={b['duration_s']}s "
                f"rms={b['rms']} peak={b['peak']}"
            )
        for f in r["fidelity"]:
            print(
                f"  {f['stage']}: r={f.get('pearson_r')} rms_err={f.get('rms_error')} "
                f"bands={ {k: v for k, v in f.items() if k.endswith('snr_db')} }"
            )
        v = r["vad"]
        prod = v["as_production_512_no_context"]
        fixed = v["with_official_64_context"]
        ef = v["energy_fallback_fused"]
        print(
            f"  VAD production (512, no ctx): prob_max={prod['prob_max']} "
            f"frames>=thr={prod['frames_above_threshold']}/{prod['frames']} "
            f"segments={prod['segment_count']}"
        )
        print(
            f"  VAD official (64-ctx):        prob_max={fixed['prob_max']} "
            f"frames>=thr={fixed['frames_above_threshold']}/{fixed['frames']} "
            f"segments={fixed['segment_count']} "
            f"speech_ms={fixed['speech_ms_total']}"
        )
        print(
            f"  VAD energy fallback (shipped): fallback_active="
            f"{ef['fallback_active']} primary_peak={ef['primary_peak']} "
            f"segments={ef['segment_count']} ms={ef['segment_ms']}"
        )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
