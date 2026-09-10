"""M1 Hungarian ASR benchmark (Qwen3-ASR-0.6B on the TARGET machine).

Machine: TARGET (Windows 11 + RTX 5070, models downloaded, torch cu128).
This sandbox (Linux, no GPU) can only IMPORT this module; the actual run
requires the GPU machine. All heavy imports are guarded so that importing
this file anywhere is safe.

Method:
  1. Load a 16 kHz mono WAV recording of the verbatim M1 test sentence
     (supplied via --wav; recording instructions are printed when missing).
  2. Feed the audio to app.asr.AsrEngine in quasi-streaming chunks
     (600 ms batches -> partial transcripts), then flush() for the final one.
  3. Score the final transcript with tests/benchmark/metrics.py:
     WER, CER, per-token accuracy, accent preservation.
  4. Print a metrics table and a VRAM-peak hint (nvidia-smi command).

Exit criteria (milestones doc Section 7.6.1): WER < 25% AND CER < 15%
  -> exit 0 on success, 1 on failure, 2 on missing prerequisites.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests/<kind>/x.py -> 3 levels up)

try:
    from tests.benchmark.metrics import accent_preservation, cer, token_accuracy, wer
except ImportError:  # direct run: script dir (tests/benchmark) already on sys.path
    from metrics import accent_preservation, cer, token_accuracy, wer  # type: ignore

# VERBATIM M1 Hungarian ASR test sentence (spec Section 19.1 / milestones 7.6.1).
# DO NOT EDIT: the misspelled "iylet" and the name "Hojsz Tamás" are intentional
# OOV/robustness probes, part of the benchmark definition.
HU_REFERENCE = (
    "A flat white és a tejeskávé egyaránt espresso-alapú italok tejjel. "
    "Mindazonáltal egyértelműen különböznek a tej mennyiségében és textúrájában, "
    "valamint az általános ízegyensúlyban. A flat white úgy lett kialakítva, hogy "
    "kiemelje az espressót. Nagyon vékony réteg finom mikrohabot használ, amely "
    "sima, szinte sík felületet ad. Ennek köszönhetően a kávé gazdag íze és aromája "
    "erőteljesebben érvényesül. Az italt általában kisebb csészében, kevesebb "
    "tejjel tálalják, mint a lattét. Ezzel szemben a tejeskávé nagyobb arányban "
    "tartalmaz forró, gőzölt tejet. Vastagabb a habrétege is, ami krémesebb és "
    "lágyabb jelleget ad neki. Ez tompítja az espresso keserűségét és savasságát. "
    "Ezért is olyan közkedvelt és könnyen iható tejes kávé. Első iylet 1890-ben "
    "készítette Hojsz Tamás nevű tudós."
)

# Key tokens tracked by the milestone exit criteria (6.6.1) plus OOV probes.
KEY_TOKENS = [
    "flat white",
    "espresso",
    "tejeskávé",
    "mikrohab",
    "lattét",
    "1890",
    "Hojsz Tamás",
]

WER_LIMIT = 0.25   # M1 exit criterion
CER_LIMIT = 0.15   # M1 exit criterion


def load_wav_mono_16k(path: Path) -> "list[Any] | Any":
    """Read a 16 kHz mono PCM WAV with stdlib `wave`, return float32 samples.

    Supports 16-bit PCM (the format produced by sounddevice recording below).
    Any other sample rate is rejected: ASR requires exactly 16 kHz.
    """
    with wave.open(str(path), "rb") as reader:
        n_channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        width = reader.getsampwidth()
        frames = reader.readframes(reader.getnframes())
    if sample_rate != 16000:
        raise ValueError(f"WAV must be 16 kHz, got {sample_rate} Hz (record with --sample_rate 16000)")
    if n_channels != 1:
        raise ValueError(f"WAV must be mono, got {n_channels} channels")
    if width != 2:
        raise ValueError(f"WAV must be 16-bit PCM, got sample width {width} bytes")
    try:
        import numpy as np  # lazy: only needed when actually running on the target machine
    except ImportError as exc:  # pragma: no cover - sandbox import check path
        raise RuntimeError("numpy is required to run the benchmark (missing here)") from exc
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def print_recording_instructions() -> None:
    """How to capture the reference sentence as 16 kHz mono WAV (sounddevice)."""
    print("No --wav file supplied. Record the reference sentence first, e.g.:")
    print()
    print("  import sounddevice as sd, wave")
    print(f"  TEXT = {HU_REFERENCE!r}")
    print("  print('Olvasd fel hangosan, 2 mp mulva indul a felvetel...')")
    print("  import time; time.sleep(2.0)")
    print("  audio = sd.rec(int(75 * 16000), samplerate=16000, channels=1, dtype='int16')")
    print("  sd.wait()  # max 75 s; a mondat kb. 60-70 s")
    print("  with wave.open('hu_reference.wav', 'wb') as w:")
    print("      w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)")
    print("      w.writeframes(audio.tobytes())")
    print()
    print("Then run: python tests/benchmark/asr_benchmark_hu.py --wav hu_reference.wav")
    print("The sentence (copy it exactly, including the misspelled 'iylet'):")
    print(f"  {HU_REFERENCE}")


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 Hungarian ASR benchmark (target machine)")
    parser.add_argument("--wav", help="path to a 16 kHz mono WAV recording of the reference sentence")
    parser.add_argument("--config", help="optional YAML config (config/voicemem_config.yaml)")
    args = parser.parse_args()

    if not args.wav:
        print_recording_instructions()
        return 2

    from app.config import AgentConfig

    cfg = AgentConfig()
    cfg.apply_env()
    if args.config and Path(args.config).exists():
        try:
            cfg = AgentConfig.from_yaml(Path(args.config))
        except (TypeError, ValueError, RuntimeError) as exc:
            print(f"WARN: YAML config load failed ({exc}); using env + defaults")

    try:
        import app.asr  # heavy deps inside are guarded by its own is_available()
    except ImportError:
        print("app.asr is not importable — run this on the target machine from the repo root.")
        return 2

    engine = app.asr.AsrEngine(cfg)
    if not engine.is_available():
        print("AsrEngine unavailable: torch/transformers missing or model dir incomplete.")
        print(f"Model dir: {cfg.asr_model_dir} (run scripts/download_models.ps1)")
        return 2

    wav_path = Path(args.wav)
    if not wav_path.is_file():
        print(f"WAV file not found: {wav_path}")
        return 2
    try:
        samples = load_wav_mono_16k(wav_path)
    except (ValueError, RuntimeError) as exc:
        print(f"Cannot load WAV: {exc}")
        return 2

    audio_seconds = len(samples) / cfg.sample_rate
    chunk = cfg.asr_chunk_samples  # 600 ms quasi-streaming batches
    first_partial_s: float | None = None
    partials: list[str] = []
    engine.reset()
    t_start = time.perf_counter()
    for offset in range(0, len(samples), chunk):
        partial = engine.feed(samples[offset : offset + chunk])
        if partial and first_partial_s is None:
            first_partial_s = time.perf_counter() - t_start
        if partial:
            partials.append(partial)
    transcript = engine.flush()
    final_latency_s = time.perf_counter() - t_start

    metrics: dict[str, Any] = {
        "wer": wer(HU_REFERENCE, transcript),
        "cer": cer(HU_REFERENCE, transcript),
        "token_accuracy": token_accuracy(HU_REFERENCE, transcript, KEY_TOKENS),
        "accent_preservation": accent_preservation(HU_REFERENCE, transcript),
        "audio_seconds": round(audio_seconds, 2),
        "first_partial_s": None if first_partial_s is None else round(first_partial_s, 3),
        "final_latency_s": round(final_latency_s, 3),
        "rtf": round(final_latency_s / audio_seconds, 3) if audio_seconds > 0 else None,
    }

    print("=" * 72)
    print("M1 ASR benchmark (Hungarian, Qwen3-ASR-0.6B)")
    print("=" * 72)
    print(f"wav           : {wav_path}")
    print(f"model dir     : {cfg.asr_model_dir}")
    print(f"audio length  : {metrics['audio_seconds']} s")
    print("-" * 72)
    print(f"WER           : {metrics['wer'] * 100:.2f}%   (limit {WER_LIMIT * 100:.0f}%)")
    print(f"CER           : {metrics['cer'] * 100:.2f}%   (limit {CER_LIMIT * 100:.0f}%)")
    print(f"accent kept   : {metrics['accent_preservation'] * 100:.2f}%")
    print(f"first partial : {metrics['first_partial_s']} s")
    print(f"final latency : {metrics['final_latency_s']} s")
    print(f"RTF           : {metrics['rtf']}")
    print("-" * 72)
    print("token accuracy (case-insensitive, multi-word tokens must stay contiguous):")
    for token, ok in metrics["token_accuracy"].items():
        print(f"  {'OK      ' if ok else 'MISS    '} {token}")
    print("-" * 72)
    print("Transcript:")
    print(f"  {transcript}")
    print("-" * 72)
    print("VRAM peak: run on the target machine in a second terminal:")
    print("  nvidia-smi --query-gpu=memory.used --format=csv -l 1")
    print("  (record the peak while this benchmark runs; steady budget is ~9.6-10.0 GB with the Gemma 4 12B LLM)")

    passed = metrics["wer"] < WER_LIMIT and metrics["cer"] < CER_LIMIT
    print("=" * 72)
    print(f"RESULT: {'PASS' if passed else 'FAIL'} "
          f"(WER < 25% and CER < 15% — milestones doc Section 7.6.1)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
