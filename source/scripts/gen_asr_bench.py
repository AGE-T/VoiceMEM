"""Generate the ASR benchmark WAV corpus (TASK-A Phase 5).

Builds a reproducible, known-good speech corpus at 16 kHz mono PCM16 from
reference texts (Hungarian + English) with EXACT reference transcripts in a
JSON manifest, so WER/CER per engine is a deterministic function of
(file, engine).

Sources (v2 — after the z-ai TTS Hungarian validation failure):

* Hungarian files: Piper ``hu_HU-anna-medium`` (proper Hungarian
  phonemisation, fully local/deterministic — validated by parakeet + the
  z-ai ASR agreeing on a transcript close to the reference).
* English file: the z-ai TTS service (validated: parakeet transcribes it
  near-verbatim; English synthesis is correct where Hungarian was not).
* ``real_windows_capture``: copied from upload/asr_test_last.wav with the
  CONSENSUS reference transcript (parakeet v3 + z-ai ASR agree to within
  normalisation) — the real MixPre line-in capture that the field Qwen
  engine transcribed as repetition garbage.

Corpus requirements (task contract): short Hungarian sentence, normal
Hungarian sentence, long Hungarian sentence, English sentence, speech plus
silence, numbers, names, Hungarian with an English word.

TTS sources run at 22.05/24 kHz; the canonical ASR input rate is 16 kHz, so
the generator resamples (band-limited FFT resampler — ground-truth quality,
this is corpus PREPARATION, not the chain under test) and writes
``<id>.wav`` + ``manifest.json`` into ``data/asr_bench/``.
"""

from __future__ import annotations

import json
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "data" / "asr_bench"
TARGET_RATE = 16000

#: id -> (reference text, lang, source). Reference text is EXACTLY what WER
#: is measured against (normalised: single spaces, stripped).
CORPUS: list[tuple[str, str, str, str]] = [
    ("hu_short", "Szia, hogy vagy ma?", "hu", "piper"),
    (
        "hu_normal",
        "Jó napot kívánok, ma reggel Budapestre utaztam vonattal, és a találkozó után be akartam nézni a kedvenc könyvesboltomba.",
        "hu",
        "piper",
    ),
    (
        "hu_long",
        "Tegnap este sokáig beszélgettünk a jövő nyári terveinkről, és bár eredetileg csak egy rövid heti kirándulást terveztünk a Balatonra, végül úgy döntöttünk, hogy inkább két hetet töltünk el a tó északi partján, ahol már tavaly is annyira jól éreztük magunkat, mert ott van az a kis családi panzió, amelynek a teraszáról pont a naplemente látszik, és a reggelik is ott a legjobbak.",
        "hu",
        "piper",
    ),
    (
        "en_sentence",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "en",
        "zai",
    ),
    (
        "hu_numbers",
        "Száznegyvenkét forintot fizettem, majd hétszázharminc oldalas könyvet vettem, és tizenkilenc percet vártam a buszra.",
        "hu",
        "piper",
    ),
    (
        "hu_names",
        "Kovács Anna és Nagy Béla Szegeden találkozott Szabó Évával a Tisza partján.",
        "hu",
        "piper",
    ),
    (
        "hu_en_mixed",
        "Holnap reggel meetingem van az irodában, utána küldök egy reportot a managernek.",
        "hu",
        "piper",
    ),
    (
        "hu_with_silence",
        "Ez egy mondat, amelyet csend követ.",
        "hu",
        "piper",
    ),
]

#: Real Windows capture (v0.5.2 field): 9.984 s, 16 kHz, MixPre line-in via
#: the full browser chain. Qwen3-ASR (production engine at the time)
#: transcribed it as "让让让让让。"; the consensus reference is what two
#: independent engines (parakeet-tdt-0.6b-v3, z-ai ASR) agree on.
REAL_CAPTURE = (
    "/home/z/my-project/upload/asr_test_last.wav",
    "Ez egy teszt. Most kipróbáljuk, hogy működik-e az ASR-rendszer, de "
    "szerintem kurvára nem működik úgy.",
)

_PIPER_VOICE = "/tmp/piper_hu_HU-anna-medium.onnx"


def piper_tts(text: str, source_rate: int) -> np.ndarray:
    """Synthesise with the local Piper Hungarian voice (22.05 kHz).

    Long texts are synthesised SENTENCE BY SENTENCE (memory bound: the
    sandbox has 4 GB RAM and whole-paragraph synthesis of the long corpus
    entry gets OOM-killed; per-sentence calls stay small).
    """
    from piper import PiperVoice

    voice = piper_tts._voice
    if voice is None:
        voice = PiperVoice.load(_PIPER_VOICE)
        piper_tts._voice = voice
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    parts: list[np.ndarray] = []
    for sentence in sentences or [text]:
        chunks = list(voice.synthesize(sentence))
        if not chunks:
            raise RuntimeError(f"piper produced no audio for {sentence[:40]!r}")
        rate = chunks[0].sample_rate
        assert rate == source_rate, f"piper rate {rate} != {source_rate}"
        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        # small inter-sentence pause keeps the corpus speech natural
        gap = np.zeros(int(0.15 * source_rate), dtype=np.float32)
        parts.append(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)
        parts.append(gap)
    return np.concatenate(parts[:-1]) if len(parts) > 1 else parts[0]


piper_tts._voice = None  # type: ignore[attr-defined]


def zai_tts(text: str, source_rate: int) -> np.ndarray:
    """Synthesise with the z-ai CLI (24 kHz; English only — Hungarian output
    was VALIDATED as gibberish and is not used)."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        proc = subprocess.run(
            ["z-ai", "tts", "-i", text, "-o", str(tmp)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size < 2000:
            raise RuntimeError(f"z-ai TTS failed for {text[:40]!r}")
        with wave.open(str(tmp), "rb") as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2
            assert w.getframerate() == source_rate
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0
    finally:
        tmp.unlink(missing_ok=True)


def bandlimited_resample(x: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Windowed-sinc resampler (ground-truth corpus PREPARATION quality).

    Kaiser-windowed sinc, 64 taps per output sample, linear-phase fractional
    offsets. Replaces the FFT zero-padding variant: 22050 -> 16000 is the
    rational factor 320/441 and the FFT form would upsample by 320 FIRST
    (a 51M-point transform, >1.5 GB — the corpus generator's original OOM).
    This form's memory is O(output length).
    """
    if from_sr == to_sr:
        return x.astype(np.float32)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    ratio = to_sr / from_sr
    n_out = int(np.floor(x.size * ratio))
    taps = 64
    beta = 8.6  # Kaiser beta ~ -80 dB sidelobes
    pos = (np.arange(n_out, dtype=np.float64) + 0.5) / ratio - 0.5  # source-space centers
    base = np.floor(pos).astype(np.int64)
    frac = (pos - base).reshape(-1, 1)
    offsets = np.arange(taps, dtype=np.float64).reshape(1, -1) - (taps // 2) + 0.5
    # sinc kernel evaluated at (offset - frac)
    arg = offsets - frac
    kernel = np.sinc(arg) * np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (arg / (taps / 2.0)) ** 2))) / np.i0(beta)
    idx = base.reshape(-1, 1) + np.arange(taps, dtype=np.int64).reshape(1, -1) - (taps // 2)
    # reflect-pad out-of-range indices (edge-safe)
    idx = np.clip(idx, 0, x.size - 1)
    weights = kernel
    # normalise each row (sinc tails at edges)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    y = (x[idx] * weights).sum(axis=1)
    return y.astype(np.float32)


def write_wav16k(path: Path, x16: np.ndarray) -> None:
    pcm = (np.clip(x16, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm.tobytes())


def synth_one(file_id: str, text: str, source: str, out: Path) -> dict:
    """Synthesise ONE corpus file in a SUBPROCESS (memory isolation).

    The sandbox has 4 GB RAM and Piper's onnxruntime arena transiently
    spikes >1.5 GB on some sentences; a long-lived process that keeps the
    voice loaded across files gets OOM-killed. A fresh interpreter per file
    bounds every spike to a short-lived process. Returns the manifest entry.
    """
    import os
    import subprocess as sp

    code = (
        "import sys, json, numpy as np\n"
        f"sys.path.insert(0, {str(_ROOT / 'scripts')!r})\n"
        "from gen_asr_bench import piper_tts, zai_tts, bandlimited_resample, "
        "write_wav16k\n"
        "payload = json.loads(sys.stdin.read())\n"
        "x, rate = (piper_tts(payload['text'], 22050), 22050) "
        "if payload['source'] == 'piper' else "
        "(zai_tts(payload['text'], 24000), 24000)\n"
        "x16 = bandlimited_resample(x, rate, 16000)\n"
        f"write_wav16k({str(out)!r}, x16)\n"
        "print(json.dumps({'samples': int(x16.size), "
        "'rms': float(np.sqrt(np.mean(x16 ** 2)))}))\n"
    )
    env = dict(os.environ)
    # onnxruntime's multi-thread arena transiently allocates 1.4-1.9 GB on
    # this 4 GB sandbox (OOM-killed); single-threaded synthesis peaks at
    # ~300 MB. Piper synthesis is single-stream anyway.
    env["OMP_NUM_THREADS"] = "1"
    proc = sp.run(
        [sys.executable, "-u", "-c", code],
        input=json.dumps(
            {"id": file_id, "text": text, "source": source}, ensure_ascii=False
        ),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess synthesis failed for {file_id}: {proc.stderr[-300:]}"
        )
    stats = json.loads(proc.stdout.strip().splitlines()[-1])
    return stats


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "corpus": "voicemem-agent ASR benchmark v2 (piper HU + zai EN)",
        "sample_rate": TARGET_RATE,
        "channels": 1,
        "dtype": "pcm16",
        "note": (
            "Known-reference speech corpus for engine-vs-engine comparison "
            "(WER/CER/latency/RTF). Hungarian: Piper hu_HU-anna-medium "
            "(local, deterministic). English: z-ai TTS (validated). "
            "real_windows_capture: real MixPre line-in field capture, "
            "consensus reference transcript (parakeet v3 + z-ai ASR)."
        ),
        "files": {},
    }

    for file_id, text, lang, source in CORPUS:
        out = OUT / f"{file_id}.wav"
        x16_samples = synth_one(file_id, text, source, out)["samples"]
        if file_id == "hu_with_silence":
            # 2 s trailing silence appended in the parent process (tiny)
            import wave as _wave

            with _wave.open(str(out), "rb") as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            pcm = np.concatenate(
                (pcm, np.zeros(int(2.0 * TARGET_RATE), dtype=np.int16))
            )
            with _wave.open(str(out), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(TARGET_RATE)
                w.writeframes(pcm.tobytes())
            x16_samples = int(pcm.size)
        rms = float(
            np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2))
        ) if file_id == "hu_with_silence" else None
        entry = {
            "wav": out.name,
            "text": text,
            "language": lang,
            "source": source,
            "duration_s": round(x16_samples / TARGET_RATE, 3),
            "samples": int(x16_samples),
        }
        if rms is not None:
            entry["rms"] = round(rms, 5)
        manifest["files"][file_id] = entry
        print(f"  {file_id}: {x16_samples / TARGET_RATE:.2f}s [{source}]")

    # real capture (copy + consensus reference)
    real_src = Path(REAL_CAPTURE[0])
    if real_src.is_file():
        out = OUT / "real_windows_capture.wav"
        shutil.copyfile(real_src, out)
        with wave.open(str(out), "rb") as w:
            n = w.getnframes()
        manifest["files"]["real_windows_capture"] = {
            "wav": out.name,
            "text": REAL_CAPTURE[1],
            "language": "hu",
            "source": "real MixPre line-in (Windows field capture, v0.5.2)",
            "duration_s": round(n / TARGET_RATE, 3),
            "samples": int(n),
            "reference_basis": (
                "consensus transcript (parakeet-tdt-0.6b-v3 and an independent "
                "cloud ASR agree; the production Qwen engine returned "
                "repetition garbage on this same file)"
            ),
        }
        print(f"  real_windows_capture: {n / TARGET_RATE:.2f}s [field]")

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"corpus written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
