"""ASR regression forensic — the on-target matrix cell (v0.10.1, P0).

WHY THIS SCRIPT EXISTS
======================
The v0.10.0 P0 field report: Hungarian speech transcribed as fluent Russian
("Я им работаю, валь.") or unrelated English ("store school") on the target
machine (Windows + RTX 5070, CUDA), while the same machine produced correct
Hungarian under v0.9.2.

The sandbox-side forensics (2026-09-15) PROVED, on the pinned model +
production code path:

  * the v0.9.2 -> v0.10.0 product delta does NOT touch the audio path
    (complete tree diff: only memory-semantics files changed);
  * transformers 5.9.0 .. 5.17.0 on CPU produce CORRECT, byte-identical
    Hungarian transcripts on the whole bench corpus (6 files);
  * the production Silero VAD segmentation is correct on the same corpus.

What the sandbox CANNOT prove is the CUDA execution leg (no GPU there).
THIS script runs the missing matrix cell ON THE ACTUAL MACHINE:

  1. dumps the real dependency state (torch/CUDA/device, transformers,
     tokenizers, onnxruntime, numpy + the ASR-relevant pip freeze);
  2. verifies the Parakeet model files on disk against the pinned-revision
     SHA256 values (MODELS.lock.json revision 541d1f99...);
  3. runs the shipped bench corpus through the PRODUCTION engine path on
     the CONFIGURED device (cuda) and, for comparison, on cpu;
  4. flags wrong-language output (Cyrillic) and computes a character error
     rate against the reference transcripts.

Run it with the product's own venv (the START.bat environment):

    .venv\\Scripts\\python.exe scripts\\asr_regression_forensic.py

Output: logs/asr_forensic_<timestamp>/report.json + console summary.
The report is the evidence the fix decision needs — send it back with the
wrong-language WAV dumps (logs/asr_wrong_language_*.wav).

This script is READ-ONLY with respect to the product: it changes no
configuration, writes only into logs/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

#: SHA256 of every file of the pinned Parakeet revision (541d1f99c6b0...)
#: as verified from a fresh pinned download on 2026-09-15 (sandbox).
PINNED_MODEL_SHA256 = {
    "config.json": "e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e",
    "generation_config.json": "b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0",
    "model.safetensors": "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
    "processor_config.json": "8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7",
    "tokenizer.json": "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
    "tokenizer_config.json": "0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874",
}

CORPUS = [
    ("hu_short", "Szia, hogy vagy ma?"),
    ("hu_with_silence", "Ez egy mondat, amelyet csend követ."),
    ("hu_normal",
     "Jó napot kívánok, ma reggel Budapestre utaztam vonattal, és a "
     "találkozó után be akartam nézni a kedvenc könyvesboltomba."),
    ("hu_names",
     "Kovács Anna és Nagy Béla Szegeden találkozott Szabó Évával a Tisza "
     "partján."),
    ("hu_long",
     "Tegnap este sokáig beszélgettünk a jövő nyári terveinkről, és bár "
     "eredetileg csak egy rövid heti kirándulást terveztünk a Balatonra, "
     "végül úgy döntöttünk, hogy inkább két hetet töltünk el a tó északi "
     "partján, ahol már tavaly is annyira jól éreztük magunkat, mert ott van "
     "az a kis családi panzió, amelynek a teraszáról pont a naplemente "
     "látszik, és a reggelik is ott a legjobbak."),
    ("real_windows_capture",
     "Ez egy teszt. Most kipróbáljuk, hogy működik-e az ASR-rendszer, de "
     "szerintem kurvára nem működik úgy."),
]


def _norm(text: str) -> str:
    """Normalise a transcript for CER: case-fold, strip accents/punct/space."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _cer(expected: str, got: str) -> float:
    """Character error rate (edit distance / len(expected))."""
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


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= ch <= "\u04FF" for ch in text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def env_dump() -> dict:
    import numpy

    env: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
            env["cuda_capability"] = ".".join(
                map(str, torch.cuda.get_device_capability(0))
            )
            env["cuda_version"] = torch.version.cuda
            env["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception as exc:  # noqa: BLE001
        env["torch"] = f"IMPORT FAILED: {exc}"
    for mod in ("transformers", "tokenizers", "onnxruntime", "soundfile"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            env[mod] = f"IMPORT FAILED: {exc}"
    return env


def main() -> None:
    from app.config import AgentConfig
    from app.asr_core import AudioBuffer, select_engine

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = _ROOT / "logs" / f"asr_forensic_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"schema": "asr-regression-forensic/1", "generated": ts}

    # -- 1) environment ------------------------------------------------------
    report["env"] = env_dump()
    print(json.dumps(report["env"], indent=1, ensure_ascii=False))

    # -- 2) model identity ---------------------------------------------------
    cfg = AgentConfig.from_yaml(_ROOT / "config" / "voicemem_config.yaml")
    engine = select_engine(cfg)
    status = engine.status()
    model_dir = Path(status["model_dir"])
    model_files: dict = {}
    for name, expected_sha in PINNED_MODEL_SHA256.items():
        p = model_dir / name
        if not p.is_file():
            model_files[name] = {"present": False, "sha256": ""}
            continue
        actual = _sha256(p)
        model_files[name] = {
            "present": True,
            "sha256": actual,
            "matches_pinned_revision": actual == expected_sha,
            "size_bytes": p.stat().st_size,
        }
    report["model"] = {
        "model_dir": str(model_dir),
        "configured_device": status["device"],
        "files": model_files,
        "all_match": all(v.get("matches_pinned_revision") for v in model_files.values()),
    }
    print(f"[model] dir={model_dir} all_sha_match={report['model']['all_match']}")

    # -- 3) corpus matrix: configured device (+ cpu comparison) ---------------
    devices = [status["device"]] if status["device"] == "cpu" else [status["device"], "cpu"]
    runs = {}
    for device in devices:
        cfg.asr_device = device
        engine_d = select_engine(cfg)
        rows = []
        for name, expected in CORPUS:
            wav = _ROOT / "data" / "asr_bench" / f"{name}.wav"
            if not wav.is_file():
                rows.append({"name": name, "error": "wav missing"})
                continue
            buf = AudioBuffer.from_wav(wav).validated(stage="asr_input")
            t0 = time.perf_counter()
            res = engine_d.transcribe(buf)
            wall = (time.perf_counter() - t0) * 1000.0
            text = res.text
            row = {
                "name": name,
                "status": res.status.value,
                "transcript": text,
                "cyrillic": _has_cyrillic(text),
                "cer": round(_cer(expected, text), 4),
                "inference_ms": round(res.inference_ms, 1),
                "wall_ms": round(wall, 1),
            }
            if res.error is not None:
                row["error"] = res.error.detail[:300]
            rows.append(row)
            flag = " CYRILLIC!" if row["cyrillic"] else ""
            print(
                f"[{device}] {name}: {res.status.value}{flag} "
                f"cer={row['cer']:.3f} {text[:60]!r}",
                flush=True,
            )
        runs[device] = rows
    report["runs"] = runs

    # -- 4) verdict -----------------------------------------------------------
    problems = []
    for device, rows in runs.items():
        for row in rows:
            if row.get("cyrillic"):
                problems.append(f"{device}/{row['name']}: CYRILLIC transcript "
                                f"(wrong language): {row['transcript'][:60]!r}")
            elif row.get("status") == "ok" and row.get("cer", 1.0) > 0.45:
                problems.append(f"{device}/{row['name']}: CER {row['cer']:.3f} "
                                f"(unrelated transcript?): {row['transcript'][:60]!r}")
            elif row.get("status") != "ok":
                problems.append(f"{device}/{row['name']}: status={row.get('status')} "
                                f"error={row.get('error', '')[:120]}")
    if not report["model"]["all_match"]:
        problems.append("model files do NOT match the pinned revision SHA256")
    report["verdict"] = {"clean": not problems, "problems": problems}
    print("\n=== VERDICT ===")
    if problems:
        for p in problems:
            print("  PROBLEM:", p)
        print("  -> collect logs/asr_wrong_language_*.wav and this report.json")
    else:
        print("  CLEAN: correct Hungarian on every corpus file, model identity "
              "verified, no wrong-language output.")

    with open(out_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(f"\nreport -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
