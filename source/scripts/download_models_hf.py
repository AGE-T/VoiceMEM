#!/usr/bin/env python3
"""scripts/download_models_hf.py - M1+M2+M3 model downloader (HF Python API only).

Purpose: download the M1+M2+M3 strict model set into the project ``models/``
tree using the huggingface_hub PYTHON API (``hf_hub_download`` /
``snapshot_download``). This module REPLACES the previous CLI-executable
based download.

WHY the CLI was removed (v0.1.6 release blocker):
  The ``download`` entrypoint of the HF command line tool prints a
  deprecation warning that contains an EMOJI character. Under Windows
  PowerShell with a CP1252 console codepage the child Python process
  crashes with UnicodeEncodeError BEFORE the actual download starts, so
  no model ever arrives. The fix is architectural: no CLI subprocess, no
  CLI output parsing, no PATH dependency - only the in-process Python API
  of ``huggingface_hub`` invoked through the project venv interpreter
  (``.venv\\Scripts\\python.exe``). The Hugging Face CLI is NOT a runtime
  dependency of this project anymore.

What this script does (all driven by MODELS.lock.json, schema_version 2):
  1. Reads the repo-root MODELS.lock.json (creates the default lock file
     if it is missing - self-heal, same content as the shipped one).
  2. For every model entry (M1: llm/asr/embedding/vad/tts; M2: emotion -
     emotion2vec+ base, 4 runtime files, >1 GB model.pt floor; M3:
     speaker - speechbrain ECAPA, 5 runtime files, 80 MiB ckpt floor):
       - snapshot=false -> ``hf_hub_download`` per listed file into
         ``target_dir`` (honours ``pinned_revision``);
       - snapshot=true  -> ``snapshot_download`` of the whole repo into
         ``target_dir`` (honours ``pinned_revision``);
       - already-present, size-verified files are SKIPPED (idempotent,
         zero network traffic when everything is in place);
       - the LLM is a single pinned official repo with a pinned
         sha256 (v0.4.16: the LLM is NOT auto-downloaded - see the llm note).
         NO alternate LLM, LLM FALLBACK: NONE); the VAD keeps its
         mirror (tphakala re-host ->
         silero original with flatten; piper voices: nested repo path ->
         flat layout fallback) - the one-click rule is preserved.
  3. RETRY: every download is retried (default 3 attempts, 5s/15s
     backoff). ``hf_hub_download``/``snapshot_download`` RESUME broken
     transfers automatically, so a re-run continues where it stopped.
  4. VERIFICATION: the script only reports success when every mandatory
     file actually EXISTS on disk with at least ``min_bytes`` size (the
     LLM GGUF has a >4 GB floor) and every snapshot entry has its probe
     file plus ``min_total_mb`` total payload. A missing/thin file is a
     hard failure (exit code 1) - the download step is never "successful"
     on a broken state.
  5. Prints a clear ASCII-only status table and machine-readable result
     lines (``ALL_MODELS_OK`` / ``MODELS_FAILED: <components>``).

Usage (the user NEVER runs this by hand - scripts/download_models.ps1
and the installer call it with the project venv interpreter):
    .venv\\Scripts\\python.exe scripts\\download_models_hf.py --root <repo>
    ... --only vad,tts          (selective repair of components)
    ... --verify-only           (idempotent presence check, no downloads)
    ... --sha256                (print sha256 of each verified file)

Output is pure ASCII on purpose: the script must be safe under any
Windows console codepage (CP437/CP1252). Set PYTHONIOENCODING=utf-8 in
the environment for belt-and-braces (the PowerShell wrapper does).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_DEFAULT_LOCK: Dict[str, Any] = {
    "schema_version": 2,
    "_note": (
        "Pinned model manifest for the M1+M2 strict model set (one-click "
        "bootstrap reads this file for local presence checks - see "
        "scripts/bootstrap.ps1 Test-ModelsPresent). pinned_revision "
        "'main' tracks the repo default branch; replace it with a full "
        "commit SHA for strict reproducibility - the resolved revision "
        "is recorded into INSTALL_MANIFEST.json by "
        "scripts/write_install_manifest.py after the download. "
        "snapshot=true means a full repo snapshot; presence is probed "
        "via snapshot_probe plus a total-size floor (min_total_mb). The "
        "M2 emotion entry (emotion2vec+ base) is CPU-only and optional "
        "at runtime - a missing emotion model degrades the pipeline to "
        "M1 behaviour, never blocks startup."
    ),
    "models": [
        {
            "component": "asr",
            "repo": "Qwen/Qwen3-ASR-0.6B",
            "target_dir": "models/asr/qwen3-asr-0.6b",
            "files": ["(full repo snapshot)"],
            "snapshot": True,
            "snapshot_probe": "config.json",
            "min_total_mb": 500,
            "pinned_revision": "main",
        },
        {
            "component": "embedding",
            "repo": "intfloat/multilingual-e5-small",
            "target_dir": "models/embedding/multilingual-e5-small",
            "files": ["(full repo snapshot)"],
            "snapshot": True,
            "snapshot_probe": "config.json",
            "min_total_mb": 200,
            "pinned_revision": "main",
        },
        {
            "component": "emotion",
            "repo": "emotion2vec/emotion2vec_plus_base",
            "_note": (
                "M2 prosody emotion analyzer (90M, CPU via funasr "
                "AutoModel). Official HF re-host by the emotion2vec "
                "team; weights under the FunASR Model Open Source "
                "License Agreement v1.1 (attribution - keep the model "
                "name; see LICENSES.md). Only the 4 runtime files are "
                "downloaded - repo images and the example/ folder are "
                "skipped."
            ),
            "target_dir": "models/emotion/emotion2vec-plus-base",
            "files": ["model.pt", "config.yaml", "tokens.txt", "configuration.json"],
            "snapshot": False,
            "pinned_revision": "main",
            "min_bytes": {
                "model.pt": 1073741824,
                "config.yaml": 1000,
                "tokens.txt": 50,
                "configuration.json": 100,
            },
        },
        {
            "component": "speaker",
            "repo": "speechbrain/spkrec-ecapa-voxceleb",
            "_note": (
                "M3 speaker embedding model (ECAPA-TDNN, 192-dim, CPU via "
                "the speechbrain pip package; Apache-2.0, not gated). All 5 "
                "runtime files are downloaded; the repo example audio files "
                "are skipped. Loaded from the LOCAL directory at runtime."
            ),
            "target_dir": "models/speaker/ecapa-voxceleb",
            "files": [
                "embedding_model.ckpt",
                "hyperparams.yaml",
                "mean_var_norm_emb.ckpt",
                "classifier.ckpt",
                "label_encoder.txt",
            ],
            "snapshot": False,
            "pinned_revision": "main",
            "min_bytes": {
                "embedding_model.ckpt": 80000000,
                "hyperparams.yaml": 500,
                "mean_var_norm_emb.ckpt": 500,
                "classifier.ckpt": 1048576,
                "label_encoder.txt": 10240,
            },
        },
        {
            "component": "vad",
            "repo": "tphakala/silero-vad",
            "target_dir": "models/vad/silero-vad",
            "files": ["silero_vad.onnx"],
            "snapshot": False,
            "pinned_revision": "main",
            "min_bytes": {"silero_vad.onnx": 1048576},
        },
        {
            "component": "tts",
            "repo": "rhasspy/piper-voices",
            "target_dir": "models/tts/piper",
            "files": [
                "hu_HU-anna-medium.onnx",
                "hu_HU-anna-medium.onnx.json",
                "hu_HU-berta-medium.onnx",
                "hu_HU-berta-medium.onnx.json",
                "hu_HU-imre-medium.onnx",
                "hu_HU-imre-medium.onnx.json",
                "en_US-lessac-medium.onnx",
                "en_US-lessac-medium.onnx.json",
            ],
            "snapshot": False,
            "pinned_revision": "main",
            "min_bytes": {
                "hu_HU-anna-medium.onnx": 1048576,
                "hu_HU-anna-medium.onnx.json": 1,
                "hu_HU-berta-medium.onnx": 1048576,
                "hu_HU-berta-medium.onnx.json": 1,
                "hu_HU-imre-medium.onnx": 1048576,
                "hu_HU-imre-medium.onnx.json": 1,
                "en_US-lessac-medium.onnx": 1048576,
                "en_US-lessac-medium.onnx.json": 1,
            },
        },
    ],
}

# --- pinned model constants (must stay in sync with MODELS.lock.json) --------
# v0.4.16: the ONE AND ONLY LLM is Qwen3.6 35B A3B IQ4_XS - an OPERATOR-PLACED
# GGUF (~19 GB, tested source hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS)
# that is NEVER auto-downloaded by this script: it may live on any drive and is
# selected with the web UI "LLM model" picker (config/llm_model.json) or
# scripts/find_qwen_gguf.ps1, or placed at models/llm/qwen3.6-35b-a3b/. There is
# NO fallback profile. VAD keeps its mirror (not an LLM).
VAD_PRIMARY = ("tphakala/silero-vad", "silero_vad.onnx")
VAD_FALLBACK = ("silero/silero-vad", "v6.2.1/silero_vad.onnx")

# Retry defaults: 3 attempts total, sleeping 5s / 15s between them.
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 5.0
RETRY_BACKOFF_FACTOR = 3.0


def _print_flush(msg: str) -> None:
    print(msg, flush=True)


def _mb(num_bytes: float) -> str:
    return "%.1f MB" % (num_bytes / (1024.0 * 1024.0))


# ------------------------------------------------------------------ lock ----

def load_lock(root: Path) -> Tuple[Dict[str, Any], Path]:
    """Load MODELS.lock.json from the repo root; self-heal the default."""
    lock_path = root / "MODELS.lock.json"
    if not lock_path.is_file():
        _print_flush(
            "[INFO] MODELS.lock.json not found - writing the default "
            "schema_version 2 lock (self-heal)."
        )
        lock_path.write_text(
            json.dumps(REPO_DEFAULT_LOCK, indent=2, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        return REPO_DEFAULT_LOCK, lock_path
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _print_flush("[FAIL] MODELS.lock.json is not valid JSON: %s" % exc)
        sys.exit(2)
    if not isinstance(lock.get("models"), list) or not lock["models"]:
        _print_flush("[FAIL] MODELS.lock.json has no 'models' list.")
        sys.exit(2)
    return lock, lock_path


# ------------------------------------------------------------- hf imports ---

def import_huggingface_hub() -> Any:
    """Import huggingface_hub with a clear, actionable error message."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        _print_flush(
            "[FAIL] The 'huggingface_hub' Python library is not installed "
            "in this interpreter."
        )
        _print_flush(
            "       FIX: re-run START.bat - the installer installs it "
            "automatically into .venv (one-click rule)."
        )
        sys.exit(3)
    from huggingface_hub import hf_hub_download, snapshot_download

    return hf_hub_download, snapshot_download


# --------------------------------------------------------------- retry ------

def with_retries(
    label: str,
    func: Callable[[], Any],
    attempts: int,
    base_delay: float,
) -> bool:
    """Run func() with retry + exponential backoff. True on success."""
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            func()
            return True
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer raises many
            if attempt >= attempts:
                _print_flush(
                    "[FAIL] %s: gave up after %d attempt(s): %s"
                    % (label, attempt, exc)
                )
                return False
            _print_flush(
                "[RETRY] %s: attempt %d/%d failed (%s) - retrying in %.0fs "
                "(resume is automatic)."
                % (label, attempt, attempts, exc, delay)
            )
            time.sleep(delay)
            delay *= RETRY_BACKOFF_FACTOR
    return False


# ------------------------------------------------------------- utilities ----

def dir_total_bytes(path: Path) -> int:
    """Total size of real payload files under path, skipping .cache/hidden."""
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(path)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue  # .cache/huggingface metadata / partial blobs
        total += item.stat().st_size
    return total


def file_ok(path: Path, min_bytes: int) -> Tuple[bool, int]:
    if not path.is_file():
        return False, 0
    size = path.stat().st_size
    return size >= min_bytes, size


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_nested(root_dir: Path) -> int:
    """Move files from subdirectories into root_dir (piper/vad layout).

    Idempotent: files already at the root are untouched; a target
    collision is skipped (first file wins). Returns moved file count.
    """
    moved = 0
    for item in sorted(root_dir.rglob("*")):
        if not item.is_file():
            continue
        if item.parent == root_dir:
            continue
        rel = item.relative_to(root_dir)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue  # .cache/huggingface metadata - leave it alone
        target = root_dir / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))
        moved += 1
    # Remove the (now empty) subdirectory skeleton, deepest first.
    for sub in sorted(
        (d for d in root_dir.rglob("*") if d.is_dir()),
        key=lambda d: len(d.parts),
        reverse=True,
    ):
        try:
            next(sub.iterdir())
        except StopIteration:
            sub.rmdir()
        except OSError:
            pass
    return moved


def piper_nested_path(flat_name: str) -> str:
    """Map a flat piper voice file name to its nested HF repo path.

    'hu_HU-anna-medium.onnx' -> 'hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx'
    'en_US-lessac-medium.onnx.json' -> 'en/en_US/lessac/medium/en_US-...'
    """
    stem = flat_name
    for suffix in (".onnx.json", ".onnx"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    lang, _speaker, _quality = stem.split("-", 2)
    top = lang.split("_")[0]
    # e.g. 'hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx'
    return "%s/%s/%s/%s/%s" % (top, lang, _speaker, _quality, flat_name)


# ------------------------------------------------------------ verification ---

def min_bytes_map(entry: Dict[str, Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    raw = entry.get("min_bytes") or {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            try:
                result[str(name)] = int(value)
            except (TypeError, ValueError):
                result[str(name)] = 1
    return result


def verify_entry(root: Path, entry: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems ('' list = entry verified)."""
    problems: List[str] = []
    target_dir = root / str(entry["target_dir"])
    snapshot = bool(entry.get("snapshot"))
    if snapshot:
        probe = target_dir / str(entry.get("snapshot_probe", "config.json"))
        ok, _size = file_ok(probe, 1)
        if not ok:
            problems.append("snapshot probe missing: %s" % probe)
            return problems
        try:
            floor = int(entry.get("min_total_mb", 0)) * 1024 * 1024
        except (TypeError, ValueError):
            floor = 0
        total = dir_total_bytes(target_dir)
        if total < floor:
            problems.append(
                "snapshot payload too small: %s < %d MB floor"
                % (_mb(total), int(entry.get("min_total_mb", 0)))
            )
        return problems
    mins = min_bytes_map(entry)
    for name in entry.get("files", []):
        path = target_dir / str(name)
        ok, size = file_ok(path, mins.get(str(name), 1))
        if not ok:
            if path.is_file():
                problems.append("file too small: %s (%d bytes)" % (path, size))
            else:
                problems.append("file missing: %s" % path)
    return problems


# ------------------------------------------------------------- downloaders --

class ModelDownloader:
    def __init__(self, root: Path, lock: Dict[str, Any], verify_sha: bool,
                 retry_attempts: int, retry_delay: float) -> None:
        self.root = root
        self.lock = lock
        self.verify_sha = verify_sha
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delay = retry_delay
        self.hf_hub_download, self.snapshot_download = import_huggingface_hub()
        # Keep the transfer layer quiet and deterministic: no telemetry.
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        self.downloaded: List[str] = []
        # --- source provenance (reproducibility, v0.3.1) --------------------
        # Which repo ACTUALLY served each component, the requested
        # revision and the resolved commit SHA. Persisted to
        # models/.download_sources.json and merged into INSTALL_MANIFEST.json
        # by scripts/write_install_manifest.py (v0.3.1 field report: the
        # source of every component must be recorded, not just the lock).
        self.source_records: Dict[str, Any] = self._load_source_records()
        self._last_source: Optional[Tuple[str, Optional[str]]] = None

    # -- source provenance -------------------------------------------------

    SOURCE_RECORD_RELPATH = "models/.download_sources.json"

    def _source_record_path(self) -> Path:
        return self.root / self.SOURCE_RECORD_RELPATH

    def _load_source_records(self) -> Dict[str, Any]:
        """Merge base: the existing record file (idempotent re-runs)."""
        path = self._source_record_path()
        if not path.is_file():
            return {"schema_version": 1, "components": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("components"), dict):
                data.setdefault("schema_version", 1)
                return data
        except (OSError, ValueError):
            pass
        return {"schema_version": 1, "components": {}}

    def write_source_records(self) -> Path:
        """Persist the merged source records (best-effort, never raises)."""
        path = self._source_record_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.source_records, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            _print_flush("[WARN] source record not written: %s" % exc)
        return path

    def _resolved_commit(self, repo_id: str,
                         revision: Optional[str]) -> Optional[str]:
        """Best-effort commit SHA for the revision actually downloaded."""
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415 - lazy import

            info = HfApi().repo_info(repo_id=repo_id, revision=revision or "main")
            return getattr(info, "sha", None)
        except Exception:  # noqa: BLE001 - metadata lookup never fails a download
            return None

    def _record_component_source(self, entry: Dict[str, Any]) -> None:
        """Record the ACTUAL download source of one component (on success)."""
        component = str(entry.get("component") or entry.get("repo"))
        repo_id, revision = self._last_source or (
            str(entry.get("repo")), entry.get("pinned_revision"),
        )
        target_dir = self.root / str(entry.get("target_dir", ""))
        files: Dict[str, Dict[str, Any]] = {}
        for name in [str(f) for f in entry.get("files", [])]:
            path = target_dir / name
            if path.is_file():
                try:
                    files[name] = {"size_bytes": path.stat().st_size}
                except OSError:
                    continue
        if not files and target_dir.is_dir():
            # snapshot components: whatever actually landed (docs skipped)
            for path in sorted(target_dir.rglob("*")):
                if (path.is_file() and path.name not in (".gitkeep", "README.md")
                        and ".download_sources.json" not in path.name):
                    rel = path.relative_to(target_dir).as_posix()
                    files[rel] = {"size_bytes": path.stat().st_size}
        resolved = self._resolved_commit(repo_id, revision)
        self.source_records["components"][component] = {
            "source_repo": repo_id,
            "requested_revision": revision or "main",
            "resolved_commit": resolved,
            "target_dir": str(entry.get("target_dir", "")),
            "files": files,
        }
        _print_flush("[SRC ] %s <- %s (revision: %s, commit: %s)"
                     % (component, repo_id, revision or "main",
                        resolved or "unresolved"))
        self._last_source = None

    # -- low level wrappers (retry + resume + local destination + revision) --

    def _hub_file(self, repo_id: str, filename: str, local_dir: Path,
                  revision: Optional[str]) -> bool:
        label = "hf_hub_download %s :: %s" % (repo_id, filename)
        _print_flush("[GET ] %s (revision: %s)" % (label, revision or "main"))

        def run() -> None:
            self.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(local_dir),
            )

        ok = with_retries(label, run, self.retry_attempts, self.retry_delay)
        if ok:
            self.downloaded.append("%s :: %s" % (repo_id, filename))
            self._last_source = (repo_id, revision)
        return ok

    def _hub_snapshot(self, repo_id: str, local_dir: Path,
                      revision: Optional[str]) -> bool:
        label = "snapshot_download %s" % repo_id
        _print_flush("[GET ] %s (revision: %s)" % (label, revision or "main"))

        def run() -> None:
            self.snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(local_dir),
            )

        ok = with_retries(label, run, self.retry_attempts, self.retry_delay)
        if ok:
            self.downloaded.append("%s :: (full repo snapshot)" % repo_id)
            self._last_source = (repo_id, revision)
        return ok

    # -- component handlers --------------------------------------------------

    def _download_llm(self, entry: Dict[str, Any]) -> bool:
        """v0.4.16: the LLM is NOT auto-downloaded (operator-placed GGUF).

        The ONE AND ONLY LLM is Qwen3.6 35B A3B IQ4_XS (~19 GB GGUF,
        tested source hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS).
        It lives wherever the operator keeps it (any drive) and is pointed
        at via the web UI "LLM model" picker (config/llm_model.json),
        scripts/find_qwen_gguf.ps1 -SetEnv, or by placing it at
        models/llm/qwen3.6-35b-a3b/. No auto-download, no mirror chain, no
        fallback profile. Downloading 19 GB unprompted during bootstrap
        would be wrong - this handler only prints the guidance.
        """
        _print_flush(
            "[SKIP] llm: the Qwen3.6 35B A3B IQ4_XS GGUF is OPERATOR-PLACED"
            " (never auto-downloaded). Select it with the web UI 'LLM model'"
            " picker (any drive -> config/llm_model.json) or"
            " scripts/find_qwen_gguf.ps1, or place it at"
            " models/llm/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-IQ4_XS.gguf."
            " Tested source: https://huggingface.co/bartowski/"
            "Qwen_Qwen3.6-35B-A3B-GGUF (tag IQ4_XS)."
        )
        return True

    def _download_snapshot(self, entry: Dict[str, Any]) -> bool:
        target = self.root / str(entry["target_dir"])
        revision = entry.get("pinned_revision") or "main"
        return self._hub_snapshot(str(entry["repo"]), target, revision)

    def _download_vad(self, entry: Dict[str, Any]) -> bool:
        target = self.root / str(entry["target_dir"])
        revision = entry.get("pinned_revision") or "main"
        repo, filename = VAD_PRIMARY
        if self._hub_file(repo, filename, target, revision):
            return True
        repo, filename = VAD_FALLBACK
        _print_flush("[INFO] VAD mirror fallback: silero/silero-vad "
                     "v6.2.1 (nested path + flatten).")
        if self._hub_file(repo, filename, target, "v6.2.1"):
            moved = flatten_nested(target)
            if moved:
                _print_flush("[OK  ] VAD flattened: %d file(s) moved to "
                             "models/vad/silero-vad/." % moved)
            return True
        return False

    def _download_tts(self, entry: Dict[str, Any]) -> bool:
        target = self.root / str(entry["target_dir"])
        revision = entry.get("pinned_revision") or "main"
        repo = str(entry["repo"])
        ok_all = True
        for flat_name in [str(f) for f in entry.get("files", [])]:
            nested = piper_nested_path(flat_name)
            if self._hub_file(repo, nested, target, revision):
                continue
            _print_flush("[INFO] piper flat-path fallback for: %s" % flat_name)
            if self._hub_file(repo, flat_name, target, revision):
                continue
            ok_all = False
        moved = flatten_nested(target)
        if moved:
            _print_flush("[OK  ] piper voices flattened: %d file(s) moved "
                         "to models/tts/piper/." % moved)
        return ok_all

    def _download_generic(self, entry: Dict[str, Any]) -> bool:
        if bool(entry.get("snapshot")):
            return self._download_snapshot(entry)
        target = self.root / str(entry["target_dir"])
        revision = entry.get("pinned_revision") or "main"
        ok_all = True
        for flat_name in [str(f) for f in entry.get("files", [])]:
            if not self._hub_file(str(entry["repo"]), flat_name, target,
                                  revision):
                ok_all = False
        return ok_all

    # -- entry point -----------------------------------------------------

    def process_entry(self, entry: Dict[str, Any], verify_only: bool) -> bool:
        component = str(entry.get("component") or entry.get("repo"))
        target_dir = self.root / str(entry["target_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        problems = verify_entry(self.root, entry)
        if not problems:
            _print_flush("[SKIP] %s: already present and verified "
                         "(idempotent - no network)." % component)
            return True
        if verify_only:
            for problem in problems:
                _print_flush("[FAIL] %s: %s" % (component, problem))
            return False
        _print_flush("")
        _print_flush("---- %s: %s -> %s" % (component, entry.get("repo"),
                                            entry.get("target_dir")))
        handler = {
            "llm": self._download_llm,
            "vad": self._download_vad,
            "tts": self._download_tts,
        }.get(component, self._download_generic)
        if component in ("asr", "embedding"):
            handler = self._download_snapshot
        ok = handler(entry)
        problems = verify_entry(self.root, entry)
        if problems:
            for problem in problems:
                _print_flush("[FAIL] %s: %s" % (component, problem))
            return False
        if ok:
            self._record_component_source(entry)
            _print_flush("[OK  ] %s: verified on disk." % component)
        return ok


# -------------------------------------------------------------------- main --

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M1 model set downloader (huggingface_hub Python API)."
    )
    parser.add_argument("--root", default="",
                        help="repo root (default: parent of this script)")
    parser.add_argument("--only", default="",
                        help="comma-separated components to download/verify")
    parser.add_argument("--verify-only", action="store_true",
                        help="only verify local presence, never download")
    parser.add_argument("--sha256", action="store_true",
                        help="print sha256 of every verified file (slow)")
    parser.add_argument("--retry-attempts", type=int,
                        default=DEFAULT_RETRY_ATTEMPTS,
                        help="download attempts per file (default: 3)")
    parser.add_argument("--retry-delay", type=float,
                        default=DEFAULT_RETRY_BASE_DELAY,
                        help="first retry delay in seconds (default: 5)")
    args = parser.parse_args(argv)

    started = time.time()
    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent.parent

    _print_flush("=" * 62)
    _print_flush("VoiceMem model download (huggingface_hub Python API)")
    _print_flush("Root        : %s" % root)
    _print_flush("Mode        : %s" % ("verify-only" if args.verify_only
                                       else "download + verify"))
    _print_flush("PyTorch-free: no CLI, no PATH dependency, no output parsing.")
    _print_flush("=" * 62)

    lock, _lock_path = load_lock(root)
    entries = lock["models"]
    only = [c.strip() for c in args.only.split(",") if c.strip()]
    if only:
        known = {str(e.get("component")) for e in entries}
        unknown = [c for c in only if c not in known]
        if unknown:
            _print_flush("[FAIL] --only has unknown components: %s"
                         % ", ".join(unknown))
            return 2
        entries = [e for e in entries if str(e.get("component")) in only]
        _print_flush("[INFO] --only active: %s" % ", ".join(only))

    downloader = ModelDownloader(
        root=root,
        lock=lock,
        verify_sha=args.sha256,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
    )

    results: Dict[str, bool] = {}
    for entry in entries:
        component = str(entry.get("component") or entry.get("repo"))
        try:
            results[component] = downloader.process_entry(
                entry, verify_only=args.verify_only
            )
        except KeyboardInterrupt:
            _print_flush("")
            _print_flush("[STOP] interrupted by user - partial transfers "
                         "resume on the next run.")
            return 130

    # -- final verification table (exit code depends ONLY on real files) -----
    _print_flush("")
    _print_flush("---- File verification (%d component(s))" % len(entries))
    failures: List[str] = []
    for entry in entries:
        component = str(entry.get("component") or entry.get("repo"))
        target_dir = root / str(entry["target_dir"])
        problems = verify_entry(root, entry)
        status = "OK  " if not problems else "FAIL"
        if problems:
            failures.append(component)
        _print_flush("  %s : %s -> %s" % (status, component, target_dir))
        if bool(entry.get("snapshot")):
            _print_flush("         payload: %s (probe: %s)"
                         % (_mb(dir_total_bytes(target_dir)),
                            entry.get("snapshot_probe")))
        else:
            mins = min_bytes_map(entry)
            for name in entry.get("files", []):
                path = target_dir / str(name)
                ok, size = file_ok(path, mins.get(str(name), 1))
                line = "         %s (%s)" % (name, _mb(size) if ok else
                                            "MISSING/TOO SMALL")
                if args.sha256 and ok:
                    line += " sha256=%s" % sha256_of(path)
                _print_flush(line)

    _print_flush("")
    _print_flush("Downloaded this run : %d item(s)"
                 % len(downloader.downloaded))
    # Source provenance (v0.3.1): persist which repo ACTUALLY served each
    # component - the install manifest merges this into every model entry.
    if downloader.source_records["components"]:
        record_path = downloader.write_source_records()
        _print_flush("Source record      : %s" % record_path)
    elapsed = time.time() - started
    _print_flush("Elapsed             : %.1f s" % elapsed)
    if failures:
        _print_flush("MODELS_FAILED: %s" % ", ".join(failures))
        _print_flush(
            "FIX: re-run START.bat (idempotent - finished files are "
            "skipped, broken transfers resume, mirrors are retried)."
        )
        return 1
    _print_flush("ALL_MODELS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
