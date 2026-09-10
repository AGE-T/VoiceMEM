"""Write the M0 INSTALL_MANIFEST.json (spec Section 23, reproducibility).

The manifest records EVERYTHING needed to re-create the same M0 environment:
project version + git commit, OS, Python, GPU (via torch when available),
the voicemem pinned clone, every model file under models/ (size + sha256),
the installed Python package versions and the offline flag state.

Design:
- Pure stdlib (``platform``, ``hashlib``, ``json``, ``importlib.metadata``);
  ``torch`` is imported OPTIONALLY inside a function (sandbox has no torch).
- The whole logic lives in ``build_manifest(root, fake_gpu=False)`` so unit
  tests can drive it against a temporary directory; ``main()`` only parses
  CLI arguments and writes the JSON file (thin I/O layer).
- Deterministic output: ``json.dumps(..., indent=2, sort_keys=True)`` and a
  stable (path-sorted) model list, so two runs on the same tree produce the
  same bytes apart from ``generated_at``.

Usage:
    python scripts/write_install_manifest.py                 # repo root, default out
    python scripts/write_install_manifest.py --out other.json
    python scripts/write_install_manifest.py --fake-gpu      # GPU block with fake data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional

#: Files inside models/ that are repo structure docs / provenance records,
#: NOT model assets (``.download_sources.json`` is written by the downloader
#: to record which repo ACTUALLY served each component - mirror chains!).
_SKIPPED_FILENAMES = frozenset({".gitkeep", "README.md", ".download_sources.json"})

#: Subdirectory of models/ that is a CACHE root (not an asset component);
#: its contents are huggingface_hub internals, huge and non-authoritative.
_SKIPPED_MODEL_SUBDIRS = frozenset({"hf"})

#: Known component -> Hugging Face repo mapping (M0 Section 7 layout).
_HF_REPO_BY_COMPONENT = {
    "asr": "Qwen/Qwen3-ASR-0.6B",
    "llm": "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
    "embedding": "intfloat/multilingual-e5-small",
    "vad": "snakers4/silero-vad",
    "tts": "rhasspy/piper-voices",
    "emotion": None,  # M2 placeholder — no model may ever live here in M0/M1
}

_SHA256_CHUNK = 1 << 20  # 1 MiB chunks: large GGUF files stay memory-safe


def build_manifest(root: Path, fake_gpu: bool = False) -> dict[str, Any]:
    """Assemble the manifest dict for the given project ``root``.

    Never raises for missing optional pieces: git, voicemem, torch and the
    metadata API all degrade to ``None``/empty values on failure.
    """
    root = Path(root).resolve()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": {
            "name": "voicemem-agent",
            "version": _read_version(root),
            "git_commit": _git_commit(root),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "gpu": _gpu_block(fake_gpu),
        "torch": _torch_block(fake_gpu),
        "voicemem": _voicemem_block(root),
        "models": _collect_models(root),
        "dependencies": _installed_dependencies(),
        "offline": {
            "hf_hub_offline": _env_flag("HF_HUB_OFFLINE"),
            "transformers_offline": _env_flag("TRANSFORMERS_OFFLINE"),
        },
    }


def write_manifest(manifest: dict[str, Any], out_path: Path) -> Path:
    """Thin I/O layer: dump the manifest as sorted, indented JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def _read_version(root: Path) -> str:
    """Project version — the VERSION file is the single source of truth."""
    version_file = root / "VERSION"
    if version_file.is_file():
        version = version_file.read_text(encoding="utf-8").strip()
        if version:
            return version
    return "0.0.0-unknown"


def _git_commit(repo_dir: Path) -> Optional[str]:
    """HEAD commit sha of ``repo_dir`` (None when not a git checkout)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _gpu_block(fake_gpu: bool) -> Optional[dict[str, Any]]:
    """GPU description: fake data, real torch probe, or None (no torch)."""
    if fake_gpu:
        return {
            "name": "NVIDIA GeForce RTX 5070",
            "cuda_available": True,
            "compute_capability": "12.0",
            "total_vram_gb": 12.0,
        }
    torch = _import_torch()
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return {
                "name": platform.node() or "unknown",
                "cuda_available": False,
                "compute_capability": None,
                "total_vram_gb": None,
            }
        props = torch.cuda.get_device_properties(0)
        major, minor = getattr(props, "major", 0), getattr(props, "minor", 0)
        return {
            "name": props.name,
            "cuda_available": True,
            "compute_capability": f"{major}.{minor}",
            "total_vram_gb": round(props.total_memory / (1024 ** 3), 1),
        }
    except Exception:  # noqa: BLE001 - torch probing must never crash the manifest
        return None


def _torch_block(fake_gpu: bool) -> Optional[dict[str, Any]]:
    """Torch version + CUDA build string (None without torch / fake data)."""
    if fake_gpu:
        return {"version": "2.7.0+cu128", "cuda": "12.8"}
    torch = _import_torch()
    if torch is None:
        return None
    try:
        return {
            "version": str(torch.__version__),
            "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        }
    except Exception:  # noqa: BLE001
        return None


def _import_torch() -> Any:
    """Optional torch import (sandbox has none — returns None, never raises)."""
    try:
        import torch  # noqa: PLC0415 - deliberately lazy heavy import

        return torch
    except ImportError:
        return None


def _voicemem_block(root: Path) -> dict[str, Any]:
    """Controlled VoiceMem source state (v0.5.0 ownership model).

    The vendor tree ``vendor/voicemem`` is OUR controlled fork of
    xzf-thu/VoiceMem — it ships with the repository and the release ZIP; the
    installer pip-installs it from there (NO upstream clone, NO fetch, NO
    main-branch fallback). Identity facts (all recorded explicitly; the pip
    package metadata version is NOT an identity marker):

      * ``upstream_commit``  - the pinned upstream base commit
        (authoritative identity; from VOICEMEM_PIN.json, cross-checked
        against the runtime attribute ``voicemem.CONTROLLED_UPSTREAM_COMMIT``)
      * ``upstream_tag``     - the pinned upstream tag (v0.0.1)
      * ``package_version``  - Python package metadata (``voicemem`` dist,
        e.g. 0.2.3 — upstream never bumped it; informational only)
      * ``runtime_import_path``  - the resolved ``voicemem.__file__`` the
        installed environment actually imports
      * ``pin_verified``     - True when the import path is under
        vendor/voicemem AND the runtime commit attribute matches the pin
      * ``local_patches``    - the controlled-fork patch ledger (IDs)

    Legacy keys ``version`` / ``commit`` / ``requested_ref`` / ``git_tag`` /
    ``git_commit`` are preserved with pin-derived values so old readers keep
    working (``git_tag``/``git_commit`` mirror the pin because the vendor
    tree is deliberately NOT a git checkout).
    """
    vendor_dir = root / "vendor" / "voicemem"
    if not vendor_dir.is_dir():
        return {"version": "not_installed", "commit": None}
    pin = _voicemem_pin(root)
    upstream_commit = pin.get("upstream_commit")
    upstream_tag = pin.get("upstream_tag")
    try:
        version = importlib_metadata.version("voicemem")
    except importlib_metadata.PackageNotFoundError:
        version = "0.2.3-vendor-controlled"
    runtime_path, runtime_commit, pin_verified = _voicemem_runtime_state(
        vendor_dir, upstream_commit
    )
    return {
        "version": str(version),
        "commit": upstream_commit,
        "requested_ref": upstream_tag,
        "git_tag": upstream_tag,
        "git_commit": upstream_commit,
        "package_version": str(version),
        "controlled": bool(pin.get("status") == "controlled-fork"),
        "upstream_repo": pin.get("upstream_repo"),
        "upstream_commit": upstream_commit,
        "upstream_tag": upstream_tag,
        "pin_file": "VOICEMEM_PIN.json",
        "runtime_import_path": runtime_path,
        "runtime_commit_attr": runtime_commit,
        "pin_verified": pin_verified,
        "local_patches": [p.get("id") for p in pin.get("local_patches", [])],
    }


def _voicemem_pin(root: Path) -> dict[str, Any]:
    """Read VOICEMEM_PIN.json ({} when absent/corrupt — never raises)."""
    pin_file = root / "VOICEMEM_PIN.json"
    if not pin_file.is_file():
        return {}
    try:
        data = json.loads(pin_file.read_text(encoding="utf-8"))
        prov = data.get("provenance") or {}
        return {
            "status": data.get("status"),
            "upstream_repo": prov.get("upstream_repo"),
            "upstream_commit": prov.get("upstream_commit"),
            "upstream_tag": prov.get("upstream_tag"),
            "local_patches": data.get("local_patches") or [],
        }
    except (OSError, ValueError):
        return {}


def _voicemem_runtime_state(
    vendor_dir: Path, pin_commit: Optional[str]
) -> tuple[Optional[str], Optional[str], bool]:
    """Import voicemem in THIS interpreter and resolve its true identity.

    Returns ``(voicemem.__file__, CONTROLLED_UPSTREAM_COMMIT, verified)``.
    The import is cheap (the package is PEP 562 lazy); on any failure the
    tuple is (None, None, False) — the manifest then records the mismatch
    for verify_m1 to fail loudly on.
    """
    try:
        import voicemem  # noqa: PLC0415 - identity probe, executed at manifest time
    except Exception:  # noqa: BLE001 - any import failure is a state to record
        return None, None, False
    try:
        path = str(Path(voicemem.__file__).resolve())  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return None, None, False
    runtime_commit = getattr(voicemem, "CONTROLLED_UPSTREAM_COMMIT", None)
    under_vendor = ("vendor/voicemem" in path.replace("\\", "/")) or (
        "vendor\\voicemem" in path
    )
    verified = bool(
        pin_commit
        and runtime_commit == pin_commit
        and under_vendor
        and bool(getattr(voicemem, "CONTROLLED_FORK", False))
    )
    return path, runtime_commit, verified


def _git_exact_tag(repo_dir: Path) -> Optional[str]:
    """Tag pointing EXACTLY at HEAD (None when HEAD is untagged/detached)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "describe", "--tags",
             "--exact-match", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _collect_models(root: Path) -> list[dict[str, Any]]:
    """Every model asset file under root/models (skipping docs + hf cache).

    Deterministic order: sorted by POSIX path. Component name = the first
    path segment below ``models/`` (asr/llm/tts/vad/embedding/emotion).
    v0.3.1: every entry is enriched with the ACTUAL download source
    (``source_repo`` / ``source_revision`` / ``source_commit``) when the
    downloader wrote ``models/.download_sources.json`` (mirror-chain
    provenance - the field report's Qwen GGUF came from unsloth, not the
    canonical Qwen repo).
    """
    models_dir = root / "models"
    sources = _load_download_sources(root)
    if not models_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(models_dir.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(models_dir)
        if rel.parts and rel.parts[0] in _SKIPPED_MODEL_SUBDIRS:
            continue
        if path.name in _SKIPPED_FILENAMES:
            continue
        component = rel.parts[0] if rel.parts else "unknown"
        record = sources.get(component)
        entry: dict[str, Any] = {
            "component": component,
            "path": (Path("models") / rel).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "hf_repo": _HF_REPO_BY_COMPONENT.get(component),
        }
        if isinstance(record, dict):
            entry["source_repo"] = record.get("source_repo")
            entry["source_revision"] = record.get("requested_revision")
            entry["source_commit"] = record.get("resolved_commit")
        entries.append(entry)
    return entries


def _load_download_sources(root: Path) -> dict[str, Any]:
    """Component -> actual-download-source map from the downloader record."""
    path = root / "models" / ".download_sources.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    components = data.get("components") if isinstance(data, dict) else None
    return components if isinstance(components, dict) else {}


def _sha256(path: Path) -> str:
    """Chunked sha256 hex digest (1 MiB reads — GGUF-safe)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_SHA256_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _installed_dependencies() -> dict[str, str]:
    """Installed distribution versions via importlib.metadata (pip-free)."""
    try:
        return {
            str(dist.metadata["Name"]): dist.version
            for dist in importlib_metadata.distributions()
            if dist.metadata["Name"]
        }
    except Exception:  # noqa: BLE001 - metadata API must never crash the manifest
        return {}


def _env_flag(name: str) -> bool:
    """Truthy env flag ('1'/'true'/'yes', case-insensitive)."""
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: --root (default: repo root above scripts/), --out, --fake-gpu."""
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="project root to describe (default: the repo containing this script)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON path (default: <root>/INSTALL_MANIFEST.json)",
    )
    parser.add_argument(
        "--fake-gpu",
        action="store_true",
        help="fill the GPU/torch blocks with fake RTX 5070 data (testing)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    out_path = args.out if args.out is not None else root / "INSTALL_MANIFEST.json"

    manifest = build_manifest(root, fake_gpu=args.fake_gpu)
    write_manifest(manifest, out_path)

    models = manifest["models"]
    print(f"INSTALL_MANIFEST written: {out_path}")
    print(f"  project : {manifest['project']['name']} v{manifest['project']['version']}")
    print(f"  models  : {len(models)} file(s), {sum(m['size_bytes'] for m in models)} bytes")
    print(f"  deps    : {len(manifest['dependencies'])} distribution(s)")
    print(f"  gpu     : {'fake' if args.fake_gpu else ('real' if manifest['gpu'] else 'unavailable')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
