"""Setup verification for the M1 runtime environment.

Machine: ANY. This script imports cleanly in the development sandbox (no GPU,
no models) and on the target machine (Windows 11 + RTX 5070). Heavy imports
(torch) are guarded; the llama-server /health endpoint is probed with urllib
(2 s timeout) so no extra dependency is needed.

All asset paths are resolved through AgentConfig properties, so they
AUTOMATICALLY follow the M0 directory layout (models/asr/..., models/llm/...,
models/tts/piper, models/vad/silero-vad, models/embedding/...) - there are no
hardcoded legacy paths here. The HF environment variables (HF_HOME,
HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE) are also probed and reported as
warnings when missing (the offline guarantee depends on them).

Note: the primary installer/verifier pair is scripts/install_m1.ps1 ->
scripts/verify_m1.ps1; this script is the low-level asset lister.

Exit codes:
  0 - every on-disk file asset of AgentConfig.check_runtime_assets() exists
  1 - at least one file asset is MISSING
  2 - invalid configuration (AgentConfig.validate() reported errors)

Usage:
  python scripts/verify_setup.py [--root PATH] [--config PATH] [--json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import AgentConfig  # noqa: E402  (pure stdlib, safe everywhere)

ASSET_PATHS: dict[str, Callable[[AgentConfig], Optional[Path]]] = {
    # v0.4.17: llm_model_file is Optional (None = no model configured).
    "llama_model": lambda cfg: cfg.llm_model_file,
    "piper_executable": lambda cfg: cfg.piper_exe_path,
    "silero_vad": lambda cfg: cfg.silero_vad_path,
    "hu_voice": lambda cfg: cfg.voices_dir / f"{cfg.tts_hu_voice}.onnx",
    "en_voice": lambda cfg: cfg.voices_dir / f"{cfg.tts_en_voice}.onnx",
    "asr_model": lambda cfg: cfg.asr_model_dir / "config.json",
    # New M0 layout key (optional until AgentConfig adds it to check_runtime_assets).
    "embedding_model": lambda cfg: (
        getattr(cfg, "embedding_model_dir", cfg.models_dir / "embedding" / "multilingual-e5-small")
        / "config.json"
    ),
    # New M0 layout key (optional): memory root (memory/sqlite, qdrant, backups).
    "memory_root": lambda cfg: getattr(
        cfg, "memory_root_path", cfg.root / "memory"
    ),
}

# HF environment variables the offline guarantee relies on.
ENV_PROBE_NAMES = ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VOICEMEM_HOME")


def build_config(args: argparse.Namespace) -> tuple[AgentConfig, list[str]]:
    """Build the AgentConfig from YAML (optional), env vars and --root override."""
    warnings: list[str] = []
    cfg: AgentConfig
    if args.config:
        try:
            cfg = AgentConfig.from_yaml(Path(args.config))
        except (TypeError, ValueError, RuntimeError, FileNotFoundError) as exc:
            warnings.append(
                f"could not load YAML config {args.config}: {exc}; "
                "falling back to env + defaults"
            )
            cfg = AgentConfig()
            cfg.apply_env()
    else:
        cfg = AgentConfig()
        cfg.apply_env()
    if args.root:  # explicit CLI override wins over env / defaults
        cfg.root = Path(args.root)
    return cfg, warnings


def probe_env() -> dict[str, bool]:
    """Presence check of the HF/offline environment variables."""
    return {name: bool(os.environ.get(name)) for name in ENV_PROBE_NAMES}


def probe_llama_server(cfg: AgentConfig) -> tuple[bool, str]:
    """GET {host}:{port}/health with a 2 s timeout via stdlib urllib."""
    url = cfg.llama_server_health_url
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:  # nosec - fixed loopback URL
            ok = 200 <= resp.status < 300
            return ok, f"HTTP {resp.status}" if ok else f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - any transport error means unreachable
        return False, type(exc).__name__


def probe_torch() -> dict[str, Any]:
    """Guarded torch/CUDA probe (never imported in the sandbox)."""
    info: dict[str, Any] = {"torch": "NOT INSTALLED", "torch_version": None, "cuda": "n/a"}
    try:
        import torch  # heavy, guarded
    except ImportError:
        return info
    info["torch"] = "OK"
    info["torch_version"] = torch.__version__
    info["cuda"] = "OK" if torch.cuda.is_available() else "NO CUDA DEVICE"
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    return info


def probe_voicemem() -> bool:
    """Check the voicemem package without importing (executing) it."""
    return importlib.util.find_spec("voicemem") is not None


def probe_voicemem_controlled(root: Path) -> dict[str, Any]:
    """v0.5.0: controlled-source pin probe (read-only, no package import).

    Checks the repository carries the controlled vendor tree + pin file and
    that the pin records the expected upstream commit. The full runtime
    (import path + CONTROLLED_UPSTREAM_COMMIT) verification happens in
    verify_m1.ps1 and scripts/verify_voicemem_pin.py with the package
    installed; this probe works even in the sandbox without voicemem.
    """
    expected = "e8384e087bd2f44eb05fc7ae1a3c525ea8244179"
    vendor_pkg = root / "vendor" / "voicemem" / "voicemem" / "__init__.py"
    pin_file = root / "VOICEMEM_PIN.json"
    out: dict[str, Any] = {
        "vendor_source": vendor_pkg.is_file(),
        "pin_file": pin_file.is_file(),
        "upstream_commit": None,
        "upstream_tag": None,
        "expected_commit": expected,
        "pin_matches": False,
    }
    if pin_file.is_file():
        try:
            pin = json.loads(pin_file.read_text(encoding="utf-8"))
            prov = pin.get("provenance") or {}
            out["upstream_commit"] = prov.get("upstream_commit")
            out["upstream_tag"] = prov.get("upstream_tag")
            out["pin_matches"] = prov.get("upstream_commit") == expected
        except (OSError, ValueError):
            pass
    return out


def probe_emotion(cfg: AgentConfig) -> dict[str, Any]:
    """M2 emotion analyzer probe: funasr presence + model files + loadability.

    Never imports funasr in the sandbox (guarded); a missing model or
    missing funasr is a DEGRADED (M1) state, not an error.
    """
    info: dict[str, Any] = {
        "funasr": "NOT INSTALLED",
        "model_files": False,
        "available": False,
        "detail": "",
    }
    try:
        import funasr  # heavy, guarded

        info["funasr"] = f"OK ({getattr(funasr, '__version__', '?')})"
    except ImportError as exc:
        info["detail"] = f"funasr import failed: {exc}"
        return info
    except Exception as exc:  # noqa: BLE001
        info["detail"] = f"funasr import raised: {exc}"
        return info
    from app.emotion import EmotionAnalyzer

    analyzer = EmotionAnalyzer(
        cfg.emotion_model_dir,
        window_s=cfg.emotion_window_s,
        sample_rate=cfg.sample_rate,
    )
    info["model_files"] = analyzer.files_present()
    info["available"] = analyzer.is_available()
    if not info["model_files"]:
        info["detail"] = f"model missing: {cfg.emotion_model_dir}"
    return info


def probe_speaker(cfg: AgentConfig) -> dict[str, Any]:
    """M3 speaker recognizer probe: speechbrain presence + model files.

    Never imports speechbrain in the sandbox (guarded); a missing model or
    missing speechbrain is a DEGRADED (M1/M2) state, not an error. The
    registry presence is informational (0 registered speakers = every
    voice falls back to the default user id until registration).
    """
    info: dict[str, Any] = {
        "speechbrain": "NOT INSTALLED",
        "model_files": False,
        "available": False,
        "registered_speakers": 0,
        "detail": "",
    }
    try:
        import speechbrain  # heavy, guarded

        info["speechbrain"] = f"OK ({getattr(speechbrain, '__version__', '?')})"
    except ImportError as exc:
        info["detail"] = f"speechbrain import failed: {exc}"
        return info
    except Exception as exc:  # noqa: BLE001
        info["detail"] = f"speechbrain import raised: {exc}"
        return info
    from app.speaker import SpeakerEmbedder, SpeakerRegistry

    embedder = SpeakerEmbedder(
        cfg.speaker_model_dir,
        window_s=cfg.speaker_window_s,
        sample_rate=cfg.sample_rate,
    )
    info["model_files"] = embedder.files_present()
    info["available"] = embedder.is_available()
    if not info["model_files"]:
        info["detail"] = f"model missing: {cfg.speaker_model_dir}"
    try:
        registry = SpeakerRegistry(cfg.speaker_registry_file)
        info["registered_speakers"] = registry.count()
    except Exception:  # noqa: BLE001 - informational only
        pass
    return info


def asset_path(name: str, cfg: AgentConfig) -> str:
    """Resolve a display path for an asset key (tolerant to new keys)."""
    resolver = ASSET_PATHS.get(name)
    if resolver is None:
        return "(path resolved inside AgentConfig)"
    # v0.4.17: a None path means "no model configured" — never show the
    # string "None" (it would look like a real path named None).
    value = resolver(cfg)
    return str(value) if value is not None else "(not configured)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="runtime root override (default: env/OS default)")
    parser.add_argument("--config", help="optional YAML config path (config/voicemem_config.yaml)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    cfg, warnings = build_config(args)
    errors = cfg.validate()
    assets = cfg.check_runtime_assets()
    server_ok, server_detail = probe_llama_server(cfg)
    torch_info = probe_torch()
    voicemem_ok = probe_voicemem()
    vm_pin = probe_voicemem_controlled(Path(cfg.root))
    emotion_info = probe_emotion(cfg)
    speaker_info = probe_speaker(cfg)
    env_info = probe_env()

    missing_env = [name for name, present in env_info.items() if not present]
    for name in missing_env:
        warnings.append(
            f"environment variable {name} is not set - the offline mode / "
            "path resolution relies on it (run scripts/install_m1.ps1 or "
            "dot-source config/env.local.ps1)"
        )

    report: dict[str, Any] = {
        "root": str(cfg.root),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "platform": platform.platform(),
        "config_errors": errors,
        "warnings": warnings,
        "assets": {},
        "llama_server": {
            "url": cfg.llama_server_health_url,
            "reachable": server_ok,
            "detail": server_detail,
        },
        "voicemem_package": voicemem_ok,
        "voicemem_controlled": vm_pin,
        "torch": torch_info,
        "emotion": emotion_info,
        "speaker": speaker_info,
        "llm_model_file": str(cfg.llm_model_file) if cfg.llm_model_file else "",
        "env": env_info,
        "hf_home": str(Path(env_info.get("HF_HOME", "") or (Path(cfg.root) / "models" / "hf"))),
    }
    for name, ok in assets.items():
        report["assets"][name] = {"ok": ok, "path": asset_path(name, cfg)}

    all_assets_ok = all(assets.values())
    if errors:
        exit_code = 2
    elif all_assets_ok:
        exit_code = 0
    else:
        exit_code = 1
    report["exit_code"] = exit_code

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return exit_code

    # Human-readable checklist table.
    print("=" * 72)
    print("VoiceMemAgent M1 - setup verification")
    print("=" * 72)
    print(f"root          : {cfg.root}")
    print(f"python        : {report['python']}")
    print(f"platform      : {report['platform']}")
    print("-" * 72)
    print(f"{'asset':16s} {'status':10s} path")
    for name, ok in assets.items():
        status = "OK" if ok else "MISSING"
        print(f"{name:16s} {status:10s} {asset_path(name, cfg)}")
    print("-" * 72)
    print(f"voicemem pkg  : {'OK' if voicemem_ok else 'NOT INSTALLED'} (install: scripts\\install_m1.ps1)")
    pin_state = "OK" if (vm_pin["vendor_source"] and vm_pin["pin_file"] and vm_pin["pin_matches"]) else "MISSING/PIN-MISMATCH"
    print(f"voicemem pin  : {pin_state} (controlled vendor/voicemem + VOICEMEM_PIN.json -> {vm_pin['upstream_commit']})")
    emotion_state = (
        "OK (M2 active)" if emotion_info["available"] else
        "DEGRADED (M1 mode - funasr or model missing)"
    )
    print(
        f"emotion (M2)  : {emotion_state} funasr={emotion_info['funasr']} "
        f"model_files={'OK' if emotion_info['model_files'] else 'MISSING'} "
        f"dir={cfg.emotion_model_dir}"
    )
    speaker_state = (
        "OK (M3 active)" if speaker_info["available"] else
        "DEGRADED (M1/M2 mode - speechbrain or model missing)"
    )
    print(
        f"speaker (M3)  : {speaker_state} speechbrain={speaker_info['speechbrain']} "
        f"model_files={'OK' if speaker_info['model_files'] else 'MISSING'} "
        f"registered={speaker_info['registered_speakers']} dir={cfg.speaker_model_dir}"
    )
    print(f"llama-server  : {'OK' if server_ok else 'UNREACHABLE'} ({server_detail}) at {cfg.llama_server_health_url}")
    print(f"torch         : {torch_info['torch']}"
          + (f" ({torch_info['torch_version']})" if torch_info['torch_version'] else ""))
    print(f"cuda          : {torch_info['cuda']}"
          + (f" ({torch_info.get('cuda_device', '')})" if torch_info.get("cuda_device") else ""))
    print(f"hf env        : " + ", ".join(
        f"{name}={'set' if present else 'NOT SET'}" for name, present in env_info.items()))
    for warning in warnings:
        print(f"WARNING       : {warning}")
    for error in errors:
        print(f"CONFIG ERROR  : {error}")
    print("-" * 72)
    if exit_code == 0:
        print("RESULT: all file assets present - setup looks complete.")
        print("Next: scripts\\start_agent.ps1 (it auto-starts the llama-server)")
    else:
        print("RESULT: incomplete - run scripts\\install_m1.ps1 (or scripts\\download_models.ps1).")
    print("=" * 72)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
