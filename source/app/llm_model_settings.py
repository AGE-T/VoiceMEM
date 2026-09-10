"""Persisted LLM model (GGUF) selection — v0.4.16 "portable model path".

WHY: GGUF files are tens of gigabytes; the operator keeps them wherever
there is space (D:\\AI\\Models, E:\\LLM, an Ollama blob store, an LM Studio
dir). The configuration therefore stores ONLY the absolute filesystem path
— the file is NEVER copied into the project, never uploaded, never
duplicated. llama-server reads it in place on whatever drive it lives on.

PERSISTENCE (the exact pattern of app/voice_settings.py): the selection is
a small JSON runtime-state file, ``config/llm_model.json`` next to
``voicemem_config.yaml``:

    {
      "llm_model_path": "D:\\AI\\Models\\Qwen\\Qwen3.6-35B-A3B-IQ4_XS.gguf",
      "selected_at": "2026-09-05T12:34:56",
      "model_name": "Qwen_Qwen3.6-35B-A3B",
      "quant": "IQ4_XS",
      "file_size": 20472638160
    }

It is written atomically (temp file + ``os.replace``) and read leniently
(corrupt/missing -> empty selection, NEVER blocks startup). The release
build EXCLUDES it from the ZIP (runtime state, like
``config/voice_settings.json``), so it survives BOTH an application
restart AND an extract-over upgrade. The env ``VOICEMEM_LLM_MODEL_SETTINGS``
redirects the file (tests, exotic installs) without touching the layout.

PRECEDENCE (single source of truth, mirrored by scripts/start_llama_server.ps1
and scripts/verify_m1.ps1):
  1. ``config/llm_model.json`` -> ``llm_model_path`` (UI selection) — wins
     over the profile so the operator never edits PowerShell by hand;
  2. ``LLAMA_MODEL_PATH`` env var (config/env.local.ps1 profile default);
  3. a GGUF ACTUALLY PRESENT in models/llm/qwen3.6-35b-a3b/ (magic-
     validated, any file name, never a guessed path). When the directory
     holds no complete GGUF there is NO default — the honest "no model
     configured" state (v0.4.17: the v0.4.16 chain returned a project-
     default PATH THAT DID NOT EXIST, which the UI displayed as if it were
     the configured model — misleading; removed per field report).

Clearing the selection (UI "Reset to profile") returns control to 2/3.

Pure stdlib; nothing here touches the network or copies model bytes.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Settings file (relative to the repo root), next to voicemem_config.yaml.
SETTINGS_RELPATH = Path("config") / "llm_model.json"

#: Marker the llama-server starter writes on EVERY launch with the exact
#: resolved model path; the web UI cross-checks it against this selection
#: ("llama-server loads the SAME file").
RESOLVED_MODEL_MARKER = Path("logs") / "llama-server.resolved-model.json"

#: Source labels of the resolution chain (UI + logs use these).
SOURCE_USER = "user-selection"
SOURCE_ENV = "env-profile"
SOURCE_DEFAULT = "project-default"
SOURCE_NONE = "none"

#: Where an operator may place the LLM GGUF by hand (the ONLY project-local
#: default location; the file name inside is irrelevant — magic decides).
OPERATOR_LLM_DIR = Path("models") / "llm" / "qwen3.6-35b-a3b"


def settings_path(root: Path) -> Path:
    """Absolute settings-file path (env override aware, lenient)."""
    env_path = os.environ.get("VOICEMEM_LLM_MODEL_SETTINGS", "").strip()
    if env_path:
        return Path(env_path)
    return Path(root) / SETTINGS_RELPATH


class LlmModelSettings:
    """The persisted user selection of the LLM GGUF file (path only)."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else None
        #: Absolute path of the selected GGUF ("" = no user selection; the
        #: env-profile / project default applies).
        self.llm_model_path: str = ""
        #: Capture metadata at select time (display + sanity, never trust
        #: for validation — the file is re-validated live on every read).
        self.selected_at: str = ""
        self.model_name: str = ""
        self.quant: str = ""
        self.file_size: int = 0
        self._path_override: Optional[Path] = None

    # -- construction ----------------------------------------------------------- #

    @classmethod
    def load(cls, root: Path) -> "LlmModelSettings":
        """Read the persisted selection (lenient); defaults when absent."""
        path = settings_path(Path(root))
        data: dict[str, Any] = {}
        try:
            if path.is_file():
                raw = json.loads(path.read_text("utf-8"))
                if isinstance(raw, dict):
                    data = raw
        except Exception as exc:  # noqa: BLE001 - never block startup
            logger.warning("llm model settings unreadable (%s): %s", path, exc)
        settings = cls(root=root)
        settings.llm_model_path = str(data.get("llm_model_path", "") or "").strip()
        settings.selected_at = str(data.get("selected_at", "") or "")
        settings.model_name = str(data.get("model_name", "") or "")
        settings.quant = str(data.get("quant", "") or "")
        try:
            settings.file_size = int(data.get("file_size", 0) or 0)
        except (TypeError, ValueError):
            settings.file_size = 0
        if os.environ.get("VOICEMEM_LLM_MODEL_SETTINGS", "").strip():
            settings._path_override = path
        return settings

    # -- persistence --------------------------------------------------------------- #

    @property
    def path(self) -> Optional[Path]:
        if self._path_override is not None:
            return self._path_override
        return (self.root / SETTINGS_RELPATH) if self.root else None

    def save(self) -> Path:
        """Persist the selection (atomic write). Raises OSError."""
        target = self.path
        if target is None:
            raise OSError("llm model settings have no root directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "llm_model_path": self.llm_model_path,
            "selected_at": self.selected_at,
            "model_name": self.model_name,
            "quant": self.quant,
            "file_size": self.file_size,
        }
        fd, tmp = tempfile.mkstemp(
            dir=str(target.parent), prefix=".llm_model.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.info("llm model selection persisted: %r", payload)
        return target

    def clear(self) -> Path:
        """Reset to the profile default: empty the persisted selection."""
        self.llm_model_path = ""
        self.selected_at = ""
        self.model_name = ""
        self.quant = ""
        self.file_size = 0
        return self.save()

    # -- application ----------------------------------------------------------------- #

    def apply_selection(
        self, path: str, model_name: str = "", quant: str = "", file_size: int = 0
    ) -> None:
        """Record + persist a validated selection (called by the REST layer)."""
        self.llm_model_path = str(path).strip()
        self.selected_at = datetime.now().isoformat(timespec="seconds")
        self.model_name = str(model_name or "")
        self.quant = str(quant or "")
        self.file_size = int(file_size or 0)
        self.save()

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_model_path": self.llm_model_path,
            "selected_at": self.selected_at,
            "model_name": self.model_name,
            "quant": self.quant,
            "file_size": self.file_size,
        }


def user_selected_llm_model_path(root: Path) -> Optional[str]:
    """The persisted user selection, or None when empty/unset (lenient).

    Read on every call (tiny file) so a selection made in the UI is visible
    to the very next ``AgentConfig.llm_model_file`` resolution — including
    the config-bridge subprocesses (start_llama_server.ps1 / verify_m1.ps1),
    which import the app fresh.
    """
    try:
        settings = LlmModelSettings.load(Path(root))
    except Exception:  # noqa: BLE001
        return None
    return settings.llm_model_path or None


def operator_default_candidates(root: Path) -> list[str]:
    """Complete, loadable GGUFs the operator placed in the project dir.

    v0.4.17 "no misleading default": the project default is whatever VALID
    GGUF (magic-validated via the header, extension-agnostic — an Ollama
    blob works) is ACTUALLY PRESENT in ``models/llm/qwen3.6-35b-a3b/``.
    Split shards are excluded (llama-server cannot load one directly).
    Zero candidates -> NO default; two or more -> ambiguous (the operator
    picks with the UI instead of us silently choosing).
    """
    from app.gguf import read_gguf_metadata

    try:
        directory = Path(root) / OPERATOR_LLM_DIR
        if not directory.is_dir():
            return []
        found: list[str] = []
        for entry in sorted(directory.iterdir()):
            try:
                if not entry.is_file():
                    continue
                info = read_gguf_metadata(entry)
                if info.ok and not info.is_split_shard:
                    found.append(str(entry))
            except OSError:
                continue
        return found
    except OSError:
        return []


def resolve_llm_model_path(
    root: Path, env_value: Optional[str] = None
) -> tuple[str, str]:
    """Resolve the ACTIVE GGUF path: user-selection > env profile > a
    GGUF actually present in the operator dir.

    Returns ``(absolute_path, source)`` where source is one of the SOURCE_*
    labels. ``("", SOURCE_NONE)`` means NO model is configured — the UI
    shows "no model configured" (never a phantom default path). Mirrored
    by the PowerShell starter's own resolution — keep the two in sync
    (scripts/start_llama_server.ps1 "LLM-GGUF-PATH FELOLDAS" block).
    """
    selected = user_selected_llm_model_path(Path(root))
    if selected:
        return selected, SOURCE_USER
    if env_value is None:
        env_value = os.environ.get("LLAMA_MODEL_PATH", "").strip()
    if env_value:
        return str(env_value), SOURCE_ENV
    candidates = operator_default_candidates(Path(root))
    if len(candidates) == 1:
        return candidates[0], SOURCE_DEFAULT
    return "", SOURCE_NONE


def read_llama_server_loaded_model(root: Path) -> dict[str, Any]:
    """Read the starter's resolved-model marker (lenient; {} when absent).

    ``scripts/start_llama_server.ps1`` writes
    ``logs/llama-server.resolved-model.json`` on every launch with the exact
    ``--model`` path it passed to llama-server; this is the cross-check that
    the RUNNING server serves the SAME file the UI has selected.
    """
    marker = Path(root) / RESOLVED_MODEL_MARKER
    try:
        if marker.is_file():
            raw = json.loads(marker.read_text("utf-8"))
            if isinstance(raw, dict):
                return raw
    except Exception as exc:  # noqa: BLE001
        logger.warning("resolved-model marker unreadable (%s): %s", marker, exc)
    return {}
