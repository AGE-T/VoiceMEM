"""Piper voice selection settings (UI task: selectable assistant voices).

The Voice section of the web UI exposes three controls:

* ``mode``      — ``auto`` | ``hu`` | ``en`` (default ``auto``)
* ``hu_voice``  — which Hungarian Piper voice speaks Hungarian replies
* ``en_voice``  — which English Piper voice speaks English replies

Resolution rule (:meth:`VoiceSettings.resolve`):

* ``auto`` — the ACTUAL RESPONSE LANGUAGE decides: a Hungarian reply uses the
  selected Hungarian voice, an English reply the selected English voice.
* ``hu``   — every reply is spoken with the selected Hungarian voice.
* ``en``   — every reply is spoken with the selected English voice.

Runtime flow (web backend): LLM response -> language detection ->
:meth:`VoiceSettings.resolve` -> Piper -> audio. Nothing here touches the
network; Piper is a local subprocess.

PERSISTENCE: selections are stored in ``config/voice_settings.json`` (next to
``voicemem_config.yaml``) so they survive an application restart. The file is
small, JSON, written atomically and read leniently (a corrupt/missing file
falls back to the config defaults, never blocks startup). The absolute
override env ``VOICEMEM_VOICE_SETTINGS`` redirects the file (tests, exotic
installs) without touching the root layout.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from app.config import AgentConfig
from app.text_utils import LANG_EN, LANG_HU

logger = logging.getLogger(__name__)

#: Voice language modes offered by the UI.
VOICE_MODES = ("auto", LANG_HU, LANG_EN)

#: Installed Piper voices (MODELS.lock.json tts entry) with friendly labels.
#: The primary UI label is ALWAYS the friendly name; the Piper file stem is
#: only an internal identifier.
VOICE_LABELS: dict[str, str] = {
    "hu_HU-anna-medium": "Anna",
    "hu_HU-berta-medium": "Berta",
    "hu_HU-imre-medium": "Imre",
    "en_US-lessac-medium": "Lessac",
}

DEFAULT_HU_VOICE = "hu_HU-anna-medium"
DEFAULT_EN_VOICE = "en_US-lessac-medium"

#: Preview sentences (local Piper only, no network).
PREVIEW_SENTENCES: dict[str, str] = {
    LANG_HU: "Szia Thomas, ez egy hangteszt.",
    LANG_EN: "Hello Thomas, this is a voice test.",
}

#: Settings file (relative to the repo root), next to voicemem_config.yaml.
SETTINGS_RELPATH = Path("config") / "voice_settings.json"


def voice_label(voice_id: str) -> str:
    """Friendly display name for a Piper voice id (``hu_HU-anna-medium`` -> ``Anna``)."""
    if voice_id in VOICE_LABELS:
        return VOICE_LABELS[voice_id]
    stem = str(voice_id).removesuffix(".onnx")
    # hu_HU-anna-medium -> "Anna"; unknown-voice -> "Unknown voice"
    parts = stem.split("-")
    core = parts[1] if len(parts) >= 2 else stem
    return core[:1].upper() + core[1:] if core else stem


def language_of_voice(voice_id: str) -> str:
    """Language tag of a Piper voice id (``hu_*`` -> ``hu``, everything else -> ``en``)."""
    return LANG_HU if str(voice_id).startswith("hu_") else LANG_EN


def language_name(language: str) -> str:
    """``hu`` -> ``Hungarian``, ``en`` -> ``English`` (UI status text)."""
    return "Hungarian" if language == LANG_HU else "English"


def available_voices(config: AgentConfig) -> list[str]:
    """Voice ids of the LOCALLY INSTALLED Piper voices (voices_dir *.onnx stems).

    When no voice is installed (DEMO mode / piper missing) the four standard
    ids are returned so the UI stays usable; synthesis then reports the real
    availability separately.
    """
    try:
        voices_dir = config.voices_dir
        installed = sorted(path.stem for path in voices_dir.glob("*.onnx"))
    except Exception:  # noqa: BLE001 - config paths must never crash the UI
        installed = []
    return installed or list(VOICE_LABELS)


def _first_of_language(ids: list[str], language: str) -> Optional[str]:
    for voice_id in ids:
        if language_of_voice(voice_id) == language:
            return voice_id
    return None


class VoiceSettings:
    """Persisted Piper voice selection (mode + per-language voice)."""

    def __init__(
        self,
        mode: str = "auto",
        hu_voice: str = DEFAULT_HU_VOICE,
        en_voice: str = DEFAULT_EN_VOICE,
        root: Optional[Path] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.root = Path(root) if root else None
        self.config = config
        #: Absolute settings file override (env VOICEMEM_VOICE_SETTINGS).
        self._path_override: Optional[Path] = None
        self.available = available_voices(config) if config else list(VOICE_LABELS)
        self.mode = mode if mode in VOICE_MODES else "auto"
        self.hu_voice = self._validated(hu_voice, LANG_HU)
        self.en_voice = self._validated(en_voice, LANG_EN)

    # -- validation ----------------------------------------------------------- #

    def _validated(self, voice_id: str, language: str) -> str:
        """Keep *voice_id* when installed and of *language*; else fall back."""
        voice_id = str(voice_id or "").strip()
        if voice_id in self.available and language_of_voice(voice_id) == language:
            return voice_id
        fallback = _first_of_language(self.available, language)
        if fallback is not None:
            return fallback
        default = DEFAULT_HU_VOICE if language == LANG_HU else DEFAULT_EN_VOICE
        return default

    # -- construction ---------------------------------------------------------- #

    @classmethod
    def load(cls, root: Path, config: Optional[AgentConfig] = None) -> "VoiceSettings":
        """Read the persisted settings (lenient); defaults when absent.

        Path resolution: ``VOICEMEM_VOICE_SETTINGS`` (absolute file) wins over
        ``<root>/config/voice_settings.json`` — tests use the override so they
        never touch the repository's real selection.
        """
        env_path = os.environ.get("VOICEMEM_VOICE_SETTINGS", "").strip()
        path = Path(env_path) if env_path else Path(root) / SETTINGS_RELPATH
        data: dict[str, Any] = {}
        try:
            if path.is_file():
                raw = json.loads(path.read_text("utf-8"))
                if isinstance(raw, dict):
                    data = raw
        except Exception as exc:  # noqa: BLE001 - never block startup
            logger.warning("voice settings file unreadable (%s): %s", path, exc)
        settings = cls(
            mode=str(data.get("mode", "auto")),
            hu_voice=str(data.get("hu_voice", DEFAULT_HU_VOICE)),
            en_voice=str(data.get("en_voice", DEFAULT_EN_VOICE)),
            root=root,
            config=config,
        )
        if env_path:
            settings._path_override = path
        return settings

    # -- persistence ------------------------------------------------------------- #

    @property
    def path(self) -> Optional[Path]:
        if self._path_override is not None:
            return self._path_override
        return (self.root / SETTINGS_RELPATH) if self.root else None

    def save(self) -> Path:
        """Persist to the settings file (atomic write). Raises OSError."""
        target = self.path
        if target is None:
            raise OSError("voice settings have no root directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": self.mode,
            "hu_voice": self.hu_voice,
            "en_voice": self.en_voice,
        }
        fd, tmp = tempfile.mkstemp(
            dir=str(target.parent), prefix=".voice_settings.", suffix=".tmp"
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
        logger.info("voice settings persisted: %s", payload)
        return target

    # -- resolution ----------------------------------------------------------------- #

    def resolve(self, language: str) -> tuple[str, str]:
        """(voice_id, spoken_language) for a reply detected as *language*.

        mode ``auto``: the response language picks the per-language voice
        (Hungarian reply -> selected Hungarian voice, English reply ->
        selected English voice). mode ``hu``/``en``: the forced language's
        voice speaks EVERY reply.
        """
        if self.mode == LANG_HU:
            return self.hu_voice, LANG_HU
        if self.mode == LANG_EN:
            return self.en_voice, LANG_EN
        if language == LANG_EN:
            return self.en_voice, LANG_EN
        return self.hu_voice, LANG_HU

    # -- UI payload ------------------------------------------------------------------ #

    def catalog(self) -> dict[str, list[str]]:
        """Installed voice ids split by language (UI dropdown contents)."""
        hu = [v for v in self.available if language_of_voice(v) == LANG_HU]
        en = [v for v in self.available if language_of_voice(v) == LANG_EN]
        return {LANG_HU: hu, LANG_EN: en}

    def to_dict(self) -> dict[str, Any]:
        """Full /api/voice payload (settings + catalog + labels)."""
        cat = self.catalog()
        installed = bool(self.config and _real_voices_installed(self.config))
        return {
            "mode": self.mode,
            "hu_voice": self.hu_voice,
            "en_voice": self.en_voice,
            "hu_voices": cat[LANG_HU],
            "en_voices": cat[LANG_EN],
            "labels": {v: voice_label(v) for v in self.available},
            "installed": installed,
            "preview_sentences": dict(PREVIEW_SENTENCES),
        }


def _real_voices_installed(config: AgentConfig) -> bool:
    """True when at least one Piper .onnx voice file exists on disk."""
    try:
        return any(config.voices_dir.glob("*.onnx"))
    except Exception:  # noqa: BLE001
        return False
