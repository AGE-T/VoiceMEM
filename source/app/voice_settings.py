"""Supertonic 3 voice selection settings (UI task: selectable assistant voices).

v0.7.0: the Voice section now selects Supertonic 3 preset voice styles
(``F1``–``F5``, ``M1``–``M5``) instead of Piper voices. Every preset
speaks BOTH Hungarian and English (the model is multilingual; the voice
is a style embedding, not a per-language model), so the per-language
selectors are a UI preference, not a technical constraint.

The Voice section of the web UI exposes three controls:

* ``mode``      — ``auto`` | ``hu`` | ``en`` (default ``auto``)
* ``hu_voice``  — which preset speaks Hungarian replies
* ``en_voice``  — which preset speaks English replies

Resolution rule (:meth:`VoiceSettings.resolve`):

* ``auto`` — the ACTUAL RESPONSE LANGUAGE decides: a Hungarian reply uses
  the selected Hungarian preset, an English reply the English preset.
* ``hu``   — every reply is spoken with the selected Hungarian preset.
* ``en``   — every reply is spoken with the selected English preset.

Runtime flow (web backend): LLM response -> language detection ->
:meth:`VoiceSettings.resolve` -> Supertonic 3 -> audio. Nothing here touches
the network; Supertonic runs ONNX Runtime locally (app/tts_supertonic.py).

PERSISTENCE: selections are stored in ``config/voice_settings.json`` (next to
``voicemem_config.yaml``) so they survive an application restart. The file is
small, JSON, written atomically and read leniently (a corrupt/missing file
falls back to the config defaults, never blocks startup). The absolute
override env ``VOICEMEM_VOICE_SETTINGS`` redirects the file (tests, exotic
installs) without touching the root layout.

v0.7.0 MIGRATION: a legacy ``voice_settings.json`` written by the Piper era
(``hu_HU-anna-medium`` etc.) fails validation and falls back to the Supertonic
default — the old ids simply do not exist as presets.
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

#: Supertonic 3 preset voices (voice_styles/*.json in the model dir) with
#: friendly labels. The primary UI label is the preset id + character; the
#: character descriptions follow the official voices documentation
#: (supertonic-py docs/voices.md).
VOICE_LABELS: dict[str, str] = {
    "F1": "F1 · Nyugodt női",
    "F2": "F2 · Derűs női",
    "F3": "F3 · Beszélő női",
    "F4": "F4 · Magabiztos női",
    "F5": "F5 · Kedves női",
    "M1": "M1 · Élénk férfi",
    "M2": "M2 · Mély férfi",
    "M3": "M3 · Tekintélyes férfi",
    "M4": "M4 · Barátságos férfi",
    "M5": "M5 · Meleg férfi",
}

DEFAULT_HU_VOICE = "F1"
DEFAULT_EN_VOICE = "F1"

#: Preview sentences (local Supertonic only, no network).
PREVIEW_SENTENCES: dict[str, str] = {
    LANG_HU: "Szia Thomas, ez egy hangteszt.",
    LANG_EN: "Hello Thomas, this is a voice test.",
}

#: Settings file (relative to the repo root), next to voicemem_config.yaml.
SETTINGS_RELPATH = Path("config") / "voice_settings.json"


def voice_label(voice_id: str) -> str:
    """Friendly display name for a preset id (``F1`` -> "F1 · Nyugodt női")."""
    if voice_id in VOICE_LABELS:
        return VOICE_LABELS[voice_id]
    stem = str(voice_id).strip()
    return stem if stem else "Unknown voice"


def language_of_voice(voice_id: str) -> str:
    """Language tag of a voice selection.

    Supertonic presets are language-agnostic (every preset speaks both hu and
    en), so this reports the UI-side grouping of the id: preset ids are valid
    for BOTH languages; the return value only feeds the legacy per-language
    validation fallbacks (``_first_of_language`` keeps working unchanged).
    """
    return LANG_EN if str(voice_id).startswith("en_") else LANG_HU


def language_name(language: str) -> str:
    """``hu`` -> ``Hungarian``, ``en`` -> ``English`` (UI status text)."""
    return "Hungarian" if language == LANG_HU else "English"


def available_voices(config: AgentConfig) -> list[str]:
    """Preset ids of the LOCALLY INSTALLED voice styles (voice_styles/*.json).

    When none is installed (DEMO mode / assets missing) the full preset list
    is returned so the UI stays usable; synthesis then reports the real
    availability separately.
    """
    try:
        voices_dir = config.supertonic_voices_dir
        installed = sorted(path.stem for path in voices_dir.glob("*.json"))
    except Exception:  # noqa: BLE001 - config paths must never crash the UI
        installed = []
    return installed or list(VOICE_LABELS)


def _first_of_language(ids: list[str], language: str) -> Optional[str]:
    for voice_id in ids:
        if language_of_voice(voice_id) == language:
            return voice_id
    return None


class VoiceSettings:
    """Persisted Supertonic preset selection (mode + per-language preset)."""

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
        """Keep *voice_id* when it is an installed preset; else fall back.

        Supertonic presets are language-AGNOSTIC (every preset speaks both
        hu and en), so any installed preset is valid for either selector;
        legacy Piper ids (``hu_HU-…``/``en_US-…``) simply do not exist as
        presets and fail this check — the v0.7.0 migration path.
        """
        voice_id = str(voice_id or "").strip()
        if voice_id in self.available:
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
        """Installed preset ids per language (UI dropdown contents).

        Every Supertonic preset speaks both languages, so BOTH dropdowns
        offer the full preset list.
        """
        presets = list(self.available)
        return {LANG_HU: presets, LANG_EN: presets}

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
    """True when at least one Supertonic voice-style JSON exists on disk."""
    try:
        return any(config.supertonic_voices_dir.glob("*.json"))
    except Exception:  # noqa: BLE001
        return False
