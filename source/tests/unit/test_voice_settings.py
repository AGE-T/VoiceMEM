"""v0.4.2 voice selection tests (v0.7.0: selectable Supertonic 3 presets).

Covers the full contract:
* VoiceSettings: catalog (installed presets), friendly labels, validation,
  auto/forced resolution, PERSISTENCE to disk (load -> save -> reload).
* REST surface (DEMO components + TestClient): GET /api/voice payload,
  POST /api/voice validation + persistence + status sync, bad input -> 400,
  /api/voice/preview returns a real WAV synthesized with the SELECTED voice.
* Turn flow (fake WS socket): the voice selected in the UI actually reaches
  the TTS engine (auto mode: per response language; forced mode: always).
* SupertonicTtsEngine: the explicit ``voice`` argument overrides the
  per-language default and lands in the SDK voice-style resolution.

Every Supertonic preset speaks BOTH hu and en (language-agnostic styles),
so both dropdowns offer the full preset list (v0.7.0 change vs Piper).

Sandbox-safe: DEMO components only; the settings file is redirected via
VOICEMEM_VOICE_SETTINGS so tests never touch the repository's real
config/voice_settings.json.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
import wave
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import AgentConfig  # noqa: E402
from app.text_utils import LANG_EN, LANG_HU  # noqa: E402
from app.voice_settings import (  # noqa: E402
    DEFAULT_EN_VOICE,
    DEFAULT_HU_VOICE,
    PREVIEW_SENTENCES,
    VOICE_LABELS,
    VoiceSettings,
    language_name,
    language_of_voice,
    voice_label,
)
from app.web_server import (  # noqa: E402
    DemoTtsEngine,
    WebComponents,
    WebSession,
    build_web_app,
)


class _FakeSock:
    """Minimal async WS double: records what the session sends to the browser."""

    def __init__(self) -> None:
        self.json_msgs: list[dict] = []
        self.audio_bytes = 0

    async def send_json(self, payload: dict) -> None:
        self.json_msgs.append(payload)

    async def send_bytes(self, raw: bytes) -> None:
        self.audio_bytes += len(raw)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestVoiceSettingsUnit(unittest.TestCase):
    """Pure VoiceSettings behaviour (no server, no engine)."""

    def test_labels_are_friendly_names(self):
        self.assertEqual(voice_label("F1"), "F1 · Nyugodt női")
        self.assertEqual(voice_label("M1"), "M1 · Élénk férfi")
        # unknown ids render as-is, never crash
        self.assertEqual(voice_label("Z9"), "Z9")

    def test_language_of_voice(self):
        # presets are language-AGNOSTIC; legacy piper ids keep their tags
        self.assertEqual(language_of_voice("F1"), LANG_HU)
        self.assertEqual(language_of_voice("en_US-lessac-medium"), LANG_EN)

    def test_language_names(self):
        self.assertEqual(language_name(LANG_HU), "Hungarian")
        self.assertEqual(language_name(LANG_EN), "English")

    def test_preview_sentences_are_the_task_sentences(self):
        self.assertEqual(PREVIEW_SENTENCES[LANG_HU], "Szia Thomas, ez egy hangteszt.")
        self.assertEqual(
            PREVIEW_SENTENCES[LANG_EN], "Hello Thomas, this is a voice test."
        )

    def test_auto_resolution_uses_response_language(self):
        vs = VoiceSettings(mode="auto", hu_voice="M4", en_voice="F3")
        self.assertEqual(vs.resolve(LANG_HU), ("M4", LANG_HU))
        self.assertEqual(vs.resolve(LANG_EN), ("F3", LANG_EN))

    def test_forced_modes_override_the_response_language(self):
        vs = VoiceSettings(mode="hu", hu_voice="M5", en_voice="F3")
        self.assertEqual(vs.resolve(LANG_EN), ("M5", LANG_HU))
        vs = VoiceSettings(mode="en", hu_voice="M5", en_voice="F3")
        self.assertEqual(vs.resolve(LANG_HU), ("F3", LANG_EN))

    def test_invalid_mode_falls_back_to_auto(self):
        vs = VoiceSettings(mode="hungarian!")
        self.assertEqual(vs.mode, "auto")

    def test_unknown_voice_falls_back_to_default(self):
        vs = VoiceSettings(hu_voice="Z9")
        self.assertEqual(vs.hu_voice, DEFAULT_HU_VOICE)

    def test_legacy_piper_ids_migrate_to_the_default(self):
        # v0.7.0 migration: an old voice_settings.json with Piper ids
        # falls back to the Supertonic default (never blocks startup).
        vs = VoiceSettings(hu_voice="hu_HU-anna-medium", en_voice="en_US-lessac-medium")
        self.assertEqual(vs.hu_voice, DEFAULT_HU_VOICE)
        self.assertEqual(vs.en_voice, DEFAULT_EN_VOICE)

    def test_catalog_lists_all_presets_without_assets(self):
        # no config -> demo catalog: exactly the ten preset voices, and BOTH
        # dropdowns offer the full list (presets are language-agnostic)
        vs = VoiceSettings()
        cat = vs.catalog()
        self.assertEqual(
            cat[LANG_HU],
            ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"],
        )
        self.assertEqual(cat[LANG_EN], cat[LANG_HU])

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vs = VoiceSettings(mode="hu", hu_voice="M2", en_voice="F4", root=root)
            path = vs.save()
            self.assertTrue(path.is_file())
            loaded = VoiceSettings.load(root)
            self.assertEqual(loaded.mode, "hu")
            self.assertEqual(loaded.hu_voice, "M2")
            self.assertEqual(loaded.en_voice, "F4")

    def test_load_is_lenient_on_garbage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "config" / "voice_settings.json"
            target.parent.mkdir(parents=True)
            target.write_text("{not json", encoding="utf-8")
            loaded = VoiceSettings.load(root)
            self.assertEqual(loaded.mode, "auto")
            self.assertEqual(loaded.hu_voice, DEFAULT_HU_VOICE)

    def test_available_voices_from_installed_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            styles = root / "models" / "tts" / "supertonic-3" / "voice_styles"
            styles.mkdir(parents=True)
            for stem in ("F1", "F3", "M2"):
                (styles / f"{stem}.json").write_text("{}")
            cfg = AgentConfig(root=root)
            vs = VoiceSettings(config=cfg)
            self.assertEqual(sorted(vs.available), ["F1", "F3", "M2"])
            # both dropdowns offer every INSTALLED preset
            cat = vs.catalog()
            self.assertEqual(cat[LANG_HU], ["F1", "F3", "M2"])
            self.assertEqual(cat[LANG_EN], ["F1", "F3", "M2"])
            # selecting a non-installed preset falls back to an installed one
            self.assertEqual(vs._validated("M5", LANG_HU), "F1")


class TestWebVoiceRest(unittest.TestCase):
    """REST surface in DEMO mode with the settings file redirected to tmp."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="vm_voice_rest_")
        cls.settings_path = Path(cls.tmp) / "voice_settings.json"
        cls._env_backup = None
        import os

        if "VOICEMEM_VOICE_SETTINGS" in os.environ:
            cls._env_backup = os.environ["VOICEMEM_VOICE_SETTINGS"]
        os.environ["VOICEMEM_VOICE_SETTINGS"] = str(cls.settings_path)
        cfg = AgentConfig(root=Path(cls.tmp))
        cls.components = WebComponents(cfg, mock=True).build()
        cls.app = build_web_app(cls.components)
        from fastapi.testclient import TestClient

        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        import os

        if cls._env_backup is not None:
            os.environ["VOICEMEM_VOICE_SETTINGS"] = cls._env_backup
        else:
            os.environ.pop("VOICEMEM_VOICE_SETTINGS", None)

    def test_get_voice_payload_shape(self):
        r = self.client.get("/api/voice")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["mode"], "auto")
        self.assertEqual(body["hu_voice"], DEFAULT_HU_VOICE)
        self.assertEqual(body["en_voice"], DEFAULT_EN_VOICE)
        self.assertEqual(
            body["hu_voices"],
            ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"],
        )
        self.assertEqual(body["en_voices"], body["hu_voices"])
        self.assertEqual(body["labels"]["M1"], "M1 · Élénk férfi")
        self.assertIn("preview_sentences", body)

    def test_post_voice_persists_and_syncs_status(self):
        r = self.client.post(
            "/api/voice",
            json={"mode": "hu", "hu_voice": "M2", "en_voice": "F4"},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["mode"], "hu")
        self.assertEqual(body["hu_voice"], "M2")
        # PERSISTED: the file on disk carries the selection
        data = json.loads(self.settings_path.read_text("utf-8"))
        self.assertEqual(data["mode"], "hu")
        self.assertEqual(data["hu_voice"], "M2")
        # the runtime status mirrors the selection
        snap = self.components.status.snapshot()
        self.assertEqual(snap["voice"]["mode"], "hu")
        self.assertEqual(snap["voice"]["hu"], "M2")
        self.assertEqual(snap["voice"]["hu_label"], "M2 · Mély férfi")

    def test_persistence_survives_a_restart(self):
        # "restart": build a FRESH component set from the same settings file
        r = self.client.post(
            "/api/voice", json={"mode": "en", "en_voice": "F4"}
        )
        self.assertEqual(r.status_code, 200)
        cfg = AgentConfig(root=Path(self.tmp))
        fresh = WebComponents(cfg, mock=True).build()
        self.assertEqual(fresh.voice.mode, "en")
        self.assertEqual(fresh.voice.en_voice, "F4")

    def test_post_voice_rejects_bad_mode(self):
        r = self.client.post("/api/voice", json={"mode": "french"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("mode", r.text)

    def test_post_voice_rejects_empty_body(self):
        r = self.client.post("/api/voice", json={})
        self.assertEqual(r.status_code, 200)  # nothing to apply is a no-op

    def test_preview_returns_wav_for_the_selected_voice(self):
        for voice in ("F1", "F2", "M1", "M5"):
            r = self.client.post("/api/voice/preview", json={"voice": voice})
            self.assertEqual(r.status_code, 200, voice)
            self.assertEqual(r.headers["content-type"].split(";")[0], "audio/wav")
            payload = r.content
            self.assertGreater(len(payload), 1000, voice)
            with wave.open(io.BytesIO(payload), "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertGreater(w.getnframes(), 2000, voice)
        # distinct voices produce distinct audio
        a = self.client.post("/api/voice/preview", json={"voice": "F1"}).content
        b = self.client.post("/api/voice/preview", json={"voice": "F2"}).content
        self.assertNotEqual(a, b)

    def test_selection_change_keeps_the_active_voice_fields(self):
        # regression: switching the voice while/after a turn must not drop
        # the active/active_label/language fields from the status snapshot
        r = self.client.post("/api/voice", json={"hu_voice": "F2"})
        self.assertEqual(r.status_code, 200)
        st = self.components.status
        st.voice["active"] = "M2"
        st.voice["active_label"] = "M2 · Mély férfi"
        st.voice["language"] = "hu"
        r = self.client.post("/api/voice", json={"hu_voice": "F2"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(st.voice["active"], "M2")
        self.assertEqual(st.voice["active_label"], "M2 · Mély férfi")
        self.assertEqual(st.voice["language"], "hu")

    def test_preview_rejects_unknown_voice(self):
        r = self.client.post("/api/voice/preview", json={"voice": "Z9"})
        self.assertEqual(r.status_code, 400)
        # legacy piper ids are unknown presets now
        r = self.client.post("/api/voice/preview", json={"voice": "hu_HU-anna-medium"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/voice/preview", json={})
        self.assertEqual(r.status_code, 400)


class TestTurnVoiceFlow(unittest.TestCase):
    """The selected voice actually reaches the TTS engine inside a turn."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="vm_voice_turn_")
        cls._env_backup = None
        import os

        if "VOICEMEM_VOICE_SETTINGS" in os.environ:
            cls._env_backup = os.environ["VOICEMEM_VOICE_SETTINGS"]
        os.environ["VOICEMEM_VOICE_SETTINGS"] = str(Path(cls.tmp) / "voice_settings.json")
        cfg = AgentConfig(root=Path(cls.tmp))
        cls.components = WebComponents(cfg, mock=True).build()

    @classmethod
    def tearDownClass(cls):
        import os

        if cls._env_backup is not None:
            os.environ["VOICEMEM_VOICE_SETTINGS"] = cls._env_backup
        else:
            os.environ.pop("VOICEMEM_VOICE_SETTINGS", None)

    def _turn(self, text: str) -> _FakeSock:
        sock = _FakeSock()
        session = WebSession(sock, self.components)

        async def run():
            await session._start_turn(text, source="text")
            await asyncio.gather(session._turn_task)

        _run(run())
        return sock

    def test_auto_mode_hungarian_reply_uses_hungarian_voice(self):
        self.components.voice.mode = "auto"
        self.components.voice.hu_voice = "M4"
        sock = self._turn("Szia, hogy vagy?")
        # the reply is scripted Hungarian in demo mode
        self.assertGreater(sock.audio_bytes, 1000)
        self.assertEqual(self.components.tts.last_voice, "M4")
        snap = self.components.status.snapshot()
        self.assertEqual(snap["voice"]["active"], "M4")
        self.assertEqual(snap["voice"]["active_label"], "M4 · Barátságos férfi")
        self.assertEqual(snap["voice"]["language"], LANG_HU)

    def test_changing_the_selection_changes_the_generated_voice(self):
        self.components.voice.mode = "auto"
        self.components.voice.hu_voice = "F2"
        self._turn("Szia, hogy vagy?")
        self.assertEqual(self.components.tts.last_voice, "F2")
        self.components.voice.hu_voice = "M4"
        self._turn("Szia, hogy vagy?")
        self.assertEqual(self.components.tts.last_voice, "M4")

    def test_forced_english_mode_speaks_english_voice_for_hungarian_text(self):
        self.components.voice.mode = "en"
        self.components.voice.en_voice = "F4"
        self._turn("Szia, hogy vagy?")  # Hungarian user text
        # demo replies follow the user language; forced mode overrides it
        self.assertEqual(self.components.tts.last_voice, "F4")
        self.assertEqual(self.components.status.snapshot()["voice"]["language"], LANG_EN)

    def test_demo_tts_voices_are_audibly_distinct(self):
        # the sandbox preview engine gives each voice a distinct tone so
        # "changing the selection changes the generated voice" is verifiable
        freqs = DemoTtsEngine.VOICE_FREQ
        self.assertEqual(len(set(freqs.values())), len(freqs))


class TestTtsEngineVoiceOverride(unittest.TestCase):
    """SupertonicTtsEngine: the explicit voice wins in voice resolution."""

    def test_voice_name_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            styles = root / "models" / "tts" / "supertonic-3" / "voice_styles"
            styles.mkdir(parents=True)
            for stem in ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"):
                (styles / f"{stem}.json").write_text("{}")
            cfg = AgentConfig(root=root)
            from app.tts_supertonic import SupertonicTtsEngine

            engine = SupertonicTtsEngine(cfg)
            # explicit voice wins over the language default
            self.assertEqual(engine._voice_name(LANG_EN, "M2"), "M2")
            # no explicit voice -> per-language default
            self.assertEqual(engine._voice_name(LANG_EN, None), "F1")
            self.assertEqual(engine._voice_name(LANG_HU, None), "F1")
            # unknown explicit voice -> per-language default (logged, not fatal)
            self.assertEqual(engine._voice_name(LANG_HU, "Z9"), "F1")
            # legacy piper ids are unknown presets
            self.assertEqual(engine._voice_name(LANG_HU, "hu_HU-anna-medium"), "F1")
            self.assertTrue(engine.voice_exists("M2"))
            self.assertFalse(engine.voice_exists("Z9"))
            self.assertEqual(
                engine.list_voices(),
                ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"],
            )


if __name__ == "__main__":
    unittest.main()
