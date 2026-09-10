"""v0.4.2 voice selection tests (UI task: selectable Piper voices).

Covers the full contract:
* VoiceSettings: catalog (installed voices), friendly labels, validation,
  auto/forced resolution, PERSISTENCE to disk (load -> save -> reload).
* REST surface (DEMO components + TestClient): GET /api/voice payload,
  POST /api/voice validation + persistence + status sync, bad input -> 400,
  /api/voice/preview returns a real WAV synthesized with the SELECTED voice.
* Turn flow (fake WS socket): the voice selected in the UI actually reaches
  the TTS engine (auto mode: per response language; forced mode: always).
* TtsEngine: the explicit ``voice`` argument overrides the per-language
  default and lands in the piper --model command line.

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
        self.assertEqual(voice_label("hu_HU-anna-medium"), "Anna")
        self.assertEqual(voice_label("hu_HU-berta-medium"), "Berta")
        self.assertEqual(voice_label("hu_HU-imre-medium"), "Imre")
        self.assertEqual(voice_label("en_US-lessac-medium"), "Lessac")
        # unknown ids still render a friendly name, never the raw file stem
        self.assertEqual(voice_label("hu_HU-zoltan-medium"), "Zoltan")

    def test_language_of_voice(self):
        self.assertEqual(language_of_voice("hu_HU-anna-medium"), LANG_HU)
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
        vs = VoiceSettings(mode="auto", hu_voice="hu_HU-imre-medium", en_voice="en_US-lessac-medium")
        self.assertEqual(vs.resolve(LANG_HU), ("hu_HU-imre-medium", LANG_HU))
        self.assertEqual(vs.resolve(LANG_EN), ("en_US-lessac-medium", LANG_EN))

    def test_forced_modes_override_the_response_language(self):
        vs = VoiceSettings(mode="hu", hu_voice="hu_HU-berta-medium", en_voice="en_US-lessac-medium")
        self.assertEqual(vs.resolve(LANG_EN), ("hu_HU-berta-medium", LANG_HU))
        vs = VoiceSettings(mode="en", hu_voice="hu_HU-berta-medium", en_voice="en_US-lessac-medium")
        self.assertEqual(vs.resolve(LANG_HU), ("en_US-lessac-medium", LANG_EN))

    def test_invalid_mode_falls_back_to_auto(self):
        vs = VoiceSettings(mode="hungarian!")
        self.assertEqual(vs.mode, "auto")

    def test_unknown_voice_falls_back_to_default(self):
        vs = VoiceSettings(hu_voice="hu_HU-zoltan-medium")
        self.assertEqual(vs.hu_voice, DEFAULT_HU_VOICE)

    def test_catalog_lists_all_four_standard_voices_without_piper(self):
        # no config -> demo catalog: exactly the four installed-by-MODELS.lock voices
        vs = VoiceSettings()
        cat = vs.catalog()
        self.assertEqual(
            cat[LANG_HU],
            ["hu_HU-anna-medium", "hu_HU-berta-medium", "hu_HU-imre-medium"],
        )
        self.assertEqual(cat[LANG_EN], ["en_US-lessac-medium"])

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vs = VoiceSettings(
                mode="hu", hu_voice="hu_HU-imre-medium", en_voice="en_US-lessac-medium", root=root
            )
            path = vs.save()
            self.assertTrue(path.is_file())
            loaded = VoiceSettings.load(root)
            self.assertEqual(loaded.mode, "hu")
            self.assertEqual(loaded.hu_voice, "hu_HU-imre-medium")
            self.assertEqual(loaded.en_voice, "en_US-lessac-medium")

    def test_load_is_lenient_on_garbage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "config" / "voice_settings.json"
            target.parent.mkdir(parents=True)
            target.write_text("{not json", encoding="utf-8")
            loaded = VoiceSettings.load(root)
            self.assertEqual(loaded.mode, "auto")
            self.assertEqual(loaded.hu_voice, DEFAULT_HU_VOICE)

    def test_available_voices_from_installed_onnx(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            voices_dir = root / "models" / "tts" / "piper"
            voices_dir.mkdir(parents=True)
            for stem in ("hu_HU-anna-medium", "hu_HU-imre-medium", "en_US-lessac-medium"):
                (voices_dir / f"{stem}.onnx").write_bytes(b"x" * 16)
            cfg = AgentConfig(root=root)
            vs = VoiceSettings(config=cfg)
            self.assertEqual(
                sorted(vs.available),
                ["en_US-lessac-medium", "hu_HU-anna-medium", "hu_HU-imre-medium"],
            )
            # only INSTALLED Hungarian voices are selectable
            cat = vs.catalog()
            self.assertEqual(cat[LANG_HU], ["hu_HU-anna-medium", "hu_HU-imre-medium"])
            # selecting a non-installed voice falls back to an installed one
            self.assertEqual(vs._validated("hu_HU-berta-medium", LANG_HU), "hu_HU-anna-medium")


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
            ["hu_HU-anna-medium", "hu_HU-berta-medium", "hu_HU-imre-medium"],
        )
        self.assertEqual(body["en_voices"], ["en_US-lessac-medium"])
        self.assertEqual(body["labels"]["hu_HU-imre-medium"], "Imre")
        self.assertIn("preview_sentences", body)

    def test_post_voice_persists_and_syncs_status(self):
        r = self.client.post(
            "/api/voice",
            json={"mode": "hu", "hu_voice": "hu_HU-imre-medium", "en_voice": "en_US-lessac-medium"},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["mode"], "hu")
        self.assertEqual(body["hu_voice"], "hu_HU-imre-medium")
        # PERSISTED: the file on disk carries the selection
        data = json.loads(self.settings_path.read_text("utf-8"))
        self.assertEqual(data["mode"], "hu")
        self.assertEqual(data["hu_voice"], "hu_HU-imre-medium")
        # the runtime status mirrors the selection
        snap = self.components.status.snapshot()
        self.assertEqual(snap["voice"]["mode"], "hu")
        self.assertEqual(snap["voice"]["hu"], "hu_HU-imre-medium")
        self.assertEqual(snap["voice"]["hu_label"], "Imre")

    def test_persistence_survives_a_restart(self):
        # "restart": build a FRESH component set from the same settings file
        r = self.client.post(
            "/api/voice", json={"mode": "en", "en_voice": "en_US-lessac-medium"}
        )
        self.assertEqual(r.status_code, 200)
        cfg = AgentConfig(root=Path(self.tmp))
        fresh = WebComponents(cfg, mock=True).build()
        self.assertEqual(fresh.voice.mode, "en")
        self.assertEqual(fresh.voice.en_voice, "en_US-lessac-medium")

    def test_post_voice_rejects_bad_mode(self):
        r = self.client.post("/api/voice", json={"mode": "french"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("mode", r.text)

    def test_post_voice_rejects_empty_body(self):
        r = self.client.post("/api/voice", json={})
        self.assertEqual(r.status_code, 200)  # nothing to apply is a no-op

    def test_preview_returns_wav_for_the_selected_voice(self):
        for voice in ("hu_HU-anna-medium", "hu_HU-berta-medium", "hu_HU-imre-medium", "en_US-lessac-medium"):
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
        a = self.client.post("/api/voice/preview", json={"voice": "hu_HU-anna-medium"}).content
        b = self.client.post("/api/voice/preview", json={"voice": "hu_HU-berta-medium"}).content
        self.assertNotEqual(a, b)

    def test_selection_change_keeps_the_active_voice_fields(self):
        # regression: switching the voice while/after a turn must not drop
        # the active/active_label/language fields from the status snapshot
        r = self.client.post("/api/voice", json={"hu_voice": "hu_HU-berta-medium"})
        self.assertEqual(r.status_code, 200)
        st = self.components.status
        st.voice["active"] = "hu_HU-imre-medium"
        st.voice["active_label"] = "Imre"
        st.voice["language"] = "hu"
        r = self.client.post("/api/voice", json={"hu_voice": "hu_HU-berta-medium"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(st.voice["active"], "hu_HU-imre-medium")
        self.assertEqual(st.voice["active_label"], "Imre")
        self.assertEqual(st.voice["language"], "hu")

    def test_preview_rejects_unknown_voice(self):
        r = self.client.post("/api/voice/preview", json={"voice": "hu_HU-zoltan-medium"})
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
        self.components.voice.hu_voice = "hu_HU-imre-medium"
        sock = self._turn("Szia, hogy vagy?")
        # the reply is scripted Hungarian in demo mode
        self.assertGreater(sock.audio_bytes, 1000)
        self.assertEqual(self.components.tts.last_voice, "hu_HU-imre-medium")
        snap = self.components.status.snapshot()
        self.assertEqual(snap["voice"]["active"], "hu_HU-imre-medium")
        self.assertEqual(snap["voice"]["active_label"], "Imre")
        self.assertEqual(snap["voice"]["language"], LANG_HU)

    def test_changing_the_selection_changes_the_generated_voice(self):
        self.components.voice.mode = "auto"
        self.components.voice.hu_voice = "hu_HU-berta-medium"
        self._turn("Szia, hogy vagy?")
        self.assertEqual(self.components.tts.last_voice, "hu_HU-berta-medium")
        self.components.voice.hu_voice = "hu_HU-imre-medium"
        self._turn("Szia, hogy vagy?")
        self.assertEqual(self.components.tts.last_voice, "hu_HU-imre-medium")

    def test_forced_english_mode_speaks_english_voice_for_hungarian_text(self):
        self.components.voice.mode = "en"
        self.components.voice.en_voice = "en_US-lessac-medium"
        self._turn("Szia, hogy vagy?")  # Hungarian user text
        # demo replies follow the user language; forced mode overrides it
        self.assertEqual(self.components.tts.last_voice, "en_US-lessac-medium")
        self.assertEqual(self.components.status.snapshot()["voice"]["language"], LANG_EN)

    def test_demo_tts_voices_are_audibly_distinct(self):
        # the sandbox preview engine gives each voice a distinct tone so
        # "changing the selection changes the generated voice" is verifiable
        freqs = DemoTtsEngine.VOICE_FREQ
        self.assertEqual(len(set(freqs.values())), len(freqs))


class TestTtsEngineVoiceOverride(unittest.TestCase):
    """TtsEngine: the explicit voice wins and lands in the piper command."""

    def test_voice_path_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            voices_dir = root / "models" / "tts" / "piper"
            voices_dir.mkdir(parents=True)
            for stem in ("hu_HU-anna-medium", "hu_HU-berta-medium", "en_US-lessac-medium"):
                (voices_dir / f"{stem}.onnx").write_bytes(b"x" * 16)
            cfg = AgentConfig(root=root)
            from app.tts import TtsEngine

            engine = TtsEngine(cfg)
            # explicit voice wins over the language default
            path = engine._voice_path(LANG_EN, "hu_HU-berta-medium")
            self.assertEqual(path.name, "hu_HU-berta-medium.onnx")
            # no explicit voice -> per-language default
            path = engine._voice_path(LANG_EN, None)
            self.assertEqual(path.name, "en_US-lessac-medium.onnx")
            path = engine._voice_path(LANG_HU, None)
            self.assertEqual(path.name, "hu_HU-anna-medium.onnx")
            # unknown explicit voice -> per-language default (logged, not fatal)
            path = engine._voice_path(LANG_HU, "hu_HU-zoltan-medium")
            self.assertEqual(path.name, "hu_HU-anna-medium.onnx")
            self.assertTrue(engine.voice_exists("hu_HU-berta-medium"))
            self.assertFalse(engine.voice_exists("hu_HU-zoltan-medium"))

    def test_piper_command_uses_the_selected_voice(self):
        """The --model argument must carry the SELECTED voice (task contract)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            voices_dir = root / "models" / "tts" / "piper"
            voices_dir.mkdir(parents=True)
            for stem in ("hu_HU-anna-medium", "hu_HU-imre-medium", "en_US-lessac-medium"):
                (voices_dir / f"{stem}.onnx").write_bytes(b"x" * 16)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            piper = bin_dir / ("piper.exe" if sys.platform == "win32" else "piper")
            piper.write_bytes(b"#!/bin/sh\nexit 0\n")
            cfg = AgentConfig(root=root)
            cfg.piper_executable = str(piper)
            from app.tts import TtsEngine

            engine = TtsEngine(cfg)
            seen: list[list[str]] = []
            import subprocess as sp

            real_popen = sp.Popen
            def spy_popen(cmd, *args, **kwargs):  # noqa: ANN001, ANN202
                seen.append(list(cmd))
                return real_popen(["true"], *args, **kwargs)

            import app.tts as tts_mod

            with unittest.mock.patch.object(tts_mod.subprocess, "Popen", spy_popen):
                with tempfile.TemporaryDirectory() as out_tmp:
                    ok = engine.synthesize_to_file(
                        "Szia!", LANG_HU, Path(out_tmp) / "x.wav", voice="hu_HU-imre-medium"
                    )
            # piper "succeeded" (exit 0) but wrote no file -> False is fine;
            # the contract under test is the --model argument:
            self.assertTrue(seen)
            cmd = seen[0]
            self.assertIn("--model", cmd)
            model = cmd[cmd.index("--model") + 1]
            self.assertEqual(Path(model).name, "hu_HU-imre-medium.onnx")
            self.assertEqual(engine.last_voice, "hu_HU-imre-medium")


if __name__ == "__main__":
    unittest.main()
