"""Unit tests for the LOCAL web backend (app/web_server.py).

Sandbox-safe: uses the DEMO (mock) components only — no torch, no onnx, no
llama-server, no network. The WebSocket end-to-end flow (with a real
llama-server stub over localhost HTTP) lives in
tests/integration/test_web_e2e.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.web_server import (  # noqa: E402
    COMPONENT_KEYS,
    DEFAULT_WEB_PORT,
    DemoLlmClient,
    DemoMemoryLayer,
    DemoTtsEngine,
    EnergyVad,
    PipelineStatus,
    WebComponents,
    build_web_app,
    clean_rb_content,
    float32_to_pcm16_bytes,
    load_config,
    pcm16_to_float32,
    resample_linear,
    sanitize_space_name,
    semantic_emotion_label,
)

import numpy as np  # noqa: E402


def _tmp_config(tmpdir: str) -> object:
    """AgentConfig rooted in a temp dir (isolated memory + demo store)."""
    from app.config import AgentConfig

    cfg = AgentConfig(root=Path(tmpdir))
    return cfg


class TestAudioHelpers(unittest.TestCase):
    def test_pcm_roundtrip(self):
        x = np.array([0.0, 0.5, -0.5, 0.99, -0.99], dtype=np.float32)
        raw = float32_to_pcm16_bytes(x)
        back = pcm16_to_float32(raw)
        self.assertEqual(back.size, 5)
        self.assertTrue(np.allclose(back, x, atol=0.001))

    def test_pcm_odd_bytes(self):
        self.assertEqual(pcm16_to_float32(b"\x01").size, 0)
        self.assertEqual(float32_to_pcm16_bytes(np.zeros(0)), b"")

    def test_resample_identity_and_ratio(self):
        x = np.ones(16000, dtype=np.float32)
        self.assertEqual(resample_linear(x, 16000, 16000).size, 16000)
        up = resample_linear(x, 16000, 24000)
        self.assertEqual(up.size, 24000)
        down = resample_linear(up, 24000, 16000)
        self.assertEqual(down.size, 16000)

    def test_resample_empty(self):
        self.assertEqual(resample_linear(np.zeros(0, dtype=np.float32), 16000, 24000).size, 0)


class TestStatusTracking(unittest.TestCase):
    def test_component_lifecycle(self):
        st = PipelineStatus()
        self.assertEqual(set(st.components.keys()), set(COMPONENT_KEYS))
        c = st.components["asr"]
        c.set_ready("Qwen3")
        self.assertEqual(c.state, "ready")
        c.begin()
        self.assertEqual(c.state, "processing")
        ms = c.end()
        self.assertGreaterEqual(ms, 0.0)
        self.assertEqual(c.state, "ready")
        self.assertEqual(c.to_dict()["key"], "asr")
        c.fail("boom")
        self.assertEqual(c.state, "error")
        self.assertEqual(c.to_dict()["error"], "boom")

    def test_snapshot_shape(self):
        snap = PipelineStatus().snapshot()
        self.assertIn("components", snap)
        self.assertIn("llama", snap)
        self.assertIn("mode", snap)


class TestSanitize(unittest.TestCase):
    def test_sanitize(self):
        self.assertEqual(sanitize_space_name("demo"), "demo")
        self.assertEqual(sanitize_space_name("My Space!"), "MySpace")
        self.assertEqual(sanitize_space_name("a/b\\c"), "abc")
        self.assertEqual(sanitize_space_name(""), "")
        self.assertEqual(sanitize_space_name("x" * 50), "x" * 32)


class TestSemanticEmotion(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(semantic_emotion_label(""), ("", 0.0, 0.0))

    def test_negative_text(self):
        # markers from app.emotion's teacher-context lists
        label, v, a = semantic_emotion_label("I don't understand, this is too difficult")
        self.assertNotEqual(label, "")
        self.assertLess(v, 0.0)

    def test_positive_text(self):
        label, v, a = semantic_emotion_label("Great, thanks, I understand now")
        self.assertGreater(v, 0.3)


class TestCleanRb(unittest.TestCase):
    def test_prefix_and_suffix_stripped(self):
        raw = "[2026-08-24] ⚠ 避免重复：Something to avoid next time（下次：be brief）"
        out = clean_rb_content(raw)
        self.assertNotIn("[", out)
        self.assertNotIn("下次", out)
        self.assertNotIn("avoid next time（", out)


class TestDemoMemoryLayer(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="vm_web_test_")
        self.cfg = _tmp_config(self.tmp)
        self.mem = DemoMemoryLayer(self.cfg, PipelineStatus())

    def test_default_space(self):
        spaces = self.mem.list_spaces()
        self.assertEqual(spaces[0]["id"], "demo")
        self.assertEqual(self.mem.active, "demo")

    def test_create_and_switch(self):
        created = self.mem.create_space("work projects")
        self.assertEqual(created["id"], "workprojects")
        self.assertEqual(self.mem.use_space("work projects"), "workprojects")
        self.assertEqual(self.mem.active, "workprojects")

    def test_duplicate_rejected(self):
        with self.assertRaises(FileExistsError):
            self.mem.create_space("demo")

    def test_ingest_search_snapshot(self):
        self.mem.ingest("Peter lives in Budapest and loves running", "ok")
        result = self.mem.search("where does Peter live?")
        hits = result.hits
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].memory_id, "m1")
        cls = result.classification
        self.assertIsInstance(cls.slots, list)  # must be a LIST not chars
        self.assertIn("Peter", cls.entities)
        snap = self.mem.snapshot()
        self.assertEqual(len(snap["left"]), 1)
        self.assertGreaterEqual(len(snap["right"]), 0)
        # persistence: a fresh layer over the same root sees the same data
        mem2 = DemoMemoryLayer(self.cfg, PipelineStatus())
        self.assertEqual(len(mem2.snapshot()["left"]), 1)


class TestDemoEngines(unittest.TestCase):
    def test_demo_llm_streams_words(self):
        import asyncio

        llm = DemoLlmClient()

        async def run():
            out = []
            async for w in llm.chat_stream(
                [{"role": "user", "content": "Hello there"}]
            ):
                out.append(w)
            return "".join(out)

        reply = asyncio.run(run())
        self.assertIn("Hello there", reply)
        self.assertEqual(len(llm.calls), 1)

    def test_demo_llm_json(self):
        import asyncio

        title = asyncio.run(
            DemoLlmClient().chat_json([{"role": "user", "content": "talking about Budapest city"}])
        )
        self.assertIn("title", title)

    def test_demo_tts(self):
        tts = DemoTtsEngine()
        pcm = tts.synthesize("Hello world", "en")
        self.assertIsNotNone(pcm)
        self.assertEqual(pcm.dtype, np.int16)
        self.assertGreater(pcm.size, tts.SAMPLE_RATE * 0.3)
        self.assertIsNone(tts.synthesize("", "en"))

    def test_energy_vad(self):
        vad = EnergyVad()
        loud = np.full(512, 0.5, dtype=np.float32)
        silent = np.zeros(512, dtype=np.float32)
        self.assertGreater(vad.prob(loud), 0.5)
        self.assertEqual(vad.prob(silent), 0.0)


class TestWebAppRest(unittest.TestCase):
    """REST surface over the FastAPI app in DEMO mode (TestClient)."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.tmp = tempfile.mkdtemp(prefix="vm_web_app_")
        cfg = _tmp_config(cls.tmp)
        cfg.enable_speaker = True  # verify the web layer forces it OFF
        components = WebComponents(cfg, mock=True).build()
        cls.components = components
        cls.app = build_web_app(components)
        from fastapi.testclient import TestClient

        cls.client = TestClient(cls.app)

    def test_speaker_disabled_by_default(self):
        self.assertFalse(self.components.config.enable_speaker)

    def test_index_serves_ui(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("VoiceMem", r.text)
        self.assertIn("micSel", r.text)  # microphone selector present
        self.assertIn("pipeStrip", r.text)  # pipeline debug view present

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["mode"], "demo")
        self.assertEqual(body["llama"]["url"], "http://127.0.0.1:8080/v1")

    def test_pipeline(self):
        r = self.client.get("/api/pipeline")
        body = r.json()
        self.assertEqual(set(body["components"].keys()), set(COMPONENT_KEYS))
        for key, comp in body["components"].items():
            self.assertIn(comp["state"], ("ready", "processing", "error", "missing", "mocked", "init"))
            self.assertIn("last_ms", comp)

    def test_spaces_crud(self):
        r = self.client.get("/api/spaces")
        self.assertEqual(r.status_code, 200)
        self.assertIn("demo", [s["id"] for s in r.json()["spaces"]])
        r = self.client.post("/api/spaces", json={"name": "lab"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["id"], "lab")
        # duplicate -> 409
        r = self.client.post("/api/spaces", json={"name": "lab"})
        self.assertEqual(r.status_code, 409)
        # invalid -> 400
        r = self.client.post("/api/spaces", json={"name": "!!!"})
        self.assertEqual(r.status_code, 400)
        # switch
        r = self.client.post("/api/spaces/lab/use")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["active"], "lab")
        # unknown space is created on the fly (registry semantics)
        r = self.client.post("/api/spaces/other/use")
        self.assertEqual(r.status_code, 200)

    def test_memories_shape(self):
        r = self.client.get("/api/memories")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("left", body)
        self.assertIn("right", body)

    def test_classify(self):
        r = self.client.post("/api/classify", json={"query": "running in the park"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsInstance(body["slots"], list)
        self.assertIsInstance(body["entities"], list)

    def test_title_and_lang(self):
        r = self.client.post("/api/title", json={"text": "we talked about Budapest"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("title", r.json())
        r = self.client.post("/api/lang", json={"lang": "en"})
        self.assertEqual(r.status_code, 200)

    def test_audio_404(self):
        r = self.client.get("/api/audio/whatever")
        self.assertEqual(r.status_code, 404)


class TestNoOpenAi(unittest.TestCase):
    """OPENAI RUNTIME CALLS: MUST BE 0 — static audit of the web backend."""

    def test_module_has_no_openai_import(self):
        import inspect

        import app.web_server as web_server

        src = inspect.getsource(web_server)
        self.assertNotIn("import openai", src)
        self.assertNotIn("from openai", src)
        self.assertNotIn("api.openai.com", src)

    def test_llm_url_is_local_llama_server(self):
        cfg = load_config("")
        self.assertIn("127.0.0.1", cfg.llama_server_url)
        self.assertIn(":8080", cfg.llama_server_url)
        self.assertEqual(cfg.llama_server_port, 8080)

    def test_default_port(self):
        self.assertEqual(DEFAULT_WEB_PORT, 8787)


if __name__ == "__main__":
    unittest.main()
