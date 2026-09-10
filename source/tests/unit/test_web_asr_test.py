"""Unit tests for the v0.4.8 mic -> ASR one-shot test (/api/asr-test).

The v0.4.7 field log: 2000 mic frames arrived, the VAD level stayed 0.00 for
93 seconds, speech_start never fired, ASR was never called, no transcript —
"I speak but nothing happens", with no way to tell WHERE the chain died
(the v0.4.7 peak log line was gated on peak >= 0.05, so the silent case
printed nothing at all).

v0.4.8 answers it with a dedicated endpoint + UI button that bypass
VAD/WS/turn state completely:

  * POST /api/asr-test — the browser records a clip (the same
    device/downsample/PCM16 path the live uplink uses) and the server runs
    it through level stats, a fresh VAD pass (would the live gate fire?)
    and the real ASR inference, then answers with a stage verdict
    (too_short / silent_mic / asr_error / asr_empty / ok) plus the
    transcript — which is also written to web-server.log.
  * GET /api/asr-test/audio — the saved recording (logs/asr_test_last.wav)
    so the user can LISTEN to what the mic actually sent.
  * The rolling VAD-peak report now ALWAYS logs (the near-zero case gets
    its own "mic uplink carries silence" warning), and the session snapshot
    carries vad_peak + vad_threshold for the Pipeline panel.

Sandbox-safe: DEMO components + fakes only; no torch/onnx/voicemem.
"""

from __future__ import annotations

import asyncio
import base64
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from app.asr import AsrEngine  # noqa: E402
from app.config import AgentConfig  # noqa: E402
from app.web_server import (  # noqa: E402
    PIPE_SAMPLE_RATE,
    WEB_SAMPLE_RATE,
    PipelineStatus,
    WebComponents,
    WebSession,
    _run_asr_test,
    _write_asr_test_wav,
    build_web_app,
)


def _tmp_config(tmpdir: str) -> AgentConfig:
    return AgentConfig(root=Path(tmpdir))


def _mock_components(tmpdir: str) -> WebComponents:
    cfg = _tmp_config(tmpdir)
    cfg.enable_speaker = False
    return WebComponents(cfg, mock=True).build()


def _pcm16_b64(samples: "np.ndarray", rate: int = WEB_SAMPLE_RATE) -> str:
    """float32 [-1, 1] -> base64 PCM16 little-endian (the uplink wire format)."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode()


def _silence(seconds: float, rate: int = WEB_SAMPLE_RATE) -> "np.ndarray":
    return np.zeros(int(rate * seconds), dtype=np.float32)


def _speechish(seconds: float, rate: int = WEB_SAMPLE_RATE) -> "np.ndarray":
    """Amplitude-modulated noise — audible to the energy VAD, not real speech."""
    n = int(rate * seconds)
    rng = np.random.default_rng(3)
    t = np.arange(n, dtype=np.float32) / rate
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)  # 3 Hz envelope
    return (0.16 * env * rng.standard_normal(n)).astype(np.float32)


class _FakeSock:
    def __init__(self) -> None:
        self.json: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.json.append(payload)

    async def send_bytes(self, raw: bytes) -> None:
        self.json.append({"type": "__binary__", "bytes": len(raw)})


class _HangingSock(_FakeSock):
    """A socket whose receive() never returns — run() blocks in its read
    loop exactly like a live browser connection."""

    async def receive(self) -> dict:
        await asyncio.sleep(3600)
        return {"type": "websocket.disconnect"}


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/asr-test — verdicts, stats, error guards (DEMO mode TestClient)
# ═══════════════════════════════════════════════════════════════════════════


class TestAsrTestEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="vm_asrtest_api_")
        cls.components = _mock_components(cls.tmp)
        cls.app = build_web_app(cls.components)
        from fastapi.testclient import TestClient

        cls.client = TestClient(cls.app)

    def test_silence_names_the_silent_mic_stage(self):
        """The v0.4.7 field case: frames flow, the audio is zeros. The verdict
        must say silent_mic (device/mute/privacy — NOT a model problem)."""
        r = self.client.post(
            "/api/asr-test", json={"audio_b64": _pcm16_b64(_silence(2.0))}
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["verdict"], "silent_mic")
        self.assertEqual(body["transcript"], "")
        self.assertEqual(body["audio"]["rms"], 0.0)
        self.assertFalse(body["vad_would_fire"])
        # the recording is kept — listening back is the fastest diagnosis
        wav = Path(self.tmp) / "logs" / "asr_test_last.wav"
        self.assertTrue(wav.is_file())
        with wave.open(str(wav)) as w:
            self.assertEqual(w.getframerate(), PIPE_SAMPLE_RATE)
        # the session/pipeline panel carries the last test summary
        self.assertIn("asr_test_last", self.components.status.session)
        self.assertIn("silent_mic", self.components.status.session["asr_test_last"])

    def test_speechish_gives_demo_transcript_and_vad_would_fire(self):
        r = self.client.post(
            "/api/asr-test", json={"audio_b64": _pcm16_b64(_speechish(2.0))}
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["verdict"], "ok")
        self.assertTrue(body["demo"])
        self.assertIn("DEMO", body["transcript"])
        self.assertGreater(body["audio"]["rms"], 0.005)
        # the demo energy VAD is 0/1 — audible noise reads as fire=True
        self.assertTrue(body["vad_would_fire"])
        self.assertGreater(body["vad"]["peak"], 0.5)
        self.assertGreater(body["vad"]["frames_above"], 0)

    def test_too_short_clip_verdict(self):
        """300 ms - 1 s clips: 200 + verdict too_short (a 400 would look like
        an API misuse to the user; the endpoint answers friendly instead)."""
        r = self.client.post(
            "/api/asr-test", json={"audio_b64": _pcm16_b64(_speechish(0.1))}
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["verdict"], "too_short")

    def test_400_on_missing_audio_key(self):
        r = self.client.post("/api/asr-test", json={})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/asr-test", json={"audio_b64": ""})
        self.assertEqual(r.status_code, 400)

    def test_400_on_invalid_base64(self):
        r = self.client.post("/api/asr-test", json={"audio_b64": "!!!not-b64!!!"})
        self.assertEqual(r.status_code, 400)

    def test_400_on_non_object_body(self):
        r = self.client.post(
            "/api/asr-test", content=b"garbage", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(r.status_code, 400)

    def test_400_on_under_50ms(self):
        r = self.client.post(
            "/api/asr-test", json={"audio_b64": _pcm16_b64(_speechish(0.02))}
        )
        self.assertEqual(r.status_code, 400)

    def test_audio_endpoint_serves_the_recording(self):
        # POST first: unittest runs methods alphabetically, so an earlier
        # test's recording cannot be relied on here.
        r = self.client.post(
            "/api/asr-test", json={"audio_b64": _pcm16_b64(_speechish(0.8))}
        )
        self.assertEqual(r.status_code, 200)
        r = self.client.get("/api/asr-test/audio")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "audio/wav")
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.assertGreater(len(r.content), 44)

    def test_audio_endpoint_404_without_a_recording(self):
        tmp = tempfile.mkdtemp(prefix="vm_asrtest_404_")
        from fastapi.testclient import TestClient

        client = TestClient(build_web_app(_mock_components(tmp)))
        r = client.get("/api/asr-test/audio")
        self.assertEqual(r.status_code, 404)

    def test_response_shape_is_the_ui_contract(self):
        r = self.client.post(
            "/api/asr-test", json={"audio_b64": _pcm16_b64(_speechish(1.0))}
        )
        body = r.json()
        for key in (
            "mode",
            "verdict",
            "transcript",
            "demo",
            "duration_ms",
            "asr_wall_ms",
            "audio",
            "vad",
            "vad_would_fire",
            "wav_url",
            "error",
            "recorded_at",
        ):
            self.assertIn(key, body)
        for key in ("peak", "avg", "threshold", "frames_above", "frames"):
            self.assertIn(key, body["vad"])
        self.assertEqual(body["mode"], "demo")


# ═══════════════════════════════════════════════════════════════════════════
# _run_asr_test core — the stage logic, straight (no HTTP layer)
# ═══════════════════════════════════════════════════════════════════════════


class TestRunAsrTestCore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vm_asrtest_core_")
        self.c = _mock_components(self.tmp)

    def test_vad_stats_reflect_the_configured_threshold(self):
        result = _run_asr_test(self.c, base64.b64decode(_pcm16_b64(_speechish(1.5))))
        # v0.4.11: the AgentConfig default (and the shipped YAML) is 0.25.
        self.assertEqual(result["vad"]["threshold"], 0.25)
        self.assertEqual(result["duration_ms"], 1500)

    def test_duration_is_measured_after_the_resample(self):
        # 2 s @ 24 kHz must be reported as ~2000 ms, not ~2000*(24/16)
        result = _run_asr_test(self.c, base64.b64decode(_pcm16_b64(_silence(2.0))))
        self.assertEqual(result["duration_ms"], 2000)

    def test_silence_sets_the_asr_test_event(self):
        _run_asr_test(self.c, base64.b64decode(_pcm16_b64(_silence(1.0))))
        events = self.c.status.session["events"]
        self.assertTrue(any("asr test" in e for e in events))


# ═══════════════════════════════════════════════════════════════════════════
# VAD observability gap: the SILENT case must log (v0.4.7 printed nothing)
# ═══════════════════════════════════════════════════════════════════════════


class TestVadPeakObservability(unittest.TestCase):
    def _feed(self, tmpdir: str, frames: int, samples: "np.ndarray") -> WebSession:
        comp = _mock_components(tmpdir)
        sock = _FakeSock()
        session = WebSession(sock, comp)

        async def scenario() -> None:
            for _ in range(frames):
                await session._on_vad_frame(samples)

        asyncio.run(scenario())
        return session

    def test_silent_uplink_logs_the_silence_warning(self):
        """The v0.4.7 log gap: 150+ silent frames printed NOTHING because the
        peak report was gated on peak >= 0.05. The near-zero case is the most
        important one — it must now log a warning naming the suspects."""
        self.tmp = tempfile.mkdtemp(prefix="vm_vadpeak_test_")
        with self.assertLogs("app.web_server", level="WARNING") as cm:
            session = self._feed(self.tmp, 160, np.zeros(512, dtype=np.float32))
        self.assertTrue(
            any("mic uplink carries silence" in line for line in cm.output),
            f"silence warning missing from {cm.output}",
        )
        s = session._c.status.session
        self.assertIn("vad_peak", s)
        self.assertEqual(s["vad_peak"], 0.0)
        self.assertIn("vad_threshold", s)
        self.assertEqual(s["vad_threshold"], 0.25)

    def test_snapshot_carries_peak_and_threshold(self):
        snap = PipelineStatus().snapshot()
        self.assertIn("vad_peak", snap["session"])
        self.assertIn("vad_threshold", snap["session"])

    def test_session_start_resets_peak(self):
        self.tmp = tempfile.mkdtemp(prefix="vm_vadreset_test_")
        comp = _mock_components(self.tmp)
        sock = _HangingSock()
        session = WebSession(sock, comp)
        session._c.status.session["vad_peak"] = 0.9  # stale value from a
        # previous session — run() must zero it on connect.

        async def scenario() -> None:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(session.run(), timeout=0.15)

        asyncio.run(scenario())
        s = session._c.status.session
        self.assertEqual(s["vad_peak"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# AsrEngine.transcribe_utterance (the one-shot path used by the endpoint)
# ═══════════════════════════════════════════════════════════════════════════


class TestTranscribeUtterance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vm_oneshot_")
        self.engine = AsrEngine(_tmp_config(self.tmp))

    def test_empty_clip_returns_empty_without_loading(self):
        """No model load for zero samples — the endpoint may hand over an
        empty buffer after its own guards; that is not an ASR failure."""
        self.assertEqual(self.engine.transcribe_utterance(np.zeros(0)), "")

    def test_raises_with_install_hint_in_the_dependency_free_sandbox(self):
        """The install-hint error path is only REACHABLE in a dependency-free
        environment (no torch/transformers): with the ASR stack installed the
        engine legitimately proceeds to a model load — a different code path
        with its own failure surface. Guard the scope instead of hanging or
        downloading in a torch-equipped gate venv (v0.5.2: the release gate
        now runs the full memory-safety battery, which REQUIRES torch)."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            _deps_present = True
        except Exception:
            _deps_present = False
        if _deps_present:
            self.skipTest("ASR deps installed — the dependency-free hint "
                          "path is not reachable in this environment")
        with self.assertRaises(RuntimeError) as cm:
            self.engine.transcribe_utterance(np.zeros(16000, dtype=np.float32))
        self.assertIn("torch", str(cm.exception))
        # last_error carries the reason (truncated to 200 chars) for the chip
        self.assertIn(str(cm.exception)[:200], self.engine.last_error)


# ═══════════════════════════════════════════════════════════════════════════
# _write_asr_test_wav — the "listen to what the mic actually sent" artifact
# ═══════════════════════════════════════════════════════════════════════════


class TestWriteAsrTestWav(unittest.TestCase):
    def test_writes_a_playable_16k_mono_wav(self):
        tmp = tempfile.mkdtemp(prefix="vm_wav_test_")
        cfg = _tmp_config(tmp)
        out = _write_asr_test_wav(cfg, np.zeros(16000, dtype=np.float32))
        self.assertIsNotNone(out)
        assert out is not None
        with wave.open(str(out)) as w:
            self.assertEqual(w.getframerate(), 16000)
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getnframes(), 16000)

    def test_never_raises_on_bad_input(self):
        tmp = tempfile.mkdtemp(prefix="vm_wav_test2_")
        cfg = _tmp_config(tmp)
        self.assertIsNone(_write_asr_test_wav(cfg, None))
        self.assertIsNone(_write_asr_test_wav(cfg, np.zeros(0, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
