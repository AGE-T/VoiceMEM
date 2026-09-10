"""Unit tests for the v0.4.19 mic capture probe + browser diagnostics.

The v0.4.18 field report: the WebSocket uplink was alive ("mic uplink
started: first frame 4096 bytes", thousands of frames) but every sample
the backend received was digital silence (VAD peak 0.00, ASR test rms
0.0, verdict silent_mic) — while the browser-side capture code was
byte-identical to the last field-verified version. The zero-audio case
had NO stage name: browser source? browser PCM conversion? transport?

v0.4.19 answers it with the minimal diagnostic path the investigation
prescribed — three RMS values measured on the SAME captured clip:

  1. browser source RMS (Float32, BEFORE any conversion)
  2. browser PCM RMS (AFTER downsample + Int16 — the exact wire bytes)
  3. backend PCM RMS (decoded from the bytes that arrived)

  * WS JSON frame {"type": "mic_probe"} — same transport as the live
    uplink; the answer is a {"type": "mic_probe_result"} frame.
  * POST /api/mic-probe — the HTTP twin (probe works without a session).
  * WS JSON frame {"type": "mic_diag"} — the browser's rolling capture
    stats, logged next to the backend's own VAD peak lines.
  * POST /api/mic-diag — the capture summary (device label, track
    readyState/enabled/muted, track settings, constraint, ctx rate).

NO VAD, NO models, NO thresholds are involved — the verdict names the
broken stage by arithmetic alone, so it works identically in DEMO mode.

Sandbox-safe: DEMO components + fakes only; no torch/onnx/voicemem.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import AgentConfig  # noqa: E402
from app.web_server import (  # noqa: E402
    WEB_SAMPLE_RATE,
    WebComponents,
    WebSession,
    _mic_probe_core,
    build_web_app,
)


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════


def _sine_b64(freq: float = 440.0, seconds: float = 1.0, amp: float = 0.5) -> str:
    n = int(WEB_SAMPLE_RATE * seconds)
    samples = [
        int(amp * 32767.0 * math.sin(2.0 * math.pi * freq * i / WEB_SAMPLE_RATE))
        for i in range(n)
    ]
    return base64.b64encode(struct.pack(f"<{n}h", *samples)).decode()


def _zeros_b64(seconds: float = 1.0) -> str:
    n = int(WEB_SAMPLE_RATE * seconds)
    return base64.b64encode(struct.pack(f"<{n}h", *([0] * n))).decode()


def _mock_components(tmpdir: str) -> WebComponents:
    cfg = AgentConfig(root=Path(tmpdir))
    cfg.enable_speaker = False
    return WebComponents(cfg, mock=True).build()


class _HangingSock:
    """A socket whose receive() never returns — run() blocks in its read
    loop exactly like a live browser connection."""

    def __init__(self) -> None:
        self.json: list[dict] = []
        self.bytes: list[int] = []

    async def send_json(self, payload: dict) -> None:
        self.json.append(payload)

    async def send_bytes(self, raw: bytes) -> None:
        self.bytes.append(len(raw))

    async def receive(self) -> dict:
        await asyncio.sleep(3600)
        return {"type": "websocket.disconnect"}


# ═══════════════════════════════════════════════════════════════════════════
# _mic_probe_core — the arithmetic + verdict decision tree
# ═══════════════════════════════════════════════════════════════════════════


class TestMicProbeCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="vm_probe_core_")
        cls.components = _mock_components(cls.tmp)

    def test_real_signal_all_three_nonzero(self):
        """The proof the field log needs: browser src RMS / browser PCM RMS /
        backend RMS — all clearly non-zero and in the same ballpark."""
        res = _mic_probe_core(
            self.components,
            _sine_b64(),
            src_rms=0.353,
            pcm_rms=0.352,
            note="unit sine",
        )
        self.assertEqual(res["verdict"], "ok")
        self.assertAlmostEqual(res["backend_rms"], 0.3535, places=3)
        self.assertAlmostEqual(res["backend_peak"], 0.5, places=2)
        self.assertEqual(res["backend_samples"], WEB_SAMPLE_RATE)
        # non-zero samples: only exact sine zeros are 0 (few per cycle)
        self.assertGreater(res["backend_nz"], WEB_SAMPLE_RATE * 0.9)
        self.assertEqual(res["browser_src_rms"], 0.353)
        self.assertEqual(res["browser_pcm_rms"], 0.352)
        self.assertEqual(res["sample_rate"], WEB_SAMPLE_RATE)
        self.assertEqual(res["duration_ms"], 1000)

    def test_browser_source_silent_names_the_stage(self):
        """The v0.4.18 field case: zeros everywhere. The verdict must point
        BELOW the app — the browser never captured a signal."""
        res = _mic_probe_core(
            self.components, _zeros_b64(), src_rms=0.0, pcm_rms=0.0
        )
        self.assertEqual(res["verdict"], "browser_source_silent")
        self.assertEqual(res["backend_rms"], 0.0)
        self.assertEqual(res["backend_nz"], 0)

    def test_browser_conversion_silent(self):
        """Source non-zero, browser PCM zero: the browser-side conversion
        produced zeros (the Int16 serialization / scaling)."""
        res = _mic_probe_core(
            self.components, _zeros_b64(), src_rms=0.05, pcm_rms=0.0
        )
        self.assertEqual(res["verdict"], "browser_conversion_silent")

    def test_transport_silent(self):
        """Browser measured non-zero source AND non-zero PCM, but the backend
        decodes zeros: the frames were zeroed in transport (framing bug)."""
        res = _mic_probe_core(
            self.components, _zeros_b64(), src_rms=0.05, pcm_rms=0.04
        )
        self.assertEqual(res["verdict"], "transport_silent")

    def test_no_audio(self):
        res = _mic_probe_core(self.components, "", src_rms=0.0, pcm_rms=0.0)
        self.assertEqual(res["verdict"], "no_audio")
        self.assertEqual(res["backend_samples"], 0)

    def test_odd_byte_length_is_handled(self):
        """A truncated frame (odd bytes) must not raise — the last odd byte
        is dropped and the rest is measured."""
        raw = base64.b64decode(_sine_b64(seconds=0.01))
        res = _mic_probe_core(
            self.components,
            base64.b64encode(raw[:-1]).decode(),
            src_rms=0.1,
            pcm_rms=0.1,
        )
        self.assertEqual(res["backend_samples"], (len(raw) - 1) // 2)
        self.assertEqual(res["verdict"], "ok")

    def test_non_numeric_browser_values_never_raise(self):
        """The browser reports come over JSON: strings/None must not break
        the verdict tree."""
        res = _mic_probe_core(
            self.components, _sine_b64(), src_rms="oops", pcm_rms=None
        )
        self.assertEqual(res["verdict"], "ok")
        self.assertEqual(res["browser_src_rms"], "oops")

    def test_log_line_carries_the_triple(self):
        """The field-log contract: one line with all three RMS values."""
        import logging

        res = _mic_probe_core(
            self.components, _sine_b64(), src_rms=0.353, pcm_rms=0.352
        )
        with self.assertLogs("app.web_server", level="INFO") as logs:
            from app.web_server import _log_mic_probe

            _log_mic_probe(self.components, res)
        line = " ".join(logs.output)
        self.assertIn("mic-probe:", line)
        self.assertIn("browser src RMS", line)
        self.assertIn("backend PCM RMS", line)
        self.assertIn("verdict ok", line)
        # the pipeline events panel gets its own row
        events = self.components.status.session.get("events", [])
        self.assertTrue(any("mic probe" in str(e) for e in events))

    def test_log_never_raises_on_broken_components(self):
        from app.web_server import _log_mic_probe

        class _Broken:
            def __getattr__(self, name):
                raise RuntimeError("boom")

        _log_mic_probe(_Broken(), {"verdict": "ok"})  # must not raise

    def test_label_lands_in_result_and_log_line(self):
        """v0.4.20: the A/B phase label (A·constrained / B·minimal) must
        reach BOTH the result dict and the log line, so one grep over
        web-server.log shows both capture variants side by side."""
        import logging

        res = _mic_probe_core(
            self.components,
            _sine_b64(),
            src_rms=0.353,
            pcm_rms=0.352,
            label="A·constrained",
        )
        self.assertEqual(res["label"], "A·constrained")
        self.assertEqual(res["verdict"], "ok")
        with self.assertLogs("app.web_server", level="INFO") as logs:
            from app.web_server import _log_mic_probe

            _log_mic_probe(self.components, res)
        line = " ".join(logs.output)
        self.assertIn("mic-probe [A·constrained]:", line)
        self.assertIn("browser src RMS", line)
        self.assertIn("verdict ok", line)

    def test_label_is_clamped_to_40_chars(self):
        res = _mic_probe_core(
            self.components,
            _sine_b64(),
            src_rms=0.353,
            pcm_rms=0.352,
            label="X" * 100,
        )
        self.assertEqual(len(res["label"]), 40)

    def test_missing_label_keeps_the_old_log_shape(self):
        """Backward contract: without a label the log line is exactly the
        v0.4.19 shape (no dangling brackets)."""
        res = _mic_probe_core(
            self.components, _sine_b64(), src_rms=0.353, pcm_rms=0.352
        )
        self.assertEqual(res["label"], "")
        with self.assertLogs("app.web_server", level="INFO") as logs:
            from app.web_server import _log_mic_probe

            _log_mic_probe(self.components, res)
        line = " ".join(logs.output)
        self.assertIn("mic-probe: browser src RMS", line)
        self.assertNotIn("mic-probe [", line)


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/mic-probe + POST /api/mic-diag (DEMO-mode TestClient)
# ═══════════════════════════════════════════════════════════════════════════


class TestMicProbeEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="vm_probe_api_")
        cls.components = _mock_components(cls.tmp)
        cls.app = build_web_app(cls.components)
        from fastapi.testclient import TestClient

        cls.client = TestClient(cls.app)

    def test_http_probe_ok(self):
        r = self.client.post(
            "/api/mic-probe",
            json={"audio_b64": _sine_b64(), "src_rms": 0.353, "pcm_rms": 0.352},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["verdict"], "ok")
        self.assertGreater(body["backend_rms"], 0.3)
        self.assertEqual(body["browser_src_rms"], 0.353)

    def test_http_probe_carries_the_ab_label(self):
        """v0.4.20: both A/B phases post with their label; the answer (and
        the log line) must carry it back so the two variants are
        distinguishable in web-server.log."""
        with self.assertLogs("app.web_server", level="INFO") as logs:
            r = self.client.post(
                "/api/mic-probe",
                json={
                    "audio_b64": _sine_b64(),
                    "src_rms": 0.353,
                    "pcm_rms": 0.352,
                    "label": "B·minimal",
                },
            )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["label"], "B·minimal")
        line = " ".join(logs.output)
        self.assertIn("mic-probe [B·minimal]:", line)

    def test_http_probe_silent(self):
        r = self.client.post(
            "/api/mic-probe",
            json={"audio_b64": _zeros_b64(), "src_rms": 0.0, "pcm_rms": 0.0},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["verdict"], "browser_source_silent")

    def test_400_on_missing_audio(self):
        r = self.client.post("/api/mic-probe", json={})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/mic-probe", json={"audio_b64": ""})
        self.assertEqual(r.status_code, 400)

    def test_400_on_invalid_base64(self):
        r = self.client.post("/api/mic-probe", json={"audio_b64": "!!!not-b64!!!"})
        self.assertEqual(r.status_code, 400)

    def test_400_on_garbage_body(self):
        r = self.client.post(
            "/api/mic-probe",
            content=b"garbage",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(r.status_code, 400)

    def test_mic_diag_endpoint_logs_and_answers_ok(self):
        payload = {
            "source": "live",
            "device": "Test Mic (unit)",
            "ready_state": "live",
            "enabled": True,
            "muted": False,
            "ctx_rate": 48000,
            "track_settings": {
                "sampleRate": 48000,
                "channelCount": 1,
                "sampleSize": 16,
                "echoCancellation": True,
                "autoGainControl": True,
                "noiseSuppression": False,
            },
            "constraint": '{"audio":{}}',
        }
        with self.assertLogs("app.web_server", level="INFO") as logs:
            r = self.client.post("/api/mic-diag", json=payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True})
        line = " ".join(logs.output)
        self.assertIn("browser mic capture:", line)
        self.assertIn("Test Mic (unit)", line)
        self.assertIn("48000 Hz", line)

    def test_mic_diag_never_500s_on_garbage(self):
        r = self.client.post("/api/mic-diag", content=b"{{{{")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True})


# ═══════════════════════════════════════════════════════════════════════════
# WS text frames: {"type": "mic_diag"} logging + {"type": "mic_probe"} reply
# ═══════════════════════════════════════════════════════════════════════════


class TestWsMicFrames(unittest.TestCase):
    def _session(self) -> tuple[WebSession, WebComponents, _HangingSock]:
        tmp = tempfile.mkdtemp(prefix="vm_probe_ws_")
        components = _mock_components(tmp)
        sock = _HangingSock()
        session = WebSession(sock, components)
        return session, components, sock

    def test_mic_diag_frame_is_logged(self):
        session, components, _ = self._session()
        frame = json.dumps(
            {
                "type": "mic_diag",
                "frames": 120,
                "src_rms": 0.041,
                "src_peak": 0.31,
                "pcm_rms": 0.039,
                "pcm_peak": 0.30,
                "nz": 40000,
                "samples": 98304,
                "ctx_rate": 48000,
                "silent": False,
                "device": "WS Test Mic",
            }
        )
        with self.assertLogs("app.web_server", level="INFO") as logs:
            asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
                session._on_text(frame)
            )
        line = " ".join(logs.output)
        self.assertIn("browser mic diag:", line)
        self.assertIn("src rms 0.041", line)
        self.assertIn("pcm rms 0.039", line)
        self.assertIn("WS Test Mic", line)

    def test_mic_probe_frame_answers_mic_probe_result(self):
        session, components, sock = self._session()
        frame = json.dumps(
            {
                "type": "mic_probe",
                "audio_b64": _sine_b64(),
                "src_rms": 0.353,
                "pcm_rms": 0.352,
            }
        )
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            session._on_text(frame)
        )
        replies = [m for m in sock.json if m.get("type") == "mic_probe_result"]
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["verdict"], "ok")
        self.assertGreater(replies[0]["backend_rms"], 0.3)

    def test_mic_probe_frame_carries_the_ab_label(self):
        """v0.4.20: the WS probe frame of each A/B phase carries its label;
        the reply (and the backend log) must name the phase."""
        session, components, sock = self._session()
        frame = json.dumps(
            {
                "type": "mic_probe",
                "audio_b64": _sine_b64(),
                "src_rms": 0.243,
                "pcm_rms": 0.242,
                "label": "A·constrained",
                "note": "ws probe A·constrained",
            }
        )
        with self.assertLogs("app.web_server", level="INFO") as logs:
            asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
                session._on_text(frame)
            )
        replies = [m for m in sock.json if m.get("type") == "mic_probe_result"]
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["label"], "A·constrained")
        self.assertEqual(replies[0]["verdict"], "ok")
        line = " ".join(logs.output)
        self.assertIn("mic-probe [A·constrained]:", line)

    def test_mic_probe_frame_with_bad_b64_answers_error_not_silence(self):
        """A broken probe must answer with an error frame — never a fake
        ok, never silence."""
        session, components, sock = self._session()
        frame = json.dumps(
            {"type": "mic_probe", "audio_b64": "###", "src_rms": 0.1}
        )
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            session._on_text(frame)
        )
        replies = [m for m in sock.json if m.get("type") == "mic_probe_result"]
        self.assertEqual(len(replies), 1)
        self.assertIn(replies[0].get("verdict", ""), ("no_audio",))
        self.assertEqual(replies[0].get("backend_samples"), 0)

    def test_user_text_still_dispatches_untouched(self):
        """Regression: the v0.4.13/v0.4.14 text-turn path must work exactly
        as before — the new frame types are additive, not a rewrite."""
        import time

        session, components, sock = self._session()
        from app.mock_components import MockAsrEngine

        components.make_asr = lambda: MockAsrEngine(queue=["Szia!"])  # type: ignore[method-assign]

        async def run() -> None:
            await session._on_text('{"type": "user_text", "text": "Szia!"}')
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                trail = " ".join(components.status.session.get("events", []))
                if "answer done" in trail:
                    break
                await asyncio.sleep(0.05)
            turn = session._turn_task
            if turn is not None and not turn.done():
                await asyncio.wait_for(asyncio.shield(turn), timeout=15)

        asyncio.run(asyncio.wait_for(run(), timeout=40))
        # the turn ran: user_transcript went out with source=text
        ut = [m for m in sock.json if m.get("type") == "user_transcript"]
        self.assertEqual(len(ut), 1)
        self.assertEqual(ut[0]["source"], "text")

    def test_unknown_types_still_ignored(self):
        session, components, sock = self._session()
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            session._on_text('{"type": "whatever"}')
        )
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            session._on_text("[1,2,3]")
        )
        self.assertEqual(
            [m for m in sock.json if m.get("type") == "mic_probe_result"], []
        )


# ═══════════════════════════════════════════════════════════════════════════
# UI contract markers — the browser half of the diagnostic path
# ═══════════════════════════════════════════════════════════════════════════


class TestUiMarkers(unittest.TestCase):
    """The capture layer must actually be IN the page (the v0.4.11 stale-page
    lesson: a missing marker means the browser runs old JS)."""

    PAGE = (Path(__file__).resolve().parent.parent.parent / "web" / "voicemem.html").read_text(
        encoding="utf-8"
    )

    def test_capture_probe_button_and_panel_exist(self):
        for marker in (
            'id="micProbeBtn"',
            'id="micDiag"',
            'id="micProbeOut"',
            "acquireMic(",
            "measureAudioFrame(e,",
            "maybeSendMicDiag(",
            "type:'mic_probe'",
            "type:'mic_diag'",
            "/api/mic-probe",
            "/api/mic-diag",
        ):
            self.assertIn(marker, self.PAGE, f"missing UI marker: {marker}")

    def test_watchdog_and_stale_device_guard_exist(self):
        for marker in (
            "MIC_SILENT_PEAK",
            "stale deviceId",
            "micSilentRetry",
            "startMicCapture(true)",
            "srcPeak<MIC_SILENT_PEAK",
        ):
            self.assertIn(marker, self.PAGE, f"missing watchdog marker: {marker}")

    def test_all_three_capture_paths_use_the_instrumented_layer(self):
        """mic test / ASR test / live uplink / the A/B probe phase all go
        through the measured capture — no path bypasses the
        instrumentation."""
        # 4 real call sites: mic-test / asr-test / live / probePhase (the
        # ONE probe-phase call serves BOTH the A-constrained and the
        # B-minimal {audio:true} variant — the minimal flag is the argument)
        self.assertEqual(self.PAGE.count("await acquireMic("), 4)
        self.assertGreaterEqual(self.PAGE.count("measureAudioFrame(e,"), 4)

    def test_ab_probe_exists(self):
        """v0.4.20: the A/B comparison — constrained capture vs the minimal
        {audio:true} retry, side by side, with the explicit Q1 answer."""
        for marker in (
            "function probePhase(tag,minimal,rateInfo)",
            "function renderABProbeResult(A,B,rateInfo)",
            "micProbeRecordingA",
            "micProbeRecordingB",
            "micProbeQ1",
            "micProbeABVerdictAZero",
            "micProbeABVerdictBothSilent",
            "micProbeRateRow",
            "label:label",
            "device_id_status",
            ".probe .ph",
        ):
            self.assertIn(marker, self.PAGE, f"missing A/B probe marker: {marker}")
        # phase B really is the minimal constraint: {audio:true}, no flags
        self.assertIn("acquireMic('probe-'+tag.toLowerCase(),minimal)", self.PAGE)
        self.assertIn("A=await probePhase('A',false,rateInfo)", self.PAGE)
        self.assertIn("B=await probePhase('B',true,rateInfo)", self.PAGE)
        self.assertIn("constraint={audio:true};", self.PAGE)

    def test_version_bumped(self):
        self.assertIn("PAGE_VERSION='0.5.2'", self.PAGE)


if __name__ == "__main__":
    unittest.main()
