"""End-to-end integration tests for the LOCAL web backend.

Covers (task TESTS list):
  * web backend startup                 -> build_web_app + uvicorn on ephemeral port
  * health endpoint                     -> GET /api/health 200
  * WebSocket connection                -> /ws handshake, session_ready
  * text chat                           -> user_text -> transcript/hits/deltas/done + AUDIO
  * local LLM request                   -> a REAL stub llama-server over localhost HTTP
                                            (SSE streaming /v1/chat/completions)
  * local JSON request                  -> chat_json via /api/title (response_format json_object)
  * Piper audio generation              -> the TTS stage returns PCM16 audio to the browser
                                            (DemoTtsEngine here; the real Piper is exercised on
                                            the target machine by scripts/smoke_test_web.py)
  * microphone capability path          -> WS binary frames drive VAD -> ASR -> turn
  * no OpenAI runtime calls             -> socket auditor: any non-loopback connect FAILS

Sandbox-safe: the heavy components are DEMO stand-ins; the LLM is the REAL
LlmClient (httpx) pointed at an in-process stub that mimics llama.cpp's
OpenAI-compatible API. Nothing leaves 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import websockets  # noqa: E402

from app.config import AgentConfig  # noqa: E402
from app.llm import LlmClient  # noqa: E402
from app.web_server import WebComponents, build_web_app  # noqa: E402

_STUB_REPLY = (
    "Rendben, ertem. Ez a valasz a helyi llama-server stubbol jon, "
    "SSE darabonkent erkezik. folytatjuk."
)


class _LlamaStubHandler(BaseHTTPRequestHandler):
    """Mimics llama.cpp llama-server: /health, /v1/chat/completions (SSE + JSON)."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        want_stream = bool(payload.get("stream"))
        want_json = (payload.get("response_format") or {}).get("type") == "json_object"
        if want_json:
            content = json.dumps({"title": "stub summary title"})
        else:
            content = _STUB_REPLY
        if want_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for word in content.split(" "):
                chunk = {
                    "id": "stub",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": word + " "}}],
                }
                data = f"data: {json.dumps(chunk)}\n\n".encode()
                self.wfile.write(hex(len(data))[2:].encode() + b"\r\n" + data + b"\r\n")
                self.wfile.flush()
            term = b"data: [DONE]\n\n"
            self.wfile.write(hex(len(term))[2:].encode() + b"\r\n" + term + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            body = json.dumps(
                {
                    "id": "stub",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


class _LlamaStub:
    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _LlamaStubHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"


def _pcm16_sine(seconds: float, freq: float = 220.0, amp: float = 0.6, sr: int = 24000) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    x = np.sin(2 * np.pi * freq * t) * amp
    return (x * 32767).astype(np.int16).tobytes()


def _silence(seconds: float, sr: int = 24000) -> bytes:
    return np.zeros(int(sr * seconds), dtype=np.int16).tobytes()


class TestWebE2E(unittest.TestCase):
    """Full-stack over a real uvicorn server on an ephemeral port."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.tmp = tempfile.mkdtemp(prefix="vm_web_e2e_")
        cls.stub = _LlamaStub()
        cls.stub.start()

        cfg = AgentConfig(root=Path(cls.tmp))
        # point the REAL LlmClient at the local stub llama-server
        cfg.llama_server_host = "127.0.0.1"
        cfg.llama_server_port = cls.stub.port
        cls.components = WebComponents(cfg, mock=True).build()
        # Swap in the REAL LLM client (httpx -> localhost stub).
        cls.components.llm = LlmClient(cfg)

        cls.app = build_web_app(cls.components)
        import uvicorn

        cls.config = uvicorn.Config(cls.app, host="127.0.0.1", port=0, log_level="error")
        cls.server = uvicorn.Server(cls.config)
        cls.port = cls.config.port  # 0 -> picked by the OS
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        # wait for the server socket
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", cls.port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.1)
        # uvicorn with port=0 logs the real port; find it from the server state
        for _ in range(50):
            if cls.server.servers:
                break
            time.sleep(0.1)
        if cls.server.servers:
            cls.port = cls.server.servers[0].sockets[0].getsockname()[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        cls.stub.stop()

    # -- helpers -------------------------------------------------------------- #

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws"

    # -- tests ---------------------------------------------------------------- #

    def test_health_endpoint(self):
        import httpx

        with httpx.Client(timeout=5) as client:
            r = client.get(self._url("/api/health"))
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        # the REAL LlmClient health-checked the stub llama-server
        self.assertTrue(body["llama"]["healthy"])
        self.assertEqual(body["llama"]["url"], self.stub.base_url)

    def test_ui_page_served(self):
        import httpx

        with httpx.Client(timeout=5) as client:
            r = client.get(self._url("/"))
        self.assertEqual(r.status_code, 200)
        self.assertIn("VoiceMem", r.text)

    def test_websocket_text_chat_with_local_llm_and_audio(self):
        """The golden path: typed turn -> stub llama-server SSE -> audio bytes."""

        async def run():
            events = []
            audio_bytes = 0
            async with websockets.connect(self._ws_url(), max_size=2**22) as ws:
                ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                self.assertEqual(ready["type"], "session_ready")
                events.append(ready["type"])
                # drain pipeline_status
                await asyncio.sleep(0.3)

                await ws.send(json.dumps({"type": "user_text", "text": "Szia, jo napot!"}))
                deadline = time.time() + 15
                while time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                    except asyncio.TimeoutError:
                        break
                    if isinstance(msg, bytes):
                        audio_bytes += len(msg)
                        self.assertGreater(len(msg), 100)  # real PCM chunk
                        continue
                    data = json.loads(msg)
                    events.append(data["type"])
                    if data["type"] == "user_transcript":
                        self.assertEqual(data["source"], "text")
                        self.assertEqual(data["text"], "Szia, jo napot!")
                    if data["type"] == "answer_delta":
                        self.assertTrue(data["text"])
                    if data["type"] == "answer_done":
                        self.assertIn("timings", data)
                        break
            return events, audio_bytes

        events, audio_bytes = asyncio.run(run())
        for expected in (
            "session_ready",
            "user_transcript",
            "memory_hits",
            "answer_start",
            "answer_delta",
            "answer_done",
        ):
            self.assertIn(expected, events)
        self.assertGreater(audio_bytes, 10000)  # Piper-stage audio reached the browser

    def test_websocket_microphone_path(self):
        """Mic frames: loud sine (speech) -> hangover silence -> VAD end -> turn."""
        from app.mock_components import MockAsrEngine

        self.components.make_asr = lambda: MockAsrEngine(queue=["Ez egy mikrofon teszt."])

        async def run():
            events = []
            async with websockets.connect(self._ws_url(), max_size=2**22) as ws:
                ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                self.assertEqual(ready["type"], "session_ready")
                # let the session's audio loop spin up
                await asyncio.sleep(0.2)
                for _ in range(6):  # ~1.0 s of loud speech
                    await ws.send(_pcm16_sine(0.17))
                    await asyncio.sleep(0.02)
                for _ in range(6):  # >1 s of silence -> hangover -> SPEECH_END
                    await ws.send(_silence(0.17))
                    await asyncio.sleep(0.02)
                deadline = time.time() + 15
                while time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                    except asyncio.TimeoutError:
                        break
                    if isinstance(msg, bytes):
                        continue
                    data = json.loads(msg)
                    if data["type"] in (
                        "partial_transcript",
                        "user_transcript",
                        "memory_hits",
                        "answer_done",
                    ):
                        events.append((data["type"], data.get("text", "")))
                    if data["type"] == "answer_done":
                        break
            return events

        events = asyncio.run(run())
        types = [e[0] for e in events]
        self.assertIn("user_transcript", types)
        transcript = [t for ty, t in events if ty == "user_transcript"][0]
        self.assertEqual(transcript, "Ez egy mikrofon teszt.")  # ASR ran (mock queue)
        self.assertIn("answer_done", types)

    def test_websocket_mic_turn_with_hung_emotion_still_answers(self):
        """v0.4.13: the emotion-init hang that ate the v0.4.12 field turns.

        The field report: 'the ASR test transcribes real Hungarian, the text
        is sent to the agent, the transcript appears - then nothing reaches
        the LLM.' Root cause: _run_turn awaited the emotion analyzer
        construction with NO timeout (funasr/torch import). Here the same
        turn runs over the REAL WebSocket with REAL binary mic frames and
        the REAL httpx LLM client against the stub llama-server, while the
        emotion analyzer construction HANGS - the turn must still deliver
        the full stage trail, an LLM answer and TTS audio bytes.
        """
        import unittest.mock

        from app.mock_components import MockAsrEngine

        self.components.make_asr = lambda: MockAsrEngine(
            queue=["Ez egy test, a mikrofonon be van kapcsolva."]
        )

        def hung_emotion_analyzer():
            time.sleep(30.0)  # simulates the funasr/torch import stall
            return None

        original_analyzer = self.components.emotion_analyzer
        self.components.emotion_analyzer = hung_emotion_analyzer  # type: ignore[method-assign]

        async def run():
            events = []
            audio_bytes = 0
            try:
                async with websockets.connect(self._ws_url(), max_size=2**22) as ws:
                    ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    self.assertEqual(ready["type"], "session_ready")
                    await asyncio.sleep(0.2)
                    for _ in range(6):  # ~1.0 s of loud speech
                        await ws.send(_pcm16_sine(0.17))
                        await asyncio.sleep(0.02)
                    for _ in range(6):  # silence -> hangover -> SPEECH_END
                        await ws.send(_silence(0.17))
                        await asyncio.sleep(0.02)
                    deadline = time.time() + 20
                    while time.time() < deadline:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=3)
                        except asyncio.TimeoutError:
                            break
                        if isinstance(msg, bytes):
                            audio_bytes += len(msg)
                            continue
                        data = json.loads(msg)
                        events.append((data["type"], data))
                        if data["type"] == "answer_done":
                            break
            finally:
                self.components.emotion_analyzer = original_analyzer  # type: ignore[method-assign]
            return events, audio_bytes

        # cut the init budget to 0.3 s so the test runs fast; the REAL
        # production constant is 5.0 s (see _EMOTION_INIT_TIMEOUT_S)
        with unittest.mock.patch("app.web_server._EMOTION_INIT_TIMEOUT_S", 0.3):
            events, audio_bytes = asyncio.run(asyncio.wait_for(run(), timeout=40))
        types = [e[0] for e in events]
        # the turn ran the full pipeline despite the hang
        self.assertIn("user_transcript", types)
        self.assertIn("answer_delta", types)
        self.assertIn("answer_done", types)
        done = [d for t, d in events if t == "answer_done"][0]
        self.assertFalse(done.get("error"), done.get("error"))
        self.assertGreater(audio_bytes, 10000, "TTS audio never reached the browser")
        # the stage trail names the blocked stage AND the recovery
        import httpx

        with httpx.Client(timeout=5) as client:
            snap = client.get(self._url("/api/pipeline")).json()
        trail = " | ".join(snap["session"].get("events", []))
        for stage in (
            "turn received",
            "memory start",
            "emotion start",
            "emotion done (init timed out",
            "memory done",
            "llm start",
            "llm first token",
            "llm done",
            "tts start",
            "tts done",
            "answer done",
        ):
            self.assertIn(stage, trail, f"stage {stage!r} missing: {trail}")

    def test_websocket_asr_empty_is_reported(self):
        """v0.4.1: speech with an EMPTY transcript must reach the browser.

        Before the fix this turn was dropped silently - the exact "I speak,
        nothing happens" field report. The browser now gets asr_empty and
        the /api/pipeline snapshot carries the event trail + mic stats.
        """
        from app.mock_components import MockAsrEngine

        self.components.make_asr = lambda: MockAsrEngine(queue=[])  # flush -> ""

        async def run():
            events = []
            async with websockets.connect(self._ws_url(), max_size=2**22) as ws:
                ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                self.assertEqual(ready["type"], "session_ready")
                await asyncio.sleep(0.2)
                for _ in range(6):  # ~1.0 s of loud speech
                    await ws.send(_pcm16_sine(0.17))
                    await asyncio.sleep(0.02)
                for _ in range(6):  # silence -> hangover -> SPEECH_END
                    await ws.send(_silence(0.17))
                    await asyncio.sleep(0.02)
                deadline = time.time() + 10
                while time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                    except asyncio.TimeoutError:
                        break
                    if isinstance(msg, bytes):
                        continue
                    data = json.loads(msg)
                    if data["type"] in ("asr_empty", "user_transcript", "answer_done"):
                        events.append((data["type"], data))
                        if data["type"] == "asr_empty":
                            break
            return events

        events = asyncio.run(run())
        types = [e[0] for e in events]
        self.assertIn("asr_empty", types)
        payload = dict(next(e[1] for e in events if e[0] == "asr_empty"))
        self.assertGreaterEqual(payload["audio_ms"], 400)
        # user_transcript / answer_done must NOT appear - no text, no turn
        self.assertNotIn("user_transcript", types)
        self.assertNotIn("answer_done", types)

        # the pipeline snapshot shows the mic uplink + the event trail
        import httpx

        with httpx.Client(timeout=5) as client:
            snap = client.get(self._url("/api/pipeline")).json()
        session = snap["session"]
        self.assertGreater(session["mic_frames"], 0)
        self.assertTrue(any("ASR EMPTY" in e for e in session["events"]))
        self.assertEqual(snap["components"]["asr"]["state"], "error")
        self.assertIn("no transcript", snap["components"]["asr"]["error"])

    def test_pipeline_snapshot_carries_session_diagnostics(self):
        import httpx

        with httpx.Client(timeout=5) as client:
            snap = client.get(self._url("/api/pipeline")).json()
        session = snap["session"]
        for key in (
            "connected",
            "mic_frames",
            "vad_level",
            "vad_in_speech",
            "speech_ms",
            "utterances",
            "turns",
            "last_event",
            "events",
        ):
            self.assertIn(key, session)

    def test_title_uses_local_llm_json_mode(self):
        import httpx

        with httpx.Client(timeout=10) as client:
            r = client.post(
                self._url("/api/title"), json={"text": "we spoke about Budapest and the park"}
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["title"], "stub summary title")

    def test_llm_client_json_and_error_paths(self):
        async def run():
            llm = LlmClient(self.components.config)
            # json_object mode against the stub
            data = await llm.chat_json([{"role": "user", "content": "summarise"}])
            self.assertEqual(data["title"], "stub summary title")
            # streaming against the stub
            words = []
            async for w in llm.chat_stream([{"role": "user", "content": "hi"}]):
                words.append(w)
            self.assertIn("llama-server", "".join(words))
            await llm.aclose()
            # dead port -> LlmUnavailableError
            from app.config import AgentConfig as _AC

            dead_cfg = _AC(root=Path(self.tmp))
            dead_cfg.llama_server_port = 1  # nothing listens here
            dead_llm = LlmClient(dead_cfg)
            try:
                async for _ in dead_llm.chat_stream([{"role": "user", "content": "x"}]):
                    pass
                raised = False
            except Exception as exc:
                raised = isinstance(exc, Exception)
                self.assertIn("Llm", type(exc).__name__)
            self.assertTrue(raised)
            self.assertFalse(await dead_llm.health_check())
            await dead_llm.aclose()

        asyncio.run(run())

    def test_no_openai_runtime_calls(self):
        """OPENAI RUNTIME CALLS: MUST BE 0.

        Audits every socket connection during a full text turn: anything that
        tries to leave 127.0.0.1 raises (and fails the test). The only
        outbound connection must be the llama-server stub on localhost.
        """
        loopback_only = []
        original_connect = socket.socket.connect

        def guarded_connect(self, address, *args, **kwargs):
            host = address[0] if isinstance(address, tuple) else str(address)
            if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::"):
                loopback_only.append(host)
                raise AssertionError(f"NON-LOCAL CONNECTION ATTEMPT: {address}")
            return original_connect(self, address, *args, **kwargs)

        socket.socket.connect = guarded_connect
        try:

            async def run():
                async with websockets.connect(self._ws_url(), max_size=2**22) as ws:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                    await ws.send(json.dumps({"type": "user_text", "text": "offline audit"}))
                    deadline = time.time() + 15
                    while time.time() < deadline:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=3)
                        except asyncio.TimeoutError:
                            break
                        if isinstance(msg, bytes):
                            continue
                        if json.loads(msg).get("type") == "answer_done":
                            break

            asyncio.run(run())
        finally:
            socket.socket.connect = original_connect
        self.assertEqual(loopback_only, [])

    def test_tts_audio_is_valid_pcm16(self):
        """Piper-stage output contract: int16 mono at the engine sample rate."""
        import tempfile

        from app.web_server import float32_to_pcm16_bytes, resample_linear

        pcm = self.components.tts.synthesize("hello", "en")
        self.assertIsNotNone(pcm)
        self.assertEqual(pcm.dtype, np.int16)
        # web path: resample 22050 -> 24000 and pack PCM16 LE
        f32 = pcm.astype(np.float32) / 32768.0
        out = float32_to_pcm16_bytes(resample_linear(f32, 22050, 24000))
        self.assertGreater(len(out), 1000)
        self.assertEqual(len(out) % 2, 0)
        # wrap into a real WAV and read it back with the stdlib wave reader
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(out)
            with wave.open(path, "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getframerate(), 24000)
                frames = w.readframes(w.getnframes())
                self.assertGreater(len(frames), 1000)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
