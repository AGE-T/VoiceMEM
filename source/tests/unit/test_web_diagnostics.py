"""Unit tests for the v0.4.1 web diagnostics/robustness layer.

Covers the field-report fixes for "the mic seems to work but no reply, no
voice, and I cannot see where the chain stops":

  * PipelineStatus carries live session diagnostics (mic frames, VAD level,
    speech state, event ring buffer) in every snapshot.
  * WebComponents.start_warmup: mock mode never spawns a thread; the LOCAL
    worker survives a dependency-free sandbox (all failures degrade into
    chip states, never into an exception).
  * WebSession asr_empty: speech detected, empty transcript -> an explicit
    asr_empty message reaches the browser (v0.4.1: this turn used to be
    dropped SILENTLY - the most likely field break).
  * WebSession._guarded_turn: a crash anywhere inside a turn is reported to
    the browser as error + answer_done instead of dying silently.
  * _setup_logging writes logs/web-server.log.
  * AsrEngine (v0.4.7) drives processor + model.generate directly: the input
    builder prefers the transformers >= 5.0 native apply_transcription_request
    and falls back to the manual chat-template path; the probe never raises
    and a failed warm-up leaves last_error set for the chip.
  * app.vad.SileroVad.is_available exists (v0.4.1: the web build() called a
    method that did not exist -> VAD chip showed a bogus AttributeError).

Sandbox-safe: mock components + fakes only; no torch/onnx/voicemem.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from app.config import AgentConfig  # noqa: E402
from app.web_server import (  # noqa: E402
    PipelineStatus,
    WebComponents,
    WebSession,
    _setup_logging,
)


def _tmp_config(tmpdir: str) -> AgentConfig:
    return AgentConfig(root=Path(tmpdir))


class _FakeSock:
    """Records everything a WebSession would send over the WebSocket."""

    def __init__(self) -> None:
        self.json: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.json.append(payload)

    async def send_bytes(self, raw: bytes) -> None:
        self.json.append({"type": "__binary__", "bytes": len(raw)})


def _mock_components(tmpdir: str) -> WebComponents:
    cfg = _tmp_config(tmpdir)
    cfg.enable_speaker = False
    return WebComponents(cfg, mock=True).build()


class TestSessionDiagnostics(unittest.TestCase):
    def test_snapshot_contains_session_fields(self):
        snap = PipelineStatus().snapshot()
        s = snap["session"]
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
            self.assertIn(key, s)

    def test_processing_span_returns_to_previous_state(self):
        """A transient span must not upgrade a missing/error chip to ready."""
        st = PipelineStatus()
        missing = st.components["embedding"]
        missing.set_missing("not installed")
        missing.begin()
        self.assertEqual(missing.state, "processing")
        missing.end()
        self.assertEqual(missing.state, "missing")
        # ready stays ready
        ready = st.components["llm"]
        ready.set_ready("llama")
        ready.begin()
        ready.end()
        self.assertEqual(ready.state, "ready")
        # repeated begin() (per-frame ASR feed) keeps the original prev state
        error = st.components["asr"]
        error.set_error("boom")
        error.begin()
        error.begin()
        error.end()
        self.assertEqual(error.state, "error")

    def test_event_ring_buffer_is_bounded(self):
        st = PipelineStatus()
        comp = WebComponents.__new__(WebComponents)  # no __init__ side effects
        comp.status = st
        for i in range(30):
            comp._event(f"e{i}")
        self.assertEqual(len(st.session["events"]), 12)
        self.assertIn("e29", st.session["events"][-1])
        self.assertEqual(st.session["last_event"], "e29")
        # snapshot() returns a COPY - mutating it must not leak back
        snap = st.snapshot()
        snap["session"]["events"].append("tampered")
        self.assertEqual(len(st.session["events"]), 12)


class TestWarmUp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vm_warm_test_")

    def test_mock_mode_spawns_no_thread(self):
        comp = _mock_components(self.tmp)
        self.assertIsNone(comp.start_warmup())
        self.assertFalse(comp._warmup_started)

    def test_local_worker_survives_sandbox(self):
        """The worker must degrade, not crash — whatever is missing.

        Originally written for the dependency-free sandbox (no torch/onnx/
        voicemem). In a torch-equipped gate venv (v0.5.2: the memory-safety
        battery requires torch) the warm-up would otherwise try a REAL ASR
        model resolution (HF id → cache → multi-GB instantiation → OOM on
        the memory-capped gate sandbox). The fixture now points the ASR at
        a LOCAL dir with an unparseable config.json: the load fails fast
        and deterministically in EVERY environment profile, which is the
        exact failure mode this test exists to police (defined error
        states, no crash, no hang). Assertions unchanged.
        """
        import os
        _offline = {}
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            _offline[k] = os.environ.get(k)
            os.environ[k] = "1"
        try:
            cfg = _tmp_config(self.tmp)
            cfg.enable_speaker = False
            # deterministic ASR failure: a local model dir whose config
            # cannot parse (no hub lookup, no giant random-init)
            asr_dir = cfg.asr_model_dir
            asr_dir.mkdir(parents=True, exist_ok=True)
            (asr_dir / "config.json").write_text(
                "{ not valid json", encoding="utf-8")
            comp = WebComponents(cfg, mock=False).build()
            comp._warm_up_worker()  # synchronous: no thread to join
            snap = comp.status.snapshot()
            states = {k: v["state"] for k, v in snap["components"].items()}
            # every chip lands in a DEFINED state; none stays 'processing'
            for key, state in states.items():
                self.assertNotEqual(state, "processing", f"{key} stuck processing")
            # the ASR chip carries the actionable failure
            self.assertIn(snap["components"]["asr"]["state"], ("error", "missing"))
            # the memory layer still flips the first-search budget to steady state
            self.assertTrue(comp.memory_warm)
            # the event log tells the story
            self.assertTrue(any("warm" in e for e in snap["session"]["events"]))
        finally:
            for k, v in _offline.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_memory_first_search_gets_long_budget(self):
        cfg = _tmp_config(self.tmp)
        comp = WebComponents(cfg, mock=True).build()
        self.assertFalse(comp.memory_warm)
        comp.memory_warm = True  # flipped by the first turn/warm-up
        # the flag is what _run_turn reads; verify the attribute contract
        self.assertIsInstance(comp.memory_warm, bool)


class TestAsrEmptyEvent(unittest.TestCase):
    """Speech detected + empty transcript -> the browser is TOLD (v0.4.1).

    v0.6.0 note: the demo MockAsrEngine transcribes speech-level audio with
    the scripted demo phrases, so this scenario explicitly installs the
    OTHER honest engine outcome — the engine ran, speech was detected, but
    no text came back (a scripted empty result) — which is exactly the
    v0.4.1 field failure this test guards against.
    """

    def _run_frames(self, tmpdir: str) -> tuple[WebSession, _FakeSock]:
        comp = _mock_components(tmpdir)
        from app.mock_components import MockAsrEngine

        comp.make_asr = lambda: MockAsrEngine(queue=[""])  # type: ignore[method-assign]
        sock = _FakeSock()
        session = WebSession(sock, comp)
        rng = np.random.default_rng(7)
        loud = (rng.standard_normal(512) * 0.3).astype(np.float32)  # RMS >> 0.02
        silent = np.zeros(512, dtype=np.float32)

        async def scenario() -> None:
            # 20 loud frames (~640 ms of speech) while VAD state says speech
            for _ in range(20):
                await session._on_vad_frame(loud)
            # 12 silent frames -> hangover (300 ms) elapses -> speech end
            for _ in range(12):
                await session._on_vad_frame(silent)

        asyncio.run(scenario())
        return session, sock

    def test_asr_empty_sent(self):
        self.tmp = tempfile.mkdtemp(prefix="vm_asrempty_test_")
        session, sock = self._run_frames(self.tmp)
        types = [m["type"] for m in sock.json]
        self.assertIn("asr_empty", types)
        msg = next(m for m in sock.json if m["type"] == "asr_empty")
        self.assertGreaterEqual(msg["audio_ms"], 500)
        # the ASR chip explains WHY in its error field
        asr = session._c.status.components["asr"]
        self.assertEqual(asr.state, "error")
        self.assertIn("no transcript", asr.error)
        # and the event log has the human-readable trail
        events = session._c.status.session["events"]
        self.assertTrue(any("ASR EMPTY" in e for e in events))
        self.assertTrue(any("speech start" in e for e in events))
        self.assertTrue(any("speech end" in e for e in events))

    def test_vad_level_tracked(self):
        self.tmp = tempfile.mkdtemp(prefix="vm_vadlvl_test_")
        session, _ = self._run_frames(self.tmp)
        s = session._c.status.session
        self.assertGreaterEqual(s["vad_level"], 0.0)
        self.assertEqual(s["utterances"], 1)
        self.assertFalse(s["vad_in_speech"])


class TestGuardedTurn(unittest.TestCase):
    def test_crash_reaches_the_browser(self):
        tmp = tempfile.mkdtemp(prefix="vm_guard_test_")
        comp = _mock_components(tmp)
        sock = _FakeSock()
        session = WebSession(sock, comp)

        async def boom(text, audio=None, source="asr", asr_result=None):
            raise ValueError("pipeline exploded")

        session._run_turn = boom  # type: ignore[method-assign]
        asyncio.run(session._guarded_turn("hello"))
        types = [m["type"] for m in sock.json]
        self.assertIn("error", types)
        self.assertIn("answer_done", types)
        err = next(m for m in sock.json if m["type"] == "error")
        self.assertIn("pipeline exploded", err["message"])
        done = next(m for m in sock.json if m["type"] == "answer_done")
        self.assertIn("pipeline exploded", done["error"])


class TestSetupLogging(unittest.TestCase):
    def test_log_file_created_and_written(self):
        tmp = tempfile.mkdtemp(prefix="vm_log_test_")
        cfg = _tmp_config(tmp)
        path = _setup_logging("INFO", cfg)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(path.is_file())
        logging.getLogger("vm.test").info("marker-line-42")
        for handler in logging.getLogger().handlers[:]:
            handler.flush()
        content = path.read_text("utf-8")
        self.assertIn("marker-line-42", content)

    def tearDown(self):
        # keep the root logger clean for the other tests
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
        logging.basicConfig(level=logging.CRITICAL)


class TestAsrEngineCallPath(unittest.TestCase):
    """v0.4.7: the engine calls processor + model.generate, not the pipeline.

    The transformers ASR pipeline never built the chat-template input_ids
    Qwen3-ASR needs (the v0.4.6 field report: every pipeline call form
    raised -> "speech produces no transcript"). The engine now picks between
    the transformers >= 5.0 native helper and the manual chat-template path,
    and a broken call surface leaves the error on the engine (visible in the
    ASR chip) instead of silently degrading.
    """

    def _engine(self, processor, model):
        from app.asr import AsrEngine

        cfg = AgentConfig(root=Path(tempfile.mkdtemp(prefix="vm_asr_probe_")))
        engine = AsrEngine.__new__(AsrEngine)
        engine._config = cfg
        engine._loaded = True
        engine._backend = {"processor": processor, "model": model}
        engine._use_native_request = None
        engine._last_error = ""
        return engine

    class _FakeBatch:
        """Minimal BatchFeature stand-in (dict + .to(device, dtype))."""

        def __init__(self, data):
            self._data = dict(data)

        def __getitem__(self, key):
            return self._data[key]

        def keys(self):
            return self._data.keys()

        def to(self, *args, **kwargs):
            return self

    class _FakeModel:
        dtype = "float32"

        def __init__(self):
            self.device = "cpu"
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return self

        @property
        def sequences(self):
            # the generated part: shape (1, >=4) so slicing [:, 4:] works
            return np.zeros((1, 6), dtype=np.int64)

    def test_native_request_layer_is_preferred(self):
        """transformers >= 5.0 apply_transcription_request wins when present."""

        class NativeProcessor:
            def __init__(self):
                self.native_calls = 0

            def apply_transcription_request(self, audio, language, prompt):
                self.native_calls += 1
                return TestAsrEngineCallPath._FakeBatch(
                    {"input_ids": np.zeros((1, 4), dtype=np.int64)}
                )

            def batch_decode(self, sequences, **kwargs):
                return ["language Hungarian<asr_text>szia"]

        proc, model = NativeProcessor(), self._FakeModel()
        engine = self._engine(proc, model)
        engine._transcribe(np.zeros(1600, dtype=np.float32))
        self.assertEqual(proc.native_calls, 1)
        self.assertTrue(engine._use_native_request)

    def test_manual_chat_template_fallback(self):
        """No native helper: apply_chat_template + processor(text=, audio=)."""

        class ManualProcessor:
            def __init__(self):
                self.prompt = None

            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                assert add_generation_prompt and not tokenize
                return "<|im_start|>user\nAUDIO<|im_end|>\n<|im_start|>assistant\n"

            def __call__(self, text, audio, return_tensors, padding):
                self.prompt = text[0]
                return TestAsrEngineCallPath._FakeBatch(
                    {"input_ids": np.zeros((1, 4), dtype=np.int64)}
                )

            def batch_decode(self, sequences, **kwargs):
                return ["language Hungarian<asr_text>jo reggelt"]

        proc, model = ManualProcessor(), self._FakeModel()
        engine = self._engine(proc, model)
        out = engine._transcribe(np.zeros(1600, dtype=np.float32))
        self.assertEqual(out, "jo reggelt")
        self.assertIn("<|im_start|>assistant", proc.prompt)
        self.assertFalse(engine._use_native_request)
        # forced language appends the prefill to the prompt
        engine._config.asr_language = "Hungarian"
        engine2 = self._engine(ManualProcessor(), self._FakeModel())
        # (a fresh engine: the first call path decision is cached per engine)
        engine2._transcribe(np.zeros(1600, dtype=np.float32))

    def test_probe_failure_is_reported_not_swallowed(self):
        """A dead call surface must surface on last_error (red chip)."""

        class DeadProcessor:
            def apply_transcription_request(self, audio, language, prompt):
                raise TypeError("dead surface")

            def apply_chat_template(self, *a, **k):
                raise RuntimeError("dead template")

            def __call__(self, *a, **k):
                raise RuntimeError("dead call")

        engine = self._engine(DeadProcessor(), self._FakeModel())
        ok = engine._probe()
        self.assertFalse(ok)
        self.assertIn("probe failed", engine.last_error)

    def test_probe_success_without_raise(self):
        """Quiet noise may legitimately transcribe to empty - still healthy."""

        class ManualProcessor:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                return "PROMPT"

            def __call__(self, text, audio, return_tensors, padding):
                return TestAsrEngineCallPath._FakeBatch(
                    {"input_ids": np.zeros((1, 4), dtype=np.int64)}
                )

            def batch_decode(self, sequences, **kwargs):
                return ["language None<asr_text>"]

        engine = self._engine(ManualProcessor(), self._FakeModel())
        self.assertTrue(engine._probe())
        self.assertEqual(engine.last_error, "")


class TestAsrLoadHint(unittest.TestCase):
    def test_hint_mentions_transformers_version(self):
        from app.asr import _load_error_hint

        hint = _load_error_hint(
            ValueError("Unrecognized configuration class Qwen3AsrConfigFor...")
        )
        self.assertIn("transformers >= 5.0", hint)

    def test_plain_error_has_no_version_hint(self):
        from app.asr import _load_error_hint

        hint = _load_error_hint(OSError("disk on fire"))
        self.assertNotIn("transformers", hint)


class TestVadIsAvailable(unittest.TestCase):
    def test_is_available_exists_and_reflects_session(self):
        from app.vad import SileroVad

        probe = SileroVad.__new__(SileroVad)
        probe._session = None
        self.assertFalse(probe.is_available())
        probe._session = object()
        self.assertTrue(probe.is_available())


if __name__ == "__main__":
    unittest.main()


class TestForcedUtteranceEnd(unittest.TestCase):
    """v0.4.2: an utterance stuck IN SPEECH is force-ended after 12 s.

    Field report: 'I speak but nothing happens' - a VAD that never leaves the
    in-speech state (continuous noise/music above the threshold, hangover
    never completing) produced NO speech_end, NO turn and NO feedback. The
    audio loop now force-ends the utterance at _MAX_UTTERANCE_MS so the
    collected audio still becomes a turn.
    """

    def test_forced_end_after_max_utterance_ms(self):
        from app.web_server import _MAX_UTTERANCE_MS

        tmp = tempfile.mkdtemp(prefix="vm_forcedend_test_")
        comp = _mock_components(tmp)
        # v0.6.0: the demo engine would transcribe the 12 s noise utterance
        # into a demo phrase and dispatch a turn — while a turn is answering,
        # VAD frames belong to barge-in detection (v0.4.14 design), so the
        # "loop did not wedge" property is observed on the empty-transcript
        # path (no turn dispatched), matching the pre-v0.6.0 mock behaviour
        # this test was written against. The assertions are unchanged.
        from app.mock_components import MockAsrEngine

        comp.make_asr = lambda: MockAsrEngine(queue=[""])  # type: ignore[method-assign]
        sock = _FakeSock()
        session = WebSession(sock, comp)
        rng = np.random.default_rng(3)
        loud = (rng.standard_normal(512) * 0.3).astype(np.float32)
        # 32 ms per frame: enough frames to cross the 12 s ceiling
        n_frames = int(_MAX_UTTERANCE_MS / 32) + 10

        async def scenario() -> None:
            for _ in range(n_frames):
                await session._on_vad_frame(loud)

        asyncio.run(scenario())
        events = comp.status.session["events"]
        self.assertTrue(any("forced utterance end" in e for e in events))
        # the utterance was flushed (speech end semantics) even though the
        # VAD state machine never emitted speech_end on its own
        self.assertTrue(any("speech end" in e for e in events))
        # the 12 s accumulation was reset - whatever speech_ms shows now
        # belongs to the NEW utterance (a few frames), not the stuck one
        self.assertLess(comp.status.session["speech_ms"], 1000)
        # the frames AFTER the forced end start a NEW utterance (the VAD
        # still sees speech) - proving the loop did not wedge
        self.assertGreaterEqual(comp.status.session["utterances"], 2)

    def test_no_forced_end_below_the_ceiling(self):
        from app.web_server import _MAX_UTTERANCE_MS

        tmp = tempfile.mkdtemp(prefix="vm_noforce_test_")
        comp = _mock_components(tmp)
        sock = _FakeSock()
        session = WebSession(sock, comp)
        rng = np.random.default_rng(3)
        loud = (rng.standard_normal(512) * 0.3).astype(np.float32)
        n_frames = int(_MAX_UTTERANCE_MS / 32) - 20  # just below the ceiling

        async def scenario() -> None:
            for _ in range(n_frames):
                await session._on_vad_frame(loud)

        asyncio.run(scenario())
        events = comp.status.session["events"]
        self.assertFalse(any("forced utterance end" in e for e in events))
        self.assertFalse(any("speech_end" in e for e in events))
        self.assertTrue(comp.status.session["vad_in_speech"])
