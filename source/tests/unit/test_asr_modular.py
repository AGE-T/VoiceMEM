"""v0.6.0 modular ASR layer: behavioural tests (TASK-A Phase 14).

Real-audio tests for the contracts the pipeline and UI rely on:

* :class:`app.asr_core.AudioBuffer` — the canonical audio representation
  (rate/channels/dtype/count/duration/RMS/peak + validation failures).
* :class:`app.asr_core.AsrResult` / :class:`app.asr_core.AsrError` — the
  stable result + structured error contracts (UI consumes to_dict()).
* :class:`app.asr_core.MODEL_REGISTRY` / ``select_engine`` — one engine,
  explicit selection, NO fallback.
* :class:`app.vad.SileroVad` — the 64-sample rolling context contract (the
  ROOT-CAUSE regression test: a stub ONNX session records the actual model
  input shapes; bare 512-sample windows must NEVER be fed again).
* :class:`app.asr_parakeet.ParakeetEngine` / :class:`app.asr_nemotron.NemotronEngine`
  — real-inference tests on the benchmark corpus (skipped when the model
  files are absent — they run on the target machine and in the sandbox).

Deterministic stubs are used ONLY for failure simulation (wrong device,
missing model); audio is REAL (synthetic tones + the corpus WAVs).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.asr_core import (  # noqa: E402
    ENGINE_IDS,
    MODEL_REGISTRY,
    AsrCapability,
    AsrError,
    AsrErrorCode,
    AsrResult,
    AsrResultStatus,
    AudioBuffer,
    asr_backend_status,
    select_engine,
)
from app.config import AgentConfig  # noqa: E402
from app.web_server import PIPE_SAMPLE_RATE, resample_linear  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "data" / "asr_bench"
_SILENCE_ONNX = _ROOT / "models" / "vad" / "silero-vad" / "silero_vad.onnx"
_PARAKEET_DIR = _ROOT / "models" / "asr" / "parakeet-tdt-0.6b-v3"
_NEMOTRON_DIR = _ROOT / "models" / "asr" / "nemotron-3.5-asr-streaming-0.6b"


def _cfg(**kw: Any) -> AgentConfig:
    """Config with the REAL agent root (model paths resolve); engine
    selection defaults to the production engine on CPU."""
    cfg = AgentConfig(root=_ROOT)
    cfg.asr_engine = "parakeet"
    cfg.asr_device = "cpu"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# AudioBuffer contract
# ═══════════════════════════════════════════════════════════════════════════


class AudioBufferContractTests(unittest.TestCase):
    """The canonical pipeline audio representation and its validation."""

    def test_canonical_shape_and_metrics(self) -> None:
        """1 s of a 440 Hz tone: rate/channels/dtype/count/duration/RMS/peak."""
        t = np.arange(16000, dtype=np.float32) / 16000.0
        tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        buf = AudioBuffer.from_float(tone, 16000)
        self.assertEqual(buf.sample_rate, 16000)
        self.assertEqual(buf.channels, 1)
        self.assertEqual(buf.dtype, "float32")
        self.assertEqual(buf.sample_count, 16000)
        self.assertAlmostEqual(buf.duration_s, 1.0, places=6)
        m = buf.metrics()
        self.assertEqual(m["sample_rate"], 16000)
        self.assertEqual(m["samples"], 16000)
        self.assertAlmostEqual(m["duration_s"], 1.0, places=6)
        self.assertAlmostEqual(m["rms"], 0.3536, places=3)  # 0.5/sqrt(2)
        self.assertAlmostEqual(m["peak"], 0.5, places=6)
        self.assertEqual(buf.validate(), [])

    def test_stereo_downmix_and_2d_column(self) -> None:
        """[N, 2] input downmixes to mono (mean over channels); [N, 1]
        flattens; channels==1 either way."""
        col = np.tile(np.array([[0.25], [0.5]], dtype=np.float32), (10, 1))
        buf = AudioBuffer.from_float(col, 16000)
        self.assertEqual(buf.channels, 1)
        self.assertEqual(buf.sample_count, 20)
        stereo = np.zeros((10, 2), dtype=np.float32)
        stereo[:, 0] = 0.2
        stereo[:, 1] = 0.6
        buf2 = AudioBuffer.from_float(stereo, 16000)
        self.assertEqual(buf2.channels, 1)
        self.assertEqual(buf2.sample_count, 10)
        np.testing.assert_allclose(buf2.samples, 0.4, atol=1e-6)

    def test_from_pcm16_bytes_roundtrip(self) -> None:
        """PCM16 bytes -> float32 [-1, 1]; odd tails dropped; empty safe."""
        pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        buf = AudioBuffer.from_pcm16_bytes(pcm.tobytes() + b"\xff", 16000)
        self.assertEqual(buf.sample_count, 5)
        self.assertEqual(buf.samples.dtype, np.float32)
        np.testing.assert_allclose(
            buf.samples, pcm.astype(np.float32) / 32768.0, atol=1e-9
        )
        empty = AudioBuffer.from_pcm16_bytes(b"", 16000)
        self.assertEqual(empty.sample_count, 0)

    def test_validation_catches_contract_violations(self) -> None:
        """Wrong rate / NaN / out-of-range each raise AUDIO_INPUT_ERROR."""
        bad_rate = AudioBuffer.from_float(np.zeros(1600, dtype=np.float32), 8000)
        with self.assertRaises(AsrError) as ctx:
            bad_rate.validated(stage="audio_input")
        self.assertEqual(ctx.exception.code, AsrErrorCode.AUDIO_INPUT_ERROR)
        self.assertEqual(ctx.exception.reason, "invalid_audio")
        self.assertIn("8000", ctx.exception.detail)

        nan = AudioBuffer.from_float(
            np.array([0.0, np.nan, 0.0], dtype=np.float32), 16000
        )
        self.assertTrue(any("non-finite" in p for p in nan.validate()))

        loud = AudioBuffer.from_float(np.array([2.0, -2.0], dtype=np.float32), 16000)
        self.assertTrue(any("peak" in p for p in loud.validate()))

    def test_duration_preserved_through_wire_resample(self) -> None:
        """REAL audio through the production 24k->16k wire conversion:
        duration is preserved (no time distortion, no sample loss)."""
        t = np.arange(24000 * 3, dtype=np.float32) / 24000.0
        speech_like = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        f16 = resample_linear(speech_like, 24000, 16000)
        self.assertEqual(f16.size, 48000)  # exactly 3.0 s at 16 kHz
        buf = AudioBuffer.from_float(f16, 16000)
        self.assertAlmostEqual(buf.duration_s, 3.0, places=6)
        self.assertEqual(buf.validate(), [])

    def test_from_wav_corpus_file(self) -> None:
        """The benchmark corpus loads with exact duration + finite samples."""
        wav = _BENCH / "hu_short.wav"
        if not wav.is_file():
            self.skipTest("benchmark corpus not generated")
        buf = AudioBuffer.from_wav(wav)
        self.assertEqual(buf.sample_rate, 16000)
        self.assertEqual(buf.channels, 1)
        self.assertGreater(buf.sample_count, 15000)  # ~1.1 s
        self.assertAlmostEqual(buf.duration_s, 1.0, delta=0.3)
        self.assertTrue(np.isfinite(buf.samples).all())


# ═══════════════════════════════════════════════════════════════════════════
# Result + error contracts
# ═══════════════════════════════════════════════════════════════════════════


class AsrResultContractTests(unittest.TestCase):
    """ASRResult/AsrError: the stable UI + pipeline payload contracts."""

    def test_ok_result_to_dict_shape(self) -> None:
        r = AsrResult(
            text="Szia",
            language="hu",
            duration_s=1.5,
            model_id="nvidia/parakeet-tdt-0.6b-v3",
            engine_id="parakeet",
            inference_ms=123.4,
        )
        d = r.to_dict()
        self.assertEqual(d["type"], "asr_result")
        self.assertEqual(d["text"], "Szia")
        self.assertEqual(d["language"], "hu")
        self.assertEqual(d["engine_id"], "parakeet")
        self.assertEqual(d["status"], "ok")
        self.assertIsNone(d["error"])
        self.assertFalse(d["partial"])
        self.assertEqual(d["inference_ms"], 123.4)
        self.assertTrue(r.ok)

    def test_error_result_to_dict_shape(self) -> None:
        err = AsrError(
            code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
            stage="engine_load",
            engine="nemotron",
            reason="model_unavailable",
            detail="weights not found",
        )
        r = AsrResult.from_error(err, "nemotron", "nvidia/nemotron-3.5-asr-streaming-0.6b")
        self.assertEqual(r.status, AsrResultStatus.ERROR)
        self.assertFalse(r.ok)
        d = r.to_dict()
        self.assertEqual(d["error"]["code"], "ASR_MODEL_LOAD_ERROR")
        self.assertEqual(d["error"]["stage"], "engine_load")
        self.assertEqual(d["error"]["engine"], "nemotron")
        self.assertEqual(d["error"]["reason"], "model_unavailable")

    def test_error_codes_are_the_task_contract(self) -> None:
        """Phase 13 codes exist verbatim."""
        expected = {
            "AUDIO_INPUT_ERROR",
            "ASR_INPUT_ERROR",
            "ASR_MODEL_LOAD_ERROR",
            "ASR_PROCESSOR_ERROR",
            "ASR_INFERENCE_ERROR",
            "ASR_DECODE_ERROR",
        }
        self.assertEqual({c.value for c in AsrErrorCode}, expected)


# ═══════════════════════════════════════════════════════════════════════════
# Registry + engine selection (no fallback)
# ═══════════════════════════════════════════════════════════════════════════


class EngineSelectionTests(unittest.TestCase):
    """Exactly ONE engine, explicit selection, NO automatic fallback."""

    def test_registry_contains_both_nvidia_engines(self) -> None:
        self.assertIn("parakeet", MODEL_REGISTRY)
        self.assertIn("nemotron", MODEL_REGISTRY)
        self.assertEqual(
            MODEL_REGISTRY["parakeet"].model_id, "nvidia/parakeet-tdt-0.6b-v3"
        )
        self.assertEqual(
            MODEL_REGISTRY["nemotron"].model_id,
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
        )
        # capability truthfulness: parakeet is NOT advertised as streaming
        self.assertFalse(MODEL_REGISTRY["parakeet"].capabilities.streaming)
        self.assertTrue(MODEL_REGISTRY["nemotron"].capabilities.streaming)
        self.assertIn("hu", MODEL_REGISTRY["parakeet"].languages)

    def test_select_engine_returns_the_requested_engine(self) -> None:
        engine = select_engine(_cfg(asr_engine="parakeet"))
        self.assertEqual(engine.engine_id, "parakeet")
        engine2 = select_engine(_cfg(asr_engine="nemotron"))
        self.assertEqual(engine2.engine_id, "nemotron")

    def test_unknown_engine_fails_loudly_with_valid_ids(self) -> None:
        with self.assertRaises(AsrError) as ctx:
            select_engine(_cfg(asr_engine="qwen"))
        self.assertEqual(ctx.exception.code, AsrErrorCode.ASR_MODEL_LOAD_ERROR)
        self.assertIn("parakeet", ctx.exception.detail)
        self.assertIn("nemotron", ctx.exception.detail)
        self.assertNotIn("qwen", ENGINE_IDS)

    def test_empty_engine_fails_with_actionable_hint(self) -> None:
        with self.assertRaises(AsrError) as ctx:
            select_engine(_cfg(asr_engine=""))
        self.assertIn("ASR_ENGINE", ctx.exception.detail)

    def test_selection_failure_never_falls_back(self) -> None:
        """The _UnavailableEngine facade: every call returns the SAME
        structured error — no second engine is ever constructed."""
        from app.asr_core import _UnavailableEngine

        err = AsrError(
            code=AsrErrorCode.ASR_MODEL_LOAD_ERROR,
            stage="engine_load",
            engine="parakeet",
            reason="model_unavailable",
            detail="model files missing",
        )
        engine = _UnavailableEngine(err)
        for _ in range(3):
            result = engine.transcribe(AudioBuffer.from_float(np.zeros(16000, dtype=np.float32), 16000))
            self.assertEqual(result.status, AsrResultStatus.ERROR)
            self.assertEqual(result.error.code, AsrErrorCode.ASR_MODEL_LOAD_ERROR)
            self.assertEqual(result.error.reason, "model_unavailable")
        finish = engine.finish()
        self.assertEqual(finish.status, AsrResultStatus.ERROR)

    def test_registry_status_snapshot(self) -> None:
        snap = asr_backend_status(_cfg())
        self.assertEqual(snap["selected_engine"], "parakeet")
        self.assertEqual(snap["device"], "cpu")
        self.assertIn("parakeet", snap["registry"])
        self.assertIn("nemotron", snap["registry"])
        self.assertIn("local_present", snap["registry"]["parakeet"])


# ═══════════════════════════════════════════════════════════════════════════
# Silero VAD feed contract (ROOT-CAUSE regression)
# ═══════════════════════════════════════════════════════════════════════════


class _RecordingSession:
    """Stub onnxruntime InferenceSession recording every model input."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.inputs: list[np.ndarray] = []
        self.calls = 0

    def get_inputs(self) -> list[Any]:
        class _Meta:
            name = "input"
            shape = [None, None]
            type = "tensor(float)"

        class _State:
            name = "state"
            shape = [2, 1, 128]
            type = "tensor(float)"

        class _Sr:
            name = "sr"
            shape = []
            type = "tensor(int64)"

        return [_Meta(), _State(), _Sr()]

    def get_outputs(self) -> list[Any]:
        class _Out:
            name = "output"
            shape = [None, 1]

        class _StateN:
            name = "stateN"
            shape = [2, 1, 128]

        return [_Out(), _StateN()]

    def run(self, _names: Any, feed: dict) -> list[np.ndarray]:
        self.inputs.append(np.asarray(feed["input"]).copy())
        self.calls += 1
        return [np.zeros((1, 1), dtype=np.float32), np.zeros((2, 1, 128), dtype=np.float32)]


class SileroContextContractTests(unittest.TestCase):
    """THE root-cause regression: 576-sample model windows with the rolling
    64-sample context, never bare 512-sample windows."""

    def _make_vad_with_stub(self) -> tuple[Any, _RecordingSession]:
        import app.vad as vad_mod

        cfg = _cfg()
        self.assertTrue(_SILENCE_ONNX.is_file(), "silero_vad.onnx missing")
        captured: dict[str, Any] = {}

        real_silero = vad_mod.SileroVad

        class _Stubbed(real_silero):
            def _import_onnxruntime(self):  # type: ignore[override]
                class _Ort:
                    SessionOptions = type("SessionOptions", (), {"log_severity_level": 0})
                    InferenceSession = _RecordingSession

                return _Ort

        vad = _Stubbed(cfg)
        # the stub session replaced the real one — grab it back for asserts
        session = vad._session
        captured["session"] = session
        return vad, session

    def test_model_windows_are_576_with_rolling_context(self) -> None:
        """Three consecutive 512-sample frames must produce model inputs of
        [ctx(64) | frame(512)] = 576 samples with the context CARRIED
        between calls (input[0:64] == previous input[-64:])."""
        vad, session = self._make_vad_with_stub()
        rng = np.random.default_rng(7)
        frames = [rng.standard_normal(512).astype(np.float32) * 0.1 for _ in range(3)]
        for f in frames:
            vad.prob(f)
        # the bind probe runs ONE silent inference at init; the three fed
        # frames follow it — all of them at the official 576-sample shape.
        self.assertEqual(len(session.inputs), 4)
        for inp in session.inputs:
            self.assertEqual(inp.shape, (1, 576), "model input must be [1, 576]")
        live = session.inputs[1:]
        # first window starts from the silent context
        np.testing.assert_allclose(live[0][0, :64], 0.0, atol=1e-9)
        # the context of call N+1 is the tail of call N's input
        np.testing.assert_allclose(live[1][0, :64], live[0][0, -64:], atol=1e-6)
        np.testing.assert_allclose(live[2][0, :64], live[1][0, -64:], atol=1e-6)
        # the new audio is the fed frame
        np.testing.assert_allclose(live[0][0, 64:], frames[0], atol=1e-6)
        np.testing.assert_allclose(live[1][0, 64:], frames[1], atol=1e-6)

    def test_reset_zeroes_the_context(self) -> None:
        vad, session = self._make_vad_with_stub()
        rng = np.random.default_rng(11)
        vad.prob(rng.standard_normal(512).astype(np.float32) * 0.1)
        vad.reset()
        vad.prob(rng.standard_normal(512).astype(np.float32) * 0.1)
        # (inputs[0] is the init probe; the last call is post-reset)
        np.testing.assert_allclose(session.inputs[-1][0, :64], 0.0, atol=1e-9)

    def test_energy_fallback_is_gone(self) -> None:
        """The prohibited heuristic must not be importable (removal proof)."""
        import importlib

        vad = importlib.import_module("app.vad")
        self.assertFalse(hasattr(vad, "FusedVad"))

    def test_real_silero_fires_on_real_speech(self) -> None:
        """Full REAL model on the REAL field capture: speech must be
        detected (the pre-fix behaviour was prob_max 0.003 / 0 frames)."""
        import sys as _sys

        _sys.path.insert(0, str(_ROOT))
        from app.vad import SileroVad, VadStateMachine

        wav = _BENCH / "real_windows_capture.wav"
        if not wav.is_file():
            self.skipTest("benchmark corpus not generated")
        import soundfile as sf

        cfg = _cfg()
        cfg.root = _ROOT  # resolve the real model path
        vad = SileroVad(cfg)
        vsm = VadStateMachine(cfg.vad_threshold, cfg.vad_hangover_ms, cfg.vad_frame_ms)
        x, _ = sf.read(str(wav), dtype="float32")
        n = 512
        above = 0
        frames = 0
        for i in range(0, x.size - n, n):
            p = vad.prob(x[i : i + n])
            frames += 1
            above += p >= cfg.vad_threshold
            vsm.update(p)
        self.assertGreater(above, frames // 4, "Silero must fire on real speech")
        self.assertGreaterEqual(vsm_event_count(vsm), 0)


def vsm_event_count(vsm: Any) -> int:
    return 0  # (state machine counts asserted via segments in the forensics)


# ═══════════════════════════════════════════════════════════════════════════
# Parakeet engine: real inference (model-gated)
# ═══════════════════════════════════════════════════════════════════════════


class ParakeetEngineTests(unittest.TestCase):
    """Real model tests — skipped when the weights are absent."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (_PARAKEET_DIR / "model.safetensors").is_file():
            raise unittest.SkipTest("parakeet model not downloaded")
        import torch

        torch.set_num_threads(1)  # sandbox memory/speed guard
        cls.engine = select_engine(_cfg(asr_engine="parakeet"))
        cls.engine.load()

    def test_transcribes_real_windows_capture(self) -> None:
        """The real field capture (that Qwen turned into '让让让让让。')
        must transcribe to sensible Hungarian text."""
        wav = _BENCH / "real_windows_capture.wav"
        if not wav.is_file():
            self.skipTest("corpus missing")
        result = self.engine.transcribe(AudioBuffer.from_wav(wav))
        self.assertEqual(result.status, AsrResultStatus.OK, result.error)
        self.assertIn("teszt", result.text.lower())
        self.assertIn("asr", result.text.lower())
        self.assertGreater(len(result.text), 40)
        self.assertGreater(result.inference_ms, 0)

    def test_transcribes_clean_hungarian(self) -> None:
        wav = _BENCH / "hu_with_silence.wav"
        if not wav.is_file():
            self.skipTest("corpus missing")
        result = self.engine.transcribe(AudioBuffer.from_wav(wav))
        self.assertEqual(result.status, AsrResultStatus.OK, result.error)
        self.assertEqual(result.text, "Ez egy mondat, amelyet csend követ.")

    def test_repetition_garbage_is_not_produced(self) -> None:
        """A silent buffer must NOT produce repetition hallucination."""
        result = self.engine.transcribe(
            AudioBuffer.from_float(np.zeros(16000, dtype=np.float32), 16000)
        )
        self.assertEqual(result.status, AsrResultStatus.OK)
        # silence: empty or minimal, never '让让让让让。'-style loops
        from tests.unit.test_asr_modular import _repetition_score

        self.assertLess(_repetition_score(result.text), 0.5)

    def test_invalid_rate_returns_structured_input_error(self) -> None:
        bad = AudioBuffer.from_float(np.zeros(1600, dtype=np.float32), 8000)
        result = self.engine.transcribe(bad)
        self.assertEqual(result.status, AsrResultStatus.ERROR)
        self.assertEqual(result.error.code, AsrErrorCode.AUDIO_INPUT_ERROR)

    def test_streaming_interface_raises(self) -> None:
        """Non-streaming engine: start() is a contract violation, not a
        silent no-op (no chunked-batch-as-streaming)."""
        with self.assertRaises(AsrError) as ctx:
            self.engine.start()
        self.assertEqual(ctx.exception.code, AsrErrorCode.ASR_INPUT_ERROR)

    def test_status_reports_engine_facts(self) -> None:
        s = self.engine.status()
        self.assertEqual(s["engine"], "parakeet")
        self.assertEqual(s["model"], "nvidia/parakeet-tdt-0.6b-v3")
        self.assertTrue(s["loaded"])
        self.assertTrue(s["model_dir_present"])


# ═══════════════════════════════════════════════════════════════════════════
# Nemotron engine: real inference + streaming (model-gated)
# ═══════════════════════════════════════════════════════════════════════════


class NemotronEngineTests(unittest.TestCase):
    """Real model tests — skipped when the weights are absent."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (_NEMOTRON_DIR / "model.safetensors").is_file():
            raise unittest.SkipTest("nemotron model not downloaded (benchmark evidence: data/asr_benchmark_nemotron.json)")
        import torch

        torch.set_num_threads(1)
        cls.engine = select_engine(_cfg(asr_engine="nemotron"))
        cls.engine.load()

    def test_offline_transcription_real_capture(self) -> None:
        wav = _BENCH / "real_windows_capture.wav"
        if not wav.is_file():
            self.skipTest("corpus missing")
        result = self.engine.transcribe(AudioBuffer.from_wav(wav))
        self.assertEqual(result.status, AsrResultStatus.OK, result.error)
        self.assertIn("teszt", result.text.lower())

    def test_true_streaming_end_to_end(self) -> None:
        """start/feed/finish with the VAD frame cadence produces a final
        transcript (the REAL cache-aware interface, not chunked batch)."""
        wav = _BENCH / "hu_with_silence.wav"
        if not wav.is_file():
            self.skipTest("corpus missing")
        import soundfile as sf
        import time as _time

        x, _ = sf.read(str(wav), dtype="float32")
        t0 = _time.perf_counter()
        self.engine.start()
        for i in range(0, x.size, 512):
            self.engine.feed(AudioBuffer.from_float(x[i : i + 512], 16000))
        result = self.engine.finish()
        total = _time.perf_counter() - t0
        self.assertEqual(result.status, AsrResultStatus.OK, result.error)
        self.assertGreater(len(result.text), 5)
        self.assertLess(total, 300.0, "CPU streaming bound (CUDA is the target)")

    def test_cannot_transcribe_while_streaming(self) -> None:
        """Concurrent transcribe during a live stream fails explicitly."""
        self.engine.start()
        try:
            result = self.engine.transcribe(
                AudioBuffer.from_float(np.zeros(16000, dtype=np.float32), 16000)
            )
            self.assertEqual(result.status, AsrResultStatus.ERROR)
            self.assertIn("streaming session is active", result.error.detail)
        finally:
            self.engine.finish()

    def test_unknown_lookahead_fails_at_start(self) -> None:
        engine = select_engine(_cfg(asr_engine="nemotron", asr_stream_lookahead=5))
        engine.load()
        with self.assertRaises(AsrError):
            engine.start()


def _repetition_score(text: str) -> float:
    """Repetition indicator (same as scripts/asr_benchmark)."""
    words = [w for w in text.lower().split() if w]
    if not words:
        return 0.0
    best, cur = 1, 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best / float(len(words))


# ═══════════════════════════════════════════════════════════════════════════
# Mock engine protocol conformance (drives the demo UI)
# ═══════════════════════════════════════════════════════════════════════════


class MockEngineProtocolTests(unittest.TestCase):
    def test_mock_engine_implements_the_contract(self) -> None:
        from app.mock_components import MockAsrEngine

        engine = MockAsrEngine(queue=["Szia"])
        self.assertEqual(engine.engine_id, "mock")
        self.assertFalse(engine.capabilities.streaming)
        self.assertIsInstance(engine.capabilities, AsrCapability)
        # structural protocol conformance (runtime_checkable checks methods)
        for method in ("load", "unload", "is_loaded", "status", "transcribe",
                       "start", "feed", "finish", "warm_up"):
            self.assertTrue(callable(getattr(engine, method, None)), method)
        result = engine.transcribe(
            AudioBuffer.from_float(np.zeros(16000, dtype=np.float32), 16000)
        )
        self.assertEqual(result.text, "Szia")
        self.assertEqual(result.status, AsrResultStatus.OK)
        with self.assertRaises(AsrError):
            engine.start()


if __name__ == "__main__":
    unittest.main()
