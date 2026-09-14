"""Unit tests for the Supertonic 3 TTS engine (app/tts_supertonic.py).

v0.7.0 production TTS contract:
  * ONE engine (supertonic3), NO Piper fallback — failures raise
    SupertonicTtsError / return None-False, logged;
  * the SDK is loaded ONCE lazily and kept alive (never per utterance);
  * synthesize -> int16 mono at config.output_sample_rate (44100);
  * synthesize_to_file -> 44.1 kHz 16-bit mono WAV;
  * M2 length_scale maps to SDK speed (1/length_scale, clamped 0.7–2.0);
  * voice resolution: explicit preset wins, unknown -> per-language default;
  * stop() sets an abandon flag that discards the NEXT result but is reset
    by the following call (barge-in contract);
  * edge-silence trim compensates the vocoder's 0.35–0.9 s padding.

The real SDK/model is NEVER touched here: the engine's SDK import is
monkeypatched with a fake (module-injection via sys.modules), matching the
mock-first convention of the rest of the suite (the deep real-model smoke
lives in tests/validation/test_feature_tts.py and the standalone
scripts/validate_supertonic.py gate).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.config import AgentConfig  # noqa: E402
from app.tts_supertonic import (  # noqa: E402
    SupertonicTtsEngine,
    SupertonicTtsError,
    _float32_to_int16,
    _trim_edge_silence,
)
from app.text_utils import LANG_EN, LANG_HU  # noqa: E402

ENV_KEYS = (
    "VOICEMEM_HOME",
    "SUPERTONIC_MODEL_PATH",
    "SUPERTONIC_SPEED",
    "SUPERTONIC_STEPS",
    "SUPERTONIC_TRIM_SILENCE",
    "TTS_ENGINE",
    "TTS_HU_VOICE",
    "TTS_EN_VOICE",
    "PIPER_VOICES_PATH",
    "PIPER_EXECUTABLE",
)


class _FakeSdkTTS:
    """Stand-in for supertonic.TTS: records calls, returns fixed audio."""

    instances: list["_FakeSdkTTS"] = []
    load_calls = 0

    def __init__(self, model: str = "supertonic-3", model_dir: Any = None,
                 auto_download: bool = True, **_: Any) -> None:
        assert model == "supertonic-3"
        assert auto_download is False, "runtime must never auto-download"
        assert model_dir is not None
        self.model_dir = str(model_dir)
        self.sample_rate = 44100
        self.voice_style_names = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
        self.calls: list[dict[str, Any]] = []
        _FakeSdkTTS.instances.append(self)
        _FakeSdkTTS.load_calls += 1

    def get_voice_style(self, voice_name: str) -> dict[str, str]:
        return {"voice": voice_name}

    def synthesize(self, text: str, voice_style: Any, total_steps: int = 8,
                   speed: float = 1.05, max_chunk_length: Any = None,
                   silence_duration: float = 0.3, lang: Any = None,
                   verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(
            {
                "text": text,
                "style": voice_style,
                "steps": total_steps,
                "speed": speed,
                "lang": lang,
            }
        )
        n = 22050  # 0.5 s of audio at 44.1 kHz
        wav = np.zeros((1, n), dtype=np.float32)
        # speech in the middle so the trim has something to keep
        wav[0, 5000:15000] = 0.3 * np.sin(np.linspace(0, 40.0, 10000))
        return wav, np.array([0.5])

    def save_audio(self, wav: np.ndarray, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        x = np.asarray(wav).reshape(-1)
        with wave.open(output_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(_float32_to_int16(x).tobytes())


class _EnvNeutral(unittest.TestCase):
    def setUp(self) -> None:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        self.addCleanup(self._restore)
        for key in ENV_KEYS:
            if key in os.environ:
                del os.environ[key]

    def _restore(self) -> None:
        pass


def _fake_layout(root: Path) -> Path:
    """Create a fake Supertonic model dir (onnx + voice styles)."""
    model_dir = root / "models" / "tts" / "supertonic-3"
    (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
    for name, size in (
        ("tts.json", 8000),
        ("unicode_indexer.json", 277000),
        ("duration_predictor.onnx", 3_700_000),
        ("text_encoder.onnx", 36_400_000),
        ("vector_estimator.onnx", 256_500_000),
        ("vocoder.onnx", 101_400_000),
    ):
        (model_dir / "onnx" / name).write_bytes(b"x" * size)
    styles = model_dir / "voice_styles"
    styles.mkdir(parents=True, exist_ok=True)
    for v in ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]:
        (styles / f"{v}.json").write_text('{"voice": "%s"}' % v)
    return model_dir


class SupertonicEngineTests(_EnvNeutral):
    def _engine_with_fake_sdk(self, root: Path | None = None) -> tuple[SupertonicTtsEngine, Path, _FakeSdkTTS]:
        import shutil

        tmp = root or Path(tempfile.mkdtemp(prefix="vm_tts_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        model_dir = _fake_layout(tmp)
        cfg = AgentConfig(root=tmp)
        engine = SupertonicTtsEngine(cfg)
        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
            # load through the public path
            pcm = engine.synthesize("Szia!", LANG_HU)
            self.assertIsNotNone(pcm)
        fake = _FakeSdkTTS.instances[-1]
        return engine, tmp, fake

    def test_config_defaults(self) -> None:
        cfg = AgentConfig()
        self.assertEqual(cfg.tts_engine, "supertonic3")
        self.assertEqual(cfg.tts_hu_voice, "F1")
        self.assertEqual(cfg.tts_en_voice, "F1")
        self.assertEqual(cfg.output_sample_rate, 44100)
        self.assertEqual(cfg.supertonic_speed, 1.05)
        self.assertEqual(cfg.supertonic_steps, 8)
        self.assertTrue(cfg.supertonic_trim_silence)
        self.assertEqual(cfg.supertonic_model_dir.name, "supertonic-3")

    def test_config_validation_rejects_piper(self) -> None:
        cfg = AgentConfig()
        cfg.tts_engine = "piper"
        errors = cfg.validate()
        self.assertTrue(any("supertonic3" in e for e in errors))

    def test_config_validation_rejects_wrong_rate(self) -> None:
        cfg = AgentConfig()
        cfg.output_sample_rate = 22050
        errors = cfg.validate()
        self.assertTrue(any("44100" in e for e in errors))

    def test_is_available_requires_all_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_layout(root)
            cfg = AgentConfig(root=root)
            engine = SupertonicTtsEngine(cfg)
            self.assertTrue(engine.is_available())
            # break one onnx module
            (cfg.supertonic_model_dir / "onnx" / "vocoder.onnx").unlink()
            self.assertFalse(engine.is_available())
            # restore, break a default voice style
            (cfg.supertonic_model_dir / "onnx" / "vocoder.onnx").write_bytes(b"x" * 1000)
            (cfg.supertonic_voices_dir / "F1.json").unlink()
            self.assertFalse(engine.is_available())

    def test_list_voices_and_voice_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_layout(root)
            cfg = AgentConfig(root=root)
            engine = SupertonicTtsEngine(cfg)
            self.assertEqual(
                engine.list_voices(),
                ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"],
            )
            self.assertTrue(engine.voice_exists("M3"))
            self.assertFalse(engine.voice_exists("hu_HU-anna-medium"))
            self.assertFalse(engine.voice_exists(""))

    def test_synthesize_returns_int16_mono_44100(self) -> None:
        engine, _root, fake = self._engine_with_fake_sdk()
        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
            pcm = engine.synthesize("Szia, minden rendben?", LANG_HU)
        self.assertIsNotNone(pcm)
        self.assertEqual(pcm.dtype, np.int16)
        self.assertTrue(len(pcm) > 0)
        self.assertEqual(fake.calls[-1]["lang"], "hu")

    def test_language_mapping(self) -> None:
        engine, _root, fake = self._engine_with_fake_sdk()
        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
            engine.synthesize("Hello there.", LANG_EN)
            self.assertEqual(fake.calls[-1]["lang"], "en")
            engine.synthesize("Helló, ismeretlen.", "xx")
            self.assertEqual(fake.calls[-1]["lang"], "hu")  # unknown -> HU parity

    def test_model_loaded_once_and_reused(self) -> None:
        engine, _root, _fake = self._engine_with_fake_sdk()
        before = _FakeSdkTTS.load_calls
        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
            for _ in range(4):
                pcm = engine.synthesize("Ismételt szintézis.", LANG_HU)
                self.assertIsNotNone(pcm)
        self.assertEqual(_FakeSdkTTS.load_calls, before, "engine must load ONCE")

    def test_explicit_voice_overrides_default(self) -> None:
        engine, _root, fake = self._engine_with_fake_sdk()
        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
            engine.synthesize("Hangváltás.", LANG_HU, voice="M5")
        self.assertEqual(fake.calls[-1]["style"], {"voice": "M5"})
        self.assertEqual(engine.last_voice, "M5")

    def test_unknown_explicit_voice_falls_back_to_default(self) -> None:
        engine, _root, fake = self._engine_with_fake_sdk()
        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
            engine.synthesize("Hangváltás.", LANG_EN, voice="hu_HU-anna-medium")
        self.assertEqual(fake.calls[-1]["style"], {"voice": "F1"})
        self.assertEqual(engine.last_voice, "F1")

    def test_length_scale_maps_to_speed(self) -> None:
        engine, _root, fake = self._engine_with_fake_sdk()
        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
            engine.synthesize("Lassabban.", LANG_HU, length_scale=1.25)
            self.assertAlmostEqual(fake.calls[-1]["speed"], 0.8, places=3)
            engine.synthesize("Gyorsabban.", LANG_HU, length_scale=0.8)
            self.assertAlmostEqual(fake.calls[-1]["speed"], 1.25, places=3)
            # out-of-band clamps to the SDK band
            engine.synthesize("Túl lassú.", LANG_HU, length_scale=10.0)
            self.assertAlmostEqual(fake.calls[-1]["speed"], 0.7, places=6)
            engine.synthesize("Túl gyors.", LANG_HU, length_scale=0.01)
            self.assertAlmostEqual(fake.calls[-1]["speed"], 2.0, places=6)
            # None -> configured default
            engine.synthesize("Alap.", LANG_HU, length_scale=None)
            self.assertAlmostEqual(fake.calls[-1]["speed"], 1.05, places=6)

    def test_empty_text_returns_none(self) -> None:
        engine, _root, _fake = self._engine_with_fake_sdk()
        self.assertIsNone(engine.synthesize("", LANG_HU))
        self.assertIsNone(engine.synthesize("   \n  ", LANG_EN))

    def test_missing_assets_raise_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AgentConfig(root=Path(tmp))  # no model files at all
            engine = SupertonicTtsEngine(cfg)
            self.assertFalse(engine.is_available())
            self.assertIsNone(engine.synthesize("Szia", LANG_HU))
            # the SAME explicit error repeats (no retry storm, no fallback)
            self.assertIsNone(engine.synthesize("Szia", LANG_HU))
            self.assertIn("Supertonic 3", engine._load_error)

    def test_synthesis_exception_returns_none(self) -> None:
        engine, _root, _fake = self._engine_with_fake_sdk()

        class _Broken(_FakeSdkTTS):
            def synthesize(self, *a: Any, **kw: Any) -> Any:
                raise RuntimeError("ONNX exploded")

        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module(_Broken)}):
            engine._tts = None
            engine._load_error = None
            pcm = engine.synthesize("Hiba teszt.", LANG_HU)
        self.assertIsNone(pcm)

    def test_stop_flag_discards_next_result_then_resets(self) -> None:
        engine, _root, _fake = self._engine_with_fake_sdk()

        class _Stopping(_FakeSdkTTS):
            """Simulates barge-in arriving WHILE the synthesis runs."""

            def synthesize(self, *a: Any, **kw: Any) -> Any:
                engine.stop()  # barge-in lands mid-synthesis
                return super().synthesize(*a, **kw)

        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module(_Stopping)}):
            engine._tts = None
            engine._load_error = None
            pcm = engine.synthesize("Ezt eldobjuk.", LANG_HU)
            self.assertIsNone(pcm, "in-flight stop must discard the result")
        # the flag resets on the NEXT call (new request supersedes stop)
        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
            engine._tts = None
            engine._load_error = None
            pcm2 = engine.synthesize("Ezt már nem dobjuk el.", LANG_HU)
            self.assertIsNotNone(pcm2)

    def test_stop_before_new_request_is_superseded(self) -> None:
        # Piper contract parity: stop() followed by a NEW synthesize call
        # means the new request supersedes the earlier stop.
        engine, _root, _fake = self._engine_with_fake_sdk()
        with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
            engine.stop()
            pcm = engine.synthesize("Új kérés a stop után.", LANG_HU)
            self.assertIsNotNone(pcm)

    def test_synthesize_to_file_writes_valid_wav(self) -> None:
        engine, _root, _fake = self._engine_with_fake_sdk()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.wav"
            with patch.dict(sys.modules, {"supertonic": _fake_sdk_module()}):
                ok = engine.synthesize_to_file("Fájl teszt.", LANG_HU, out)
            self.assertTrue(ok)
            with wave.open(str(out), "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 44100)
                self.assertGreater(w.getnframes(), 100)

    def test_synthesize_to_file_failure_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AgentConfig(root=Path(tmp))
            engine = SupertonicTtsEngine(cfg)
            out = Path(tmp) / "no_model.wav"
            self.assertFalse(engine.synthesize_to_file("x", LANG_HU, out))
            self.assertFalse(out.exists())

    def test_status_detail(self) -> None:
        engine, _root, _fake = self._engine_with_fake_sdk()
        self.assertIn("Supertonic 3", engine.status_detail())
        self.assertIn("ONNX CPU", engine.status_detail())


class TrimSilenceTests(unittest.TestCase):
    def test_trims_edges_keeps_speech(self) -> None:
        rate = 44100
        # 0.5 s silence + 1.0 s speech + 0.5 s silence
        x = np.zeros(int(rate * 2.0), dtype=np.float32)
        x[int(rate * 0.5) : int(rate * 1.5)] = 0.2
        out = _trim_edge_silence(x, rate)
        self.assertLess(len(out), len(x))
        self.assertGreater(len(out), int(rate * 0.9))
        # speech survives
        self.assertGreater(float(np.max(np.abs(out))), 0.1)

    def test_noop_on_already_trimmed(self) -> None:
        rate = 44100
        x = 0.2 * np.ones(int(rate * 0.5), dtype=np.float32)
        out = _trim_edge_silence(x, rate)
        self.assertEqual(len(out), len(x))

    def test_never_raises_on_garbage(self) -> None:
        self.assertEqual(len(_trim_edge_silence(np.zeros(10, dtype=np.float32))), 10)

    def test_float32_to_int16_clips(self) -> None:
        x = np.array([0.0, 0.5, 1.5, -1.5], dtype=np.float32)
        y = _float32_to_int16(x)
        self.assertEqual(y.dtype, np.int16)
        self.assertEqual(int(y[2]), 32767)
        self.assertEqual(int(y[3]), -32767)


def _fake_sdk_module(cls: type = _FakeSdkTTS) -> Any:
    """Build a fake 'supertonic' module object for sys.modules injection."""
    import types

    mod = types.ModuleType("supertonic")
    mod.TTS = cls  # type: ignore[attr-defined]
    return mod


if __name__ == "__main__":
    unittest.main()
