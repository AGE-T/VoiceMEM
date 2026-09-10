"""Unit tests for app/emotion.py (M2 emotion intelligence).

Everything here runs in the dependency-free sandbox: the pure mapping /
fusion / memory parts plus the analyzer wired to a FAKE model factory
(no funasr/torch is ever imported).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from app.emotion import (
    DEFAULT_PROSODY_WEIGHT,
    EMOTION2VEC_LABELS,
    EMOTION2VEC_LABELS_ZH_EN,
    TEACHER_EMOTIONS,
    EmotionAnalyzer,
    EmotionMemory,
    EmotionResult,
    fuse_emotion,
    map_raw_label,
    prosody_valence_arousal,
    raw_label_valence_arousal,
    semantic_valence_arousal,
    tail_window,
    teacher_label_from_va,
)


def _result(
    raw: str = "angry",
    valence: float = -0.8,
    arousal: float = 0.8,
    **kwargs: Any,
) -> EmotionResult:
    return EmotionResult(
        raw_label=raw,
        label=map_raw_label(raw),
        valence=valence,
        arousal=arousal,
        **kwargs,
    )


class LabelMappingTests(unittest.TestCase):
    """Raw emotion2vec+ categories -> teacher-facing categories."""

    def test_all_nine_raw_labels_map_to_teacher_labels(self) -> None:
        for raw in EMOTION2VEC_LABELS:
            self.assertIn(map_raw_label(raw), TEACHER_EMOTIONS)

    def test_negative_high_arousal_maps_to_frustrated(self) -> None:
        for raw in ("angry", "disgusted", "fearful"):
            self.assertEqual(map_raw_label(raw), "frustrated")

    def test_specific_mappings(self) -> None:
        self.assertEqual(map_raw_label("happy"), "happy")
        self.assertEqual(map_raw_label("surprised"), "happy")
        self.assertEqual(map_raw_label("sad"), "sad")
        for raw in ("neutral", "other", "unknown", "<unk>"):
            self.assertEqual(map_raw_label(raw), "neutral")

    def test_bilingual_tokens_txt_forms(self) -> None:
        for bilingual in EMOTION2VEC_LABELS_ZH_EN:
            self.assertIn(map_raw_label(bilingual), TEACHER_EMOTIONS)
        self.assertEqual(map_raw_label("生气/angry"), "frustrated")
        self.assertEqual(map_raw_label("开心/happy"), "happy")

    def test_unknown_values_never_raise(self) -> None:
        self.assertEqual(map_raw_label(""), "neutral")
        self.assertEqual(map_raw_label("whatever"), "neutral")
        self.assertEqual(map_raw_label(None), "neutral")

    def test_valence_arousal_anchors(self) -> None:
        self.assertEqual(raw_label_valence_arousal("angry"), (-0.8, 0.8))
        self.assertEqual(raw_label_valence_arousal("neutral"), (0.0, 0.0))
        self.assertEqual(raw_label_valence_arousal("难过/sad"), (-0.7, -0.4))
        self.assertEqual(raw_label_valence_arousal("nonsense"), (0.0, 0.0))


class DistributionTests(unittest.TestCase):
    """prosody_valence_arousal: softmax distribution -> weighted point."""

    def test_one_hot_matches_the_anchor(self) -> None:
        for i, raw in enumerate(EMOTION2VEC_LABELS):
            scores = [0.0] * len(EMOTION2VEC_LABELS)
            scores[i] = 1.0
            self.assertEqual(
                prosody_valence_arousal(list(EMOTION2VEC_LABELS), scores),
                raw_label_valence_arousal(raw),
            )

    def test_mixed_distribution_is_the_weighted_average(self) -> None:
        labels = ["angry", "neutral"]
        scores = [0.75, 0.25]
        v, a = prosody_valence_arousal(labels, scores)
        self.assertAlmostEqual(v, 0.75 * -0.8 + 0.25 * 0.0)
        self.assertAlmostEqual(a, 0.75 * 0.8 + 0.25 * 0.0)

    def test_degenerate_inputs_return_neutral(self) -> None:
        self.assertEqual(prosody_valence_arousal([], []), (0.0, 0.0))
        self.assertEqual(prosody_valence_arousal(["angry"], []), (0.0, 0.0))
        self.assertEqual(prosody_valence_arousal(["angry", "sad"], [1.0]), (0.0, 0.0))

    def test_unnormalised_scores_are_tolerated(self) -> None:
        # scores sum to 2.0 - the function normalises by the total
        v, a = prosody_valence_arousal(["happy", "sad"], [1.2, 0.8])
        self.assertAlmostEqual(v, 0.6 * 0.8 + 0.4 * -0.7)
        self.assertAlmostEqual(a, 0.6 * 0.5 + 0.4 * -0.4)


class SemanticHeuristicTests(unittest.TestCase):
    """Transcript markers -> semantic valence/arousal anchor."""

    def test_no_signal_is_neutral(self) -> None:
        self.assertEqual(semantic_valence_arousal(""), (0.0, 0.0))
        self.assertEqual(semantic_valence_arousal("hello there"), (0.0, 0.0))

    def test_hungarian_frustration_markers(self) -> None:
        for text in (
            "nem értem ezt a szabályt",
            "nem ertem ezt a szabalyt",
            "ez nekem túl bonyolult",
            "megint elrontottam",
        ):
            self.assertEqual(
                semantic_valence_arousal(text), (-0.5, 0.4), text
            )

    def test_english_frustration_markers(self) -> None:
        for text in (
            "I do not understand this at all",
            "this is so difficult",
            "why is this wrong again",
        ):
            self.assertEqual(semantic_valence_arousal(text), (-0.5, 0.4))

    def test_positive_markers(self) -> None:
        for text in (
            "köszönöm szépen",
            "köszi, értem",
            "thanks, I understand now",
            "great, got it",
        ):
            self.assertEqual(semantic_valence_arousal(text), (0.6, 0.3))

    def test_tie_is_neutral(self) -> None:
        # one negative + one positive marker -> tie -> neutral
        self.assertEqual(
            semantic_valence_arousal("nem értem, de köszönöm"), (0.0, 0.0)
        )


class TeacherLabelTests(unittest.TestCase):
    """teacher_label_from_va decision thresholds."""

    def test_thresholds(self) -> None:
        self.assertEqual(teacher_label_from_va(-0.68, 0.64), "frustrated")
        self.assertEqual(teacher_label_from_va(-0.6, 0.1), "sad")
        self.assertEqual(teacher_label_from_va(0.72, 0.42), "happy")
        self.assertEqual(teacher_label_from_va(0.0, 0.0), "neutral")
        self.assertEqual(teacher_label_from_va(-0.1, 0.5), "neutral")
        self.assertEqual(teacher_label_from_va(0.2, 0.0), "neutral")


class FusionTests(unittest.TestCase):
    """fuse_emotion: the spec 8.3.3 0.6/0.4 linear blend."""

    def test_none_prosody_short_circuits_to_none(self) -> None:
        self.assertIsNone(fuse_emotion(None, "nem értem"))

    def test_default_blend_math(self) -> None:
        prosody = _result("angry", valence=-0.8, arousal=0.8)
        fused = fuse_emotion(prosody, "I do not understand this", )
        self.assertIsNotNone(fused)
        # semantic = (-0.5, 0.4)
        self.assertAlmostEqual(fused.valence, 0.6 * -0.8 + 0.4 * -0.5)
        self.assertAlmostEqual(fused.arousal, 0.6 * 0.8 + 0.4 * 0.4)
        self.assertEqual(fused.label, "frustrated")
        self.assertTrue(fused.fused)
        self.assertEqual(fused.raw_label, "angry")

    def test_neutral_transcript_blends_toward_zero(self) -> None:
        # neutral semantic anchor (0, 0) shrinks the prosody values by 0.4
        prosody = _result("happy", valence=0.8, arousal=0.5)
        fused = fuse_emotion(prosody, "ez egy mondat")
        self.assertAlmostEqual(fused.valence, 0.6 * 0.8)
        self.assertAlmostEqual(fused.arousal, 0.6 * 0.5)
        self.assertEqual(fused.label, "happy")  # 0.48 >= 0.25

    def test_custom_weight(self) -> None:
        prosody = _result("angry", valence=-0.8, arousal=0.8)
        fused = fuse_emotion(prosody, "thanks", prosody_weight=0.0)
        # pure semantic side
        self.assertAlmostEqual(fused.valence, 0.6)
        self.assertAlmostEqual(fused.arousal, 0.3)

    def test_weight_is_clamped(self) -> None:
        prosody = _result("angry", valence=-0.8, arousal=0.8)
        fused = fuse_emotion(prosody, "x", prosody_weight=7.0)
        self.assertAlmostEqual(fused.valence, -0.8)  # clamped to 1.0
        fused = fuse_emotion(prosody, "x", prosody_weight=-3.0)
        self.assertAlmostEqual(fused.valence, 0.0)  # clamped to 0.0... on "x" neutral

    def test_latency_and_confidence_are_carried_over(self) -> None:
        prosody = _result("sad", valence=-0.7, arousal=-0.4, confidence=0.77, latency_ms=42.0)
        fused = fuse_emotion(prosody, "unalmas")
        self.assertAlmostEqual(fused.confidence, 0.77)
        self.assertAlmostEqual(fused.latency_ms, 42.0)


class TailWindowTests(unittest.TestCase):
    """tail_window: the last N seconds of the utterance."""

    def test_short_audio_is_returned_unchanged(self) -> None:
        audio = np.zeros(8000, dtype=np.float32)  # 0.5 s
        out = tail_window(audio, 16000, 5.0)
        self.assertEqual(out.size, 8000)

    def test_long_audio_is_cut_to_the_window(self) -> None:
        audio = np.arange(16 * 16000, dtype=np.float32)  # 16 s
        out = tail_window(audio, 16000, 5.0)
        self.assertEqual(out.size, 5 * 16000)
        np.testing.assert_array_equal(out, audio[-80000:])

    def test_duck_typed_object_without_size_passes_through(self) -> None:
        class NoSize:
            pass

        obj = NoSize()
        self.assertIs(tail_window(obj, 16000, 5.0), obj)


class EmotionResultTests(unittest.TestCase):
    """EmotionResult.to_dict shape (locked JSON contract)."""

    def test_to_dict_keys_and_rounding(self) -> None:
        result = _result("angry", valence=-0.68123, arousal=0.64222, confidence=0.9123, latency_ms=41.87)
        payload = result.to_dict()
        self.assertEqual(
            set(payload.keys()),
            {"raw_label", "label", "valence", "arousal", "confidence", "fused", "latency_ms"},
        )
        self.assertEqual(payload["raw_label"], "angry")
        self.assertEqual(payload["label"], "frustrated")
        self.assertEqual(payload["valence"], -0.681)
        self.assertEqual(payload["latency_ms"], 41.9)
        self.assertFalse(payload["fused"])


class EmotionMemoryTests(unittest.TestCase):
    """JSONL emotion log + should_store thresholds + fact text."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "emotion_log.jsonl"
        self.memory = EmotionMemory(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_record_turn_writes_valid_json_lines(self) -> None:
        fused = fuse_emotion(_result("angry"), "I do not understand this")
        self.assertTrue(self.memory.record_turn(fused, "nem értem"))
        self.assertTrue(self.memory.path.is_file())
        lines = self.memory.path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertIn("transcript", entry)
        self.assertEqual(entry["label"], "frustrated")
        self.assertIn("ts", entry)
        self.assertIn("stored_as_fact", entry)

    def test_record_never_raises_on_bad_path(self) -> None:
        broken = EmotionMemory(Path("/nonexistent-root/x/y/emotion.jsonl"))
        fused = fuse_emotion(_result("neutral"), "x")
        self.assertFalse(broken.record_turn(fused, "x"))  # best effort -> False

    def test_should_store_thresholds(self) -> None:
        frustrated_strong = _result("angry", valence=-0.68, arousal=0.64)
        frustrated_weak = _result("angry", valence=-0.3, arousal=0.3)
        happy_strong = _result("happy", valence=0.8, arousal=0.5)
        neutral = _result("neutral", valence=0.0, arousal=0.0)
        self.assertTrue(EmotionMemory.should_store(frustrated_strong))
        self.assertFalse(EmotionMemory.should_store(frustrated_weak))
        self.assertTrue(EmotionMemory.should_store(happy_strong))
        self.assertFalse(EmotionMemory.should_store(neutral))

    def test_fact_text_contains_label_and_quote(self) -> None:
        fused = fuse_emotion(_result("angry"), "why is I have went wrong")
        fact = EmotionMemory.fact_text(fused, "why is I have went wrong")
        self.assertIn("frustrated", fact)
        self.assertIn("why is I have went wrong", fact)
        self.assertIn("valence=", fact)


class _FakeAutoModel:
    """funasr AutoModel stand-in returning a canned generate() result."""

    calls: list[Any] = []

    def __init__(self, result: Optional[dict]) -> None:
        self._result = result
        self.generate_kwargs: list[dict] = []

    def generate(self, data: Any, **kwargs: Any) -> list[dict]:
        self.generate_kwargs.append(kwargs)
        self.__class__.calls.append(data)
        if self._result is None:
            raise RuntimeError("fake model failure")
        return [dict(self._result)]


class EmotionAnalyzerTests(unittest.TestCase):
    """Analyzer with a fake model factory (sandbox: no funasr/torch).

    ``_to_tensor`` is patched to the identity: the real method imports
    torch, which is deliberately absent from the sandbox (CONTRACT 2).
    """

    @staticmethod
    def _identity_tensor(window: Any) -> Any:
        return window

    def _patch_tensor(self):
        """Context: _to_tensor -> identity (avoids importing torch)."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            original = EmotionAnalyzer._to_tensor
            EmotionAnalyzer._to_tensor = staticmethod(EmotionAnalyzerTests._identity_tensor)
            try:
                yield
            finally:
                EmotionAnalyzer._to_tensor = original

        return _ctx()

    def _analyzer(
        self, model: Optional[_FakeAutoModel], with_files: bool = True
    ) -> EmotionAnalyzer:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model_dir = Path(tmp.name)
        if with_files:
            for name in EmotionAnalyzer.REQUIRED_FILES:
                (model_dir / name).write_bytes(b"x" * 8)

        def factory(_path: Path) -> _FakeAutoModel:
            if model is None:
                raise ImportError("no funasr in sandbox")
            return model

        analyzer = EmotionAnalyzer(
            model_dir, window_s=5.0, sample_rate=16000, model_factory=factory
        )
        return analyzer

    def test_files_present_and_missing(self) -> None:
        analyzer = self._analyzer(_FakeAutoModel({}), with_files=True)
        self.assertTrue(analyzer.files_present())
        analyzer2 = self._analyzer(_FakeAutoModel({}), with_files=False)
        self.assertFalse(analyzer2.files_present())

    def test_is_available_false_without_funasr_import(self) -> None:
        # files exist but `import funasr` fails in the sandbox -> False
        analyzer = self._analyzer(_FakeAutoModel({}), with_files=True)
        self.assertFalse(analyzer.is_available())

    def test_analyze_parses_labels_and_scores(self) -> None:
        result = {
            "key": "utt1",
            "labels": list(EMOTION2VEC_LABELS),
            "scores": [0.05, 0.02, 0.01, 0.9, 0.01, 0.0, 0.005, 0.004, 0.001],
        }
        model = _FakeAutoModel(result)
        analyzer = self._analyzer(model)
        with self._patch_tensor():
            out = analyzer.analyze(np.zeros(16000, dtype=np.float32))
        self.assertIsNotNone(out)
        self.assertEqual(out.raw_label, "happy")
        self.assertEqual(out.label, "happy")
        self.assertAlmostEqual(out.confidence, 0.9 / 1.0, places=6)
        self.assertAlmostEqual(out.latency_ms, 0.0, delta=50.0)
        # distribution-weighted valence: dominated by happy anchor
        self.assertGreater(out.valence, 0.5)
        self.assertFalse(out.fused)

    def test_analyze_accepts_bilingual_labels(self) -> None:
        result = {
            "labels": list(EMOTION2VEC_LABELS_ZH_EN),
            "scores": [0.9, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005],
        }
        model = _FakeAutoModel(result)
        analyzer = self._analyzer(model)
        with self._patch_tensor():
            out = analyzer.analyze(np.zeros(16000, dtype=np.float32))
        self.assertIsNotNone(out)
        self.assertEqual(out.raw_label, "生气/angry")
        self.assertEqual(out.label, "frustrated")

    def test_analyze_empty_audio_returns_none(self) -> None:
        analyzer = self._analyzer(_FakeAutoModel({}))
        self.assertIsNone(analyzer.analyze(np.zeros(0, dtype=np.float32)))

    def test_analyze_model_failure_degrades_to_none(self) -> None:
        analyzer = self._analyzer(_FakeAutoModel(None))
        self.assertIsNone(analyzer.analyze(np.zeros(16000, dtype=np.float32)))
        # the failure is remembered: the factory is not retried
        self.assertIsNone(analyzer.analyze(np.zeros(16000, dtype=np.float32)))

    def test_analyze_odd_shapes_degrade_to_none(self) -> None:
        for bad in ({}, {"labels": []}, "not-a-list", [{"labels": "x"}]):
            model = _FakeAutoModel(bad if isinstance(bad, dict) else {"labels": ["a"], "scores": [1.0]})
            model._result = bad
            analyzer = self._analyzer(model)
            with self._patch_tensor():
                self.assertIsNone(
                    analyzer.analyze(np.zeros(16000, dtype=np.float32)), repr(bad)
                )

    def test_tail_window_is_applied(self) -> None:
        result = {
            "labels": list(EMOTION2VEC_LABELS),
            "scores": [0.05, 0.02, 0.01, 0.9, 0.01, 0.0, 0.005, 0.004, 0.001],
        }
        model = _FakeAutoModel(result)
        analyzer = self._analyzer(model)
        audio = np.zeros(16 * 16000, dtype=np.float32)  # 16 s utterance
        with self._patch_tensor():
            analyzer.analyze(audio)
        sent = _FakeAutoModel.calls[-1]
        self.assertEqual(int(np.asarray(sent).size), 5 * 16000)

    def test_generate_kwargs_are_utterance_classification(self) -> None:
        result = {
            "labels": list(EMOTION2VEC_LABELS),
            "scores": [0.05, 0.02, 0.01, 0.9, 0.01, 0.0, 0.005, 0.004, 0.001],
        }
        model = _FakeAutoModel(result)
        analyzer = self._analyzer(model)
        with self._patch_tensor():
            analyzer.analyze(np.zeros(16000, dtype=np.float32))
        kwargs = model.generate_kwargs[0]
        self.assertEqual(kwargs.get("granularity"), "utterance")
        self.assertEqual(kwargs.get("extract_embedding"), False)

    def test_warm_up_preloads_the_model(self) -> None:
        result = {
            "labels": list(EMOTION2VEC_LABELS),
            "scores": [0.05, 0.02, 0.01, 0.9, 0.01, 0.0, 0.005, 0.004, 0.001],
        }
        model = _FakeAutoModel(result)
        analyzer = self._analyzer(model)
        self.assertTrue(analyzer.warm_up())
        self.assertEqual(model.generate_kwargs, [])  # loaded, not analyzed
        # a failing factory warm-up reports False and degrades
        analyzer2 = self._analyzer(None)
        self.assertFalse(analyzer2.warm_up())


class ModuleImportHygieneTests(unittest.TestCase):
    """CONTRACT rule 2: no heavy import at module level (sandbox safety)."""

    def test_emotion_module_imports_without_heavy_deps(self) -> None:
        import app.emotion as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        # top-level (unindented) import lines must not name heavy packages
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if line.startswith(" ") or line.startswith("\t"):
                continue  # indented = inside a function -> lazy, allowed
            for heavy in ("funasr", "torch", "onnxruntime", "sounddevice", "transformers"):
                self.assertNotIn(heavy, line, f"top-level heavy import: {line}")


if __name__ == "__main__":
    unittest.main()
