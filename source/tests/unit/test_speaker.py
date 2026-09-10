"""Unit tests for app/speaker.py (M3 speaker recognition).

Everything here runs in the dependency-free sandbox: the pure cosine /
identification / averaging math, the JSON registry persistence, and the
embedder/recognizer wired to a FAKE model factory (no speechbrain/torch is
ever imported).
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

from app.speaker import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_REGISTRATION_MIN_S,
    DEFAULT_WINDOW_S,
    ECAPA_EMBEDDING_DIM,
    UNKNOWN_USER_ID,
    SpeakerEmbedder,
    SpeakerIdentification,
    SpeakerRecognizer,
    SpeakerRegistry,
    average_embeddings,
    cosine_similarity,
    identify_speaker,
    tail_window,
)


def _emb(seed: int, dim: int = 8, scale: float = 1.0) -> list[float]:
    """Deterministic pseudo-embedding (stable per seed)."""
    rng = np.random.default_rng(seed)
    return [float(x) * scale for x in rng.standard_normal(dim)]


class _FakeModel:
    """Fake speechbrain interface: encode_batch -> fixed (1, 1, dim)."""

    def __init__(self, embedding: list[float], fail: bool = False) -> None:
        self.embedding = embedding
        self.fail = fail
        self.calls = 0

    def encode_batch(self, tensor: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError("fake encode failure")
        arr = np.asarray(self.embedding, dtype=np.float32).reshape(1, 1, -1)
        return arr


def _fake_factory(embedding: list[float], fail: bool = False):
    def factory(model_dir: Path) -> _FakeModel:
        return _FakeModel(embedding, fail=fail)
    return factory


# --------------------------------------------------------------------- math #


class CosineSimilarityTests(unittest.TestCase):
    """Pure cosine math (no numpy, scale-invariant, never raises)."""

    def test_identical_vectors_score_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_scale_invariance(self) -> None:
        self.assertAlmostEqual(
            cosine_similarity([2, 0], [7, 0]), cosine_similarity([1, 0], [1, 0])
        )

    def test_empty_and_mismatch_never_raise(self) -> None:
        self.assertEqual(cosine_similarity([], []), 0.0)
        self.assertEqual(cosine_similarity([1, 2], [1]), 0.0)
        self.assertEqual(cosine_similarity(None, [1]), 0.0)  # type: ignore[arg-type]

    def test_zero_vector_returns_zero(self) -> None:
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)


class IdentifySpeakerTests(unittest.TestCase):
    """Spec 9.3.2 identification: best cosine with threshold."""

    def test_best_match_above_threshold_wins(self) -> None:
        refs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        result = identify_speaker([1.0, 0.1], refs, threshold=0.5)
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "a")
        self.assertTrue(result.known)
        self.assertEqual(result.best_id, "a")
        self.assertEqual(result.registered, 2)
        self.assertGreater(result.similarity, 0.5)

    def test_below_threshold_is_unknown_not_error(self) -> None:
        refs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        # [1, 1] is ~0.707 from both axes: above the default 0.5 threshold
        # but clearly below a stricter 0.8 one -> unknown, no exception.
        result = identify_speaker([1.0, 1.0], refs, threshold=0.8)
        self.assertIsNotNone(result)
        self.assertIsNone(result.id)
        self.assertFalse(result.known)
        self.assertIsNotNone(result.best_id)  # best candidate still reported
        self.assertLess(result.similarity, 0.8)

    def test_missing_embedding_returns_none(self) -> None:
        """A dead analyzer (None embedding) is NOT an 'unknown' record."""
        self.assertIsNone(identify_speaker(None, {"a": [1.0]}, 0.5))
        self.assertIsNone(identify_speaker([], {"a": [1.0]}, 0.5))

    def test_empty_registry_returns_unknown(self) -> None:
        result = identify_speaker([1.0, 0.0], {}, threshold=0.5)
        self.assertIsNotNone(result)
        self.assertIsNone(result.id)
        self.assertEqual(result.registered, 0)
        self.assertEqual(result.similarity, 0.0)

    def test_default_threshold_is_spec_value(self) -> None:
        self.assertEqual(DEFAULT_MATCH_THRESHOLD, 0.5)

    def test_identification_dict_shape(self) -> None:
        refs = {"a": [1.0, 0.0]}
        result = identify_speaker([1.0, 0.0], refs)
        assert result is not None
        payload = result.to_dict()
        self.assertEqual(
            set(payload.keys()),
            {"id", "known", "similarity", "best_id", "registered", "latency_ms"},
        )
        self.assertTrue(payload["known"])


class AverageEmbeddingsTests(unittest.TestCase):
    """Registration averaging (spec 9.3.3)."""

    def test_elementwise_mean(self) -> None:
        self.assertEqual(average_embeddings([[1, 1], [3, 3]]), [2.0, 2.0])

    def test_empty_returns_empty(self) -> None:
        self.assertEqual(average_embeddings([]), [])
        self.assertEqual(average_embeddings([[], [1]]), [1.0])

    def test_shape_mismatch_rows_skipped(self) -> None:
        self.assertEqual(average_embeddings([[1, 2], [9, 9, 9]]), [1.0, 2.0])

    def test_averaging_smooths_outliers(self) -> None:
        """Mean of several chunks beats any single noisy sample (the spec's
        rationale for ~10 s registrations)."""
        refs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        mean = average_embeddings(refs)
        self.assertAlmostEqual(mean[0], 2 / 3)
        self.assertAlmostEqual(mean[1], 2 / 3)


class TailWindowTests(unittest.TestCase):
    """The last N seconds of the utterance (spec 9.3.1)."""

    def test_shorter_audio_returned_whole(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        out = tail_window(audio, 16000, 5.0)
        self.assertEqual(int(np.asarray(out).size), 100)

    def test_longer_audio_cut_to_window(self) -> None:
        audio = np.arange(16000 * 7, dtype=np.float32)
        out = tail_window(audio, 16000, 5.0)
        self.assertEqual(int(np.asarray(out).size), 16000 * 5)
        self.assertEqual(float(out[0]), 2 * 16000)  # the LAST 5 s

    def test_duck_typed_sizeless_input_passes_through(self) -> None:
        self.assertEqual(tail_window([1, 2, 3], 16000, 5.0), [1, 2, 3])

    def test_default_window_is_spec_upper_bound(self) -> None:
        self.assertEqual(DEFAULT_WINDOW_S, 5.0)


# ---------------------------------------------------------------- registry #


class SpeakerRegistryTests(unittest.TestCase):
    """JSON persistence of reference embeddings."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "speaker_registry.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_file_starts_empty(self) -> None:
        registry = SpeakerRegistry(self.path)
        self.assertEqual(registry.speaker_ids(), [])
        self.assertEqual(registry.count(), 0)
        self.assertEqual(registry.references(), {})

    def test_register_persists_and_roundtrips(self) -> None:
        registry = SpeakerRegistry(self.path)
        ok = registry.register("thomas", _emb(1), samples=3, audio_seconds=10.4)
        self.assertTrue(ok)
        # a NEW instance reads the same file (persistence, not in-memory)
        fresh = SpeakerRegistry(self.path)
        self.assertEqual(fresh.speaker_ids(), ["thomas"])
        refs = fresh.references()
        self.assertEqual(len(refs["thomas"]), len(_emb(1)))
        self.assertAlmostEqual(refs["thomas"][0], _emb(1)[0])

    def test_reregistering_overwrites(self) -> None:
        registry = SpeakerRegistry(self.path)
        registry.register("thomas", _emb(1))
        registry.register("thomas", _emb(2))
        refs = registry.references()
        self.assertAlmostEqual(refs["thomas"][0], _emb(2)[0])
        self.assertEqual(registry.count(), 1)

    def test_remove_speaker(self) -> None:
        registry = SpeakerRegistry(self.path)
        registry.register("thomas", _emb(1))
        self.assertTrue(registry.remove("thomas"))
        self.assertEqual(registry.count(), 0)
        self.assertFalse(registry.remove("thomas"))  # already gone

    def test_empty_embedding_rejected(self) -> None:
        registry = SpeakerRegistry(self.path)
        self.assertFalse(registry.register("", _emb(1)))
        self.assertFalse(registry.register("thomas", []))
        self.assertFalse(registry.register("thomas", None))  # type: ignore[arg-type]

    def test_corrupt_file_degrades_to_empty(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        registry = SpeakerRegistry(self.path)
        self.assertEqual(registry.speaker_ids(), [])

    def test_registry_schema_in_file(self) -> None:
        registry = SpeakerRegistry(self.path)
        registry.register("thomas", _emb(1))
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 1)
        self.assertIn("embedding", data["speakers"]["thomas"])
        self.assertEqual(data["speakers"]["thomas"]["samples"], 1)


# ----------------------------------------------------------------- embedder #


class SpeakerEmbedderTests(unittest.TestCase):
    """Embedder with a FAKE speechbrain factory (no heavy imports)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self._tmp.name) / "ecapa"
        self.model_dir.mkdir(parents=True)
        for name in SpeakerEmbedder.REQUIRED_FILES:
            (self.model_dir / name).write_bytes(b"x" * 16)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_required_files_contract(self) -> None:
        self.assertEqual(
            SpeakerEmbedder.REQUIRED_FILES,
            (
                "embedding_model.ckpt",
                "hyperparams.yaml",
                "mean_var_norm_emb.ckpt",
                "classifier.ckpt",
                "label_encoder.txt",
            ),
        )

    def test_files_present(self) -> None:
        embedder = SpeakerEmbedder(self.model_dir)
        self.assertTrue(embedder.files_present())
        (self.model_dir / "hyperparams.yaml").unlink()
        self.assertFalse(embedder.files_present())

    def test_is_available_false_without_speechbrain(self) -> None:
        """The sandbox has no speechbrain -> is_available is False even with
        the files present (guarded import, CONTRACT rule 2)."""
        embedder = SpeakerEmbedder(self.model_dir)
        self.assertFalse(embedder.is_available())

    def test_embed_with_fake_model_factory(self) -> None:
        target = _emb(11, dim=ECAPA_EMBEDDING_DIM)
        embedder = SpeakerEmbedder(self.model_dir, model_factory=_fake_factory(target))
        self.assertTrue(embedder.warm_up())
        audio = np.zeros(16000 * 3, dtype=np.float32)
        result = embedder.embed(audio)
        self.assertEqual(len(result), len(target))
        np.testing.assert_allclose(result, target, rtol=1e-6)  # float32 pass

    def test_embed_returns_none_on_model_failure(self) -> None:
        embedder = SpeakerEmbedder(
            self.model_dir, model_factory=_fake_factory(_emb(1), fail=True)
        )
        embedder.warm_up()
        self.assertIsNone(embedder.embed(np.zeros(16000, dtype=np.float32)))

    def test_embed_empty_audio_returns_none(self) -> None:
        embedder = SpeakerEmbedder(self.model_dir, model_factory=_fake_factory(_emb(1)))
        embedder.warm_up()
        self.assertIsNone(embedder.embed(np.zeros(0, dtype=np.float32)))
        self.assertIsNone(embedder.embed(None))

    def test_failed_load_is_remembered_not_retried(self) -> None:
        calls = {"count": 0}

        def factory(model_dir: Path) -> Any:
            calls["count"] += 1
            raise RuntimeError("load failed")

        embedder = SpeakerEmbedder(self.model_dir, model_factory=factory)
        self.assertFalse(embedder.warm_up())
        self.assertIsNone(embedder.embed(np.zeros(16000, dtype=np.float32)))
        self.assertIsNone(embedder.embed(np.zeros(16000, dtype=np.float32)))
        self.assertEqual(calls["count"], 1)  # one attempt only

    def test_flatten_accepts_nested_shapes(self) -> None:
        flatten = SpeakerEmbedder._flatten
        arr = np.asarray([[2.0, 3.0]], dtype=np.float32)
        self.assertEqual(flatten(arr), [2.0, 3.0])
        self.assertEqual(flatten([1.0, 2.0]), [1.0, 2.0])
        self.assertIsNone(flatten(None))
        self.assertIsNone(flatten("bad"))  # type: ignore[arg-type]

    def test_embed_applies_tail_window(self) -> None:
        seen: dict[str, Any] = {}

        class _WindowProbe(_FakeModel):
            def encode_batch(self, tensor: Any) -> Any:
                seen["samples"] = int(tensor.shape[-1])
                return super().encode_batch(tensor)

        target = _emb(3, dim=4)

        def factory(model_dir: Path) -> _WindowProbe:
            return _WindowProbe(target)

        embedder = SpeakerEmbedder(self.model_dir, window_s=2.0, model_factory=factory)
        embedder.warm_up()
        audio = np.zeros(16000 * 6, dtype=np.float32)  # 6 s -> last 2 s
        embedder.embed(audio)
        self.assertEqual(seen["samples"], 16000 * 2)


# --------------------------------------------------------------- recognizer #


class SpeakerRecognizerTests(unittest.TestCase):
    """Facade: embedder + registry + threshold (fake factory)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self._tmp.name) / "ecapa"
        self.model_dir.mkdir(parents=True)
        for name in SpeakerEmbedder.REQUIRED_FILES:
            (self.model_dir / name).write_bytes(b"x" * 16)
        self.registry_path = Path(self._tmp.name) / "registry.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _recognizer(
        self, embedding: list[float], threshold: float = 0.5
    ) -> SpeakerRecognizer:
        embedder = SpeakerEmbedder(
            self.model_dir, model_factory=_fake_factory(embedding)
        )
        registry = SpeakerRegistry(self.registry_path)
        return SpeakerRecognizer(
            embedder, registry, threshold=threshold, sample_rate=16000
        )

    def test_registration_and_identification_roundtrip(self) -> None:
        """The golden M3 path: register 'thomas', then every identical chunk
        identifies as thomas."""
        target = _emb(21, dim=ECAPA_EMBEDDING_DIM)
        recognizer = self._recognizer(target)
        ok, seconds = recognizer.register("thomas", [np.zeros(16000 * 5, dtype=np.float32)])
        self.assertTrue(ok)
        self.assertAlmostEqual(seconds, 5.0)
        result = recognizer.identify(np.zeros(16000, dtype=np.float32))
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "thomas")
        self.assertAlmostEqual(result.similarity, 1.0)
        self.assertGreater(result.latency_ms, 0.0)

    def test_identification_with_no_registry_is_unknown(self) -> None:
        recognizer = self._recognizer(_emb(5))
        result = recognizer.identify(np.zeros(16000, dtype=np.float32))
        self.assertIsNotNone(result)
        self.assertIsNone(result.id)
        self.assertEqual(result.registered, 0)

    def test_identification_with_dead_embedder_returns_none(self) -> None:
        embedder = SpeakerEmbedder(
            self.model_dir, model_factory=_fake_factory(_emb(1), fail=True)
        )
        recognizer = SpeakerRecognizer(
            embedder, SpeakerRegistry(self.registry_path)
        )
        # embed fails -> identify returns None (not an unknown record)
        result = recognizer.identify(np.zeros(16000, dtype=np.float32))
        self.assertIsNone(result)

    def test_threshold_is_configurable(self) -> None:
        recognizer = self._recognizer(_emb(7), threshold=0.99)
        self.assertEqual(recognizer.threshold, 0.99)

    def test_registration_from_multiple_chunks_averages(self) -> None:
        """Multiple chunks -> the registry stores the MEAN embedding."""
        chunks = [np.zeros(16000, dtype=np.float32) for _ in range(3)]
        target = _emb(9, dim=6)
        recognizer = self._recognizer(target)
        ok, seconds = recognizer.register("anna", chunks)
        self.assertTrue(ok)
        self.assertAlmostEqual(seconds, 3.0)
        refs = recognizer.registry.references()
        # all chunks embed identically -> the mean equals one embedding
        # (float32 precision through the fake model)
        np.testing.assert_allclose(refs["anna"], target, rtol=1e-6)

    def test_registration_empty_chunks_fails(self) -> None:
        recognizer = self._recognizer(_emb(2))
        ok, seconds = recognizer.register("x", [])
        self.assertFalse(ok)
        self.assertEqual(seconds, 0.0)

    def test_short_registration_logs_warning(self) -> None:
        recognizer = self._recognizer(_emb(4))
        with self.assertLogs("app.speaker", level="WARNING") as captured:
            recognizer.register("thomas", [np.zeros(16000, dtype=np.float32)])
        self.assertTrue(
            any("10 s" in rec.getMessage() for rec in captured.records)
        )

    def test_default_registration_floor_matches_spec(self) -> None:
        self.assertEqual(DEFAULT_REGISTRATION_MIN_S, 10.0)

    def test_unknown_user_id_constant(self) -> None:
        self.assertEqual(UNKNOWN_USER_ID, "voice_user")


class ImportHygieneTests(unittest.TestCase):
    """The module must stay importable without speechbrain/torch (CONTRACT 2)."""

    def test_no_heavy_imports_at_module_level(self) -> None:
        import app.speaker as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        # only UNINDENTED (module-level) import lines are checked - heavy
        # imports inside methods are the lazy pattern by design
        for line in source.splitlines():
            if line.startswith(("import torch", "import speechbrain",
                                "from torch", "from speechbrain")):
                self.fail(f"module-level heavy import: {line}")


if __name__ == "__main__":
    unittest.main()
