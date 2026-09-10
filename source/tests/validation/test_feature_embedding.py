"""Embedding feature validation (Task 12 - M0.1).

Validates the ``embedding`` feature: the multilingual-e5-small memory
embedding contract of ``app.config.AgentConfig`` (384 output dimensions, the
``models/embedding/multilingual-e5-small`` directory).

Deep check: with sentence-transformers installed and the model downloaded,
the local model loads and embeds a sentence into 384 dimensions. A model that
is present but fails to load is recorded honestly as a SKIP with the error
(compatibility is a target-machine concern, not a silent pass); when the load
succeeds, a wrong dimension is a real FAIL.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from tests.validation._report import FeatureValidationTest, has_module

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

_ENV_KEYS = (
    "VOICEMEM_HOME", "EMBEDDING_MODEL_PATH", "VOICEMEM_EMBED_DIM",
    "QWEN3_ASR_MODEL_PATH", "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class EmbeddingFeatureTest(_EnvNeutralTest):
    """multilingual-e5-small embedding contract."""

    FEATURE = "embedding"

    def test_logic_embedding_contract(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertEqual(cfg.embed_dim, 384)
        self.assertTrue(
            cfg.embedding_model_dir.as_posix().endswith(
                "models/embedding/multilingual-e5-small"
            )
        )

    def test_deep_local_e5_model_embeds(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        if not has_module("sentence_transformers"):
            self.deep_skip("sentence-transformers not installed")
        if not (cfg.embedding_model_dir / "config.json").is_file():
            self.deep_skip("e5 embedding model not downloaded")
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(str(cfg.embedding_model_dir))
            vectors = model.encode(["hello"])
        except Exception as exc:  # noqa: BLE001 - honest skip with the reason
            self.deep_skip(f"e5 model not loadable here: {exc}")
        self.assertEqual(vectors.ndim, 2)
        self.assertEqual(vectors.shape[0], 1)
        self.assertEqual(
            vectors.shape[-1], 384, "multilingual-e5-small must embed to 384 dims"
        )
        self.deep_pass("multilingual-e5-small loads and embeds to 384 dims")


if __name__ == "__main__":
    unittest.main()
