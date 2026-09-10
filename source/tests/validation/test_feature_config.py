"""Configuration feature validation (Task 12 - M0.1).

Validates the ``config`` feature: ``app.config.AgentConfig`` loaded from the
repo's real ``config/voicemem_config.yaml`` must resolve to the M0/M1 contract
values (audio rates, VAD/barge-in tunables, model names, llama-server endpoint
and the M0 Section 7 path layout), must stay M1-strict (M2/M3 flags off,
text_mode) and must fall back to valid defaults when the YAML path does not
exist. Environment variables consumed by ``apply_env()`` are neutralized so
the outer shell cannot leak into the validation.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

#: Env vars consumed by AgentConfig.apply_env() (mirrors the V1 spec 18.1 set).
_ENV_KEYS = (
    "VOICEMEM_HOME", "VOICEMEM_MEMORY_ROOT", "VOICEMEM_EMBED_DIM",
    "VOICEMEM_LOG_LEVEL", "VOICEMEM_ENABLE_EMOTION", "VOICEMEM_ENABLE_SPEAKER",
    "PIPER_VOICES_PATH", "PIPER_EXECUTABLE", "TTS_HU_VOICE", "TTS_EN_VOICE",
    "SILERO_VAD_PATH", "QWEN3_ASR_MODEL_PATH", "LLAMA_MODEL_PATH",
    "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT", "LLAMA_CONTEXT_SIZE",
    "LLAMA_N_GPU_LAYERS", "LLAMA_CACHE_TYPE_K", "LLAMA_CACHE_TYPE_V",
    "OPENAI_BASE_URL", "OPENAI_MODEL",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class ConfigFeatureTest(_EnvNeutralTest):
    """The real repo YAML resolves to the M0/M1 contract."""

    FEATURE = "config"

    def test_logic_yaml_values_match_contract(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertEqual(cfg.validate(), [], "real config must be valid")
        # M2 contract: emotion ON by default (8.6 flag); M3 speaker ON (9.5 flag).
        self.assertTrue(cfg.enable_emotion, "enable_emotion is True in M2 (8.6 flag)")
        self.assertTrue(cfg.enable_speaker, "enable_speaker is True in M3 (9.5 flag)")
        self.assertEqual(cfg.voicemem_mode, "text_mode")
        self.assertTrue(cfg.offline)
        # Audio contract.
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertEqual(cfg.output_sample_rate, 22050)
        # VAD / barge-in tunables (v0.4.11: vad_threshold 0.5 -> 0.25).
        self.assertEqual(cfg.vad_threshold, 0.25)
        self.assertEqual(cfg.vad_hangover_ms, 300)
        self.assertEqual(cfg.barge_in_threshold, 0.3)
        self.assertEqual(cfg.barge_in_min_speech_ms, 500)
        # M2 emotion tunables.
        self.assertEqual(cfg.emotion_window_s, 5.0)
        self.assertAlmostEqual(cfg.emotion_fusion_prosody_weight, 0.6)
        self.assertAlmostEqual(cfg.emotion_slow_length_scale, 1.1)
        self.assertAlmostEqual(cfg.emotion_store_threshold, 0.5)
        self.assertEqual(
            cfg.emotion_model_name, "emotion2vec/emotion2vec_plus_base"
        )
        # M3 speaker tunables.
        self.assertAlmostEqual(cfg.speaker_match_threshold, 0.5)
        self.assertEqual(cfg.speaker_window_s, 5.0)
        self.assertAlmostEqual(cfg.speaker_registration_min_s, 10.0)
        self.assertEqual(
            cfg.speaker_model_name, "speechbrain/spkrec-ecapa-voxceleb"
        )
        # Model identities.
        self.assertEqual(cfg.asr_model_name, "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(cfg.tts_hu_voice, "hu_HU-anna-medium")
        self.assertEqual(cfg.tts_en_voice, "en_US-lessac-medium")
        # llama-server endpoint (loopback).
        self.assertEqual(cfg.llama_server_host, "127.0.0.1")
        self.assertEqual(cfg.llama_server_port, 8080)

    def test_logic_yaml_paths_match_m0_layout(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertEqual(cfg.root, REPO_ROOT, "root must be the repo root")
        self.assertEqual(cfg.models_dir, cfg.root / "models")
        self.assertEqual(cfg.voices_dir, cfg.models_dir / "tts" / "piper")
        self.assertTrue(cfg.asr_model_dir.as_posix().endswith("models/asr/qwen3-asr-0.6b"))
        # v0.4.17: NO phantom project-default path. llm_model_file resolves
        # to a Path ONLY when a model is ACTUALLY configured (user selection
        # / env / a GGUF really present in models/llm/qwen3.6-35b-a3b/);
        # in the clean repo (no 20 GB file committed) it is None — the
        # honest "no model configured" state, never a non-existent default.
        if cfg.llm_model_file is not None:
            self.assertTrue(
                cfg.llm_model_file.as_posix().endswith(".gguf")
                or "sha256-" in cfg.llm_model_file.name,
                "a resolved LLM path must be a real, operator-chosen GGUF",
            )
        self.assertTrue(cfg.silero_vad_path.as_posix().endswith("silero_vad.onnx"))
        self.assertTrue(
            cfg.embedding_model_dir.as_posix().endswith("models/embedding/multilingual-e5-small")
        )
        self.assertEqual(cfg.memory_root_path, cfg.root / "memory")

    def test_logic_missing_yaml_falls_back_to_defaults(self):
        cfg = AgentConfig.from_yaml(REPO_ROOT / "config" / "does_not_exist.yaml")
        self.assertEqual(cfg.validate(), [], "defaults must be valid on their own")
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertEqual(cfg.output_sample_rate, 22050)
        self.assertEqual(cfg.voicemem_mode, "text_mode")
        self.assertEqual(cfg.models_dir, cfg.root / "models")
        self.assertEqual(cfg.memory_root_path, cfg.root / "memory")


if __name__ == "__main__":
    unittest.main()
