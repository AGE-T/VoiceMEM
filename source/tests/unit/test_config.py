"""Unit tests for app.config (AgentConfig) — stdlib unittest + PyYAML.

Covers the M0 repo-root default (``default_root()`` -> THIS repository),
the M0 Section 7 model layout properties (models/asr|llm|tts|vad|embedding),
the new M0 fields (model_root/data_root/logs_root, llm_parallel, audio
 devices) and the env-override chain.

Environment variables are neutralized (set to empty strings, which the config
treats as unset) via ``unittest.mock.patch.dict`` so no outer-environment
pollution can leak into the assertions; patch.dict restores everything.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import app.config as config_module
from app.config import AgentConfig, default_root

#: Repository root — two levels above this test file's package (tests/unit/).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Every environment variable AgentConfig reads — neutralized to "" (falsy) in tests.
CONFIG_ENV_KEYS = (
    "LLAMA_SERVER_HOST",
    "LLAMA_SERVER_PORT",
    "LLAMA_MODEL_PATH",
    "LLAMA_CONTEXT_SIZE",
    "LLAMA_N_GPU_LAYERS",
    "LLAMA_CACHE_TYPE_K",
    "LLAMA_CACHE_TYPE_V",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "VOICEMEM_MEMORY_ROOT",
    "VOICEMEM_EMBED_DIM",
    "VOICEMEM_HOME",
    "PIPER_EXECUTABLE",
    "PIPER_VOICES_PATH",
    "TTS_HU_VOICE",
    "TTS_EN_VOICE",
    "VOICEMEM_LOG_LEVEL",
    "VOICEMEM_ENABLE_EMOTION",
    "VOICEMEM_ENABLE_SPEAKER",
    "SILERO_VAD_PATH",
    "QWEN3_ASR_MODEL_PATH",
    "EMBEDDING_MODEL_PATH",
)


def neutral_env(**overrides: str) -> dict[str, str]:
    """Env mapping with every config variable neutralized, plus overrides."""
    env = {key: "" for key in CONFIG_ENV_KEYS}
    env.update(overrides)
    return env


class DefaultsTests(unittest.TestCase):
    """Default construction and validate()."""

    def test_defaults_valid(self) -> None:
        """A default config is valid and matches the M1 component choices."""
        with patch.dict(os.environ, neutral_env()):
            cfg = AgentConfig()
        self.assertEqual(cfg.validate(), [])
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertEqual(cfg.output_sample_rate, 22050)
        self.assertEqual(cfg.voicemem_mode, "text_mode")
        self.assertTrue(cfg.enable_emotion)   # M2 default: on (8.6 flag)
        self.assertTrue(cfg.enable_speaker)   # M3 default: on (9.5 flag)
        self.assertEqual(cfg.vad_frame_samples, 512)      # 32 ms @ 16 kHz
        self.assertEqual(cfg.asr_chunk_samples, 9600)     # 600 ms @ 16 kHz
        self.assertEqual(cfg.llama_server_port, 8080)
        self.assertEqual(cfg.asr_model_name, "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(cfg.llama_server_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(cfg.llama_server_health_url, "http://127.0.0.1:8080/health")

    def test_m0_default_fields(self) -> None:
        """New M0 dataclass fields: model/data/logs roots, parallel, audio."""
        with patch.dict(os.environ, neutral_env()):
            cfg = AgentConfig()
        self.assertEqual(cfg.model_root, "models")
        self.assertEqual(cfg.data_root, "data")
        self.assertEqual(cfg.logs_root, "logs")
        self.assertEqual(cfg.llm_parallel, 1)
        self.assertEqual(cfg.audio_input_device, "")   # "" = system default
        self.assertEqual(cfg.audio_output_device, "")

    def test_m0_model_layout_properties(self) -> None:
        """M0 Section 7 layout: every component sits under models/ by kind."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, neutral_env(VOICEMEM_HOME=tmp)):
                cfg = AgentConfig()
            root = cfg.root
            self.assertEqual(cfg.models_dir, root / "models")
            self.assertEqual(cfg.voices_dir, root / "models" / "tts" / "piper")
            self.assertEqual(
                cfg.silero_vad_path, root / "models" / "vad" / "silero-vad" / "silero_vad.onnx"
            )
            self.assertEqual(cfg.asr_model_dir, root / "models" / "asr" / "qwen3-asr-0.6b")
            # v0.4.17: NO phantom default path. Without a user selection, env
            # var or a GGUF ACTUALLY PRESENT in models/llm/qwen3.6-35b-a3b/,
            # llm_model_file is None (the honest "no model configured" state —
            # the v0.4.16 chain returned a non-existent path here).
            self.assertIsNone(cfg.llm_model_file)
            # A GGUF actually placed in the operator dir (any file name —
            # magic decides, an Ollama sha256-... blob qualifies) is the
            # default; a broken file (magic without a valid header) is not:
            op_dir = root / "models" / "llm" / "qwen3.6-35b-a3b"
            op_dir.mkdir(parents=True, exist_ok=True)
            canonical = op_dir / "Qwen3.6-35B-A3B-IQ4_XS.gguf"
            canonical.write_bytes(b"GGUF" + b"\x00" * 64)  # broken header
            self.assertIsNone(cfg.llm_model_file)
            import struct

            canonical.write_bytes(
                b"GGUF" + struct.pack("<IIQQ", 3, 0, 0, 0) + b"\x00" * 32
            )  # minimal valid GGUF header
            self.assertEqual(cfg.llm_model_file, canonical)
            self.assertEqual(
                cfg.embedding_model_dir, root / "models" / "embedding" / "multilingual-e5-small"
            )

    def test_m2_emotion_layout_and_tunables(self) -> None:
        """M2: emotion model dir under models/emotion, log under data/."""
        with patch.dict(os.environ, neutral_env()):
            cfg = AgentConfig()
        root = cfg.root
        self.assertEqual(
            cfg.emotion_model_dir, root / "models" / "emotion" / "emotion2vec-plus-base"
        )
        self.assertEqual(cfg.emotion_log_path, root / "data" / "emotion_log.jsonl")
        self.assertEqual(cfg.emotion_window_s, 5.0)
        self.assertAlmostEqual(cfg.emotion_fusion_prosody_weight, 0.6)
        self.assertAlmostEqual(cfg.emotion_slow_length_scale, 1.1)
        self.assertAlmostEqual(cfg.emotion_store_threshold, 0.5)
        self.assertEqual(cfg.emotion_model_name, "emotion2vec/emotion2vec_plus_base")
        # check_runtime_assets stays M1-hard; emotion has its own soft check
        self.assertNotIn("emotion", " ".join(cfg.check_runtime_assets().keys()))
        self.assertTrue(set(cfg.check_emotion_assets()) >= {
            "emotion_model_pt", "emotion_config", "emotion_tokens",
        })

    def test_m0_memory_data_logs_defaults(self) -> None:
        """Memory lives in <root>/memory (NOT memoryspace); data/ and logs/."""
        with patch.dict(os.environ, neutral_env()):
            cfg = AgentConfig()
        self.assertEqual(cfg.memory_root_path, cfg.root / "memory")
        self.assertEqual(cfg.data_dir, cfg.root / "data")
        self.assertEqual(cfg.logs_dir, cfg.root / "logs")

    def test_model_root_field_moves_the_whole_layout(self) -> None:
        """model_root/data_root/logs_root fields relocate every derived path."""
        with patch.dict(os.environ, neutral_env()):
            cfg = AgentConfig(model_root="m", data_root="d", logs_root="l")
        self.assertEqual(cfg.models_dir, cfg.root / "m")
        self.assertEqual(cfg.voices_dir, cfg.root / "m" / "tts" / "piper")
        self.assertEqual(cfg.embedding_model_dir, cfg.root / "m" / "embedding" / "multilingual-e5-small")
        self.assertEqual(cfg.data_dir, cfg.root / "d")
        self.assertEqual(cfg.logs_dir, cfg.root / "l")

    def test_memory_root_field_override(self) -> None:
        with patch.dict(os.environ, neutral_env()):
            cfg = AgentConfig(memory_root="custom/mem")
        self.assertEqual(cfg.memory_root_path, Path("custom/mem"))

    def test_embedding_model_path_field_and_env(self) -> None:
        """embedding dir: field override, then EMBEDDING_MODEL_PATH env wins."""
        with patch.dict(os.environ, neutral_env()):
            cfg = AgentConfig(embedding_model_path="custom/embed")
            self.assertEqual(cfg.embedding_model_dir, Path("custom/embed"))
        with patch.dict(os.environ, neutral_env(EMBEDDING_MODEL_PATH="/env/embed")):
            cfg2 = AgentConfig()
            # the property reads the env LIVE — assert inside the patch scope
            self.assertEqual(cfg2.embedding_model_dir, Path("/env/embed"))

    def test_validate_rejects_zero_llm_parallel(self) -> None:
        cfg = AgentConfig()
        cfg.llm_parallel = 0
        self.assertTrue(any("llm_parallel" in e for e in cfg.validate()))
        cfg.llm_parallel = 2
        self.assertEqual(cfg.validate(), [])

    def test_validate_rejects_multi_modal_mode(self) -> None:
        cfg = AgentConfig()
        cfg.voicemem_mode = "multi_modal"
        errors = cfg.validate()
        self.assertTrue(any("voicemem_mode" in e for e in errors))

    def test_validate_accepts_emotion_flag_m2(self) -> None:
        """M2: enable_emotion is a legal config state (False = M1-identical)."""
        cfg = AgentConfig()
        self.assertTrue(cfg.enable_emotion)
        self.assertEqual(cfg.validate(), [])
        cfg.enable_emotion = False
        self.assertEqual(cfg.validate(), [])  # M1-identical mode also valid

    def test_validate_rejects_bad_emotion_tunables(self) -> None:
        for field, bad in (
            ("emotion_window_s", 0.0),
            ("emotion_window_s", 60.0),
            ("emotion_fusion_prosody_weight", 1.5),
            ("emotion_fusion_prosody_weight", -0.1),
            ("emotion_slow_length_scale", 0.9),
            ("emotion_slow_length_scale", 3.0),
            ("emotion_store_threshold", 1.5),
        ):
            cfg = AgentConfig()
            setattr(cfg, field, bad)
            self.assertTrue(
                any(field in e for e in cfg.validate()),
                f"{field}={bad} was accepted",
            )

    def test_validate_accepts_speaker_flag(self) -> None:
        """M3: enable_speaker=True is now the legal default (9.5 flag)."""
        cfg = AgentConfig()
        cfg.enable_speaker = True
        self.assertFalse(any("enable_speaker" in e for e in cfg.validate()))

    def test_validate_rejects_bad_speaker_fields(self) -> None:
        """M3 field ranges: threshold (0,1], window [0.5,30], reg_min [2,60]."""
        cfg = AgentConfig()
        cfg.speaker_match_threshold = 0.0
        self.assertTrue(any("speaker_match_threshold" in e for e in cfg.validate()))
        cfg = AgentConfig()
        cfg.speaker_match_threshold = 1.5
        self.assertTrue(any("speaker_match_threshold" in e for e in cfg.validate()))
        cfg = AgentConfig()
        cfg.speaker_window_s = 0.1
        self.assertTrue(any("speaker_window_s" in e for e in cfg.validate()))
        cfg = AgentConfig()
        cfg.speaker_registration_min_s = 1.0
        self.assertTrue(any("speaker_registration_min_s" in e for e in cfg.validate()))
        # boundaries are valid
        cfg = AgentConfig()
        cfg.speaker_match_threshold = 0.5
        cfg.speaker_window_s = 5.0
        cfg.speaker_registration_min_s = 10.0
        self.assertEqual(cfg.validate(), [])

    def test_validate_rejects_bad_port(self) -> None:
        for bad_port in (0, -1, 65536, 99999):
            cfg = AgentConfig()
            cfg.llama_server_port = bad_port
            self.assertTrue(
                any("llama_server_port" in e for e in cfg.validate()),
                f"port {bad_port} was accepted",
            )
        # boundary ports are valid
        for good_port in (1, 65535):
            cfg = AgentConfig()
            cfg.llama_server_port = good_port
            self.assertEqual(cfg.validate(), [])

    def test_validate_rejects_wrong_sample_rate(self) -> None:
        cfg = AgentConfig()
        cfg.sample_rate = 44100
        self.assertTrue(any("sample_rate" in e for e in cfg.validate()))


class ApplyEnvTests(unittest.TestCase):
    """apply_env() parsing of LLAMA_SERVER_PORT and OPENAI_BASE_URL."""

    def test_llama_server_port_env(self) -> None:
        with patch.dict(os.environ, neutral_env(LLAMA_SERVER_PORT="9001")):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertEqual(cfg.llama_server_port, 9001)

    def test_openai_base_url_with_v1_suffix(self) -> None:
        url = "http://127.0.0.1:8080/v1"
        with patch.dict(os.environ, neutral_env(OPENAI_BASE_URL=url)):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertEqual(cfg.llama_server_host, "127.0.0.1")
        self.assertEqual(cfg.llama_server_port, 8080)
        self.assertEqual(cfg.llama_server_url, "http://127.0.0.1:8080/v1")

    def test_openai_base_url_without_v1_suffix(self) -> None:
        with patch.dict(os.environ, neutral_env(OPENAI_BASE_URL="http://localhost:9999")):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertEqual(cfg.llama_server_host, "localhost")
        self.assertEqual(cfg.llama_server_port, 9999)

    def test_base_url_overrides_explicit_port(self) -> None:
        """OPENAI_BASE_URL is applied after LLAMA_SERVER_PORT, so it wins."""
        with patch.dict(
            os.environ,
            neutral_env(
                LLAMA_SERVER_PORT="9001",
                OPENAI_BASE_URL="http://127.0.0.1:8080/v1",
            ),
        ):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertEqual(cfg.llama_server_host, "127.0.0.1")
        self.assertEqual(cfg.llama_server_port, 8080)

    def test_enable_flags_from_env(self) -> None:
        """VOICEMEM_ENABLE_EMOTION/SPEAKER on/off parsing (M3: both legal)."""
        with patch.dict(
            os.environ,
            neutral_env(VOICEMEM_ENABLE_EMOTION="true", VOICEMEM_ENABLE_SPEAKER="1"),
        ):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertTrue(cfg.enable_emotion)   # M2: legal
        self.assertTrue(cfg.enable_speaker)   # M3: legal
        errors = cfg.validate()
        self.assertFalse(any("enable_emotion" in e for e in errors))
        self.assertFalse(any("enable_speaker" in e for e in errors))
        with patch.dict(os.environ, neutral_env(VOICEMEM_ENABLE_EMOTION="0")):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertFalse(cfg.enable_emotion)  # explicit off
        with patch.dict(os.environ, neutral_env(VOICEMEM_ENABLE_SPEAKER="0")):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertFalse(cfg.enable_speaker)  # explicit off

    def test_speaker_paths_from_env(self) -> None:
        """SPEAKER_MODEL_PATH + VOICEMEM_SPEAKER_THRESHOLD env overrides."""
        with patch.dict(os.environ, neutral_env()):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertEqual(
            cfg.speaker_model_dir,
            cfg.models_dir / "speaker" / "ecapa-voxceleb",
        )
        self.assertEqual(cfg.speaker_registry_file, cfg.data_dir / "speaker_registry.json")
        with patch.dict(
            os.environ,
            neutral_env(SPEAKER_MODEL_PATH="/custom/speaker-model"),
        ):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertEqual(str(cfg.speaker_model_dir), "/custom/speaker-model")


class FromYamlTests(unittest.TestCase):
    """from_yaml() loading, unknown-key tolerance and error handling."""

    def test_from_yaml_loads_known_keys(self) -> None:
        content = "\n".join(
            [
                "llama_server_port: 9999",
                "vad_threshold: 0.62",
                "llm_temperature: 0.3",
                "tts_hu_voice: hu_HU-test-medium",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voicemem_config.yaml"
            path.write_text(content, encoding="utf-8")
            with patch.dict(os.environ, neutral_env()):
                cfg = AgentConfig.from_yaml(path)
        self.assertEqual(cfg.llama_server_port, 9999)
        self.assertAlmostEqual(cfg.vad_threshold, 0.62)
        self.assertAlmostEqual(cfg.llm_temperature, 0.3)
        self.assertEqual(cfg.tts_hu_voice, "hu_HU-test-medium")
        self.assertEqual(cfg.validate(), [])

    def test_from_yaml_ignores_unknown_keys(self) -> None:
        content = "llama_server_port: 7777\nunknown_bogus_key: ignored\nnested: {a: 1}\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voicemem_config.yaml"
            path.write_text(content, encoding="utf-8")
            with patch.dict(os.environ, neutral_env()):
                cfg = AgentConfig.from_yaml(path)  # must not raise
        self.assertEqual(cfg.llama_server_port, 7777)

    def test_from_yaml_ignores_project_and_paths_sections(self) -> None:
        """The M0 documentation sections (project:/paths:) are skipped, and
        the paths: values do NOT leak into AgentConfig fields."""
        content = "\n".join(
            [
                "project:",
                "  name: voicemem-agent",
                "  version: 0.1.0",
                "  offline: true",
                "paths:",
                "  model_root: models",
                "  memory_root: memory",
                "  data_root: data",
                "  logs_root: logs",
                "  hf_home: models/hf",
                "app:",
                "  llama_server_port: 7777",
                "  llm_parallel: 2",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voicemem_config.yaml"
            path.write_text(content, encoding="utf-8")
            with patch.dict(os.environ, neutral_env()):
                cfg = AgentConfig.from_yaml(path)
        self.assertEqual(cfg.llama_server_port, 7777)
        self.assertEqual(cfg.llm_parallel, 2)          # app: section IS loaded
        self.assertEqual(cfg.model_root, "models")     # field default, not leaked
        self.assertEqual(cfg.memory_root, "")
        self.assertEqual(cfg.data_root, "data")
        self.assertEqual(cfg.logs_root, "logs")
        self.assertEqual(cfg.validate(), [])

    def test_from_yaml_tolerates_app_section(self) -> None:
        content = "\n".join(
            [
                "llm_temperature: 0.2",
                "app:",
                "  llama_server_port: 7777",
                "  vad_threshold: 0.7",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voicemem_config.yaml"
            path.write_text(content, encoding="utf-8")
            with patch.dict(os.environ, neutral_env()):
                cfg = AgentConfig.from_yaml(path)
        self.assertEqual(cfg.llama_server_port, 7777)
        self.assertAlmostEqual(cfg.vad_threshold, 0.7)

    def test_from_yaml_missing_file_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.yaml"
            with patch.dict(os.environ, neutral_env()):
                cfg = AgentConfig.from_yaml(missing)
        self.assertEqual(cfg.llama_server_port, 8080)
        self.assertEqual(cfg.validate(), [])

    def test_from_yaml_rejects_non_mapping_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("- just\n- a\n- list\n", encoding="utf-8")
            with patch.dict(os.environ, neutral_env()):
                with self.assertRaises(ValueError):
                    AgentConfig.from_yaml(path)


class RootResolutionTests(unittest.TestCase):
    """VOICEMEM_HOME-driven root resolution."""

    def test_root_resolution_via_voicemem_home_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, neutral_env(VOICEMEM_HOME=tmp)):
                cfg = AgentConfig()
            self.assertEqual(cfg.root, Path(tmp))
            self.assertEqual(cfg.bin_dir, Path(tmp) / "bin")
            self.assertEqual(cfg.models_dir, Path(tmp) / "models")
            self.assertEqual(cfg.voices_dir, Path(tmp) / "models" / "tts" / "piper")
            self.assertEqual(cfg.memory_root_path, Path(tmp) / "memory")

    def test_apply_env_updates_root(self) -> None:
        with patch.dict(os.environ, neutral_env()):
            cfg = AgentConfig()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, neutral_env(VOICEMEM_HOME=tmp)):
                cfg.apply_env()
            self.assertEqual(cfg.root, Path(tmp))

    def test_default_root_is_the_repo_root(self) -> None:
        """M0: the default root is THIS repository (the dir holding app/).

        No Windows drive-letter default, no <repo>/runtime fallback —
        the repo root is the single operating unit (spec Section 26).
        """
        with patch.dict(os.environ, neutral_env()):
            root = default_root()
        self.assertEqual(root, REPO_ROOT)
        self.assertEqual(root, Path(config_module.__file__).resolve().parent.parent)
        self.assertTrue((root / "app" / "config.py").is_file())


class RuntimeAssetsTests(unittest.TestCase):
    """check_runtime_assets() key set and file detection."""

    def test_keys_present_and_values_bool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, neutral_env(VOICEMEM_HOME=tmp)):
                cfg = AgentConfig()
                assets = cfg.check_runtime_assets()
        self.assertEqual(
            set(assets),
            {"llama_model", "piper_executable", "silero_vad", "hu_voice", "en_voice", "asr_model"},
        )
        self.assertTrue(all(isinstance(v, bool) for v in assets.values()))

    def test_all_assets_detected_when_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, neutral_env(VOICEMEM_HOME=tmp)):
                cfg = AgentConfig()
                piper_name = "piper.exe" if platform.system() == "Windows" else "piper"
                # v0.4.17: the LLM asset must be a VALID GGUF for the
                # operator-dir scan to accept it (magic decides).
                import struct

                llm_file = cfg.root / "models" / "llm" / "qwen3.6-35b-a3b" / "m.gguf"
                llm_file.parent.mkdir(parents=True, exist_ok=True)
                llm_file.write_bytes(
                    b"GGUF" + struct.pack("<IIQQ", 3, 0, 0, 0) + b"\x00" * 32
                )
                files = [
                    cfg.bin_dir / piper_name,
                    cfg.silero_vad_path,
                    cfg.voices_dir / f"{cfg.tts_hu_voice}.onnx",
                    cfg.voices_dir / f"{cfg.tts_en_voice}.onnx",
                    cfg.asr_model_dir / "config.json",
                    llm_file,
                ]
                for asset in files:
                    asset.parent.mkdir(parents=True, exist_ok=True)
                    asset.write_bytes(
                        b"GGUF" + struct.pack("<IIQQ", 3, 0, 0, 0) + b"\x00" * 32
                        if asset == llm_file
                        else b""
                    )
                assets = cfg.check_runtime_assets()
        missing = [name for name, ok in assets.items() if not ok]
        self.assertEqual(missing, [], f"assets not detected: {missing}")


if __name__ == "__main__":
    unittest.main()
