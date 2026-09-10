"""Agent configuration (M0 repo-root principle + M1 tunables).

Single config object shared by every component. Resolution order:
  1. YAML file (``config/voicemem_config.yaml`` — optional; its ``app:``
     section maps flat onto AgentConfig fields; unknown top-level sections
     such as ``project:``/``paths:`` are ignored for compatibility)
  2. Environment variables (Windows: ``config/env.local.ps1`` sets them; names mirror
     the V1 spec Section 18.1 so VoiceMem itself can consume the same variables)
  3. Dataclass defaults

M0 principle (spec Section 26): the REPO ROOT is the single operating unit.
``default_root()`` therefore resolves to this repository's root (the parent
of ``app/``); the ``VOICEMEM_HOME`` environment variable still overrides it
for exotic setups, but nothing depends on machine-global locations — the
old Windows drive-letter default and the ``<repo>/runtime`` fallback are
gone.

Pure stdlib — importable on any machine without heavy dependencies.
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

try:
    import yaml

    _HAS_YAML = True
except ImportError:  # pragma: no cover - PyYAML missing on a bare target machine
    _HAS_YAML = False


APP_NAME = "voicemem-agent"


def _parse_env_int(name: str, value: str) -> int:
    """v0.4.9 (report issue #10): parse one integer env value with a clear error.

    A raw ``int(os.environ[...])`` on a bad value raises a bare ValueError
    ('invalid literal for int() with base 10: 'abc') with no hint about WHICH
    variable was wrong — an operator setting LLAMA_SERVER_PORT=808O then got a
    cryptic crash. This wrapper names the variable, echoes the bad value and
    shows a correct example instead.
    """
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {name} must be a whole number "
            f"(got: {value!r}). Example: set {name}=8080"
        ) from None


def _parse_env_float(name: str, value: str) -> float:
    """v0.4.9: parse one float env value with a clear error (see _parse_env_int)."""
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {name} must be a number "
            f"(got: {value!r}). Example: set {name}=0.5"
        ) from None


def default_root() -> Path:
    """Project root: $VOICEMEM_HOME if set, else THIS repo's root (M0 §26).

    The repo root is the directory containing ``app/``, ``config/``, ``models/``
    etc. — resolved from this file's location, so it works from any CWD and on
    any OS. Everything (models, memory, data, logs) lives under it.
    """
    env = os.environ.get("VOICEMEM_HOME")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent


@dataclass
class AgentConfig:
    """All tunables of the M1/M2/M3 agent."""

    # --- paths (M0: relative to the repo root; see spec Sections 7/26) ---
    root: Path = field(default_factory=default_root)
    model_root: str = "models"       # all model assets (asr/llm/tts/vad/embedding)
    data_root: str = "data"          # smoke-test audio, benchmark inputs
    logs_root: str = "logs"          # runtime logs

    # --- audio ---
    sample_rate: int = 16000          # mic input (VAD + ASR), 16 kHz mono
    output_sample_rate: int = 22050   # Piper output, 22.05 kHz mono int16
    channels: int = 1
    audio_input_device: str = ""      # "" = system default mic (sounddevice)
    audio_output_device: str = ""     # "" = system default speaker

    # --- VAD (Silero v6.2.1) ---
    # v0.4.11: 0.5 -> 0.25 — quiet line-in/room speech can peak at 0.25-0.5
    # and never crossed the old 0.5 gate (field log: 93 s of speaking, peak
    # 0.003). Clean speech still scores 0.6-0.95, so 0.25 only widens the
    # sensitivity margin; a ~0.00 peak on real speech is an input-chain
    # problem (browser DSP / resampling), NOT a threshold problem.
    vad_threshold: float = 0.25       # speech start/end threshold
    vad_hangover_ms: int = 300        # silence required before SPEECH_END
    vad_frame_ms: int = 32            # Silero v5/v6 ONNX expects 512 samples @ 16 kHz
    # v0.4.14: energy fallback behind Silero (see app.vad.FusedVad). When the
    # primary VAD is DEAF on the capture channel (v0.4.13 field report: real,
    # ASR-transcribable line-in speech scored 0.003 — the live mic path never
    # dispatched a turn), speech-LEVEL audio still opens the gate through an
    # energy detector, so the automatic mic -> ASR -> agent transition can
    # never die on Silero's opinion alone. Healthy-Silero channels are
    # unaffected: the fallback is passive whenever Silero fires.
    vad_energy_fallback: bool = True
    barge_in_threshold: float = 0.30  # speech prob during TTS playback
    barge_in_min_speech_ms: int = 500 # sustained speech needed to trigger barge-in

    # --- ASR (Qwen3-ASR-0.6B, PyTorch cu128, GPU) ---
    asr_model_name: str = "Qwen/Qwen3-ASR-0.6B"
    asr_model_path: str = ""          # optional local dir override (offline loading)
    asr_device: str = "cuda"
    asr_chunk_ms: int = 600           # quasi-streaming batch size
    # v0.4.7: transcription language. "" = auto-detect (the model reports the
    # detected language itself - 30 languages incl. Hungarian). Force a
    # language with its full English name ("Hungarian", "English", ...) or an
    # ISO code ("hu", "en") when auto-detect misfires.
    asr_language: str = ""
    asr_max_new_tokens: int = 256     # generation cap per transcription call

    # --- LLM (llama.cpp llama-server, OpenAI-compatible) ---
    # v0.4.16: the ONE AND ONLY production LLM is Qwen3.6 35B A3B IQ4_XS
    # (selected via config/env.local.ps1 -> OPENAI_MODEL + LLAMA_MODEL_PATH,
    # or the UI model picker writing config/llm_model.json; see the
    # LLM-MODELPROFIL block there). There is NO fallback profile and no
    # switching back to any previous model. LLM MODEL COUNT at runtime: 1;
    # no silent switch, no cloud fallback, no second model.
    llama_server_host: str = "127.0.0.1"
    llama_server_port: int = 8080
    llm_model_name: str = "qwen3.6-35b-a3b"
    llm_model_path: str = ""          # only used by scripts/start_llama_server.ps1
    llm_context_size: int = 8192
    llm_n_gpu_layers: int = -1
    llm_parallel: int = 1             # llama-server --parallel slots
    llm_cache_type_k: str = "q8_0"
    llm_cache_type_v: str = "q8_0"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    # v0.4.4: llama-server's chat handler defaults ``enable_thinking`` to
    # TRUE; hybrid-reasoning models then answer INSIDE the
    # thought channel -> empty ``content`` -> 0-char replies (the v0.4.3
    # text-mode outage). The client sends request-level thinking
    # suppression (chat_template_kwargs + reasoning_effort "none") unless
    # this is explicitly turned off.
    llm_disable_thinking: bool = True

    # --- TTS (Piper subprocess) ---
    piper_executable: str = ""        # resolved to root/bin/piper[.exe] when empty
    tts_hu_voice: str = "hu_HU-anna-medium"
    tts_en_voice: str = "en_US-lessac-medium"
    tts_first_chunk_chars: int = 24   # first sentence window (low TTFB)
    tts_chunk_chars: int = 80         # later chunks

    # --- VoiceMem ---
    memory_root: str = ""             # resolved to root/memory when empty (M0)
    user_id: str = "voice_user"       # M3 hook — fixed value in M1/M2
    top_k: int = 5
    voicemem_mode: str = "text_mode"  # NEVER "multi_modal" (scene recognition excluded)
    embed_dim: int = 384              # multilingual-e5-small
    embedding_model_path: str = ""    # optional embedding dir override (env EMBEDDING_MODEL_PATH)

    # --- M2 emotion intelligence (emotion2vec+ base, CPU) ---
    enable_emotion: bool = True       # M2 default; False -> M1-identical pipeline
    emotion_model_name: str = "emotion2vec/emotion2vec_plus_base"  # HF repo (docs)
    emotion_model_path: str = ""      # optional dir override (env EMOTION2VEC_MODEL_PATH)
    emotion_window_s: float = 5.0      # prosody window: last N s of the utterance
    emotion_fusion_prosody_weight: float = 0.6  # spec 8.3.3 blend (semantic = 1 - w)
    emotion_slow_length_scale: float = 1.1      # Piper --length_scale when frustrated
    emotion_store_threshold: float = 0.5        # fact persistence threshold (8.3.4)

    # --- M3 speaker recognition (SpeechBrain ECAPA, CPU) ---
    enable_speaker: bool = True       # M3 default; False -> M1/M2-identical pipeline
    speaker_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"  # HF repo (docs)
    speaker_model_path: str = ""      # optional dir override (env SPEAKER_MODEL_PATH)
    speaker_match_threshold: float = 0.5      # spec 9.3.2 ECAPA identification threshold
    speaker_window_s: float = 5.0     # embedding window: last N s of the utterance (9.3.1)
    speaker_registration_min_s: float = 10.0   # spec 9.3.3 clean-speech registration floor
    speaker_registry_path: str = ""   # optional registry JSON override (resolved to data/)

    # --- misc ---
    offline: bool = True
    log_level: str = "INFO"

    # ------------------------------------------------------------------ helpers

    @property
    def llama_server_url(self) -> str:
        return f"http://{self.llama_server_host}:{self.llama_server_port}/v1"

    @property
    def llama_server_health_url(self) -> str:
        return f"http://{self.llama_server_host}:{self.llama_server_port}/health"

    @property
    def bin_dir(self) -> Path:
        return self.root / "bin"

    @property
    def models_dir(self) -> Path:
        """All model assets live here (M0 §7 layout: asr/llm/tts/vad/embedding)."""
        return self.root / self.model_root

    @property
    def data_dir(self) -> Path:
        """Smoke-test audio + benchmark inputs (installer writes here)."""
        return self.root / self.data_root

    @property
    def logs_dir(self) -> Path:
        """Runtime logs root."""
        return self.root / self.logs_root

    @property
    def voices_dir(self) -> Path:
        env = os.environ.get("PIPER_VOICES_PATH")
        if env:
            return Path(env)
        return self.models_dir / "tts" / "piper"

    @property
    def piper_exe_path(self) -> Path:
        if self.piper_executable:
            return Path(self.piper_executable)
        name = "piper.exe" if platform.system() == "Windows" else "piper"
        return self.bin_dir / name

    @property
    def silero_vad_path(self) -> Path:
        env = os.environ.get("SILERO_VAD_PATH")
        if env:
            return Path(env)
        return self.models_dir / "vad" / "silero-vad" / "silero_vad.onnx"

    @property
    def asr_model_dir(self) -> Path:
        env = os.environ.get("QWEN3_ASR_MODEL_PATH")
        if env:
            return Path(env)
        return self.models_dir / "asr" / "qwen3-asr-0.6b"

    @property
    def memory_root_path(self) -> Path:
        env = os.environ.get("VOICEMEM_MEMORY_ROOT")
        if env:
            return Path(env)
        if self.memory_root:
            return Path(self.memory_root)
        return self.root / "memory"

    @property
    def llm_model_file(self) -> Optional[Path]:
        # v0.4.16: PORTABLE MODEL PATH. GGUF files are tens of GB and may
        # live on ANY drive (D:\AI\Models, E:\LLM, an Ollama blob store —
        # C:\AI_HOME\models\blobs\sha256-... blobs load DIRECTLY: llama.cpp
        # checks the GGUF magic, never the file name). Resolution (mirrored
        # by scripts/start_llama_server.ps1 and scripts/verify_m1.ps1 — keep
        # the three in sync):
        #   1. config/llm_model.json -> llm_model_path (the UI "LLM model"
        #      Browse button persists the operator's pick here; survives
        #      restart AND extract-over upgrade);
        #   2. LLAMA_MODEL_PATH env var (the env.local.ps1 profile value);
        #   3. a GGUF ACTUALLY PRESENT in models/llm/qwen3.6-35b-a3b/
        #      (magic-validated, any file name).
        # v0.4.17: NO phantom default — when nothing resolves this is None
        # (the honest "no model configured" state) instead of a project-
        # default path that does not exist on disk (the UI used to display
        # that non-existent path as if it were the configured model).
        # The file is read IN PLACE — never copied into the repo.
        from app.llm_model_settings import (
            operator_default_candidates,
            user_selected_llm_model_path,
        )

        selected = user_selected_llm_model_path(self.root)
        if selected:
            return Path(selected)
        env = os.environ.get("LLAMA_MODEL_PATH")
        if env:
            return Path(env)
        # v0.4.17: the ONE AND ONLY LLM is Qwen3.6 35B A3B IQ4_XS (operator-
        # placed GGUF, Ollama-tested source
        # hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS; the web UI model
        # picker persists any absolute path into config/llm_model.json, and
        # scripts/identify_ollama_blob.ps1 pins the exact Ollama blob).
        # LLM MODEL COUNT at runtime stays 1 - no fallback chain, no second
        # model; NO file present = None (LLM ERROR; the hint tells the
        # operator to select the GGUF with the picker or run the identify /
        # finder scripts). Legacy Qwen3 4B/8B GGUFs that may still sit in an
        # old models/llm/ directory are deliberately IGNORED.
        candidates = operator_default_candidates(self.root)
        if len(candidates) == 1:
            return Path(candidates[0])
        return None

    @property
    def embedding_model_dir(self) -> Path:
        """multilingual-e5-small — VoiceMem long-term memory embeddings (M0 §7)."""
        env = os.environ.get("EMBEDDING_MODEL_PATH")
        if env:
            return Path(env)
        if self.embedding_model_path:
            return Path(self.embedding_model_path)
        return self.models_dir / "embedding" / "multilingual-e5-small"

    @property
    def emotion_model_dir(self) -> Path:
        """emotion2vec+ base — M2 prosody analyzer (models/emotion, CPU)."""
        env = os.environ.get("EMOTION2VEC_MODEL_PATH")
        if env:
            return Path(env)
        if self.emotion_model_path:
            return Path(self.emotion_model_path)
        return self.models_dir / "emotion" / "emotion2vec-plus-base"

    @property
    def emotion_log_path(self) -> Path:
        """Per-turn emotion JSONL log (data/emotion_log.jsonl, spec 8.3.4)."""
        return self.data_dir / "emotion_log.jsonl"

    @property
    def speaker_model_dir(self) -> Path:
        """SpeechBrain ECAPA - M3 speaker embedder (models/speaker, CPU)."""
        env = os.environ.get("SPEAKER_MODEL_PATH")
        if env:
            return Path(env)
        if self.speaker_model_path:
            return Path(self.speaker_model_path)
        return self.models_dir / "speaker" / "ecapa-voxceleb"

    @property
    def speaker_registry_file(self) -> Path:
        """Registered speaker reference embeddings (data/speaker_registry.json)."""
        if self.speaker_registry_path:
            return Path(self.speaker_registry_path)
        return self.data_dir / "speaker_registry.json"

    @property
    def vad_frame_samples(self) -> int:
        return int(self.sample_rate * self.vad_frame_ms / 1000)

    @property
    def asr_chunk_samples(self) -> int:
        return int(self.sample_rate * self.asr_chunk_ms / 1000)

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AgentConfig":
        """Load YAML, apply env overrides on top, validate."""
        data: dict[str, Any] = {}
        path = Path(path)
        if path.exists():
            if not _HAS_YAML:
                raise RuntimeError(
                    "PyYAML is required to load a YAML config: pip install pyyaml"
                )
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config root must be a mapping: {path}")
            data = loaded
        # tolerate the full spec YAML (with voicemem: block) — take the app section if present
        if "app" in data and isinstance(data["app"], dict):
            data = {**{k: v for k, v in data.items() if k != "app"}, **data["app"]}
        cfg = cls(**{k: v for k, v in data.items() if _is_field(cls, k)})
        cfg.apply_env()
        cfg.root = Path(cfg.root)
        errors = cfg.validate()
        if errors:
            raise ValueError("Invalid config: " + "; ".join(errors))
        return cfg

    def apply_env(self) -> None:
        """Override from environment variables (spec Section 18.1 names).

        v0.4.9 (report issue #10): numeric variables are parsed through
        ``_parse_env_int``/``_parse_env_float`` — a bad value raises a
        ValueError that NAMES the variable and shows a correct example,
        instead of the old cryptic 'invalid literal for int()' (or, for
        VOICEMEM_SPEAKER_THRESHOLD, a SILENT fallback to the default that hid
        the operator's typo entirely). v0.4.9 also honours AUDIO_INPUT_DEVICE
        / AUDIO_OUTPUT_DEVICE for the sounddevice device selection and
        fallback chain in app.audio_io.
        """
        if v := os.environ.get("LLAMA_SERVER_HOST"):
            self.llama_server_host = v
        if v := os.environ.get("LLAMA_SERVER_PORT"):
            self.llama_server_port = _parse_env_int("LLAMA_SERVER_PORT", v)
        if v := os.environ.get("LLAMA_MODEL_PATH"):
            self.llm_model_path = v
        if v := os.environ.get("LLAMA_CONTEXT_SIZE"):
            self.llm_context_size = _parse_env_int("LLAMA_CONTEXT_SIZE", v)
        if v := os.environ.get("LLAMA_N_GPU_LAYERS"):
            self.llm_n_gpu_layers = _parse_env_int("LLAMA_N_GPU_LAYERS", v)
        if v := os.environ.get("LLAMA_CACHE_TYPE_K"):
            self.llm_cache_type_k = v
        if v := os.environ.get("LLAMA_CACHE_TYPE_V"):
            self.llm_cache_type_v = v
        if v := os.environ.get("OPENAI_MODEL"):
            self.llm_model_name = v
        if v := os.environ.get("LLM_DISABLE_THINKING", "").strip().lower():
            self.llm_disable_thinking = v not in ("0", "false", "no", "off")
        # v0.4.14: the energy fallback behind Silero can be turned off for a
        # channel where it is not wanted (VAD_ENERGY_FALLBACK=0).
        if v := os.environ.get("VAD_ENERGY_FALLBACK", "").strip().lower():
            self.vad_energy_fallback = v not in ("0", "false", "no", "off")
        if v := os.environ.get("OPENAI_BASE_URL"):
            self._apply_base_url(v)
        if v := os.environ.get("VOICEMEM_MEMORY_ROOT"):
            self.memory_root = v
        if v := os.environ.get("EMBEDDING_MODEL_PATH"):
            self.embedding_model_path = v
        if v := os.environ.get("VOICEMEM_EMBED_DIM"):
            self.embed_dim = _parse_env_int("VOICEMEM_EMBED_DIM", v)
        if v := os.environ.get("VOICEMEM_HOME"):
            self.root = Path(v)
        if v := os.environ.get("PIPER_EXECUTABLE"):
            self.piper_executable = v
        if v := os.environ.get("TTS_HU_VOICE"):
            self.tts_hu_voice = v
        if v := os.environ.get("TTS_EN_VOICE"):
            self.tts_en_voice = v
        if v := os.environ.get("VOICEMEM_LOG_LEVEL"):
            self.log_level = v
        if v := os.environ.get("EMOTION2VEC_MODEL_PATH"):
            self.emotion_model_path = v
        if os.environ.get("VOICEMEM_ENABLE_EMOTION", "").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            self.enable_emotion = True
        if os.environ.get("VOICEMEM_ENABLE_EMOTION", "").strip().lower() in (
            "0", "false", "no", "off",
        ):
            self.enable_emotion = False
        if os.environ.get("VOICEMEM_ENABLE_SPEAKER", "").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            self.enable_speaker = True
        if os.environ.get("VOICEMEM_ENABLE_SPEAKER", "").strip().lower() in (
            "0", "false", "no", "off",
        ):
            self.enable_speaker = False
        if v := os.environ.get("SPEAKER_MODEL_PATH"):
            self.speaker_model_path = v
        if v := os.environ.get("VOICEMEM_SPEAKER_THRESHOLD"):
            # v0.4.9: a typo here used to be SILENTLY ignored (the old handler
            # swallowed the ValueError and kept the default) — now it is a
            # named, actionable error at startup (report issue #10).
            self.speaker_match_threshold = _parse_env_float(
                "VOICEMEM_SPEAKER_THRESHOLD", v
            )
        if v := os.environ.get("AUDIO_INPUT_DEVICE", "").strip():
            self.audio_input_device = v
        if v := os.environ.get("AUDIO_OUTPUT_DEVICE", "").strip():
            self.audio_output_device = v

    def _apply_base_url(self, url: str) -> None:
        """Parse http://host:port/v1 (OPENAI_BASE_URL pointing at llama-server)."""
        stripped = url.rstrip("/")
        if stripped.endswith("/v1"):
            stripped = stripped[: -len("/v1")]
        if "://" in stripped:
            stripped = stripped.split("://", 1)[1]
        if ":" in stripped:
            host, port = stripped.rsplit(":", 1)
            if port.isdigit():
                self.llama_server_host = host
                self.llama_server_port = int(port)

    # ------------------------------------------------------------------ checks

    def validate(self) -> list[str]:
        """Return a list of configuration errors (empty list == valid)."""
        errors: list[str] = []
        if self.voicemem_mode != "text_mode":
            errors.append(
                f"voicemem_mode must be 'text_mode' (got {self.voicemem_mode!r}) — "
                "multi_modal is excluded in all milestones"
            )
        if not (0.5 <= self.emotion_window_s <= 30.0):
            errors.append("emotion_window_s out of range [0.5, 30]")
        if not (0.0 <= self.emotion_fusion_prosody_weight <= 1.0):
            errors.append("emotion_fusion_prosody_weight out of range [0, 1]")
        if not (1.0 <= self.emotion_slow_length_scale <= 2.0):
            errors.append("emotion_slow_length_scale out of range [1.0, 2.0]")
        if not (0.0 <= self.emotion_store_threshold <= 1.0):
            errors.append("emotion_store_threshold out of range [0, 1]")
        if not (0.0 < self.speaker_match_threshold <= 1.0):
            errors.append("speaker_match_threshold out of range (0, 1]")
        if not (0.5 <= self.speaker_window_s <= 30.0):
            errors.append("speaker_window_s out of range [0.5, 30]")
        if not (2.0 <= self.speaker_registration_min_s <= 60.0):
            errors.append("speaker_registration_min_s out of range [2, 60]")
        if not (0.0 < self.vad_threshold <= 1.0):
            errors.append("vad_threshold out of range (0, 1]")
        if not (0.0 < self.barge_in_threshold <= 1.0):
            errors.append("barge_in_threshold out of range (0, 1]")
        if self.vad_hangover_ms < 0 or self.barge_in_min_speech_ms < 0:
            errors.append("hangover/barge-in durations must be >= 0")
        if not (1 <= self.llama_server_port <= 65535):
            errors.append("llama_server_port out of range")
        if self.sample_rate != 16000:
            errors.append("sample_rate must be 16000 (VAD/ASR requirement)")
        if self.output_sample_rate != 22050:
            errors.append("output_sample_rate must be 22050 (Piper voices are 22.05 kHz)")
        if self.top_k < 1:
            errors.append("top_k must be >= 1")
        if self.llm_parallel < 1:
            errors.append("llm_parallel must be >= 1 (llama-server --parallel slots)")
        if self.asr_max_new_tokens < 16:
            errors.append("asr_max_new_tokens must be >= 16")
        if len(self.asr_language) > 32:
            errors.append("asr_language must be empty (auto) or a short language name")
        return errors

    def check_runtime_assets(self) -> dict[str, bool]:
        """Existence check of every on-disk asset the REAL (non-mock) mode needs.

        M2 note: the emotion model is NOT part of the hard list - a missing
        emotion model degrades to the M1 pipeline (EMOTION block omitted)
        instead of blocking startup (modularity contract 8.6 item 5).
        """
        assets: dict[str, bool] = {
            # v0.4.17: None (no model configured) is honestly "missing".
            "llama_model": bool(
                self.llm_model_file and self.llm_model_file.is_file()
            ),
            "piper_executable": self.piper_exe_path.is_file(),
            "silero_vad": self.silero_vad_path.is_file(),
            "hu_voice": (self.voices_dir / f"{self.tts_hu_voice}.onnx").is_file(),
            "en_voice": (self.voices_dir / f"{self.tts_en_voice}.onnx").is_file(),
            "asr_model": (self.asr_model_dir / "config.json").is_file(),
        }
        return assets

    def check_emotion_assets(self) -> dict[str, bool]:
        """M2 emotion asset checklist (informational, never blocks startup)."""
        model_dir = self.emotion_model_dir
        return {
            "emotion_model_pt": (model_dir / "model.pt").is_file(),
            "emotion_config": (model_dir / "config.yaml").is_file(),
            "emotion_tokens": (model_dir / "tokens.txt").is_file(),
        }

    def check_speaker_assets(self) -> dict[str, bool]:
        """M3 speaker asset checklist (informational, never blocks startup).

        A missing ECAPA model or an empty registry degrades the pipeline to
        the fixed ``user_id`` (M1/M2 behaviour) - modularity 9.5 item 5/6.
        """
        model_dir = self.speaker_model_dir
        return {
            "speaker_model": (model_dir / "embedding_model.ckpt").is_file(),
            "speaker_hparams": (model_dir / "hyperparams.yaml").is_file(),
            "speaker_registry": self.speaker_registry_file.is_file(),
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["root"] = str(self.root)
        data["llama_server_url"] = self.llama_server_url
        return data


def _is_field(cls: type, name: str) -> bool:
    return any(f.name == name for f in fields(cls))
