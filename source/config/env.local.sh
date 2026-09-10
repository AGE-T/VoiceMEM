#!/usr/bin/env bash
# config/env.local.sh - M0 environment variables for bash dev environments
# (Linux/macOS/WSL; mirrors config/env.local.ps1 and the V1 spec Section 18.1,
# with the M0 repo-root principle and the M1 strictness: Qwen3-ASR-0.6B).
#
# M0 principle (spec Section 26): the REPO ROOT is the single operating unit -
# Root is derived from this script's own location (config/ -> parent = repo
# root), so the script is machine-independent.
#
# Usage (source it so the variables land in the current shell):
#     source config/env.local.sh
#
# NOTE: do NOT source this before downloading models - HF_HUB_OFFLINE=1 blocks
# downloads. Order: scripts/download_models.ps1 (online) -> source this file
# (offline mode).
#
# The OPENAI_* values are DUMMY values: the official `openai` python lib
# requires them (used by the voicemem package); there is NO real API key -
# the endpoint is the local llama-server (127.0.0.1:8080).

# Repo root = parent of the directory holding this script.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Root="$(dirname "$_SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# HF cache isolation (M0): the HuggingFace cache goes into the project
# (models/hf), NOT the user profile (~/.cache/huggingface).
# ---------------------------------------------------------------------------
export HF_HOME="$Root/models/hf"                    # HF cache root
export HF_HUB_CACHE="$Root/models/hf/hub"           # hub snapshots/blobs
export TRANSFORMERS_CACHE="$Root/models/hf/transformers"

# ---------------------------------------------------------------------------
# Offline mode (spec 11.3 / 18.1) - set AFTER the downloads.
# ---------------------------------------------------------------------------
export HF_HUB_OFFLINE=1               # huggingface_hub: no network access
export TRANSFORMERS_OFFLINE=1         # transformers: no runtime model download
export TOKENIZERS_PARALLELISM=false   # tokenizer warning suppression

# ---------------------------------------------------------------------------
# VoiceMem configuration (consumed by the voicemem package AND AgentConfig)
# ---------------------------------------------------------------------------
# VOICEMEM_HOME overrides the project root (OPTIONAL - the default already
# IS the repo root; only set it if you really want to relocate):
# export VOICEMEM_HOME="$Root"
export VOICEMEM_MEMORY_ROOT="$Root/memory"          # memory root (sqlite/qdrant/backups)
export VOICEMEM_EMBED_DIM=384                       # local E5 (multilingual-e5-small, 384-d)
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"   # llama.cpp llama-server (loopback)
export OPENAI_API_KEY="not-needed-but-required-by-openai-lib"  # dummy, NO real key
export OPENAI_MODEL="qwen3.6-35b-a3b"  # v0.4.16: az EGYETLEN LLM = Qwen3.6 35B A3B IQ4_XS (l. env.local.ps1 LLM-MODELPROFIL)
export TTS_BACKEND="local"                          # Piper instead of OpenAI TTS

# ---------------------------------------------------------------------------
# Model path overrides (OPTIONAL - the AgentConfig property defaults already
# point at the M0 Section 7 layout):
# models/asr/qwen3-asr-0.6b, models/tts/piper, models/vad/silero-vad,
# models/llm/qwen3.6-35b-a3b, models/embedding/multilingual-e5-small
# ---------------------------------------------------------------------------
# export QWEN3_ASR_MODEL_PATH="$Root/models/asr/qwen3-asr-0.6b"
# export PIPER_VOICES_PATH="$Root/models/tts/piper"
# export SILERO_VAD_PATH="$Root/models/vad/silero-vad/silero_vad.onnx"
# export EMBEDDING_MODEL_PATH="$Root/models/embedding/multilingual-e5-small"
# M2 (emotion2vec) - EXCLUDED in M0/M1, FORBIDDEN to enable:
# export EMOTION2VEC_PATH="$Root/models/emotion"

# ---------------------------------------------------------------------------
# llama.cpp server settings (mirrors scripts/start_llama_server.ps1)
# v0.4.16: PORTABLE MODEL PATH - the GGUF may live on ANY drive. Select the
# actual file with the web UI "LLM model" section (Browse...) -> the
# absolute path lands in config/llm_model.json (that override has priority
# over this line), or run scripts/find_qwen_gguf.ps1 -SetEnv (rewrites the
# ps1 line). The GGUF is never copied into the project.
# ---------------------------------------------------------------------------
export LLAMA_SERVER_HOST=127.0.0.1
export LLAMA_SERVER_PORT=8080
# v0.4.17: the phantom project-relative default is COMMENTED OUT (it did
# not exist on the target machine - the cause of the v0.4.15 field
# "GGUF model not found" failure). Configure via the web UI picker /
# identify_ollama_blob.ps1 -Select (config/llm_model.json) or un-comment
# this line with a REAL absolute path:
# export LLAMA_MODEL_PATH="D:/AI/Models/Qwen3.6-35B-A3B-IQ4_XS.gguf"
export LLAMA_CONTEXT_SIZE=8192
export LLAMA_N_GPU_LAYERS=26   # partial offload: ~19 GB model > 12 GB VRAM
export LLM_DISABLE_THINKING=1  # hybrid-reasoning: thinking channel OFF
export LLAMA_CACHE_TYPE_K=q8_0
export LLAMA_CACHE_TYPE_V=q8_0

# ---------------------------------------------------------------------------
# v0.4.16/v0.4.17 LLM-MODELPROFIL (SINGLE profile: Qwen3.6 35B A3B IQ4_XS;
# mirror of config/env.local.ps1). There is NO fallback profile and NO
# second model: Qwen3.6 35B A3B IQ4_XS is the ONE AND ONLY production LLM
# (tested source: hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS).
# v0.4.17: the model may be an OLLAMA CONTENT-ADDRESSED BLOB
# (sha256-... in the Ollama blobs dir) - llama.cpp checks the GGUF magic,
# never the file name, so the blob loads IN PLACE (no copy, no rename);
# identify the exact blob with scripts/identify_ollama_blob.ps1.
# Nothing configured = NO default (the starter fails loudly).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------------------------
# Qdrant embedded (local storage used by the memory - a directory, not a server)
# ---------------------------------------------------------------------------
export QDRANT_PATH="$Root/memory/qdrant"

# ---------------------------------------------------------------------------
# Telemetry OFF (HF / mem0 / general) - nothing may leave the machine
# ---------------------------------------------------------------------------
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export MEM0_ANONYMIZED_TELEMETRY=false
export MEM0_TELEMETRY=false

# M2 preparation (FunASR cache) - NOT used in M0/M1, kept commented:
# export FUNASR_CACHE_DIR="$Root/.funasr_cache"

echo "env.local.sh loaded. Root=$Root"
echo "HF cache isolated: HF_HOME=$Root/models/hf"
echo "Offline mode ON (HF_HUB_OFFLINE=1) - start a fresh shell without sourcing this to download models."
