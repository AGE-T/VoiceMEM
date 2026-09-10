#!/usr/bin/env python3
"""Sandbox replica of scripts/build_release.ps1 (Linux-side release builder).

Mirrors the PowerShell builder step-for-step:
  1. (gate) the full unittest suite must have passed -> releases are only
     built from a green tree (the caller runs the gate; this script records
     the numbers it is given).
  2. staged CHANGELOG entry (the ZIP carries its own version's entry).
  3. version sync is already done (VERSION/pyproject/yaml).
  4. staging: allowlist copy of app/ config/ scripts/ tests/ web/ + root
     files + M0 placeholder dirs; every staged *.bat normalized to CRLF.
  5. BUILD_INFO.json into the stage root.
  6. ZIP with a single top-level folder VoiceMemAgent_v<version>/.
  6b. SELF-CHECK: the finished ZIP is opened and verified (placeholder
      dirs, own CHANGELOG entry, cumulative v0.3.5/v0.3.6 markers, and the
      new v0.4.1 markers). A failed self-check DELETES the zip - no false
      success is ever published.
  7. sha256 sidecar file.
  8. RELEASE_INDEX.json + repo CHANGELOG.md + BUILD_HISTORY.json updated.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RELEASES = REPO / "releases"

NEW_VERSION = "0.5.2"
PREV_VERSION = "0.5.1"
ZIP_NAME = f"VoiceMemAgent_v{NEW_VERSION}.zip"

PLACEHOLDER_DIRS = [
    "models/asr/qwen3-asr-0.6b",
    "models/llm/qwen3.6-35b-a3b",   # v0.4.14: the ONE AND ONLY LLM dir (README + .gitkeep; the GGUF is operator-placed, any drive - v0.4.16 picker)
    "models/tts/piper",
    "models/vad/silero-vad",
    "models/embedding/multilingual-e5-small",
    "models/hf",
    "models/emotion",
    "models/speaker",
    "memory/sqlite",
    "memory/qdrant",
    "memory/backups",
]

ROOT_FILES = [
    "README.md", "CHANGELOG.md", "LICENSES.md", "CONTRACT.md", "VERSION",
    "INSTALL_MANIFEST.example.json", "requirements.txt", "requirements.lock",
    "pyproject.toml", ".gitignore", "START.bat", "MODELS.lock.json",
    # v0.5.0: the controlled VoiceMem ownership artifacts ship in the ZIP
    "VOICEMEM_PIN.json", "UPSTREAM_POLICY.md",
]

NOTES = (
    "v0.5.2 TASK 1.5 MEMORY SAFETY GATE (the memory engine can no longer "
    "silently destroy information): three controlled-fork patches close the "
    "two P0 findings of the independent Claude audit (VoiceMEM_Audit_v1.0, "
    "audited v0.5.0 @ 57fb6a7) plus the graph-orphan P2. VM-LOCAL-007 (CD-1, "
    "P0): right-brain run_cleanup DELETE - the LLM heartnote-cleanup decision "
    "on the Ingest hot path - now routes through _execute_heartnote_deletes "
    "with the SAME opt-in gate as the left brain (VOICEMEM_ALLOW_MEMORY_ "
    "DELETE=1, DEFAULT DENY, loud log; the supersede branch stays "
    "non-destructive). VM-LOCAL-008 (CD-2, P0): UPDATE is NON-DESTRUCTIVE "
    "append + explicit supersession - the new observation becomes a NEW "
    "mem0 row (supersedes + session/event-date provenance), the old row is "
    "never text-rewritten (only gains superseded_by/superseded_at metadata "
    "via mem0's merge-update, text and created_at preserved), the JSON "
    "mirror appends instead of overwriting, cognitive annotation targets "
    "the NEW id, search marks hits with superseded_by and ranks the current "
    "value first, and _dedupe_near never collapses an explicit supersession "
    "pair - a previous fact can never become unrecoverable just because an "
    "LLM decided UPDATE. VM-LOCAL-009 (F-H): a permitted left-brain delete "
    "cascades memories/entity_memory_links/memory_tags/graph_entity_memories "
    "(FK-safe order) so no stale graph truth survives. BEHAVIOURAL REGRESSION "
    "BATTERY tests/integration/test_memory_safety.py (15 tests, REAL vendor "
    "+ REAL mem0/qdrant/E5 store, deterministic mock LLM for the DECISIONS "
    "only): RB gate default-deny/=0-deny/=1-delete-with-anchor-cascade; the "
    "restaurant A->B UPDATE scenario (old text survives, supersession "
    "explicit both directions, provenance on both rows, mirror lossless, "
    "current-first ranking + historical hit recoverable, both graph rows); "
    "LB gate default-deny-non-destructive/=1-full-cascade-no-orphans; "
    "production import resolution + PYTHONPATH shadow DETECTED by the pin "
    "check. Gate env upgrade: the release gate now runs the torch-equipped "
    "profile with the production offline env (HF_HUB_OFFLINE=1, like "
    "env.local) per package + chunked (memory-capped sandbox); two "
    "env-profile-dependent tests gained honest scope guards (ASR "
    "dependency-free hint test skips when deps present; the warm-up "
    "degradation test gets a deterministic broken-local-ASR fixture) and "
    "the pre-existing v0.5.1 PAGE_VERSION staleness (page stayed 0.5.0) is "
    "fixed (now 0.5.2, test green again). CD-6 mirror source tree: the "
    "GitHub mirror gains the full first-party source tree under source/ "
    "(additive, zips preserved) via the extended standing sync policy. NO "
    "retrieval redesign, NO recency/temporal/confidence changes, NO "
    "bi-temporal model, NO learning layer, ASR->LLM chain/E5/Piper/"
    "llama-server/UI untouched - TASK 2 (rich retrieval) starts only after "
    "this gate."
)
#: v0.4.3-era retired-model markers were REMOVED in v0.4.16 (single-model policy;
#: the self-check now asserts their ABSENCE instead - see V0416_ABSENT_FILES).

#: v0.4.4 field-fix markers: the thinking suppression (all layers), the
#: transformers floor + installer guard, and the E5 local pin.
V044_MARKERS = {
    "app/llm.py": [
        "_thinking_control_kwargs",
        "parse_sse_reasoning_delta",
        '"reasoning_effort": "none"',
        "LLM reply was empty",
    ],
    "app/web_server.py": [
        "_verify_llm_reply",
        "reply verified",
        "LLM startup probe",
    ],
    "app/voicemem_bridge.py": [
        "pin_e5_local_model",
        "VOICEMEM_E5_MODEL",
    ],
    "app/config.py": [
        "llm_disable_thinking",
    ],
    "scripts/start_llama_server.ps1": [
        '"--reasoning", "off"',
    ],
    "scripts/install_m1.ps1": [
        "$TransformersFloor",
        "v >= (5, 0)",
    ],
    "scripts/verify_m1.ps1": [
        '"chat_template_kwargs":{"enable_thinking":false}',
        '"reasoning_effort":"none"',
    ],
    "config/env.local.ps1": [
        "VOICEMEM_E5_MODEL",
    ],
    "config/.env.example": [
        "VOICEMEM_E5_MODEL",
        "LLM_DISABLE_THINKING",
    ],
    "requirements.txt": [
        "transformers>=5.0",
    ],
    "config/voicemem_config.yaml": [
        "llm_disable_thinking: true",
    ],
}

V044_REQUIRED_FILES = [
    "tests/unit/test_llm_reasoning.py",
    "tests/unit/test_e5_local_model.py",
    "tests/unit/test_web_llm_probe.py",
    "tests/validation/test_feature_thinking.py",
]

#: v0.4.5 markers: the version-aware bootstrap probe (the ASR self-heal),
#: the prompt-budget guard, and the actionable ASR repair hint.
V045_MARKERS = {
    "scripts/bootstrap.ps1": [
        "$TfProbe",
        "transformers.__version__",
        "(5, 0)",
        "else 5",
        "Qwen3-ASR",
    ],
    "scripts/install_m1.ps1": [
        "22 lepes, idempotens",
    ],
    "app/web_server.py": [
        "_cap_memory_context",
        "_fit_history_budget",
        "_PROMPT_CHAR_BUDGET",
        "_MEMORY_CONTEXT_MAX_CHARS",
    ],
    "app/asr.py": [
        "START.bat",
    ],
}

V045_REQUIRED_FILES = [
    "tests/unit/test_prompt_budget.py",
]

#: v0.4.6 markers: the British-English UI (with the zh locale really gone),
#: and the headline bug fixes from the two-agent code review.
V046_MARKERS = {
    "web/voicemem.html": [
        "en-GB",
        "searchChats",
        "KIND_LABEL",
    ],
    "app/web_server.py": [
        "llm_probe_failed",
    ],
    "app/speaker.py": [
        "embed_windows",
    ],
    "app/tts.py": [
        "stop raced with spawn",
    ],
    "app/main.py": [
        "forced_end",
    ],
    "tests/validation/test_feature_release.py": [
        "_latest_entry",
    ],
}

V046_REQUIRED_FILES: list[str] = []

#: v0.4.7 markers: the official processor+generate ASR call path, the
#: honest warm-up chip, the mic/VAD observability lines, the Chinese-slot
#: display translation, and the vendor prompt patch + memory migration.
V047_MARKERS = {
    "app/asr.py": [
        "apply_transcription_request",
        "_parse_asr_output",
        "_import_model_class",
        "Qwen3ASRForConditionalGeneration",
        "probe failed",
    ],
    "app/web_server.py": [
        "localise_slot",
        "_SLOT_EN",
        "mic uplink started",
        "VAD never reached the speech threshold",
        "ASR warm-up failed",
    ],
    "app/config.py": [
        "asr_language",
        "asr_max_new_tokens",
    ],
    "scripts/install_m1.ps1": [
        "patch_voicemem_english.py",
        '$TransformersFloor = "5.0"',
    ],
    "scripts/bootstrap.ps1": [
        "patch_voicemem_english.py",
        "v >= (5, 0)",
    ],
    "scripts/start_agent.ps1": [
        "localise_memories.py",
    ],
    "requirements.txt": [
        "transformers>=5.0",
    ],
    "pyproject.toml": [
        "transformers>=5.0",
    ],
}

V047_REQUIRED_FILES = [
    "scripts/patch_voicemem_english.py",
    "scripts/localise_memories.py",
    "tests/unit/test_asr_parse.py",
]

#: v0.4.8 markers: the mic -> ASR one-shot test endpoint + UI panel, the
#: always-logging VAD peak report, and the live session fields.
V048_MARKERS = {
    "app/web_server.py": [
        "/api/asr-test",
        "_run_asr_test",
        "transcribe_utterance",
        "asr_test_last.wav",
        "mic uplink carries silence",
        "vad_peak",
    ],
    "app/asr.py": [
        "transcribe_utterance",
        "ASR one-shot",
    ],
    "web/voicemem.html": [
        "asrTestBtn",
        "asrTestOut",
        "asrTestVerdict",
        "asr-test/audio",
        "asrTestVerdictOkNoVad",
    ],
}

V048_REQUIRED_FILES = [
    "tests/unit/test_web_asr_test.py",
]

#: v0.4.10 markers: the original-repo ASR repair (feature extractor swap +
#: thinker config + key remap) and the browser capture fixes.
V0410_MARKERS = {
    "app/asr.py": [
        "_parse_thinker_layout",
        "_repaired_qwen_config",
        "_ORIGINAL_QWEN_KEY_MAPPING",
        "key_mapping=dict(_ORIGINAL_QWEN_KEY_MAPPING)",
        "_ensure_qwen3_feature_extractor",
        "_qwen3_feature_extractor_class",
        "_resolve_n_window",
        "low_cpu_mem_usage=True",
    ],
    "web/voicemem.html": [
        "noiseSuppression:false",
        "anti-aliased downsampler",
    ],
}

V0410_REQUIRED_FILES = [
    "tests/unit/test_asr_original_repo.py",
]

#: v0.4.11 markers: the VAD sensitivity change must ship in BOTH config
#: layers (the YAML the real backend loads + the dataclass default the web
#: DEMO and the fallback path use) - and the docs/tests must follow.
V0411_MARKERS = {
    "config/voicemem_config.yaml": [
        "vad_threshold: 0.25",
        "v0.4.11: 0.5 -> 0.25",
    ],
    "app/config.py": [
        "vad_threshold: float = 0.25",
    ],
    "app/web_server.py": [
        "st.session[\"vad_threshold\"] = round(float(self.config.vad_threshold), 3)",
    ],
    "README.md": [
        "0,25 / 300 ms",
    ],
    "tests/validation/test_feature_vad.py": [
        "assertEqual(cfg.vad_threshold, 0.25)",
    ],
    "tests/validation/test_feature_config.py": [
        "assertEqual(cfg.vad_threshold, 0.25)",
    ],
    "tests/unit/test_web_asr_test.py": [
        'assertEqual(result["vad"]["threshold"], 0.25)',
    ],
}

#: v0.4.12 markers: stale-page detection + raw capture + send-as-turn.
V0412_MARKERS = {
    "web/voicemem.html": [
        "const PAGE_VERSION='0.4.14';",
        "uiVerStale",
        "id=\"rawCapChk\"",
        "?{echoCancellation:false,noiseSuppression:false,autoGainControl:false}",
        "asrTestSendRow.hidden=!txt",
        "{lang:'en',page:PAGE_VERSION}",
        "asrTestVerdictDeaf",
    ],
    "app/web_server.py": [
        "web UI page version",
        "STALE PAGE, reload needed",
    ],
    "tests/unit/test_web_ui_v0412.py": [
        "PAGE_VERSION",
        "rawCapChk",
        "asrTestSend",
        "asrTestVerdictDeaf",
    ],
}

#: v0.4.13 markers: emotion-init timeout + full turn stage trail + load lock.
V0413_MARKERS = {
    "app/web_server.py": [
        "_EMOTION_INIT_TIMEOUT_S = 5.0",
        "asyncio.wait_for(",
        "emotion done (init timed out",
        "turn received (source=",
        "memory start",
        "llm start (",
        "llm first token (",
        "tts start (chunk 1",
        "tts done (",
    ],
    "app/emotion.py": [
        "_load_lock",
    ],
    "config/env.local.ps1": [
        "LLM-MODELPROFIL",
        "qwen3.6-35b-a3b",
    ],
    "web/voicemem.html": [
        "slice(-24)",
    ],
    "tests/unit/test_web_turn_stages.py": [
        "STAGES",
        "init timed out",
    ],
    "tests/integration/test_web_e2e.py": [
        "test_websocket_mic_turn_with_hung_emotion_still_answers",
    ],
}

#: v0.4.14 markers: energy-fallback VAD + automatic mic dispatch + Qwen profile.
V0414_MARKERS = {
    "app/vad.py": [
        "class FusedVad:",
        "fallback_active",
        "primary_peak",
        "last_primary",
    ],
    "app/web_server.py": [
        "FusedVad(silero, self.config)",
        "ASR turn dispatch (source=asr)",
        "ASR flush start",
        "ASR final transcript:",
        "mic frame received",
        "ASR feed (frames flowing",
        "del events[:-48]",
        "vad_energy_fallback",
        "last_primary",
    ],
    "app/config.py": [
        "vad_energy_fallback: bool = True",
        "qwen3.6-35b-a3b",
    ],
    "config/env.local.ps1": [
        'OPENAI_MODEL = "qwen3.6-35b-a3b"',
        "Qwen3.6-35B-A3B-IQ4_XS.gguf",
        'LLAMA_N_GPU_LAYERS = "26"',
        'LLM_DISABLE_THINKING = "1"',
        "EGYETLEN profil",   # v0.4.16: single-profile wording (was: VISSZAILLESZTETT profil)
    ],
    "app/teacher_persona.py": [
        "natural, friendly voice assistant",
    ],
    "web/voicemem.html": [
        "const PAGE_VERSION=",   # page version follows the backend (stale-page detection sync); exact value pinned by V0416
        "energy-fallback gate",
        "asrTestVadSilero",
    ],
    "tests/unit/test_fused_vad.py": [
        "FusedVad",
        "DEAF primary",
    ],
    "tests/unit/test_web_mic_dispatch.py": [
        "ASR turn dispatch",
        "energy fallback",
    ],
    "tests/e2e_deaf_server.py": [
        "_DeafSilero",
    ],
    "models/llm/qwen3.6-35b-a3b/README.md": [
        "the ACTIVE runtime LLM profile",
    ],
}

#: v0.4.15 markers: the exact-Qwen-GGUF discovery gate (find_qwen_gguf.ps1)
#: + every wired pointer. The active profile itself is UNCHANGED (v0.4.14's
#: Qwen3.6 35B A3B IQ4_XS markers above still pin it).
V0415_MARKERS = {
    "scripts/find_qwen_gguf.ps1": [
        "EXACT Qwen3.6 35B A3B IQ4_XS GGUF locator",
        "hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS",
        "MODEL FILE MISSING",
        "Read-GgufIdentity",
        "general.file_type",
        "$Iq4XsFileType  = 30",   # v0.4.16: corrected (29 was IQ2_M)
        "-CopyToRepo",
        "-SetEnv",
        "Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf",
    ],
    "config/env.local.ps1": [
        "find_qwen_gguf.ps1",
    ],
    "config/env.local.sh": [
        "find_qwen_gguf.ps1",
    ],
    "scripts/start_llama_server.ps1": [
        "find_qwen_gguf.ps1",
    ],
    "scripts/verify_m1.ps1": [
        "find_qwen_gguf.ps1",
    ],
    "models/llm/qwen3.6-35b-a3b/README.md": [
        "find_qwen_gguf.ps1",
        "DO NOT GUESS THE FILENAME",
    ],
    "README.md": [
        "find_qwen_gguf.ps1",
    ],
    "web/voicemem.html": [
        "const PAGE_VERSION=",
    ],
}

#: v0.4.16 markers: the portable LLM model path (picker + persistence +
#: resolution chain + stale-server restart) and the single-model policy
#: (Gemma removed everywhere active).
V0416_MARKERS = {
    "app/gguf.py": [
        "GGUF_MAGIC = 0x46554747",
        "IQ4_XS_FILE_TYPE = 30",
        "def read_gguf_metadata",
        "def expected_model_check",
    ],
    "app/native_picker.py": [
        "GetOpenFileNameW",
        "GGUF_FILTER",
        "def pick_file",
        "def is_supported",
    ],
    "app/llm_model_settings.py": [
        "SETTINGS_RELPATH",
        "llm_model_path",
        "def user_selected_llm_model_path",
        "def resolve_llm_model_path",
        "def read_llama_server_loaded_model",
        "RESOLVED_MODEL_MARKER",
    ],
    "app/config.py": [
        "user_selected_llm_model_path",
        "the ONE AND ONLY production LLM",
    ],
    "app/web_server.py": [
        "api_llm_model",
        "api_llm_model_browse",
        "api_llm_model_inspect",
        "api_llm_model_select",
        "api_llm_model_reset",
        "apply_llm_model_selection",
        "reset_llm_model_selection",
        "picker_supported",
    ],
    "scripts/start_llama_server.ps1": [
        "llm_model.json",
        "llama-server.resolved-model.json",
        "Get-LlamaServedModel",
        "Stop-LlamaServerOnPort",
        "MODELLUT FELULIRAS",
    ],
    "scripts/verify_m1.ps1": [
        "llm_model.json",
        "user selection (config/llm_model.json)",
    ],
    "scripts/find_qwen_gguf.ps1": [
        "$Iq4XsFileType  = 30",
        "KEY LENGTH IS u64",
    ],
    "web/voicemem.html": [
        'const PAGE_VERSION=\'0.4.17\';',
        'id="llmBrowseBtn"',
        "llmLoadStatus",
        "/api/llm-model/select",
        "llmServingNote",
    ],
    "MODELS.lock.json": [
        "operator-placed",
        "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
    ],
    "scripts/download_models_hf.py": [
        "OPERATOR-PLACED",
    ],
    "models/llm/qwen3.6-35b-a3b/README.md": [
        "ONE AND ONLY runtime LLM",
        "PORTABLE MODEL PATH",
    ],
    "tests/unit/test_llm_model_settings.py": [
        "TestGgufReader",
        "TestLlmModelSettings",
        "TestLlmModelRest",
    ],
    "tests/validation/test_feature_llm_policy.py": [
        "SingleLlmPolicyFeatureTest",
        "test_logic_no_gemma_fallback_on_active_surfaces",
    ],
}

#: v0.4.16: files that must NOT ship in the ZIP (the Gemma scripts were
#: deleted; runtime state must never ship).
V0416_ABSENT_FILES = [
    "scripts/smoke_test_gemma.py",
    "scripts/quality_test_gemma.py",
    "config/llm_model.json",
    "models/llm/gemma-4-12b/README.md",
]

#: v0.4.17 markers: Ollama blob identification + direct llama-server load
#: proofs + the honest no-phantom-default model state.
V0417_MARKERS = {
    "scripts/identify_ollama_blob.ps1": [
        "[string]$OllamaRoot",
        "OLLAMA_MODELS",
        "manifests",
        "sha256-",
        "general.split.no",
        "Get-FileHash",
        "llm_model.json",
        "llama-server",
        '30 = "IQ4_XS"',
        "Status       : File found",
    ],
    "app/gguf.py": [
        "SPLIT_NO_KEY",
        "general.split.count",
        "is_split_shard",
        "def llama_server_load_verdict",
        "ZERO-indexed",
    ],
    "app/llm_model_settings.py": [
        "SOURCE_NONE",
        "OPERATOR_LLM_DIR",
        "def operator_default_candidates",
    ],
    "app/config.py": [
        "def llm_model_file",
        "NO phantom default",
    ],
    "app/native_picker.py": [
        "Ollama blobs",
        "sha256-*",
    ],
    "app/web_server.py": [
        "split_shard",
        "llama_server_load_verdict",
        "log_path_found",
        "completion_ok",
        "served_model_id",
    ],
    "scripts/start_llama_server.ps1": [
        "Invoke-LlamaLoadProof",
        "Get-LlamaLoadLogLine",
        "Invoke-LlamaCompletionProbe",
        "Write-LlamaResolvedMarker",
        "Select-String",
        "chat/completions",
        "identify_ollama_blob.ps1",
        "nincs konfiguralva LLM modell",
    ],
    "scripts/verify_m1.ps1": [
        "Get-OperatorDefaultGguf",
        "identify_ollama_blob.ps1",
        "blob-stem",
        "LLM load proof",
        "Selected LLM : Qwen3.6 35B A3B IQ4_XS",
    ],
    "scripts/find_qwen_gguf.ps1": [
        "INSERTED after OPENAI_MODEL",
    ],
    "config/env.local.ps1": [
        "KIKOMMENTEZVE",
    ],
    "config/env.local.sh": [
        "COMMENTED OUT",
    ],
    "web/voicemem.html": [
        "llmStatusNone",
        "llmStatusShard",
        "llmLoadableYes",
        "llmServedId",
        "llmLogProof",
        "llmCompletionOk",
        "llmPickOllamaHint",
        "split_shard",
    ],
    "models/llm/qwen3.6-35b-a3b/README.md": [
        "OLLAMA BLOBS LOAD DIRECTLY",
        "identify_ollama_blob.ps1",
        "general.split.no",
    ],
    "tests/unit/test_llm_model_settings.py": [
        "TestGgufSplitShardDetection",
        "test_operator_dir_ambiguity_is_not_a_default",
        "test_config_llm_model_file_operator_default",
    ],
    "tests/validation/test_feature_llm_policy.py": [
        "test_logic_identify_script_exists_and_is_ollama_aware",
        "test_logic_identify_script_never_copies_the_blob",
        "test_logic_app_code_has_no_hardcoded_ai_home",
        "test_logic_starter_records_runtime_load_proof",
        "test_logic_picker_lists_ollama_blobs",
    ],
}


#: v0.4.18 markers: the GGUF metadata VALUE TYPE table fix (the v0.4.16/17
#: walkers numbered 8=BOOL/9=STRING/10=ARRAY/7=FLOAT64/13=FLOAT16; the real
#: spec is 7=BOOL/8=STRING/9=ARRAY/10=UINT64/11=INT64/12=FLOAT64, no type 13).
V0418_MARKERS = {
    "scripts/identify_ollama_blob.ps1": [
        "llama-server load verdict (v0.4.18)",
        "v0.4.18 FIX (field report 2026-09)",
        "$GgufScalar",
        "function Read-GgufString",
        "function Skip-GgufValue",
        "GGUFValueType",
        "there is NO type 13",
        "# STRING: capture for identity keys",
        "# UINT32: file_type / split keys",
        "absurd GGUF string length",
        "absurd GGUF array count",
    ],
    "scripts/find_qwen_gguf.ps1": [
        "v0.4.18 FIX",
        "$GgufScalar",
        "function Skip-GgufValue",
        "there is NO type 13",
        "# STRING: capture for identity keys",
    ],
    "app/gguf.py": [
        "v0.4.18 PARSER FIX",
        "_SCALAR_SIZES",
        "_SCALAR_FMTS",
        "_skip_array",
        "_MAX_ARRAY_COUNT",
        "there is NO type 13",
    ],
    "tests/unit/test_llm_model_settings.py": [
        "the type ids below follow the ACTUAL GGUF spec table",
        "def kv_u64",
        "def kv_f32",
        "def kv_u16",
    ],
}

V0419_MARKERS = {
    "web/voicemem.html": [
        "v0.4.19",
        "MIC_SILENT_PEAK",
        "function acquireMic(reason,minimal)",
        "function measureAudioFrame(e,st,withPcm,ctxRate)",
        "stale deviceId",
        "startMicCapture(true)",
        "function maybeSendMicDiag()",
        "type:'mic_diag'",
        "type:'mic_probe'",
        'id="micProbeBtn"',
        'id="micDiag"',
        "PAGE_VERSION='0.5.2'",
    ],
    "app/web_server.py": [
        "def _mic_probe_core(",
        "def _log_mic_probe(",
        "def _log_mic_diag(",
        "async def _handle_mic_probe(",
        'msg_type == "mic_diag"',
        'msg_type == "mic_probe"',
        '"/api/mic-probe"',
        '"/api/mic-diag"',
        "browser src RMS %s / browser PCM RMS %s /",
        "browser mic diag: frames %s, src rms %s",
        "browser mic capture: device %r",
    ],
    "tests/unit/test_web_mic_probe.py": [
        "the minimal diagnostic path",
        "browser_source_silent",
        "browser_conversion_silent",
        "transport_silent",
        "test_capture_probe_button_and_panel_exist",
        "test_watchdog_and_stale_device_guard_exist",
    ],
}

#: v0.4.20 markers: the A/B capture probe — both phases (constrained vs
#: minimal {audio:true}), the explicit Q1 answer, the comparison verdict,
#: the deviceId validity row and the backend label passthrough.
V0420_MARKERS = {
    "web/voicemem.html": [
        "v0.4.20",
        "function probePhase(tag,minimal,rateInfo)",
        "function renderABProbeResult(A,B,rateInfo)",
        "A=await probePhase('A',false,rateInfo)",
        "B=await probePhase('B',true,rateInfo)",
        "micProbeRecordingA",
        "micProbeRecordingB",
        "micProbeQ1",
        "micProbeABVerdictAZero",
        "micProbeABVerdictBothSilent",
        "micProbeRateRow",
        "device_id_status",
        "label:label",
        ".probe .ph",
    ],
    "app/web_server.py": [
        'label: str = ""',
        '"label": label[:40]',
        'mic-probe%s: browser src RMS',
        'label=str(data.get("label", "") or "")[:40]',
        'label=str(body.get("label", "") or "")[:40]',
    ],
    "tests/unit/test_web_mic_probe.py": [
        "test_ab_probe_exists",
        "test_label_lands_in_result_and_log_line",
        "test_http_probe_carries_the_ab_label",
        "test_mic_probe_frame_carries_the_ab_label",
        "PAGE_VERSION='0.5.2'",
    ],
}

#: v0.4.21 markers: the explicit no-thinking production alignment — both
#: layers (server --reasoning off + request-level switches) at every
#: request-construction site, the benchmark's no-thinking verification, the
#: pinned server build whose reasoning API was verified live, and the
#: no-Ollama-runtime contract.
V0421_MARKERS = {
    "scripts/start_llama_server.ps1": [
        '"--reasoning", "off"',
    ],
    "MODELS.lock.json": [
        '"tag": "b10717"',
    ],
    "scripts/localise_memories.py": [
        '"chat_template_kwargs": {"enable_thinking": False}',
        '"reasoning_effort": "none"',
    ],
    "scripts/smoke_test_web.py": [
        '"chat_template_kwargs": {"enable_thinking": False}',
        '"reasoning_effort": "none"',
        "reasoning_content",
        "and not reasoning",
    ],
    "tests/benchmark/llm_benchmark.py": [
        "THINKING_TAG_MARKERS",
        "reply_has_thinking_block",
        "reasoning_eaten_errors",
        "empty_replies",
        "first_token_s",
        "_capture_request",
        "NO-THINKING VERDICT",
    ],
    "tests/validation/test_feature_thinking.py": [
        "ThinkingAlignmentV0421ContractTests",
        "test_localise_memories_carries_the_switches",
        "test_smoke_test_round_trip_mirrors_production",
        "test_benchmark_verifies_the_no_thinking_profile",
        "test_no_ollama_runtime_dependency",
    ],
}


#: v0.5.0 markers: the controlled VoiceMem ownership model.
V052_MARKERS = {
    "vendor/voicemem/voicemem/rightbrain/brain.py": [
        "VM-LOCAL-007",
        "_execute_heartnote_deletes",
        "BLOCKED by the controlled-fork",
    ],
    "vendor/voicemem/voicemem/leftbrain/mem0_backend_store.py": [
        "VM-LOCAL-008",
        "UPDATE is NON-DESTRUCTIVE",
        "superseded_by",
    ],
    "vendor/voicemem/voicemem/leftbrain/memory_repository.py": [
        "VM-LOCAL-009",
        "supersedes",
    ],
    "vendor/voicemem/voicemem/leftbrain/cognitive_graph/store.py": [
        "delete_memory_record",
    ],
    "vendor/voicemem/voicemem/__init__.py": [
        "VM-LOCAL-007",
        "VM-LOCAL-008",
        "VM-LOCAL-009",
    ],
    "VOICEMEM_PIN.json": [
        "VM-LOCAL-007",
        "VM-LOCAL-008",
        "VM-LOCAL-009",
    ],
    "tests/integration/test_memory_safety.py": [
        "CD-1 REGRESSION",
        "RightBrainDeleteGateTests",
        "UpdateHistoryPreservationTests",
        "LeftBrainDeleteGateTests",
        "VendorImportResolutionTests",
    ],
    "web/voicemem.html": [
        "const PAGE_VERSION='0.5.2';",
    ],
}

V050_MARKERS = {
    "VOICEMEM_PIN.json": [
        "controlled-fork",
        "e8384e087bd2f44eb05fc7ae1a3c525ea8244179",
        "VM-LOCAL-001",
        "VM-LOCAL-005",
        "UPSTREAM-961efe8",
    ],
    "UPSTREAM_POLICY.md": [
        "CONTROLLED FORK",
        "Local E5 is the only memory embedder",
        "Adoption review procedure",
    ],
    "vendor/voicemem/voicemem/__init__.py": [
        "CONTROLLED_FORK = True",
        "CONTROLLED_UPSTREAM_COMMIT",
        "CONTROLLED_PATCHES",
    ],
    "vendor/voicemem/voicemem/orchestrator.py": [
        "_LOCAL_E5_CACHE_KEY",
        "VM-LOCAL-001",
    ],
    "vendor/voicemem/voicemem/leftbrain/brain.py": [
        "VM-LOCAL-002",
    ],
    "vendor/voicemem/voicemem/utils/defaults.py": [
        "VM-LOCAL-003",
    ],
    "vendor/voicemem/voicemem/rightbrain/traits_store.py": [
        "_WARNED_VEC",
        "rb_traits embedding FAILED",
        "last_query_dim",
    ],
    "vendor/voicemem/voicemem/leftbrain/mem0_backend_store.py": [
        "VOICEMEM_ALLOW_MEMORY_DELETE",
        "_as_date",
    ],
    "scripts/install_m1.ps1": [
        "VEZERELT FORRAS",
        "VOICEMEM_PIN.json",
        "CONTROLLED_UPSTREAM_COMMIT",
        "e8384e087bd2f44eb05fc7ae1a3c525ea8244179",
    ],
    "scripts/verify_m1.ps1": [
        "CONTROLLED_UPSTREAM_COMMIT",
        "vezerlt forras + pin",
    ],
    "scripts/verify_voicemem_pin.py": [
        "EMBEDDING-REGRESSION-OK",
        "EXPECTED_COMMIT",
    ],
    "scripts/assess_trait_embeddings.py": [
        "mode=ro",
        "DRY RUN ONLY",
    ],
    "scripts/write_install_manifest.py": [
        "_voicemem_pin",
        "pin_verified",
        "runtime_import_path",
    ],
    "app/voicemem_bridge.py": [
        "def embedding_factory():",
        "embedding=embedding_factory",
    ],
    "tests/unit/test_voicemem_controlled.py": [
        "PinFileContractTests",
        "VendorPatchMarkerTests",
        "DeploymentChainSourceTests",
        "SubprocessRuntimeProbeTests",
        "ManifestVoicememBlockTests",
        "AssessTraitEmbeddingsTests",
    ],
}

#: v0.4.9 markers: the ten code-analysis-report fixes (P0 #1-5 + P1 #6-10).
V049_MARKERS = {
    "app/pipeline.py": [
        "_commit_reply_with_retry",
        "retry_pending_commits",
        "memory_status",
        "fit_prompt_budget",
        "_pending_commits",
    ],
    "app/asr.py": [
        "with _BACKEND_CACHE_LOCK, guard:",
    ],
    "app/voicemem_bridge.py": [
        "verify_e5_local_model",
        "COMMIT_COMMITTED",
        "_may_retry_vm",
        "_prune_vm_cache",
        "_VM_CACHE_MAX",
        "fit_prompt_budget",
    ],
    "app/audio_io.py": [
        "_open_stream",
        "_mic_open_error_hint",
        "falling back to the SYSTEM DEFAULT input device",
    ],
    "app/llm.py": [
        "sse_lines",
        "first_line_s",
        "NO data ",
    ],
    "app/vad.py": [
        "flush_prob",
        "_frame_buf",
    ],
    "app/web_server.py": [
        "_cancel_session_tasks",
        "_session_tasks",
        "self._spawn",
    ],
    "app/config.py": [
        "_parse_env_int",
        "_parse_env_float",
        "AUDIO_INPUT_DEVICE",
    ],
    "app/teacher_persona.py": [
        "fit_prompt_budget",
        "_PROMPT_CHAR_BUDGET",
        "estimate_prompt_chars",
    ],
}

V041_HTML_MARKERS = [
    "asr_empty",
    "vad_level",
    "mic frames",
]

V041_SERVER_MARKERS = [
    "start_warmup",
    "web-server.log",
    "vad_level",
    "is_available",
    "_guarded_turn",
    "asr_empty",
]

V042_HTML_MARKERS = [
    "voiceModeSel",
    "voiceHuSel",
    "voiceEnSel",
    "voiceHuPrev",
    "voiceEnPrev",
    "voiceActive",
    "loadVoiceSettings",
    "previewVoice",
    "api/voice/preview",
]

V042_SERVER_MARKERS = [
    "VoiceSettings",
    "apply_voice_settings",
    "api/voice/preview",
    "PREVIEW_SENTENCES",
    "_MAX_UTTERANCE_MS",
    "last_empty_utterance.wav",
]

CUMULATIVE_MARKERS = {
    "web/voicemem.html": ["micSel", "pipeStrip", "XTransformPort", "asrBypassed", "SAMPLE_RATE=24000"],
    "app/web_server.py": ["WebSession", "user_text", "pipeline_status"],
    "scripts/smoke_test_web.py": ["smoke"],
    "START.bat": ["127.0.0.1:8787", "Unknown start mode"],
}


def write_text_no_bom(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def crlf_normalize_bat(path: Path) -> None:
    raw = path.read_bytes().decode("ascii", errors="replace")
    norm = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    path.write_bytes(norm.encode("ascii", errors="replace"))


def copy_tree(src: Path, dst: Path) -> None:
    """Allowlist copy mirroring robocopy /E /XD __pycache__ .pytest_cache
    /XF *.pyc *.pyo .env - no build junk ever enters the ZIP.

    v0.4.2: config/voice_settings.json is RUNTIME STATE (the user's persisted
    voice selection on their machine) - it must never ship inside a release.
    """
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache",
                                                  "*.pyc", "*.pyo", ".env",
                                                  "voice_settings.json",
                                                  "llm_model.json",
                                                  "*.egg-info", "build", "dist",
                                                  "voicemem_memory*", "results"))
    # v0.4.2 + v0.4.16: persisted RUNTIME STATE never ships in a release
    # (the user's selections survive an extract-over upgrade instead).
    for stale_name in ("voice_settings.json", "llm_model.json"):
        stale = dst / stale_name
        if stale.is_file():
            stale.unlink()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _strip_version_entry(text: str, version: str) -> str:
    """Remove an existing '## [<version>]' entry block (idempotent insert).

    The entry block runs from its '## [x.y.z]' heading to the next '## ['
    heading (or EOF). Rebuilding/re-stamping a version replaces the old
    entry instead of duplicating it.
    """
    heading = f"## [{version}]"
    idx = text.find(heading)
    if idx == -1:
        return text
    rest = text[idx + len(heading):]
    nxt = rest.find("\n## [")
    end = len(text) if nxt == -1 else idx + len(heading) + nxt + 1
    return text[:idx] + text[end:]


def build() -> int:
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stage_root = Path(tempfile.mkdtemp(prefix=f"voicemem_build_{NEW_VERSION}_"))
    stage = stage_root / f"VoiceMemAgent_v{NEW_VERSION}"
    stage.mkdir(parents=True, exist_ok=True)
    zip_path = RELEASES / ZIP_NAME

    # gate numbers (from the caller's green run)
    gate_total = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    deep_pass = sys.argv[2] if len(sys.argv) > 2 else "release, runtime-deps"
    deep_skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if gate_total <= 0:
        print("BUILD FAILED: no green test-gate numbers supplied "
              "(usage: build_release_sandbox.py <total_tests> <deep_pass> <deep_skip>)")
        return 1

    ch_entry = (
        f"## [{NEW_VERSION}] - {today}\n"
        f"### Kiadas (ZIP-build)\n"
        f"- {NOTES}\n"
        f"- Tesztkapu: PASS a build ELOTT (teljes tesztkeszlet: {gate_total} teszt; "
        f"deep validacio: {deep_pass}; SKIP: {deep_skip} db)\n"
        f"- Kiadas: ``releases/{ZIP_NAME}`` + ``{ZIP_NAME}.sha256`` (UTC: {built_at}).\n\n"
    )

    # ---------------- staging ----------------
    # v0.5.0: vendor/ (the controlled VoiceMem source) SHIPS in the ZIP —
    # the installer pip-installs from it instead of cloning upstream.
    # v0.5.2: docs/ (verification evidence + gate records) ship too (the
    # v0.5.1 zip set the precedent) — the release ZIP carries its own
    # evidence.
    stage_list = ["app", "config", "scripts", "tests", "web", "vendor"]
    if (REPO / "docs").is_dir():
        stage_list.insert(1, "docs")
    for d in stage_list:
        src = REPO / d
        if not src.is_dir():
            print(f"BUILD FAILED: staging source missing: {d}")
            return 1
        copy_tree(src, stage / d)
    for f in ROOT_FILES:
        src = REPO / f
        if src.is_file():
            shutil.copy2(src, stage / f)
        else:
            print(f"[warn] staging: {f} missing, skipped")
    for json_name in ("RELEASE_INDEX.json", "BUILD_HISTORY.json"):
        src = REPO / "releases" / json_name
        if src.is_file():
            (stage / "releases").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, stage / "releases" / json_name)

    for bat in stage.rglob("*.bat"):
        crlf_normalize_bat(bat)
    n_bats = len(list(stage.rglob("*.bat")))

    for d in PLACEHOLDER_DIRS:
        target = stage / d
        target.mkdir(parents=True, exist_ok=True)
        keep = target / ".gitkeep"
        if not keep.exists():
            keep.write_bytes(b"")
        repo_readme = REPO / d / "README.md"
        if repo_readme.is_file():
            shutil.copy2(repo_readme, target / "README.md")

    # staged CHANGELOG gets its own entry BEFORE zipping
    staged_ch = stage / "CHANGELOG.md"
    if not staged_ch.is_file():
        print("BUILD FAILED: staged CHANGELOG.md missing")
        return 1
    ch_text = staged_ch.read_text(encoding="utf-8")
    marker = "# Változatonapló (CHANGELOG) — voicemem-agent"
    # idempotent insert: a pre-added placeholder entry for THIS version is
    # REPLACED, never duplicated (rebuilds stamp the final numbers in place)
    ch_text = _strip_version_entry(ch_text, NEW_VERSION)
    if marker in ch_text:
        head, rest = ch_text.split(marker, 1)
        # insert the entry after the intro block (before the first existing '## [')
        idx = rest.find("## [")
        if idx == -1:
            rest = ch_entry + rest
        else:
            rest = rest[:idx] + ch_entry + rest[idx:]
        staged_ch.write_text(head + marker + rest, encoding="utf-8", newline="")
    else:
        staged_ch.write_text(ch_entry + ch_text, encoding="utf-8", newline="")

    n_files = sum(1 for p in stage.rglob("*") if p.is_file())

    # ---------------- BUILD_INFO ----------------
    build_info = {
        "schema_version": 1,
        "name": "voicemem-agent",
        "version": NEW_VERSION,
        "built_from_version": PREV_VERSION,
        "milestone": "M0",
        "built_at_utc": built_at,
        "git_commit": git_commit(),
        "generator": "sandbox replica of scripts/build_release.ps1",
        "test_gate": "unittest discover (unit + integration + benchmark + validation)",
        "test_gate_exit_code": 0,
        "validation_deep_pass": deep_pass,
        "validation_deep_skipped": deep_skip,
        "validation_strict": False,
        "notes": NOTES,
    }
    write_text_no_bom(stage / "BUILD_INFO.json", json.dumps(build_info, indent=2) + "\n")

    # ---------------- zip ----------------
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(stage_root).as_posix())

    # ---------------- self-check ----------------
    root_prefix = f"VoiceMemAgent_v{NEW_VERSION}/"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist()]
            for d in PLACEHOLDER_DIRS:
                if not any(n.startswith(root_prefix + d + "/") for n in names):
                    raise AssertionError(f"self-check: '{d}/' missing from the ZIP")
            ch_in_zip = zf.read(root_prefix + "CHANGELOG.md").decode("utf-8")
            if f"## [{NEW_VERSION}]" not in ch_in_zip:
                raise AssertionError("self-check: ZIP CHANGELOG lacks its own entry")
            # cumulative markers (v0.3.5 web payload, v0.3.6 CRLF START.bat)
            for rel, markers in CUMULATIVE_MARKERS.items():
                content = zf.read(root_prefix + rel)
                text = content.decode("utf-8", errors="replace")
                for m in markers:
                    if m not in text:
                        raise AssertionError(f"self-check: {rel} lacks marker {m!r}")
            start_bat = zf.read(root_prefix + "START.bat").decode("ascii", errors="replace")
            if "\r\n" not in start_bat:
                raise AssertionError("self-check: START.bat in the ZIP is not CRLF")
            # v0.4.1 markers
            html = zf.read(root_prefix + "web/voicemem.html").decode("utf-8", errors="replace")
            web_server = zf.read(root_prefix + "app/web_server.py").decode("utf-8", errors="replace")
            vad = zf.read(root_prefix + "app/vad.py").decode("utf-8", errors="replace")
            asr = zf.read(root_prefix + "app/asr.py").decode("utf-8", errors="replace")
            for m in V041_HTML_MARKERS:
                if m not in html:
                    raise AssertionError(f"self-check: voicemem.html lacks v0.4.1 marker {m!r}")
            for m in V041_SERVER_MARKERS:
                if m not in web_server:
                    raise AssertionError(f"self-check: web_server.py lacks v0.4.1 marker {m!r}")
            if "def is_available" not in vad:
                raise AssertionError("self-check: vad.py lacks is_available")
            # v0.4.7: the pipeline call-form probe is GONE (replaced by the
            # processor + generate path); assert the new call layers.
            for m in ("apply_transcription_request", "_parse_asr_output",
                      "apply_chat_template", "_ASR_TEXT_TAG", "asr_language"):
                if m not in asr:
                    raise AssertionError(f"self-check: asr.py lacks the v0.4.7 marker {m!r}")
            # v0.4.2 markers: voice selection + the 8B LLM swap
            for m in V042_HTML_MARKERS:
                if m not in html:
                    raise AssertionError(f"self-check: voicemem.html lacks v0.4.2 marker {m!r}")
            for m in V042_SERVER_MARKERS:
                if m not in web_server:
                    raise AssertionError(f"self-check: web_server.py lacks v0.4.2 marker {m!r}")
            voice_settings_mod = root_prefix + "app/voice_settings.py"
            if voice_settings_mod not in names:
                raise AssertionError("self-check: app/voice_settings.py missing from the ZIP")
            vs_src = zf.read(voice_settings_mod).decode("utf-8", errors="replace")
            for m in ("hu_HU-anna-medium", "hu_HU-berta-medium", "hu_HU-imre-medium",
                      "en_US-lessac-medium", "Szia Thomas, ez egy hangteszt."):
                if m not in vs_src:
                    raise AssertionError(f"self-check: voice_settings.py lacks marker {m!r}")
            lock = zf.read(root_prefix + "MODELS.lock.json").decode("utf-8", errors="replace")
            # v0.4.16: the lock must NOT contain an llm entry at all (the
            # Qwen3.6 GGUF is operator-placed) and must NOT reference Gemma
            # (the fallback profile was removed) or any legacy Qwen3 LLM.
            import json as _json
            _lock = _json.loads(lock)
            _comps = [str(m.get("component", "")) for m in _lock["models"]]
            if "llm" in _comps:
                raise AssertionError("self-check: MODELS.lock.json must NOT auto-download the LLM (operator-placed)")
            for m in ("gemma", "Qwen3-8B", "Qwen/Qwen3-8B"):
                if m.lower() in lock.lower():
                    raise AssertionError(f"self-check: MODELS.lock.json still references {m!r}")
            if "fallback_repos" in lock:
                raise AssertionError("self-check: MODELS.lock.json still declares fallback repos")
            for banned in ("Qwen3-8B", "qwen3-8b", "Qwen3-4B", "qwen3-4b", "Qwen/Qwen3-8B"):
                if banned in lock:
                    raise AssertionError(f"self-check: MODELS.lock.json still references {banned!r}")
            if "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF" not in lock:
                raise AssertionError("self-check: MODELS.lock.json must document the exact tested Qwen source")
            config_py = zf.read(root_prefix + "app/config.py").decode("utf-8", errors="replace")
            # v0.4.14+: the ACTIVE profile is Qwen3.6 35B A3B IQ4_XS (env-
            # overridable; v0.4.16: single-model policy, NO fallback profile)
            if 'llm_model_name: str = "qwen3.6-35b-a3b"' not in config_py:
                raise AssertionError("self-check: app/config.py does not pin the active qwen3.6-35b-a3b profile")
            # v0.4.17: the resolution is llm_model.json > env > operator-dir
            # scan; a phantom non-existent default path is BANNED (the honest
            # None state applies when nothing is configured - the exact
            # model is identified by scripts/identify_ollama_blob.ps1 or the
            # web UI picker, not by a hardcoded file name in app code).
            if "Qwen3.6-35B-A3B-IQ4_XS.gguf" in config_py:
                raise AssertionError(
                    "self-check: app/config.py hardcodes a project-default GGUF file name "
                    "(v0.4.17 removed the misleading phantom default; the operator-dir "
                    "scan resolves ANY valid GGUF placed there)"
                )
            if "operator_default_candidates" not in config_py:
                raise AssertionError(
                    "self-check: app/config.py must use the operator-dir GGUF scan "
                    "(imported from app/llm_model_settings.py)"
                )
            if "qwen3-8b" in config_py or "qwen3-4b" in config_py:
                raise AssertionError("self-check: app/config.py still references a legacy Qwen3 LLM")
            # v0.4.16: the retired Gemma scripts must NOT ship, and the
            # persisted llm_model.json runtime state must NOT ship either.
            for absent in V0416_ABSENT_FILES:
                if root_prefix + absent in names:
                    raise AssertionError(
                        f"self-check: {absent} must NOT ship in the ZIP"
                    )
            for runtime_state in (
                root_prefix + "config/voice_settings.json",
                root_prefix + "config/llm_model.json",
            ):
                if runtime_state in names:
                    raise AssertionError(
                        f"self-check: runtime state {runtime_state} must NOT ship in the ZIP"
                    )
            tts_src = zf.read(root_prefix + "app/tts.py").decode("utf-8", errors="replace")
            if "voice: Optional[str] = None" not in tts_src:
                raise AssertionError("self-check: tts.py lacks the voice override parameter")
            # v0.4.4: the three field fixes must ship in full
            for rel, markers in V044_MARKERS.items():
                src = zf.read(root_prefix + rel).decode("utf-8", errors="replace")
                for m in markers:
                    if m not in src:
                        raise AssertionError(
                            f"self-check: {rel} lacks v0.4.4 marker {m!r}"
                        )
            for rel in V044_REQUIRED_FILES:
                if root_prefix + rel not in names:
                    raise AssertionError(
                        f"self-check: v0.4.4 regression test file missing: {rel}"
                    )
            # v0.4.5: the ASR self-heal + prompt budget must ship in full
            for rel, markers in V045_MARKERS.items():
                src_text = zf.read(root_prefix + rel).decode("utf-8", errors="replace")
                for m in markers:
                    if m not in src_text:
                        raise AssertionError(
                            f"self-check: {rel} lacks v0.4.5 marker {m!r}"
                        )
            for rel in V045_REQUIRED_FILES:
                if root_prefix + rel not in names:
                    raise AssertionError(
                        f"self-check: v0.4.5 regression test file missing: {rel}"
                    )
            # v0.4.6: the British-English UI + the code-review fixes must
            # ship in full
            for rel, markers in V046_MARKERS.items():
                src_text = zf.read(root_prefix + rel).decode("utf-8", errors="replace")
                for m in markers:
                    if m not in src_text:
                        raise AssertionError(
                            f"self-check: {rel} lacks v0.4.6 marker {m!r}"
                        )
            for rel in V046_REQUIRED_FILES:
                if root_prefix + rel not in names:
                    raise AssertionError(
                        f"self-check: v0.4.6 regression test file missing: {rel}"
                    )
            # v0.4.7: the ASR call path + Chinese-memory localisation must
            # ship in full
            for rel, markers in V047_MARKERS.items():
                src_text = zf.read(root_prefix + rel).decode("utf-8", errors="replace")
                for m in markers:
                    if m not in src_text:
                        raise AssertionError(
                            f"self-check: {rel} lacks v0.4.7 marker {m!r}"
                        )
            for rel in V047_REQUIRED_FILES:
                if root_prefix + rel not in names:
                    raise AssertionError(
                        f"self-check: v0.4.7 file missing: {rel}"
                    )
            # v0.4.8: the mic -> ASR test must ship in full
            for rel, markers in V048_MARKERS.items():
                src_text = zf.read(root_prefix + rel).decode("utf-8", errors="replace")
                for m in markers:
                    if m not in src_text:
                        raise AssertionError(
                            f"self-check: {rel} lacks v0.4.8 marker {m!r}"
                        )
            for rel in V048_REQUIRED_FILES:
                if root_prefix + rel not in names:
                    raise AssertionError(
                        f"self-check: v0.4.8 file missing: {rel}"
                    )
            # v0.4.9: the ten code-analysis-report fixes must ship in full
            for rel, markers in V049_MARKERS.items():
                src_text = zf.read(root_prefix + rel).decode("utf-8", errors="replace")
                for m in markers:
                    if m not in src_text:
                        raise AssertionError(
                            f"self-check: {rel} lacks v0.4.9 marker {m!r}"
                        )
            # v0.4.10: the original-repo ASR repair + browser capture fixes
            for rel, markers in V0410_MARKERS.items():
                src_text = zf.read(root_prefix + rel).decode("utf-8", errors="replace")
                for m in markers:
                    if m not in src_text:
                        raise AssertionError(
                            f"self-check: {rel} lacks v0.4.10 marker {m!r}"
                        )
            for rel in V0410_REQUIRED_FILES:
                if root_prefix + rel not in names:
                    raise AssertionError(
                        f"self-check: v0.4.10 regression test file missing: {rel}"
                    )
            # v0.4.11: the VAD sensitivity change must ship in BOTH config
            # layers (YAML + dataclass default) with the docs/tests in sync
            for rel, markers in V0411_MARKERS.items():
                src_text = zf.read(root_prefix + rel).decode("utf-8", errors="replace")
                for m in markers:
                    if m not in src_text:
                        raise AssertionError(
                            f"self-check: {rel} lacks v0.4.11 marker {m!r}"
                        )
            # v0.4.12: stale-page detection + raw capture + send-as-turn
            for rel, markers in {**V0412_MARKERS, **V0413_MARKERS, **V0414_MARKERS, **V0415_MARKERS, **V0416_MARKERS, **V0417_MARKERS, **V0418_MARKERS, **V0419_MARKERS, **V0420_MARKERS, **V0421_MARKERS, **V050_MARKERS, **V052_MARKERS}.items():
                src_text = zf.read(root_prefix + rel).decode("utf-8", errors="replace")
                for m in markers:
                    if m not in src_text:
                        raise AssertionError(
                            f"self-check: {rel} lacks v0.4.12 marker {m!r}"
                        )
            if root_prefix + "tests/unit/test_web_ui_v0412.py" not in names:
                raise AssertionError(
                    "self-check: v0.4.12 UI regression test file missing "
                    "from the ZIP"
                )
            # v0.4.14: the energy-fallback VAD + mic-dispatch regression
            # tests and the E2E deaf-channel launcher must ship
            for must in (
                "tests/unit/test_fused_vad.py",
                "tests/unit/test_web_mic_dispatch.py",
                "tests/e2e_deaf_server.py",
                "models/llm/qwen3.6-35b-a3b/README.md",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.4.14 file missing from the ZIP: {must}"
                    )
            # v0.4.15: the exact-Qwen-GGUF discovery script must ship
            for must in (
                "scripts/find_qwen_gguf.ps1",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.4.15 file missing from the ZIP: {must}"
                    )
            # v0.4.16: the model-picker modules + policy tests must ship
            for must in (
                "app/gguf.py",
                "app/native_picker.py",
                "app/llm_model_settings.py",
                "tests/unit/test_llm_model_settings.py",
                "tests/validation/test_feature_llm_policy.py",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.4.16 file missing from the ZIP: {must}"
                    )
            # v0.4.17: the Ollama-blob identifier must ship
            for must in (
                "scripts/identify_ollama_blob.ps1",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.4.17 file missing from the ZIP: {must}"
                    )
            # v0.4.6 localisation guarantee: the web UI must contain ZERO
            # Han characters (CJK Unified Ideographs). The only legal Han
            # in the repo lives in app/emotion.py's model token table and
            # the _RB_SUFFIX_RE cleanup regex - neither is user-visible.
            if any("\u4e00" <= ch <= "\u9fff" for ch in html):
                raise AssertionError(
                    "self-check: web/voicemem.html still contains Han "
                    "characters - the UI must be pure English"
                )
            starter_src = zf.read(
                root_prefix + "scripts/start_llama_server.ps1"
            ).decode("ascii", errors="replace")
            if starter_src.replace("\r\n", "\n").count("\n") and "\r\n" not in starter_src:
                raise AssertionError("self-check: start_llama_server.ps1 lost its CRLF endings")
    except AssertionError as exc:
        zip_path.unlink(missing_ok=True)
        print(f"BUILD FAILED - self-check: {exc} (the zip was deleted)")
        return 1

    # ---------------- sha256 ----------------
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest().upper()
    write_text_no_bom(RELEASES / f"{ZIP_NAME}.sha256", f"{sha}  {ZIP_NAME}\n")
    size = zip_path.stat().st_size

    # ---------------- index / changelog / history ----------------
    index_path = RELEASES / "RELEASE_INDEX.json"
    entries = []
    if index_path.is_file():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8")).get("releases", [])
        except Exception:  # noqa: BLE001
            entries = []
    # idempotent: a rebuild REPLACES this version's index entry (final numbers)
    entries = [e for e in entries if e.get("version") != NEW_VERSION]
    if True:
        entries.append({
            "version": NEW_VERSION,
            "built_at_utc": built_at,
            "zip": ZIP_NAME,
            "sha256": sha,
            "size_bytes": size,
            "notes": NOTES,
            "git_commit": build_info["git_commit"],
            "test_gate": "run_tests.ps1",
            "test_gate_exit_code": 0,
            "validation_deep_pass": deep_pass,
            "validation_deep_skipped": deep_skip,
            "validation_strict": False,
        })
    write_text_no_bom(index_path, json.dumps(
        {"schema_version": 1, "releases": entries}, indent=2, ensure_ascii=False) + "\n")
    # newest-first order (a rebuild appends at the end - re-sort explicitly)
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))

        def _key(e):
            return tuple(int(x) for x in str(e.get("version", "0")).split("."))

        data["releases"] = sorted(data.get("releases", []), key=_key, reverse=True)
        write_text_no_bom(index_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - ordering is cosmetic, never fail the build
        pass

    repo_ch = REPO / "CHANGELOG.md"
    ch_text = repo_ch.read_text(encoding="utf-8")
    ch_text = _strip_version_entry(ch_text, NEW_VERSION)
    marker = "# Változatonapló (CHANGELOG) — voicemem-agent"
    head, rest = ch_text.split(marker, 1)
    idx = rest.find("## [")
    rest = rest[:idx] + ch_entry + rest[idx:] if idx != -1 else ch_entry + rest
    repo_ch.write_text(head + marker + rest, encoding="utf-8", newline="")

    hist_path = RELEASES / "BUILD_HISTORY.json"
    history = []
    if hist_path.is_file():
        try:
            history = json.loads(hist_path.read_text(encoding="utf-8")).get("attempts", [])
        except Exception:  # noqa: BLE001
            history = []
    history.append({
        "at_utc": built_at,
        "outcome": "success",
        "version": NEW_VERSION,
        "zip": ZIP_NAME,
        "sha256": sha,
        "test_gate_exit_code": 0,
        "deep_validation": f"pass: {deep_pass}; skip: {deep_skip}",
    })
    write_text_no_bom(hist_path, json.dumps({"attempts": history}, indent=2, ensure_ascii=False) + "\n")

    shutil.rmtree(stage_root, ignore_errors=True)
    print(f"BUILD SUCCESS - VoiceMemAgent {NEW_VERSION}")
    print(f"  {PREV_VERSION} -> {NEW_VERSION}")
    print(f"  ZIP    : releases/{ZIP_NAME} ({size/1024:.1f} KB, {n_files} files, {n_bats} .bat CRLF)")
    print(f"  SHA256 : {sha}")
    print(f"  Deep   : PASS: {deep_pass} (SKIP: {deep_skip})")
    print(f"  Changelog + index + history updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
