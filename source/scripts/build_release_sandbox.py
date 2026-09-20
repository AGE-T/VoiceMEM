#!/usr/bin/env python3
"""Sandbox replica of scripts/build_release.ps1 (Linux-side release builder).

Mirrors the PowerShell builder step-for-step:
  1. (gate) v0.8.1 gate-order fix: a GREEN releases/gate_record.json from
     scripts/run_release_gate.py is REQUIRED, and its source-tree
     fingerprint must match the CURRENT tree — the exact tree that is
     packaged is provably the tree that passed the final release gate (the
     v0.8.0 flow ran the gate BEFORE the VERSION bump and shipped a
     never-gated PAGE_VERSION defect). The gate numbers in BUILD_INFO /
     CHANGELOG / RELEASE_INDEX come from the record — caller-supplied
     numbers are no longer accepted.
  2. staged CHANGELOG entry (the ZIP carries its own version's entry; the
     entry's gate numbers are stamped FROM the gate record).
  3. version sync is already done (VERSION/pyproject/yaml + the generated
     page version via scripts/sync_page_version.py) — the gate has seen it.
  4. staging: allowlist copy of app/ config/ scripts/ tests/ web/ + root
     files + M0 placeholder dirs; every staged *.bat normalised to CRLF.
  5. BUILD_INFO.json into the stage root.
  6. ZIP with a single top-level folder VoiceMemAgent_v<version>/.
  6b. SELF-CHECK: the finished ZIP is opened and verified (placeholder
      dirs, own CHANGELOG entry, VERSION == PAGE_VERSION, cumulative
      markers, and the per-version markers). A failed self-check DELETES
      the zip - no false success is ever published.
  7. sha256 sidecar file.
  8. RELEASE_INDEX.json + repo CHANGELOG.md + BUILD_HISTORY.json updated.

Usage:
    python scripts/run_release_gate.py            # FIRST: gate + record
    python scripts/build_release_sandbox.py [gate_record.json]
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

NEW_VERSION = "0.10.7"
PREV_VERSION = "0.10.6"
ZIP_NAME = f"VoiceMemAgent_v{NEW_VERSION}.zip"

PLACEHOLDER_DIRS = [
    "models/asr/qwen3-asr-0.6b",     # v0.6.0: LEGACY non-production migration module dir (app/asr.py); the engine registry never loads it
    "models/asr/parakeet-tdt-0.6b-v3",     # v0.6.0: the DEFAULT production ASR engine
    "models/asr/nemotron-3.5-asr-streaming-0.6b",  # v0.6.0: selectable streaming engine (CUDA target)
    "models/llm/qwen3.6-35b-a3b",   # v0.4.14: the ONE AND ONLY LLM dir (README + .gitkeep; the GGUF is operator-placed, any drive - v0.4.16 picker)
    "models/tts/supertonic-3",   # v0.7.0: Supertonic 3 onnx assets (installer downloads; Piper retired)
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
    "requirements.lock.json", "pyproject.toml", ".gitignore", "START.bat",
    "MODELS.lock.json",
    # v0.5.0: the controlled VoiceMem ownership artifacts ship in the ZIP
    "VOICEMEM_PIN.json", "UPSTREAM_POLICY.md",
]

NOTES = (
    "v0.10.7 TTS N1/N2 DETERMINISZTIKUS JAVITO KIADAS (operator order: a "
    "2026-09-20 konszolidalt audit KET ujonnan azonositott hibaosztalyanak "
    "KIZAROLAGOS javitasa - semmi mas; audit/VoiceMEM_tts_N1_N2_forensic). "
    "N1 (angol kolcsonzo MAGYAR szavat nyel el): a 'meeting' tipusu "
    "ee-bizonyiteku kolcsonzo a trailing budgettel magyar szavakat "
    "soprit az EN spanbe ('A 2026-os meeting fontos...' -> en 'meeting "
    "fontos'). GYOKER: a magyar tartalmaszavak egy resze ortografialag "
    "láthatatlan a szint-szintu osztalyozo elott (nincs ekezet, nincs "
    "stopword-bejegyzes, nincs eros digraf). JAVITAS: MAGYAR "
    "MAGANHANGZO-HARMONIA OR (a nativ magyar szo maganhangzoi vagy "
    "mind hatso a/o/u, vagy mind elolso e/i/o-umlaut; a vegyes "
    "elolso+hatso kombinacio jelentos reszt angol sajatsag - "
    "'base'/'later'/'afterparty'): _hu_harmonic() or a KEPT budget "
    "(trailing ES leading) aln: legalabb 2 harmonikus maganhangzoju "
    "semleges jeloltet SOHA nem abszorbeal az EN spanbe; egymaganhangzos "
    "tokenek mentesek ('up', 'next' valtozatlanul abszorbealodik). N2 "
    "(ekezet nelkuli magyar host EN-re billen): a chunk-szintu "
    "szamitas a magyar nevelot ('a') angol stopword-kent szamolja "
    "('Holnap lesz a meeting a csapattal.' -> hu 2 vs en 2 dontoen "
    "EN), es a 'holnap' hianyzott a HU_PLAIN_WORDS F-O kategoriabol. "
    "JAVITAS: (1) 'holnap' belepett a tablaba (a 'mikor'/'hol' "
    "osztalya); (2) donto-szabaly finomitas: ekezet nelkuli dontetlennel "
    "a magyar HOST-KONTEXTUS funkcioszav-bizonyitek gyoz (az idzett "
    "regionben levo tokenek - pl. 'The word \"szia\" means hello here.' "
    "- NEM szavaznak a hostrol: a regiongep sajat spant ad nekik; a "
    "szamitas lazy - csak a dontetlen uton fut). TESZTEK: +16 uj "
    "(tests/unit/test_tts_n1_n2_matrix.py: N1 operator-eset + "
    "melleknev/ige/fonev-korulmenyek + tobbszoros kolcsonso-budget-"
    "szamitas + leading tukor + differencial-vedelmek; N2 operator-eset "
    "+ audit-varians + ekezet nelkuli funkcioszavas hostok + tobb "
    "kolcsonso + ekezetes kontrollok + tiszta EN/HU differencialok); "
    "a ket dokumentalt hamis-pozitiv pin TUDATOSAN korrigalva (mindketto "
    "magyar szo EN-spanbe soprasa = az N1 osztaly maga: 'Nyisd ki a "
    "chat ablakot.' most egyetlen HU span; 'Holnap see you later.' most "
    "hu 'Holnap ' + en 'see you later.'). NEM VALTOZOTT (szerzodes): "
    "nincs lexikon, nincs LLM, nincs uj detektalos rendszer, "
    "engine/chunking/hang; a vedett angol frazisteszek ('touch base', "
    "'catch up', 'see you later', 'cut to the chase', 'hit the ground "
    "running', 'burn the midnight oil') valtozatlanul EN-be iranyitanak; "
    "a keteertelmu kolcsonzo-vedelmek (hazai/auto/tea/euro/projekt/"
    "only/really/city/money/many) es a 'break a leg' nulla-bizonyiteku "
    "korlat valtozatlan. DOKUMENTALT MARADVANYOK (határon kivul): a "
    "csupa 'a' nevelo utazasa a kolcsonzo utan ('meeting a csapattal' "
    "-> en 'meeting a'); az ekezet nelkuli DIsharmonikus magyar "
    "tartalmaszavak (fiu/kavics-osztaly); a funkcioszo nelkuli mondatok "
    "host-billenese. ELOSZLAS (mert sandbox, a 22-mondatos N1/N2 korpusz "
    "4400 mintaja): median 46.4 -> 49.8 us (+7.3%), p90 61.1 -> 64.3 us "
    "- a detektor mikrosecundum-osztalyu marad, a dontetlen-ag lazy. "
    "GATE: a javitott fan GREEN - 1537 teszt (+16 az uj matrix), 0 "
    "regresszio, 2 pinned sandbox env-gap, fingerprint 28749983..."
)

#: v0.6.0 markers: the modular ASR engine contract - asr_core (AudioBuffer,
#: AsrResult, AsrError, registry, select_engine), the two NVIDIA adapters,
#: the official-context SileroVad feed, the engine-contract web-server leg,
#: the generic stage events, and the CHAIN UI. Merged LAST so it overrides
#: every older block for the same file (the FusedVad/energy-fallback era is
#: history).
V060_MARKERS = {
    "app/asr_core.py": [
        "class AudioBuffer",
        "class AsrResult",
        "class AsrError",
        "class AsrCapability",
        "MODEL_REGISTRY",
        "def select_engine",
        "class _UnavailableEngine",
        "ASR_ENGINE",
    ],
    "app/asr_parakeet.py": [
        "ParakeetForTDT",
        "Non-streaming engine: feeding is a contract violation",
        "AudioBuffer",
    ],
    "app/asr_nemotron.py": [
        "lookahead",
        "TextIteratorStreamer",
        "AudioBuffer",
    ],
    "app/vad.py": [
        "class SileroVad:",
        "64-sample rolling CONTEXT",
        "def reset",
    ],
    "app/web_server.py": [
        "select_engine",
        "_UnavailableEngine",
        "def _stage_event",
        "stage_event",
        "user_transcript",
        "asr_empty",
    ],
    "app/config.py": [
        "asr_engine",
    ],
    "config/voicemem_config.yaml": [
        "asr_engine",
        "parakeet",
    ],
    "web/voicemem.html": [
        "const PIPE_KEYS=['mic','vad','asr','memory','embedding','llm','tts']",
        "const CHAIN_LIVE={}",
        "renderChainErrorLine",
        "stage_event",
        "const PAGE_VERSION=",
        "micDiagHead",
        "llmColHead",
    ],
    "tests/unit/test_asr_modular.py": [
        "AudioBuffer",
        "AsrResult",
        "select_engine",
        "no fallback",
    ],
    "docs/TASK_A_ASR_GATE.md": [
        "ASR Exit Gate",
        "parakeet",
    ],
    "app/mock_components.py": [
        "class MockAsrEngine",
        "engine_id = \"mock\"",
    ],
}

#: v0.6.1 markers: the DELIVERY-LAYER hotfix - the bootstrap manifest
#: finally lists the production ASR (the v0.6.0 release shipped the modular
#: engine but the MODELS.lock.json asr entry still said Qwen, so the field
#: bootstrap never downloaded parakeet and the runtime failed with the
#: cryptic "Unrecognized processing class"). Merged LAST.
V061_MARKERS = {
    "MODELS.lock.json": [
        "nvidia/parakeet-tdt-0.6b-v3",
        "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    ],
    "scripts/download_models_hf.py": [
        "nvidia/parakeet-tdt-0.6b-v3",
    ],
    "scripts/bootstrap.ps1": [
        # v0.10.1: the probe floor is 5.9 (where ParakeetForTDT exists)
        "(5, 9)",
    ],
    "scripts/install_m1.ps1": [
        "5.9",
    ],
    "scripts/verify_m1.ps1": [
        "parakeet-tdt-0.6b-v3",
    ],
    "requirements.txt": [
        # v0.10.1: exact pin (P0 forensic)
        "transformers==5.17.0",
    ],
    "app/asr_parakeet.py": [
        "does not contain",
        "MODELS.lock.json",
    ],
    "app/config.py": [
        "_selected_asr_model_present",
    ],
    "scripts/write_install_manifest.py": [
        "nvidia/parakeet-tdt-0.6b-v3",
    ],
}

#: v0.6.2 markers: the installer step-12 pin-read fix + the production-ASR
#: banners. The pin check must read provenance.upstream_commit (a root-level
#: read is always $null -> deterministic step-12 failure on every first real
#: installer run); the bootstrap/install summaries must name Parakeet, not
#: the retired Qwen3 ASR. Merged LAST (overrides every older block).
V062_MARKERS = {
    "scripts/install_m1.ps1": [
        "$PinJson.provenance.upstream_commit",
        "v0.6.2 (FIELD REPORT",
        "NVIDIA Parakeet TDT 0.6B v3",
        "olvasott: $PinCommitShown",
    ],
    "scripts/bootstrap.ps1": [
        "NVIDIA Parakeet TDT 0.6B v3",
    ],
    "tests/unit/test_voicemem_controlled.py": [
        "InstallerPinReadRegressionTests",
        "test_step12_passes_on_the_real_pin_file",
        "test_step12_fails_loudly_on_a_wrong_commit",
    ],
    "web/voicemem.html": [
        "const PAGE_VERSION="
    ],
}

#: v0.6.3 markers: the ALL-IN-ONE LLM profile baked into the release ZIP
#: (the former separate LLMConfig package applied at build time). Every
#: production-config surface must carry the measured profile ngl20 + 32K,
#: and the decision evidence must ship inside the ZIP under docs/.
#: Merged LAST (overrides every older block for the same file).
V063_MARKERS = {
    "config/voicemem_config.yaml": [
        "llm_context_size: 32768",
        "llm_n_gpu_layers: 20",
        "PRODUKCIÓS LLM-PROFIL",
    ],
    "config/env.local.ps1": [
        'LLAMA_CONTEXT_SIZE = "32768"',
        'LLAMA_N_GPU_LAYERS = "20"',
    ],
    "config/env.local.sh": [
        "LLAMA_CONTEXT_SIZE=32768",
        "LLAMA_N_GPU_LAYERS=20",
    ],
    "scripts/start_llama_server.ps1": [
        '$FbNgl = "20"',
        '$FbCtx = "32768"',
        "--reasoning\", \"off\"",
    ],
    "scripts/verify_m1.ps1": [
        '"-ngl", "20"',
        '"-c", "32768"',
    ],
    "web/voicemem.html": [
        "const PAGE_VERSION="
    ],
    "docs/llm_config_ngl20-c32768/identity.json": [
        "ngl20 + 32768 context",
    ],
    "docs/llm_config_ngl20-c32768/RECOVERY.md": [
        "n_gpu_layers 26 -> 20",
        "context 8192 -> 32768",
    ],
}

#: v0.6.4 markers: the external-audit fixes (CD-3 HU/EN temporal,
#: CD-4 recency sort, OCC-1 occurrence counting, F-B provenance render,
#: SEC-1 loopback guard, OFF-1 local_files_only, BAT-1 process guard)
#: inside the release ZIP.
V064_MARKERS = {
    "vendor/voicemem/voicemem/leftbrain/time_expand.py": [
        "_LATIN_SPANS",
        "tegnapel\u0151tt",
        "next week",
        "VM-LOCAL-010",
    ],
    "vendor/voicemem/voicemem/leftbrain/local_memory_store.py": [
        "def parse_date_values",
        "def date_overlap_bonus_values",
        "def recency_bonus",
        "VM-LOCAL-011",
        "occurrence_count",
    ],
    "vendor/voicemem/voicemem/leftbrain/mem0_backend_store.py": [
        "def count_occurrence",
        "h.base_score + h.recency_boost",
        "VM-LOCAL-012",
    ],
    "vendor/voicemem/voicemem/leftbrain/memory_repository.py": [
        "def count_occurrence",
        "VM-LOCAL-012",
    ],
    "vendor/voicemem/voicemem/utils/common/voice_input.py": [
        'r.event == "NONE" and r.memory_id',
        "omit-NONE",
    ],
    "vendor/voicemem/voicemem/__init__.py": [
        "VM-LOCAL-010",
        "VM-LOCAL-011",
        "VM-LOCAL-012",
    ],
    "app/voicemem_bridge.py": [
        "def hit_provenance_suffix",
    ],
    "app/web_server.py": [
        "VOICEMEM_ALLOW_REMOTE_WEB",
        "def hit_provenance_suffix",
    ],
    "app/asr_parakeet.py": [
        "local_files_only=True",
    ],
    "app/asr_nemotron.py": [
        "local_files_only=True",
    ],
    "scripts/start_llama_server.ps1": [
        "function Get-LlamaServerProcesses",
        "[switch]$ForceKillExisting",
    ],
    "tests/unit/test_time_expand_hu.py": [
        "class TimeExpandHungarianTests",
    ],
    "tests/integration/test_memory_safety.py": [
        "class OccurrenceExplicitNoneTests",
        "class OccurrenceOmitNoneTests",
    ],
    "CONTRACT.md": [
        "v0.6.4 kieg\u00e9sz\u00edt\u0151",
    ],
    "README.md": [
        "OPER\u00c1TORI D\u00d6NT\u00c9S",
    ],
}

#: v0.4.3-era retired-model markers were REMOVED in v0.4.16 (single-model policy;
#: the self-check now asserts their ABSENCE instead - see V0416_ABSENT_FILES).

#: v0.7.0 markers: the Supertonic 3 TTS migration - the engine adapter,
#: the no-fallback contract, the config/voice-settings/web integration,
#: the MODELS.lock archive entry, and the Piper retirement, inside the ZIP.
V070_MARKERS = {
    "app/tts_supertonic.py": [
        "TTS engine = Supertonic 3",
        "auto_download=False",
        "class SupertonicTtsEngine",
        "_speed_from_length_scale",
        "no fallback",
        "_trim_edge_silence",
        "voice: Optional[str] = None",
    ],
    "app/config.py": [
        "supertonic3",
        "supertonic_model_dir",
        "supertonic_voices_dir",
        "output_sample_rate must be 44100",
    ],
    "app/voice_settings.py": [
        "DEFAULT_HU_VOICE = \"F1\"",
        "language-AGNOSTIC",
    ],
    "app/web_server.py": [
        "from app.tts_supertonic import SupertonicTtsEngine as TtsEngine",
        "Supertonic 3 model assets incomplete",
    ],
    "config/voicemem_config.yaml": [
        "tts_engine: \"supertonic3\"",
        "supertonic_steps: 8",
        "output_sample_rate: 44100",
    ],
    "MODELS.lock.json": [
        "supertone-oss-archive/supertonic-3",
        "aafc6e32416a594460b32413efc49d7fe4ce6d46",
    ],
    "requirements.lock": [
        "supertonic==1.3.1",
    ],
    "scripts/download_models_hf.py": [
        "supertone-oss-archive/supertonic-3",
    ],
    "scripts/install_m1.ps1": [
        "supertonic",
    ],
    "tests/unit/test_tts_supertonic.py": [
        "class SupertonicEngineTests",
        "class TrimSilenceTests",
    ],
    "LICENSES.md": [
        "OpenRAIL-M",
    ],
    "web/voicemem.html": [
        "const PAGE_VERSION=",
        "Supertonic 3 since v0.7.0",
        "Preview this voice (local Supertonic 3)",
    ],
}

#: v0.7.1 markers: the installer parse-fix hotfix - the restored closing
#: brace + hotfix comment in install_m1.ps1, the bin_dir bridge fix in
#: start_llama_server.ps1, and the ps_lint guard itself, inside the ZIP.
#: NOTE: "web/voicemem.html" is deliberately NOT a key here - the dict
#: merge in the self-check loop replaces whole entries per file, and
#: V070's html marker set must stay the effective one.
V071_MARKERS = {
    "scripts/install_m1.ps1": [
        "supertonic",  # kept from V070 (this entry replaces it in the merge)
        "v0.7.1 (field report #5): a v0.7.0-es piper-eltavolitas",
        "scripts/ps_lint.py most gyartas elott is ellenorzi",
    ],
    "scripts/start_llama_server.ps1": [
        "v0.7.1 (field report #5): AgentConfig.bin_dir",
        '"exe": str(c.bin_dir),',
    ],
    "scripts/ps_lint.py": [
        "def check_ps_source",
        "PS7_ONLY_OPERATORS",
        "v0.7.1 field report",
    ],
    "tests/validation/test_feature_scripts.py": [
        "test_logic_all_ps1_are_structurally_balanced",
        "ps_lint",
    ],
}


#: v0.7.2 markers: the canonical LLM configuration - the yaml file, the
#: ONE loader, the AgentConfig wiring, the generated command line, the
#: startup summary, and the removed duplicate copies, inside the ZIP.
V072_MARKERS = {
    "config/llm_config.yaml": [
        "THE canonical LLM runtime configuration",
        "gpu_layers: 20",
        "context_size: 32768",
        "reasoning:",
        "enabled: false",
    ],
    "app/llm_config.py": [
        "the ONE loader (v0.7.2)",
        "def load_llm_config",
        "def llama_server_args",
        "def match_llama_server_log",
        "ENV_OVERRIDES",
    ],
    "app/config.py": [
        "def materialise_llm_runtime",
        "cfg.materialise_llm_runtime(config_dir=path.parent)",
        "v0.7.2 — REMOVED duplicate LLM VALUE parsing",
    ],
    "app/llm.py": [
        "v0.7.2: when the config carries the canonical runtime",
        "return runtime.thinking_control_kwargs()",
    ],
    "app/main.py": [
        "config.materialise_llm_runtime()",
        "config.llm_config_summary()",
    ],
    "app/web_server.py": [
        "cfg.materialise_llm_runtime()",
        "cfg.llm_config_summary()",
    ],
    "scripts/start_llama_server.ps1": [
        "$LlamaArgs += @(\"--reasoning\", $ReasoningFlag)",
        '"reasoning": (runtime.reasoning_server_arg',
        "reasoning = \"off\"",
    ],
    "config/voicemem_config.yaml": [
        "KANONIKUS FORRÁS KÖLTÖZÖTT",
    ],
    "tests/unit/test_llm_config.py": [
        "test_from_yaml_materialises_the_canonical_runtime",
        "test_canonical_values_beat_voicemem_yaml_duplicates",
        "test_thinking_kwargs_come_from_the_runtime_single_source",
    ],
}
V080_MARKERS = {
    "vendor/voicemem/voicemem/rightbrain/traits_store.py": [
        "TRAIT_REINFORCE_STEP",
        "effective_trait_confidence",
        "occurrence_count",
        "first_seen",
        "last_seen",
    ],
    "vendor/voicemem/voicemem/rightbrain/brain.py": [
        "TRAIT_CONF_RANK_FLOOR",
        "eff_confidence",
    ],
    "vendor/voicemem/voicemem/orchestrator.py": [
        "_maybe_archive_cold_memories",
        "archive_cold_last_run",
    ],
    "vendor/voicemem/voicemem/rightbrain/store.py": [
        "TTL_EXPIRY_DAYS",
        "ttl_expired",
    ],
    "app/text_utils.py": [
        "HU_PLAIN_WORDS",
        "HU_DIGRAPHS",
        "EN_ONLY_LETTERS",
    ],
    "tests/unit/test_traits_observation.py": [
        "test_confidence_asymptotic",
        "test_ranking_floor_never_zeroes",
        "test_legacy_schema_migrated",
    ],
    "tests/unit/test_archive_ttl_wiring.py": [
        "test_throttle_blocks_second_run_within_20h",
        "test_rows_survive_non_destructively",
    ],
    "MODELS.lock.json": [
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "394d7e6b8c193baf42e965c97b13894e9de4afd8",
    ],
    "VOICEMEM_PIN.json": [
        "VM-LOCAL-013",
        "VM-LOCAL-014",
    ],
    "CONTRACT.md": [
        "v0.8.0 kiegészítő",
    ],
}

#: v0.8.1 markers: the stabilization release — the typed retrieval result
#: contract + its three consumers, the page-version single source, the
#: gate-order fix (fingerprint-bound gate record required by the build),
#: and the British-English guard on the visible surfaces.
V081_MARKERS = {
    "app/retrieval_contract.py": [
        "class TraitObservationInfo",
        "def trait_fields_payload",
        "def trait_prompt_suffix",
        "TRAIT_PAYLOAD_KEYS",
    ],
    "app/web_server.py": [
        "trait_fields_payload",
        "_SLOT_EN",
        "localise_slot",
    ],
    "app/voicemem_bridge.py": [
        "trait_prompt_suffix",
    ],
    "scripts/run_release_gate.py": [
        "KNOWN_ENV_FAILURES",
        "def classify_failures",
        "gate_record.json",
    ],
    "scripts/sync_page_version.py": [
        "const PAGE_VERSION=",
    ],
    "scripts/release_tree.py": [
        "def tree_fingerprint",
    ],
    "tests/unit/test_retrieval_contract.py": [
        "trait_fields_payload",
        "trait_prompt_suffix",
    ],
    "tests/unit/test_release_gate_order.py": [
        "_verify_gate_record",
        "test_line_endings_preserved",
    ],
    "tests/unit/test_british_english.py": [
        "BritishEnglishSurfaceTests",
    ],
    "CONTRACT.md": [
        "v0.8.1 kiegészítő",
    ],
}

#: v0.9.0 markers: the memory-semantics mechanisms — the stance scanner,
#: the three-outcome merge decision, the supersession chain columns, the
#: current-first retrieval ordering, the contract extension and the
#: measured-matrix regression guard. Markers pin the MECHANISM (the v0.8.1
#: lesson: never pin a value the next release legitimately changes).
V090_MARKERS = {
    "vendor/voicemem/voicemem/rightbrain/stance.py": [
        "def classify_stance",
        "def observation_stance",
        "def topic_overlaps",
        "def is_presuppositional",
    ],
    "vendor/voicemem/voicemem/rightbrain/traits_store.py": [
        "SUPERSEDE_MIN_SIM",
        "def _best_active_match",
        "def _superseded_agreeing_match",
        "def _is_stale_replay",
        "superseded_by TEXT NOT NULL DEFAULT ''",
    ],
    "vendor/voicemem/voicemem/rightbrain/brain.py": [
        '"superseded_by"',
        '"stance"',
    ],
    "vendor/voicemem/voicemem/__init__.py": [
        "VM-LOCAL-015",
    ],
    "VOICEMEM_PIN.json": [
        "VM-LOCAL-015",
    ],
    "app/retrieval_contract.py": [
        "superseded_by: str",
        'bits.append("superseded")',
        '"stance", "supersedes", "superseded_by", "superseded_at",',
    ],
    "app/web_server.py": [
        '"superseded": bool(getattr(t, "superseded_by", "") or "")',
    ],
    "web/voicemem.html": [
        "x.superseded",
    ],
    "docs/SEMANTIC_MATRIX.md": [
        "OBSERVATION STORE NOT REQUIRED",
        "SUPERSEDE_MIN_SIM",
    ],
    "CONTRACT.md": [
        "Jelentés-semantikai elnevezések (v0.9.0 — VM-LOCAL-015)",
    ],
    "tests/unit/test_stance_cues.py": [
        "HungarianRequiredCases",
    ],
    "tests/unit/test_trait_semantics.py": [
        "TraitSemanticsTests",
        "test_contradiction_freezes_old_row",
    ],
    "tests/unit/test_retrieval_contract_semantics.py": [
        "PromptSuffixTests",
    ],
    "tests/unit/test_fact_supersession_semantics.py": [
        "FactSupersessionSemanticsTests",
    ],
    "tests/validation/test_feature_memory_semantics.py": [
        "MemorySemanticsValidation",
        "_MEASURED",
    ],
    "scripts/measure_semantic_matrix.py": [
        "def embed_passages",
    ],
}

#: v0.9.2 markers: the conversation-first background memory gate, the
#: extraction/conflict output limits, the prompt reduction + addendum
#: reorder, and the unified 1200-character canonical memory budget.
V092_MARKERS = {
    "app/background_memory.py": [
        "class BackgroundMemoryGate",
        "USER CONVERSATION HAS ABSOLUTE PRIORITY",
        "VOICEMEM_BG_GRACE_S",
    ],
    "app/web_server.py": [
        "v0.9.2 (PART 5): conversation-first background memory scheduling",
        "_memory_gate.arm(\"speech\")",
        "await self._memory_gate.submit(user_text, reply)",
        "v0.9.2 (PART 11) CANONICAL MEMORY BUDGET",
    ],
    "vendor/voicemem/voicemem/leftbrain/extract_facts_openai.py": [
        "VOICEMEM_EXTRACT_MAX_TOKENS",
        "VOICEMEM_RESOLVE_MAX_TOKENS",
        "_warn_if_truncated",
        "prompt_addendum() + \"\\n\\n\" + user_content",
    ],
    "tests/unit/test_extraction_output_limits.py": [
        "test_truncated_output_fails_closed",
        "test_extract_request_carries_max_tokens",
    ],
    "tests/unit/test_background_memory_gate.py": [
        "test_ingest_waits_while_turn_active",
        "test_speech_rearm_inside_grace_defers",
    ],
    "tests/unit/test_prompt_reduction.py": [
        "test_required_signals_present",
        "test_forbidden_removals_stay_removed",
    ],
}

#: v0.10.0 markers: the deterministic temporal classification module, the
#: shared stance vocabulary extended with "future", the event-time ISO
#: anchor fix, cessation closing valid_until, the denied-DELETE
#: supersession fallback, temporal ranking in the canonical search path,
#: render markers in both copies, the memory_history() consolidation hook,
#: and the memory-semantics document.
V0100_MARKERS = {
    "vendor/voicemem/voicemem/leftbrain/temporal.py": [
        "def fact_stance",
        "def fact_validity",
        "def query_temporal_intent",
        "def fact_status",
        "def temporal_rank_tier",
        "zero LLM calls, zero prompt changes",
    ],
    "vendor/voicemem/voicemem/rightbrain/stance.py": [
        '"future"',
        "_FUTURE_CUES",
    ],
    "vendor/voicemem/voicemem/leftbrain/mem0_backend_store.py": [
        "query_temporal_intent, row_status, temporal_rank_tier",
    ],
    "vendor/voicemem/voicemem/leftbrain/memory_repository.py": [
        "def memory_history",
    ],
    "vendor/voicemem/voicemem/orchestrator.py": [
        "v0.10",
        "isoformat",
    ],
    "app/voicemem_bridge.py": [
        "[v0.10 PHASE 6] temporal-semantic markers",
    ],
    "app/web_server.py": [
        "[v0.10 PHASE 6] temporal-semantic markers",
        "v0.10: fact-side temporal semantics",
    ],
    "app/retrieval_contract.py": [
        "[v0.10] ``future`` = not-yet-valid forward-looking observation",
    ],
    "docs/MEMORY_SEMANTICS.md": [
        "no Observation Store was introduced",
        "DERIVED, never stored",
    ],
    "tests/unit/test_temporal_semantics.py": [
        "RankingTierTests",
        "test_legacy_sort_order_is_byte_identical",
    ],
    "tests/integration/test_temporal_memory.py": [
        "SupersessionChainTests",
        "ObservationTimeTests",
        "TraitFutureStanceTests",
    ],
}

#: v0.10.1 markers: the P0 ASR forensic release — the idle-VAD report fix
#: (below the state machine + explicit condition), the wrong-language
#: evidence dump, the asr_dump_utterances flag, the exact transformers pin
#: (installer + bootstrap floor reality 5.9), the on-target forensic
#: script, and the two new regression suites.
V0101_MARKERS = {
    "app/web_server.py": [
        "v0.10.1 (P0 ASR forensic fix): this block moved BELOW the state",
        "elif peak < float(self._config.vad_threshold):",
        "asr_wrong_language_",
        "def _has_cyrillic",
    ],
    "app/config.py": [
        "asr_dump_utterances: bool = False",
    ],
    "config/voicemem_config.yaml": [
        "asr_dump_utterances: false",
    ],
    "requirements.txt": [
        "transformers==5.17.0",
    ],
    "scripts/install_m1.ps1": [
        "$TransformersPin = \"5.17.0\"",
    ],
    "scripts/bootstrap.ps1": [
        "sys.exit(0 if v >= (5, 9) else 5)",
    ],
    "scripts/asr_regression_forensic.py": [
        "PINNED_MODEL_SHA256",
        "asr-regression-forensic/1",
    ],
    "tests/unit/test_vad_idle_report.py": [
        "IdleVadReportTests",
    ],
    "tests/integration/test_real_asr_hungarian.py": [
        "RealParakeetHungarianTests",
    ],
}

#: v0.10.2 markers: the user-priority release - the vendor cooperative
#: cancellation gate, the BackgroundMemoryGate v2 (cancel-on-arm + requeue +
#: idle policy), the resolve-leg timeout, the ASR language diagnostic mode,
#: the openai exact pin, and the regression battery covering all of it.
V0102_MARKERS = {
    "vendor/voicemem/voicemem/utils/common/llm_bg_gate.py": [
        "BG_CANCEL = threading.Event()",
        "class BackgroundCancelledError(BaseException)",
        "def bg_chat_create",
    ],
    "app/background_memory.py": [
        "VOICEMEM_BG_IDLE_S",
        "def note_activity",
        "BackgroundCancelledError as exc:",
    ],
    "app/web_server.py": [
        'self._memory_gate.note_activity("vad-frame")',
        'self._memory_gate.note_activity("llm-delta")',
    ],
    "vendor/voicemem/voicemem/leftbrain/extract_facts_openai.py": [
        'leg="memory-extraction"',
        'leg="conflict-resolution"',
        '"timeout": 60.0, "max_retries": 2',
    ],
    "app/asr_parakeet.py": [
        "def _resolve_language_mode",
        '"language_mode": self._language_mode',
    ],
    "requirements.txt": [
        "openai==3.14.0",
    ],
    "scripts/install_m1.ps1": [
        '$OpenAiPin = "3.14.0"',
    ],
    "requirements.lock.json": [
        "voicemem-critical-path-lock/1",
    ],
    "scripts/llm_slot_forensic.py": [
        "llm-slot-forensic/1",
    ],
    "tests/unit/test_llm_bg_gate.py": [
        "BgChatCreateTests",
    ],
    "tests/unit/test_background_memory_gate.py": [
        "UserPriorityCancellationTests",
        "IdlePolicyTests",
    ],
    "tests/unit/test_asr_language_mode.py": [
        "EngineLanguageModeTests",
    ],
}

#: v0.10.3 markers: the upstream-audit selective ports - the TTS speech
#: normalisation (the reported U+201E crash), the playback-tail hearing
#: window + first-audio grace + supersede force-stop, the two-level
#: concurrent TTS synthesis pipeline, the interrupted-turn history entries,
#: and the vendor retrieval-quality quotas (heartnote cap, response_experience
#: off, query-gated recency).
V0103_MARKERS = {
    "app/text_utils.py": [
        "def normalize_for_speech",
        '"\\u201E": \'"\'',
        "_SPEECH_CHAR_MAP",
    ],
    "app/web_server.py": [
        "def _playback_tail_active",
        "def _account_sent_audio",
        "_PLAYBACK_TAIL_SLACK_S",
        "_TTS_MAX_PARALLEL_CHUNKS",
        "[interrupted]",
        "normalize_for_speech(chunk)",
    ],
    "vendor/voicemem/voicemem/leftbrain/temporal.py": [
        "def wants_recency",
        "_RECENCY_QUERY_CUES",
    ],
    "vendor/voicemem/voicemem/leftbrain/mem0_backend_store.py": [
        "q_wants_recency = wants_recency(q)",
    ],
    "vendor/voicemem/voicemem/rightbrain/brain.py": [
        '"situation_pattern": max(0, int(os.environ.get("VOICEMEM_RB_HEARTNOTE_MAX", "2")))',
        "VOICEMEM_RB_WRITE_EXPERIENCE",
    ],
    "tests/unit/test_upstream_audit_ports.py": [
        "NormalizeForSpeechTests",
        "PlaybackTailTests",
        "SourceQuotaTests",
    ],
}

#: v0.10.4 markers: the additive forensic logging — the gate armed/
#: released/open/ingest-started INFO lines (background_memory), the
#: prosody-wait diag line (web_server), the llm-slot-forensic/2 schema
#: with the offline S8 product-trace phase and the deterministic
#: verdict classes, and the two forensic test files. Logging only —
#: zero control-flow change (the release contract).
V0104_MARKERS = {
    "app/background_memory.py": [
        "background memory gate armed (reason=%s)",
        "background memory gate released (turn ended; grace=%.1fs idle=%.1fs)",
        "background memory gate open (idle window held; ",
        "background memory ingest started (turn_no=%d, %d queued, ",
    ],
    "app/web_server.py": [
        "emotion prosody done (waited)",
        "emotion prosody pending (late",
    ],
    "scripts/llm_slot_forensic.py": [
        "llm-slot-forensic/2",
        "def s8_product_trace",
        "CONTENTION_CONFIRMED",
        "PREPROCESSING_DOMINATED",
        "REAL_LLM_LATENCY",
    ],
    "tests/unit/test_llm_slot_forensic.py": [
        "class ParseTests",
        "class RebuildTests",
        "class MetricsTests",
        "class StreamProbeTests",
    ],
    "tests/unit/test_background_memory_gate.py": [
        "GateForensicLoggingTests",
    ],
}

#: v0.10.5 markers: the TTS code-switching fix - span-level language
#: routing for embedded foreign phrases at both call sites (web + CLI),
#: the segmentation machinery in text_utils, and the 42-test suite.
V0105_MARKERS = {
    "app/text_utils.py": [
        "def segment_language_spans",
        "EN_CONTRACTIONS",
        "def _classify_word",
        "def _scan_regions",
        "def _region_language",
    ],
    "app/web_server.py": [
        "segment_language_spans",
        "v0.10.4 code-switching: span segmentation ONLY in auto mode",
    ],
    "app/pipeline.py": [
        "segment_language_spans",
        "v0.10.4 code-switching: span segmentation ONLY in auto mode",
    ],
    "tests/unit/test_tts_code_switching.py": [
        "class SegmentPureLanguageTests",
        "class SegmentQuotedForeignTests",
        "class SegmentEdgeCaseTests",
        "class WebSynthesizeChunkTests",
        "class PipelineSpeakChunkTests",
    ],
}

#: v0.10.7 markers: the N1/N2 deterministic fix - the Hungarian
#: vowel-harmony guard (_hu_harmonic + HU_BACK_VOWELS/HU_FRONT_VOWELS)
#: on both budget absorptions, the "holnap" HU_PLAIN_WORDS entry, the
#: lazy host-context tie-break (_hu_host_context_function_evidence),
#: and the N1/N2 regression matrix.
V0107_MARKERS = {
    "app/text_utils.py": [
        "HU_BACK_VOWELS",
        "HU_FRONT_VOWELS",
        "def _hu_harmonic",
        "def _hu_host_context_function_evidence",
        "[v0.10.6 N2 fix] \"holnap\" joins the category",
    ],
    "tests/unit/test_tts_n1_n2_matrix.py": [
        "class N1TrailingHungarianWordTests",
        "class N2HostLanguageFlipTests",
        "test_n1_operator_case_meeting_fontos",
        "test_n2_operator_case",
    ],
}

#: v0.10.6 markers: the leading-neutral absorption fix (edac5c9) - the
#: leading budget + min-evidence shield + _absorb_leading guards in
#: text_utils, and the IdiomLeadingAbsorptionTests battery that pins
#: them (idiom heads join their evidence runs; loanword neutrals stay
#: host).
V0106_MARKERS = {
    "app/text_utils.py": [
        "PHRASE_RUN_LEADING_BUDGET",
        "PHRASE_RUN_LEADING_MIN_EVIDENCE",
        "def _absorb_leading",
    ],
    "tests/unit/test_tts_code_switching.py": [
        "class IdiomLeadingAbsorptionTests",
        "class SentenceStreamChunkBoundaryTests",
        "test_idiom_cut_to_the_chase_in_hungarian",
        "test_idiom_break_a_leg_hungarian_host_is_documented_limitation",
        "test_single_evidence_run_never_absorbs_leading_neutral",
        "test_hyphen_trimmed_candidate_never_absorbed",
    ],
}

#: v0.9.1 markers: the web VAD onset fix - the SPEECH_START trigger frame
#: is CAPTURED (no early return in the speech_start branch), CLI parity is
#: pinned by a dedicated regression test, and the gate baseline records the
#: venv-drift env-gap pins.
V091_MARKERS = {
    "app/web_server.py": [
        "v0.9.1 (web VAD onset fix): NO early return here",
        "matching the CLI path (app/main.py appends it",
    ],
    "tests/unit/test_web_vad_trigger_frame.py": [
        "TestTriggerFrameRetention",
        "test_cli_collection_parity",
        "test_trigger_frame_is_first_captured_frame",
    ],
    "scripts/run_release_gate.py": [
        "degraded-mode simulation no longer degrades (stash-verified)",
    ],
}

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
        # v0.7.2 realignment: the flag is GENERATED from the canonical
        # config (llm.reasoning.enabled=false -> --reasoning off via the
        # bridge + $ReasoningFlag); the hardcoded literal is retired.
        '$LlamaArgs += @("--reasoning", $ReasoningFlag)',
    ],
    "scripts/install_m1.ps1": [
        # v0.10.1: the floor guard became the EXACT pin (P0 forensic)
        "$TransformersPin",
        "v == (5, 17)",
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
        # v0.10.1: exact pin (P0 forensic)
        "transformers==5.17.0",
    ],
    "config/llm_config.yaml": [
        # v0.7.2 realignment: the thinking flag moved to the canonical
        # llm_config.yaml (llm.reasoning.enabled: false = thinking OFF,
        # the single switch behind the server flag + request kwargs);
        # the voicemem_config.yaml duplicate key is REMOVED.
        "enabled: false",
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
        # v0.10.1: the probe floor is 5.9 (ParakeetForTDT reality)
        "(5, 9)",
        "else 5",
        "ParakeetForTDT",
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
    "app/tts_supertonic.py": [
        "TTS engine = Supertonic 3",
        "no fallback",
        "voice: Optional[str] = None",
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
        # v0.10.1: the floating floor became the EXACT pin (P0 forensic)
        '$TransformersPin = "5.17.0"',
    ],
    "scripts/bootstrap.ps1": [
        "patch_voicemem_english.py",
        "v >= (5, 9)",
    ],
    "scripts/start_agent.ps1": [
        "localise_memories.py",
    ],
    "requirements.txt": [
        # v0.10.1: exact pin (P0 forensic)
        "transformers==5.17.0",
    ],
    "pyproject.toml": [
        # v0.10.1: exact pin (P0 forensic)
        "transformers==5.17.0",
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
        # v0.6.0: transcribe_utterance was replaced by the ENGINE CONTRACT
        # call (transcribe(AudioBuffer) / finish()); the marker below pins it
        "engine contract",
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
        "const PAGE_VERSION=",
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
    # v0.6.0: FusedVad (the v0.4.14 energy/gain fallback gate) is DELETED -
    # Silero is the single decision path, fed with the OFFICIAL 64-sample
    # rolling context (the forensic root-cause fix). Markers updated.
    "app/vad.py": [
        "class SileroVad:",
        "64-sample rolling CONTEXT",
        "reset()",
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
        'LLAMA_N_GPU_LAYERS = "20"',
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
    # v0.6.0: tests/unit/test_fused_vad.py is DELETED with FusedVad (the
    # energy fallback is gone from production; the deaf-VAD regression lives
    # on in test_vad_state_machine.py + test_asr_modular.py).
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
        "const PAGE_VERSION="
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
        "test_version_bumped"
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


def _verify_gate_record() -> dict:
    """v0.8.1 gate-order fix: load + verify releases/gate_record.json.

    The record must be GREEN, its source-tree fingerprint must equal the
    CURRENT tree's fingerprint (any post-gate change — a version bump, a
    'small fix', a page edit — invalidates it), and the record's version
    must equal NEW_VERSION and the tree's VERSION file. This is the
    mechanical guarantee that THE EXACT TREE PACKAGED IS THE TREE GATED.
    """
    record_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        RELEASES / "gate_record.json"
    )
    if not record_path.is_file():
        print(
            f"BUILD FAILED: no gate record at {record_path}. Run "
            f"'python scripts/run_release_gate.py' on the FINAL tree first "
            f"(v0.8.1 gate-order rule: the gate runs on the exact tree to "
            f"be released, AFTER the VERSION/CHANGELOG/page-version update)."
        )
        return {}
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"BUILD FAILED: gate record unreadable: {exc}")
        return {}
    if record.get("verdict") != "GREEN":
        print(
            "BUILD FAILED: gate record verdict is "
            f"{record.get('verdict')!r} (regressions: "
            f"{record.get('regressions')}) — the tree is not green."
        )
        return {}
    if record.get("version") != NEW_VERSION:
        print(
            f"BUILD FAILED: gate record version {record.get('version')!r} != "
            f"builder NEW_VERSION {NEW_VERSION!r} — bump/re-gate mismatch "
            f"(the gate must run AFTER the VERSION update)."
        )
        return {}
    sys.path.insert(0, str(REPO / "scripts"))
    from release_tree import tree_fingerprint, tree_version

    fingerprint, _n = tree_fingerprint(REPO)
    if record.get("tree_fingerprint") != fingerprint:
        print(
            "BUILD FAILED: tree changed since the gate ran "
            f"(record {str(record.get('tree_fingerprint'))[:16]}… vs now "
            f"{fingerprint[:16]}…). Re-run scripts/run_release_gate.py on "
            "the current tree. THE EXACT TREE PACKAGED MUST BE THE TREE "
            "THAT PASSED THE GATE."
        )
        return {}
    tree_ver = tree_version(REPO)
    if tree_ver != NEW_VERSION:
        print(
            f"BUILD FAILED: tree VERSION file {tree_ver!r} != NEW_VERSION "
            f"{NEW_VERSION!r} — sync the version files first, then re-gate."
        )
        return {}
    # page version must be in sync too (the P1-1 defect class)
    page = (REPO / "web" / "voicemem.html").read_bytes()
    want = f"const PAGE_VERSION='{NEW_VERSION}';".encode("ascii")
    if want not in page:
        print(
            f"BUILD FAILED: web/voicemem.html PAGE_VERSION literal is not "
            f"{NEW_VERSION!r} — run scripts/sync_page_version.py, then "
            f"re-run the gate (the gate pins VERSION == PAGE_VERSION)."
        )
        return {}
    return record


def build() -> int:
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stage_root = Path(tempfile.mkdtemp(prefix=f"voicemem_build_{NEW_VERSION}_"))
    stage = stage_root / f"VoiceMemAgent_v{NEW_VERSION}"
    stage.mkdir(parents=True, exist_ok=True)
    zip_path = RELEASES / ZIP_NAME

    # v0.8.1 gate-order fix: gate numbers come from the VERIFIED record of a
    # green run on THIS EXACT TREE (fingerprint-bound) — never from the
    # caller's memory. See _verify_gate_record.
    record = _verify_gate_record()
    if not record:
        return 1
    gate_total = int(record.get("totals", {}).get("tests", 0))
    deep_pass = record.get("deep_validation_pass") or "—"
    deep_skip = int(record.get("deep_validation_skipped", 0))
    env_gap = len(record.get("env_gap_failures", {}))
    gate_skipped = int(record.get("totals", {}).get("skipped", 0))
    if gate_total <= 0:
        print("BUILD FAILED: gate record has no test total")
        return 1

    ch_entry = (
        f"## [{NEW_VERSION}] - {today}\n"
        f"### Kiadas (ZIP-build)\n"
        f"- {NOTES}\n"
        f"- Tesztkapu: PASS a VEGSO FAJAN (v0.8.1 kapu-sorrend: verzio/"
        f"CHANGELOG/page-verzio FRISSITES UTAN futott a teljes gate; "
        f"releases/gate_record.json fingerprint-kotott): teljes "
        f"tesztkeszlet {gate_total} teszt (SKIP: {gate_skipped} db; "
        f"ismeretes sandbox kornyezeti hiba: {env_gap or 0} db — a "
        f"pinned-bazisban dokumentalt, celgepen zold); deep validacio: "
        f"{deep_pass}; SKIP: {deep_skip} db)\n"
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
        "test_gate": "run_release_gate.py (unit + integration + validation)",
        "test_gate_exit_code": 0,
        "gate_record": {
            "ran_at_utc": record.get("ran_at_utc"),
            "verdict": record.get("verdict"),
            "tree_fingerprint": record.get("tree_fingerprint"),
            "tests": gate_total,
            "failures": record.get("totals", {}).get("failures", 0),
            "errors": record.get("totals", {}).get("errors", 0),
            "skipped": gate_skipped,
            "env_gap_failures": env_gap,
            "regressions": record.get("regressions", []),
            "note": (
                "v0.8.1 gate-order rule: the gate ran on the EXACT tree that "
                "was packaged (fingerprint-bound record; every failure is a "
                "pinned+documented sandbox environment gap, green on the "
                "target machine)"
            ),
        },
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
            # v0.8.1 (post-audit P1-1): the ZIP's page version MUST equal the
            # ZIP's VERSION — the v0.8.0 release shipped PAGE_VERSION='0.7.2'
            # inside the v0.8.0 ZIP and every session showed a false stale-
            # page toast. Belt and braces on top of the serve-time injection.
            zip_version = zf.read(root_prefix + "VERSION").decode(
                "utf-8"
            ).strip()
            zip_html = zf.read(root_prefix + "web/voicemem.html").decode(
                "utf-8", errors="replace"
            )
            want_page = f"const PAGE_VERSION='{zip_version}';"
            if want_page not in zip_html:
                raise AssertionError(
                    "self-check: ZIP web/voicemem.html PAGE_VERSION != ZIP "
                    f"VERSION (expected {want_page!r}) — the served page "
                    "would show a false staleness warning"
                )
            # v0.10.0: the temporal-semantics module, the memory-semantics
            # document, and BOTH regression suites must ship in the ZIP.
            for must in (
                "vendor/voicemem/voicemem/leftbrain/temporal.py",
                "docs/MEMORY_SEMANTICS.md",
                "tests/unit/test_temporal_semantics.py",
                "tests/integration/test_temporal_memory.py",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.10.0 file missing from the ZIP: {must}"
                    )
            # v0.10.1: the P0 forensic deliverables must ship in the ZIP.
            for must in (
                "scripts/asr_regression_forensic.py",
                "tests/unit/test_vad_idle_report.py",
                "tests/integration/test_real_asr_hungarian.py",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.10.1 file missing from the ZIP: {must}"
                    )
            # v0.10.2: the LLM-priority deliverables must ship in the ZIP.
            for must in (
                "vendor/voicemem/voicemem/utils/common/llm_bg_gate.py",
                "app/background_memory.py",
                "scripts/llm_slot_forensic.py",
                "requirements.lock.json",
                "tests/unit/test_llm_bg_gate.py",
                "tests/unit/test_background_memory_gate.py",
                "tests/unit/test_asr_language_mode.py",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.10.2 file missing from the ZIP: {must}"
                    )
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
            # v0.7.0: Supertonic presets (F1-F5, M1-M5) replaced the Piper
            # voice ids; the preview sentence stays the task sentence.
            for m in ("F1", "M1", "Nyugodt n\u0151i", "Szia Thomas, ez egy hangteszt."):
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
            tts_src = zf.read(root_prefix + "app/tts_supertonic.py").decode("utf-8", errors="replace")
            if "voice: Optional[str] = None" not in tts_src:
                raise AssertionError("self-check: tts_supertonic.py lacks the voice override parameter")
            if "auto_download=False" not in tts_src:
                raise AssertionError("self-check: tts_supertonic.py must keep offline loading (auto_download=False)")
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
            for rel, markers in {**V0412_MARKERS, **V0413_MARKERS, **V0414_MARKERS, **V0415_MARKERS, **V0416_MARKERS, **V0417_MARKERS, **V0418_MARKERS, **V0419_MARKERS, **V0420_MARKERS, **V0421_MARKERS, **V050_MARKERS, **V052_MARKERS, **V060_MARKERS, **V061_MARKERS, **V062_MARKERS, **V063_MARKERS, **V064_MARKERS, **V070_MARKERS, **V071_MARKERS, **V072_MARKERS, **V080_MARKERS, **V081_MARKERS, **V090_MARKERS, **V091_MARKERS, **V092_MARKERS, **V0100_MARKERS, **V0101_MARKERS, **V0102_MARKERS, **V0103_MARKERS, **V0104_MARKERS, **V0105_MARKERS, **V0106_MARKERS, **V0107_MARKERS}.items():
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
            # v0.4.14: the mic-dispatch regression tests and the E2E
            # deaf-channel launcher must ship (v0.6.0: test_fused_vad.py is
            # DELETED together with the FusedVad energy-fallback gate)
            for must in (
                "tests/unit/test_web_mic_dispatch.py",
                "tests/e2e_deaf_server.py",
                "models/llm/qwen3.6-35b-a3b/README.md",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.4.14 file missing from the ZIP: {must}"
                    )
            # v0.6.0: the modular ASR engine contract must ship
            for must in (
                "app/asr_core.py",
                "app/asr_parakeet.py",
                "app/asr_nemotron.py",
                "tests/unit/test_asr_modular.py",
                "tests/unit/test_vad_state_machine.py",
                "tests/e2e_fake_mic.js",
                "docs/TASK_A_ASR_GATE.md",
                "scripts/asr_benchmark.py",
                "scripts/asr_forensics.py",
                "scripts/gen_asr_bench.py",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.6.0 file missing from the ZIP: {must}"
                    )
            # v0.7.0: the Supertonic 3 TTS migration must ship
            for must in (
                "app/tts_supertonic.py",
                "tests/unit/test_tts_supertonic.py",
                "tests/unit/test_voice_settings.py",
                "tests/validation/test_feature_tts.py",
                "tests/benchmark/tts_benchmark.py",
                "scripts/validate_supertonic.py",
                "scripts/asr_roundtrip_tts.py",
                "models/tts/supertonic-3/README.md",
                "models/tts/supertonic-3/.gitkeep",
            ):
                if root_prefix + must not in names:
                    raise AssertionError(
                        f"self-check: v0.7.0 file missing from the ZIP: {must}"
                    )
            # v0.7.0: the RETIRED Piper module must NOT ship
            if root_prefix + "app/tts.py" in names:
                raise AssertionError(
                    "self-check: app/tts.py (Piper) must NOT ship - the "
                    "production TTS is Supertonic 3 (no fallback)"
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
            # v0.7.1 (field report #5): every zipped .ps1 must pass the
            # PowerShell structural lint (brace/string/here-string balance,
            # PS7-only operator scan). The v0.7.0 build shipped
            # install_m1.ps1 with ONE missing closing brace and only the
            # target machine's Windows PowerShell 5.1 parser caught it -
            # this check makes that defect class unshippable forever.
            sys.path.insert(0, str(REPO / "scripts"))
            from ps_lint import check_ps_source  # repo-local stdlib tokenizer

            installer_src = zf.read(root_prefix + "scripts/install_m1.ps1")
            if b"\r\n" not in installer_src:
                raise AssertionError(
                    "self-check: install_m1.ps1 lost its CRLF endings"
                )
            ps1_names = [
                n for n in names
                if n.startswith(root_prefix) and n.endswith(".ps1")
            ]
            if len(ps1_names) < 10:
                raise AssertionError(
                    "self-check: implausibly few .ps1 files in the ZIP"
                )
            for n in ps1_names:
                raw = zf.read(n)
                if b"piper_exe_path" in raw:
                    raise AssertionError(
                        f"self-check: {n} references the removed "
                        "AgentConfig.piper_exe_path attribute"
                    )
                lint_problems = check_ps_source(
                    raw.decode("ascii", errors="replace")
                )
                if lint_problems:
                    raise AssertionError(
                        f"self-check: {n} fails the PowerShell structural "
                        f"lint (would be a ParserError on Windows): "
                        f"{lint_problems[:3]}"
                    )
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
            "test_gate": "run_release_gate.py (v0.8.1 gate-order fix)",
            "test_gate_exit_code": 0,
            "gate_record": {
                "ran_at_utc": record.get("ran_at_utc"),
                "verdict": record.get("verdict"),
                "tree_fingerprint": record.get("tree_fingerprint"),
                "tests": gate_total,
                "env_gap_failures": env_gap,
                "regressions": len(record.get("regressions", [])),
            },
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
