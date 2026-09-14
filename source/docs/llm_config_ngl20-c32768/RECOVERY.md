# RECOVERY — VoiceMemAgent_v0.6.0_LLMConfig_ngl20-c32768

Production LLM configuration change: **llama-server n_gpu_layers 26 -> 20,
context 8192 -> 32768**. Everything else in the LLM profile is UNCHANGED
(parallel 1, KV cache q8_0/q8_0, temperature 0.7, reasoning off at BOTH
layers, Qwen3.6 35B A3B IQ4_XS, llama.cpp b10717).

## Commit / base

- Baseline commit: `34c5df9` (the restored workspace state)
- Change commit:   `dd9fccd` (this package)
- Full diff:       `patches/production-config-change.patch` (34c5df9..dd9fccd)
- Changed files:   `patches/files/...` (full copies, extract-over-install safe
  EXCEPT env.local.ps1 - see the note below about LLAMA_MODEL_PATH)

## Why (measured decision)

| profile | TTFT | total |
|---|---|---|
| ngl20 + 8192  | ~4.90 s | ~7.53 s |
| ngl20 + 32768 | ~4.89 s | ~7.82 s |

The 32K context costs ~nothing measured, and production requests above 8K
tokens already occurred (context overrun). Production target: **ngl20 + 32768**.

## New effective llama-server command line

```
F:\Voicemem\VoiceMemAgent\bin\llama-server.exe ^
  --model "<resolved GGUF path>" ^
  --host 127.0.0.1 --port 8080 ^
  -ngl 20 -c 32768 --parallel 1 ^
  --cache-type-k q8_0 --cache-type-v q8_0 ^
  --temp 0.7 --metrics --no-webui --reasoning off
```

`<resolved GGUF path>` is the operator-configured selection (resolution order:
`config/llm_model.json` > `LLAMA_MODEL_PATH` env > the models dir). The exact
blob on the target machine loads IN PLACE:
`C:\AI_HOME\models\blobs\sha256-afc7238af403ed454b7846454091b5e38b07575bdbb64c3e86777414dde4193c`
This change does NOT touch model selection - verify with the
`logs/llama-server.resolved-model.json` marker after startup.

## Files changed (functional values)

| file | old | new |
|---|---|---|
| `config/voicemem_config.yaml` | `llm_context_size: 8192`, `llm_n_gpu_layers: 26` | `32768`, `20` |
| `config/env.local.ps1` | `$env:LLAMA_CONTEXT_SIZE = "8192"`, `$env:LLAMA_N_GPU_LAYERS = "26"` | `"32768"`, `"20"` |
| `config/env.local.sh` | `LLAMA_CONTEXT_SIZE=8192`, `LLAMA_N_GPU_LAYERS=26` | `32768`, `20` |
| `scripts/start_llama_server.ps1` | fallback `$FbNgl = "26"`, `$FbCtx = "8192"` | `"20"`, `"32768"` |
| `scripts/verify_m1.ps1` | direct fallback `"-ngl","-1"`, `"-c","8192"` | `"-ngl","20"`, `"-c","32768"` |

Plus consistency-only updates: `scripts/build_release_sandbox.py` (release
self-check marker), `tests/validation/test_feature_scripts.py` (expectations
realigned to the new profile), `models/llm/qwen3.6-35b-a3b/README.md` +
`README.md` (current-state profile docs). Historical field records in README.md
were deliberately left as-is.

## APPLY (field machine, F:\Voicemem\VoiceMemAgent)

1. Close the agent (Ctrl+C in the llama-server + agent windows).
2. Copy the nine files from `patches/files/` over the install
   (same relative paths) - **EXCEPT `config/env.local.ps1`**:
   if your `env.local.ps1` has a LIVE `$env:LLAMA_MODEL_PATH = "..."` line
   (written by `find_qwen_gguf.ps1 -SetEnv` / `identify_ollama_blob.ps1
   -SetEnv` / `-Select`), edit ONLY these two lines instead of copying the file:
   `$env:LLAMA_CONTEXT_SIZE = "32768"` and `$env:LLAMA_N_GPU_LAYERS = "20"`.
   (`config/llm_model.json`, if present, is untouched by this package.)
3. Run `START.bat` normally.

## VERIFY ON THE TARGET MACHINE (operator checklist)

1. BEFORE start:
   `tasklist /FI "IMAGENAME eq llama-server.exe"`  -> 0 instances
2. Run `START.bat` (the normal production flow). Wait for the
   "PASS: a llama-server felfutott" line before doing anything else.
3. AFTER start:
   `tasklist /FI "IMAGENAME eq llama-server.exe"`  -> EXACTLY 1 instance.
   Two instances = re-run of START.bat during the (minutes-long) model load
   (the known /health-guard blind window) - stop the YOUNGER one:
   `Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
    Select ProcessId,CreationDate,CommandLine`
4. `logs/llama-server.out.log` / `.err.log` must report:
   - `offloaded 20/42 layers to GPU`   (n_gpu_layers)
   - `n_ctx = 32768` and `n_ctx_slot = 32768`
   - `llama_kv_cache: size = ... ( 32768 cells, 42 layers, 1/1 seqs), K (q8_0): ..., V (q8_0): ...`
   - `init: chat template, thinking = 0`   (reasoning off)
   - exactly one listening slot (`/props` -> `total_slots: 1`)
5. Model identity (no silent substitution):
   `logs/llama-server.resolved-model.json` -> the sha256-afc7238... blob path;
   `GET http://127.0.0.1:8080/v1/models` -> the same path as served id;
   the starter's three load proofs (served id + own log line + completion
   round-trip) all PASS in the starter transcript.
6. Application request flags: the web UI "LLM" panel / `logs/web-server.log`
   show the request path with `chat_template_kwargs.enable_thinking=false` +
   `reasoning_effort "none"` (see evidence/server-log-wire-dump.txt for the
   server-side view: `reasoning_format` + closed thought channel in the
   generation prompt).
7. Normal smoke test (normal interaction ONLY - no forensic tooling):
   speak a turn -> ASR transcript appears -> memory (VoiceMEM/E5 chips) ->
   LLM reply -> TTS audio. No context-init error, no layer-fit error.
8. Record the LLM timing from the NORMAL application logs
   (`logs/web-server.log` LLM lines: first-token / total) - production
   observation only; this package deliberately does NOT run benchmarks.

## Sandbox verification already done (this package)

- Config resolution (the starter's python-bridge replica): ctx 32768 / ngl 20 /
  parallel 1 / temp 0.7 / q8_0/q8_0 - BOTH with the production env AND with
  the yaml alone (evidence/bridge_proof.txt).
- Live llama.cpp **b10717 a32af33de** (the EXACT MODELS.lock.json pin, ubuntu
  build) started with the production command line on a Qwen3-0.6B Q4_K_M
  stand-in (the 19 GB production GGUF cannot run in the sandbox): log reports
  `n_ctx = 32768`, `n_ctx_slot = 32768`, `offloaded 20/29 layers` (the flag
  semantics; the production 42-layer model reports 20/42), K/V q8_0,
  `thinking = 0`, `total_slots: 1`; EXACTLY ONE llama-server process during
  the proof, 0 after clean stop (evidence/server-log-*.txt, process_check.txt).
- `app/llm.py` LlmClient (the production request path) live round trip:
  REQUEST-FLAGS `{"chat_template_kwargs": {"enable_thinking": false},
  "reasoning_effort": "none"}`, non-empty content reply, 0 reasoning chunks
  (evidence/llm_client_proof.txt).
- Targeted tests: 159/159 green
  (evidence/targeted_tests.txt). `test_feature_release` has 2 PRE-EXISTING
  failures (missing `releases/*.zip` after the sandbox storage rollback) -
  proven identical at git HEAD `34c5df9` before this change.
- Sandbox stand-in timing (NOT production-representative): TTFT 0.09-0.35 s,
  total 0.46-0.63 s on CPU with the 0.6B model. The production numbers are the
  measured ngl20 rows in the table above.

## ROLLBACK

Method A (repo): `git revert dd9fccd`
Method B (field install, manual line edits - restore the OLD profile):
- `config/voicemem_config.yaml`:
  `llm_context_size: 8192`, `llm_n_gpu_layers: 26`
- `config/env.local.ps1`:
  `$env:LLAMA_CONTEXT_SIZE = "8192"`, `$env:LLAMA_N_GPU_LAYERS = "26"`
- `scripts/start_llama_server.ps1` fallback:
  `$FbNgl = "26"`, `$FbCtx = "8192"`
- `scripts/verify_m1.ps1` direct fallback: `"-ngl", "-1"`, `"-c", "8192"`
  (the pre-change values, including the stale -1)
Then restart `START.bat` and confirm `n_ctx = 8192` / `offloaded 26/42`.
Method C: re-extract `config/` from the v0.6.0 release ZIP
(`VoiceMemAgent_v0.6.0.zip`, SHA-256 9C55D9E194F8E15DDF2EF01481E12CA5993C769126B1A86BCF731B2A0CE46C4),
keeping your `LLAMA_MODEL_PATH` / `llm_model.json` selection intact.

## Scope guard (what this change does NOT touch)

Model file/blob, model selection, llama-server binary/version, ASR, TTS,
memory semantics, prompts (the 14000-char prompt budget deliberately stays),
UI, forensic tooling, vendor/. The known Budapest-location memory issue is
NOT addressed here (noted for later, out of scope by instruction).
