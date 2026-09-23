# Production llama.cpp b11073 runtime upgrade — VoiceMem v0.10.7

**Status: production runtime upgrade.** This pack replaces the production
llama.cpp server runtime (pinned **b10717**, `llama-server 0.3.0-dev (build
10717, commit a32af33de)`) in `bin\` with the **already installed and already
verified** llama.cpp **b11073** build (`llama-server 0.4.1-dev (build 11073,
commit 1aa2954bd)` — expected build identity **`b11073-1aa2954bd`**), while
keeping, byte for byte:

- the **Qwen3.6 35B A3B IQ4_XS** model at
  `C:\AI_HOME\models\blobs\sha256-afc7238af403ed454b7846454091b5e38b07575bdbb64c3e86777414dde4193c`
  (never moved, re-downloaded, re-quantised or modified);
- the **VoiceMem application code** (no `.py` / `.ps1` / config change —
  verified by a before/after hash snapshot of every source file);
- the **production endpoint** `127.0.0.1:8080` and the **production profile**
  (`-ngl 16 -c 16000 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0
  --temp 0.7 --reasoning off --metrics --no-webui`);
- the **existing production startup mechanism**
  (`scripts\start_llama_server.ps1` behind `START.bat`).

**This is a SERVER RUNTIME upgrade, NOT a model change.** The purpose is a
clean production baseline of **OLD MODEL + NEW SERVER**, to be compared
against the previous **OLD MODEL + OLD SERVER** behaviour (the ~637-char
repeated-generation / Hungarian-English TTS oscillation class observed in the
experimental production-chain run). No prompt, memory, language-routing,
TTS-routing or VoiceMem behaviour is modified by this pack.

**The pack contains NO binaries and performs NO download.** The runtime
source is the experimental install that the rev-2 installer pack already
placed and verified on this machine (`experimental\llama_b11073\bin\`,
24/24 SHA-256 verified against the pin table + the `1aa2954bd` build identity
inside `llama-server-impl.dll`).

```
VoiceMem v0.10.7
   |
   +-- bin\                                    <- PRODUCTION runtime (b10717 -> b11073 by this pack)
   |     +-- llama-server.exe + 23 pinned DLLs/notices   (replaced)
   |     +-- cudart*/cublas* CUDA runtime DLLs           (PRESERVED - the b11073 ggml-cuda.dll resolves them here)
   |
   +-- experimental\llama_b11073\              <- the verified runtime SOURCE (unchanged, reused in place)
   |
   +-- backups\runtime_pre_b11073_<ts>\        <- timestamped rollback anchor (created by this pack)
   |
   +-- ops\llama_b11073_production_upgrade\    <- this pack
```

## Pack contents (scripts + documentation ONLY)

| File | Role |
| --- | --- |
| `RUN_PRODUCTION_UPGRADE.bat` | **one-click entry**: runs the orchestrator with the Bypass policy (the `START.bat` 64-bit PowerShell resolution), prints the evidence ZIP path and the rollback command |
| `upgrade_production_b11073.ps1` | the orchestrator: preflight → audit → stop → backup → install → verify → start (through the existing starter) → server battery (health / models / Hungarian chat / streaming / JSON / fingerprint) → VoiceMem integration (endpoint gate + guided production-chain turns with log capture) → regression → evidence report. Modes: `-VerifyOnly`, `-AuditOnly`, `-ProbeOnly`, `-SkipIntegration`, `-NoAutoRollback`, `-FunctionsOnly` |
| `rollback_production_b11073.ps1` | standalone rollback: restore `bin\` from the timestamped backup, restart through the existing starter, verify (`/health`, `/v1/models`, a real completion, the live fingerprint must NOT be b11073) |
| `config/prod_upgrade_expected.json` | the expected identity table (b11073 / 1aa2954bd / asset SHA-256 / the 24 per-file pins, cross-checked against `experimental\llama_b11073\config\b11073.pins.json` and the install `runtime_manifest.json` — all three must agree), the documented production profile and the model path |
| `SHA256SUMS.txt` | checksums of the pack files |

## Prerequisites

1. An installed VoiceMem **v0.10.7** tree with the **experimental b11073
   runtime already installed and verified** (the rev-2 installer pack:
   `experimental\llama_b11073\bin\` present with its
   `config\runtime_manifest.json`). If the manifest is missing, re-run
   `experimental\llama_b11073\install_b11073.ps1` first (idempotent, no
   re-download).
2. The production model present at the blob path above.
3. 64-bit Windows PowerShell 5.1 or PowerShell 7 (the `START.bat` chain
   already guarantees this).
4. Free disk space for the timestamped backup of `bin\` (the runtime +
   CUDA DLL set, a few hundred MB).

## Running the upgrade (exact commands)

From a normal command prompt in the project root:

```bat
cd /d F:\Voicemem\VoiceMemAgent

:: (recommended first) pre-flight identity proof of the runtime source,
:: no changes, no server touched:
ops\llama_b11073_production_upgrade\RUN_PRODUCTION_UPGRADE.bat -VerifyOnly

:: (optional) record the current production state only, no changes:
ops\llama_b11073_production_upgrade\RUN_PRODUCTION_UPGRADE.bat -AuditOnly

:: THE UPGRADE (one click, interactive during the integration stage):
ops\llama_b11073_production_upgrade\RUN_PRODUCTION_UPGRADE.bat
```

Direct PowerShell equivalents (policy-independent):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\upgrade_production_b11073.ps1 -VerifyOnly
powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\upgrade_production_b11073.ps1
```

During **[S8] integration** the script starts the normal web backend (or uses
the running one), verifies its LLM endpoint resolves to
`http://127.0.0.1:8080/v1`, then asks you to perform three turns in the web
UI (`http://127.0.0.1:8787/`):

1. a short **Hungarian conversational request** (voice),
2. a Hungarian request **containing English expressions** (voice),
3. one **normal voice turn** through the full chain (VAD → ASR → VoiceMem →
   E5 → LLM → TTS).

The `[chain]` trail (`logs\web-server.log`) is captured for exactly this
window and the following metrics are extracted into the evidence: ASR
transcript, LLM first-token latency, total LLM latency, reply length, TTS
chunk count, first-audio latency, long-generation (>400 chars) flags,
repeated-generation suspicion flags (>600 chars — the class of the observed
~637-char oscillation), and any `:8081`/`:8082` endpoint leakage (which must
be zero).

## What the upgrade does, stage by stage

| Stage | What happens | Fails how |
| --- | --- | --- |
| S0 preflight | the pack config × experimental pin table × install manifest cross-check (all three must agree); the SOURCE runtime verified 24/24 SHA-256 + build id `1aa2954bd` | loud abort, nothing touched |
| S1 audit | old `llama-server.exe` SHA-256 + version info, `--version` output (**informational only — never a gate**, the rev-1 experimental launcher lesson), in-binary anchors, live PID/command line/`/health`/`/v1/models`, CUDA DLL hashes, model integrity marks (size + mtime + head/tail hash), a hash snapshot of every python source + launcher + config | abort if the audited command line deviates from the documented profile (no silent parameter change) or the model is missing |
| S2 stop | only the `llama-server.exe` listening on 8080 is stopped (never unrelated processes); the port is confirmed free | loud abort if the port owner is something else |
| S3 backup | full timestamped copy of `bin\` into `backups\runtime_pre_b11073_<ts>\` with a per-file hash manifest | abort if the backup cannot be verified |
| S4 install | the 24 pinned files are copied into `bin\`; stale llama.cpp-family files not in the pinned set are removed (they are safe in the backup); the CUDA runtime DLLs are **preserved** (the b11073 `ggml-cuda.dll` resolves `cublas64_13.dll` from `bin\` — the exact mechanism the experimental launcher proved on this machine) | abort + rollback on any file error |
| S5 verify | the installed set verified again (24/24 + build id), CUDA presence checked | rollback |
| S6 start | the server is started **through the existing `scripts\start_llama_server.ps1`** (its own window, config-driven, unchanged); `/health` waited for; then the listener PID must be `bin\llama-server.exe`, the relaunched command line must comply with the documented profile, and `/v1/models` must serve the same model | rollback |
| S7 battery | `/health`; `/v1/models`; a Hungarian chat completion (non-stream); streaming (first token + clean `[DONE]`); a JSON completion (`response_format: json_object`); `system_fingerprint == b11073-1aa2954bd` | rollback on any failure |
| S8 integration | the normal web backend started/used, its LLM endpoint verified as `http://127.0.0.1:8080/v1` (an experimental-session backend pointing at 8081/8082 is replaced by a normal one), then the guided turns + chain-log metrics | rollback on connection/endpoint failure; behavioural observations are recorded, not gated |
| S9 regression | port 8080 still bound by `bin\llama-server.exe`, model unchanged, b11073 identity re-confirmed, the python-source/config hash snapshot compared (must be identical), model file integrity marks unchanged, no `:8081`/`:8082` in the production command line, `START.bat` + starter intact | rollback for the hard gates; source/model anomalies are reported as CRITICAL warnings |
| S10 report | `REPORT.md` + `stage_results.json` + raw responses + the chain-log slice + `prod_b11073_upgrade_evidence.zip` in `logs\prod_b11073_upgrade_<ts>\` | — |

## Rollback

Automatic: any server-startup / CUDA / `/health` / `/v1/models` / chat /
streaming / VoiceMem-connection failure restores the timestamped backup and
restarts the old production server (disable with `-NoAutoRollback` for
diagnosis). Manual:

```bat
cd /d F:\Voicemem\VoiceMemAgent
powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\rollback_production_b11073.ps1
:: or with an explicit backup:
powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\rollback_production_b11073.ps1 -BackupDir "F:\Voicemem\VoiceMemAgent\backups\runtime_pre_b11073_<ts>"
```

The rollback verifies `/health`, `/v1/models` (same Qwen3.6 model), a real
completion round-trip, and that the live `system_fingerprint` is **no longer**
b11073.

## Idempotency

Re-running the upgrade on an already-upgraded tree is safe: if `bin\` already
passes the 24/24 + build-id verification, no backup/replace happens and the
run continues with verify → start → battery → integration.

## Evidence — what to send back

After a successful run:

```
logs\prod_b11073_upgrade_<ts>\prod_b11073_upgrade_evidence.zip
```

contains `REPORT.md` (before/after identity, SHA-256s, command lines, test
results, rollback status, modified-file list), `audit_before.json`,
`source_snapshot_before/after.json`, `chat_test.json`, `streaming_test.json`,
`json_test.json`, `integration/web-server.slice.log`, the master log and the
console transcript. Please send the ZIP back (chat upload) so it can be
published to the audit record on the mirror.

## Notes and known limits

- `--version` output is collected and logged **informationally** in the
  audit; the identity gates are the 24-file SHA-256 pin table, the
  `1aa2954bd` build id inside `llama-server-impl.dll`, and the live
  `system_fingerprint` — never a version banner (the rev-1 experimental
  launcher bug is deliberately not repeated here).
- `MODELS.lock.json` still records the b10717 pin after this runtime swap.
  That file belongs to the product source and is deliberately NOT modified
  by this pack; recording the new runtime pin belongs to a future product
  release.
- The experimental 8081 profile is untouched; nothing in this pack points
  VoiceMem at 8081/8082 (verified as a regression gate).
- The three integration turns are performed by the operator through the web
  UI (the voice chain needs the microphone); the script captures and analyses
  the `[chain]` trail rather than synthesising a turn.
