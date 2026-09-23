# Experimental llama.cpp b11073 runtime — installer/configurator pack (NOT production)

**Status: EXPERIMENTAL — reversible — no production release.** This pack
integrates and validates the operator-tested newer llama.cpp build **b11073**
(llama-server `0.4.1-dev (build 11073, commit 1aa2954bd)`) next to the
untouched VoiceMem **v0.10.7** production baseline (pinned **b10717**,
`llama-server 0.3.0-dev (build 10717, commit a32af33de)`).

**The pack contains NO binaries.** Following the SAME MODEL as the existing
VoiceMem installation (`START.bat` → bootstrap downloads the production
runtime), the b11073 runtime is **downloaded at installation time** from its
authoritative release source, with the exact release, asset and SHA-256
pinned, and the install fails loudly on any mismatch — a different version
is never silently used.

```
VoiceMem v0.10.7
   |
   +-- production llama-server (b10717, bin\llama-server.exe, 127.0.0.1:8080)  UNTOUCHED
   |
   +-- experimental llama-server (b11073, experimental\llama_b11073\bin\, 127.0.0.1:8081)  THIS
```

## Pack contents (scripts + documentation ONLY)

| File | Role |
| --- | --- |
| `RUN_EXPERIMENTAL.bat` | **one-click entry**: installs if needed → enables → starts the verified experimental server (window 1) → waits for `/health` → starts the app session (window 2) |
| `install_b11073.ps1` | installer: pinned download → SHA-256 gate → staged extraction → 24-file measured closure → per-file hash + build-identity verification → manifest (re-running it on an installed runtime REPAIRS a missing manifest without re-downloading) |
| `configure_b11073.ps1` | configurator: `-Enable` (operator-local override + generated app-session script) / `-Disable` (stop server + remove override) / status |
| `start_llama_server_experimental.ps1` | launcher (rev 2): verifies the pinned runtime BEFORE start (24-file SHA-256 against `runtime_manifest.json`, cross-checked against the pack pin table, + in-binary build identity), starts the server, waits for `/health`, then verifies the LIVE identity (listener PID = the launched child + `system_fingerprint`); `-VerifyOnly` = pre-flight identity proof without starting |
| `verify_b11073.ps1` | end-to-end verification (install integrity, production-untouched pins, listener identity, live server, client transport; `-ThroughAppClient` runs the real app client) |
| `check_environment.ps1` | read-only prerequisite probe (runtime files, pinned hashes, `cublas64_13.dll` resolution, VC++/UCRT, driver, model resolution) |
| `config/b11073.pins.json` | the pinned runtime identity: tag, asset URL, asset SHA-256, per-file SHA-256 table, experimental profile values |
| `DEPENDENCY_INSPECTION.md` | the measured evidence behind the 24-file closure (every DLL classified A = target-provided / B = application-local) |
| `logs/` | installer + launcher logs land here |

NOT in the pack (by design): `llama-server.exe`, llama.cpp binaries, ggml
DLLs, CUDA DLLs, NVIDIA driver files, GGUF models, Python/venv, the VoiceMem
application. Nothing else is needed: the installer downloads the runtime, and
the target machine already provides the NVIDIA/CUDA environment.

## Prerequisites

1. An installed VoiceMem **v0.10.7** tree (`START.bat` ran at least once:
   `.venv`, models, production `bin\llama-server.exe` + cudart DLL set).
2. 64-bit Windows PowerShell 5.1 or PowerShell 7 (the `START.bat` chain
   already guarantees this).
3. Internet access at INSTALLATION time (GitHub release download) — or a
   pre-downloaded copy of the pinned asset passed via `-ZipPath`.
4. `cublas64_13.dll` resolvable at RUN time — the single CUDA DLL the
   runtime imports. Expected source: the existing production `bin\`
   (the b10717 cudart install) and/or a system CUDA 13.x. The installer and
   the launcher PROBE and report it; they never package or modify it.

## Install (from the VoiceMemAgent v0.10.7 root)

```powershell
# 1) extract this pack into the v0.10.7 root so that
#    experimental\llama_b11073\ appears next to START.bat

# 2) install the pinned runtime (~150 MB download, SHA-256-gated):
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\install_b11073.ps1

#    offline / slow link variant (the zip is STILL hash-verified):
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\install_b11073.ps1 -ZipPath C:\Downloads\llama-b11073-bin-win-cuda-13.4-x64.zip

# 3) optional, any time: read-only prerequisite probe
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\check_environment.ps1
```

What the installer does (and never does):

* downloads **only** `llama-b11073-bin-win-cuda-13.4-x64.zip` (150,093,926
  bytes, SHA-256 `85c1b874180faec412ccbba16ee0833062c28e2bf1390b12ed30f7dc6d7d79c4`)
  from the llama.cpp **b11073** release — never the cudart companion (the
  target already provides CUDA 13.x);
* verifies SHA-256 **before** extraction and fails closed on mismatch —
  including a size check, a per-file hash gate after extraction, and a
  build-identity check (the `1aa2954bd` commit string inside
  `llama-server-impl.dll`);
* installs the **24-file measured closure** into
  `experimental\llama_b11073\bin\` (see `DEPENDENCY_INSPECTION.md`);
* writes `config\runtime_manifest.json` and an install log under `logs\`;
* re-run on an **already installed** runtime: verifies the 24 hashes again
  and **repairs a missing manifest in place, without re-downloading**;
* **never** writes anywhere outside `experimental\llama_b11073\` — the
  production `bin\llama-server.exe` hash is recorded and the install logs
  prove it unchanged.

## Enable / configure (operator-local override only)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\configure_b11073.ps1 -Enable
```

`-Enable` verifies the runtime is installed, writes the flag
`config\experimental_enabled.json` and **generates**
`experimental\llama_b11073\start_voicemem_experimental.ps1` — the
operator-local app session that:

1. dot-sources the standard `config\env.local.ps1` first (offline mode, HF
   cache isolation, memory roots — the full standard environment);
2. overrides only the LLM endpoint through the **existing** mechanism
   (`app/config.py apply_env`): `LLAMA_SERVER_HOST=127.0.0.1`,
   `LLAMA_SERVER_PORT=8081`, and `OPENAI_BASE_URL=http://127.0.0.1:8081/v1`
   (the vendor bridge legs follow it — and because it is set,
   `scripts\start_agent.ps1` will NOT re-source `env.local.ps1`, so the
   override survives the bootstrap chain);
3. starts the app through the **existing starter**
   `scripts\start_agent.ps1 -Web -NoServer` — `-NoServer` is the documented
   switch for "someone already started the llama-server manually", which is
   exactly the experimental case (the production 8080 auto-start is skipped,
   so the 19 GB production model is not loaded twice).

No production config file is modified — not `config\llm_config.yaml`, not
`config\env.local.ps1`, not `MODELS.lock.json`.

## Start (the validated experimental profile)

**One click** (installs/enables if needed, then starts both windows):

```bat
experimental\llama_b11073\RUN_EXPERIMENTAL.bat
```

Manual, two windows:

```powershell
# window 1 — the experimental llama-server (b11073 on 127.0.0.1:8081):
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\start_llama_server_experimental.ps1

# pre-flight identity proof alone (no server start):
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\start_llama_server_experimental.ps1 -VerifyOnly

# window 2 — the application pointed at it (generated by -Enable):
experimental\llama_b11073\start_voicemem_experimental.ps1
```

Generated server command line (the operator-tested baseline — ngl 99 /
ctx 16000 / parallel 1 / threads 12 / reasoning off / KV q8_0 / temp 0.7,
`--metrics --no-webui`):

```
experimental\llama_b11073\bin\llama-server.exe --model <resolved GGUF> --host 127.0.0.1 --port 8081 -ngl 99 -c 16000 --parallel 1 -t 12 --cache-type-k q8_0 --cache-type-v q8_0 --temp 0.7 --reasoning off --metrics --no-webui --verbose
```

**How the launcher verifies the runtime (rev 2 — identity chain, NOT a
`--version` banner):** before start, `runtime_manifest.json` (written by the
installer) is cross-checked against the pack's `config\b11073.pins.json`,
then **every one of the 24 pinned files** (llama-server.exe + the required
DLL set + the OpenMP license notice) must match its pinned SHA-256, and
`llama-server-impl.dll` must carry the b11073 build identity string
`1aa2954bd` (the same byte-scan the installer used on the target). Anything
else is rejected before start — the launcher accepts ONLY the exact pinned
artifact. `--version` output is still collected and logged, informational
only. After `/health` turns OK, the LIVE identity is verified: the process
LISTENING on 8081 must be exactly the child this launcher started (listener
PID via `Get-NetTCPConnection`/`netstat`, `Win32_Process` executable-path
cross-check), and the running server must self-identify as b11073 through
its `system_fingerprint` (a real 1-token completion must report
`b11073-…`, retried up to 3 times). The idempotent "already running" exit
passes the same live-identity checks. The effective configuration (resolved
model, flags, PID, hashes, verification results) is written to
`logs\launcher-config_<timestamp>.log`.

## Verify

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\verify_b11073.ps1
# structural only (no server needs to run):
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\verify_b11073.ps1 -SkipServer
# + the REAL application client (app\llm.py through AgentConfig + env override):
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\verify_b11073.ps1 -ThroughAppClient
```

The verification proves: the b11073 binary was downloaded and every file's
SHA-256 matches the pinned table; the build identity is b11073; the
production baseline is untouched (production exe hash, `MODELS.lock.json`
still b10717, `VERSION` still 0.10.7, canonical config still ngl 20 /
ctx 32768); `/health` on 8081 is OK; the process listening on 8081 is the
pinned experimental binary; the running server identifies as b11073; and
the OpenAI-compatible chat-completions transport the application uses
answers.

## Rollback (trivial)

```powershell
# 1) stop the experimental server + remove the override + restore report:
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\configure_b11073.ps1 -Disable
```

After `-Disable` the app targets `127.0.0.1:8080` again in any new terminal
(the override only ever lived inside the generated session script — nothing
persisted anywhere else). The normal `START.bat` flow never changed. For a
full cleanup, delete `experimental\llama_b11073\` — no production file,
config value, version number or release artefact was ever changed.

## Fix history

**Rev 2 (2026-09-23) — launcher identity-verification fix.** Field feedback
from the target: the rev-1 launcher ran `llama-server.exe --version` and
required the output to contain "11073"; on the real target (Windows
PowerShell 5.1) that check failed with an empty/non-matching result even
though the installed runtime was the correctly pinned, hash-verified b11073
build. Fix: the gate is no longer the `--version` banner. The launcher now
verifies the established authoritative identity — the pinned
`runtime_manifest.json` + per-file SHA-256 (all 24 files) + the
`1aa2954bd` build identity inside `llama-server-impl.dll` — BEFORE start,
and the LIVE server identity (listener PID + `system_fingerprint`) AFTER
`/health`. Verification was strengthened, not weakened; the experimental
profile, port 8081, the model, the production runtime and the application
code are all unchanged. The installer additionally learned to repair a
missing manifest in place (no re-download).

**Exact target-machine retry when the runtime is ALREADY installed** (the
already-downloaded files are valid and are NOT re-downloaded):

```bat
cd /d F:\Voicemem\VoiceMemAgent

:: 1) overwrite the pack scripts with the fixed pack (extract the ZIP over
::    the root - it contains ONLY scripts + docs; bin\ and the manifest stay):
powershell -NoProfile -Command "Expand-Archive -LiteralPath $env:USERPROFILE\Downloads\VoiceMemAgent_v0.10.7_LlamaB11073_Installer.zip -DestinationPath . -Force"

:: 2) (idempotent, no re-download; repairs the manifest if it is missing)
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\install_b11073.ps1

:: 3) pre-flight identity proof (optional but recommended):
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\start_llama_server_experimental.ps1 -VerifyOnly

:: 4) one click:
experimental\llama_b11073\RUN_EXPERIMENTAL.bat
```

(Step 1 assumes the fixed ZIP was downloaded and verified with its published
SHA-256; adjust the path as needed. A completely fresh machine uses the same
commands — step 2 then downloads the pinned runtime.)

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| installer: `SHA-256 ELTERES` | corrupted or wrong download — delete and retry, or check `-ZipPath`. NEVER continue with a different version. |
| installer: `cublas64_13.dll NINCS a loader keresesi utvonalan` (warning) | the production cudart set is not in `bin\` yet — run `START.bat` once (installs b10717 + cudart), or install a system CUDA 13.x. |
| launcher: server exits instantly, `0xC0000135` in the log | missing `cublas64_13.dll` — see above; run `check_environment.ps1` for the probe. |
| launcher: `a telepitett runtime NEM a pin-elt b11073: <file> (vart …, kapott …)` | a runtime file changed/was replaced — the launcher only accepts the exact pinned artifact. Delete `experimental\llama_b11073\bin\` and re-run `install_b11073.ps1 -Force`. |
| launcher: `a(z) 8081/-on figyelo PID … NEM a pin-elt b11073 binaris` | something else is listening on 8081 — the launcher refuses to trust (and never kills) unknown listeners. Check `netstat -ano | findstr :8081`, or `configure_b11073.ps1 -Disable`. |
| launcher: `a 8081-es szerver NEM azonosithato b11073-kent (fingerprint = '…')` | the verified binary started a server that does not self-identify as b11073 — report it; meanwhile `configure_b11073.ps1 -Disable`. |
| `--version` prints nothing / no build tag | EXPECTED on the target — informational only since rev 2; the identity is proven by the SHA-256 manifest + build id + live fingerprint instead. |
| CUDA minor-version note | the b11073 build targets CUDA 13.4 while the production cudart set is 13.3 — NVIDIA's documented same-major compatibility; the cuBLAS symbols used are stable across 13.x (see `DEPENDENCY_INSPECTION.md`). If a target ever fails to load, the isolated fallback is to extract `cudart-llama-bin-win-cuda-13.4-x64.zip` into `experimental\llama_b11073\bin\` (the exe directory takes loader precedence) — production stays untouched. |
| app still on 8080 | you launched the normal `START.bat` (production session) instead of `RUN_EXPERIMENTAL.bat` / the generated `start_voicemem_experimental.ps1` — that is by design. |

## Sandbox / Linux side

```bash
bash experimental/llama_b11073/start_experimental_8081.sh
# (adapts: no GPU -> -ngl is a no-op on the CPU-only build; every value
#  env-overridable via LLAMA_EXPERIMENTAL_{BIN,MODEL,HOST,PORT,NGL,CTX,THREADS,LOG})
```

## Validated facts (see the audit report for the full evidence)

* API-compatible with the existing client: /health, /v1/models, /props,
  /slots, /metrics, SSE streaming shape, JSON mode, request-level thinking
  suppression — all proven live through the REAL `app/llm.py` client
  (audit: `audit/VoiceMEM_llama_b11073_experimental/`).
* `response_format={"type":"json_object"}` is GRAMMAR-ENFORCED on b11073
  (3/3 clean JSON), while the pinned b10717 emitted markdown-fenced JSON on
  2/3 identical probes — b11073 matches the documented client contract
  better.
* `--reasoning off` / `--cache-type-k/-v` / `-t` / `-ngl` flag surface:
  identical between the two builds (both `--help` outputs verified).
* `-c 16000` is rounded up to `n_ctx = 16128` (multiple of 256) — same
  rounding on both builds.
* Sandbox A/B (identical flags, same stand-in model, sequential runs):
  latency parity within noise; the queue-behind-background-leg outlier
  class persists identically (architectural, not runtime-versioned);
  cancellation/barge-in transport behaves identically (slot frees
  sub-second on both).
* The operator's "~13 tok/s and substantially better latency" observation
  was made on the TARGET machine with the 19 GB production model and GPU
  offload — the sandbox (CPU-only, 2 cores, 1.7 B stand-in) can neither
  confirm nor refute it. See the report's Known Limitations.
* The installer's script logic was additionally executed in this sandbox
  under PowerShell 7.4.6 on Linux against a clean extracted v0.10.7 tree
  (download → hash gate → extraction → per-file verification → enable/
  disable/verify, production-untouched proof) — see the worklog; the
  Windows-specific parts (process start of the PE server, listener PID /
  `.venv` client probes) are the operator's target-side steps documented
  above.
* Rev 2 verification evidence: the pinned asset was re-downloaded and its
  SHA-256 re-matched the pin (`85c1b874…d79c4`, 150,093,926 bytes); all 24
  per-file hashes re-verified against the extracted asset; the `1aa2954bd`
  build identity confirmed present in `llama-server-impl.dll` (and the
  production `a32af33de` absent); the launcher's `-VerifyOnly` pre-flight
  executed green against the real files, and the tamper tests (corrupted
  exe / swapped impl.dll / forged manifest) all failed closed as designed.

## Artefacts

* `audit/VoiceMEM_llama_b11073_experimental/REPORT.md` — the full report
  (A/B + forensic results, known limitations, recommendation).
* `tests/unit/test_experimental_llama_runtime.py` — validation-only tests
  pinning the isolation contract (launcher baselines, env override +
  rollback, production-untouched pins, installer pin-table validity,
  live-captured b11073 SSE fixture parsed through the production parser,
  the rev-2 identity-verification contract).
* `tools/inspect_pe_closure.py` — the PE import-closure inspection tool
  behind `DEPENDENCY_INSPECTION.md`.
