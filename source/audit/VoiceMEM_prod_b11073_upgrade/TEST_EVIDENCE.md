# VoiceMEM production b11073 upgrade pack — sandbox execution test evidence

**Pack:** `VoiceMemAgent_v0.10.7_ProdB11073_Upgrade.zip` (ops/llama_b11073_production_upgrade/)
**Date:** 2026-09-23/24 · **Environment:** Linux sandbox, PowerShell 7.4.6 (`pwsh`), real pinned b11073 asset re-downloaded and verified (SHA-256 `85c1b874…d79c4`, 150,093,926 bytes, 24/24 per-file pins, build id `1aa2954bd` present / `b11073` literal absent in `llama-server-impl.dll` — re-proving the pin table live).

The sandbox cannot run the Windows PE binaries, so the test harness uses a **mock llama server** (Python stdlib, `prctl`-renamed to `llama-server`, OpenAI-shaped `/health`, `/v1/models`, `/v1/chat/completions` non-stream/SSE/json_object with a configurable `system_fingerprint`) plus a **stand-in starter** that derives the served fingerprint from the actual `bin\llama-server-impl.dll` content (b11073 build id → `b11073-1aa2954bd`, else `b10717-a32af33de`) — i.e. the "server identity follows the installed runtime", exactly like the real pair.

## Results

| # | Test | Result |
| --- | --- | --- |
| T0 | helper chain sanity: `Get-ListenerPids` (Get-NetTCPConnection → netstat → ss fallbacks), `Get-ProcInfo` name resolution, mock HTTP surface, the pack's own `Invoke-StreamProbe` (first chunk + `[DONE]`) | **PASS** |
| T1 | `-VerifyOnly` on a healthy tree: pack config × experimental pins × install manifest three-way agreement + source 24/24 + build id → exit 0, no changes | **PASS** |
| T2 | 34 unit tests of the pure helpers: command-line tokenizer/parser (quoted paths, `name=value` forms), documented-profile compliance (PASS case + ngl/ctx/model/port/missing-flag deviation catches), fingerprint gate (exact / variant / b10717 / empty), served-model containment, Hungarian prompt bodies (JSON `\uXXXX` decode), mojibake repair | **PASS 34/34** |
| T3 | full run (audit → stop → backup → install → verify → start via the starter → battery → regression → report): exit 0; the 24 real pinned files installed with hashes verified; stale `ggml-cpu-nehalem.dll` + `llama-cli.exe` removed; CUDA DLLs (`cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll`) preserved; battery C/D/E/F green (Hungarian chat, SSE first-token + clean `[DONE]`, grammar-valid JSON, `system_fingerprint = b11073-1aa2954bd`); regression OK; `REPORT.md` + `stage_results.json` + evidence ZIP written | **PASS** |
| T4 | auto-rollback on identity failure (fault-injected fingerprint `b99999-deadbeef`): the [F] gate failed → rollback executed → `bin\llama-server.exe` restored **hash-identical** to the audited b10717 → server restarted and verified → orchestrator exit 1 with `ROLLBACK EXECUTED and VERIFIED` | **PASS** |
| T5 | idempotency: second run on the already-upgraded tree detects `bin\` = the pinned b11073 set → backup/replace skipped → still exit 0 | **PASS** |
| T6 | `-ProbeOnly` against the running b11073 server: health + model + chat + fingerprint → exit 0 | **PASS** |
| T6b | `-ProbeOnly` against a b10717 server: exit 1, diagnostics name the fingerprint | **PASS** |
| T7 | standalone `rollback_production_b11073.ps1 -BackupDir …`: restore (hash-verified against the backup manifest) → restart via the starter → `/health`, `/v1/models`, completion round-trip → live fingerprint `b10717-a32af33de` (NOT b11073 — the rollback proof) → exit 0 | **PASS** |
| T8 | tamper detection: one byte flipped in the experimental source → `-VerifyOnly` exit 1 naming the file + expected/actual hash; the full run also refuses **before touching `bin\`** (bin hash unchanged) | **PASS** |

## Bugs found and fixed by these tests (the honest list)

1. **S3 backup copied nothing**: `Copy-Item -LiteralPath <dir>\*` takes the wildcard literally (no expansion) — the empty backup was then caught by the backup-verification gate and rolled back exactly as designed. Fixed to `-Path`.
2. **Rollback could accept an empty backup**: when both the manifest and the backup exe hash were unreadable, `"" -eq ""` passed. Now a backup without `llama-server.exe` fails loudly.
3. **Auto-rollback could hang on some platforms**: `Start-Process -Wait` also waits for the descendant tree (the rollback's starter child supervises the server forever). Replaced with a direct-child poll with a deadline.
4. **Rollback lacked `-ModelPath` passthrough** (and leaf extraction used a Windows-only path API). Both fixed; the auto-rollback now forwards the model path.
5. **`$sPid` collided with the read-only `$PID` automatic variable** (case-insensitive) — renamed.
6. **An unterminated double-quoted string containing backticks** (a ` ``` ` fence test) broke the parser — moved to single quotes.
7. **Test-harness-only findings**: `-WindowStyle` is unsupported on Linux pwsh (removed — the rollback console is now visible, which is better transparency anyway); piped stdout keeps detached children alive (the target's BAT runs in its own console, unaffected).

## Honest limits (Windows-only paths reviewed, not executed here)

Exactly as documented for the rev-2 pack: the PE process start, `Win32_Process` command-line capture, the profile-compliance gates over live command lines, the real `scripts\start_llama_server.ps1` flow (config bridge, log redirection, load proofs) and the VoiceMem web-backend integration stage run **only on the target machine**. They are PS 5.1-constrained by construction (no ternary/null-coalescing, EAP-safe native capture, ASCII+CRLF sources) and the operator's `RUN_PRODUCTION_UPGRADE.bat` run is the final proof, as the README documents.
