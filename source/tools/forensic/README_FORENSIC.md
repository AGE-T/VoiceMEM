# TEMPORARY forensic tools (delete after the LLM latency investigation)

**Status: forensic diagnostics only. Nothing here is production code, is
referenced by the installer, the tests, the release build or any runtime
script. Do not extend the release ZIP manifest for these files.**

## llm_latency_forensics.py

One-shot llama-server latency forensic benchmark for the VoiceMem production
chain (Qwen3.6 35B A3B IQ4_XS on the RTX 5070). Read the module docstring
for the full phase description. It:

- never modifies production config/source/installer/manifests,
- never benchmarks or kills the production llama-server by default
  (`--stop-production-server` opts in),
- references the 19 GB GGUF in place (never copies it),
- keeps thinking disabled exactly as production does (`--reasoning off` +
  request-level `chat_template_kwargs.enable_thinking=false` +
  `reasoning_effort:"none"`),
- benchmarks only FRESH llama-server processes on a dedicated port (8180),
  treats the FIRST request after startup as the only authoritative
  measurement, and reads every effective runtime value from the server's own
  startup log (never from the intended command line).

Run it on the field machine, from the repo root:

```powershell
# recommended: stop the production llama-server window first (Ctrl+C), then:
.\.venv\Scripts\python.exe tools\forensic\llm_latency_forensics.py

# or let the tool stop the production server for the benchmark phases:
.\.venv\Scripts\python.exe tools\forensic\llm_latency_forensics.py --stop-production-server
```

Outputs (all under `logs/`, all small):

- `logs/llm_runtime_forensic_report.md` - the full report (rewrites the
  stage-1 analysis report with field measurements),
- `logs/llm_runtime_forensic_results.json` - every raw measurement,
- `logs/llm_forensic_real_payload.json` - the captured real request,
- `logs/llm_forensic_test_<config>.log` - each test server's startup log.

Useful flags: `--quick` (small matrix), `--skip-short/--skip-real/
--skip-ollama`, `--synthetic-memory` (skip the real VoiceMem search),
`--model PATH` / `--server-exe PATH` overrides, `--request-timeout S`.

Restart production afterwards: double-click `START.bat` (or run
`scripts\start_llama_server.ps1` in its own window).

---

# STAGE 3: capture ONE REAL production slow turn + direct replay

The stage-2 benchmark proved what FRESH servers do with a REPRESENTATIVE
payload — it did **not** reproduce the production 30-50 s first-token delay
(its history entries were synthetic; no real slow turn was ever replayed).
Stage 3 closes exactly that gap. **No ngl/context matrix is run here.**

## What gets captured (per LLM turn, append-only JSONL)

`logs/llm_production_trace.jsonl`, one JSON object per line:

- `session` - process + llama-server identity (health/props/metrics +
  effective-config lines from the llama-server logs + llama-server.exe
  command line on Windows)
- `start` - `llm_request_start`: the exact messages array (verbatim, deep
  copy), message stats (chars per role, history entries), resolved request
  parameters, whether the httpx client was already warm
- `wire` - the EXACT JSON body handed to httpx (captured inside the
  transport, i.e. the literal bytes sent to llama-server)
- `headers` - response headers received (HTTP status, send elapsed)
- `end` - every timing marker (see below) + `/metrics` counters before and
  after the request + the llama-server's own log lines produced during the
  turn (slot state, `print_timing`) + error/cancellation info

Timing markers (seconds relative to `llm_request_start` of that turn):

```
llm_request_start -> body_start -> http_request_start -> http_headers_received
                  -> first_sse_line -> first_content_token -> stream_end
                  -> llm_request_end
```

`http_request_start` is measured inside the httpx transport — the moment
the request actually leaves the application. `/metrics` deltas give the
server-side seconds (`llamacpp:prompt_seconds_total`) and the exact prompt
token count for THE production turn.

## Operator runbook (field machine, F:\Voicemem\VoiceMemAgent)

1. Start VoiceMem as usual (`START.bat`), wait for the web UI.
2. Close ONLY the web-backend console window (the one running python); the
   llama-server window keeps running.
3. Double-click `tools\forensic\CAPTURE_START.bat` (same backend, now with
   the LLM trace active; the browser opens automatically).
4. Use VoiceMem NORMALLY — talk/typing as always. Every LLM request is
   captured. Wait for a slow turn (the 30-50 s one). If the slowest captured
   turn is still below 30 s, KEEP USING VoiceMem — the replay tool says so
   explicitly and the result stays marked "diagnostic only, NOT a
   reproduction".
5. Double-click `tools\forensic\REPLAY_CAPTURED_TURN.bat` — it replays the
   EXACT captured slow request (default: the slowest turn) TWICE:
   - WARM: directly against the running llama-server (never restarted,
     never reconfigured), and
   - FRESH: against a NEW llama-server process built from the production
     process's own command line (only the port differs; the fresh process
     is stopped afterwards). A warm-cache hit can never masquerade as a
     fast cold request.
   It then writes the full comparison (22 measured points), the A-G
   decision-tree classification, the MEASURED FACTS / INFERENCES /
   UNRESOLVED QUESTIONS separation and the A-P verdict to
   `logs\llm_production_capture_result.md` + `.json`.
   The fresh start loads the 19 GB model again — expect minutes. If the
   fresh server cannot fit VRAM next to the production server, the tool
   records the failure with guidance: close the production llama-server
   window (Ctrl+C), then run
   `REPLAY_CAPTURED_TURN.bat --replay fresh-only` (it MERGES the warm
   result already captured), then restart `START.bat`.
6. To return to plain production: Ctrl+C, then `START.bat`. Nothing to
   uninstall — no production file was modified.

Advanced: `replay_captured_request.py --trace-id <id> | --mode last` to pick
a different turn; `--replay warm|fresh-only|both` (default both); `--url`
to override the warm endpoint (default: the trace's own production URL);
`--fresh-port` (default 8181); `--slow-threshold` (default 30 s).

## Decision tree the replay result answers

The report classifies the turn into one of the measured cases:

- CASE A: production slow, fresh replay fast → the model/server does not
  reproduce the delay → application runtime / request handling / HTTP
  transport / event loop / locking / queueing / streaming (or one-time
  server state). No model configuration change.
- CASE B: production slow, fresh replay slow too → the request itself or
  the llama-server configuration produces the delay.
- CASE C: fresh slow but warm replay very fast (cache hit) → KV cache state
  is decisive — production receives a cold/ineffective cache.
- CASE D/G: the server produces the first token quickly but the
  application receives it late (first SSE line → first content token gap)
  → SSE parsing / buffering / event loop / stream consumption.
- CASE E: the LLM call is delayed BEFORE the HTTP request even starts
  (llm_request_start → http_request_start) → application-side blocking.
- CASE F: headers/first SSE line arrive tens of seconds after the request
  → inside llama-server (prefill / request handling).
- `/metrics` prompt_seconds delta ≈ the whole delay → the delay is INSIDE
  llama-server (prefill); ≈ 0 → it never reached the server.

The exact payload is preserved in the trace (start + wire events, including
the literal wire text), so the replayed request is byte-identical where
technically possible and the equivalence is verified in the report.

## Files (all temporary, all outside production paths)

- `llm_trace_instrumentation.py` - runtime wrapper for
  `app.llm.LlmClient.chat_stream` + httpx transport hooks; fail-open
- `run_web_with_llm_trace.py` - the forensic launcher (production start +
  instrumentation)
- `replay_captured_request.py` - exact replay + 13-point comparison
- `CAPTURE_START.bat` / `REPLAY_CAPTURED_TURN.bat` - double-click helpers

## Pre-flight: exactly ONE llama-server.exe

Two simultaneous llama-server instances (e.g. START.bat started twice while
the 19 GB model was still loading) corrupt every latency measurement and
can themselves produce tens-of-seconds prefill delays on a 12 GB VRAM GPU.
Before the capture session, verify in Task Manager — or:

```powershell
Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
  Select-Object ProcessId, CreationDate, CommandLine
```

Exactly one entry with `--port 8080` is expected; kill any other instance
(`Stop-Process -Id <PID>`) and wait for `/health` 200 before capturing.
