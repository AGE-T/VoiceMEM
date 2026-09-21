# Experimental llama.cpp b11073 runtime — isolated profile (NOT production)

**Status: EXPERIMENTAL — reversible — no production release.** This profile
integrates and validates the operator-tested newer llama.cpp build **b11073**
(llama-server `0.4.1-dev (build 11073, commit 1aa2954bd)`) next to the
untouched VoiceMem **v0.10.7** production baseline (pinned **b10717**,
`llama-server 0.3.0-dev (build 10717, commit a32af33de)`).

```
VoiceMem v0.10.7
   |
   +-- production llama-server (b10717, bin\llama-server.exe, 127.0.0.1:8080)  UNTOUCHED
   |
   +-- experimental llama-server (b11073, bin\llama-server-b11073\, 127.0.0.1:8081)  THIS
```

## What is NOT touched (the hard guarantees)

* the pinned production binary (MODELS.lock.json `tools.llama-server` stays
  `b10717`) — the experimental executable lives in a **separate path**;
* `config/llm_config.yaml` (the canonical production profile: ngl 20,
  ctx 32768, parallel 1, KV q8_0, reasoning off) — the experimental profile
  deliberately carries its OWN values in the launchers only;
* `scripts/start_llama_server.ps1` (the production starter) — unmodified;
* `VERSION` (0.10.7), prompt semantics, memory semantics, cancellation
  semantics, TTS, ASR, retrieval — zero application-code changes.

## Exact launcher commands

**Target machine (Windows) — the operator-tested baseline:**

```powershell
# one-time: download the b11073 release asset matching your CUDA,
# e.g. llama-b11073-bin-win-cuda-13.3-x64.zip, and extract it into
# bin\llama-server-b11073\  (the production bin\ stays as-is)
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\start_llama_server_experimental.ps1
```

Generated server command line (ngl 99 / ctx 16000 / parallel 1 / threads 12 /
reasoning off / KV q8_0 / temp 0.7, `--metrics --no-webui`):

```
bin\llama-server-b11073\llama-server.exe --model <resolved GGUF> --host 127.0.0.1 --port 8081 -ngl 99 -c 16000 --parallel 1 -t 12 --cache-type-k q8_0 --cache-type-v q8_0 --temp 0.7 --reasoning off --metrics --no-webui --verbose
```

**Sandbox / Linux side (this validation):**

```bash
bash experimental/llama_b11073/start_experimental_8081.sh
# (adapts: no GPU -> -ngl is a no-op on the CPU-only build; threads default
#  to the physical core count; every value env-overridable via
#  LLAMA_EXPERIMENTAL_{BIN,MODEL,HOST,PORT,NGL,CTX,THREADS,LOG})
```

## Exact application-side override (existing mechanism, zero code change)

The VoiceMem client surface derives every endpoint from
`config.llama_server_url`; `app/config.py apply_env` has honoured
`LLAMA_SERVER_HOST` / `LLAMA_SERVER_PORT` since v0.4.x:

```bash
export LLAMA_SERVER_HOST=127.0.0.1
export LLAMA_SERVER_PORT=8081
# then start the agent normally — the Python client (app/llm.py) AND every
# vendor bridge leg (mem0, via OPENAI_BASE_URL) target the experimental server
```

**The port is never hardcoded in the application** — it lives only in the
launcher default (overridable) and the two env variables above.

## Rollback / recovery note

1. Stop the experimental server (Ctrl+C / close its window / `pkill`).
2. Remove the two env variables from the agent session (`unset
   LLAMA_SERVER_HOST LLAMA_SERVER_PORT` / `Remove-Item Env:...` or simply a
   fresh terminal) — the agent then targets the production server on
   127.0.0.1:8080 again, with byte-identical behaviour.
3. Optionally delete `bin\llama-server-b11073\` (and on the sandbox
   `/tmp/llama/llama-b11073/`). Nothing else exists to undo: no production
   file, config value, version number or release artefact was changed.

## Validated facts (see the audit report for the full evidence)

* API-compatible with the existing client: /health, /v1/models, /props,
  /slots, /metrics, SSE streaming shape, JSON mode, request-level thinking
  suppression — all proven live through the REAL `app/llm.py` client.
* `response_format={"type":"json_object"}` is GRAMMAR-ENFORCED on b11073
  (3/3 clean JSON), while the pinned b10717 emitted markdown-fenced JSON on
  2/3 identical probes — b11073 matches the documented client contract
  better (app/llm.py documents constrained generation as the expected
  behaviour).
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

## Artefacts

* `audit/VoiceMEM_llama_b11073_experimental/REPORT.md` — the full report
  (A/B + forensic results, known limitations, recommendation).
* `audit/VoiceMEM_llama_b11073_experimental/evidence/` — transport audit,
  benchmark JSONs, server logs, forensic reports, comparison tables.
* `tests/unit/test_experimental_llama_runtime.py` — validation-only tests
  pinning the isolation contract (launcher baselines, env override +
  rollback, production-untouched pins, live-captured b11073 SSE fixture
  parsed through the production parser).
