# LLM runtime forensic report - VoiceMem / Qwen3.6 35B A3B IQ4_XS / RTX 5070

- Stage 1 (agent-side): production code traced, request reconstructed and
  tokenized with the production tokenizer, pinned llama.cpp b10717 build
  behaviour verified live, operator-reported field measurements organised.
- Stage 2 (field run, operator-reported): the full benchmark matrix ran on
  the field machine and rewrote the field copy of this report. **Caveat
  (stage-2, critical):** its "real workload replay" payload was
  REPRESENTATIVE, not a recording of an actual production turn (the user
  utterance was the default "no log transcript found" one; the history
  entries were "representative synthetic (real history is in-memory
  only)"). Its best result (ngl 20 + 32768: ~4.89 s TTFT / ~7.82 s total)
  therefore does **NOT** reproduce or explain the production 30-50 s
  first-token delay. Production first-token was **NOT MEASURED** in the
  available logs. Treat stage-2 numbers as supporting material only.
- Stage 3 (this update, agent-side): the real-turn capture instrumentation
  is BUILT and VALIDATED (below); the field capture is the pending step.
- **No production file was ever modified.**
- Raw stage-1 data: `logs/llm_runtime_forensic_results.json`
- Evidence labels: **[MEASURED-CODE]** read directly from the production
  source; **[MEASURED-SANDBOX]** measured here against the real pinned
  llama.cpp build / real tokenizer / full pipeline; **[MEASURED-FIELD]**
  numbers reported by the operator from the real machine; **[OPERATOR-
  REPORTED STAGE-2]** field benchmark numbers with the stage-2 payload
  caveat; **[INFERRED]** analysis conclusion - not a measurement;
  **[PENDING-FIELD]** to be measured on the field machine.

---

## 0. STAGE 3 - real production turn capture + dual direct replay (the current step)

**Goal:** capture ONE REAL production turn that actually experienced the
30-50 s first-token delay - the exact request at the production boundary
immediately before `c.llm.chat_stream(messages)` - then send THAT EXACT
request directly to llama-server **TWICE** (warm running server + a fresh
process with the same production configuration) and compare, so the exact
point where the 30-50 s are spent is isolated. No ngl/context matrix. No
synthetic history. No default utterance.

**Built and validated end-to-end in the sandbox (MEASURED-SANDBOX):**

| component | file | what it does |
|---|---|---|
| runtime instrumentation | `tools/forensic/llm_trace_instrumentation.py` | wraps `app.llm.LlmClient.chat_stream` + httpx transport at RUNTIME; captures the exact payload (deep copy + the literal wire text `exact_wire_body_raw`), every timing marker, /metrics deltas, the llama-server log lines of the turn; **zero production files modified** |
| forensic launcher | `tools/forensic/run_web_with_llm_trace.py` (+ `CAPTURE_START.bat`) | starts the PRODUCTION web backend (same `app.web_server.main()`, same args) with the instrumentation installed; fail-open |
| dual replay + comparison | `tools/forensic/replay_captured_request.py` (+ `REPLAY_CAPTURED_TURN.bat`) | replays the EXACT captured request (default: slowest turn) TWICE: (1) WARM against the RUNNING production server (never restarted), (2) FRESH against a NEW llama-server process rebuilt from the production process's own command line (only the port differs; stopped afterwards). Writes the 22-point comparison, the A-G decision-tree classification, the MEASURED FACTS / INFERENCES / UNRESOLVED QUESTIONS separation and the A-P verdict. If the fresh server cannot fit VRAM next to the production server, the failure is recorded with guidance and `--replay fresh-only` MERGES the warm result afterwards |

Markers per turn (relative to `llm_request_start`):
`body_start -> http_request_start -> http_headers_received -> first_sse_line
-> first_content_token -> stream_end -> llm_request_end` (the app-side
completion, stamped BEFORE forensic bookkeeping) `-> forensic_end_rel` (the
instrumentation's own overhead), plus
`/metrics` deltas (`llamacpp:prompt_tokens_total`,
`llamacpp:prompt_seconds_total`, `llamacpp:tokens_predicted_total`,
`llamacpp:tokens_predicted_seconds_total`) = the server-side prefill and
generation seconds and the EXACT token counts of the production request,
plus the server's own `slot print_timing` lines from the request window
(verified: print_timing is INFO-level in b10717 - it appears WITHOUT
`--verbose`, so the fresh server keeps the exact production command line).

**Sandbox validation evidence v2 (Qwen3-1.7B stand-in, pinned b10717, full
production flag set, NO --verbose, port 8180, validation run
`.zscripts/stage3_validation_run2.sh`):**
- 2 traced turns through the REAL `LlmClient` + REAL
  `teacher_persona.build_messages` payload: 6 messages / 2316 chars;
  `exact_wire_body_raw` captured (2705 chars) and the httpx-style
  re-serialization is BYTE-IDENTICAL to the wire (sha256 proof);
- all 8 markers present incl. `forensic_end_rel`; app completion
  31.8557 s <= forensic end 31.8644 s (instrumentation overhead 0.009 s);
- turn-1 (cold): headers 0.398 s, first SSE line 13.4465 s, first token
  13.4466 s, total 31.8557 s; `/metrics` delta: 661 prompt tokens,
  13.049 s prompt seconds, 52.3 tok/s prefill, 96 tokens / 18.4 s
  generation - matching the server's own
  `print_timing: prompt eval time = 13052.50 ms / 661 tokens`;
- WARM replay of the exact bytes: first token 0.148 s, prompt-token delta
  ~1 -> correctly flagged "KV/prompt CACHE HIT" (sha256(sent) == sha256(wire));
- fresh-start FAILURE path (forced bad --server-exe): failure + operator
  guidance recorded honestly, warm kept, classification stays pending;
- FRESH replay via `--replay fresh-only` (production server closed first =
  the exact VRAM-contingency operator flow): the tool started a NEW server
  from the ps-captured production command line (port 8181, ready 6.2 s),
  replayed the exact bytes cold: first token 13.023 s vs production
  12.963 s, print_timing prompt 13004 ms, MERGED the previous warm result
  -> classified **CASE C** (fresh slow + warm cache hit) with the correct
  next measurement ("cache_reuse / slot state lines");
- NOT_SLOW path (default 30 s threshold, 13 s stand-in turn): explicit
  "NOT A REPRODUCTION - diagnostic only" banner + verdict A = NO;
- [chain] web-server.log correlation: matched by content_chars, chain
  first-token delta 12.963 s vs instrumented 12.9629 s (agrees);
- launcher smoke test: instrumented web backend booted (mock mode) with
  "instrumentation: ACTIVE", session event written with the full config;
- `uninstall()` restores the original methods.

**Known instrumentation overhead (honest):** one localhost `/metrics` GET
before and after each LLM request (0.009 s measured inside the validation
turn; negligible against a 30-50 s delay; recorded between
`llm_request_end_rel` and `forensic_end_rel`).

**Operator runbook (field machine):** see
`tools/forensic/README_FORENSIC.md` stage-3 section - START.bat as usual,
close only the web-backend console, run `tools\forensic\CAPTURE_START.bat`,
use VoiceMem normally until a slow (30-50 s) turn happens, then run
`tools\forensic\REPLAY_CAPTURED_TURN.bat` (warm + fresh; expect minutes for
the 19 GB fresh load; if the fresh server cannot fit VRAM next to the
production server, close the production llama-server console and re-run
`REPLAY_CAPTURED_TURN.bat --replay fresh-only`, which MERGES the warm
result). Output:
`logs/llm_production_capture_result.md` / `.json` (the 22-point comparison,
A-G classification, A-P verdict)
+ `logs/llm_production_trace.jsonl` (the exact captured requests, including
the literal wire text). If the slowest turn is still below 30 s the result
says so explicitly and stays marked diagnostic-only - keep using VoiceMem
until the real 30-50 s turn occurs.

---

## 1. Production request path (traced from the code)

**[MEASURED-CODE]** `app/web_server.py` -> `build_system_prompt` ->
`_fit_history_budget` -> `build_messages` -> `c.llm.chat_stream(messages)` ->
`app/llm.py` -> `POST http://127.0.0.1:8080/v1/chat/completions` (SSE).

Exact request construction (`WebSession` turn handler, lines ~2749-2830):

```
memory_context = WebSession._memory_context(result)   # 5 LB + 3 RB hits, hard cap 1200 chars
memory_context = _cap_memory_context(memory_context)  # second cap, 6000 chars (no-op at 1200)
system_prompt  = build_system_prompt(memory_context, emotion_label?, ...)  # base ~894 chars
history        = _fit_history_budget(system_prompt, self._history[-8:], user_text)  # 14000-char budget
messages       = build_messages(user_text, system_prompt, history=history)
async for delta in c.llm.chat_stream(messages): ...
```

Request payload (`app/llm.py::chat_stream`): `model`, `messages`,
`stream: true`, `temperature: 0.7`, `max_tokens: 512`,
`chat_template_kwargs: {enable_thinking: false}`, `reasoning_effort: "none"`.
httpx client: connect timeout 3 s, read timeout 120 s. Thinking is disabled
BOTH at the server (`--reasoning off`, start_llama_server.ps1) and at the
request level - two independent switches that both set
`enable_thinking = false`.

## 2. Actual request size (tokenized with the real Qwen3.6-35B-A3B tokenizer)

**[MEASURED-SANDBOX]** The tokenizer of `Qwen/Qwen3.6-35B-A3B` (the exact
model family, fetched from HF; the 19 GB GGUF itself stays external) was used
to render the production chat template (thinking disabled) over payloads
rebuilt with the PRODUCTION functions (`_cap_memory_context`,
`_fit_history_budget`, `build_system_prompt`, `build_messages`):

| variant | messages | total chars | templated prompt tokens | chars/token | fits 8192 (ctx - 512)? |
|---|---|---|---|---|---|
| minimal cold turn (no memory, no history) | 2 | 913 | **192** | 4.76 | yes (7488 headroom) |
| typical turn (memory 1192 ch + emotion + 4-entry history) | 6 | 2952 | **808** | 3.65 | yes (6872 headroom) |
| budget-max (memory 1192 ch + emotion + 8-entry history) | 10 | 4759 | **1431** | 3.33 | yes (6249 headroom) |
| short bench-style prompt (user msg only) | 2 | 946 | **203** | 4.66 | yes |

- Hungarian/English mixed text on this tokenizer: **3.3-4.8 chars/token** -
  the code comment's "~2-3 chars/token" is conservative.
- The 14000-char prompt budget converts to at most ~4200 templated tokens -
  **[INFERRED]** context overflow is NOT the latency cause at the current
  budgets (the historical 10459-token rejection predates the v0.4.5 caps).
- **[PENDING-FIELD]** the stage-2 run tokenizes the exact captured payload
  via the server's own metrics counters (exact, GGUF-embedded tokenizer).

## 3. Production llama-server configuration

**[MEASURED-CODE]** from `config/voicemem_config.yaml` +
`config/env.local.ps1` + `scripts/start_llama_server.ps1` (argument assembly,
lines 737-762):

```
bin\llama-server.exe
  --model "<resolved: config/llm_model.json (UI picker) > LLAMA_MODEL_PATH >
            models\llm\qwen3.6-35b-a3b\*.gguf - the field machine resolves the
            C:\AI_HOME Ollama blob sha256-afc7238... in place>"
  --host 127.0.0.1 --port 8080
  -ngl 26                (config llm_n_gpu_layers / env LLAMA_N_GPU_LAYERS)
  -c 8192                (config llm_context_size)
  --parallel 1
  --cache-type-k q8_0 --cache-type-v q8_0
  --temp 0.7 --metrics --no-webui --reasoning off
```

- Logs: `logs\llama-server.out.log` / `logs\llama-server.err.log`
  (the effective runtime values must be read from there), plus
  `logs\llama-server.resolved-model.json` (the actual loaded file marker).
- The config comment itself says: `llm_n_gpu_layers: 26  # RÉSZLEGES
  offload: ~19 GB modell > 12 GB VRAM`.
- No `-fa` flag is passed; see §5 for why flash attention is nevertheless ON.

## 4. Model architecture facts (Qwen3.6 35B A3B, from the model repo config)

**[MEASURED-SANDBOX]** `num_hidden_layers = 40`, MoE with 256 experts / 8
active per token, hidden 2048, 16 Q heads / **2 KV heads**, head_dim 256,
vocab 248,320, max position 262,144, chat template with `enable_thinking`.

**[INFERRED]** VRAM arithmetic (estimates, to be replaced by the log-measured
buffer sizes in stage 2):
- weights ~19 GB / 40 layers ≈ ~470 MB per layer (embeddings extra);
- `-ngl 26` → 26 × ~470 MB ≈ **12.2 GB of weights alone**, plus KV cache
  (~330 MB at 8192 q8_0) plus compute buffers (~0.5-1.5 GB) → **cannot fit
  the 12 GB RTX 5070** → the observed `failed to fit params to free device
  memory: n_gpu_layers already set by user to 26`;
- `-ngl 20` → ~9.4 GB + buffers ≈ 10.5-11 GB → borderline fits;
- `-ngl 16-18` → 7.5-8.5 GB + buffers → comfortable;
- KV cache q8_0 ≈ 41 KiB/token → 8192 ctx ≈ 330 MB, 32768 ctx ≈ 1.32 GB
  (a 32768 context consumes ~1 GB MORE VRAM than 8192).

## 5. Pinned llama.cpp build behaviour (verified live with b10717)

**[MEASURED-SANDBOX]** The exact pinned build (`llama-server 0.3.0-dev build
10717, the same binary family as bin\llama-server.exe`) was executed in the
sandbox with the production flag set. Its startup log proves:

```
llama_init_from_model: enabling flash_attn since it is required for quantized V cache
llama_context: flash_attn            = enabled
srv    load_model: initializing, n_slots = 1, n_ctx_slot = 8192, kv_unified = 'false'
slot   operator(): id  0 | task 1 | new prompt, n_ctx_slot = 8192, n_keep = 0, task.n_tokens = 430
slot print_timing: id  0 | task 1 | prompt eval time = 7304.80 ms / 430 tokens (58.87 tokens per second)
```

- `--cache-type-v q8_0` **forces flash attention ON** in this build
  (regardless of the `auto` default) - so production already runs with FA.
- `n_ctx_slot` IS printed by this build and equals `-c` when `--parallel 1`;
  an 8192 slot context with `-c 32768` requested would indicate the request
  never reached this server - **[PENDING-FIELD]** the stage-2 matrix tests
  32768 explicitly and reads the effective slot context per run.
- Per-request server-side timings (`slot print_timing`) and
  `/metrics` counters (`llamacpp:prompt_tokens_total`,
  `llamacpp:prompt_seconds_total`, ...) are available - the harness uses
  them for exact prompt token counts and server-side seconds.
- A repeated identical request after the first shows `prompt eval time /
  1 tokens` (KV/prompt cache hit, ttft ~0.15 s in the sandbox) - which is
  why the benchmark methodology mandates FRESH processes and treats only
  the FIRST request as authoritative.

## 6. Known field measurements (operator-reported)

**[MEASURED-FIELD]** short manual benchmarks (same short Hungarian prompt,
thinking disabled, single requests):

| config | total | prompt tok/s | gen tok/s |
|---|---|---|---|
| ngl 0 + ctx 8192 | ~4.11 s | ~23.1 | ~11.9 |
| ngl 8 + ctx 8192 | ~3.48 s | ~25.9 | ~15.7 |
| ngl 12 + ctx 8192 | ~3.56 s | ~24.8 | ~16.4 |
| ngl 16 + ctx 8192 | **~2.98 s** | ~30.5 | ~19.8 |
| ngl 20 + ctx 8192 | ~3.30 s | ~28.7 | ~23.1 |
| ngl 26 + ctx 8192 | **~14.24 s** | **~4.43** | **~4.47** |
| ngl 18 + requested 32768 | ~2.56 s | ~33.1 | ~21.0 |

- A later production llama-server log showed `n_ctx_slot = 8192` even though
  `-c 32768` had been requested in a benchmark - the effective-context
  question stays open until stage 2.
- Ollama MAXI reference (same underlying blob, thinking disabled):
  total ~0.83 s; load ~0.005 s; prompt_eval 28 tokens in ~0.212 s
  (**~132 tok/s**); eval 28 tokens in ~0.566 s (**~49.5 tok/s**);
  /api/ps: size_vram ~9.62 GB, context_length 32768. A thinking-enabled run
  produced ~1378 tokens in ~28.7 s - thinking must stay disabled (it is, on
  both layers, in production).
- Production VoiceMem turns: ~30-50+ s before the first LLM token; a logged
  example: llm start -> ~39.5 s to first token -> ~40.5 s to LLM done. Other
  turns stalled >40 s with no first token.

## 7. The latency arithmetic that explains the production numbers

**[INFERRED]** (arithmetic over [MEASURED-FIELD] rates and [MEASURED-SANDBOX]
token counts - the stage-2 run replaces it with direct measurement):

| scenario | prompt tokens | at ngl 26 (4.43 tok/s) | at ngl 16-20 (25-33 tok/s) | at Ollama rate (132 tok/s) |
|---|---|---|---|---|
| cold turn (real production early turns) | 192 | **43.3 s** | 5.8-7.7 s | 1.5 s |
| typical turn | 808 | 182 s | 24-32 s | 6.1 s |
| budget-max turn | 1431 | 323 s | 43-57 s | 10.8 s |

- 39.5 s (production first token) x 4.43 tok/s ≈ **175 tokens** - almost
  exactly the real cold-turn prompt size (192 tokens). The ngl 26
  configuration + the real request size quantitatively REPRODUCES the
  observed 30-50 s production latency.
- The manual "good" configs (ngl 16-20) were measured on a **~40-token**
  prompt; on the real 800-1400-token workload they would still cost 24-57 s
  to the first token - **[INFERRED]** the primary bottleneck is the prompt
  processing (prefill) rate of the partially-offloaded MoE, not a single bad
  flag: the fit-failing ngl 26 makes it catastrophic; even a fitting split
  stays 4-10x slower than Ollama's reference.
- Ollama fits the same 19 GB model in ~9.62 GB VRAM and still reaches
  ~132 tok/s prefill / ~49.5 tok/s generation - proof that the hardware can
  do far better with a smarter layer split.

## 8. Hypothesis status

| hypothesis | status | evidence |
|---|---|---|
| GPU layer placement / VRAM misfit (ngl 26) | **ruled IN (primary)** | fit warning in the production log; 3.2-4.8x collapse at ngl 26 [MEASURED-FIELD]; §4 arithmetic |
| Prompt size (history/memory) | **ruled IN (amplifier, not root cause)** | real prompts are 192-1431 tokens [MEASURED-SANDBOX]; at Ollama-class rates they cost only 1.5-10.8 s |
| Context size 8192 vs 32768 | **pending** | 32768 adds ~1 GB KV -> worse fit; effective slot context unverified [PENDING-FIELD] |
| KV cache / batch / ubatch / threads | **pending (recorded, not tuned)** | b10717 defaults: n_batch 2048, n_ubatch 512; recorded by the harness per run |
| Flash attention | **ruled OUT (already ON)** | b10717 forces FA for quantized V cache [MEASURED-SANDBOX] |
| Thinking channel | **ruled OUT** | disabled at BOTH server and request level [MEASURED-CODE]; the 28.7 s Ollama case was thinking-enabled |
| Context overflow of the real request | **ruled OUT at current budgets** | worst measured case 1431 + 512 < 8192 [MEASURED-SANDBOX] |
| Streaming / client overhead | **ruled OUT** | httpx SSE; the server's own `prompt eval time` dominates [MEASURED-SANDBOX] |

## 9. Field benchmark protocol (stage 2 - the authoritative measurements)

`tools/forensic/llm_latency_forensics.py` (temporary forensic tool, validated
end-to-end in the sandbox against the same pinned b10717 build):

1. Inspect the RUNNING production server: exact process command line,
   effective values from its startup log, resolved-model marker, nvidia-smi,
   /health /v1/models /props, and the real turn latency already recorded in
   `logs/web-server.log` (`[chain] llm start` / `llm first token (X ms)` /
   `llm done`).
2. Capture the exact real request with the production functions + the REAL
   VoiceMem memory search + the last real user utterance from the log
   (read-only; writes only `logs/llm_forensic_real_payload.json`).
3. Short matrix, FRESH server per config, first request authoritative:
   ngl {0,8,12,16,18,20,26} x 8192, then ngl {16,18,20} x 32768, port 8180.
4. REAL workload replay: ngl {16,18,20} x {8192,32768} + the exact
   production config (ngl 26) - first-token latency, exact prompt tokens
   (metrics counters), server-side seconds, context-overflow detection.
5. Ollama reference: /api/ps fields + one equivalent real-prompt run with
   thinking disabled.
6. Rewrites this report from the measurements.

Harness validation evidence (sandbox, same b10717 build, Qwen3-1.7B stand-in
on CPU): all phases executed; effective values parsed from the live log
(`flash_attn = enabled`, `n_ctx_slot = 8192`, `offloaded X/29`, batch
2048/512); exact prompt token counts via metrics deltas (430 tokens);
cache-warm second request demonstrated (1 token, ~0.15 s) and excluded from
comparison; the report generator produced every required section. The
production server is never benchmarked or killed by default; the 19 GB GGUF
is referenced in place, never copied.

## 10. Recommended configuration (candidate - NOT applied, pending stage 2)

**[INFERRED]** expected winner (to be confirmed by the stage-2 real-workload
matrix):

```
llm_n_gpu_layers: 16-18        (fits VRAM with margin; 20 is borderline)
llm_context_size: 8192         (real worst-case prompt 1431 tok + 512 answer
                                leaves 6249 headroom; 32768 costs ~1 GB VRAM
                                and worsens the fit)
llm_cache_type_k/v: q8_0       (keep; forces FA on this build)
llm_parallel: 1, temp 0.7, max_tokens 512, thinking disabled (unchanged)
```

Next-step experiments AFTER the proven config lands (out of scope for this
forensic task): explicit `-fa on`, ubatch tuning for prefill, a layer split
matching Ollama's ~9.6 GB VRAM footprint.

## 11. Confidence levels

- HIGH: request path, payload shape, token counts, production command
  composition, b10717 build behaviour (all measured from code / the real
  build / the real tokenizer).
- MEDIUM: the causal chain "ngl 26 fit failure -> CPU-offloaded prefill ->
  30-50 s first token" (strong numerical fit §7, but no in-place A/B yet -
  stage 2 closes this).
- The representative history entries in the stage-2 payload are synthetic
  and labelled; memory/system/user text come from the production path.

## 12. Unresolved questions

- [PENDING-FIELD] Why a production log showed `n_ctx_slot = 8192` with
  `-c 32768` requested (stage 2 measures effective slot context per run).
- [PENDING-FIELD] The exact post-"failed to fit" fallback behaviour of
  b10717 on the RTX 5070 (which buffers move to CPU) - the harness records
  the evidence lines verbatim.
- [PENDING-FIELD] The best REAL-workload config by first-token latency
  (ngl 16 vs 18 vs 20; 8192 vs 32768).
- [PENDING-FIELD] Ollama MAXI's current /api/ps fields on the day of the run.
- Why Ollama's split reaches ~132 tok/s prefill where llama-server's best
  manual config reached ~33 (engine scheduling, not hardware).

---

## Verdict (stage 3 update - the state after the stage-2 field benchmark)

**PROVEN CONFIGURATION**

- NOTHING is proven for the real production workload yet. The stage-2
  winner (ngl 20 + 32768, ~4.89 s TTFT / ~7.82 s total
  [OPERATOR-REPORTED STAGE-2]) was measured on a REPRESENTATIVE payload
  (synthetic history, default utterance) - it is NOT evidence that the
  production 30-50 s problem is solved or explained. No configuration
  change is recommended or applied until stage 3 captures and replays a
  real slow turn.

**REAL WORLD FIRST TOKEN**

- Production (operator logs): **~30-50 s** (logged example 39.5 s; other
  turns stalled >40 s with no first token). NOT MEASURED inside the
  available logs at token level - the stage-3 instrumentation now measures
  it exactly. Best tested config: only representative-payload numbers
  exist (see above caveat).

**REAL WORLD TOTAL LATENCY**

- Production (operator logs): ~40.5 s for a short reply. Direct measurement
  PENDING-FIELD (stage 3).

**MAIN BOTTLENECK**

- NOT PROVEN for a real production turn. Stage-1 arithmetic
  ([INFERRED], §7) still points at prefill of the partially-offloaded MoE
  at ngl 26, but the real slow request has never been captured, replayed
  or attributed. The stage-3 markers
  (`http_request_start` / `prompt_seconds` delta / streaming gaps)
  decide between: app path, server state, queueing, streaming, or the
  request/config itself.

**EFFECTIVE CONTEXT**

- Production config: 8192. [OPERATOR-REPORTED STAGE-2]: on the field
  benchmark servers, 32768 became genuinely active (n_ctx = n_ctx_slot =
  32768). The RUNNING production server's own value is still to be read
  from its startup log by the stage-3 session record.

**EFFECTIVE GPU LAYERS**

- Production requests ngl 26 of 40 layers; the actual offloaded count and
  the fit-failure fallback are in the production startup log - the stage-3
  session record captures them verbatim (PENDING-FIELD).

**WHETHER 32768 CONTEXT IS ACTUALLY ACTIVE**

- [OPERATOR-REPORTED STAGE-2] YES on the benchmark servers (32768 runs
  showed n_ctx = n_ctx_slot = 32768). The production server runs -c 8192
  per config; its effective slot context is re-read by stage 3.

**WHETHER ngl 26 MUST BE REMOVED**

- NOT PROVEN for the real production turn. ngl 26 cannot fit the 12 GB
  card by arithmetic ([INFERRED], §4) and collapsed the short benchmark to
  ~4.4 tok/s / ~43.56 s total [OPERATOR-REPORTED STAGE-2], but the real
  slow request was never replayed with it. Stage 3's per-request measured
  prefill rate + fit-failure lines will decide. NO production change
  until then.

---

## Next single step (stage 3, operator)

1. Copy the stage-3 forensic bundle into F:\Voicemem\VoiceMemAgent
   (tools\forensic\...).
2. START.bat as usual; close only the web-backend console;
   run `tools\forensic\CAPTURE_START.bat`; use VoiceMem normally.
3. When a 30-50 s turn happens, run
   `tools\forensic\REPLAY_CAPTURED_TURN.bat`.
4. Send back `logs\llm_production_trace.jsonl` +
   `logs\llm_production_capture_result.md/.json` - they contain the exact
   captured slow request, both latencies, and the 13-point comparison.
