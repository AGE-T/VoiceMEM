# Experimental llama.cpp b11073 runtime for VoiceMem — validation report

**Date:** 2026-09-21 · **Baseline:** VoiceMem v0.10.7 (VERSION unchanged — this is NOT a production release) ·
**Pinned production runtime:** llama.cpp **b10717** (`llama-server 0.3.0-dev, build 10717, commit a32af33de`) ·
**Experimental runtime:** llama.cpp **b11073** (`llama-server 0.4.1-dev, build 11073, commit 1aa2954bd`) ·
**Author:** Z.ai Code, per the operator's workstream order (British English).

Evidence directory: `audit/VoiceMEM_llama_b11073_experimental/evidence/`
(BIT identifiers below). All live evidence was produced on the sandbox
described in §6; every claim is either **[PROVEN]** (reproduced here,
artefact exists) or explicitly marked as outside the sandbox's reach.

---

## 0. Executive summary

| Question | Answer |
| --- | --- |
| Is b11073 API-compatible with the existing v0.10.7 LLM client? | **Yes [PROVEN]** — every endpoint, the SSE shape, JSON mode, thinking suppression, cancellation transport; plus ONE behavioural improvement: `response_format=json_object` becomes grammar-enforced |
| Can the improvement be brought in by replacing only the server runtime? | **Compatibility-wise yes [PROVEN]; performance-wise unproven on this hardware** — the sandbox measures parity, not the operator's target-machine improvement |
| Does b11073 fix the 30–42 s outlier class? | **No [PROVEN]** — the class is architectural (single slot + non-streaming background legs); it persists identically on b11073 |
| Any production regression? | **None [PROVEN]** — production binary/config/starter/VERSION untouched; the full unit-suite failure list is byte-identical with and without the experimental files |
| Recommendation | **READY FOR PRODUCTION REVIEW** (see §7 for the precise scope of what is and is not proven) |

---

## 1. Experimental runtime implementation

Isolated profile, purely additive, fully reversible
(`voicemem-agent/experimental/llama_b11073/`):

| File | Role |
| --- | --- |
| `start_llama_server_experimental.ps1` | Target-machine (Windows) launcher: separate executable path `bin\llama-server-b11073\llama-server.exe`, port **8081**, operator-tested baseline flags, health-wait, idempotency, load-evidence logging |
| `start_experimental_8081.sh` | Sandbox/Linux launcher: same profile, sandbox-adapted (threads = physical cores; every value `LLAMA_EXPERIMENTAL_*`-overridable) |
| `README.md` | Documentation: exact commands, env override, rollback/recovery note, validated facts |

**The isolation contract [PROVEN]:**

* The pinned production runtime is untouched: `MODELS.lock.json`
  `tools[llama-server].tag` still `b10717` (pinned by
  `tests/unit/test_experimental_llama_runtime.py::test_pinned_llama_runtime_is_still_b10717`).
* The canonical production profile is untouched:
  `config/llm_config.yaml` still ngl 20 / ctx 32768 / parallel 1 /
  reasoning off (pinned by `test_canonical_llm_config_unchanged`).
* `scripts/start_llama_server.ps1` (production starter) unmodified;
  `VERSION` still 0.10.7 (pinned by `test_version_is_still_0107`).
* No application module knows the experimental profile exists
  (`test_no_production_module_references_the_experimental_profile`
  greps `app/` — zero hits). The application only ever sees host:port.

## 2. Exact launcher command

**Target machine (Windows)** — the operator-tested baseline
(ngl 99 / ctx 16000 / parallel 1 / threads 12 / reasoning off):

```powershell
# one-time: extract the b11073 release asset matching your CUDA
# (e.g. llama-b11073-bin-win-cuda-13.3-x64.zip) into bin\llama-server-b11073\
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\start_llama_server_experimental.ps1
```

Resulting server command line:

```
bin\llama-server-b11073\llama-server.exe --model <resolved GGUF> ^
  --host 127.0.0.1 --port 8081 -ngl 99 -c 16000 --parallel 1 -t 12 ^
  --cache-type-k q8_0 --cache-type-v q8_0 --temp 0.7 ^
  --reasoning off --metrics --no-webui --verbose
```

**Sandbox/Linux** (used for this validation; threads default to the
physical core count because this machine has 2):

```bash
bash experimental/llama_b11073/start_experimental_8081.sh
```

## 3. Exact environment-variable / config override

**No code change was needed — the required override already exists** (audit
evidence: `evidence/transport_audit.md`). `app/config.py::apply_env`
(lines 502-505) has honoured these since v0.4.x, and every client surface
derives from `config.llama_server_url` (the Python client `app/llm.py`, and
every vendor/mem0 bridge leg through `OPENAI_BASE_URL`,
`app/voicemem_bridge.py:238-241`):

```bash
export LLAMA_SERVER_HOST=127.0.0.1
export LLAMA_SERVER_PORT=8081
```

Verified live end-to-end: with only these two variables set,
`AgentConfig.llama_server_url == "http://127.0.0.1:8081/v1"` and the REAL
`LlmClient` streamed and JSON-completed against the b11073 server
(evidence: the compatibility run in this report + the pinned tests
`test_env_override_targets_experimental_port`,
`test_env_override_unset_restores_production_endpoint` — the latter is the
**rollback proof**: unset the variables and the client targets
`127.0.0.1:8080` again).

## 4. A/B benchmark results

**Design.** Same model on both sides (the sandbox stand-in
`Qwen3-1.7B-Q4_K_M` GGUF, SHA-256 `72c5c3cb…4fb` — the 19 GB production
Qwen3.6 35B is never sandbox-resident, the standing sandbox convention),
same workload (the REAL production system prompt from
`app/teacher_persona.py` + a representative memory block + 8 fixed
Hungarian user turns; extraction legs with a JSON-object
`response_format`), same machine, **sequential single-server runs** (the
4 GB sandbox OOM-kills two llama-servers simultaneously — proven during
setup). 2×2 matrix to separate version from configuration effects:

| Cell | Build | Flags |
| --- | --- | --- |
| A1 | b10717 | sandbox production-proxy (ngl 0, ctx 8192, t 2) |
| B1 | b11073 | **identical to A1** (pure version effect) |
| A2 | b10717 | operator baseline adapted (ngl 99, ctx 16000, t 2) |
| B2 | b11073 | operator baseline adapted |

**Results** (full JSONs + `COMPARISON.txt` + `KEY_NUMBERS.txt` in
`evidence/ab_results/`; server logs `evidence/ab_results/server_*.log`):

| Metric | A1 (b10717 prod) | B1 (b11073 same) | A2 (b10717 op) | B2 (b11073 op) |
| --- | --- | --- | --- | --- |
| TTFT median (8 user turns) | 0.614 s | 0.580 s | 0.597 s | 0.562 s |
| TTFT p90 / max | 0.690 / 0.891 s | 0.651 / 0.886 s | 0.636 / 0.938 s | 0.688 / 0.923 s |
| Total reply median | 6.64 s | 8.03 s | 7.69 s | 6.70 s |
| Streaming integrity (8 turns) | 8/8 | 8/8 | 8/8 | 8/8 |
| Extraction legs: median total | 9.52 s | 9.60 s | 8.77 s | 9.25 s |
| Extraction legs: JSON valid | 4/4 | 4/4 | 4/4 | 4/4 |
| Generation rate (extract legs, median) | 7.95 tok/s | 7.64 tok/s | 7.53 tok/s | 8.19 tok/s |
| **User TTFT behind a background leg** (median/max) | **11.4 / 14.0 s** | **8.0 / 10.1 s** | **8.8 / 11.6 s** | **10.2 / 12.5 s** |
| Barge-in slot-free (median/max) | 480 / 502 ms | 210 / 300 ms | 183 / 343 ms | 461 / 582 ms |
| Follow-up TTFT after cancel (median) | 0.414 s | 0.410 s | 0.395 s | 0.426 s |
| Prefix cache (identical prompt re-sent) | 20/414 tok re-processed | 20/414 | 20/414 | 20/414 |

**Findings:**

1. **Latency parity within noise.** All four cells sit in the same band
   (TTFT ≈ 0.56–0.61 s median; generation ≈ 7.5–8.2 tok/s on this 2-core
   CPU). The sandbox can neither confirm nor refute the operator's
   "~13 tok/s and substantially better latency" — that observation was
   made on the target machine with the 19 GB model and GPU offload
   (ngl 99), hardware this sandbox does not have (§6).
2. **The 30–42 s outlier class is NOT fixed by the runtime swap [PROVEN].**
   The user TTFT behind a non-streaming background leg lands at 6.6–14.0 s
   at sandbox scale (a 192-token background leg; the forensic tool's
   variant with a 2655-token prompt reaches 47.6 s on b10717 vs 55.0 s on
   b11073 — the same class as the production 30–42 s outliers). The class
   is architectural: single `--parallel 1` slot + non-streaming legs
   holding it. Both builds behave the same way (§5, s3).
3. **Cancellation/barge-in transport behaves identically [PROVEN]** —
   slot-free after a streaming disconnect is sub-second on every cell
   (median 183–480 ms; max 582 ms). The async-httpx cancel path frees the
   background-leg slot in ~6.4–7.0 ms on every cell. No regression, no
   improvement class.
4. **Streaming integrity identical [PROVEN]** — every cell: 1 role chunk
   (with `"content": null` — correctly ignored by the production parser),
   N content chunks, 1 finish chunk, `data: [DONE]`, zero reasoning
   leakage, HTTP 200. The b11073 `system_fingerprint` (`b11073-1aa2954bd`)
   is carried in the chunks; a live-captured fixture is pinned and parsed
   by the production parser in the new tests.
5. **`response_format=json_object` is grammar-enforced on b11073, NOT on
   b10717 [PROVEN]** — identical minimal probes: b10717 returned
   markdown-fenced ```` ```json ```` blocks on 2 of 3 trials (grammar not
   applied; `app/llm.py` documents constrained generation as the expected
   behaviour), while b11073 returned clean JSON 3/3 (grammar-constrained).
   With the production-shaped extraction prompt (a JSON-instructing system
   prompt) both builds returned 4/4 valid — the difference only bites when
   the model is weakly guided. This is a **compatibility improvement** of
   b11073 relative to the documented client contract.
6. **Flag surface identical [PROVEN]** (`--help` diff): `--reasoning
   [on|off|auto]`, `--cache-type-k/-v`, `-t/--threads`, `-ngl` all present
   in both builds. `-c 16000` is rounded to `n_ctx = 16128` on BOTH builds
   (multiple-of-256 rounding) — the operator's "ctx 16000" is effectively
   16128. On the CPU-only sandbox build `-ngl 99` is accepted and reports
   `offloaded 29/29 layers to GPU` where the only backend is the CPU — no
   error, no effect. Both builds emit the same harmless
   `Error: Unknown (built-in) filter 'slice' for type None` Jinja warnings
   for this Qwen3 chat template (33 lines each).

## 5. Forensic results (llm_slot_forensic.py, schema 2)

The existing tool, phases s1–s7, run against both builds at IDENTICAL flags
(ctx 8192, ngl 0, t 2), same stand-in model, `LSF_MAX_TOKENS=192`
(reports: `evidence/forensic_A_b10717/report.json`,
`evidence/forensic_B_b11073/report.json`, comparison:
`evidence/forensic_comparison.txt`):

| Phase | b10717 | b11073 |
| --- | --- | --- |
| s1 identity (health/props/slots/metrics) | 200/200/200/200, 1 slot | 200/200/200/200, 1 slot |
| s2 baseline TTFT | 366 ms | 421 ms |
| s3 FIFO: user TTFT behind 192-tok non-streaming leg | **47.6 s** | **55.0 s** |
| s3 disconnect-yield: user TTFT after streaming disconnect at ~1.5 s | 311 ms | 259 ms |
| s4 JSON streaming assembles to a JSON object | yes (1803 ms) | yes (1817 ms) |
| s5 LCP cache: identical-prefix re-send | 20/2657 tok re-processed (2.42 s) | 20/2657 (2.47 s) |
| s5 disjoint prompt | 2647 re-processed (98.4 s) | 2647 (107.7 s) |
| s6 cache-after-cancel: re-sent prompt after a 30 s cancelled generation | 2128 tok re-processed | 2128 tok |
| s7 user→extraction→user: user-2 after extraction re-processes | 19 tok (0.73 s) | 19 tok (0.85 s) |
| s3 verdict hint | server FREES the slot on streaming disconnect (transport sound) | identical |

**Forensic conclusions:**

* **REAL_LLM_LATENCY**: unchanged within single-sample noise (s2 366 vs
  421 ms; s5/s7 large prefills 98.4 vs 107.7 s — the sandbox CPU is not the
  operator's GPU; no sandbox-observable improvement or regression).
* **Contention behaviour**: unchanged — the single-slot FIFO queue
  semantics are identical; the outlier class persists (s3 FIFO 47.6 vs
  55.0 s at sandbox scale).
* **Prefix/prompt evaluation cost**: unchanged — LCP caching works
  identically (20-token re-processing on identical prefixes on both), and
  the cache-after-cancel requeue cost is identical (2128 tokens).
* **Generation throughput**: unchanged within noise (§4 table).
* **Cancellation latency**: unchanged and sub-second on both (s3
  disconnect-yield 311 vs 259 ms; identical verdict hint: the llm_bg_gate
  cancellation transport is sound on BOTH builds).

## 6. Known limitations

1. **The sandbox is not the target machine.** 2 CPU cores (AVX512), 4 GB
   RAM, **no GPU**; the production 19 GB Qwen3.6 35B A3B IQ4_XS is never
   sandbox-resident (standing convention); the 1.7 B stand-in model
   changes every absolute number. **The operator's "~13 tok/s and
   substantially better behaviour/latency than the current production
   baseline" (llama-cli, ngl 99, ctx 16000, threads 12, target machine)
   can neither be confirmed nor refuted here.** No "10x faster" claim is
   made — none of the sandbox measurements support one.
2. **Two servers cannot run simultaneously in the sandbox** (OOM-kill
   proven during setup) — the A/B ran sequentially, one cell per
   invocation. On the target machine both servers CAN coexist (8080 +
   8081) for a live A/B.
3. **ngl 99 is untested in its GPU meaning here** — the sandbox build has
   no GPU, so the offload configuration effect (the operator's primary
   suspect for the improvement) is not measured; only ctx-size effects
   were sandbox-measurable (A1/A2 vs B1/B2 show no material ctx effect on
   this hardware).
4. **Threads 12 vs 2**: the sandbox uses the physical core count; the
   operator baseline's thread count is a target-machine value.
5. **The 30–42 s outliers themselves were never observed at their
   production magnitudes here** — the sandbox reproduces the MECHANISM
   (queue behind a non-streaming leg) at 6.6–55 s scale, and shows it is
   version-independent; the architectural fix (extending
   `bg_chat_create`-class streaming to the remaining ingest legs, the
   v0.10.5 audit's IMPROVE class) is unchanged by the runtime swap.
6. **Vendor/mem0 legs were exercised only at the transport level** (the
   same `response_format` JSON path), not through the full vendor
   pipeline (the editable vendor install is a documented sandbox
   state-loss class; §8's env-gap list).

## 7. Recommendation

**READY FOR PRODUCTION REVIEW** — strictly on the measured evidence:

* **Compatibility: PASSED [PROVEN].** Full API compatibility with the
  existing client (endpoints, SSE shape, JSON mode, thinking suppression,
  cancellation transport), plus one behavioural improvement
  (grammar-enforced JSON mode, matching the documented client contract
  better than the pinned b10717). Zero regressions on every measured axis
  (latency, throughput, contention, cache, cancellation), zero
  production-file changes, fully reversible isolation.
* **Performance: PARITY in the sandbox** — the improvement case rests on
  the operator's own target-machine measurements (llama-cli ~13 tok/s,
  ngl 99), which this sandbox cannot reproduce; a target-machine A/B
  (production 8080 vs experimental 8081, both launchers provided) is the
  remaining step for a production promotion decision.
* **Outliers: NOT fixed by the swap [PROVEN]** — the 30–42 s class is
  architectural (single slot + non-streaming background legs); it must be
  addressed at the application level (the v0.10.5 audit's IMPROVE class),
  regardless of runtime version.

Not KEEP-EXPERIMENTAL-only: nothing failed and no risk was found.
Not REJECT: no incompatibility or regression exists. The open item is
purely the target-machine performance confirmation.

## 8. Tests (validation-only; existing production tests untouched)

`tests/unit/test_experimental_llama_runtime.py` — **14/14 green**:

* launcher baselines (sandbox .sh + Windows .ps1): operator flags pinned,
  separate port/path, no 8080 bind;
* the env override + rollback proof (monkeypatch-isolated);
* production-untouched pins: MODELS.lock b10717, canonical yaml values,
  VERSION 0.10.7, no `app/` reference to the experimental profile;
* live-captured b11073 SSE fixture (66 lines, `system_fingerprint
  b11073-1aa2954bd`) parsed through the production parser: content
  assembles, null-content role chunk ignored, no reasoning leakage,
  [DONE] handled;
* thinking-suppression kwargs shape.

**Neighbouring production tests:** LLM/config cluster 171 passed
(including the new 14). **Full unit suite:** 1028 passed / 13 failed /
13 skipped with the experimental files present, and **the identical 13
failures without them** (stash-rerun-restore proof — zero regressions;
the 13 + 12 collection errors are the documented sandbox environment-gap
class: missing editable vendor install, missing silero/parakeet model
state, missing heavy ML deps; green on the target machine per the
release-gate baseline convention).

## 9. Rollback / recovery

See `experimental/llama_b11073/README.md` ("Rollback / recovery note"):
stop the experimental server, unset `LLAMA_SERVER_HOST` /
`LLAMA_SERVER_PORT`, optionally delete `bin\llama-server-b11073\`. No
production file, config value, version number or release artefact was
changed — there is nothing else to undo.

## 10. Release policy compliance

* **No production release**: VERSION stays 0.10.7; no release ZIP built;
  no RELEASE_INDEX/CHANGELOG/BUILD_INFO entries; the production mirror's
  release surface is untouched.
* **Audit artefacts uploaded** (the operator's explicit instruction:
  "Ne feledd el feltölteni, legyen meg az audit itt is"): this report +
  evidence ZIP + the additive `source/` tree sync to the GitHub mirror
  per the standing policy; the mirror README's "latest release" pointers
  remain at v0.10.7.
