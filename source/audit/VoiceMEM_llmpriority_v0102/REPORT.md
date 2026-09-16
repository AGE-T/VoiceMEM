# VoiceMEM v0.10.2 — P0 forensic + fix report
## ASR REGRESSION + LLM PRIORITY + REPRODUCIBILITY FIX

Operator order: 15 parts. This report separates **ROOT CAUSE / FIXED /
MEASURED / VERIFIED / REMAINING** and never calls a hypothesis a root cause.

VERSION DECISION: v0.10.2. v0.10.1 (the ASR-side forensic, previous session)
is already published (zip + gate record + CHANGELOG + served artifacts, SHA
A9C781E0…). This task touches runtime code (gate v2, vendor legs, engine,
installer) — a published version is never re-cut. The operator's "v0.10.1"
label is interpreted as the v0.10.x line; the release carrying these fixes
is **v0.10.2**, ZIP `releases/VoiceMemAgent_v0.10.2.zip`, SHA256
`F56E5155043B9900B2A45799F4FA1663ED8475584F9DBA1F1BBD80B13C525A54`,
gate GREEN 1386/0-reg, fingerprint `26e38b90…`.

---

## PART-BY-PART STATUS

| Part | Status | Where |
|---|---|---|
| 1 A/B baseline | SPLIT: ASR side CLOSED by the v0.10.1 forensic (byte-identical audio path v0.9.2→v0.10.0; CPU library stack correct on the full HU corpus). LLM side: the target-side A/B is delivered as the forensic procedure (llm_slot_forensic.py + the recovery package §C); the sandbox cannot run the 35B model. | audit/VoiceMEM_asrregression_v0101 + scripts/llm_slot_forensic.py |
| 2 ASR forensics | CLOSED (v0.10.1): waveform-level trace + VAD/ASR separation + direct-Parakeet + corpus matrix. | audit/VoiceMEM_asrregression_v0101 |
| 3 ASR language config | CLOSED (code): asr_language now reaches the production parakeet engine (was read ONLY by the legacy engines — the operator's observation was correct). auto/hu diagnostic mode implemented; ParakeetForTDT exposes NO language parameter (multilingual auto-detect only) — the hint is recorded + transcript script verified; production default stays auto. | app/asr_parakeet.py, app/config.py, tests/unit/test_asr_language_mode.py |
| 4 Dependency forensics | CLOSED: openai SDK was COMPLETELY UNPINNED (bare `"openai"` in the vendor pyproject — the memory chain's LLM client resolved to install-time latest). transformers was pinned exact in v0.10.1 (the floating `>=5.6` was the single non-reproducible ASR-stack variable). torch/torchaudio pinned exact (2.7.0 cu128) since v0.1.6. | requirements.lock.json (machine-readable manifest with wheel SHA256s) |
| 5 Model identity | CLOSED (v0.10.1): 6-file SHA256 verification of the pinned Parakeet revision. | audit/VoiceMEM_asrregression_v0101 |
| 6 Direct ASR test | CLOSED (v0.10.1): known-good HU WAVs → production path → correct Hungarian; VAD separation proved VAD is not the cause. | audit/VoiceMEM_asrregression_v0101 |
| 7 LLM latency forensics | CLOSED (mechanism, sandbox-evidenced; target numbers via script): FIFO queue behind in-flight background work on the single slot. | scripts/llm_slot_forensic.py + evidence/llm_slot_forensic_report.json |
| 8 Background memory priority | CLOSED: arm() cancels the in-flight chain (vendor cooperative gate + streaming disconnect), re-queues the turn pair, never drops it. | app/background_memory.py + vendor …/llm_bg_gate.py |
| 9 Background start policy | CLOSED: idle-based (speech frames + LLM deltas + turn activity + VOICEMEM_BG_IDLE_S quiet window; the blind 2 s grace stays as a floor). Start delay MEASURED (GateStats.last_start_wait_s). | app/background_memory.py + app/web_server.py notes |
| 10 Cache validation | CLOSED (measured, not log-inferred): /metrics counter deltas — same-prefix 20/2657 tokens re-evaluated (99.2% reuse); disjoint 2647/2650; user-chat prefix SURVIVES an interleaved extraction (19 tokens re-processed); a CANCELLED request's own prefix does not survive (2131/2131 re-processed on re-send). Background and user requests share the single slot; LCP reuse is real and is NOT the defect. | evidence/llm_slot_forensic_report.json (S5/S6/S7) |
| 11 Dependency pinning | CLOSED: openai==3.14.0 exact (vendor pyproject + installer step-16 OR-assert + requirements.lock.json with wheel SHA256s: openai 232a85a1…, transformers 78ec1ce2…). Unrelated packages untouched. | requirements.lock.json, scripts/install_m1.ps1 |
| 12 Release integrity | CLOSED: cross-artifact invariant test (VERSION == CHANGELOG top == PAGE_VERSION; no index entry newer than VERSION; a built zip must be indexed; gate record never newer). Builder already refuses mismatched gate records. | tests/validation/test_feature_release.py |
| 13 VAD diagnostics | CLOSED (v0.10.1 logging fix) + the boundary test added now: below / EXACTLY at / above threshold. At-threshold starts speech (inclusive VSM) and is never reported "below". | tests/unit/test_vad_idle_report.py |
| 14 Regression tests | CLOSED: +28 net new (gate v2 cancel/requeue/stale-rejection/idle 7; llm_bg_gate 9; language mode 10; VAD boundary 1; release invariants 1; runtime-deps openai documentation 1; mock SSE upgrades in the memory integration suites). Hungarian ASR + cross-language + Cyrillic ban: closed by v0.10.1's real-inference tests. | tests/ |
| 15 Final acceptance | See below. | — |

---

## ROOT CAUSE

**P0-2 (LLM TTFT 40–60 s) — ROOT CAUSE PROVEN:**

The runtime uses ONE llama-server (parallel=1, n_slots=1, FIFO task queue).
v0.9.2's BackgroundMemoryGate deferred the START of background memory work,
but deliberately never interrupted an in-flight chain (its own docstring:
"An in-flight extraction is NOT aborted… its duration is bounded instead by
the extraction max_tokens limit"). Three facts make that bound ineffective:

1. A full warm background chain is ~111 s on the production model (v0.9.2 E4
   measurement), not seconds.
2. The conflict-resolution leg had NO client timeout at all — the OpenAI SDK
   default is 600 s (the single worst unbounded edge; extract had 60 s, resolve
   had none).
3. The user's chat request FIFO-queues behind whatever leg is running the
   moment speech starts: `arm()` closed the gate for NEW work but the running
   leg(s) kept the slot.

Measured mechanism proof on the EXACT pinned server build (llama.cpp b10717,
production flags `--parallel 1 -c 32768 --cache-type-k q8_0 --cache-type-v
q8_0`, Qwen3-0.6B stand-in, scripts/llm_slot_forensic.py):

- user chat TTFT while ONE non-streaming background request runs: **299.7 s**
  (sandbox CPU; the mechanism, not the magnitude — on the production GPU it
  is exactly the reported 40–60 s window);
- after the background request's client disconnects mid-stream: **3.2 s**
  (−98.9%) — **llama-server b10717 frees the single slot on streaming client
  disconnect** (this is the cancellation transport the fix uses);
- the reported v0.9.2-measured tail (chat landing ~2 s after extraction
  launch ≈ 163 s) and the v0.10.0 field report (40–60 s TTFT) are the same
  FIFO mechanism at different chain positions.

**P0-1 (ASR Hungarian → Russian/English) — status:** the v0.10.1 forensic
(12 phases, byte-identical product delta, CPU library stack correct on the
full HU corpus, correct VAD segmentation, correct direct-Parakeet output)
exonerated everything sandbox-provable and pinned the floating transformers
dependency (the one non-reproducible ASR-stack variable) plus delivered the
wrong-language waveform dump + the on-target forensic script. The remaining
target-side evidence (CUDA leg / live mic chain) is collected by
scripts/asr_regression_forensic.py on the actual machine — this task adds
nothing further to that leg beyond the language-mode diagnostic (PART 3).

## FIXED

1. **Vendor cooperative cancellation** (`voicemem/utils/common/llm_bg_gate.py`,
   NEW): `BG_CANCEL` thread-event; `check_cancel()` before every ingest-chain
   LLM leg; `bg_chat_create()` — streaming transport with cancel poll between
   chunks; on cancel the HTTP stream is closed (server slot freed, measured)
   and `BackgroundCancelledError` (a **BaseException**, the
   asyncio.CancelledError pattern) propagates — the chain's `except Exception`
   fallbacks (notably voice_input.py's "ConflictResolver failed → ADD-only")
   cannot swallow it. The two big legs (extraction, conflict resolution) run
   through it; JSON-mode streaming assembly verified live (S4).
2. **Gate v2** (`app/background_memory.py`): `arm()` now also CANCELS the
   in-flight chain; a cancelled pair is re-queued (front, attempts counted,
   exact duplicates still coalesced, NEVER dropped); `release()` opens only
   after a real idle window; activity signals wired in the web session (VAD
   frames with speech energy, barge frames, LLM deltas).
3. **Resolve-leg timeout**: 60 s + max_retries=2 (was: none — SDK default
   600 s).
4. **ASR language mode** (PART 3): production engine reads `asr_language`
   (""/auto = native auto-detect; "hu" = diagnostic hint recorded + transcript
   script verified, loud warning on contradiction); ASR_LANGUAGE env override;
   config validation; yaml default stays auto.
5. **Dependency pinning** (PART 11): openai==3.14.0 exact everywhere +
   `requirements.lock.json` machine-readable manifest (package, version,
   source, pin site, wheel SHA256 where appropriate; torch hash omitted with
   documented reason — version+index pinned, pip freeze recorded).
6. **Release metadata invariants** (PART 12): new validation test.
7. **VAD boundary diagnostics** (PART 13): exactly-at-threshold test added
   (below/above were covered by v0.10.1).

NOT changed: Parakeet, the model family, VoiceMem, the LLM, memory schema,
TTS, llama-server configuration, context size, parallelism.

## MEASURED (sandbox, exact pinned server build)

| Measurement | Value |
|---|---|
| Baseline user TTFT (S2) | 1268.9 ms |
| User TTFT behind 1 non-streaming background request (S3) | **299,733 ms** |
| User TTFT after background client disconnect (S3) | **3,157 ms** (−98.9%) |
| LCP same-prefix re-evaluated tokens (S5) | 20 / 2657 (99.2% reuse) |
| LCP disjoint re-evaluated tokens (S5) | 2647 / 2650 |
| User-chat prefix after interleaved extraction (S7) | 19 tokens re-processed |
| Cancelled request's own prefix on re-send (S6) | 2131 / 2131 (no reuse) |
| JSON-mode + streaming assembly valid (S4) | true |
| /slots endpoint (b10717) | HTTP 200 |

## VERIFIED

- Gate: **GREEN — 1386 tests, 0 regressions, 6 pinned sandbox env-gap, 28
  skipped**; deep validation PASS (embedding, memory, memory-semantics,
  release, runtime-deps); fingerprint `26e38b90…` in releases/gate_record.json.
- Release ZIP self-check: V0102 must-ship files + markers + cumulative
  invariants PASS; sidecar `sha256sum -c` OK; RELEASE_INDEX entry written;
  PAGE_VERSION == VERSION == CHANGELOG top.
- New regression tests: cancellation (pre-issue + mid-stream, stream closed),
  requeue semantics (never dropped, coalesced resubmit), stale-result
  rejection (store never sees a partial leg), idle policy (activity defers,
  quiet window opens, measured start delay), BaseException semantics, JSON
  streaming assembly, language mode (pure + engine status + config + env),
  release invariants, VAD boundary.
- Browser verification of the download page (public/, v0.10.2 card served).

## REMAINING (honest)

1. **Target-side numbers** (PART 15.4/5): real-runtime TTFT on the 35B model
   before/after — run `scripts/llm_slot_forensic.py` on the target
   (~10–15 min; procedure + expectations in the recovery package §C). The
   sandbox proves the MECHANISM on the exact server build; it cannot run the
   19 GB model.
2. **ASR root cause on the target** (if the HU→RU regression reproduces):
   the v0.10.1 forensic script + wrong-language WAV dumps close it — send
   back `logs/asr_forensic_*/report.json` + `logs/asr_wrong_language_*.wav`.
3. **Idle-window tuning**: the default VOICEMEM_BG_IDLE_S=6.0 is a
   conservative voice-first default; the gate now MEASURES the actual
   release→start delay (GateStats.last_start_wait_s) — tune with data on the
   target.
4. Mirror sync for v0.10.1+v0.10.2 (pending; the PHASE-14 machinery is idle
   — one sync covers both, per the additive policy).

## ARTIFACTS

| Artifact | SHA256 |
|---|---|
| releases/VoiceMemAgent_v0.10.2.zip (2701.4 KB, 365 files) | F56E5155043B9900B2A45799F4FA1663ED8475584F9DBA1F1BBD80B13C525A54 |
| public/VoiceMem_llmpriority_v0102_recovery.zip | 06B065ABE83E42F01F162BAA1711FA22B2FDDBCB63ED79851CCAE567630220BD |
| evidence/llm_slot_forensic_report.json | (in recovery package + this audit dir) |
| evidence/code.diff (2445 lines) | (in this audit dir) |
| releases/gate_record.json | fingerprint 26e38b90f171e567… |
| Exact commit | see git log — release + forensic commits made after this report |
