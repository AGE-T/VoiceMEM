# VoiceMEM 0.10.5 CONSOLIDATED AUDIT — EXECUTIVE SUMMARY

**Date:** 2026-09-20 · **Tree audited:** git HEAD `a3e3a8a`
(= v0.10.5 release + one unreleased production fix) · **Discipline:**
audit only — zero production code changes, zero merges, zero DB mutations,
zero LLM/embedding runs. All conclusions tagged `[PROVEN]` / `[INFERENCE]` /
`[UNKNOWN]`. Full evidence in the five companion reports under
`voicemem-agent/audit/`.

---

## A. EXACT CURRENT BASELINE

v0.10.5 (VERSION, pyproject, config yaml, RELEASE_INDEX all agree), released
2026-09-18 from commit `0a135a6`, zip SHA-256 `401814AC…` (verified
byte-identical to sidecar, self-consistent inner BUILD_INFO, gate GREEN
1478/0/4). **The working tree differs from the release by exactly one
production change** — the TTS leading-absorption fix `edac5c9`
(2026-09-19, `app/text_utils.py`). Working tree otherwise clean; vendor
controlled-fork pin `e8384e0` intact. `[PROVEN]`

## B. GIT REPOSITORY FINDINGS

History is linear and release-disciplined (v0.10.0→v0.10.5, every release
build-gated; dedicated recovery packages exist for v0.10.1–v0.10.4 only).
Three findings needing operator attention:
1. **The `edac5c9` fix is unreleased** — the public v0.10.5 zip does NOT
   contain it, and `TTSForensicFix.zip` is only the audit evidence package.
   A version build is required to distribute it.
2. The fix's gate run was **RED-environment** (sandbox model state loss),
   with stash-compared fix-innocence proof — the fix has never been gated on
   a fully restored environment; the next build must be.
3. Cosmetic: stale `BUILD_INFO.json` (0.7.1) in the tree; UUID-subject
   snapshot commits; tracked `dev.pid`. `[PROVEN]`

## C. UPSTREAM CHANGES WORTH PORTING

**Exactly one small item: the `/api/debug/log` browser→backend error-reporting
hook** (idea from upstream's experimental RTC branch `b36b4fa`, adapted to
our logging stack, ~30 lines, app-layer only, optional). Upstream main is
**static since 2026-09-05** — nothing else new exists to port. `[PROVEN]`

## D. UPSTREAM CHANGES TO REJECT OR DEFER

**Reject:** mem0 embedder passthrough (`a8d5c76` — violates the E5-only
invariant); the memory-language chain (`a48e90f`/`6f282e8`/`0472c41` —
mono-language per space, en/zh only, contradicts our bilingual single store);
Breeze TTS (licence/remote); the RTC transport branch; zh-demo display
cosmetics. **Defer (with triggers):** attribution schema shrink; reversible
barge pause (hard worklet dependency); trait script guard; heard-text
timeline. Everything functional we compete on — we are already ahead
(VM-LOCAL-001…021 + the v0.10.3 port trio). `[PROVEN]`

## E. MIXED-LANGUAGE ROOT PROBLEM

Chunk-level detection misroutes *embedded* foreign phrases; solved by the
evidence-gated span mechanism (v0.10.5) plus leading absorption (`edac5c9`).
The residual root problem is **zero-orthographic-evidence idioms** ("break a
leg"): unfixable without a lexicon, deliberately host-routed, operator-
endorsed trade-off. This audit additionally found **two new deterministic
defect classes**: N1 (`ee`-bearing loans like "meeting" sweeping a following
HU word into the EN span — "A 2026-os meeting fontos…" routes "meeting
fontos" to EN) and N2 (accent-free HU host flips — "Holnap lesz a meeting…"
detects EN as host). Both are word-level table fixes within the standing
constraints. `[PROVEN — reproduced; traffic magnitude UNKNOWN]`

## F. IS A SEPARATE PHRASE DETECTOR ACTUALLY REQUIRED?

**No.** The evidence-gated phrase-run mechanism *is* the phrase detector —
shipped, bounded, 81 pinned tests, correct on every evidence-bearing class.
A lexicon/statistical detector is operator-forbidden today and unnecessary
for the evidence-bearing classes. `[PROVEN]`

## G. IS LLM STRUCTURED LANGUAGE OUTPUT VIABLE?

**No, not in the TTS hot path — structurally.** Segmentation input exists
only after the LLM emits a chunk; TTS needs it immediately; on the
single-slot llama-server any second request queues behind the whole remaining
generation (proven 30–163 s classes) or delays first audio by its own
round-trip (+6–17 s/chunk best case). All seven approaches (A–G) evaluated;
every in-path variant fails streaming, latency or reliability. The only
defensible LLM shape is **off-path telemetry** (G). `[PROVEN mechanism]`

## H. SAFEST HYBRID ARCHITECTURE

Deterministic evidence-gated segmentation remains the **sole hot-path
authority** (27–133 µs/call, zero concurrency surface, invariant-preserving
join). Optional future: a bounded, operator-re-authorised idiom list gated
like existing rules (technically best-in-class; forbidden today); LLM
validation only as offline QA. Fallback on any failure is always the
deterministic path. `[PROVEN + INFERENCE]`

## I. SESSION CONTINUITY ROOT PROBLEM

Web `_history` (8-entry window) dies with the WebSocket; CLI has no history
at all; nothing survives restarts except extracted facts in semantic memory,
which cannot express "what we were doing / what remains open". The real
requirement is exactly **one compact previous-conversation reminder** —
nothing more. `[PROVEN]`

## J. SHOULD SESSIONTRACKER REMAIN DORMANT?

**Yes.** It stores lifecycle only (turn counter + opaque session token), no
content — it cannot serve the requirement; activating it would fire
non-cancellable 15 s-timeout LLM bursts inline in the next session's
background chain (slot-blocking class), and its tables never drain
(`rb_slot_long`, `subgraph_pool`, `graph_query_activations` have no DELETE
paths). Upstream's own examples never use it either. `[PROVEN]`

## K. IS A SMALL SESSION CHECKPOINT SUFFICIENT?

**Yes** — it matches the requirement exactly: bounded (~669 B representative),
advisory, replaceable, independent of semantic memory, immune to history
explosion by construction. `[INFERENCE from PROVEN facts]`

## L. WHERE SHOULD THE CHECKPOINT LIVE?

In the **existing space KV** (`kv(k TEXT PK, v TEXT)`), key
**`session_checkpoint:<user_id>`** — the only identity dimension that exists
(user_id ≡ space; no language/assistant dimension exists). Measured:
`kv_get` p50 0.065–0.070 ms, `kv_set` p50 0.070 ms, JSON parse 0.003 ms.
Generation: one bounded gated LLM digest at session end (disconnect/idle/
exit), deterministic fallback, hard timeout; **failure can never break the
conversation flow**. `[PROVEN lane fitness; INFERENCE final schema]`

## M. WHAT SHOULD BE IMPLEMENTED NEXT (priority order)

1. **Release the `edac5c9` fix** through a proper version build, gate run on
   a restored environment, CHANGELOG stamp, mirror sync. (Closes the
   release-integrity gap — B.)
2. **N1/N2 deterministic detector fixes** (word-level table corrections,
   µs-class, test-pinnable; operator triage first).
3. **KV-lane SESSION_CHECKPOINT** (Lane B) per the consolidated report —
   smallest safe continuity solution, with its §7 test battery.
4. **`/api/debug/log` browser error hook** (E1) — optional, tiny, useful for
   all future field forensics.
5. **Operator runs the backlog SQL pack** on a copy of the production DB
   (read-only; script already delivered) to convert the backlog magnitudes
   from UNKNOWN to measured.

## N. WHAT SHOULD NOT BE IMPLEMENTED

LLM structured segmentation in the TTS hot path; SessionTracker/Flush
activation (Lane A); any lexicon/statistical phrase detector without explicit
operator re-authorisation; the upstream memory-language chain; the RTC
transport; per-turn summarisation; a new database or new identity dimensions
for the checkpoint; chunking changes (operator-forbidden).

## O. WHAT MUST BE MEASURED ON THE REAL RTX 5070 ENVIRONMENT FIRST

1. **Full release gate on a restored environment** for the `edac5c9`-carrying
   build (the RED-environment class must not ship).
2. **Real-traffic frequency of N1/N2** (from production logs) — sizes the
   detector fix's value.
3. **Checkpoint digest leg quality/time** (one gated bounded LLM call on
   the real model) — implementation-time experiment, timeout-bounded.
4. **Backlog magnitudes** via the delivered SQL pack on a DB copy
   (`rb_slot_long`, `subgraph_pool`, `graph_query_activations`, `touched_refs`
   namespaces, KV census) — closes every remaining UNKNOWN in the session
   reports.
5. If any LLM segmentation variant is ever reconsidered: on-target TTFT and
   grammar-on/off decode rate for a ~300-token constrained request, plus a
   labelled span corpus (none exists today).

---

### Report index

| Report | Path |
|---|---|
| Git repository forensic | `audit/VoiceMEM_git_repository_forensic/REPORT.md` |
| Upstream selective merge | `audit/VoiceMEM_current_upstream_selective_merge/REPORT.md` |
| TTS phrase architecture | `audit/VoiceMEM_tts_phrase_architecture/REPORT.md` |
| Session continuity consolidated | `audit/VoiceMEM_session_continuity_consolidated/REPORT.md` |
| Backlog forensic (2026-09-20 + addendum) | `audit/VoiceMEM_session_backlog_forensic/REPORT.md` |
| Prior art (verified, still valid) | `audit/VoiceMEM_dormant_session_pipeline_audit/REPORT.md`, `docs/VoiceMEM_Session_Continuity_Architecture_Study.md`, `audit/VoiceMEM_tts_codeswitch_v0105_{forensic,unquoted}/`, `audit/VoiceMEM_upstreamaudit_v0103/` |
