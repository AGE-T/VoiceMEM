# VoiceMEM — SESSION CONTINUITY CONSOLIDATED AUDIT

**Date:** 2026-09-20
**Type:** Audit only. No implementation, no production code change, no
SESSION_CHECKPOINT introduced, no SessionTracker activated, no `session_id`
passed, no `Flush()`, no LLM, no embedding, no DB mutation.
**Inputs consolidated and verified against the current tree (HEAD `a3e3a8a`):**
1. `docs/VoiceMEM_Session_Continuity_Architecture_Study.md` (2026-09-19)
2. `audit/VoiceMEM_dormant_session_pipeline_audit/REPORT.md` (2026-09-19)
3. `audit/VoiceMEM_session_backlog_forensic/REPORT.md` (2026-09-20)
**Verification result:** `git diff 51ad195..HEAD --stat -- voicemem-agent/app
voicemem-agent/vendor` is **empty** — all three documents describe the exact
live tree. 40 spot-checked claims: **all CONFIRMED**, none stale (the only
notes are cosmetic: one line-count nit and two analytical-premise refinements
already documented in the TTS-phrase report). `[PROVEN]`
This report answers task §§10–14, 16–17 and the ten §19 session questions,
and carries the decision tree forward. All file references: `vendor/voicemem/`
unless prefixed `app/`.

---

## 1. WHAT "SESSION CONTINUITY" ACTUALLY MEANS HERE (task §10)

The verified requirement is **not** transcript reproduction. It is:

> "At the start of a new conversation, the bot should have a compact, reliable
> reminder of what the previous conversation was about and what remains open."

**What survives today** `[PROVEN]`:

| Boundary | Web lane (`app/web_server.py`) | CLI lane (`app/pipeline.py`, `app/main.py`) |
|---|---|---|
| turn-to-turn | in-memory `_history` (WebSession, init `[]` :1869; prompt uses last 8 entries :158, :3088-3091; 14000-char budget :186, :213-234) | **nothing** — `build_messages(history=None)` (pipeline.py:353) |
| WebSocket reconnect | **lost** (per-connection object; browser has no reconnect — `voicemem.html:4040` — and sends no session identifier) | n/a |
| process restart | **lost** | **lost** |
| app restart | **lost** | **lost** |
| "session expiry" | no expiry concept exists | none |
| new session, same space | `_history` empty; **semantic memory only** (vendor store persists) | same |

The vendor memory chain persists across everything (SQLite + JSON mirror),
but it stores **extracted facts**, not conversational state — it cannot
reconstruct "what we were doing". The gap is real and precisely bounded:
**one compact, bounded, per-user reminder of the previous conversation.**

## 2. THE DORMANT VENDOR MACHINERY (task §11) — verified answers

All answers re-verified against the current tree this audit:

1. **Where session_id exists / is passed / populated:** it exists only in the
   vendor pipeline signature (`orchestrator.py` Ingest, `record_turn`) and in
   row metadata (`graph_query_activations.session_id`). **The app never
   populates it** — zero hits for `session_id` in `app/` + `web/`; the web
   ingest (`web_server.py:708-721`) and the CLI bridge
   (`voicemem_bridge.py:640-646`) omit it. `[PROVEN]`
2. **Flush() called?** Only by the vendor facade (`core.py:126`), the unused
   `memory_api.py:125` wrapper, and two test teardowns. **No production
   caller anywhere, ours or upstream's.** `[PROVEN]`
3. **record_turn stores content?** **Lifecycle only** — a turn counter and
   the last opaque session token in `session_state (user_id, last_session_id,
   turn_count, updated_at)`; plus boundary bookkeeping rows in `touched_refs`.
   No topic, no state, no transcript. `[PROVEN — schema read]`
4. **Does subgraph/query activation state persist?** Yes: `rb_slot_long`,
   `subgraph_pool`, `graph_query_activations` rows persist **forever** in the
   space SQLite — `rb_slot_long` drains only via the session-boundary path
   (never taken: orchestrator.py:1436), `subgraph_pool` only via
   `RunSubgraphCheckpoint` (brain.py:704-706), and `graph_query_activations`
   has **no DELETE path at all** (the only DELETE in the whole vendor tree is
   `pop_touched`, session_tracker.py:130). `[PROVEN]`
5. **Does activation create additional LLM work?** Yes — the boundary batch
   fires 3 `_llm_json` judges per candidate (subgraph_manager.py:342/377/400)
   plus 1 schema refresh per slot, 1 long-term attribution per slot, and 1+M
   short-term attribution per entity — **all plain non-streaming 15 s-timeout
   `create()` calls** (`orchestrator.py:609-651`); the cooperative-cancel
   transport (`bg_chat_create`/`check_cancel`) exists **only** in the
   extraction chain (`extract_facts_openai.py:27-28, 397, 654`). `[PROVEN]`
6. **Can that work block the single LLM slot / run during foreground?**
   Activation hooks `_finish_ingest`, which runs inside the background gate
   worker — i.e. inline in the *new session's* background chain. The legs are
   **not cancellable**, so a user speaking mid-burst cannot reclaim the slot
   until the current leg finishes (bounded per leg at 10–15 s, but the burst
   is a queue of them). The measured analogue (uncancellable non-streaming
   vendor leg) produced the 299.7 s / 40–60 s TTFT classes. `[PROVEN — the
   non-cancellable transport; INFERENCE — burst magnitude on real data]`
7. **Unbounded growth?** Yes as a class: `rb_slot_long` + `subgraph_pool`
   never drain; `graph_query_activations` grows with every retrieval ×
   activated entity, `session_id` always NULL, no deletion. Magnitude:
   `[UNKNOWN]` — no production DB exists in this sandbox (see the backlog
   forensic + its addendum); the read-only SQL pack for the operator is the
   closing step.
8. **Production examples using it?** Upstream's own shipped examples, web UI
   and evals all omit `session_id`/`Flush` — the machinery is dormant
   upstream too. `[PROVEN — grep of the upstream clone at pin]`
9. **Is it a general session-lifecycle infrastructure or vendor-internal?**
   Vendor-internal: a **turn/session-boundary trigger ledger** whose design
   purpose (module docstring) is to *trigger batch jobs of the left and right
   brain* — attribution and subgraph checkpointing — not to remember
   conversational content. `[PROVEN]`

**Verdict on activation (Lane A): rejected for this requirement.** It does
not store content, so it cannot serve the requirement by itself; activating
it would wake a chain of non-cancellable LLM bursts competing for the single
slot, and its backlog/unbounded-growth mechanics are structurally proven even
if their magnitude is unmeasured. The Session Continuity Architecture
Study's Option F2 (wire the dormant machinery) is confirmed as the wrong
foundation; the KV-lane checkpoint (D/F1) stands. `[INFERENCE from PROVEN
facts — same conclusion as all three source documents]`

## 3. THE SMALLEST SAFE SESSION NOTE (task §12)

**Storage locations compared:**

| Option | Verdict | Evidence |
|---|---|---|
| A. existing VoiceMem KV (`kv(k TEXT PK, v TEXT)`, space.py:120-124) | **chosen** — O(1) point read, upstream-provided, already used for 4 housekeeping keys (`leftbrain_store`, `rb_cleanup_last_count`, `audio_cleanup_last_run`, `archive_cold_last_run`); micro-bench: 669 B representative checkpoint, `kv_get` hit p50 0.065–0.068 ms, `kv_set` p50 0.070 ms, JSON parse p50 0.0033 ms, round-trip equality OK (re-run this audit on a scratch DB: 0.065 ms — reproduces the recorded numbers) | `[PROVEN]` |
| B. space JSON | viable but the KV lane is strictly simpler and measured | — |
| C. dedicated session JSON file | new file-surface + crash-consistency questions the KV already solves | — |
| D. dedicated SQLite table | a new database object for one bounded row — over-engineering; violates "no new database unless proven inadequate" | — |
| E. semantic memory | **rejected**: would pollute retrieval, is not bounded per user, and semantically is not a memory ("what we were doing" is not a fact about the world) | — |
| F. hybrid | unnecessary at this size | — |

**Minimum useful schema** (from the study, unchanged in verification):

```json
{ "session_id": "...", "timestamp": "...", "topic": "...",
  "current_state": "...", "decisions": [], "open_items": [],
  "last_user_goal": "...", "next_step": "...", "language": "..." }
```
bounded to a few hundred bytes — the representative serialised example
measured 669 B. Single replaceable row per user; writer-bound size;
easy to validate/read/delete; independent from semantic memory; immune to
history explosion by construction. `[PROVEN — size/latency facts;
INFERENCE — schema adequacy]`

**Namespace (task §17 — space isolation):** the only identity dimension in
the actual architecture is `user_id` (web spaces = `webspace_<name>` user
ids on one shared SQLite root; CLI = `voice_user`/per-speaker; **no
language or assistant dimension exists anywhere**). KV keys are **global
per SQLite file with no user column** — identity must be embedded in the key.
Therefore `session_checkpoint:<user_id>` is the correct and sufficient
namespace today: user_id ≡ space on the web lane and is the stores' own
isolation boundary; `session_checkpoint:<space>:<user>` would double-encode.
Two concurrent WebSessions on one space degrade to last-writer-wins —
benign for a single advisory row (and F2 activation would be far worse:
`session_changed` would fire on every alternating ingest). `[PROVEN — key
scope; INFERENCE — namespace sufficiency]`

## 4. GENERATION AND LIFECYCLE (task §§13–14)

**How the note should be produced** (options evaluated against the
production constraints — single slot, gate, crash behaviour):

| Candidate | Verdict |
|---|---|
| A. deterministic extraction | weakest content quality; always available |
| B. **one bounded LLM summary at session end** | **chosen** — the only LLM touch, one bounded call, run as a gate-able background leg |
| C. at inactivity timeout | equivalent trigger variant of B |
| D. hybrid deterministic + LLM | deterministic skeleton + LLM polish — viable refinement, not required for v1 |
| E. periodic rolling checkpoint | rejected without a demonstrated requirement (continuous summarisation is explicitly excluded by the task) |

**Lifecycle discipline (from the study, re-verified):**

```
SESSION START (first turn of a WebSession / CLI run)
  → kv_get(session_checkpoint:<user_id>)  [O(1), ~0.07 ms]
  → freshness check (timestamp age; expired → ignore)
  → if present: inject a SHORT preamble into the system prompt
  → normal flow (behaviour with no checkpoint == today's, exactly)
conversation proceeds
  → session becomes inactive (Web: WS disconnect or idle timeout;
     CLI: process exit path)
  → generate/update checkpoint (ONE bounded LLM call, submitted through
     the background gate; hard timeout; deterministic fallback on failure)
  → kv_set (replace-in-place)
next session → load checkpoint
```

- **Inactivity detection:** Web — WebSocket disconnect (the only reliable
  signal; `_history` dies there today anyway) or an idle timer; CLI —
  exit/Ctrl+C path. Perfect session-end detection is **not** required: a
  checkpoint written at timeout/disconnect is an *advisory reminder*, not a
  transaction.
- **Crash / checkpoint-generation failure:** log → ignore → conversation
  unaffected. **The failure iron law holds by design: a checkpoint error can
  never break the normal flow** (read path: missing key = today's behaviour;
  write path is background and timeout-bounded). `[PROVEN — read-path
  semantics; design guarantee for the write path]`
- **Expiry:** data-level TTL via the `timestamp` field (advisory staleness
  bound, e.g. days — an old checkpoint should expire rather than mislead).
- **How many may exist:** exactly one per user_id (replace-in-place). No
  history of checkpoints — bounded by construction.
- **Concurrency:** single advisory row, last-writer-wins; no locking needed
  beyond SQLite's own.

## 5. THE TEN DECISION QUESTIONS (task §19, session)

1. **Is existing semantic memory sufficient?** No — it stores facts, not
   conversational state; it cannot produce "what we were doing / what remains
   open". `[PROVEN — store semantics]`
2. **Is SessionTracker useful for our actual requirement?** No — it stores
   no content; it is a trigger ledger. `[PROVEN]`
3. **Does it solve content continuity?** No. `[PROVEN]`
4. **Would activating it introduce new LLM work?** Yes — the boundary batch
   (3 judges/candidate + per-slot refresh/long-term + per-entity short-term),
   non-cancellable, inline in the next session's background chain. `[PROVEN]`
5. **Is a small application-level session checkpoint sufficient?** Yes —
   it matches the requirement exactly (compact reminder, bounded, advisory).
   `[INFERENCE]`
6. **Where stored?** The existing space KV, key `session_checkpoint:<user_id>`.
   `[PROVEN — lane fitness; INFERENCE — final key]`
7. **How generated?** One bounded LLM digest at session end/inactivity, via
   the background gate; deterministic fallback; hard timeout.
8. **When updated?** At session boundary (disconnect/idle/exit) — never
   per-turn.
9. **Minimum schema?** §3 (nine fields, ~669 B representative).
10. **What deferred?** Lane A activation (rejected, revisit only if the
    requirement ever becomes *vendor attribution/subgraph* work rather than
    conversation continuity); multi-checkpoint history; per-assistant or
    per-language checkpoint dimensions (no such identity exists today).

## 6. DECISION TREE

```
Requirement: "remember where we left off" at next session start
├── Need vendor batch attribution / subgraph checkpointing?  → NO (different
│   problem; if ever yes, first run the backlog SQL pack + a Flush dry-run
│   on a COPY of the production DB — see the backlog forensic §7)
├── Need full transcript replay?                             → NO (history window
│   + semantic memory already cover in-session context)
└── Need a compact previous-conversation reminder?           → YES
      → KV-lane SESSION_CHECKPOINT (single bounded row per user_id)
        → read at session start (O(1), ~0.07 ms, expiry-checked)
        → write at session boundary (one gated LLM digest, deterministic
          fallback, hard timeout; failure never breaks the flow)
        → validate: unit tests for schema/fallback/expiry; integration test
          that a missing/corrupt checkpoint leaves behaviour byte-identical
```

## 7. TEST STRATEGY (task §21, session portion)

Required before any implementation: new session (no checkpoint → behaviour
identical to today, pinned by test); continuation (checkpoint loaded,
preamble injected once); reconnect/process-restart (checkpoint survives;
_history correctly does not); checkpoint replacement (single row, new wins);
malformed checkpoint (log + ignore); generation failure (timeout → fallback,
no flow impact); expired checkpoint (ignored); multiple spaces/users
(`webspace_A` vs `webspace_B` vs `voice_user` — no leakage); concurrent
sessions on one space (last-writer-wins, no crash); shutdown during
generation (orphan-free: either a complete new row or the old one).

## 8. WHAT REMAINS UNKNOWN (honest ledger)

- **Production backlog magnitudes** (`rb_slot_long`, `subgraph_pool`,
  `graph_query_activations`, `touched_refs` per namespace) — no production DB
  in this sandbox; the read-only SQL pack (backlog forensic §7) on a **copy**
  of the operator's `memory/memory.sqlite` is the single closing step. This
  does not affect the Lane-B decision (the checkpoint is a new, bounded key).
- First-burst cost of a hypothetical Lane-A activation on real data —
  `[UNKNOWN]`, and moot given the rejection.
- LLM digest quality for the checkpoint (which model leg, prompt, token
  budget) — an implementation-time experiment, gated and timeout-bounded by
  design.

## 9. COMPLIANCE

Production code changes: **0**. Commits of code: **0**. `Flush()`: **0**.
`session_id` passed: **0**. SessionTracker activation: **0**.
LLM/embedding runs: **0**. DB mutations: **0** (the KV micro-bench ran on a
scratch DB in `/tmp`). Prior-artefact modifications: only the dated
re-verification addendum appended to the backlog forensic report.
