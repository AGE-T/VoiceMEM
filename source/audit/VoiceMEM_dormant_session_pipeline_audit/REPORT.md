# VoiceMEM — DORMANT SESSION PIPELINE AUDIT

**Date:** 2026-09-19
**Type:** Audit only. No implementation, no production code change, no commit,
no dependency/config change, no SESSION_CHECKPOINT introduced, no
SessionTracker activated. Production behaviour untouched.
**Question under audit:** is the dormant vendor SessionTracker / `session_id`
/ `Flush()` infrastructure the right foundation for session continuity, or
is an app-level bounded SESSION_CHECKPOINT in the existing space KV the
correct direction?
**Companion document:** `docs/VoiceMEM_Session_Continuity_Architecture_Study.md`
(2026-09-19) — the placement study whose Option F2 ("wire the dormant vendor
session machinery") is exactly what this audit evaluates. That study
provisionally recommended the KV-lane checkpoint (D/F1) and explicitly
deferred the F2 activation decision pending runtime evidence; this document
is that deferred examination.

All statements below are **direct code-reading evidence** with file:line
references unless explicitly marked `[inference]` or `[needs runtime
measurement]`. The vendor tree is the controlled fork at pin `e8384e0`
(`VOICEMEM_PIN.json`); "orchestrator.py" etc. refer to
`vendor/voicemem/voicemem/…` unless prefixed `app/`.

---

## 1. SESSIONTRACKER — COMPLETE SEMANTICS

### 1.1 What it is (file: `utils/common/session_tracker.py`, 134 lines)

A SQLite-backed **turn/session-boundary trigger ledger**. Its own module
docstring states its purpose: track per-user ingest turns + session
boundaries *"供左右脑批处理任务触发用"* — to trigger batch jobs of the left
and right brain. Two designed trigger signals:

- every 3rd ingest (`is_third_turn`) → right-brain short-term attribution;
- `session_id` change (`session_changed`) → right-brain long-term
  attribution + left-brain subgraph checkpoint.

**It stores no conversational content.** No topic, no state, no decisions, no
open items, no transcript. A `session_id` is an opaque string token to it.
It cannot, by itself, remember "where we left off" — it can only *detect*
boundaries and *count*.

Schema (session_tracker.py:39-56), in the **space sqlite** (the single
`<memory_root>/<dir>.sqlite`, orchestrator.py:497-499 → `_space.db()`):

```sql
session_state (user_id TEXT PRIMARY KEY, last_session_id TEXT,
               turn_count INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL)
touched_refs  (user_id, namespace, ref, created_at,
               PRIMARY KEY (user_id, namespace, ref))
```

Thread-safety: a **new sqlite connection per operation**, WAL pragma per
connection (session_tracker.py:33-37). Construction: lazy singleton per
Orchestrator (facade) in `self._cache` (orchestrator.py:493-500) — one
instance per facade, all on the same file.

### 1.2 API surface and exact semantics

| Method | Lines | Semantics |
|---|---|---|
| `record_turn(user_id, session_id)` | 58-91 | Called **once per `Ingest()`** (orchestrator.py:1342). Returns `{turn_count, session_changed, is_third_turn}`. `session_changed` is True **only when old AND new session_id are both non-empty and differ** (line 81). Docstring: *"if the caller never passes session_id, this trigger mechanism never starts, it cannot mis-fire"* (63-66). |
| `get_current_session(user_id)` | 93-101 | Last recorded session_id, else `None`. Consumed by `compute_rho()` session-scoping (see 1.5). |
| `touch(user_id, namespace, ref)` | 105-111 | `INSERT OR IGNORE` into the ledger. |
| `count_touched` / `pop_touched` | 113-133 | Count / take-and-clear a namespace's accumulated refs. |

Critical transition detail (line 81): the **first** `record_turn` that ever
receives a non-null session_id after a NULL history returns
`session_changed=False` (old value is NULL). Activation is therefore
inert on day one; the boundary fires from the **second** distinct session
onward. Deterministic from code — no measurement needed.

### 1.3 What the boundary batch actually runs

`session_changed=True` → `Orchestrator._run_session_boundary_batch()`
(orchestrator.py:1368-1369, 1413-1440), which runs **inline inside the
ingest that detected the change**:

1. **`RunSubgraphCheckpoint()`** (orchestrator.py:716-718 →
   leftbrain/brain.py:704-724 → `SubgraphManager.run_for_retrieved_pool`,
   slot_split/subgraph_manager.py:146-177):
   pops the accumulated `subgraph_pool` touched refs (retrieval accounting
   from `Search`), builds an entity co-occurrence graph, greedy densest-
   subset peeling with the ρ(H) synergy score, then — if above
   `MIN_SYNERGY_THRESHOLD` — **three independent LLM verification calls**
   (relevance / importance / completeness; `_judge_subgraph`,
   subgraph_manager.py:263-408, `self._llm(...)` at 342/377/400). On
   approval: **creates a dynamic slot, migrates entity `slot_ref`s, and
   re-tags memories** (`tag_fn` → `upsert_memory_tags`). This *changes
   future retrieval routing* — it is a semantic-memory mutation.
2. **`_refresh_schema_descriptions()`** (leftbrain/brain.py:898-959): for
   every slot with ≥3 memories and ≥`_SCHEMA_DESC_MIN_NEW` new since last
   refresh — **one LLM call each** writing a ≤40-word synthesis into
   `slot_summaries`. These summaries are **read on every search**
   (leftbrain/brain.py:808-810; memory_repository_v2.py:307-308) and ride
   `SearchResult.related_summaries` into the prompt. Activation therefore
   changes the **content of the memory context block** on subsequent turns.
3. **Long-term attribution** (rightbrain/attribution_manager.py:100-118):
   for every touched `rb_slot_long` slot — one `_summarize_slot` LLM call
   (120-139) rewriting the slot's persona description.

`Flush()` (orchestrator.py:1446-1457) = the same boundary batch **plus**
short-term attribution on the remaining `rb_entity_short` refs. Docstring:
call once *"when the conversation/session formally ends"* to back-fill the
last session (which has no successor ingest to trigger it). Idempotent when
no refs are pending.

Short-term attribution (`run_short_term`, attribution_manager.py:36-75) —
per entity: one `_summarize_entity` LLM call + one `_refine_memory_item`
LLM call per un-refined memory item. Its trigger today is
`n ≥ SHORT_TERM_MIN_MEMORIES` (default **20**, env
`VOICEMEM_ATTRIBUTION_MIN_MEMORIES`, orchestrator.py:77) **or**
`session_changed` (orchestrator.py:1360).

### 1.4 The touched-refs namespaces — who writes, who drains

| Namespace | Written by | Drained by | State in our deployment |
|---|---|---|---|
| `rb_pending_memories` | orchestrator.py:1355-1358 (memory ids + heartnote id per ingest) | the 20-memories trigger / `session_changed` (1360-1361) | **active** (threshold path) |
| `rb_entity_short` | rightbrain/brain.py:717 (per trait-entity write) | short-term trigger (1362) and `Flush()` (1453) | **active** (threshold path) |
| `rb_slot_long` | rightbrain/brain.py:718 (per trait-slot write) | **only** the boundary batch (1436) | **accumulates forever — never drained** |
| `subgraph_pool` | leftbrain/brain.py:687-689 (`Search` → `_record_subgraph_activation`, every real retrieval) | **only** `RunSubgraphCheckpoint` | **accumulates forever — never drained** |

This is a factual correction to the shorthand "the vendor session machinery
is dormant": `record_turn` **runs on every ingest today** (with
`session_id=None`, so `turn_count` increments and `session_state` rows
exist per user_id), and every `Search` records query activations into
`graph_query_activations` with `session_id NULL`
(leftbrain/brain.py:697-700). The **dormant** parts are precisely:
`session_changed` (never fires — no session_id is ever passed), the whole
boundary batch (subgraph checkpoint, schema-refresh-via-boundary,
long-term attribution), and `Flush()` (no production caller). Consequence:
on a long-lived production space, `rb_slot_long` and `subgraph_pool` hold
an **unknown-magnitude backlog** that the first activated boundary batch
would attempt to process in one synchronous run (`pop_touched` takes
everything). `[needs runtime measurement: production DB row counts]`

Note: `is_third_turn` is **dead code in the orchestrator** — it appears
only in session_tracker.py and its docstring (grep across the vendor tree).
The comment at orchestrator.py:1344-1347 records why: measured 7.0 s of an
18.4 s ingest when attribution ran every turn — the vendor itself moved
from a turn-count trigger to a memory-count trigger. That is direct
evidence the vendor's own evolution was *away* from per-turn batch work.

### 1.5 Which components activate in the presence of session_id

- `get_current_session()` becomes non-None → `compute_rho()` scopes Q to
  the current session's query activations instead of the user's full
  history (graph_entity_store.py:139-174; the docstring cites the paper's
  Algorithm 1 ρ(H): "Q is the current session's query set").
- `voice_input.py` threads session_id into `repo.update_memory(...)` /
  `count_occurrence(...)` (voice_input.py:679, 697, 712, 734) where it is
  **metadata only** (`meta["session_id"]`, `meta["last_observed_session"]`,
  mem0_backend_store.py:365-366, 450-451). Readable turn ids
  (`turn_ids.format_turn_id`) become session-scoped.
- The audio perceiver keeps session-scoped speaker-binding state
  (`_session_person_pin` / `_person_origin_session`,
  utils/audio/perceiver.py:170-177) — **irrelevant in our deployment**:
  both surfaces construct the facade in `text_mode`
  (app/web_server.py:625, app/voicemem_bridge.py:501), no audio reaches
  the vendor.
- `paper_emotion_detector.py:152` hardcodes `session_id="ingest"` —
  not on our path.

**LLM calls that activation would add** (all through
`Orchestrator._llm_json` / `_llm_text`, orchestrator.py:609-651): 3 per
subgraph candidate above threshold, 1 per schema-refresh-eligible slot,
1 per long-term-attribution slot, (1 + un-refined items) per
short-attribution entity. **Embeddings: none new** — the subgraph legs use
SQL/graph data plus precomputed entity embeddings; attribution legs are
text-only. **Threads/queues/daemons: none new** — the batch runs inline in
the caller's thread. But see §4.B/J: every one of these LLM calls is a
**plain, non-streaming, 15 s-timeout OpenAI-client call** — only the two
big extraction legs are cancel-aware (`bg_chat_create`/`check_cancel`,
used exclusively in leftbrain/extract_facts_openai.py:27-28, 397, 654;
grep proof across the vendor tree).

---

## 2. UPSTREAM / ORIGINAL INTENT

**Evidence status: the upstream repository history is NOT available in this
environment.** The vendor tree ships vendored (`vendor/voicemem`, no `.git`);
`VOICEMEM_PIN.json` records only that it is a controlled fork of
`https://github.com/xzf-thu/VoiceMem.git` at commit `e8384e0…` (tag
v0.0.1, package 0.2.3, fork created 2026-09-10). When it appeared, which
commit introduced it, and whether a later upstream change deliberately
stopped using it **cannot be determined here — explicitly flagged**. The
app repository's own git log contains no commit discussing the session
machinery's dormancy. What follows is intent reconstructed from code text
only, marked as such:

- `[inference from docstrings]` The tracker was built as the **batch-scheduler
  seam** for attribution and subgraph work — the module docstring names the
  two consumers (right-brain attribution, `core.py::RunSubgraphCheckpoint`),
  and `graph_query_activations` carries a `session_id` column + index
  explicitly for the paper's session-scoped ρ formula.
- `[inference]` `Flush()` exists because `session_changed` can only be
  inferred *forward* ("see the first ingest of the next session",
  orchestrator.py:1419-1421): the last session before a conversation ends
  has no successor ingest, so an explicit end-of-conversation call is needed.
- `[inference]` The vendor's `memory_api.py` (the one-line integration for
  external chat systems) exposes `flush()` as the "conversation formally
  ends" hook (memory_api.py:122-125) — i.e. the designed caller of `Flush()`
  is an **integrator's chat loop**, not a voice pipeline. Our app does not
  use `memory_api.py` at all (it drives the `VoiceMem` facade directly).
- `[inference]` The fork's own integration tests call `vm.flush()` purely
  as best-effort resource release (tests/integration/test_memory_safety.py:
  353, test_temporal_memory.py:334) — not as a behavioural consumer.
- `[inference]` No app-side document, changelog entry, or comment states a
  deliberate decision to leave the machinery unwired; the honest reading is
  **it was never wired**, rather than deliberately switched off.

### 2.1 ADDENDUM (2026-09-20, backlog-forensic session) — upstream history located

The statement above ("upstream repository history is NOT available") was
written before the full upstream clone at `tool-results/voicemem-src` was
identified in this workspace. It **is** available (remote
`https://github.com/xzf-thu/VoiceMem.git`, HEAD = pin `e8384e0`, 2026-08-27,
26 commits). What it actually contains — checked read-only this session:

- The repo's code arrived in **bulk import commits** ("Import current
  contents from lang-jiaqi/Voicemem_open", 2026-08-25). `session_tracker.py`
  has **exactly one** commit (the import); `orchestrator.py` has two (both
  imports). There is **no per-feature commit history** for the session
  machinery — it appeared fully formed. The one later code sync
  (`801f9a3`, "Sync latest changes from development repo", 2026-08-26)
  touched README/web assets only.
- Every later commit is README/paper-link/demo-video work — 26 commits
  total, none touching session code.
- **No shipped caller uses the machinery**: the upstream examples
  (`examples/01_memory.py`, `examples/03_simple_agent_with_voicemem_
  memory.py`) call `vm.ingest(...)` **without** `session_id` and never call
  `Flush()` (the `flush()` in example 03 is a TTS chunk-buffer closure, not
  the vendor API); the vendor's own web demo (`web/run.py`) and the
  evaluation harness (`evaluation/`) likewise have zero `session_id` /
  `Flush()` usage. The only caller-facing surface remains
  `memory_api.Memory.flush()` (memory_api.py:122-125).
- The vendor ships `voicemem_memoryspace/{demo,myvoicemem}/.gitkeep` —
  empty placeholders, no data.

Net effect on §2's conclusions: the flagged inferences stand, now with
positive evidence — the public upstream history contains **no evolution and
no usage** of the session machinery, so "designed as a library capability
with a thin integrator wrapper, never wired by any shipped caller" is the
strongest supportable reading. The "which commit introduced it / was it
deliberately deactivated" questions remain unanswerable (single-import
provenance) — now as a proven fact about the history's shape rather than an
availability gap.

---

## 3. CURRENT APPLICATION CALL GRAPH

### 3.1 Web (primary surface)

```
Browser WS connect
  → WebSession.__init__            (app/web_server.py:1854-1919)
      _history: list[dict] = []    (:1869)          ← full session transcript, RAM only
      BackgroundMemoryGate(_gated_memory_ingest)    (:1906-1912) — session-owned
  → run() loop                      (:2065-2153)
      user speech → VAD → ASR (or typed user_text)
      → _start_turn → memory_gate.arm(source)       (:2791; VAD arm :2366)
      → _guarded_turn → _run_turn                   (:2831+)
          memory search: MemoryLayer.search → vm.search(query, emotion)
                        (web_server.py:703-707)      ← vendor Search, 0 LLM
                          └─ _record_subgraph_activation EVERY retrieval
                             (orchestrator.py:853 → forwarder :712-714 →
                             leftbrain/brain.py:679-702): touches
                             subgraph_pool + records query activations
                             with session_id=None
          system prompt + self._history[-8:]         (:3088-3091, _HISTORY_WINDOW=8 :158,
                                                       budget :213-234)
          LLM streaming (single llama-server slot) → chunked TTS
          reply end:
            _history.append(user, assistant)         (:3320-3321)
            memory_gate.submit(user_text, reply)     (:3329)
      gate worker (app/background_memory.py:391-469): waits grace 2 s + idle 6 s
        → asyncio.to_thread(_gated_memory_ingest)
          → RealMemoryLayer.ingest(text, reply, wait=True)   (web_server.py:2659-2666)
            → vm.ingest(text, agent_reply=..., async_facts=False)  (:709-721)
              ← session_id NEVER passed — vendor Ingest(session_id=None)
                 (orchestrator.py:1002-1013 accepts it; nothing sends it)
  → disconnect / close (msg websocket.disconnect or receive() error)
      → finally: _cancel_session_tasks()             (:2137-2153, 2668-2692)
        — cancels turn/audio/gate-worker tasks; _history is LOST;
          the gate's still-queued (user_text, reply) pairs DIE with the
          worker task (background_memory.py:439-446 — "session teardown:
          let the queue die with the task")
```

**Where a session_id could be minted / where it is lost today:**
mint at `WebSession.__init__` (one per WS connection — the natural unit);
plumb through `MemoryLayer.ingest(text, reply, wait=…, session_id=…)`
(web_server.py:709-721 — one added parameter) into
`vm.ingest(..., session_id=...)`. Today it is lost because
`RealMemoryLayer.ingest` does not accept or forward one. **Where Flush could
be called:** `run()`'s finally block (after/instead of
`_cancel_session_tasks`) — but note the gate worker is already dead there,
so `Flush()` would run synchronously on the event loop or need a fresh
server-level task.

### 3.2 CLI

```
process start (app/main.py:561+ run_real)
  → MicStream → _vad_loop → utterance_q → _turn_loop      (:765-799)
  → pipeline.handle_utterance                             (app/pipeline.py:268-400)
      _retrieve_memory → bridge.process_turn → vm.search  (voicemem_bridge.py:549-577)
      build_messages(transcript, system_prompt)           (pipeline.py:353)
        ← NO history parameter — every CLI turn is context-free
      LLM streaming → TTS
      _commit_reply_with_retry → bridge.commit_reply       (pipeline.py:406-441)
        → vm.ingest(user_text, agent_reply=reply, async_facts=True)
          (voicemem_bridge.py:638-650) ← session_id NEVER passed; the
            vendor's own fire-and-forget daemon thread finishes the chain
  Ctrl+C → CancelledError → finally                       (main.py:818-828)
      cancel tasks, llm.aclose(), "Viszlát!" — nothing persisted;
      pipeline._pending_commits (failed-commit queue) dies with the process
```

No idle timeout exists on either surface (no WS ping/timeout configuration
in web_server.py; the CLI is a live mic loop). **No explicit
conversation-reset protocol message exists** (the app's WS text protocol
handles only `user_text`, `mic_diag`, `mic_probe` — web_server.py:2155-2179).

### 3.3 Storage topology (matters for everything below)

Both surfaces construct facades with the **same `memory_root`**
(`cfg.memory_root_path` → `<repo>/memory` — app/config.py:307-314;
web_server.py:629, voicemem_bridge.py:505). "Spaces" on the web are
**user_id scoping, not directory scoping**: every facade gets
`user_id = webspace_<name>` on the shared root
(web_server.py:613-631). Therefore: ONE sqlite file
(`memory/memory.sqlite`) shared by web + CLI + all spaces; the `kv` table
in it has **global string keys** (no user column — space.py:120-124), while
`session_state`/`touched_refs` are keyed by `user_id` and thus properly
isolated per space/speaker.

---

## 4. WHAT WOULD HAPPEN IF WE ACTIVATED THE VENDOR SESSION LIFECYCLE

### A) Adding `session_id` to `ingest()` calls

- `record_turn` begins recording real session identity. First-ever
  non-null → `session_changed=False` (old is NULL — session_tracker.py:81),
  so activation day is inert; the **second** session's first ingest fires
  the first boundary batch.
- Memory rows start carrying `session_id` / `last_observed_session`
  metadata (mem0_backend_store.py:365-366, 450-451) — additive metadata,
  no store behaviour change.
- `compute_rho` scopes to session queries (semantic change in subgraph
  scoring, but subgraph scoring itself is dormant today).
- Foreground-path latency: **unchanged** — record_turn is 2-3 SQL
  statements already on the ingest path (it runs today with None).

### B) `Flush()` activation — which vendor mechanisms start

The boundary batch (§1.3): subgraph checkpoint (3 LLM calls per eligible
candidate + possible dynamic-slot creation + memory re-tagging), schema
refresh (1 LLM per eligible slot), long-term attribution (1 LLM per touched
slot), plus short-term attribution leftovers (1 + N LLM calls). **All are
plain non-streaming calls** (orchestrator.py:609-651) with 15 s timeouts —
the exact class the llm-priority audit measured as the worst
foreground-blocking mechanism (see §10).

### C) `session_changed` — side effects

By vendor design the boundary batch for session N runs **inside the first
ingest of session N+1** ("inferred from seeing the next session's first
ingest" — orchestrator.py:1419-1421). With the web gate wiring, that is
**inside the new session's gate worker**, i.e. the *new* session's second
turn can queue behind the *old* session's boundary work on the single
llama-server slot, and the legs are not cooperatively cancellable (§1.5).

### D) Subgraph checkpoint — does it auto-start?

No. Only via the boundary batch / `Flush`. But its **input pool is already
accumulating** today (`subgraph_pool` never drained — §1.4), and
`graph_query_activations` rows accumulate with NULL session ids. The first
activated boundary would consume the whole backlog at once.
`[needs runtime measurement: backlog size on the production DB]`

### E) Schema refresh — does it auto-start?

The boundary-batch writer (leftbrain/brain.py:898) is dormant. **Nuance
found by this audit:** an *independent opportunistic writer exists and is
live* — `LeftBrainMemoryRepositoryV2._maybe_update_slot_summary`
(memory_repository_v2.py:322-341) fires a daemon-thread LLM call on
fallback-path searches when a slot gained ≥10 new memories
(`repo.search` is reached from `Rank`'s supplement/fallback branches —
leftbrain/brain.py:491, 499). So `slot_summaries` are not strictly empty
today; activating the boundary batch adds the second, larger writer.

### F) Attribution (short-term)

Already active at the 20-memories threshold (§1.4). `session_changed`
would add a second trigger; `Flush()` would add a third.

### G) Attribution (long-term)

Dormant (`rb_slot_long` never drained). Activation runs it over the whole
accumulated backlog at the first boundary.

### H) Background LLM / embedding — new resource contention?

No new embeddings (§1.5), no new threads/queues. New **LLM load**: one
burst per session boundary whose size is backlog- and slot-count-dependent.
Worst-case shape: dozens of sequential non-cancellable 15 s-timeout calls
on the single slot.

### I) Concurrency with BackgroundMemoryGate / llm_bg_gate

- Gate-serialized ingests: the boundary batch inside the gated ingest is
  serialized behind the conversation like any background chain **for
  starts** — but `arm()`'s cooperative cancellation does **not** reach these
  legs (`check_cancel`/`bg_chat_create` exist only in
  extract_facts_openai.py). The measured mitigation (streaming disconnect
  frees the slot in 3.2 s — audit REPORT §FIXED) **does not apply** to
  them; a user speaking mid-batch FIFO-queues (the 40-60 s production
  window class).
- A `Flush()` at WS close runs **outside any gate** (the session's gate is
  dead) — no idle discipline, no cancellation, worst shape: the *next*
  session's first turn queues behind the previous session's flush.

### J) Latency

Foreground user-path latency is unaffected by `record_turn` itself (SQL,
sub-ms, already running). The risk is **conditional burst latency** at
boundaries, in the measured worst class: the v0102 forensic measured one
non-streaming background request holding the single slot → user chat TTFT
**299.7 s (sandbox reproduction; the production-GPU manifestation is the
reported 40-60 s window)** — audit/VoiceMEM_llmpriority_v0102/REPORT.md:63,
121. The boundary batch is N such requests back-to-back.
`[needs runtime measurement: production-like Flush() wall time + LLM call
count on a copy of the real space]`

### K) Memory semantics

Activation changes the **read path over time**: schema-refresh summaries
enter every subsequent `SearchResult.related_summaries` (brain.py:808-810)
→ the memory context block's content changes; subgraph checkpoint may
create dynamic slots and re-tag memories → future slot routing changes;
long-term attribution rewrites rb slot descriptions → persona prompt
content changes. These are **semantic-memory behaviour changes**, not
mere bookkeeping. They may be *desirable* improvements — but they are
exactly the class of change the operator's stability discipline requires
to be evaluated on its own evidence, not adopted as a side effect of a
session-continuity feature.

### L) Failure behaviour

Each leg catches its own exceptions and prints (e.g.
orchestrator.py:1425-1426, 1432-1433, 1438-1440, 1456-1457) — a failed
boundary batch does not corrupt the store, and a failed `Flush()` raises
nothing outward. A mid-batch **process crash** leaves completed legs'
writes and drops the rest (no transaction spans the batch); the touched
refs already popped are lost from the ledger (pop-then-process), so those
items will never be re-processed — acceptable for batch jobs, worth
knowing for planning. `Flush()` blocking/timeout at close: each LLM call
has a 15 s client timeout; there is no overall budget, no watchdog.

### ★ Highlighted risk — "turning on a switch that lights up other rooms"

Activating `session_id` + `Flush()` for session continuity **is not a
narrow actuation**: it simultaneously enables (i) the subgraph
dynamic-slot subsystem with memory re-tagging, (ii) the boundary-gated
schema-refresh writer, (iii) long-term attribution over an unmeasured
backlog, (iv) session-scoped ρ scoring — each with its own LLM cost,
its own writes into semantic memory, and **no cooperative cancellation**.
The checkpoint feature needs **none** of these. This coupling is the
central finding of the audit: the vendor "session lifecycle" is a
**batch-consolidation scheduler**, not a session-continuity primitive.

---

## 5. THE TWO ARCHITECTURES COMPARED

**Lane A — activate the dormant vendor session lifecycle and build on it:**
`session_id` → `SessionTracker` → `Flush()` → vendor session machinery →
SESSION_CHECKPOINT riding that machinery (e.g. a checkpoint column/table
next to `session_state`, written by a boundary hook).

**Lane B — keep the vendor lifecycle dormant; app-level checkpoint:**
app-side SESSION_CHECKPOINT → existing space KV (`kv_get`/`kv_set`) → own
bounded schema → O(1) read at session start → write at session boundary.

Characteristic-by-characteristic (no scores, no ranking):

| Axis | Lane A (activate vendor lifecycle) | Lane B (app-level KV checkpoint) |
|---|---|---|
| Implementation complexity | App: mint session_id, plumb through 2 layers, wire Flush at 2 close paths; **plus** boundary-batch budgeting/cancellation work to make it safe (that work does not exist today and is non-trivial: the legs are not cancel-aware — a vendor patch or an acceptance of the measured contention class) | App: writer + reader/renderer + freshness policy at the same 2 close/start paths; no vendor change |
| Upstream divergence | Activation itself is app-side (the API already exists), but making the boundary batch safe on a single-slot server plausibly requires vendor patches (cancel-awareness for `_llm_json`/`_llm_text` legs) → new VM-LOCAL entries, permanent rebase surface | Zero vendor change; the kv lane is already shipped and already used by the fork (throttles) |
| Production behaviour change | **Large and semantic**: new LLM batch subsystems, slot summaries/dynamic slots/persona descriptions begin changing memory content and future retrieval | None when the checkpoint is absent/corrupt; a bounded prompt block when present |
| Runtime overhead (steady state) | record_turn SQL (already running today) | one sub-ms kv read per session start; optional sub-ms per-turn accumulator write |
| Runtime overhead (boundary) | one LLM burst per session boundary, non-cancellable, measured worst class | one sub-ms kv write; optional one bounded LLM digest (cancellable/timeoutable by design, with deterministic fallback) |
| LLM overhead | 3 LLM/candidate + 1/slot + 1/slot + (1+M)/entity per boundary (+backlog burst at first activation) | 0 required (deterministic digest); ≤1 optional per session end |
| Embedding overhead | none new | none |
| Concurrency risk | boundary legs ignore `BG_CANCEL`; a `Flush()` at close has no gate at all | digest call designed behind the same idle/cancel discipline; checkpoint write is SQL, not LLM |
| Failure modes | failed legs print and drop; popped-then-crashed ledger refs are lost forever; backlog burst unbounded a priori | corrupt JSON → `kv_get` returns default → block omitted (self-heals on next write); write failure prints and is non-fatal (space.py:137-145) |
| Lifecycle handling | session identity is first-class (vendor tables); turn counts for free | session identity is app-minted (uuid/iso-ts); turn counts for free (the app already knows its turns) |
| Web support | requires gate-death-at-close workaround for Flush | natural (WS close finally block) |
| CLI support | natural (process finally) | natural (process finally) |
| Process restart / crash recovery | previous session's un-flushed batch work is silently skipped (by design) | previous checkpoint persists (dated); per-turn accumulator optionally bounds loss |
| Space switching | `session_state` rows are per user_id — isolation is native | kv keys must embed the space/user id (the table is global-keyed — §6) |
| Multi-session (two tabs) | tracker is last-writer-wins per user_id; boundaries interleave | last-writer-wins on one key; timestamps make interleaving visible |
| TTL / freshness | none in the vendor machinery | data-level: `ended_at` + age policy (data, not code) |
| Testability | needs the vendor batch systems live (or heavily mocked) to test the boundary path | writer/reader/freshness are pure app units; round-trip test across a simulated restart |
| Backward compatibility | activation changes retrieval content over time (schema summaries, dynamic slots) — old baselines (memperf cardinality, prompt-token measurements) drift | none (prompt block only, bounded, guarded by the existing 14 000-char budget) |
| Recovery | n/a beyond above | fallback chain: fresh → dated-stale → deterministic → absent |
| Future extensibility | vendor-native session ledger (per-session rows, queryable) if ever needed | single key → last-N ring is a bounded list in the same key |
| Upstream compatibility | activation couples the app to vendor batch internals whose evolution is upstream's | the app depends only on `kv_get`/`kv_set` (a shipped, used API) and its own code |

**What Lane A genuinely offers that Lane B lacks:** a first-class,
queryable per-session ledger in the place the vendor already owns "session"
(SessionTracker tables), vendor-native turn ids, and — the real prize —
the un-dorming of schema refresh / long-term attribution / subgraph
checkpoint as *memory-quality features in their own right*. **What Lane B
lacks in exchange:** nothing for the stated capability (a compact
"where we left off" state); the checkpoint needs identity addressing, not
batch scheduling.

**Key structural fact:** the two lanes are **not mutually exclusive and
not alternatives for the same job**. Lane A is a *memory-consolidation*
decision; Lane B is a *conversation-state* decision. A checkpoint in the KV
lane is unaffected if the vendor lifecycle is later activated; the vendor
lifecycle provides no storage a checkpoint needs.

---

## 6. THE KV STORAGE LANE (deep dive)

Location: the `kv` table inside the space sqlite
(`utils/common/space.py:118-145`).

- **Schema:** `kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)` — created
  idempotently per connection (:120-124). No user/space column, no TTL
  column, no size column. **Keys are global per sqlite file**; since all
  surfaces and spaces share one root in this deployment (§3.3), the
  checkpoint key must embed the identity itself, e.g.
  `session_checkpoint:webspace_demo` / `session_checkpoint:voice_user`.
- **Access pattern:** a **fresh sqlite connection per call**
  (`_kv_conn`, :120-124); `kv_get` = SELECT + `json.loads` with
  **every exception swallowed → default** (:127-134); `kv_set` = single
  `INSERT … ON CONFLICT(k) DO UPDATE` inside the `with` transaction
  (:137-145), failures printed, never raised.
- **Atomicity/transactions:** one-statement upsert inside one connection's
  transaction — atomic at the sqlite level; no torn JSON on the read side
  (partial writes either commit or don't).
- **Concurrency:** per-call connections + SQLite file locking; the file is
  persistent WAL (SessionTracker sets `journal_mode=WAL` per connection on
  the same file, :36, and WAL is a file-level persistent property), so
  readers don't block the writer. Existing contention precedent: the same
  table already carries the left-brain JSON mirror
  (`leftbrain_store`, memory_repository.py:117-138) — a potentially **large**
  blob — written during ingests, alongside the throttle keys. A
  ≤1-2 KB checkpoint value is comfortably within demonstrated usage.
- **Existing keys (collision check, exhaustive):** `leftbrain_store`
  (memory_repository.py:117), `rb_cleanup_last_count`
  (rightbrain/brain.py:896-903), `audio_cleanup_last_run`
  (orchestrator.py:1474-1487), `archive_cold_last_run`
  (orchestrator.py:1506-1520). A namespaced reserved key
  (`session_checkpoint:<uid>`) has **no collision risk**; versioning is a
  `"schema": 1` field inside the JSON; freshness is an `ended_at` field
  (data, not schema).
- **Corruption / missing key:** corrupt JSON → `kv_get` returns the
  default → the block is omitted → the next write self-heals. Missing key
  → same. **Crash during write:** atomic upsert — either old or new value.
- **Cleanup/TTL:** none exists in the lane (no mechanism, none needed for
  a single overwritten key). **Size limits:** none (TEXT); the practical
  bound is the writer's own cap.

**Conclusion of §6:** a single bounded JSON checkpoint fits the lane
safely, by construction and by precedent (the fork already uses it for
exactly this class of small persistent state).

---

## 7. THE SESSION-END DIGEST MODEL (C + A) — analysis only, nothing implemented

The model proposed by the previous study: **per-turn deterministic
accumulator (A) + one bounded LLM digest at session end (C) + hard timeout
+ deterministic fallback.**

- **What can be deterministic:** turn count, started_at/ended_at,
  session_id, last user utterance (truncated), last assistant reply
  (truncated) — all maintainable per turn with sub-ms kv writes to bound
  crash loss. The *abstracted* fields (topic/state/decisions/open
  items/next step) are where LLM adds value; a deterministic floor can
  only heuristically fill them (or leave them empty).
- **Is LLM needed?** For the *abstraction quality* the operator's example
  implies — yes, one call. For the *capability* — no (deterministic
  digest still gives wording-level continuity, Option-E material).
- **Sensible token budget:** prompt = six field definitions + the last
  ~10-15 turn pairs (the app holds the full transcript in RAM:
  `WebSession._history`), capped to a few K chars; output `max_tokens`
  ~200-300; `response_format json_object`. Well under the 10.4K-token
  extraction class already running per turn pair.
- **How per-turn LLM summarization is avoided:** by construction — the
  accumulator is pure bookkeeping; the only LLM call is the boundary
  digest. (The vendor's own history — is_third_turn abandoned because
  attribution cost 7.0 s/18.4 s per ingest, §1.4 — is the standing
  evidence that per-turn batch LLM work is the wrong shape.)
- **Fit with BackgroundMemoryGate / llm_bg_gate:** the gate is
  **session-owned and dies at WS close** (worker task cancelled;
  background_memory.py:439-446) — the digest **cannot ride the gate at the
  moment it is needed** (close). Two workable shapes (design-time
  decision, not made here): (a) bounded synchronous digest inside the
  close path (hard timeout, e.g. 5-10 s; the user is gone); (b) a
  short-lived **server-level** task after close, running under the same
  idle/cancel discipline. Either way the pattern should be
  **write-then-upgrade**: write the deterministic checkpoint FIRST
  (sub-ms), then attempt the LLM digest and overwrite on success —
  a cancelled/failed digest then costs only abstraction quality, never
  presence.
- **Non-blocking of the next session's foreground path:** guaranteed by
  (i) the checkpoint write being SQL, (ii) the digest being
  timeout-bounded and (ideally) cancel-on-arm aware, (iii) the next
  session's read being O(1) kv regardless of whether the upgrade
  happened.

---

## 8. SESSION BOUNDARY DETECTION

| Boundary | Detectable? | Flush/digest possible? | Crash behaviour | Fallback |
|---|---|---|---|---|
| WEB WS disconnect | **Yes, reliably** — run() finally (web_server.py:2086-2153) on close frame or TCP error | Yes — app code in the finally (gate is dead; use a bounded sync or server-level task) | nothing runs | previous checkpoint (dated) |
| Browser close | Yes — arrives as WS close/EOF, same finally | same | same | same |
| Server shutdown | **Partially** — ASGI shutdown closes connections; the finally path is not guaranteed to complete under shutdown time pressure `[needs runtime verification]` | same, time-boxed | same | same |
| Idle timeout (web) | **Does not exist** — no server-side WS timeout; a session stays open indefinitely (and `_history` grows unboundedly in RAM until disconnect) | n/a | n/a | n/a |
| Explicit conversation reset | **Protocol does not exist** (only `user_text`/`mic_diag`/`mic_probe`) | would need a new message type | — | — |
| CLI Ctrl+C | **Yes** — asyncio cancellation → finally (main.py:818-828) | Yes — in the finally | nothing runs | previous checkpoint |
| CLI process exit / kill -9 | No handler today (an atexit hook would be needed) | only via such a hook | nothing runs | previous checkpoint |
| Machine restart | No | No | nothing runs | previous checkpoint |

**Is perfect session-end detection required for session continuity? No.**
The checkpoint is a dated, overwrite-superseded nicety; the fallback chain
(fresh → dated-stale → deterministic → absent) already tolerates a missed
boundary. The per-turn deterministic accumulator (§7) bounds the loss of a
crashed session to its tail. The one honest gap: **a session that never
disconnects** (tab left open) has no boundary — acceptable for a
personal-assistant rhythm, and identical for both lanes (the vendor's
`session_changed` equally relies on a *next* ingest existing).

---

## 9. SPACE SWITCHING

- `use_space()` is a **global active-space mutation**
  (web_server.py:689-701; REST `POST /api/spaces/{name}/use` :4404-4411):
  it swaps `self._active`; facades are cached per space on the shared root.
- `SessionTracker` state is per-facade singleton on the shared file, rows
  keyed by `user_id` (`webspace_<name>`) — native per-space isolation.
- **KV checkpoints:** same file, global keys → the key **must** embed the
  space/user id (§6). With that, switching space mid-UI-session and
  reading the new space's checkpoint for the next turn is a consistent,
  cheap behaviour (lookup keyed by the space active at read/write time —
  an explicit design decision, flagged in §14).
- **Cross-space leakage:** not possible for semantic memory (user_id row
  isolation) and not possible for the checkpoint (distinct keys). The
  checkpoint is a prompt block only; no store interaction exists to leak
  through. One residual nuance: the gate worker of a live session resolves
  `_facade(self._active)` **per call** — a mid-session switch already
  affects where the *next* pair ingest lands (existing behaviour,
  unchanged by either lane).

---

## 10. PERFORMANCE / CONCURRENCY

Standing constraints (from the memory-performance and llm-priority audits):
foreground LLM must never block on background work; background LLM must
be cancellable; `BackgroundMemoryGate` / `llm_bg_gate` must not degrade;
E5 embedding contention must not worsen.

**Activating the SessionTracker (the call itself):** no new LLM, no new
embedding, no new thread/queue — `record_turn` is already on the ingest
path today with `None`; passing a real id changes a comparison, not the
cost. **This part is statically decidable and harmless.**

**Activating the boundary batch / Flush:** statically derivable cost
structure (§1.3), magnitudes not measurable here:

- LLM legs: 3 per above-threshold subgraph candidate; 1 per
  schema-eligible slot (~13 static slots + dynamic, those with ≥3 memories
  and enough new); 1 per long-term-attribution slot; (1 + M) per
  short-attribution entity.
- All legs: plain `client.chat.completions.create` (non-streaming,
  15 s client timeout) — **not cancel-aware** (only
  extract_facts_openai.py:27-28/397/654 use the cooperative gate).
- Measured mechanism (v0102 forensic, production server build): one
  non-streaming background request → user chat TTFT **299.7 s sandbox /
  the reported 40-60 s production window**; streaming client disconnect
  frees the slot in 3.2 s — the mitigation that these legs cannot use.
- First-activation backlog burst: `rb_slot_long` + `subgraph_pool`
  accumulate unboundedly today (§1.4) — the first boundary processes
  everything popped in one run. `[needs runtime measurement on the
  production DB: touched_refs counts per namespace]`
- Foreground slot contention: the boundary batch inside the gated ingest
  delays the *new* session's turns behind non-cancellable work; a
  post-close `Flush()` has no gate at all (§4.I).

No estimates are offered where measurement is required: the production
wall-time of one `Flush()` on a real space, and the first-boundary backlog
size, are the two numbers that must be measured before any activation
decision.

---

## 11. FAILURE / RECOVERY (checkpoint lane)

| Failure | Effect | Handling (existing or by construction) |
|---|---|---|
| Corrupt checkpoint JSON | unreadable | `kv_get` returns default on any exception (space.py:127-134) → log + ignore + normal flow; self-heals on next write |
| Missing checkpoint / key | absent | block omitted; behaviour identical to today |
| Half-finished write | old or new value only | single-statement atomic upsert |
| Process crash mid-session | no session-end write | previous checkpoint persists (dated); optional per-turn accumulator bounds the tail loss |
| Machine restart | same | same |
| LLM digest timeout/failure | no abstraction | deterministic digest written regardless; `digest_method` provenance field |
| Embedding timeout | **not applicable** — the checkpoint never embeds |
| SQLite lock contention | brief | per-call connections, WAL file; precedent: `leftbrain_store` blob already written during ingests |
| Space deletion | checkpoint goes with it | the kv table lives in the space's sqlite — copy-folder semantics preserved |
| Schema change / old checkpoint with newer app | parse risk | `schema: 1` field gates rendering; unknown version → omit |
| Two tabs writing at close | last writer wins | timestamps make interleaving visible; degradation documented |

**Governing principle (satisfied by the KV lane):** a checkpoint failure
must never break the normal VoiceMEM conversation flow — with the kv
lane, the failure modes are "read returns nothing" and "write prints and
continues", both already the vendor's behaviour.

---

## 12. UPSTREAM DIVERGENCE

- **Lane A:** the *activation* is app-side only (the vendor API already
  accepts `session_id`; `Flush()` is public) — no patch needed to switch
  it on. However: (i) building the checkpoint *on* the vendor machinery
  (a column/table beside `session_state`) is a VM-LOCAL patch with
  permanent rebase cost; (ii) making the activated batch safe on a
  single-slot server plausibly requires vendor patches (cancel-aware
  boundary legs) — more VM-LOCAL surface; (iii) the app becomes coupled
  to vendor batch internals (namespace names, batch semantics) that
  upstream evolves.
- **Lane B:** the app depends on `kv_get`/`kv_set` — a **shipped,
  documented, already-used vendor API** — and nothing else. All new code
  is app-side. No new abstraction is created for beauty's sake; the lane
  is the one the fork already uses for exactly this class of state.
- **Stable extension point already present:** `kv_get`/`kv_set` **is**
  the vendor's stable extension point for small persistent state. The
  checkpoint is a new *consumer* of it, not a new mechanism.

---

## 13. RELATIONSHIP TO THE CURRENT ARCHITECTURE

The checkpoint must not become: a semantic fact store, an embedding store,
emotional memory, a Learning Engine, or a second conversation history.

- **KV lane (B):** structurally outside every store the semantic path
  touches — never embedded, never retrieved, never ranked, never cleaned
  up, never extracted from (the ingest pair stream is unchanged; the
  checkpoint is only ever a system-prompt block). The Learning Engine
  plan's read-only-evidence boundary is unaffected.
- **Lane A additions** (schema summaries, dynamic slots, persona
  rewrites) are semantic-memory changes *by design* — orthogonal to the
  checkpoint, desirable or not on their own merits, and requiring their
  own validation cycle.
- **Existing surfaces:** Web `_history` (8-window) and the CLI's
  context-free turns remain exactly as they are under both lanes; the
  checkpoint's injection window (while `_history` is young / first turn)
  composes with, not replaces, the existing window.

---

## 14. OPEN QUESTIONS — what is proven, what is inferred, what is not decidable here

**Proven by code reading (this audit):**
- SessionTracker's full semantics, tables, and trigger conditions (§1);
  it is a trigger ledger, not a session-content store.
- The exact contents and LLM cost structure of the boundary batch and
  `Flush()`; the legs' non-cancellability (only extraction legs are
  cancel-aware).
- The precise dormancy split: turn counting + touched-refs accumulation
  are LIVE; session identity, boundary batch, `Flush()`, and their
  subsystems are DORMANT; `rb_slot_long`/`subgraph_pool` never drain.
- No production `Flush()` caller; the vendor's designed caller is the
  integrator-facing `memory_api.Memory.flush()`, which our app does not
  use. `is_third_turn` is dead code in the orchestrator.
- The app call graphs (web + CLI), the exact points where session_id is
  lost, where Flush could hook, and what dies at close (gate queue,
  `_history`, pending CLI commits).
- KV lane schema/behaviour, existing keys (exhaustive), global-key
  scope, corruption/atomicity semantics.
- No idle timeout, no reset protocol, no space-directory isolation
  (user_id isolation only) in the current deployment.

**Inference (flagged):** upstream intent of the session machinery (no
git history available — §2); the vendor's own move away from per-turn
attribution as evidence about batch-cost philosophy.

**Not decidable from code / needs runtime measurement:**
- Production wall-time and LLM-call count of one `Flush()` on a copy of
  the real memory space (boundary-batch duration).
- The dormant backlog size (`touched_refs` per namespace,
  `graph_query_activations` rows) on the production DB.
- User-visible first-turn TTFT impact of a boundary burst on the RTX 5070
  machine (the mechanism is proven; the production magnitude is not).
- Digest producer latency/quality delta (LLM vs deterministic) on the
  production model.
- Server-shutdown completion of the WS finally path.

**Operator decisions (product, not code):** injection window
(first-turn vs first-N vs whole session); TTL default; CLI parity in the
first cut; last-N ring vs single slot; UI surfacing ("welcome back"
card); space-switch read policy (checkpoint of the newly active space
applies to the next turn — yes/no).

**Standing questions from the previous study, resolved here:**
- *F2 / current_session anchor activation* — examined; its runtime cost
  is the non-cancellable boundary-batch class, and its value is
  memory-consolidation, not continuity (§4, §5).
- *Digest producer / execution shape* — analysed (§7): write-then-upgrade,
  bounded, cancellable, deterministic floor; cannot ride the gate at close.
- *CLI parity / last-N / UI surfacing / TTL* — remain operator decisions.

---

## 15. FINAL TECHNICAL RECOMMENDATION

*(Characteristics, not scores — per the operator's standing rule.)*

### A) "What do we know for sure?"

1. The vendor SessionTracker is a **batch-trigger ledger**: it can detect
   boundaries and count turns, but it stores no session content — it is
   not, and cannot become, a "where we left off" store without the same
   kind of app-side writer that the KV lane needs anyway.
2. Activating `session_id` + `Flush()` does not just create a session-end
   hook: it **simultaneously activates** the subgraph dynamic-slot
   subsystem (with memory re-tagging), the boundary-gated schema-refresh
   writer, and long-term attribution — a bundle of non-cancellable,
   single-slot LLM work in the measured worst foreground-blocking class,
   fed on day one by an unmeasured, never-drained backlog
   (`rb_slot_long`, `subgraph_pool`).
3. The checkpoint capability needs **none** of that machinery: one O(1)
   read at session start, one sub-ms write at session boundary, zero
   embeddings, zero required LLM.
4. The KV lane is an existing, shipped, already-used vendor API with
   failure semantics that match the required "never break the flow"
   principle (read → default on any error; write → non-fatal).

### B) "What is the main technical risk?"

**Activating the dormant vendor lifecycle as a means to the session-
continuity end** — i.e. accepting, as a side effect of a small feature,
a semantic-memory behaviour change (slot summaries/dynamic slots/persona
rewrites altering future retrieval content), a non-cancellable LLM burst
at exactly the wrong moment (the new session's first ingest or an
ungated post-close moment), and a first-activation backlog burst of
unknown magnitude on the single llama-server slot. The measured mechanism
(299.7 s sandbox TTFT behind ONE non-streaming request; 40-60 s production
window) is the proven shape of exactly this leg class.

### C) "Which direction fits the current production architecture better, and why?"

**The application-level bounded SESSION_CHECKPOINT in the existing KV
lane (Lane B).** It is the direction consistent with every hard constraint
the current architecture has accumulated: zero vendor divergence (the
fork's own established pattern of using the kv lane for small persistent
state), zero new LLM/embedding load, O(1) reads off the critical path,
complete structural separation from semantic memory (it cannot become a
second memory system because it never enters any store, ranking, quota,
or cleanup), and graceful degradation to today's behaviour by
construction. Lane A's genuine value — the memory-consolidation features
it would un-dorm — is real but **is a different feature**, with its own
runtime evidence burden, and can be decided later without affecting this
design: the two lanes are orthogonal, not competing implementations of
the same capability.

### D) "What must still be proven before implementation?"

(Per audit-only discipline — validation steps, not implementation.)
1. **Backlog sizing:** read the production space sqlite —
   `touched_refs` counts per namespace, `graph_query_activations` row
   count — to size what a first activated boundary batch would consume
   (this number also matters for any future Lane-A decision).
2. **Boundary-batch dry run:** on a **copy** of the production memory
   space, on the production-like machine, run one controlled `Flush()`
   offline and measure wall time + LLM call count + result quality
   (dynamic slots created, summaries written).
3. **Digest micro-measurement:** one bounded digest call on the real
   model (latency + quality vs the deterministic floor).
4. **kv micro-measurement:** `kv_get`/`kv_set` round-trip on the real
   sqlite (expected sub-ms; confirm under concurrent ingest).
5. **Operator decisions:** injection window, TTL, CLI parity, UI
   surfacing, space-switch read policy (§14).

### E) "What is the next smallest audit/validation step?"

**The production-DB backlog inspection (D.1) — a read-only SQLite query
session against a copy of the space database** (`SELECT namespace,
COUNT(*) FROM touched_refs GROUP BY namespace; SELECT COUNT(*) FROM
graph_query_activations;` plus the session_state row for the user).
It is zero-risk, zero-code, and it is the single fact that most reduces
uncertainty in *both* directions: it bounds the first-activation burst
(Lane A's main risk) and completes the evidence base for the checkpoint
writer's session-boundary choice (Lane B's only timing-sensitive
design point). The second step, in the same session, is the offline
`Flush()` dry run on that same copy (D.2), which converts Lane A from
"unmeasured" to "measured" for any future decision about un-dorming the
vendor batch machinery — independent of, and after, the checkpoint
direction already being clear.

---

*Evidence base: direct reading of `vendor/voicemem/voicemem/`
(session_tracker.py, orchestrator.py, core.py, memory_api.py,
utils/common/space.py, utils/common/llm_bg_gate.py, utils/common/turn_ids.py,
utils/common/voice_input.py, leftbrain/brain.py,
leftbrain/memory_repository.py, leftbrain/memory_repository_v2.py,
leftbrain/mem0_backend_store.py, leftbrain/slot_split/subgraph_manager.py,
leftbrain/slot_split/graph_entity_store.py, rightbrain/brain.py,
rightbrain/attribution_manager.py, utils/audio/perceiver.py),
`app/` (web_server.py, pipeline.py, voicemem_bridge.py,
background_memory.py, main.py, config.py, teacher_persona.py),
`app/background_memory.py`, tests/integration/{test_memory_safety,
test_temporal_memory}.py, `VOICEMEM_PIN.json`, `UPSTREAM_POLICY.md`,
`audit/VoiceMEM_llmpriority_v0102/REPORT.md`, and
`docs/VoiceMEM_Session_Continuity_Architecture_Study.md`.
No file outside this report was modified; nothing committed.*
