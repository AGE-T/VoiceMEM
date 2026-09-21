# VoiceMEM — SESSION PIPELINE BACKLOG FORENSIC AUDIT

**Date:** 2026-09-20
**Type:** READ-ONLY production-like DB forensic. Zero production-code change,
zero commit, zero DB mutation, zero `Flush()`, zero LLM calls, zero
embedding runs, zero `session_id` passed, zero SessionTracker activation.
**Companion document:** `audit/VoiceMEM_dormant_session_pipeline_audit/REPORT.md`
(2026-09-19) — the code-level dormant-session audit whose §14/§15 named the
"production-DB backlog inspection" as the next validation step. This document
is that step, executed to the extent the current environment permits.

**Headline finding (stated up front, evidence below):** the VoiceMEM
production SQLite database **does not exist in this environment**. This
sandbox runs the web backend in `--mock` DEMO mode (JSON-backed demo memory
store, vendor facade never constructed), the CLI does not run here, and no
`voicemem_memoryspace/` or `memory/*.sqlite` file exists anywhere on this
machine. The production DB lives on the operator's Windows machine
(`memory/memory.sqlite` under the agent repo, by `app/config.py:307-314`).
Consequently **no dormant-backlog row counts can be measured here**; what this
forensic delivers instead is (i) the verified identification of every
relevant DB/table/namespace and its exact growth mechanics, (ii) the proof of
which namespaces are structurally unbounded (never drained), (iii) a
ready-to-run **read-only SQL pack** for the operator to execute on a **copy**
of the production DB, and (iv) the scratch-DB KV micro-measurements
explicitly permitted by the task's §13.

All file references are to `voicemem-agent/` unless prefixed. Vendor file
references are to the controlled fork at pin `e8384e0` (`VOICEMEM_PIN.json`).

---

## 1. DATABASES AND SPACES — identification, no assumed paths

### 1.1 Where the production DB is (by config, not assumption)

- `AgentConfig.memory_root` (app/config.py:207) resolves via
  `memory_root_path` (app/config.py:307-314): `VOICEMEM_MEMORY_ROOT` env →
  explicit `memory_root` field → **`<repo>/memory`** default.
- The vendor space layer derives the single sqlite per root as
  `<root>/<dirname>.sqlite` (`_pick`, vendor
  `voicemem/utils/common/space.py:87-100`; `db()` :103-105) — i.e. **one
  file: `memory/memory.sqlite`** for the whole agent (web + CLI), regardless
  of how many web "spaces" exist, because spaces are **user_id scoping, not
  directory scoping**: every web space is a facade with
  `user_id = webspace_<name>` on the shared root (app/web_server.py:552-557,
  audit-1 §3.3), and the CLI bridge uses `voice_user` by default
  (app/config.py:208) with per-speaker facades (`{user_id: vm}` dict,
  app/voicemem_bridge.py:57, :382-410).
- The same file carries **all** structured stores (space.py:225-244 describe()
  inventory): `memories`, `entities`, `entity_edges`, `memory_tags`,
  `slot_profiles`, `graph_entities`, right-brain tables
  (`rb_slots`, `rb_entities`, …), the `kv` table, `session_state`,
  `touched_refs`, `graph_query_activations`, slot-split tables.

### 1.2 What actually exists in THIS environment (exhaustive search)

| Candidate | Finding | Evidence |
|---|---|---|
| `voicemem-agent/memory/*.sqlite` | **absent** — `memory/` contains only `sqlite/.gitkeep`, `backups/.gitkeep`, `qdrant/.gitkeep` | directory listing; `find` across `/home/z`, `/tmp` (no `*.sqlite`/`*.db`/WAL/SHM outside the two below) |
| any `voicemem_memoryspace/` directory | **absent** (the only one on the machine is the upstream clone's own `.gitkeep` placeholders: `tool-results/voicemem-src/voicemem_memoryspace/{demo,myvoicemem}/.gitkeep`) | filesystem search; `git ls-files` in the clone |
| git history: was a DB ever committed? | **never** — `git log --all --diff-filter=A -- 'voicemem-agent/memory/**.sqlite' '**.db'` returns nothing | git |
| running web backend | `python3 -m app.web_server --mock --host 127.0.0.1 --port 8787` (pid 1392) — **mock mode**: `DemoMemoryLayer`, vendor facade **never constructed** (app/web_server.py:1257-1271 early-`return`s after installing demo components; `RealMemoryLayer` is only reachable on the non-mock path, :1278) | `ps aux`; supervisor `mini-services/voicemem-web/index.js:34` |
| demo memory store | `data/web_demo_memory.json` (8 787 B) — pure-JSON `spaces.demo.left/right` list, no SQLite, no session machinery | file read |
| CLI | not running in this sandbox | `ps aux` |
| `/home/z/my-project/db/custom.db` | **not VoiceMEM** — the Next.js showcase scaffold's Prisma DB (`User`/`Post` models, 0 rows) | `prisma/schema.prisma`; read-only sqlite query |

**Conclusion of §1:** the "production-like DB" for this forensic is, in this
environment, **nonexistent by design** (DEMO mode exists precisely because
the sandbox has no GPU/models/llama-server — supervisor header comment). There
is no production/test/demo DB mixing to untangle here: there is no SQLite DB
at all. The production DB is on the operator's machine at the path derived
in §1.1; every measurement below that needs it is therefore delivered as a
ready-to-run pack instead of a number.

---

## 2. SESSIONTRACKER STATE (structure + observed)

Schema (vendor `voicemem/utils/common/session_tracker.py:39-56`, verified in
full this session):

```sql
session_state (user_id TEXT PRIMARY KEY, last_session_id TEXT,
               turn_count INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL)
touched_refs  (user_id, namespace, ref, created_at,
               PRIMARY KEY (user_id, namespace, ref))
```

- `record_turn(user_id, session_id)` runs **once per `Ingest()`**
  (orchestrator.py:1342) — today always with `session_id=None`, so in the
  production DB: exactly **one `session_state` row per user_id**
  (`webspace_<name>` per web space, `voice_user`/per-speaker for CLI), with
  `last_session_id = NULL` forever and a monotonically increasing
  `turn_count` (session_tracker.py:58-91; `session_changed` requires both
  ids non-null and different, :81).
- `get_current_session()` therefore returns `None` for every user
  (session_tracker.py:93-101) → every `graph_query_activations` row is
  written with `session_id NULL` (see §5).
- **Observed state:** not observable here — no DB (§1.2). Structural
  expectation for the operator's production DB: `SELECT * FROM
  session_state` returns one row per space/speaker, all with
  `last_session_id IS NULL`.

---

## 3. TOUCHED_REFS FORENSIC (the highlighted namespace ledger)

### 3.1 The four namespaces — writers, drainers, and their liveness today

| Namespace | Written by (verified) | Drained by (verified) | Liveness in current deployment |
|---|---|---|---|
| `rb_pending_memories` | every ingest: one touch per new left-brain memory id + heartnote id (orchestrator.py:1353-1358) | the `≥ SHORT_TERM_MIN_MEMORIES` trigger (**default 20**, env `VOICEMEM_ATTRIBUTION_MIN_MEMORIES`, orchestrator.py:77) pops it (orchestrator.py:1359-1361) | **LIVE and bounded** — oscillates 0→20 |
| `rb_entity_short` | every right-brain trait-entity write (rightbrain/brain.py:717) | popped by the same 20-memories trigger (orchestrator.py:1362) **and** by `Flush()` (orchestrator.py:1453) — `Flush()` never called in production | **quasi-bounded** — drained on each 20-memory window; residual between windows |
| `rb_slot_long` | every right-brain trait-slot write (rightbrain/brain.py:718) | **only** `_run_session_boundary_batch()` (orchestrator.py:1436) — which only fires on `session_changed` / `Flush()` | **NEVER drained — unbounded** |
| `subgraph_pool` | **every real retrieval**: one touch per distinct hit memory id (leftbrain/brain.py:677-689, `_record_subgraph_activation`, called from `Search()` — orchestrator.py:853) | **only** `RunSubgraphCheckpoint` (leftbrain/brain.py:704-711 pops the whole pool) — only reachable via the boundary batch / `Flush()` | **NEVER drained — unbounded** |

`touch` is `INSERT OR IGNORE` on the (user_id, namespace, ref) primary key
(session_tracker.py:105-111) — so the row count per namespace is the count of
**distinct refs**, not of events. The only `DELETE` on `touched_refs` anywhere
in the vendor+app tree is `pop_touched` (session_tracker.py:122-133; grep
across `vendor/` + `app/` — exactly one hit).

### 3.2 Growth pattern (reconstructed from code; timestamps allow runtime
reconstruction on the real DB)

- `subgraph_pool`: +distinct retrieved memory ids per turn with retrieval.
  Ceiling = number of **distinct memories ever retrieved** in that space.
  Monotonic; `created_at` per row allows the operator to reconstruct
  accumulation over time (`SELECT date(created_at), COUNT(*) … GROUP BY 1`).
- `rb_slot_long`: +distinct slot ids written. Ceiling = number of distinct
  rb slots (~13 static + dynamic, audit-1 §1.3) — **small cardinality, but
  each ref costs 1 long-term-attribution LLM call when finally drained**
  (orchestrator.py:1436-1438).
- `rb_entity_short`: grows between 20-memory windows; residual at any moment
  ≈ entities touched since the last window.
- Per-space distribution: rows are keyed by `user_id` — web spaces and CLI
  speakers never share rows.

### 3.3 The "soha nem ürített állapot" question

**Structurally proven: yes for `rb_slot_long` and `subgraph_pool`.** Both
namespaces have exactly one drain path each, and that path is reachable only
through `session_changed` (never fires — no `session_id` is ever passed) or
`Flush()` (no production caller; the only shipped caller-facing wrapper is
`memory_api.Memory.flush()` at memory_api.py:122-125, which the app does not
use — audit-1 §2, re-verified in the upstream clone this session). Whether
the *production rows* have actually accumulated is exactly what the SQL pack
(§7) measures on the operator's copy.

---

## 4. SUBGRAPH_POOL — what a hypothetical first Flush would consume

`RunSubgraphCheckpoint` (leftbrain/brain.py:704-724) pops the **entire**
accumulated pool in one call (`pop_touched` takes everything,
session_tracker.py:122-133), then: builds the entity co-occurrence graph over
those memory ids, peels greedy densest subsets scoring ρ(H), and for every
candidate **above `MIN_SYNERGY_THRESHOLD = 0.05`** (subgraph_manager.py:39;
below-threshold candidates are skipped **without LLM**, :19, :226) runs
**three LLM verification calls** (relevance :342 / importance :377 /
completeness :400). On approval it creates a dynamic slot, migrates entity
`slot_ref`s, and re-tags memories (audit-1 §1.3).

- **Backlog shape (static derivation):** one first-activation run processes
  `N = |subgraph_pool|` distinct memory ids, where N is bounded by the
  number of distinct retrieved memories in the space. The LLM-call count is
  `3 × (candidates above ρ 0.05)` — **not derivable from N without running
  the graph step** (it depends on entity overlap structure).
- **Duplicate/orphan rows:** impossible by construction (PK dedup); there is
  no second writer to diverge.
- **Wall-time: not estimable from row counts** (per the task's rule). It is
  the sum of graph build + `3 × LLM-candidate` sequential, non-cancellable,
  15 s-timeout calls (audit-1 §4.B/J) — measurable only in the offline dry
  run on a copy (audit-1 §15.D.2).

---

## 5. GRAPH_QUERY_ACTIVATIONS

DDL (graph_entity_store.py:87-95, plus the `session_id` ALTER and
`idx_gqa_user_session` index at :107-111 — the in-code comment documents a
real historical migration failure on an old demo DB, evidence the column was
added after the table first shipped):

```sql
graph_query_activations (id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
  query_id TEXT NOT NULL, entity_id TEXT NOT NULL, session_id TEXT,
  created_at TEXT NOT NULL)   -- + idx (user_id, query_id), idx (user_id, session_id)
```

- **Writer:** every real retrieval records one row **per activated
  graph-entity** of the hit memories (leftbrain/brain.py:690-702 →
  `record_query_activation`, graph_entity_store.py:119-137; a fresh
  `query_id = uuid4().hex` per retrieval). `session_id` is threaded from
  `get_current_session()` → **always NULL today**.
- **Reader:** `compute_rho` (graph_entity_store.py:139-174) — with
  `session_id=None` it falls back to **the user's full history** as Q.
- **Growth:** unbounded, proportional to (retrievals × activated entities);
  **nothing ever deletes from it** (grep: no `DELETE FROM
  graph_query_activations` anywhere in the tree) — it is a permanent
  activation log, growing even in the current dormant-session deployment.
- **Session linkage today:** every row has `session_id IS NULL` — the
  column is present, indexed, and never populated. On the production DB the
  pack's `GROUP BY session_id` should return a single NULL group; anything
  else would contradict the dormancy analysis.

---

## 6. RB_SLOT_LONG

Rows = distinct rb slot ids touched by right-brain writes
(rightbrain/brain.py:718), one row per (user_id, slot). Cardinality is small
(static slot set ≈ 13 + dynamic slots, audit-1 §1.3), but this namespace is
the **long-term-attribution backlog**: at the first boundary each popped slot
costs one `_summarize_slot` LLM call rewriting the slot's persona
description (orchestrator.py:1436-1438; attribution_manager.py:100-139 per
audit-1). Accumulation is monotonic (never drained, §3.1); the cost is
bounded by slot cardinality, not by row count growth.

---

## 7. BACKLOG SIZE ESTIMATION — honest answer + the read-only SQL pack

**Measured here:** nothing can be — the production DB is absent (§1.2).
**Estimates by row count → forbidden by the task; wall-time from row count →
not offered.** What can be stated statically (all derivations above):

- A hypothetical first `Flush()` would consume, in one synchronous run:
  the whole `subgraph_pool` (graph build + up to `3×candidates` LLM calls),
  the whole `rb_slot_long` (1 LLM call per slot), the residual
  `rb_entity_short` (1 + un-refined-items LLM calls per entity), and would
  also run `_refresh_schema_descriptions()` (1 LLM call per
  schema-eligible slot) (orchestrator.py:1413-1457).
- The namespaces that are **structurally unbounded** are `rb_slot_long`,
  `subgraph_pool`, `graph_query_activations` (§3-§6). `rb_pending_memories`
  is bounded by the live 20-trigger; `rb_entity_short` is
  window-quasi-bounded.

### 7.1 READ-ONLY SQL PACK (for the operator, to run on a **copy** of
`memory/memory.sqlite`; every statement is a SELECT or PRAGMA)

```sql
-- belt-and-braces on the copy (keep the original untouched):
PRAGMA query_only = ON;

-- (1) touched_refs backlog, per user (space) and namespace
SELECT user_id, namespace, COUNT(*) AS refs
FROM touched_refs GROUP BY user_id, namespace ORDER BY refs DESC;

-- (2) accumulation-over-time reconstruction (subgraph_pool focus)
SELECT user_id, date(created_at) AS day, COUNT(*) AS new_refs
FROM touched_refs WHERE namespace IN ('subgraph_pool','rb_slot_long')
GROUP BY user_id, date(created_at) ORDER BY day;

-- (3) session_state census (expect: last_session_id IS NULL everywhere)
SELECT user_id, last_session_id, turn_count, updated_at FROM session_state;

-- (4) graph_query_activations volume + range + session linkage
SELECT COUNT(*) AS rows, MIN(created_at), MAX(created_at)
FROM graph_query_activations;
SELECT user_id, COUNT(*) FROM graph_query_activations GROUP BY user_id;
SELECT session_id, COUNT(*) FROM graph_query_activations GROUP BY session_id;
   -- expected: single group with session_id NULL (dormancy invariant)

-- (5) KV census: key names + value sizes (leftbrain_store expected largest)
SELECT k, length(v) AS bytes FROM kv ORDER BY bytes DESC;

-- (6) KV corruption sweep (read-only; json_valid is SQLite built-in)
SELECT k FROM kv WHERE v IS NULL OR v = '' OR NOT json_valid(v);

-- (7) ceilings for the backlog namespaces
SELECT COUNT(*) AS distinct_retrieved_memories_ceiling FROM memories;
SELECT user_id, COUNT(*) AS slots FROM rb_slots GROUP BY user_id;

-- (8) oldest/newest touched rows (is it "soha nem ürített"?)
SELECT namespace, MIN(created_at) AS oldest, MAX(created_at) AS newest
FROM touched_refs GROUP BY namespace;
```

Statements (1), (4) and (5) are the audit-critical numbers; (2) and (8)
answer the "folyamatosan nő?" question from data; (6) feeds §10; (7) turns
the counts into interpretation.

---

## 8. HISTORY / LEFTBRAIN / KV — the KV lane census (code-level, exhaustive)

The `kv` table (DDL: `kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)`,
space.py:120-124) has exactly **five writers/readers in the entire
vendor+app tree** (grep for `kv_get|kv_set`, exhaustive):

| Key | Written/read by | Purpose | Size class |
|---|---|---|---|
| `leftbrain_store` | leftbrain/memory_repository.py:117, :119-138 | full left-brain JSON mirror, rewritten on ingests (`_KV_KEY`, :117) | **scales with memory count — the largest blob in the lane** |
| `rb_cleanup_last_count` | rightbrain/brain.py:896-903, :1043 | right-brain cleanup **throttle** (count-based) | tiny int |
| `audio_cleanup_last_run` | orchestrator.py:1474-1487 | audio-archive cleanup **throttle** (daily-once) | ISO timestamp |
| `archive_cold_last_run` | orchestrator.py:1506-1520 | cold-memory archive **throttle** (daily-once, VM-LOCAL-014) | ISO timestamp |

(These three are the "throttle keys" the task asked to look for — the word
"throttle" itself does not appear in the code; the daily-once / count-gated
kv pattern is the throttling mechanism, orchestrator.py:1335-1339 documents
the pattern for the archive leg.)

- **Observed KV state:** not observable here (no DB). Structural expectation
  on the production DB: the four keys above, with `leftbrain_store` by far
  the largest value.
- **Bounded SESSION_CHECKPOINT fit:** a representative checkpoint of the
  operator's target shape (schema 1, session_id, topic, state, decisions,
  open_items, last_user_goal, next_step, turn_count, started/ended_at,
  digest_method) serializes to **669 bytes** (measured, §13) — the same
  lane already carries `leftbrain_store`, which grows with every ingested
  memory. A constant ~0.7 KB, once-per-session-overwritten value is strictly
  inside demonstrated usage.

---

## 9. KV KEY SCOPE — global keys on the shared root

Verified from the schema and call sites (§8):

- The `kv` table has **no user/space column** — keys are **global per sqlite
  file** (space.py:120-124). Since all web spaces + CLI speakers share one
  root/one file (§1.1), a naive key like `session_checkpoint` would be
  shared by **every** space and speaker.
- The existing convention is indeed identity-free: all four current keys are
  single global values; e.g. `rb_cleanup_last_count` is one row shared by
  all `user_id`s — acceptable for housekeeping (cleanup gating is
  file-level), but **exactly the pattern a per-space checkpoint must NOT
  follow** (cross-space last-writer-wins).
- **No collision-protection mechanism exists** beyond key-string choice.
- **Conceptual (explicitly NOT final, NOT implemented) key proposal**, per
  the task: `session_checkpoint:<user_id>` where `<user_id>` is the **same
  identity the stores already use** — `webspace_<name>` (web) /
  `voice_user` or per-speaker id (CLI). This reuses the exact isolation
  boundary of `session_state`/`touched_refs` (both keyed by `user_id`,
  session_tracker.py:41-56) without inventing a new convention, and makes
  space deletion/copy-folder semantics identical to every other store.
  Versioning: a `"schema": 1` field inside the JSON; freshness: `ended_at` +
  age policy (data, not code). This matches audit-1 §6's suggestion and
  adds the identity-equality argument.

---

## 10. CORRUPTION / FAILURE FORENSIC

No production data exists to inspect (§1.2) — the structural findings, and
the checks the operator should run on the copy (they are statements (5)/(6)
of the §7 pack):

- **Malformed JSON / NULL / empty value:** `kv_get` swallows **every**
  exception and returns the default (space.py:127-134) — a corrupt value
  manifests as "key absent", never as an error on the conversation path.
  `json_valid(v)` (pack statement 6) detects it read-only.
- **Extrême large blob:** `leftbrain_store` is the expected maximum
  (pack statement 5 sizes it). No size limit exists in the lane (TEXT;
  space.py:118-145).
- **Torn/partial write:** not possible at the sqlite level — one-statement
  upsert inside one connection transaction (space.py:137-145).
- **Duplicated logical key:** impossible by primary key; the realistic
  duplication risk is **two identities hashing to the same key string**,
  which the `session_checkpoint:<user_id>` scheme avoids by construction
  (§9).
- **Suspicious data:** nothing observable here; on the real DB, statement (4)
  returning any non-NULL `session_id` group, or `session_state.last_session_id`
  non-NULL, would contradict the dormancy analysis and should be escalated.

---

## 11. SPACE SWITCH / CROSS-CONTAMINATION

- Semantic-memory isolation is per-`user_id` rows in the shared file
  (§1.1) — switching the active web space rebinds the facade; no
  cross-space rows exist to leak (audit-1 §9, re-verified via the
  `user_id` keying of every table inventoried in space.py:225-244).
- KV is the one lane **without** built-in isolation (§9): current keys are
  global-by-intent (housekeeping); a checkpoint key without identity would
  be readable/writable across spaces — the report's proposed
  `session_checkpoint:<user_id>` scope makes the checkpoint's isolation
  identical to the stores' isolation **by using the same identity string**.
- No existing cross-space collision exists today (only four global
  housekeeping keys, §8) — confirmed at code level; the pack's statement (5)
  verifies it on the real data.

---

## 12. PRODUCTION-RUNTIME RISK — the seven questions

1. **Is there an actual dormant backlog?** Not measurable in this
   environment (no production DB, §1.2). **Structurally: yes, on any
   long-lived production space** — `rb_slot_long` and `subgraph_pool` have
   no drain in the current deployment (§3.1), and `graph_query_activations`
   grows on every retrieval with no deletion path (§5). The magnitude is
   precisely what the §7 pack measures on the operator's copy.
2. **How big?** Unknown from here; ceilings are derivable
   (§3.2/§4/§6: `subgraph_pool` ≤ distinct retrieved memories;
   `rb_slot_long` ≤ slot count ≈ 13 + dynamic; `graph_query_activations` ∝
   retrievals × activated entities).
3. **Where concentrated?** In the single `memory/memory.sqlite`, in rows
   keyed by the busiest `user_id` (web spaces / CLI speakers) — pack
   statements (1)/(4) give the per-user distribution.
4. **Evidence of continuous growth?** Code-level monotonicity is proven
   (writers on every retrieval/ingest; the only `DELETE` is `pop_touched`;
   nothing deletes `graph_query_activations`). Runtime growth-rate evidence
   comes from pack statement (2)/(8) (`created_at` per row).
5. **What would a hypothetical Flush touch?** The enumerated legs of
   §4/§6/§7: the whole popped backlog in one synchronous, non-cancellable
   run — plus schema-refresh and short-attribution leftovers
   (orchestrator.py:1413-1457).
6. **Is activation dangerous first-run on this data?** The **mechanism**
   risk stands as audit-1 §4.B/J stated it (non-cancellable 15 s-timeout LLM
   legs in the measured worst foreground-blocking class; first activation
   processes the entire backlog at once). The **data** needed to size it is
   exactly what this environment cannot provide — so the honest verdict is:
   the risk is proven as a class, unproven as a magnitude, and the §7 pack
   on a copy is the mandatory pre-decision step. One **new** nuance from
   this forensic softens one sub-risk: `rb_entity_short` is drained by the
   live 20-memories window (§3.1), so the truly-unknown backlog is carried
   by `rb_slot_long` (bounded cardinality) and `subgraph_pool`
   (memory-count-bounded) — the activation burst is bounded by space
   size, not infinite; but per-leg cost is LLM-bound, and slot-cardinality
   × 15 s worst case is already minutes on a single-slot server.
7. **Does this change audit-1's recommendation?** **No.** Audit-1
   recommended the app-level bounded KV checkpoint (Lane B) and named the
   backlog inspection as the next step; this forensic converts that step
   into a ready-to-run pack, confirms the structural growth analysis
   first-hand at the DB level, and found nothing that contradicts the
   Lane-B direction. Lane A (activation) remains "measure first, decide
   later, independently of the checkpoint feature".

---

## 13. KV MICRO-MEASUREMENT (scratch DB only — production DB nonexistent/untouched)

Environment: sandbox, Python 3.12.14, `/tmp/vm_kbench/scratch_root`
(freshly created scratch root; vendor `space.py` loaded standalone — it is
dependency-free, so no vendor package initialisation ran; **no LLM, no
embedding, no SessionTracker was instantiated** — the benchmark exercises
only `kv_get`/`kv_set`/`json`, which are pure SQLite+JSON operations).
These numbers characterise the **cost shape** on this machine; they are not
production-hardware representative and are not offered as such.

Representative bounded checkpoint (schema 1, all target fields populated):
**669 bytes** serialized (static write-size evidence).

| Operation (n=300, per-call fresh sqlite connection included) | min | p50 | p95 | max |
|---|---|---|---|---|
| `kv_get` — key present | 0.058 ms | 0.068 ms | 0.131 ms | 1.07 ms |
| `kv_get` — key missing (default path) | 0.047 ms | 0.056 ms | 0.082 ms | 1.54 ms |
| `kv_set` — overwrite (scratch DB) | 0.061 ms | 0.070 ms | 0.104 ms | 1.09 ms |
| `json.loads` of the 669 B checkpoint | 0.0033 ms | 0.0033 ms | 0.0035 ms | 0.019 ms |

Round-trip `kv_set → kv_get` object-equality verified. **No write benchmark
was run on any production DB** (none exists); the write column above is the
scratch-DB measurement the task's §13 explicitly authorises ("külön scratch
DB"). Conclusion: the checkpoint lane's read cost is sub-millisecond in
shape, ~4 orders of magnitude below the ~100 ms-class semantic search
(audit-1 §1.4 / memperf evidence), and its write is a single atomic upsert.

---

## 14. FINAL VERDICT

### A) PROVEN FACTS
- The production VoiceMEM SQLite DB **does not exist in this environment**
  (§1.2) — the sandbox web backend is `--mock` DEMO mode with a JSON demo
  store; the CLI is absent; no sqlite file exists anywhere; git history
  never contained one.
- The expected production DB location is `memory/memory.sqlite` (single
  file, user_id-scoped spaces, §1.1).
- `session_state`: one row per user_id, `last_session_id` NULL forever in
  the current deployment (§2).
- `touched_refs` namespace liveness: `rb_pending_memories` bounded by the
  live 20-trigger; `rb_entity_short` window-quasi-bounded; **`rb_slot_long`
  and `subgraph_pool` never drained**; the only `DELETE` in the tree is
  `pop_touched` (§3).
- `graph_query_activations`: one row per (retrieval × activated entity),
  `session_id` always NULL, **no deletion path at all** (§5).
- A first `Flush()` consumes the entire popped backlog in one synchronous,
  non-cancellable run: subgraph leg (3 LLM/candidate above ρ 0.05),
  schema-refresh (1 LLM/slot), long-term attribution (1 LLM/slot),
  short-term leftovers (§4/§6/§7).
- KV lane: exactly four existing keys (`leftbrain_store` + three
  housekeeping throttle keys), global-per-file scope, no user column;
  `kv_get` failure → default; `kv_set` single atomic upsert (§8-§10).
- Upstream (clone at pin e8384e0, 26 commits, single-import provenance):
  no shipped caller — examples, web demo, evaluation — ever passes
  `session_id` or calls `Flush()`; the only wrapper is
  `memory_api.flush()` (§2 of audit-1 + this session's clone verification).
- Scratch KV micro-measurements: sub-ms read/write, 669 B checkpoint (§13).

### B) OBSERVED DB STATE
- Here: **none** — no DB file, no rows, nothing to observe (§1.2). The demo
  lane is `data/web_demo_memory.json` (8 787 B, Hungarian demo strings); the
  only other sqlite on the machine is the unrelated Next.js Prisma scaffold.

### C) BACKLOG RISK
- Proven as a **class** (monotonic writers, absent drains, single-shot
  consumer, non-cancellable LLM legs); **unmeasured as a magnitude** — the
  decisive data lives in the operator's `memory/memory.sqlite` and is
  one `sqlite3` session away via the §7 pack. Bounded by construction:
  `rb_slot_long` by slot cardinality (≈13 + dynamic), `subgraph_pool` by
  distinct retrieved memories, `graph_query_activations` by retrieval count.

### D) KV EVIDENCE
- The lane is real, shipped, already used for exactly this class of small
  persistent state (including throttles), failure-tolerant by construction,
  and sized comfortably: a 669 B checkpoint against a `leftbrain_store`
  blob that scales with the memory count (§8, §13). Key scope must embed
  identity (§9).

### E) REMAINING UNKNOWNS
- All row counts and growth curves on the production DB (pack §7).
- The Flush wall-time / LLM-call count on real data (offline dry run on a
  copy — audit-1 §15.D.2, still the follow-up after the pack).
- Production-hardware KV numbers (RTX 5070 machine's disk/CPU) — the §13
  numbers are shape, not production latency.

### F) IMPACT ON PREVIOUS ARCHITECTURE DECISION
- **None that reverses it.** Audit-1's Lane-B (app-level bounded
  SESSION_CHECKPOINT in the KV lane) recommendation is unchanged and
  marginally strengthened (KV cost shape measured; identity-scoping
  argument sharpened; `rb_entity_short` found to be window-drained, which
  narrows the true unknown backlog to `rb_slot_long` + `subgraph_pool`).
  Lane-A activation remains gated on the pack + dry run, unchanged.

### G) NEXT SAFEST STEP
- **Operator-side, zero-risk, zero-code:** copy `memory/memory.sqlite` from
  the production machine, open the copy read-only
  (`sqlite3 "file:copy.sqlite?mode=ro"`), and run the §7 pack. Those
  numbers close every "unknown" in (E) except the dry run, and are the
  single input both the checkpoint design (Lane B) and any future Lane-A
  decision still lack. The subsequent step (only if Lane A is ever
  seriously considered) remains the offline `Flush()` dry run on that same
  copy — audit-1 §15.D.2.

---

## 15. COMPLIANCE PROOF (audit-only discipline)

- **Production code changes: 0** — no file under `vendor/`, `app/`,
  `scripts/`, `mini-services/`, or any runtime path was modified (git status
  shows only this report, the audit-1 report + its §2 addendum, and the
  worklog as work products of this audit session).
- **Commits: 0** — nothing staged or committed.
- **DB mutations: 0** — the production DB was never touched (it does not
  exist here); the only writes performed were to the task-authorised
  scratch DB at `/tmp/vm_kbench/scratch_root/`.
- **`Flush()` runs: 0. LLM calls: 0. Embedding runs: 0. SessionTracker
  activations: 0** — `session_tracker.py` was read, never instantiated;
  no `session_id` was passed anywhere.
- Read-only discipline on external data: the upstream clone and every
  sqlite touched in this session were opened via `file:…?mode=ro` or were
  read-only file operations.

---

*Evidence base (all verified first-hand this session unless attributed):
vendor `voicemem/utils/common/session_tracker.py` (full read),
`utils/common/space.py` (full read), `leftbrain/slot_split/
graph_entity_store.py:60-179`, `leftbrain/brain.py:672-716`,
`leftbrain/memory_repository.py:108-142`, `rightbrain/brain.py:710-722,
896-903, 1043`, `orchestrator.py:1332-1461, 1474-1520`, `leftbrain/
slot_split/subgraph_manager.py:19, 39, 226, 342-400`; app `config.py:207-208,
307-314`, `web_server.py:552-560, 1245-1289, 1257-1271`; upstream clone
`tool-results/voicemem-src` (git remote xzf-thu/VoiceMem, HEAD e8384e0,
26 commits; `examples/01_memory.py`, `examples/03_simple_agent_with_
voicemem_memory.py`, `web/run.py` greps; `memory_api.py:122-125`);
supervisor `mini-services/voicemem-web/index.js`; filesystem searches;
scratch benchmark `/tmp/vm_kbench/bench.py` (preserved as
`audit/VoiceMEM_session_backlog_forensic/tools/kv_micro_bench.py`);
`audit/VoiceMEM_dormant_session_pipeline_audit/REPORT.md` (companion,
code-level findings re-verified where cited).*

---

## ADDENDUM — RE-VERIFICATION AGAINST THE CURRENT TREE (2026-09-20)

**Context:** the consolidated 0.10.5 audit (task §§15–16, 18) required this
report's evidence to be re-verified against the live tree and the production
environment question to be re-answered. Executed read-only; no DB mutation,
no LLM, no embedding.

- **Tree drift since this report:** `git diff 51ad195..HEAD --stat --
  voicemem-agent/app voicemem-agent/vendor` is **empty** — the report
  describes the exact live tree. `[PROVEN]`
- **KV micro-bench reproducibility:** re-run on a fresh scratch DB
  (`/tmp/vm_kbench_11a/`, same script class): 669 B representative
  checkpoint; `kv_get` hit p50 **0.065 ms** (recorded: 0.068 ms — noise);
  round-trip object equality OK. The recorded numbers stand. `[PROVEN]`
- **Production DB availability re-checked:** still absent in this
  environment (`memory/` contains only `.gitkeep` placeholders; the sandbox
  web backend runs `--mock`; exhaustive filesystem sweep found no
  VoiceMEM SQLite anywhere). All backlog magnitudes therefore remain
  `[UNKNOWN — production DB not available]`, and **§7's read-only SQL pack
  on a copy of the operator's `memory/memory.sqlite` remains the single
  closing step** for every unmeasured quantity in this report.
- **Upstream pressure check:** no commit since the vendor pin touches
  `session_tracker.py`, `subgraph_manager.py` or the schema-refresh path —
  upstream main is static (last code commit 2026-09-05), so nothing upstream
  can have changed the backlog mechanics since this report was written.
  `[PROVEN — file-level + -S sweeps]`
- **Structural findings re-confirmed with line evidence:** `rb_slot_long`
  (rightbrain/brain.py:718) drains only via orchestrator.py:1436 (the
  session-boundary path, never taken); `subgraph_pool`
  (leftbrain/brain.py:679-683) only via `RunSubgraphCheckpoint`
  (brain.py:704-706); `graph_query_activations` (DDL
  slot_split/graph_entity_store.py:87-95, writer :119-137, `session_id`
  always NULL via `get_current_session`) has no DELETE path; the only DELETE
  in the vendor tree remains `pop_touched` (session_tracker.py:130); KV key
  census remains exactly 4 keys on a `k TEXT PRIMARY KEY` table with no user
  column (space.py:120-124). `[PROVEN]`
- **Conclusion unchanged:** dormant backlog risk is **proven as a class,
  unmeasured as a magnitude**; the KV lane is **structurally fit** for a
  bounded SESSION_CHECKPOINT (identity embedded in the key). The consolidated
  verdicts live in
  `audit/VoiceMEM_session_continuity_consolidated/REPORT.md`.

*Addendum evidence: the verification table (40 claims) recorded in
`worklog.md` (Task ID 11-a), including the scratch bench re-run.*
