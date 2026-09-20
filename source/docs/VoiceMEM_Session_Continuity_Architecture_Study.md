# VoiceMEM — Session Continuity / Session State Architecture Study

**Date:** 2026-09-19
**Type:** Architecture study (AUDIT ONLY)
**Scope:** How a separate, highly abstracted SESSION_STATE / SESSION_CHECKPOINT
("where did we leave off") capability could be realised in the current
VoiceMEM architecture.
**Status of this document:** analysis and provisional direction only.
No implementation, no refactoring, no schema change, no new DB, no new vector
store, no queue, no Learning Engine design, no HA-specific logic — per the
operator's explicit constraints.

---

## 1. Current Architecture

### 1.1 The two conversational surfaces

| | Web (primary) | CLI (`app/main.py` + `app/pipeline.py`) |
|---|---|---|
| Session unit | one `WebSession` per WebSocket connection (`web_server.py:4490-4507`) | one process run |
| Conversation history | `WebSession._history: list[dict]` (`web_server.py:1869`), in-memory, last 8 entries (`_HISTORY_WINDOW`, `:158`) | **none** — every CLI turn is `system + user` only (`pipeline.py:353`; the `history=` parameter of `build_messages` is never passed on this path) |
| Session identity | **no session id exists anywhere in `app/` or `web/`** | none |
| End-of-session handling | `_cancel_session_tasks` (`web_server.py:2668-2692`) cancels turn/audio/gate tasks; **nothing is persisted** | Ctrl-C → cancel tasks, close LLM; **nothing is persisted; queued failed memory commits die with the process** |
| Browser-side state | chat transcript lives in a JS array only; `localStorage` holds the mic device id and raw-capture flag, nothing conversational (`voicemem.html:1471-1485, 2348, 2422`) | n/a |

### 1.2 The turn pipeline (web)

```
speech → VAD → ASR → user_text
  → memory search  = vm.search(user_text, emotion)   [vendor Search; 0 LLM; E5 embed + qdrant + SQL rank]
  → memory context = top-5 facts + top-3 right-brain notes, provenance-suffixed, capped at 1200 chars
  → system prompt  = persona + "Long-term memory about this user:" + memory block + emotion block
  → history       = self._history[-8:]  (dropped oldest-first if over the 14 000-char prompt budget)
  → LLM streaming (single llama-server slot, Qwen3.6 35B A3B)
  → TTS chunked synthesis → browser
  → on reply end: _history.append(user, assistant); BackgroundMemoryGate.submit(user_text, reply)
```

Key constants (hardcoded, not config): `_HISTORY_WINDOW = 8`,
`_MEMORY_CONTEXT_MAX_CHARS = 1200`, `_PROMPT_CHAR_BUDGET = 14000`
(`web_server.py:158, 185-186`). The guard was designed for an 8 192-token
server; the current `config/llm_config.yaml` declares `context_size: 32768`
— the guard remains deliberately conservative.

### 1.3 The memory layers (vendor tree, pin e8384e0 + 22 VM-LOCAL patches)

* **Left brain — facts.** mem0/Qdrant vector rows + space sqlite mirror;
  temporal semantics (stance, validity intervals, supersession chains),
  statuses derived at read time (`docs/MEMORY_SEMANTICS.md`). Search is
  0-LLM: `(-temporal_tier, not superseded_by, base + recency_boost)`.
* **Right brain.** `right_brain_memories` rows:
  * **heartnotes** — one row per ingest turn, `content` = the user's
    verbatim utterance, event-time `created_at`, emotion/entities/inner-OS
    in metadata. This is VoiceMEM's episodic memory.
  * **response_experience** — reply-experience rows (write path gated OFF
    by default since v0.10.3, `VOICEMEM_RB_WRITE_EXPERIENCE=1`).
  Retrieved by **anchor routing** (entity/emotion anchors + fallbacks),
  quota-capped per turn; **no embeddings for heartnotes**.
* **Traits** — `rb_traits`: durable person-judgments with E5 claim vectors,
  evidence chain, occurrence counts, asymptotic confidence, read-side decay
  (90 d grace / 180 d half-life), non-destructive supersession.
* **Session tracking** — `SessionTracker` (`utils/common/session_tracker.py`):
  sqlite tables `session_state (user_id, last_session_id, turn_count, updated_at)`
  and `touched_refs`. `record_turn()` returns `{turn_count, session_changed,
  is_third_turn}`; `session_changed` is true **only when old and new
  session_ids are both non-empty and differ** — the docstring itself notes:
  *"if the caller never passes session_id, this trigger mechanism never
  starts"*.
* **Session-boundary batch + Flush** — `Orchestrator._run_session_boundary_batch`
  (subgraph checkpoint + per-slot schema description refresh + right-brain
  long-term attribution) fires on `session_changed`; `VoiceMem.Flush()` is
  the idempotent "conversation formally ended" complement for the **last**
  session (which has no successor ingest to trigger it).
* **Generic kv store** — the space sqlite has a `kv (k TEXT PRIMARY KEY, v TEXT)`
  table with `kv_get`/`kv_set` helpers (`utils/common/space.py:118-145`),
  already used by the fork for throttle state (`audio_cleanup_last_run`,
  `archive_cold_last_run`, `rb_cleanup_last_count`). This is an existing,
  application-agnostic, O(1) persistence lane inside the memory space.
* **memory_history()** — `leftbrain/memory_repository.py:246-295`: the
  v0.10 PHASE 9 hook; reads the full supersession chain of ONE memory id.
  No runtime consumer (only an integration test).

### 1.4 Background memory processing

`BackgroundMemoryGate` (`app/background_memory.py`): per-session queue of
`(user_text, reply)` pairs; arms on the first VAD frame / turn start (also
cooperatively cancelling in-flight vendor LLM legs); opens only after
grace 2.0 s + idle 6.0 s of true quiet; the ingest chain runs synchronously
in a worker thread (`async_facts=False`). The vendor `Ingest` per turn does
extraction + conflict resolution + heartnote + trait writes; the remaining
non-streaming legs (attribution, inner-OS, slot tagging, schema refresh,
cleanup) are the documented foreground-blocking class when they hold the
single llama-server slot (memperf-0105, reproduced: user TTFT = whole leg,
worst 31.2 s).

### 1.5 Persistence layout (one memory space)

```
<memory_root>/<space>.sqlite      ALL structured storage (facts mirror, right brain,
                                  traits, session_tracker, kv, mem0 history/messages,
                                  slot profiles, graph tables …)
<memory_root>/vectors/            qdrant text-vector store (left brain)
<memory_root>/multi_modal/        audio derivatives
```

The web app constructs ONE vendor facade per memory space with
`user_id = webspace_<safe>` and a shared `memory_root`
(`web_server.py:613-631`); the CLI bridge routes by speaker/space
(`voicemem_bridge.py:500-508`). All surfaces ultimately share the same
space sqlite — hence the same `kv` table.

---

## 2. Existing Session/History Capabilities (inventory)

What exists today, and how far each goes toward "session continuity":

| Mechanism | Where | State in our deployment | What it actually provides |
|---|---|---|---|
| `WebSession._history` (8 entries) | app, per WS connection | **active** | in-window conversational coherence within ONE connection; lost at close |
| CLI history | — | **absent** | nothing (every turn is context-free) |
| Vendor `_exchanges` deque(maxlen=2) | `orchestrator.py:273` | active (in-process) | reply pairing for ingest disambiguation; not a continuity surface |
| `SessionTracker.session_state` | vendor sqlite | **dormant** (no session_id ever passed by app) | turn counting + boundary *detection* only — stores no content |
| `SessionTracker.touched_refs` | vendor sqlite | active (the 20-new-memories threshold part) | batch-attribution triggers |
| `Flush()` / session-boundary batch | vendor | **dormant** (never called) | subgraph checkpoint, per-slot schema refresh, long-term attribution — the designed "conversation formally ended" hook |
| `memory_history()` | vendor | present, unused at runtime | supersession-chain reads (consolidation hook, not session state) |
| Extraction prompt `## Summary` section | `mem0_additive_prompt_build.py:101` | present, **always empty** | a ready-made injection slot for a prior-conversations digest — the live caller never passes `summary=` |
| mem0 `history`/`messages` tables | space sqlite | written, **never read** | durable add/update/delete event log + full chat payloads — a latent transcript substrate |
| `slot_profiles.summary` / `related_summaries` | vendor | partially active (refresh is boundary-gated → dormant) | per-slot rolling syntheses (≤40 words) riding `SearchResult` |
| Cross-session semantic recall (`vm.search`) | vendor | **active** | THE only continuity mechanism that works today: whatever facts/traits/heartnotes are semantically close to the new session's first utterance |
| Browser chat persistence | web UI | **absent** | reload = transcript gone (facts re-fetched via `/api/memories`) |
| Heartnote `ttl='session'` class, `current_session` AnchorType, `turn_ids.py` | vendor | present/unused | reserved vocabulary that was never wired to a session concept |

**Verdict:** there is **no existing mechanism** that stores or restores a
compact "where did we left off" state. The durable layer (facts, traits,
heartnotes) crosses sessions, but its recall is *query-driven*: whether the
previous session's context resurfaces depends entirely on the semantic
distance between the new first utterance and stored rows — not on any
notion of "the previous session". The vendor's own session machinery
(`session_id` → `SessionTracker` → boundary batch → `Flush`) exists, is
generic, and is fully dormant in our deployment.

---

## 3. Problem Definition

**Wanted:** when a new session starts, VoiceMEM should briefly know:

* where we were (topic),
* what the current state was,
* which decisions were made,
* what remained open,
* the last important user goal,
* the natural next step.

**What SESSION_STATE is NOT** (per the operator's definition):

* not the full conversation history, not a transcript;
* not full episodic memory (heartnotes already are that);
* not the knowledge base (facts already are that);
* not the Learning Engine (separate workstream, explicitly out of scope);
* not an application-specific workflow store — it must remain a **universal
  conversation/session-continuity capability** (no Home Assistant logic, no
  Goal Catalog, no business rules).

**Hard requirements derived from the brief:**

1. Highly abstracted, compact state (≈ 1-2 sentences up to a small
   structured record; the brief suggests a 200-500-token ceiling).
2. New-session startup must add **no perceptible context/latency overhead**
   (O(1)-style lookup; retrieval/embedding only if justified).
3. Must not degrade the runtime memory performance (the measured read path:
   search P50 ≈ 100 ms at 118 facts / 177 ms at 4 000 facts).
4. Must not become a second general-purpose memory system.
5. Must be indistinguishable-from-today behaviour when absent (graceful
   degradation).

---

## 4. Candidate Architectures (overview)

| Option | One-line shape | Vendor change? | New LLM call? | Lookup cost | Divergence |
|---|---|---|---|---|---|
| A | session state as a new memory type/namespace inside the existing stores | yes (schema/ranking) | read: no; write: optional | retrieval-shaped | higher |
| B | session checkpoint as a separate record extending the session/history structures (`SessionTracker`) | yes (additive) | optional 1/session-end | O(1) SQL | low-moderate |
| C | session summaries as special elements of the existing episodic memory (heartnote/experience rows) | mostly no (semantics reuse) | optional | anchor-routed (indirect) | moderate (semantic overload) |
| D | session state in a separate persistence lane under the same VoiceMEM API — concretely the existing space `kv` table | **no** (app-side use of an existing vendor API) | optional 1/ session-end | O(1) kv read | **minimal** |
| E | minimal extension of the existing session/history structure (persist `_history` tail) | no | no | O(1) | minimal (but product-intent conflict) |
| F1 (additional) | = D made concrete: a single namespaced kv key holding the compact JSON checkpoint | no | optional | O(1) | minimal |
| F2 (additional, complementary) | wire the dormant vendor session machinery (`session_id` through ingest + `Flush()` at session end) — plumbing any option can use, not itself a checkpoint store | small app-side (+ optional vendor patch-free usage) | activates attribution legs | n/a | low, but runtime consequences |

Options are analysed below against the same fourteen questions: where it
lives, what it stores, how it is created, when updated, when read, how it
enters the new session's context, how it leaves the context, stale-state
handling, separation from normal memory, runtime cost, new LLM call?,
embedding?, offline behaviour, long-term behaviour.

---

## 5. Option A — Session state as a separate memory type / namespace in the current VoiceMEM storage

**Where it would live.** A new row class inside the existing stores — e.g.
a new `memory_class = 'session_state'` in `right_brain_memories`, or facts
in the left brain written with a marker (source/metadata namespace
`session_checkpoint`). The stores themselves (space sqlite + qdrant) are
reused; only a new "type" is introduced.

**What it would store.** The compact digest fields (topic / state /
decisions / open items / next step), one row per session.

**How it would be created / updated.** A session-end write through the
normal `Ingest`-adjacent paths or a direct store call, tagged with the
namespace; updates = new row per session (append-only) or in-place rewrite
of the latest row.

**When it would be read / how it enters the new session context.** This is
Option A's structural problem: everything in these stores is reached
through **retrieval** (vector search on the left; anchor-routed search on
the right). A session checkpoint would surface only when the new utterance
semantically matches it — the exact failure mode the capability is meant to
remove. Making it "always included at session start" requires either a
special-case branch in the render path (filter by marker, force-include)
or a dedicated query — i.e. the "memory type" degenerates into a direct
lookup with extra steps, while inheriting every store behaviour around it.

**How it would leave the context.** Natural (it stops matching / newer rows
outrank it) — but "leaving" would again be similarity-driven, not
lifecycle-driven.

**Stale-state handling.** Inherited partially (supersession, `updated_at`),
but the digest is not a factual claim; the fact machinery (stance,
validity intervals, occurrence counts) has no meaning for it.

**Separation from normal memory.** Weak. The checkpoint would compete for
top-k slots with real facts, participate in cleanup quotas (every +50
heartnotes triggers LLM cleanup), and — if stored as facts — risk being
mutated by the conflict-resolution/supersession machinery or by a later
extraction that misreads it.

**Runtime cost.** Read: one extra candidate in the search space (negligible
per row, but it occupies a top-k slot when it does surface). Write: one
store write per session end.

**New LLM call / embedding?** Digest production: optional one per session
end. Read: none — but note a left-brain placement would **embed** the
digest (E5), which is pure waste for something that needs identity lookup,
plus it changes the vector store cardinality measured in the memperf
baselines.

**Offline behaviour.** Same as the stores (fully local); the digest LLM call
would need the local llama-server (single slot contention, see §11).

**Long-term behaviour.** Session rows accumulate (append-per-session) or
are overwritten; without a retention rule the namespace grows and keeps
entering similarity space.

**Characteristics summary:** *fits the storage substrate but fights its
access model*; the stores are similarity-addressed, the checkpoint is
identity-addressed. Higher semantic-coupling risk, no retrieval needed,
higher validation burden to prove it never pollutes fact ranking,
cleanup quotas or supersession chains.

---

## 6. Option B — Session checkpoint as a separate session/history record

**Where it would live.** A first-class record in the session/history
structures: either extending `SessionTracker.session_state` (additive
column, e.g. `checkpoint TEXT`) or a sibling table
(`session_checkpoints`) in the same space sqlite. This is the
vendor-native placement: the vendor already owns "session" as a concept
here.

**What it would store.** `(user_id, session_id, started_at, ended_at,
turn_count, checkpoint_json)` — the compact structured digest plus the
session identity/ledger fields the table already tracks.

**How it would be created / updated.** `SessionTracker` already sits on
the ingest path (`record_turn` per ingest). The checkpoint write would
naturally hang off the same machinery: a `write_checkpoint(user_id,
session_id, payload)` method called at session end (and optionally a
cheap per-turn refresh of the deterministic fields).

**When read / entering the new session context.** `get_checkpoint(user_id)`
→ O(1) SQL by `user_id` (or `user_id + is_latest`) → rendered into the
first-turn system prompt as its own labelled block. Direct lookup, no
retrieval, exactly the SessionTracker access pattern (`get_current_session`
already exists as the template, `session_tracker.py:93-101`).

**Leaving the context.** Explicit lifecycle: the block is rendered while
the new session's history is young (e.g. first N turns or until the
session's own turns displace it), and the record is superseded by the
next session-end write. No similarity dependence.

**Stale-state handling.** Natural home for the staleness fields:
`ended_at` timestamp, optional `expires`/age policy, supersession = the
next row for the same user. One row per session gives a queryable history
of checkpoints (last-N) at no extra cost.

**Separation from normal memory.** Strong: a dedicated table is not part
of fact/trait/heartnote space; no vector store, no ranking participation,
no cleanup quotas.

**Runtime cost.** Negligible: one indexed SQL read at session start
(sub-ms), one SQL write per session end. No embedding, no retrieval.

**New LLM call?** Only the optional session-end digest (one per session;
deterministic fallback otherwise).

**Offline behaviour.** Fully local; the digest LLM call uses the single
local llama-server slot (same constraint as every other background LLM
work — see §11).

**Long-term behaviour.** Table grows one row per session (bytes); a
retention rule (keep last K, or age out) is one SQL statement. Predictable.

**Cost of the option:** it is a **vendor-tree change** — a new VM-LOCAL
patch (additive: ALTER TABLE or a new table + two accessor methods). The
fork's pin/patch machinery handles exactly this, but every vendor change
adds rebase surface against upstream (`UPSTREAM_POLICY.md` adoption review;
invariant 5: memory data is never migrated by a code merge — an additive
column/table respects this, no migration of existing rows is needed).

**Characteristics summary:** *the vendor-native, structurally cleanest
record semantics* (session identity + digest in one place, O(1) lookup,
clean supersession) at the price of **one additive vendor patch** and the
ongoing divergence bookkeeping that implies.

---

## 7. Option C — Session summaries as special elements of the existing episodic memory

**Where it would live.** Inside the right brain's episodic rows: a
heartnote-class or experience-class row whose `content` is the session
digest (the reserved `response_experience` class and even the reserved
`current_session` AnchorType / `ttl='session'` TTL class show the vendor
reserved vocabulary for exactly this direction without ever wiring it).

**What it would store.** One digest row per session, event-time `created_at`
= session end, `condition_text` = "at session start", metadata carrying
the structured fields.

**How created / updated.** Reusing `RightBrain.write`-adjacent paths at
session end (or a direct store upsert keyed by a synthetic anchor).

**When read / entering the new session context.** Through the normal
right-brain search: anchor-routed retrieval. To make the digest reliably
present at session start it would need a **synthetic always-on anchor**
(the `current_session` AnchorType was reserved for precisely this shape
and never constructed) — i.e. a routing special case, or reliance on the
`global_style` fallback anchor that every query plan already carries.

**Leaving the context.** Quota system: situation_pattern rows are capped at
2 per turn (plus response_experience 0 by default, profile 3) — the digest
would *occupy an episodic slot every turn* while active, displacing real
episodic notes; "leaving" happens by TTL (`ttl='session'` = 1-day
read-side expiry — a class that exists but is test-only today).

**Stale-state handling.** TTL class + `updated_at`; no real supersession
semantics (it is not a claim about the person).

**Separation from normal memory.** Weakest of the SQL-based options: the
digest rows live in the episodic table, participate in heartnote counts
(every +50 heartnotes → LLM cleanup pass over ALL heartnotes — the digest
would both inflate the counter and be "cleaned"/superseded by the cleanup
LLM, which is allowed to propose deletions for it), and inherit the
verbatim-content expectations of the table (heartnote `content` is by
convention the *user's verbatim words* — a digest breaks that invariant
for every consumer that relies on it, including attribution batching and
the SC-1a-documented one-time content refinement).

**Runtime cost.** Read: inside the existing right-brain search (no extra
call, but consumes quota). Write: one row per session.

**New LLM call / embedding?** Digest production optional; heartnotes are
deliberately **not** embedded (anchor-routed) so no embedding waste — but
also no identity lookup.

**Offline behaviour.** Local; same single-slot caveat for the digest.

**Long-term behaviour.** Digest rows age out via TTL or accumulate as
history; cleanup interactions are the main long-term unknown.

**Characteristics summary:** *reuses existing machinery but overloads
episodic semantics* (verbatim-content convention, cleanup quotas, anchor
routing) — the capability is identity-addressed while the substrate is
anchor-addressed. Lower structural divergence than it looks, because the
required special cases (synthetic anchor, quota carve-out, cleanup
exemption) each erode the "no new mechanism" advantage.

---

## 8. Option D — Session state in a separate persistence layer under the same VoiceMEM API

**Where it would live.** The memory space **already contains** a separate
persistence lane: the `kv` table in the space sqlite
(`utils/common/space.py:118-145`), accessed through the vendor's own API
(`kv_get` / `kv_set`), already used by the fork for throttle state.
A checkpoint is one namespaced key, e.g.
`kv["session_checkpoint:<user_id>"] = { …compact JSON… }`.
The layer is separate from facts/traits/heartnotes by construction, yet it
lives in the same sqlite file, the same space, the same backup/copy story
("copy the folder, the checkpoint travels with the memory").

**What it would store.** The compact structured record (see §19 data
model): schema version, session timestamps, turn count, topic, state,
decisions, open items, last user goal, next step, and provenance of how
the digest was produced (`llm` / `deterministic`).

**How created / updated.** App-side at session end: a small
"session-checkpoint writer" that (a) assembles deterministic fields from
the turns it already holds (`_history`, the gate's submitted pairs), (b)
optionally asks the local LLM for the abstracted digest (one bounded call),
(c) `kv_set`s the JSON. Update = overwrite the same key (single-slot
supersession). Optionally a cheap per-turn refresh of the deterministic
fields (last topic/turn count) to bound crash loss — sub-ms writes.

**When read / entering the new session context.** App-side at session
start: one `kv_get` (sub-ms, indexed primary key) when the WS session is
created or at the first turn; render the (freshness-gated) block into the
system prompt as its own labelled section, e.g.:

```
Where we left off (last session, ended 2026-09-18, 21:40 — may be outdated):
- Topic: VoiceMEM memory audit
- State: independent audit completed
- Decisions: no major architecture change; no cache; M1/M2 need real-machine validation
- Open: run field validation; decide M1/M2
- Next step: field validation on a production-like machine
```

Direct lookup; the normal `vm.search` runs exactly as today, untouched.

**Leaving the context.** Explicit: include the block only while the new
session's own history is below a threshold (e.g. first 2-3 turns / while
`_history` is empty) or for the whole session — a one-line policy either
way; the record itself is superseded by the next session-end write.

**Stale-state handling.** All staleness machinery is data, not
infrastructure: `ended_at` timestamp always rendered; an age policy
(include with date marker up to N days; then omit); supersession by
overwrite; no confidence scoring (nothing is measured — the digest is a
rendered prompt block, not a ranked memory).

**Separation from normal memory.** Complete: nothing in the fact/trait/
heartnote/vector stores ever sees the checkpoint. The ingest pair stream
(`(user_text, reply)`) is unchanged, so there is no path by which
checkpoint text leaks into extraction. (Verified: the checkpoint is only
ever a *system-prompt* block; `Ingest` receives only the turn pair.)

**Runtime cost.** Read: one sqlite PK lookup (<1 ms) once per session.
Write: one sqlite upsert per session end (+ optional per-turn sub-ms
refresh). Zero steady-state cost; zero effect on the measured search
path (P50 ≈ 100 ms at 118 facts).

**New LLM call?** Optional one per session end (the digest). The
deterministic fallback (truncate last user utterance + last reply + the
accumulated fields) means the capability works with zero LLM involvement.

**Embedding?** None. Ever. The checkpoint is not a vector citizen.

**Offline behaviour.** Fully local (sqlite); the only network/LLM
dependency is the optional digest call, which degrades to the
deterministic digest.

**Long-term behaviour.** One key, overwritten per session: constant size,
no accumulation, no retention job. If a last-N ring is ever wanted, the
key becomes a small list (bounded, e.g. last 3) — still one kv entry.

**Cost of the option:** **zero vendor change** — the app calls an
existing vendor API (`kv_get`/`kv_set`) with the memory root it already
holds. All new code is app-side (a writer, a reader/renderer, a freshness
policy, tests).

**Characteristics summary:** *a separate lane that already exists in the
architecture, used from the app side* — minimal change, O(1) lookup, no
retrieval, no embedding, no vendor divergence, complete separation from
semantic memory. Its "record semantics" are thinner than Option B (a
kv string vs. a first-class session row): no queryable session ledger,
identity = the key string.

---

## 9. Option E — Minimal extension of the existing session/history structure

**Where it would live.** The existing structure IS `WebSession._history`
(the 8-entry in-memory window). The minimal extension: persist the tail of
`_history` at session end (to kv or a file), restore it into `_history`
at session start.

**What it would store.** The last ≤8 raw transcript entries (optionally
trimmed to the char budget), plus the session end timestamp.

**How created / updated.** One write at WS close (and CLI exit), one read
at session start. No LLM, no digest.

**When read / entering context.** Restored into `_history` itself — the
existing window machinery (8-entry window, 14 000-char budget,
`_fit_history_budget`) then treats restored entries exactly like live
ones. Zero new prompt code.

**Leaving the context.** Natural: the window scrolls as the new session
produces turns.

**Stale-state handling.** Timestamp on restore (optional marker on the
first restored entry); the 8-entry window bounds size by itself.

**Separation from normal memory.** Complete (it never touches the stores).

**Runtime cost.** Negligible (one small read/write per session).

**New LLM call / embedding?** None.

**Offline / long-term.** Fully local; constant bounded size.

**The honest problem with E:** it restores a **transcript**, and the
operator's definition explicitly says SESSION_STATE is *not* the
conversation history and not a transcript. E gives continuity of *wording*
("…as I was saying before the reload") rather than continuity of *state*
(decisions/open items/next step). It also consumes prompt budget with raw
turns (up to several hundred tokens of verbatim text vs. a ~60-150-token
digest), and its quality does not improve with LLM involvement (there is
nothing to abstract — it IS the raw material).

**Characteristics summary:** *the smallest possible change and a genuine
continuity effect, but it answers a different requirement* (transcript
restoration, not state abstraction). Fits as a stopgap or as the
"deterministic fallback" data source for a digest producer rather than as
the capability itself.

---

## 10. Additional Option(s)

### 10.1 Option F1 — the concrete kv-checkpoint (D made explicit)

Option D *is* this, but for clarity the concrete shape studied forward:
one namespaced kv key per user/space, compact JSON, written at session end
(deterministic fields + optional one-shot LLM digest with deterministic
fallback), read once at session start, rendered as a bounded, dated,
clearly-labelled prompt section, freshness-gated, superseded by overwrite.
All details are those of §8. This is listed separately only because the
operator asked for options "the current code makes natural" — and this
one is natural precisely because the fork already uses the kv lane for
exactly this class of small persistent state (cleanup throttles).

### 10.2 Option F2 — wiring the dormant vendor session machinery (complementary plumbing, not a store)

Not a checkpoint store: the activation of what already exists. Pass a
`session_id` through `ingest()` (the vendor API already accepts it,
`orchestrator.py:1008`), and call `Flush()` at session end (the vendor
designed it as the idempotent "conversation formally ended" call). Effects,
all proven by code reading:

* `SessionTracker.record_turn` starts receiving real session identity →
  `session_changed` fires at real boundaries;
* the session-boundary batch runs: left-brain subgraph checkpoint, per-slot
  schema description refresh (the `slot_profiles.summary` machinery that is
  dormant today), right-brain long-term attribution;
* `Flush()` covers the last session (which has no successor ingest).

**Runtime caveat (documented, from the memperf-0105 audit):** the boundary
batch's legs include the **non-streaming LLM class** (attribution, schema
refresh) that holds the single llama-server slot — running them
synchronously at WS close is safe for the *closing* session (the user is
gone) but can contend with the *next* session's first turn if they overlap.
Any F2 wiring needs the same idle-gate/time-budget treatment as the
existing background work (or a bounded synchronous budget at close).

F2 is orthogonal to the checkpoint placement: any of A/B/D/E can ride on
it (the session-boundary moment it creates is the natural checkpoint
write trigger), but none *requires* it — a checkpoint writer can be called
directly at WS close/CLI exit without activating any vendor batch.

---

## 11. Performance Considerations

### 11.1 Size: what is the reasonable maximum?

Anchors from the measured system:

* the whole memory funnel today renders at ≈123 tokens on a realistic
  store (v0.9.2 cardinality benchmark; hard cap 1 200 chars);
* the history window is 8 entries under a 14 000-char prompt budget
  (designed for an 8 192-token context; the server now declares 32 768 —
  the guard stays conservative);
* the user's own ceiling: 200-500 tokens.

**Analysis:** the checkpoint is injected *in addition to* the memory block
and (initially) *instead of* history. A structured digest of
**300-500 characters (≈100-200 tokens)** carries the six fields with room
for 2-3 items per field — the example in the operator's brief renders at
≈90 tokens. Recommendation-shaped guidance (not a decision): keep the
*rendered* block ≤ ~500 chars; store internally as JSON (schema-versioned)
so the render policy can evolve without touching stored data. 1-2 sentences
per field is the natural granularity; a 200-token ceiling forces the
abstraction the capability wants anyway.

### 11.2 Retrieval vs. direct lookup

* Retrieval (embedding search) is **not needed and not justified**: the
  checkpoint is a single identity-addressed object per user/space.
* Direct lookup is **O(1)** in every SQL-based option: kv PK read or an
  indexed `session_state`/`session_checkpoints` row — sub-millisecond,
  against a measured search path of P50 ≈ 100 ms (118 facts) / 177 ms
  (4 000 facts). The checkpoint read is three orders of magnitude below
  the work the first turn already does.
* Options A/C are the exceptions: their read path is similarity/anchor
  shaped and cannot guarantee presence (§5, §7).

### 11.3 First-audio / first-token latency

The kv/SQL read can happen at WebSession construction (before the first
utterance exists) — it adds nothing to the first turn's critical path. If
read at first turn instead, it is sub-ms inside a turn that already spends
≥100 ms on search + seconds on LLM. No perceptible overhead either way.

### 11.4 Write-side cost

One sqlite upsert at session end (sub-ms; the space sqlite is already
WAL-per-call from the stores' perspective). The optional per-turn
deterministic refresh is also sub-ms per turn — bounded, and skippable.

### 11.5 The one real cost: the optional LLM digest at session end

Evidence-based sizing: the ingest extraction prompt is the ~10.4K-token
class on the single llama-server slot; a session digest prompt is far
smaller (the six fields + the session's turn pairs, typically well under
2K tokens), and it runs when the closing session no longer needs the
slot. Risks to manage (same class as the memperf IMPROVE findings):
(a) it must NOT run as a non-cancellable non-streaming leg that a new
session's first turn queues behind — it needs the same idle-gate
discipline or a post-close, time-budgeted execution;
(b) a hard timeout with the deterministic fallback keeps the checkpoint
path non-blocking by construction.

### 11.6 Steady-state invariant

With the checkpoint absent, deleted, or corrupt, every code path behaves
exactly as today (the read returns nothing; the render omits the block).
The capability is pure addition on the read side.

---

## 12. Lifecycle

### 12.1 Session start

```
NEW SESSION (WS connect / CLI start)
  → session checkpoint lookup        [O(1): kv_get / SQL row; <1 ms]
  → freshness gate                   [age policy; date always rendered]
  → optional normal memory retrieval [vm.search — exactly as today]
  → context construction             [persona + memory block + CHECKPOINT BLOCK + history]
  → LLM
```

**Automatic or conditional?** Analysis of the cases:

* **completely new session (same user/space):** include — this is the
  capability's purpose.
* **short-reopened session (e.g. tab reload seconds later):** include —
  the checkpoint was just written from that very session; with a restored
  transcript (Option E hybrid) this is the "welcome back" case. With a
  digest-only design the short-gap case is still served (the digest
  describes the session that just ended).
* **long-after new session:** include **with the age marker**; beyond the
  TTL, omit. The date is always rendered so the LLM (and user) can weigh
  it.
* **brand-new space / first ever session:** no checkpoint exists → block
  omitted, zero cost.
* **different user/space:** the key is per user/space — each space has its
  own checkpoint; no cross-contamination.

Injection window policy (open question §18): first-turn-only vs.
first-N-turns vs. whole session. The structurally simple choice consistent
with the existing architecture: inject while `_history` is empty (the
first turn of every WS session) — i.e. exactly when the system today has
*no* conversational context. Keeping it for the whole session is equally
one line and keeps "next step" visible throughout; the cost is ~100-200
tokens per turn against a 14 000-char budget (measured-not-binding).

### 12.2 Session end (production of the state)

Analysis of the operator's A-E menu:

| Trigger | LLM cost | Latency risk | Analysis |
|---|---|---|---|
| A) update after every turn | 0 (deterministic fields) | none (sub-ms kv/SQL write) | viable ONLY as a deterministic accumulator (turn count, timestamps, last topic = current utterance head); a per-turn LLM summary is explicitly unjustified — it would put a background LLM call behind *every* turn on the single slot (the measured contention class) |
| B) periodic checkpoint (every N turns) | 1 LLM per N turns | queues behind conversation | the gate already exists to serialize such work; still adds avoidable slot traffic mid-conversation |
| C) session end checkpoint | 1 LLM per session | none for the closing session; budgeted for the next | the natural point: the turn pairs are complete, the user is gone, the material (`_history`/gate pairs) is about to be discarded |
| D) on important state change | heuristic + maybe LLM | mid-conversation | requires an importance detector — new mechanism, scope creep |
| E) combination | — | — | **C + (optional A)**: per-turn deterministic accumulation to bound crash loss, plus the session-end digest (LLM with deterministic fallback) |

**Is the existing background pipeline suited?** Partially. The
BackgroundMemoryGate is *session-owned* (its worker dies in
`_cancel_session_tasks` at WS close) — a session-end job cannot ride it
at the moment of close. Two workable shapes (to be decided at design
time, both app-side): (a) a bounded synchronous job at close (digest with
a hard timeout; deterministic fallback on timeout), or (b) a short-lived
server-level (non-session) task with the same idle discipline the gate
uses. Shape (a) is simplest and bounds the close path; shape (b) keeps
the close path instant. F2's `Flush()` (if ever activated) has the same
two shapes.

---

## 13. Freshness / Staleness

The operator's cases, mapped to mechanisms that need **no coupling to
semantic memory**:

| Checkpoint state | Mechanism |
|---|---|
| yesterday | included; date rendered |
| one week old | included (default TTL ≥ 7 d) with the date marker; beyond TTL → omitted |
| a month old | omitted by TTL (or included-with-warning if the policy is raised) |
| partially refuted by new turns | no invalidation machinery — the prompt block is framed as "last session state; verify against what the user says now"; the *facts* layer does what it already does (supersession) for durable knowledge; the next session-end checkpoint overwrites |
| already closed items | same: static digest + overwrite on next checkpoint |
| no longer relevant | same: date framing + TTL + overwrite; the LLM's instruction block tells it to drop stale framing |

Mechanisms available **without** conflating SESSION_STATE with semantic
memory: `ended_at` timestamp (always), age policy/TTL (config), supersession
(single-slot overwrite; or last-N ring), and — optionally — a `digest_method`
provenance field (`llm` / `deterministic`) that a future reader could treat
as a soft confidence proxy. What is deliberately NOT introduced: similarity
checks against the fact store, semantic invalidation, confidence scoring —
each would couple the checkpoint to the memory layer the operator wants it
separated from.

Multi-session interleaving (two tabs): the checkpoint records
`started_at`/`ended_at`; interleaved sessions are detectable from the
timestamps even though the store is last-writer-wins (§14).

---

## 14. Failure / Recovery

| Failure | Effect | Mitigation / fallback |
|---|---|---|
| session crash / process crash mid-session | no session-end write happens | previous checkpoint remains (or none); optional per-turn deterministic accumulator bounds the loss to the tail; degradation = today's behaviour |
| checkpoint **write** failure (sqlite error) | `kv_set` catches and logs (existing behaviour, `space.py:137-145`) | next session reads the previous/none; non-fatal by construction |
| checkpoint read failure / corrupt JSON | `kv_get` returns the default on any exception | treated as absent → block omitted; the corruption self-heals on the next write |
| **LLM digest** failure / timeout | no abstracted digest | deterministic digest (accumulated fields + truncated last exchange) — the checkpoint is written regardless, marked `digest_method: "deterministic"` |
| partial session (one turn, then disconnect) | checkpoint written from 1 turn pair | valid (topic/state reflect that turn); if a policy "min turns" is wanted, it is one condition |
| concurrent sessions (two tabs, same space) | both write at close; last writer wins | timestamps make interleaving visible; degradation is a checkpoint describing one of the two sessions — acceptable and documented; per-space keying matches how spaces already work (the active-space switch is already a global mutation, same class) |
| duplicate checkpoints | impossible by construction (single key / one row per session identity) | — |
| write lands while a reader reads | sqlite WAL per-call connections | no torn reads (single-statement upsert; JSON loads guarded) |

**How much does SESSION_STATE always need to exist?** Not critical-path.
It is a continuity nicety layered on a system whose durable knowledge is
already in the fact/trait stores. The design invariant: **every consumer
must behave identically when the checkpoint is missing** (empty render,
no prompt slot, no latency). Fallback chain: fresh checkpoint → stale
checkpoint with date → deterministic digest → none.

---

## 15. Upstream Divergence

Pin: `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` + 22 VM-LOCAL patches
(`VOICEMEM_PIN.json`), policy `UPSTREAM_POLICY.md` (hard invariants: local
E5 only; no new OpenAI runtime dep; no masked failures; HU/EN preserved;
memory data never migrated by code merge; Qwen3.6 on local llama-server;
DELETE opt-in).

| Option | Vendor change | Divergence character |
|---|---|---|
| A | schema/ranking semantics in fact/heartnote stores | **higher** — touches the stores upstream keeps evolving; every upstream change to search/ranking/cleanup needs a re-check against the checkpoint's special cases; also touches invariant-adjacent surfaces (fact ranking purity) |
| B | additive: `session_state` column or sibling table + 2 accessors | **low-moderate** — one new VM-LOCAL patch, additive, no migration (invariant 5 respected); `SessionTracker` is a quiet corner of upstream; rebase cost is small but real and permanent |
| C | mostly none (reuses classes) — but needs quota carve-outs/synthetic anchor to be reliable | **moderate on paper, higher in practice** — the special cases erode the no-change claim and live inside tables upstream actively evolves (response_experience is already fork-gated VM-LOCAL-017) |
| D / F1 | **none** — app-side calls to the existing `kv_get`/`kv_set` API | **minimal** — no vendor file touched; the kv lane is already part of the shipped tree and already used by the fork for throttle state; nothing to rebase |
| E | none (app + optional kv/file) | **minimal** — app-only |
| F2 | none required (vendor API already exposes `session_id` + `Flush()`); pure app wiring | **minimal in code**, but *activates* vendor behaviours (boundary batch legs) with real runtime consequences (§10.2) — an operational divergence more than a code divergence |

**The "second memory system" test:** A and C risk *becoming* one (their
rows live in, and evolve with, the semantic stores). B and D/F1 are a
*record*, not a *system*: one compact object, one accessor pair, no
ranking, no retrieval, no extraction. E restores transcript data — a
cache, not a system.

Preference order of the policy axes (natural in current architecture /
minimal local change / no upstream break / no second memory system)
points to D/F1 first, E alongside, B next, C then A. This is a
characteristic statement, not a ranking verdict — the trade-offs per
option are the sections above.

---

## 16. Implementation Complexity (for sizing only — nothing implemented now)

| Option | New code (rough) | Touch points | Test surface |
|---|---|---|---|
| A | store tagging + render special-case + retention | vendor stores, search/render, cleanup exemptions | fact-ranking purity, cleanup interactions, supersession isolation — the largest validation matrix |
| B | vendor table/column + accessors + app writer/reader | vendor `session_tracker.py`, app session lifecycle | vendor regression suite + pin/patch bookkeeping (VOICEMEM_PIN entry, identity block) |
| C | special row writer + synthetic anchor + quota carve-out + cleanup exemption | vendor right brain (several seams) | heartnote-convention consumers, quota behaviour, cleanup pass |
| D / F1 | app-side writer (deterministic + optional digest) + reader/renderer + freshness policy + tests | app session lifecycle only (WebSession close/start; CLI exit/start); config knob | unit (writer/fallback/freshness), integration (round-trip across a simulated restart), prompt-budget guard test |
| E | persist/restore `_history` tail | app session lifecycle only | round-trip + budget tests (smallest) |
| F2 | session-id mint/propagation + Flush call at close (budgeted) | app ingest path, WS close, CLI exit | boundary-batch activation (needs its own runtime evidence — the non-streaming-leg class) |

All options share the same small render/policy core (freshness gate +
prompt block + budget guard), which is where most of the value and most
of the tests live regardless of placement.

---

## 17. Risks

1. **Misleading stale state** — the digest says "next step: X" and the
   user has moved on. Mitigations: date always rendered, TTL, prompt
   framing ("may be outdated"), overwrite-on-next-session. Residual risk
   accepted: a static digest cannot know it was refuted.
2. **Prompt budget erosion** — the block adds ~100-200 tokens/turn while
   active. Guarded by the existing 14 000-char budget; measured
   not-binding at current densities; must be covered by a budget test.
3. **Slot contention at session end** — the optional digest (and F2's
   boundary legs) are LLM work on the single llama-server slot; without
   the gate/timeout discipline this recreates the measured
   foreground-blocking class. Mitigation: hard timeout + deterministic
   fallback + post-close/idle execution.
4. **Multi-tab last-writer-wins** — checkpoint describes one of several
   interleaved sessions. Accepted degradation (documented), detectable
   via timestamps.
5. **Scope creep toward the Learning Engine** — the digest must stay
   conversation-level and user-facing; any "what the system learned"
   semantics is a different workstream (explicitly out of scope; the
   LEARNING_ENGINE_INTEGRATION_PLAN keeps VoiceMem read-only evidence).
6. **Semantic-memory coupling** (Options A/C) — quota occupation,
   cleanup interactions, supersession pollution; the reason those
   options carry higher validation burden.
7. **Silent permanence** — a checkpoint that stops being written (a bug
   in the close path) would serve an ever-staler state; the TTL turns
   this into graceful disappearance, and `digest_method`/`ended_at`
   make it observable in diagnostics.
8. **Vendor rebase cost** (Option B) — permanent but small; one more
   patch to re-apply at every pin review.

---

## 18. Open Questions

1. **Injection window:** first turn only / first N turns / whole session?
2. **TTL default:** 7 days (include-with-date) then omit — or longer,
   given a personal assistant's rhythm? Operator decision.
3. **Digest producer:** LLM at session end (with timeout + fallback) vs.
   deterministic-only (fields + truncated last exchange)? The LLM digest
   is where the "decisions / open items / next step" abstraction quality
   comes from; the deterministic floor is where the reliability comes
   from. Is the quality delta worth one more background LLM call per
   session on the single slot?
4. **Execution shape of the session-end job:** bounded synchronous at WS
   close vs. short-lived server-level task (§12.2)?
5. **CLI parity:** is wiring the CLI exit path (pending-commit drain +
   checkpoint write) part of the first cut, or web-first?
6. **Last-N ring vs. single slot:** does any product need the previous
   2-3 sessions, or always just the last?
7. **UI surfacing:** should the checkpoint render as a "welcome back"
   card in the web UI (it already exists server-side for the prompt)?
8. **F2 activation:** independently desirable (it un-dorms schema refresh
   and long-term attribution), but its runtime evidence (boundary-batch
   legs at close) must be gathered first — separate decision, separate
   task.
9. **Space-switch semantics:** the checkpoint is per memory space; when
   the user switches spaces mid-UI-session (already a global mutation),
   should the freshly-read checkpoint of the new space apply to the next
   turn? (Consistent answer: yes — lookup is cheap and keyed.)

---

## 19. Recommended Direction (PROVISIONAL)

**Characteristic comparison (no scores, no "best"):**

* **Option D / F1 (kv-checkpoint)** has the following characteristics: zero
  vendor change; an already-existing persistence lane the fork already
  uses for the same class of state; O(1) identity lookup with no
  retrieval/embedding; complete structural separation from semantic
  memory; graceful degradation to today's behaviour; constant size; and
  the thinnest record semantics of the SQL options (a keyed string — no
  first-class session ledger).
* **Option B (session record in/next to `SessionTracker`)** has these
  trade-offs: the strongest record semantics (session identity + digest +
  a queryable per-session history in the place the vendor already owns
  "session"); one additive vendor patch and its permanent rebase cost;
  otherwise the same O(1)/no-retrieval/separation profile as D.
* **Option E (history-tail persistence)** has these characteristics: the
  smallest possible change and genuine continuity of wording, but it
  restores a *transcript*, which the capability definition explicitly
  excludes; its natural role is as the deterministic data source for a
  digest producer, or as a stopgap.
* **Option A (memory type/namespace)** and **Option C (episodic elements)**
  share the structural mismatch: identity-addressed state on
  similarity-addressed substrates; reliable presence requires special
  cases that erode their low-change appeal, and both couple the checkpoint
  to stores whose evolution (upstream) and machinery (cleanup, quotas,
  supersession) were not designed for it.

**PROVISIONAL RECOMMENDATION (technical fit only):**

The direction that fits the current VoiceMEM architecture is **Option D /
F1**: a compact, schema-versioned JSON checkpoint in the existing space
`kv` table, keyed per user/space, written at session end by an app-side
writer (deterministic fields always; one bounded optional LLM digest with
a deterministic fallback), read once at session start through the same
vendor API, rendered as a dated, freshness-gated, explicitly-labelled
prompt section while the new session has no history of its own, and
superseded by the next session-end write. Option B is the natural
*upgrade path* if a first-class session ledger ever becomes a product
need: the kv key and a `session_checkpoints` row carry the same payload,
so a later move is a data copy, not a redesign. Option E's
history-persistence may be adopted *inside* the writer as the
deterministic fallback material. F2 (session_id + Flush wiring) is a
separate, independently-evidenced decision that any placement benefits
from but none requires.

---

## FINAL ANSWER — "Can VoiceMEM support a compact persistent last-session state without creating a second general-purpose memory system?"

**Yes.** The architecture already contains every part needed; nothing
about the capability requires a second memory system.

* **How it fits the current architecture:** the memory space already has
  a generic, application-agnostic persistence lane — the `kv` table in
  the space sqlite (`voicemem/utils/common/space.py:118-145`) — that the
  fork already uses for small persistent state (cleanup throttles). A
  session checkpoint is the same class of citizen: one namespaced key per
  user/space, written and read through the existing vendor API from the
  app side. It never enters the vector store, the fact tables, the trait
  tables, or the heartnote tables; it is never retrieved, never embedded,
  never ranked, never cleaned up.

* **Minimal change:** app-side only — (1) a session-end writer
  (~tens of lines) hooked where the WS session is torn down (and the CLI
  exit path if wanted), (2) a session-start reader + renderer (~tens of
  lines) plus a freshness gate, (3) one prompt-budget guard test. Zero
  vendor changes, zero schema changes, zero new dependencies, zero
  migration. (The optional LLM digest reuses the existing local
  llama-server; the deterministic fallback needs no LLM at all.)

* **Data model (per user/space, one JSON object):**

  ```json
  {
    "schema": 1,
    "session": {"started_at": "…", "ended_at": "…", "turns": 12, "session_id": "…"},
    "topic": "VoiceMEM memory audit",
    "current_state": "Independent audit completed",
    "decisions": ["no major architecture change", "no cache", "M1/M2 require real-machine validation"],
    "open_items": ["run field validation", "decide on M1/M2"],
    "last_user_goal": "field validation on a production-like machine",
    "next_step": "field validation on a production-like machine",
    "digest_method": "llm" | "deterministic"
  }
  ```

  All fields free-text or string lists — no application-specific
  vocabulary, no Goal Catalog, no workflow states. Anything richer than
  this shape is OUT OF SCOPE for SESSION_STATE.

* **When written:** at session end (WS close / CLI exit), always the
  deterministic core, plus the optional one-shot LLM digest with a hard
  timeout; a per-turn deterministic refresh (turn count, timestamps,
  last topic) is optional crash-loss bounding. Single key → each new
  session's write supersedes the previous.

* **When read:** once at session start (WebSession construction or first
  turn), through the same `kv_get` the throttles use; sub-millisecond.

* **How it stays small:** the stored object is bounded by construction
  (one session, six fields); the rendered block is capped
  (≤ ~500 chars / ~100-200 tokens, well inside the 14 000-char prompt
  budget alongside the 1 200-char memory block); the store is one key
  (constant size — no growth, no retention job); and the injection
  window ends when the session's own history exists.

* **Runtime performance guarantee:** the read is one indexed sqlite
  lookup (< 1 ms — three orders of magnitude under the measured
  ~100 ms search path) and can run before the first utterance exists;
  the write is one sub-ms upsert at close; the optional digest is one
  bounded background-class call at close (timeout + fallback, never on
  the conversation's critical path); and with the checkpoint absent,
  deleted, or corrupt, every path behaves exactly as it does today.

**One architectural honesty note:** this study found that the vendor's
own session machinery (`session_id` → `SessionTracker` → boundary batch →
`Flush()`) is complete and dormant in our deployment. Wiring it (Option
F2) is not required for the checkpoint and carries its own runtime
evidence burden — but it is the natural *companion* decision, because the
session boundary it would create is precisely the moment a checkpoint
writer wants to run.

---

*Study artifacts: based on direct code reading of `app/web_server.py`,
`app/pipeline.py`, `app/voicemem_bridge.py`, `app/background_memory.py`,
`app/teacher_persona.py`, the vendored `voicemem` tree at pin e8384e0
(orchestrator, core, memory_api, session_tracker, space, right-brain
stores, left-brain repository, mem0 backend, fusion demo),
`docs/MEMORY_SEMANTICS.md`, `UPSTREAM_POLICY.md`, `VOICEMEM_PIN.json`, and
the measured evidence of the v0.9.2 / memperf-0105 audits. No production
or runtime code was modified in producing this document.*
