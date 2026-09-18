# VoiceMem ↔ MAXIM Learning Engine Integration Plan

Status: Architecture plan — updated with the completed v0.10.5 implementation audit
Audit baseline: VoiceMemAgent v0.10.5 + TTS code-switching phase 2 (2026-09-18)
Scope: VoiceMem ↔ MAXIM V4 Learning Engine ↔ Knowledge Bundle
Owner: Architecture
Companions: `docs/NEXT_ARCHITECTURE_WORKPLAN.md` (sequencing), `docs/BRITISH_ENGLISH_AND_LOCALISATION.md` (Workstream B), `audit/VoiceMEM_arch_prep_v0105/REPORT.md` (full evidence: the 15-section final report)

## 1. Purpose

This document defines how the existing VoiceMem memory system shall connect to the MAXIM Learning Engine without creating a second competing memory system.

The core architectural decision is:

- **VoiceMem is the evidence and experience layer.**
- **The Learning Engine is the learning, validation, approval and mutation layer.**
- **The Knowledge Bundle is the canonical persistent knowledge boundary consumed by MAXIM.**

VoiceMem remains responsible for remembering what happened. The Learning Engine decides whether observed evidence represents knowledge worth proposing and, after authorisation, mutates the canonical Knowledge Bundle.

## 2. Existing Architecture We Preserve

MAXIM already defines a controlled learning lifecycle:

Observation → Candidate Knowledge → Validation → Learning Proposal → User Decision → Knowledge Mutation → Knowledge Bundle Update → Version Creation

Persistent user knowledge has one writer: the Learning Engine.

VoiceMem must therefore remain read-only from the Learning Engine's perspective. It supplies evidence; it does not become an alternative writer of MAXIM knowledge.

## 3. Responsibility Split

| Layer | Primary question |
|---|---|
| VoiceMem | What happened before? What evidence do we have? |
| Learning Adapter | Which VoiceMem evidence is relevant to learning? |
| Learning Engine | Is there enough evidence to form a candidate and should we propose it? |
| Knowledge Bundle | What persistent knowledge is currently canonical? |
| Parser / Knowledge Base | What language pattern is present and what goal may it represent? |
| Goal Resolver | What does the user want now? |
| Planner | What should the system do? |
| Home Assistant Gateway | How is the approved plan executed? |

## 4. Audited VoiceMem Capability Matrix (2026-09-18)

Every memory type was inspected in the live tree (file:line inventory: final report §2). Primary MAXIM role per structure:

| Structure | Lives in | Persists how | MAXIM role (primary / secondary) |
|---|---|---|---|
| Facts | vendor left-brain + Qdrant `voicemem` collection + kv-JSON mirror | permanent, append-only, supersession-chained | canonical persistent knowledge / evidence (metadata provenance) |
| Episodic memories | `right_brain_memories` (heartnotes, agent-reply rows, sound-only rows) | SQLite, permanent (TTL is a read-side filter) | observation / historical |
| Traits (5 slots) | `rb_traits` + `rb_evidence` | SQLite, permanent (read-side decay) | canonical persistent knowledge / candidate source |
| Preferences | no dedicated store — fact slot tag + graph `preference` entities + trait slot | permanent | canonical persistent knowledge / candidate source |
| Stance | deterministic classifier at ingest; stored on fact/trait rows | rides parent rows | evidence |
| Emotion | app `data/emotion_log.jsonl` + vendor emotion graph | JSONL append-only + SQLite | observation / evidence |
| Heartnotes | `right_brain_memories` (memory_class=heartnote) | SQLite, permanent | observation / candidate source |
| Routines | `scene_observations` + `routines` tables → synthetic fact on threshold | SQLite + Qdrant | candidate source (the cleanest existing learning pattern) / canonical knowledge (resulting fact) |
| Corrections | deterministic cue tables → reaction path; content corrections → fact supersession | ephemeral signal → persistent rows | observation / candidate source |
| Supersession | three isomorphic chains (facts / traits / heartnotes), never destructive | permanent | historical / evidence |
| Evidence | `rb_evidence` rows, `evidence_memory_ids`, emotion episode evidence | permanent, append-only | evidence |
| Temporal information | ISO fields + read-side decay + derived `row_status` | rides parent rows | historical |
| Occurrence counts | fact metadata + trait columns | permanent | evidence |
| Confidence | heterogeneous (traits asymptotic reinforcement; facts inert 0.9; entities/edges own scales) | rides parent rows | evidence (explicitly not a probability) |
| response_experience | rb rows — write OFF by default since v0.10.3 (env `VOICEMEM_RB_WRITE_EXPERIENCE`) | SQLite, permanent | canonical knowledge (currently disabled) |
| Cognitive graph | entities / edges / links / slots / affective edges / activations | SQLite, permanent | canonical persistent knowledge |
| Entity / relation / aliases | graph tables + `voiceprint_registry.json` | SQLite + JSON, permanent | canonical persistent knowledge / candidate source |
| Goals | no goal store — fact slot tag `goals` + anchor types | Qdrant + SQLite | candidate source |
| Session / conversation history | `WebSession._history` (last 8 turns, ephemeral), session tracker, mem0 history tables (write-only, never read back) | mixed | observation (the primary candidate substrate) |

Ephemeral, not learning-safe: playback/barge ledgers, VAD peaks, `DemoMemoryLayer` (mock).

## 5. Learning Event — Minimal Correct Contract (Audit-Derived)

Field-by-field status on the current tree:

| Contract field | Status | Where / gap |
|---|---|---|
| event id | MISSING | no observation-event id exists (mem0 row uuid is a fact id, not an event id) |
| user reference | EXISTS | `user_id` on every row |
| session reference | PARTIAL | param exists through Ingest but the web path never passes one |
| turn reference | PARTIAL | fact metadata `turn_id` exists; rb `evidence_turn_ids` column exists but never populated by the live path |
| timestamp | EXISTS | event-time anchors (`time_start`/`created_at`/`first_seen`/`at`, four distinct meanings kept separate in docs/MEMORY_SEMANTICS.md) |
| source | PARTIAL | two vocabularies (fact metadata vs rb hit source), no unified enum |
| source memory ids | PARTIAL | `evidence_memory_ids`, `rb_evidence.cause_id`, graph evidence ids — inconsistently filled |
| source turn ids | MOSTLY MISSING | only the unused `evidence_turn_ids` column |
| event type | MISSING | closest: resolver event (ADD/UPDATE/DELETE/NONE), stance class, memory_class |
| subject | PARTIAL | `user_id`; entities have ids; fact subjects live inside the text |
| context | PARTIAL | heartnote metadata.agent_reply, CurrentSignals, emotion fusion inputs — scattered |
| observation | PARTIAL | verbatim quotes survive in several places (rb_evidence.quote, emotion-log transcript), none canonical |
| evidence | EXISTS for traits / MISSING for facts | rb_evidence rows; facts carry only flat metadata |
| candidate type | MISSING | no candidate concept anywhere |
| candidate value | MISSING | closest realised values: trait claim, fact text, routine row |
| confidence | EXISTS (heterogeneous) | traits asymptotic, facts inert 0.9, entities/edges own scales |
| status | EXISTS (derived) | `row_status` current/historical/superseded/future — computed at read, never stored |

Minimal contract v1 (interface only; grants no write access to anything):

```json
{
  "schema": "voicemem.learning_event/1",
  "event_id": "<uuid4>",
  "user_id": "webspace_<name>",
  "session_id": "<SessionTracker id — requires web-path wiring>",
  "turn_id": "<fact metadata turn_id — requires rb population>",
  "observed_at": "<ISO 8601 event time>",
  "source": "voice | agent_reply | sound_only | routine | reaction",
  "source_memory_ids": ["..."],
  "source_turn_ids": ["..."],
  "event_type": "repeated_phrase | repeated_alias | correction | preference_signal | execution_outcome | response_feedback | routine_signal",
  "subject": "...",
  "context": "...",
  "observation": "<verbatim>",
  "evidence": [{"quote": "...", "at": "...", "memory_id": "..."}],
  "candidate_type": "phrase_goal | alias | correction | preference | ...",
  "candidate_value": "...",
  "confidence": 0.0,
  "status": "emitted"
}
```

Audit insights that shaped the minimal contract:

1. The fact layer is already a near-complete LearningEvent sink — it lacks only event-id / event-type / candidate semantics, and those can ride the existing additive-metadata channel exactly as stance and temporal status did in v0.10 (zero schema migration, provenance preserved).
2. `session_id`/`turn_id` propagation from the web layer is the single wiring gap to close for full event provenance.
3. The two `source` vocabularies must be unified by the LearningEvent enum above.
4. Confidence semantics are heterogeneous by design — the Learning Engine must treat confidence per candidate_type, never cross-type.

Emission point (proposal): `BackgroundMemoryGate` — the natural deferred-scheduling hook with proven queue/coalesce/cancel semantics. The adapter listens AFTER persistence, emits events, never writes back.

## 6. Safe Candidate Sources vs Forbidden Sources

(a) phrase → goal — SAFE: `WebSession._history` (user_text, reply) pairs; fact rows tagged slot `goals` (provenance-rich). FORBIDDEN: `slot_profiles`/`slot_summaries` (LLM-aggregated rewrites, lossy provenance), `dynamic_slots` (auto-emerging persistent taxonomy).

(b) phrase → alias / entity — SAFE (read-only): `entities` + `aliases` + `entity_memory_links` + `evidence_memory_ids`; `voiceprint_registry.json` (user-confirmed bindings — the gold standard). FORBIDDEN: calling `upsert_entity` / `get_or_create_entity_semantic` from an adapter (mutates the graph; stays in the ingest path).

(c) correction learning — SAFE: the deterministic cue tables (`_CORRECTION_CUES` / `_DISSATISFIED_CUES` / `_APPRECIATION_CUES`), fact supersession chains read via `memory_history()` (both sides preserved), `data/emotion_log.jsonl` (append-only, has transcript). NOT sources: `response_experience` rows and `rb_traits` (already conclusions — using them as candidates would double-count).

## 7. Existing Single-Step Learning Behaviours (Risk Register)

Six existing behaviours already turn observations into persistent knowledge in one step. The Learning Adapter must NOT extend this pattern; remediation is a separate operator decision (listed for awareness, deliberately not changed by this plan):

1. Trait writes from LLM judgment at ingest (merged-extraction traits + attribution `user_trait`) — mitigated by the deterministic MERGE/SUPERSEDE/SEPARATE gate, but no candidate/approval status exists.
2. `run_cleanup` heartnote supersession — an LLM decision mutates persistent metadata with no audit trail beyond the marking.
3. `ConflictResolver` UPDATE/DELETE — LLM decision rewrites the fact chain immediately (lossless by construction, but no candidate stage).
4. `AttributionManager` long/short-term runs + `_refresh_schema_descriptions` — LLM rewrites entity/slot descriptions in place.
5. `DynamicSlotStore` / SubgraphManager — emergent persistent slot taxonomy from retrieval statistics.
6. Triplicated fact storage (Qdrant + kv mirror + graph wrappers) — a consistency risk, not a learning one.

## 8. Learning Phases

### Learning V1 — phrase / alias / correction learning (first practical layer)

- Emit LearningEvents from the gate after persistence (append-only JSONL, the emotion-log pattern).
- Candidates live ONLY in the Learning Engine; VoiceMem never writes back.
- Wiring: session_id/turn_id propagation (the only production touch), read-only adapter, tests.
- Exact integration points: `app/background_memory.py` (post-persist hook), `app/web_server.py` (session/turn wiring), new `app/learning_adapter.py` (read-only emitter), `tests/unit/test_learning_events.py`.

### Learning V2 — preferences / persistent corrections
### Learning V3 — execution outcome learning

The chain user request → goal → execution → result → user feedback is recorded as EVIDENCE first (the outcome event type above); repeated, validated outcomes may later generate a candidate preference or policy.

### Learning V4 — procedure / routine / skill learning

The existing routine mechanism (distinct_days >= 3 → synthetic fact) already demonstrates the pattern deterministically; the Learning Engine version turns the same signal into a candidate + approval instead of a direct fact.

### Learning V5 — knowledge consolidation

`memory_history()` is the documented consolidation hook (many observations → few stable knowledge objects, via the Learning Engine).

## 9. Knowledge Base / Goal Catalog / Missing Information Matrix Integration

The existing MAXIM Knowledge Base follows one principle: Phrase → Goal, in one Knowledge Base, not separate static and learned stores.

- Learned records enter the same canonical Knowledge Base after Learning Engine approval, carrying `source: "learned"` provenance so the runtime can distinguish manual and learned knowledge without separate runtime stores.
- Goal Catalog: VoiceMem facts tagged `goals` are EVIDENCE for goal learning, never a runtime goal store. The catalog is mutated only by the Learning Engine after approval.
- Missing Information Matrix: lives on the MAXIM side; it may query VoiceMem read-only (what evidence exists) but never writes to it.
- Knowledge Bundle: the sole persistent-knowledge boundary; VoiceMem never bypasses it.

## 10. What VoiceMem Should NOT Learn Directly

Do not promote every observed fact into persistent operational knowledge. Temporary mood, one-off behaviour, single events and transient runtime state remain memory/evidence unless a validated learning rule says otherwise. Learning must remain selective.

## 11. Governance Rules

- The MAXIM Learning Engine remains the sole writer of persistent user knowledge.
- Every implicit learning path follows: Observation → Candidate → Validation → Proposal → User Decision → Mutation.
- Every persistent mutation must be auditable, reversible and versioned.
- No Learning Engine component may execute a Home Assistant action.

## 12. Implementation Preparation (Concrete)

Before runtime learning, create and test: the LearningEvent schema, the VoiceMem → Learning Adapter interface, the candidate knowledge model, evidence provenance, candidate conflict checks, the Learning Proposal model, the Knowledge Bundle mutation interface, version/rollback fixtures, Goal Catalog integration tests, Missing Information Matrix integration tests. Runtime behaviour must not change until these boundaries are tested.

## 13. Reusable Read-Only Components (Audit-Verified)

Fact retrieval + provenance: `Mem0BackendStore.search` (tier-ranked hits with temporal/supersession/occurrence fields), `memory_history()` chain walker, `list_entries`. Deterministic classifiers (zero LLM): `leftbrain/temporal.py` (`classify_fact_temporal`, `row_status`, `query_temporal_intent`), `rightbrain/stance.py`. Trait reads: `TraitStore.search_scored`, `effective_trait_confidence`, `app/retrieval_contract.py`. Turn/session plumbing: `SessionTracker`, `turn_ids.format_turn_id`, `BackgroundMemoryGate`. Correction detection: `_reaction_signals` + cue tables (pure functions). Right-brain reads: `RightBrainStore` getters, `RoutineStore.list_routines`. Graph reads: `CognitiveGraphStore` entity/edge/link getters. Append-only sinks: `EmotionMemory.record_turn` (the JSONL pattern to copy for LearningEvents).

## 14. Long-Term Architecture

```text
                         VOICEMEM
               Evidence / Experience Layer
                         |
                         | Learning Events (append-only, read-only adapter)
                         v
                  LEARNING ADAPTER
                         |
                         v
                  LEARNING ENGINE
          Candidate / Validation / Proposal
             Approval / Mutation / Versioning
                         |
                         v
                  KNOWLEDGE BUNDLE
             Canonical Persistent Knowledge
                         |
                         v
                  MAXIM RUNTIME
      Parser → KB → Goal → Need Resolution → Plan
                         |
                         v
                   HA Gateway
```

The architecture intentionally avoids a second independent learning database.

## 15. Unresolved Architectural Decisions

1. Where do emitted LearningEvents live before the Learning Engine exists? (proposal: append-only JSONL, the emotion-log pattern — not a database, no query surface, rotation policy)
2. Does the Learning Adapter run in-process (web_server) or as a mini-service?
3. Candidate dedup keys for phrase → goal (normalised phrase vs embedding similarity).
4. Should the six §7 single-step behaviours be retrofitted with candidate status eventually, or accepted as vendor-embedded? (operator decision; not in V1)
5. Turn-id population for rb rows (`evidence_turn_ids`) — wire now or with the adapter?
6. Language handling of candidate values (HU/EN mixed phrases — normalisation policy).
