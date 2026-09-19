# VoiceMem v0.10.5 Architecture Preparation — Final Report

- Date: 2026-09-18
- Baseline: VoiceMemAgent v0.10.5 + TTS code-switching phase 2 (tree at commit 8c8bc93)
- Inputs: VoiceMemAgent_v0.10.5_learning_prep.zip (427 files — the v0.10.5 release tree carrying the three draft architecture contracts, reviewed and updated rather than replaced) + the live tree
- Deliverables: this report; docs/LEARNING_ENGINE_INTEGRATION_PLAN.md; docs/BRITISH_ENGLISH_AND_LOCALISATION.md; docs/NEXT_ARCHITECTURE_WORKPLAN.md; tools/asian_script_scan.py; tools/asian_script_baseline.json; tests/unit/test_asian_script_guard.py
- Phase rule honoured: audit + planning only — ZERO production code changes, no VERSION bump, no release.

## 1. Current capabilities

VoiceMem today: a Hungarian-first voice agent with a two-brain vendored memory library (left brain: extracted facts in Qdrant + SQLite mirrors with stance, temporal validity, supersession chains, occurrence counts; right brain: heartnotes, traits in five slots with evidence rows, response_experience (write disabled since v0.10.3), situation patterns), a cognitive graph (entities, aliases, relations, slot profiles), routine detection (scene observations → threshold-crossing synthetic facts), deterministic stance/temporal classifiers, emotion fusion with an append-only JSONL log, session/conversation history (ephemeral web window + persistent trackers), background memory ingestion gated behind an idle window (BackgroundMemoryGate), and the v0.10.5 TTS code-switching span routing (quoted + unquoted phrase runs). Retrieval is tier-ranked with provenance rendering; the LLM/ASR/TTS stack is local (llama baseline ngl 16 / ctx 16000 / parallel 1).

## 2. Capability matrix

See docs/LEARNING_ENGINE_INTEGRATION_PLAN.md §4 for the full 19-structure matrix (structure → location → persistence → MAXIM role). Summary of the classification outcome:

- Canonical persistent knowledge (already durable): facts, traits, preferences (via carriers), cognitive graph, entities/aliases, response_experience (disabled).
- Observation (raw capture): episodic memories/heartnotes, emotion, session/conversation history (the primary candidate substrate).
- Evidence: rb_evidence rows, evidence_memory_ids chains, stance, occurrence counts, confidence, supersession chains.
- Candidate sources: routines (the cleanest existing pattern), corrections, goals-tagged facts, entity aliases.
- Historical/contextual: temporal fields, row_status.
- Not suitable for learning: playback/barge ledgers, VAD peaks, DemoMemoryLayer.

## 3. Existing reusable components

docs/LEARNING_ENGINE_INTEGRATION_PLAN.md §13 — retrieval with provenance (Mem0BackendStore.search, memory_history()), deterministic classifiers (temporal.py, stance.py), trait reads + typed contract (retrieval_contract.py), session/turn plumbing (SessionTracker, format_turn_id, BackgroundMemoryGate), correction cue tables, right-brain/graph read getters, the append-only JSONL sink pattern (EmotionMemory.record_turn).

## 4. Architecture gaps

1. No LearningEvent semantics: no event id, no event type, no candidate concept anywhere.
2. session_id/turn_id provenance: the web path never passes session_id into ingest; evidence_turn_ids exists but is never populated.
3. Two disjoint `source` vocabularies (fact metadata vs rb hit source).
4. No Learning Engine, candidate model, proposal/approval flow, or Knowledge Bundle writer — they exist only in the MAXIM architecture, not in VoiceMem.
5. Six existing single-step learning behaviours (below) have no candidate/approval stage.
6. British-English residual: 2 Chinese comments in product code + 2 advisory leak channels.

## 5. Learning Event proposal

docs/LEARNING_ENGINE_INTEGRATION_PLAN.md §5 — the minimal contract `voicemem.learning_event/1` (16 fields) + the field-by-field gap table + four shaping insights: (1) the fact layer is already a near-complete event sink — event-id/type/candidate semantics ride the additive-metadata channel exactly as stance did in v0.10; (2) session/turn wiring is the single provenance gap; (3) unify the two source vocabularies; (4) confidence is heterogeneous by design — treat per candidate type. Emission point: a post-persist hook on BackgroundMemoryGate; the adapter is read-only and never writes back.

## 6. VoiceMem → Learning Engine data flow

```text
user turn → web pipeline → reply
   → BackgroundMemoryGate (idle window)
   → vm.ingest → persistence (facts/traits/heartnotes/graph)
   → [NEW] post-persist hook → Learning Adapter (read-only)
   → Learning Events (append-only JSONL)
   → (future, MAXIM side) Learning Engine: candidate → validation
     → proposal → user decision → mutation → Knowledge Bundle → version
```

VoiceMem semantics do not change; the adapter listens after persistence and emits; nothing flows back.

## 7. Goal Catalog integration

VoiceMem facts tagged slot `goals` + anchor types are EVIDENCE for goal learning, never a runtime goal store. The Goal Catalog (MAXIM) is mutated only by the Learning Engine after approval. V1 phrase→goal candidates are mined from `WebSession._history` (user_text, reply) pairs + goals-tagged facts (provenance-rich); slot_profiles/slot_summaries/dynamic_slots are explicitly FORBIDDEN as sources (LLM-aggregated, lossy provenance, or emergent taxonomy).

## 8. Missing Information Matrix integration

The MIM lives on the MAXIM side. It may query VoiceMem read-only (which evidence exists for a subject) to decide what to ask next; it never writes to VoiceMem and never becomes a retrieval trigger inside VoiceMem (no speculative retrieval).

## 9. Knowledge Bundle integration

The Knowledge Bundle remains the sole canonical persistent-knowledge boundary with the Learning Engine as its only writer. Learned records enter the same canonical Knowledge Base (Phrase → Goal philosophy, one KB) carrying `source: "learned"` provenance so the runtime distinguishes manual and learned knowledge without separate runtime stores. VoiceMem never bypasses it; there is no second learning database.

## 10. British English localisation findings

Deterministic scan (570 files, 11,576 Asian-script runs): category A (product-facing) EMPTY — the five labels (喜好与厌恶/表达风格/思维模式/应对方式/情绪) are localised at both UI payload sites, excluded from TTS and the app-side LLM prompt. Category F (27 runs, preserved): the five slot enums, eight bilingual emotion2vec tokens, advisory strip markers, fullwidth regex literals. Category B: 100 vendor-format fixtures (legitimate) + 2 queued Chinese comments (app/web_server.py:537, app/emotion.py:229). C 11,332 vendor/model; D 110 forensic; E 5 runtime. Residual advisory channels: the `raw` payload field (debug-only), `localise_slot()` unknown-slot pass-through, vendor stdout. Full details: docs/BRITISH_ENGLISH_AND_LOCALISATION.md §3-§5.

## 11. Exact files requiring modification (when implementation begins)

Step 1 (localisation): app/web_server.py (comment translation + `raw` field hardening + `localise_slot` English fallback), app/emotion.py (comment translation).
Step 2 (LearningEvent emission): app/web_server.py (session_id/turn_id wiring into ingest), app/background_memory.py (post-persist hook), new app/learning_adapter.py, new tests/unit/test_learning_events.py, runtime artifact data/learning_events.jsonl (gitignored).
Explicitly NOT modified: vendor/**, app/tts_supertonic.py, ASR/VAD/LLM paths, config llama baseline, Home Assistant execution architecture, the web wire format.

## 12. Proposed tests

Shipped with this phase: tests/unit/test_asian_script_guard.py (4 tests — zone cleanliness, canonical labels, baseline ratchet, planted-violation detection).
Step 2: tests/unit/test_learning_events.py — event schema validity, session/turn provenance propagation, adapter read-only guarantee (no store writes), event-log append-only guarantee, disabled-adapter behavioural identity (memory outputs identical).
Step 3+ (MAXIM side): contract tests against the LearningEvent JSONL.

## 13. Proposed implementation sequence

Step 0 (DONE — this phase): audit + documentation + scanner.
Step 1: the two comment translations + two hardening items (one commit, full gate).
Step 2: behaviour-neutral LearningEvent emission layer (wiring + adapter + JSONL + tests; gate must show identical memory outputs with the adapter disabled).
Step 3: Learning Engine side (MAXIM repo, out of scope here).
Step 4: Learning V1 runtime enablement (phrase→goal, phrase→alias, correction) behind an explicit switch, default OFF.
Step 5: V2+ per the phases in docs/LEARNING_ENGINE_INTEGRATION_PLAN.md §8.
Every step follows the Golden Rule: full test gate, zero new regressions, recovery package, SHA256, worklog, mirror synchronisation, verification.

## 14. Unresolved architectural decisions

docs/LEARNING_ENGINE_INTEGRATION_PLAN.md §15 — (1) event-log location before the Learning Engine exists (proposal: append-only JSONL); (2) adapter process model (in-process vs mini-service); (3) candidate dedup keys; (4) retrofit or accept the six single-step learning behaviours; (5) turn-id population timing; (6) candidate value language normalisation.

## 15. Explicit list of things that should NOT be changed

- No second persistent learning database; no Learning Engine write access from VoiceMem; no Knowledge Bundle bypass.
- No autonomous persistent learning; no silent learned-knowledge writes.
- No LLM/ASR/TTS behaviour changes; no Supertonic engine changes; no llama baseline changes (ngl 16 / ctx 16000 / parallel 1).
- No Home Assistant execution architecture changes; no speculative retrieval.
- No VoiceMem semantic changes merely to fit the Learning Engine.
- No version bump or release for documentation-only changes.
- Category-F compatibility identifiers stay verbatim (persistence/tokeniser/vendor-format contracts).
- Historical forensic evidence, vendor content and user-generated content are never mutated for localisation.

## Final status

- **Workstream A: NOT READY for runtime learning** (the Learning Engine does not exist yet; the six §14 decisions are open). The behaviour-neutral LearningEvent emission layer (Step 2) is READY to implement on the VoiceMem side.
- **Workstream B: READY for the queued items** (Step 1: two comment translations + two hardening items; full-gate rule applies).
- Gate evidence on the final tree: 1501 tests (unit 1107 / integration 123 / validation 271), 0 new regressions, 4 pinned environment-gap failures identical to the v0.10.5 baseline. audit/VoiceMEM_tts_codeswitch_v0105_unquoted/evidence/gate_run_20260918.txt + the S-scanner worklog record.
