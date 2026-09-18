# Next Architecture Workplan — Learning + Localisation

Status: audit + planning phase COMPLETE (2026-09-18); runtime implementation NOT started
Baseline: VoiceMemAgent v0.10.5 (+ TTS code-switching phase 2)
Full evidence: audit/VoiceMEM_arch_prep_v0105/REPORT.md

## Workstream A — VoiceMem ↔ MAXIM Learning

Completed by the 2026-09-18 audit (docs/LEARNING_ENGINE_INTEGRATION_PLAN.md):

1. Capability matrix for all 19 memory structures (§4) — MAXIM role assigned to each.
2. LearningEvent contract derived from what exists (§5) — the minimal correct contract, the field-by-field gap table, the emission point proposal.
3. Safe vs forbidden candidate sources per V1 domain (§6).
4. Risk register of six existing single-step learning behaviours (§7).
5. V1-V5 phase plan with exact integration points (§8, §12).
6. Reusable read-only component list (§13).
7. Six unresolved architectural decisions (§15) — these need operator/architecture decisions BEFORE runtime learning.

Remaining for V1 (the behaviour-neutral emission layer): see the Implementation Sequence below.

## Workstream B — British English / Asian-Script Localisation

Completed by the 2026-09-18 audit + scanner:

1. Complete deterministic inventory: 11,576 occurrences classified A-F (docs/BRITISH_ENGLISH_AND_LOCALISATION.md §3).
2. Category A (product-facing) is EMPTY — the five labels are localised at both UI payload sites, excluded from TTS and the app-side LLM prompt.
3. Category F compatibility identifier set documented (§4).
4. Deterministic regression scanner implemented: tools/asian_script_scan.py + tools/asian_script_baseline.json + tests/unit/test_asian_script_guard.py (§7).

Remaining (implementation phase, each small and isolated):

5. Translate the two category-B comments (app/web_server.py:537, app/emotion.py:229).
6. Harden the `raw` payload field (strip server-side or mark debug-only).
7. Harden `localise_slot()` (warn + English fallback for unmapped non-ASCII slot values).

## Implementation Sequence (Proposed)

Step 0 (done): architecture audit + this documentation + the scanner.
Step 1: the two category-B comment translations + the two Workstream-B hardening items (one commit, full gate, no version bump needed beyond the next scheduled release).
Step 2: LearningEvent emission layer — session_id/turn_id wiring (web path), `app/learning_adapter.py` read-only emitter, append-only JSONL event log, `tests/unit/test_learning_events.py`. Behaviour-neutral: zero change to memory semantics; the gate must show identical memory outputs with the adapter disabled.
Step 3: Learning Engine side (MAXIM repo, out of scope here): candidate model, validation, proposal, approval UI, Knowledge Bundle mutation, versioning.
Step 4: Learning V1 runtime enablement (phrase → goal, phrase → alias, correction) behind an explicit switch, default OFF.
Step 5: V2+ per docs/LEARNING_ENGINE_INTEGRATION_PLAN.md §8.

Every implementation step follows the Golden Rule: full test gate, zero new regressions, recovery package, SHA256, worklog, mirror synchronisation, verification.

## Proposed Tests

- Existing (new, shipped with this phase): tests/unit/test_asian_script_guard.py (scanner gate).
- Step 1: extend test_british_english.py scope if needed; no new suite.
- Step 2: tests/unit/test_learning_events.py — event schema validity, session/turn provenance propagation, adapter read-only guarantee (no write to any store), event-log append-only guarantee, disabled-adapter behavioural identity (memory outputs byte-identical).
- Step 3+: defined in the MAXIM repo; integration contract tests against the LearningEvent JSONL.

## Files Requiring Modification (Exact, When Implementation Begins)

Step 1: app/web_server.py (comment + `raw` field + `localise_slot` fallback), app/emotion.py (comment).
Step 2: app/web_server.py (session_id/turn_id wiring into ingest), app/background_memory.py (post-persist hook), new app/learning_adapter.py, new data/learning_events.jsonl (runtime artifact, gitignored), new tests/unit/test_learning_events.py.
Explicitly NOT modified: vendor/**, app/tts_supertonic.py, ASR/VAD/LLM paths, config.yaml llama baseline (ngl 16 / ctx 16000 / parallel 1), Home Assistant execution architecture, the web wire format.

## Unresolved Architectural Decisions

The six decisions in docs/LEARNING_ENGINE_INTEGRATION_PLAN.md §15 (event-log location, adapter process model, dedup keys, the six single-step behaviours, turn-id wiring timing, candidate language normalisation).

## Explicit Do-Not-Change List

- No second persistent learning database.
- No Learning Engine write access from VoiceMem; no Knowledge Bundle bypass.
- No autonomous persistent learning; no silent learned-knowledge writes.
- No LLM/ASR/TTS behaviour changes; no Supertonic engine changes; no llama baseline changes.
- No Home Assistant execution architecture changes.
- No speculative retrieval.
- No VoiceMem semantic changes merely to fit the Learning Engine.
- No version bump for documentation-only changes.

## Status

- Workstream A: **NOT READY** for runtime learning (the Learning Engine itself does not exist yet; unresolved decisions §15). The behaviour-neutral emission layer (Step 2) is READY to implement on the VoiceMem side.
- Workstream B: **READY** for the queued items (Step 1: two comment translations + two hardening items; full-gate rule applies).
- Release rule: no production release for documentation-only changes; implementation follows the Golden Rule.
