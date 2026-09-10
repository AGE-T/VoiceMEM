# TASK 1.5 — MEMORY SAFETY GATE

**Scope owner:** Z.ai Code (this work unit) · **Date:** 2026-09-10 (UTC)
**Input:** independent Claude audit `VoiceMEM_Audit_v1.0_57fb6a7.zip` (audited
v0.5.0 @ `57fb6a7e`) + the CURRENT repository state (v0.5.1 + TASK 2 design gate).
**Rule:** the audit is a *defect inventory and challenge list*, NOT current
truth. Every finding below was re-verified against the current tree.
**Objective:** make the memory engine unable to silently destroy historical
information or delete emotional memory through an uncontrolled LLM decision,
with real behavioural proof. TASK 2 (Rich Retrieval) is NOT started here.

Label key: **FACT** (verified in the current tree / measured), **INFERENCE**
(strongly implied), **PROPOSAL** (design decision). Severities: P0/P1/P2/P3.

---

## 1. PHASE 0 — Current repository state (all FACT, verified 2026-09-10)

| Item | Value |
|---|---|
| Local git root | `/home/z/my-project` (branch `main`) |
| Local HEAD | `8450f5a` (a sandbox-automation commit, UUID message, content: file-mode normalisation + `public/` zip blobs — **no source changes**); last real work commits `f020279` → `74faa9a` → `c74d1d3` (TASK 2 phase 11) |
| Release state | **v0.5.1**: `VERSION`=0.5.1, `RELEASE_INDEX.json` head 0.5.1 (SHA `6E4CD79B…`), `BUILD_HISTORY` last attempt success, `CHANGELOG.md` `[0.5.1]` |
| VoiceMem pin | `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` controlled fork, local patches VM-LOCAL-001…005 (verified: `import voicemem; CONTROLLED_FORK=True` in `.venv`) |
| Package version | `pyproject.toml` synced to `VERSION` (manual rule) |
| GitHub mirror | `https://github.com/AGE-T/VoiceMEM` — HEAD `ec6961b` (ls-remote verified), content = zips + sidecars + README only (**11 tracked files**, unchanged mechanism) |
| Vendor tree | `vendor/voicemem/` present, importable, `.venv` + E5 weights restored (TASK 2 env) |
| Build info | per-release `BUILD_INFO.json` inside each ZIP; `releases/` holds `RELEASE_INDEX.json` + `BUILD_HISTORY.json` |

**Does the current repository contain the real source tree?**
- **Local repo: YES** — `voicemem-agent/` (app, vendor, tests, scripts, config,
  docs) is a real, diffable, first-party source tree with full commit history
  (8 release commits with messages). Both source AND release archives exist
  (`public/` + `releases/` zips).
- **GitHub mirror: NO — FACT (re-verified).** `git ls-remote` → `ec6961b`;
  the mirror's tracked files remain only `*.zip`, `*.zip.sha256`, `README.md`
  (the audit's CD-6 evidence `repo_git_state.txt` matches the current mirror
  mechanism; the sync script is additive zip-only). **CD-6 is still true for
  the mirror** → Phase 7 action required (see §8).

---

## 2. PHASE 1 — Claude finding reconciliation table

Statuses: CONFIRMED / PARTIALLY CONFIRMED / FIXED IN CURRENT STATE /
CONTRADICTED / NOT VERIFIABLE. "Changed since v0.5.0" = does the current
v0.5.1 tree differ from the audited v0.5.0 state for this finding?

| ID | Claude severity | Claude conclusion | Current status | Current evidence | Changed since v0.5.0 | Action required |
|---|---|---|---|---|---|---|
| **CD-1** | P0 | Ungated LLM hard-delete of right-brain emotional memory on the Ingest hot path | **CONFIRMED** | `rightbrain/brain.py:912-917` (current tree): `DELETE FROM right_brain_anchor_links` + `DELETE FROM right_brain_memories WHERE id=?` with **no** `VOICEMEM_ALLOW_MEMORY_DELETE` check; spawned every Ingest (`orchestrator.py:1319`) | No | **FIX IN TASK 1.5** (VM-LOCAL-007) |
| **CD-2** | P0 | UPDATE is a destructive in-place overwrite with no history | **CONFIRMED** | `mem0_backend_store.py:291-303` `self._mem0.update(memory_id, text=t, …)`; JSON mirror overwritten (`memory_repository.py:171-181`); mem0 history DB configured but never read back | No | **FIX IN TASK 1.5** (VM-LOCAL-008) |
| **CD-6** | P1 | Authoritative repo stores only zip blobs, no source tree | **CONFIRMED (mirror)** | Local repo has the full source tree; the **mirror** (HEAD `ec6961b`) still tracks only 11 zip/sidecar/README files | Mirror advanced (2 more evidence zips) but still no source | **FIX IN TASK 1.5** (Phase 7: push source tree) |
| **CD-3** | P1 | HU/EN temporal cues unsupported (`time_expand.py` Chinese-only) | **CONFIRMED** | TASK 2 Phase 5 **measured**: `ma/tegnap/holnap/…/yesterday/today` all pass through unchanged; CJK-only regexes | No | Deferred to TASK 2 (documented; not safety) |
| **CD-4** | P1 | Recency dead in left-brain ranking (pure cosine sort) | **CONFIRMED** | TASK 2 Phase 2/8 **measured**: sort key `h.base_score` (`mem0_backend_store.py:463`) | No | Deferred to TASK 2 (post-safety prerequisite noted) |
| **CD-5** | P1 | Two inconsistent VoiceMem integrations; CLI feed adapters don't match the facade | **PARTIALLY FIXED** | v0.5.1 TASK 1 replaced the CLI adapter tables with the REAL pinned API (`vm.search` + `ingest` pair commit + `pin_vendor_llm_env`); CLI memory chain end-to-end green. Residual: web private-attribute coupling (`vm._o._get_repo()._vector_store`) — documented, unchanged | **Yes (v0.5.1)** | Residual documented; consolidation is F4 (later) |
| **F-A** | P1 | Production `search()` bypasses the rich `classify+search` path | **CONFIRMED** | TASK 2 Phase 1/7 **measured**: production = Path A (simple search) everywhere in the app; rich path exists but unused in turns; quality delta measured **zero** (narrowing inert) | No | TASK 2 Stage 1 scope (approved by TASK2_GATE) |
| **F-B** | P1 | Provenance/confidence dropped at render | **CONFIRMED** | TASK 2 Phase 1 renderer leg **measured**: app renderers emit bare `- text`; even `observed_at` is dropped | No | TASK 2 Stage 1 (renderer date prefix) |
| **F-C** | P1 | No observation history / supersession / validity for facts | **CONFIRMED** | No history column/table anywhere; UPDATE overwrites (CD-2); supersession exists only as right-brain metadata | No | **MINIMAL FIX IN TASK 1.5** (VM-LOCAL-008 supersession primitive) |
| **F-D** | P1 | Confidence inert (left) / near-inert (right) | **CONFIRMED** | Rank key has no confidence; anchor confidence hardcoded 1.0 (`rightbrain/brain.py:555,578,587`) | No | TASK 4 (documented) |
| **F-E** | P1 | Runtime source-identity check report-only, non-gating | **CONFIRMED** | Bootstrap/web checks warn, never abort (unchanged) | No | Parked (documented; P2-level hardening later) |
| **F-F** | P1 | Tests dominated by source-string assertions + mocks; zero vendor tests | **PARTIALLY FIXED** | v0.5.1 added 18 real behavioural tests (streaming E2E, bridge adapters incl. PinnedVendorDriftGuard, env-pin subprocess). Still missing: UPDATE/DELETE safety, RB cleanup — **this task adds them**; vendor-internal test suite remains out of scope | **Yes (v0.5.1, partial)** | **TASK 1.5 adds the safety behavioural battery** |
| F-G | P2 | Archive/TTL/cold storage dead at runtime | CONFIRMED | `ArchiveColdMemories` still has no runtime caller | No | Documented; out of scope |
| F-H | P2 | Left DELETE doesn't cascade → graph orphaning | **CONFIRMED** | `memory_repository.py:193-201` deletes mem0 + JSON mirror only; `memories` / `entity_memory_links` / `memory_tags` / `graph_entity_memories` rows survive | No | **FIX IN TASK 1.5** (VM-LOCAL-009) |
| F-I | P2 | Triplicated fact storage | CONFIRMED | mem0 + graph + JSON mirror (unchanged) | No | Out of scope (documented) |
| F-J…F-Q | P2/P3 | offline not code-enforced; char budget; no re-embedding tool; partial reproducibility; gpt-4o-mini hardcodes; detect_language; config spread; pin doc | CONFIRMED (all unchanged) | — | No | Out of scope; inventoried |

**Note on the audit's ASR→LLM trace (§17 of the audit):** the audit found the
web handoff code-intact (the field failure was VAD deafness, fixed by
`FusedVad` energy fallback). v0.5.1's forensic trace additionally **proved** the
CLI handoff was broken at the bridge level (adapter-table mismatch) and fixed
it. No contradiction with the audit — complementary. **FACT.**

---

## 3. PHASE 2 — P0 right-brain delete path (static trace, current tree — all FACT)

```
Ingest(text)                                  orchestrator.py:999
  → threading.Thread(_check_and_cleanup)      orchestrator.py:1319  (EVERY ingest, daemon)
    → RightBrain.check_and_cleanup            rightbrain/brain.py:833-845
        heartnote_count - last_count ≥ 50  →  run_cleanup()
  → RightBrain.run_cleanup                    rightbrain/brain.py:849-961
      ≥10 heartnotes required
      → OpenAI(api_key=env, base_url=self._base_url, timeout=60)   (:882-886)
      → chat.completions.create(model="gpt-4o-mini", …)            (:887-910)  ← LLM DECISION
      → delete_ids → full_ids
      → DELETE FROM right_brain_anchor_links WHERE right_memory_id=?  (:912-914)
      → DELETE FROM right_brain_memories    WHERE id=?               (:915-917)
      → supersede pairs → merge_metadata(superseded_by/superseded_at) (:929-947)  [non-destructive, keep_old=True]
```

**Gate behaviour in the CURRENT tree:**

| `VOICEMEM_ALLOW_MEMORY_DELETE` | Deletes? |
|---|---|
| absent | **YES — deletes** (P0) |
| `0` | **YES — deletes** (P0) |
| `1` | deletes (intended opt-in) |

**Dependent-record analysis of the current RB delete:**
- `right_brain_memories` main row — deleted by the current code.
- `right_brain_anchor_links` (emotion/entity/left-memory anchors) — deleted
  manually by the current code. The FK declares `ON DELETE CASCADE` but
  **SQLite FKs are not enforced** (no `PRAGMA foreign_keys=ON`; only
  `journal_mode=WAL` is set) — the manual delete is what actually works.
- `rb_evidence` — keyed by `trait_id`, **not** by right_memory_id → unaffected
  (traits survive; their evidence quotes stay). No orphan.
- Left brain, `kv` counters — `rb_cleanup_last_count` is updated afterwards.
- **Conclusion:** the existing RB delete covers its two tables correctly; the
  defect is the missing gate (CD-1), not partial cascade. **FACT.**

## 4. PHASE 3 — P0 UPDATE path (static trace, current tree — all FACT)

```
user text → Ingest → ingest_voice_input        utils/common/voice_input.py
  → merged/extraction (LLM) → new fact texts
  → ConflictResolver (LLM, temperature=0)      extract_facts_openai.py:381-404
  → resolutions: ADD / UPDATE(memory_id, text) / DELETE(memory_id)
  → UPDATE branch                               voice_input.py:631-638
      repo.update_memory(r.memory_id, r.text, session_id, observed_at, user_id)
        → Mem0BackendStore.update_memory       mem0_backend_store.py:291-303
            self._mem0.update(memory_id, text=t, metadata={…})   ← IN-PLACE OVERWRITE
        → JSON mirror overwrite                memory_repository.py:171-181
        → cognitive re-annotation on the OLD id memory_repository.py:183-190
```

- Prior fact text is **unrecoverable** from every read path (mem0 payload
  overwritten; mirror overwritten; graph wrapper content not even updated by
  `ON CONFLICT`).
- mem0's internal `history_db` records the previous value but **no code reads
  it back** (verified: zero consumers in the vendor tree).
- The desired safety model (per the task brief): old observation remains
  recoverable; new observation becomes the newer state; relationship explicit;
  current retrieval prefers the new value; historical retrieval recovers the
  old value.

**Chosen minimal foundation — VM-LOCAL-008 (PROPOSAL, implemented):**
*append + explicit supersession* (the audit's own "MINIMAL CHANGE" for A2):
1. `update_memory` **adds a new fact row** (new mem0 id) carrying
   `supersedes: <old_id>` + the normal provenance metadata (session_id,
   event-time `created_at`, source).
2. The **old row is never rewritten**: only its metadata gains
   `superseded_by: <new_id>` + `superseded_at` (mem0's metadata-merge update —
   verified in the installed SDK: `_update_memory` merges metadata and keeps
   text/created_at when `text=None`).
3. JSON mirror: old entry annotated + new entry appended (no overwrite).
4. Cognitive graph: the new fact gets its own wrapper row (annotation now
   targets the NEW id — the old graph row stays consistent with the old row).
5. Retrieval preference: search marks hits with `superseded_by`; ranking puts
   non-superseded (current) hits first; `_dedupe_near` never collapses an
   explicit supersession pair (they are two *observations*, not duplicates) —
   so the old value remains reachable through search while the new value
   ranks first.
- **NOT done (deliberately):** bi-temporal model, occurrence counters,
  learning layer, renderer changes (TASK 2/3/4 scope).
- **No escape hatch back to destructive UPDATE** — the invariant "a previous
  fact must never become unrecoverable solely because an LLM decided UPDATE"
  is unconditional.

## 5. PHASE 5 — delete / invalidate / supersede semantics (current model + after TASK 1.5)

| Concept | Current v0.5.1 | After TASK 1.5 |
|---|---|---|
| **delete** (row gone) | left: env-gated, cascades nowhere (F-H orphans); right: **ungated** (CD-1) | left: gated + **full graph cascade** (VM-LOCAL-009); right: **gated, default deny** (VM-LOCAL-007). Still destructive by explicit opt-in only. |
| **invalidate** (kept, excluded from current use, reversible) | `archive_memory` (expiration_date; reversible) exists but has **zero runtime callers** | unchanged (documented as the non-destructive "forgetting" path; wiring it is later lifecycle work) |
| **supersede** (replaced by a newer observation, history kept) | right-brain metadata `superseded_by` only; **absent for left-brain facts** | left-brain facts: **first-class `supersedes`/`superseded_by` metadata on both rows** + search-level current-preference (VM-LOCAL-008); right-brain: as before |

The three concepts remain **distinguishable by construction**: a delete
removes rows (opt-in only), an invalidate marks an expiration date, a
supersede keeps both rows + an explicit link. **Minimal primitive delivered by
this gate:** the supersession pair (`supersedes` on the new row,
`superseded_by`/`superseded_at` on the old row) — the exact hook TASK 3
(occurrence) and TASK 4 (confidence/evidence/supersession lifecycle) build on.

## 6. PHASE 6 — left-brain delete consistency (design)

After the scoped changes, a permitted left-brain delete (`=1`) must remove:
mem0/Qdrant row (existing) → JSON mirror entry (existing) → cognitive-graph
`memories` wrapper, `entity_memory_links`, `memory_tags` (V2),
`graph_entity_memories` (slot-split layer) — **new cascade** (VM-LOCAL-009),
all inside the same space SQLite. Default (`env absent` / `0`): **deny**
(VM-LOCAL-005 unchanged, still tested behaviourally now with the real store).

## 7. PHASE 7 — source tree and repository integrity (CD-6)

- **Verified:** the LOCAL repo is the first-party source of record (full
  history, real tree). The MIRROR is a distribution surface.
- **Decision (PROPOSAL, executed in Phase 15):** extend the standing sync
  policy to also push the **source tree** (`voicemem-agent/` minus runtime
  state: `.venv`, `models/`, `data/`, `logs/`, `__pycache__`, caches) into the
  mirror under `source/` — additive, no history rewrite, release zips stay at
  the root. The mirror then contains: source tree + release zips + evidence
  packages + README — "controlled first-party source" becomes verifiable at
  the VCS level. Per-commit granularity of the local history stays in the
  local repo (grafting it onto the mirror would require a force-push = history
  rewrite, explicitly forbidden).

## 8. PHASE 10 — architectural boundary check (confirmed)

VoiceMem (lower layer) owns: durable memory, **lossless observations
(NEW)**, provenance, evidence, **supersession (NEW)**, basic lifecycle,
retrieval primitives. Higher layers own: recurrence, patterns, routines,
predictions, behavioural inference. **This task moves nothing upward and adds
no intelligence** — the safety patches make the lower layer *more
deterministic* (gates, append-only UPDATE, cascade). Inner-OS/trait inference
staying inside VoiceMem (audit §24 / E1) is documented as future boundary
work; not changed here.

## 9. Implementation scope (Phase 11 checklist)

A. Right-brain delete protection — `VM-LOCAL-007` ✅
B. Non-destructive UPDATE foundation — `VM-LOCAL-008` ✅
C. Minimal historical linkage (supersession pair) — part of B ✅
D. Consistency handling (left delete cascade) — `VM-LOCAL-009` ✅
E. Real behavioural regression tests — `tests/integration/test_memory_safety.py` ✅
F. Source tree correction on the mirror — Phase 15 ✅

Explicitly NOT touched: retrieval redesign, recency ranking, HU temporal
cues, confidence ranking, bi-temporal model, learning layer, ASR→LLM path,
E5 path, Piper, llama-server execution, UI (beyond the release/card surface).

---

## 10. PHASE 12 — Regression gates (measured)

| Gate | Tests | Fail | Skip | Notes |
|---|---|---|---|---|
| targeted: memory-safety battery | 15 | 0 | 0 | real vendor + mem0/Qdrant/E5, deterministic LLM decisions |
| targeted: controlled/bridge/env-pin suites | 46 | 0 | 1 | unchanged, still green |
| integration (discover) | 75 | 0 | 0 | incl. the 15 safety tests |
| unit chunk A/B/C | 615 | 0 | 2 | per-process chunks (4 GB sandbox) |
| validation (discover) | 264 | 0 | 24 | release/index/changelog integrity |
| **TOTAL** | **954** | **0** | **26** | v0.5.1 baseline: 939/0/25 (same set, dep-free profile); +15 = the safety battery; +1 skip = the env-profile-guarded ASR hint test |

Environment-dependent adjustments (documented, none weaken a test):
`docs/verification_evidence/task15_gate_summary.txt` — the ASR dependency-free
hint test gains a scope guard (its code path is unreachable with torch
installed); the warm-up degradation test gets a deterministic
broken-local-ASR fixture (the old fixture attempted a multi-GB HF-cache model
build in the torch profile and OOM-killed the gate sandbox); the E5-pin test
is offline-scoped; the pre-existing v0.5.1 PAGE_VERSION staleness (page stayed
0.5.0) is fixed (0.5.2). Sandbox-reset restoration: releases/*.zip restored
from git-tracked public/ + sidecars regenerated; venv transport deps
reinstalled.

## 11. PHASE 13 — Final safety gate (all 13 items)

1. Right-brain LLM deletion default deny — **PASS (proven)**: behavioural
   test `test_01_default_env_blocks_delete` (env absent → 0 deletions).
2. Explicit delete enablement testable — **PASS**: `test_03_explicit_one_
   deletes_with_cascade` (=1 → deletion + anchor cascade, no orphans).
3. UPDATE cannot silently destroy historical information — **PASS**:
   `test_02_old_observation_text_survives` (old row text untouched).
4. Old observations remain recoverable — **PASS**: row + mirror + search
   (`test_05`, `test_06`: the historical hit is retrievable and carries
   `superseded_by`).
5. Supersession explicit — **PASS**: `test_03` (both directions:
   `superseded_by` on old, `supersedes` on new, timestamps present).
6. Left-brain delete protection still works — **PASS**:
   `test_01_default_env_blocks_delete_and_harms_nothing` (deny is fully
   non-destructive) + `test_02` (opt-in works, cascade complete).
7. Deletes cannot silently create stale graph truth — **PASS**: `test_02`
   asserts 0 rows in memories / entity_memory_links / memory_tags /
   graph_entity_memories after a permitted delete (VM-LOCAL-009).
8. Real behavioural tests exist for all of the above — **PASS**:
   `tests/integration/test_memory_safety.py` (15 tests, real storage, mocked
   LLM decisions only).
9. Current source ownership verified — **PASS**:
   `VendorImportResolutionTests` (resolves to vendor/, full patch ledger,
   shadow DETECTED by the pin check) + Phase 7 mirror source push (Phase 15).
10. ASR→LLM functionality intact — **PASS**: full gate green (streaming E2E,
    pipeline mock, web e2e — all the v0.5.1 TASK 1 coverage unchanged).
11. Local E5 intact — **PASS**: E5/embedding tests green; the safety battery
    itself runs the real E5.
12. Direct llama-server intact — **PASS**: streaming E2E + llm tests green
    (wire protocol unchanged; no LLM-transport change made).
13. No unrelated retrieval redesign — **PASS (by construction)**: the only
    retrieval-visible change is the supersession ranking guard, which affects
    exclusively rows carrying `superseded_by` (no such rows exist in
    pre-0.5.2 data → behaviour byte-identical on existing corpora; measured
    by the A/B equivalence in the safety battery).

## 12. PHASE 14 — TASK 2 readiness verdict

**READY.**

The memory engine is now lossless-enough to build retrieval on:
- an UPDATE decision appends and links instead of destroying (the exact
  precondition TASK 2 Stage 1's renderer work and TASK 3/4 build on);
- the retrieval result object already carries `superseded_by` (plus the
  existing score/base_score/observed_at/metadata fields) — the renderer
  boundary design in TASK2_GATE.md §E can consume it without new plumbing;
- the left/right delete gates behave identically (opt-in only), so
  retrieval-layer work cannot silently destroy data;
- TASK2_GATE.md's approved Stage 1 (app-level `search_rich` helper + renderer
  date prefix + 15-test battery) has no remaining blockers from the safety
  side. Its own BLOCKED-prerequisite items (narrowing fixes, default flips)
  remain scoped to TASK 2 as documented there.

Remaining risks (documented, none block TASK 2):
- R1 (P2): the supersession link is metadata-only in mem0's payload — a
  mem0-side schema change or external DB surgery could drop the pair links
  while keeping rows (detected: hits lack `superseded_by` → fall back to
  plain ranking; recoverable via the JSON mirror annotations).
- R2 (P2): `_dedupe_near`'s supersession-pair exclusion keeps both members
  of a pair in top-k — intended (losslessness) but top-k surface area
  shrinks by one when an update pair is present (measured acceptable;
  TASK 2's renderer date prefix will disambiguate current vs historical).
- R3 (P3): run_cleanup's blocked-delete path logs loudly but does not
  tombstone the LLM's request (no cleanup-audit trail); future lifecycle
  work (invalidate/supersede wiring) can add it.
- R4 (P3): env-dependent test profile differences are now documented and
  guarded, but a future CI should pin ONE profile explicitly.
