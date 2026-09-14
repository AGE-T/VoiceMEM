# TASK 2 — Rich Retrieval DESIGN GATE (v0.5.1 state)

**Scope**: `vm.search()` simple primitive vs `classify → search(slots, entities)` rich path.
**Rule set**: measurement over assumption; FACT / INFERENCE / PROPOSAL labels on material conclusions.
**Evidence base** (all from THIS session, current repo `cc43859` @ v0.5.1, vendor pin `e8384e0`):
- `docs/verification_evidence/task2_retrieval_baseline.json` (106,969 B) — 16-query × 2-path matrix, controlled 8-fact corpus, real E5 + real mem0/qdrant store, deterministic mock LLM
- `docs/verification_evidence/task2_callgraph.json` — runtime `sys.setprofile` call graphs of both paths
- `docs/verification_evidence/task2_performance.json` — 20-run latency stats (mean/median/p95)
- Harnesses: `scripts/task2_retrieval_baseline.py`, `scripts/task2_callgraph_trace.py` (measurement-only; no implementation was modified)

---

## A. Current baseline (FACT)

| | PATH A — `vm.search(q)` | PATH B — `classify(q)` → `search(q, slots, entities)` |
|---|---|---|
| Used by | CLI bridge (`_retrieve_user_turn`), web voice turn (`MemoryLayer.search`, +`emotion=`), forensic driver | **Nobody in the app.** Vendor's own `Memory.recall/inject`, `PrimeSubgraphFromQuery` |
| Internal mode | `fallback` (pure vector over full library) | `slot-only` |
| Slot filter | skipped (slots=[] → no SQL at all) | SQL executed — but returns ALL memories (see B1) |
| Entity narrowing | skipped | executed (fuzzy match + 1-hop neighbours) — unioned back to full set (see B2) |
| Rank / dedup / rb / heat | identical | identical |
| rank-1 correctness | **13/14** targeted queries | **13/14** (same misses: HU→EN multi-condition "Q3 audit" = rank 2 both) |
| Hits returned | always exactly top_k=5 (no relevance threshold; EMPTY control also returns 5) | same |
| Latency (median, warm, mock LLM) | 58–73 ms | 64–77 ms (classify ≈ 4–8 ms mock) |
| 20-run p95 (real E5/store) | 81–85 ms | 80–90 ms |
| Cross-language HU↔EN | rank 1 both directions (multilingual E5) | same |
| Extra data | `classification` = empty QueryClassification | `classification` = real slots+entities; `related_summaries` populated |
| rb_hits | trait queries return structured rb hits (source, priority, slot_name, trait_id) | same |

First-call cold latency: ~2.5 s (E5 load + store init) — measured once per process; steady state is the table above.

Runtime call graphs (divergence): Path A enters `Orchestrator.Search` with `slots=None` → `SearchCogGraph` skips the slot SQL entirely → `final_ids=∅` → `Rank()` does full-library `repo.search`. Path B first runs `LeftBrain.Classify` → `QuerySlotClassifier.classify` (one LLM call) → then the same `Search` with slots → `store_v2.memory_ids_for_slots_v2` (SQL) → `_search_data_impl` (entity graph walk: `find_entities_by_name_fuzzy` → `memory_ids_for_entities` → `neighbor_entity_ids` → `_pool_mode`) → `time_question_kind` → `get_macro_related_slots`/`get_slot_summaries` (related summaries, Path B only) → same `Rank`. Both converge at `Rank` → `mem0_backend_store.search` (E5 query embed + cosine + lexical/time/date bonuses) → `_dedupe_near` → heat recording → subgraph accounting → `SearchResult` assembly. Right-brain search runs in a `ThreadPoolExecutor` worker concurrent with `Rank` in both paths (anchored on `activated_names`, which Path B can populate via entity activation).

## B. Exact weaknesses (FACT unless noted)

1. **The slot filter is structurally a no-op** (measured + static): `memory_repository_v2._tag_memory_slots_v2` auto-tags EVERY base slot whose cosine ≥ 0.20 to the fact text — with local E5 all seven anchors score ≈0.72–0.77, so every memory carries all 7 `memory_tags` rows; the LLM re-tag (`_llm_tag_memories`) upserts the correct slot at 0.95 but **upsert never deletes the spurious rows**; `memory_ids_for_slots_v2` ignores confidence entirely. ⇒ `slot_mem_ids` = whole corpus for ANY slot.
2. **Entity narrowing is inert in the default pool mode**: `_search_data_impl` returns `entity_mids | slot_mem_ids` (union) unless `_pool_mode()=="strict"` (opt-in env) — with B1 the union is always the whole corpus. Measured `final_candidate_ids=8/8` even with a matching entity ("Q3 audit").
3. **`vm.search()` never classifies** — by construction (`slots or []`), so the rich machinery (slot summaries, entity activation for rb anchoring, narrowing) is unreachable from the primitive the app actually calls.
4. **Renderer information loss (measured)**: vendor `build_memory_context` renders `- [observed_at] text` + rb block; the app's two renderers (bridge `_extract_memory_context`, web `_memory_context`) render bare `- text` — **even `observed_at` is dropped**, plus score, base_score, time_boost, attributed_to, metadata, slot, entities, classification.
5. **No relevance threshold**: every query — including the nonsense control — returns exactly `top_k` nearest neighbours; "no evidence" is only signalled by the low-confidence hint (rb_directive) when the left brain is empty, not by hit scores.
6. **Temporal: HU/EN invisible** (measured): `expand_relative_dates` and `time_question_kind`/`query_dates` are Chinese-only (CJK regexes + CJK word table). `ma/tegnap/holnap/tegnap este/ma reggel/múlt héten/jövő héten/yesterday/today/tomorrow/last week/next week` all pass through unchanged; no widening, no date boost. Temporal facts are retrieved purely by E5 vector similarity (works when the relative wording overlaps the fact text — measured rank 1 — but nothing constrains the answer to the asked day).
7. **Speaker: no-op in text mode** (measured): zero `speaker:<id>` tags exist (they are written by the audio/voiceprint path only); `speaker_filter=` exists on the API but filters nothing.
8. **Production-path inconsistency (PHASE 7 audit)**: web voice turn = Path A(+emotion); CLI = Path A; forensic driver = Path A; `Memory.recall/inject` = Path B; **no test exercises the composition**; the web UI's `_hits_payload` advertises slots/entities that are always empty on real turns, while the DemoMemoryLayer (mock) ALWAYS fabricates a classification — the mock web UI and the real web UI disagree.
9. **(INFERENCE)** The only measured differences Path B buys today: +1 LLM call (cost), real classification data (UI/API surface), related-slot summaries, and entity-activated rb anchoring. Zero hit-quality difference on the controlled corpus.

## C. Proposed API — three designs compared (PHASE 3)

The key question is NOT "how do we replace `vm.search()`" but "how does the primitive stay backward compatible while richer behaviour is exposed safely and explicitly".

| | **D1 — `search(q, mode="rich")`** (vendor kwarg) | **D2 — app-level `search_rich(vm, q)` helper** (composition of the two public primitives) | **D3 — caller-side composition per site** |
|---|---|---|---|
| Backward compat | OK but requires a **vendor patch** (mode param does not exist) — grows the patch ledger | **Perfect** — vendor untouched (pin-safe, UPSTREAM_POLICY-compliant, same philosophy as the v0.5.1 TASK 1 fix) | Perfect |
| Caller ambiguity | **High** — `mode="rich"` vs explicitly passing `slots=`/`entities=` (both already exist as params!) → precedence rules, stringly-typed growth | **None** — two named primitives; the simple one's signature is byte-identical | None at API level |
| Testability | OK | **Best** — one composition point, A/B comparison is a one-liner | Worst — must test each site's hand-rolled copy |
| Future extensibility (TASK 3/4) | mode-string sprawl ("rich", "rich+temporal"…) | helper can grow params without touching the primitive | every enhancement re-implemented per site |
| Hidden behaviour risk | Medium (future default flips; vendor drift) | **Low** — opt-in, additive, default unchanged | Low |
| CLI compatibility | callers edited anyway | callers edited anyway, one shared import | three edits |
| Web compatibility | same | same | three edits |
| Learning layer compat | vendor change ripples into internals | untouched | untouched |
| Performance impact | same +1 classify LLM call when mode set | same, explicit and visible | same |
| Consistency fix (B8) | partial | **by construction** — CLI+web share one implementation | **worsens drift risk** |

**RECOMMENDED: D2.** FACT-basis: the measured quality delta between the paths is currently **zero** (B1/B2 make narrowing inert), so rich retrieval must enter as an *explicit, visible-cost opt-in* — never as a silent default or a mode flag on the primitive. D1 would modify the pinned vendor for a feature whose measured benefit is presently nil; D3 recreates exactly the drift the PHASE 7 audit found. D2 is one ~30-line app-side helper:

```
app/memory_retrieval.py (PROPOSAL — not implemented)
    def search_rich(vm, query, *, top_k=None, emotion=None, speaker_filter=None) -> SearchResult
        # 1. classification = vm.classify(query)          (public primitive)
        # 2. return vm.search(query, slots=classification.slots,
        #                      entities=classification.entities, ...)  (public primitive, unchanged)
        # returns the vendor SearchResult object unchanged — single source of truth
```
CLI bridge and web `MemoryLayer` import it; both keep Path A by default in this gate's scope.

## D. Result model proposal (PHASE 4)

**FACT**: the vendor `SearchResult` is ALREADY a structured result object — `hits: MemorySearchHit(memory_id, text, score, base_score, time_boost, observed_at, attributed_to, metadata)` + `classification` + `related_summaries` + `slot_mem_ids`/`final_candidate_ids` + `search_mode` + `rb_hits: RightBrainHit(content, source, priority, metadata)` + `rb_directive` + `scene_directive`/`current_scene` + `timing`. The loss happens **only at rendering**, not in the result model.

**PROPOSAL**:
- Do NOT build a new app-level result object (duplication, drift, zero measured need).
- `score`, `base_score`, `time_boost`, `timing` stay **internal ranking metadata** — they must NOT reach the LLM context (the user's own constraint: don't expose internal ranking implementation details; they invite the model to rationalise numbers).
- `observed_at` **should** reach the LLM context (dates are user-meaningful provenance, and the vendor's own proven renderer includes them — the app renderers are the outliers).
- `attributed_to`/`metadata`/`slot`/`entities`/`classification` remain available to the API/UI layer (web `_hits_payload` already surfaces a subset) but not the LLM context.

## E. Rendering boundary (PHASE 4)

**FACT (measured)**: three renderers exist — vendor `build_memory_context` (`- [date] text` + framed rb block), bridge `_extract_memory_context` and web `_memory_context` (byte-identical behaviour: bare `- text`, rb cleaned, 1200-char cap — dates dropped).

**PROPOSAL**: keep the app renderers' shape (it is the TASK-1-proven working pattern) and add exactly ONE field to the fact lines: the `[observed_at]` date prefix, matching the vendor's proven format:

```
- [2026-09-09] A felhasználó tegnap este moziba ment Annával.
- The user finished the quarterly report for the Q3 audit last week   (no date → no bracket)
+ rb block (unchanged, already cleaned+framed semantics)
```
The boundary rule: **retrieval result object = vendor SearchResult (structured, complete); LLM context = text + observed_at + rb content; everything else internal.**

## F. Temporal behaviour (PHASE 5, FACT)

Recognition: NONE for HU/EN (B6). Storage: `observed_at` is stored per fact (metadata.time_start + created_at) and rendered by the vendor renderer, absent in the app renderers. Filtering: none at query time (only `date_overlap_bonus` on CJK date literals). Boosting: `time_question_kind` (CJK) gates the widening + `_TIME_WEIGHT` lexical boost — never fires for HU/EN. The system "merely stores the query text" for HU/EN temporal questions — retrieval succeeds only via vector similarity on shared relative wording.

**PROPOSAL (deferred, separately gated)**: HU/EN relative-date recognition is the highest-value future enhancement. App-side query pre-expansion (rewrite before `vm.search`, mirroring the vendor's own in-place expansion contract: "only the retrieval copy changes") is pin-safe; a vendor table extension is a patch-ledger decision. NOT part of this gate's implementation.

## G. Speaker & language (PHASE 6, FACT)

- HU memory → HU query: rank 1. EN→EN: rank 1. HU memory → EN query: rank 1. EN memory → HU query: rank 1. **Multilingual behaviour is intact and must be preserved** — any rich-retrieval change that rewrites the query must remain language-neutral (or verified against this 4-way matrix).
- Speaker: `speaker_filter` is API surface only; no tags exist in text mode (audio path writes them); no narrowing (measured identical results). No speaker recognition is proposed (out of scope, per rules).

## H. Compatibility assessment

| Surface | Today | With D2 Stage 1 | Risk |
|---|---|---|---|
| CLI bridge | Path A | unchanged (helper imported, not yet wired) | none |
| Web voice turn | Path A + emotion | unchanged | none |
| Web `/api/classify` | vendor classify | unchanged | none |
| Forensic driver | Path A via bridge | unchanged | none |
| Vendor internals | pin e8384e0 | **untouched** — no new patch ledger entries | none |
| Tests (939/0/25 gate) | green | additive tests only | none |
| DemoMemoryLayer (mock web) | fabricates classification | unchanged | none |
| M3/learning layer | n/a | untouched | none |

## I. Migration strategy (PROPOSAL)

- **Stage 1 (this gate's scope)**: add `app/memory_retrieval.py::search_rich` + renderer date prefix + full test battery (J). No caller flips. Ship as v0.5.2.
- **Stage 2 (separately gated)**: flip ONE surface (web voice turn) to `search_rich` behind a config default-off flag; A/B measure in the field (the harness from this gate is the measurement tool).
- **Stage 3 (separately gated, prerequisite B1/B2 fix decided first)**: consider flipping CLI; consider vendor-side narrowing fixes (slot-tag threshold, strict pool mode, tag cleanup) — each its own patch-ledger entry with gate.

## J. Regression risks

1. Renderer date prefix changes the LLM prompt content → output drift on date-sensitive queries (mitigate: additive line format, keep 1200-char cap, A/B in tests).
2. `search_rich` adds +1 LLM call per turn when wired → latency on the voice path (measured: classifier is a short completion; TASK 1 full-env first token 39.6 ms baseline re-usable).
3. Any future default flip hits B1/B2 (narrowing inert) — flipping defaults before fixing narrowing buys cost, not quality (this is the central evidence-based warning of this gate).
4. Cross-language regression risk if query rewriting is ever added (G matrix is the guard).
5. `related_summaries`/classification data reaching the UI changes what users see in slot badges (today always empty) — cosmetic, verify in Stage 2.

## Test strategy (PHASE 10 — designed, NOT implemented)

**Unit** (fast, mock embedder or deterministic harness):
1. `search_rich` returns vendor SearchResult unchanged when classify returns empty (degradation path).
2. `search_rich` passes slots+entities through exactly (wiretap mock asserts the LLM payload + search kwargs).
3. `search_rich` accepts/forwards top_k, emotion, speaker_filter.
4. Renderer emits `[date]` prefix when observed_at present, none when empty; 1200-cap preserved; rb block unchanged.
5. Renderer boundary: score/base_score/time_boost/metadata NEVER appear in rendered context (negative assertion).
6. Determinism: identical inputs → identical rendered context (hash comparison, 3 runs).

**Integration** (real E5 + store, mock LLM — the baseline harness env):
7. Path A/B equivalence contract: rank-1 13/14 both (the measured baseline as a regression pin).
8. HU/EN/cross-language 4-way matrix rank-1 (language guard).
9. Temporal corpus queries still rank-1 (vector-only temporal as-is; documents current behaviour).
10. Empty control still returns top_k hits with no exception (documents no-threshold behaviour).
11. `vm.search()` signature unchanged: positional/keyword calls from v0.5.1 code still work (backward-compat pin).
12. Speaker filter leg: identical results with/without (documents text-mode no-op).

**End-to-end**:
13. CLI forensic driver: `search_rich` path produces non-empty memory_context (TASK 1 harness re-run).
14. Web voice turn with a feature flag OFF: byte-identical behaviour to v0.5.1 (prompt-level diff).
15. Web `_hits_payload` surfaces slots/entities when fed a Path B result.

No existing test weakened or deleted; all 939 current tests must stay green.

---

## GATE DECISION

**PROCEED TO IMPLEMENTATION** — scoped to Stage 1 only (D2 helper + renderer date prefix + test battery H/I; no caller flips, no vendor changes, no default changes).

Evidence basis: (a) D2 is additive and pin-safe with zero default-change risk on a system whose full gate is 939/0/25; (b) the rich path's measured benefit is currently ZERO (B1/B2) — therefore it must be exposed as an explicit opt-in with visible cost, which is exactly what Stage 1 delivers; (c) flipping defaults or fixing narrowing (vendor patches) is **BLOCKED — PREREQUISITE REQUIRED** (the narrowing no-op must be fixed and field-validated first, as its own gated change).

*Prepared from the current verified state (v0.5.1, repo `cc43859`). All FACTs traceable to the three evidence JSONs listed above.*
