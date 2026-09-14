# Memory Semantic Matrix — v0.9.0

**Scope:** contradiction, supersession and validity semantics for the trait
memory layer (`rb_traits`), with the fact layer's existing supersession
(VM-LOCAL-008) demonstrated as the architectural reference.

**Implementation:** vendor patch **VM-LOCAL-015** (`voicemem/rightbrain/
stance.py` new, `traits_store.py`, `rightbrain/brain.py`), app contract
extension `app/retrieval_contract.py`. All thresholds below are **measured**
on this machine with the pinned local `intfloat/multilingual-e5-small`
(384-d) — re-measure with `scripts/measure_semantic_matrix.py` after any
embedder change (a validation test pins the numbers).

---

## 1. The measured evidence (why cosine alone is unsafe)

| Case | Existing | Incoming | Cosine (E5, measured) | ≥0.95 merge? |
|---|---|---|---|---|
| A reinforce (en) | Thomas likes motorcycles | Thomas likes motorcycles | 1.0000 | yes (correct) |
| A reinforce (hu) | Thomas szereti a motorokat | Thomas szereti a motorokat | 1.0000 | yes (correct) |
| B paraphrase (en) | Thomas likes motorcycles | Thomas enjoys riding motorcycles | 0.9764 | yes (correct) |
| B paraphrase (hu) | Thomas szereti a motorokat | Thomas imád motorozni | 0.9803 | yes (correct) |
| **C antipathy (hu)** | Thomas szereti a motorokat | Thomas utálja a motorokat | **0.9685** | **yes — HAZARD** |
| **E past (hu)** | Thomas szereti a motorokat | Thomas régen szerette a motorokat | **0.9821** | **yes — HAZARD** |
| **E past (en)** | Thomas likes motorcycles | Thomas used to like motorcycles | **0.9865** | **yes — HAZARD** |
| **F qualified (hu)** | Thomas szereti a motorokat | …, kivéve télen | **0.9552** | **yes — HAZARD** |
| **flip-back (hu)** | már nem szereti a motorokat | megint szereti a motorokat | **0.9521** | **yes — HAZARD** |
| C negation (en) | Thomas likes motorcycles | Thomas no longer likes motorcycles | 0.9142 | no (but ≥0.88 retrieval) |
| C negation (en2) | Thomas likes motorcycles | Thomas does not like motorcycles | 0.9206 | no (but ≥0.88 retrieval) |
| C negation (hu) | Thomas szereti a motorokat | Thomas már nem szereti… | 0.9422 | no (but ≥0.88 retrieval) |
| C negation (hu2) | Thomas szereti a motorokat | Thomas nem szereti… | 0.9496 | no (but ≥0.88 retrieval) |
| C antipathy (en) | Thomas likes motorcycles | Thomas hates motorcycles | 0.9066 | no (but ≥0.88 retrieval) |
| claim pos↔neg | likes motorcycles | no longer likes motorcycles | 0.9198 | no |
| claim pos↔antipathy | likes motorcycles | hates motorcycles | 0.9149 | no |
| D replace (trait) | likes coffee | gave up coffee | 0.8824 | no |
| D replace (trait2) | likes coffee | stopped liking coffee | 0.9023 | no |
| flip-back (en) | no longer likes motorcycles | likes motorcycles again | 0.9271 | no |
| E temporal (en) | Thomas owns a Renault | Thomas previously owned a BMW | 0.8890 | no |
| E temporal (hu) | Thomasnak Renault-ja van | …korábban BMW-je volt | 0.8703 | no |
| F qualified (en) | Thomas likes motorcycles | …except during winter | 0.9338 | no |
| G uncertain (en) | Thomas likes motorcycles | Thomas might no longer like… | 0.9130 | no |
| G uncertain (hu) | Thomas szereti a motorokat | Thomas talán már nem szereti… | 0.9378 | no |
| C cross-lang | Thomas szereti a motorokat | Thomas no longer likes motorcycles | 0.8447 | no |
| C cross-lang (2) | Thomas likes motorcycles | Thomas már nem szereti a motorokat | 0.8114 | no |
| noise (topic) | likes motorcycles | no longer likes bicycles | 0.8876 | no |
| noise (topic2) | likes coffee | gave up tea | 0.8317 | no |

**Conclusion (FACT):** paraphrase (0.976–0.98), antipathy (0.9685), past
tense (0.982–0.987), qualified HU (0.9552) and flip-back HU (0.9521)
**occupy the same cosine range**. No threshold separates agreement from
contradiction. Cosine similarity is retrieval evidence, not semantic
agreement — the v0.8.1 merge would reinforce the very trait the incoming
observation negates (five measured pairs above 0.95).

**Cross-language (FACT):** full-sentence HU↔EN pairs fall below both gates
(0.81–0.84). In the production pipeline the *stored claims* are extractor
labels (English patterns, `merged_extraction.py`), so store-level pairs are
EN↔EN ≈ 0.92 — inside the supersede band. The language truth travels in the
evidence quote, which the stance scanner reads language-independently.

## 2. The semantic model

### 2.1 Stance (the deterministic signal layer)

`voicemem/rightbrain/stance.py` classifies ONE text (claim or quote) into:

| stance | meaning | example cues (EN / HU) |
|---|---|---|
| `pos` | current-state positive affect | likes, loves, enjoys / szereti, kedveli, imádja |
| `neg` | current-state negative: negation or antipathy | no longer, not anymore, gave up, stopped, does not like, hates / már nem, nem szeret…, utálja, abbahagyta |
| `past` | statement about a PAST state (temporal coexistence) | used to, previously, formerly / régen, korábban, volt, -ttem/-ttam/-otte/-ette/-tte endings |
| `qualified` | scoped/conditional statement | except, not always, only when / kivéve, nem mindig, csak, este |
| `uncertain` | hedged change of state | might, maybe, perhaps / talán, lehet hogy |
| `""` | no cue found (neutral/legacy) | gives examples before conclusions |

Scan order: **uncertain > qualified > past > neg > pos** (the first match
wins). "talán már nem szeretem" → uncertain (uncertainty never becomes a
hard replacement); "kivéve télen" → qualified; "régen szerettem, de már
nem" → past (documented limit: position-aware mixed-sentence parsing is
not implemented — the outcome is the safe separate-state, never a false
reinforcement, never a flip).

`observation_stance(claim, quote)` scans BOTH the distilled claim and the
user's original quote: the extractor may normalise a negation out of the
label; the quote is ground truth and is never missed.

**Documented gaps (deliberate, per the task's "prefer existing structured
signals; document the exact gap"):**

* surface cues only — semantic opposition with no marker and no lexicon
  word is invisible ("adores" vs "despises" outside the lexicon);
* Hungarian past detection is the explicit adverbs + double-t/vowel verb
  endings — agglutinative edge cases can be missed (the single-t endings
  were REJECTED: they terminate present-tense forms like "szeretem");
* the affect lexicon is finite (like/love/enjoy/prefer vs
  hate/dislike/avoid + HU equivalents);
* stance is one class per text — a sentence carrying both a past and a
  current clause is classified by the scan order above.

### 2.2 The merge decision (TraitStore.add, v0.9.0)

Three **explicit** outcomes (Phase 4 requirement):

```
MERGE (reinforce)     same-stance (or neutral input) AND cosine ≥ 0.95
                      → v0.8.0 semantics: evidence append, occurrence+1,
                        asymptotic confidence, first_seen preserved
SUPERSEDE (flip)      stance pair == {pos, neg} AND cosine ≥ band AND
                      topic tokens overlap AND not a stale replay
                      → NEW row becomes current (supersedes = old id);
                        old row gains ONLY superseded_by/superseded_at —
                        occurrence/confidence frozen (a contradiction is
                        never counted as confirmation)
SEPARATE (defer)      everything else (qualified/past/uncertain stance,
                      cross-stance, below band, topic mismatch, stale
                      replay with no historical row)
                      → new independent node; both states coexist and
                        stay retrievable; resolution is deferred
```

**Bands (measured, not invented):**

| constant | value | anchor |
|---|---|---|
| `MERGE_THRESHOLD` | 0.95 | unchanged v0.8.1 value (topic identity) |
| `SUPERSEDE_MIN_SIM` | 0.90 | measured contradiction cluster floor (antipathy-en 0.9066, stopped-liking 0.9023) |
| `SUPERSEDE_MIN_SIM_PRESUPPOSITION` | 0.88 | measured E5 topical-hit floor (the codebase's own calibration) + presuppositional negation ("no longer"/"gave up"/"már nem" presuppose the state they replace — "gave up coffee"↔"likes coffee" measured 0.8824) |

**Topic-overlap guard:** a supersede additionally requires a shared
content-word stem (≥4 chars, affect/negation/stopwords excluded, HU
inflection suffixes stripped, prefix-matching for cross-language stems).
Measured noise pairs ("likes motorcycles"↔"no longer likes bicycles"
0.8876) must never supersede each other.

**Stale-replay guard (event-time):** a flip is refused when the incoming
observation's event time is OLDER than the target's latest event time
(the newest evidence `created_at`, falling back to `last_seen`). A refused
flip with an agreeing historical row on the supersession chain attaches
its evidence to that frozen historical row (the replay is recorded, the
current state is untouched, no new "current-looking" row appears);
without a historical row it becomes a separate node. Without timestamps
the guard fails open on time — the stance gate remains the hard safety
(live sessions always carry `observed_at`).

### 2.3 Validity / current state (Phase 6 — smallest model)

No state enum was added. Validity is **structural**:

* a trait is **historical** iff `superseded_by != ''` (it was replaced);
* a trait is **current** iff `superseded_by == ''`;
* chains are append-only: `S(pos) ← N(neg) ← R(pos …)` — every flip
  appends a row and marks its predecessor; rapid alternation grows the
  chain, the last observation is always current, all intermediate states
  stay recoverable with their own evidence and dates;
* a superseded row is **never deleted** (supersession ≠ deletion — the
  delete path remains the separately-gated VM-LOCAL-005/007 operation);
* `stance` records the observation class that created the row.

Legacy rows (pre-v0.9.0 databases): additive idempotent `ALTER TABLE`
migration (stance/supersedes/superseded_by/superseded_at default '');
stance is lazily inferred from the stored claim text at decision time.

### 2.4 Confidence and occurrence semantics (Phase 7)

* reinforcement still raises confidence (asymptotic, unchanged);
* **a contradiction never raises it** — the superseded row's confidence
  and occurrence_count are frozen at flip time (proven by test);
* a superseded trait never behaves as active evidence: retrieval orders
  it after every active match and the prompt marks it `superseded`;
* stale/superseded are distinguishable at retrieval (contract fields);
* confidence is still never rendered as a probability in the prompt
  (v0.8.1 discipline, unchanged);
* **no confidence-reduction mechanism was added** — decay remains the
  read-time `effective_trait_confidence` (90-day grace, 180-day
  half-life). Reduction would need a semantic justification that does
  not exist yet (how much does one contradiction weigh? — unanswered, so
  not implemented).

### 2.5 Retrieval (Phase 8)

`search_scored` orders `(not superseded, -sim)` — current first, exactly
the left-brain VM-LOCAL-008 pattern. Superseded rows **remain in the
result set** (historically recoverable) but never outrank an active match
for a current-state query. `_rb_trait_hits` filters the min-sim gate
per-row (the active-first order broke the old "sorted desc, break early"
assumption), applies a ×0.75 priority demotion to superseded traits (the
heartnote convention) and surfaces the semantic state in hit metadata.

### 2.6 Prompt and payload (Phase 9)

* the app prompt line renders `[… | superseded]` for a historical trait
  — the fact-side wording (one term, one meaning); absence of the marker
  = current (existing convention);
* the confidence floats are still never rendered;
* the web payload carries `stance`/`supersedes`/`superseded_by`/
  `superseded_at` through the uniform `trait_fields_payload` schema;
  the mind-map marks superseded nodes ("former · …" in the card footer).

## 3. Cross-layer notes

* **Facts (left brain):** unchanged — the resolver-driven UPDATE →
  VM-LOCAL-008 append+supersession already implements cases D/E; the
  behaviour is now pinned by `tests/unit/test_fact_supersession_semantics.py`
  (explicit replacement, historical recoverability, current-first
  retrieval, occurrence never fires on UPDATE).
* **Heartnotes:** the run_cleanup LLM supersession marking (metadata
  `superseded_by`/`superseded_at`, ×0.75 demotion, "outdated" annotation)
  is unchanged and remains the third isomorphic supersession pattern.
* **Archive/TTL (Phase 15):** traits have **no archive and no TTL** —
  decay is read-time only and never hides rows, so no interaction exists
  by construction. For facts, the archive interaction is pinned by tests:
  archiving a superseded row hides it from search (data retained,
  unarchive restores); an archived current row is invisible to the
  resolver, so a new observation starts fresh (documented: reactivation
  of an archived row is not implemented — `unarchive_memory` is the
  manual path).
* **Observation Store (Phase 14):** decision recorded in §5 below.

## 4. Test coverage

* `tests/unit/test_stance_cues.py` (30) — the cue scanner, Phase 10 list;
* `tests/unit/test_trait_semantics.py` (32) — the Phase 12 matrix (15
  scenarios) + Phase 13 failure modes (rapid alternation, restart,
  duplicate ingest, stale replay, user isolation, topic guard, legacy
  migration) with a degenerate cosine-1.0 embedder (the hardest regime —
  the stance gate is the only protection);
* `tests/unit/test_retrieval_contract_semantics.py` (12) — the contract
  extension (prompt marker, payload keys, neutral defaults);
* `tests/unit/test_fact_supersession_semantics.py` (8) — the fact-side
  reference behaviour + Phase 15 archive interactions;
* `tests/validation/test_feature_memory_semantics.py` (2) — the measured
  E5 matrix as a regression guard (deep-skip when the model is absent).

## 5. Observation Store decision (Phase 14)

**OBSERVATION STORE NOT REQUIRED** (for the semantics this task required).

Evidence from the implemented use cases: every semantic requirement —
contradiction refusal, explicit supersession, historical preservation,
current/historical distinction at retrieval, replay handling, chain
navigation — is representable in the existing model:

* trait observation history = `rb_evidence` rows (quote, event time,
  emotion, left-brain cause link) — one row per observation, append-only;
* supersession chains = `supersedes`/`superseded_by`/`superseded_at`
  columns on `rb_traits`;
* fact history = mem0 rows + VM-LOCAL-008 metadata;
* heartnote history = metadata + run_cleanup marking.

A separate observation store would duplicate `rb_evidence` (traits),
mem0's own append history (facts) and the cognitive-graph links, and
would add a fourth supersession pattern to maintain. Revisit when a
concrete requirement exceeds this model (e.g. cross-trait contradiction
queries, observation-level retraction, or per-observation provenance
chains for multiple speakers on one trait).
