# VoiceMEM v0.10 — Memory Semantics

**Scope:** the fact layer (left brain, mem0/Qdrant rows) and the trait layer
(right brain, `rb_traits`). The cognitive graph is out of scope. This
document defines what each semantic term MEANS in this system, what is
measured/tested, and — just as important — what is deliberately NOT
supported yet.

The v0.10 design decision (see `audit/VoiceMEM_v010/PHASE0_BASELINE_AND_SEMANTIC_MAP.md`):
**no Observation Store was introduced.** Every semantic below is carried by
ADDITIVE metadata on the existing rows and DERIVED at read time — no
schema migration, no destructive change, v0.9.2 data stays readable.

---

## The domain-neutral observation primitive

Every stored observation (fact or trait) carries:

| Field | Facts (mem0 payload metadata) | Traits (`rb_traits` column) | Meaning |
|---|---|---|---|
| subject / predicate / value | in the fact text | `claim` (+`slot`) | who/what the row is about |
| stance | `stance` | `stance` | polarity/temporal class, see below |
| temporal validity | `valid_from`, `valid_until` | (stance only) | when the statement itself says it applies |
| observation time | `time_start` (event) / `created_at` (event when supplied) / `last_observed_at` | `first_seen`, `last_seen`, evidence `created_at` | when it was OBSERVED |
| row modification time | mem0 top-level `updated_at` | `updated_at` | when the ROW was last written |
| confirmation count | `occurrence_count` | `occurrence_count` | independent confirmations |
| confidence | annotator default 0.9 (fact ranking does not use it) | `confidence` (asymptotic reinforcement) | confirmation strength — **not a probability** |
| provenance | `turn_id`, `source`, `speaker_entity_map`, `affect` | evidence `quote`/`emotion`/`cause` | where it came from |
| supersession | `supersedes` / `superseded_by` / `superseded_at` | same columns | the chain |

**The four time meanings are distinct and never conflated:**

1. **observation time** — when the user said it (`time_start`; the v0.10
   anchor fix makes the live path emit a real ISO datetime — pre-v0.10 it
   was a time-of-day string every parser rejected);
2. **row modification time** — when the stored row was last written (mem0
   `updated_at`, traits `updated_at`) — e.g. a supersede-marking or an
   occurrence increment modifies the row WITHOUT being a new observation;
3. **temporal validity** — when the STATEMENT says it holds
   (`valid_from`/`valid_until` — only set when the statement itself bounds
   it);
4. **currency** — whether the row is the current belief of record
   (structural: the supersession chain position + validity interval vs
   today).

## Statuses — DERIVED, never stored

`current | historical | superseded | future` are computed at read time
(`leftbrain/temporal.py::row_status`):

* **superseded** — `superseded_by` is set (the chain closed this row);
* **future** — `valid_from` > today, or `stance == "future"` (indefinite
  future);
* **historical** — `valid_until` < today, or `stance == "past"`
  (self-declared past: "used to");
* **current** — everything else.

**uncertain is a FACET, not a status** — an uncertain observation is
current knowledge rendered as uncertain. It never merges into, flips, or
outranks a confirmed row.

## The stance vocabulary (facts AND traits — one vocabulary)

`pos` · `neg` · `past` · `qualified` · `uncertain` · `future` · `""`

Classified deterministically (regex, 0 LLM, `rightbrain/stance.py`), scan
order `uncertain > qualified > past > neg > future > pos`, EN+HU cues. The
claim OR the original quote may carry it — the quote is ground truth (the
extractor normalises "szeretni fogom" → "szereti"; the future signal
usually survives only in the quote).

| Class | Example (EN) | Example (HU) | Stored effect |
|---|---|---|---|
| pos | "Thomas likes coffee" | "Tamás szereti a kávét" | current row |
| neg | "no longer likes coffee" | "már nem szereti a kávét" | supersedes the pos row; the OLD row's interval closes at the observation date |
| past | "used to like tea" | "régen szerette a teát" | separate historical row; never claims currency |
| qualified | "likes tea except in winter" | "kivéve télen szereti" | separate scoped row |
| uncertain | "may not like tea" | "talán már nem szereti a teát" | separate row, rendered `uncertain`; NEVER replaces a confirmed row |
| future | "will like fishing" | "szeretni fogja a horgászatot" | `valid_from` = the date in the statement ("" = indefinite); ranked below current for present-tense questions |

## PHASE 2 semantic classes — how each is represented

* **A. likes X now** → row, stance `pos`, status current.
* **B. used to like X** → own row, stance `past`, status historical (even
  with nothing superseded — it never claimed to be current).
* **C. no longer likes X** → ConflictResolver UPDATE → new `neg` row
  (current) + old row superseded + **`valid_until` = observation date on
  the old row** (the structure records WHEN the state ended, not just
  that it ended). The mem0-official resolver can also emit DELETE for
  cessation — the fork's delete gate blocks destruction, and v0.10 adds
  the **denied-DELETE → supersession fallback**: the cessation is
  appended as the new current row instead of being silently dropped.
* **D. may not like X** → stance `uncertain`, separate row, rendered with
  the `uncertain` marker; never flips/replaces.
* **E. will like X** → stance `future` + `valid_from` (dated) or ""
  (indefinite). NOT current: ranked below current rows for present-tense
  questions; a later confirmed statement replaces it via the normal
  chain. A future DATED EVENT ("flight on 2026-12-01") is deliberately
  NOT future status — it is current knowledge about a future event.
* **F. liked X in January but not now** → the January observation keeps
  its event date; the cessation supersedes it and closes its interval;
  date-overlap retrieval recovers the January state for past-intent
  queries.
* **G. liked X repeatedly, then changed** → each confirmation increments
  `occurrence_count` (+`last_observed_at`); the flip freezes the count
  on the historical row (traits since v0.9.0; facts: the count stays on
  the superseded row and the chain keeps it retrievable).

## Retrieval semantics (canonical path, cardinality UNCHANGED: top_k 5 + rb 3)

The single canonical search (`mem0_backend_store.search`, reached through
`Search()`) now ranks by:

```
(-temporal_tier, not superseded_by, base_score + recency_boost)
```

The query's temporal INTENT (now/past/future; default now; EN+HU cues)
picks the tier map:

* **current-intent** — current(0) < historical = future(1) < superseded(2)
* **past-intent** — historical = superseded(0) < current(1) < future(2)
* **future-intent** — future(0) < current(1) < historical(2)

Intent reorders, NEVER filters — a historical row still surfaces when it is
the only relevant one. **Legacy equivalence (proven by randomised test):**
rows without temporal keys (all pre-v0.10 data) sort in EXACTLY the v0.9.2
order — the tier degenerates to the old `(not superseded_by, …)` partition.

The provenance suffix now renders the markers compactly:
`[<date> | 3x confirmed | past | superseded | by X]` — `past`,
`future from 2027-01-01` / `future`, `uncertain`, `qualified`. Absent
markers = current confirmed knowledge (the existing convention). The
1200-char render cap is unchanged — markers live within it.

## Confidence (documented boundary, v0.10 decision)

Trait confidence is **confirmation strength** (asymptotic reinforcement
`c' = c + (1−c)·0.30` per agreeing observation, start 0.9; read-side decay
90d grace/180d halflife) — NOT a calibrated probability; it is therefore
deliberately not rendered into the LLM prompt. Fact confidence (annotator
default) does not participate in fact ranking at all.

**v0.10 decision: temporal state does NOT reduce confidence.** No
measurable justification exists for a probabilistic coupling; the existing
model already prevents post-flip reinforcement (frozen rows). Superseded
rows keep their frozen confidence as the historical record of how
confirmed the OLD state was.

## What the system deliberately does NOT support yet (v0.11 candidates)

1. **Event-time supersession guard on facts.** Currency follows WRITE
   order; a backdated replay of an old statement can re-open a chain
   (both rows keep their true event times — the structure a future guard
   needs is now stored). The trait side HAS this guard (`_is_stale_replay`)
   but fails open without parseable observation times.
2. **Replay detection for identical re-observations.** Re-speaking an old
   statement verbatim counts as a new confirmation (occurrence++), not as
   a replay — the wall-clock path cannot distinguish them.
3. **Auto-promotion of indefinite futures.** "Will like X" with no date
   stays future forever until superseded; no clock job promotes it when
   the (unknown) date passes.
4. **Trait-side validity intervals.** Traits classify future/past via
   stance only; no `valid_from`/`valid_until` columns on `rb_traits`.
5. **Consolidation engine.** The chain + intervals + `memory_history()`
   hook exist; the observation → current → historical compaction itself is
   future work (v0.11).
6. **Cross-chain reasoning.** Each supersession chain is independent; no
   link between "likes A→B" and "dislikes A" claims about the same topic.

## Backward compatibility

* **Existing v0.9.2 data:** readable as-is. Pre-v0.10 rows have no
  `stance`/`valid_*` keys → status `current`/`superseded` only → the
  EXACT v0.9.2 sort order and render output (randomised proof +
  render-equality tests).
* **Empty database:** zero-state search returns `[]` exactly as before
  (covered by the existing suites).
* **Partially migrated data:** every key is read independently with
  neutral defaults; a row with `stance` but no intervals behaves like the
  stance alone.
* **No migration was needed** (schemaless metadata + derived statuses).

## Where the code lives

* `voicemem/leftbrain/temporal.py` — the v0.10 core (classification,
  statuses, intent, tiers; deterministic, 0 LLM).
* `voicemem/rightbrain/stance.py` — the shared stance vocabulary (+`future`).
* `voicemem/utils/common/voice_input.py` — per-fact classification at
  ingest; UPDATE interval-closure; denied-DELETE fallback.
* `voicemem/leftbrain/mem0_backend_store.py` — temporal fields on hits;
  tier ranking; `update_memory(valid_until=…, new_meta=…)`.
* `voicemem/leftbrain/memory_repository.py` — mirror sync;
  `memory_history()` (PHASE 9 hook).
* `app/web_server.py` + `app/voicemem_bridge.py` — render markers
  (mirrored copies); `app/retrieval_contract.py` — trait contract.
* Tests: `tests/unit/test_temporal_semantics.py` (37),
  `tests/integration/test_temporal_memory.py` (39, real E5 + mem0/Qdrant).
