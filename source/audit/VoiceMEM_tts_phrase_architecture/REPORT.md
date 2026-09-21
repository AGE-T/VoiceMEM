# VoiceMEM — TTS PHRASE ARCHITECTURE AUDIT

**Date:** 2026-09-20
**Type:** Audit only. No production code change, no commit of code, no engine
or SDK change. Pure-python read-only lab checks ran on **copies** under
`/tmp/tts_phrase_lab/` (matrix scripts + outputs preserved there); no models,
no LLM, no embedding, no DB were run or touched.
**Question set (task §§6–9, §19.1-10, §21):** is the current deterministic
span segmentation sufficient; is a separate phrase detector necessary; could
LLM structured language output replace or augment it; what is the safest
hybrid; what are the real latency/concurrency costs.
**Baseline audited:** `voicemem-agent/app/text_utils.py` at tree HEAD
`a3e3a8a` (= v0.10.5 + the post-release leading-absorption fix `edac5c9`).
All line references are to this exact file unless prefixed.

---

## 1. THE CURRENT ALGORITHM (as shipped, verified by direct read)

`detect_language(text)` (:83-129) — chunk-level single label. Empty→`en`
(:94-95). Scoring: HU diacritic **+3**; `HU_STOPWORDS` (24-entry, :24-31)
**+2**; `HU_PLAIN_WORDS` (F-O accent-free table, :37-43) **+2**; `EN_STOPWORDS`
(37-entry, :57-63) **+1**; ≥3 tokens with ≥50% HU digraphs → **+2 hu**
(:116-119); any `q` → **+2 en** (:122-123; `w` deliberately excluded).
`hu>en→hu`, `en>hu→en`, tie → `hu` iff a diacritic present else `en`
(:125-129).

`_classify_word(token)` (:191-224) — word-level priority ladder:
① digits/`://`/`@` → neutral ② HU diacritic → `hu` ③ `EN_CONTRACTIONS`
(50-entry, :149-161) → `en` ④ EN consonant cluster (`ch ck sh th ph wh gh qu`)
or `q` → `en` ⑤ EN vowel cluster (`ou ee oo oa ue ui ei`) → `en` —
`au/ea/eu/ai` deliberately **absent** (autó/tea/euro/hazai) ⑥ HU
stopwords/plain words → `hu` ⑦ HU strong digraphs (`sz cs gy zs`) → `hu` —
`ny/ly/ty` excluded (only/really/city) ⑧ `EN_WORD_STOPWORDS` (= EN_STOPWORDS
minus `a`, `is`) → `en` ⑨ neutral.

`_scan_regions` (:227-276) — greedy paired `"`; parens at nesting level 1;
single quotes only under alnum-boundary proof (contraction apostrophes never
delimit, :259-269); unclosed dropped. `_region_language` (:279-299) — no
foreign evidence→host; foreign-only→foreign; both→`detect_language` arbitrates,
split only on a foreign verdict.

**Phrase runs** (the unquoted mechanism): `PHRASE_RUN_MIN_WORDS=2` (:320);
`PHRASE_RUN_TRAILING_BUDGET={en:1, hu:0}` (:328);
`PHRASE_RUN_LEADING_BUDGET={en:1, hu:0}` (:336, the post-release fix);
`PHRASE_RUN_LEADING_MIN_EVIDENCE=2` (:343 — the loanword shield:
single-evidence runs like "touch base" NEVER absorb);
`PHRASE_RUN_MAX_RUNS=3` (:348 — fragmentation guard); clause-final
punctuation blocks (`.!?;`, :352). `_iter_phrase_words` (:374-406):
hyphen-trimmed signal-free words (HU-suffixed `"base-t"` stays host);
evidence-bearing heads stay whole (`check-in`). `_phrase_runs` (:409-462):
runs start at foreign evidence, extend over evidence freely, over neutrals
only within the trailing budget and never across clause-final punctuation;
`_absorb_leading` (:465-500) absorbs ONE preceding signal-free word iff
budget>0 ∧ run has ≥2 evidence words ∧ candidate purely alphabetic ∧ not
hyphen-trimmed ∧ not clause-final.

`segment_language_spans` (:509-611) — host normalised (≠`en`→`hu`); level-1
regions + level-2 runs in host gaps; >3 runs → all dropped (:574-575);
whitespace shards absorbed; adjacent same-language spans merged; **hard
invariant `"".join(spans) == text`** (:538-539); empty → `[(text, host)]`.
Forced voice modes skip segmentation at the **caller** (auto-only:
`web_server.py:3384-3385`, `pipeline.py:720-721`). `SentenceStream`
(:812-903): first chunk 24 chars / later 80 (`config.py:197-198`),
digit-dot holdback (:883-889). `normalize_for_speech` (:662-699) is applied
only on the web path (`web_server.py:3364`) — a documented CLI divergence.
TTS: per-span language, same resolved voice, PCM concat, 2-slot synth
semaphore (`web_server.py:147`), ordered sender → **playback order is
invariant by construction**. `[PROVEN — all read from the current tree]`

---

## 2. VERIFICATION OF THE PRIOR AUDITS AGAINST THE CURRENT TREE

| Audit claim | Verdict |
|---|---|
| v0104: detect weights/tables | ✅ `text_utils.py:100-129` `[PROVEN]` |
| v0104: `VoiceSettings.resolve` voice_settings.py:239 | ✅ exact |
| v0104: 2 parallel chunk workers + ordered sender | ✅ now `web_server.py:147` + `:3118-3177` |
| v0104: cited line numbers for `normalize_for_speech`/`detect_language` call sites | ⚠️ line-drift only (now :662 / :3367) — cosmetic staleness, no behavioural change |
| v0.10.5-forensic: leading budget {en:1,hu:0} + MIN_EVIDENCE=2 + `_absorb_leading` guards | ✅ `:336`, `:343`, `:487-500` `[PROVEN]` |
| v0.10.5-forensic: T5 routing change exactly one span; T6/T7/T10 byte-identical; 59→59 calls | ✅ lab-reproduced, matches recorded evidence |
| v0.10.5-forensic: 81 tests in `test_tts_code_switching.py` | ✅ 81 test functions `[PROVEN]` |
| unquoted audit: R1–R6 constants and guards | ✅ all present at stated values |
| unquoted audit: F2–F5 post-fix matrix | ✅ lab output matches `repro_after_fix_20260918.txt` byte-for-byte |
| unquoted audit: "<15 µs/call" | ⚠️ optimistic — its own recorded evidence says 21–127 µs; our lab: 27–133 µs |
| unquoted audit: FP row "I visited Győr today" (assumes EN host) | ⚠️ premise stale — actual host is `hu` (Győr +3); real output `[en]'I visited' + [hu]' Győr today.'` — still a mis-route, different reason |
| unquoted audit: operator constraints (no lexicon, no new detection system, no chunking change) | ✅ present and quoted in §5 |

**Only tree change since the audits = `edac5c9`** (the leading-absorption
fix). No stale *behavioural* claims beyond the two analytical premises noted.

---

## 3. MANDATORY TEST MATRIX (executed on real code in the lab)

Full span outputs preserved in `/tmp/tts_phrase_lab/matrix_output.txt`.
Condensed:

### 3.1 Correct behaviour classes `[PROVEN]`

| Case | Result |
|---|---|
| "Please take care." / "We need to catch up tomorrow." | host=en, single EN span ✅ |
| "Rendben, see you later." | `hu 'Rendben, '` + `en 'see you later.'` ✅ (the original field case) |
| "Szerintem ezt most cut to the chase módon kellene megoldani." | head absorbed: `en 'cut to the chase'` ✅ |
| "Hit the road, aztán induljunk." | `en 'Hit the road,'` ✅ (leading absorption works) |
| "Burn the midnight oil, aztán folytatjuk." | `en 'Burn the midnight oil,'` ✅ |
| ambiguous ×11 (hazai, autó, auto, tea, euro, projekt, only, really, city, money, many) in HU sentences | all single-HU ✅ (anti-shredding tables work) |
| contractions (I'd, don't, won't; "we'll see" in HU) | EN spans, contractions never split ✅ |
| "teammeetinggel" (single suffixed token) | stays HU ✅ (min-2-words shield) |
| URLs / emails / `x_val = 42` | neutral rule ①, no exception ✅ |
| "Vettem egy iPhone 15-t." | `en 'iPhone 15'`, HU suffix stays host ✅ defensible |
| "a meeting" (single-word loan) | stays with host ✅ |
| "Szeretnék egy gyors touch base-t veled." | `en 'touch base'`, `-t veled.` host ✅ |
| fragmentation guard (4 runs) | all runs dropped → single host span ✅ |
| loanword shield "Kedd touch base volt." | `en 'touch base'` without absorbing "Kedd" ✅ |
| quoted EN in HU / `(follow up)` parens | quoted→EN ✅; paren zero-evidence stays host (documented class) |
| punctuation-heavy mixed sentence | single HU, no exception, join invariant holds ✅ |
| "Please take care." after an HU sentence, as **separate SentenceStream chunk** | chunk 2 alone → host=en ✅ (production chunking splits at the sentence dot — verified) |

### 3.2 Documented limitations (known, accepted, pinned by tests) `[PROVEN]`

| Case | Result |
|---|---|
| zero-evidence idioms embedded in HU ("break a leg", "take care", "give up", "turn off", "make sense", "take place", "pay attention") | stay on the HU model — **operator-forbidden to fix** without a lexicon (§5) |
| "Majd catch up tomorrow, jó?" | `en 'catch up'` only; "tomorrow" (signal-free) left HU — trailing budget = 1 by design |
| "quick brown fox teszt" | `en 'quick brown'`; "fox" stays host |
| "(follow up)" paren zero-evidence | host |
| 24-char first-chunk window splitting an idiom | documented residual; chunking change is operator-forbidden |
| repeated phrase "…megint see you later…" | 2nd occurrence absorbs the HU adverb ("megint see you later,") — the known leading-FP class, pinned in the forensic suite |

### 3.3 NEW defect classes found by this audit (not in any prior report)

| # | Case | Output | Root cause |
|---|---|---|---|
| **N1** | "A 2026-os meeting fontos lesz mindenkinnek." | `hu 'A 2026-os '` + **`en 'meeting fontos'`** + `hu '…'` | "meeting" carries `ee` vowel-cluster EN evidence at word level — the audits call it signal-neutral, which is wrong at word level; the trailing budget then sweeps the Hungarian word "fontos" into the EN span. Deterministic, µs-class fixable within the existing table design. `[PROVEN — reproduced]` |
| **N2** | "Holnap lesz a meeting a csapattal." (accent-free HU + EN stopwords) | **host=en** — the whole HU sentence on the EN model | `a`×2 scores EN-stopword vs `lesz`+2 HU → tie → no diacritic → EN. Same class: "Holnap touch base." → EN host. Deterministic tie-break fix possible. `[PROVEN — reproduced]` |

Both N1/N2 are **deterministic word-level scoring fixes inside the existing
table design** — no lexicon, no new detection system — and therefore *within*
the operator's standing constraints. Magnitude in real traffic:
`[UNKNOWN]` (no production DB here to measure frequency).

---

## 4. LLM STRUCTURED LANGUAGE SEGMENTATION — CAN IT REPLACE THE DETECTOR?

### 4.1 The structural argument (decisive) `[PROVEN mechanism]`

Segmentation input only exists **after** the LLM emits a chunk, and TTS needs
it immediately. On the production **single-slot** llama-server
(`parallel: 1`, `config/llm_config.yaml`; qwen3.6-35b-a3b, ctx 32768,
temp 0.7, thinking suppressed — `llm.py:174-214`), a second request **queues
behind the entire remaining generation** of the reply stream — proven by the
S8 correlation class (user TTFT = full remaining background leg, worst
31.2 s; 163.2 s behind a 10.4k-token prefill). An LLM segmentation call in
the TTS hot path therefore either serialises behind the reply or delays first
audio by its own full round-trip.

### 4.2 Approach-by-approach evaluation

Transport facts verified first: the app already uses request-level
`response_format json_object` (`llm.py:385`); the vendor background legs
forward `response_format` through `bg_chat_create`
(`llm_bg_gate.py:150-230`, streaming + cooperative cancel); no GBNF/grammar
usage exists anywhere in the repo; server-wide `--json-schema`/grammar flags
would constrain **every** response and break plain streaming (documented at
`llm.py:369-373`).

| Approach | Latency impact | Streaming | Failure mode | Parser | Verdict |
|---|---|---|---|---|---|
| **A** request-level `json_schema` | +prefill+TTFT+grammar-constrained decode ≈ **+6–17 s per chunk best case** (slot free); unbounded behind the streaming reply (common case) | incompatible — JSON envelope delays first audio until parse completes | schema-valid but wrong spans | moderate | **REJECTED (hot path)** |
| **B** free-form JSON in the reply | first audio = full-reply time (+10–40 s); 512-token budget can truncate the JSON | destroys streaming TTS | markers leak into spoken text | moderate | worst option |
| **C** inline `[hu]…[/hu]` tags | +first-tag-close only (~0.5–2 s) — least-bad | partially compatible | one unclosed tag swaps languages for the rest of the reply; tag chars eat the 24-char first-chunk window | trivial | risky on a mid-size local model; loses to deterministic on every axis |
| **D** tool/function-style | same as A (extra call) or B (in-reply) | as A/B | brittle parsing on non-flagship models | moderate | dominated by A |
| **E** prompt-only format | as B | no | no enforcement — fences/quotes drift | trivial | rejected |
| **F** server-level grammar | global, breaks streaming | breaks | server-wide | n/a | rejected (request-level form A is the usable variant, and it is rejected above) |
| **G** hybrid LLM + deterministic validation | same as A **if on the hot path; ≈0 off-path** (telemetry/post-hoc) | only off-path | malformed → deterministic fallback (safe) | ~80 lines | the **only defensible LLM shape — off the hot path** |

Latency decomposition `[INFERENCE from measured tok/s bands, mechanism PROVEN]`:
current steady-state first audio ≈ 6–8 s total (ASR ~0.5 s; memory ≤1.6 s
budget; LLM TTFT ~3.2 s; 24-char decode ~0.7–1.4 s; **segmentation
0.00003–0.0002 s**; first-span synth 1.5–2.7 s sandbox-CPU). Segmentation is
≈0.002% of the budget. Any LLM variant multiplies the two dominant terms.

### 4.3 Concurrency cost `[PROVEN class]`

Deterministic segmentation: pure CPU, zero shared resources, zero
cancellation surface. Any LLM path: contends for the single slot against
(1) the reply stream itself and (2) the background memory legs
(extraction + conflict resolution, which already use the cooperative-cancel
transport — every cancelled segmentation attempt is wasted work; the reply
cannot be cancelled for segmentation's sake). Per-chunk calls multiply the
contention ×5–8 per reply.

---

## 5. IS A SEPARATE PHRASE DETECTOR NECESSARY? OPERATOR CONSTRAINTS (quoted)

The question is already answered by the field evidence and by the standing
operator constraints, which this audit re-verified verbatim in the unquoted
audit (§4.4):

> "**No English dictionary / phrase list.** Detection is orthographic
> evidence + stopword tables (all pre-existing, all word-level). Phrases with
> zero signal words (e.g. `follow up` in HU text …) stay host: documented
> limitation, **preferred over an unauditable lexicon**."

…plus the forensic audit §5 rejecting "a new statistical detector" and
forbidding chunking changes ("NE változtasd meg az egész chunking
rendszert"). `[PROVEN]`

**Conclusion:** the evidence-gated phrase-run mechanism *is* the phrase
detector — it exists, it is bounded, it is regression-pinned (81 tests), and
it routes correctly every phrase class that carries orthographic evidence.
A *separate* detector (lexicon/statistical) is **not necessary** and is
operator-forbidden today; the zero-evidence tail is a documented, deliberate
trade-off.

---

## 6. HYBRID ARCHITECTURES COMPARED

| Hybrid | Latency | Reliability | Failure mode | Fallback | Streaming | Concurrency | Verdict |
|---|---|---|---|---|---|---|---|
| (i) LLM spans → deterministic validator → deterministic fallback | full round-trip **if in-path**; ≈0 off-path | high (validator) | safe degrade | deterministic | only off-path | slot contention if in-path | viable **only as offline QA/telemetry** |
| (ii) deterministic + LLM only for ambiguous spans | +1–6 s on affected chunks best case; unbounded behind the stream | medium | per-span fallback | deterministic | poor fit | extra slot requests | rejected — ambiguity signal barely exists and cost lands on the hot path |
| (iii) deterministic + LLM validation when confidence low | as (ii) | — | — | — | — | — | **rejected: no confidence signal exists; inventing one = a new detection system (operator-forbidden)** |
| (iv) deterministic + phrase-level sequence model (tiny n-gram/lexicon) | **µs** | high, deterministic | silent mis-route | host | perfect | zero | **best-in-class technically, operator-FORBIDDEN today** — needs explicit re-authorisation |

**Recommended architecture:** keep (iv)'s shape *conceptually* on file, but
implement nothing now. The deterministic evidence-gated detector remains the
sole hot-path authority. LLM involvement, if ever desired, is confined to
post-hoc telemetry (approach G off-path) — never between the LLM stream and
TTS.

---

## 7. THE TEN DECISION QUESTIONS (task §19)

1. **Is the existing deterministic span segmentation sufficient?** For its
   declared scope, **yes** `[PROVEN: three audits, 81 pinned tests, this
   50+ case matrix]`. Insufficient only for: the zero-evidence idiom tail
   (deliberate, operator-endorsed trade-off), 3+-word trailing-neutral
   phrases, single-token suffixed phrases — plus the two NEW defect classes
   N1/N2 (§3.3), both fixable deterministically.
2. **Is a phrase detector actually necessary?** A *separate* one: **no**.
   The evidence-gated phrase-run mechanism is the phrase detector and is
   already shipped and regression-pinned.
3. **Could structured LLM output replace it?** **No** — structural proof in
   §4.1: any LLM call serialises behind the reply on the single-slot server
   or delays first audio by its own round-trip; failure handling converges to
   the deterministic path anyway, which is itself the proof the deterministic
   path must remain authoritative.
4. **Could it augment it?** Only off the hot path (post-hoc telemetry,
   next-turn caches) or via an operator-re-authorised bounded idiom list
   gated like the existing rules. Nothing in the hot path.
5. **Safest hybrid?** Deterministic stays sole hot-path authority; optional
   (iv)-lite bounded lexicon only after operator re-authorisation;
   LLM validation only as offline QA.
6. **Fallback if LLM output is malformed?** The deterministic segmentation —
   always available, µs-class, invariant-preserving. (This is the design
   today: there is no LLM in the path to fail.)
7. **During streaming?** Chunks arrive per sentence / 24–80 chars;
   segmentation runs per chunk in µs; idioms straddling the first-chunk
   window split across chunks (documented residual; chunking change
   forbidden); forced voice modes skip segmentation; barge-in cancels unsent
   spans (`web_server.py:3171-3177`).
8. **Latency cost?** Current: 27–133 µs per call, 0 extra LLM calls; +1
   synthesis call per extra span (~1.5–2.7 s sandbox-CPU, amortised by the
   2-slot pump; first audio unchanged when chunks start in the host
   language). LLM path: +6–17 s per chunk best case, unbounded worst case.
9. **Concurrency cost?** Deterministic: zero. Any LLM path: single-slot
   contention against the reply stream and background memory legs (proven
   30–163 s classes), multiplied per chunk.
10. **What needs benchmarking before implementation?** Nothing for the
    deterministic path (already measured). If any LLM variant is pursued
    despite §4: on-target TTFT + grammar-on/off decode rate for a ~300-token
    constrained request on the production model; span-agreement rate vs the
    deterministic detector on a **labelled corpus (none exists — build it
    first)**; queue-behind-streaming measurement
    (`scripts/llm_slot_forensic.py` already exists for this).

---

## 8. TEST STRATEGY (pre-implementation requirements, task §21)

The existing `tests/unit/test_tts_code_switching.py` (81 tests) already
covers: pure HU/EN, quoted EN, unquoted evidence phrases, ambiguous words,
suffixes, contractions, punctuation, URLs/emails, capitalisation, repeated
phrases, multi-span sentences, join invariant, forced modes, web `_synthesize_chunk`
and CLI `_speak_chunk` integration sequences. **Required additions before
any N1/N2 fix ships:**

- N1 class: `meeting`-type `ee`-bearing loans followed by HU words (matrix:
  "meeting fontos", "weekend lesz", "green tea-t iszom") — pin the fix and
  the no-regression of "catch up tomorrow" (trailing budget must still work
  when the trailing word is genuinely English).
- N2 class: accent-free HU sentences whose only EN signal is stopwords
  ("Holnap lesz a…", "Most megint…" with `a`/`megint` balance) — pin host
  detection; verify no EN-host regression on genuinely English short
  sentences ("We need to catch up tomorrow." must stay EN).
- Product-name/`iPhone 15` class and the §3 matrix rows already reproduced
  here should be promoted from lab evidence into pinned tests where absent.
- Latency regression: keep the µs-budget assertion pattern
  (`measure_segmentation_latency_runs.py`).

---

## 9. RECOMMENDATION (decision tree)

```
Mixed-language routing defect reported?
├── Span boundary wrong AND phrase carries orthographic evidence
│     → fix inside the existing word-level tables + absorption budgets
│       (classes N1/N2 are exactly this — deterministic, µs, test-pinnable)
├── Phrase has ZERO orthographic evidence (break a leg)
│     → today: documented host-routing (operator-endorsed)
│     → only change with explicit operator re-authorisation of a bounded
│       idiom list (approach iv-lite); never via LLM in the hot path
├── Chunk-boundary splits an idiom
│     → pinned residual; chunking changes are operator-forbidden
└── LLM-structured segmentation proposed
      → REJECT for the hot path (§4.1 structural proof);
        permit only as off-path telemetry (G)
```

---

## 10. COMPLIANCE

Production code changes: **0**. Engine/SDK/config changes: **0**.
LLM/embedding/TTS model runs: **0**. Database access: **0**. The lab in
`/tmp/tts_phrase_lab/` operates on **copies**; its scripts are preserved
there and mirrored into this audit's evidence notes in `worklog.md`
(Task ID 6-a).
