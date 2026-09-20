# VoiceMemAgent — N1/N2 TTS deterministic detector defects: forensic reproduction + minimal deterministic fix

Forensic report — British English. Date: 2026-09-20.
Task: "WORKSTREAM 2 — N1/N2 FORENSIC REPRODUCTION + DETERMINISTIC FIX"
(the two defect classes identified by the 2026-09-20 consolidated
audit, audit/VoiceMEM_tts_phrase_architecture §3.3).

Scope discipline: the smallest deterministic correction consistent with
the existing architecture. No lexicon, no LLM, no phrase detector, no
idiom list, no new model, no runtime service, no chunking/engine/voice
change. The two identified defect classes only.

---

## 1. Reproduction (before any production change) `[PROVEN]`

Reproducer: `tools/reproduce.py` — exercises the exact production path
the TTS call sites use (`normalize_for_speech` → `detect_language` →
`segment_language_spans`). Pre-fix evidence:
`evidence/reproduce_prefix.txt`; post-fix: `evidence/reproduce_postfix.txt`.

### N1 — EN-evidence loan absorbs a following Hungarian word

```
"A 2026-os meeting fontos lesz mindenkinnek."
  host: hu
  spans: hu 'A 2026-os ' | en 'meeting fontos' | hu ' lesz mindenkinnek.'
```

"fontos" (Hungarian, "important") was read with the English voice.

### N2 — accent-free Hungarian host flips to English

```
"Holnap lesz a meeting a csapattal."
  host: en   <-- the whole Hungarian sentence on the EN voice
"Holnap touch base."
  host: en
```

The failing classes were pinned by a new focused regression matrix
BEFORE the production change (`tests/unit/test_tts_n1_n2_matrix.py`,
16 tests): **9 failures** on the pre-fix tree, exactly the N1/N2
classes (evidence: `evidence/prefix_matrix_run.txt`).

## 2. Root-cause trace `[PROVEN — directly reproduced from the current implementation]`

All paths verified by execution, not inferred from the prior audit.

### N1 mechanism (`_phrase_runs` + `_absorb_leading`)

The trailing budget (`PHRASE_RUN_TRAILING_BUDGET[en] = 1`) absorbs
ANY word that classifies as signal-free (`_classify_word → ""`).
A Hungarian content word without diacritics, without stopword/plain-word
membership and without a strong digraph — "fontos" (o, o), "marad"
(a, a), "ablakot" (a, a, o) — is **orthographically invisible** to the
word classifier, so the budget swept it into the English span after an
EN-evidence loan ("meeting" carries `ee`, an `EN_VOWEL_CLUSTERS` member).
The leading budget (`_absorb_leading`) had the mirror exposure: a
signal-free Hungarian word immediately before a ≥2-evidence run was
absorbed ("Holnap see you later." read "Holnap" in English).

The protected counter-cases (why the budget exists at all): English
phrase bodies are also signal-free — "touch **base**", "see you
**later**", "catch **up**" — so the budget cannot simply be zeroed.

### N2 mechanism (`detect_language`)

Chunk-level scoring counts the Hungarian definite article "a" as an
English stopword (the word-level table deliberately removed "a"/"is";
the chunk-level table kept them). In "Holnap lesz a meeting a
csapattal.": hu = lesz(+2) = 2, en = a(+1) + a(+1) = 2 → **tie**; the
tie-break returns EN when no Hungarian diacritic is present → the whole
sentence gets the English voice. The audit's second case ("Holnap
touch base.") is the zero-evidence variant: "holnap" is absent from
`HU_PLAIN_WORDS` (the F-O accent-free function-word category contains
"hol" and "mikor" but not "holnap"), so hu = 0, en = 0 → tie → EN.

## 3. The fix (app/text_utils.py only — three surgical changes)

1. **Hungarian vowel-harmony guard** (N1): native Hungarian words are
   vowel-harmonic — all vowels back (a, á, o, ó, u, ú) or all front
   (e, é, i, í, ö, ő, ü, ű); front/back mixtures in one word are
   essentially an English trait ("base", "later", "afterparty").
   `_hu_harmonic(token)` (≥2 vowels, one class) now guards BOTH budget
   absorptions (trailing and leading): a Hungarian-looking candidate is
   never swept into an English span. Single-vowel tokens are exempt —
   they carry no harmony signal either way, so the protected English
   particles ("up", "next") keep their historical absorption.
2. **"holnap" table entry** (N2): joined `HU_PLAIN_WORDS` — the same
   F-O category as "mikor"/"hol" (temporal adverb, accent-free). This
   fixes the zero-evidence variant by score, and independently makes
   "Holnap" carry host evidence at word level (blocking leading
   absorption).
3. **Tie-break refinement** (N2): on a no-diacritic tie, Hungarian
   HOST-CONTEXT function-word evidence (`_hu_host_context_function_
   evidence`) beats the EN side's article counting. Tokens inside
   set-off regions (quotes/parentheses — foreign material under
   discussion, e.g. 'The word "szia" means hello here.') do not vote
   on the host. Computed lazily on the tie path only.

### Overcorrection check (WS2-3) — behaviours that depend on the old rules

The full existing battery was run against the fix. Two pinned
behaviours changed — both are the N1 class itself, corrected
consciously (tests updated with documentation):

| Case | Before | After |
|---|---|---|
| "Nyisd ki a chat ablakot." (documented FP pin) | en 'chat ablakot.' | one hu span ("chat" follows the single-word loan policy; "ablakot" stays host) |
| "Holnap see you later." (documented FP pin) | whole text en | hu 'Holnap ' + en 'see you later.' |
| 'The word "szia" means hello here.' (WebSynthesizeChunk) | en host + hu region | **unchanged** — an early tie-break draft flipped it; the set-off-region exclusion was designed specifically to preserve it |
| hazai / autó / tea / euro / projekt / only / really / city / money / many | — | unchanged (all pinned tests green) |
| pure HU, pure EN, quoted EN, unquoted EN, contractions, punctuation, URLs, e-mail, hyphen suffixes, forced modes, streaming boundaries | — | unchanged (full suite green, zero regressions) |
| "touch base" / "catch up" / "see you later" / "cut to the chase" / "hit the ground running" / "burn the midnight oil" routing | — | unchanged ("base"/"later" disharmonic; "up"/"cut"/"hit"/"burn" single-vowel-exempt) |
| "A meeting afterparty fontos volt." | en 'meeting afterparty' | unchanged (budget accounting: the disharmonic "afterparty" consumes the budget, harmonic "fontos" stays host) |

### Documented residuals (out of the two defect classes — NOT fixed)

* The bare article "a" riding into an EN span after a loan
  ("meeting a csapattal" → en 'meeting a') — the token is ambiguous
  with the English article "a"; a separate decision class requiring
  its own forensic evidence.
* Accent-free DISHARMONIC Hungarian content words (fiú/kavics/
  "napirend"-class) — invisible to the harmony guard.
* Sentences with NO Hungarian function words at all ("A meeting fontos
  volta meglepett.") still host-flip via article counting — content
  words are not table-able without a lexicon (operator-forbidden).
* "break a leg" zero-evidence idiom class — unchanged documented
  limitation.

## 4. Regression coverage

New focused deterministic matrix: `tests/unit/test_tts_n1_n2_matrix.py`
(16 tests — N1: the operator case + adjective/verb/noun/equivalent
contexts/multiple-loans budget accounting/leading mirror/differential
protections; N2: the operator case + audit variant + accent-free
function-word hosts + several loans + accented controls + the
"touch base" control + pure-EN/pure-HU differentials).

## 5. Performance (hot path) `[PROVEN — measured]`

`tools/perf_bench.py`: detect_language + segment_language_spans over
the 22-sentence N1/N2 corpus, 200 repeats (4,400 samples):

| | median | p90 | mean | max |
|---|---|---|---|---|
| before | 46.36 µs | 61.05 µs | 55.77 µs | 20,336 µs (one-off scheduler artifact) |
| after | 49.76 µs | 64.34 µs | 59.16 µs | 16,377 µs (same artifact class) |

+3.4 µs median (+7.3%) on a corpus deliberately saturated with budget
candidates; the tie-break is lazy (non-tie chunks pay zero). The
detector remains microsecond-class — no statistical or LLM mechanism
was introduced.

## 6. Full gate `[PROVEN]`

Full release gate on the fixed tree: **GREEN** — 1,537 tests
(floor 1,090; +16 = the new matrix), **0 regressions**, 2 pinned
sandbox env-gap failures (the documented baseline subset), 30 skipped,
deep validation PASS [embedding, memory-semantics, release,
runtime-deps, vad] (skip 15), fingerprint 28749983ae978d6e… (349 files).

## 7. Release recommendation

The fix corrects field-visible TTS mispronunciation classes (Hungarian
words read with English phonetics; whole Hungarian sentences on the
English voice) — the same severity class as the v0.10.4/v0.10.5
releases, which each shipped as their own gated patch release per the
repository's standing policy. The N1/N2 fix is a new production change
on top of v0.10.6 and is NOT bundled into it: **release as v0.10.7**
through the v0.8.1 gate-order (version metadata first, full gate on the
bumped tree, fingerprint-bound build), with the standard recovery
package and mirror synchronisation.

## 8. Artefacts

* `tools/reproduce.py` — the pre/post reproducer
* `tools/perf_bench.py` — the hot-path benchmark
* `evidence/reproduce_prefix.txt` / `reproduce_postfix.txt`
* `evidence/prefix_matrix_run.txt` — the 9 pinned pre-fix failures
* `evidence/perf_before_fix.json` / `perf_after_fix.json`
* `tests/unit/test_tts_n1_n2_matrix.py` — the regression matrix
* Production change: `app/text_utils.py` (harmony guard, "holnap" entry,
  tie-break refinement) + two consciously corrected documented-FP pins
  in `tests/unit/test_tts_code_switching.py`
