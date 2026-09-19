# TTS code-switching, phase 2: UNQUOTED embedded English phrases (v0.10.5 field gap)

- **Date**: 2026-09-18
- **Scope**: TTS span segmentation only (`app/text_utils.py`). LLM/ASR untouched,
  Supertonic engine/SDK untouched, voice pipeline untouched.
- **Trigger**: production field report on v0.10.5 — the quoted-phrase fix
  (v0.10.4/5, audit `VoiceMEM_tts_codeswitch_v0104`) works, but **unquoted**
  embedded English phrases are still read with the Hungarian model.
  Observed: `SEE YOU LATER` spoken with Hungarian grapheme values
  ("se ju..."); same for `touch base`, `catch up`.
- **Version policy**: no VERSION bump (per order). Change ships in-tree;
  release packaging happens later.

---

## 1. Production log audit (2026-09-18, sessions ~19:31 / 19:33 / 19:34 / 19:35)

### 1.1 What is available locally

| Source | Coverage (UTC) | TTS language decisions | LLM reply text |
|---|---|---|---|
| `logs/web-server.log` | 2026-09-18 11:24 → **16:37** (ends) | yes — `[chain] TTS voice: F1 · Nyugodt női (Hungarian, mode auto)` per synthesis call | **no** (INFO level logs lengths only: `llm done (205 chars …)`) |
| `data/emotion_demo_log.jsonl` | up to 16:38 | no TTS decisions (emotion/ASR pipeline) | transcripts of user audio only |
| `.zscripts/s8_session_record.jsonl` | 12:15 → 12:22 | no TTS | no (S8 latency work) |
| `/tmp/vm_web_e2e_*`, `/tmp/t_v10_*` | 16:38–16:39 | test fixtures of the v0.10.5 release gate | mock replies only |

The reported sessions at **19:31–19:35 local (Europe/Budapest) = 17:31–17:35 UTC**
ran on the **user-side production install**, after the sandbox log had ended
(16:37:40 UTC, last entry). No per-session artifacts for 17:31–17:35 UTC exist in
this sandbox (verified: no file in `/home/z`, `/tmp` modified in that window).
The sandbox-side TTS decision log that DOES exist for 2026-09-18 is the v0.10.5
release-verification E2E at 16:30 UTC, which exercised the **quoted** case:

```
16:30:21,539 tts start (chunk 1, 28 chars)
16:30:21,539 TTS voice: F1 · Nyugodt női (Hungarian, mode auto)   <- host prefix
16:30:21,686 TTS voice: F1 · Nyugodt női (Hungarian, mode auto)
16:30:21,700 TTS voice: F1 · Nyugodt női (English, mode auto)    <- quoted EN region
16:30:21,859 TTS voice: F1 · Nyugodt női (Hungarian, mode auto)   <- host suffix
16:30:21,909 tts done (4 chunks, 421 ms)
```

This confirms the v0.10.5 fix works **exactly when the phrase is set off by
quotes** — and it confirms the mechanism of the gap reported now: every
UNQUOTED word sequence is one host span.

### 1.2 Correlation method (why the failure is still fully provable)

The reply text is not logged at INFO level, so the correlation between LLM
reply text and TTS decisions is reconstructed from the **deterministic
decision path** plus a **local reproduction on the exact reported sentences**
(`tools/repro_unquoted_codeswitch.py`, evidence
`evidence/repro_before_fix_20260918.txt`):

- the chunk-level language (host) comes from `detect_language(chunk)`;
- the span cut points come from `segment_language_spans(chunk, host)`;
- both functions are pure and dependency-free, so their exact production
  behaviour on any given sentence is reproducible offline, byte-for-byte.

Reproduction on v0.10.5 HEAD (`b7160ce`), sentences from the field report:

| Case | Sentence | host | spans | decision |
|---|---|---|---|---|
| F1 | `A "touch base" egy gyakori angol kifejezés.` | hu | 3 | quoted region → **en** (v0.10.5 fix OK) |
| F2 | `Szeretnék egy gyors touch base-t veled.` | hu | **1** | **whole chunk hu — BUG** |
| F3 | `Szerintem később catch up-olhatunk.` | hu | **1** | **whole chunk hu — BUG** |
| F4 | `Rendben, see you later.` | hu | **1** | **whole chunk hu — BUG** |
| F5 | `Rendben, SEE YOU LATER!` | hu | **1** | **whole chunk hu — BUG** |

This matches the field report exactly: Supertonic-HU grapheme-to-phoneme reads
`SEE YOU LATER` with Hungarian letter values → "se ju…". F2–F5 are the
reported production failures; F1 proves the quoted path is not the regressor.

### 1.3 Which generated phrases were synthesized HU vs EN (2026-09-18)

- **EN (correct)**: phrases inside paired quotes/parentheses (16:30 E2E
  session, per-call voice log above); whole replies that are pure English.
- **HU (wrong, the production gap)**: `touch base`, `catch up`,
  `see you later` / `SEE YOU LATER` when **unquoted** inside a Hungarian
  reply — i.e. the natural LLM style "Rendben, see you later.",
  "Szeretnék egy gyors touch base-t veled.", "Szerintem később catch
  up-olhatunk."

---

## 2. Exact current failure mechanism

`segment_language_spans` (v0.10.4/5) cuts the chunk ONLY at paired
set-off delimiters (`_scan_regions`: `"…"`, `'…'`, `(…)`). For text outside
those regions it emits **one span labelled with the host language — regardless
of the words' languages**. The v0.10.4 audit documented this as deliberate:

> "unquoted code-switching ("touch base-elni" without quotes) stays with the
> host language: word-level runs cannot place phrase boundaries without a
> lexicon."

That policy is now insufficient for the actual conversational use case:
the LLM inserts English formulae **without** quotes in most turns (quotes
appear in maybe 1 of 10 code-switches in casual speech).

Failure trace, `Szeretnék egy gyors touch base-t veled.`:

1. `detect_language`: `szeretnék`(á,é +3) `egy`(HU stopword +2) `gyors`(0)
   `touch`(0) `base-t`(0) `veled`(0) → **hu 5 : en 0** → host `hu`.
2. `_scan_regions`: no quotes/parens → no regions.
3. `segment_language_spans`: single span `("Szeretnék egy gyors touch
   base-t veled.", "hu")`.
4. `_synthesize_chunk` calls `tts.synthesize(span, "hu", …)` once.
5. Supertonic HU G2P reads the English graphemes with Hungarian letter
   values → unintelligible ("táucs bézété…").

Root cause class: **the unit of language routing (the set-off region) is
narrower than the unit of code-switching (the multi-word phrase)**.

## 3. Failure classification (ordered by the operator's taxonomy)

**A. Explicit quoted foreign spans** — handled since v0.10.4/5 (F1 above).
Not part of this gap; regression-guarded by the existing 42-test suite.

**B. Unquoted but clearly foreign multi-word phrases** — THE production gap
(F2–F5). Fix scope of this change: `touch base`, `catch up`, `see you
later` must route to EN when the host is HU, without quotes.

**C. Ambiguous loanwords (single words)** — `projekt`, `meeting`, `hazai`,
`tea`, `autó`. Policy (confirmed by the operator): **stay with the host
language unless strong evidence**. A single neutral word never re-routes —
this is the anti-over-detection requirement, and it also keeps Hungarian
loans written without diacritics safe.

**D. Legitimate Hungarian text containing English-looking character
patterns** — e.g. `A technika fejlődik` (ch inside `technika`), `hazai`
(English-looking "ai"), `tea` ("ea"), `autó` ("au"). Must stay HU. Guards:
`ai`/`au`/`ea`/`eu` are NOT English vowel-cluster evidence (v0.10.4 table
design); single words never re-route (new rule, §4); a Hungarian-evidence
word always terminates a run.

## 4. Revised segmentation policy: evidence-gated phrase runs

### 4.1 Design goal

Route unquoted foreign **phrases** (category B) to the foreign model while
keeping categories C and D with the host — **without** adding an English
dictionary (compact evidence mechanism only), and without fragmenting
ordinary sentences.

### 4.2 The phrase-run mechanism

A **phrase run** is a maximal sequence of consecutive words inside a
host-language span that satisfies ALL of:

- **R1 multi-word**: at least 2 whitespace-words (a single word NEVER
  re-routes — kills category C and D single-word cases).
- **R2 strong foreign evidence**: at least one word carries positive
  English evidence — the same v0.10.4 signal tables, applied per
  subtoken: English contractions (`I'd`, `let's`), consonant clusters
  (`ch ck sh th ph wh gh qu`), vowel clusters (`ou ee oo oa ue ui ei`),
  non-HU English stopwords, or `q`. No dictionary, no phrase list.
- **R3 host consistency**: no word inside the run carries Hungarian
  evidence (diacritic / HU stopword / HU plain word / strong digraph
  `sz cs gy zs`). A Hungarian-evidence word always terminates the run —
  categories C and D can never be swept in.
- **R4 boundaries**: the run starts at the first foreign-evidence word
  (no leading neutral sweep) and may extend over at most
  **one trailing neutral word** — and only if the previous run word does
  not end with sentence-final punctuation (`. ! ? ;`). The trailing
  budget exists because English phrase bodies carry signal-free content
  words (`touch **base**`, `catch **up**`, `see you **later**` — "base",
  "up", "later" have no orthographic English signal on their own).
  Trailing budget is **1 for foreign=EN** and **0 for foreign=HU**:
  Hungarian words are orthographically distinctive (diacritics/digraphs),
  so they never need a neutral tail, while English content words do.
- **R5 hyphen rule**: a word containing an interior hyphen contributes
  only its pre-final-hyphen head to the run when that head is
  signal-neutral (budget word); Hungarian derivational/case suffixes
  (`-t`, `-olhatunk`, `-től`) stay with the host span. A head that itself
  carries foreign evidence stays whole (English hyphen compounds:
  `check-in`, `catch-up`).
- **R6 fragmentation cap**: if more than **3** runs are detected in one
  chunk, NO run switching happens for that chunk (the whole chunk stays
  host — today's behaviour). Pathological LLM output (word-lists,
  alternating slogans) must not turn one chunk into 10+ synthesis calls.

Quoted/parenthesized regions keep the v0.10.4/5 mechanism unchanged and
take precedence; runs are only detected in host-language text between
regions. Whitespace-only spans created by cutting are absorbed into the
adjacent span (never a synthesis call for pure whitespace). The hard
invariant `"".join(span texts) == text` (no character lost, added or
reordered) is preserved — playback ordering cannot change by construction.

### 4.3 Behaviour on the required matrix

| Sentence | host | spans | routing |
|---|---|---|---|
| `Szeretnék röviden beszélni veled az új projektről.` | hu | 1 | all HU (projekt = category C) |
| `A "touch base" egy gyakori angol kifejezés.` | hu | 3 | HU / **EN** quoted / HU (unchanged v0.10.5) |
| `Szeretnék egy gyors touch base-t veled.` | hu | 3 | HU `Szeretnék egy gyors ` / **EN `touch base`** / HU `-t veld.` |
| `Szerintem később catch up-olhatunk.` | hu | 3 | HU `Szerintem később ` / **EN `catch up`** / HU `-olhatunk.` |
| `Rendben, see you later.` | hu | 2 | HU `Rendben, ` / **EN `see you later.`** |
| `I'd like to touch base with you.` | en | 1 | single EN span (no HU evidence words) |
| `Let's catch up later.` | en | 1 | single EN span |
| `See you later.` | en | 1 | single EN span |
| `projekt` / `meeting` / `hazai` / `tea` / `autó` (in HU) | hu | 1 | HU — single words never re-route |

Hungarian suffix fragments (`-t`, `-olhatunk.`) remain with the host:
Hungarian morphology is pronounced correctly by the HU model and would be
butchered by EN G2P; the cost is a slightly choppy case-ending —
intelligibility of the English phrase (the actual reported failure) is
guaranteed.

### 4.4 What is deliberately NOT done

- **No English dictionary / phrase list.** Detection is orthographic
  evidence + stopword tables (all pre-existing, all word-level). Phrases
  with zero signal words (e.g. `follow up` in HU text — `follow` and `up`
  carry no orthographic English evidence) stay host: documented
  limitation, preferred over an unauditable lexicon.
- **No character-level or single-word switching** (operator requirement).
- **No changes to** `detect_language`, the voice resolver, the engine,
  the callers (`web_server._synthesize_chunk`, `pipeline._speak_chunk`
  already synthesize per span since v0.10.4/5), forced voice modes,
  barge-in, PCM concat, or the browser wire format.

## 5. False-positive risks

| Risk | Example | Mitigation | Residual |
|---|---|---|---|
| Hungarian loan with EN cluster + next neutral word | `chat ablak` (chat=ch) | needs BOTH words signal-free-of-HU + ≥1 EN signal + budget ≤1 | `ablak` read with EN phonetics — same documented worst case as the quoted path (v0.10.4); rare in practice |
| Sentence-initial stray English word + neutral HU word | `Kedd touch base` | leading budget 0 — run starts at `touch` | `Kedd` stays HU, correct |
| Neutral sweep across sentence boundary | `touch. Base?` | sentence-final punctuation blocks budget extension | run invalid (single word) → host |
| Alternating pathological text | 5+ runs | R6 cap: whole chunk stays host | none |
| English hyphen compound split | `check-in`, `catch-up` | evidence-bearing head stays whole (R5) | suffixed evidence verbs (`catch-up-olhatunk`) keep the suffix in EN — rare, intelligible |
| Proper noun + neutral word in EN host | `I visited Győr today` | foreign=HU budget is 0 → lone HU word forms no run | `Győr` read by EN model — standard foreign-name handling |
| Numbers/URLs/e-mails | `a 2026-os évkönyv` | neutral by `_classify_word` | none |

## 6. Expected latency impact

- **Segmentation cost**: O(words) pure-stdlib scan. v0.10.5 baseline was
  3.7–4.1 µs/call for pure chunks, 9.4 µs for mixed quoted. The run pass
  adds one tokenization of host spans: expected < 15 µs/call worst case —
  3 orders of magnitude below one synthesis call. Measured in
  `evidence/segmentation_latency_20260918.txt` (§7).
- **Synthesis calls**: pure HU/EN chunks (the vast majority): **1 call,
  unchanged**. A chunk with one embedded unquoted phrase: 1 → 3 calls
  (host prefix + phrase + host suffix). Same as the already-shipped quoted
  behaviour (16:30 E2E: 4 chunks → 421 ms total TTS).
- **First audio latency**: unchanged in the common layout — the FIRST span
  is the host prefix, so the first synthesis call still starts immediately
  with the same leading text; measured first-span times in §7.
- **Playback continuity**: unchanged mechanism — span PCM parts are
  concatenated into ONE payload per chunk (v0.10.4/5 invariant), ordering
  preserved by construction; only the language per segment differs.

## 7. Test matrix and measurement plan

Unit (extend `tests/unit/test_tts_code_switching.py`):
the 8 mandatory sentences + suffix variants + caps + punctuation
boundaries + budget edges + cap + hyphen compounds + whitespace absorption
+ text-preservation invariant battery + segmentation latency guard.

Integration (existing harness, no new fixtures):
per-span `(text, language)` call sequences via `WebSession._synthesize_chunk`
+ `VoicePipeline._speak_chunk` (MockTtsEngine), forced modes unchanged.

Engine measurement (this audit, `evidence/`):
real Supertonic CPU synthesis of F2–F4 old-vs-new — span count, synthesis
calls, per-call and total TTS latency, first-audio latency, concatenated
PCM non-empty (continuity proxy).

Release gate: full unit + integration suites; no regression vs. the
v0.10.5 GREEN baseline (1478 tests, 4 pinned env-gap).

## 8. Result summary (filled after implementation)

See §4.3 table for final behaviour, `evidence/repro_after_fix_20260918.txt`
for the reproduction matrix on the fixed tree, and
`evidence/synthesis_latency_20260918.txt` for engine measurements.
