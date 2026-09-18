# VoiceMEM TTS Code-Switching Audit — v0.10.4 field report

- **Date**: 2026-09-18
- **Trigger**: production bug report — English phrases embedded inside
  Hungarian sentences are synthesized by the Hungarian Supertonic model and
  become unintelligible; some mixed positions produce audible garbage.
- **Scope**: TTS path only (`response text → chunking → language detection →
  language selection → Supertonic normalization → synthesis → playback`).
  No LLM / ASR / runtime / voice-config changes. Supertonic 3, CPU provider,
  fully local — unchanged.
- **Production version NOT bumped** (stays 0.10.4 until sign-off).

---

## A. Current TTS segmentation behaviour

The full production TTS path (web streaming, `app/web_server.py`):

1. **Response text** — the LLM streams deltas; `reply += delta` accumulates.
2. **Chunking** — `app/text_utils.SentenceStream` (`app/web_server.py:3097`)
   buffers deltas and emits a chunk per completed *sentence*, or at a
   comma/space boundary when the buffer exceeds `tts_chunk_chars` (80;
   first chunk 24). A chunk is therefore one sentence or a sub-sentence —
   **language can only switch between chunks, never inside one**.
3. **Language detection** — per chunk, `detect_language(chunk)`
   (`app/web_server.py:3350`) returns ONE label for the WHOLE chunk.
4. **Language selection** — `VoiceSettings.resolve(language)`
   (`app/voice_settings.py:239`) maps the detected language to
   `(voice_id, spoken_language)`; in forced modes (`hu`/`en`) it REPLACES the
   detected language with the forced one.
5. **Supertonic input normalization** — `normalize_for_speech(chunk)`
   (`app/text_utils.py:179`) maps typographic characters („ " ' ' – …) to
   ASCII before detection/synthesis (v0.10.3 fix).
6. **Synthesis** — ONE call: `c.tts.synthesize(chunk, language, …)` →
   `SupertonicTtsEngine.synthesize` → SDK `engine.synthesize(..., lang="hu"|"en")`
   (`app/tts_supertonic.py:259-277`). The `lang` parameter selects the
   language model for the ENTIRE text of the call.
7. **Playback ordering** — a pump starts each chunk's synthesis as the LLM
   emits it (≤ `_TTS_MAX_PARALLEL_CHUNKS = 2` workers); a single ordered
   sender drains the per-chunk PCM queues strictly in chunk order
   (`app/web_server.py:3117-3176`). Ordering is queue-driven and cannot
   reorder.

The CLI pipeline path (`app/pipeline.py:_speak_chunk`, line 680) is the same
pattern with the same defect: detect once per chunk → one synthesis call.

## B. Current language detection behaviour

`app/text_utils.detect_language(text)` is a **whole-text, single-label**
heuristic (HU/EN):

- HU diacritic-bearing token: +3 HU; HU stopwords/plain words: +2 HU;
  EN stopwords: +1 EN; digraph density (sz/cs/gy/ny/ly/ty/zs in ≥50% of
  ≥3 tokens): +2 HU; any `q`: +2 EN; ties → HU iff any diacritic present,
  else EN; empty → EN.

**Can it provide span-level boundaries? NO.** It returns exactly one of
`"hu"` / `"en"` for the entire input. It has no notion of position, spans or
runs. The tables inside it (diacritics, stopwords, digraphs) are reusable as
*word-level signals*, but the function itself cannot answer "which part of
this text is English". A new, separate span-segmentation function is
required.

## C. Exact failure mechanism

`A "touch base" egy gyakori angol kifejezés.` forms ONE SentenceStream chunk
(one sentence). In `_synthesize_chunk`:

```
detect_language('A "touch base" egy gyakori angol kifejezés.')
  → tokens: a touch base egy gyakori angol kifejezés
  → HU: egy (+2 stopword) + kifejezés (+3 diacritic é) = 5
  → EN: a (+1 stopword) = 1
  → "hu"  (correct for the sentence as a whole — Hungarian dominates)
```

The single label `hu` is passed to `SupertonicTtsEngine.synthesize`, which
calls the SDK with `lang="hu"` for the whole chunk. Supertonic's Hungarian
grapheme-to-phoneme mapping then reads the English graphemes with Hungarian
letter values — "touch base" is rendered approximately as
*/tɒttʃ bɒʃɛ/ — phonetically wrong, unintelligible ("audible but
unintelligible sound"). The same mechanism inverts for embedded Hungarian
inside an English chunk. Adjacent chunks that differ in dominant language
(HU→EN→HU→EN in the production log) switch correctly — confirming selection
happens at chunk granularity only, and embedded spans inside a chunk are
impossible to express today.

**Root cause (one line)**: language is selected exactly once per TTS chunk
(one `lang` per `engine.synthesize` call), while the unit of code-switching is
the embedded *phrase*, not the chunk.

## D. Proposed segmentation algorithm

New pure-stdlib function in `app/text_utils.py`:

```
segment_language_spans(text, host_language) -> list[(span_text, span_lang)]
```

**Policy** (deliberately narrow — see rejected alternatives):

1. **Host language** = the existing chunk-level detection (unchanged), after
   `VoiceSettings.resolve` (forced modes keep their forced language and NO
   span splitting — the forced mode's documented contract).
2. **Explicit set-off regions are the only split candidates**: paired
   `"`…`"` quotes (all typographic forms normalized to ASCII first),
   paired `'…'` single quotes (with letter-boundary guards so contractions
   like `I'd`, `don't` are never treated as delimiters), and paired
   parentheses `(`…`)`.
3. **Region classification — evidence-gated**:
   - every word inside the region is classified by a *word-level* signal
     classifier (see below); counts per language give
     `ev_hu`, `ev_en`;
   - `foreign` = the language that is not the host;
   - `ev[foreign] == 0` → region stays with the host (no split);
   - `ev[foreign] > 0 and ev[host] == 0` → region becomes a foreign span;
   - both sides have evidence → the existing weighted `detect_language`
     arbitrates; it wins only if it picks the foreign language.
   The gate kills false positives where the quote carries **no** foreign
   evidence (e.g. `A "hazai" megoldás` — "hazai" is a native, signal-free
   Hungarian word; the old tie-break alone would have routed it to EN).
4. **Word-level signal classifier** (new tables, priority order):
   1. token contains a digit / `://` / `@` → **neutral** (numbers, URLs,
      e-mails never carry language);
   2. contains a HU diacritic (á é í ó ö ő ú ü ű) → **hu** (strongest —
      `chatelünk` is Hungarian despite `ch`);
   3. EN contraction table (`i'd, don't, i'm, we're, it's, let's, …`) → **en**;
   4. contains an EN consonant cluster `ch ck sh th ph wh gh qu` or `q` → **en**;
   5. contains an EN vowel cluster `ou ee oo oa ue ui ei` → **en**
      (`au`/`ea`/`eu`/`ai` are excluded — `autó`, `tea`, `euro` are Hungarian
      loans and `hazai`-class `-ai` adjectives are everyday Hungarian);
   6. HU stopwords / plain-word tables → **hu**;
   7. contains a safe HU digraph `sz cs gy zs` → **hu**
      (`ny/ly/ty` excluded — `only`, `really`, `city`, `money` are English);
   8. EN stopword table minus `a`, `is` (Hungarian function words) → **en**;
   9. otherwise → **neutral**.
5. **Segment assembly**: the region (including its delimiters) becomes one
   span; all text outside regions stays host-language; adjacent
   same-language spans merge; `"".join(span_texts) == text` is a hard
   invariant (no character is lost, added or reordered).
6. **Synthesis**: the caller synthesizes each span sequentially with its
   own `lang` but the SAME host voice (voice continuity — one speaker
   quoting a foreign phrase, not a different person), concatenates the PCM
   (per-span edge-trim keeps the vocoder's 0.35–0.9 s padding out of the
   middle of the sentence), and returns it as ONE chunk to the existing
   ordered sender — **playback ordering is untouched by construction**.

**Why quotes/parens only (rejected alternatives)**:

- *Unquoted foreign word runs* (e.g. `Szeretnék röviden touch base-elni
  veled…`): word-level runs cannot place phrase boundaries without a
  lexicon — `touch` is EN-signal-bearing but `base` is signal-neutral, so
  every joining rule either splits the phrase (`EN "touch" | HU "base"`) or
  swallows the following Hungarian (`EN "touch base-elni veled a projekt
  miatt"`). Hungarian loanwords (`technika`, `pech` contain `ch`) also false-
  positive. The task marks this input conditional; our policy therefore
  treats unquoted code-switching as host-language (today's behaviour,
  documented).
- *Character/letter-level switching*: explicitly forbidden by requirements
  (and phonetically catastrophic).
- *Cloud/ML span detectors*: forbidden (no cloud dependency).

## E. Expected edge cases

| Case | Behaviour |
| --- | --- |
| `A "hazai" megoldás` (quoted native, signal-free) | no split — host HU (evidence gate) |
| `A "technika" szó` (quoted HU loan with `ch`) | EN span (documented FP; rare; intelligible) |
| `The Hungarian word "projekt" is used here.` | no split — `projekt` is signal-neutral; EN reading ≈ HU pronunciation |
| `Az "I'm ready" azt jelenti…` | EN span (contraction table) |
| `A 'touch base' egy…` (single quotes) | EN span (letter-boundary guarded pairing) |
| Unmatched `"` (quote spanning chunks) | no region → host, graceful |
| `"3.14"` / `"https://x.com"` / `"a@b.com"` | neutral → no split, no exception |
| `A "touch" és "check" dolog` (two regions) | both EN spans, host text between |
| Forced voice mode (`hu`/`en`) | exactly today's behaviour (no splitting) |
| Pure HU / pure EN chunk | single span — byte-identical path to v0.10.4 |
| Empty region `""` / `()` | no words → no split |
| Nested quotes | greedy pairing degrades to sequential regions (conservative) |

## F. Latency impact

- **Segmentation itself** (measured, evidence/segmentation_latency_20260918.txt):
  3.7 µs pure 50-char chunk, 9.4 µs mixed 44-char 3-span chunk, 80 µs for a
  synthetic 362-char 19-span worst case — production chunks are ≤80 chars
  (`tts_chunk_chars`), so the real cost is single-digit microseconds.
- **Pure chunks** (the overwhelming majority): span list = `[(chunk, host)]`
  → exactly ONE synthesis call, as today. **Zero added latency.**
- **Mixed chunks**: N sequential Supertonic calls instead of 1 (N = spans,
  typically 3: host + foreign + host). Reference points from
  `data/supertonic_validation/performance.json` (2-vCPU sandbox, *not* the
  target machine): 17 chars ≈ 2.48 s, 75 chars ≈ 4.22 s synthesis — the
  foreign span is short (a phrase), so the added cost is ≈ one short-call
  synthesis (~2.5 s sandbox / substantially less on the target CPU), and it
  is amortized by the existing chunk pump (≤2 parallel chunk syntheses, the
  browser is usually still playing the previous chunk). First-audio latency
  is unaffected whenever the chunk begins in the host language (the host
  span synthesizes first; only the foreign phrase waits for its own call).
- **No engine change**: `SupertonicTtsEngine` and the SDK call signature are
  untouched — one chunk now makes 1–3 calls through the identical code path.

## G. Test matrix

For every case verify: (1) detected language / span boundaries, (2) selected
TTS language per span, (3) generated audio exists, (4) no unsupported-character
exception, (5) chunk ordering preserved.

| # | Input | Expected spans (text → lang) |
| --- | --- | --- |
| 1 | `Szeretnék röviden beszélni veled az új projektről.` | 1× HU |
| 2 | `I'd like to talk to you about the new project.` | 1× EN |
| 3 | `A "touch base" egy gyakori angol kifejezés.` | HU + EN(`touch base`) + HU |
| 4 | `A "catch up" hasonló jelentésű ebben a mondatban.` | HU + EN(`catch up`) + HU |
| 5 | `The Hungarian word "projekt" is used here.` | 1× EN (no split) |
| 6 | `Szeretnék röviden touch base-elni veled a projekt miatt.` | 1× HU (unquoted policy: host) |
| 7 | `The word "szeretnék" means "I would like" here.` | host detects **HU** (pinned HU-bias: one `szeretnék` = 5 > four EN stopwords = 4) → HU + EN(`"I would like"`) + HU; the pure EN-host case is covered by `The word "szia" means hello here.` → EN + HU(`"szia"`) + EN |
| 8 | `Az "I'm ready" kifejezés angolul…` + typographic `„…”`/`’` variants | HU + EN + HU; no exception |
| 9 | apostrophes, quotes, parentheses, commas, hyphens, numbers, `https://…`, `a@b.com` | neutral/host; no exception; text preserved |
| 10 | forced mode `hu` / `en` | single span, forced language |
| 11 | pure-EN quoted EN (`The "catch up" phrase`) | no split |
| 12 | `I’d` / `don’t` / `I’m` / `we’re` (U+2019) | tokenized whole; EN context stays EN |

Integration level: `WebSession._synthesize_chunk` (mock components — assert
the mock engine's recorded per-call `(text, language)` sequence, non-empty
PCM returned, text concatenation equals the normalized chunk) and
`VoicePipeline._speak_chunk` (CLI path, same assertions).
