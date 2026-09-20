# VoiceMem v0.10.5 — TTS Code-Switching Forensic + Fix
**Operator case:** "VOICE MEM — TTS CODE-SWITCHING FORENSIC + FIX" (the four-idiom report)
**Tree:** HEAD 6535497 (post v0.10.5 + memory-audit commits), VERSION 0.10.5
**Date:** 2026-09-19 (UTC)
**Discipline:** forensic first — NO phrase-detector change before the root cause was
proven; the fix below was implemented only after the trace reproduced the failure.

---

## 1. Baseline and scope

| Item | Value |
|---|---|
| Repo | `/home/z/my-project/voicemem-agent`, git `main` @ `6535497` |
| Version | 0.10.5 (no version bump in this task — operator order is fix+tests+recovery, not a release) |
| Changed production code | `app/text_utils.py` ONLY (one function extended, one helper added, two constants) |
| Changed tests | `tests/unit/test_tts_code_switching.py` (+20 tests, 10 subtests) |
| Touched but NOT part of the fix | `audit/VoiceMEM_tts_codeswitch_v0105_forensic/` (this package), `releases/gate_record.json` (gate rerun) |
| Not changed | `app/tts_supertonic.py` (engine), `app/web_server.py`, `app/pipeline.py`, `app/voice_settings.py`, chunking, Supertonic SDK, any ASR/LLM/memory path |

The four reported idioms: **"cut to the chase"**, **"hit the ground running"**,
**"burn the midnight oil"**, **"break a leg"**.

---

## 2. The traced pipeline (every stage, production code)

The web speak path (the production path the operator hears):

```
raw LLM deltas
  → SentenceStream.add_delta(delta)                    [web_server.py:3210; params 24/80, config.py:197-198]
  → chunk
  → normalize_for_speech(chunk)                        [web_server.py:3364]
  → detect_language(chunk)            = HOST_LANGUAGE  [web_server.py:3367]
  → VoiceSettings.resolve(host)       = SELECTED_VOICE [web_server.py:3371-3380; voice_settings.py:239-253]
      · mode auto → per-language preset by CHUNK language (this also logs the
        "TTS voice: F1 · … (English, mode auto)" diag line)
  → segment_language_spans(chunk, host) (auto mode only) [web_server.py:3383-3385]
      · level 1: quoted/parenthesized regions          [text_utils.py:481-501]
      · level 2: unquoted phrase runs (_phrase_runs)   [text_utils.py:409-462]
  → per span: tts.synthesize(span_text, span_lang, length_scale, voice_id)
                                                       [web_server.py:3389-3394]
  → SupertonicTtsEngine._synthesize_f32: lang = _LANG_MAP[span_lang] ("hu"/"en")
                                                       [tts_supertonic.py:259-285]
  → engine.synthesize(..., lang=lang)  = ACTUAL TTS CALL (ONNX, per span)
```

The CLI path (`pipeline.py:685-741`) is the same logic without the
`normalize_for_speech` first step (pre-existing divergence, out of scope).

**Forensic instrumentation** (`tools/forensic_trace.py`): drives the REAL
`SentenceStream`/`normalize_for_speech`/`detect_language`/`VoiceSettings`/
`segment_language_spans` with a recording TTS stub; the ~10-line glue replicates
`web_server._synthesize_chunk` line-for-line (quoted in the script header).
Per chunk it logs the operator's required fields: `TEXT`,
`HOST_LANGUAGE`, `DETECTED_ENGLISH_SPANS`, `SPAN_LANGUAGE`, `ROUTED_LANGUAGE`,
`SELECTED_VOICE`, `FINAL_TTS_TEXT` — plus the exact `mode auto` diag line.
Three streaming regimes per sentence: 3-char deltas, 8-char deltas,
whole-sentence (the no-split control).

---

## 3. FORENSIC RESULTS — the 10 mandatory sentences

Full evidence: `evidence/forensic_trace_before_fix.{json,txt}` (pre-fix) and
`evidence/forensic_trace_after_fix.{json,txt}` (post-fix), 30 case×regime runs each.

### 3.1 Pre-fix state (the reproduced defect)

| Case | Sentence | Pre-fix routing | Verdict |
|---|---|---|---|
| T1 | "Let's cut to the chase." | 1 chunk, host=en, 1 EN call | **PASS** (whole chunk EN) |
| T2 | "Sometimes you just have to hit the ground running." | 2 chunks, both host=en | **PASS** |
| T3 | "We had to burn the midnight oil." | 1-2 chunks, host=en | **PASS** |
| T4 | "Before the show, everyone said break a leg." | 2 chunks, both host=en (chunk 2 "everyone said break a leg." → en 2-0: "the"+"a" vs 0 HU — "a" is NOT a HU stopword, only "az" is) | **PASS** |
| T5 | "Most már tényleg cut to the chase, és menjünk tovább." | host=hu; EN span = **"to the chase,"** only — **"cut" stayed in the HU span** (whole-sentence); under streaming the 24-char first window cut after "cut to", so chunk 1 "Most már tényleg cut to" was **all HU** and chunk 2 carried "the chase" EN | **FAIL — idiom amputated** |
| T6 | "Ma először checkeljük a resultot, … next step-ről." | host=hu, zero EN evidence ("checkeljük" HU by diacritic, "statuszt" HU by "sz" digraph, "resultot"/"next"/"step" neutral) → 1 HU span | **DESIRED** (loanwords stay host) |
| T7 | "A mai test során …" | same as T6; "test" ambiguous but "során" HU | **DESIRED** |
| T8 | "Holnap lesz a meeting, utána pedig catch up." | host=hu; "meeting" = single word → stays HU; "catch up" → EN run | **DESIRED** (the exact differential the operator asked for) |
| T9 | "Touch base után megbeszéljük a projektet." | "Touch base" → EN; "projektet" stays HU | **DESIRED** |
| T10 | "Ez a projekt fontos, de most nem akarok meetinget." | all HU ("meetinget" single word) | **DESIRED** |

Supplementary probes (same class, not in the 10): in HU context
"hit the ground running" routed **EN "the ground running" + HU "hit"**;
"burn the midnight oil" routed **EN "the midnight oil" + HU "burn"**;
"break a leg" carried **zero orthographic evidence → all HU**.

### 3.2 A–E classification of where the pipeline breaks

* **A) English span not recognized — YES, for "break a leg" only.**
  break/a/leg are ALL signal-free ("a" is deliberately not an EN word-stopword:
  it is the Hungarian definite article; the anti-shredding rule from v0.10.4).
  No evidence → no run → no quoted-region split either.
* **B) Recognized but not segmented as a span — YES, THE primary defect.**
  `_phrase_runs` started runs AT the first foreign-evidence word. The idiom
  heads "cut"/"hit"/"burn" are orthographically signal-free, so the run began
  at "to"/"the" and the head was amputated into the host span. The idiom was
  spoken half-Hungarian: HU("… cut ") + EN("to the chase").
* **C) Wrong voice/language routing — NO.** Every detected span routed to the
  correct Supertonic language; the voice stays the chunk's resolved preset by
  design ("one speaker quoting a foreign phrase", pinned by
  `test_voice_continuity_across_spans`).
* **D) normalize_for_speech corrupts — NO.** All ten sentences are
  ASCII-normal; the mapping is identity here (verified: `FINAL_TTS_TEXT`
  equals the normalized chunk text in every call).
* **E) Synthesis/output defect — NO.** The recording stub (and the pinned
  `data/supertonic_validation` report for the real engine) shows the exact
  span text+lang reaching the engine call. The engine is not touched.

**Compounding artifact (documented, not fixed):** the SentenceStream
first-chunk window (24 chars, the low-TTFB design) cuts at the last space
before the limit, language-blind. Any foreign phrase straddling the window
splits across chunks (T5 under streaming: "cut to" | "the chase"). This
affects "touch base" today equally — it is a pre-existing chunker property,
not an idiom-class regression; fixing it needs phrase-aware chunking
(operator-forbidden: "NE változtasd meg az egész chunking rendszert").

### 3.3 The idiom vs loanword distinction (operator §3)

Confirmed by the trace as the correct discriminator, now enforced by two
gates:

* **Clear multi-word English constructions** carry ≥1 positive English
  evidence word INSIDE the phrase ("to"/"the" function words; "chase" ch-;
  "ground" ou-; "midnight" gh-) → they form runs and (after the fix) absorb
  their signal-free heads.
* **Ambiguous single-word English loanwords** ("test", "check", "result",
  "status", "project", "meeting", and HU-morphologized "checkeljük",
  "resultot", "statuszt", "meetinget", "next step-ről") carry no run-forming
  evidence → single words NEVER re-route; multi-word sequences without
  positive evidence ("next step-ről") never form runs. T6/T7/T10 stay 100 %
  HU — unchanged by the fix (pinned by new regression tests).

---

## 4. "MODE AUTO" — the full decision path (operator §4)

"English / mode auto" in the log is the per-chunk diag line
`web_server.py:3377-3380`:

```
TTS voice: F1 · Nyugodt női (English, mode auto)
```

`mode` is `VoiceSettings.mode` — a VOICE-SELECTION mode, persisted in
`config/voice_settings.json`, default `auto` (`voice_settings.py:52-53`).
Decision path with `mode auto`:

1. **phrase detection:** mode auto is the GATE —
   `web_server.py:3384-3385` / `pipeline.py:720-721` call
   `segment_language_spans` ONLY when mode is auto (or no voice settings).
   Forced `hu`/`en` skip segmentation entirely (explicit single-language
   override, pinned by `test_forced_hu/en_mode_keeps_single_language`).
2. **voice selection:** `VoiceSettings.resolve(host)` maps the CHUNK-level
   detected language to the per-language preset (HU reply → `hu_voice`,
   EN reply → `en_voice`). The log line reports exactly this resolution.
3. **language routing:** `resolve()` returns the chunk language as the
   baseline; the per-span languages from segmentation override it per span.
4. **Supertonic language selection:** per SPAN —
   `tts_supertonic.py:266` `lang = _LANG_MAP[span_lang]` ("hu"/"en") —
   NOT per chunk, NOT per voice.
5. **chunk synthesis:** per-span calls, concatenated PCM, one ordered payload.

**Proven conclusion:** "mode auto" is neither correct nor incorrect per se —
it is a routing POLICY flag. The log line "(English, mode auto)" means only
"this CHUNK was detected English at chunk level and the EN preset was
selected". It says NOTHING about per-span routing inside the chunk: a
Hungarian-host chunk with an English idiom span still logs
"(Hungarian, mode auto)" while the idiom synthesizes with `lang=en`.
The mispronunciation the operator heard was never a "mode auto" problem —
the mode gated correctly; the DETECTOR (root cause §5) amputated the idioms.

---

## 5. BUG CLASSIFICATION (operator §5)

**Primary: SEGMENTATION BUG** — `_phrase_runs` had no leading-neutral
absorption: a run started at the first foreign-evidence word, so
signal-free English idiom heads ("cut", "hit", "burn") stayed in the host
span. Causal chain:

```
evidence-gated design (protects ambiguous loanwords)
  → run must START at an evidence word
    → "cut to the chase" run = "to the chase"
      → "cut" left in the HU span → HU grapheme-to-phoneme reads "cut"
        → the idiom is spoken half-Hungarian (the field report)
```

**Secondary: DETECTION GAP (documented, NOT fixed — operator-forbidden):**
"break a leg" has zero orthographic English evidence. Every mechanism that
would catch it is rejected: an idiom lexicon (not general), loosening the
"a"-is-Hungarian rule (shreds Hungarian), a new statistical detector (forbidden
"új language detection rendszert"). It routes correctly in English-host
chunks (T4 passes) and stays host in Hungarian chunks — pinned as a
documented limitation with a dedicated regression test so any future change
is a conscious decision.

**Not bugs:** routing (C), normalization (D), synthesis (E), state/ordering,
logging. The chunk-boundary split is a documented pre-existing chunker
property (§3.2), not introduced here.

---

## 6. THE MINIMAL FIX (operator §6)

`app/text_utils.py`, three additive pieces, no existing rule loosened:

1. `PHRASE_RUN_LEADING_BUDGET = {LANG_EN: 1, LANG_HU: 0}` — the mirror of
   the trailing budget (one word for EN, none for HU).
2. `PHRASE_RUN_LEADING_MIN_EVIDENCE = 2` — absorption requires ≥2
   foreign-evidence words INSIDE the run. This is the loanword shield:
   "touch base"/"catch up" (1 evidence) NEVER absorb their leading neutral
   ("Holnap touch base" keeps "Holnap" host — pinned).
3. `_absorb_leading()` — absorbs ONE immediately-preceding signal-free word
   into a formed run, with the guards:
   * host-evidence words block (unchanged semantics),
   * the candidate must be purely alphabetic (digits/URLs never absorb:
     "A 2026-os team meeting…" keeps "2026-os" host),
   * hyphen-trimmed (Hungarian-suffixed) candidates never absorb
     ("next step-ről" stays host),
   * a candidate ending in clause-final punctuation blocks
     ("…a cut. To the chase" — "cut." stays host), the exact structural
     mirror of the trailing rule's clause guard,
   * budget of one word, EN direction only (HU runs keep budget 0 —
     "We had a gyors tempó" keeps "a" with the EN host).

**Why it is general and application-agnostic:** it adds no lexicon, no new
detector, no language pair beyond the existing HU/EN machinery — it closes
an asymmetry between the trailing and leading sides of the SAME evidence-run
rule, discovered by the forensic trace.

**Explicitly NOT done (operator's NE list honoured):** no new language
detection system; no new TTS pipeline; no Supertonic engine change; no
chunking change; no loosened ambiguous-word rules (T6/T7/T10 byte-identical
before/after — zero routing change in all 29 of 30 case×regime combos).

**Residual, documented:** (a) "break a leg" in HU context stays host
(§5); (b) a phrase straddling the 24-char first-chunk window splits across
chunks (§3.2) — pinned by `SentenceStreamChunkBoundaryTests` as current
behaviour so any future change is conscious.

---

## 7. REGRESSION SUITE (operator §7)

`tests/unit/test_tts_code_switching.py`: 61 → 81 tests (10 subtests),
organized as:

* **The four idioms, explicit regression tests:** T1–T4 (pure EN, host
  routing) + the three evidence-bearing idioms embedded in HU sentences
  (whole-idiom EN spans, exact span sequences) + "break a leg" pinned both
  ways (EN host = single EN span; HU host = documented limitation).
* **The 10-sentence mandatory matrix:** per-case expected language
  sequences + text-preservation invariant; loanword battery (the containing
  span must be host); the clear-phrase/loanword differential (T8/T9).
* **Guard battery for the new rule:** single-evidence runs never absorb
  ("Holnap touch base"); digit tokens never absorb ("2026-os");
  hyphen-trimmed candidates never absorb (white-box `_phrase_runs` check);
  clause-final candidates block; sentence-initial heads absorb
  ("Kész. Cut to the chase"); documented leading false-positive class pinned
  ("Holnap see you later." — the mirror of the existing trailing-FP test).
* **Streaming chunker documentation tests:** the deterministic 24-char
  boundary of T5 and per-chunk routing of the residual.
* **Integration levels:** web `WebSession._synthesize_chunk` — the real
  per-span call sequence + voice continuity for T5/hit/burn; CLI
  `VoicePipeline._speak_chunk` — the same span sequence (the TTS mock
  records every `(text, language, length_scale, voice)` call).
* **Pre-existing categories re-verified green:** pure HU, pure EN, quoted EN,
  unquoted EN, punctuation, contractions, URLs/e-mails, multiple phrases per
  sentence, run cap, forced modes (the full prior 61-test battery, unchanged).

---

## 8. PERFORMANCE (operator §8)

From the 30-combo forensic matrix (before → after):

| Metric | Before | After | Δ |
|---|---|---|---|
| Total synthesis calls (10 cases × 3 regimes) | 59 | 59 | **0** |
| Detected EN spans | 15 | 15 | 0 (content changed in 1) |
| Routing changes | — | — | exactly ONE: T5 whole-sentence "to the chase," → "cut to the chase," |
| First-call text (first-audio driver) | "Most már tényleg cut " | "Most már tényleg " | 4 chars SHORTER (first audio equal or faster) |
| Segmentation+normalize+detect | 72.5 µs/sentence | 71.9-72.8 µs/sentence | noise (O(1) per run added) |

Engine-side reference (pinned `data/supertonic_validation/report.json`,
2026-09-13, real ONNX engine): ~1.3-2.0 s per synthesis call at ~14.3 chars/s
— three orders of magnitude above the segmentation cost. The fix adds NO
synthesis call and does not change any span count; total synthesized
characters per case are identical, so total synthesis latency is unchanged.
(The real engine could not be re-run in this sandbox — the ONNX assets are
absent after the documented sandbox state loss; the pinned validation data
stands as the engine reference.)

---

## 9. TEST GATE (operator §10) — honest accounting

| Layer | Result |
|---|---|
| `tests/unit/test_tts_code_switching.py` | **109 passed** (81 in-file + text_utils 28), 10 subtests, 0 failed, 1.4 s |
| `tests/unit/test_text_utils.py` | 28 passed |
| `tests/integration/test_pipeline_mock.py` | **47 passed**, 37.5 s |
| Adjacent suites (speaker, tts_supertonic, asr_language_mode, british_english) | **86 passed** |
| Full unit suite (pytest) | 1095 passed / 9 failed / 13 skipped |
| Official release gate (`scripts/run_release_gate.py`) | **RED — environment, NOT the fix** (see below) |

**Gate RED decomposition (proven, not assumed):** the identical gate run on
the PRE-FIX tree (both production files stashed) produces the **byte-identical
failure set**: 14 pinned env-gap (asr_modular Silero/Parakeet model files
absent — sandbox state loss; install-manifest venv identity; voicemem-bridge
venv extras) + 17 "not in baseline" failures = temporal-memory suites failing
in `setUpClass` on `from openai import OpenAI` (venv extras absent — same
class documented in the v0.10.5 worklog), `test_prompt_reduction` isolated
runner, and 2 release-zip tests (tree/zip drift that predates this task: the
v0.10.5 zip was built before the phase-2 tests were added). **Zero failures
appear or disappear due to this fix.** Evidence: `evidence/gate_run_forensic.txt`.

---

## 10. ROOT CAUSE / FIX / WHY / REGRESSION PREVENTED

**ROOT CAUSE:** The unquoted phrase-run detector required a run to START at
a foreign-evidence word. English idiom heads are orthographically
signal-free (cut/hit/burn — no EN cluster, vowel pair, stopword or "q"), so
the run began one word late and the head was amputated into the Hungarian
host span: the idiom was synthesized half by the HU model, half by the EN
model ("cut" HU + "to the chase" EN). Zero-evidence idioms ("break a leg")
never formed a run at all.

**FIX:** leading-neutral absorption in `_phrase_runs` — one signal-free word
immediately before a run is merged into it, gated on ≥2 foreign-evidence
words in the run, alphabetic-only candidates, no host evidence, no
clause-final punctuation, budget 1 (EN) / 0 (HU) — the structural mirror of
the existing trailing budget with the evidence gate as the loanword shield.

**WHY IT WORKS:** it restores the head word to the same evidence run without
touching any detection table: the three evidence-bearing idioms now route as
whole EN spans; every ambiguous-loanword protection is provably untouched
(the same gate that fixes the idioms is what blocks "Holnap touch base");
synthesis call counts are identical (59→59) because absorption only MOVES a
word between two adjacent spans of the same chunk.

**WHAT REGRESSION IT PREVENTS:** the exact field report — HU-embedded
English idioms spoken half-Hungarian. The regression suite additionally pins
the two documented residuals (zero-evidence idioms; chunk-boundary splits)
and the full prior 61-test battery, so both over-detection and
under-detection regressions are caught by CI.

---

## 11. Files

* Production: `app/text_utils.py` (+63 lines: 2 constants, 1 helper, run-start extension, docs)
* Tests: `tests/unit/test_tts_code_switching.py` (+20 tests, 10 subtests)
* Forensic package: `audit/VoiceMEM_tts_codeswitch_v0105_forensic/`
  * this `REPORT.md`
  * `TEST_RESULTS.md` (verbatim suite outputs)
  * `WORKLOG.md` (task log)
  * `tools/forensic_trace.py` (the reusable forensic harness)
  * `evidence/`: before/after trace JSON+TXT, gate run, `SHA256SUMS.txt`
