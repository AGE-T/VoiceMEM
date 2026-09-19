# WORKLOG — Task: TTS CODE-SWITCHING FORENSIC + FIX (v0.10.5)

Operator case: "VOICE MEM — TTS CODE-SWITCHING FORENSIC + FIX" — the four
idioms ("cut to the chase", "hit the ground running", "burn the midnight oil",
"break a leg") mispronounced / mis-routed.

## Sequence

1. **Orientation.** Read the worklog (4229 lines, last entries: memory audit
   memperf-0105 complete, audit-only). Located the TTS pipeline: web path
   `web_server.py` `_synthesize_chunk` (3098-3213 speak worker, 3331-3411
   chunk synthesis), CLI path `pipeline.py` `_speak_chunk` (685-773), engine
   `tts_supertonic.py`, voice resolution `voice_settings.py`, detection +
   segmentation `text_utils.py`. Confirmed the existing 61-test suite has
   ZERO coverage of the four idioms (the coverage gap).
2. **Forensic harness BEFORE any code change.**
   `audit/VoiceMEM_tts_codeswitch_v0105_forensic/tools/forensic_trace.py` —
   drives the REAL SentenceStream (24/80), normalize_for_speech,
   detect_language, VoiceSettings (auto), segment_language_spans with a
   recording TTS stub; logs TEXT / HOST_LANGUAGE / DETECTED_ENGLISH_SPANS /
   SPAN_LANGUAGE / ROUTED_LANGUAGE / SELECTED_VOICE / FINAL_TTS_TEXT + the
   "mode auto" diag line, per chunk, for 3 streaming regimes.
3. **Reproduction (pre-fix).** T1-T4 (pure EN): PASS at every stage. T5
   (HU context): the idiom is amputated — EN span "to the chase," only,
   "cut" stays in the HU span (whole-sentence regime); under streaming the
   deterministic 24-char first window cuts after "cut to" so chunk 1 is all
   HU. Supplementary probes: hit/burn same class; "break a leg" carries ZERO
   orthographic evidence (all-HU routing). T6-T10: desired conservative
   loanword behaviour (verified word by word). A-E classification: B
   (segmentation) primary; A for zero-evidence idioms; C/D/E clean.
4. **"mode auto" investigation.** Traced to `voice_settings.py` (a voice
   policy flag, default auto, persisted) + the diag line web_server.py:3377.
   Effects documented: gates span segmentation (forced hu/en skip it); picks
   the per-chunk preset; per-SPAN Supertonic lang comes from the span, not
   the mode. The log line is chunk-level only — proven NOT a bug indicator.
5. **Root cause fixed minimally** (`app/text_utils.py`): leading-neutral
   absorption — `PHRASE_RUN_LEADING_BUDGET` (EN 1 / HU 0, mirror of
   trailing), `PHRASE_RUN_LEADING_MIN_EVIDENCE = 2` (the loanword shield:
   single-evidence runs like "touch base"/"catch up" never absorb),
   `_absorb_leading()` with guards: host-evidence stop, alphabetic-only
   (digits/URLs never absorb), hyphen-trimmed (HU-suffixed) candidates
   rejected, clause-final punctuation blocks. No detection table, no
   chunking, no engine, no new system touched.
6. **Verification matrix.** Protection battery run live before writing
   tests: "Holnap touch base" intact; "2026-os" host; "Ez nem touch.
   Base?" single span; "chat ablakot" FP unchanged; the three
   evidence-bearing idioms whole-EN; "break a leg" pinned as documented
   limitation.
7. **Regression suite** (+20 tests, 10 subtests): four idioms explicit
   (both hosts), 10-sentence mandatory matrix (language sequences +
   loanword battery + differential), guard battery, SentenceStream
   boundary documentation tests, web `_synthesize_chunk` and CLI
   `_speak_chunk` integration sequences. 109 passed (+ text_utils 28),
   pipeline_mock 47, adjacent suites 86.
8. **Performance.** 59→59 synthesis calls over the 30-combo matrix
   (Δ=0); exactly ONE routing change (T5 whole-sentence idiom whole-EN);
   first-call text 4 chars SHORTER; segmentation 72 µs/sentence
   (before=after within noise).
9. **Gate honesty.** Official release gate RED on BOTH pre-fix and post-fix
   trees with the byte-identical failure set (14 pinned env-gap + 17
   sandbox-state classes: temporal-memory `import openai` venv-extras,
   Silero/Parakeet model files absent, isolated runner, release-zip
   drift). Zero failures attributable to the fix. Evidence captured.
10. **Deliverables.** REPORT.md (16 sections incl. mode-auto decision path,
    A-E classification, causal chain, NOT-done list), TEST_RESULTS.md,
    WORKLOG.md, SHA256SUMS.txt, before/after forensic evidence, the
    reusable forensic harness.

## Status

FIX COMPLETE per the operator's seven-point definition: (1) reproduced
(forensic trace, both regimes), (2) root cause identified (run-start
evidence rule amputates signal-free idiom heads), (3) minimal fix landed
(one function + guards, no rule loosened), (4) the four idioms work in
their operator test contexts (T1-T4 green; three of four whole-EN in HU
sentences; "break a leg" documented limitation in HU context — unfixable
without a lexicon, pinned), (5) ambiguous-word behaviour not regressed
(T6/T7/T10 byte-identical, guard tests), (6) relevant suites fully green,
(7) recovery package built.
