# TASK A/B — ASR Exit Gate + Performance Summary (v0.6.0)

**Scope**: modular ASR architecture (TASK A) + UI observability (TASK B).
**Baseline**: v0.5.2 @ `bb144fd` (Memory Safety Gate — FROZEN, untouched by this work unit).
**Decision authority**: measured evidence on the identical corpus (`data/asr_bench/`), not reputation.

---

## 1. Forensic root causes (Phase 0+1) — CLOSED

| Finding | Evidence | Verdict |
|---|---|---|
| Audio conversion chain (browser 48 kHz track → 96 kHz ctx → 24 kHz wire → 16 kHz server) | Pearson r=0.998067, RMS err 0.0034 vs signal RMS 0.0546, duration preserved exactly 9.984 s | **TRANSPARENT at 96 kHz ctx** — NOT the broken boundary |
| Silero VAD fed 512-sample windows WITHOUT the official 64-sample rolling context | production feed prob_max=0.0031 (0/312 frames speech) vs official-context feed prob_max=1.000, 235/312 frames, 2 natural segments (2496+3520 ms) on the SAME real capture | **ROOT CAUSE #1 (FACT)**: deaf VAD → v0.4.14 energy-fallback hack → fragmented ~320–384 ms "utterances" |
| Qwen3-ASR-0.6B integration on the same valid audio | field result "让让让让让。" (repetition garbage); independent cloud ASR reads the same file as valid Hungarian | **ROOT CAUSE #2 (FACT)**: Qwen integration fails on valid speech — engine retired, not "tuned" |

Full report: `data/asr_forensics_report.json`. Real field specimen: `upload/asr_test_last.wav`
(16 kHz mono PCM16, 9.984 s, RMS 0.0546, peak 0.4696 — the v0.4.19 Windows capture).

---

## 2. Performance summary (Phase 15, measured on the sandbox CPU)

Both engines driven through the **production contract** (`scripts/asr_benchmark.py`,
identical corpus, same `AudioBuffer → transcribe` path the web server uses).

| Metric | Parakeet TDT 0.6B v3 | Nemotron 3.5 ASR Streaming 0.6B |
|---|---|---|
| HU WER mean (6 HU files) | **17.1 %** | 43.4 % |
| HU CER mean | **8.8 %** | 20.1 % |
| EN WER | **15.4 %** | 23.1 % |
| Real Windows capture WER | **11.8 %** | 41.2 % |
| hu_with_silence WER | **0.0 %** | 55.6 % |
| Empty outputs | 0 / 9 | 0 / 9 |
| Stability (identical re-runs) | 9 / 9 | 9 / 9 |
| Load time | 11.5 s | 14.7 s |
| RSS delta | +472 MB | +469 MB |
| Offline RTF (aggregate, CPU) | 0.80 | 0.80 |
| True streaming | no (contract-honest: start/feed/finish raise) | **yes** (cache-aware, verified end-to-end: 4.47 s file → streamed decode, partials via TextIteratorStreamer) |

Evidence: `data/asr_benchmark_parakeet.json`, `data/asr_benchmark_nemotron.json`.

**Decision (measurement-based): `parakeet` is the production default** (`asr.engine: parakeet`,
`ASR_ENGINE` env override). 2.5× better Hungarian accuracy on the identical corpus at the
same CPU RTF. `nemotron` stays registered + selectable with its genuine streaming capability
documented — its CPU RTF for streaming is unusable in production (RTF 16.5 measured on the
GPU-less sandbox), CUDA is its intended execution config.

Known accuracy notes (documented, not hidden): parakeet writes digits where the reference has
number words ("142" vs "száznegyvenkét" — a normalisation artifact in WER scoring, CER 0.39
on that file), and degrades on 1.08 s ultra-short clips (WER 0.5, CER 0.0 — segmentation
floor). Neither is a hallucination or a repetition failure.

---

## 3. The 16-item ASR Exit Gate

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Engine interface `load/unload/status/transcribe` (+ `start/feed/finish` for genuinely streaming engines), one authoritative path | **PASS** | `app/asr_core.py` protocol; `app/asr_parakeet.py`, `app/asr_nemotron.py` |
| 2 | `AsrResult` (text/language/duration/model_id/engine_id/status/error/…) is the ONLY result contract crossing engine→pipeline/UI | **PASS** | `asr_core.AsrResult.to_dict()`; `web_server` `user_transcript.asr`, `asr_empty.asr`; UI renders it engine-agnostically |
| 3 | `AudioBuffer` canonical contract (16 kHz mono float32, validate/metrics, from_pcm16/from_wav); model-specific preprocessing stays INSIDE adapters | **PASS** | `asr_core.AudioBuffer`; processors used only inside the two adapters |
| 4 | VAD: Silero is the single decision path, official 64-sample context feed contract, energy/gain fallback REMOVED (not tuned) | **PASS** | `app/vad.py` SileroVad (576-sample model windows); `FusedVad` deleted; `vad_energy_fallback` config + `VAD_ENERGY_FALLBACK` env removed |
| 5 | Zero semantic fallbacks in the ASR stage (no partial-join on final failure, no engine switch, non-empty ≠ success) | **PASS** | `web_server._on_vad_frame` speech-end block: `transcribe`/`finish` → structured result; empty turn only on explicit failure |
| 6 | Qwen removed from production: not in `MODEL_REGISTRY`, not imported by any production module | **PASS** | registry = parakeet+nemotron only; `app/asr.py` retained ONLY as the clearly-marked non-production migration module for its historical tests (docstring + config comment say NON-PRODUCTION) |
| 7 | Explicit selection `ASR_ENGINE`/`ASR_DEVICE`/yaml; unknown id or load failure = loud structured error, NEVER auto-switch | **PASS** | `asr_core.select_engine`; `_UnavailableEngine` facade returns the same `ASR_MODEL_LOAD_ERROR` on every call; `ASR_DEVICE=cuda` without CUDA → `reason=cuda_unavailable` |
| 8 | Structured error codes (AUDIO_INPUT_ERROR, ASR_INPUT_ERROR, ASR_MODEL_LOAD_ERROR, ASR_PROCESSOR_ERROR, ASR_INFERENCE_ERROR, ASR_DECODE_ERROR) with stage/engine/reason/diagnostics | **PASS** | `asr_core.AsrError`/`AsrErrorCode`; `_asr_error_payload` maps exceptions into stage events |
| 9 | Both NVIDIA engines benchmarked through the official transformers-native integration on the shared corpus | **PASS** | §2 table; both JSONs |
| 10 | Production engine selected on measurements (not reputation) | **PASS** | parakeet default; decision recorded here + CHANGELOG |
| 11 | Streaming honesty: non-streaming engines do NOT fake streaming (start/feed/finish fail loudly); nemotron's streaming is the real cache-aware path | **PASS** | `asr_parakeet.start` raises `AsrError(streaming_unsupported)`; `asr_nemotron` queue-bridge + exact official mel-chunk sizes (1+8·lookahead trimmed, 8·(lookahead+1)) |
| 12 | Empty output is a first-class visible outcome (`asr_empty` + `stage asr/failed`), never a silent drop | **PASS** | `web_server` speech-end block; UI `asr_empty` toast with reason |
| 13 | Behavioural tests: unit gate green after the contract migration (3 stale tests fixed to the new contract, not weakened) | **PASS** | 627 unit tests, 0 failures, 3 skips (env-profile guards); integration gate re-run |
| 14 | UI observability: ASR transcript persisted in chat as a user turn (with generic asr dict), live CHAIN states MIC→VAD→ASR→VoiceMEM→E5→LLM→TTS, late stages never DONE after an early failure | **PASS** | `web/voicemem.html` `user_transcript` handler, `CHAIN_LIVE`/`renderChainFromLive`/`renderChainErrorLine` |
| 15 | UI hygiene: partials are transient (input line, no chat spam); MIC diagnostics + LLM panels collapsible with bounded internal scroll; nothing pushed out of the viewport | **PASS** | `partial_transcript` → `s.ui.input` only; `.micdiag .col-body{max-height:240px}`, `.llm-col-body{max-height:260px}`; `.center{min-height:0}`, `.p-in{flex:0 0 auto}` |
| 16 | Browser-verified end-to-end (real mic uplink frames → VAD → ASR → turn → answer → TTS) + landing page + lint + dev.log clean | **PASS** | e2e fake-mic harness `tests/e2e_fake_mic.js` through headless Chromium: `logs/web-server.log` full chain at 22:06 (`speech start → memory 8 hits → e5 → llm first token 15 ms → tts 5 chunks → answer done 474 ms`); screenshots `voicemem-v060-micturn-{desktop,mobile}.png`; ESLint clean; dev.log: no runtime errors on `/` |

**Memory Safety freeze (constraint check):** vendor tree untouched this unit (`vendor/` files
have zero diffs in the v0.6.0 commit); no schema/UPDATE/DELETE/supersession/retrieval
changes; `test_memory_safety.py` re-run in the v0.6.0 gate.

---

## 4. What was NOT done (deliberate)

* **Whisper Large V3 Turbo CT2** — future candidate only, per the work order; not implemented.
* **24 kHz→16 kHz server resample quality (P3 nit)** — measured transparent at the field's
  96 kHz context (r=0.998); the 44.1 kHz fractional-ratio linear interpolation is a known
  weakness left as documented P3, not a field failure.
* **Nemotron CUDA benchmark** — no GPU in this sandbox; CPU streaming RTF (16.5) honestly
  recorded; CUDA is its production execution config (`ASR_ENGINE=nemotron ASR_DEVICE=cuda`).

## 5. Engine selection card (authoritative registry)

| engine_id | model | streaming | production |
|---|---|---|---|
| `parakeet` | nvidia/parakeet-tdt-0.6b-v3 @541d1f99 (CC-BY-4.0) | no | **default** |
| `nemotron` | nvidia/nemotron-3.5-asr-streaming-0.6b @ea30d66d (OpenMDW-1.1) | yes (cache-aware) | selectable, CUDA target |
