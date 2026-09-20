# VoiceMEM — CURRENT UPSTREAM SELECTIVE MERGE AUDIT

**Date:** 2026-09-20
**Type:** Audit only. No merge, no rebase, no cherry-pick, no blind import.
Read-only (the single permitted side effect: a read-only `git fetch` of the
public upstream repository to refresh refs).
**Question:** which upstream changes since our pin are worth porting into the
0.10.5 baseline, and which must stay out?
**Method:** actual diff inspection of every commit since the pin
(`git show <sha>` per commit), cross-referenced against `VOICEMEM_PIN.json`
(21 local patches + 4 ported fixes) and the v0.10.3 upstream audit
(`audit/VoiceMEM_upstreamaudit_v0103/REPORT.md`, which reviewed the same
52-commit window ending at `6cacb3c`).
**Decision criteria:** functional benefit, architectural compatibility,
correctness, latency, concurrency impact, operational risk, maintenance cost,
validability — never popularity, commit size or recency.

---

## 1. UPSTREAM STATE (evidence)

| Item | Value | Evidence |
|---|---|---|
| Pin (our vendor base) | `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` | `VOICEMEM_PIN.json`; gitlink `tool-results/voicemem-src` `[PROVEN]` |
| Fetch (2026-09-20) | **SUCCESS** — exit 0 | `git fetch origin` in the upstream clone `[PROVEN]` |
| origin/main HEAD | `6cacb3c1e7fca2679f3bf95451f6b480efd9a783` — "Add files via upload" (PNG asset only, 2026-09-14). **Last code commit `a450911` (2026-09-05)** | `git rev-parse origin/main` `[PROVEN]` |
| Commits since pin | **52 — the identical set the v0.10.3 audit reviewed; ZERO new code on main since that audit** | `git log --oneline e8384e0..origin/main \| wc -l` = 52; SHA set matches `audit/VoiceMEM_upstreamaudit_v0103/evidence/upstream_history_since_pin.txt` `[PROVEN]` |
| New material since the v0103 audit | Branch **`feature/local-rtc-transport`** = `b36b4fa` "Add experimental local RTC transport" (2026-09-03, +936/−29, merge-base `42c3fae` — branched *before* the barge-in/session-context work, contains none of it) | fetch output `* [new branch]` `[PROVEN]` |
| Site branches | `site/try-it-now` (4ab242b), `site/try-polish` (e336b5f) — fully contained in main; stale Pages pointers, nothing new | `git log origin/main..origin/site/*` empty `[PROVEN]` |
| Special-attention negatives | **No commit since the pin touches `session_tracker.py`, `subgraph_manager.py` or `_refresh_schema_descriptions`** — SessionTracker / schema refresh / dynamic subgraph have zero upstream movement | file-level + `git log -S` sweeps `[PROVEN]` |

**Headline:** upstream main is **static since the v0103 audit**. Our divergence
is ahead on every functional axis we compete on. The only genuinely new upstream
artefact is an unmerged experimental RTC branch whose sole broadly-compatible
idea is a small browser-error reporting hook.

---

## 2. SELECTIVE MERGE MATRIX

Legend — Action: **AP** = ALREADY PRESENT (ported or superior local
equivalent), **KL** = KEEP LOCAL (our divergence is better), **SP** =
SELECTIVE PORT, **PWM** = PORT WITH MODIFICATION, **D** = DEFER (with
trigger), **R** = REJECT, **R(N/A)** = no functional content.

| Upstream commit | Area | What it changes (from the diff) | Dependency chain | Useful to us? | Compatibility | Risk | Action |
|---|---|---|---|---|---|---|---|
| 22 README/news/assets/docs commits (afb6b10, 76cc1f5, d587d42, 41b1c8e, 23fd385, f636f75, 1f342cc, df3bca6, f50cc5e, 1b7cc9a, 4290831, 2ba2874, ab6f9d3, 7b22b18, edd0b05, 758d076, e57ac44, 9be64eb, 800afc5, 3fe6d98, 42c3fae, 6cacb3c) | docs/assets | README copy, SVG/PNG assets, repo renames, GH-Pages doc PR, final image re-upload | none | no | n/a | none | **R (N/A)** |
| `9193c78` | TTS | TTS becomes 9th injectable util (`vm.utils.get("tts")`; providers openai/local-piper/voxcpm/**breeze**); `BaseTTS.stream(text, instruction)` per-sample chunking; env-tunable first/sentence cut points | root of web chain | partly | our TTS is app-layer **by invariant** (span-level HU/EN routing); Breeze = remote + non-commercial weights; our `first_chunk_chars=24` already equivalent | Breeze: licence + network | **KL** (slot) / **R** (Breeze) |
| `7f842a8` | prompt/session | (1) two-level concurrent synth; (2) prosody → TTS `instruction` param; (3) `_history_block/_push_history` per-space deque (200 ch/turn) rendered in system block; (4) `memory_api max_rb` 3→5; (5) LCS echo detect | ←9193c78 | (1) yes | (1) **AP** since v0.10.3 (2-slot semaphore); (2) Supertonic ONNX has no instruction param; (3) our native 8-entry history + 14k budget gives 99.2% llama prefix-cache reuse; (4) own rendering contract; (5) our mic suppression prevents answering-phase echo | low | **AP**(1) / **D**(2) / **KL**(3,4) / **D**(5) |
| `d9fa443` | retrieval | multiplicative `_recency_weight` on `observed_at` (30-day halflife, floor 0.75) | ←e3cc965 | already have richer | superseded by VM-LOCAL-011 (additive bonus, undated=0, supersession-first) + VM-LOCAL-018 | none (live) | **AP** (adapted) |
| 6 RB web-display commits (`0ce0d72`, `a74d8a9`, `1dc9bbb`, `365f782`, `4504192`, `130089b`) | web display | `_RB_HUMANIZE_PROMPT` iterations (first person, speaker name, interface language, "what I'll do" suffix) — display-only | sequential same-day chain | no | zh demo UI; our UI has its own renderer + `_SLOT_EN` localisation | none | **R** |
| `fa537a9` | retrieval/attribution | `response_experience` quota 1→0; attribution JSON drops `assistant_did`/`next_time`; write path **deleted** | ←d9fa443 era; →333dbcb | quota+gate already live | ours gated (VM-LOCAL-017), env-reversible; attribution LLM call kept | shrink = prompt-change risk | **AP**(quota+gate) / **D**(schema shrink) |
| `333dbcb` | retrieval | heartnote slot cap = 2 (`VOICEMEM_RB_HEARTNOTE_MAX`) | ←fa537a9 | already live | identical semantics via `_apply_source_quota` | none | **AP** |
| `388c7dd` | cleanup | drops imports orphaned by fa537a9's deletion | ←fa537a9 | no | our write path is gated-retained → imports still used | none | **R (N/A)** |
| `f99569e` | playback | AudioWorklet player (adaptive jitter 80→320 ms, drain/clear, underrun telemetry) + caption sync | prerequisite of 6c085d4 pause/resume | maybe | our player = scheduled `createBufferSource` (gapless since our v0.10.3 concurrent synth); bigger browser change | medium (browser rework) | **D** |
| `e3cc965` | storage | `_as_date` validates `observed_at` | enables d9fa443 | ported at fork | none | none | **AP** |
| `ab97cf5` | UPDATE semantics | UPDATE defaults to ADD (append); `VOICEMEM_APPLY_UPDATE=1` restores overwrite | independent | no — strictly richer locally | VM-LOCAL-008: append + supersession + provenance + current-first ranking | none (live) | **KL** (obsoleted by our work) |
| `a507978` | retrieval | `wants_recency()` cue gate (zh/en) — attribute queries keep pure similarity | ←d9fa443 | already live (adapted) | ours: additive bonus + **HU cues** (temporal.py:299-328) | none | **AP** |
| `961efe8` | embeddings | `_embed_text` routes through injected embedder | root of embed chain | ported (we then removed the OpenAI fallback entirely — VM-LOCAL-001) | E5-only invariant | none | **AP** |
| `91d2e42` | storage | dim-mismatch guards (cosine/graph_entity/traits) | ←961efe8 | ported | tagged `[ports upstream 91d2e42]` locally | none | **AP** |
| `f535f9d` | retrieval | trait sim threshold bound to embedder dims (384→0.88) | ←961efe8 | ported | `rightbrain/brain.py:252` tagged | none | **AP** |
| `089cb5e` | tooling | `tools/reembed.py` (88 lines) | ←a8d5c76 | no | single embedder (E5-only); we have read-only `assess_trait_embeddings.py` | none | **R (N/A)** |
| `a8d5c76` | embeddings | unknown providers pass through to mem0 EmbedderFactory (ollama/hf/gemini/bedrock/azure/vertex/together/lmstudio/fastembed/langchain) | →089cb5e, d0ab392 | **no** | **violates the E5-only local invariant** (`VOICEMEM_PIN.json` embedding_policy) | high (invariant) | **R** |
| `d0ab392` | docs | README embedding-provider docs | ←a8d5c76 | no | n/a | none | **R (N/A)** |
| `7581656` | barge-in | `hearing()` = sent-audio ledger tail window; grace from first audio sent; force-stop on superseding turn; interrupted-turn retention | ←7f842a8 | yes — this was our exact bug class | our v0.10.3 port: `_playback_tail_active` (web_server.py:2694) + `_account_sent_audio` (:2716) + force-stop (:2744-2772) | none (live since v0.10.3) | **AP** |
| `7c22ac7` | config | new `llm_config.py`: 5 roles, env precedence, import-time env fix | independent | no | our `llm_config.yaml` + llama-server role resolver predates and covers (OpenAI-centric upstream) | none | **KL** |
| `a48e90f` | memory language | `voicemem/lang.py`: process-global `memory_language` (**en/zh only**), `label_rule()`, trait script-mismatch drop guard, localised trait prompts, `space.describe()` language field | root of lang chain | mechanism no; guard idea minor | **contradicts our bilingual single-store design** (E5 multilingual; HU/EN product); Hungarian unsupported upstream | high (architecture) | **R** (mechanism) |
| `6f282e8` (+merge `35bb79e`) | memory language | localised attribution prompts + emotion line; script-mismatch drop before reaction-trait write | ←a48e90f | same as above | our attribution prompt already English (VM-LOCAL-EN lineage); guard idea only | same | **R** (mechanism) |
| `0472c41` | memory language | `resolve_for_space()` (space-level language, constructor-order independent); `display_emotion` zh→en map | ←a48e90f, 6f282e8 | no | per-space mono-language ≠ our design; app layer already localises display; HU uncovered | high | **R** |
| `6c085d4` | session/barge-in | (1) `SessionBuffer` per (session,space), turn removed on `persistent_memory_created` via new `on_complete` callback; (2) `queue_remember_turn` asyncio.Lock write serialization; (3) **reversible barge candidates** (pause at VAD onset, confirm/reject/resume, noise-turn discard); (4) interrupted flag | ←f99569e (worklet pause/resume), ←7581656 | (4) yes | (1) we keep turns by design (native bounded history); (2) our llm_bg_gate serialises **and cancels** — strictly richer; (3) hard browser dependency; (4) **AP** (adapted: heard fragment + `[interrupted]`, web_server.py:3256-3274) | (3) medium | **KL**(1,2) / **D**(3) / **AP**(4) |
| `a450911` | playback | `voicemem/audio_timing.py` + `web/audio_timeline.py`: playback-position→text mapping (provider alignment > segment ratio > per-char-cost EWMA, CJK-calibrated); `SessionTurn.interrupted` | ←6c085d4, ←f99569e | maybe later | Supertonic provides **no timestamps** → EWMA path only; costs CJK-calibrated (HU/EN need recalibration); our sent≈heard + `[interrupted]` adequate | medium, speculative | **D** |
| `b36b4fa` (branch `feature/local-rtc-transport`) | transport | Go WebRTC gateway (pion v4; browser↔gateway Opus/SRTP; gateway↔Python WS IPC; ICE UDP 8791/TCP 8792; drop-oldest queue) + `RTCTransport` (PyAV, 24k↔48k Opus, 20 ms pacing) + `--transport` flag + RTC-only BARGE_VAD_MS=120 fast stop + `/api/debug/log` browser-error endpoint + stdout logging tee | branched off 42c3fae (contains none of 7581656/6c085d4) | debug-log idea only | transport **R**: unmerged experimental branch; Go build dependency + 2 new ports (attack surface); Opus framing **bypasses our 24 kHz PCM sent-audio ledger** (breaks `_playback_tail_active` hearing accounting); browser-AEC echo strategy conflicts with our mic-suppression design; zh-demo browser code | transport: high | **R** (transport, tee, BARGE_VAD) / **SP** (debug-log idea) |

---

## 3. GROUPING BY FUNCTIONAL AREA

- **Retrieval / query behaviour** — fully absorbed: VM-LOCAL-011/016/017/018 +
  fork-creation ports. Only open sub-item: `fa537a9`'s attribution-prompt
  schema shrink (**D** — measure background slot pressure first).
- **UPDATE / memory semantics** — ours strictly richer (VM-LOCAL-008
  supersession lineage).
- **Session context & state persistence** — all upstream variants are
  in-memory and inferior to our design direction (native bounded history +
  prefix cache + the studied KV-lane checkpoint). **KEEP LOCAL** across the
  board.
- **Barge-in / interrupted context** — our v0.10.3 port set covers the
  measured failure classes; the reversible-pause branch has a hard worklet
  dependency and no measured need. **D**.
- **Memory language** — **REJECT as a mechanism** (mono-language per space vs
  our bilingual-by-design single store; en/zh only, no Hungarian; E5 is already
  cross-lingual). Two ideas quarantined for reconsideration (§5).
- **TTS / streaming** — app-layer span routing invariant untouched; Breeze
  rejected (licence/remote).
- **Transport (RTC)** — rejected; only the debug-log hook survives (§4).
- **SessionTracker / schema refresh / dynamic subgraph** — **zero upstream
  movement since pin**; the dormant-machinery conclusions of the
  session-continuity audits stand with no upstream pressure to revisit.

---

## 4. THE ONE RECOMMENDED PORT — E1: browser→backend error reporting

**`/api/debug/log` (idea from `b36b4fa`, adapted)**

- **What:** one POST route in `app/web_server.py` (truncate kind≤40,
  message≤12k, page≤500; log at WARNING with peer host via the existing
  logging stack → `logs/web-server.log`) + a `report()` fetch (keepalive,
  catch-all) wired into our **existing** `window.onerror` /
  `unhandledrejection` handlers (`web/voicemem.html:751-757`) and the on-page
  error banner.
- **Why:** browser-side failures (JS errors, WS close causes) are currently
  visible only as an on-page banner; this puts them into the server log next
  to the pipeline-stage timings already instrumented — the exact gap the
  S8-correlation class of forensics keeps hitting.
- **Adaptations vs upstream:** upstream prints to a stdout tee; we use the
  app's logging stack. No `logging_utils.py`. No transport machinery.
- **Cost/risk:** ~30 lines, app-layer only, zero vendor changes, zero new
  dependencies, fire-and-forget (no critical-path impact).
- **Test requirements:** unit test for route truncation; manual browser error
  injection. Rollback: delete the route + the fetch call.
- **Classification:** optional, production-non-critical. `[INFERENCE]`
  (benefit) on `[PROVEN]` mechanism facts.

---

## 5. DEFERRED BUNDLE (explicit triggers, not this cycle)

| Item | Source | Reconsideration trigger |
|---|---|---|
| `fa537a9` attribution schema shrink | upstream main | gate stats show background slot pressure attributable to the attribution call |
| Reversible barge pause + AudioWorklet | `6c085d4` + `f99569e` (hard dep) | field reports of false barge-ins during the playback tail window |
| Trait script-consistency drop guard | `a48e90f`/`6f282e8` idea | field reports of non-English labels leaking into `rb_traits` |
| Heard-text playback timeline | `a450911` | users report `[interrupted]` misrepresenting what they heard on long replies |

None is blocked on anything upstream (main is static); each is a local
decision gated on production evidence.

---

## 6. STILL MUST NOT BE IMPORTED

`a8d5c76` (E5-only invariant), Breeze TTS (licence + remote), the `lang.py`
chain (architecture), the RTC transport branch, the zh-demo display layer,
`7c22ac7` (our llama role config predates and covers it), `089cb5e`
(single-embedder environment).

---

## 7. COMPLIANCE

- Production code changes: **0**. Merges/rebases/cherry-picks: **0**.
- The only git side effect: read-only `git fetch origin` in the upstream
  clone (refs refresh, no working-tree change).
- All verdicts above are backed by actual diff inspection recorded in the
  working notes merged into `worklog.md` (Task ID 4-a).
