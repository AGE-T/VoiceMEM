# VoiceMem v0.5.0 — Controlled Migration VERIFICATION Report

**Task**: Verify the controlled VoiceMem migration + establish the rich retrieval
baseline (19 phases). This session VERIFIED; it did not modify the engine.
**Verified state**: `VERSION 0.5.0`, repo commit chain
`2b406bd` (v0.5.0 source) → `c192492` (browser verification) → `4849174` (lint) →
`a2758c2` (worklog) → `9a0d3d1` (HEAD at verification time; runtime test data only).
**VoiceMem identity**: controlled fork `vendor/voicemem`, pinned upstream commit
`e8384e087bd2f44eb05fc7ae1a3c525ea8244179` (= tag `v0.0.1`, package version `0.2.3`
— the version is NOT the identity; the commit is).

Every claim below was re-verified in THIS session (not trusted from the prior
report). Evidence files ship alongside this report.

---

## A. Deployment verification (FACT)

- `vendor/voicemem` exists in-repo (105 files), tracked by the main repo
  (root `/home/z/my-project`), **no `.git` inside the vendor tree** — there is no
  fetch surface. `vendor/.clone_ref` (the old clone-era marker) does NOT exist.
- `VOICEMEM_PIN.json` exists and records the pinned commit, the embedding
  policy, the local patch ledger (VM-LOCAL-001…006 + VM-LOCAL-EN) and the four
  ported upstream fixes.
- Runtime identity block verified: `voicemem.CONTROLLED_FORK = True`,
  `CONTROLLED_UPSTREAM_COMMIT = e8384e087bd2f44eb05fc7ae1a3c525ea8244179`,
  `CONTROLLED_PATCHES` = the 10-entry ledger.
- `scripts/verify_voicemem_pin.py` passes in ALL THREE modes:
  `source` (patch markers, no OpenAI embeddings call, DELETE guard, vendor not a
  clone), `runtime` (import resolves under `vendor/voicemem`, fork + commit
  match), `embedding` (instrumented fake-openai proof; see F).

## B. Installer (FACT)

`scripts/install_m1.ps1` step 12 now installs `pip install -e vendor\voicemem`
and FAILS when: the vendor package is missing, the pin file is missing, the pin
commit mismatches, or the English-patch marker is absent. The old
clone-at-v0.0.1-with-main-fallback logic is GONE (grep-verified: the only
remaining `git clone` strings are comments and user-facing hints to clone OUR
repo). `scripts/build_release.ps1` + `build_release_sandbox.py` stage `vendor`
into the ZIP (105 vendor files present in the v0.5.0 ZIP — verified by listing).

## C. venv (FACT)

`voicemem-agent/.venv` (Python 3.12.14) contains exactly ONE voicemem
distribution: the editable install `__editable__.voicemem-0.2.3.pth` whose
finder maps to `vendor/voicemem`. `PYTHONPATH` is empty; the bare `sys.path`
does not contain the vendor dir (the editable finder provides it). The main
sandbox venv (`/home/z/.venv`) and system `python3` have NO voicemem. No
shadowing exists. (The .venv carries a dependency SUBSET for the runtime E2E
scripts; the full test gate runs against the main sandbox venv in degraded
mode — see O/P.)

## D. Runtime (FACT)

Both app entry layers inject the local embedder:
- `app/voicemem_bridge.py` (CLI) — `embedding=` factory returns
  `LocalE5Embedder` (parity fix, v0.5.0).
- `app/web_server.py` (web) — same injection; the status panel reports
  "VoiceMem controlled fork @ e8384e0 (vendor, text_mode, per space)" or
  "NOT the controlled fork".
- `bootstrap.ps1` gates on the full pin probe (a stray site-packages voicemem
  can no longer pass the boot check) and errors loudly when the
  English-patch marker is missing.
- `verify_m1.ps1` check 6b: import → `vendor\voicemem`, `CONTROLLED_FORK`,
  commit equality — with an actionable repair message.

## E. Exact commit (FACT)

Pinned upstream base: `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` (tag `v0.0.1`).
Repo release commit: `2b406bd` (v0.5.0 CONTROLLED VOICEMEM OWNERSHIP).
Vendor tree vs upstream reference clone diff: EXACTLY 10 files differ, all
mapped to the documented ledger:
`__init__.py` (VM-LOCAL-006), `leftbrain/brain.py` (VM-LOCAL-002),
`leftbrain/mem0_backend_store.py` (VM-LOCAL-005),
`leftbrain/merged_extraction.py` (VM-LOCAL-EN),
`leftbrain/slot_split/graph_entity_store.py` (UPSTREAM-91d2e42 port),
`orchestrator.py` (VM-LOCAL-001 + 961efe8 intent), `rightbrain/brain.py`
(UPSTREAM-f535f9d port, verified: `_TRAIT_MIN_SIM_BY_DIM = {384: 0.88}`),
`rightbrain/traits_store.py` (VM-LOCAL-004 + ports),
`utils/common/_graph_common.py` (91d2e42 cosine guard),
`utils/defaults.py` (VM-LOCAL-003). **No undocumented divergence.**

## F. Embeddings (FACT)

- Source mode: `_embed_uncached` is local-E5-only; the OpenAI embeddings API
  call is REMOVED; all three defaults (orchestrator, left-brain repo,
  utils/defaults factory) route to `LocalE5Embedder`.
- Instrumented proof (`verify_voicemem_pin.py --embedding`): a fake `openai`
  module that EXPLODES on any `OpenAI()` construction + a fake 384-d local E5
  → `_embed_text` routes to the injected embedder; the no-injection default is
  the local E5; **the OpenAI client is NEVER constructed**. The proof runs
  without any network.
- Live proof: the real-E5 runtime E2E (H) exercises the same path with the
  actual `intfloat/multilingual-e5-small` (384-d, unit-norm).

## G. Trait embedding status (FACT)

`scripts/assess_trait_embeddings.py` (read-only, `--strict`, `--dry-run-backfill`)
on the fresh runtime DB: 2 traits, **0 NULL, 0 stale-dimension, histogram
{384: 2}**, per-user and per-slot breakdowns healthy, backfill plan EMPTY.
On synthetic defect DBs (unit tests) NULL + stale rows are correctly detected
and planned. NO backfill was executed anywhere.
**Production note**: the sandbox `memory/` tree is empty (user data lives on
the target machine); run the same read-only tool there — it ships in the ZIP.

## H. Clean install result (FACT)

v0.5.0 ZIP (`releases/VoiceMemAgent_v0.5.0.zip`, SHA256
`45C1FEBA29DC61492AA148B167C31E60509B7EEEEDE003F33348F430254CEB40`, 1915147
bytes, same hash at `public/` and `public/releases/`) → extracted to
`/tmp/clean_verify_v050` → FRESH venv → `pip install --no-deps -e vendor/voicemem`
→ `import voicemem` resolves to the EXTRACTED vendor tree → fork True, commit =
pin → `verify_voicemem_pin.py` source mode OK → `write_install_manifest.py`
records `pin_verified: true`, `upstream_commit: e8384e0…`,
`runtime_import_path: …/vendor/voicemem/voicemem/__init__.py`,
`package_version: 0.2.3`. **A clean machine cannot install anything other than
the controlled source.**

Runtime sanity on the agent venv with the REAL E5 + the deterministic mock LLM
(`tests/e2e_mock_llm.py`, port 18080): ingest → retrieve → repeated related
ingest → dual retrieval → trait creation → trait embeddings 384-d/0-NULL → HU,
EN, cross-language queries → **RUNTIME SANITY OK** (observed_at dates populate,
`[2026-09-10]` prefixes present — the e3cc965 fix is live).

## I. Retrieval baseline (FACT, re-measured this session)

Same 4-memory corpus, 7 queries (HU/EN/cross-language/trait), paths A and B
(`/tmp/voicemem_retrieval_baseline.json`, shipped as evidence):

| | PATH A — plain `vm.search(q)` | PATH B — `classify → search(slots, entities)` |
|---|---|---|
| mode | `fallback` (pure vector) | `slot-only` |
| slot classification | none (`slots=[]`) | yes (~4 ms with the mock classifier) |
| slot_mem_ids / final_ids | 0 / 0 | 4 / 4 |
| hits | 3 | 3 |
| rank of expected | 1 (all 7 queries) | 1 (all 7 queries) |
| latency | 39–106 ms | classify 4 ms + search 62–85 ms |
| rb trait hits | trait queries return rb hits | same |

Root cause of the Claude audit's "bypass" (confirmed by measurement): the
plain public `search()` NEVER classifies — `LeftBrain.search` receives
`slots or []`, so it falls back to pure vector by construction. The rich path
exists and is used by `Memory.recall/inject` and stream turn speculation;
`vm.search` and the web manual search do NOT use it.

## J. Renderer baseline (FACT, re-measured this session)

`SearchResult.hit` carries: `memory_id, text, score, base_score, time_boost,
observed_at, attributed_to, metadata` (+ slot/entity context in the caller).
`build_memory_context` renders ONLY `- [observed_at] text` lines (+ rb hits
capped at 3 with the "HOW TO SPEAK TO THIS USER" framing): 269 chars for 3
hits. NOT rendered: `score, base_score, time_boost, attributed_to` (speaker),
`metadata` (source/provenance), slot, entity — the measured information loss.

## K. HU/EN (FACT)

HU fact + HU query → rank 1; EN fact + EN query → rank 1; Hungarian trait
sentence → trait stored, embedded (384-d), retrieved via HU and EN trait
queries. HU/EN behaviour preserved (no language coercion; the multilingual E5
is a design component). HU/EN content is NOT translated and NOT migrated.

## L. Cross-language (FACT)

`kedvenc kávé minden reggel` (HU query → EN-normalised memory) rank 1;
`my favourite morning coffee Budapest` rank 1. Same entity, two languages, no
identity split.

## M. Upstream review (FACT — ledger verified)

`UPSTREAM_POLICY.md` documents 51 commits after the pin, with the 5-way
classification: ADOPTED now (961efe8, 91d2e42, f535f9d, e3cc965 — ported and
ledger-verified); DEFERRED with triggers (d9fa443+a507978 recency; 333dbcb
heartnote cap; ab97cf5 append-default; language-per-space trio — would
ValueError on `hu`; 7c22ac7 schema→slots rename — breaking; 6c085d4
on_complete; 089cb5e re-embed tool — our own read-only assessor covers it);
REJECTED (a8d5c76 mem0 provider adapter; their TTS/web-demo/streaming stack).
Hard invariants: local E5 only, no new OpenAI runtime dep, no masked failures,
HU/EN preserved, no data migration by code merge, Qwen3.6 35B A3B IQ4_XS on
llama-server, DELETE opt-in.

## N. Next-step recommendations (PROPOSAL)

1. **Rich retrieval for the web manual search**: inject `LocalQueryClassifier`
   (~10 ms, 0 LLM) into `web_server.search` behind an explicit flag; measure
   HU slot-classification quality at production scale FIRST.
2. **Renderer**: add a compact speaker/source marker (`[user]`/`[assistant]`)
   and a confidence grade when present; NOT raw score numbers.
3. **Traits first_seen/last_seen/occurrence_count** — the cheapest primitives
   with real consumers (UI + merge policy); every other primitive from the
   Phase 13 list stays parked until it has a consumer.
4. **heartnote cleanup hardcodes `model="gpt-4o-mini"`** (rightbrain/brain.py:885)
   — registered risk: on llama-server this 404s and silently no-ops at ≥50
   heartnotes. Fix = route through the configured OpenAI-compatible client.
5. **PARKED BLOCKER (highest priority, out of scope here)**: mic→LLM chain
   break — trace ASR transcript → pipeline handoff → VoiceMem/conversation
   processing → prompt construction → LLM request → llama-server → first
   response token BEFORE changing anything.

## O. Tests executed (this session, FACT)

- `verify_voicemem_pin.py` — source, runtime, embedding modes (3× exit 0).
- `tests/unit/test_voicemem_controlled.py` — 31 tests via the agent venv.
- Full gate `python -m unittest discover -s tests -t .` (main sandbox venv,
  degraded mode, as at build time) — see P.
- `verify_voicemem_runtime.py` — real-E5 runtime E2E against the mock LLM.
- `assess_trait_embeddings.py` — read-only on the runtime DB (+ `--strict`).
- Clean-install simulation: ZIP → fresh venv → editable install → identity +
  pin + manifest checks.

## P. Test results (FACT)

- **Full gate: 921 tests, OK, 24 skipped — reproduced exactly as recorded in
  `BUILD_HISTORY.json` for v0.5.0** (`test_gate_exit_code: 0`, deep validation
  "pass: release, runtime-deps, runtime-venv; skip: 24").
- Controlled-fork suite: 31/31 OK (30.5 s).
- Runtime E2E: RUNTIME SANITY OK; all 7 queries rank 1 on both retrieval paths.
- Trait assessment: 0 NULL / 0 stale / {384: 2}.
- Note: running the gate with the vendor dir injected into PYTHONPATH makes 4
  environment-assumption tests fail ("voicemem NOT importable in the sandbox
  process") — that is expected, not a defect; the sanctioned gate environment
  is the degraded-mode run above.

## Q. Residual risks (FACT/INFERENCE)

1. **PARKED: mic→LLM chain break** (HIGH, out of scope, untouched by design) —
   mic→PCM→VAD→ASR verified working in earlier sessions; the break is between
   successful ASR output and the final LLM request; exact failure point not
   yet established. Evidence preserved in v0.4.19–v0.4.21 diagnostics.
2. **heartnote `gpt-4o-mini` hardcode** (MEDIUM) — silent no-op path once ≥50
   heartnotes accumulate on llama-server.
3. **Production trait-embedding health** (UNKNOWN until measured) — run
   `assess_trait_embeddings.py` read-only on the target machine's memory DB;
   the sandbox has no user data. Old NULL/stale rows (pre-fix) may exist there
   and need an EXPLICITLY approved backfill (`--dry-run-backfill` plan first).
4. **Recency signal** (MEDIUM) — v0.0.1 Rank has NO recency weighting
   (upstream d9fa443+a507978 deferred, needs HU cue extension); memories rank
   purely by similarity+heat today.
5. **Slot-split graph dormant** (LOW) — revived by VM-LOCAL-001 but only
   written when the app schedules `Flush()`/session boundaries (it never does
   today); no data loss, no action needed this phase.
6. **Tool-results reference clone** (NONE) — `tool-results/voicemem-src` is a
   submodule pinned at the same upstream commit e8384e0 with local file-mode
   noise; it never ships and no install path reads it.

---

## Phase 19 — the ten engineering decisions (written answers)

1. **Is the controlled deployment chain complete?** YES (FACT). ZIP ships the
   vendor source; the installer editable-installs it and pin-verifies; import
   resolves under vendor with the pinned commit; the manifest records
   `pin_verified: true`; bootstrap/verify_m1 re-probe at boot; no path clones
   upstream (Phase 2 classification: ACTIVE = none).
2. **Is the OpenAI embedding defect thoroughly fixed?** YES (FACT). The call
   is removed from source; three defaults route to local E5; the instrumented
   fake-openai proof passes; the live runtime E2E embeds with the real E5.
   Regression-locked by `test_voicemem_controlled.py` (would fail on return).
3. **Does the runtime guarantee local E5?** YES (FACT) when anything is
   injected (both app layers inject it), and the no-injection default is the
   same local E5 (VM-LOCAL-002/003). The only way out is a caller deliberately
   injecting a different embedder.
4. **Is the public `search()` bypass confirmed?** YES (FACT, measured). Plain
   `vm.search` = pure-vector fallback (`slots or []` by construction); the
   Claude audit's finding reproduces on every query.
5. **Should rich retrieval become the default?** NOT the low-level primitive
   (DECISION). The rich path is already the default on the official one-line
   API (`Memory.recall/inject`) and turn speculation. Flipping `vm.search`
   itself is DEFERRED until HU classification quality is measured at
   production scale (PROPOSAL: LocalQueryClassifier behind an explicit flag
   in the web manual search first).
6. **What renderer changes?** Minimal (DECISION): compact speaker/source
   marker + confidence grade when present; keep scores OUT of the prompt
   (noise); measure token cost before/after. observed_at is already rendered.
7. **Which memory primitives next?** `first_seen / last_seen /
   occurrence_count` on traits (DECISION) — cheap, real consumers (UI + merge
   policy). Everything else (supersedes, valid_from/until, confidence
   plumbing, recorded_at surfacing) stays parked until a consumer exists.
8. **Which subsystems stay untouched?** Entity stores: KEEP all three
   (cognitive graph = authoritative; slot-split = dormant-but-revivable; rb
   entity graph = dead upstream on our paths — remove nothing). Lifecycle:
   no changes this phase except the registered heartnote model-name risk.
   Recency: defer with the upstream commits. No memory learning (Phase 12).
9. **Which upstream changes are worth adopting?** The four already ported.
   Next candidates, in order: 6c085d4 (on_complete freshness signal),
   d9fa443+a507978 recency (AFTER the HU cue extension + baseline rerun),
   333dbcb heartnote cap (with the model-name fix). NEVER: their TTS/web
   demo/streaming stack, a8d5c76 provider adapter.
10. **Which changes stay local forever?** The no-OpenAI-embeddings divergence
    (VM-LOCAL-001), the local-E5 defaults (002/003), the loud trait-embedding
    failure (004), the DELETE opt-in guard (005), the runtime identity block
    (006), and the English extraction localisation (VM-LOCAL-EN) — each is
    either stricter than upstream or project-specific by design.

---

*Verification session: 2026-09-10. No engine source, installer, or memory data
was modified by this verification. Evidence: `docs/verification_evidence/`
(shipped with the recovery package).*
