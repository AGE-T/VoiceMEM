# PROVENANCE — the project-controlled VoiceMem implementation

This directory is the **authoritative VoiceMem memory implementation for
VoiceMemAgent**. It is a source-owned vendor tree, NOT a live git clone:

* the application installs it with `pip install -e vendor\voicemem`
  (installer step 12) — the installer never clones, fetches or checks out
  anything for VoiceMem;
* every modification below is first-party code, reviewed and tested in THIS
  repository (`voicemem-agent`), tracked by THIS repository's git history;
* rollback = `git revert` in `voicemem-agent`; upstream comparison = diff
  against the pinned upstream snapshot (see "Comparing with upstream").

## Base snapshot (immutable pin)

| Field    | Value |
|----------|-------|
| Upstream | https://github.com/xzf-thu/VoiceMem |
| Tag      | `v0.0.1` |
| Commit   | `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` |
| Taken    | 2026-09 (release v0.4.22 vendor move) |

Excluded from the upstream tree when vendoring (no code references them):

* `voicemem/assets/*.wav` — 968 KB of README/example demo audio;
  unreferenced by the package (`rg 'voicemem/assets' voicemem/` is empty);
  `voicemem.sample_audio()` falls back gracefully when absent.
* upstream top-level `docs/ evaluation/ examples/ finetune/ web/ scripts/
  tests/ tools/` — demo and research harnesses, not part of the installed
  package (`[tool.setuptools.packages.find] include = ["voicemem*"]`).

Everything else — including `voicemem/data/*.txt` (runtime prompt files,
declared as package data in `pyproject.toml`) — is a byte-exact copy of the
pinned commit, except the local modifications listed below (7 as of the write-guard fix).

## Local modifications (first-party, in this repo's history)

All changes below are marked in-source with
`CONTROLLED-VENDOR LOCAL FIX (voicemem-agent; upstream analogue: …)` blocks.

| # | File | Change | Why | Upstream analogue |
|---|------|--------|-----|-------------------|
| 1 | `voicemem/orchestrator.py` | `_embed_text` honours the injected embedder (`embed_query_text` preferred, `embed_texts` fallback) before the OpenAI embeddings API | The stock v0.0.1 hard-wired the remote call for slot anchors, graph entities and right-brain traits even when `VoiceMem(embedding=…)` installed a local embedder. Our llama-server has no `/v1/embeddings`, so those calls always failed: `rb_traits.embedding` NULL, trait merge dead, RB retrieval empty, slot-tagging results discarded. **Invariant established: with an injected embedder, NO memory-pipeline embedding call leaves the local model.** | `961efe8` "Use the injected embedder everywhere" (inspected, reimplemented) |
| 2 | `voicemem/rightbrain/brain.py` | `trait_min_sim(dim)`: retrieval gate 0.88 for 384-dim (local multilingual E5), 0.45 otherwise; env `VOICEMEM_RB_TRAIT_MIN_SIM` overrides both | E5 compresses cosine similarity into a narrow high band (real hits 0.89–0.92, noise 0.82–0.86). The stock 0.45 (OpenAI-tuned) would let noise traits flood RB retrieval once fix #1 makes trait vectors real. | `f535f9d` "Bind trait threshold to embedder" (inspected, reimplemented) |
| 3 | `voicemem/rightbrain/traits_store.py` | `search_scored` records `last_query_dim` (read by the gate above) and warns once when stored trait vectors have a stale dimension | Makes the dimension-keyed gate possible; makes an embedder switch VISIBLE instead of silently shrinking retrieval. | `f535f9d` + `91d2e42` (inspected, reimplemented) |
| 4 | `voicemem/utils/common/_graph_common.py` | `cosine()` returns 0.0 for unequal lengths | The stock `zip` silently truncates to the shorter vector; after any embedder switch, entity dedup would merge on meaningless truncated dot products. | `91d2e42` (inspected, reimplemented) |
| 5 | `voicemem/leftbrain/slot_split/graph_entity_store.py` | `find_similar_entity` skips + warns once on dimension-mismatched stored vectors | Same failure class as #4, at the entity-store layer. | `91d2e42` (inspected, reimplemented) |
| 6 | `voicemem/leftbrain/merged_extraction.py` | The trait/emotion extraction prompt block is localised to English (British spelling); the five slot VALUES stay the upstream enum | HU/EN conversations otherwise get Chinese trait labels and emotion words written into the memory graph (v0.4.6 field report). Previously applied post-clone by `scripts/patch_voicemem_english.py`; now first-party. **Equivalence proof: running that script against this tree reports "already localised" (exit 0).** | none (project-specific) |
| 7 | `voicemem/utils/common/voice_input.py` | ingest decision guards: UPDATE downgraded to ADD by default (`VOICEMEM_APPLY_UPDATE=1` opt-in); DELETE not executed by default, suppression logged (`VOICEMEM_APPLY_DELETE=1` opt-in) | LLM conflict-resolver decisions that destroy history get deterministic guards. Upstream measured 158 UPDATEs: 34% no-ops, 11% information loss, ASR noise overwriting good memories; a hard DELETE is equally unrecoverable, yet upstream guards only UPDATE (ab97cf5, v0.0.2). Memory text, embeddings and provenance stay intact; duplicates remain a recoverable dedup problem. | `ab97cf5` "Append instead of overwrite" (inspected; UPDATE part adopted, DELETE part has no upstream analogue) |

Deliberately NOT adopted from upstream (as of 2026-09):

* `961efe8`'s siblings `a8d5c76` (mem0 provider passthrough) and `089cb5e`
  (`tools/reembed.py`) — we inject `LocalE5Embedder` directly and hold no
  stale-dimension vectors (our NULL-trait recovery is a separate, future
  operation; see `UPSTREAM_POLICY.md`).
* everything on upstream `main` beyond the commits above (language-per-space,
  unified llm_config, response_experience removal, recency weighting, …) —
  each requires individual review under the policy in `UPSTREAM_POLICY.md`.

## Comparing with upstream

```bash
git clone https://github.com/xzf-thu/VoiceMem.git /tmp/voicemem-upstream
git -C /tmp/voicemem-upstream checkout e8384e087bd2f44eb05fc7ae1a3c525ea8244179
diff -r --exclude assets --exclude .git \
  /tmp/voicemem-upstream/voicemem  <this-dir>/voicemem
# expected: only the six controlled modifications above (and their markers)
```

## Policy

Upstream is a REFERENCE SOURCE, not an authority. See the repository-level
`UPSTREAM_POLICY.md` (voicemem-agent root) for the monitoring cadence and the
ADOPT / PARTIALLY ADOPT / REIMPLEMENT LOCALLY / DEFER / REJECT decision rules.
