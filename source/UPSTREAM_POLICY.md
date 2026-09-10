# VoiceMem Upstream Policy

**Status: CONTROLLED FORK.** This repository owns its VoiceMem memory engine.
The authoritative source is `vendor/voicemem` in this repository — a controlled
fork of upstream `xzf-thu/VoiceMem` pinned at commit
`e8384e087bd2f44eb05fc7ae1a3c525ea8244179` (tag `v0.0.1`, package version
`0.2.3`). The exact identity and patch ledger is `VOICEMEM_PIN.json`.

The upstream repository is a REFERENCE, never a dependency. No deployment path
clones it, fetches it, or checks out any ref from it. Installation is
`pip install -e vendor\voicemem` from the tree that ships with this repository.

## Hard invariants (never negotiable in an adoption review)

1. **Local E5 is the only memory embedder.** Every VoiceMem memory embedding
   operation routes through the explicitly injected local multilingual E5
   (`intfloat/multilingual-e5-small`, 384-d, offline), or the same local E5
   via `LocalE5Embedder` when nothing is injected. A candidate upstream change
   that reintroduces a remote OpenAI embeddings call, an embedding-endpoint
   fallback, or a second embedding model is REJECTED by default.
2. **No new OpenAI runtime dependency.** The `openai` client is used only for
   OpenAI-compatible CHAT completions against our local llama-server.
3. **No masked failures.** A failing embedder must be loud (log + measurable
   via `scripts/assess_trait_embeddings.py`), never silently degraded.
4. **HU/EN behaviour is preserved.** No global `memory_language` coercion; the
   multilingual E5 is a design component, not an accident; different languages
   must not create separate identities for the same entity.
5. **Memory data is never migrated by a code merge.** Schema/behaviour changes
   touching stored data require a dry-run report first and explicit operator
   approval (see `scripts/assess_trait_embeddings.py --dry-run-backfill`).
6. **The LLM stays Qwen3.6 35B A3B IQ4_XS on local llama-server** (project-level
   policy; the vendor package's LLM calls go to `OPENAI_BASE_URL`).
7. **DELETE stays opt-in** (`VOICEMEM_ALLOW_MEMORY_DELETE=1`); the reversible
   path is `archive_memory()`.

## Adoption review procedure

When a new upstream change looks relevant:

1. **Identify** the upstream commit (sha, title, files) from the upstream repo
   or its releases. NEVER merge a branch or pull `main`.
2. **Understand intent** by reading the commit message and the diff. Do not
   port blindly: our fork's files carry local patches that the diff will not
   know about (`VOICEMEM_PIN.json` lists them).
3. **Classify** the change:
   - **ADOPT** — port the change onto `vendor/voicemem`, preserving local
     patches; add a `ported_upstream_fixes` entry in `VOICEMEM_PIN.json`;
     add/extend a regression test.
   - **PARTIALLY ADOPT** — port the applicable subset; document exactly what
     was not taken and why.
   - **REIMPLEMENT LOCALLY** — solve the problem our own way (e.g. when the
     upstream fix depends on architecture we do not have).
   - **DEFER** — record the candidate in the worklog with the trigger that
     would reopen it.
   - **REJECT** — record the reason (violates an invariant, targets their
     demo stack, etc.).
4. **Update** `VOICEMEM_PIN.json` (local_patches / ported_upstream_fixes
   ledger) and the controlled-fork identity block in
   `voicemem/__init__.py` (`CONTROLLED_PATCHES`).
5. **Test** — vendor-focused regression tests must stay green; the full gate
   runs before a release.

## Review history

| Date | Upstream | Change | Decision |
|------|----------|--------|----------|
| 2026-09-01 | 961efe8 | Use the injected embedder everywhere | ADOPT (as VM-LOCAL-001 with a stricter local divergence: no-OpenAI default) |
| 2026-09-01 | 91d2e42 | Guard vector dimension mismatch | ADOPT (print → logging locally) |
| 2026-09-01 | f535f9d | Bind trait threshold to embedder | ADOPT |
| 2026-09-01 | e3cc965 | Fix memory event dates | ADOPT |
| 2026-09-01 | 089cb5e | re-embed tool (tools/reembed.py) | DEFER — assessment via our own read-only `scripts/assess_trait_embeddings.py`; actual backfill only after explicit operator approval |
| 2026-09-01 | d9fa443 + a507978 | recency weighting in Rank | DEFER — retrieval-behaviour change; needs the retrieval baseline + HU cue extension before adoption |
| 2026-09-01 | 333dbcb | cap heartnote seats in right-brain context | DEFER — behaviour change, measure first |
| 2026-09-01 | ab97cf5 | append instead of overwrite (UPDATE→ADD default) | DEFER — conflict-semantics change; data-safety attractive, needs decision + tests |
| 2026-09-01 | a8d5c76 | mem0 embedder provider adapter | REJECT for now — we inject LocalE5Embedder directly; adds provider surface we do not use |
| 2026-09-03 | a48e90f / 0472c41 / 6f282e8 | memory language per space (SUPPORTED = en/zh only) | DEFER — architecture noted; requires `hu` support work; current HU/EN behaviour deliberately preserved |
| 2026-09-03 | 7c22ac7 | unify model configuration (schema→slots rename) | DEFER — 27-file refactor; record the rename as a future breaking change |
| 2026-09-05 | 6c085d4 | ingest on_complete callback | DEFER — useful freshness signal; adopt with a dedicated change |
| — | 9193c78, f99569e, 7581656, a450911, 0ce0d72 etc. | TTS backends / web demo / streaming UI | REJECT — their demo stack; we own TTS (Piper) and the web UI |

## Refresh procedure (re-vendor from a newer upstream pin)

Only when a deliberate decision is made to move the upstream pin:

1. Read the new pin's diff against `e8384e0` (upstream compare URL or a
   throwaway clone OUTSIDE this repo, e.g. `/tmp`).
2. Re-apply the `local_patches` and still-wanted `ported_upstream_fixes`
   from `VOICEMEM_PIN.json` onto the new tree.
3. Re-run `scripts/patch_voicemem_english.py` (idempotent).
4. Update `VOICEMEM_PIN.json` + the identity block in `voicemem/__init__.py`.
5. Run the vendor regression tests + full gate.
6. Update this document's review history.

**Never** `git fetch`/`git pull`/`git checkout <ref>` inside `vendor/voicemem`
(the tree is not a git checkout — any such command can only fail or, worse,
indicate someone replaced the controlled source with a clone).
