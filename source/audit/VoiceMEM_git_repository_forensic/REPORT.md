# VoiceMEM — GIT REPOSITORY FORENSIC AUDIT

**Date:** 2026-09-20
**Type:** Audit only. Read-only. No history rewrite, no rebase, no squash, no merge,
no push performed as part of the audit itself (the post-audit evidence push is a
separate, additive step ordered by the operator).
**Scope:** Baseline identification (task §1) and Git repository forensics (task §2)
of the first-party tree at `/home/z/my-project`.
**Evidence discipline:** every important claim carries `[PROVEN]`, `[INFERENCE]`
or `[UNKNOWN]` with its source. The audit was executed against git HEAD
`a3e3a8af7d52a2cef786b573cfebd16af5b8bc4c`.

---

## 1. BASELINE IDENTIFICATION

### 1.1 Exact current baseline

| Property | Value | Evidence |
|---|---|---|
| Production version | **0.10.5** | `voicemem-agent/VERSION` = `0.10.5`; `pyproject.toml [project] version = "0.10.5"`; `config/voicemem_config.yaml project.version: "0.10.5"`; `releases/RELEASE_INDEX.json` newest entry 0.10.5 `[PROVEN]` |
| Git commit | **a3e3a8af7d52a2cef786b573cfebd16af5b8bc4c** (branch `main`) | `git rev-parse HEAD` `[PROVEN]` |
| Working tree | **Clean for all production paths** | `git status --porcelain`: only `m tool-results/voicemem-src` (gitlink content drift, §2.6) and four untracked `worklog.*.md` subagent notes (merged into `worklog.md` by this audit, then removed) `[PROVEN]` |
| Tags | One: `voicemem-ownership-baseline` | annotated tag; message is a UUID placeholder, not a release marker — it predates the current development line `[PROVEN]` |
| Remotes | **None configured** on this repository | `git remote -v` empty. The distribution mirror `AGE-T/VoiceMEM` is a *separate* repository synchronised by `scripts/sync_github_mirror.sh` (§1.4) `[PROVEN]` |
| Total commits | 184 | `git log --oneline | wc -l` `[PROVEN]` |
| Vendor pin | `vendor/voicemem` = controlled fork of `xzf-thu/VoiceMem` at upstream commit `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` (tag v0.0.1, package 0.2.3), forked 2026-09-10; 21 local patches + 4 ported upstream fixes recorded in `VOICEMEM_PIN.json` | `VOICEMEM_PIN.json` (read in full) `[PROVEN]` |
| Upstream clone | `tool-results/voicemem-src` — a git **gitlink** (mode 160000) pinned at exactly `e8384e08…` | `git ls-files -s | grep 160000` `[PROVEN]` |

### 1.2 Whether the working tree differs from the expected 0.10.5 release

**It does — by exactly one production change.** `[PROVEN]`

- The released artefact `public/VoiceMemAgent_v0.10.5.zip` (SHA-256
  `401814AC02E81E6258C9BF8C8E6D3656482769ED56AE789CCBDA8AA6BBE7C01C`,
  byte-identical to its sidecar — verified) contains
  `app/text_utils.py` with SHA-256 `817b1a3a…`.
- The current working tree's `app/text_utils.py` has SHA-256 `739791c7…`.
- The single intervening commit is `edac5c9` (2026-09-19, "TTS code-switching
  forensic + fix: unquoted idiom heads absorbed into their evidence runs"),
  the leading-neutral absorption fix (`app/text_utils.py` only).

**Consequence (release-integrity finding):** the leading-absorption fix exists
only in the first-party tree. `VoiceMemAgent_v0.10.5_TTSForensicFix.zip`
(28,611 B, SHA-256 verified against its sidecar) is an **audit evidence
package** (`VoiceMEM_tts_codeswitch_v0105_forensic/` report + tools + traces) —
it does **not** carry a patched `text_utils.py`. An operator installing from
`public/VoiceMemAgent_v0.10.5.zip` therefore still receives the pre-fix
segmentation behaviour. The fix is **unreleased as a product version**;
distribution requires a version bump + build (0.10.5-r1 or 0.10.6) through
`scripts/build_release_sandbox.py` / `build_release.ps1`. `[PROVEN]`

### 1.3 Release and gate identity of the released artefact

- The zip's inner `BUILD_INFO.json` is self-consistent: version 0.10.5,
  built `2026-09-18T16:41:23Z`, git commit `0a135a6c…`, gate record
  GREEN, 1478 tests, 0 regressions, 4 pinned env-gap failures, tree
  fingerprint `634ac479…`. `[PROVEN]`
- `releases/RELEASE_INDEX.json` records the same zip hash, size 2,876,103 B
  and the same gate record — the served artefact, the index and the tree agree.
  `[PROVEN]`
- **Hygiene finding:** the working tree's `voicemem-agent/BUILD_INFO.json` is
  **stale at version 0.7.1** (built 2026-09-13). The build regenerates it inside
  the zip; the tree copy is not refreshed post-build. Harmless to runtime
  (nothing reads it at runtime — it is a build artefact), but it misrepresents
  the tree's release identity to a human reader. `[PROVEN]`

### 1.4 Mirror and recovery/release integrity

- **Mirror** (`github.com/AGE-T/VoiceMEM`): `git ls-remote` (authenticated,
  read-only) shows `refs/heads/main = a7ddce56996db3abaaf5b90071d7a31e60578276`.
  The landing page badge `GITHUB_MIRROR_HEAD = 'a7ddce5'`
  (`src/app/page.tsx:188`) matches. The mirror is current **through the TTS
  forensic-fix sync** (commit `16f24c3`, 2026-09-19: source tree +
  `TTSForensicFix.zip`). The three later audit-only commits (`51ad195`,
  `382100d`, `a3e3a8a`) had **not** been mirrored at audit time — this audit's
  closing evidence push closes that gap. `[PROVEN]`
- **Evidence packages in `public/`** — all SHA-256-verified against sidecars
  during this audit: `v0.10.5.zip` (401814AC…), `v0.10.5_MirrorSync.zip`
  (A00C96BA…), `v0.10.5_MemoryPerformanceAudit.zip` (5ceb6bd1…, mirror-confirmed
  as `974eee0` in the mirror repo per commit `6535497`),
  `v0.10.5_TTSForensicFix.zip` (b5ba4726…). `[PROVEN]`
- **Recovery packages:** `recovery/` holds `v0102-llmpriority`, `v0103`,
  `v0104`. **No dedicated `recovery/v0105/` package exists.** For v0.10.5 the
  rollback anchors are: the release zip + sidecar, the git history, the
  MirrorSync evidence and the audit trails — the dedicated
  `RECOVERY.md`+`identity.json` convention used for v0.10.1–v0.10.4 was not
  repeated for v0.10.5 or for the post-release fix. `[PROVEN]` (absence)
  *[Operator decision whether to backfill one before the next release.]*

### 1.5 Uncommitted / untracked files affecting production behaviour

**None.** `[PROVEN]` The only working-tree differences are (a) the gitlink
content drift inside `tool-results/voicemem-src` (§2.6, mode-only, non-production),
and (b) this audit's own untracked notes before merging. No file under
`voicemem-agent/app/`, `voicemem-agent/vendor/`, `voicemem-agent/config/`,
`voicemem-agent/web/`, `mini-services/`, `src/` or `scripts/` is modified
relative to HEAD.

---

## 2. GIT REPOSITORY FORENSIC

### 2.1 Chronology of the current development line

Releases in this repository follow a strict build-gated discipline
(`releases/RELEASE_INDEX.json` + `releases/BUILD_HISTORY.json`, every release
entry carrying its gate record). The 0.10.x line, condensed from the full log
(184 commits; earlier 0.4.x–0.9.x history is intact and consistent):

| Commit | Date (2026) | Subject (abbreviated) | Class |
|---|---|---|---|
| `7b4d3d8` | 09-15 | v0.10 PHASE 0 read-only baseline + semantic map | audit |
| `324a372` | 09-15 | v0.10.0 memory semantics & temporal memory | release (1350 tests, 0 reg) |
| `a394b69` | 09-15 | v0.10.0 release executed (zip 472FB043…) | release |
| `8625071` | 09-15 | v0.10.0 recovery package | recovery |
| `d9a80ac` | 09-15 | v0.10.1 P0 ASR forensic release (transformers exact pin) | release (1358/0) |
| `b50e9d0` | 09-15 | v0.10.2 P0 LLM-PRIORITY release (bg cancellation, gate v2) | release (1386/0) |
| `7eaac4b` | 09-16 | v0.10.3 upstream deep-audit selective port | release (1407/0) |
| `6d6cd8b` | 09-17 | v0.10.4 forensic logging release | release (1436/0) |
| `37baefb` | 09-18 | **v0.10.5 release: TTS code-switching span routing** | release (1478/0) |
| `b7160ce` | 09-18 | v0.10.5 release verification evidence (browser) | audit |
| `16cf08a` | 09-18 | TTS code-switching phase 2: gate evidence (1497/0) | audit |
| `e597cd3` | 09-18 | architecture prep (learning-engine plan, localisation) | audit (1501/0) |
| `5ba3246` | 09-18 | memory-layer performance & stability audit | audit-only |
| `6535497` | 09-18 | mirror: MemoryPerformanceAudit evidence package | mirror |
| `edac5c9` | 09-19 | **TTS forensic fix: unquoted idiom heads (production change)** | fix (unreleased) |
| `16f24c3` | 09-19 | mirror badge restamp f138325→a7ddce5 | mirror |
| `51ad195` | 09-19 | Session Continuity Architecture Study (1004 lines) | audit |
| `382100d` | 09-19 | Dormant Session Pipeline Audit (897 lines) | audit |
| `a3e3a8a` | 09-20 | Session Backlog Forensic (572 lines) + TTS forensic evidence | audit |

Interleaved are UUID-subject commits (`8c8bc93`, `0a135a6`, `926b721`, …) —
continuation-session snapshots produced by the development environment at
session boundaries (§2.5). `[PROVEN]`

### 2.2 Local-only functional changes (first-party vs upstream)

The vendor delta is fully enumerated in `VOICEMEM_PIN.json`:
VM-LOCAL-001…021 + VM-LOCAL-EN, plus ported upstream fixes
(961efe8, 91d2e42, f535f9d, e3cc965). The **app layer**
(`voicemem-agent/app/`, absent upstream) is entirely first-party. Release
0.10.5 + the post-release fix are first-party TTS work. `[PROVEN]`

### 2.3 Potentially duplicated / overlapping work

- **Two TTS code-switching fixes exist in history**: `bee5fea` (v0.10.4,
  quoted-region span routing) and `37baefb` (v0.10.5, the evidence-gated
  redesign) plus `edac5c9` (leading absorption). These are **layered, not
  duplicated**: v0.10.5 replaced v0.10.4's shipped mechanism
  (the v0.10.4 zip served for one day; v0.10.5 shipped two days later), and
  `edac5c9` extends v0.10.5 without replacing it. `[PROVEN]`
- **Upstream overlap**: v0.10.3 already ported the applicable upstream fixes
  (heartnote cap, response_experience gate, recency narrowing); the upstream
  matrix in the companion report
  (`audit/VoiceMEM_current_upstream_selective_merge/REPORT.md`) confirms no
  remaining duplication pressure — upstream main is static since 2026-09-05.
  `[PROVEN]`

### 2.4 History anomalies

1. **UUID-subject commits** (e.g. `a3e3a8a`, `382100d`, `51ad195`,
   `bb828da`, `8c8bc93`, …) — environment-generated session snapshots. Their
   contents are legitimate (audit artefacts, pid files), but they carry no
   human-readable subjects. Cosmetic hygiene item only. `[PROVEN]`
2. **`.zscripts/dev.pid` is tracked** and changes in snapshot commits —
   runtime noise inside version control. Cosmetic. `[PROVEN]`
3. **`tool-results/` is tracked**, including raw bash outputs and the upstream
   clone as a gitlink. This is deliberate (forensic evidence retention) and the
   gitlink is a *feature* — the upstream pin is reproducible from the
   repository itself. `[PROVEN]`
4. **Annotated tag with UUID message** (`voicemem-ownership-baseline`) —
   predates the release line, message body is a placeholder. Cosmetic. `[PROVEN]`
5. **No merge commits, no cherry-picks, no reverts** on the main line —
   the history is linear. `[PROVEN]`

### 2.5 The post-release gate record (verdict RED — environment class)

The forensic-fix session's official gate run
(`audit/VoiceMEM_tts_codeswitch_v0105_forensic/evidence/gate_run_forensic.txt`)
records **VERDICT: RED** (14 pinned env-gap + 17 failures: Silero/Parakeet
model files absent after a sandbox state-loss, venv-extras import failures,
temporal-memory `setUpClass` on `from openai import OpenAI`). The fix's
innocence is proven in `TEST_RESULTS.md` §5: (a) the identical 4-file subset
run with and without the fix (`git stash`) yields the identical failure set;
(b) the pre-fix and post-fix official gate runs are byte-identical in their
failure sets. **Implication:** the *unreleased* fix has never been validated on
a fully-populated environment; the next release build must re-run the gate on
the real target (models restored) before shipping. `[PROVEN]`

### 2.6 `tool-results/voicemem-src` drift (the only dirty path)

`git status` reports modified content in the gitlink. Inside the clone:
159 files differ **in file mode only** (100644→100755; `git diff --shortstat`
= 0 insertions, 0 deletions) — a side effect of clone/checkout on this
filesystem. No content drift; the pin remains `e8384e08…`. `[PROVEN]`

### 2.7 GIT REPOSITORY AUDIT — summary block

- **Canonical baseline:** v0.10.5 release commit `0a135a6` (zip 401814AC…,
  gate GREEN 1478/0) + one unreleased production fix `edac5c9`
  (tree HEAD `a3e3a8a`); tree-clean; mirror `AGE-T/VoiceMEM` @ `a7ddce5`
  (current through the forensic-fix sync; audit commits mirrored by this
  audit's closing push).
- **Significant release lineage:** v0.10.0 → v0.10.5, every release
  build-gated and evidence-packaged (§2.1).
- **Local-only functional changes:** app layer + VM-LOCAL-001…021 vendor
  ledger (§2.2).
- **Upstream-derived changes:** four ported fixes + the v0.10.3 port trio
  (§2.3; companion upstream report).
- **Audit-only changes:** v010 baseline, memory-performance, S8 correlation,
  session-continuity trio, TTS audits — all additive under `audit/` and
  `docs/`.
- **Recovery/release infrastructure:** MirrorSync packages per release;
  dedicated recovery packages exist for v0.10.1–v0.10.4 only.
- **Potentially duplicated work:** none found (layered fixes, §2.3).
- **Potentially stale work:** the v0.10.4 code-switching implementation
  (superseded by v0.10.5 within its own release lineage — retained as
  history, correctly); `BUILD_INFO.json` tree copy (0.7.1).
- **History anomalies:** UUID snapshot commits, tracked `dev.pid`,
  UUID-message tag (§2.4) — all cosmetic.
- **Recommended Git hygiene actions (audit-only recommendations, nothing
  executed):** (1) refresh `BUILD_INFO.json` in the tree or stop tracking the
  build-stamped copy; (2) ship the `edac5c9` fix through a proper version
  build once the gate is green on a restored environment; (3) consider a
  v0.10.5 recovery-package backfill or an explicit decision note; (4) keep the
  UUID-snapshot pattern out of future release-critical commits where
  practical. No rebase/squash/rewrite is recommended or performed.
- **Changes that should NOT be touched:** the entire `audit/` evidence
  history, the vendor pin + `VOICEMEM_PIN.json` ledger, the release index and
  build history, and the v0.10.2 background-cancellation semantics
  (LLM-priority invariant).

---

## 3. COMPLIANCE

- Production code changes by this audit: **0**. Commits: only this audit's
  additive artefacts (reports + worklog) — ordered by the operator.
- Database changes: **0** (no production DB exists in this environment —
  see the backlog-forensic addendum). LLM/embedding runs: **0**.
- History rewrites / merges / pushes-as-audit-action: **0** (the single
  post-audit evidence push to the mirror is the operator-ordered additive
  upload, §1.4).
