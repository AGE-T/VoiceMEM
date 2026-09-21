# VoiceMemAgent v0.10.6 — RELEASE REPORT (Workstream 1)

Audit / release record — British English. Date: 2026-09-20.
Task: "WORKSTREAM 1 — RELEASE THE EXISTING TTS FIX (commit edac5c9)".

## 1. Released version

**0.10.6** — derived from the repository's actual release convention
(sequential patch versions: RELEASE_INDEX 0.10.0 → 0.10.5; the tree
VERSION was 0.10.5 with exactly one unreleased production change, so
the patch release carrying it is 0.10.6).

- Release commit: **124b958** (local repository, branch main)
- Build-time HEAD: 64edfef (recorded in the ZIP's BUILD_INFO)
- ZIP: `releases/VoiceMemAgent_v0.10.6.zip` — 2,923,900 B, 373 files
- **SHA256: A310E679F5F6C1A3043F0DB736607DAB3691410B80CDBFA8CEB00E9864046CCC**
- Recovery package: `VoiceMemAgent_v0.10.6_recovery.zip`
  **SHA256: C6D9080709647B1B85D5CB13AF2B2BDEE60750547745A237F3FFD9DF7A3E8B4F**
  (26,732 B; RECOVERY.md + identity.json + evidence/{changelog_head,
  code.diff, gate_record} + mirrorsync/{push log, fresh-clone
  verification, mirror git log})

## 2. edac5c9 inclusion proof [PROVEN]

- edac5c9 changes `voicemem-agent/app/text_utils.py` only (in
  production code): `PHRASE_RUN_LEADING_BUDGET` (en 1 / hu 0),
  `PHRASE_RUN_LEADING_MIN_EVIDENCE = 2`, `_absorb_leading()` with the
  four guards (host-evidence block, alphabetic-only, no
  hyphen-trimmed suffixes, clause-final punctuation block) — plus the
  61 → 81 test battery and the forensic evidence package.
- The working tree's `app/text_utils.py` was SHA-256
  `739791c7ec3cd321289fae696bc63707901e338dd9b43985d05e9d0fb68a5eca`,
  byte-identical to the edac5c9-committed version (verified before any
  release work; zero drift between the fix and the release).
- The 0.10.6 ZIP's `app/text_utils.py` is byte-identical to that tree
  file, and differs from the v0.10.5 ZIP's copy — the fix is
  demonstrably NEW in the released artefact.
- The implementation was NOT modified at any point of this release
  (no release-gating investigation found a defect requiring correction).

## 3. Gate result

**GREEN — genuinely, reproducibly.** Run on the exact tree that was
packaged (v0.8.1 gate-order: VERSION / pyproject / yaml / page-version /
CHANGELOG entry updated FIRST; the record is fingerprint-bound):

- Total tests: **1521** (floor 1090) — unit 1127, integration 123,
  validation 271
- Failures: 2, **both pinned sandbox environment-gaps** (baseline-
  documented, green on the target machine):
  `test_install_manifest.BuildManifestTests.test_voicemem_controlled_fork_identity_from_pin`,
  `test_voicemem_bridge.VoiceMemBridgeDegradedTests.test_is_available_false_without_package`
- **Regressions: 0**
- Skipped: 30
- Deep validation: PASS [embedding, memory-semantics, release,
  runtime-deps, vad]; SKIP 15
- Tree fingerprint: `672434143af31d709a6b74999044a5e9d3d443357d08890c02a37490d6bc3bc1` (348 files)
- The gate result was reproduced twice (identical numbers) on the
  stable configuration before the build consumed the record.

## 4. Environment status

The previous RED gate was environment/state-loss only. Restored before
this release (all pins verified after install):

- torch 2.7.0+cpu, transformers 5.17.0 (exact), onnxruntime 1.23.0
  (exact), openai 3.14.0 (exact), sentence-transformers 6.1.0,
  mem0ai 2.1.0, qdrant-client 1.19.1; numpy remained 2.1.3 (locked)
- E5 model restored at the pinned revision 614241f6… (496 MB, the
  sentence-transformers file set) — the 16 test_temporal_memory
  failures of the previous gate are green again (39/39)
- Qwen3.6 tokenizer restored at 995ad96e (`/home/z/vmforensic/qwen_tok`)
  — the prompt-size contract measures 6,265 tokens again (test green)
- silero VAD ONNX restored at the pinned revision 394d7e6b…
- All 28 released ZIPs restored into `releases/` with SHA-256
  verification against RELEASE_INDEX (the deep-zip validation is green)

Environment-only deviation, documented: the `pip install -e
vendor/voicemem` variant (which additionally enables the "memory"
deep-validation feature, matching the v0.10.5-era evidence) was tested
and REJECTED — it destabilises this 4 GB sandbox (kernel OOM-kill of
the unit package mid-run; dmesg evidence captured). The stable
configuration without it is the release basis; the deep-validation
skip count (15 vs 14 at v0.10.5) reflects this. No product behaviour
is involved.

## 5. Release integrity

- VERSION = pyproject = config yaml = PAGE_VERSION = CHANGELOG top
  entry = BUILD_INFO (inside the ZIP) = RELEASE_INDEX newest entry =
  **0.10.6** — the release-metadata invariant test is green.
- ZIP self-check passed: all cumulative version markers (v0.4.12 →
  v0.10.5) plus the new V0106 markers (the three fix symbols in
  text_utils.py; the two new test classes and guard tests in
  test_tts_code_switching.py).
- The stale tree `BUILD_INFO.json` (a 0.7.1-era leftover flagged by the
  consolidated audit) was refreshed from the 0.10.6 build — the tree no
  longer carries two conflicting release identities. (Not part of the
  ZIP staging set; a tree-only housekeeping change.)
- Staged for download: `public/VoiceMemAgent_v0.10.6.zip` + sidecar and
  `public/releases/…` (the served paths), byte-identical to
  `releases/…`.

## 6. Mirror

- Sync run via `scripts/sync_github_mirror.sh` (standing additive
  policy; token env-file-only, never in any artefact).
- Mirror HEAD: **4cf11af → dae46e4** (AGE-T/VoiceMEM, main).
- Pushed: the product ZIP + sidecar, the recovery ZIP + sidecar, the
  regenerated README (v0.10.6 as the latest stable release) and the
  full `source/` tree at the 0.10.6 release state.
- Anonymous fresh-clone verification: PASS — both new ZIPs
  byte-identical; README metadata correct; source/VERSION = 0.10.6;
  source/app/text_utils.py byte-identical to the released tree;
  source/BUILD_INFO.json = 0.10.6.
- MirrorSync evidence package: `VoiceMemAgent_v0.10.6_MirrorSync.zip`
  (SHA256 8083E4753734D546D4FB28A0547C4304E5A25DD99BCAAC3DE039E19FE03A5AE4).

## 7. Remaining release-integrity warnings

1. The two pinned sandbox env-gap failures remain (documented in the
   gate baseline; they pass on the target machine with the operator-
   placed model weights and the installed manifest build).
2. The deep-validation "memory" feature is skipped in this sandbox
   (4 GB RAM; the enabling install is OOM-fragile here) — it is an
   evidence nicety, not a gate criterion; the target machine is
   unaffected.
3. The parakeet ASR weights remain operator-placed (never present in
   the sandbox); the placeholder-state contract tests cover that path.

No other warnings. Workstream 1 is closed: the edac5c9 fix is
released, gated, mirrored and recoverable.
