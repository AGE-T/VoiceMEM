# VoiceMEM — VoiceMemAgent Releases

Distribution mirror for **VoiceMemAgent**, the one-click, offline-capable
voice memory agent. This repository hosts the versioned release packages
and their verification artifacts — the canonical build source lives in
the agent workspace.

**Latest stable release: `v0.10.7`**

---

## Packages

| File | What it is | Size |
| --- | --- | --- |
| `VoiceMemAgent_v0.10.7.zip` | **The product.** Full agent source, machine-independent (no runtime state inside: no `.venv`, models, or memory). Install = unzip + `START.bat`. | 2.7 MB |
| `VoiceMemAgent_v0.5.0_GitHubSync.zip` | **First authenticated sync evidence**: token permission check, push output, fresh-clone verification of the mirror state after the sync. | 6 KB |
| `VoiceMemAgent_v0.5.0_MirrorCheck.zip` | **The mirror check**: proof that this GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides, git clone evidence). Verdict: `PASS`. | 8 KB |
| `VoiceMemAgent_v0.5.0_SyncSetup.zip` | **Standing sync policy + infrastructure**: the sync script, the procedure document, and the setup-run evidence (every future update goes to the GitHub mirror). | 12 KB |
| `VoiceMemAgent_v0.5.0_Verification.zip` | **The verification record** (not a product release): independent verification report — phases + engineering decisions, evidence files (pin verification, clean-install manifest, retrieval baseline, trait assessment, test gate), `identity.json` and `RECOVERY.md`. | 29 KB |
| `VoiceMemAgent_v0.5.1_ForensicTrace.zip` | **TASK 1 ASR→LLM forensic trace**: the proven break point (CLI bridge adapter tables referenced non-existent vendor API — every CLI turn ran memoryless), the smallest-correct-fix patch, the post-fix trace matrix (S1–S7b green; thinking-only fails loud), and the full test-gate record. | 41 KB |
| `VoiceMemAgent_v0.5.1_Task2Gate.zip` | **TASK 2 rich-retrieval DESIGN GATE** (measurement-only): the authoritative baseline of both retrieval paths, runtime call graphs, the API design comparison (recommended: app-level search_rich), the rendering/temporal/speaker audits and the 15-test plan. | 35 KB |
| `VoiceMemAgent_v0.5.2_MemorySafetyGate.zip` | **TASK 1.5 memory-safety gate** (v0.5.2): the Claude-audit reconciliation, the three safety patches (right-brain delete gate; non-destructive UPDATE with explicit supersession; left delete graph cascade), the P0 before/after proof, the 15-test behavioural battery (real vendor + real store) and the 954/0/26 release gate. | 139 KB |
| `VoiceMemAgent_v0.6.0_ASRGate.zip` | **Evidence package** for the `ASRGate.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 566 KB |
| `VoiceMemAgent_v0.6.0_LLMConfig_ngl20-c32768.zip` | **Evidence package** for the `ngl20-c32768.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 151 KB |
| `VoiceMemAgent_v0.6.0_Recovery.zip` | **Evidence package** for the `Recovery.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 215 KB |
| `VoiceMemAgent_v0.6.1_ASRHotfix.zip` | **Evidence package** for the `ASRHotfix.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 31 KB |
| `VoiceMemAgent_v0.6.2_PinHotfix.zip` | **Evidence package** for the `PinHotfix.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 14 KB |
| `VoiceMemAgent_v0.7.0_TTSMigration.zip` | **Evidence package** for the `TTSMigration.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 2.0 MB |
| `VoiceMemAgent_v0.10.0_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 9 KB |
| `VoiceMemAgent_v0.10.2_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 10 KB |
| `VoiceMemAgent_v0.10.3_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 2 KB |
| `VoiceMemAgent_v0.10.4_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 9 KB |
| `VoiceMemAgent_v0.10.4_S8LatencyCorrelation.zip` | **Evidence package** for the `S8LatencyCorrelation.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 531 KB |
| `VoiceMemAgent_v0.10.5_MemoryPerformanceAudit.zip` | **Evidence package** for the `MemoryPerformanceAudit.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 48 KB |
| `VoiceMemAgent_v0.10.5_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 11 KB |
| `VoiceMemAgent_v0.10.5_TTSForensicFix.zip` | **Evidence package** for the `TTSForensicFix.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 28 KB |
| `VoiceMemAgent_v0.10.6_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 5 KB |
| `VoiceMemAgent_v0.10.6_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 26 KB |
| `VoiceMemAgent_v0.10.7_LlamaB11073Experimental.zip` | **Evidence package** for the `LlamaB11073Experimental.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 74 KB |
| `VoiceMemAgent_v0.10.7_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 5 KB |
| `VoiceMemAgent_v0.10.7_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 17 KB |
| `VoiceMem_asrregression_v0101_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 33 KB |
| `VoiceMem_forensic_v0104_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 38 KB |
| `VoiceMem_llmconfig_v072_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 34 KB |
| `VoiceMem_llmpriority_v0102_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 36 KB |
| `VoiceMem_memoryfixes_v080_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 13 KB |
| `VoiceMem_memorysemantics_v090_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 61 KB |
| `VoiceMem_runtimeopt_v092_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 45 KB |
| `VoiceMem_stabilization_v081_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 52 KB |
| `VoiceMem_temporal_v010_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 36 KB |
| `VoiceMem_upstreamaudit_v0103_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 44 KB |
| `VoiceMem_vadfix_v091_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 37 KB |

## Verify your download

```bash
sha256sum VoiceMemAgent_v0.10.7.zip VoiceMemAgent_v0.5.0_GitHubSync.zip VoiceMemAgent_v0.5.0_MirrorCheck.zip VoiceMemAgent_v0.5.0_SyncSetup.zip VoiceMemAgent_v0.5.0_Verification.zip VoiceMemAgent_v0.5.1_ForensicTrace.zip VoiceMemAgent_v0.5.1_Task2Gate.zip VoiceMemAgent_v0.5.2_MemorySafetyGate.zip VoiceMemAgent_v0.6.0_ASRGate.zip VoiceMemAgent_v0.6.0_LLMConfig_ngl20-c32768.zip VoiceMemAgent_v0.6.0_Recovery.zip VoiceMemAgent_v0.6.1_ASRHotfix.zip VoiceMemAgent_v0.6.2_PinHotfix.zip VoiceMemAgent_v0.7.0_TTSMigration.zip VoiceMemAgent_v0.10.0_MirrorSync.zip VoiceMemAgent_v0.10.2_MirrorSync.zip VoiceMemAgent_v0.10.3_MirrorSync.zip VoiceMemAgent_v0.10.4_MirrorSync.zip VoiceMemAgent_v0.10.4_S8LatencyCorrelation.zip VoiceMemAgent_v0.10.5_MemoryPerformanceAudit.zip VoiceMemAgent_v0.10.5_MirrorSync.zip VoiceMemAgent_v0.10.5_TTSForensicFix.zip VoiceMemAgent_v0.10.6_MirrorSync.zip VoiceMemAgent_v0.10.6_recovery.zip VoiceMemAgent_v0.10.7_LlamaB11073Experimental.zip VoiceMemAgent_v0.10.7_MirrorSync.zip VoiceMemAgent_v0.10.7_recovery.zip VoiceMem_asrregression_v0101_recovery.zip VoiceMem_forensic_v0104_recovery.zip VoiceMem_llmconfig_v072_recovery.zip VoiceMem_llmpriority_v0102_recovery.zip VoiceMem_memoryfixes_v080_recovery.zip VoiceMem_memorysemantics_v090_recovery.zip VoiceMem_runtimeopt_v092_recovery.zip VoiceMem_stabilization_v081_recovery.zip VoiceMem_temporal_v010_recovery.zip VoiceMem_upstreamaudit_v0103_recovery.zip VoiceMem_vadfix_v091_recovery.zip
```

Expected:

```text
c5a6c4e30056b0ee3a636cd7cdc6315155ab1334f04406ba57acba3a7b684dc3  VoiceMemAgent_v0.10.7.zip
1b4596351d8ce45e0f750e1c21c0693366fdb0ab9ab3c0a627335e06bb353847  VoiceMemAgent_v0.5.0_GitHubSync.zip
89bc2933843301e1317708e2c4b98083ef3129d149611c945a9ee9dd6a44bc14  VoiceMemAgent_v0.5.0_MirrorCheck.zip
bce2979501864e54281e9615d563d5a0f3148c690b53e105060f58ff251be559  VoiceMemAgent_v0.5.0_SyncSetup.zip
aa880796031a5ada388a75cdae26befc723d6e37988589d4a0bf6b91331db631  VoiceMemAgent_v0.5.0_Verification.zip
68b3c0dbf751bc61ac38458e7efde739b6de7e978e3d6f990382e6ce704caedc  VoiceMemAgent_v0.5.1_ForensicTrace.zip
b24ea62f65ce77bdaf94fb843e10e8c1b88c7e1fcd37c064806e948d538abbbb  VoiceMemAgent_v0.5.1_Task2Gate.zip
4d9d3dc7e384b45a22ab5edf3bd6476748d5e0c0d0364ef279800121f098c8cf  VoiceMemAgent_v0.5.2_MemorySafetyGate.zip
54274f8dec40e068e6aeba3b29aa3187241c0c7f2c54fb7d15872a5250d9a2f9  VoiceMemAgent_v0.6.0_ASRGate.zip
e36b9abed3cecde6b94adb69acfcf2fea144293c3cbc4f94b1af4a2b0e583aae  VoiceMemAgent_v0.6.0_LLMConfig_ngl20-c32768.zip
5803e435538c1fa08baf00cc6dc5b810c2af380dbfd05d9dddbc49be0a058614  VoiceMemAgent_v0.6.0_Recovery.zip
9fc73340a376928e6952a274b66e50931de8d42ad64cccbe8eaa9f04ff9f534e  VoiceMemAgent_v0.6.1_ASRHotfix.zip
258dac460815501d706bbe9c237605c9752cba96a980307eed299bc7f87ad66f  VoiceMemAgent_v0.6.2_PinHotfix.zip
6a5e832a42030747449caecb0109877e3bcce1ed8896f786b8ea8355b764d3a8  VoiceMemAgent_v0.7.0_TTSMigration.zip
5ca6eec78f3079515d62d4cdb12418ab5a4030002cda8d800e980322de83c20b  VoiceMemAgent_v0.10.0_MirrorSync.zip
dfa214e1dc38659a6a4779b1b112cbba98b512bacbb5cd8359aed560e9c7ab8b  VoiceMemAgent_v0.10.2_MirrorSync.zip
1ece3fce271a7cb41cdf8e845a194dc29c78d6b1ad1b1e38b8220956b79a52bf  VoiceMemAgent_v0.10.3_MirrorSync.zip
dbb35358eb78a071f084a46088894cf2e0c2c2e6b17fc0918735c6e538f64787  VoiceMemAgent_v0.10.4_MirrorSync.zip
f9ace279cebe98abb8f438352ce724ad63f0fec7f8bbd22fb8308b055b422146  VoiceMemAgent_v0.10.4_S8LatencyCorrelation.zip
5ceb6bd176f9ed309b6f67a9112629a5b3282b4e13cd6965e0c9c1be74eb8e28  VoiceMemAgent_v0.10.5_MemoryPerformanceAudit.zip
a00c96ba5319babd643984eb1ef4ebd40ca216a7d2b17ab844fb257dc601394f  VoiceMemAgent_v0.10.5_MirrorSync.zip
b5ba47263d7388bb03d599260f21040c2a86d5ce4da863b1e393ad1c75787712  VoiceMemAgent_v0.10.5_TTSForensicFix.zip
8083e4753734d546d4fb28a0547c4304e5a25dd99bcaac3de039e19fe03a5ae4  VoiceMemAgent_v0.10.6_MirrorSync.zip
c6d9080709647b1b85d5cb13af2b2bdee60750547745a237f3ffd9df7a3e8b4f  VoiceMemAgent_v0.10.6_recovery.zip
d28f7f8c275828318a7bd12a581bc2e7e8f27aac61a2f3c2d5cd9fe8e5e6e5b5  VoiceMemAgent_v0.10.7_LlamaB11073Experimental.zip
e0f9f7c911e476db55265e71ea1a69819b426e2a289fdf93c99939f74ba051c7  VoiceMemAgent_v0.10.7_MirrorSync.zip
8b601582901bfd4e9f4a37253346140d025f7318c487038b2fa9c0b1f4bfa70e  VoiceMemAgent_v0.10.7_recovery.zip
59f0df85711dd0eb6b6d4a2848878c47c421a204a504d31963d2b74c0571bb9c  VoiceMem_asrregression_v0101_recovery.zip
514ecdd8010e78efcc6c5cd1dd31b062c41b6a8d85019882fa597a442274842c  VoiceMem_forensic_v0104_recovery.zip
f11b7aaf013be3db5f1c379ce309c3b297152e4e093e18652a98672fb85e2687  VoiceMem_llmconfig_v072_recovery.zip
06b065abe83e42f01f162baa1711fa22b2fddbcb63ed79851ccae567630220bd  VoiceMem_llmpriority_v0102_recovery.zip
cc54ffbb29af3b4cad14bf03da9f5ad288389958bc8001d21cb204fec1dcb28e  VoiceMem_memoryfixes_v080_recovery.zip
25d35796d6abbce8e2325f2401218d8fe65ede79f71e2c255cfb1e37c86a5a5c  VoiceMem_memorysemantics_v090_recovery.zip
f8052ecb67451be38dfddfc428ff3ec6f18976a6f8865f0051182c18a168487e  VoiceMem_runtimeopt_v092_recovery.zip
b4218a4c6a1f862791371a646a3f6281fb02285320b0fa625341cb572d5fb1fc  VoiceMem_stabilization_v081_recovery.zip
e5ac3e542790cdfc6f1649502910513bad2e886cab0482cc887845c9d2b29cee  VoiceMem_temporal_v010_recovery.zip
4b7990ec4f366648d1d944f10b2bfa3a01f15279735df8bb7196c66f2d848fc5  VoiceMem_upstreamaudit_v0103_recovery.zip
1336bd647415d357c85bb6dde1801f5ef936217575a23d74e18c71533ecac5f4  VoiceMem_vadfix_v091_recovery.zip
```

(`.sha256` sidecar files are included next to each ZIP.)

## Install (Windows)

1. Unzip `VoiceMemAgent_v0.10.7.zip` into any folder (e.g. `F:\Voicemem\VoiceMemAgent`).
2. Double-click **`START.bat`** — that is the single user entry point.
3. Wait for the bootstrap: Python check, `.venv`, dependencies, HF tooling,
   model download (`MODELS.lock.json`), configuration, smoke tests.
4. If everything is green, the agent starts — you can talk to it.

Advanced modes: `START.bat mock` · `check` · `benchmark` · `repair` · `build`.
At a single end-to-end failure, re-run `START.bat repair`.

## The verification story (v0.10.7)

- **Vendor identity:** `vendor/voicemem` pinned at
  `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` (= upstream tag `v0.0.1`;
  the `0.2.3` package version is **not** the identity anchor).
- **Pin verification:** 3-mode script (source / runtime / embedding), with
  an instrumented fake OpenAI client proving the OpenAI client is never
  constructed — all green.
- **Clean install:** fresh ZIP → new venv → editable install → import
  identity → `pin_verified: true` in the manifest.
- **Test gate (v0.10.5 release gate):** 1478 passed / 0 regressions /
  28 skipped (4 pinned sandbox env-gap — documented in the pinned
  baseline, green on the target machine); deep validation PASS
  (embedding, memory, memory-semantics, release, runtime-deps, vad) on
  fingerprint
  634ac479ab3b230f28775ed16a4601cb324cbb49b0c38a94ec14d32907672c08
  — the stamped record ships inside the ZIP (`releases/gate_record.json`).
- **Local vendor patches:** 24 files vs the pinned upstream
  (21 modified + 3 added: right-brain stance, temporal ranking,
  background-LLM cancellation, query-gated recency); patch ledger
  `VOICEMEM_PIN.json` (VM-LOCAL-001…021 + EN) + the v0.10.x
  CHANGELOG entries — permanently local, never upstreamed.
- Full details: open the `*_Verification.zip` → `docs/` report.

## Mirrors

- This repository is the **external distribution mirror**, synced with
  write access granted by the owner (token-scoped, revocable).
- Standing sync policy: every future update (new release + evidence
  packages) is pushed here automatically (`scripts/sync_github_mirror.sh`).

## Source tree (`source/`)

- The full **first-party source-controlled tree** (the VoiceMemAgent
  workspace: app/, vendor/voicemem controlled fork, tests/, scripts/,
  web/, config/, docs/ + root files) lives under `source/` — synced
  additively with every update since 2026-09-10 (TASK 1.5, closing
  audit finding CD-6: the mirror previously tracked only ZIP blobs).
- Runtime state is NOT mirrored (no `.venv`, models, data, memory,
  logs); release ZIPs stay at the repository root.
- The ZIP under `source/releases/` is intentionally excluded — the
  root-level ZIP **is** the release artifact.

---

## Magyar összefoglaló

Ez a repó a **VoiceMemAgent** kiadásainak külső tükre. A legfrissebb
stabil verzió a **v0.10.7**. Mit találhat itt:

- **`VoiceMemAgent_v0.10.7.zip`** — maga a termék. Gépfüggetlen forrás,
  telepítés: kicsomagolás + `START.bat` dupla kattintás, minden mást a
  bootstrap elintézi.
- **Verifikációs / evidencia ZIP-ek** — a kiadások független
  ellenőrzésének dokumentumai és a tükör szinkron bizonyítékai.

Letöltés után ellenőrizd a SHA-256 összegeket (fenti táblázat). A vendor
identitás a `e8384e0…` pin (nem a 0.2.3-as csomagverzió).
