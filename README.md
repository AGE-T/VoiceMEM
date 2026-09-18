# VoiceMEM — VoiceMemAgent Releases

Distribution mirror for **VoiceMemAgent**, the one-click, offline-capable
voice memory agent. This repository hosts the versioned release packages
and their verification artifacts — the canonical build source lives in
the agent workspace.

**Latest stable release: `v0.10.5`**

---

## Packages

| File | What it is | Size |
| --- | --- | --- |
| `VoiceMemAgent_v0.10.5.zip` | **The product.** Full agent source, machine-independent (no runtime state inside: no `.venv`, models, or memory). Install = unzip + `START.bat`. | 2.7 MB |
| `VoiceMemAgent_v0.10.0_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 9 KB |
| `VoiceMemAgent_v0.10.2_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 10 KB |
| `VoiceMemAgent_v0.10.3_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 2 KB |
| `VoiceMemAgent_v0.10.4_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 9 KB |
| `VoiceMemAgent_v0.10.4_S8LatencyCorrelation.zip` | **Evidence package** for the `S8LatencyCorrelation.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 531 KB |
| `VoiceMemAgent_v0.10.5_MemoryPerformanceAudit.zip` | **Evidence package** for the `MemoryPerformanceAudit.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 48 KB |
| `VoiceMemAgent_v0.10.5_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 11 KB |
| `VoiceMem_asrregression_v0101_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 33 KB |
| `VoiceMem_forensic_v0104_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 38 KB |
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
sha256sum VoiceMemAgent_v0.10.5.zip VoiceMemAgent_v0.10.0_MirrorSync.zip VoiceMemAgent_v0.10.2_MirrorSync.zip VoiceMemAgent_v0.10.3_MirrorSync.zip VoiceMemAgent_v0.10.4_MirrorSync.zip VoiceMemAgent_v0.10.4_S8LatencyCorrelation.zip VoiceMemAgent_v0.10.5_MemoryPerformanceAudit.zip VoiceMemAgent_v0.10.5_MirrorSync.zip VoiceMem_asrregression_v0101_recovery.zip VoiceMem_forensic_v0104_recovery.zip VoiceMem_llmpriority_v0102_recovery.zip VoiceMem_memoryfixes_v080_recovery.zip VoiceMem_memorysemantics_v090_recovery.zip VoiceMem_runtimeopt_v092_recovery.zip VoiceMem_stabilization_v081_recovery.zip VoiceMem_temporal_v010_recovery.zip VoiceMem_upstreamaudit_v0103_recovery.zip VoiceMem_vadfix_v091_recovery.zip
```

Expected:

```text
401814ac02e81e6258c9bf8c8e6d3656482769ed56ae789ccbda8aa6bbe7c01c  VoiceMemAgent_v0.10.5.zip
5ca6eec78f3079515d62d4cdb12418ab5a4030002cda8d800e980322de83c20b  VoiceMemAgent_v0.10.0_MirrorSync.zip
dfa214e1dc38659a6a4779b1b112cbba98b512bacbb5cd8359aed560e9c7ab8b  VoiceMemAgent_v0.10.2_MirrorSync.zip
1ece3fce271a7cb41cdf8e845a194dc29c78d6b1ad1b1e38b8220956b79a52bf  VoiceMemAgent_v0.10.3_MirrorSync.zip
dbb35358eb78a071f084a46088894cf2e0c2c2e6b17fc0918735c6e538f64787  VoiceMemAgent_v0.10.4_MirrorSync.zip
f9ace279cebe98abb8f438352ce724ad63f0fec7f8bbd22fb8308b055b422146  VoiceMemAgent_v0.10.4_S8LatencyCorrelation.zip
5ceb6bd176f9ed309b6f67a9112629a5b3282b4e13cd6965e0c9c1be74eb8e28  VoiceMemAgent_v0.10.5_MemoryPerformanceAudit.zip
a00c96ba5319babd643984eb1ef4ebd40ca216a7d2b17ab844fb257dc601394f  VoiceMemAgent_v0.10.5_MirrorSync.zip
59f0df85711dd0eb6b6d4a2848878c47c421a204a504d31963d2b74c0571bb9c  VoiceMem_asrregression_v0101_recovery.zip
514ecdd8010e78efcc6c5cd1dd31b062c41b6a8d85019882fa597a442274842c  VoiceMem_forensic_v0104_recovery.zip
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

1. Unzip `VoiceMemAgent_v0.10.5.zip` into any folder (e.g. `F:\Voicemem\VoiceMemAgent`).
2. Double-click **`START.bat`** — that is the single user entry point.
3. Wait for the bootstrap: Python check, `.venv`, dependencies, HF tooling,
   model download (`MODELS.lock.json`), configuration, smoke tests.
4. If everything is green, the agent starts — you can talk to it.

Advanced modes: `START.bat mock` · `check` · `benchmark` · `repair` · `build`.
At a single end-to-end failure, re-run `START.bat repair`.

## The verification story (v0.10.5)

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
stabil verzió a **v0.10.5**. Mit találhat itt:

- **`VoiceMemAgent_v0.10.5.zip`** — maga a termék. Gépfüggetlen forrás,
  telepítés: kicsomagolás + `START.bat` dupla kattintás, minden mást a
  bootstrap elintézi.
- **Verifikációs / evidencia ZIP-ek** — a kiadások független
  ellenőrzésének dokumentumai és a tükör szinkron bizonyítékai.

Letöltés után ellenőrizd a SHA-256 összegeket (fenti táblázat). A vendor
identitás a `e8384e0…` pin (nem a 0.2.3-as csomagverzió).
