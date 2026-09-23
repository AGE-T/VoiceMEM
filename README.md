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
| `VoiceMemAgent_v0.10.4_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 9 KB |
| `VoiceMemAgent_v0.10.4_S8LatencyCorrelation.zip` | **Evidence package** for the `S8LatencyCorrelation.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 531 KB |
| `VoiceMemAgent_v0.10.5_MemoryPerformanceAudit.zip` | **Evidence package** for the `MemoryPerformanceAudit.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 48 KB |
| `VoiceMemAgent_v0.10.5_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 11 KB |
| `VoiceMemAgent_v0.10.5_TTSForensicFix.zip` | **Evidence package** for the `TTSForensicFix.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 28 KB |
| `VoiceMemAgent_v0.10.6_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 5 KB |
| `VoiceMemAgent_v0.10.6_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 26 KB |
| `VoiceMemAgent_v0.10.7_LlamaB11073Experimental.zip` | **Evidence package** for the `LlamaB11073Experimental.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 74 KB |
| `VoiceMemAgent_v0.10.7_LlamaB11073_Installer.zip` | **The experimental llama.cpp b11073 installer/configurator pack (rev 2)** — scripts + documentation ONLY (no binaries): downloads the pinned b11073 release at install time, SHA-256-gated; the launcher verifies the pinned runtime identity (24-file manifest + build id + live fingerprint) and never gates on `--version`. | 50 KB |
| `VoiceMemAgent_v0.10.7_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 5 KB |
| `VoiceMemAgent_v0.10.7_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 17 KB |
| `VoiceMem_forensic_v0104_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 38 KB |

## Verify your download

```bash
sha256sum VoiceMemAgent_v0.10.7.zip VoiceMemAgent_v0.10.4_MirrorSync.zip VoiceMemAgent_v0.10.4_S8LatencyCorrelation.zip VoiceMemAgent_v0.10.5_MemoryPerformanceAudit.zip VoiceMemAgent_v0.10.5_MirrorSync.zip VoiceMemAgent_v0.10.5_TTSForensicFix.zip VoiceMemAgent_v0.10.6_MirrorSync.zip VoiceMemAgent_v0.10.6_recovery.zip VoiceMemAgent_v0.10.7_LlamaB11073Experimental.zip VoiceMemAgent_v0.10.7_LlamaB11073_Installer.zip VoiceMemAgent_v0.10.7_MirrorSync.zip VoiceMemAgent_v0.10.7_recovery.zip VoiceMem_forensic_v0104_recovery.zip
```

Expected:

```text
c5a6c4e30056b0ee3a636cd7cdc6315155ab1334f04406ba57acba3a7b684dc3  VoiceMemAgent_v0.10.7.zip
dbb35358eb78a071f084a46088894cf2e0c2c2e6b17fc0918735c6e538f64787  VoiceMemAgent_v0.10.4_MirrorSync.zip
f9ace279cebe98abb8f438352ce724ad63f0fec7f8bbd22fb8308b055b422146  VoiceMemAgent_v0.10.4_S8LatencyCorrelation.zip
5ceb6bd176f9ed309b6f67a9112629a5b3282b4e13cd6965e0c9c1be74eb8e28  VoiceMemAgent_v0.10.5_MemoryPerformanceAudit.zip
a00c96ba5319babd643984eb1ef4ebd40ca216a7d2b17ab844fb257dc601394f  VoiceMemAgent_v0.10.5_MirrorSync.zip
b5ba47263d7388bb03d599260f21040c2a86d5ce4da863b1e393ad1c75787712  VoiceMemAgent_v0.10.5_TTSForensicFix.zip
8083e4753734d546d4fb28a0547c4304e5a25dd99bcaac3de039e19fe03a5ae4  VoiceMemAgent_v0.10.6_MirrorSync.zip
c6d9080709647b1b85d5cb13af2b2bdee60750547745a237f3ffd9df7a3e8b4f  VoiceMemAgent_v0.10.6_recovery.zip
d28f7f8c275828318a7bd12a581bc2e7e8f27aac61a2f3c2d5cd9fe8e5e6e5b5  VoiceMemAgent_v0.10.7_LlamaB11073Experimental.zip
c844ad7168f29dc772911ce11a16040b532fb13de0ff8672c273f7afad52529a  VoiceMemAgent_v0.10.7_LlamaB11073_Installer.zip
e0f9f7c911e476db55265e71ea1a69819b426e2a289fdf93c99939f74ba051c7  VoiceMemAgent_v0.10.7_MirrorSync.zip
8b601582901bfd4e9f4a37253346140d025f7318c487038b2fa9c0b1f4bfa70e  VoiceMemAgent_v0.10.7_recovery.zip
514ecdd8010e78efcc6c5cd1dd31b062c41b6a8d85019882fa597a442274842c  VoiceMem_forensic_v0104_recovery.zip
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
