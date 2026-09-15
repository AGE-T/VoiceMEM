# VoiceMEM — VoiceMemAgent Releases

Distribution mirror for **VoiceMemAgent**, the one-click, offline-capable
voice memory agent. This repository hosts the versioned release packages
and their verification artifacts — the canonical build source lives in
the agent workspace.

**Latest stable release: `v0.9.2`**

---

## Packages

| File | What it is | Size |
| --- | --- | --- |
| `VoiceMemAgent_v0.9.2.zip` | **The product.** Full agent source, machine-independent (no runtime state inside: no `.venv`, models, or memory). Install = unzip + `START.bat`. | 2.5 MB |
| `VoiceMemAgent_v0.5.0_GitHubSync.zip` | **First authenticated sync evidence**: token permission check, push output, fresh-clone verification of the mirror state after the sync. | 6 KB |
| `VoiceMemAgent_v0.5.0_MirrorCheck.zip` | **The mirror check**: proof that this GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides, git clone evidence). Verdict: `PASS`. | 8 KB |
| `VoiceMemAgent_v0.5.0_SyncSetup.zip` | **Standing sync policy + infrastructure**: the sync script, the procedure document, and the setup-run evidence (every future update goes to the GitHub mirror). | 12 KB |
| `VoiceMemAgent_v0.5.0_Verification.zip` | **The verification record** (not a product release): independent verification report — phases + engineering decisions, evidence files (pin verification, clean-install manifest, retrieval baseline, trait assessment, test gate), `identity.json` and `RECOVERY.md`. | 29 KB |
| `VoiceMemAgent_v0.5.1_ForensicTrace.zip` | **TASK 1 ASR→LLM forensic trace**: the proven break point (CLI bridge adapter tables referenced non-existent vendor API — every CLI turn ran memoryless), the smallest-correct-fix patch, the post-fix trace matrix (S1–S7b green; thinking-only fails loud), and the full test-gate record. | 41 KB |
| `VoiceMemAgent_v0.5.1_Task2Gate.zip` | **TASK 2 rich-retrieval DESIGN GATE** (measurement-only): the authoritative baseline of both retrieval paths, runtime call graphs, the API design comparison (recommended: app-level search_rich), the rendering/temporal/speaker audits and the 15-test plan. | 35 KB |
| `VoiceMemAgent_v0.5.2_MemorySafetyGate.zip` | **TASK 1.5 memory-safety gate** (v0.5.2): the Claude-audit reconciliation, the three safety patches (right-brain delete gate; non-destructive UPDATE with explicit supersession; left delete graph cascade), the P0 before/after proof, the 15-test behavioural battery (real vendor + real store) and the 954/0/26 release gate. | 139 KB |
| `VoiceMemAgent_v0.6.0_ASRGate.zip` | **Evidence package** for the `ASRGate.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 566 KB |
| `VoiceMem_llmconfig_v072_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 34 KB |
| `VoiceMem_memoryfixes_v080_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 13 KB |
| `VoiceMem_memorysemantics_v090_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 61 KB |
| `VoiceMem_runtimeopt_v092_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 45 KB |
| `VoiceMem_stabilization_v081_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 52 KB |
| `VoiceMem_vadfix_v091_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 37 KB |

## Verify your download

```bash
sha256sum VoiceMemAgent_v0.9.2.zip VoiceMemAgent_v0.5.0_GitHubSync.zip VoiceMemAgent_v0.5.0_MirrorCheck.zip VoiceMemAgent_v0.5.0_SyncSetup.zip VoiceMemAgent_v0.5.0_Verification.zip VoiceMemAgent_v0.5.1_ForensicTrace.zip VoiceMemAgent_v0.5.1_Task2Gate.zip VoiceMemAgent_v0.5.2_MemorySafetyGate.zip VoiceMemAgent_v0.6.0_ASRGate.zip VoiceMem_llmconfig_v072_recovery.zip VoiceMem_memoryfixes_v080_recovery.zip VoiceMem_memorysemantics_v090_recovery.zip VoiceMem_runtimeopt_v092_recovery.zip VoiceMem_stabilization_v081_recovery.zip VoiceMem_vadfix_v091_recovery.zip
```

Expected:

```text
3a9caaf44fbfae6fce8449e781e42a6007434d9c062c37270c5d8b566dacbf05  VoiceMemAgent_v0.9.2.zip
1b4596351d8ce45e0f750e1c21c0693366fdb0ab9ab3c0a627335e06bb353847  VoiceMemAgent_v0.5.0_GitHubSync.zip
89bc2933843301e1317708e2c4b98083ef3129d149611c945a9ee9dd6a44bc14  VoiceMemAgent_v0.5.0_MirrorCheck.zip
bce2979501864e54281e9615d563d5a0f3148c690b53e105060f58ff251be559  VoiceMemAgent_v0.5.0_SyncSetup.zip
aa880796031a5ada388a75cdae26befc723d6e37988589d4a0bf6b91331db631  VoiceMemAgent_v0.5.0_Verification.zip
68b3c0dbf751bc61ac38458e7efde739b6de7e978e3d6f990382e6ce704caedc  VoiceMemAgent_v0.5.1_ForensicTrace.zip
b24ea62f65ce77bdaf94fb843e10e8c1b88c7e1fcd37c064806e948d538abbbb  VoiceMemAgent_v0.5.1_Task2Gate.zip
4d9d3dc7e384b45a22ab5edf3bd6476748d5e0c0d0364ef279800121f098c8cf  VoiceMemAgent_v0.5.2_MemorySafetyGate.zip
54274f8dec40e068e6aeba3b29aa3187241c0c7f2c54fb7d15872a5250d9a2f9  VoiceMemAgent_v0.6.0_ASRGate.zip
f11b7aaf013be3db5f1c379ce309c3b297152e4e093e18652a98672fb85e2687  VoiceMem_llmconfig_v072_recovery.zip
cc54ffbb29af3b4cad14bf03da9f5ad288389958bc8001d21cb204fec1dcb28e  VoiceMem_memoryfixes_v080_recovery.zip
25d35796d6abbce8e2325f2401218d8fe65ede79f71e2c255cfb1e37c86a5a5c  VoiceMem_memorysemantics_v090_recovery.zip
f8052ecb67451be38dfddfc428ff3ec6f18976a6f8865f0051182c18a168487e  VoiceMem_runtimeopt_v092_recovery.zip
b4218a4c6a1f862791371a646a3f6281fb02285320b0fa625341cb572d5fb1fc  VoiceMem_stabilization_v081_recovery.zip
1336bd647415d357c85bb6dde1801f5ef936217575a23d74e18c71533ecac5f4  VoiceMem_vadfix_v091_recovery.zip
```

(`.sha256` sidecar files are included next to each ZIP.)

## Install (Windows)

1. Unzip `VoiceMemAgent_v0.9.2.zip` into any folder (e.g. `F:\Voicemem\VoiceMemAgent`).
2. Double-click **`START.bat`** — that is the single user entry point.
3. Wait for the bootstrap: Python check, `.venv`, dependencies, HF tooling,
   model download (`MODELS.lock.json`), configuration, smoke tests.
4. If everything is green, the agent starts — you can talk to it.

Advanced modes: `START.bat mock` · `check` · `benchmark` · `repair` · `build`.
At a single end-to-end failure, re-run `START.bat repair`.

## The verification story (v0.9.2)

- **Vendor identity:** `vendor/voicemem` pinned at
  `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` (= upstream tag `v0.0.1`;
  the `0.2.3` package version is **not** the identity anchor).
- **Pin verification:** 3-mode script (source / runtime / embedding), with
  an instrumented fake OpenAI client proving the OpenAI client is never
  constructed — all green.
- **Clean install:** fresh ZIP → new venv → editable install → import
  identity → `pin_verified: true` in the manifest.
- **Test gate:** 1253 passed / 0 regressions / 13 skipped
  (pinned env-gap: GPU-bound ASR/VAD/TTS legs; semantics- and
  memory-gate suites included; deep validation PASS on fingerprint
  a74af6c54a85f288d8d9b3ea7dc556d4).
- **Local vendor patches:** exactly 21 files vs upstream (20 modified
  + 1 added, right-brain stance), all mapped in a documented
  ledger — permanently local, never upstreamed.
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
stabil verzió a **v0.9.2**. Mit találhat itt:

- **`VoiceMemAgent_v0.9.2.zip`** — maga a termék. Gépfüggetlen forrás,
  telepítés: kicsomagolás + `START.bat` dupla kattintás, minden mást a
  bootstrap elintézi.
- **Verifikációs / evidencia ZIP-ek** — a kiadások független
  ellenőrzésének dokumentumai és a tükör szinkron bizonyítékai.

Letöltés után ellenőrizd a SHA-256 összegeket (fenti táblázat). A vendor
identitás a `e8384e0…` pin (nem a 0.2.3-as csomagverzió).
