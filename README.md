# VoiceMEM — VoiceMemAgent Releases

Distribution mirror for **VoiceMemAgent**, the one-click, offline-capable
voice memory agent. This repository hosts the versioned release packages
and their verification artifacts — the canonical build source lives in
the agent workspace.

**Latest stable release: `v0.6.0`**

---

## Packages

| File | What it is | Size |
| --- | --- | --- |
| `VoiceMemAgent_v0.6.0.zip` | **The product.** Full agent source, machine-independent (no runtime state inside: no `.venv`, models, or memory). Install = unzip + `START.bat`. | 2.1 MB |
| `VoiceMemAgent_v0.5.0_GitHubSync.zip` | **First authenticated sync evidence**: token permission check, push output, fresh-clone verification of the mirror state after the sync. | 6 KB |
| `VoiceMemAgent_v0.5.0_MirrorCheck.zip` | **The mirror check**: proof that this GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides, git clone evidence). Verdict: `PASS`. | 8 KB |
| `VoiceMemAgent_v0.5.0_SyncSetup.zip` | **Standing sync policy + infrastructure**: the sync script, the procedure document, and the setup-run evidence (every future update goes to the GitHub mirror). | 12 KB |
| `VoiceMemAgent_v0.5.0_Verification.zip` | **The verification record** (not a product release): independent verification report — phases + engineering decisions, evidence files (pin verification, clean-install manifest, retrieval baseline, trait assessment, test gate), `identity.json` and `RECOVERY.md`. | 29 KB |
| `VoiceMemAgent_v0.5.1_ForensicTrace.zip` | **TASK 1 ASR→LLM forensic trace**: the proven break point (CLI bridge adapter tables referenced non-existent vendor API — every CLI turn ran memoryless), the smallest-correct-fix patch, the post-fix trace matrix (S1–S7b green; thinking-only fails loud), and the full test-gate record. | 41 KB |
| `VoiceMemAgent_v0.5.1_Task2Gate.zip` | **TASK 2 rich-retrieval DESIGN GATE** (measurement-only): the authoritative baseline of both retrieval paths, runtime call graphs, the API design comparison (recommended: app-level search_rich), the rendering/temporal/speaker audits and the 15-test plan. | 35 KB |
| `VoiceMemAgent_v0.5.2_MemorySafetyGate.zip` | **TASK 1.5 memory-safety gate** (v0.5.2): the Claude-audit reconciliation, the three safety patches (right-brain delete gate; non-destructive UPDATE with explicit supersession; left delete graph cascade), the P0 before/after proof, the 15-test behavioural battery (real vendor + real store) and the 954/0/26 release gate. | 139 KB |
| `VoiceMemAgent_v0.6.0_ASRGate.zip` | **Evidence package** for the `ASRGate.zip` work unit (GOLDEN RULE deliverable: RECOVERY.md + identity.json + evidence). | 566 KB |

## Verify your download

```bash
sha256sum VoiceMemAgent_v0.6.0.zip VoiceMemAgent_v0.5.0_GitHubSync.zip VoiceMemAgent_v0.5.0_MirrorCheck.zip VoiceMemAgent_v0.5.0_SyncSetup.zip VoiceMemAgent_v0.5.0_Verification.zip VoiceMemAgent_v0.5.1_ForensicTrace.zip VoiceMemAgent_v0.5.1_Task2Gate.zip VoiceMemAgent_v0.5.2_MemorySafetyGate.zip VoiceMemAgent_v0.6.0_ASRGate.zip
```

Expected:

```text
9c55d9e194f8e15ddf2ef01481e12ca5993c769126b1a86cbcf731b2a0ce46c4  VoiceMemAgent_v0.6.0.zip
1b4596351d8ce45e0f750e1c21c0693366fdb0ab9ab3c0a627335e06bb353847  VoiceMemAgent_v0.5.0_GitHubSync.zip
89bc2933843301e1317708e2c4b98083ef3129d149611c945a9ee9dd6a44bc14  VoiceMemAgent_v0.5.0_MirrorCheck.zip
bce2979501864e54281e9615d563d5a0f3148c690b53e105060f58ff251be559  VoiceMemAgent_v0.5.0_SyncSetup.zip
aa880796031a5ada388a75cdae26befc723d6e37988589d4a0bf6b91331db631  VoiceMemAgent_v0.5.0_Verification.zip
68b3c0dbf751bc61ac38458e7efde739b6de7e978e3d6f990382e6ce704caedc  VoiceMemAgent_v0.5.1_ForensicTrace.zip
b24ea62f65ce77bdaf94fb843e10e8c1b88c7e1fcd37c064806e948d538abbbb  VoiceMemAgent_v0.5.1_Task2Gate.zip
4d9d3dc7e384b45a22ab5edf3bd6476748d5e0c0d0364ef279800121f098c8cf  VoiceMemAgent_v0.5.2_MemorySafetyGate.zip
54274f8dec40e068e6aeba3b29aa3187241c0c7f2c54fb7d15872a5250d9a2f9  VoiceMemAgent_v0.6.0_ASRGate.zip
```

(`.sha256` sidecar files are included next to each ZIP.)

## Install (Windows)

1. Unzip `VoiceMemAgent_v0.6.0.zip` into any folder (e.g. `F:\Voicemem\VoiceMemAgent`).
2. Double-click **`START.bat`** — that is the single user entry point.
3. Wait for the bootstrap: Python check, `.venv`, dependencies, HF tooling,
   model download (`MODELS.lock.json`), configuration, smoke tests.
4. If everything is green, the agent starts — you can talk to it.

Advanced modes: `START.bat mock` · `check` · `benchmark` · `repair` · `build`.
At a single end-to-end failure, re-run `START.bat repair`.

## The verification story (v0.6.0)

- **Vendor identity:** `vendor/voicemem` pinned at
  `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` (= upstream tag `v0.0.1`;
  the `0.2.3` package version is **not** the identity anchor).
- **Pin verification:** 3-mode script (source / runtime / embedding), with
  an instrumented fake OpenAI client proving the OpenAI client is never
  constructed — all green.
- **Clean install:** fresh ZIP → new venv → editable install → import
  identity → `pin_verified: true` in the manifest.
- **Test gate:** 921 passed / 24 skipped, reproduced in a downgraded
  (no vendor PYTHONPATH) main-sandbox venv.
- **Local vendor patches:** exactly 10 files vs upstream, all mapped in a
  documented ledger — permanently local, never upstreamed.
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
stabil verzió a **v0.6.0**. Mit találhat itt:

- **`VoiceMemAgent_v0.6.0.zip`** — maga a termék. Gépfüggetlen forrás,
  telepítés: kicsomagolás + `START.bat` dupla kattintás, minden mást a
  bootstrap elintézi.
- **Verifikációs / evidencia ZIP-ek** — a kiadások független
  ellenőrzésének dokumentumai és a tükör szinkron bizonyítékai.

Letöltés után ellenőrizd a SHA-256 összegeket (fenti táblázat). A vendor
identitás a `e8384e0…` pin (nem a 0.2.3-as csomagverzió).
