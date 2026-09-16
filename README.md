# VoiceMEM — VoiceMemAgent Releases

Distribution mirror for **VoiceMemAgent**, the one-click, offline-capable
voice memory agent. This repository hosts the versioned release packages
and their verification artifacts — the canonical build source lives in
the agent workspace.

**Latest stable release: `v0.10.2`**

---

## Packages

| File | What it is | Size |
| --- | --- | --- |
| `VoiceMemAgent_v0.10.2.zip` | **The product.** Full agent source, machine-independent (no runtime state inside: no `.venv`, models, or memory). Install = unzip + `START.bat`. | 2.6 MB |
| `VoiceMemAgent_v0.10.0_MirrorSync.zip` | **The mirror-sync evidence** for this release: PAT permission check (redacted), sync dry-run, push log, and a fresh anonymous-clone verification that the GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides). | 9 KB |
| `VoiceMem_asrregression_v0101_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 33 KB |
| `VoiceMem_llmpriority_v0102_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 36 KB |
| `VoiceMem_memoryfixes_v080_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 13 KB |
| `VoiceMem_memorysemantics_v090_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 61 KB |
| `VoiceMem_runtimeopt_v092_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 45 KB |
| `VoiceMem_stabilization_v081_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 52 KB |
| `VoiceMem_temporal_v010_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 36 KB |
| `VoiceMem_vadfix_v091_recovery.zip` | **The recovery package** for this release: `RECOVERY.md` + `identity.json` + evidence (the GREEN gate record, the code diff, changed files, test output, the release ZIP hash) — the versioned, independently restorable rollback anchor (what changed, how to verify it, how to roll back). | 37 KB |

## Verify your download

```bash
sha256sum VoiceMemAgent_v0.10.2.zip VoiceMemAgent_v0.10.0_MirrorSync.zip VoiceMem_asrregression_v0101_recovery.zip VoiceMem_llmpriority_v0102_recovery.zip VoiceMem_memoryfixes_v080_recovery.zip VoiceMem_memorysemantics_v090_recovery.zip VoiceMem_runtimeopt_v092_recovery.zip VoiceMem_stabilization_v081_recovery.zip VoiceMem_temporal_v010_recovery.zip VoiceMem_vadfix_v091_recovery.zip
```

Expected:

```text
f56e5155043b9900b2a45799f4fa1663ed8475584f9dba1f1bbd80b13c525a54  VoiceMemAgent_v0.10.2.zip
5ca6eec78f3079515d62d4cdb12418ab5a4030002cda8d800e980322de83c20b  VoiceMemAgent_v0.10.0_MirrorSync.zip
59f0df85711dd0eb6b6d4a2848878c47c421a204a504d31963d2b74c0571bb9c  VoiceMem_asrregression_v0101_recovery.zip
06b065abe83e42f01f162baa1711fa22b2fddbcb63ed79851ccae567630220bd  VoiceMem_llmpriority_v0102_recovery.zip
cc54ffbb29af3b4cad14bf03da9f5ad288389958bc8001d21cb204fec1dcb28e  VoiceMem_memoryfixes_v080_recovery.zip
25d35796d6abbce8e2325f2401218d8fe65ede79f71e2c255cfb1e37c86a5a5c  VoiceMem_memorysemantics_v090_recovery.zip
f8052ecb67451be38dfddfc428ff3ec6f18976a6f8865f0051182c18a168487e  VoiceMem_runtimeopt_v092_recovery.zip
b4218a4c6a1f862791371a646a3f6281fb02285320b0fa625341cb572d5fb1fc  VoiceMem_stabilization_v081_recovery.zip
e5ac3e542790cdfc6f1649502910513bad2e886cab0482cc887845c9d2b29cee  VoiceMem_temporal_v010_recovery.zip
1336bd647415d357c85bb6dde1801f5ef936217575a23d74e18c71533ecac5f4  VoiceMem_vadfix_v091_recovery.zip
```

(`.sha256` sidecar files are included next to each ZIP.)

## Install (Windows)

1. Unzip `VoiceMemAgent_v0.10.2.zip` into any folder (e.g. `F:\Voicemem\VoiceMemAgent`).
2. Double-click **`START.bat`** — that is the single user entry point.
3. Wait for the bootstrap: Python check, `.venv`, dependencies, HF tooling,
   model download (`MODELS.lock.json`), configuration, smoke tests.
4. If everything is green, the agent starts — you can talk to it.

Advanced modes: `START.bat mock` · `check` · `benchmark` · `repair` · `build`.
At a single end-to-end failure, re-run `START.bat repair`.

## The verification story (v0.10.2)

- **Vendor identity:** `vendor/voicemem` pinned at
  `e8384e087bd2f44eb05fc7ae1a3c525ea8244179` (= upstream tag `v0.0.1`;
  the `0.2.3` package version is **not** the identity anchor).
- **Pin verification:** 3-mode script (source / runtime / embedding), with
  an instrumented fake OpenAI client proving the OpenAI client is never
  constructed — all green.
- **Clean install:** fresh ZIP → new venv → editable install → import
  identity → `pin_verified: true` in the manifest.
- **Test gate (v0.10.2 release gate):** 1386 passed / 0 regressions /
  28 skipped (6 pinned sandbox env-gap — documented in the pinned
  baseline, green on the target machine); deep validation PASS
  (embedding, memory, memory-semantics, release, runtime-deps) on
  fingerprint
  768cc7b5292de2b4c7a601092232b21d6534ec2e6b43d686426b86f90416c1ad
  — the stamped record ships inside the ZIP (`releases/gate_record.json`).
- **Local vendor patches:** 24 files vs the pinned upstream
  (21 modified + 3 added: right-brain stance, temporal ranking,
  background-LLM cancellation); patch ledger `VOICEMEM_PIN.json`
  (VM-LOCAL-001…015 + EN) + the v0.10.x CHANGELOG entries —
  permanently local, never upstreamed.
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
stabil verzió a **v0.10.2**. Mit találhat itt:

- **`VoiceMemAgent_v0.10.2.zip`** — maga a termék. Gépfüggetlen forrás,
  telepítés: kicsomagolás + `START.bat` dupla kattintás, minden mást a
  bootstrap elintézi.
- **Verifikációs / evidencia ZIP-ek** — a kiadások független
  ellenőrzésének dokumentumai és a tükör szinkron bizonyítékai.

Letöltés után ellenőrizd a SHA-256 összegeket (fenti táblázat). A vendor
identitás a `e8384e0…` pin (nem a 0.2.3-as csomagverzió).
