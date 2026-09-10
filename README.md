# VoiceMEM — VoiceMemAgent Releases

Distribution mirror for **VoiceMemAgent**, the one-click, offline-capable
voice memory agent. This repository hosts the versioned release packages
and their verification artifacts — the canonical build source lives in
the agent workspace.

**Latest stable release: `v0.5.0`** (controlled VoiceMem foundation —
independently re-verified).

---

## Packages

| File | What it is | Size |
| --- | --- | --- |
| `VoiceMemAgent_v0.5.0.zip` | **The product.** Full agent source, machine-independent (no runtime state inside: no `.venv`, models, or memory). Install = unzip + `START.bat`. | 1.9 MB |
| `VoiceMemAgent_v0.5.0_Verification.zip` | **The verification record** (not a product release): independent re-verification report of the v0.5.0 foundation — A–Q phases + ten engineering decisions, 5 evidence files (pin verification, clean-install manifest, A/B retrieval baseline, trait assessment, test gate 921 OK / 24 skipped), `identity.json` and `RECOVERY.md`. | 29 KB |
| `VoiceMemAgent_v0.5.0_MirrorCheck.zip` | **The mirror check**: proof that this GitHub mirror is byte-identical to the locally produced artifacts (SHA-256 tables from both sides, git clone evidence, ZIP content listings). Verdict: `PASS`. | 8 KB |

## Verify your download

```bash
sha256sum VoiceMemAgent_v0.5.0.zip VoiceMemAgent_v0.5.0_Verification.zip VoiceMemAgent_v0.5.0_MirrorCheck.zip
```

Expected:

```text
45c1feba29dc61492aa148b167c31e60509b7eeeede003f33348f430254ceb40  VoiceMemAgent_v0.5.0.zip
aa880796031a5ada388a75cdae26befc723d6e37988589d4a0bf6b91331db631  VoiceMemAgent_v0.5.0_Verification.zip
89bc2933843301e1317708e2c4b98083ef3129d149611c945a9ee9dd6a44bc14  VoiceMemAgent_v0.5.0_MirrorCheck.zip
```

(`.sha256` sidecar files are included next to each ZIP.)

## Install (Windows)

1. Unzip `VoiceMemAgent_v0.5.0.zip` into any folder (e.g. `F:\Voicemem\VoiceMemAgent`).
2. Double-click **`START.bat`** — that is the single user entry point.
3. Wait for the bootstrap: Python check, `.venv`, dependencies, HF tooling,
   model download (`MODELS.lock.json`), configuration, smoke tests.
4. If everything is green, the agent starts — you can talk to it.

Advanced modes: `START.bat mock` · `check` · `benchmark` · `repair` · `build`.
At a single end-to-end failure, re-run `START.bat repair`.

## The verification story (v0.5.0)

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
  documented ledger (VM-LOCAL-001..006, VM-LOCAL-EN + 4 port patches) —
  permanently local, never upstreamed.
- Full details: open `VoiceMemAgent_v0.5.0_Verification.zip` →
  `docs/VERIFICATION_REPORT_v0.5.0.md`.

## Mirrors

- This repository is the **external distribution mirror**, synced with
  write access granted by the owner (token-scoped, revocable).
- The download page inside the build environment links here directly.

---

## Magyar összefoglaló

Ez a repó a **VoiceMemAgent** kiadásainak külső tükre. A legfrissebb
stabil verzió a **v0.5.0**. Mit találhat itt:

- **`VoiceMemAgent_v0.5.0.zip`** — maga a termék. Gépfüggetlen forrás,
  telepítés: kicsomagolás + `START.bat` dupla kattintás, minden mást a
  bootstrap elintéz.
- **`VoiceMemAgent_v0.5.0_Verification.zip`** — a v0.5.0 alap
  független ellenőrzésének teljes dokumentuma (A–Q riport, 10 mérnöki
  döntés, 5 bizonyítékfájl, helyreállítási útmutató).
- **`VoiceMemAgent_v0.5.0_MirrorCheck.zip`** — annak bizonyítéka, hogy
  ez a tükör bájtról bájtra azonos az eredetileg legyártott csomagokkal.

Letöltés után ellenőrizd a SHA-256 összegeket (fenti táblázat). A vendor
identitás a `e8384e0…` pin (nem a 0.2.3-as csomagverzió). A teljes
verifikációs történet a Verifikációs ZIP `docs/` mappájában olvasható.
