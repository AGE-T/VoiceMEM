# b11073 DELTA RUNTIME PACK — VoiceMem v0.10.7 (experimental, NOT production)

**What this is:** the minimal, target-machine delta runtime for the
already-validated experimental llama.cpp **b11073** integration —
`llama-server.exe` plus **only** the application-local DLLs its dependency
closure requires, and the **validated Windows launcher**. It contains **no
CUDA, no NVIDIA driver, no cudart, no model files, no VoiceMem application
files, no Python** — the target machine already provides all of those.

Companion evidence: `DEPENDENCY_INSPECTION.md` (how every DLL was classified
and what is NOT packaged, with the measurement method), `SHA256SUMS.txt`
(integrity manifest). Full background: the audit report at
`audit/VoiceMEM_llama_b11073_experimental/REPORT.md` in the VoiceMEM mirror.

---

## Gyorsindítás (unpack and go, magyarul)

1. **Csomagold ki** ezt a ZIP-et a **VoiceMemAgent_v0.10.7 gyökérbe** (oda,
   ahol a `START.bat` van). Két mappát hoz létre/egészít ki:
   `bin\llama-server-b11073\` (a futtatókörnyezet) és
   `experimental\llama_b11073\` (az indító + ez a dokumentáció).
   A gyártási `bin\llama-server.exe` (b10717) **érintetlen marad**.
2. **Duplán kattints** ide: `experimental\llama_b11073\RUN_EXPERIMENTAL.bat`
   — ez lefuttatja a csak-olvasás korfeltétel-probát (`check_environment.ps1`),
   láthatóvá teszi a célgépen már meglévő CUDA DLL-eket
   (`cublas64_13.dll`, jellemzően a gyártási `bin\`-ből), majd átadja a
   irányítást a **validált** indítónak (`start_llama_server_experimental.ps1`,
   bájtról bájtra az audited példány).
3. A szerver **127.0.0.1:8081**-en indul (ngl 99 / ctx 16000 / parallel 1 /
   threads 12 / reasoning off / KV q8_0 — az operátor által tesztelt profil).
   Az ablak nyitva tartja a folyamatot.
4. **Csatlakoztasd az agentet** (meglévő env-felülírás, semmi kódváltozás):
   ```powershell
   $env:LLAMA_SERVER_HOST = "127.0.0.1"
   $env:LLAMA_SERVER_PORT = "8081"
   .\scripts\start_agent.ps1
   ```

## Rollback (teljes visszavonás)

1. Állítsd le az experimentális szervert (Ctrl+C / ablak bezárása).
2. Töröld a két env-változót az agent sessionből
   (`Remove-Item Env:LLAMA_SERVER_HOST, Env:LLAMA_SERVER_PORT`, vagy egyszerűen
   új terminál) — az agent újra a `127.0.0.1:8080`-as gyártási szervert
   célozza, bájtról bájtra változatlan viselkedéssel.
3. Opcionálisan töröld: `bin\llama-server-b11073\` + a kicsomagolt pack-fájlok.
   Semmi mást nem kell visszavonni: gyártási fájl, konfiguráció, verziószám
   nem változott.

## Prerequisites (category A — NOT in the pack, by design)

| Prerequisite | Why it is not packaged |
| --- | --- |
| `cublas64_13.dll` | **The only CUDA DLL the runtime needs.** The target already provides it (production `bin\` from the b10717 cudart install, and/or a system CUDA 13.x). The wrapper only adds the existing location to the child PATH. |
| NVIDIA driver (any recent) | System component; the production CUDA server already runs on this machine. |
| `msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll` | VC++ 2015–2022 redistributable — system-level; the production b10717 server already proves they are present. |
| UCRT (`api-ms-win-crt-*`, `ucrtbase.dll`) | Built into Windows 10/11. |
| The LLM GGUF | Never packaged (VoiceMem convention — operator-placed, resolved via `config\llm_model.json` / `LLAMA_MODEL_PATH` / default profile path). |

`check_environment.ps1` probes every one of these on the target and prints
where each resolves from; it also verifies every packaged file against
`SHA256SUMS.txt`.

## What is in the pack (category B — measured, not guessed)

`bin\llama-server-b11073\`: `llama-server.exe` + its static import closure
(`llama-server-impl.dll`, `llama-common.dll`, `llama.dll`, `mtmd.dll`,
`ggml.dll`, `ggml-base.dll`, `libomp.dll`) + the **dynamically loaded ggml
backends** (`ggml-cuda.dll` — without it the server silently runs CPU-only;
the 14 `ggml-cpu-*.dll` microarchitecture variants) + `LICENSE-LLVM-OpenMP`
(licenses `libomp.dll`). The full classification with import edges:
`DEPENDENCY_INSPECTION.md`.

## Known limitations (honest scope)

* The delta pack was **statically verified in the sandbox** (import closure,
  integrity, backend-discovery evidence); the **Windows execution itself was
  not verifiable here** (Linux sandbox, no GPU). Target-side verification:
  `check_environment.ps1` (prerequisites + hashes) and the launcher's own
  health-wait (`/health` 200 + the b11073 version banner in the log).
* b11073 ships no Windows CUDA 13.3 asset; this pack is the **CUDA 13.4**
  build. Its single CUDA dependency is major-version-named
  (`cublas64_13.dll`), satisfied by the target's CUDA 13.3 set through
  NVIDIA's minor-version compatibility (same major). If a newer
  `cublas64_13.dll` is ever needed, drop it next to
  `bin\llama-server-b11073\llama-server.exe` (app dir wins the loader order).
* Not a production release: VoiceMem stays at v0.10.7; the experimental
  runtime remains fully reversible (see Rollback).

— Generated 2026-09-22, per the delta-pack order. Source asset integrity and
the full classification evidence: `DEPENDENCY_INSPECTION.md`.
