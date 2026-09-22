# DEPENDENCY INSPECTION — b11073 delta runtime pack

**Date:** 2026-09-22 · **Subject:** `llama-server.exe` from
`llama-b11073-bin-win-cuda-13.4-x64.zip` (llama.cpp release **b11073**,
prerelease, published 2026-09-21T15:30:06Z) · **Method:** PE import-table
analysis (`objdump -p`, pei-x86-64) with recursive closure from
`llama-server.exe`, plus in-binary string evidence for runtime-discovered
backends · **Purpose:** decide exactly what a target-machine DELTA pack must
contain, with every DLL classified before packaging (per the delta-pack
order: classify A = target-provided / B = application-local, then package
only B).

## 0. Source integrity [PROVEN]

| Item | Value |
| --- | --- |
| Asset | `llama-b11073-bin-win-cuda-13.4-x64.zip` |
| Size | 150,093,926 bytes |
| SHA-256 (downloaded) | `85c1b874180faec412ccbba16ee0833062c28e2bf1390b12ed30f7dc6d7d79c4` |
| SHA-256 (GitHub API digest) | `sha256:85c1b874…` — **MATCH** |
| Build identity in-binary | commit string `1aa2954bd` present in `llama-server-impl.dll` (matches the b11073 release commit; the validated audit ran the same release's ubuntu build: `llama-server 0.4.1-dev (build 11073, commit 1aa2954bd)`) |
| Machine | `pei-x86-64` |

The CUDA-runtime companion asset (`cudart-llama-bin-win-cuda-13.4-x64.zip`)
was **intentionally NOT downloaded and NOT packaged** — the delta-pack order
forbids packaging CUDA components the target already provides.

## 1. Static import closure of llama-server.exe

b11073 uses a split layout: `llama-server.exe` is a thin wrapper importing
`llama-server-impl.dll`; the closure (recursive) is:

```
llama-server.exe
└─ llama-server-impl.dll
   ├─ llama-common.dll ── llama.dll ── ggml.dll ── ggml-base.dll ── libomp.dll
   ├─ mtmd.dll          (multimodal support compiled into the server)
   ├─ ggml.dll, ggml-base.dll
   └─ (system: KERNEL32, WS2_32, CRYPT32, MSVCP140, VCRUNTIME140[,_1],
      api-ms-win-crt-* ×11)
```

**Every application-local DLL in the closure is packaged.** Nothing in the
closure is a CUDA DLL.

## 2. DLL classification (the required A/B decision)

### Category B — application-local, PACKAGED (24 files)

| File | Size | Included because |
| --- | ---: | --- |
| `llama-server.exe` | 86,016 | the executable itself |
| `llama-server-impl.dll` | 8,904,704 | static import of the exe (server core) |
| `llama-common.dll` | 7,821,824 | static import (common runtime) |
| `llama.dll` | 3,167,744 | static import (llama API) |
| `mtmd.dll` | 1,770,496 | static import of `llama-server-impl.dll` |
| `ggml.dll` | 79,872 | static import (backend registry + loader) |
| `ggml-base.dll` | 796,672 | static import (ggml core) |
| `libomp.dll` | 768,000 | static import of `ggml-base.dll` (LLVM OpenMP) |
| `ggml-cuda.dll` | 145,057,280 | **dynamic backend** (see §3) — the GPU path; without it the server silently runs CPU-only and `-ngl 99` loses its meaning |
| `ggml-cpu-*.dll` × 14 | ≈ 18,500,000 total | **dynamic backends** (see §3) — CPU dispatch variants; `ggml.dll` selects at runtime by CPU microarchitecture |
| `LICENSE-LLVM-OpenMP` | 1,562 | license notice for `libomp.dll` (Apache-2.0 WITH LLVM-exception) — redistribution compliance |

CPU variant set (14 files): `alderlake, cannonlake, cascadelake, cooperlake,
haswell, icelake, ivybridge, piledriver, sandybridge, sapphirerapids,
skylakex, sse42, x64, zen4` — the loader picks the best match, `x64` is the
portable baseline. (~18.5 MB total.)

### Category A-SYSTEM — target-provided, NOT packaged (documented prerequisite)

`KERNEL32.dll`, `WS2_32.dll`, `ADVAPI32.dll` (cpu variants), `SHELL32.dll`,
`CRYPT32.dll`, `PSAPI.DLL` (libomp) — Windows OS.
`api-ms-win-crt-*.dll` × 11 + `ucrtbase.dll` — UCRT, built into Windows 10/11.
`MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll` — VC++ 2015–2022
redistributable (system-level; the production b10717 `llama-server.exe`
depends on the same set and already runs on the target, proving presence).

### Category A-NVIDIA — target-provided CUDA, NOT packaged

**Exactly one CUDA DLL appears in the entire dependency tree:**

| DLL | Imported by | Note |
| --- | --- | --- |
| `cublas64_13.dll` | `ggml-cuda.dll` (static import) | the only dynamic CUDA dependency; cuBLAS is not statically linkable, everything else (cudart) is statically linked into `ggml-cuda.dll`. No `nvrtc`, no `cublasLt`, no `cudart64_13.dll` import — verified against the import tables AND in-binary strings. |

Resolution on the target: `cublas64_13.dll` is major-version-named (CUDA
13.x), and the production install already placed the CUDA 13.3 set flat into
`bin\` (`install_m1.ps1` extracts `cudart-llama-bin-win-cuda-13.3-x64.zip`
there; its DLL-set validation checks `ggml-cuda*/cudart*/cublas*`). The b11073
build (13.4) against a 13.3 runtime is NVIDIA's documented same-major
minor-version compatibility; the cuBLAS calls used
(`cublasCreate_v2`, `cublasGemmEx`, `cublasGemmBatchedEx`,
`cublasGemmStridedBatchedEx` — in-binary evidence) are stable across 13.x.
`RUN_EXPERIMENTAL.bat` therefore prepends `<root>\bin\` to the child PATH so
the **existing** DLL is found without packaging it; a system-wide CUDA 13.x
on PATH is an equally valid source. The pack directory itself is always
searched first by the Windows loader, so dropping a newer `cublas64_13.dll`
next to the exe (if ever needed) takes precedence.

## 3. Dynamic backend evidence (why ggml-cuda.dll is packaged)

`ggml-cuda.dll` and the `ggml-cpu-*.dll` variants are **not** in any static
import table — they are discovered at runtime. In-binary evidence in
`ggml.dll`: `ggml_backend_load`, `ggml_backend_load_best`,
`ggml_backend_load_all`, `ggml_backend_load_all_from_path`. Consequence:

* without `ggml-cuda.dll` present in the exe directory, the server starts
  **CPU-only** (silent degradation — `-ngl 99` becomes meaningless) — so it is
  category B by requirement, not by import table;
* without any `ggml-cpu-*.dll`, no CPU backend registers — the full variant
  set is shipped as upstream distributes it (~20 MB, loader picks).

## 4. Excluded by inspection (present in the source zip, NOT needed)

| Excluded | Reason |
| --- | --- |
| `ggml-rpc.dll`, `ggml-rpc-server.exe` | RPC backend: no importer in the closure, unused by VoiceMem |
| `llama-cli.exe` + `llama-cli-impl.dll`, `llama.exe`, `llama-bench*`, `llama-perplexity*`, `llama-quantize*`, `llama-completion*`, `llama-batched-bench*`, `llama-fit-params*`, `llama-imatrix.exe`, `llama-tokenize.exe`, `llama-tts.exe`, `llama-results.exe`, `llama-gguf-split.exe`, `llama-mtmd-cli.exe`, `llama-mtmd-debug.exe`, `llama-gemma3-cli.exe`, `llama-llava-cli.exe`, `llama-minicpmv-cli.exe`, `llama-qwen2vl-cli.exe` | none is in `llama-server.exe`'s import closure (tools for other workflows) |

No CUDA DLL, no driver file, no model file, no VoiceMem application file, no
Python component is packaged — per the delta-pack order.

## 5. Verification performed

**In the sandbox (this document's evidence):**
1. downloaded asset SHA-256 == GitHub API digest [PROVEN];
2. recursive import closure extracted with `objdump -p` from the actual
   binaries [PROVEN];
3. classification complete: every DLL in the closure is either packaged (B)
   or documented (A-SYSTEM / A-NVIDIA); no UNCLASSIFIED import remains;
4. dynamic-backend mechanism and build identity confirmed in-binary strings
   [PROVEN];
5. the final pack re-extracted and re-inspected: closure check + hash
   manifest verify (see the build log in the worklog).

**On the target (the steps the sandbox cannot run):**
1. `check_environment.ps1` — pack presence, `SHA256SUMS.txt` integrity,
   `cublas64_13.dll` resolution (reports the path it will load from),
   VC++/UCRT presence, driver sanity, model resolution;
2. the validated launcher's own gate — `/health` within the wait window and
   the version banner (`0.4.1-dev (build 11073, commit 1aa2954bd)`) in
   `logs\llama-server-experimental.out.log`;
3. `/props` and `/slots` (the same live checks the validation audit used).
