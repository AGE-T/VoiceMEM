<#
===============================================================================
experimental/llama_b11073/check_environment.ps1 — DELTA RUNTIME PACK probe

READ-ONLY prerequisite + integrity verification for the b11073 delta runtime
pack, to be run ON THE TARGET MACHINE (Windows). It checks, in order:

  1. PACK FILES  — every category-B file the dependency inspection requires
     (bin\llama-server-b11073\llama-server.exe + its application-local DLLs +
     the dynamically loaded ggml backends) is present;
  2. INTEGRITY   — SHA256SUMS.txt (shipped in this folder) verifies every
     packaged file byte-for-byte (Get-FileHash);
  3. CUDA PRE    — cublas64_13.dll (the ONLY CUDA DLL the runtime needs —
     see DEPENDENCY_INSPECTION.md) is found somewhere the Windows loader
     will look: the pack dir, System32, or PATH. NOT packaged by design:
     the target machine already provides it (production bin\ from the
     b10717 cudart install and/or a system CUDA 13.x);
  4. VC++ / UCRT — msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll,
     ucrtbase.dll (target-provided; the production b10717 server already
     proves this machine has them — informational, not a gate);
  5. GPU DRIVER  — nvidia-smi presence (informational only);
  6. MODEL       — the LLM model resolution the launcher will use
     (config\llm_model.json > LLAMA_MODEL_PATH > default profile path).

Exit codes: 0 = ready (warnings allowed) / 2 = HARD FAIL (missing pack file,
integrity mismatch, or no cublas64_13.dll anywhere the loader searches).

This script changes NOTHING. It is part of the experimental delta pack;
production (bin\llama-server.exe b10717, config, scripts, VERSION) is not
touched.
===============================================================================
#>
param(
    [switch]$Quiet
)

$ErrorActionPreference = "Continue"
$ScriptDir = $PSScriptRoot
$Root = (Get-Item $ScriptDir).Parent.Parent.FullName
$PackBin = Join-Path $Root "bin\llama-server-b11073"

$fails  = New-Object System.Collections.Generic.List[string]
$warns  = New-Object System.Collections.Generic.List[string]

function Result([string]$Label, [bool]$Ok, [string]$Detail, [switch]$Hard) {
    if ($Ok) {
        if (-not $Quiet) { Write-Host ("  [OK]    {0,-46} {1}" -f $Label, $Detail) }
    } else {
        if ($Hard) {
            $fails.Add("$Label : $Detail")
            Write-Host ("  [FAIL]  {0,-46} {1}" -f $Label, $Detail) -ForegroundColor Red
        } else {
            $warns.Add("$Label : $Detail")
            if (-not $Quiet) { Write-Host ("  [WARN]  {0,-46} {1}" -f $Label, $Detail) -ForegroundColor Yellow }
        }
    }
}

if (-not $Quiet) {
    Write-Host "=== b11073 DELTA RUNTIME PACK — korfeltetel-proba (read-only) ==="
    Write-Host ("  root : {0}" -f $Root)
    Write-Host ""
    Write-Host "[1/6] Pack fajlok (bin\llama-server-b11073\):"
}

# --- [1/6] category-B runtime files (see DEPENDENCY_INSPECTION.md) -----------
$PackFiles = @(
    "llama-server.exe", "llama-server-impl.dll", "llama-common.dll",
    "llama.dll", "mtmd.dll", "ggml.dll", "ggml-base.dll", "libomp.dll",
    "ggml-cuda.dll",
    "ggml-cpu-alderlake.dll",   "ggml-cpu-cannonlake.dll",
    "ggml-cpu-cascadelake.dll", "ggml-cpu-cooperlake.dll",
    "ggml-cpu-haswell.dll",     "ggml-cpu-icelake.dll",
    "ggml-cpu-ivybridge.dll",   "ggml-cpu-piledriver.dll",
    "ggml-cpu-sandybridge.dll", "ggml-cpu-sapphirerapids.dll",
    "ggml-cpu-skylakex.dll",    "ggml-cpu-sse42.dll",
    "ggml-cpu-x64.dll",         "ggml-cpu-zen4.dll",
    "LICENSE-LLVM-OpenMP"
)
foreach ($f in $PackFiles) {
    $p = Join-Path $PackBin $f
    if (Test-Path -LiteralPath $p) {
        if (-not $Quiet) { Write-Host ("  [OK]    {0,-46} {1:N0} bytes" -f $f, (Get-Item $p).Length) }
    } else {
        Result $f $false "HIANYZIK a bin\llama-server-b11073\ mabbol" -Hard
    }
}

if (-not $Quiet) { Write-Host ""; Write-Host "[2/6] Integritas (SHA256SUMS.txt):" }
# --- [2/6] integrity manifest -------------------------------------------------
$SumsFile = Join-Path $ScriptDir "SHA256SUMS.txt"
if (-not (Test-Path -LiteralPath $SumsFile)) {
    Result "SHA256SUMS.txt" $false "nincs meg (a csomag resze kell legyen)" -Hard
} else {
    $nOk = 0; $nBad = 0; $nMiss = 0
    foreach ($line in Get-Content -LiteralPath $SumsFile) {
        if ($line -notmatch '^\s*([0-9A-Fa-f]{64})\s+(.+?)\s*$') { continue }
        $expected = $Matches[1]
        $rel = $Matches[2] -replace '^\.\\', '' -replace '^/', ''
        $target = Join-Path $ScriptDir $rel
        if (-not (Test-Path -LiteralPath $target)) {
            $nMiss++; $fails.Add("integritas: hianyzo fajl $rel"); continue
        }
        $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        if ($actual -eq $expected.ToUpper()) { $nOk++ }
        else { $nBad++; $fails.Add("integritas: HASH ELTER: $rel") }
    }
    if ($nBad -eq 0 -and $nMiss -eq 0) {
        if (-not $Quiet) { Write-Host ("  [OK]    {0,-46} {1}/{1} fajl egyezik" -f "SHA-256 mind", $nOk) }
    } else {
        Write-Host ("  [FAIL]  SHA-256: {0} OK, {1} elter, {2} hianyzik" -f $nOk, $nBad, $nMiss) -ForegroundColor Red
    }
}

if (-not $Quiet) { Write-Host ""; Write-Host "[3/6] CUDA elofeltetel (cublas64_13.dll — NEM a csomag resze):" }
# --- [3/6] cublas64_13.dll — the ONLY CUDA dependency (target-provided) -------
$CudaName = "cublas64_13.dll"
$Found = $null
# Windows loader order (approximation): app dir, System32, then PATH in order.
$SearchDirs = New-Object System.Collections.Generic.List[string]
$SearchDirs.Add($PackBin)
$SearchDirs.Add((Join-Path $env:windir "System32"))
$SearchDirs.AddRange(($env:Path -split ';' | Where-Object { $_ }))
foreach ($d in $SearchDirs) {
    $cand = Join-Path $d $CudaName
    if (Test-Path -LiteralPath $cand) { $Found = $cand; break }
}
if ($Found) {
    $ver = ""
    try { $ver = (Get-Item $Found).VersionInfo.FileVersion } catch { }
    if (-not $Quiet) {
        Write-Host ("  [OK]    {0,-46} {1}" -f $CudaName, $Found)
        if ($ver) { Write-Host ("          file version: {0}" -f $ver) }
    }
    $prodBin = Join-Path $Root "bin"
    if ($Found -like "$prodBin*") {
        if (-not $Quiet) { Write-Host "          (forras: a gyartasi b10717 cudart letoltes — ez a tervezett mod)" -ForegroundColor DarkGray }
    }
} else {
    Result $CudaName $false ("NINCS a loader keresesi utvonalan (csomagdir/System32/PATH). Varhato hely: <root>\bin a gyartasi cudart letoltesbol, vagy rendszer-szintu CUDA 13.x") -Hard
}

if (-not $Quiet) { Write-Host ""; Write-Host "[4/6] VC++ / UCRT futtatokornyezet (rendszer — informacios):" }
# --- [4/6] VC++ redist + UCRT (informational; production server proves them) --
foreach ($dll in @("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "ucrtbase.dll")) {
    $p = Join-Path (Join-Path $env:windir "System32") $dll
    if (Test-Path -LiteralPath $p) {
        if (-not $Quiet) { Write-Host ("  [OK]    {0,-46} System32" -f $dll) }
    } else {
        Result $dll $false "nincs a System32-ben (a gyartasi szerver futasabol ki kellett volna derulnie)" -Hard
    }
}

if (-not $Quiet) { Write-Host ""; Write-Host "[5/6] GPU driver (nvidia-smi — informacios):" }
# --- [5/6] GPU driver (informational) -----------------------------------------
$smi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($smi) {
    try {
        $gpu = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
        if (-not $Quiet) { Write-Host ("  [OK]    nvidia-smi: {0}" -f $gpu) }
    } catch {
        Result "nvidia-smi" $false "megvan, de nem valaszol" 
    }
} else {
    Result "nvidia-smi" $false "nincs a PATH-on (a driver maga lehet rendben)" 
}
if (-not $Quiet) { Write-Host ""; Write-Host "[6/6] LLM modell feloldas (a launcher azonos sorrendje — informacios):" }
# --- [6/6] model resolution (informational; launcher fails loudly if absent) --
$Model = $null; $ModelSource = ""
$LlmModelJson = Join-Path $Root "config\llm_model.json"
if (Test-Path -LiteralPath $LlmModelJson) {
    try {
        $Sel = Get-Content -LiteralPath $LlmModelJson -Raw | ConvertFrom-Json
        if ($Sel.model -and (Test-Path -LiteralPath $Sel.model)) {
            $Model = [string]$Sel.model; $ModelSource = "config/llm_model.json"
        }
    } catch { }
}
if (-not $Model -and $env:LLAMA_MODEL_PATH -and (Test-Path -LiteralPath $env:LLAMA_MODEL_PATH)) {
    $Model = [string]$env:LLAMA_MODEL_PATH; $ModelSource = "LLAMA_MODEL_PATH env"
}
if (-not $Model) {
    $Default = Join-Path $Root "models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf"
    if (Test-Path -LiteralPath $Default) { $Model = $Default; $ModelSource = "default profil ut" }
}
if ($Model) {
    if (-not $Quiet) { Write-Host ("  [OK]    {0}" -f $Model); Write-Host ("          forras: {0}" -f $ModelSource) }
} else {
    Result "LLM modell" $false "nem feloldhato (config\llm_model.json > LLAMA_MODEL_PATH > models\llm\qwen3.6-35b-a3b\)" -Hard
}

# --- summary ------------------------------------------------------------------
Write-Host ""
if ($fails.Count -gt 0) {
    Write-Host ("EREDMENY: HIBA — {0} hiba, {1} figyelmeztetes. NEM indithato biztonsagosan." -f $fails.Count, $warns.Count) -ForegroundColor Red
    exit 2
} else {
    if ($warns.Count -gt 0) {
        Write-Host ("EREDMENY: RENDBEN FIGYELMEZTETESEKKEL ({0}) — a szerver indithato." -f $warns.Count) -ForegroundColor Yellow
    } else {
        Write-Host "EREDMENY: RENDBEN — minden korfeltetel teljesul, a szerver indithato." -ForegroundColor Green
    }
    exit 0
}
