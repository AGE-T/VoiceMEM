<#
===============================================================================
experimental/llama_b11073/check_environment.ps1 - INSTALL/RUNTIME PREREQUISITE PROBE

READ-ONLY prerequisite + integrity verification for the experimental b11073
runtime, to be run ON THE TARGET MACHINE (Windows) any time after
install_b11073.ps1. It checks, in order:

  1. RUNTIME FILES  - every file the measured dependency closure requires
     (experimental\llama_b11073\bin\llama-server.exe + its application-local
     DLLs + the dynamically loaded ggml backends) is present;
  2. INTEGRITY    - every installed file's SHA-256 matches the pinned table
     (config\b11073.pins.json);
  3. CUDA PRE     - cublas64_13.dll (the ONLY CUDA DLL the runtime needs -
     see DEPENDENCY_INSPECTION.md) is found somewhere the Windows loader
     will look once the launcher has prepended the production bin\ to the
     child PATH: the experimental bin\, the production bin\, System32, or
     the current PATH. NOT installed by the pack or installer by design:
     the target machine already provides it (production bin\ from the
     b10717 cudart install and/or a system CUDA 13.x);
  4. VC++ / UCRT  - msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll,
     ucrtbase.dll (target-provided; the production b10717 server already
     proves this machine has them - informational, not a gate);
  5. GPU DRIVER   - nvidia-smi presence (informational only);
  6. MODEL        - the LLM model resolution the launcher will use
     (config\llm_model.json > LLAMA_MODEL_PATH > default profile path).

Exit codes: 0 = ready (warnings allowed) / 2 = HARD FAIL (missing runtime
file, integrity mismatch, or no cublas64_13.dll anywhere the loader searches).

This script changes NOTHING. Production (bin\llama-server.exe b10717,
config, scripts, VERSION) is not touched.

NOTE: deliberately ASCII (PowerShell 5.1 reads BOM-less files as Windows-1252).
===============================================================================
#>

param(
    [switch]$Quiet
)

$ErrorActionPreference = "Continue"
$ScriptDir = $PSScriptRoot
$Root = (Get-Item -LiteralPath $ScriptDir).Parent.Parent.FullName
$BinDir = Join-Path $ScriptDir "bin"
$ProdBin = Join-Path $Root "bin"

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
    Write-Host "=== EXPERIMENTAL b11073 - korfeltetel-proba (read-only; installer modell) ==="
    Write-Host ("  root   : {0}" -f $Root)
    Write-Host ("  runtime: {0}" -f $BinDir)
    Write-Host ""
    Write-Host "[1/6] Runtime fajlok (experimental\llama_b11073\bin\):"
}

# --- [1/6] category-B runtime files (see DEPENDENCY_INSPECTION.md) ----------------
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
$Missing = 0
foreach ($f in $PackFiles) {
    $p = Join-Path $BinDir $f
    if (Test-Path -LiteralPath $p) {
        if (-not $Quiet) { Write-Host ("  [OK]    {0,-46} {1:N0} bytes" -f $f, (Get-Item -LiteralPath $p).Length) }
    } else {
        $Missing++
        Result $f $false "HIANYZIK - futtasd: install_b11073.ps1" -Hard
    }
}
if ($Missing -gt 0 -and -not $Quiet) {
    Write-Host "          (a runtime letoltese: install_b11073.ps1; NINCS binaris a pack-ban)" -ForegroundColor DarkGray
}

if (-not $Quiet) { Write-Host ""; Write-Host "[2/6] Integritas (config\b11073.pins.json pin-tabla):" }
# --- [2/6] integrity against the pinned per-file table -----------------------------
$PinsFile = Join-Path $ScriptDir "config\b11073.pins.json"
if (-not (Test-Path -LiteralPath $PinsFile)) {
    Result "b11073.pins.json" $false "nincs meg (a csomag resze kell legyen)" -Hard
} elseif ($Missing -gt 0) {
    Result "integritas" $false "kihagyva (hianyzo runtime fajlok)" 
} else {
    $nOk = 0; $nBad = 0
    try {
        $Pins = Get-Content -LiteralPath $PinsFile -Raw | ConvertFrom-Json
        foreach ($f in $Pins.runtime.files) {
            $target = Join-Path $BinDir ([string]$f.name)
            if (-not (Test-Path -LiteralPath $target)) { $nBad++; $fails.Add("integritas: hianyzo fajl " + $f.name); continue }
            $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
            if ($actual -eq ([string]$f.sha256).ToUpper()) { $nOk++ }
            else { $nBad++; $fails.Add("integritas: HASH ELTER: " + $f.name) }
        }
    } catch {
        Result "b11073.pins.json" $false ("nem olvashato: " + $_.Exception.Message) -Hard
    }
    if ($nBad -eq 0 -and $nOk -gt 0) {
        if (-not $Quiet) { Write-Host ("  [OK]    {0,-46} {1}/{1} fajl hash-e egyezik" -f "SHA-256 mind", $nOk) }
    } elseif ($nOk -gt 0 -or $nBad -gt 0) {
        Write-Host ("  [FAIL]  SHA-256: {0} OK, {1} elter" -f $nOk, $nBad) -ForegroundColor Red
    }
}

if (-not $Quiet) { Write-Host ""; Write-Host "[3/6] CUDA elofeltetel (cublas64_13.dll - NEM a csomag resze):" }
# --- [3/6] cublas64_13.dll - the ONLY CUDA dependency (target-provided) -------------
# Search order mirrors what the launcher makes the loader do: experimental bin\
# first (exe dir), then the production bin\ (the launcher PREPENDS it to the
# child PATH), then System32, then the rest of the current PATH.
$CudaName = "cublas64_13.dll"
$Found = $null
$SearchDirs = New-Object System.Collections.Generic.List[string]
$SearchDirs.Add($BinDir)
$SearchDirs.Add($ProdBin)
if ($env:windir) { $SearchDirs.Add((Join-Path $env:windir "System32")) }
$SearchDirs.AddRange(@([Environment]::GetEnvironmentVariable("PATH") -split ([System.IO.Path]::PathSeparator) | Where-Object { $_ }))
foreach ($d in $SearchDirs) {
    $cand = Join-Path $d $CudaName
    if (Test-Path -LiteralPath $cand) { $Found = $cand; break }
}
if ($Found) {
    $ver = ""
    try { $ver = (Get-Item -LiteralPath $Found).VersionInfo.FileVersion } catch { }
    if (-not $Quiet) {
        Write-Host ("  [OK]    {0,-46} {1}" -f $CudaName, $Found)
        if ($ver) { Write-Host ("          file version: {0}" -f $ver) }
        if ($Found -like "$ProdBin*") {
            Write-Host "          (forras: a gyartasi b10717 cudart letoltes - ez a tervezett mod)" -ForegroundColor DarkGray
        }
    }
} else {
    Result $CudaName $false ("NINCS a loader keresesi utvonalan (experimental bin / gyartasi bin / System32 / PATH). Varhato hely: <root>\bin a gyartasi cudart letoltesbol (START.bat egyszeri futtatasa), vagy rendszer-szintu CUDA 13.x") -Hard
}

if (-not $Quiet) { Write-Host ""; Write-Host "[4/6] VC++ / UCRT futtatokornyezet (rendszer - informacios):" }
# --- [4/6] VC++ redist + UCRT (informational; production server proves them) --------
if (-not $env:windir) {
    if (-not $Quiet) { Write-Host "  [----]  kihagyva (nem Windows - a sandbox tesztfutas)" }
} else {
    foreach ($dll in @("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "ucrtbase.dll")) {
        $p = Join-Path (Join-Path $env:windir "System32") $dll
        if (Test-Path -LiteralPath $p) {
            if (-not $Quiet) { Write-Host ("  [OK]    {0,-46} System32" -f $dll) }
        } else {
            Result $dll $false "nincs a System32-ben (a gyartasi szerver futasabol ki kellett volna derulnie)" -Hard
        }
    }
}

if (-not $Quiet) { Write-Host ""; Write-Host "[5/6] GPU driver (nvidia-smi - informacios):" }
# --- [5/6] GPU driver (informational) -----------------------------------------------
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
if (-not $Quiet) { Write-Host ""; Write-Host "[6/6] LLM modell feloldas (a launcher azonos sorrendje - informacios):" }
# --- [6/6] model resolution (informational; launcher fails loudly if absent) --------
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
    Write-Host ("EREDMENY: HIBA - {0} hiba, {1} figyelmeztetes. NEM indithato biztonsagosan." -f $fails.Count, $warns.Count) -ForegroundColor Red
    exit 2
} else {
    if ($warns.Count -gt 0) {
        Write-Host ("EREDMENY: RENDBEN FIGYELMEZTETESEKKEL ({0}) - a szerver indithato." -f $warns.Count) -ForegroundColor Yellow
    } else {
        Write-Host "EREDMENY: RENDBEN - minden korfeltetel teljesul, a szerver indithato." -ForegroundColor Green
    }
    exit 0
}
