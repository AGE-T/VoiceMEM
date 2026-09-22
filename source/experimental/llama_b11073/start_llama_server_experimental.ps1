<#
===============================================================================
experimental/llama_b11073/start_llama_server_experimental.ps1

ISOLATED EXPERIMENTAL llama.cpp runtime launcher (target machine / Windows).

PURPOSE: run the operator-tested NEWER llama.cpp build (b11073) NEXT TO the
untouched production runtime, on a SEPARATE executable path
(experimental\llama_b11073\bin\, installed by install_b11073.ps1) and a
SEPARATE port (default 127.0.0.1:8081), so the whole VoiceMem application
can be pointed at it with the EXISTING environment-only override. The
recommended way is the GENERATED session script
(start_voicemem_experimental.ps1, created by configure_b11073.ps1 -Enable):

    experimental\llama_b11073\start_voicemem_experimental.ps1

which loads the standard env, sets the three override variables and starts
the app through the existing starter. Manual equivalent:

    $env:LLAMA_SERVER_HOST = "127.0.0.1"
    $env:LLAMA_SERVER_PORT = "8081"
    $env:OPENAI_BASE_URL   = "http://127.0.0.1:8081/v1"   # vendor legs + start_agent.ps1 env-guard
    .\START.bat    (NOTE: START.bat would ALSO auto-start the production
                    server on 8080; the generated session script uses
                    scripts\start_agent.ps1 -Web -NoServer instead to avoid
                    loading the 19 GB production model twice.)

REVERTING THE EXPERIMENT (full rollback):
    1. experimental\llama_b11073\configure_b11073.ps1 -Disable
       (stops this server, removes the override, prints the restore steps)
    2. Optionally delete experimental\llama_b11073\ (the runtime + scripts).

PRODUCTION IS NOT TOUCHED BY THIS SCRIPT:
    - it NEVER reads or writes config\llm_config.yaml (the canonical file
      keeps generating the production command line);
    - it NEVER touches bin\llama-server.exe (the pinned b10717 production
      binary) or any production DLL - the experimental executable lives in
      experimental\llama_b11073\bin\;
    - it does NOT edit scripts\start_llama_server.ps1;
    - it binds 127.0.0.1:8081 (never 8080), so the production server can run
      at the same time if the operator wants a live A/B on the same machine.

EXPERIMENTAL PROFILE - the operator-tested baseline (2026-09-20 manual test;
~13 tok/s generation observed with llama-cli on the target machine). This is
the FIRST experimental candidate, NOT an optimised one, and deliberately NOT
the canonical production profile (ngl 20 / ctx 32768):

    ngl 99 | ctx 16000 | parallel 1 | threads 12 | reasoning off
    (+ KV cache q8_0 / temp 0.7 kept equal to production for comparability)

CUDA RESOLUTION (see DEPENDENCY_INSPECTION.md): the runtime imports exactly
one CUDA DLL, cublas64_13.dll, which is NOT installed next to the exe - the
target machine already provides it. The launcher therefore PREPENDS the
production bin\ directory to the CHILD process PATH (never copies or
modifies any CUDA component); a system-wide CUDA 13.x on PATH is an equally
valid source.

MODEL RESOLUTION is IDENTICAL to production (same model - the A/B contract):
    config\llm_model.json (web UI selection) > LLAMA_MODEL_PATH env >
    models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf

NOTE: deliberately ASCII (PowerShell 5.1 reads BOM-less files as Windows-1252).
===============================================================================
#>

param(
    [int]$Port = 8081,
    [int]$WaitSec = 180
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$Root      = (Get-Item -LiteralPath $ScriptDir).Parent.Parent.FullName
$LogsDir   = Join-Path $ScriptDir "logs"

# --- experimental binary (SEPARATE path; production bin\ is never touched) -------
$Exe = Join-Path $ScriptDir "bin\llama-server.exe"
if (-not (Test-Path -LiteralPath $Exe)) {
    Write-Host "HIBA: az experimentalis binaris nem letezik: $Exe" -ForegroundColor Red
    Write-Host "JAVITAS: futtasd eloszor a telepitot (a runtime NINCS a pack-ban, letoltes toltesik):" -ForegroundColor Yellow
    Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\install_b11073.ps1" -ForegroundColor Yellow
    Write-Host "A gyartasi bin\llama-server.exe (b10717) NINCS erintve." -ForegroundColor Yellow
    exit 1
}

# --- effective-configuration log (this launch's exact resolved state) ------------
if (-not (Test-Path -LiteralPath $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null }
$Stamp    = Get-Date -Format "yyyyMMdd_HHmmss"
$CfgLog   = Join-Path $LogsDir ("launcher-config_" + $Stamp + ".log")
function Log([string]$m) {
    try { Add-Content -LiteralPath $CfgLog -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $m) } catch { }
}
Log ("launcher start; script dir = " + $ScriptDir + "; root = " + $Root)
Log ("binary = " + $Exe)
try { Log ("binary sha256 = " + (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash) } catch { }

# --- VERSION VERIFICATION #1: the binary itself must be b11073 --------------------
$VersionLine = ""
try { $VersionLine = [string](& $Exe --version 2>&1 | Select-Object -First 1) } catch { }
Log ("--version = " + $VersionLine)
Write-Host "=== EXPERIMENTAL llama-server (b11073) - EZ NEM A GYARTASI SZERVER ===" -ForegroundColor Magenta
Write-Host ("  binary : {0}" -f $Exe)
if ($VersionLine -ne "") { Write-Host ("          {0}" -f $VersionLine) }
if ($VersionLine -notmatch "11073") {
    Write-Host ("HIBA: a binaris VERZIOJA NEM b11073! Kapott: " + $VersionLine) -ForegroundColor Red
    Write-Host "NEM INDUL EL - soha ne futtass masik verziot a kiserleti profilban." -ForegroundColor Red
    Write-Host "JAVITAS: torold az experimental\llama_b11073\bin mappat es futtasd ujra a telepitot" -ForegroundColor Yellow
    Write-Host "         (install_b11073.ps1 -Force), amely SHA-256 alapjan pin-eli a b11073-at." -ForegroundColor Yellow
    exit 1
}

# --- model resolution (same order as the production starter) ----------------------
$Model = ""
$ModelSource = ""
$LlmModelJson = Join-Path $Root "config\llm_model.json"
if (Test-Path -LiteralPath $LlmModelJson) {
    try {
        $Sel = Get-Content -LiteralPath $LlmModelJson -Raw | ConvertFrom-Json
        if ($Sel.model -and (Test-Path -LiteralPath $Sel.model)) {
            $Model = [string]$Sel.model
            $ModelSource = "config/llm_model.json (web UI kivalasztas)"
        }
    } catch { }
}
if (-not $Model -and $env:LLAMA_MODEL_PATH -and (Test-Path -LiteralPath $env:LLAMA_MODEL_PATH)) {
    $Model = [string]$env:LLAMA_MODEL_PATH
    $ModelSource = "LLAMA_MODEL_PATH env"
}
if (-not $Model) {
    $Default = Join-Path $Root "models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf"
    if (Test-Path -LiteralPath $Default) {
        $Model = $Default
        $ModelSource = "default profil ut"
    }
}
if (-not $Model) {
    Write-Host "HIBA: nincs feloldhato LLM modell (config\llm_model.json > LLAMA_MODEL_PATH > models\llm\qwen3.6-35b-a3b\)." -ForegroundColor Red
    exit 1
}
Log ("model = " + $Model + " (source: " + $ModelSource + ")")

# --- experimental profile (operator-tested baseline) ------------------------------
$Ngl = "99"
if ($env:LLAMA_EXPERIMENTAL_NGL) { $Ngl = [string]$env:LLAMA_EXPERIMENTAL_NGL }
$Ctx = "16000"
if ($env:LLAMA_EXPERIMENTAL_CTX) { $Ctx = [string]$env:LLAMA_EXPERIMENTAL_CTX }
$Threads = "12"
if ($env:LLAMA_EXPERIMENTAL_THREADS) { $Threads = [string]$env:LLAMA_EXPERIMENTAL_THREADS }

$Host_ = "127.0.0.1"

Write-Host ("  model  : {0}  ({1})" -f $Model, $ModelSource)
Write-Host ("  bind   : {0}:{1}   (a gyartasi 8080 NINCS erintve)" -f $Host_, $Port)
Write-Host ("  profil : ngl {0} | ctx {1} | parallel 1 | threads {2} | reasoning off | KV q8_0" -f $Ngl, $Ctx, $Threads)

# --- CUDA prerequisite: prepend the production bin\ to the CHILD PATH --------------
# cublas64_13.dll (the ONLY CUDA import of ggml-cuda.dll) resolves from the
# existing production cudart install and/or a system CUDA 13.x. The pack and
# the installer never place CUDA DLLs next to the exe; this PATH extension is
# the only mechanism, and it applies to the CHILD process alone.
$ProdBin = Join-Path $Root "bin"
$CudaSource = ""
$CudaSearch = @($ProdBin)
if ($env:windir) { $CudaSearch += (Join-Path $env:windir "System32") }
foreach ($D in $CudaSearch) {
    if (Test-Path -LiteralPath (Join-Path $D "cublas64_13.dll")) { $CudaSource = $D; break }
}
if (-not $CudaSource) {
    foreach ($D in @([Environment]::GetEnvironmentVariable("PATH") -split ([System.IO.Path]::PathSeparator) | Where-Object { $_ })) {
        if (Test-Path -LiteralPath (Join-Path $D "cublas64_13.dll")) { $CudaSource = $D; break }
    }
}
if ($CudaSource) {
    Write-Host ("  cuda   : cublas64_13.dll <- {0} (gyermek-PATH bovites)" -f $CudaSource)
    Log ("cublas64_13.dll resolves from: " + $CudaSource)
} else {
    Write-Host "  FIGYELMEZTETES: cublas64_13.dll NINCS a keresesi utvonalon!" -ForegroundColor Yellow
    Write-Host "  A szerver indulasa valoszinuleg 0xC0000135-tel elhal (hianyzo DLL)." -ForegroundColor Yellow
    Write-Host "  Varhato forras: a gyartasi bin\ cudart letoltes (START.bat egyszeri futtatasa)" -ForegroundColor Yellow
    Write-Host "  vagy rendszer-szintu CUDA 13.x. Reszletek: README.md." -ForegroundColor Yellow
    Log "cublas64_13.dll NOT FOUND on search path (warning)"
}
$env:PATH = "$ProdBin;$env:PATH"
Log ("child PATH prepended with: " + $ProdBin)

# --- idempotent start: never a second server on 8081 ------------------------------
function Test-Health {
    param([string]$H, [int]$P, [int]$T = 3)
    try {
        $R = Invoke-WebRequest -Uri ("http://{0}:{1}/health" -f $H, $P) -Method Get -TimeoutSec $T -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}
if (Test-Health $Host_ $Port) {
    Write-Host "MAR FUT egy szerver a ${Host_}:${Port} cimen - nem inditok masodikat." -ForegroundColor Yellow
    Log "health already OK before start; idempotent exit"
    exit 0
}

$ErrLog = Join-Path $LogsDir "llama-server-experimental.err.log"
$OutLog = Join-Path $LogsDir "llama-server-experimental.out.log"

$Args = @(
    "--model", $Model,
    "--host", $Host_, "--port", "$Port",
    "-ngl", $Ngl, "-c", $Ctx, "--parallel", "1", "-t", $Threads,
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--temp", "0.7", "--reasoning", "off", "--metrics", "--no-webui", "--verbose"
)
Write-Host ("Inditas: {0} {1}" -f $Exe, ($Args -join " "))
Write-Host ("Logok  : {0} / {1}" -f $ErrLog, $OutLog)
Write-Host ("Konfig : {0}" -f $CfgLog)
Log ("effective command: " + $Exe + " " + ($Args -join " "))
Log ("effective env overrides: NGL=" + $Ngl + " CTX=" + $Ctx + " THREADS=" + $Threads + " PORT=" + $Port)

$Proc = Start-Process -FilePath $Exe -ArgumentList $Args -WorkingDirectory $Root `
    -RedirectStandardError $ErrLog -RedirectStandardOutput $OutLog -PassThru
Log ("started pid " + $Proc.Id)

# --- /health wait (same primary success condition as the production starter) -------
$Deadline = (Get-Date).AddSeconds($WaitSec)
while ((Get-Date) -lt $Deadline) {
    if ($Proc.HasExited) {
        Write-Host ("HIBA: az experimentalis llama-server kilepett (kod {0}). Log: {1}" -f $Proc.ExitCode, $ErrLog) -ForegroundColor Red
        Log ("server exited early, code " + $Proc.ExitCode)
        if (Test-Path -LiteralPath $ErrLog) {
            Get-Content -LiteralPath $ErrLog -Tail 8 | ForEach-Object { Write-Host ("          " + $_) -ForegroundColor DarkYellow }
        }
        Write-Host "Gyakori ok: hianyzo cublas64_13.dll (l. fent) vagy serult modellfajl." -ForegroundColor Yellow
        exit 1
    }
    if (Test-Health $Host_ $Port) {
        Write-Host "OK: experimentalis szerver ELO a ${Host_}:${Port} cimen." -ForegroundColor Green
        Log "health OK"

        # --- VERSION VERIFICATION #2: the RUNNING server must be b11073 ------------
        $Fingerprint = ""
        try {
            $Resp = Invoke-RestMethod -Uri ("http://{0}:{1}/v1/chat/completions" -f $Host_, $Port) `
                -Method Post -ContentType "application/json" `
                -Body (ConvertTo-Json -Compress -InputObject ([ordered]@{
                    messages = @(@{ role = "user"; content = "ping" })
                    max_tokens = 1
                    temperature = 0.0
                })) -TimeoutSec 60
            $Fingerprint = [string]$Resp.system_fingerprint
        } catch {
            Log ("fingerprint probe failed: " + $_.Exception.Message)
        }
        if ($Fingerprint -match "b11073") {
            Write-Host ("  VERZIO-ELLENORZES OK: a futtato szerver rendszerujjlenyomat = " + $Fingerprint) -ForegroundColor Green
            Log ("system_fingerprint = " + $Fingerprint + " (b11073 CONFIRMED LIVE)")
        } else {
            Write-Host ("HIBA: a 8081-es szerver NEM a b11073! system_fingerprint = '" + $Fingerprint + "'") -ForegroundColor Red
            Write-Host "Allitsd le es nezd meg, mi fut a 8081-en (configure_b11073.ps1 -Disable)." -ForegroundColor Yellow
            Log ("version verification FAILED live: fingerprint = '" + $Fingerprint + "'")
            exit 1
        }

        Write-Host "Visszaallas (rollback): configure_b11073.ps1 -Disable" -ForegroundColor Gray
        Write-Host "Ez az ablak nyitva tartja a folyamatot (Ctrl+C = leallitas)." -ForegroundColor Gray
        Wait-Process -Id $Proc.Id
        exit 0
    }
    Start-Sleep -Milliseconds 500
}
Write-Host "HIBA: az /health nem valt elerhetove ${WaitSec} mp alatt. Log: $ErrLog" -ForegroundColor Red
Log "health wait timed out"
Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
exit 1
