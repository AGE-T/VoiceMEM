<#
===============================================================================
experimental/llama_b11073/start_llama_server_experimental.ps1

ISOLATED EXPERIMENTAL llama.cpp runtime launcher (target machine / Windows).

PURPOSE: run the operator-tested NEWER llama.cpp build (b11073) NEXT TO the
untouched production runtime, on a SEPARATE executable path and a SEPARATE
port (default 127.0.0.1:8081), so the whole VoiceMem application can be
pointed at it with the EXISTING environment-only override (NO application
code change — app/config.py apply_env already honours these):

    $env:LLAMA_SERVER_HOST = "127.0.0.1"
    $env:LLAMA_SERVER_PORT = "8081"
    .\scripts\start_agent.ps1        (or START.bat in the second window)

REVERTING THE EXPERIMENT (full rollback):
    1. Stop this server (Ctrl+C in its window, or close the window).
    2. Remove-Item Env:LLAMA_SERVER_HOST, LLAMA_SERVER_PORT in the agent
       session (or just open a fresh terminal) — the agent then targets the
       production server on 127.0.0.1:8080 again, byte-for-byte unchanged.
    3. Optionally delete bin\llama-server-b11073\ (the experimental binary).

PRODUCTION IS NOT TOUCHED BY THIS SCRIPT:
    - it NEVER reads or writes config/llm_config.yaml (the canonical file
      keeps generating the production command line);
    - it NEVER touches bin\llama-server.exe (the pinned b10717 production
      binary) — the experimental executable lives in bin\llama-server-b11073\;
    - it does NOT edit scripts/start_llama_server.ps1;
    - it binds 127.0.0.1:8081 (never 8080), so the production server can run
      at the same time if the operator wants a live A/B on the same machine.

EXPERIMENTAL PROFILE — the operator-tested baseline (2026-09-20 manual test;
~13 tok/s generation observed with llama-cli on the target machine). This is
the FIRST experimental candidate, NOT an optimised one, and deliberately NOT
the canonical production profile (ngl 20 / ctx 32768):

    ngl 99 | ctx 16000 | parallel 1 | threads 12 | reasoning off
    (+ KV cache q8_0 / temp 0.7 kept equal to production for comparability)

ONE-TIME SETUP (b11073 binary, separate path — download it yourself from
the llama.cpp b11073 release, asset llama-b11073-bin-win-cuda-13.3-x64.zip
matching your CUDA, and extract into bin\llama-server-b11073\; the production
bin\ stays as-is). The script checks the presence of
bin\llama-server-b11073\llama-server.exe before doing anything else.

Model resolution is IDENTICAL to production (same model — the A/B contract):
    config\llm_model.json (web UI selection) > LLAMA_MODEL_PATH env >
    models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf
===============================================================================
#>

param(
    [int]$Port = 8081,
    [int]$WaitSec = 180
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))

# --- experimental binary (SEPARATE path; production bin\ is never touched) ---
$LlamaBinDir = Join-Path $Root "bin\llama-server-b11073"
$Exe = Join-Path $LlamaBinDir "llama-server.exe"
if (-not (Test-Path -LiteralPath $Exe)) {
    Write-Host "HIBA: az experimentalis binaris nem letezik: $Exe" -ForegroundColor Red
    Write-Host "Letoltes: llama.cpp release b11073 asset 'llama-b11073-bin-win-cuda-13.3-x64.zip'" -ForegroundColor Yellow
    Write-Host "(vagy a sajat CUDA-dnak megfelelo valtozat), csomagold ki ide. A gyartasi" -ForegroundColor Yellow
    Write-Host "bin\llama-server.exe (b10717) NINCS erintve." -ForegroundColor Yellow
    exit 1
}

# --- model resolution (same order as the production starter) ----------------
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

# --- experimental profile (operator-tested baseline) --------------------------
$Ngl = "99"
if ($env:LLAMA_EXPERIMENTAL_NGL) { $Ngl = [string]$env:LLAMA_EXPERIMENTAL_NGL }
$Ctx = "16000"
if ($env:LLAMA_EXPERIMENTAL_CTX) { $Ctx = [string]$env:LLAMA_EXPERIMENTAL_CTX }
$Threads = "12"
if ($env:LLAMA_EXPERIMENTAL_THREADS) { $Threads = [string]$env:LLAMA_EXPERIMENTAL_THREADS }

$Host_ = "127.0.0.1"

Write-Host "=== EXPERIMENTAL llama-server (b11073) — EZ NEM A GYARTASI SZERVER ===" -ForegroundColor Magenta
Write-Host ("  binary : {0}" -f $Exe)
& $Exe --version 2>&1 | Select-Object -First 1 | ForEach-Object { Write-Host ("          {0}" -f $_) }
Write-Host ("  model  : {0}  ({1})" -f $Model, $ModelSource)
Write-Host ("  bind   : {0}:{1}   (a gyartasi 8080 NINCS erintve)" -f $Host_, $Port)
Write-Host ("  profil : ngl {0} | ctx {1} | parallel 1 | threads {2} | reasoning off | KV q8_0" -f $Ngl, $Ctx, $Threads)
Write-Host ""
Write-Host "Az alkalmazast igyanitsd ra (MEGLEVO env-feluliras, semmi kodvaltozas):" -ForegroundColor Cyan
Write-Host ('  $env:LLAMA_SERVER_HOST = "{0}"; $env:LLAMA_SERVER_PORT = "{1}"' -f $Host_, $Port) -ForegroundColor Cyan
Write-Host ""

function Test-Health {
    param([string]$H, [int]$P, [int]$T = 3)
    try {
        $R = Invoke-WebRequest -Uri ("http://{0}:{1}/health" -f $H, $P) -Method Get -TimeoutSec $T -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

# Idempotens inditas: ha mar fut EGY 8081-es experimentalis szerver, nem
# inditunk masodikat (a gyartasi :8080-t sosem piszaljuk).
if (Test-Health $Host_ $Port) {
    Write-Host "MAR FUT egy szerver a ${Host_}:${Port} cimen — nem inditok masodikat." -ForegroundColor Yellow
    exit 0
}

$ErrLog = Join-Path $Root "logs\llama-server-experimental.err.log"
$OutLog = Join-Path $Root "logs\llama-server-experimental.out.log"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

$Args = @(
    "--model", $Model,
    "--host", $Host_, "--port", "$Port",
    "-ngl", $Ngl, "-c", $Ctx, "--parallel", "1", "-t", $Threads,
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--temp", "0.7", "--reasoning", "off", "--metrics", "--no-webui", "--verbose"
)
Write-Host ("Inditas: {0} {1}" -f $Exe, ($Args -join " "))
Write-Host ("Logok  : {0} / {1}" -f $ErrLog, $OutLog)

$Proc = Start-Process -FilePath $Exe -ArgumentList $Args -WorkingDirectory $Root `
    -RedirectStandardError $ErrLog -RedirectStandardOutput $OutLog -PassThru

# --- /health varas (a gyartasi starterrel azonos elsodleges sikerfeltetel) ---
$Deadline = (Get-Date).AddSeconds($WaitSec)
while ((Get-Date) -lt $Deadline) {
    if ($Proc.HasExited) {
        Write-Host ("HIBA: az experimentalis llama-server kilepett (kod {0}). Log: {1}" -f $Proc.ExitCode, $ErrLog) -ForegroundColor Red
        exit 1
    }
    if (Test-Health $Host_ $Port) {
        Write-Host "OK: experimentalis szerver ELO a ${Host_}:${Port} cimen." -ForegroundColor Green
        Write-Host "Visszaallas (rollback): allitsd le, majd torold a LLAMA_SERVER_* env-valtozokat." -ForegroundColor Gray
        Write-Host "Ez az ablak nyitva tartja a folyamatot (Ctrl+C = leallitas)." -ForegroundColor Gray
        Wait-Process -Id $Proc.Id
        exit 0
    }
    Start-Sleep -Milliseconds 500
}
Write-Host "HIBA: az /health nem valt elerhetove ${WaitSec} mp alatt. Log: $ErrLog" -ForegroundColor Red
Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
exit 1
