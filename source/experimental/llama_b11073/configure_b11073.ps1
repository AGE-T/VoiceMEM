<#
===============================================================================
experimental/llama_b11073/configure_b11073.ps1 - CONFIGURATOR (enable/disable)

Enables or disables the EXPERIMENTAL llama.cpp b11073 runtime for VoiceMem
using an OPERATOR-LOCAL environment override only. Production config files
are NEVER modified:

    - config\llm_config.yaml      (canonical production profile)  UNTOUCHED
    - config\env.local.ps1        (production environment)         UNTOUCHED
    - MODELS.lock.json, VERSION, bin\                              UNTOUCHED

WHAT -Enable DOES (all inside experimental\llama_b11073\):
    1. verifies the b11073 runtime is installed (install_b11073.ps1 ran);
    2. writes the flag  config\experimental_enabled.json;
    3. GENERATES the operator-local app-session script
       experimental\llama_b11073\start_voicemem_experimental.ps1 which:
         - dot-sources the STANDARD config\env.local.ps1 first (offline
           mode, HF cache isolation, memory roots - the full standard env);
         - then overrides ONLY the LLM endpoint via the EXISTING env
           mechanism (app/config.py apply_env):
               LLAMA_SERVER_HOST = 127.0.0.1
               LLAMA_SERVER_PORT = 8081            (production stays 8080)
               OPENAI_BASE_URL   = http://127.0.0.1:8081/v1
             (the third one matters: the vendor bridge legs follow it, and
              it is also the guard variable of scripts\start_agent.ps1 -
              because it is set, start_agent will NOT re-source
              env.local.ps1, so the override SURVIVES the bootstrap);
         - starts the app through the EXISTING starter
           scripts\start_agent.ps1 -Web -NoServer   (-NoServer: the
           experimental llama-server is already running from its own
           launcher; production auto-start on 8080 is skipped).

WHAT -Disable DOES (the trivial rollback):
    1. stops the experimental llama-server (only processes whose executable
       lives under experimental\llama_b11073\bin\ - nothing else);
    2. removes the enable flag + the generated app-session script;
    3. prints the restore steps: the app targets 127.0.0.1:8080 again in
       any NEW terminal (no env override persists anywhere - the override
       existed only inside the generated session script).

USAGE:
    powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\configure_b11073.ps1 -Enable
    powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\configure_b11073.ps1 -Disable
    powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\configure_b11073.ps1          # status

EXIT CODES: 0 = ok | 1 = failure.

NOTE: deliberately ASCII (PowerShell 5.1 reads BOM-less files as Windows-1252).
===============================================================================
#>

param(
    [switch]$Enable,
    [switch]$Disable
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$Root      = (Get-Item -LiteralPath $ScriptDir).Parent.Parent.FullName
$BinDir    = Join-Path $ScriptDir "bin"
$ConfigDir = Join-Path $ScriptDir "config"
$FlagFile  = Join-Path $ConfigDir "experimental_enabled.json"
$AppScript = Join-Path $ScriptDir "start_voicemem_experimental.ps1"
$Exe       = Join-Path $BinDir "llama-server.exe"
$ProdBin   = Join-Path $Root "bin"

function Write-Ok([string]$m)   { Write-Host ("  [OK]    " + $m) -ForegroundColor Green }
function Write-Warn2([string]$m){ Write-Host ("  [WARN]  " + $m) -ForegroundColor Yellow }
function Write-Err2([string]$m) { Write-Host ("  [FAIL]  " + $m) -ForegroundColor Red }

function Test-Health([string]$H, [int]$P) {
    try {
        $R = Invoke-WebRequest -Uri ("http://{0}:{1}/health" -f $H, $P) -Method Get -TimeoutSec 2 -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

# ------------------------------- STATUS (default) -------------------------------
if (-not ($Enable -or $Disable)) {
    Write-Host "=== EXPERIMENTAL llama.cpp b11073 - ALLAPOT ===" -ForegroundColor Magenta
    if (Test-Path -LiteralPath $Exe) { Write-Ok ("runtime telepitve: " + $Exe) }
    else { Write-Warn2 ("runtime NINCS telepitve - futtasd: install_b11073.ps1") }
    if (Test-Path -LiteralPath $FlagFile) { Write-Ok "kiserlet ENGEDELYEZVE (experimental_enabled.json jelen)" }
    else { Write-Host "  [----]  kiserlet NINCS engedelyezve (gyartasi 8080 a default)" }
    if (Test-Health "127.0.0.1" 8081) { Write-Ok "experimentalis szerver ELO (127.0.0.1:8081/health OK)" }
    else { Write-Host "  [----]  experimentalis szerver nem fut" }
    if (Test-Health "127.0.0.1" 8080) { Write-Ok "gyartasi szerver ELO (127.0.0.1:8080/health OK)" }
    else { Write-Host "  [----]  gyartasi szerver nem fut" }
    exit 0
}

# ------------------------------- ENABLE -----------------------------------------
if ($Enable) {
    Write-Host "=== EXPERIMENTAL llama.cpp b11073 - ENGEDELYEZES ===" -ForegroundColor Magenta

    if (-not (Test-Path -LiteralPath $Exe)) {
        Write-Err2 ("a b11073 runtime nincs telepitve: " + $Exe)
        Write-Host "JAVITAS: eloszor futtasd a telepitot:" -ForegroundColor Yellow
        Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\install_b11073.ps1" -ForegroundColor Yellow
        exit 1
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Root "config\env.local.ps1"))) {
        Write-Err2 "nem VoiceMemAgent gyokerben fut a szkript (config\env.local.ps1 hianyzik)"
        exit 1
    }

    New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null

    # 1) enable flag (operator-local state, NOT a production config file)
    $Flag = [ordered]@{
        enabled_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        endpoint_host  = "127.0.0.1"
        endpoint_port  = 8081
        note           = "operator-local override only; production config files untouched; rollback = configure_b11073.ps1 -Disable"
    }
    $Flag | ConvertTo-Json | Set-Content -LiteralPath $FlagFile -Encoding ASCII
    Write-Ok ("flag: " + $FlagFile)

    # 2) the generated operator-local app-session script (the WHOLE override)
    $Gen = @'
# ==============================================================================
# AUTO-GENERATED by configure_b11073.ps1 -Enable  (experimental llama.cpp b11073)
# Operator-local app session: the VoiceMem application targets the EXPERIMENTAL
# llama-server on 127.0.0.1:8081 for THIS session only. No production config
# file is modified. Rollback: close this window + configure_b11073.ps1 -Disable.
# ==============================================================================
$VmRoot = "<VMROOT>"

# 1) the FULL standard environment first (offline mode, HF cache isolation,
#    memory roots, telemetry opt-outs) - exactly what the normal flow loads:
. (Join-Path (Join-Path $VmRoot "config") "env.local.ps1")

# 2) the EXPERIMENTAL endpoint override - the EXISTING env mechanism
#    (app/config.py apply_env honours HOST/PORT; the vendor bridge legs follow
#    OPENAI_BASE_URL; setting it also stops scripts\start_agent.ps1 from
#    re-sourcing env.local.ps1, so this override survives the bootstrap):
$env:LLAMA_SERVER_HOST = "127.0.0.1"
$env:LLAMA_SERVER_PORT = "8081"                  # a gyartas tovabbra is 8080
$env:OPENAI_BASE_URL   = "http://127.0.0.1:8081/v1"

Write-Host ""
Write-Host "=== VOICEMEM - EXPERIMENTALIS LLM SZEZION (b11073 @ 127.0.0.1:8081) ===" -ForegroundColor Magenta
Write-Host "Elobetel: az experimentalis llama-servernek futnia kell!"
Write-Host "  experimental\llama_b11073\start_llama_server_experimental.ps1" -ForegroundColor Cyan
Write-Host "Visszaallas: uj terminal + configure_b11073.ps1 -Disable" -ForegroundColor Gray
Write-Host ""

# 3) the app through the EXISTING starter (-NoServer: the experimental server
#    is already running from its own launcher; the production auto-start on
#    8080 is deliberately skipped for this session):
$PsExe = (Get-Process -Id $PID).Path
& $PsExe -NoProfile -ExecutionPolicy Bypass -File (Join-Path (Join-Path $VmRoot "scripts") "start_agent.ps1") -Web -NoServer
exit $LASTEXITCODE
'@
    $Gen = $Gen.Replace("<VMROOT>", $Root)
    [System.IO.File]::WriteAllText($AppScript, $Gen.Replace("`n", "`r`n"))
    Write-Ok ("generalva: " + $AppScript)

    Write-Host ""
    Write-Host "ENGEDELYEZVE. Inditasi sorrend:" -ForegroundColor Green
    Write-Host "  1. experimental\llama_b11073\start_llama_server_experimental.ps1   (b11073 a 8081-en)"
    Write-Host "  2. experimental\llama_b11073\start_voicemem_experimental.ps1      (az app erre a szerverre nez)"
    Write-Host "     (a normal START.bat folyamat NEM valtozik - az tovabbra is a 8080-as gyartasi szerverre nezet.)"
    Write-Host "Visszaallas: configure_b11073.ps1 -Disable"
    exit 0
}

# ------------------------------- DISABLE (rollback) ------------------------------
if ($Disable) {
    Write-Host "=== EXPERIMENTAL llama.cpp b11073 - VISSZAALLITAS (disable) ===" -ForegroundColor Magenta

    # 1) stop the experimental server - ONLY processes whose exe lives in
    #    experimental\llama_b11073\bin\ (never anything else):
    $BinPrefix = $BinDir.ToLower() + [System.IO.Path]::DirectorySeparatorChar
    $Stopped = 0
    foreach ($Proc in (Get-Process -Name "llama-server" -ErrorAction SilentlyContinue)) {
        try { $PPath = [string]$Proc.Path } catch { $PPath = "" }
        if ($PPath -and $PPath.ToLower().StartsWith($BinPrefix)) {
            Stop-Process -Id $Proc.Id -Force
            Write-Ok ("leallitva: PID " + $Proc.Id + " (" + $PPath + ")")
            $Stopped++
        }
    }
    if ($Stopped -eq 0) {
        if (Test-Health "127.0.0.1" 8081) {
            Write-Warn2 "valami ELO van a 8081-en, de nem az experimentalis binaris utvonalabol - nem allitom le."
            Write-Host "          (nezd meg: netstat -ano | findstr :8081)" -ForegroundColor Yellow
        } else {
            Write-Host "  [----]  az experimentalis szerver nem futott"
        }
    }

    # 2) remove the enable flag + the generated app-session script
    foreach ($F in @($FlagFile, $AppScript)) {
        if (Test-Path -LiteralPath $F) {
            Remove-Item -LiteralPath $F -Force
            Write-Ok ("torolve: " + $F)
        }
    }

    # 3) restore report
    Write-Host ""
    Write-Host "VISSZAALLITVA. A gyartasi mukodes helyreallt:" -ForegroundColor Green
    Write-Host "  - az app uj terminalban a 127.0.0.1:8080 gyartasi szerverre nez (env-feluliras sehol sem perzisztalodott);"
    Write-Host "  - a normal START.bat folyamat vegig valtozatlan volt;"
    Write-Host "  - a gyartasi bin\llama-server.exe (b10717), a CUDA DLL-ek, config\llm_config.yaml,"
    Write-Host "    config\env.local.ps1, MODELS.lock.json es a VERSION sosem valtozott."
    Write-Host "  Opcionalis teljes takaritas: torold az experimental\llama_b11073\ mappat."
    if (Test-Health "127.0.0.1" 8080) { Write-Ok "a gyartasi szerver ELO (127.0.0.1:8080/health OK)" }
    else { Write-Host "  [----]  a gyartasi szerver most nem fut (inditsd START.bat-tal)" }
    exit 0
}
