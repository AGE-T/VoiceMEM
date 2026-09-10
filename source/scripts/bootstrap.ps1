<#
===============================================================================
scripts/bootstrap.ps1 - ONE-CLICK bootstrap orchestrator (Task 13, M0.2)

USER-FACING CONTRACT (the one-click rule):
  The user double-clicks START.bat, and START.bat calls THIS script with a
  mode. The user never runs any other script manually, never creates a venv,
  never runs pip, never installs the Hugging Face CLI, never downloads a
  model, never edits config, never sets environment variables manually.

MODES (-Mode):
  run        (default) ensure environment, then start the voice agent
  mock       ensure environment, then run the scripted 3-turn demo
  check      ensure environment, then full verification (verify + tests)
  benchmark  ensure environment, then run the latency benchmark suite
  repair     force re-verification + automatic repair of the environment
  build      versioned release ZIP via scripts\build_release.ps1
             (the release test gate runs BEFORE the ZIP is created)

WHAT IT DOES (first run - the full one-click flow):
  1.  resolve the project root from its own location (the project is
      relocatable - any drive, any folder, no hardcoded paths)
  2.  find Python 3.11 (search order: py -3.11, python3.11, python);
      verify 3.11.x AND 64-bit; clean error otherwise (no tracebacks)
  3.  fast probes: .venv health, dependency imports, HF CLI tooling,
      bin\llama-server.exe + bin\piper.exe, MODELS.lock.json presence
      check, config\voicemem_config.yaml + .env
  4.  if anything is missing (or first run, or -Mode repair): run the
      idempotent 18-step installer scripts\install_m1.ps1 as a CHILD
      process - it creates .venv, installs dependencies, installs the
      Hugging Face tooling, downloads the M1 model set, writes config,
      runs the smoke tests and the manifest. Missing parts are repaired
      automatically; ready parts are skipped (never re-downloaded).
  5.  dispatch the requested mode (see MODES)
  6.  print the final install summary and start the agent

IDEMPOTENCY (.install_state.json):
  Every step updates .install_state.json (python_ready, venv_ready,
  dependencies_ready, hf_tooling_ready, binaries_ready, models_ready,
  config_ready, manifest_ready, smoke_tests_passed, ...). A second or
  third START.bat run skips the installer entirely ("environment already
  ready"), verifies, and starts the agent. An interrupted install is
  resumed by simply running START.bat again.

ERROR UX:
  Primary errors are short, clean, actionable ([ERROR] + Detected +
  Required + next step). Python tracebacks and technical detail go to
  logs\bootstrap.log, never to the primary error output.

LOG (logs\bootstrap.log):
  every run logs timestamp, mode, OS, Python version, GPU, probe results,
  installer runs, test results, errors/warnings and the final status.

CHILD PROCESSES: install_m1.ps1 / start_agent.ps1 / verify_m1.ps1 /
run_tests.ps1 / build_release.ps1 always run as child PowerShell processes
(the SAME 64-bit engine that runs this bootstrap - self exe first, then the
native System32 Windows PowerShell, then PATH pwsh) so an exit inside them
cannot kill the bootstrap, and exit codes propagate cleanly.

64-BIT RULE: START.bat resolves a 64-bit engine (64-bit PowerShell 7
first, then the native System32 Windows PowerShell - the generic
'powershell' PATH lookup is never used). THIS script enforces the rule
with an [Environment]::Is64BitProcess guard and refuses to run in a
32-bit process: WOW64 file redirection would silently corrupt the whole
install chain (System32 -> SysWOW64 redirection, 32-bit Python discovery,
GPU tool detection).

NOTE: this file is intentionally pure ASCII - PowerShell 5.1 without a BOM
would read accented characters as Windows-1252.

Hasznalat (fejlesztoi):
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Mode mock
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Mode repair
#>
param(
    [ValidateSet('run', 'cli', 'mock', 'check', 'benchmark', 'repair', 'build')]
    [string]$Mode = 'run',
    [string]$Root = '',
    [string]$Notes = '',
    [string]$Bump = '',
    [switch]$StrictValidation
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

# ------------------------------------------------------------------ utf-8 ---
# v0.3.4 (field report #4): the python children write their stdout through
# PIPES (captured by this script and its children) - Python then encodes
# with the ANSI code page (cp1252 on the target machine), and the Hungarian
# text (u+0171 etc. in the --check checklist) crashed with
# UnicodeEncodeError before a single line could print. PYTHONIOENCODING
# sets ONLY the stdio encoding (deliberately NOT PYTHONUTF8: file-I/O
# defaults stay locale-based), and the shared console goes UTF-8 so
# PowerShell decodes the python output correctly.
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# ------------------------------------------------------------------ paths ---

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
$RepoRoot     = $Root
$VenvDir      = Join-Path $Root '.venv'
$VenvPython   = Join-Path $Root '.venv\Scripts\python.exe'
$StateFile    = Join-Path $Root '.install_state.json'
$LogFile      = Join-Path $Root 'logs\bootstrap.log'
$ModelsLock   = Join-Path $Root 'MODELS.lock.json'
$ManifestFile = Join-Path $Root 'INSTALL_MANIFEST.json'
$YamlFile     = Join-Path $Root 'config\voicemem_config.yaml'
$EnvExample   = Join-Path $Root 'config\.env.example'
$EnvFile      = Join-Path $Root 'config\.env'
$LlamaExe     = Join-Path $Root 'bin\llama-server.exe'
$PiperExe     = Join-Path $Root 'bin\piper.exe'
$Installer    = Join-Path $Root 'scripts\install_m1.ps1'
$StartAgent   = Join-Path $Root 'scripts\start_agent.ps1'
$VerifyM1     = Join-Path $Root 'scripts\verify_m1.ps1'
$RunTests     = Join-Path $Root 'scripts\run_tests.ps1'
$BuildRelease = Join-Path $Root 'scripts\build_release.ps1'

# ------------------------------------------------- MOTW / zone mark ----
# v0.3.2: remove the Mark-of-the-Web (Zone.Identifier) from all repo
# scripts. A release ZIP downloaded with a browser and extracted with
# Explorer propagates the internet zone mark to every extracted file;
# a plain RemoteSigned session then blocks the .ps1 files with "not
# digitally signed". The START.bat chain already runs everything with
# -ExecutionPolicy Bypass, but MANUAL script runs (README examples)
# hit the block - remove the mark here, once, on every entry path.
# (The same block lives in install_m1.ps1 + verify_m1.ps1.)
Get-ChildItem -Path (Join-Path $Root 'scripts') -Filter '*.ps1' -File -ErrorAction SilentlyContinue |
    Unblock-File -ErrorAction SilentlyContinue

# --------------------------------------------------------- 64-bit guard ----
# START.bat resolves a 64-bit PowerShell engine (PowerShell 7 first, then
# the native System32 Windows PowerShell). This is the enforcement point:
# a 32-bit engine would silently corrupt the whole install chain (WOW64
# System32->SysWOW64 file redirection, 32-bit Python discovery, GPU tool
# detection), so 32-bit is EXPLICITLY forbidden here as well.
if (-not [Environment]::Is64BitProcess) {
    $SelfPath = ''
    try { $SelfPath = (Get-Process -Id $PID).Path } catch { }
    try {
        $LogDir = Join-Path $Root 'logs'
        if (-not (Test-Path $LogDir)) {
            New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        }
        Add-Content -Path (Join-Path $LogDir 'bootstrap.log') -Value ('[{0}] FATAL: 32-bit PowerShell process refused ({1}) - a 64-bit engine is required.' -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $SelfPath) -ErrorAction SilentlyContinue
    } catch { }
    Write-Host ''
    Write-Host '[ERROR] VoiceMem Agent must run in a 64-bit PowerShell process.' -ForegroundColor Red
    Write-Host '        Detected : ' $SelfPath ' (32-bit engine)'
    Write-Host '        Required : 64-bit Windows PowerShell 5.1 or PowerShell 7+.'
    Write-Host '        Next step: double-click START.bat again - it always resolves a 64-bit engine.'
    exit 2
}

# Robust GPU detection library (PATH -> standard locations -> exec ->
# query -> WMI fallback + diagnostics); shared with install_m1.ps1.
. (Join-Path $Root 'scripts\gpu_check.ps1')

if (-not (Test-Path (Join-Path $Root 'app'))) {
    Write-Host '[ERROR] The project root is invalid (app\ folder not found):' $Root
    Write-Host '        Extract or clone the FULL project, then double-click START.bat again.'
    exit 1
}

# logs\ must exist before the first Write-BLog (a fresh ZIP extract has no
# logs directory yet - the installer only creates it later, in its step 6).
if (-not (Test-Path (Join-Path $Root 'logs'))) {
    New-Item -ItemType Directory -Path (Join-Path $Root 'logs') -Force | Out-Null
}

# ----------------------------------------------------------- console/log ---

function Write-BLog {
    param([string]$Level, [string]$Message)
    $Stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    try {
        Add-Content -Path $LogFile -Value ("{0} [{1}] {2}" -f $Stamp, $Level, $Message) -Encoding UTF8
    } catch { }
}

function Write-BStep { param([string]$Msg) Write-Host '' ; Write-Host ("== {0}" -f $Msg) -ForegroundColor Cyan; Write-BLog 'INFO' $Msg }
function Write-BOk   { param([string]$Msg) Write-Host ("   [OK]    {0}" -f $Msg) -ForegroundColor Green; Write-BLog 'OK' $Msg }
function Write-BWarn { param([string]$Msg) Write-Host ("   [WARN]  {0}" -f $Msg) -ForegroundColor Yellow; Write-BLog 'WARN' $Msg }
function Write-BErr  { param([string]$Msg) Write-Host ("   [ERROR] {0}" -f $Msg) -ForegroundColor Red; Write-BLog 'ERROR' $Msg }

function Fail-Bootstrap {
    param([string]$Message, [string]$Detected = '', [string]$Required = '', [string]$NextStep = '')
    Write-Host ''
    Write-Host '[ERROR]' $Message -ForegroundColor Red
    if ($Detected -ne '') {
        Write-Host ''
        Write-Host 'Detected:'
        Write-Host ("    {0}" -f $Detected)
    }
    if ($Required -ne '') {
        Write-Host ''
        Write-Host 'Required:'
        Write-Host ("    {0}" -f $Required)
    }
    if ($NextStep -eq '') { $NextStep = 'Fix the problem above, then double-click START.bat again.' }
    Write-Host ''
    Write-Host $NextStep
    Write-Host 'Technical details: logs\bootstrap.log'
    Write-Host ''
    Write-BLog 'ERROR' ("bootstrap failed: {0} | detected: {1}" -f $Message, $Detected)
    try { Set-StateFlag 'last_result' ('error: ' + $Message) } catch { }
    exit 1
}

# ------------------------------------------------------------------ state ---

function New-DefaultState {
    $State = [ordered]@{
        schema_version       = 1
        version              = ''
        python_ready         = $false
        venv_ready           = $false
        dependencies_ready   = $false
        hf_tooling_ready     = $false
        binaries_ready       = $false
        models_ready         = $false
        config_ready         = $false
        manifest_ready       = $false
        smoke_tests_passed   = $false
        voicemem_ready       = $false
        python_version       = ''
        python_exe           = ''
        gpu_name             = ''
        cuda_ready           = $false
        runs_count           = 0
        last_run_at_utc      = ''
        last_mode            = ''
        last_result          = ''
    }
    return $State
}

function Read-InstallState {
    $State = New-DefaultState
    if (Test-Path $StateFile) {
        try {
            $Parsed = Get-Content -Path $StateFile -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($Prop in $Parsed.PSObject.Properties) {
                if ($State.Contains($Prop.Name)) { $State[$Prop.Name] = $Prop.Value }
            }
        } catch {
            Write-BLog 'WARN' ('.install_state.json is unreadable - using defaults. Detail: ' + $_.Exception.Message)
        }
    }
    return $State
}

function Save-InstallState {
    param($State)
    try {
        $Json = $State | ConvertTo-Json -Depth 4
        [IO.File]::WriteAllText($StateFile, $Json + "`n", (New-Object System.Text.UTF8Encoding($false)))
    } catch {
        Write-BLog 'WARN' ('could not write .install_state.json: ' + $_.Exception.Message)
    }
}

function Set-StateFlag {
    param([string]$Name, $Value)
    $Script:State[$Name] = $Value
    Save-InstallState $Script:State
}

function Read-StateFlag {
    param([string]$Name)
    return $Script:State[$Name]
}

# ---------------------------------------------------------- child engine ---

function Get-ChildPs {
    # Prefer the CURRENT engine (bootstrap itself passed the 64-bit guard
    # above, so children inherit the verified bitness); then the native
    # System32 Windows PowerShell; then PATH pwsh. The generic PATH
    # powershell.exe lookup is the LAST resort only (from a verified
    # 64-bit process PATH resolution is 64-bit as well).
    $SelfExe = ''
    try { $SelfExe = (Get-Process -Id $PID).Path } catch { }
    if ($SelfExe -and (Test-Path $SelfExe)) { return $SelfExe }
    $Native = ''
    if ($env:windir) {
        $Native = Join-Path $env:windir 'System32\WindowsPowerShell\v1.0\powershell.exe'
    }
    if ($Native -and (Test-Path $Native)) { return $Native }
    if (Get-Command pwsh -ErrorAction SilentlyContinue) { return 'pwsh' }
    if (Get-Command powershell.exe -ErrorAction SilentlyContinue) { return 'powershell.exe' }
    return ''
}

function Invoke-ChildScript {
    param([string]$ScriptPath, [string[]]$ScriptArgs = @())
    if (-not (Test-Path $ScriptPath)) {
        Fail-Bootstrap ('internal script missing: ' + $ScriptPath) '' '' 'Re-extract or re-clone the full project, then double-click START.bat again.'
    }
    $Ps = Get-ChildPs
    if ($Ps -eq '') {
        Fail-Bootstrap 'No PowerShell engine found (powershell.exe / pwsh).'
    }
    $AllArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $ScriptPath) + @($ScriptArgs)
    # Out-Host keeps the child's progress on the user's console while keeping
    # the function's return value clean (just the child's exit code).
    & $Ps @AllArgs | Out-Host
    return $LASTEXITCODE
}

# ------------------------------------------------------------- probes ------

function Find-ProjectPython {
    # Search order (user spec): py.exe -3.11, python3.11, python.
    # Returns @{ Ok; Exe; Extra; VersionText; Source; Bitness; SeenVersion }
    # - never installs anything.
    $Found = @{ Ok = $false; Exe = ''; Extra = @(); VersionText = ''; Source = ''; Bitness = 0; SeenVersion = '' }
    $Candidates = @(
        @{ Exe = 'py';         Extra = @('-3.11'); Source = 'py.exe -3.11' },
        @{ Exe = 'python3.11'; Extra = @();        Source = 'python3.11' },
        @{ Exe = 'python';     Extra = @();        Source = 'python' }
    )
    $SeenVersion = ''
    foreach ($Cand in $Candidates) {
        $Exe = $Cand.Exe
        if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) { continue }
        $VersionText = ''
        try {
            $Out = & $Exe @($Cand.Extra) --version 2>$null
            if ($LASTEXITCODE -eq 0) { $VersionText = ((@($Out) -join ' ')).Trim() }
        } catch { }
        if ([string]::IsNullOrWhiteSpace($VersionText)) { continue }
        if ($VersionText -match 'Python 3\.11\.\d+') {
            $Found = @{
                Ok = $true; Exe = $Exe; Extra = @($Cand.Extra)
                VersionText = $VersionText; Source = $Cand.Source
                Bitness = 0; SeenVersion = $SeenVersion
            }
            break
        }
        if ($SeenVersion -eq '') { $SeenVersion = $VersionText }
    }
    if (-not $Found.Ok) {
        $Found.SeenVersion = $SeenVersion
        return $Found
    }
    # 64-bit check (a 32-bit 3.11 is NOT usable for this project).
    # The probe code MUST NOT contain embedded double quotes: Windows
    # PowerShell 5.1 does not escape them when passing arguments to native
    # executables, so Python would receive calcsize(P) with the quotes
    # stripped -> NameError -> exit 1 -> a false "0 bit" failure. The
    # expressions below are therefore quote-free.
    $Found.Bitness = 0
    try {
        $Bits = & $Found.Exe @($Found.Extra) -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $Bits) {
            $BitsText = ((@($Bits) -join ' ')).Trim()
            if ($BitsText -match '^(\d+)$') { $Found.Bitness = [int]$Matches[1] }
        }
    } catch { }
    if ($Found.Bitness -eq 0) {
        # secondary probe (also quote-free) if the first one failed.
        try {
            $Arch = & $Found.Exe @($Found.Extra) -c 'import platform; print(platform.architecture()[0])' 2>$null
            if ($LASTEXITCODE -eq 0 -and $Arch) {
                $ArchText = ((@($Arch) -join ' ')).Trim()
                if ($ArchText -match '^(\d+)bit$') { $Found.Bitness = [int]$Matches[1] }
            }
        } catch { }
    }
    if ($Found.Bitness -ne 64) {
        $BitLabel = 'unknown'
        if ($Found.Bitness -gt 0) { $BitLabel = [string]$Found.Bitness }
        Fail-Bootstrap '64-bit Python 3.11 is required.' `
            ('{0} ({1} bit)' -f $Found.VersionText, $BitLabel) `
            '64-bit Python 3.11.x' `
            'Please install the 64-bit Python 3.11.x installer from python.org and run START.bat again.'
    }
    return $Found
}

function Test-VenvReady {
    if (-not (Test-Path $VenvPython)) { return $false }
    try {
        $Out = & $VenvPython --version 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $Text = ((@($Out) -join ' ')).Trim()
        return ($Text -match 'Python 3\.11\.')
    } catch { return $false }
}

function Test-DepsReady {
    # Import probe of the KEY runtime packages (torch + onnxruntime
    # included: a missing heavy dependency must route to the installer, not
    # fail at agent start). v0.3.4 (field report #4): onnxruntime MUST be
    # probed here - NO other dependency path installs it (voicemem pulls
    # sherpa-onnx, not onnxruntime), so without this probe a venv that is
    # missing it passes as "ready" and the agent dies at SileroVad init.
    # v0.4.0 (web UI): fastapi + uvicorn + websockets are probed too - the
    # LOCAL web backend (app/web_server.py, port 8787) dies at import time
    # without them; an existing venv self-repairs via one START.bat click.
    if (-not (Test-Path $VenvPython)) { return $false }
    $Probe = 'import yaml, numpy, httpx, soundfile, torch, onnxruntime, fastapi, uvicorn, websockets'
    # try/catch: with $ErrorActionPreference='Stop' a native command writing
    # to a redirected stderr stream can raise NativeCommandError in PS 5.1.
    try { & $VenvPython -c $Probe *> $null } catch { }
    if ($LASTEXITCODE -ne 0) {
        # capture the detail once for the log (never the primary message)
        $Detail = ''
        try { $Detail = ((& $VenvPython -c $Probe 2>&1 | Out-String)) } catch { }
        if (-not [string]::IsNullOrWhiteSpace($Detail)) {
            Write-BLog 'WARN' ('dependency import probe failed: ' + $Detail.Trim())
        }
        return $false
    }
    # v0.4.7: transformers import + VERSION floor in one
    # probe. The PyPI voicemem package pins transformers==4.52.3, so an
    # in-place upgraded venv keeps that version FOREVER: an import-only
    # probe passed ("dependencies importable"), the bootstrap SKIPPED the
    # installer, and step 16 (the v0.4.7 transformers>=5.0 guard) never
    # ran -> transformers lacks the native qwen3_asr module -> "ASR not
    # green", voice mode unavailable, text-only conversation. Exit code 5
    # = importable but older than the floor -> force the installer ->
    # step 16 upgrades it -> ASR loads again (self-healing START.bat).
    # No embedded double quotes (PS 5.1 native-argument quoting); the ''
    # pairs are escaped single quotes for the Python string literals.
    $TfProbe = 'import sys, transformers; v = tuple(int(x) for x in transformers.__version__.split(''+'')[0].split(''.'')[:2]); sys.exit(0 if v >= (5, 0) else 5)'
    try { & $VenvPython -c $TfProbe *> $null } catch { }
    if ($LASTEXITCODE -eq 0) { return $true }
    if ($LASTEXITCODE -eq 5) {
        $TfVersion = 'unknown'
        try { $TfVersion = ((& $VenvPython -c 'import transformers; print(transformers.__version__)' 2>$null | Out-String)).Trim() } catch { }
        if ([string]::IsNullOrWhiteSpace($TfVersion)) { $TfVersion = 'unknown' }
        Write-BLog 'WARN' ('transformers ' + $TfVersion + ' < 5.0 in the venv - Qwen3-ASR (qwen3_asr) has no native module with it; installer step 16 will upgrade it automatically')
        Write-BWarn ('transformers ' + $TfVersion + ' < 5.0 (Qwen3-ASR floor) -> will be upgraded automatically (ASR repair)')
    } else {
        Write-BLog 'WARN' 'transformers import/version probe failed - installer step 16 will repair it'
        Write-BWarn 'transformers missing or broken -> will be installed automatically'
    }
    return $false
}

function Test-HfToolingReady {
    # v0.1.7: a modellletoltes a huggingface_hub KONYVTARAT hasznalja
    # (Python API: hf_hub_download / snapshot_download) - a Hugging Face
    # CLI mar NEM runtime dependency (a CLI emoji-t nyomott ki, CP1252
    # konzolon UnicodeEncodeError - v0.1.6 release blocker). Ezert itt
    # NEM CLI exe-t, hanem a venv-ben importalhato konyvtarat keressuk.
    if (-not (Test-Path $VenvPython)) { return $false }
    try { & $VenvPython -c 'import huggingface_hub' *> $null } catch { }
    return ($LASTEXITCODE -eq 0)
}

function Test-BinariesReady {
    return ((Test-Path $LlamaExe) -and (Test-Path $PiperExe))
}

function Test-ModelsPresent {
    # MODELS.lock.json-driven presence check (never downloads anything).
    # Returns @{ Ok; Missing; Present }.
    $Result = @{ Ok = $false; Missing = @(); Present = @() }
    if (-not (Test-Path $ModelsLock)) {
        $Result.Missing = @('MODELS.lock.json')
        return $Result
    }
    try {
        $Lock = Get-Content -Path $ModelsLock -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        $Result.Missing = @('MODELS.lock.json (invalid JSON)')
        return $Result
    }
    foreach ($M in @($Lock.models)) {
        $Component = [string]$M.component
        if ([string]::IsNullOrWhiteSpace($Component)) { $Component = [string]$M.repo }
        $Dir = Join-Path $Root ([string]$M.target_dir)
        $EntryOk = $true
        if ([bool]$M.snapshot) {
            $ProbePath = Join-Path $Dir ([string]$M.snapshot_probe)
            if (-not (Test-Path -LiteralPath $ProbePath)) { $EntryOk = $false }
            if ($EntryOk) {
                $Total = 0
                if (Test-Path $Dir) {
                    $Sum = (Get-ChildItem -Path $Dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
                    if ($Sum) { $Total = [long]$Sum }
                }
                $MinBytes = [long]$M.min_total_mb * 1MB
                if ($Total -lt $MinBytes) { $EntryOk = $false }
            }
        } else {
            $MinMap = @{}
            if ($M.min_bytes) {
                foreach ($P in $M.min_bytes.PSObject.Properties) { $MinMap[$P.Name] = [long]$P.Value }
            }
            foreach ($F in @($M.files)) {
                $FilePath = Join-Path $Dir ([string]$F)
                if (-not (Test-Path -LiteralPath $FilePath)) { $EntryOk = $false; break }
                $Len = (Get-Item -LiteralPath $FilePath).Length
                $Min = [long]1
                if ($MinMap.ContainsKey([string]$F)) { $Min = [long]$MinMap[[string]$F] }
                if ($Len -lt $Min) { $EntryOk = $false; break }
            }
        }
        if ($EntryOk) { $Result.Present += $Component } else { $Result.Missing += $Component }
    }
    $Result.Ok = ($Result.Missing.Count -eq 0)
    return $Result
}

function Test-ConfigReady {
    return ((Test-Path $YamlFile) -and (Test-Path $EnvFile))
}

function Test-LlamaHealth {
    try {
        $R = Invoke-WebRequest -Uri 'http://127.0.0.1:8080/health' -Method Get -TimeoutSec 2 -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

# ---------------------------------------------------------------- summary ---

function Write-FinalSummary {
    param([string]$Closing = 'Starting VoiceMem Agent...')
    $PyVer = [string](Read-StateFlag 'python_version')
    if ($PyVer -eq '') { $PyVer = '3.11.x' }
    $GpuName = [string](Read-StateFlag 'gpu_name')
    if ([string]::IsNullOrWhiteSpace($GpuName)) { $GpuName = 'not detected' }
    $Cuda = 'READY'
    if (-not (Read-StateFlag 'cuda_ready')) { $Cuda = 'not verified' }
    $VoiceMem = 'READY'
    if (-not (Read-StateFlag 'voicemem_ready')) { $VoiceMem = 'NOT INSTALLED (degraded mode - memory context is empty)' }
    Write-Host ''
    Write-Host '========================================' -ForegroundColor Green
    Write-Host 'VoiceMem Agent'
    Write-Host 'Installation complete'
    Write-Host '========================================' -ForegroundColor Green
    Write-Host ''
    Write-Host 'Project:'
    Write-Host ("    {0}" -f $Root)
    Write-Host 'Python:'
    Write-Host ("    {0}" -f $PyVer)
    Write-Host 'Virtual environment:'
    Write-Host '    READY'
    Write-Host 'GPU:'
    Write-Host ("    {0}" -f $GpuName)
    Write-Host 'CUDA:'
    Write-Host ("    {0}" -f $Cuda)
    Write-Host 'Dependencies:'
    Write-Host '    READY'
    Write-Host 'Models:'
    Write-Host '    READY'
    Write-Host 'VoiceMem:'
    Write-Host ("    {0}" -f $VoiceMem)
    Write-Host 'ASR:'
    Write-Host '    Qwen3 ASR 0.6B'
    Write-Host 'LLM:'
    Write-Host '    Qwen3.6 35B A3B IQ4_XS (az EGYETLEN LLM - nincs visszaesi profil;'
    Write-Host '    a GGUF barmelyik meghajton: web UI "LLM model" picker / find_qwen_gguf.ps1)'
    Write-Host 'TTS:'
    Write-Host '    Piper HU + EN'
    Write-Host 'Offline runtime:'
    Write-Host '    READY'
    Write-Host 'Smoke tests:'
    Write-Host '    PASS'
    Write-Host ''
    Write-Host '========================================' -ForegroundColor Green
    Write-Host $Closing
    Write-Host '========================================' -ForegroundColor Green
}

# ------------------------------------------------------------------- main ---

$ExitCode = 1
$Script:State = Read-InstallState

try {
    $RunId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    Write-BLog 'INFO' ("==============================================================")
    Write-BLog 'INFO' ("bootstrap run {0} | mode: {1}" -f $RunId, $Mode)

    Write-Host ''
    Write-Host '==================================================================='
    Write-Host ' VoiceMem Agent - one-click bootstrap'
    Write-Host ('  mode : ' + $Mode)
    Write-Host ('  root : ' + $Root)
    Write-Host '==================================================================='

    # OS + GPU info into the log (spec: the bootstrap log records these).
    try {
        $OsVersion = [System.Environment]::OSVersion.Version
        Write-BLog 'INFO' ('OS: ' + [System.Environment]::OSVersion.ToString())
    } catch { }
    $GpuName = ''
    $GpuLogged = $false
    try {
        $SmiProbe = Find-NvidiaSmi
        if ($SmiProbe.Found) {
            $GpuQ = Invoke-NvidiaSmiQuery -SmiPath ([string]$SmiProbe.Path)
            if ($GpuQ.Ok) {
                $GpuName = [string]$GpuQ.Name
                Write-BLog 'INFO' ('GPU: ' + $GpuQ.Name + ' | driver ' + $GpuQ.Driver + ' | VRAM ' + $GpuQ.VramMiB + ' MiB | nvidia-smi via ' + $SmiProbe.Origin + ' -> ' + $SmiProbe.Path)
                $GpuLogged = $true
            } else {
                Write-BLog 'WARN' ('GPU: nvidia-smi found (' + $SmiProbe.Path + ') but query failed (' + $GpuQ.Code + ') - the installer step 4 runs the full validation')
                $GpuLogged = $true
            }
        }
    } catch { }
    if (-not $GpuLogged) {
        Write-BLog 'WARN' 'GPU: not detected via nvidia-smi (PATH + standard locations) - the installer step 4 runs the WMI fallback + full diagnostics'
    }
    if ($GpuName -ne '') {
        Set-StateFlag 'gpu_name' $GpuName
    }

    # ---- Step 1: Python 3.11 check (always, clean failure) -----------------
    Write-BStep 'Step 1/5: Python 3.11 check'
    $Py = Find-ProjectPython
    if (-not $Py.Ok) {
        $Seen = $Py.SeenVersion
        if ([string]::IsNullOrWhiteSpace($Seen)) { $Seen = 'no Python found on PATH' }
        Fail-Bootstrap 'Python 3.11 is required.' $Seen 'Python 3.11.x (64-bit)' 'Please install Python 3.11.x from python.org and run START.bat again.'
    }
    Write-BOk ('{0} ({1}, 64-bit)' -f $Py.VersionText, $Py.Source)
    Set-StateFlag 'python_ready' $true
    Set-StateFlag 'python_version' $Py.VersionText
    $PyExeText = (@($Py.Exe) + @($Py.Extra)) -join ' '
    Set-StateFlag 'python_exe' $PyExeText

    # ---- Steps 2-3: fast probes (idempotency) ------------------------------
    Write-BStep 'Step 2/5: environment probes (idempotency - ready parts are skipped)'
    $VenvOk = Test-VenvReady
    $DepsOk = $false
    $HfOk = $false
    $BinsOk = $false
    if ($VenvOk) {
        $DepsOk = Test-DepsReady
        $HfOk = Test-HfToolingReady
        $BinsOk = Test-BinariesReady
    }
    $Models = Test-ModelsPresent
    $ConfigOk = Test-ConfigReady

    Set-StateFlag 'venv_ready' $VenvOk
    Set-StateFlag 'dependencies_ready' $DepsOk
    Set-StateFlag 'hf_tooling_ready' $HfOk
    Set-StateFlag 'binaries_ready' $BinsOk
    Set-StateFlag 'models_ready' $Models.Ok
    Set-StateFlag 'config_ready' $ConfigOk

    if ($VenvOk) { Write-BOk '.venv ready (Python 3.11)' } else { Write-BWarn '.venv missing or broken -> will be created' }
    if ($DepsOk) { Write-BOk 'dependencies importable' } else { Write-BWarn 'one or more dependencies missing -> will be installed' }
    if ($HfOk) { Write-BOk 'huggingface_hub library available (Python API - no CLI dependency)' } else { Write-BWarn 'huggingface_hub library missing -> will be installed automatically' }
    if ($BinsOk) { Write-BOk 'bin\llama-server.exe + bin\piper.exe present' } else { Write-BWarn 'runtime binaries missing -> will be downloaded' }
    if ($Models.Ok) {
        Write-BOk ('model set complete: ' + ($Models.Present -join ', '))
    } else {
        Write-BWarn ('missing models -> download: ' + ($Models.Missing -join ', '))
        Write-BLog 'INFO' ('model presence check - missing: ' + ($Models.Missing -join ', ') + ' | present: ' + ($Models.Present -join ', '))
    }
    if ($ConfigOk) { Write-BOk 'config ready (voicemem_config.yaml + .env)' } else { Write-BWarn 'config incomplete -> will be completed' }

    # Cheap auto-repairs that never need the full installer (spec 17):
    if (-not (Test-Path $EnvFile)) {
        if (Test-Path $EnvExample) {
            Copy-Item -Path $EnvExample -Destination $EnvFile -Force
            Write-BOk 'config\.env generated from .env.example'
            Write-BLog 'INFO' 'generated config\.env from config\.env.example'
            $ConfigOk = Test-ConfigReady
            Set-StateFlag 'config_ready' $ConfigOk
        }
    }

    # ---- Step 3/5: install / repair decision -------------------------------
    $SmokePassed = [bool](Read-StateFlag 'smoke_tests_passed')
    $NeedInstall = $false
    if ($Mode -eq 'repair') {
        $NeedInstall = $true
        Write-BStep 'Step 3/5: REPAIR mode - running the idempotent installer (fixes anything broken)'
    } elseif ((-not $VenvOk) -or (-not $DepsOk) -or (-not $HfOk) -or (-not $BinsOk) -or (-not $Models.Ok) -or (-not $ConfigOk) -or (-not $SmokePassed)) {
        $NeedInstall = $true
        Write-BStep 'Step 3/5: first-time setup or missing components - running the installer'
        Write-Host '    The one-time setup downloads dependencies and models (~4 GB total).'
        Write-Host '    This can take 15-60 minutes depending on your connection.'
        Write-Host '    Re-running START.bat later SKIPS everything that is already ready.'
    } else {
        Write-BStep 'Step 3/5: environment already ready - skipping the installer (idempotent bootstrap)'
    }

    if ($NeedInstall) {
        Write-BLog 'INFO' ('running installer: ' + $Installer)
        $InstallExit = Invoke-ChildScript $Installer @('-Root', $Root)
        Write-BLog 'INFO' ('installer exit code: ' + $InstallExit)
        if ($InstallExit -ne 0) {
            Fail-Bootstrap ('Installation failed (installer exit code {0}).' -f $InstallExit) `
                'See the [FAIL] / INSTALLATION FAILED lines above.' `
                'A complete, healthy install.' `
                'Follow the Fix hint above (usually: install Python 3.11 / NVIDIA driver / restore network), then double-click START.bat again.'
        }

        # pip bootstrap refresh (spec 9: pip, setuptools, wheel up to date)
        Write-Host '    pip/setuptools/wheel refresh...'
        try { & $VenvPython -m pip install --upgrade pip setuptools wheel *> $null } catch { }
        if ($LASTEXITCODE -ne 0) {
            Write-BWarn 'pip/setuptools/wheel upgrade failed (non-fatal - core install already succeeded)'
        }

        # Re-probe after the installer and update the state flags.
        Write-Host ''
        Write-Host '    Post-install verification...'
        $VenvOk = Test-VenvReady
        $DepsOk = Test-DepsReady
        $HfOk = Test-HfToolingReady
        $BinsOk = Test-BinariesReady
        $Models = Test-ModelsPresent
        $ConfigOk = Test-ConfigReady
        Set-StateFlag 'venv_ready' $VenvOk
        Set-StateFlag 'dependencies_ready' $DepsOk
        Set-StateFlag 'hf_tooling_ready' $HfOk
        Set-StateFlag 'binaries_ready' $BinsOk
        Set-StateFlag 'models_ready' $Models.Ok
        Set-StateFlag 'config_ready' $ConfigOk

        $PostOk = ($VenvOk -and $DepsOk -and $HfOk -and $BinsOk -and $Models.Ok -and $ConfigOk)
        if (-not $PostOk) {
            $Bad = @()
            if (-not $VenvOk) { $Bad += '.venv' }
            if (-not $DepsOk) { $Bad += 'dependencies' }
            if (-not $HfOk) { $Bad += 'huggingface_hub library' }
            if (-not $BinsOk) { $Bad += 'runtime binaries' }
            if (-not $Models.Ok) { $Bad += ('models: ' + ($Models.Missing -join ', ')) }
            if (-not $ConfigOk) { $Bad += 'config' }
            Fail-Bootstrap 'The installer finished, but these components are still not ready.' ($Bad -join '; ') 'all components READY' 'Run START.bat repair, and check logs\bootstrap.log if it keeps failing.'
        }

        # GPU/CUDA + VoiceMem state flags (from the live environment now).
        try { & $VenvPython -c 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 3)' *> $null } catch { }
        $CudaOk = ($LASTEXITCODE -eq 0)
        Set-StateFlag 'cuda_ready' $CudaOk
        Write-BLog 'INFO' ('torch CUDA available: ' + $CudaOk)
        # v0.5.0: not just importable - the import must resolve to the
        # CONTROLLED source under vendor\voicemem with the pinned runtime
        # identity (a stray site-packages voicemem would pass the old
        # bare-import check while shadowing the fork).
        try { & $VenvPython -c "import voicemem,sys;sys.exit(0 if (voicemem.CONTROLLED_FORK and voicemem.CONTROLLED_UPSTREAM_COMMIT=='e8384e087bd2f44eb05fc7ae1a3c525ea8244179' and 'vendor'+chr(92)+'voicemem' in __import__('pathlib').Path(voicemem.__file__).resolve().__str__()) else 1)" *> $null } catch { }
        $VmOk = ($LASTEXITCODE -eq 0)
        Set-StateFlag 'voicemem_ready' $VmOk
        if ($VmOk) {
            Write-BLog 'INFO' 'voicemem importable AND pin-verified (controlled fork @ e8384e0)'
        } else {
            Write-BLog 'ERROR' 'voicemem import failed or is NOT the controlled fork (vendor path / commit mismatch) - degraded memory mode; re-run the installer (START.bat repair)'
        }

        # The installer's step 15 (verify_m1 -WithServer) is the smoke gate.
        Set-StateFlag 'smoke_tests_passed' $true
        $SmokePassed = $true
        Write-BOk 'install + verification complete'
    }

    # Manifest auto-repair (spec 17: manifest missing -> generate manifest).
    if (-not (Test-Path $ManifestFile)) {
        $ManifestScript = Join-Path $Root 'scripts\write_install_manifest.py'
        if ((Test-Path $ManifestScript) -and (Test-Path $VenvPython)) {
            Write-Host ''
            Write-Host '   INSTALL_MANIFEST.json missing - regenerating...'
            try { & $VenvPython $ManifestScript --root $Root --out $ManifestFile *> $null } catch { }
            if ($LASTEXITCODE -eq 0) {
                Write-BOk 'INSTALL_MANIFEST.json regenerated'
            } else {
                Write-BWarn 'manifest regeneration failed (non-fatal)'
            }
        }
    }
    Set-StateFlag 'manifest_ready' (Test-Path $ManifestFile)

    # ---- Step 4/5: state bookkeeping ---------------------------------------
    Write-BStep 'Step 4/5: state saved (.install_state.json)'

    # v0.4.7: vendored VoiceMem trait-prompt localisation (ALWAYS checked,
    # idempotent, seconds). v0.5.0: the vendor tree is the CONTROLLED fork
    # (no .git - the old gate skipped this check entirely for a git-less
    # vendor!). The localisation is folded into the fork (VM-LOCAL-EN); this
    # boot check now verifies the marker is present: a MISSING marker means
    # the vendor tree is not the controlled fork - a loud ERROR (the
    # English-prompt guarantee is part of the fork identity).
    $VmDir = Join-Path $Root 'vendor\voicemem'
    $VmPkg = Join-Path $VmDir 'voicemem\__init__.py'
    $VmPatch = Join-Path $Root 'scripts\patch_voicemem_english.py'
    if ((Test-Path $VmPkg) -and (Test-Path $VmPatch)) {
        Write-BLog 'INFO' 'controlled VoiceMem source: English-patch marker check (idempotent)'
        & $VenvPython $VmPatch $VmDir 2>&1 | ForEach-Object { Write-BLog 'INFO' ("[patch-voicemem] " + $_) }
        if ($LASTEXITCODE -ne 0) {
            Write-BLog 'ERROR' 'voicemem English patch marker MISSING - the vendor\voicemem tree is not the controlled fork (extract the full release ZIP; see VOICEMEM_PIN.json)'
        }
    }
    try {
        $VersionFile = Join-Path $Root 'VERSION'
        if (Test-Path $VersionFile) {
            $VerText = (Get-Content $VersionFile -Raw).Trim()
            Set-StateFlag 'version' $VerText
        }
    } catch { }
    Set-StateFlag 'runs_count' ([int](Read-StateFlag 'runs_count') + 1)
    Set-StateFlag 'last_run_at_utc' (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    Set-StateFlag 'last_mode' $Mode
    Set-StateFlag 'last_result' 'ok'
    Write-BLog 'INFO' ('bootstrap environment ready (mode: {0})' -f $Mode)

    # ---- Step 5/5: mode dispatch --------------------------------------------
    Write-BStep ('Step 5/5: mode dispatch ({0})' -f $Mode)

    switch ($Mode) {
        'run' {
            Write-FinalSummary 'Starting VoiceMem Agent (local web UI)...'
            # v0.4.0: the normal user flow starts the LOCAL WEB UI (browser ->
            # local backend -> agent pipeline -> llama-server). The agent CLI
            # is still available via START.bat cli (python -m app.main).
            $ExitCode = Invoke-ChildScript $StartAgent @('--web')
            Write-BLog 'INFO' ('web agent finished with exit code {0}' -f $ExitCode)
            Write-Host ''
            if ($ExitCode -eq 0) {
                Write-Host 'Web UI stopped normally. Double-click START.bat to open it again.' -ForegroundColor Green
            } else {
                Write-Host ('Web backend exited with code {0} - see the messages above.' -f $ExitCode) -ForegroundColor Yellow
            }
        }
        'cli' {
            Write-FinalSummary 'Starting VoiceMem Agent (console CLI)...'
            $ExitCode = Invoke-ChildScript $StartAgent @()
            Write-BLog 'INFO' ('cli agent finished with exit code {0}' -f $ExitCode)
        }
        'mock' {
            Write-FinalSummary 'Starting VoiceMem Agent (mock demo - no mic, no GPU models)...'
            $ExitCode = Invoke-ChildScript $StartAgent @('-NoServer', '--mock')
            Write-BLog 'INFO' ('mock demo finished with exit code {0}' -f $ExitCode)
        }
        'check' {
            Write-Host ''
            Write-Host 'Full verification: assets + smoke (verify_m1.ps1 -WithServer)...'
            $VerifyExit = Invoke-ChildScript $VerifyM1 @('-Root', $Root, '-WithServer')

            Write-Host ''
            Write-Host 'Full verification: test suite (run_tests.ps1)...'
            $TestsExit = Invoke-ChildScript $RunTests @()
            Write-Host ''
            Write-Host '========================================'
            Write-Host ' VoiceMem Agent verification summary'
            Write-Host '========================================'
            if ($VerifyExit -eq 0) { Write-Host ' Assets + smoke : PASS' -ForegroundColor Green }
            else { Write-Host (' Assets + smoke : FAIL (exit {0})' -f $VerifyExit) -ForegroundColor Red }
            if ($TestsExit -eq 0) { Write-Host ' Test suite     : PASS' -ForegroundColor Green }
            else { Write-Host (' Test suite     : FAIL (exit {0})' -f $TestsExit) -ForegroundColor Red }
            Write-Host ' Validation report: logs\validation_report.json'

            if (($VerifyExit -eq 0) -and ($TestsExit -eq 0)) {
                $ExitCode = 0
                Write-BLog 'INFO' 'check mode: PASS'
            } else {
                $ExitCode = 2
                Write-BLog 'ERROR' ('check mode: FAIL (verify={0}, tests={1})' -f $VerifyExit, $TestsExit)
                Write-Host ' Run START.bat repair to fix a broken environment.' -ForegroundColor Yellow
            }
        }
        'benchmark' {
            Write-Host ''
            Write-Host 'Benchmark suite: end-to-end latency (mock pipeline)...'
            Push-Location $Root
            try {
                & $VenvPython 'tests\benchmark\latency_benchmark.py'
                $LatExit = $LASTEXITCODE
            } finally { Pop-Location }
            Write-BLog 'INFO' ('latency benchmark exit code: {0}' -f $LatExit)
            if ($LatExit -ne 0) {
                Write-Host ('Latency benchmark FAILED (exit {0}).' -f $LatExit) -ForegroundColor Red
                $ExitCode = 2
            } else {
                $ExitCode = 0
                Write-Host 'Latency benchmark: PASS (p50/p95 limits)' -ForegroundColor Green
            }
            # Optional component benchmarks - best effort, warnings only.
            if ($BinsOk -and $Models.Ok) {
                foreach ($Bench in @('tests\benchmark\tts_benchmark.py', 'tests\benchmark\asr_benchmark_hu.py')) {
                    Write-Host ''
                    Write-Host ('Optional benchmark: {0} (best effort)...' -f $Bench)
                    Push-Location $Root
                    try {
                        & $VenvPython $Bench
                        $BExit = $LASTEXITCODE
                    } finally { Pop-Location }
                    Write-BLog 'INFO' ('{0} exit code: {1}' -f $Bench, $BExit)
                    if ($BExit -ne 0) { Write-BWarn ('{0} did not pass - see output above (non-fatal)' -f $Bench) }
                }
                if (Test-LlamaHealth) {
                    Write-Host ''
                    Write-Host 'llama-server is running - LLM benchmark...'
                    Push-Location $Root
                    try {
                        & $VenvPython 'tests\benchmark\llm_benchmark.py'
                        $LlmExit = $LASTEXITCODE
                    } finally { Pop-Location }
                    Write-BLog 'INFO' ('llm benchmark exit code: {0}' -f $LlmExit)
                    if ($LlmExit -ne 0) { Write-BWarn 'llm benchmark did not pass (non-fatal)' }
                } else {
                    Write-Host ''
                    Write-Host 'llama-server not running - LLM benchmark skipped.'
                    Write-Host 'Run START.bat once (it starts the server), then START.bat benchmark again.'
                }
            } else {
                Write-Host ''
                Write-Host 'Component benchmarks skipped (binaries/models not all present).'
            }
            Write-Host ''
            Write-Host 'Benchmark results are written under data\benchmarks\.'
        }
        'repair' {
            # The installer already ran above (repair forces NeedInstall).
            # Post-repair smoke: the scripted mock demo must pass.
            Write-Host ''
            Write-Host 'Post-repair smoke: scripted mock demo...'
            $MockExit = Invoke-ChildScript $StartAgent @('-NoServer', '--mock')
            if ($MockExit -eq 0) {
                Write-BOk 'repair complete - the mock demo passes'
                Write-FinalSummary 'Repair finished. Double-click START.bat to talk.'
                $ExitCode = 0
            } else {
                Write-BErr ('repair finished, but the mock demo still fails (exit {0})' -f $MockExit)
                Write-Host 'See logs\bootstrap.log and the messages above.' -ForegroundColor Yellow
                $ExitCode = 1
            }
            Write-BLog 'INFO' ('repair mode result: {0}' -f $ExitCode)
        }
        'build' {
            Write-Host ''
            Write-Host 'Versioned release build (the full test suite runs BEFORE the ZIP)...'
            $BuildArgs = @()
            if ($Notes -ne '') { $BuildArgs += @('-Notes', $Notes) }
            if ($Bump -ne '') { $BuildArgs += @('-Bump', $Bump) }
            if ($StrictValidation) { $BuildArgs += '-StrictValidation' }
            $ExitCode = Invoke-ChildScript $BuildRelease $BuildArgs
            Write-BLog 'INFO' ('build mode exit code: {0}' -f $ExitCode)
            if ($ExitCode -eq 0) {
                Write-Host ''
                Write-Host 'Release ZIP written to releases\ (see the output above).' -ForegroundColor Green
            } else {
                Write-Host ''
                Write-Host 'Build FAILED - no ZIP was produced. See the output above.' -ForegroundColor Red
            }
        }
    }
}
catch {
    $ExitCode = 1
    $CleanMsg = $_.Exception.Message
    $FullDetail = $_.Exception.ToString()
    try { $FullDetail = $FullDetail + "`n" + $_.ScriptStackTrace } catch { }
    Write-BLog 'ERROR' ('unexpected bootstrap error: ' + $FullDetail)
    try { Set-StateFlag 'last_result' ('error: ' + $CleanMsg) } catch { }
    Write-Host ''
    Write-Host ('[ERROR] Unexpected bootstrap error: {0}' -f $CleanMsg) -ForegroundColor Red
    Write-Host 'Technical details: logs\bootstrap.log'
    Write-Host 'Run START.bat repair to attempt an automatic fix.'
    Write-Host ''
}

exit $ExitCode
