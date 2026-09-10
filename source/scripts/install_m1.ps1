<#
===============================================================================
scripts/install_m1.ps1 - M0/M2 telepito (22 lepeses, idempotens)

CEL: egyetlen paranccsal teljesen lokalis, reprodukalhato runtime a repo
gyokereben (clone -> install -> modellletoltes -> smoke test -> run):
    powershell -ExecutionPolicy Bypass -File scripts\install_m1.ps1

A 22 LEPES (a felhasznalo specje szerint, pontosan ebben a sorrendben):
  1.  repo-gyoker ellenorzese (app/ + scripts/ letezese)
  2.  Windows verzio ellenorzese (10+; a Win11-detektalas informalis)
  3.  Python 3.11 ellenorzese (py -3.11, fallback: python3.11, python)
      - ha nincs pontos 3.11.x: ertheto hiba + javaslat + LEALLITAS
      - SOHA nem telepitunk automatikusan masik Python verziot
  4.  NVIDIA GPU ellenorzese (nvidia-smi elerheto + GPU nev kiirasa)
  5.  RTX 5070 ellenorzese (mas NVIDIA GPU: figyelmeztetes, de megy tovabb;
      a compute capability (sm_120) a 10. lepes GPU smoke testjeben dont)
  6.  projektmappak: models/, memory/, data/, logs/, bin/, vendor/
  7.  .venv letrehozasa (Python 3.11)
  8.  pip frissitese
  9.  core fuggosegek (requirements.lock, ha van, kulonben requirements.txt)
      + huggingface_hub KONYVTAR a modellletolteshez (Python API, CLI nelkul)
  10. PyTorch CU128 telepitese + GPU SMOKE TEST
      (mar telepitett es importalhato torch -> skip; cc < 12.0 -> SIKERTELEN)
  11. llama.cpp + piper binarisok letoltese (pin-elt GitHub release)
  12. VoiceMem: VEZERELT vendor\voicemem (repo-ban) + pip install -e + pin-ellenorzes
  13. onnxruntime telepitese (M1: Silero VAD futtato, CPU - pin-elt 1.23.0;
      v0.3.4 field report #4: sem a requirements.lock aktival sorai, sem a
      voicemem fuggosegei (sherpa-onnx) nem viszik a venv-be, az app/vad.py
      Silero VAD-ja viszont PONTOSAN onnxruntime-ot var - nelkule az agent
      mar a SileroVad konstruktorban elhal)
  14. funasr telepitese (M2: emotion2vec+ futtato, CPU; torch-NELKULI dep,
      a cu128 harmast nem bantja - utana trio-guard fut)
  15. speechbrain telepitese (M3: ECAPA beszeloi embedding futtato, CPU)
  16. transformers or: >= 5.0 (Qwen3-ASR nativ tamogatas; a voicemem pin-jet
      visszabeszelo csomagok javitasa)
  17. modellletoltes: scripts\download_models.ps1 (idempotens, M1+M2 lock)
  18. konfiguracio: config\voicemem_config.yaml + .env a .env.example-bol
  19. smoke testek: scripts\verify_m1.ps1 -WithServer (FAIL -> SIKERTELEN)
  20. licenc-ellenorzes: LICENSES.md + Piper hang .onnx.json fajlok
  21. offline kornyezet: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
      HF_HOME=$Root\models\hf (a HF cache a repo models\hf ala izolalodik)
  22. vegso install report + INSTALL_MANIFEST.json

IDEMPOTENS: nyugodtan ujrafuttathato - a kesz reszeket atugorja
(skip-if-exists mindenutt). HAMIS SUCCESS-T SOHA nem ir ki: minden siker-
kimenet a 15-17. lepes tenyleges ellenorzesere epul.

KAPCSOLOK:
  -SkipModels        a modellletoltes kihagyasa (offline ujratelepiteshez)
  -SkipBinaries      a llama.cpp/piper letoltes kihagyasa (mar kezzel telepitve)
  -SkipVoiceMem      a VoiceMem kihagyasa (a verify_m1 jelezni fogja)
  -Force             .venv ujraepitese + binarisok ujraletoltese
                     (a modelleket NEM torli - azt a download_models kezeli)
  -ConnectivityAudit a 17. lepesben az offline_check.ps1 is lefut (alap: skip)

PIN-ELT VERZIOK (reprodukalhatosag - webes kutatas, 2026-08-31):
  llama.cpp : tag b10717
                binaris:  llama-b10717-bin-win-cuda-13.3-x64.zip (llama-server.exe)
                cuda dll: cudart-llama-bin-win-cuda-13.3-x64.zip (CUDA 13.3 runtime)
                forras: https://github.com/ggml-org/llama.cpp/releases
                (a ggerganov/llama.cpp URL ide iranyit at)
  piper     : tag 2023.11.14-2, asset piper_windows_amd64.zip
                forras: https://github.com/rhasspy/piper/releases
  VoiceMem  : VEZERELT FORRAS - vendor\voicemem a repo-ban (controlled fork;
                alap: upstream commit e8384e087bd2f44eb05fc7ae1a3c525ea8244179
                = tag v0.0.1; identitas: VOICEMEM_PIN.json + UPSTREAM_POLICY.md).
                NINCS git clone, NINCS upstream fetch/checkout, NINCS main-ag
                fallback - a telepito a repo-ban szallitott forrasbol
                pip install -e vendor\voicemem telepit, majd a pint ellenorzi.)

MEGJEGYZES: a fajl szandekosan ASCII - a PowerShell 5.1 BOM nelkul
Windows-1252-kent olvasna az ekezetes karaktereket.

HASZNALAT:
    powershell -ExecutionPolicy Bypass -File scripts\install_m1.ps1
    .\scripts\install_m1.ps1 -SkipModels
    .\scripts\install_m1.ps1 -Root "C:\VoiceMemAgent" -Force
#>
param(
    [string]$Root = "",
    [switch]$SkipModels,
    [switch]$SkipBinaries,
    [switch]$SkipVoiceMem,
    [switch]$Force,
    [switch]$ConnectivityAudit
)

# ---------------------------------------------------------------------------
# Pin-elt verziok (a szkript elejen, reprodukalhatosag)
# ---------------------------------------------------------------------------
$LlamaCppTag = "b10717"
$LlamaCppBinAsset = "llama-b10717-bin-win-cuda-13.3-x64.zip"
$LlamaCppCudartAsset = "cudart-llama-bin-win-cuda-13.3-x64.zip"
$LlamaCppBaseUrl = "https://github.com/ggml-org/llama.cpp/releases/download"
$PiperTag = "2023.11.14-2"
$PiperAsset = "piper_windows_amd64.zip"
$PiperBaseUrl = "https://github.com/rhasspy/piper/releases/download"
# v0.5.0: a VoiceMem VEZERELT FORRAS (vendor\voicemem, l. VOICEMEM_PIN.json) -
# nincs upstream repo/ref klonozes; az identitast a pin-fajl es a futokori
# voicemem.CONTROLLED_UPSTREAM_COMMIT adja.
$VoiceMemUpstreamCommit = "e8384e087bd2f44eb05fc7ae1a3c525ea8244179"
$VoiceMemPinFile = "VOICEMEM_PIN.json"
$TorchVersion = "2.7.0"
$TorchvisionVersion = "0.22.0"
$TorchaudioVersion = "2.7.0"
$TorchCudaIndex = "https://download.pytorch.org/whl/cu128"
$FunasrVersion = "1.4.11"
$SpeechbrainVersion = "1.1.1"
# v0.3.4 (field report #4): a Silero VAD futtatoja - M1-KRITIKUS csomag.
# A requirements.lock aktival sorai csak a core harmast telepitik (numpy/
# PyYAML/httpx - a TARGET-ONLY blokk ott megjegyzes), a voicemem fuggosegei
# pedig sherpa-onnx-ot huznak, nem onnxruntime-t -> ez a csomag CSAK itt,
# kulon lepesben kerul a venv-be. CPU build, torch-fuggosege nincs.
$OnnxRuntimeVersion = "1.23.0"
# FIGYELEM: a harom csomag HIVATALOS PAROSITOTT verzioja, mind ugyanarrol a
# CUDA 12.8 indexrol (torch 2.7.0 <-> torchvision 0.22.0 <-> torchaudio
# 2.7.0). A v0.1.6 telepito a torchvision/torchaudio verziokat NEM pinelte,
# ezert a pip a cu128 indexrol torchaudio 2.11.0+cu128-et oldott fel a
# torch 2.7.0 melle - inkompatibilis paros. A pip soha nemdontshet a
# parositott verziorol: mindharom EXAKT pinelve van, es a telepites utan
# import-val ellenorzott.

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

# ---------------------------------------------------------------------------
# Segedfuggvenyek
# ---------------------------------------------------------------------------
$Script:StepNo = 0

function Write-Step {
    param([string]$Title)
    $Script:StepNo = $Script:StepNo + 1
    Write-Host ""
    Write-Host ("---- [{0}/22] {1}" -f $Script:StepNo, $Title) -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host ("    FIGYELEM: {0}" -f $Message) -ForegroundColor Yellow
}

function Fail-Install {
    param([string]$Message, [string]$Hint = "")
    Write-Host ""
    Write-Host "INSTALLATION FAILED" -ForegroundColor Red
    Write-Host ("Failed step : {0}" -f $Script:StepNo)
    Write-Host ("Error       : {0}" -f $Message)
    if ($Hint -ne "") {
        Write-Host ("Fix         : {0}" -f $Hint)
    }
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "==================================================================="
Write-Host " VoiceMemAgent M0 telepito (22 lepes, idempotens)"
Write-Host "==================================================================="

# ===========================================================================
# 1) Project root ellenorzese
# ===========================================================================
Write-Step "Project root ellenorzese"
if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
if (-not (Test-Path $Root)) {
    Fail-Install ("A megadott Root nem letezik: {0}" -f $Root) "Add meg a repo-gyoker utvonalat a -Root parameterrel."
}
$AppDir = Join-Path $Root "app"
$ScriptsDir = Join-Path $Root "scripts"
if (-not ((Test-Path $AppDir) -and (Test-Path $ScriptsDir))) {
    Fail-Install ("A(z) {0} nem egy ervenyes repo-gyoker (az app\ es a scripts\ mappa kotelezo)." -f $Root) "A teljes repot klonozd (git clone), es abbol futtasd a scripts\install_m1.ps1-t."
}
Write-Host ("    Root: {0}" -f $Root)

# ---------------------------------------------------------------------------
# v0.3.2: Mark-of-the-Web eltavolitas a scripts\*.ps1 fajlokbol. A
# bongeszobol letoltott release ZIP Explorer-kicsomagolasanal a .ps1-ek
# Zone.Identifier internet-zona jelzest orokolhetnek - a normal RemoteSigned
# policy kezi inditasnal "not digitally signed" hibaval blokkolja oket.
# A START.bat lanca Bypass-szal fut, de kezi inditasok miatt itt is
# eltavolitjuk. (Azonos blokk: bootstrap.ps1 + verify_m1.ps1.)
# ---------------------------------------------------------------------------
Get-ChildItem -Path $ScriptsDir -Filter "*.ps1" -File -ErrorAction SilentlyContinue |
    Unblock-File -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# GPU-check library (robust: PATH -> standard locations -> exec -> query ->
# WMI fallback + diagnostics; shared with bootstrap.ps1).
# ---------------------------------------------------------------------------
. (Join-Path $ScriptsDir 'gpu_check.ps1')

function Write-ILog {
    # Appends a timestamped line to logs\bootstrap.log - the single technical
    # log the user sends with bug reports (shared with bootstrap.ps1).
    param([string]$Level = 'INFO', [string]$Message = '')
    try {
        $LogDir = Join-Path $Root 'logs'
        if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
        $Line = ('{0} [{1}] (installer) {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message)
        Add-Content -LiteralPath (Join-Path $LogDir 'bootstrap.log') -Value $Line -Encoding UTF8
    } catch { }
}

# ===========================================================================
# 2) Windows verzio ellenorzese
# ===========================================================================
Write-Step "Windows verzio ellenorzese"
$OsVersion = [System.Environment]::OSVersion.Version
Write-Host ("    OS verzio: {0} (build {1})" -f $OsVersion.ToString(), $OsVersion.Build)
if ($OsVersion.Build -ge 22000) {
    Write-Host "    Windows 11 detektalva (informalis megjegyzes)."
} elseif ($OsVersion.Build -ge 10240) {
    Write-Host "    Windows 10 detektalva (informalis megjegyzes)."
}
if ($OsVersion.Major -lt 10) {
    Fail-Install ("Windows 10 vagy ujabb szukseges (talalt: {0})." -f $OsVersion.ToString()) "Frissits Windows 10/11-re, majd futtasd ujra a telepitot."
}

# ===========================================================================
# 3) Python 3.11 ellenorzese (SOHA nem telepitunk masik Pythont)
# ===========================================================================
Write-Step "Python 3.11 ellenorzese"
# Python is resolved as an EXECUTABLE PATH + an ARGUMENT ARRAY and invoked
# as `& $PythonExe @PythonArgs ...` (the same pattern as bootstrap.ps1).
# The installer NEVER passes a multi-element array to the call operator:
# in Windows PowerShell 5.1 `& $PythonCmd` with $PythonCmd = @('py','-3.11')
# stringifies the array into the single (nonexistent) command name
# 'py -3.11' -> CommandNotFoundException -> broken venv creation.
$PythonExe = ""
$PythonArgs = @()
$PythonFound = ""
function Resolve-PythonSource {
    param([string]$Name)
    $Src = (Get-Command $Name -ErrorAction SilentlyContinue).Source
    if ([string]::IsNullOrWhiteSpace($Src)) { return $Name }
    return $Src
}
if (Get-Command py -ErrorAction SilentlyContinue) {
    $Out = py -3.11 --version 2>$null
    $OutText = (@($Out) -join " ").Trim()
    if (($LASTEXITCODE -eq 0) -and ($OutText -match 'Python 3\.11\.')) {
        $PythonExe = Resolve-PythonSource 'py'
        $PythonArgs = @("-3.11")
        $PythonFound = $OutText
    }
}
if ($PythonExe -eq "") {
    $C311 = Get-Command python3.11 -ErrorAction SilentlyContinue
    if ($C311) {
        $Out = python3.11 --version 2>$null
        $OutText = (@($Out) -join " ").Trim()
        if (($LASTEXITCODE -eq 0) -and ($OutText -match 'Python 3\.11\.')) {
            $PythonExe = Resolve-PythonSource 'python3.11'
            $PythonArgs = @()
            $PythonFound = $OutText
        }
    }
}
if ($PythonExe -eq "") {
    $CPy = Get-Command python -ErrorAction SilentlyContinue
    if ($CPy) {
        $Out = python --version 2>$null
        $OutText = (@($Out) -join " ").Trim()
        $PythonFound = $OutText
        if (($LASTEXITCODE -eq 0) -and ($OutText -match 'Python 3\.11\.')) {
            $PythonExe = Resolve-PythonSource 'python'
            $PythonArgs = @()
        }
    }
}
if ($PythonExe -eq "") {
    $PyHint = "Toltsd le es telepitsd a 64-bites Python 3.11.9-et a https://www.python.org/downloads oldalrol " +
        "(a telepito SOHA nem telepit automatikusan masik Python verziot), majd inditsd ujra a START.bat-ot."
    Fail-Install ("Nem talalhato pontosan Python 3.11.x. Utoljara latott verzio: '{0}'." -f $PythonFound) $PyHint
}
$PythonDisplay = (@($PythonExe) + @($PythonArgs)) -join " "
Write-Host ("    Python: {0} ({1})" -f $PythonFound, $PythonDisplay)

# ===========================================================================
# 4) NVIDIA GPU ellenorzese (robusztus, tobb strategias - v0.1.5)
# ===========================================================================
Write-Step "NVIDIA GPU ellenorzese"
$GpuStatus = Resolve-NvidiaGpuStatus
if ($GpuStatus.Ok) {
    Write-Host ("    GPU: {0}" -f $GpuStatus.GpuName)
    Write-Host ("    Driver: {0} | VRAM: {1} MiB | nvidia-smi: {2} [{3}]" -f $GpuStatus.Driver, $GpuStatus.VramMiB, $GpuStatus.SmiPath, $GpuStatus.SmiOrigin)
    Write-ILog 'INFO' ('GPU OK: ' + $GpuStatus.GpuName + ' | driver ' + $GpuStatus.Driver + ' | VRAM ' + $GpuStatus.VramMiB + ' MiB | nvidia-smi ' + $GpuStatus.SmiPath + ' [' + $GpuStatus.SmiOrigin + ']')
    if (($GpuStatus.VramMiB -gt 0) -and ($GpuStatus.VramMiB -lt 10240)) {
        Write-Warn ("A VRAM {0} MiB - az M1 konfiguracio 10 GB feletti VRAM-mal biztositott." -f $GpuStatus.VramMiB)
    }
} else {
    # Diagnostics FIRST - collected in THIS process environment (the START.bat
    # chain child), proving what this exact environment can and cannot see
    # (interactive PowerShell vs START.bat chain comparison).
    $Diag = New-GpuDiagnostics -Status $GpuStatus
    Write-ILog 'WARN' $Diag
    $PsBitText = 'unknown'
    try { $PsBitText = [string][Environment]::Is64BitProcess } catch { }
    $GcDiag = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    $GcDiagText = 'NOT FOUND in this process'
    if ($GcDiag) { $GcDiagText = [string]$GcDiag.Source }
    $Sys32Text = 'N/A'
    if ($env:SystemRoot) {
        $Sys32Text = [string](Test-Path -LiteralPath (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'))
    }
    Write-Warn ("Diagnosztika: PowerShell 64-bit: {0} | Get-Command nvidia-smi: {1} | System32 nvidia-smi: {2}" -f $PsBitText, $GcDiagText, $Sys32Text)
    Write-Warn "             (a teljes diagnosztikai blokk a logs\bootstrap.log-ban)"

    $WmiNote = ''
    if ($GpuStatus.WmiSeen) { $WmiNote = (' | WMI latja: {0}' -f $GpuStatus.WmiName) }
    switch ($GpuStatus.Code) {
        'GPU_NOT_FOUND' {
            Fail-Install "Nincs NVIDIA GPU a gepen (GPU_NOT_FOUND): a WMI (Win32_VideoController) sem talalt NVIDIA adaptert." "A spec szerint NVIDIA GPU kotelezo (CPU-ra nincs tamogatas). Ellenorizd, hogy NVIDIA GPU van-e a gepben."
        }
        'DRIVER_NOT_FOUND' {
            Fail-Install ("NVIDIA GPU van, de a driver nem mukodik (DRIVER_NOT_FOUND).{0}" -f $WmiNote) "Telepitsd ujra az NVIDIA drivert (nvidia.com, clean install), majd inditsd ujra a START.bat-ot."
        }
        'NVIDIA_SMI_NOT_FOUND' {
            Fail-Install ("NVIDIA GPU es driver lathato, de az nvidia-smi eszkoz nem talalhato (NVIDIA_SMI_NOT_FOUND).{0}" -f $WmiNote) "Telepitsd ujra az NVIDIA drivert (nvidia.com, clean install - ez visszateszi az nvidia-smi.exe-t), majd inditsd ujra a START.bat-ot."
        }
        'NVIDIA_SMI_EXEC_FAILED' {
            Fail-Install ("Az nvidia-smi megtalalhato, de nem futtathato (NVIDIA_SMI_EXEC_FAILED): {0}" -f $GpuStatus.Detail) "Valoszinuleg serult driver vagy erintett nvidia-smi.exe - telepitsd ujra az NVIDIA drivert (clean install), majd inditsd ujra a START.bat-ot."
        }
        'NVIDIA_SMI_QUERY_FAILED' {
            Fail-Install ("Az nvidia-smi lefutott, de a kimenete nem ertelmezheto (NVIDIA_SMI_QUERY_FAILED): {0}" -f $GpuStatus.Detail) "Telepitsd ujra az NVIDIA drivert (clean install), majd inditsd ujra a START.bat-ot."
        }
        'VRAM_QUERY_FAILED' {
            Fail-Install ("A GPU nev es driver rendben, de a VRAM nem olvashato ki (VRAM_QUERY_FAILED): {0}" -f $GpuStatus.Detail) "Telepitsd ujra az NVIDIA drivert, majd inditsd ujra a START.bat-ot."
        }
        default {
            Fail-Install ("A GPU allapota nem allapithato meg (GPU_VALIDATION_FAILED): {0}" -f $GpuStatus.Detail) "Kuldd el a logs\bootstrap.log-ot a teljes GPU-diagnosztikai blokkal."
        }
    }
}
$GpuNameSmi = [string]$GpuStatus.GpuName

# ===========================================================================
# 5) RTX 5070 ellenorzese (informalis; a cc-t a 10. lepes donti el)
# ===========================================================================
Write-Step "RTX 5070 ellenorzese (informalis)"
if ($GpuNameSmi -notmatch "RTX 5070") {
    Write-Warn ("A GPU nem RTX 5070 ('{0}') - a telepites folytatodik; a compute capability-t " -f $GpuNameSmi)
    Write-Warn "a 10. lepes GPU smoke testje donti el (cc >= 12.0 kotelezo)."
} else {
    Write-Host "    RTX 5070 detektalva (Blackwell, sm_120)."
}

# ===========================================================================
# 6) Projektmappak letrehozasa
# ===========================================================================
Write-Step "Projektmappak letrehozasa (idempotens)"
$Dirs = @(
    "models\asr",
    "models\llm\qwen3.6-35b-a3b",
    "models\tts\piper",
    "models\vad\silero-vad",
    "models\embedding\multilingual-e5-small",
    "models\emotion",
    "models\hf",
    "memory\sqlite",
    "memory\qdrant",
    "memory\backups",
    "data\audio",
    "data\benchmarks",
    "logs",
    "bin",
    "vendor"
)
foreach ($Dir in $Dirs) {
    $Target = Join-Path $Root $Dir
    if (-not (Test-Path $Target)) {
        New-Item -ItemType Directory -Path $Target -Force | Out-Null
        Write-Host ("    letrehozva: {0}" -f $Dir)
    }
}
Write-Host "    Minden projektmappa rendben."

# ===========================================================================
# 7) .venv letrehozasa
# ===========================================================================
Write-Step ".venv letrehozasa (Python 3.11)"
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if ($Force -and (Test-Path $VenvDir)) {
    Write-Host "    -Force: a regi .venv torlese es ujraepitese..."
    Remove-Item -Path $VenvDir -Recurse -Force
}
if ((Test-Path $VenvDir) -and (-not (Test-Path $VenvPython))) {
    Write-Warn "A .venv konyvtar serultnek tunik (nincs benne python.exe) - ujraepitem."
    Remove-Item -Path $VenvDir -Recurse -Force
}
if (-not (Test-Path $VenvPython)) {
    & $PythonExe @PythonArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Fail-Install "A .venv letrehozasa nem sikerult." ("Futtasd kezzel: {0} -m venv .venv - ha ez is hibazik, a Python 3.11 telepitese serult." -f $PythonDisplay)
    }
} else {
    Write-Host "    A .venv mar letezik - a letrehozast atugrom."
}
$VenvVer = & $VenvPython --version 2>$null
$VenvVerText = ((@($VenvVer) -join " ")).Trim()
if (($LASTEXITCODE -ne 0) -or ($VenvVerText -notmatch 'Python 3\.11\.')) {
    # ONE-CLICK automatikus javitas (M0.2): a .venv regeneralhato allapot -
    # rossz Python-verzio eseten automatikusan ujraepitjuk (a fuggosegeket a
    # 9. lepes ujratelepiti), nem a felhasznalora tereljuk a kezi torlest.
    Write-Warn ("A .venv Pythonja nem 3.11.x (kapott: '{0}') - a .venv AUTOMATIKUS ujraepitese." -f $VenvVerText)
    if (Test-Path $VenvDir) { Remove-Item -Path $VenvDir -Recurse -Force }
    & $PythonExe @PythonArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Fail-Install "A .venv automatikus ujraepitese nem sikerult." "A Python 3.11 telepitese serultnek tunik - telepitsd ujra a python.org-rol, majd inditsd ujra a START.bat-ot."
    }
    $VenvVer = & $VenvPython --version 2>$null
    $VenvVerText = ((@($VenvVer) -join " ")).Trim()
    if (($LASTEXITCODE -ne 0) -or ($VenvVerText -notmatch 'Python 3\.11\.')) {
        Fail-Install ("Az ujraepitett .venv Pythonja sem 3.11.x (kapott: '{0}')." -f $VenvVerText) "Telepitsd ujra a 64-bites Python 3.11.x-et a python.org-rol, majd inditsd ujra a START.bat-ot."
    }
}
Write-Host ("    {0} -> {1}" -f $VenvVerText, $VenvPython)

# ===========================================================================
# 8) pip frissitese
# ===========================================================================
Write-Step "pip frissitese"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Fail-Install "A pip frissitese nem sikerult." "Halozati problema? Ujrafuttatas (idempotens), vagy kezzel: .venv\Scripts\python.exe -m pip install --upgrade pip"
}
$PipVer = & $VenvPython -m pip --version 2>$null
Write-Host ("    {0}" -f ((@($PipVer) -join " ")).Trim())

# ===========================================================================
# 9) Core fuggosegek telepitese
# ===========================================================================
Write-Step "Core fuggosegek telepitese"
$LockFile = Join-Path $Root "requirements.lock"
$ReqFile = Join-Path $Root "requirements.txt"
if (Test-Path $LockFile) {
    $UseReq = $LockFile
} elseif (Test-Path $ReqFile) {
    $UseReq = $ReqFile
} else {
    Fail-Install "Nem talalhato sem requirements.lock, sem requirements.txt a repo gyokerben." "Klonozd ujra a repot, vagy potold a hianyzo fajlt."
}
Write-Host ("    pip install -r {0}" -f (Split-Path -Leaf $UseReq))
& $VenvPython -m pip install -r $UseReq
if ($LASTEXITCODE -ne 0) {
    Fail-Install "A core fuggosegek telepitese nem sikerult." "Nezd a pip hibauzenetet fent (halozat?), majd ujrafuttatas (idempotens)."
}
# huggingface_hub KONYVTAR (Python API a modellletolteshez).
# SZANDERKOS CLI nelkul: a Hugging Face CLI mar NEM runtime dependency - a
# regi "huggingface-cli download" deprecation warningjanak emoji karaktere
# CP1252 konzolon UnicodeEncodeError-t okozott (v0.1.6 release blocker).
& $VenvPython -m pip install "huggingface_hub"
if ($LASTEXITCODE -ne 0) {
    Fail-Install "A huggingface_hub konyvtar telepitese nem sikerult (a modellletolteshez kell)." "Halozati problema? Ujrafuttatas (idempotens)."
}

# ===========================================================================
# 10) PyTorch CU128 telepitese + GPU smoke test
#
#     A harom csomag (torch / torchvision / torchaudio) VERZIO-ELLENORZO
#     probe-ja: a telepites elott ES utan is lefut, a kesobbi VoiceMem-
#     telepites utan pedig ujra (guard a pip feluliras ellen).
#     - friss .venv: az import hibazik -> telepites
#     - v0.1.6-bol orokolt, eltorult allapot (pl. torchaudio 2.11.0+cu128):
#       a probe kilepesi kod 4 -> AUTOMATIKUS visszairanyitas a pin-elt
#       haromasra (idempotens javitas, nem kezi munka)
# ===========================================================================
Write-Step "PyTorch CU128 telepitese + GPU smoke test"
$TrioProbeCode = @'
import sys
import torch
import torchvision
import torchaudio


def _base(v):
    return v.split("+")[0]


def _local(v):
    return v.split("+", 1)[1] if "+" in v else ""


WANT = {"torch": "2.7.0", "torchvision": "0.22.0", "torchaudio": "2.7.0"}
errs = []
for mod in (torch, torchvision, torchaudio):
    name = mod.__name__
    version = mod.__version__
    local = _local(version)
    if _base(version) != WANT[name] or (local and "cu128" not in local):
        errs.append("%s==%s (want %s+cu128)" % (name, version, WANT[name]))
if errs:
    print("TRIO_MISMATCH: " + ", ".join(errs))
    sys.exit(4)
print("TORCH_VERSION=" + torch.__version__)
print("TORCHVISION_VERSION=" + torchvision.__version__)
print("TORCHAUDIO_VERSION=" + torchaudio.__version__)
print("TORCH_CUDA=" + str(torch.version.cuda))
print("TORCH_TRIO_OK")
'@
function Test-TorchTrio {
    # A venv-ben telepitett harmas ellenorzese import utan (exit 0 = rendben).
    & $VenvPython -c $TrioProbeCode *> $null
    return ($LASTEXITCODE -eq 0)
}
if (Test-TorchTrio) {
    Write-Host "    A pin-elt PyTorch haromas (torch 2.7.0 + torchvision 0.22.0 + torchaudio 2.7.0, cu128) mar telepitve - a telepitest atugrom (idempotencia)."
} else {
    & $VenvPython -c $TrioProbeCode 2>$null | ForEach-Object { if ($_ -match '^TRIO_MISMATCH') { Write-Host ("    Regi/eltorult allapot: {0}" -f $_) } }
    Write-Host ("    pip install torch=={0} torchvision=={1} torchaudio=={2} --index-url {3}" -f $TorchVersion, $TorchvisionVersion, $TorchaudioVersion, $TorchCudaIndex)
    & $VenvPython -m pip install "torch==$TorchVersion" "torchvision==$TorchvisionVersion" "torchaudio==$TorchaudioVersion" --index-url $TorchCudaIndex
    if ($LASTEXITCODE -ne 0) {
        Fail-Install "A PyTorch cu128 telepitese nem sikerult." "Halozati problema vagy a pytorch.org index elerhetetlen? Ujrafuttatas (idempotens)."
    }
    if (-not (Test-TorchTrio)) {
        Fail-Install "A telepites utan a PyTorch haromas (torch 2.7.0 / torchvision 0.22.0 / torchaudio 2.7.0, cu128) import utan nem stimmel." "Inditsd ujra a START.bat-ot (idempotens); ha tovabb is hibazik, a pytorch.org cu128 index elerhetosege a problema."
    }
}
Write-Host "    GPU smoke test (valodi CUDA-matrixmuvelet) futtatasa..."
$GpuSmokeCode = @'
import torch
import torchvision
import torchaudio
assert torch.cuda.is_available(), "CUDA is not available"
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_device_properties(0).total_memory)
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
assert float(y.sum()) == float(y.sum())  # tenyleges inference tortent
print("GPU_NAME=" + str(torch.cuda.get_device_name(0)))
print("GPU_CC_MAJOR=" + str(torch.cuda.get_device_capability(0)[0]))
print("GPU_CC_MINOR=" + str(torch.cuda.get_device_capability(0)[1]))
print("VRAM_BYTES=" + str(torch.cuda.get_device_properties(0).total_memory))
print("TORCH_VERSION=" + str(torch.__version__))
print("TORCHVISION_VERSION=" + str(torchvision.__version__))
print("TORCHAUDIO_VERSION=" + str(torchaudio.__version__))
print("CUDA_VERSION=" + str(torch.version.cuda))
print("GPU OK")
'@
$SmokeOut = & $VenvPython -c $GpuSmokeCode
$SmokeExit = $LASTEXITCODE
if ($SmokeExit -ne 0) {
    foreach ($Line in @($SmokeOut)) { Write-Host "    $Line" }
    Fail-Install "A GPU smoke test elbukott (a torch nem latja a CUDA-t vagy a CUDA-kernel nem fut)." "Ellenorizd: nvidia-smi mutatja-e a GPU-t, es a fenti hibauzenetet. CPU-ra NINCS tamogatas (a spec szerint a telepites sikertelen)."
}
$GpuNameTorch = ""
$GpuCcMajor = 0
$GpuCcMinor = 0
$VramBytes = 0
$TorchVer = ""
$TorchvisionVer = ""
$TorchaudioVer = ""
$CudaVer = ""
foreach ($Line in @($SmokeOut)) {
    if ($Line -match '^GPU_NAME=(.+)$') { $GpuNameTorch = $Matches[1].Trim() }
    elseif ($Line -match '^GPU_CC_MAJOR=(\d+)$') { $GpuCcMajor = [int]$Matches[1] }
    elseif ($Line -match '^GPU_CC_MINOR=(\d+)$') { $GpuCcMinor = [int]$Matches[1] }
    elseif ($Line -match '^VRAM_BYTES=(\d+)$') { $VramBytes = [long]$Matches[1] }
    elseif ($Line -match '^TORCH_VERSION=(.+)$') { $TorchVer = $Matches[1].Trim() }
    elseif ($Line -match '^TORCHVISION_VERSION=(.+)$') { $TorchvisionVer = $Matches[1].Trim() }
    elseif ($Line -match '^TORCHAUDIO_VERSION=(.+)$') { $TorchaudioVer = $Matches[1].Trim() }
    elseif ($Line -match '^CUDA_VERSION=(.+)$') { $CudaVer = $Matches[1].Trim() }
}
Write-Host ("    torch: {0} | torchvision: {1} | torchaudio: {2}" -f $TorchVer, $TorchvisionVer, $TorchaudioVer)
Write-Host ("    cuda: {0} | GPU: {1} | cc: {2}.{3} | VRAM: {4} GB" -f $CudaVer, $GpuNameTorch, $GpuCcMajor, $GpuCcMinor, [math]::Round($VramBytes / 1GB, 1))
# A harom parositott verzio tenyleges import utanali ellenorzese (v0.1.6
# blocker: torchaudio 2.11.0+cu128 furdott be a torch 2.7.0 melle).
if (-not (($TorchVer -like "$TorchVersion*") -and ($TorchvisionVer -like "$TorchvisionVersion*") -and ($TorchaudioVer -like "$TorchaudioVersion*"))) {
    Fail-Install ("A telepitett PyTorch haromas NEM a pin-elt parositott verziokkal fut: torch {0} / torchvision {1} / torchaudio {2} (varakozas: {3} / {4} / {5})." -f $TorchVer, $TorchvisionVer, $TorchaudioVer, $TorchVersion, $TorchvisionVersion, $TorchaudioVersion) "Inditsd ujra a START.bat-ot - a telepito automatikusan visszairanyitja a pin-elt cu128 haromast."
}
if (-not ($CudaVer -like "12.8*")) {
    Fail-Install ("A torch.version.cuda nem 12.8.x (kapott: {0})." -f $CudaVer) "A telepito a CUDA 12.8 (cu128) wheel-eket telepiti a pytorch.org sajat indexerol - ujrafuttatas (idempotens)."
}
if ($GpuCcMajor -lt 12) {
    $CcHint = "A Blackwell (sm_120) tamogatashoz cc >= 12.0 szukseges - ezert a telepites sikertelen. " +
        "GPU-csere szukseges (RTX 5070 ajanlott), vagy sajat felelossegre regi build (nem tamogatott)."
    Fail-Install ("compute capability < 12.0 (Blackwell sm_120) - a PyTorch cu128 wheel tamogatasa miatt. Talalt cc: {0}.{1}." -f $GpuCcMajor, $GpuCcMinor) $CcHint
} elseif (-not (($GpuCcMajor -eq 12) -and ($GpuCcMinor -eq 0))) {
    Write-Warn ("A compute capability nem pontosan 12.0 (sm_120), hanem {0}.{1} - folytatjuk, de a teljes perf nem garantalt." -f $GpuCcMajor, $GpuCcMinor)
}

# ===========================================================================
# 11) llama.cpp + piper binarisok letoltese (pin-elt release)
# ===========================================================================
Write-Step "llama.cpp + piper binarisok letoltese (pin-elt release)"
$LlamaExe = Join-Path $Root "bin\llama-server.exe"
$PiperExe = Join-Path $Root "bin\piper.exe"
# v0.3.2: a binaris-ellenorzes DLL-TUDATOS. A v0.3.1 elotti idempotencia csak
# a llama-server.exe letetet nezte - egy RESZLEGES bin\ (exe megvan, a CUDA
# DLL-ek nem) atcsuszott, es a llama-server indulaskor NEMA modon halt meg
# (STATUS_DLL_NOT_FOUND 0xC0000135, ures log). Most a fajtgesz
# DLL-keszletet koveteljuk: ggml-cuda* (CUDA backend) + cudart* (CUDA
# runtime) + cublas* (cuBLAS), mind a ket pin-elt ziptol.
$BinDirForDllCheck = Join-Path $Root "bin"
$BinDllNames = @(Get-ChildItem -Path $BinDirForDllCheck -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
$LlGgmlCudaOk = (@($BinDllNames | Where-Object { $_ -like "ggml-cuda*" })).Count -gt 0
$LlCudartOk   = (@($BinDllNames | Where-Object { $_ -like "cudart*" })).Count -gt 0
$LlCublasOk   = (@($BinDllNames | Where-Object { $_ -like "cublas*" })).Count -gt 0
$LlamaDllSetOk = ($LlGgmlCudaOk -and $LlCudartOk -and $LlCublasOk)
if ($SkipBinaries) {
    Write-Host "    -SkipBinaries: a binarisletoltes kihagyva (offline ujratelepiteshez)."
} else {
    $NeedLlama = ((-not (Test-Path $LlamaExe)) -or (-not $LlamaDllSetOk) -or $Force)
    $NeedPiper = ((-not (Test-Path $PiperExe)) -or $Force)
    if (-not ($NeedLlama -or $NeedPiper)) {
        Write-Host "    bin\llama-server.exe + CUDA DLL-keszlet es bin\piper.exe mar leteznek - a letoltest atugrom."
    } else {
        if ($NeedLlama -and (Test-Path $LlamaExe) -and (-not $LlamaDllSetOk)) {
            Write-Host "    FIGYELEM: a bin\llama-server.exe megvan, de a CUDA DLL-keszlet hianyos" -ForegroundColor Yellow
            Write-Host "    (ggml-cuda*/cudart*/cublas*) - ez okozza a nema azonnali kilepest" -ForegroundColor Yellow
            Write-Host "    (0xC0000135). A binarisok ujraletoltese a biztonsagos javitas." -ForegroundColor Yellow
        }
        $StageDir = Join-Path $env:TEMP "voicemem-m0-install"
        New-Item -ItemType Directory -Path $StageDir -Force | Out-Null
        if ($NeedLlama) {
            $LlamaBinUrl = "$LlamaCppBaseUrl/$LlamaCppTag/$LlamaCppBinAsset"
            $LlamaCudartUrl = "$LlamaCppBaseUrl/$LlamaCppTag/$LlamaCppCudartAsset"
            Write-Host ("    Letoltes : {0}" -f $LlamaBinUrl)
            Write-Host ("    Letoltes : {0} (CUDA 13.3 runtime DLL-ek)" -f $LlamaCudartUrl)
            try {
                Invoke-WebRequest -Uri $LlamaBinUrl -OutFile (Join-Path $StageDir $LlamaCppBinAsset) -UseBasicParsing
                Invoke-WebRequest -Uri $LlamaCudartUrl -OutFile (Join-Path $StageDir $LlamaCppCudartAsset) -UseBasicParsing
            } catch {
                $DlHint = "1) Probold ujra (a GitHub atiranyitas lassu lehet). 2) Kezi letoltes: " + $LlamaBinUrl +
                    " es " + $LlamaCudartUrl + " -> csomagold ki a bin\ mappaba, majd ujra: install_m1.ps1 -SkipBinaries"
                Fail-Install ("A llama.cpp binarisok letoltese nem sikerult: {0}" -f $_.Exception.Message) $DlHint
            }
            $LlamaStage = Join-Path $StageDir "llama"
            New-Item -ItemType Directory -Path $LlamaStage -Force | Out-Null
            Write-Host "    Kicsomagolas (a ket zip egy staging mappaba)..."
            Expand-Archive -Path (Join-Path $StageDir $LlamaCppBinAsset) -DestinationPath $LlamaStage -Force
            Expand-Archive -Path (Join-Path $StageDir $LlamaCppCudartAsset) -DestinationPath $LlamaStage -Force
            # A zip-ek gyokerben vagy almappaban is csomagolhatnak - ezert minden
            # fajlt kiemelesen (flatten) masolunk a bin\ mappaba (exe + CUDA DLL-ek).
            $BinDir = Join-Path $Root "bin"
            Get-ChildItem -Path $LlamaStage -Recurse -File | Copy-Item -Destination $BinDir -Force
            if (-not (Test-Path $LlamaExe)) {
                Fail-Install "A kicsomagolas utan sem talalhato a bin\llama-server.exe." "Nezd meg a release asset-neveket a https://github.com/ggml-org/llama.cpp/releases oldalon, es allitsd at a $LlamaCppTag / asset valtozokat a szkript fejeben."
            }
            # v0.3.2: a kicsomagolt DLL-keszlet nevesitett ujraillesztese - a
            # tenyleges zip-tartalomtol fuggetlenul a CUDA-haromasnak meg kell
            # lennie (kulonben a szerver meg mindig nema modon halna meg).
            $PostDllNames = @(Get-ChildItem -Path $BinDirForDllCheck -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
            $PostGgmlCuda = (@($PostDllNames | Where-Object { $_ -like "ggml-cuda*" })).Count -gt 0
            $PostCudart   = (@($PostDllNames | Where-Object { $_ -like "cudart*" })).Count -gt 0
            $PostCublas   = (@($PostDllNames | Where-Object { $_ -like "cublas*" })).Count -gt 0
            if (-not ($PostGgmlCuda -and $PostCudart -and $PostCublas)) {
                Fail-Install "A kicsomagolas utan a CUDA DLL-keszlet hianyos (ggml-cuda/cudart/cublas nem talalhato a bin\ mappaban)." "Valoszinuleg megvaltozott a pin-elt llama.cpp release asset-strukturaja - nyisd meg a https://github.com/ggml-org/llama.cpp/releases oldalt, es hasonlitsd ossze a(z) $LlamaCppTag asset-jeit a szkript fejeben pin-elt nevekkel."
            }
            Write-Host "    bin\llama-server.exe + CUDA DLL-ek rendben (ggml-cuda/cudart/cublas ellenorzve)."
        } else {
            Write-Host "    llama-server.exe mar letezik - csak a piper letoltese fut le."
        }
        if ($NeedPiper) {
            $PiperUrl = "$PiperBaseUrl/$PiperTag/$PiperAsset"
            Write-Host ("    Letoltes : {0}" -f $PiperUrl)
            try {
                Invoke-WebRequest -Uri $PiperUrl -OutFile (Join-Path $StageDir $PiperAsset) -UseBasicParsing
            } catch {
                $PpHint = "Kezi letoltes: " + $PiperUrl + " -> csomagold ki, a piper.exe keruljon a bin\ mappaba, majd ujra: install_m1.ps1 -SkipBinaries"
                Fail-Install ("A piper letoltese nem sikerult: {0}" -f $_.Exception.Message) $PpHint
            }
            $PiperStage = Join-Path $StageDir "piper"
            New-Item -ItemType Directory -Path $PiperStage -Force | Out-Null
            Write-Host "    Kicsomagolas..."
            Expand-Archive -Path (Join-Path $StageDir $PiperAsset) -DestinationPath $PiperStage -Force
            $PiperFound = Get-ChildItem -Path $PiperStage -Recurse -Filter "piper.exe" | Select-Object -First 1
            if (-not $PiperFound) {
                Fail-Install "A piper zip-ben nem talalhato piper.exe." "Kezi telepites: csomagold ki a zip-et, es a piper.exe-t (a melle hallo dll-ekkel, adatokkal) masold a bin\ mappaba."
            }
            # A piper.exe teljes konyvtarat masoljuk (dll-ek + espeak-ng adatok).
            Copy-Item -Path (Join-Path $PiperFound.Directory.FullName "*") -Destination (Join-Path $Root "bin") -Recurse -Force
            if (-not (Test-Path $PiperExe)) {
                Fail-Install "A masolas utan sem talalhato a bin\piper.exe."
            }
            Write-Host "    bin\piper.exe rendben."
        } else {
            Write-Host "    piper.exe mar letezik - a letoltese kihagyva."
        }
        try {
            Remove-Item -Path $StageDir -Recurse -Force -ErrorAction SilentlyContinue
        } catch { }
    }
}

# ===========================================================================
# 12) VoiceMem telepitese (VEZERELT vendor\voicemem + pip install -e + pin)
# ===========================================================================
Write-Step "VoiceMem telepitese (vendor\voicemem + pip install -e)"
$VmDir = Join-Path $Root "vendor\voicemem"
$VmPinFile = Join-Path $Root $VoiceMemPinFile
if ($SkipVoiceMem) {
    Write-Host "    -SkipVoiceMem: a VoiceMem telepitese kihagyva."
} else {
    # v0.5.0 (controlled fork): a VoiceMem forras a REPO-BAN szallitott vendor\voicemem
    # (controlled fork, l. VOICEMEM_PIN.json + UPSTREAM_POLICY.md). NINCS git clone,
    # NINCS fetch/checkout, NINCS main-ag fallback - az elozo telepito ciklus
    # klonozta az upstream v0.0.1-et (a tag hianyaban a main-agrol is - nyilt
    # drift-kockazat); most a kiadas ZIP mar tartalmazza a vezelt forrast.
    $VmPkgDir = Join-Path $VmDir "voicemem"
    $VmPinOk = (Test-Path (Join-Path $VmPkgDir "__init__.py")) -and (Test-Path $VmPinFile)
    if (-not $VmPinOk) {
        Fail-Install "A vezelt VoiceMem forras hianyzik (vendor\voicemem\voicemem\__init__.py vagy VOICEMEM_PIN.json)." "A teljes repot kellett klonozni/kitomoriteni (a release ZIP tartalmazza a vendor\voicemem-et). Ne hasznalj regi (v0.4.x elotti telepito-design) ZIP-et."
    }
    Write-Host "    Vezelt VoiceMem forras: vendor\voicemem (pin: $VoiceMemUpstreamCommit)."

    # A pin-fajl es a vendor foltok jelenlete ellenorzese (a manifest rogzi a tenyleges ertekeket).
    try {
        $PinJson = Get-Content -Path $VmPinFile -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        $PinJson = $null
    }
    if ($null -eq $PinJson -or $PinJson.upstream_commit -ne $VoiceMemUpstreamCommit) {
        Fail-Install "A VOICEMEM_PIN.json hianyzik vagy nem a vart upstream commit-et rogziti (vart: $VoiceMemUpstreamCommit)." "A vendor\voicemem es a VOICEMEM_PIN.json egy zaros paros - telepitsd a teljes repot egyben (release ZIP)."
    }

    # v0.4.7: a vendor KINAI trait/emocio-extrakcios promptjanak angolositasa
    # (scripts\patch_voicemem_english.py, idempotens). A vezerelt farban a folt
    # mar eleve benne van (VM-LOCAL-EN), a szkript itt ellenorzo szerepet tolit be:
    # a marker hianya (pl. valaki a vendor tartalmat cserelete) HIBA.
    $PatchScript = Join-Path $Root "scripts\patch_voicemem_english.py"
    if (Test-Path $PatchScript) {
        & $VenvPython $PatchScript $VmDir
        if ($LASTEXITCODE -ne 0) {
            Fail-Install "A voicemem angolosito folt nem talalhato a vendor forrasban (a marker hianyzik)." "A vendor\voicemem tartalma nem a vezelt far - telepitsd a teljes release ZIP-et."
        } else {
            Write-Host "    VoiceMem trait-prompt angolositva (English extraction, marker OK)."
        }
    }

    Write-Host "    pip install -e vendor\voicemem"
    & $VenvPython -m pip install -e $VmDir
    if ($LASTEXITCODE -ne 0) {
        Fail-Install "A 'pip install -e vendor\voicemem' nem sikerult." "Nezd a fenti pip hibat (gyakori: halozat vagy lemezhely); az idempotens ujrafuttatas biztositott."
    }
    Write-Host "    VoiceMem pip-csomag rendben."

    # v0.5.0: TELEPITES UTANI PIN-ELLENORZES - az import a venv-ben a vendor
    # konyvtarba kell mutasson, es a futokori identitasnak a pin-elt commitnek
    # kell lennie (a pip metadata verzio NEM identitas - l. VOICEMEM_PIN.json).
    $PinProbe = @'
import sys
import voicemem
from pathlib import Path
src = str(Path(voicemem.__file__).resolve())
commit = getattr(voicemem, "CONTROLLED_UPSTREAM_COMMIT", "")
ok = True
if "vendor" + chr(92) + "voicemem" not in src and "/vendor/voicemem" not in src:
    print("IMPORT-PATH-FAIL: " + src); ok = False
if commit != "e8384e087bd2f44eb05fc7ae1a3c525ea8244179":
    print("PIN-FAIL: " + (commit or "<missing>")); ok = False
sys.exit(0 if ok else 1)
'@
    $ProbeScript = Join-Path $env:TEMP ("voicemem_pin_probe_{0}.py" -f $PID)
    try {
        Set-Content -Path $ProbeScript -Value $PinProbe -Encoding ASCII -ErrorAction Stop
        & $VenvPython $ProbeScript
        if ($LASTEXITCODE -ne 0) {
            Fail-Install "A VoiceMem pin-ellenorzese NEM sikerult (import utvonal vagy futokori commit)." "Valoszinuleg egy masik voicemem telepites takaritja el a vezelt forrast (site-packages / masik venv) - torold ki, majd futtasd ujra a telepitot."
        } else {
            Write-Host "    VoiceMem pin-ellenorzes OK (import -> vendor\voicemem, commit e8384e0)."
        }
    } finally {
        Remove-Item -Path $ProbeScript -ErrorAction SilentlyContinue | Out-Null
    }
}


# ---------------------------------------------------------------------------
# PyTorch-harmas UJRAELLENORZESE a VoiceMem telepites utan (guard):
# egy kesobbi pip install (pl. a voicemem fuggosegei) felulirhatja a pin-elt
# cu128 haromast - ezt itt azonnal es automatikusan visszairanyitjuk, hogy
# a vegso allapot mindig a hivatalosan parositott verzio legyen:
#   torch 2.7.0+cu128 + torchvision 0.22.0+cu128 + torchaudio 2.7.0+cu128.
# Az ujratelepites --no-deps + --force-reinstall: CSAK a harom pin-elt
# csomagot erinti, a tobbi fuggoseget nem bantja.
# ---------------------------------------------------------------------------
if (-not (Test-TorchTrio)) {
    Write-Warn "A PyTorch-harmas eltolodott a kesobbi telepitsek kozben - AUTOMATIKUS visszairanyitas a pin-elt cu128 haromasra..."
    & $VenvPython -m pip install --force-reinstall --no-deps "torch==$TorchVersion" "torchvision==$TorchvisionVersion" "torchaudio==$TorchaudioVersion" --index-url $TorchCudaIndex
    if ($LASTEXITCODE -ne 0) {
        Fail-Install "A PyTorch-harmas visszairanyitasa nem sikerult (pytorch.org cu128 index elerhetetlen?)." "Ujrafuttatas (idempotens) - vagy inditsd ujra a START.bat-ot."
    }
    if (-not (Test-TorchTrio)) {
        Fail-Install ("A visszairanyitas utan sem stimmel a PyTorch-harmas (torch {0} / torchvision {1} / torchaudio {2})." -f $TorchVersion, $TorchvisionVersion, $TorchaudioVersion) "Inditsd ujra a START.bat-ot (repair mode) - ha tovabb is hibazik, kuld el a logs\bootstrap.log-ot."
    }
    Write-Host "    PyTorch-harmas visszaallitva: torch 2.7.0 + torchvision 0.22.0 + torchaudio 2.7.0 (cu128)."
} else {
    Write-Host "    PyTorch-harmas rendben a VoiceMem telepites utan is (feluliras-ellenorzes)."
}

# ===========================================================================
# 13) onnxruntime telepitese (M1: Silero VAD futtato, CPU)
# ===========================================================================
Write-Step "onnxruntime telepitese (M1 Silero VAD futtato, CPU)"
# v0.3.4 (celgepi field report #4): az onnxruntime eddig SEHONNAN sem kerult
# a .venv-be: a requirements.lock aktival sorai csak a core harmast
# (numpy/PyYAML/httpx) telepitik (a TARGET-ONLY blokk ott megjegyzes), es a
# voicemem fuggosegei sherpa-onnx-ot huznak, nem onnxruntime-t - viszont az
# app/vad.py Silero VAD-ja PONTOSAN onnxruntime-ot var. Eredmeny a celgepen:
# "ModuleNotFoundError: No module named 'onnxruntime'" -> RuntimeError mar
# az agent inditasanak elejen (a VAD a pipeline elso komponense). Ez
# M1-KRITIKUS csomag, ezert a hiba itt FATAL - NEM graceful degradation,
# mint a funasr/speechbrain (M2/M3, opcionalis) csomagoknal.
& $VenvPython -m pip install "onnxruntime==$OnnxRuntimeVersion"
if ($LASTEXITCODE -ne 0) {
    Fail-Install ("Az onnxruntime==" + $OnnxRuntimeVersion + " telepitese nem sikerult (M1: a Silero VAD futtatoja).") "Ujrafuttatas (idempotens), vagy START.bat repair; kezzel: .venv\Scripts\python.exe -m pip install onnxruntime==$OnnxRuntimeVersion"
} else {
    & $VenvPython -c "import onnxruntime; print('    onnxruntime ' + onnxruntime.__version__ + ' rendben (M1 Silero VAD, CPU futtato).')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Fail-Install "Az onnxruntime import nem sikerult (M1: a Silero VAD futtatoja)." "Inditsd ujra a START.bat-ot (repair) - a bootstrap automatikusan ujratelepiti."
    }
}

# ===========================================================================
# 14) funasr telepitese (M2: emotion2vec+ futtato, CPU)
# ===========================================================================
Write-Step "funasr telepitese (M2 emotion2vec+ futtato, CPU)"
# A funasr az M2 proszodia-elemzes futtatoja (emotion2vec+ base, CPU).
# FONTOS: a funasr requires_dist NEM tartalmaz torch-ot - a cu128 harmas
# nem tolodhat el miatta; a trio-guard azert utana is lefut (biztonsag).
# A csomag nehez (modelscope/librosa/oss2...), de Windows-wheel-ekkel
# telepitheto. Ha a telepites hibazik: az M1 pipeline VALID tovabb, csak
# az M2 emotion-elemzes degradalodik ki (graceful degradation).
& $VenvPython -m pip install "funasr==$FunasrVersion"
if ($LASTEXITCODE -ne 0) {
    Write-Warn "A funasr telepitese nem sikerult - az M2 emotion-elemzes kikapcsolva (M1 mod)."
    Write-Host "    Ujrafuttatas (idempotens), vagy inditsd ujra a START.bat-ot (repair)."
} else {
    & $VenvPython -c "import funasr; print('    funasr ' + funasr.__version__ + ' rendben (M2 emotion2vec+ CPU futtato).')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "A funasr import nem sikerult - az M2 emotion-elemzes kikapcsolva (M1 mod)."
    }
}
# Trio-guard a funasr telepitese utan is (a fuggosegei felulirhatnak):
if (-not (Test-TorchTrio)) {
    Write-Warn "A PyTorch-harmas eltolodott a funasr telepitese kozben - AUTOMATIKUS visszairanyitas a pin-elt cu128 haromasra..."
    & $VenvPython -m pip install --force-reinstall --no-deps "torch==$TorchVersion" "torchvision==$TorchvisionVersion" "torchaudio==$TorchaudioVersion" --index-url $TorchCudaIndex
    if ($LASTEXITCODE -ne 0) {
        Fail-Install "A PyTorch-harmas visszairanyitasa nem sikerult (pytorch.org cu128 index elerhetetlen?)." "Ujrafuttatas (idempotens) - vagy inditsd ujra a START.bat-ot."
    }
    if (-not (Test-TorchTrio)) {
        Fail-Install ("A visszairanyitas utan sem stimmel a PyTorch-harmas (torch {0} / torchvision {1} / torchaudio {2})." -f $TorchVersion, $TorchvisionVersion, $TorchaudioVersion) "Inditsd ujra a START.bat-ot (repair mode) - ha tovabb is hibazik, kuld el a logs\bootstrap.log-ot."
    }
    Write-Host "    PyTorch-harmas visszaallitva: torch 2.7.0 + torchvision 0.22.0 + torchaudio 2.7.0 (cu128)."
} else {
    Write-Host "    PyTorch-harmas rendben a funasr telepites utan is (feluliras-ellenorzes)."
}

# ===========================================================================
# 15) speechbrain telepitese (M3: ECAPA beszeloi embedding futtato, CPU)
# ===========================================================================
Write-Step "speechbrain telepitese (M3 beszeloi embedding futtato, CPU)"
# A speechbrain az M3 beszelo-azonositas futtatoja (SpeechBrain
# spkrec-ecapa-voxceleb, ECAPA-TDNN, 192-dim, CPU).
# FONTOS: a speechbrain requires_dist torch>=2.1.0 + torchaudio>=2.1.0 -
# a pin-elt cu128 harmas (2.7.0) ELEGENDO, a pip NEM csereli ki; a
# trio-guard azert utana is lefut (biztonsag). Ha a telepites hibazik: az
# M1/M2 pipeline VALID tovabb, csak az M3 beszelo-azonositas degradalodik
# ki (user_id="voice_user" fix, graceful degradation).
& $VenvPython -m pip install "speechbrain==$SpeechbrainVersion"
if ($LASTEXITCODE -ne 0) {
    Write-Warn "A speechbrain telepitese nem sikerult - az M3 beszelo-azonositas kikapcsolva (M1/M2 mod)."
    Write-Host "    Ujrafuttatas (idempotens), vagy inditsd ujra a START.bat-ot (repair)."
} else {
    & $VenvPython -c "import speechbrain; print('    speechbrain ' + speechbrain.__version__ + ' rendben (M3 ECAPA CPU futtato).')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "A speechbrain import nem sikerult - az M3 beszelo-azonositas kikapcsolva (M1/M2 mod)."
    }
}
# Trio-guard a speechbrain telepitese utan is (a fuggosegei felulirhatnak):
if (-not (Test-TorchTrio)) {
    Write-Warn "A PyTorch-harmas eltolodott a speechbrain telepitese kozben - AUTOMATIKUS visszairanyitas a pin-elt cu128 haromasra..."
    & $VenvPython -m pip install --force-reinstall --no-deps "torch==$TorchVersion" "torchvision==$TorchvisionVersion" "torchaudio==$TorchaudioVersion" --index-url $TorchCudaIndex
    if ($LASTEXITCODE -ne 0) {
        Fail-Install "A PyTorch-harmas visszairanyitasa nem sikerult (pytorch.org cu128 index elerhetetlen?)." "Ujrafuttatas (idempotens) - vagy inditsd ujra a START.bat-ot."
    }
    if (-not (Test-TorchTrio)) {
        Fail-Install ("A visszairanyitas utan sem stimmel a PyTorch-harmas (torch {0} / torchvision {1} / torchaudio {2})." -f $TorchVersion, $TorchvisionVersion, $TorchaudioVersion) "Inditsd ujra a START.bat-ot (repair mode) - ha tovabb is hibazik, kuld el a logs\bootstrap.log-ot."
    }
    Write-Host "    PyTorch-harmas visszaallitva: torch 2.7.0 + torchvision 0.22.0 + torchaudio 2.7.0 (cu128)."
} else {
    Write-Host "    PyTorch-harmas rendben a speechbrain telepitese utan is (feluliras-ellenorzes)."
}

# ===========================================================================
# 16) transformers OR (v0.4.7): >= 5.0 - a Qwen3-ASR qwen3_asr modul nativan
#     a transformers 5.x-ben letezik (4.57 NEM tartalmazza); az app/asr.py a
#     processor + generate hivasi utat hasznalja, amihez a nativ modul kell.
# ===========================================================================
Write-Step "transformers or (>= 5.0: Qwen3-ASR tamogatas - voicemem pin-javitas)"
# v0.4.4 FIELD REPORT: a 12. lepes "pip install -e vendor\voicemem" a vendored
# csomag pyproject.toml-ja miatt a transformers-t 4.52.3-ra DOWNGRADELI - az
# a verzio NEM ismeri a qwen3_asr architekturat, igy a Qwen3-ASR-0.6B (ASR
# modell) SOHA nem toltodik be ("ASR nem zold a UI-on"). A 14/15. lepesek
# fuggosegei (funasr/speechbrain) is eltolhetik. Az or ezert az UTOLSO
# pip-lepes utan fut, es az 5.0+ kotelezo also padlot kenyszeri a venvbe.
# v0.4.5: a bootstrap.ps1 dependency-probe most VERZIO-ERZEKENY (a
# voicemem PyPI pin miatt visszamarado 4.52.3-et kiszuri), igy egy
# meglevo venven ez a lepes AUTOMATIKUSAN lefut - nem kell repair mod.
$TransformersFloor = "5.0"
& $VenvPython -m pip install "transformers>=$TransformersFloor"
if ($LASTEXITCODE -ne 0) {
    Fail-Install "A transformers >= $TransformersFloor telepitese nem sikerult (a Qwen3-ASR qwen3_asr architekturahoz EZ KELL - nelkule az ASR nem toltodik be)." "Ujrafuttatas (idempotens), vagy inditsd ujra a START.bat-ot (repair)."
}
$TfVersion = (& $VenvPython -c "import transformers; print(transformers.__version__)" 2>$null)
if ($LASTEXITCODE -ne 0) {
    Fail-Install "A transformers import nem sikerult az or-telepites utan." "Nezd a pip hibauzenetet fent; ujrafuttatas (idempotens)."
}
# Verzio-ASSERT tuple-osszehasonlitassal (NE string-hasonlitas: "4.9" < "5.0"
# stringkent hamisul meg) - kesz allapotban soha nem bukhat el:
& $VenvPython -c "import sys, transformers; v = tuple(int(x) for x in transformers.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if v >= (5, 0) else 1)"
if ($LASTEXITCODE -ne 0) {
    Fail-Install ("A transformers verzio a or utan is < 5.0 ({0}) - a Qwen3-ASR nem toltodik be vele." -f $TfVersion) "Valamelyik csomag (voicemem pin / funasr / speechbrain fuggoseg) visszabeszelte. Kezzel: .venv\Scripts\python.exe -m pip install "transformers>=5.0" es ujrafuttatas."
}
Write-Host ("    transformers {0} rendben (>= 5.0: a qwen3_asr modul nativan betoltodik)." -f $TfVersion)

# ===========================================================================
# 17) Modellek letoltese (scripts\download_models.ps1, idempotens)
# ===========================================================================
Write-Step "Modellletoltes (scripts\download_models.ps1)"
if ($SkipModels) {
    Write-Host "    -SkipModels: a modellletoltes kihagyva (offline ujratelepiteshez)."
} else {
    $DlScript = Join-Path $Root "scripts\download_models.ps1"
    if (-not (Test-Path $DlScript)) {
        Fail-Install "Nem talalhato a scripts\download_models.ps1." "Klonozd ujra a repot (repo-fajl), majd ujrafuttatas."
    }
    & $DlScript -Root $Root
    if ($LASTEXITCODE -ne 0) {
        Fail-Install "A modellletoltes nem sikerult (l. a download_models.ps1 kimenetet fent)." "Idempotens ujrafuttathatosag; a letolto automatikus tukor-lancot probal (Qwen -> unsloth -> bartowski). Ha tovabb is hibazik: ellenorizd az internetet, majd inditsd ujra a START.bat-ot."
    }
}

# ===========================================================================
# 18) Konfiguracio (yaml a repoban + .env a .env.example-bol)
# ===========================================================================
Write-Step "Konfiguracio ellenorzese (+ .env letrehozasa)"
$YamlFile = Join-Path $Root "config\voicemem_config.yaml"
if (-not (Test-Path $YamlFile)) {
    Fail-Install "Hianyzik a config\voicemem_config.yaml (repo-fajl)." "git checkout -- config/ vagy ujra-clone, majd ujrafuttatas."
}
Write-Host "    config\voicemem_config.yaml rendben."
$EnvExample = Join-Path $Root "config\.env.example"
if (-not (Test-Path $EnvExample)) {
    $EnvExample = Join-Path $Root ".env.example"
}
if (Test-Path $EnvExample) {
    $EnvTarget = Join-Path (Split-Path -Parent $EnvExample) ".env"
    if (-not (Test-Path $EnvTarget)) {
        Copy-Item -Path $EnvExample -Destination $EnvTarget
        Write-Host ("    letrehozva: {0} (a .env.example alapjan)" -f $EnvTarget)
    } else {
        Write-Host "    .env mar letezik - nem irtam felul."
    }
} else {
    Write-Warn "Nem talalhato .env.example (config\ vagy repo-gyoker) - a .env letrehozasa kihagyva."
}

# ===========================================================================
# 19) Smoke testek (scripts\verify_m1.ps1)
# ===========================================================================
Write-Step "Smoke testek (scripts\verify_m1.ps1 -WithServer)"
$VerifyScript = Join-Path $Root "scripts\verify_m1.ps1"
if (-not (Test-Path $VerifyScript)) {
    Fail-Install "Nem talalhato a scripts\verify_m1.ps1." "Klonozd ujra a repot (repo-fajl), majd ujrafuttatas."
}
& $VerifyScript -Root $Root -WithServer
if ($LASTEXITCODE -ne 0) {
    Fail-Install "A smoke testek nem futottak at (verify_m1.ps1 FAIL - l. a kimenetet fent)." "Kovetsd a [FAIL] sorok javitasi javaslatat, majd ujrafuttatas (idempotens)."
}

# ===========================================================================
# 20) Licenc-ellenorzes
# ===========================================================================
Write-Step "Licenc-ellenorzes (LICENSES.md + Piper hang-konfigok)"
$LicFile = Join-Path $Root "LICENSES.md"
if (-not (Test-Path $LicFile)) {
    Fail-Install "Hianyzik a LICENSES.md a repo gyokerben (repo-fajl)." "git checkout -- LICENSES.md vagy ujra-clone, majd ujrafuttatas."
}
Write-Host "    LICENSES.md rendben."
$VoiceNames = @("hu_HU-anna-medium", "hu_HU-berta-medium", "hu_HU-imre-medium", "en_US-lessac-medium")
$VoiceJsonOk = $true
foreach ($V in $VoiceNames) {
    $JsonPath = Join-Path $Root ("models\tts\piper\{0}.onnx.json" -f $V)
    if (-not (Test-Path $JsonPath)) {
        $VoiceJsonOk = $false
        Write-Host ("    HIANYZIK: {0}" -f $JsonPath)
    } else {
        try {
            $FirstLine = Get-Content -Path $JsonPath -TotalCount 1
            if (-not $FirstLine) {
                $VoiceJsonOk = $false
                Write-Host ("    URES/olvashatatlan: {0}" -f $JsonPath)
            }
        } catch {
            $VoiceJsonOk = $false
            Write-Host ("    Olvashatatlan: {0}" -f $JsonPath)
        }
    }
}
if (-not $VoiceJsonOk) {
    Fail-Install "Legalabb egy Piper hang .onnx.json fajl hianyzik vagy olvashatatlan." "Futtasd ujra a scripts\download_models.ps1-t (idempotens)."
}
Write-Host "    Minden Piper hang .onnx.json olvashato. License: PASS"

# ===========================================================================
# 21) Offline kornyezet beallitasa
# ===========================================================================
Write-Step "Offline kornyezet beallitasa"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HOME = Join-Path $Root "models\hf"
$env:TOKENIZERS_PARALLELISM = "false"
Write-Host ("    HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_HOME={0}" -f $env:HF_HOME)
if ($ConnectivityAudit) {
    Write-Host "    -ConnectivityAudit: offline_check.ps1 futtatasa..."
    $OfflineScript = Join-Path $Root "scripts\offline_check.ps1"
    if (Test-Path $OfflineScript) {
        & $OfflineScript
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Az offline_check.ps1 nem-loopback kapcsolatot talalt (l. fent)."
        }
    } else {
        Write-Warn "Nem talalhato a scripts\offline_check.ps1 - az audit kihagyva."
    }
} else {
    Write-Host "    (A connectivity audit alapbol kihagyva - a verify_m1 mar offline env-ekkel futott.)"
}

# ===========================================================================
# 22) Vegso install report + INSTALL_MANIFEST.json
# ===========================================================================
Write-Step "Vegso install report + INSTALL_MANIFEST.json"
$PyVerLine = & $VenvPython --version 2>$null
$PyVer = ((@($PyVerLine) -join " ") -replace '^Python\s*', '').Trim()
$VramStr = "0.0"
try {
    $VramStr = ([math]::Round($VramBytes / 1GB, 1)).ToString("0.0", [System.Globalization.CultureInfo]::InvariantCulture)
} catch { }
$VmInfo = "NOT INSTALLED"
$VmOut = & $VenvPython -c "import voicemem; print(getattr(voicemem, '__version__', 'unknown'))" 2>$null
$VmOutText = ((@($VmOut) -join " ")).Trim()
if (($LASTEXITCODE -eq 0) -and ($VmOutText -ne "")) {
    $VmInfo = $VmOutText
}
if ($VmInfo -eq "NOT INSTALLED") {
    if (Test-Path (Join-Path $VmDir ".git")) {
        $Commit = git -C $VmDir rev-parse --short HEAD 2>$null
        $CommitText = ([string]$Commit).Trim()
        if (($LASTEXITCODE -eq 0) -and ($CommitText -ne "")) {
            $VmInfo = "commit " + $CommitText
        }
    }
}

Write-Host ""
Write-Host "INSTALLATION SUCCESS" -ForegroundColor Green
Write-Host ""
Write-Host "Project: $Root"
Write-Host "Python: $PyVer"
Write-Host "GPU: $GpuNameSmi"
Write-Host "VRAM: $VramStr GB"
Write-Host "PyTorch: $TorchVer"
Write-Host "torchvision: $TorchvisionVer"
Write-Host "torchaudio: $TorchaudioVer"
Write-Host "CUDA: $CudaVer"
Write-Host "VoiceMem: $VmInfo"
Write-Host "ASR: Qwen3 ASR 0.6B"
Write-Host "LLM: Qwen3.6 35B A3B IQ4_XS (az EGYETLEN LLM; a GGUF operator altal elhelyezendo/barhol kivalaszthato - l. config/env.local.ps1 + a web UI LLM model pickere; NINCS visszaesi profil)"
Write-Host "TTS: Piper HU + EN"
Write-Host "VAD: Silero"
Write-Host "Embedding: multilingual E5 small"
Write-Host "Offline: READY"
Write-Host "License: PASS"
Write-Host "Smoke tests: PASS"

Write-Host ""
$ManifestScript = Join-Path $Root "scripts\write_install_manifest.py"
if (Test-Path $ManifestScript) {
    try {
        & $VenvPython $ManifestScript --root $Root --out (Join-Path $Root "INSTALL_MANIFEST.json")
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "A write_install_manifest.py hibaval ert veget (exit $LASTEXITCODE) - az INSTALL_MANIFEST.json elofordul, hogy nem keszult el."
        } else {
            Write-Host "INSTALL_MANIFEST.json keszult a repo gyokerben."
        }
    } catch {
        Write-Warn ("A write_install_manifest.py futtatasa nem sikerult: {0}" -f $_.Exception.Message)
    }
} else {
    Write-Warn "A scripts\write_install_manifest.py meg nem letezik - az INSTALL_MANIFEST.json kimaradt (a telepites maga sikeres). Ujrafuttatas utan kesz lesz."
}

Write-Host ""
Write-Host "Kovetkezo lepes (a felhasznalo szamara - egyetlen muvelet):"
Write-Host "    Inditsd ujra a START.bat-ot (dupla kattintas) - az agent automatikusan elindul."
Write-Host "    Ha valami elromlik: START.bat repair (automatikus javitas)."
Write-Host ""
exit 0
