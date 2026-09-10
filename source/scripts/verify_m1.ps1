<#
===============================================================================
scripts/verify_m1.ps1 - M0/M1 telepites-ellenorzo (PASS/FAIL)

MIT csinal: soronkent lefuttatja az M0 exit-criteria ellenorzeseket, es minden
sor ele [PASS] / [FAIL] / [WARN] jelzest ir. A vegso osszegzo sorban PASS/FAIL
all (a WARN NEM buktat meg). Kilepesi kod: 0 = PASS, 1 = FAIL.

FONTOS: a szkript a FUTAS ELEJEN maganak allitja be az offline env-valtozokat
(HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_HOME=$Root\models\hf) - igy az
offline viselkedes is tesztelve van (a modellbetoltesek CSAK lokalisan tortennek).

ELLENORZESI SORREND (a felhasznalo M0 specje szerint):
  1.  .venv python verzioja (3.11.x kotelezo)
  2.  pip mukodokepessege
  3.  CUDA/torch GPU smoke test (valodi matrixmuvelet)
  4.  GPU compute capability >= 12.0 (Blackwell sm_120)
  5.  modellfajlok: LLM GGUF (> 1 GB), ASR config.json, silero_vad.onnx,
      e5 config.json, 4 Piper hang (.onnx + .onnx.json)
  6.  VoiceMem import (kotelezo PASS - M0 exit criteria)
  7.  ASR modell betoltes (AutoTokenizer, local_files_only=True)
  8.  LLM szerver (v0.3.3: a /health az ELSODELEGES sikerfeltetel)
      -WithServer kapcsolo: ha a szerver nem fut, ELINDITJA a
      scripts\start_llama_server.ps1-t (minimizalt ablak), majd a
      GET http://127.0.0.1:8080/health-t LEGFELEJEBB 30 s-ig poll-olja.
      HA /health = HTTP 200 -> PASS: ez az elsodleges sikerfeltetel, a
      PowerShell starter wrapper exit kodja NEM az (ha a wrapper kozben
      nem-0 koddal lepett ki, az csak MEGJEGYZES - nem bukta).
      Ha a wrapper kilepett ES nem marad elo llama-server folyamat, a
      verify KOZVETLENUL inditja a bin\llama-server.exe-t a shellbol
      kozvetlenul levezetett konfiguracioval (ugyanazok a flagek:
      -ngl -1, -c 8192, --parallel 1, q8_0 KV cache, --temp 0.7,
      --metrics, --no-webui).
      FAIL CSAK akkor, ha (1) a llama-server folyamat tenylegesen
      leallt ES (2) a /health nem valt elerhetove a timeouton belul.
      A vegso eredmeny a kovetkezo 6 reszjelzot KULON sorban irja ki:
        1. server startup        (indulas / mar fut / fallback)
        2. health                (GET /health -> HTTP 200)
        3. chat completion       (POST /v1/chat/completions)
        4. JSON completion       (response_format json_object)
        5. model loaded          (GET /v1/models)
        6. process alive         (Get-Process llama-server)
      A python-bridge (config -> JSON) hibaja kulon WARNING - NEM
      fatal, amennyiben a shellbol kozvetlenul levezetett llama-server
      konfiguracio sikeresen mukodik. A verify altal inditott szervert
      a vegen LEALLITJA (a kulso inditast nem nyulja).
      Alapmodban (kapcsolo nelkul) csak a health-check fut - ha nem megy,
      a hibauzenet mutatja a javitasi utat.
  9.  Piper HU szintezis (bin\piper.exe -> data\audio\smoke_hu.wav, > 1 KB)
  10. Piper EN szintezis (en_US-lessac-medium -> data\audio\smoke_en.wav)
  11. memory konyvtarak irhatosaga (sqlite/qdrant/backups + write probe)
  12. konfiguracio: AgentConfig.from_yaml + validate()

HASZNALAT:
    .\scripts\verify_m1.ps1
    .\scripts\verify_m1.ps1 -WithServer       # LLM szerver auto-inditas + completion
    .\scripts\verify_m1.ps1 -Root "C:\masik_root"

MEGJEGYZES: a fajl szandekosan ASCII - a PowerShell 5.1 BOM nelkul
Windows-1252-kent olvasna az ekezetes karaktereket.
#>
param(
    [string]$Root = "",
    [switch]$WithServer
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

# --- Offline env-ek beallitasa marad elesen tesztelels celjabol ----------------
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HOME = Join-Path $Root "models\hf"
$env:TOKENIZERS_PARALLELISM = "false"

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

# --- segedfuggvenyek -----------------------------------------------------------
$Script:FailCount = 0
$Script:WarnCount = 0

function Check-Pass {
    param([string]$Name, [string]$Detail = "")
    if ($Detail -ne "") {
        Write-Host ("[PASS] {0} - {1}" -f $Name, $Detail) -ForegroundColor Green
    } else {
        Write-Host ("[PASS] {0}" -f $Name) -ForegroundColor Green
    }
}

function Check-Warn {
    param([string]$Name, [string]$Detail = "")
    $Script:WarnCount = $Script:WarnCount + 1
    if ($Detail -ne "") {
        Write-Host ("[WARN] {0} - {1}" -f $Name, $Detail) -ForegroundColor Yellow
    } else {
        Write-Host ("[WARN] {0}" -f $Name) -ForegroundColor Yellow
    }
}

function Check-Fail {
    param([string]$Name, [string]$Detail = "", [string]$Fix = "")
    $Script:FailCount = $Script:FailCount + 1
    Write-Host ("[FAIL] {0}" -f $Name) -ForegroundColor Red
    if ($Detail -ne "") { Write-Host ("       reszlet : {0}" -f $Detail) }
    if ($Fix -ne "") { Write-Host ("       javitas: {0}" -f $Fix) }
}

function Test-LlamaHealth {
    param([int]$TimeoutSec = 5)
    try {
        $R = Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -Method Get -TimeoutSec $TimeoutSec -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

function Show-LlamaLogTail {
    # v0.3.2: a STARTER SAJAT kimenetet (logs\llama-server.starter.log,
    # Start-Transcript) ELOSZOR mutatjuk - a minimizalt gyerekablak kimenete
    # (a start-szkript tenyleges hibauzenete, pl. hianyzo CUDA DLL) csak itt
    # lathato. Utana a llama-server stdout/stderr logjai (ha nem uresak).
    param([string]$RootPath, [string]$Label, [int]$N = 20)
    foreach ($LogName in @("llama-server.starter.log", "llama-server.err.log", "llama-server.out.log")) {
        $LogPath = Join-Path $RootPath ("logs\" + $LogName)
        if (Test-Path $LogPath) {
            $Tail = @(Get-Content -Path $LogPath -Tail $N -ErrorAction SilentlyContinue)
            if ($Tail.Count -gt 0) {
                Write-Host ("       {0} [{1}] utolso sorai:" -f $Label, $LogName) -ForegroundColor Yellow
                foreach ($Line in $Tail) { Write-Host ("         {0}" -f $Line) }
            }
        }
    }
}

Write-Host ""
Write-Host "==================================================================="
Write-Host (" VoiceMemAgent M1 verify (Root: {0})" -f $Root)
Write-Host "==================================================================="

# ---------------------------------------------------------------------------
# v0.3.2: Mark-of-the-Web eltavolitas a szkriptekrol. A bongeszobol letoltott
# release ZIP-bol (Explorer-kicsomagolas) a .ps1 fajlok Zone.Identifier
# internet-zona jelzest orokolhetnek - egy normal RemoteSigned session
# "not digitally signed" hibaval blokkolja oket kezi inditasnal. A START.bat
# lanca Bypass-szal fut, de kezi `.\scripts\verify_m1.ps1` inditasok miatt
# itt is eltavolitjuk. (Azonos blokk: bootstrap.ps1 + install_m1.ps1.)
# ---------------------------------------------------------------------------
$ScriptsDirToUnblock = Join-Path $Root "scripts"
Get-ChildItem -Path $ScriptsDirToUnblock -Filter "*.ps1" -File -ErrorAction SilentlyContinue |
    Unblock-File -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# 1) .venv python verzio
# ---------------------------------------------------------------------------
if (-not (Test-Path $VenvPython)) {
    Check-Fail ".venv python" ("nem talalhato: {0}" -f $VenvPython) "inditsd ujra a START.bat-ot - a bootstrap automatikusan letrehozza es javitja a kort"
} else {
    $PyVer = & $VenvPython --version 2>$null
    $PyVerText = ((@($PyVer) -join " ")).Trim()
    if (($LASTEXITCODE -eq 0) -and ($PyVerText -match 'Python 3\.11\.')) {
        Check-Pass ".venv python verzio" $PyVerText
    } else {
        Check-Fail ".venv python verzio" ("kapott: '{0}' - pontosan 3.11.x szukseges" -f $PyVerText) "inditsd ujra a START.bat-ot - a bootstrap automatikusan ujraepiti a .venv-t"
    }
}

# ---------------------------------------------------------------------------
# 2) pip mukodokepesseg
# ---------------------------------------------------------------------------
if (Test-Path $VenvPython) {
    $PipVer = & $VenvPython -m pip --version 2>$null
    $PipVerText = ((@($PipVer) -join " ")).Trim()
    if (($LASTEXITCODE -eq 0) -and ($PipVerText -ne "")) {
        Check-Pass "pip mukodokepesseg" $PipVerText
    } else {
        Check-Fail "pip mukodokepesseg" "a .venv pipje nem valaszol" "START.bat repair (automatikus javitas)"
    }
}

# ---------------------------------------------------------------------------
# 3) CUDA / torch GPU smoke test
# ---------------------------------------------------------------------------
$GpuNameTorch = ""
$GpuCcMajor = 0
$GpuCcMinor = 0
$VramBytes = 0
$TorchVer = ""
$CudaVer = ""
if (Test-Path $VenvPython) {
    $GpuSmokeCode = @'
import torch
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
print("CUDA_VERSION=" + str(torch.version.cuda))
print("GPU OK")
'@
    $SmokeOut = & $VenvPython -c $GpuSmokeCode
    $SmokeExit = $LASTEXITCODE
    $SmokeText = ((@($SmokeOut) | ForEach-Object { [string]$_ }) -join " ")
    if ($SmokeExit -eq 0) {
        foreach ($Line in @($SmokeOut)) {
            if ($Line -match '^GPU_NAME=(.+)$') { $GpuNameTorch = $Matches[1].Trim() }
            elseif ($Line -match '^GPU_CC_MAJOR=(\d+)$') { $GpuCcMajor = [int]$Matches[1] }
            elseif ($Line -match '^GPU_CC_MINOR=(\d+)$') { $GpuCcMinor = [int]$Matches[1] }
            elseif ($Line -match '^VRAM_BYTES=(\d+)$') { $VramBytes = [long]$Matches[1] }
            elseif ($Line -match '^TORCH_VERSION=(.+)$') { $TorchVer = $Matches[1].Trim() }
            elseif ($Line -match '^CUDA_VERSION=(.+)$') { $CudaVer = $Matches[1].Trim() }
        }
        Check-Pass "GPU smoke test (torch+CUDA matrixmuvelet)" ("{0} | cc {1}.{2} | {3} GB" -f $GpuNameTorch, $GpuCcMajor, $GpuCcMinor, [math]::Round($VramBytes / 1GB, 1))
    } else {
        Check-Fail "GPU smoke test (torch+CUDA matrixmuvelet)" $SmokeText "START.bat repair (torch cu128 ujratelepites); nvidia-smi ellenorzese"
    }
}

# ---------------------------------------------------------------------------
# 4) GPU compute capability >= 12.0 (Blackwell sm_120)
# ---------------------------------------------------------------------------
if ($SmokeExit -eq 0) {
    if ($GpuCcMajor -lt 12) {
        Check-Fail "GPU compute capability >= 12.0" ("talalt cc: {0}.{1} - a Blackwell (sm_120) es a PyTorch cu128 wheel tamogatasa miatt ez keves" -f $GpuCcMajor, $GpuCcMinor) "GPU-csere szukseges (RTX 5070 ajanlott)"
    } elseif (-not (($GpuCcMajor -eq 12) -and ($GpuCcMinor -eq 0))) {
        Check-Warn "GPU compute capability" ("cc {0}.{1} - nem pontosan 12.0 (sm_120), de >= 12.0, elfogadva" -f $GpuCcMajor, $GpuCcMinor)
    } else {
        Check-Pass "GPU compute capability >= 12.0" ("cc {0}.{1} (Blackwell sm_120)" -f $GpuCcMajor, $GpuCcMinor)
    }
}

# ---------------------------------------------------------------------------
# 5) Modellfajlok
# ---------------------------------------------------------------------------
# --- v0.4.17: a GGUF-ellenorzes a TELJES feloldasi lancot hasznalja
# (config/llm_model.json [web UI kivalasztas / identify_ollama_blob.ps1
# -Select] > LLAMA_MODEL_PATH env > a models\llm\qwen3.6-35b-a3b\
# konyvtarban tenylegesen levo GGUF fajl). A v0.4.16-os megteveszto
# projekt-default UT (nem lezo fajl nevenek kijelzese) megszunt: ha semmi
# nincs konfiguralva, az uzenet azt mondja (a fajl BARMELYIK meghajton
# allhat, a projektben nem kotelezo lennie; a GGUF NEM masolodik be).
function Get-OperatorDefaultGguf {
    param([string]$Dir)
    if (-not (Test-Path -LiteralPath $Dir -PathType Container)) { return "" }
    $found = @()
    try {
        foreach ($f in @(Get-ChildItem -LiteralPath $Dir -File -ErrorAction SilentlyContinue | Sort-Object Name)) {
            try {
                $fs = [System.IO.File]::OpenRead($f.FullName)
                try {
                    $b = New-Object byte[] 4
                    $n = $fs.Read($b, 0, 4)
                    if (($n -eq 4) -and ($b[0] -eq 0x47) -and ($b[1] -eq 0x47) -and ($b[2] -eq 0x55) -and ($b[3] -eq 0x46)) {
                        $found += $f.FullName
                    }
                } finally { $fs.Close() }
            } catch { }
        }
    } catch { }
    if ($found.Count -eq 1) { return $found[0] }
    return ""
}
$LlmGgufSource = "none (no model configured)"
$LlmGguf = ""
$OpDefault = Get-OperatorDefaultGguf (Join-Path $Root "models\llm\qwen3.6-35b-a3b")
if ($OpDefault) {
    $LlmGguf = $OpDefault
    $LlmGgufSource = "project default (GGUF in models\llm\qwen3.6-35b-a3b)"
}
if ($env:LLAMA_MODEL_PATH) {
    $LlmGguf = $env:LLAMA_MODEL_PATH
    $LlmGgufSource = "env profile (LLAMA_MODEL_PATH)"
}
$LlmModelJson = Join-Path $Root "config\llm_model.json"
if (Test-Path $LlmModelJson) {
    try {
        $LlmSel = Get-Content -LiteralPath $LlmModelJson -Raw -Encoding UTF8 | ConvertFrom-Json
        $LlmSelPath = [string]$LlmSel.llm_model_path
        if (-not [string]::IsNullOrWhiteSpace($LlmSelPath)) {
            $LlmGguf = $LlmSelPath
            $LlmGgufSource = "user selection (config/llm_model.json)"
        }
    } catch { }
}
if ([string]::IsNullOrWhiteSpace($LlmGguf)) {
    Check-Fail "LLM GGUF (Qwen3.6 35B A3B IQ4_XS - az EGYETLEN LLM)" "NINCS konfiguralva modell (llm_model.json / LLAMA_MODEL_PATH / models\llm\qwen3.6-35b-a3b\ - semmi)" "v0.4.17 harom ut: (a) a web UI 'LLM model' szekcio Browse... gombja (barmelyik meghajto; Ollama sha256-... blob is; az ut a config/llm_model.json-be kerul), (b) az Ollama-blob azonosito: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\identify_ollama_blob.ps1 -Select (manifest+GGUF-header+digest-ellenorzes; -Select = web UI kivalasztas), (c) az altalanos kereso: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1. NINCS visszaesi profil - a Qwen3.6 az egyetlen LLM."
} elseif (Test-Path -LiteralPath $LlmGguf) {
    $GgufLen = (Get-Item -LiteralPath $LlmGguf).Length
    if ($GgufLen -gt 6GB) {
        Check-Pass "LLM GGUF (Qwen3.6 35B A3B IQ4_XS - az EGYETLEN LLM)" ("{0} ({1} GB; forras: {2})" -f $LlmGguf, [math]::Round($GgufLen / 1GB, 2), $LlmGgufSource)
    } else {
        Check-Fail "LLM GGUF (Qwen3.6 35B A3B IQ4_XS) meret" ("a fajl kevesebb mint 6 GB ({0} byte) - serult masolas" -f $GgufLen) "valaszd ki ujra a fajlt a web UI 'LLM model' szekcioval, vagy az identify_ollama_blob.ps1 -Select kapcsoloval; NINCS visszaesi profil"
    }
} else {
    Check-Fail "LLM GGUF (Qwen3.6 35B A3B IQ4_XS - az EGYETLEN LLM)" ("nem talalhato: {0} (forras: {1})" -f $LlmGguf, $LlmGgufSource) "v0.4.17 harom ut: (a) a web UI 'LLM model' szekcio Browse... gombja (barmelyik meghajto; Ollama sha256-... blob is; az ut a config/llm_model.json-be kerul), (b) az Ollama-blob azonosito: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\identify_ollama_blob.ps1 -Select (manifest+GGUF-header+digest-ellenorzes; -Select = web UI kivalasztas), (c) az altalanos pontos-modell kereso: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1 (Ollama-store / HF-cache / letoltesek). NINCS visszaesi profil - a Qwen3.6 az egyetlen LLM."
}

$AsrConfig = Join-Path $Root "models\asr\qwen3-asr-0.6b\config.json"
if (Test-Path $AsrConfig) {
    Check-Pass "ASR config.json" $AsrConfig
} else {
    Check-Fail "ASR config.json" ("nem talalhato: {0}" -f $AsrConfig) "scripts\download_models.ps1 (M1: Qwen3-ASR-0.6B, NEM 1.7B)"
}

$VadOnnx = Join-Path $Root "models\vad\silero-vad\silero_vad.onnx"
if ((Test-Path $VadOnnx) -and ((Get-Item -LiteralPath $VadOnnx).Length -ge 1048576)) {
    Check-Pass "VAD silero_vad.onnx" $VadOnnx
} else {
    Check-Fail "VAD silero_vad.onnx" ("nem talalhato / kisebb mint 1 MB (serult?): {0}" -f $VadOnnx) "START.bat (a letolto a tphakala re-hostot, majd az eredeti silero repot probalja)"
}

$E5Config = Join-Path $Root "models\embedding\multilingual-e5-small\config.json"
if (Test-Path $E5Config) {
    Check-Pass "Embedding config.json" $E5Config
} else {
    Check-Fail "Embedding config.json" ("nem talalhato: {0}" -f $E5Config) "scripts\download_models.ps1"
}

$PiperDir = Join-Path $Root "models\tts\piper"
$VoiceNames = @("hu_HU-anna-medium", "hu_HU-berta-medium", "hu_HU-imre-medium", "en_US-lessac-medium")
$MissingVoices = @()
foreach ($V in $VoiceNames) {
    foreach ($Ext in @(".onnx", ".onnx.json")) {
        $Vp = Join-Path $PiperDir ("{0}{1}" -f $V, $Ext)
        if (-not (Test-Path $Vp)) {
            $MissingVoices += ("{0}{1}" -f $V, $Ext)
        } elseif (($Ext -eq ".onnx") -and ((Get-Item -LiteralPath $Vp).Length -lt 1048576)) {
            # a MODELS.lock.json-gal osszehangolt 1 MB-os padlo (serult letoltes)
            $MissingVoices += ("{0}{1} (tulkicsi)" -f $V, $Ext)
        }
    }
}
if ($MissingVoices.Count -eq 0) {
    Check-Pass "Piper hangok (4 hang x .onnx + .onnx.json)" $PiperDir
} else {
    Check-Fail "Piper hangok (4 hang x .onnx + .onnx.json)" ("hianyzik: {0}" -f ($MissingVoices -join ", ")) "scripts\download_models.ps1 ujrafuttatasa (idempotens)"
}

# ---------------------------------------------------------------------------
# 6) VoiceMem import + VEZERLT FORRAS + PIN ellenorzes (M0 exit criteria: PASS kotelezo)
#    v0.5.0: nem eleg az import - az importnak a vendor\voicemem forrasra kell
#    mutatnia, es a futokori identitasnak a pin-elt upstream commitre
#    (e8384e087bd2f44eb05fc7ae1a3c525ea8244179) kell allnia. A pip metadata
#    verzio (0.2.3) NEM identitas - l. VOICEMEM_PIN.json.
# ---------------------------------------------------------------------------
if (Test-Path $VenvPython) {
    $VmOut = & $VenvPython -c "import voicemem; print(getattr(voicemem, '__version__', 'unknown'))" 2>$null
    $VmText = ((@($VmOut) -join " ")).Trim()
    if (($LASTEXITCODE -eq 0) -and ($VmText -ne "")) {
        Check-Pass "VoiceMem import" $VmText
    } else {
        Check-Fail "VoiceMem import" "a 'import voicemem' nem sikerult a .venv-ben (a M0 exit criteria szerint PASS kotelezo)" "START.bat repair (a Voicemem telepites ujrafuttatasa)"
    }
    # 6b) a VEZERLT forras-utvonal es a pin (controlled fork)
    $VmProbe = @'
import sys
import voicemem
from pathlib import Path
src = str(Path(voicemem.__file__).resolve())
commit = getattr(voicemem, "CONTROLLED_UPSTREAM_COMMIT", "")
fork = bool(getattr(voicemem, "CONTROLLED_FORK", False))
ok = True
if not (fork and "vendor" + chr(92) + "voicemem" in src or "/vendor/voicemem" in src):
    print("SRC:" + src); ok = False
if commit != "e8384e087bd2f44eb05fc7ae1a3c525ea8244179":
    print("COMMIT:" + (commit or "<missing>")); ok = False
sys.exit(0 if ok else 1)
'@
    $VmProbeScript = Join-Path $env:TEMP ("voicemem_verify_pin_{0}.py" -f $PID)
    try {
        Set-Content -Path $VmProbeScript -Value $VmProbe -Encoding ASCII -ErrorAction Stop
        $VmPinOut = & $VenvPython $VmProbeScript 2>$null
        if ($LASTEXITCODE -eq 0) {
            Check-Pass "VoiceMem vezerlt forras + pin (e8384e0)" "import -> vendor\voicemem, CONTROLLED_FORK igaz, upstream commit OK"
        } else {
            $VmPinText = ((@($VmPinOut) -join " ")).Trim()
            Check-Fail "VoiceMem vezerlt forras + pin (e8384e0)" ("a voicemem mas forrasbol/committol importalodik: {0}" -f $VmPinText) "Egy masik voicemem telepites (site-packages / masik venv) takaritja el a vezelt forrast - torold ki, majd START.bat repair"
        }
    } finally {
        Remove-Item -Path $VmProbeScript -ErrorAction SilentlyContinue | Out-Null
    }
}

# ---------------------------------------------------------------------------
# 7) ASR modell betoltes (AutoTokenizer, local_files_only=True)
# ---------------------------------------------------------------------------
if ((Test-Path $VenvPython) -and (Test-Path $AsrConfig)) {
    $AsrDirPy = (Join-Path $Root "models\asr\qwen3-asr-0.6b").Replace('\', '/')
    $AsrCode = "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained(r'" + $AsrDirPy + "', local_files_only=True); print('ASR TOKENIZER OK', type(t).__name__)"
    $AsrOut = & $VenvPython -c $AsrCode 2>&1
    $AsrText = ((@($AsrOut) | ForEach-Object { [string]$_ }) -join " ")
    if (($LASTEXITCODE -eq 0) -and ($AsrText -match 'ASR TOKENIZER OK')) {
        Check-Pass "ASR modell betoltes (AutoTokenizer, offline)" $AsrText
    } else {
        Check-Fail "ASR modell betoltes (AutoTokenizer, offline)" $AsrText "scripts\download_models.ps1 ujrafuttatasa; ha a fajlok megvannak, a transformers verzioja a problema"
    }
}

# ---------------------------------------------------------------------------
# 8) LLM szerver (v0.3.3: a /health az ELSODELEGES sikerfeltetel)
#    - a starter szkript (PowerShell wrapper) exit kodja NEM sikerfeltetel
#    - elsodleges feltetel: GET http://127.0.0.1:8080/health -> HTTP 200
#    - masodlagos feltetel: a llama-server folyamat fut
#    - FAIL csak akkor, ha (1) a folyamat tenylegesen leallt ES (2) a
#      /health nem valt elerhetove a timeouton belul
#    - a vegso eredmeny 6 kulon reszjelzo: server startup / health /
#      chat completion / JSON completion / model loaded / process alive
#    - a python-bridge hibaja kulon WARNING, NEM fatal (ha a shellbol
#      kozvetlenul levezetett konfiguracioval a szerver mukodik)
# ---------------------------------------------------------------------------
$ServerProc = $null
$DirectProc = $null
$StartedServer = $false
$LlmStartupOk = $false
$LlmHealthOk = $false
$LlmChatOk = $false
$LlmJsonOk = $false
$LlmSchemaOk = $false
$LlmModelOk = $false
$LlmProcAlive = $false
$StarterExitCode = $null
$DirectExitCode = $null
$LlmHow = ""
$SlsScript = Join-Path $Root "scripts\start_llama_server.ps1"

# --- 8.0) python-bridge probe: UGYANAZ a konfig-feloldas, amit a starter
#     hasznal (AgentConfig.from_yaml -> JSON). A VERDIKTJET csak a vegso
#     LLM-eredmeny ismereteben dontjuk el (l. 8f): bridge-hiba + mukodo
#     kozvetlen konfiguracio -> WARNING, nem fatal.
$BridgeOk = $false
$BridgeDetail = ""
if (Test-Path $VenvPython) {
    $BridgeProbe = @'
import json
import os
import sys

root = os.environ.get("VOICEMEM_HOME", os.getcwd())
if root and root not in sys.path:
    sys.path.insert(0, root)
from app.config import AgentConfig

c = AgentConfig.from_yaml(os.path.join(root, "config", "voicemem_config.yaml"))
print(json.dumps({
    "exe": str(c.piper_exe_path.parent),
    "model": (str(c.llm_model_file) if c.llm_model_file else ""),
    "host": c.llama_server_host,
    "port": c.llama_server_port,
    "ctx": c.llm_context_size,
    "ngl": c.llm_n_gpu_layers,
    "parallel": getattr(c, "llm_parallel", 1),
    "temp": c.llm_temperature,
    "ck_k": c.llm_cache_type_k,
    "ck_v": c.llm_cache_type_v,
}))
'@
    # v0.3.4 (field report #4): the probe runs from an ATOMIC temp .py file -
    # the previous `python -c $BridgeProbe` lost its inner double quotes in
    # the PowerShell 5.1 native argument passing (r"config\..." arrived as
    # rconfig\... -> SyntaxError -> a false bridge WARNING on the target
    # machine). A file's content has no command-line quoting; sys.path is
    # secured via PYTHONPATH + VOICEMEM_HOME, the YAML path is absolute.
    $BridgeProbePy = Join-Path ([System.IO.Path]::GetTempPath()) ("voicemem_cfg_probe_{0}.py" -f $PID)
    Set-Content -Path $BridgeProbePy -Value $BridgeProbe -Encoding ASCII
    $PrevProbePyPath = [string]$env:PYTHONPATH
    try {
        if ([string]::IsNullOrWhiteSpace($PrevProbePyPath)) { $env:PYTHONPATH = $Root } else { $env:PYTHONPATH = "$Root;$PrevProbePyPath" }
        Push-Location $Root
        $BridgeOut = & $VenvPython $BridgeProbePy 2>&1
        $BridgeExit = $LASTEXITCODE
    } finally {
        try { Pop-Location } catch { }
        $env:PYTHONPATH = $PrevProbePyPath
        Remove-Item -Path $BridgeProbePy -Force -ErrorAction SilentlyContinue
    }
    $BridgeDetail = ((@($BridgeOut) | ForEach-Object { [string]$_ }) -join " ").Trim()
    if (($BridgeExit -eq 0) -and ($BridgeDetail -ne "")) { $BridgeOk = $true }
} else {
    $BridgeDetail = "a .venv python nem talalhato"
}

if (Test-LlamaHealth -TimeoutSec 5) {
    Check-Pass "LLM server startup" "a szerver mar fut (nem inditunk ujat - idempotens)"
    Check-Pass "LLM health (/health 200)" "http://127.0.0.1:8080/health OK (mar fut)"
    $LlmStartupOk = $true
    $LlmHealthOk = $true
    $LlmHow = "mar futott"
} elseif (-not $WithServer) {
    Check-Fail "LLM server startup" "a szerver nem fut (a /health nem valaszol)" "inditsd el egy masik ablakban: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_llama_server.ps1 - vagy futtasd a verify_m1.ps1 -WithServer kapcsoloval"
    Check-Fail "LLM health (/health 200)" "a http://127.0.0.1:8080/health cimen 5 s-en belul nem erkezett valasz" "l. az LLM server startup sor javitasi javaslatat"
} elseif (-not (Test-Path $SlsScript)) {
    Check-Fail "LLM server startup" ("nem talalhato: {0}" -f $SlsScript) "repo-fajl - klonozd ujra a repot"
    Check-Fail "LLM health (/health 200)" "a starter szkript hianyzik, a szerver nem indithato" "repo-fajl potlasa (ujra-klonozas)"
} else {
    # --- 8.1) STARTER kiserlet: a wrapper exit kodja NEM sikerfeltetel ---
    Write-Host "  -WithServer: a llama-server inditasa a starter szkripttel (minimizalt ablak)..."
    Write-Host "        v0.3.3: az elsodleges sikerfeltetel a /health 200 - a starter exit kodja nem az"
    try {
        $ServerProc = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $SlsScript, "-Root", $Root) -WindowStyle Minimized -PassThru
    } catch {
        Write-Host ("    a starter inditasa nem sikerult: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
        $ServerProc = $null
    }
    if ($ServerProc -ne $null) { $StartedServer = $true }
    $LlmHow = "starter szkript"
    $Healthy = $false
    $T0 = Get-Date
    $LastProgress = 0
    while ($true) {
        if (Test-LlamaHealth -TimeoutSec 2) { $Healthy = $true; break }
        $Elapsed = [int][math]::Floor(((Get-Date) - $T0).TotalSeconds)
        if ($Elapsed -ge 30) { break }
        if (($ServerProc -ne $null) -and ($ServerProc.HasExited)) {
            try { $StarterExitCode = [int]$ServerProc.ExitCode } catch { $StarterExitCode = 1 }
            break
        }
        if (($Elapsed - $LastProgress) -gt 5) {
            Write-Host ("    /health varakozas... {0} s" -f $Elapsed)
            $LastProgress = $Elapsed
        }
        Start-Sleep -Milliseconds 700
    }

    if (-not $Healthy) {
        $StarterDead = (($ServerProc -eq $null) -or ($ServerProc.HasExited))
        $LlamaProcsNow = @(Get-Process -Name "llama-server" -ErrorAction SilentlyContinue)
        if ($StarterDead -and ($LlamaProcsNow.Count -eq 0)) {
            # --- 8.2) KOZVETLEN FALLBACK: a starter wrapper kilepett es nincs
            # elo llama-server folyamat -> a verify MAGA inditja a
            # bin\llama-server.exe-t a shellbol kozvetlenul levezetett
            # konfiguracioval (a config alapertekeivel AZONOS flagek).
            $LlamaExe = Join-Path $Root "bin\llama-server.exe"
            if (-not (Test-Path $LlamaExe)) {
                if ($env:LLAMA_BIN) {
                    $Candidate = Join-Path $env:LLAMA_BIN "llama-server.exe"
                    if (Test-Path $Candidate) { $LlamaExe = $Candidate }
                }
            }
            # v0.4.17: ugyanaz a feloldasi lanc mint a fenti GGUF-kapu
            # (llm_model.json > env > operator-dir) - a v0.4.16-os
            # hardcode-olt default ut helyett.
            $LlamaModel = ""
            if ($LlmGguf) { $LlamaModel = $LlmGguf }
            if (-not $LlamaModel) {
                $LlamaModel = Get-OperatorDefaultGguf (Join-Path $Root "models\llm\qwen3.6-35b-a3b")
            }
            if (-not $LlamaModel) {
                Write-Host "    FALLBACK KIHAGYVA: nincs feloldhato LLM modell-fajl (l. az LLM GGUF kaput fent)." -ForegroundColor Yellow
            }
            if ($LlamaModel -and (Test-Path $LlamaExe) -and (Test-Path -LiteralPath $LlamaModel)) {
                # MEGJEGYZES: PATH-modifikacio NEM szukseges - a Windows DLL-
                # keresese eloszor a exe SAJAT konyvtaraban keres (bin\), igy a
                # llama-server.exe megtalalja a CUDA DLL-eket (ez meg a
                # gpu_check PATH-invariantajat is megorzi - l. a tesztet).
                Write-Host ("    FALLBACK: a starter wrapper kilepett (exit kod: {0}) - a llama-server.exe kozvetlen inditasa" -f $StarterExitCode)
                $ModelPathQ = '"' + $LlamaModel + '"'
                $LlamaArgs = @(
                    "--model", $ModelPathQ,
                    "--host", "127.0.0.1",
                    "--port", "8080",
                    "-ngl", "-1",
                    "-c", "8192",
                    "--parallel", "1",
                    "--cache-type-k", "q8_0",
                    "--cache-type-v", "q8_0",
                    "--temp", "0.7",
                    "--metrics",
                    "--no-webui"
                )
                $OutLog = Join-Path $Root "logs\llama-server.out.log"
                $ErrLog = Join-Path $Root "logs\llama-server.err.log"
                try {
                    $DirectProc = Start-Process -FilePath $LlamaExe -ArgumentList $LlamaArgs -WorkingDirectory $Root -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
                } catch {
                    Write-Host ("    FALLBACK inditasi hiba: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
                    $DirectProc = $null
                }
                if ($DirectProc -ne $null) {
                    $LlmHow = "kozvetlen inditas (fallback a starter hibaja utan)"
                    $T0 = Get-Date
                    $LastProgress = 0
                    while ($true) {
                        if (Test-LlamaHealth -TimeoutSec 2) { $Healthy = $true; break }
                        if ($DirectProc.HasExited) { break }
                        $Elapsed = [int][math]::Floor(((Get-Date) - $T0).TotalSeconds)
                        if ($Elapsed -ge 30) { break }
                        if (($Elapsed - $LastProgress) -gt 5) {
                            Write-Host ("    /health varakozas... {0} s" -f $Elapsed)
                            $LastProgress = $Elapsed
                        }
                        Start-Sleep -Milliseconds 700
                    }
                    if (-not $Healthy -and (-not $DirectProc.HasExited)) {
                        # a folyamat EL, de meg nem ready - amig el, addig varunk
                        # (max 60 s tobblet): FAIL csak leallt folyamat + nincs
                        # /health eseten van (v0.3.3 kontraktus)
                        Write-Host "    a folyamat el, de meg nem ready - varakozas folytatasa (max 60 s tobblet)..."
                        $TExt = Get-Date
                        while ($true) {
                            if (Test-LlamaHealth -TimeoutSec 2) { $Healthy = $true; break }
                            if ($DirectProc.HasExited) { break }
                            if (((Get-Date) - $TExt).TotalSeconds -ge 60) { break }
                            Start-Sleep -Milliseconds 700
                        }
                    }
                }
            } else {
                Write-Host ("    FALLBACK nem indithato: hianyzo fajl (exe: {0} | modell: {1})" -f $LlamaExe, $LlamaModel)
            }
        } else {
            # --- 8.3) a starter meg fut (sajat /health-varakozasa max 90 s)
            # VAGY mar elo llama-server folyamat van - a /health-ig varunk
            # (max 90 s a starter inditasatol szamitva).
            Write-Host "    a szerver meg indul - a /health varakozas folytatasa (max 90 s)..."
            while ($true) {
                if (Test-LlamaHealth -TimeoutSec 2) { $Healthy = $true; break }
                if (((Get-Date) - $T0).TotalSeconds -ge 90) { break }
                if (($ServerProc -ne $null) -and ($ServerProc.HasExited)) {
                    if ($StarterExitCode -eq $null) {
                        try { $StarterExitCode = [int]$ServerProc.ExitCode } catch { $StarterExitCode = 1 }
                    }
                    $LlamaProcsMid = @(Get-Process -Name "llama-server" -ErrorAction SilentlyContinue)
                    if ($LlamaProcsMid.Count -eq 0) { break }
                }
                Start-Sleep -Milliseconds 700
            }
        }
    }

    # --- 8.4) VERDIKT: ha /health = 200 -> PASS akkor is, ha a PowerShell
    # starter wrapper nem-0 exit koddal kilepett (az csak MEGJEGYZES).
    $UpSec = [math]::Round(((Get-Date) - $T0).TotalSeconds, 1)
    if ($Healthy) {
        $LlmStartupOk = $true
        $LlmHealthOk = $true
        Check-Pass "LLM server startup" ("felfutott, /health 200 ({0})" -f $LlmHow)
        Check-Pass "LLM health (/health 200)" ("http://127.0.0.1:8080/health - HTTP 200, {0} s alatt" -f $UpSec)
        if (($StarterExitCode -ne $null) -and ($StarterExitCode -ne 0)) {
            Write-Host ("       MEGJEGYZES: a PowerShell starter wrapper exit kodja {0}." -f $StarterExitCode)
            Write-Host "       A /health 200 az elsodleges sikerfeltetel, ezert ez PASS - a wrapper exit kodja NEM bukta."
        }
    } else {
        if ($DirectProc -ne $null) {
            try { if ($DirectProc.HasExited) { $DirectExitCode = [int]$DirectProc.ExitCode } } catch { $DirectExitCode = 1 }
        }
        $StarterExitDisp = "n/a"
        if ($StarterExitCode -ne $null) { $StarterExitDisp = [string]$StarterExitCode }
        $DirectExitDisp = "n/a"
        if ($DirectExitCode -ne $null) { $DirectExitDisp = [string]$DirectExitCode }
        $LlamaProcsEnd = @(Get-Process -Name "llama-server" -ErrorAction SilentlyContinue)
        if ($LlamaProcsEnd.Count -eq 0) {
            Check-Fail "LLM server startup" ("a llama-server folyamat leallt ES a /health nem valt elerhetove a timeouton belul (starter exit: {0}; kozvetlen inditas exit: {1})" -f $StarterExitDisp, $DirectExitDisp) "a lenti log-kivonatok: eloszor a start-szkript SAJAT kimenete (llama-server.starter.log - transcript), utana a szerver logja; a teljes logok: logs\llama-server.starter.log, logs\llama-server.err.log"
            $ZoneProbe = Get-Item -Path $SlsScript -Stream Zone.Identifier -ErrorAction SilentlyContinue
            if ($ZoneProbe) {
                Write-Host ("       FIGYELEM: {0} Internet-zona jelzest (MOTW) hordoz - egy normal session blokkolja." -f $SlsScript) -ForegroundColor Yellow
                Write-Host "       JAVITAS (kezi inditasokhoz):  Get-ChildItem .\scripts\*.ps1 | Unblock-File"
            }
            Show-LlamaLogTail -RootPath $Root -Label "llama-server"
        } else {
            Check-Fail "LLM server startup" ("a /health nem valt elerhetove a timeouton belul (a llama-server folyamat meg fut - nem ready; starter exit: {0})" -f $StarterExitDisp) "a szerver betoltes alatt lehet - l. a lenti log-kivonatokat; lassu modellbetoltes eseten ujrafuttatas"
            Show-LlamaLogTail -RootPath $Root -Label "llama-server"
        }
        Check-Fail "LLM health (/health 200)" "a http://127.0.0.1:8080/health a timeouton belul nem valaszolt" "l. az LLM server startup sor javitasi javaslatat"
    }
}
if ($LlmHealthOk) {
    # --- 8a) chat completion: POST /v1/chat/completions (nem-stream) -------
    $Body = '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Hello! Reply with one short greeting sentence."}],"max_tokens":48,"temperature":0,"stream":false,"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"none"}'
    try {
        $Resp = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" -Method Post -Body $Body -ContentType "application/json" -TimeoutSec 120
        $Content = ""
        try { $Content = [string]$Resp.choices[0].message.content } catch { }
        $Trim = $Content.Trim()
        if ($Trim -ne "") {
            $LlmChatOk = $true
            Check-Pass "LLM chat completion (POST /v1/chat/completions)" ("valasz: '{0}'" -f $Trim)
        } else {
            Check-Fail "LLM chat completion (POST /v1/chat/completions)" "a valasz ures (choices[0].message.content nincs)" "a /health megy, de a /v1/chat/completions nem valaszol - nezd a logs\llama-server.err.log fajlt"
        }
    } catch {
        Check-Fail "LLM chat completion (POST /v1/chat/completions)" ("hiba: {0}" -f $_.Exception.Message) "a /health megy, de a /v1/chat/completions nem - nezd a logs\llama-server.err.log fajlt"
    }

    # --- 8b) JSON completion: response_format json_object (REQUEST szintu) --
    #     Pontosan ez az ut hasznalja a VoiceMem _llm_json() es az
    #     app/llm.py chat_json() (llama.cpp constrained generation).
    $JsonBody = '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Return a JSON object with exactly two keys: name and language. Values: Thomas and Hungarian. Output ONLY the JSON object, no other text."}],"max_tokens":64,"temperature":0,"stream":false,"response_format":{"type":"json_object"},"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"none"}'
    try {
        $JsonResp = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" -Method Post -Body $JsonBody -ContentType "application/json" -TimeoutSec 120
        $JsonContent = ""
        try { $JsonContent = [string]$JsonResp.choices[0].message.content } catch { }
        $JsonTrim = $JsonContent.Trim()
        $JsonErr = ""
        if ($JsonTrim -eq "") {
            $JsonErr = "a valasz ures"
        } elseif ($JsonTrim.StartsWith('```')) {
            $JsonErr = "markdown fence a JSON valaszban: " + $JsonTrim
        } else {
            try {
                $Parsed = $JsonTrim | ConvertFrom-Json
                if ($Parsed -eq $null) {
                    $JsonErr = "a valasz nem JSON: " + $JsonTrim
                } elseif (-not ($Parsed.PSObject.Properties["name"] -and $Parsed.PSObject.Properties["language"])) {
                    $JsonErr = "hianyzo kulcs (name/language) a JSON-ben: " + $JsonTrim
                }
            } catch {
                $JsonErr = "ervenytelen JSON / extra szoveg: " + $JsonTrim
            }
        }
        if ($JsonErr -eq "") {
            $LlmJsonOk = $true
            Check-Pass "LLM JSON completion (response_format json_object)" ("ervenyes JSON, kulcsok rendben: name={0}, language={1}" -f $Parsed.name, $Parsed.language)
        } else {
            Check-Fail "LLM JSON completion (response_format json_object)" $JsonErr "a response_format json_object-nak ervenyes, fence-nelkuli JSON-t kell adnia - nezd a logs\llama-server.err.log fajlt"
        }
    } catch {
        Check-Fail "LLM JSON completion (response_format json_object)" ("hiba: {0}" -f $_.Exception.Message) "a json_object response_format kuldese nem sikerult - nezd a logs\llama-server.err.log fajlt"
    }

    # --- 8c) JSON completion: response_format json_schema (REQUEST szintu) --
    #     A b10717 tamogatja; ha egy jovobeli build nem tamogatna, a szerver
    #     4xx hibaval valaszol - ezt WARN-nal jelezzuk, NEM buktatjuk meg
    #     vele az M1-et (a json_object a kotelezo JSON-ut).
    $SchemaBody = '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Return a JSON object with exactly two keys: name and language. Values: Thomas and Hungarian."}],"max_tokens":64,"temperature":0,"stream":false,"response_format":{"type":"json_schema","json_schema":{"name":"user_language","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"},"language":{"type":"string"}},"required":["name","language"],"additionalProperties":false}}},"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"none"}'
    try {
        $SchemaResp = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" -Method Post -Body $SchemaBody -ContentType "application/json" -TimeoutSec 120
        $SchemaContent = ""
        try { $SchemaContent = [string]$SchemaResp.choices[0].message.content } catch { }
        $SchemaTrim = $SchemaContent.Trim()
        $SchemaErr = ""
        if ($SchemaTrim -eq "") {
            $SchemaErr = "a valasz ures"
        } elseif ($SchemaTrim.StartsWith('```')) {
            $SchemaErr = "markdown fence a JSON valaszban: " + $SchemaTrim
        } else {
            try {
                $SchemaParsed = $SchemaTrim | ConvertFrom-Json
                if ($SchemaParsed -eq $null) {
                    $SchemaErr = "a valasz nem JSON: " + $SchemaTrim
                } elseif (-not ($SchemaParsed.PSObject.Properties["name"] -and $SchemaParsed.PSObject.Properties["language"])) {
                    $SchemaErr = "hianyzo kulcs (name/language) a JSON-ben: " + $SchemaTrim
                }
            } catch {
                $SchemaErr = "ervenytelen JSON / extra szoveg: " + $SchemaTrim
            }
        }
        if ($SchemaErr -eq "") {
            $LlmSchemaOk = $true
            Check-Pass "LLM JSON completion (json_schema)" ("schema-compliant JSON: name={0}, language={1}" -f $SchemaParsed.name, $SchemaParsed.language)
        } else {
            Check-Fail "LLM JSON completion (json_schema)" $SchemaErr "a json_schema response_format-nak schema-compliant JSON-t kell adnia"
        }
    } catch {
        $Msg = [string]$_.Exception.Message
        if ($Msg -match "response_format|json_schema|400") {
            Check-Warn "LLM JSON completion (json_schema)" ("a telepitett build nem tamogatja a json_schema response_format-ot: {0}" -f $Msg) "a kotelezo JSON-ut a json_object (8b) - az PASS-olt"
        } else {
            Check-Fail "LLM JSON completion (json_schema)" ("hiba: {0}" -f $Msg) "nezd a logs\llama-server.err.log fajlt"
        }
    }

    # --- 8d) model loaded: GET /v1/models (a szerver altal betoltott modell)
    # v0.4.17: a modell lehet OLLAMA BLOB (sha256-afc723...) is - ilyenkor a
    # served id a blob NEVE (a kiterjesztes nelkuli fajlnev), NEM a
    # "qwen3.6-35b-a3b" profilnev. Elfogadott: (a) a profilnev egyezik, VAGY
    # (b) a served id egyezik a feloldott fajl nevevel (blob-stem), ES a
    # starter marker (logs/llama-server.resolved-model.json) ugyanazt a
    # modellutat irja, mint a feloldas - ez a "prove the exact file" kapu.
    try {
        $ModelsResp = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/models" -Method Get -TimeoutSec 30
        $ModelId = ""
        try { $ModelId = [string]$ModelsResp.data[0].id } catch { }
        # v0.4.17: marker-alapu keresztkontroll (a starter minden inditasnal
        # irja; tartalmazza a served id-t, a szerver sajat log-sorat es a
        # completion round-trip eredmenyet is).
        $MarkerPath = Join-Path $Root "logs\llama-server.resolved-model.json"
        $MarkerMatch = "unknown"
        $MarkerInfo = ""
        if (Test-Path -LiteralPath $MarkerPath) {
            try {
                $MarkerJson = Get-Content -LiteralPath $MarkerPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $MarkerModel = [string]$MarkerJson.model_path
                if ($MarkerModel -and $LlmGguf) {
                    if ($MarkerModel.ToLower() -eq ([string]$LlmGguf).ToLower()) { $MarkerMatch = "same" } else { $MarkerMatch = "different" }
                }
                $MarkerInfo = ("marker: model_path={0}; served_model_id={1}; log_path_found={2}; completion_ok={3}" -f $MarkerModel, [string]$MarkerJson.served_model_id, [bool]$MarkerJson.log_path_found, [bool]$MarkerJson.completion_ok)
            } catch { $MarkerInfo = "marker olvashatatlan" }
        }
        $ResolvStem = ""
        try { $ResolvStem = [System.IO.Path]::GetFileName($LlmGguf) } catch { }
        $ResolvStemNorm = ""
        $ModelIdNorm = ""
        if ($ResolvStem) { $ResolvStemNorm = $ResolvStem.ToLower() -replace '[^0-9a-z]', '' }
        if ($ModelId) { $ModelIdNorm = $ModelId.ToLower() -replace '[^0-9a-z]', '' }
        $StemMatch = ($ResolvStemNorm.Length -gt 0 -and $ModelIdNorm.Length -gt 0 -and
                     (($ModelIdNorm.Contains($ResolvStemNorm)) -or ($ResolvStemNorm.Contains($ModelIdNorm))))
        if (($ModelId -ne "") -and (($ModelId -match "qwen3\.6-35b-a3b") -or $StemMatch)) {
            $LlmModelOk = $true
            Check-Pass "LLM model loaded (GET /v1/models)" ("a szerver altal betoltott modell: {0} (Qwen3.6 35B A3B - az EGYETLEN LLM; blob-stem egyezes: {1})" -f $ModelId, $StemMatch)
        } elseif ($ModelId -ne "") {
            Check-Fail "LLM model loaded (GET /v1/models)" ("a betoltott modell NEM a Qwen3.6 35B A3B IQ4_XS: {0}" -f $ModelId) "inditsd ujra a START.bat-ot (a v0.4.16+ starter a regi peldanyt magatol leallitja es a konfigural modelllel indit ujra); nincs masik profil"
        } else {
            Check-Fail "LLM model loaded (GET /v1/models)" "a /v1/models valaszban nincs modell-azonosito (data[0].id)" "a /health megy, de a modell-lista ures - nezd a logs\llama-server.err.log fajlt"
        }
        # v0.4.17: a "betoltes-bizonyitek" blokk (a felhasznalo altal kert
        # diagnosztikai panel formaban).
        if ($MarkerInfo) {
            Write-Host ""
            Write-Host "  --- LLM betoltes-bizonyitek (logs\llama-server.resolved-model.json) ---"
            if ($LlmGguf) {
                Write-Host ("  Selected LLM : Qwen3.6 35B A3B IQ4_XS (egyetlen LLM)")
                Write-Host ("  GGUF         : {0}" -f (Split-Path -Leaf $LlmGguf))
                Write-Host ("  Path         : {0}" -f $LlmGguf)
                $FileStatus = "File NOT found"
                try { if (Test-Path -LiteralPath $LlmGguf) { $FileStatus = "File found" } } catch { }
                Write-Host ("  Status       : {0}" -f $FileStatus)
            }
            Write-Host ("  {0}" -f $MarkerInfo)
            if ($MarkerMatch -eq "different") {
                Check-Warn "LLM load proof (marker vs. feloldas)" ("a starter marker MAS fajlt ir ({0})" -f $MarkerInfo) "inditsd ujra a START.bat-ot a kivalasztott modelllel"
            } elseif ($MarkerMatch -eq "same") {
                Check-Pass "LLM load proof (marker vs. feloldas)" ("a futtato szerver a KONFIGURALT fajlt tolti ({0})" -f $MarkerInfo)
            } else {
                Check-Warn "LLM load proof (marker vs. feloldas)" "a starter marker meg nem irta felul a bizonyitek-mezoket (v0.4.17 elotti inditas?) - ujrainditas utan frissul" "futtasd ujra a START.bat-ot"
            }
        }
    } catch {
        Check-Fail "LLM model loaded (GET /v1/models)" ("hiba: {0}" -f $_.Exception.Message) "a GET /v1/models nem sikerult - nezd a logs\llama-server.err.log fajlt"
    }

    # --- 8e) process alive: a llama-server folyamat a tesztek utan is fut ---
    $LlamaProcsAlive = @(Get-Process -Name "llama-server" -ErrorAction SilentlyContinue)
    if ($LlamaProcsAlive.Count -gt 0) {
        $LlmProcAlive = $true
        $P0 = $LlamaProcsAlive[0]
        Check-Pass "LLM process alive" ("llama-server fut (PID {0}, ~{1} MB munkakeszlet)" -f $P0.Id, [math]::Round($P0.WorkingSet64 / 1MB))
    } else {
        Check-Fail "LLM process alive" "a llama-server folyamat nem talalhato (Get-Process llama-server)" "a szerver a tesztek alatt kilepett - nezd a logs\llama-server.err.log fajlt"
    }
} else {
    # a szerver nem erheto el: a maradek reszjelzok kulon-kulon jelezve
    Check-Fail "LLM chat completion (POST /v1/chat/completions)" "kihagyva - a szerver nem erheto el" "l. az LLM server startup / LLM health sorok javitasi javaslatat"
    Check-Fail "LLM JSON completion (response_format json_object)" "kihagyva - a szerver nem erheto el" "l. az LLM server startup / LLM health sorok javitasi javaslatat"
    Check-Fail "LLM model loaded (GET /v1/models)" "kihagyva - a szerver nem erheto el" "l. az LLM server startup / LLM health sorok javitasi javaslatat"
    Check-Fail "LLM process alive" "a llama-server folyamat nem talalhato vagy nem erheto el" "l. az LLM server startup / LLM health sorok javitasi javaslatat"
}

# --- 8f) python-bridge VERDIKT (a 8.0-ban mertuk ki): a bridge hibaja NEM
#     fatal, ha a shellbol kozvetlenul levezetett llama-server konfiguracio
#     sikeresen mukodik (celgepi field report #3).
$BridgeShort = $BridgeDetail
if ($BridgeShort.Length -gt 240) { $BridgeShort = $BridgeShort.Substring(0, 240) + "..." }
if ($BridgeOk) {
    Check-Pass "Konfiguracio python-bridge (config\voicemem_config.yaml)" "a starter konfig-feloldasa mukodik (AgentConfig.from_yaml + env felulirasok)"
} elseif ($LlmHealthOk) {
    Check-Warn "Konfiguracio python-bridge (config\voicemem_config.yaml)" ("a konfiguracio nem toltheto be a python-bridge-en keresztul ({0}) - NEM fatal: a shellbol kozvetlenul levezetett llama-server konfiguracio mukodik (/health 200)" -f $BridgeShort) "kesobbi javitas: .venv\Scripts\python.exe -m app.main --list-config (a repo gyokererol)"
} else {
    Check-Fail "Konfiguracio python-bridge (config\voicemem_config.yaml)" ("a konfiguracio nem toltheto be a python-bridge-en keresztul ({0})" -f $BridgeShort) "a kozvetlen (shell) konfiguracio sem vezetett elo szerverhez - futtasd ujra a telepitot, majd kezen: .venv\Scripts\python.exe -m app.main --list-config"
}

# --- a 6 reszjelzo osszefogo sora (a vegso verify eredmeny resze) ----------
$FS = "FAIL"
if ($LlmStartupOk) { $FS = "PASS" }
$FH = "FAIL"
if ($LlmHealthOk) { $FH = "PASS" }
$FC = "FAIL"
if ($LlmChatOk) { $FC = "PASS" }
$FJ = "FAIL"
if ($LlmJsonOk) { $FJ = "PASS" }
$FM = "FAIL"
if ($LlmModelOk) { $FM = "PASS" }
$FP = "FAIL"
if ($LlmProcAlive) { $FP = "PASS" }
Write-Host ""
Write-Host ("LLM resz-eredmenyek: server startup={0} | health={1} | chat completion={2} | JSON completion={3} | model loaded={4} | process alive={5}" -f $FS, $FH, $FC, $FJ, $FM, $FP)

# ---------------------------------------------------------------------------
# 9) Piper HU szintezis
# ---------------------------------------------------------------------------
$PiperExe = Join-Path $Root "bin\piper.exe"
$AudioDir = Join-Path $Root "data\audio"
New-Item -ItemType Directory -Path $AudioDir -Force | Out-Null
$HuVoiceOnnx = Join-Path $PiperDir "hu_HU-anna-medium.onnx"
$SmokeHuWav = Join-Path $AudioDir "smoke_hu.wav"
if ((Test-Path $PiperExe) -and (Test-Path $HuVoiceOnnx)) {
    # Az ekezetes szoveget ASCII-forrasbol epitjuk fel (a fajl ekezetmentes):
    $HuText = "J" + [char]0x00F3 + " reggelt, ez egy f" + [char]0x00FC + "stteszt."
    $OutputEncoding = [System.Text.Encoding]::UTF8
    $PiperOut = $HuText | & $PiperExe --model $HuVoiceOnnx --output_file $SmokeHuWav
    $PiperExit = $LASTEXITCODE
    $HuWavSize = 0
    if (Test-Path $SmokeHuWav) { $HuWavSize = (Get-Item -LiteralPath $SmokeHuWav).Length }
    if (($PiperExit -eq 0) -and ($HuWavSize -gt 1024)) {
        Check-Pass "Piper HU szintezis" ("{0} ({1} KB)" -f $SmokeHuWav, [int]($HuWavSize / 1024))
    } else {
        $PiperDetail = ((@($PiperOut) | ForEach-Object { [string]$_ }) -join " ")
        Check-Fail "Piper HU szintezis" ("kilepesi kod: {0}, wav meret: {1} byte; kimenet: {2}" -f $PiperExit, $HuWavSize, $PiperDetail) "kezi teszt: echo szoveg | bin\piper.exe --model models\tts\piper\hu_HU-anna-medium.onnx --output_file probe.wav"
    }
} else {
    Check-Fail "Piper HU szintezis" ("hianyzik: {0} vagy {1}" -f $PiperExe, $HuVoiceOnnx) "START.bat repair (binarisok + hangok automatikus potlasa)"
}

# ---------------------------------------------------------------------------
# 10) Piper EN szintezis
# ---------------------------------------------------------------------------
$EnVoiceOnnx = Join-Path $PiperDir "en_US-lessac-medium.onnx"
$SmokeEnWav = Join-Path $AudioDir "smoke_en.wav"
if ((Test-Path $PiperExe) -and (Test-Path $EnVoiceOnnx)) {
    $EnText = "Good morning, this is a smoke test."
    $OutputEncoding = [System.Text.Encoding]::UTF8
    $PiperOut2 = $EnText | & $PiperExe --model $EnVoiceOnnx --output_file $SmokeEnWav
    $PiperExit2 = $LASTEXITCODE
    $EnWavSize = 0
    if (Test-Path $SmokeEnWav) { $EnWavSize = (Get-Item -LiteralPath $SmokeEnWav).Length }
    if (($PiperExit2 -eq 0) -and ($EnWavSize -gt 1024)) {
        Check-Pass "Piper EN szintezis" ("{0} ({1} KB)" -f $SmokeEnWav, [int]($EnWavSize / 1024))
    } else {
        $PiperDetail2 = ((@($PiperOut2) | ForEach-Object { [string]$_ }) -join " ")
        Check-Fail "Piper EN szintezis" ("kilepesi kod: {0}, wav meret: {1} byte; kimenet: {2}" -f $PiperExit2, $EnWavSize, $PiperDetail2) "kezi teszt: echo szoveg | bin\piper.exe --model models\tts\piper\en_US-lessac-medium.onnx --output_file probe.wav"
    }
} else {
    Check-Fail "Piper EN szintezis" ("hianyzik: {0} vagy {1}" -f $PiperExe, $EnVoiceOnnx) "START.bat repair (binarisok + hangok automatikus potlasa)"
}

# ---------------------------------------------------------------------------
# 11) Memory konyvtarak irhatosaga
# ---------------------------------------------------------------------------
if (Test-Path $VenvPython) {
    $env:VMEM_MEMCHECK_ROOT = $Root
    $MemCode = @'
import os
from pathlib import Path
root = Path(os.environ["VMEM_MEMCHECK_ROOT"])
failures = []
for sub in ("sqlite", "qdrant", "backups"):
    d = root / "memory" / sub
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_probe"
        probe.write_text("probe", encoding="utf-8")
        content = probe.read_text(encoding="utf-8")
        probe.unlink()
        if content != "probe":
            failures.append(str(d))
    except Exception as exc:
        failures.append("{0}: {1}".format(d, exc))
if failures:
    print("MEMORY FAIL: " + ", ".join(failures))
    raise SystemExit(1)
print("MEMORY OK")
'@
    $MemOut = & $VenvPython -c $MemCode 2>&1
    $MemText = ((@($MemOut) | ForEach-Object { [string]$_ }) -join " ")
    if (($LASTEXITCODE -eq 0) -and ($MemText -match 'MEMORY OK')) {
        Check-Pass "Memory konyvtarak irhatosaga" "memory\sqlite + qdrant + backups (write probe OK)"
    } else {
        Check-Fail "Memory konyvtarak irhatosaga" $MemText "jog/lemezhiba? nezd a memory\ mappa engedelyeit"
    }
}

# ---------------------------------------------------------------------------
# 12) Konfiguracio (AgentConfig.from_yaml + validate)
# ---------------------------------------------------------------------------
if (Test-Path $VenvPython) {
    Push-Location $Root
    $CfgOut = & $VenvPython -c "from app.config import AgentConfig; c = AgentConfig.from_yaml('config/voicemem_config.yaml'); errs = c.validate(); print('CONFIG OK' if not errs else 'CONFIG FAIL: ' + '; '.join(errs))" 2>&1
    $CfgExit = $LASTEXITCODE
    Pop-Location
    $CfgText = ((@($CfgOut) | ForEach-Object { [string]$_ }) -join " ")
    if (($CfgExit -eq 0) -and ($CfgText -match 'CONFIG OK')) {
        Check-Pass "Konfiguracio (config/voicemem_config.yaml)" "AgentConfig.from_yaml + validate() OK"
    } else {
        Check-Fail "Konfiguracio (config/voicemem_config.yaml)" $CfgText "nezd a YAML-t; a hibauzenet a megtorott mezoket irja"
    }
}

# ---------------------------------------------------------------------------
# Takaritas: a verify altal inditott szerver leallitasa
# (csak akkor, ha mi inditottuk - a kulso szervert nem nyuljuk)
# ---------------------------------------------------------------------------
if ($StartedServer) {
    Write-Host ""
    Write-Host "A verify altal inditott llama-server leallitasa..."
    Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    if ($ServerProc -ne $null) {
        Stop-Process -Id $ServerProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($DirectProc -ne $null) {
        Stop-Process -Id $DirectProc.Id -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# Osszegzes
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "==================================================================="
if ($Script:FailCount -eq 0) {
    if ($Script:WarnCount -gt 0) {
        Write-Host ("VERIFY: PASS ({0} figyelmeztetes)" -f $Script:WarnCount) -ForegroundColor Green
    } else {
        Write-Host "VERIFY: PASS" -ForegroundColor Green
    }
    exit 0
} else {
    Write-Host ("VERIFY: FAIL ({0} hibas ellenorzes)" -f $Script:FailCount) -ForegroundColor Red
    exit 1
}
