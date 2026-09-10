<#
===============================================================================
scripts/measure_vram.ps1 - VRAM mero szkript (fazisonkenti nvidia-smi mintavetel)

MIT csinal: a Qwen3.6 35B A3B IQ4_XS llama-server es az egyeb GPU-szamitasi folyamatok
VRAM-felhasznalasat meri 6 fazisban. Minden fazisban az OSSZ GPU-memoria
(nvidia-smi --query-gpu=memory.used,memory.total) es a folyamatonkenti
bontas (nvidia-smi --query-compute-apps: llama-server.exe KULON az osszes
tobbi folyamatattol, pl. a python/PyTorch ASR-tol) is mintavetelre kerul -
igy a Windows/egyeb hasznalat megkulonboztetheto a llama szerveretol.
Eredmeny:
  - konzol  : fazisonkenti osszefogo tablazat + folyamatonkenti sorok,
  - fajl    : logs\vram_report.json (ConvertTo-Json, mintankenti rekordok).

FAZISOK:
  [0/5] idle              : nvidia-smi mintavetel MINDEN muvelet ELOTT
  [1/5] model_loaded      : ha a /health mar 200-t valaszol, a szerver MAR FUT
                            (megjegyezzuk, NEM inditunk masodikat); kulonben
                            inditas rejtett ablakban:
                              powershell -ExecutionPolicy Bypass -File
                              <root>\scripts\start_llama_server.ps1
                            majd /health poll max 90 s, 1 s-es lepesekkel.
                            Ha a starter gyerekfolyamat koran kilep: a
                            logs\llama-server.starter.log (a start-szkript
                            SAJAT kimenete - transcript), a
                            logs\llama-server.err.log ES a
                            logs\llama-server.out.log utolso ~15 sora a
                            konzolra kerul, majd FAIL + exit 1 (gyors-hiba).
  [2/5] first_request     : POST /v1/chat/completions (Hello, max_tokens 16,
                            temperature 0.0, stream false) -> mintavetel UTANA
  [3/5] normal_generation : streaming keres (max_tokens 200, hosszabb valaszt
                            kero prompt) Start-Job-ban, kozben 0,5 s-enkent
                            nvidia-smi mintavetel a generalas ALATT
  [4/5] json_generation   : ugyanez response_format json_object-tal -> a
                            mintavetel szinten generalas kozben
  [5/5] peak              : a mintak kozotti csucsertek (total_used_mb)

TAKARITAS: ha a szkript MAGA inditotta a szervert, a vegen LEALLITJA a sajat
folyamatait (Get-Process llama-server + a starter powershell pidje) - a
verify_m1.ps1 mintajara. Ha a szerver eleve futott, HOZZA NEM NYULUNK.

KILEPESI KODOK: 0 = PASS (meres kesz, JSON riport kiirva), 1 = FAIL
(nincs nvidia-smi / a szerver nem indult el / a HTTP keres nem sikerult).
A vegso sor mindig "VRAM REPORT: ..." formatumu, a JSON fajl utvonalaval.

HASZNALAT:
    .\scripts\measure_vram.ps1
    .\scripts\measure_vram.ps1 -Port 8080 -Samples 5
    .\scripts\measure_vram.ps1 -Root "C:\masik_root"

MEGJEGYZES: a fajl szandekosan ASCII - a PowerShell 5.1 BOM nelkul
Windows-1252-kent olvasna az ekezetes karaktereket.
#>
param(
    [string]$Root = "",
    [int]$Port = 8080,
    [int]$Samples = 5
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
if ($Samples -lt 1) { $Samples = 1 }

# --- allapot (a fazisok es a takaritas erze) --------------------------------
$Script:StartedServer = $false
$Script:StarterProc = $null
$Script:SampleList = @()
$Script:NvidiaSmi = $null

# --- segedfuggvenyek ---------------------------------------------------------

function Resolve-NvidiaSmi {
    # nvidia-smi feloldasa: PATH (Get-Command) -> standard abszolut utvonalak.
    # (Utanproba: egy gyerekfolyamat PATH-ja elterhet az interaktiv
    # shell-etol - l. scripts\gpu_check.ps1 es a v0.1.4 hibajelentes.)
    $Cmd = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($Cmd -ne $null) {
        return $Cmd.Source
    }
    $Candidates = @()
    if ($env:SystemRoot) {
        $Candidates += (Join-Path $env:SystemRoot "System32\nvidia-smi.exe")
        $Candidates += (Join-Path $env:SystemRoot "SysWOW64\nvidia-smi.exe")
    }
    if ($env:ProgramFiles) {
        $Candidates += (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    }
    $Pf86 = ${env:ProgramFiles(x86)}
    if ($Pf86) {
        $Candidates += (Join-Path $Pf86 "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    }
    foreach ($C in $Candidates) {
        if (Test-Path $C) { return $C }
    }
    return $null
}

function Test-LlamaHealth {
    param([int]$TimeoutSec = 3)
    try {
        $R = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/health" -f $Port) -Method Get -TimeoutSec $TimeoutSec -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

function Stop-OwnServer {
    # Takaritas: CSAK a sajat magunk altal inditott szervert allitjuk le
    # (verify_m1.ps1 minta) - a kulso, mar futo szervert nem nyuljuk.
    if (-not $Script:StartedServer) { return }
    Write-Host ""
    Write-Host "A meres altal inditott llama-server leallitasa..."
    Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    if ($null -ne $Script:StarterProc) {
        Stop-Process -Id $Script:StarterProc.Id -Force -ErrorAction SilentlyContinue
    }
}

function Fail-Measure {
    param([string]$Message = "")
    Write-Host ""
    if ($Message -ne "") {
        Write-Host ("FAIL: {0}" -f $Message) -ForegroundColor Red
    }
    Stop-OwnServer
    Write-Host "VRAM REPORT: FAIL - JSON riport nem keszult (a meres nem fejezodott be)" -ForegroundColor Red
    exit 1
}

function New-VramSample {
    # Egy nvidia-smi mintavetel: total/used + szamitasi folyamatok bontasa.
    # Hiba eseten $null - a hivo donti el, hogy fatalis-e.
    param([string]$Phase, [int]$Index)
    $Used = 0
    $Total = 0
    try {
        $GpuOut = & $Script:NvidiaSmi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $FirstLine = ""
        foreach ($L in @($GpuOut)) {
            if ($null -ne $L) { $FirstLine = [string]$L; break }
        }
        $Parts = $FirstLine.Split(",")
        if ($Parts.Count -lt 2) { return $null }
        if (-not [int]::TryParse($Parts[0].Trim(), [ref]$Used)) { return $null }
        if (-not [int]::TryParse($Parts[1].Trim(), [ref]$Total)) { return $null }
    } catch {
        return $null
    }
    $PerProcess = @()
    try {
        $ProcOut = & $Script:NvidiaSmi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0) {
            foreach ($Line in @($ProcOut)) {
                if ($null -eq $Line) { continue }
                $Text = ([string]$Line).Trim()
                if ($Text -eq "") { continue }
                # Oszlopok: pid, process_name, used_memory. A nev vesszot
                # tartalmazhat (utvonal) - az elso es az utolso oszlop kozott
                # minden a nevhez tartozik.
                $Cols = $Text.Split(",")
                if ($Cols.Count -lt 3) { continue }
                $ProcPid = 0
                $ProcMb = 0
                if (-not [int]::TryParse($Cols[0].Trim(), [ref]$ProcPid)) { continue }
                if (-not [int]::TryParse($Cols[$Cols.Count - 1].Trim(), [ref]$ProcMb)) { continue }
                $NameParts = @()
                for ($i = 1; $i -lt ($Cols.Count - 1); $i++) {
                    $NameParts += $Cols[$i].Trim()
                }
                $ProcName = ($NameParts -join ",")
                $Group = "other"
                if ($ProcName -like "*llama-server*") { $Group = "llama-server" }
                $PerProcess += [ordered]@{
                    pid = $ProcPid
                    process_name = $ProcName
                    used_mb = $ProcMb
                    group = $Group
                }
            }
        }
    } catch { }
    $Sample = [ordered]@{
        phase = $Phase
        sample_index = $Index
        timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        total_used_mb = $Used
        total_mb = $Total
        per_process = $PerProcess
    }
    return $Sample
}

function Add-VramSample {
    # Mintavetel + rogzites + kozvetlen konzolkiiras. Sikertelen minta: $false.
    param([string]$Phase, [int]$Index)
    $Sample = New-VramSample -Phase $Phase -Index $Index
    if ($null -eq $Sample) {
        Write-Host ("  [WARN] az nvidia-smi mintavetel nem sikerult ({0} #{1})" -f $Phase, $Index) -ForegroundColor Yellow
        return $false
    }
    $Script:SampleList += $Sample
    $LlamaMb = 0
    $OtherMb = 0
    foreach ($P in @($Sample["per_process"])) {
        if ([string]$P["group"] -eq "llama-server") {
            $LlamaMb = $LlamaMb + [int]$P["used_mb"]
        } else {
            $OtherMb = $OtherMb + [int]$P["used_mb"]
        }
    }
    Write-Host ("  {0} #{1}: total {2} MB / {3} MB | llama-server {4} MB | egyeb {5} MB" -f $Phase, $Index, [int]$Sample["total_used_mb"], [int]$Sample["total_mb"], $LlamaMb, $OtherMb)
    return $true
}

function Measure-Phase {
    # Egy fazis (nyugalmi) mintavetelei; visszater a sikeres mintak szamaval.
    param([string]$Phase, [int]$Count = 1)
    $Ok = 0
    for ($i = 1; $i -le $Count; $i++) {
        if (Add-VramSample -Phase $Phase -Index $i) { $Ok = $Ok + 1 }
    }
    return $Ok
}

function Show-LogTail {
    param([string]$Path, [int]$Tail = 15)
    if (-not (Test-Path $Path)) {
        Write-Host ("  (nincs ilyen logfajl: {0})" -f $Path)
        return
    }
    $Lines = @()
    try { $Lines = @(Get-Content -LiteralPath $Path -Tail $Tail -ErrorAction SilentlyContinue) } catch { }
    if ($Lines.Count -eq 0) {
        Write-Host ("  (a logfajl ures vagy nem olvashato: {0})" -f $Path)
        return
    }
    foreach ($L in $Lines) { Write-Host ("    {0}" -f $L) }
}

function Invoke-GenerationPhase {
    # Streaming generalas Start-Job-ban, kozben 0,5 s-enkenti mintaveteles.
    param([string]$Phase, [string]$Body)
    $Uri = ("http://127.0.0.1:{0}/v1/chat/completions" -f $Port)
    $Job = Start-Job -ScriptBlock {
        param($Uri, $Body)
        $ProgressPreference = "SilentlyContinue"
        try {
            $R = Invoke-WebRequest -Uri $Uri -Method Post -Body $Body -ContentType "application/json" -TimeoutSec 180 -UseBasicParsing
            return @{ ok = $true; status = [int]$R.StatusCode; bytes = ([string]$R.Content).Length }
        } catch {
            return @{ ok = $false; error = [string]$_.Exception.Message }
        }
    } -ArgumentList $Uri, $Body
    Write-Host "  streaming keres inditasa a hatterben (Start-Job)..."
    $Ok = 0
    $Index = 0
    while (($Job.State -eq "Running") -and ($Index -lt $Samples)) {
        $Index = $Index + 1
        if (Add-VramSample -Phase $Phase -Index $Index) { $Ok = $Ok + 1 }
        if (($Job.State -eq "Running") -and ($Index -lt $Samples)) {
            Start-Sleep -Milliseconds 500
        }
    }
    if ($Ok -eq 0) {
        # A keres minden mintavetel ELOTT vegetert - legalabb egy minta a keres
        # UTAN is ertelmes adat (a KV-cache ekkor is fent van).
        $Index = $Index + 1
        if (Add-VramSample -Phase $Phase -Index $Index) { $Ok = $Ok + 1 }
    }
    Wait-Job -Job $Job -Timeout 200 | Out-Null
    if ($Job.State -eq "Running") {
        Stop-Job -Job $Job -ErrorAction SilentlyContinue
        Remove-Job -Job $Job -Force
        Fail-Measure ("a(z) {0} fazis keres nem fejezodott be 200 s alatt" -f $Phase)
    }
    $Result = $null
    try { $Result = Receive-Job -Job $Job -ErrorAction Stop } catch { }
    Remove-Job -Job $Job -Force
    if ($null -eq $Result) {
        Fail-Measure ("a(z) {0} fazis streaming keres nem adott vissza eredmenyt" -f $Phase)
    }
    if (-not $Result["ok"]) {
        $ErrText = "ismeretlen hiba"
        try { $ErrText = [string]$Result["error"] } catch { }
        Fail-Measure ("a(z) {0} fazis streaming keres nem sikerult: {1}" -f $Phase, $ErrText)
    }
    Write-Host ("  keres kesz (HTTP {0}, {1} bajt valasz)" -f [int]$Result["status"], [int]$Result["bytes"])
    return $Ok
}

Write-Host ""
Write-Host "==================================================================="
Write-Host (" VRAM meres (Root: {0} | Port: {1} | Samples: {2})" -f $Root, $Port, $Samples)
Write-Host "==================================================================="

# ---------------------------------------------------------------------------
# nvidia-smi feloldasa (PATH + standard utvonalak)
# ---------------------------------------------------------------------------
$Script:NvidiaSmi = Resolve-NvidiaSmi
if ($null -eq $Script:NvidiaSmi) {
    Write-Host "FAIL: nem talalhato az nvidia-smi (PATH, System32, SysWOW64, NVSMI)." -ForegroundColor Red
    Write-Host "JAVITAS: telepitsd vagy frissitsd az NVIDIA drivert - az nvidia-smi a" -ForegroundColor Yellow
    Write-Host "driver resze. Kezi teszt egy PowerShell-ablakban: nvidia-smi"
    Write-Host "VRAM REPORT: FAIL - nincs nvidia-smi, a meres nem indithato" -ForegroundColor Red
    exit 1
}
Write-Host ("nvidia-smi: {0}" -f $Script:NvidiaSmi)

# ---------------------------------------------------------------------------
# [0/5] fazis: idle (nvidia-smi MINDEN muvelet ELOTT)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[0/5] fazis: idle (alapallapot, a modellbetoltes ELOTT)"
if ((Measure-Phase -Phase "idle" -Count 1) -lt 1) {
    Fail-Measure "az idle mintavetel nem sikerult - az nvidia-smi lekerdezes hibas"
}

# ---------------------------------------------------------------------------
# [1/5] fazis: model_loaded
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[1/5] fazis: model_loaded (a modell GPU-ra toltve)"
$SlsScript = Join-Path $Root "scripts\start_llama_server.ps1"
if (Test-LlamaHealth -TimeoutSec 5) {
    Write-Host ("  a szerver MAR fut a http://127.0.0.1:{0}/health cimen - nem inditunk ujat" -f $Port)
    Write-Host "  (megjegyzes: igy az idle fazis erteke mar a betoltott modellt tartalmazza)"
} elseif (-not (Test-Path $SlsScript)) {
    Fail-Measure ("nem talalhato a start_llama_server.ps1: {0}" -f $SlsScript)
} else {
    Write-Host ("  szerver inditasa rejtett ablakban: powershell -ExecutionPolicy Bypass -File {0}" -f $SlsScript)
    $Script:StarterProc = Start-Process -FilePath "powershell" -ArgumentList @("-ExecutionPolicy", "Bypass", "-File", $SlsScript) -WindowStyle Hidden -PassThru
    $Script:StartedServer = $true
    $Healthy = $false
    $Waited = 0
    $Script:StarterExitNoted = $false
    while ($Waited -lt 90) {
        Start-Sleep -Seconds 1
        $Waited = $Waited + 1
        if ($Script:StarterProc.HasExited -and (-not $Script:StarterExitNoted)) {
            $Script:StarterExitNoted = $true
            # v0.3.3: a starter exit kodja NEM sikerfeltetel - a /health az.
            # Ha a wrapper kilepett, de elo llama-server folyamat van, a
            # /health-re tovabb varunk.
            $VramLlamaAlive = @(Get-Process -Name "llama-server" -ErrorAction SilentlyContinue)
            if ($VramLlamaAlive.Count -eq 0) {
                Write-Host ("  FIGYELEM: a start_llama_server.ps1 gyerekfolyamat {0} s mulva kilepett (kilepesi kod: {1})" -f $Waited, $Script:StarterProc.ExitCode) -ForegroundColor Yellow
                Write-Host "  starter transcript (utolso 15 sor, logs\llama-server.starter.log):" -ForegroundColor Yellow
                Show-LogTail -Path (Join-Path $Root "logs\llama-server.starter.log") -Tail 15
                Write-Host "  stderr tail (utolso 15 sor, logs\llama-server.err.log):" -ForegroundColor Yellow
                Show-LogTail -Path (Join-Path $Root "logs\llama-server.err.log") -Tail 15
                Write-Host "  stdout tail (utolso 15 sor, logs\llama-server.out.log):" -ForegroundColor Yellow
                Show-LogTail -Path (Join-Path $Root "logs\llama-server.out.log") -Tail 15
                Fail-Measure "a llama-server gyors-hibaval leallt (a log-ok utolso sorai fent)"
            } else {
                Write-Host ("  MEGJEGYZES: a starter wrapper kilepett (kilepesi kod: {0}), de a llama-server folyamat el - a /health varakozas folytatodik" -f $Script:StarterProc.ExitCode) -ForegroundColor Yellow
            }
        }
        if (Test-LlamaHealth -TimeoutSec 2) { $Healthy = $true; break }
        Write-Host ("    varakozas a /health-re... {0} s" -f $Waited)
    }
    if (-not $Healthy) {
        Fail-Measure ("a /health 90 s alatt sem valaszolt (http://127.0.0.1:{0}/health)" -f $Port)
    }
    Write-Host ("  a szerver felfutott ({0} s)" -f $Waited)
}
if ((Measure-Phase -Phase "model_loaded" -Count 1) -lt 1) {
    Fail-Measure "a model_loaded mintavetel nem sikerult"
}

# ---------------------------------------------------------------------------
# [2/5] fazis: first_request (elso /v1/chat/completions hivas)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[2/5] fazis: first_request (elso valasz, stream:false)"
$BodyFirst = '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"temperature":0.0,"stream":false}'
$FirstUri = ("http://127.0.0.1:{0}/v1/chat/completions" -f $Port)
try {
    $Resp = Invoke-WebRequest -Uri $FirstUri -Method Post -Body $BodyFirst -ContentType "application/json" -TimeoutSec 120 -UseBasicParsing
    if (($Resp.StatusCode -lt 200) -or ($Resp.StatusCode -ge 300)) {
        Fail-Measure ("a first_request HTTP {0}-t adott" -f $Resp.StatusCode)
    }
    Write-Host ("  elso valasz: HTTP {0}, {1} bajt" -f [int]$Resp.StatusCode, ([string]$Resp.Content).Length)
} catch {
    Fail-Measure ("az elso /v1/chat/completions keres nem sikerult: {0}" -f $_.Exception.Message)
}
if ((Measure-Phase -Phase "first_request" -Count 1) -lt 1) {
    Fail-Measure "a first_request mintavetel nem sikerult"
}

# ---------------------------------------------------------------------------
# [3/5] fazis: normal_generation (streaming, hosszabb valasz)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/5] fazis: normal_generation (hosszabb valasz, stream:true)"
$BodyStory = '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Tell me a short story about a coffee cup in Hungarian."}],"max_tokens":200,"temperature":0.7,"stream":true}'
if ((Invoke-GenerationPhase -Phase "normal_generation" -Body $BodyStory) -lt 1) {
    Fail-Measure "a normal_generation fazisban egy mintavetel sem sikerult"
}

# ---------------------------------------------------------------------------
# [4/5] fazis: json_generation (streaming + json_object response_format)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[4/5] fazis: json_generation (stream:true + response_format json_object)"
$BodyJson = '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Return a JSON object with keys name and language, values Thomas and Hungarian."}],"max_tokens":200,"temperature":0.0,"stream":true,"response_format":{"type":"json_object"}}'
if ((Invoke-GenerationPhase -Phase "json_generation" -Body $BodyJson) -lt 1) {
    Fail-Measure "a json_generation fazisban egy mintavetel sem sikerult"
}

# ---------------------------------------------------------------------------
# [5/5] fazis: peak (a mintak csucserteke)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/5] fazis: peak (a mintak kozotti csucsertek)"
$Peak = $null
foreach ($S in $Script:SampleList) {
    if ($null -eq $Peak) { $Peak = $S; continue }
    if ([int]$S["total_used_mb"] -gt [int]$Peak["total_used_mb"]) { $Peak = $S }
}
if ($null -eq $Peak) {
    Fail-Measure "egy mintavetel sem tortent - nem szamithato peak"
}
$PeakLlama = 0
$PeakOther = 0
foreach ($P in @($Peak["per_process"])) {
    if ([string]$P["group"] -eq "llama-server") {
        $PeakLlama = $PeakLlama + [int]$P["used_mb"]
    } else {
        $PeakOther = $PeakOther + [int]$P["used_mb"]
    }
}
Write-Host ("  peak: {0} MB / {1} MB ({2} fazis, {3}. minta)" -f [int]$Peak["total_used_mb"], [int]$Peak["total_mb"], [string]$Peak["phase"], [int]$Peak["sample_index"])

# ---------------------------------------------------------------------------
# Fazisonkenti osszegzes a konzol tablazathoz es a JSON riporthoz
# ---------------------------------------------------------------------------
$PhaseNames = @("idle", "model_loaded", "first_request", "normal_generation", "json_generation")
$PhaseSummary = @()
foreach ($Ph in $PhaseNames) {
    $PhSamples = 0
    $PhMaxTotal = 0
    $PhMaxLlama = 0
    $PhMaxOther = 0
    foreach ($S in $Script:SampleList) {
        if ([string]$S["phase"] -ne $Ph) { continue }
        $PhSamples = $PhSamples + 1
        $Tot = [int]$S["total_used_mb"]
        if ($Tot -gt $PhMaxTotal) { $PhMaxTotal = $Tot }
        $Ll = 0
        $Ot = 0
        foreach ($P in @($S["per_process"])) {
            if ([string]$P["group"] -eq "llama-server") {
                $Ll = $Ll + [int]$P["used_mb"]
            } else {
                $Ot = $Ot + [int]$P["used_mb"]
            }
        }
        if ($Ll -gt $PhMaxLlama) { $PhMaxLlama = $Ll }
        if ($Ot -gt $PhMaxOther) { $PhMaxOther = $Ot }
    }
    $PhaseSummary += [ordered]@{
        phase = $Ph
        samples = $PhSamples
        max_total_used_mb = $PhMaxTotal
        max_llama_server_mb = $PhMaxLlama
        max_other_mb = $PhMaxOther
    }
}

# ---------------------------------------------------------------------------
# JSON riport: logs\vram_report.json
# ---------------------------------------------------------------------------
$LogsDir = Join-Path $Root "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}
$ReportPath = Join-Path $LogsDir "vram_report.json"
$Report = [ordered]@{
    schema = "vram_report_v1"
    generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    root = $Root
    port = $Port
    samples_requested = $Samples
    server_started_by_script = $Script:StartedServer
    samples = $Script:SampleList
    phases = $PhaseSummary
    peak = $Peak
}
$ReportJson = $Report | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($ReportPath, $ReportJson)
Write-Host ("  JSON riport kiirva: {0}" -f $ReportPath)

# ---------------------------------------------------------------------------
# Konzol: osszefogo tablazat + folyamatonkenti bontas
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "==================================================================="
Write-Host " OSSZEGZO TABLAZAT (fazisonkenti csucsertekek, MB)"
Write-Host "==================================================================="
Write-Host ("{0,-20} {1,7} {2,10} {3,12} {4,10}" -f "fazis", "mintak", "total_mb", "llama_mb", "egyeb_mb")
foreach ($Row in $PhaseSummary) {
    Write-Host ("{0,-20} {1,7} {2,10} {3,12} {4,10}" -f [string]$Row["phase"], [int]$Row["samples"], [int]$Row["max_total_used_mb"], [int]$Row["max_llama_server_mb"], [int]$Row["max_other_mb"])
}
Write-Host ("{0,-20} {1,7} {2,10} {3,12} {4,10}" -f "peak", "-", [int]$Peak["total_used_mb"], $PeakLlama, $PeakOther)

Write-Host ""
Write-Host " FOLYAMATONKENTI BONTAS (llama-server kulon az egyeb GPU-folyamoktol)"
Write-Host " (a fazis mintai kozott az adott folyamat csucserteke)"
foreach ($Ph in $PhaseNames) {
    $Agg = @()
    foreach ($S in $Script:SampleList) {
        if ([string]$S["phase"] -ne $Ph) { continue }
        foreach ($P in @($S["per_process"])) {
            $Mb = [int]$P["used_mb"]
            $Found = $false
            foreach ($A in $Agg) {
                if (($A["pid"] -eq $P["pid"]) -and ($A["process_name"] -eq $P["process_name"])) {
                    if ($Mb -gt $A["max_mb"]) { $A["max_mb"] = $Mb }
                    $Found = $true
                    break
                }
            }
            if (-not $Found) {
                $Agg += [ordered]@{
                    pid = $P["pid"]
                    process_name = $P["process_name"]
                    group = $P["group"]
                    max_mb = $Mb
                }
            }
        }
    }
    Write-Host ("  {0}:" -f $Ph)
    $HasLlama = $false
    foreach ($A in $Agg) {
        if ([string]$A["group"] -eq "llama-server") {
            Write-Host ("    - llama-server (pid {0}): {1} MB" -f [int]$A["pid"], [int]$A["max_mb"])
            $HasLlama = $true
        }
    }
    if (-not $HasLlama) {
        Write-Host "    - llama-server: nem talalhato GPU-szamitasi folyamat"
    }
    foreach ($A in $Agg) {
        if ([string]$A["group"] -ne "llama-server") {
            Write-Host ("    - egyeb: {0} (pid {1}): {2} MB" -f [string]$A["process_name"], [int]$A["pid"], [int]$A["max_mb"])
        }
    }
}

# ---------------------------------------------------------------------------
# Takaritas: CSAK a sajat magunk altal inditott szerver all le
# (a verify_m1.ps1 mintajara - a kulso szervert nem nyuljuk)
# ---------------------------------------------------------------------------
Stop-OwnServer

# ---------------------------------------------------------------------------
# Vegso sor + kilepesi kod
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("VRAM REPORT: PASS - peak {0} MB / {1} MB, {2} minta - JSON: {3}" -f [int]$Peak["total_used_mb"], [int]$Peak["total_mb"], $Script:SampleList.Count, $ReportPath) -ForegroundColor Green
exit 0
