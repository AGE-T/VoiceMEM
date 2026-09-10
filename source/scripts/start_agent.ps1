<#
===============================================================================
scripts/start_agent.ps1 - M1 agent indito (.venv\Scripts\python.exe -m app.main)

MIT csinal (a felhasznalo M0 specjenek 20. pontja szerint):
  1. ellenorzi a .venv-et (ha nincs: a telepito futtatasa kell)
  2. rovid asset-ellenorzes: .venv python -m app.main --check
  3. ellenorzi a konfiguraciot (config\voicemem_config.yaml + env.local.ps1)
  4. ha a llama-server nem fut (127.0.0.1:8080/health), UJ ABLAKBAN elinditja
     a scripts\start_llama_server.ps1-t es megvari a health-t (max 60 s,
     2 s-enkenti poll); -NoServer kapcsoloval kihagyhato, ha valaki kezzel
     mar inditotta
  5. betolti a config\env.local.ps1-t (offline env-ek) CSAK HA a fo env-
     valtozok nincsenek beallitva, majd a repo gyokererol allva elinditja:
     .venv\Scripts\python.exe -m app.main
  6. minden egyeb argumentum athalad a pythonnak (@PassThrough)

Leallitas: Ctrl+C.

JAVASOLT SMOKE SORREND:
    .\scripts\start_agent.ps1 --check        # asset-lista (verify helyett)
    .\scripts\start_agent.ps1 --mock         # 3 fordulos demo mock komponensekkel:
                                              #   1. EN nyelvtani hiba javitasa
                                              #   2. HU kerdes valasza
                                              #   3. memoria-visszaidazas
    .\scripts\start_agent.ps1 --mock --barge-in-demo   # megszakitas demo
    .\scripts\start_agent.ps1                # csak ezutan VALOS uzem

ARGUMENTUM-ATALAS: ez a szkript NINCS param() blokkja - minden argumentum
athalad a python -m app.main-nak (--check, --mock, --list-config,
--text "...", --barge-in-demo, --config, stb.). Harom kivetel:
    -Root <utvonal> / --root <utvonal>   - a runtime gyoker (env.local.ps1-nek)
    -NoServer / --no-server              - a llama-server auto-inditas kihagyasa
    egy-vesszos, csak betuket tartalmazo flag-ek automatikus atalakitasa:
    a PowerShell a -Mock alakot parameternek vennE, ezert a szkript
    -Mock -> --mock, -Check -> --check normalizalast vegez.
    (Tipp: dupla vesszot irj, az a biztonsagos forma.)

ELOFELTEL: kesz telepites (scripts\install_m1.ps1 lefuttatva - a szkript
ellenorzi a .venv-et es a konfiguraciot).

Megjegyzes: a fajl szandekosan ASCII - a PowerShell 5.1 BOM nelkul
Windows-1252-kent olvasna az ekezetes karaktereket.
#>

# --- Root feloldasa + argumentumok atalakitasa -------------------------------
# NINCS param() blokk: minden argumentum meg sem jelenik parameter-kotesben,
# igy a --mock es a -Mock alak is athalad (az utobbi normalizalva lesz).
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Root = $RepoRoot
$PassThrough = @()
$SkipNext = $false
$NoServer = $false
$WebMode = $false
for ($i = 0; $i -lt $Args.Count; $i++) {
    $Arg = [string]$Args[$i]
    if ($SkipNext) { $SkipNext = $false; continue }
    if ($Arg -eq "-Root" -or $Arg -eq "--root") {
        if (($i + 1) -lt $Args.Count) {
            $Root = [string]$Args[$i + 1]
            $SkipNext = $true
        }
        continue
    }
    if ($Arg -eq "-NoServer" -or $Arg -eq "--no-server") {
        $NoServer = $true
        continue
    }
    if ($Arg -eq "-Web" -or $Arg -eq "--web") {
        # v0.4.0: the normal user flow is the LOCAL WEB UI (port 8787) -
        # the browser opens, the full pipeline runs in the local backend.
        $WebMode = $true
        continue
    }
    if ($Arg -match '^-[A-Za-z]+$') {
        # egy-vesszos, csak betuket tartalmazo flag: -Mock -> --mock
        $PassThrough += ("--" + $Arg.Substring(1).ToLower())
        continue
    }
    $PassThrough += $Arg
}

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

# --- v0.3.4: UTF-8 stdio a pythonhoz (celgepi field report #4) ----------------
# A --check es az agent kimenetenek magyar ekezetes szovege ATIRANYITOTT
# (pipe-on elfogott) kimeneten cp1252 kodolassal UnicodeEncodeError-t adott
# a celgepen. A PYTHONIOENCODING CSAK az stdio kodolasat allitja UTF-8-ra
# (szandekosan NEM PYTHONUTF8: a fajl-I/O alapertelmezeset nem valtoztatja),
# a [Console]::OutputEncoding pedig a kozos konzolt UTF-8-ra allitja - igy a
# PowerShell is helyesen dekodolja a python kimenetet.
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# --- 1) .venv ellenorzese ------------------------------------------------------
if (-not (Test-Path $VenvPython)) {
    Write-Host "HIBA: nem talalhato a .venv python: $VenvPython" -ForegroundColor Red
    Write-Host "JAVITAS: inditsd ujra a START.bat-ot - a bootstrap automatikusan letrehozza a .venv-et es javitja a kort." -ForegroundColor Yellow
    exit 1
}

# --- 2) Modellek gyors ellenorzese (a CLI check futtatasa) --------------------
$HasSpecialFlag = $false
foreach ($A in $PassThrough) {
    if ($A -eq "--check" -or $A -eq "--mock" -or $A -eq "--list-config") {
        $HasSpecialFlag = $true
    }
}
if (-not $HasSpecialFlag) {
    Write-Host "Rovid asset-ellenorzes: .venv\Scripts\python.exe -m app.main --check"
    Push-Location $RepoRoot
    & $VenvPython -m app.main --check
    $CheckExit = $LASTEXITCODE
    Pop-Location
    if ($CheckExit -ne 0) {
        Write-Host "FIGYELEM: az asset-ellenorzes hibas kimenetet adott (l. fent)." -ForegroundColor Yellow
        Write-Host "          Ha modell/binaris hianyzik: inditsd ujra a START.bat-ot (automatikus javitas), vagy START.bat repair-t." -ForegroundColor Yellow
    }
}

# --- 3) Konfiguracio ellenorzese -----------------------------------------------
$YamlFile = Join-Path $RepoRoot "config\voicemem_config.yaml"
$EnvScript = Join-Path $RepoRoot "config\env.local.ps1"
if (-not (Test-Path $YamlFile)) {
    Write-Host "HIBA: nem talalhato a config\voicemem_config.yaml: $YamlFile" -ForegroundColor Red
    Write-Host "JAVITAS: repo-fajl hianyzik - inditsd ujra a START.bat-ot; ha nem javul, csomagold ki / klonozd ujra a projektet." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $EnvScript)) {
    Write-Host "HIBA: nem talalhato a config\env.local.ps1: $EnvScript" -ForegroundColor Red
    Write-Host "JAVITAS: repo-fajl hianyzik - inditsd ujra a START.bat-ot; ha nem javul, csomagold ki / klonozd ujra a projektet." -ForegroundColor Yellow
    exit 1
}
Write-Host "Konfiguracio rendben (config\voicemem_config.yaml + config\env.local.ps1)."

# --- 4) llama-server inditasa (ha meg nem fut) --------------------------------
function Test-LlamaHealth {
    try {
        $R = Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -Method Get -TimeoutSec 2 -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

if ($NoServer) {
    Write-Host "-NoServer: a llama-server auto-inditas kihagyva."
} elseif (Test-LlamaHealth) {
    Write-Host "llama-server mar fut (127.0.0.1:8080/health OK)."
} else {
    Write-Host "A llama-server nem fut - inditas uj ablakban (scripts\start_llama_server.ps1)..."
    $SlsScript = Join-Path $RepoRoot "scripts\start_llama_server.ps1"
    if (-not (Test-Path $SlsScript)) {
        Write-Host "HIBA: nem talalhato a scripts\start_llama_server.ps1" -ForegroundColor Red
        exit 1
    }
    Start-Process -FilePath "powershell" -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", $SlsScript) -WorkingDirectory $RepoRoot
    $Healthy = $false
    $Waited = 0
    while ($Waited -lt 60) {
        Start-Sleep -Seconds 2
        $Waited = $Waited + 2
        if (Test-LlamaHealth) { $Healthy = $true; break }
        Write-Host ("  varakozas a llama-serverre... {0} s" -f $Waited)
    }
    if ($Healthy) {
        Write-Host "llama-server felfutott."
    } else {
        Write-Host "FIGYELEM: a llama-server 60 s alatt nem valaszolt - az agent ugy is elindul," -ForegroundColor Yellow
        Write-Host "          de a valaszok helyett LlmUnavailableError-t fogsz kapni." -ForegroundColor Yellow
        # v0.3.1: a start-szkript a logs\ mappaba iranyitja a szerver stdout/stderr-t -
        # a tenyleges hibauzenetet azonnal itt is mutatjuk (nem csak "nezd az ablakot").
        # v0.3.2: eloszor a STARTER SAJAT kimenetet (transcript) mutatjuk - az
        # uj ablak lathato, de ez a futtas logja akkor is megmarad utolag.
        foreach ($LogName in @("llama-server.starter.log", "llama-server.err.log")) {
            $LogPath = Join-Path $RepoRoot ("logs\" + $LogName)
            if (Test-Path $LogPath) {
                $Tail = @(Get-Content -Path $LogPath -Tail 15 -ErrorAction SilentlyContinue)
                if ($Tail.Count -gt 0) {
                    Write-Host ("          A start-szkript / szerver utolso sorai (logs\{0}):" -f $LogName) -ForegroundColor Yellow
                    foreach ($Line in $Tail) { Write-Host ("            {0}" -f $Line) }
                }
            }
        }
        Write-Host "          (a szerver ablaka es a scripts\start_llama_server.ps1 kimenet)" -ForegroundColor Yellow
    }
}

# --- 4b) v0.4.7: memoria-lokalisacio (egyszeri, idempotens) -------------------
# A vendored VoiceMem korabban KINAIUL kerte a trait-cimkeket; a v0.4.7
# folt utan az uj memoriai angolul szuletnek, de a MAR TAROLT kinai
# allitasokat ez a lepes forditja le (llama-server / Qwen3.6). Idempotens:
# nincs Han-karakteres sor -> masodperc alatt kilep, ezert minden inditasnal
# lefut, de csak elso alkalommal vegzik munkat. A web backend ELOTT fut,
# igy az elso /api/memories mar angol cimkeket ad.
$LlScript = Join-Path $RepoRoot "scripts\localise_memories.py"
if ((Test-Path $LlScript) -and (Test-LlamaHealth)) {
    Write-Host "Memoria-lokalisacio ellenorzese (idempotens)..."
    $LlLog = Join-Path $RepoRoot "logs\localise_memories.log"
    & $VenvPython $LlScript *> $LlLog
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FIGYELEM: a memoria-lokalisacio nem futott le teljesen (llama-server elerheto?)" -ForegroundColor Yellow
        Write-Host "          A mar meglevo kinai cimkek veglegesitesehoz: .venv\Scripts\python.exe scripts\localise_memories.py" -ForegroundColor Yellow
        if (Test-Path $LlLog) { Get-Content -Path $LlLog -Tail 5 | ForEach-Object { Write-Host ("          {0}" -f $_) } }
    } else {
        $LlTail = ""
        if (Test-Path $LlLog) { $LlTail = (Get-Content -Path $LlLog -Tail 1 -ErrorAction SilentlyContinue) }
        if ($LlTail) { Write-Host ("  {0}" -f $LlTail) }
    }
}

# --- 5) env betoltese (offline mod) CSAK ha meg nincs beallitva -----------------
$NeedEnv = -not ($env:VOICEMEM_HOME -and $env:HF_HUB_OFFLINE -and $env:OPENAI_BASE_URL)
if ($NeedEnv) {
    . $EnvScript -Root $Root
}

# --- 6) Agent inditasa a repo gyokererol ----------------------------------------
Write-Host ""
Write-Host "Agent inditasa: .venv\Scripts\python.exe -m app.main"
Write-Host "Atadott argumentumok:"
if ($PassThrough.Count -eq 0) {
    Write-Host "  (nincs)"
} else {
    Write-Host "  $($PassThrough -join ' ')"
}
# --- 6b) WEB mod: lokalis web backend + bongeszo (v0.4.0) ---------------------
if ($WebMode) {
    $WebPort = 8787
    if ($env:VOICEMEM_WEB_PORT) { $WebPort = [int]$env:VOICEMEM_WEB_PORT }
    $WebUrl = "http://127.0.0.1:" + $WebPort + "/"
    Write-Host ""
    Write-Host "A lokalis web backend inditasa: .venv\Scripts\python.exe -m app.web_server @PassThrough"
    Write-Host ("A web UI cime: " + $WebUrl)
    Write-Host "Leallitas: Ctrl+C (a bongeszo lapja is zarodjon)"
    Write-Host ""

    # Start the backend as a CHILD process of this console so Ctrl+C stops it
    # together with the script (no orphan uvicorn left behind).
    $ExitCode = 1
    Push-Location $RepoRoot
    $BackendStarted = $false
    try {
        # Fire-and-track: spawn the backend, poll /api/health, open browser.
        $WebArgs = @("-u", "-m", "app.web_server", "--host", "127.0.0.1", "--port", "$WebPort") + $PassThrough
        $Proc = Start-Process -FilePath $VenvPython -ArgumentList $WebArgs `
            -WorkingDirectory $RepoRoot -PassThru -NoNewWindow
        function Test-WebHealth {
            try {
                $R = Invoke-WebRequest -Uri ("http://127.0.0.1:" + $WebPort + "/api/health") `
                    -Method Get -TimeoutSec 2 -UseBasicParsing
                if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
            } catch { }
            return $false
        }
        $WaitedWeb = 0
        while ($WaitedWeb -lt 45) {
            if ($Proc.HasExited) { break }
            if (Test-WebHealth) { $BackendStarted = $true; break }
            Start-Sleep -Milliseconds 700
            $WaitedWeb += 0.7
        }
        if ($BackendStarted) {
            Write-Host ("A web backend kesz: " + $WebUrl) -ForegroundColor Green
            Write-Host "A bongeszo automatikus megnyitasa..."
            Start-Process $WebUrl
            Write-Host ""
            Write-Host "================================================================"
            Write-Host ("  VOICEMEM WEB UI: " + $WebUrl)
            Write-Host "  A lap a kovetkezo mukodesre kesz:"
            Write-Host "   - mikrofonvalaszto + teszt (bal panel, Live input)"
            Write-Host "   - beszeld a mikrofont: VAD -> Qwen3 ASR -> memoria -> Qwen3.6 35B -> Piper"
            Write-Host "   - vagy gepyel be szoveget (ASR bypass, hibakereseshez)"
            Write-Host "   - Pipeline panel: VAD/ASR/Memory/Embedding/LLM/TTS allapot + idozitok"
            Write-Host "     es elosutesben: mikrofon-keretek, VAD-szint, esemenylog (v0.4.1)"
            Write-Host "   - Memory Space: letrehozas / valtas / grafikon / export"
            Write-Host ("  Diagnosztikai log: " + (Join-Path $RepoRoot "logs\web-server.log"))
            Write-Host "  Az elso inditasnal a modellek hatterben felmelegednek - latszik a Pipeline panelben."
            Write-Host "  Leallitas: Ctrl+C ebben az ablakban."
            Write-Host "================================================================"
            Write-Host ""
            # Wait for the child (Ctrl+C propagates to the console group).
            Wait-Process -Id $Proc.Id
            $ExitCode = $Proc.ExitCode
            if ($null -eq $ExitCode) { $ExitCode = 0 }
        } else {
            if (-not $Proc.HasExited) { try { Stop-Process -Id $Proc.Id -Force } catch { } }
            Write-Host "HIBA: a web backend 45 mp alatt nem valaszolt a /api/health-re." -ForegroundColor Red
            Write-Host "Nz meg: logs\ mappa es a fenti konzolkimenet. Gyakori ok:" -ForegroundColor Yellow
            Write-Host "  - hianyzo csomag (fastapi/uvicorn/websockets) -> START.bat repair" -ForegroundColor Yellow
            Write-Host "  - foglalt port " $WebPort "- zarj be masik 8787-es folyamatot" -ForegroundColor Yellow
            $ExitCode = 1
        }
    }
    finally {
        Pop-Location
    }
    exit $ExitCode
}

Write-Host "Leallitas: Ctrl+C"
Write-Host ""

$ExitCode = 1
Push-Location $RepoRoot
try {
    & $VenvPython -m app.main @PassThrough
    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $ExitCode
