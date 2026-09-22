<#
===============================================================================
experimental/llama_b11073/install_b11073.ps1 - INSTALLER (no binaries in pack)

Installs the VALIDATED EXPERIMENTAL llama.cpp b11073 runtime for VoiceMem
into the ISOLATED location  experimental\llama_b11073\bin\  by downloading
the pinned release asset at installation time. The pack itself contains NO
binaries, NO DLLs, NO models, NO CUDA components - only this installer, the
configurator, verification scripts and documentation.

THE SAME MODEL AS THE PRODUCTION INSTALLER (scripts/install_m1.ps1):
    pinned llama.cpp release tag + asset name, Invoke-WebRequest download,
    staged Expand-Archive, DLL-set-aware post-extract validation, fail
    loudly with an actionable hint. There is NO second dependency-management
    system: the runtime lands inside the existing VoiceMem tree.

PINNED RUNTIME IDENTITY (fail-closed - never silently use another version):
    tag     : b11073            (llama-server 0.4.1-dev, commit 1aa2954bd)
    asset   : llama-b11073-bin-win-cuda-13.4-x64.zip
              (b11073 ships no cuda-13.3 asset; 13.4 is the same-major
              successor of the production b10717 cuda-13.3 line; the 12.4
              line lacks Blackwell support)
    size    : 150,093,926 bytes
    sha256  : 85C1B874180FAEC412CCBBA16EE0833062C28E2BF1390B12ED30F7DC6D7D79C4
              (== the GitHub API digest of the release asset; verified at
              pack-build time against the actually downloaded bytes)

WHAT IS INSTALLED (measured dependency closure - see DEPENDENCY_INSPECTION.md):
    24 files, NOTHING else from the zip: llama-server.exe + the 7
    application-local DLLs of its import closure + ggml-cuda.dll + the 14
    ggml-cpu-*.dll dynamic backends + LICENSE-LLVM-OpenMP. No CUDA DLL, no
    driver, no model, no application file is ever packaged or downloaded.

WHAT IS NEVER TOUCHED (hard guarantees):
    - bin\llama-server.exe            (the pinned PRODUCTION b10717 binary)
    - bin\*.dll                       (the production cudart / CUDA set)
    - config\llm_config.yaml         (the canonical production profile)
    - config\env.local.ps1            (the production environment)
    - MODELS.lock.json, VERSION, scripts\start_llama_server.ps1
    Everything is written under experimental\llama_b11073\ ONLY.

PREREQUISITE (NOT installed by this script - the target already has it):
    cublas64_13.dll, the single CUDA DLL the runtime imports, resolves from
    the EXISTING production bin\ (b10717 cudart install) and/or a system
    CUDA 13.x. The script PROBES and reports where it will resolve from.

USAGE (from the VoiceMemAgent root, 64-bit PowerShell):
    powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\install_b11073.ps1
    ... -ZipPath <pre-downloaded llama-b11073-bin-win-cuda-13.4-x64.zip>   # offline/slow link
    ... -Force                                                              # reinstall

EXIT CODES: 0 = installed (or already installed) | 1 = failure (see log).

NOTE: this file is deliberately ASCII (PowerShell 5.1 reads BOM-less files
as Windows-1252; accented characters would be mojibake on the target).
===============================================================================
#>

param(
    [string]$ZipPath = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# --- pinned runtime identity (cross-checked against config\b11073.pins.json) --
$PinTag    = "b11073"
$PinAsset  = "llama-b11073-bin-win-cuda-13.4-x64.zip"
$PinUrl    = "https://github.com/ggml-org/llama.cpp/releases/download/b11073/llama-b11073-bin-win-cuda-13.4-x64.zip"
$PinSha256 = "85C1B874180FAEC412CCBBA16EE0833062C28E2BF1390B12ED30F7DC6D7D79C4"
$PinBytes  = 150093926
$PinCommit = "1aa2954bd"

# the measured category-B file set (see DEPENDENCY_INSPECTION.md; 24 entries)
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

# --- locations ------------------------------------------------------------------
$ScriptDir = $PSScriptRoot
$Root      = (Get-Item -LiteralPath $ScriptDir).Parent.Parent.FullName
$BinDir    = Join-Path $ScriptDir "bin"
$ConfigDir = Join-Path $ScriptDir "config"
$LogsDir   = Join-Path $ScriptDir "logs"
$PinsFile  = Join-Path $ConfigDir "b11073.pins.json"
$Manifest  = Join-Path $ConfigDir "runtime_manifest.json"

$Stamp     = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile   = Join-Path $LogsDir ("install_" + $Stamp + ".log")

function Write-Log([string]$Line) {
    Write-Host $Line
    try {
        if (-not (Test-Path -LiteralPath $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null }
        Add-Content -LiteralPath $LogFile -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $Line)
    } catch { }
}

function Fail([string]$Msg, [string]$Hint) {
    Write-Host ""
    Write-Host ("HIBA: " + $Msg) -ForegroundColor Red
    if ($Hint) { Write-Host ("JAVITAS: " + $Hint) -ForegroundColor Yellow }
    Write-Log ("FAIL: " + $Msg + " | HINT: " + $Hint)
    exit 1
}

Write-Host "=== EXPERIMENTAL llama.cpp b11073 INSTALLER (teleptio) ===" -ForegroundColor Magenta
Write-Log ("installer start; root = " + $Root)

# --- 0) elofeltetel: tenyleg egy VoiceMemAgent gyokerben vagyunk ----------------
foreach ($Required in @(
    (Join-Path $Root "START.bat"),
    (Join-Path $Root "config\env.local.ps1"),
    (Join-Path $Root "scripts\start_agent.ps1"))) {
    if (-not (Test-Path -LiteralPath $Required)) {
        Fail ("nem VoiceMemAgent gyokerben fut a telepito (hianyzik: " + $Required + ")") `
             ("csomagold ki a VoiceMemAgent_v0.10.7_LlamaB11073_Installer.zip-et a VoiceMemAgent v0.10.7 GYOKEREBE, ugy hogy az experimental\llama_b11073\ mappa a gyokerben keletkezzen, majd inditsd ujra")
    }
}
Write-Log "voiceMem root verified (START.bat + config + scripts present)"

# --- 1) pins.json betoltese + KERESZT-ELLENORZES a beagyazott pinekkel ----------
if (-not (Test-Path -LiteralPath $PinsFile)) {
    Fail ("nem talalhato a pin-fajl: " + $PinsFile) `
         ("a telepito csomag resze kell legyen (config\b11073.pins.json). Csomagold ki ujra a ZIP-et.")
}
try {
    $Pins = Get-Content -LiteralPath $PinsFile -Raw | ConvertFrom-Json
} catch {
    Fail ("a pin-fajl nem olvashato/ervenytelen JSON: " + $_.Exception.Message) "serult csomag; csomagold ki ujra a ZIP-et."
}
if ($Pins.runtime.tag -ne $PinTag -or
    $Pins.runtime.asset.url -ne $PinUrl -or
    $Pins.runtime.asset.sha256 -ne $PinSha256 -or
    [int]$Pins.runtime.asset.size_bytes -ne $PinBytes) {
    Fail ("a pin-fajl es a telepito beagyazott pinje ELTER (tag/url/sha256/size)") `
         ("a csomag belso konzisztencia-serulese - hasznalj ervenyes VoiceMemAgent_v0.10.7_LlamaB11073_Installer.zip-et. SOHA ne lepj at eltero verziora.")
}
$PinFileTable = @{}
foreach ($f in $Pins.runtime.files) { $PinFileTable[[string]$f.name] = ([string]$f.sha256).ToUpper() }
if ($PinFileTable.Count -ne $PackFiles.Count) {
    Fail ("a pin-fajl fajllista merete (" + $PinFileTable.Count + ") elter a mert 24-elemu zarastol") "serult csomag."
}
Write-Log ("pins.json cross-check OK: tag=" + $PinTag + " asset=" + $PinAsset + " sha256=" + $PinSha256.Substring(0,12) + "... files=" + $PinFileTable.Count)

# --- 2) idempotencia -------------------------------------------------------------
$ExeInstalled = Join-Path $BinDir "llama-server.exe"
if ((Test-Path -LiteralPath $ExeInstalled) -and (-not $Force)) {
    $When = "?"
    if (Test-Path -LiteralPath $Manifest) {
        try { $When = [string](Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json).installed_utc } catch { }
    }
    Write-Host ""
    Write-Host ("Mar telepitve van: " + $ExeInstalled) -ForegroundColor Yellow
    Write-Host ("  (telepites ideje: " + $When + "; ujrateszteshez: -Force)") -ForegroundColor Yellow
    Write-Host "A kovetkezo lepes: configure_b11073.ps1 -Enable"
    Write-Log "already installed; exiting without changes (use -Force to reinstall)"
    exit 0
}
if ($Force) { Write-Log "-Force: ujratesztes"; }

# --- 3) a runtime ZIP megszerzese (letoltes vagy elore letoltott fajl) -----------
$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) "voicemem-b11073-install"
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null
$ZipFile = ""

if ($ZipPath -ne "") {
    if (-not (Test-Path -LiteralPath $ZipPath)) {
        Fail ("a megadott -ZipPath nem letezik: " + $ZipPath) "add meg a teljes utvonalat a llama-b11073-bin-win-cuda-13.4-x64.zip fajlhoz."
    }
    $ZipFile = $ZipPath
    Write-Log ("using pre-downloaded zip: " + $ZipFile)
} else {
    $ZipFile = Join-Path $WorkDir $PinAsset
    Write-Host ("Letoltes: " + $PinUrl)
    Write-Log ("downloading " + $PinUrl)
    $ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest progress bar = 10x lassabb nagy fajlnal
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072 } catch { }  # TLS 1.2 (PS 5.1)
    $DlOk = $false
    for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
        try {
            Invoke-WebRequest -Uri $PinUrl -OutFile $ZipFile -UseBasicParsing
            $DlOk = $true; break
        } catch {
            Write-Host ("  letoltesi kiserlet " + $Attempt + "/3 sikertelen: " + $_.Exception.Message) -ForegroundColor Yellow
            Write-Log ("download attempt " + $Attempt + " failed: " + $_.Exception.Message)
            Start-Sleep -Seconds 3
        }
    }
    if (-not $DlOk) {
        Fail ("a(z) " + $PinTag + " runtime letoltese 3 probalkozas utan sem sikerult") `
             ("1) probald ujra (lassu GitHub atiranyitas). 2) Kezi letoltes: " + $PinUrl + " majd: install_b11073.ps1 -ZipPath <letoltott fajl>")
    }
}

# --- 4) SHA-256 ELLENORZES A KICSOMAGOLAS ELOTT (fail-closed) ---------------------
Write-Host ("SHA-256 ellenorzes (vart: " + $PinSha256.Substring(0, 16) + "...)")
$Actual = (Get-FileHash -LiteralPath $ZipFile -Algorithm SHA256).Hash
Write-Log ("zip sha256 actual=" + $Actual)
if ($Actual -ne $PinSha256) {
    Fail ("SHA-256 ELTERES - a letoltott fajl NEM a pin-elt " + $PinTag + " runtime!`n         vart:    " + $PinSha256 + "`n         kapott:  " + $Actual) `
         ("SOHA ne folytasd masik verzioval. 1) torold a fajlt es letoltsd ujra. 2) ha -ZipPath-tal futtattad: a fajl serult vagy mas release. 3) a hiba perzisztens eseten a llama.cpp b11073 release megvaltozhatott - ne hasznald, amig a pack nem frissul.")
}
$ZipLen = (Get-Item -LiteralPath $ZipFile).Length
if ($ZipLen -ne $PinBytes) {
    Fail ("meret-elteres: vart " + $PinBytes + " bajt, kapott " + $ZipLen) "a SHA-256 egyezett, de a meret nem - erdemes ujra letolteni."
}
Write-Host "  OK: a SHA-256 es a meret is egyezik a pin-elt release-essel." -ForegroundColor Green
Write-Log ("zip verified: sha256+size match (" + $PinBytes + " bytes)")

# --- 5) kicsomagolas stagingbe, majd a 24 fajl meretezett masolasa ----------------
$Stage = Join-Path $WorkDir ("stage_" + $Stamp)
New-Item -ItemType Directory -Path $Stage -Force | Out-Null
Write-Host "Kicsomagolas (staging)..."
Expand-Archive -LiteralPath $ZipFile -DestinationPath $Stage -Force
New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
$Copied = 0
foreach ($f in $PackFiles) {
    $Src = Get-ChildItem -LiteralPath $Stage -Recurse -File -Filter $f | Select-Object -First 1
    if (-not $Src) {
        Fail ("a pin-elt zip-ben hianyzik a fajl: " + $f) "valtozott a release asset-struktura; ne kezzel masolj semmit - ertesitsd a pack keszitojet."
    }
    Copy-Item -LiteralPath $Src.FullName -Destination (Join-Path $BinDir $f) -Force
    $Copied++
}
Write-Log ("copied " + $Copied + " measured-closure files into " + $BinDir)

# --- 6) kicsomagolt allapot validalasa (fajlonkenti SHA-256) ---------------------
$Bad = 0
foreach ($f in $PackFiles) {
    $T = Join-Path $BinDir $f
    if (-not (Test-Path -LiteralPath $T)) { Fail ("telepites utani ellenorzes: hianyzik " + $f) "rejtvany - futtasd ujra -Force-ral."; }
    $H = (Get-FileHash -LiteralPath $T -Algorithm SHA256).Hash
    if ($H -ne $PinFileTable[$f]) {
        Fail ("telepitesi fajl-ellenorzes HIBA: " + $f + "`n         vart:    " + $PinFileTable[$f] + "`n         kapott:  " + $H) "a kicsomagolas serult - torold az experimental\llama_b11073\bin mappat es futtasd ujra."
    }
}
Write-Host ("  OK: mind a " + $PackFiles.Count + " fajl SHA-256 hash-e egyezik a pin-elt ertekekkel.") -ForegroundColor Green
Write-Log ("per-file hash verification: " + $PackFiles.Count + "/" + $PackFiles.Count + " OK")

# --- 7) build-identitas: a binaris TENYLEG a b11073 commit-ja ---------------------
$Impl = Join-Path $BinDir "llama-server-impl.dll"
try {
    $Bytes = [System.IO.File]::ReadAllBytes($Impl)
    $Text  = [System.Text.Encoding]::ASCII.GetString($Bytes)
} catch {
    Fail ("nem olvashato a " + $Impl + ": " + $_.Exception.Message) "serult letoltes?"
}
if (-not $Text.Contains($PinCommit)) {
    Fail ("a binaris-ban NINCS meg a b11073 build-identitas (" + $PinCommit + ") - masik build!") "a SHA-256 egyezett, de az identitas-szoveg hianyzik: NE hasznald; ertesitsd a pack keszitojet."
}
Write-Host ("  OK: build-identitas '" + $PinCommit + "' megtalalva a llama-server-impl.dll-ben (b11073).") -ForegroundColor Green
Write-Log ("build identity verified in-binary: " + $PinCommit)

# --- 8) PRODUKCIO-ERVEINTETLENSSEG BIZONYITASA -----------------------------------
$ProdExe = Join-Path $Root "bin\llama-server.exe"
$ProdHash = ""
if (Test-Path -LiteralPath $ProdExe) {
    $ProdHash = (Get-FileHash -LiteralPath $ProdExe -Algorithm SHA256).Hash
    Write-Log ("production bin\llama-server.exe sha256 (before, and untouched after): " + $ProdHash)
    Write-Host ("  [info] a gyartasi bin\llama-server.exe (b10717) hash-e rogzitve: " + $ProdHash.Substring(0, 12) + "... - a telepito SOHA nem irja.")
} else {
    Write-Host "  [info] a gyartasi bin\llama-server.exe meg nincs telepitve (START.bat meg nem futott) - nincs mit erinteni." -ForegroundColor DarkGray
}

# --- 9) cublas elofeltetel PROBA (nem telepitjuk - a celgep biztositja) -----------
$CudaName = "cublas64_13.dll"
$CudaFound = ""
$SearchDirs = @($BinDir, (Join-Path $Root "bin"))
if ($env:windir) { $SearchDirs += (Join-Path $env:windir "System32") }
$SearchDirs += @([Environment]::GetEnvironmentVariable("PATH") -split ([System.IO.Path]::PathSeparator) | Where-Object { $_ })
foreach ($d in $SearchDirs) {
    if ((Test-Path -LiteralPath (Join-Path $d $CudaName))) { $CudaFound = Join-Path $d $CudaName; break }
}
if ($CudaFound) {
    Write-Host ("  [info] " + $CudaName + " feloldva: " + $CudaFound) -ForegroundColor Green
    Write-Log ("cublas prerequisite resolves from: " + $CudaFound)
} else {
    Write-Host ("  [FIGYELMEZTETES] " + $CudaName + " NINCS a loader keresesi utvonalan.") -ForegroundColor Yellow
    Write-Host "  A telepites rendben van, de a SZERVER INDITASA eltagadt (0xC0000135)." -ForegroundColor Yellow
    Write-Host "  Varhato forras: a gyartasi bin\ cudart letoltes (START.bat egyszeri futtatasa)," -ForegroundColor Yellow
    Write-Host "  vagy rendszer-szintu CUDA 13.x. Reszletek: README.md 'Elofeltetelek' fejezet." -ForegroundColor Yellow
    Write-Log ("cublas prerequisite NOT found on loader search path (warning)")
}

# --- 10) manifest + zaroveg -------------------------------------------------------
$FileHashes = @{}
foreach ($f in $PackFiles) { $FileHashes[$f] = $PinFileTable[$f] }
$ManifestObj = [ordered]@{
    component         = "llama-server-experimental"
    tag               = $PinTag
    build_commit      = $PinCommit
    asset_name        = $PinAsset
    asset_url         = $PinUrl
    asset_sha256      = $PinSha256.ToLower()
    asset_size_bytes  = $PinBytes
    installed_utc     = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    installed_by      = "VoiceMemAgent_v0.10.7_LlamaB11073_Installer"
    install_dir       = "experimental/llama_b11073/bin"
    files_sha256      = $FileHashes
    production_llama_server_sha256 = $ProdHash
}
$ManifestObj | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Manifest -Encoding ASCII
Write-Log ("manifest written: " + $Manifest)

Write-Host ""
Write-Host "TELEPITES KESZ." -ForegroundColor Green
Write-Host ("  hely     : " + $BinDir)
Write-Host ("  runtime  : llama.cpp " + $PinTag + " (0.4.1-dev, commit " + $PinCommit + "), win x64 CUDA 13.4")
Write-Host ("  fajlok   : " + $PackFiles.Count + " (mert fuggosegi zaras; l. DEPENDENCY_INSPECTION.md)")
Write-Host "  NEM turte: bin\llama-server.exe (b10717), bin\*.dll, config\llm_config.yaml,"
Write-Host "             config\env.local.ps1, MODELS.lock.json, VERSION - semmi produkcios fajl."
Write-Host ""
Write-Host "Kovetkezo lepesek:"
Write-Host "  1. experimental\llama_b11073\configure_b11073.ps1 -Enable"
Write-Host "  2. experimental\llama_b11073\start_llama_server_experimental.ps1   (port 8081)"
Write-Host "  3. experimental\llama_b11073\verify_b11073.ps1"
Write-Log "INSTALL COMPLETE"
exit 0
