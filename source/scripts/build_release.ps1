<#
===============================================================================
scripts/build_release.ps1 - verzionalt ZIP-kiadas (M0.1 release-munkafolyamat)

"Verziozott ZIP minden frissites utan": a szkript a KOTELEZO TESZTKAPU utan
kesit egy verzionalt ZIP fajlt a releases\ mappaba. A kapu szabalya
(felhasznaloi kovetelmeny): a teljes tesztkeszlet (tests/unit +
tests/integration + tests/benchmark + tests/validation) MINDEN ZIP-build
ELOTT lefut a scripts\run_tests.ps1-en at. Ha egyetlen teszt is elbukik,
NEM keszul ZIP (kilepesi kod 1), es a kiserlet a releases\BUILD_HISTORY.json
-be kerul. Hamis sikert soha nem ir ki.

Ajanlott rutin egy frissites utan (napi munkafolyamat):
    .\scripts\build_release.ps1 -Notes "mi valtozott es miert"
    (alap -Bump patch: minden frissites uj verzioszamot kap)

Lepesek:
  1/8 cel-verzio: VERSION + -Bump (patch|minor|major|none, alap: patch)
      vagy -Version x.y.z (explicit feluliras)
  2/8 TESZTKAPU: run_tests.ps1 child PowerShellben - FAIL -> abort.
      -StrictValidation: a tests\validation deep-SKIP-ek is FAIL-t okoznak
      (kiadasi minosegu buildhez; a riport: logs\validation_report.json)
  3/8 verzio-iras: VERSION + pyproject.toml [project] version +
      config/voicemem_config.yaml project.version (a tesztek ellenorzik a
      szinkront - a szinkron-szabaly egyetlen forrasa a VERSION)
  4/8 staging megengedesi-listaval: app, config, scripts, tests, web, vendor (a
      helyi web UI - v0.3.5: a ZIP a rendszer MINDEN elemet tartalmazza)
      + gyokeri dokumentumok + START.bat + MODELS.lock.json (M0.2 one-click:
      a kicsomagolt ZIP-bol a START.bat dupla kattintassal telepitheto).
      A futasi allapot (.venv, models, memory, data, logs, bin,
      releases, INSTALL_MANIFEST.json, .install_state.json, .env,
      __pycache__) NEM kerul a ZIP-be. (v0.5.0: a vendor\voicemem
      VEZERLT forras KERUL a ZIP-be - a telepito abbol telepit, l.
      VOICEMEM_PIN.json.)
      4a) M0 KONYVTARVAZ: a ZIP tartalmazza az M0 altal elvart ures
      konyvtarakat (models/asr|llm|tts|vad|embedding + models/hf,
      models/emotion + memory/sqlite|qdrant|backups) .gitkeep
      placeholder fajlokkal - a ZIP formatum nem tarol megbizhatosan
      ures konyvtarakat. A modellsulyok NEM kerulnek a ZIP-be.
      4b) CHANGELOG: a staged peldany megkapja a sajat verzioju
      bejegyzest, mielott a ZIP elkeszul - a kicsomagolt ZIP
      CHANGELOG.md-je igy mindig tartalmazza a sajat verziojat.
      4c) CRLF GARANCIA (v0.3.6): a staged *.bat fajlok Windows CRLF
      sorvegjeleket kapnak a ZIP-be kerules ELOTT - az LF-only .bat a
      cmd.exe parserenek ismert kockazata (field report: a START.bat
      elakadt a mod-menun).
  5/8 BUILD_INFO.json a ZIP gyokerebe (verzio, UTC-ido, git-commit,
      teszt- es validacio-osszefoglalo, notes)
  6/8 ZIP: releases\VoiceMemAgent_vX.Y.Z.zip (ZipFile::CreateFromDirectory)
  6b/8 ZIP SELF-CHECK: a kesz ZIP-et ki lehet nyitni es ellenorizni kell:
      az M0 konyvtarvaz minden konyvtara + a sajat verzioju CHANGELOG
      bejegyzes tenylegesen BENN van-e (release-blocker-regresszio-orto).
      Hiba eseten a hibas ZIP torlodik - hamis sikert soha nem irunk ki.
  7/8 SHA256: releases\VoiceMemAgent_vX.Y.Z.zip.sha256 (Get-FileHash)
  8/8 releases\RELEASE_INDEX.json + CHANGELOG.md (repo-peldany) +
      BUILD_HISTORY.json

Kapcsolok:
    -Bump patch|minor|major|none   alap: patch (minden frissites uj verzio)
    -Version x.y.z                 explicit verzio (felulirja a -Bump-ot)
    -Notes "szoveg"                CHANGELOG-bejegyzes leirasa
    -Force                         mar letezo verzio ujraepitese
    -StrictValidation              a deep validacios SKIP-ek is FAIL-t okoznak
    -Root <utvonal>                repo-gyoker (alap: a szkript mappajanak
                                   szuloje - a repo barmhol klonozhato)

Megjegyzes: a fajl szandekosan ASCII - PowerShell 5.1 BOM nelkul
Windows-1252-kent olvasna az ekezetes karaktereket. Elsosorban Windowsra
keszult, de dev-sandboxon (pwsh + robocopy nelkul) is futhat a Copy-Item
fallback-gal, hogy a kapu/build logika tesztelheto legyen.
#>

param(
    [string]$Bump = "patch",
    [string]$Version = "",
    [string]$Notes = "",
    [switch]$Force,
    [switch]$StrictValidation,
    [string]$Root = ""
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------- helpers ----

$Script:Failure = ""
$Script:StageRoot = ""
$Script:ZipSelfCheckOk = $false

function Write-BStep { param([string]$Msg) Write-Host "[BUILD] $Msg" -ForegroundColor Cyan }
function Write-BOk   { param([string]$Msg) Write-Host "[OK]    $Msg" -ForegroundColor Green }
function Write-BWarn { param([string]$Msg) Write-Host "[WARN]  $Msg" -ForegroundColor Yellow }
function Write-BErr  { param([string]$Msg) Write-Host "[ERROR] $Msg" -ForegroundColor Red }

function Test-SemVer([string]$V) { return ($V -match '^\d+\.\d+\.\d+$') }

function Get-Bumped([string]$V, [string]$Part) {
    $Bits = $V.Split('.')
    switch ($Part) {
        "major" { $Bits[0] = [string]([int]$Bits[0] + 1); $Bits[1] = '0'; $Bits[2] = '0' }
        "minor" { $Bits[1] = [string]([int]$Bits[1] + 1); $Bits[2] = '0' }
        "patch" { $Bits[2] = [string]([int]$Bits[2] + 1) }
        "none"  { }
        default { throw "Ervenytelen -Bump ertek: '$Part' (patch|minor|major|none)" }
    }
    return ($Bits -join '.')
}

function New-Utf8NoBom { return (New-Object System.Text.UTF8Encoding($false)) }

function Write-TextNoBom([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text, (New-Utf8NoBom))
}

function Add-ChangelogEntry([string]$Path, [string]$VersionStr, [string]$EntryText) {
    # Beszuri a kiadasi bejegyzest a CHANGELOG.md ele (a legeleje "## ["
    # elotti pozicioba). Idempotens: ha mar van a verziorol bejegyzes, nem
    # nyul hozza. Hasznalat: eloszor a STAGED peldanyon (a ZIP-ben legyen
    # benne a sajat verzioju bejegyzes), majd a sikeres build utan a
    # repo-peldanyon is.
    if (-not (Test-Path $Path)) { Fail "A CHANGELOG.md hianyzik: $Path" }
    $Ch = [IO.File]::ReadAllText($Path)
    if ($Ch -notmatch [regex]::Escape("## [$VersionStr]")) {
        $Pos = $Ch.IndexOf("## [")
        if ($Pos -ge 0) { $Ch = $Ch.Insert($Pos, $EntryText) }
        else { $Ch = $Ch.TrimEnd() + "`n`n" + $EntryText }
        [IO.File]::WriteAllText($Path, $Ch, (New-Utf8NoBom))
        return $true
    }
    return $false
}

function Fail([string]$Msg) {
    $Script:Failure = $Msg
    throw "BUILD-ABORT: $Msg"
}

function Remove-Stage {
    if ($Script:StageRoot -ne "" -and (Test-Path $Script:StageRoot)) {
        Remove-Item -Path $Script:StageRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ------------------------------------------------------------------- paths ---

if ($Root -ne "") {
    $RepoRoot = (Resolve-Path $Root).Path
}
else {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}

$VersionFile   = Join-Path $RepoRoot "VERSION"
$PyProjectFile = Join-Path $RepoRoot "pyproject.toml"
$YamlFile      = Join-Path $RepoRoot "config/voicemem_config.yaml"
$ChangelogFile = Join-Path $RepoRoot "CHANGELOG.md"
$RunTestsFile  = Join-Path $RepoRoot "scripts/run_tests.ps1"
$ReleasesDir   = Join-Path $RepoRoot "releases"
$IndexPath     = Join-Path $ReleasesDir "RELEASE_INDEX.json"
$HistoryPath   = Join-Path $ReleasesDir "BUILD_HISTORY.json"
$ReportPath    = Join-Path $RepoRoot "logs/validation_report.json"

function Add-BuildHistory([string]$Outcome, [string]$VersionStr, [hashtable]$Extra) {
    $Entry = @{
        outcome   = $Outcome
        version   = $VersionStr
        at_utc    = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        generator = "scripts/build_release.ps1"
    }
    foreach ($K in $Extra.Keys) { $Entry[$K] = $Extra[$K] }
    $Entries = @()
    if (Test-Path $HistoryPath) {
        try {
            $Raw = Get-Content $HistoryPath -Raw -Encoding UTF8
            if ($Raw -and $Raw.Trim()) {
                $Parsed = $Raw | ConvertFrom-Json
                $Entries = @($Parsed)
            }
        }
        catch { $Entries = @() }
    }
    $All = @($Entries) + @($Entry)
    $Parts = @()
    foreach ($E in $All) { $Parts += (ConvertTo-Json -InputObject $E -Depth 6) }
    $Text = "[`n    " + ($Parts -join ",`n    ") + "`n]`n"
    Write-TextNoBom $HistoryPath $Text
}

function Copy-Tree([string]$Src, [string]$Dst) {
    if (Get-Command robocopy -ErrorAction SilentlyContinue) {
        robocopy $Src $Dst /E /XD __pycache__ .pytest_cache /XF *.pyc *.pyo .env voice_settings.json *.egg-info | Out-Null
        if ($LASTEXITCODE -ge 8) { Fail "robocopy hiba: $Src (kod: $LASTEXITCODE)" }
    }
    else {
        Copy-Item -Path $Src -Destination $Dst -Recurse -Force
        Get-ChildItem -Path $Dst -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path $Dst -Recurse -Directory -Filter ".pytest_cache" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path $Dst -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
        $EnvFile = Join-Path $Dst ".env"
        if (Test-Path $EnvFile) { Remove-Item $EnvFile -Force }
        # v0.4.2: config/voice_settings.json RUNTIME ALLAPOT (perzisztalt
        # hangvalasztas) - soha nem szallhat ki release ZIP-ben.
        Get-ChildItem -Path $Dst -Recurse -File -Filter "voice_settings.json" -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------- main flow --

$CurrentVersion = ""
$NewVersion     = ""
$ZipName        = ""
$ZipPath        = ""
$ZipSize        = 0
$Hash           = ""
$GateExit       = 1
$DeepPassStr    = ""
$DeepSkipCount  = 0
$DeepSkipReasons = ""
$GitCommit      = ""
$BuiltAtUtc     = ""

try {
    # --------------------------------------------------------- 1/8 verzio ---
    Write-BStep "1/8 Cel-verzio meghatarozasa..."
    if (-not (Test-Path $VersionFile)) { Fail "A VERSION fajl hianyzik: $VersionFile" }
    if (-not (Test-Path $PyProjectFile)) { Fail "A pyproject.toml hianyzik." }
    if (-not (Test-Path $YamlFile)) { Fail "A config/voicemem_config.yaml hianyzik." }
    if (-not (Test-Path $ChangelogFile)) { Fail "A CHANGELOG.md hianyzik." }

    $CurrentVersion = (Get-Content $VersionFile -Raw).Trim()
    if (-not (Test-SemVer $CurrentVersion)) {
        Fail "A VERSION fajl tartalma ('$CurrentVersion') nem X.Y.Z alaku semver."
    }

    if ($Version -ne "") {
        if (-not (Test-SemVer $Version)) { Fail "A -Version ertek ('$Version') nem X.Y.Z alaku semver." }
        $NewVersion = $Version
    }
    else {
        $NewVersion = Get-Bumped $CurrentVersion $Bump
    }

    # verzio-visszafejlesztes tiltva
    $Cv = $CurrentVersion.Split('.'); $Nv = $NewVersion.Split('.')
    $Older = $false
    if ([int]$Nv[0] -lt [int]$Cv[0]) { $Older = $true }
    elseif ([int]$Nv[0] -eq [int]$Cv[0] -and [int]$Nv[1] -lt [int]$Cv[1]) { $Older = $true }
    elseif ([int]$Nv[0] -eq [int]$Cv[0] -and [int]$Nv[1] -eq [int]$Cv[1] -and [int]$Nv[2] -lt [int]$Cv[2]) { $Older = $true }
    if ($Older) {
        Fail "Verzio-visszafejlesztes tiltott: $CurrentVersion -> $NewVersion"
    }

    if (-not (Test-Path $ReleasesDir)) { New-Item -ItemType Directory -Path $ReleasesDir | Out-Null }
    $ZipName = "VoiceMemAgent_v$NewVersion.zip"
    $ZipPath = Join-Path $ReleasesDir $ZipName
    if ((Test-Path $ZipPath) -and (-not $Force)) {
        Fail "A releases\$ZipName mar letezik. Hasznalj nagyobb verziojat (-Bump/-Version) vagy -Force-ot."
    }
    Write-BOk "Verzio: $CurrentVersion -> $NewVersion (-Bump: $Bump)"

    # ------------------------------------------------------ 2/8 TESZTKAPU ---
    Write-BStep "2/8 TESZTKAPU: a teljes tesztkeszlet lefut a ZIP ELOTT..."
    if (-not (Test-Path $RunTestsFile)) { Fail "A scripts/run_tests.ps1 hianyzik." }

    $ChildPs = ""
    $SelfExe = ""
    try { $SelfExe = (Get-Process -Id $PID).Path } catch { }
    if ($SelfExe -and (Test-Path $SelfExe)) { $ChildPs = $SelfExe }
    elseif (Get-Command powershell.exe -ErrorAction SilentlyContinue) { $ChildPs = "powershell.exe" }
    elseif (Get-Command pwsh -ErrorAction SilentlyContinue) { $ChildPs = "pwsh" }
    else { Fail "Nem talalhato PowerShell a futtatashoz (powershell.exe / pwsh)." }

    $GateStartUtc = (Get-Date).ToUniversalTime()
    & $ChildPs -NoProfile -ExecutionPolicy Bypass -File $RunTestsFile
    $GateExit = $LASTEXITCODE
    if ($GateExit -ne 0) {
        Fail "TESZTKAPU FAIL (kilepesi kod $GateExit) - a ZIP NEM keszul el. Fixald a teszteket, majd futtasd ujra a buildet."
    }
    Write-BOk "Tesztkapu PASS (kilepesi kod 0)."

    # --- validacios riport kiolvasasa (a run_tests.ps1 irta) ---
    if (Test-Path $ReportPath) {
        try {
            $Report = Get-Content $ReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $PassList = @(); $SkipList = @()
            if ($Report.features) {
                foreach ($P in $Report.features.PSObject.Properties) {
                    if ($P.Value.status -eq "pass") { $PassList += $P.Name }
                    else { $SkipList += "$($P.Name): $($P.Value.reason)" }
                }
            }
            $DeepPassStr     = ($PassList -join ", ")
            $DeepSkipCount   = $SkipList.Count
            $DeepSkipReasons = ($SkipList -join "; ")
            try {
                $RepTime = [datetimeoffset]::Parse($Report.generated_at, [Globalization.CultureInfo]::InvariantCulture)
                if ($RepTime.UtcDateTime -lt $GateStartUtc.AddMinutes(-2)) {
                    Write-BWarn "A validation_report.json regi (generated_at: $($Report.generated_at)) - ellenorizd, hogy a mostani kapu irta."
                }
            }
            catch { Write-BWarn "A validation_report.generated_at nem ertelmezheto." }
        }
        catch { Write-BWarn "A logs/validation_report.json nem olvashato: $($_.Exception.Message)" }
    }
    else {
        Write-BWarn "Nem keletkezett logs/validation_report.json (a run_tests.ps1-nek kellett volna irnia)."
    }

    if ($StrictValidation -and $DeepSkipCount -gt 0) {
        Fail "STRICT validacio: $DeepSkipCount deep-ellenorzes SKIP-pelt. Okok: $DeepSkipReasons"
    }
    Write-BOk "Deep validacio: PASS: $DeepPassStr (SKIP: $DeepSkipCount db)."

    # --------------------------- changelog-bejegyzes szovege (a staging elott) --
    # A bejegyzes szovege itt all ossze (kapu utan - minden adat elerheto),
    # es a 4/8 lepesben a STAGED CHANGELOG-ba kerul, mielott a ZIP
    # elkeszulne: a kicsomagolt ZIP CHANGELOG.md-je igy mindig tartalmazza
    # a sajat verzioja bejegyzeset (a repo-peldany a 8/8 lepesben kapja
    # meg a sikeres build utan).
    $BuiltAtUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $Today = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
    $NoteLine = $Notes
    if ([string]::IsNullOrWhiteSpace($NoteLine)) { $NoteLine = "(nincs -Notes leiras)" }
    $ChEntry = "## [$NewVersion] - $Today`n"
    $ChEntry += "### Kiadas (ZIP-build)`n"
    $ChEntry += "- $NoteLine`n"
    $ChEntry += "- Tesztkapu: PASS a build ELOTT (teljes tesztkeszlet; deep validacio: $DeepPassStr; SKIP: $DeepSkipCount db).`n"
    $ChEntry += "- Kiadas: ``releases/$ZipName`` + ``$ZipName.sha256`` (UTC: $BuiltAtUtc).`n`n"

    # ------------------------------------------------- 3/8 verzio-iras -----
    Write-BStep "3/8 Verzio-iras: VERSION + pyproject.toml + config YAML..."
    [IO.File]::WriteAllText($VersionFile, "$NewVersion`n", (New-Utf8NoBom))

    $PP = [IO.File]::ReadAllText($PyProjectFile)
    $PPNew = [regex]::Replace($PP, '(?m)^version\s*=\s*"[^"]*"', ('version = "' + $NewVersion + '"'), 1)
    if ($PPNew -eq $PP) { Fail "A pyproject.toml version mezoje nem talalhato." }
    [IO.File]::WriteAllText($PyProjectFile, $PPNew, (New-Utf8NoBom))

    $YL = [IO.File]::ReadAllText($YamlFile)
    $YLNew = [regex]::Replace($YL, '(?m)^(\s*version:\s*")[^"]+(")', ('${1}' + $NewVersion + '${2}'), 1)
    if ($YLNew -eq $YL) { Fail "A config/voicemem_config.yaml project.version mezoje nem talalhato." }
    [IO.File]::WriteAllText($YamlFile, $YLNew, (New-Utf8NoBom))
    Write-BOk "Verzio-fajlok frissitve (szinkron-szabaly)."

    # ------------------------------------------------------ 4/8 staging ----
    Write-BStep "4/8 Staging (megengedesi-lista masolas)..."
    $Stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
    $Script:StageRoot = Join-Path ([IO.Path]::GetTempPath()) "voicemem_build_${NewVersion}_$Stamp"
    $StageRepo = Join-Path $Script:StageRoot "VoiceMemAgent_v$NewVersion"
    New-Item -ItemType Directory -Path $StageRepo -Force | Out-Null

    # v0.5.0: vendor = a VEZERLT VoiceMem forras - a ZIP-ben utazik
    foreach ($D in @("app", "config", "scripts", "tests", "web", "vendor")) {
        $Src = Join-Path $RepoRoot $D
        if (-not (Test-Path $Src)) { Fail "A staging forras hianyzik: $D" }
        Copy-Tree $Src (Join-Path $StageRepo $D)
    }
    foreach ($F in @("README.md", "CHANGELOG.md", "LICENSES.md", "CONTRACT.md", "VERSION",
                     "INSTALL_MANIFEST.example.json", "requirements.txt", "requirements.lock",
                     "pyproject.toml", ".gitignore", "START.bat", "MODELS.lock.json",
                     "VOICEMEM_PIN.json", "UPSTREAM_POLICY.md")) {
        $Src = Join-Path $RepoRoot $F
        if (Test-Path $Src) {
            Copy-Item -Path $Src -Destination (Join-Path $StageRepo $F) -Force
        }
        else {
            Write-BWarn "Staging: a(z) $F nem letezik a repoban - kihagyva."
        }
    }

    # --- 4c) CRLF GARANCIA (v0.3.6 field report): a Windows .bat fajlok
    # CRLF sorvegjelekkel szallnak ki a ZIP-ben. A repo/dev checkout
    # LF sorvegjelekkel is tartalmazhatja oket, ami a cmd.exe
    # parszolasanak ismert kockazata. Minden staged .bat fajlt
    # normalizalunk CRLF-re a ZIP elkeszulese ELOTT (ASCII-tisztan).
    $BatFiles = @(Get-ChildItem -Path $StageRepo -Recurse -Filter "*.bat" -File)
    foreach ($BatFile in $BatFiles) {
        $BatText = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($BatFile.FullName))
        $BatNorm = $BatText.Replace("`r`n", "`n").Replace("`r", "`n").Replace("`n", "`r`n")
        [IO.File]::WriteAllBytes($BatFile.FullName, [Text.Encoding]::ASCII.GetBytes($BatNorm))
    }
    Write-BOk ("CRLF garancia: {0} .bat fajl normalizalva." -f $BatFiles.Count)

    # --- 4a) M0 KONYVTARVAZ (release blocker fix): a ZIP-nek tartalmaznia
    # kell az M0 altal elvart ures konyvtarakat. A ZIP formatum nem tarol
    # megbizhatosan ures konyvtarakat, ezert minden konyvtar .gitkeep
    # placeholder-t kap (+ a repo README.md-jet, ha van). A modellsulyok
    # SOHA nem kerulnek a ZIP-be - azokat a telepito tolti le.
    $PlaceholderDirs = @(
        "models/asr/qwen3-asr-0.6b",
        "models/llm/qwen3.6-35b-a3b",
        "models/tts/piper",
        "models/vad/silero-vad",
        "models/embedding/multilingual-e5-small",
        "models/hf",
        "models/emotion",
        "models/speaker",
        "memory/sqlite",
        "memory/qdrant",
        "memory/backups"
    )
    foreach ($Dir in $PlaceholderDirs) {
        $Target = Join-Path $StageRepo $Dir
        if (-not (Test-Path $Target)) {
            New-Item -ItemType Directory -Path $Target -Force | Out-Null
        }
        $KeepFile = Join-Path $Target ".gitkeep"
        if (-not (Test-Path $KeepFile)) { Write-TextNoBom $KeepFile "" }
        $RepoDir = Join-Path $RepoRoot $Dir
        if (Test-Path (Join-Path $RepoDir "README.md")) {
            Copy-Item -Path (Join-Path $RepoDir "README.md") -Destination (Join-Path $Target "README.md") -Force
        }
    }
    Write-BOk ("M0 konyvtarvaz: {0} placeholder konyvtar a ZIP-ben." -f $PlaceholderDirs.Count)

    # --- 4b) STAGED CHANGELOG (release blocker fix): a ZIP-ben levo
    # CHANGELOG.md megkapja a sajat verzioju bejegyzest a ZIP elkeszulese
    # ELOTT (a 6b/8 self-check kesobb tenylegesen visszaidzeli).
    $StagedChangelog = Join-Path $StageRepo "CHANGELOG.md"
    if (-not (Test-Path $StagedChangelog)) { Fail "A staged CHANGELOG.md hianyzik." }
    [void](Add-ChangelogEntry $StagedChangelog $NewVersion $ChEntry)

    $StagedFiles = (Get-ChildItem -Path $StageRepo -Recurse -File | Measure-Object).Count
    Write-BOk "Staging kesz: $StagedFiles fajl."

    # ------------------------------------------------- 5/8 BUILD_INFO ------
    Write-BStep "5/8 BUILD_INFO.json osszeallitasa..."
    $GitCommit = ""
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $PrevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $Out = & git -C $RepoRoot rev-parse HEAD
            if ($LASTEXITCODE -eq 0 -and $Out) { $GitCommit = [string]$Out }
        }
        catch { $GitCommit = "" }
        $ErrorActionPreference = $PrevEap
    }
    $BuildInfo = @{
        schema_version          = 1
        name                    = "voicemem-agent"
        version                 = $NewVersion
        built_from_version      = $CurrentVersion
        milestone               = "M0"
        built_at_utc            = $BuiltAtUtc
        git_commit              = $GitCommit
        generator               = "scripts/build_release.ps1"
        test_gate               = "run_tests.ps1 (unit + integration + benchmark + validation)"
        test_gate_exit_code     = $GateExit
        validation_deep_pass    = $DeepPassStr
        validation_deep_skipped = $DeepSkipCount
        validation_strict       = [bool]$StrictValidation
        notes                   = $Notes
    }
    $BiJson = ConvertTo-Json -InputObject $BuildInfo -Depth 4
    Write-TextNoBom (Join-Path $StageRepo "BUILD_INFO.json") $BiJson

    # ---------------------------------------------------- 6/8 + 7/8 ZIP ----
    Write-BStep "6/8 ZIP keszites: releases\$ZipName"
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $StageRepo, $ZipPath, [System.IO.Compression.CompressionLevel]::Optimal, $true)
    if (-not (Test-Path $ZipPath)) { Fail "A ZIP nem jott letre: $ZipPath" }
    $ZipSize = (Get-Item $ZipPath).Length

    # ---------------------------------------------- 6b/8 ZIP SELF-CHECK ------
    # Release-blocker-regresszio-orto: a kesz ZIP-et ki kell nyitni es
    # ellenorizni kell, hogy az M0 konyvtarvaz + a sajat verzioju CHANGELOG
    # bejegyzes tenylegesen BENN van-e. Hiba eseten a hibas ZIP torlodik
    # (a catch agyban), es NEM szuletik kiadas - hamis sikert soha nem
    # irunk ki.
    Write-BStep "6b/8 ZIP self-check (M0 konyvtarvaz + sajat CHANGELOG bejegyzes)..."
    $RootPrefix = "VoiceMemAgent_v$NewVersion/"
    $ZipRead = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $EntryNames = @()
        foreach ($Ze in $ZipRead.Entries) { $EntryNames += ($Ze.FullName.Replace('\', '/')) }
        foreach ($Dir in $PlaceholderDirs) {
            $Found = $false
            foreach ($N in $EntryNames) {
                if ($N.StartsWith("$RootPrefix$Dir/")) { $Found = $true; break }
            }
            if (-not $Found) { Fail "ZIP self-check: a(z) '$Dir/' konyvtar hianyzik a ZIP-bol." }
        }
        $ChEntryZip = $ZipRead.GetEntry($RootPrefix + "CHANGELOG.md")
        if (-not $ChEntryZip) { Fail "ZIP self-check: a CHANGELOG.md hianyzik a ZIP-bol." }
        $Reader = New-Object System.IO.StreamReader($ChEntryZip.Open(), (New-Object System.Text.UTF8Encoding($false)))
        try { $ZipChText = $Reader.ReadToEnd() } finally { $Reader.Dispose() }
        if ($ZipChText -notmatch [regex]::Escape("## [$NewVersion]")) {
            Fail "ZIP self-check: a ZIP-ben levo CHANGELOG.md nem tartalmazza a(z) $NewVersion bejegyzest."
        }
    }
    finally {
        $ZipRead.Dispose()
    }
    $Script:ZipSelfCheckOk = $true
    Write-BOk "ZIP self-check OK ($($PlaceholderDirs.Count) M0 konyvtar + CHANGELOG bejegyzes)."

    Write-BStep "7/8 SHA256 checksum..."
    $Hash = (Get-FileHash -Path $ZipPath -Algorithm SHA256).Hash
    Write-TextNoBom "$ZipPath.sha256" "$Hash  $ZipName`n"

    # --------------------------------------- 8/8 index + changelog + history
    Write-BStep "8/8 RELEASE_INDEX + CHANGELOG + BUILD_HISTORY frissitese..."

    $Entries = @()
    if (Test-Path $IndexPath) {
        try {
            $Raw = Get-Content $IndexPath -Raw -Encoding UTF8
            if ($Raw -and $Raw.Trim()) {
                $Idx = $Raw | ConvertFrom-Json
                if ($Idx.releases) { $Entries = @($Idx.releases) }
            }
        }
        catch { Write-BWarn "A RELEASE_INDEX.json serult - ujrairodik."; $Entries = @() }
    }
    $AlreadyIndexed = $false
    foreach ($E in $Entries) { if ($E.version -eq $NewVersion) { $AlreadyIndexed = $true } }
    if (-not $AlreadyIndexed) {
        $NewEntry = @{
            version                 = $NewVersion
            built_at_utc            = $BuiltAtUtc
            zip                     = $ZipName
            sha256                  = $Hash
            size_bytes              = $ZipSize
            notes                   = $Notes
            git_commit              = $GitCommit
            test_gate               = "run_tests.ps1"
            test_gate_exit_code     = $GateExit
            validation_deep_pass    = $DeepPassStr
            validation_deep_skipped = $DeepSkipCount
            validation_strict       = [bool]$StrictValidation
        }
        $Entries = @($Entries) + @($NewEntry)
    }
    $Parts = @()
    foreach ($E in $Entries) { $Parts += (ConvertTo-Json -InputObject $E -Depth 4) }
    $IndexText = "{`n  `"schema_version`": 1,`n  `"releases`": [`n    " + ($Parts -join ",`n    ") + "`n  ]`n}`n"
    Write-TextNoBom $IndexPath $IndexText

    # --- CHANGELOG.md: a repo-peldany is megkapja a bejegyzest (a staged,
    # ZIP-ben levo masolat mar a 4/8 lepesben kapta meg, a ZIP elkeszulese
    # ELOTT - igy a ZIP mindig a sajat bejegyzesevel egyutt csomagolodik).
    [void](Add-ChangelogEntry $ChangelogFile $NewVersion $ChEntry)

    Add-BuildHistory "success" $NewVersion @{
        zip = $ZipName; sha256 = $Hash; test_gate_exit_code = $GateExit
        deep_validation = "pass: $DeepPassStr; skip: $DeepSkipCount"
    }

    Remove-Stage
    Write-Host ""
    Write-BOk "BUILD SUCCESS - VoiceMemAgent $NewVersion"
    Write-Host "  Verzio     : $CurrentVersion -> $NewVersion"
    Write-Host "  ZIP        : releases/$ZipName ($([math]::Round($ZipSize / 1KB, 1)) KB)"
    Write-Host "  SHA256     : $Hash"
    Write-Host "  Deep val.  : PASS: $DeepPassStr"
    Write-Host "  Deep SKIP  : $DeepSkipCount db"
    Write-Host "  Changelog  : CHANGELOG.md ($NewVersion bejegyzes)"
    Write-Host "  Index      : releases/RELEASE_INDEX.json"
    Write-Host ""
}
catch {
    $Msg = $_.Exception.Message
    if (-not ($Msg -like "BUILD-ABORT*")) {
        $Script:Failure = $Msg
        Write-BErr "Varatlan hiba: $Msg"
    }
    # Ha a ZIP mar kesz lett, de a self-check (vagy barmi mas a sikeres
    # indexeles elott) elbukott, a hibas ZIP nem maradhat a releases
    # mappaban - hamis sikert soha nem irunk ki.
    if (($Script:ZipSelfCheckOk -ne $true) -and $ZipPath -and (Test-Path $ZipPath)) {
        Remove-Item -Path $ZipPath -Force -ErrorAction SilentlyContinue
        $ShaPath = "$ZipPath.sha256"
        if (Test-Path $ShaPath) { Remove-Item -Path $ShaPath -Force -ErrorAction SilentlyContinue }
        Write-BErr "A hibas ZIP torolve lett: $ZipName"
    }
    $HistVersion = $NewVersion
    if ([string]::IsNullOrWhiteSpace($HistVersion)) { $HistVersion = "(unresolved)" }
    try {
        Add-BuildHistory "failed" $HistVersion @{
            reason = $Script:Failure; test_gate_exit_code = $GateExit
        }
    }
    catch { }
    Remove-Stage
    Write-BErr "BUILD FAILED - NEM kesult ZIP. Ok: $Script:Failure"
    exit 1
}
finally {
    Remove-Stage
}

exit 0
