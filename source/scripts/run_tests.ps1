<#
===============================================================================
scripts/run_tests.ps1 - M0/M1 tesztfuttato (a ZIP-build KOTELEZO kapuja)

MIT csinal: a repo gyokerere valt, elozetes bajtkod-ellenorzest futtat
(python -m compileall -q app - syntax-hiba gyors jelzese), majd a teljes
unittest-keszletet:
    python -m unittest discover -s tests -t .
A -t . kapcsolo a repo-gyokeret teszi top-level konyvtarnek, igy a
tests.unit.* / tests.integration.* / tests.benchmark.* /
tests.validation.* csomag-modulnevek mukodnek.
A vegen osszefogzalo, es a unittest kilepesi kodjaval kilep
(0 = minden teszt zold, nem 0 = volt hiba).

PYTHON-VALASZTAS (M0-elv): a repo .venv-e az elsodleges -
.venv\Scripts\python.exe (Windows) vagy .venv/bin/python (POSIX); ha nincs
.venv (pl. fejlesztoi sandbox vagy friss klon), a PATH-rol esik vissza
python -> python3. A ZIP-build (scripts/build_release.ps1) EZEZT a
szkriptet hivja KOTELEZO kapukent: ha egy teszt is elbukik, nem keszul ZIP.

VALIDACIOS RIPORT: a futtas alatt a VMA_VALIDATION_REPORT kornyezeti
valtozo a logs\validation_report.json-re allitodik - a
tests\validation\* tesztek a deep-ellenorzesek eredmenyet (PASS/SKIP +
ok) ide irjak, a build_release.ps1 pedig kiolvassa (a -StrictValidation
kapcsalo a SKIP-eket FAIL-le teszi). A valtozot a futtas vegen
visszaallitjuk (vagy toroljuk).

Futtathato:
  - a fejlesztoi sandboxban (Linux, GPU/hang nelkul) - a tesztek nehez
    fuggosegeket NEM hasznalnak, a deep tesztek ok-riporttal SKIP-pelnek;
  - a celgepen (Windows 11 + RTX 5070) - ott a deep tesztek a valodi
    komponenseket is ervenyesitik (modellek, GPU, llama-server health).

Hasznalat:
    .\scripts\run_tests.ps1
    (vagy kezzel: python -m unittest discover -s tests -t . -v)

Megjegyzes: a fajl szandekosan ASCII - PowerShell 5.1 BOM nelkul
Windows-1252-kent olvasna az ekezetes karaktereket.
#>

$RepoRoot = Split-Path -Parent $PSScriptRoot

# --- Python valasztas: .venv eloszor (M0-elv), utana PATH --------------------
$VenvWin = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$VenvPosix = Join-Path $RepoRoot ".venv\bin\python"
$Python = ""
if (Test-Path $VenvWin) {
    $Python = $VenvWin
}
elseif (Test-Path $VenvPosix) {
    $Python = $VenvPosix
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Python = "python"
}
elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $Python = "python3"
}
else {
    Write-Host "HIBA: nincs elerheto Python (.venv, python, python3)." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================================="
Write-Host "VoiceMem M0/M1 tesztfuttatas"
Write-Host "================================================================="
Write-Host "Repo gyoker : $RepoRoot"
Write-Host "Python      : $Python"
Write-Host "Tesztszerkezet: tests/unit + integration + benchmark + validation"

# --- Validacios riport: a deep-ellenorzesek eredmenyeide ir (build kapu) -----
$ReportPath = Join-Path $RepoRoot "logs\validation_report.json"
$PrevReport = $env:VMA_VALIDATION_REPORT
$env:VMA_VALIDATION_REPORT = $ReportPath

$ExitCode = 1
Push-Location $RepoRoot
try {
    Write-Host ""
    Write-Host "1/2 lepes: bajtkod-ellenorzes (python -m compileall -q app)..."
    & $Python -m compileall -q app
    if ($LASTEXITCODE -ne 0) {
        Write-Host "HIBA: az app/ modulok nem forditasithatok le (syntax hiba?)." -ForegroundColor Red
        $ExitCode = 1
    }
    else {
        Write-Host "OK - minden app/*.py forditasithato."
        Write-Host ""
        Write-Host "2/2 lepes: unittest futtatasa (discover -s tests -t . -v)..."
        & $Python -m unittest discover -s tests -t . -v
        $ExitCode = $LASTEXITCODE
    }
}
finally {
    Pop-Location
    if ($null -eq $PrevReport) {
        Remove-Item Env:\VMA_VALIDATION_REPORT -ErrorAction SilentlyContinue
    }
    else {
        $env:VMA_VALIDATION_REPORT = $PrevReport
    }
}

Write-Host ""
Write-Host "================================================================="
if ($ExitCode -eq 0) {
    Write-Host "OSSZEFOGLALO: minden teszt OK (kilepesi kod 0)." -ForegroundColor Green
    Write-Host "Validacios riport: $ReportPath" -ForegroundColor DarkGray
}
else {
    Write-Host "OSSZEFOGLALO: HIBA van (kilepesi kod $ExitCode) - lasd fent a reszleteket." -ForegroundColor Red
}
Write-Host "================================================================="
exit $ExitCode
