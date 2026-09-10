<#
===============================================================================
scripts/download_models.ps1 - M1+M2 modellkeszlet letoltese (ONLINE fazis)

ARCHITEKTURA (v0.1.7): a letoltest a huggingface_hub PYTHON API vegzi
(hf_hub_download / snapshot_download), a szkript egy VEKONY PowerShell
wrapper. A regi huggingface-cli.exe alapu megoldas ELTAVOLITVA, mert:

  ROOT CAUSE (v0.1.6 release blocker, 13/18 Modellletoltes lepes):
  a "huggingface-cli download" parancs deprecation warningja EMOJI
  karaktert nyom ki; Windows PowerShell CP1252 stdout mellett a gyermek
  Python processz UnicodeEncodeError-rel OSSZEOMLIK meg a tenyleges
  letoltes elott, ezert egyetlen modell sem erkezett meg.

  A javitas szabalyai:
    - NINCS CLI subprocess, NINCS CLI output parsing, NINCS PATH fuggoseg.
    - A Hugging Face CLI NEM runtime dependency (sehol nem kell telepiteni).
    - A Python interpreter mindig a projekt venv-e:
      .venv\Scripts\python.exe (a valasztas soha nem PATH-ra bizatik).
    - A letoltet a scripts\download_models_hf.py vegzi a MODELS.lock.json
      altal definialt repo/fajl adatokkal, KOZVETLENUL a projekt models\
      konyvtaraiba: retry (3 probalkozas, 5s/15s backoff), resume
      (megszakadt atvitel folytatasa), revision tamogatas
      (pinned_revision), tukor-lanc (Qwen -> unsloth -> bartowski;
      tphakala -> silero; piper nested -> flat) es kotelezo
      fajl-ellenorzes (min_bytes / min_total_mb floorok).

MIT csinal a wrapper (sorrendben):
  1. Internet-ellenorzes (a szkript csak online futhat).
  2. .venv\Scripts\python.exe ellenorzese (a .venv a telepito dolga - ha
     hianyzik, a javitas a START.bat ujrainditasa, a bootstrap magatol
     megcsinalja).
  3. ONE-CLICK SZABALY: ha a huggingface_hub KONYVTAR nem importalhato a
     venv-ben, AUTOMATIKUSAN telepitjuk (pip install "huggingface_hub" -
     a KONYVTARAT, NEM a [cli] extra-t). Soha nem adunk "telepitsd kezzel"
     utasitast a felhasznalonak.
  4. PYTHONIOENCODING=utf-8 + PYTHONUTF8=1 beallitasa a gyermek Pythonnak
     (biztonsagi or: a gyermek soha ne haljon el kodolasi hibara, akkor
     sem, ha egy konyvtar nem-ASCII karaktert nyomne ki).
  5. .venv\Scripts\python.exe -u scripts\download_models_hf.py --root ...
     futtatasa; a kilepesi kod atadasa (0 = minden kotelezo fajl a helyen
     ellenorzive; nem-0 = hiba - a letoltes NEM sikeres).

IDEMPOTENS: nyugodtan ujrafuttathato - a mar letoltott, meretileg
ellenorzott fajlokat a Python oldal atugorja (nulla halozati forgalom),
a megszakadt letolteseket folytatja (resume), a tukroket automatikusan
probalja. A vegen a szkript CSAK akkor jelez sikert, ha a MODELS.lock.json
altal kotelezett fajlok tenylegesen leteznek es a meretuk rendben.

FONTOS szabalok:
  1. Ez a szkript CSAK ONLINE futhat! Ne kapcsold ki a netet futas kozben.
  2. NE allitsd be elotte a HF_HUB_OFFLINE=1-et: az offline mod a
     letoltest BLOKKOLJA. (A telepito a letoltes UTAN allitja az offline
     env-eket.)
  3. A modellletoltes soha nem hagyatkozik a PATH-ban levo CLI-re.

A BINARISOK (llama-server.exe, piper.exe) letoltese mar a
scripts/install_m1.ps1 11. lepesenek dolga - EZ a szkript NEM tolti le oket
(ne duplikaljuk).

Ellenorzes a vegen (a teljes telepites ellenorzese):
  .\scripts\verify_m1.ps1         (vagy: scripts\install_m1.ps1 - ellenorzi)

Hasznalat:
    .\scripts\download_models.ps1                          # Root = repo-gyoker
    .\scripts\download_models.ps1 -Root "C:\masik_root"

Megjegyzes: a fajl szandekosan ASCII - a PowerShell 5.1 BOM nelkul
Windows-1252-kent olvasna az ekezetes karaktereket.

#>
param(
    [string]$Root = ""
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

# ---------------------------------------------------------------------------
# 1) Internet-ellenorzes (a szkript csak online futhat)
# ---------------------------------------------------------------------------
try {
    Invoke-WebRequest -Uri "https://huggingface.co" -Method Head -TimeoutSec 10 -UseBasicParsing | Out-Null
    Write-Host "Internet-ellenorzes OK (https://huggingface.co elerheto)."
} catch {
    Write-Host "HIBA: a https://huggingface.co nem erheto el 10 s-en belul (nincs internet / proxy / tuzfal)." -ForegroundColor Red
    Write-Host "Ez a szkript CSAK ONLINE futhat - allitsd vissza a halozatot, majd futtasd ujra." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# 2) A projekt venv Pythonjanak feloldasa (SOHA nem a PATH-rol)
# ---------------------------------------------------------------------------
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "HIBA: nem talalhato a projekt .venv Python: $VenvPython" -ForegroundColor Red
    Write-Host "JAVITAS: inditsd ujra a START.bat-ot - a bootstrap automatikusan letrehozza es javitja a kort." -ForegroundColor Yellow
    exit 1
}

# ---------------------------------------------------------------------------
# 3) huggingface_hub KONYVTAR ellenorzese (.venv elsodleges)
#
#    ONE-CLICK SZABALY: a huggingface_hub hianya SOHA nem lehet user-facing
#    hiba, ha a fuggoseg automatikusan telepitheto. HA az import nem megy,
#    itt azonnal AUTOMATIKUSAN telepitjuk a KONYVTARAT a .venv pythonjaval
#    (idempotens: ha mar telepitve van, a pip azonnal kilep).
#    FIGYELEM: a [cli] extra NINCS telepitve - a Hugging Face CLI mar NEM
#    runtime dependency (a letoltes a Python API-val tortenik).
# ---------------------------------------------------------------------------
$HfHelper = Join-Path $Root "scripts\download_models_hf.py"
if (-not (Test-Path $HfHelper)) {
    Write-Host "HIBA: nem talalhato a scripts\download_models_hf.py (repo-fajl)." -ForegroundColor Red
    Write-Host "JAVITAS: inditsd ujra a START.bat-ot (idempotens); ha tovabb is hianyzik, csomagold ki ujra a ZIP-et." -ForegroundColor Yellow
    exit 1
}
& $VenvPython -c "import huggingface_hub" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "A huggingface_hub konyvtar nem importalhato a .venv-ben - AUTOMATIKUS telepites (huggingface_hub konyvtar, CLI nelkul)..."
    & $VenvPython -m pip install "huggingface_hub"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "HIBA: a huggingface_hub automatikus telepitese nem sikerult (halozati problema?)." -ForegroundColor Red
        Write-Host "JAVITAS: ellenorizd az internetkapcsolatot, majd inditsd ujra a START.bat-ot (idempotens - a kesz reszeket atugorja)." -ForegroundColor Yellow
        exit 1
    }
    & $VenvPython -c "import huggingface_hub" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "HIBA: a telepites utan sem importalhato a huggingface_hub a .venv-ben." -ForegroundColor Red
        Write-Host "JAVITAS: inditsd ujra a START.bat-ot - a bootstrap ujrafuttatja a telepitot (idempotens)." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "huggingface_hub konyvtar automatikusan telepitve."
} else {
    Write-Host "huggingface_hub konyvtar: rendben (a .venv-ben importalhato)."
}

# ---------------------------------------------------------------------------
# 4) Kodolasi biztonsagi or a gyermek Pythonnak
#    (a gyermek soha ne omoljon el kodolasi hibara; a sajat kimenete amugy
#    is tiszta ASCII, ahol a CP437/CP1252 konzol is biztonsagos)
# ---------------------------------------------------------------------------
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# ---------------------------------------------------------------------------
# 5) A letoltes futtatasa: huggingface_hub Python API a venv interpreterrel
#    (-u: unbuffered, hogy az allapot-sorok azonnal latszodjanak)
#    A kilepesi kod a MODELS.lock.json alapjan torteno fajl-ellenorzes
#    eredmenye: 0 = minden kotelezo fajl tenylegesen a helyen.
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Modellletoltes inditasa: huggingface_hub Python API (hf_hub_download / snapshot_download)..."
& $VenvPython -u $HfHelper --root $Root
$HfExit = $LASTEXITCODE

# ---------------------------------------------------------------------------
# Befejezes: a kilepesi kod atadasa - hamis siker soha
# ---------------------------------------------------------------------------
if ($HfExit -ne 0) {
    Write-Host ""
    Write-Host "FIGYELEM: a modellletoltes HIBAVAL ert veget (kilepesi kod $HfExit)." -ForegroundColor Red
    Write-Host "Futtasd ujra a START.bat-ot (idempotens - a kesz fajlokat"
    Write-Host "atugorja, a megszakadt letolteseket folytatja, a tukroket"
    Write-Host "automatikusan probalja)."
    exit 1
}
Write-Host ""
Write-Host "Minden modell-letoltes es fajl-ellenorzes rendben."
Write-Host ""
Write-Host "MEGJEGYZES: a binarisok (llama-server.exe, piper.exe) letoltese a"
Write-Host "scripts\install_m1.ps1 11. lepesenek dolga - ez a szkript nem tolti le oket."
Write-Host "Ellenorzes: .\scripts\verify_m1.ps1"
exit 0
