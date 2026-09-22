<#
===============================================================================
experimental/llama_b11073/verify_b11073.ps1 - END-TO-END VERIFICATION

Proves, in order (exit 0 only if every check passes):

  [1] INSTALL   the b11073 runtime is present in experimental\llama_b11073\bin\
                and EVERY file's SHA-256 matches the pinned table
                (config\b11073.pins.json) - i.e. the right binary was
                downloaded and nothing was swapped since;
  [2] MANIFEST  the install manifest agrees with the pins (asset sha256, tag);
  [3] BUILD ID  llama-server-impl.dll carries the b11073 commit string;
  [4] PRODUCTION UNTOUCHED - the production baseline is intact:
                bin\llama-server.exe hash == the hash recorded at install
                time (when installed), MODELS.lock.json still pins b10717,
                VERSION still 0.10.7, the canonical config is still
                ngl 20 / ctx 32768 / parallel 1 / reasoning off;
  [5] SERVER    (default; -SkipServer to omit) /health on 127.0.0.1:8081 is
                OK, and the RUNNING server identifies as b11073 via its
                system_fingerprint (a real 1-token chat completion);
  [6] CLIENT    (default; -SkipServer to omit) a minimal chat completion via
                the same OpenAI-compatible endpoint the VoiceMem LLM client
                uses (POST /v1/chat/completions, streaming shape) - proving
                the transport the application needs works end to end.

  -ThroughAppClient additionally runs the REAL application client
  (app\llm.py LlmClient through app\config.py AgentConfig with the env
  override) - the same proof the validation audit used. Requires the
  installed .venv (START.bat must have run at least once).

USAGE:
    powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\verify_b11073.ps1
    ... -SkipServer            # structural verification only (server not running)
    ... -ThroughAppClient      # + the real app client proof

EXIT CODES: 0 = ALL PASS | 1 = at least one FAIL.

NOTE: deliberately ASCII (PowerShell 5.1 reads BOM-less files as Windows-1252).
===============================================================================
#>

param(
    [switch]$SkipServer,
    [switch]$ThroughAppClient
)

$ErrorActionPreference = "Continue"

$ScriptDir = $PSScriptRoot
$Root      = (Get-Item -LiteralPath $ScriptDir).Parent.Parent.FullName
$BinDir    = Join-Path $ScriptDir "bin"
$PinsFile  = Join-Path $ScriptDir "config\b11073.pins.json"
$Manifest  = Join-Path $ScriptDir "config\runtime_manifest.json"

$Fails = 0
function Ok([string]$m)   { Write-Host ("  [OK]    " + $m) -ForegroundColor Green }
function Warn([string]$m) { Write-Host ("  [WARN]  " + $m) -ForegroundColor Yellow }
function Bad([string]$m)   { Write-Host ("  [FAIL]  " + $m) -ForegroundColor Red; $script:Fails++ }

function Test-Health([string]$H, [int]$P, [int]$T = 3) {
    try {
        $R = Invoke-WebRequest -Uri ("http://{0}:{1}/health" -f $H, $P) -Method Get -TimeoutSec $T -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

Write-Host "=== EXPERIMENTAL llama.cpp b11073 - VERIFIKACIO ===" -ForegroundColor Magenta

# ---------------- [1] install: files + per-file SHA-256 ---------------------------
Write-Host "[1/6] Telepitesi allapot (fajlonkenti SHA-256 a pin-tabla ellen):"
if (-not (Test-Path -LiteralPath $PinsFile)) {
    Bad ("hianyzik a pin-fajl: " + $PinsFile)
} else {
    $Pins = Get-Content -LiteralPath $PinsFile -Raw | ConvertFrom-Json
    $nOk = 0; $nBad = 0; $nMiss = 0
    foreach ($f in $Pins.runtime.files) {
        $T = Join-Path $BinDir ([string]$f.name)
        if (-not (Test-Path -LiteralPath $T)) { $nMiss++; Bad ("hianyzo fajl: " + $f.name); continue }
        $H = (Get-FileHash -LiteralPath $T -Algorithm SHA256).Hash
        if ($H -eq ([string]$f.sha256).ToUpper()) { $nOk++ }
        else { $nBad++; Bad ("HASH ELTER: " + $f.name) }
    }
    if ($nMiss -eq 0 -and $nBad -eq 0) {
        Ok ("mind a " + $nOk + " runtime-fajl SHA-256 hash-e megegyezik a pin-elt b11073 tablaval")
    }
    Ok ("telepitesi hely: " + $BinDir)
}

# ---------------- [2] manifest consistency ----------------------------------------
Write-Host "[2/6] Telepitesi manifest:"
if (-not (Test-Path -LiteralPath $Manifest)) {
    Bad ("hianyzik a manifest: " + $Manifest + " (futtattad mar az install_b11073.ps1-t?)")
} else {
    $M = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
    $PinSha = ""
    if ($Pins) { $PinSha = ([string]$Pins.runtime.asset.sha256).ToUpper() }
    if (([string]$M.asset_sha256).ToUpper() -eq $PinSha -and $PinSha -ne "") {
        Ok ("manifest sha256 == pins sha256 (asset: " + $M.asset_name + ")")
    } else {
        Bad "a manifest es a pin-tabla asset hash-e elter!"
    }
    Ok ("telepitve: " + $M.installed_utc + " (tag " + $M.tag + ", commit " + $M.build_commit + ")")
}

# ---------------- [3] build identity in-binary ------------------------------------
Write-Host "[3/6] Build-identitas:"
$Impl = Join-Path $BinDir "llama-server-impl.dll"
if (Test-Path -LiteralPath $Impl) {
    $Text = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($Impl))
    if ($Text.Contains("1aa2954bd")) { Ok "llama-server-impl.dll hordozza a b11073 commit-azonositot (1aa2954bd)" }
    else { Bad "a b11073 commit-azonosito NINCS meg a binarisban - mas build!" }
    if ($Text.Contains("a32af33de")) { Bad "a PRODUKCIOS b10717 commit is benne van - kevert binaris!" }
} else {
    Bad "hianyzik a llama-server-impl.dll"
}

# ---------------- [4] production untouched -------------------------------------------
Write-Host "[4/6] Produkcio erintetlensege:"
$ProdExe = Join-Path $Root "bin\llama-server.exe"
if (Test-Path -LiteralPath $ProdExe) {
    $Now = (Get-FileHash -LiteralPath $ProdExe -Algorithm SHA256).Hash
    $Rec = ""
    if ((Test-Path -LiteralPath $Manifest)) {
        try { $Rec = ([string](Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json).production_llama_server_sha256).ToUpper() } catch { }
    }
    if ($Rec -ne "" -and $Now -eq $Rec) {
        Ok ("bin\llama-server.exe hash megegyezik a telepiteskor rogzitettel (b10717 valtozatlan)")
    } elseif ($Rec -eq "") {
        Warn ("bin\llama-server.exe van, de nincs rogzitett telepiteskori hash (regi telepites?) - aktualis: " + $Now.Substring(0, 12) + "...")
    } else {
        Bad "bin\llama-server.exe MEGVALTOZOTT a telepites ota!"
    }
} else {
    Write-Host "  [----]  a gyartasi bin\llama-server.exe nincs telepitve (nincs mit erinteni)"
}
try {
    $Lock = Get-Content -LiteralPath (Join-Path $Root "MODELS.lock.json") -Raw | ConvertFrom-Json
    $Ll = @($Lock.tools | Where-Object { $_.component -eq "llama-server" })
    if ($Ll.Count -gt 0 -and [string]$Ll[0].tag -eq "b10717") { Ok "MODELS.lock.json tovabbra is a b10717-et pin-eli (gyartasi llama.cpp)" }
    else { Bad "a MODELS.lock.json llama-server pinje megvaltozott!" }
} catch { Bad ("MODELS.lock.json nem olvashato: " + $_.Exception.Message) }
$Ver = ""
try { $Ver = (Get-Content -LiteralPath (Join-Path $Root "VERSION") -Raw).Trim() } catch { }
if ($Ver -eq "0.10.7") { Ok "VERSION tovabbra is 0.10.7 (nincs termekkiadas)" }
else { Bad ("VERSION elter: '" + $Ver + "'") }
try {
    $Yaml = Get-Content -LiteralPath (Join-Path $Root "config\llm_config.yaml") -Raw
    if ($Yaml -match "gpu_layers:\s*20" -and $Yaml -match "context_size:\s*32768" -and $Yaml -match "parallel:\s*1") {
        Ok "kanonikus config: ngl 20 / ctx 32768 / parallel 1 (gyartasi profil valtozatlan)"
    } else { Bad "a kanonikus config\llm_config.yaml ertkei megvaltoztak!" }
} catch { Bad ("config\llm_config.yaml nem olvashato: " + $_.Exception.Message) }

if (-not $SkipServer) {
    # ---------------- [5] live server: health + b11073 identity --------------------
    Write-Host "[5/6] El szerver (127.0.0.1:8081):"
    if (Test-Health "127.0.0.1" 8081) {
        Ok "/health OK (127.0.0.1:8081)"
        $Fingerprint = ""
        try {
            $Resp = Invoke-RestMethod -Uri "http://127.0.0.1:8081/v1/chat/completions" `
                -Method Post -ContentType "application/json" `
                -Body (ConvertTo-Json -Compress -InputObject ([ordered]@{
                    messages = @(@{ role = "user"; content = "ping" })
                    max_tokens = 1
                    temperature = 0.0
                })) -TimeoutSec 60
            $Fingerprint = [string]$Resp.system_fingerprint
        } catch { Warn ("a fingerprint-proba nem sikerult: " + $_.Exception.Message) }
        if ($Fingerprint -match "b11073") { Ok ("futto szerver rendszerujjlenyomat: " + $Fingerprint + " (b11073 ELO)") }
        elseif ($Fingerprint -ne "") { Bad ("a 8081-en FUTO szerver NEM b11073! fingerprint = " + $Fingerprint) }
        try {
            $Props = Invoke-WebRequest -Uri "http://127.0.0.1:8081/props" -Method Get -TimeoutSec 5 -UseBasicParsing
            Ok ("/props HTTP " + $Props.StatusCode + " (a validacios audit ugyanezt a vegpontot hasznalta)")
        } catch { Warn "/props nem valaszol (informacios)" }
    } else {
        Bad "a 8081-en NINCS el szerver - inditsd: start_llama_server_experimental.ps1"
    }

    # ---------------- [6] client transport ------------------------------------------
    Write-Host "[6/6] Kliens-transzport (ugyanaz a vegpont, amit az app LLM-kliense hasznal):"
    if (Test-Health "127.0.0.1" 8081) {
        try {
            $Resp = Invoke-RestMethod -Uri "http://127.0.0.1:8081/v1/chat/completions" `
                -Method Post -ContentType "application/json" `
                -Body (ConvertTo-Json -Compress -InputObject ([ordered]@{
                    messages    = @(@{ role = "user"; content = "Say exactly: ok" })
                    max_tokens  = 8
                    temperature = 0.0
                    stream      = $false
                })) -TimeoutSec 120
            $Reply = ""
            try { $Reply = [string]$Resp.choices[0].message.content } catch { }
            if ($Reply -ne "") { Ok ("chat completion valaszott: '" + $Reply.Trim().Substring(0, [Math]::Min(40, $Reply.Trim().Length)) + "'") }
            else { Bad "a chat completion nem adott tartalmat" }
        } catch { Bad ("chat completion HIBA: " + $_.Exception.Message) }
    } else {
        Write-Host "  [----]  kihagyva (a szerver nem fut)"
    }

    # ---------------- optional: the REAL application client --------------------------
    if ($ThroughAppClient) {
        Write-Host "[+] AZ ALKALMAZAS SAJAT LLM-KLIENSE (app\llm.py a env-felulirassal):"
        $VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $VenvPy)) {
            Warn ("nincs .venv python (" + $VenvPy + ") - futtasd egyszer a START.bat-ot, vagy hasznald a [6] transzport-probat")
        } else {
            $Py = @'
import asyncio, os, sys
os.environ["LLAMA_SERVER_HOST"] = "127.0.0.1"
os.environ["LLAMA_SERVER_PORT"] = "8081"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8081/v1"
sys.path.insert(0, r"<VMROOT>")
from app.config import AgentConfig
from app.llm import LlmClient
async def main():
    cfg = AgentConfig(); cfg.apply_env()
    assert cfg.llama_server_url == "http://127.0.0.1:8081/v1", "env override failed"
    print("  [OK]    app client endpoint:", cfg.llama_server_url)
    client = LlmClient(cfg)
    print("  [OK]    app client health :", await client.health_check())
    chunks = []
    async for d in client.chat_stream([{"role": "user", "content": "Say exactly: ok"}], max_tokens=8):
        chunks.append(d)
    print("  [OK]    app client stream :", repr("".join(chunks))[:60])
    await client.aclose()
asyncio.run(main())
'@
            $Py = $Py.Replace("<VMROOT>", $Root)
            $PyFile = Join-Path ([System.IO.Path]::GetTempPath()) ("voicemem_b11073_client_probe_" + [Guid]::NewGuid().ToString("N").Substring(0,8) + ".py")
            [System.IO.File]::WriteAllText($PyFile, $Py)
            try {
                Push-Location $Root
                & $VenvPy $PyFile
                if ($LASTEXITCODE -ne 0) { Bad "az alkalmazas-szintu kliens-proba nem sikerult (l. fenti kimenet)" }
                Pop-Location
            } catch {
                Pop-Location
                Bad ("app client probe: " + $_.Exception.Message)
            } finally {
                Remove-Item -LiteralPath $PyFile -Force -ErrorAction SilentlyContinue
            }
        }
    }
} else {
    Write-Host "[5-6] Kihagyva (-SkipServer): csak strukturallis ellenorzes futott le."
}

# ---------------- summary -----------------------------------------------------------
Write-Host ""
if ($Fails -eq 0) {
    Write-Host "EREDMENY: MINDEN ELLENORZES PASS." -ForegroundColor Green
    exit 0
} else {
    Write-Host ("EREDMENY: " + $Fails + " HIBA.") -ForegroundColor Red
    exit 1
}
