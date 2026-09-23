<#
===============================================================================
experimental/llama_b11073/start_llama_server_experimental.ps1

ISOLATED EXPERIMENTAL llama.cpp runtime launcher (target machine / Windows).

PURPOSE: run the operator-tested NEWER llama.cpp build (b11073) NEXT TO the
untouched production runtime, on a SEPARATE executable path
(experimental\llama_b11073\bin\, installed by install_b11073.ps1) and a
SEPARATE port (default 127.0.0.1:8081), so the whole VoiceMem application
can be pointed at it with the EXISTING environment-only override. The
recommended way is RUN_EXPERIMENTAL.bat (one click) or the GENERATED session
script (start_voicemem_experimental.ps1, created by configure_b11073.ps1
-Enable):

    experimental\llama_b11073\start_voicemem_experimental.ps1

Manual equivalent:

    $env:LLAMA_SERVER_HOST = "127.0.0.1"
    $env:LLAMA_SERVER_PORT = "8081"
    $env:OPENAI_BASE_URL   = "http://127.0.0.1:8081/v1"   # vendor legs + start_agent.ps1 env-guard
    .\START.bat    (NOTE: START.bat would ALSO auto-start the production
                    server on 8080; the generated session script uses
                    scripts\start_agent.ps1 -Web -NoServer instead to avoid
                    loading the 19 GB production model twice.)

REVERTING THE EXPERIMENT (full rollback):
    1. experimental\llama_b11073\configure_b11073.ps1 -Disable
       (stops this server, removes the override, prints the restore steps)
    2. Optionally delete experimental\llama_b11073\ (the runtime + scripts).

PRODUCTION IS NOT TOUCHED BY THIS SCRIPT:
    - it NEVER reads or writes config\llm_config.yaml (the canonical file
      keeps generating the production command line);
    - it NEVER touches bin\llama-server.exe (the pinned b10717 production
      binary) or any production DLL - the experimental executable lives in
      experimental\llama_b11073\bin\;
    - it does NOT edit scripts\start_llama_server.ps1;
    - it binds 127.0.0.1:8081 (never 8080), so the production server can run
      at the same time if the operator wants a live A/B on the same machine.

REVISION 2 (2026-09-23) - LAUNCHER IDENTITY-VERIFICATION FIX (field
feedback from the target): rev 1 gated the start on `llama-server.exe
--version` emitting "11073". On the real target (Windows PowerShell 5.1)
that gate failed EVEN THOUGH the correctly pinned, hash-verified b11073
runtime was installed: the build tag is not reliably capturable from the
exe (console/stderr redirection behaviour - and the binary does not even
carry the literal "b11073" string inside llama-server-impl.dll). Rev 2
replaces the banner gate with the ESTABLISHED authoritative identity
chain - the same evidence the installer verified successfully on the
target:

  BEFORE START (fail-closed):
    [V1] identity table: config\runtime_manifest.json (written by the
         installer) cross-checked against config\b11073.pins.json (the
         pack-shipped pin table; fallback when the manifest is missing);
         both must agree with the embedded expected identity (tag b11073,
         commit 1aa2954bd, asset SHA-256);
    [V2] EVERY pinned file - llama-server.exe + the required DLL set +
         the OpenMP license notice, 24 files - is present and its SHA-256
         equals the pinned value; anything else is REJECTED before start.
         This accepts ONLY the exact pinned b11073 artifact (a stronger
         guarantee than any version banner could give).
    [V3] llama-server-impl.dll carries the b11073 build identity string
         1aa2954bd (same byte-scan method the installer used successfully
         on the target).
         `--version` output is still COLLECTED AND LOGGED - informational
         only, never a gate (rev 1's gate line is gone BY DESIGN).

  AFTER START (live identity, fail-closed):
    [V4] /health turns OK (unchanged primary condition), AND
    [V5] the process LISTENING on 127.0.0.1:8081 is exactly the child this
         launcher started from the hash-verified pinned binary (listener
         PID via Get-NetTCPConnection with a netstat -ano fallback;
         Win32_Process ExecutablePath cross-check when readable), AND
    [V6] the RUNNING server self-identifies as b11073 through its
         system_fingerprint (a real 1-token completion must report
         "b11073-..." - proven supported by the b11073 compatibility
         audit and the pinned SSE fixture), retried up to 3 times.

  The idempotent "already running" exit now passes the SAME [V5]+[V6]
  identity verification before exiting 0 - an unknown listener on 8081 is
  never trusted, and never killed, by this script.

  -VerifyOnly: run [V1]-[V3] and exit without starting anything (the
  pre-flight identity proof on its own).

EXPERIMENTAL PROFILE - the operator-tested baseline (2026-09-20 manual test;
~13 tok/s generation observed with llama-cli on the target machine). This is
the FIRST experimental candidate, NOT an optimised one, and deliberately NOT
the canonical production profile (ngl 20 / ctx 32768):

    ngl 99 | ctx 16000 | parallel 1 | threads 12 | reasoning off
    (+ KV cache q8_0 / temp 0.7 kept equal to production for comparability)

CUDA RESOLUTION (see DEPENDENCY_INSPECTION.md): the runtime imports exactly
one CUDA DLL, cublas64_13.dll, which is NOT installed next to the exe - the
target machine already provides it. The launcher therefore PREPENDS the
production bin\ directory to the CHILD process PATH (never copies or
modifies any CUDA component); a system-wide CUDA 13.x on PATH is an equally
valid source.

MODEL RESOLUTION is IDENTICAL to production (same model - the A/B contract):
    config\llm_model.json (web UI selection) > LLAMA_MODEL_PATH env >
    models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf

NOTE: deliberately ASCII (PowerShell 5.1 reads BOM-less files as Windows-1252).
===============================================================================
#>

param(
    [int]$Port = 8081,
    [int]$WaitSec = 180,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$Root      = (Get-Item -LiteralPath $ScriptDir).Parent.Parent.FullName
$BinDir    = Join-Path $ScriptDir "bin"
$ConfigDir = Join-Path $ScriptDir "config"
$LogsDir   = Join-Path $ScriptDir "logs"
$Exe       = Join-Path $BinDir "llama-server.exe"
$ImplDll   = Join-Path $BinDir "llama-server-impl.dll"
$PinsFile  = Join-Path $ConfigDir "b11073.pins.json"
$Manifest  = Join-Path $ConfigDir "runtime_manifest.json"

# --- the established pinned identity (three-way consistency: this embedded
#     table must agree with the pack's pins file AND the installer manifest) --
$ExpectedTag        = "b11073"
$ExpectedCommit     = "1aa2954bd"
$ExpectedAssetSha256 = "85C1B874180FAEC412CCBBA16EE0833062C28E2BF1390B12ED30F7DC6D7D79C4"

# --- effective-configuration log (this launch's exact resolved state) ------------
if (-not (Test-Path -LiteralPath $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null }
$Stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$CfgLog = Join-Path $LogsDir ("launcher-config_" + $Stamp + ".log")
function Log([string]$m) {
    try { Add-Content -LiteralPath $CfgLog -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $m) } catch { }
}
function Fail([string]$Msg, [string]$Hint) {
    Write-Host ""
    Write-Host ("HIBA: " + $Msg) -ForegroundColor Red
    if ($Hint) { Write-Host ("JAVITAS: " + $Hint) -ForegroundColor Yellow }
    Log ("FAIL: " + $Msg + " | HINT: " + $Hint)
    exit 1
}

Log ("launcher rev 2 start; script dir = " + $ScriptDir + "; root = " + $Root)
Log ("binary = " + $Exe)

Write-Host "=== EXPERIMENTAL llama-server (b11073) - EZ NEM A GYARTASI SZERVER ===" -ForegroundColor Magenta
Write-Host ("  binary : {0}" -f $Exe)

# --- [V0] the pinned runtime must be installed ------------------------------------
if (-not (Test-Path -LiteralPath $Exe)) {
    Fail ("az experimentalis binaris nem letezik: " + $Exe) `
         ("futtasd eloszor a telepitot (a runtime NINCS a pack-ban, letoltes tortenik): powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\install_b11073.ps1 - a gyartasi bin\llama-server.exe (b10717) NINCS erintve.")
}

# --- [V1] identity table: manifest (install record) x pins (pack table) -----------
if (-not (Test-Path -LiteralPath $PinsFile)) {
    Fail ("hianyzik a pin-tabla: " + $PinsFile) `
         ("a csomag resze kell legyen (experimental\llama_b11073\config\b11073.pins.json). Csomagold ki ujra a VoiceMemAgent_v0.10.7_LlamaB11073_Installer.zip-et.")
}
try { $Pins = Get-Content -LiteralPath $PinsFile -Raw | ConvertFrom-Json } catch {
    Fail ("a pin-tabla nem olvashato/ervenytelen JSON: " + $_.Exception.Message) "serult csomag; csomagold ki ujra a ZIP-et."
}
if ([string]$Pins.runtime.tag -ne $ExpectedTag -or [string]$Pins.runtime.build_commit -ne $ExpectedCommit) {
    Fail ("a pin-tabla identitasa elter a varottol (tag/commit): '" + [string]$Pins.runtime.tag + "'/'" + [string]$Pins.runtime.build_commit + "'") `
         ("ervenytelen csomag - varott: " + $ExpectedTag + "/" + $ExpectedCommit + ". SOHA ne lepj at eltero verziora.")
}
if (([string]$Pins.runtime.asset.sha256).ToUpper() -ne $ExpectedAssetSha256) {
    Fail "a pin-tabla asset SHA-256 elter a varottol" "ervenytelen csomag."
}
$Table = @{}
foreach ($f in $Pins.runtime.files) { $Table[[string]$f.name] = ([string]$f.sha256).ToUpper() }
if ($Table.Count -ne 24) {
    Fail ("a pin-tabla fajllista merete (" + $Table.Count + ") elter a mert 24-elemu zarastol") "ervenytelen csomag."
}

$IdentitySource = ""
if (Test-Path -LiteralPath $Manifest) {
    try { $M = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json } catch {
        Fail ("a telepitesi manifest nem olvashato/ervenytelen JSON: " + $_.Exception.Message) "futtasd ujra: install_b11073.ps1 (a mar telepitett runtimeot ujrahasznos es a manifestet helyreallitja)."
    }
    if ([string]$M.tag -ne $ExpectedTag -or [string]$M.build_commit -ne $ExpectedCommit) {
        Fail ("a telepitesi manifest identitasa elter (tag/commit): '" + [string]$M.tag + "'/'" + [string]$M.build_commit + "'") "a manifest masik runtimeot rogzit - SOHA ne hasznald; futtasd ujra install_b11073.ps1 -Force-t."
    }
    if (([string]$M.asset_sha256).ToUpper() -ne $ExpectedAssetSha256) {
        Fail "a telepitesi manifest asset SHA-256 elter a pin-elt ertektol" "a manifest megvaltozott - futtasd ujra install_b11073.ps1 -Force-t."
    }
    $ManTable = @{}
    foreach ($p in $M.files_sha256.PSObject.Properties) { $ManTable[[string]$p.Name] = ([string]$p.Value).ToUpper() }
    if ($ManTable.Count -ne $Table.Count) {
        Fail ("a manifest fajlszama (" + $ManTable.Count + ") elter a pin-tablajevel (" + $Table.Count + ")") "a manifest megvaltozott - futtasd ujra install_b11073.ps1 -Force-t."
    }
    foreach ($k in @($Table.Keys)) {
        if (-not $ManTable.ContainsKey($k)) {
            Fail ("a manifestbol hianyzik a fajl: " + $k) "a manifest megvaltozott - futtasd ujra install_b11073.ps1 -Force-t."
        }
        if ($ManTable[$k] -ne $Table[$k]) {
            Fail ("a manifest es a pin-tabla elter a fajlon: " + $k) "ervenytelen csomag-allapot - futtasd ujra install_b11073.ps1 -Force-t."
        }
    }
    $IdentitySource = "runtime_manifest.json (kereszt-ellenorizve a b11073.pins.json-nel)"
} else {
    Write-Host "  FIGYELMEZTETES: a telepitesi manifest hianyzik - a csomag pin-tablaja alapjan ellenorzok." -ForegroundColor Yellow
    Write-Host "  (A manifest helyreallithato: futtasd ujra install_b11073.ps1 - nem tortenik uj letoltes.)" -ForegroundColor Yellow
    Log "runtime_manifest.json missing; falling back to the pack pin table"
    $IdentitySource = "b11073.pins.json (a manifest HIANYZIK - fallback)"
}
Log ("identity table: " + $IdentitySource)

# --- [V2] full pinned-file SHA-256 gate (the exact-artifact acceptance) -----------
Write-Host ("  ident  : {0}" -f $IdentitySource)
Write-Host ("  ellenorzes: a(z) " + $Table.Count + " pin-elt runtime-fajl SHA-256 hash-e...")
$nOk = 0
$BadFile = ""
$BadDetail = ""
foreach ($k in @($Table.Keys | Sort-Object)) {
    $T = Join-Path $BinDir $k
    if (-not (Test-Path -LiteralPath $T)) {
        $BadFile = $k; $BadDetail = "hianyzo fajl"; break
    }
    $H = (Get-FileHash -LiteralPath $T -Algorithm SHA256).Hash
    if ($H -eq $Table[$k]) {
        $nOk++
        Log ("  ok " + $k + " " + $H.Substring(0, 12) + "...")
    } else {
        $BadFile = $k; $BadDetail = ("vart " + $Table[$k] + ", kapott " + $H); break
    }
}
if ($BadFile -ne "") {
    Fail ("a telepitett runtime NEM a pin-elt b11073: " + $BadFile + " (" + $BadDetail + ")") `
         ("a(z) " + $BadFile + " elter - a kiserleti profil CSAK a pontosan pin-elt artifactot fogadja el. Torold az experimental\llama_b11073\bin mappat es futtasd ujra: install_b11073.ps1 -Force (ujra letolti es SHA-256 alapjan pin-eli a b11073-at).")
}
Write-Host ("  OK: mind a {0} fajl SHA-256 hash-e megegyezik a pin-elt b11073 tablaval." -f $nOk) -ForegroundColor Green
Log ("pre-start file verification: " + $nOk + "/" + $Table.Count + " OK")

# --- [V3] in-binary build identity (the established b11073 anchor) ----------------
try {
    $ImplBytes = [System.IO.File]::ReadAllBytes($ImplDll)
    $ImplText  = [System.Text.Encoding]::ASCII.GetString($ImplBytes)
} catch {
    Fail ("nem olvashato a " + $ImplDll + ": " + $_.Exception.Message) "serult telepites - torold az experimental\llama_b11073\bin mappat es futtasd ujra a telepitot."
}
if (-not $ImplText.Contains($ExpectedCommit)) {
    Fail ("a binaris-ban NINCS meg a b11073 build-identitas (" + $ExpectedCommit + ") - masik build!") `
         "a SHA-256 egyezett, de az identitas-szoveg hianyzik: NE hasznald; ertesitsd a pack kesztojet."
}
Write-Host ("  OK: build-identitas '" + $ExpectedCommit + "' megtalalva a llama-server-impl.dll-ben (b11073).") -ForegroundColor Green
Log ("build identity verified in-binary: " + $ExpectedCommit)

# --- informational --version capture (NOT a gate; rev 1's gate removed by design) --
$VersionInfo = ""
try {
    $OldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $RawLines = & $Exe --version 2>&1 | ForEach-Object { [string]$_ }
    $VersionInfo = (($RawLines | Where-Object { $_ -and $_.Trim() }) -join " | ").Trim()
} catch {
    $VersionInfo = ""
} finally {
    $ErrorActionPreference = $OldEap
}
Log ("--version output (INFORMATIONAL, not a gate): '" + $VersionInfo + "'")
if ($VersionInfo -ne "") {
    Write-Host ("  [info] --version: {0}" -f $VersionInfo) -ForegroundColor DarkGray
}
Write-Host "  verzio : b11073 - VERIFIKALVA (SHA-256 manifest + build-azonositas; nem --version banner)" -ForegroundColor Green

# --- -VerifyOnly: the pre-flight identity proof on its own ------------------------
if ($VerifyOnly) {
    Write-Host ""
    Write-Host "VERIFIKACIO KESZ: a pin-elt b11073 runtime integritasa rendben. (-VerifyOnly: inditas NEM tortent.)" -ForegroundColor Green
    Log "-VerifyOnly: identity verification complete; no server started"
    exit 0
}

# --- model resolution (same order as the production starter) ----------------------
$Model = ""
$ModelSource = ""
$LlmModelJson = Join-Path $Root "config\llm_model.json"
if (Test-Path -LiteralPath $LlmModelJson) {
    try {
        $Sel = Get-Content -LiteralPath $LlmModelJson -Raw | ConvertFrom-Json
        if ($Sel.model -and (Test-Path -LiteralPath $Sel.model)) {
            $Model = [string]$Sel.model
            $ModelSource = "config/llm_model.json (web UI kivalasztas)"
        }
    } catch { }
}
if (-not $Model -and $env:LLAMA_MODEL_PATH -and (Test-Path -LiteralPath $env:LLAMA_MODEL_PATH)) {
    $Model = [string]$env:LLAMA_MODEL_PATH
    $ModelSource = "LLAMA_MODEL_PATH env"
}
if (-not $Model) {
    $Default = Join-Path $Root "models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf"
    if (Test-Path -LiteralPath $Default) {
        $Model = $Default
        $ModelSource = "default profil ut"
    }
}
if (-not $Model) {
    Fail "nincs feloldhato LLM modell (config\llm_model.json > LLAMA_MODEL_PATH > models\llm\qwen3.6-35b-a3b\)."
}
Log ("model = " + $Model + " (source: " + $ModelSource + ")")

# --- experimental profile (operator-tested baseline) ------------------------------
$Ngl = "99"
if ($env:LLAMA_EXPERIMENTAL_NGL) { $Ngl = [string]$env:LLAMA_EXPERIMENTAL_NGL }
$Ctx = "16000"
if ($env:LLAMA_EXPERIMENTAL_CTX) { $Ctx = [string]$env:LLAMA_EXPERIMENTAL_CTX }
$Threads = "12"
if ($env:LLAMA_EXPERIMENTAL_THREADS) { $Threads = [string]$env:LLAMA_EXPERIMENTAL_THREADS }

$Host_ = "127.0.0.1"

Write-Host ("  model  : {0}  ({1})" -f $Model, $ModelSource)
Write-Host ("  bind   : {0}:{1}   (a gyartasi 8080 NINCS erintve)" -f $Host_, $Port)
Write-Host ("  profil : ngl {0} | ctx {1} | parallel 1 | threads {2} | reasoning off | KV q8_0" -f $Ngl, $Ctx, $Threads)

# --- helpers for the LIVE identity verification ------------------------------------
function Test-Health {
    param([string]$H, [int]$P, [int]$T = 3)
    try {
        $R = Invoke-WebRequest -Uri ("http://{0}:{1}/health" -f $H, $P) -Method Get -TimeoutSec $T -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

function Get-ListenerPids {
    # The PIDs of the processes LISTENING on the port. Primary: Get-NetTCPConnection
    # (Win8+/PS3+). Fallback: netstat -ano (the protocol token "TCP" is not
    # localised; we match the LOCAL address column ending in ":<port>" and require
    # a numeric final column - established lines never carry the server port in
    # the local column).
    param([int]$P)
    $Found = @()
    $Got = $false
    try {
        $Conns = Get-NetTCPConnection -LocalPort $P -State Listen -ErrorAction Stop
        foreach ($c in $Conns) { if ($c.OwningProcess) { $Found += [int]$c.OwningProcess } }
        $Got = $true
    } catch { }
    if (-not $Got) {
        try {
            $NsLines = & netstat -ano 2>$null
            foreach ($L in $NsLines) {
                $Cols = @($L -split '\s+' | Where-Object { $_ })
                if ($Cols.Count -ge 4 -and [string]$Cols[0] -eq "TCP" -and [string]$Cols[1] -like ("*:" + $P)) {
                    $Pv = 0
                    if ([int]::TryParse([string]$Cols[-1], [ref]$Pv)) { $Found += $Pv }
                }
            }
        } catch { }
    }
    return @($Found | Sort-Object -Unique)
}

function Get-ProcPath {
    param([int]$ProcId)
    try {
        $Wp = Get-CimInstance -ClassName Win32_Process -Filter ("ProcessId = " + $ProcId) -ErrorAction Stop
        if ($Wp) { return [string]$Wp.ExecutablePath }
    } catch { }
    try {
        $Wm = Get-WmiObject -Class Win32_Process -Filter ("ProcessId = " + $ProcId) -ErrorAction Stop
        if ($Wm) { return [string]$Wm.ExecutablePath }
    } catch { }
    return ""
}

function Get-SystemFingerprint {
    # The RUNNING server's own identity: a real 1-token chat completion must
    # report system_fingerprint "b11073-...". Proven supported by the b11073
    # compatibility audit and the pinned SSE fixture. Non-streaming first;
    # if that response carries no fingerprint, the streaming shape is parsed.
    param([string]$H, [int]$P, [int]$Attempts = 3)
    for ($A = 1; $A -le $Attempts; $A++) {
        try {
            $Resp = Invoke-RestMethod -Uri ("http://{0}:{1}/v1/chat/completions" -f $H, $P) `
                -Method Post -ContentType "application/json" `
                -Body (ConvertTo-Json -Compress -InputObject ([ordered]@{
                    messages = @(@{ role = "user"; content = "ping" })
                    max_tokens = 1
                    temperature = 0.0
                })) -TimeoutSec 90
            $Fp = [string]$Resp.system_fingerprint
            if ($Fp -ne "") { return $Fp }
        } catch {
            Log ("system_fingerprint probe (non-streaming) " + $A + "/" + $Attempts + " sikertelen: " + $_.Exception.Message)
        }
        try {
            $R = Invoke-WebRequest -Uri ("http://{0}:{1}/v1/chat/completions" -f $H, $P) `
                -Method Post -ContentType "application/json" `
                -Body (ConvertTo-Json -Compress -InputObject ([ordered]@{
                    messages = @(@{ role = "user"; content = "ping" })
                    max_tokens = 1
                    temperature = 0.0
                    stream = $true
                })) -TimeoutSec 90 -UseBasicParsing
            if ($R.Content -match '"system_fingerprint"\s*:\s*"([^"]+)"') { return $Matches[1] }
        } catch {
            Log ("system_fingerprint probe (streaming) " + $A + "/" + $Attempts + " sikertelen: " + $_.Exception.Message)
        }
        if ($A -lt $Attempts) { Start-Sleep -Seconds 3 }
    }
    return ""
}

# --- CUDA prerequisite: prepend the production bin\ to the CHILD PATH --------------
# cublas64_13.dll (the ONLY CUDA import of ggml-cuda.dll) resolves from the
# existing production cudart install and/or a system CUDA 13.x. The pack and
# the installer never place CUDA DLLs next to the exe; this PATH extension is
# the only mechanism, and it applies to the CHILD process alone.
$ProdBin = Join-Path $Root "bin"
$CudaSource = ""
$CudaSearch = @($ProdBin)
if ($env:windir) { $CudaSearch += (Join-Path $env:windir "System32") }
foreach ($D in $CudaSearch) {
    if (Test-Path -LiteralPath (Join-Path $D "cublas64_13.dll")) { $CudaSource = $D; break }
}
if (-not $CudaSource) {
    foreach ($D in @([Environment]::GetEnvironmentVariable("PATH") -split ([System.IO.Path]::PathSeparator) | Where-Object { $_ })) {
        if (Test-Path -LiteralPath (Join-Path $D "cublas64_13.dll")) { $CudaSource = $D; break }
    }
}
if ($CudaSource) {
    Write-Host ("  cuda   : cublas64_13.dll <- {0} (gyermek-PATH bovites)" -f $CudaSource)
    Log ("cublas64_13.dll resolves from: " + $CudaSource)
} else {
    Write-Host "  FIGYELMEZTETES: cublas64_13.dll NINCS a keresesi utvonalon!" -ForegroundColor Yellow
    Write-Host "  A szerver inditasa valoszinuleg 0xC0000135-tel elhal (hianyzo DLL)." -ForegroundColor Yellow
    Write-Host "  Varhato forras: a gyartasi bin\ cudart letoltes (START.bat egyszeri futtatasa)" -ForegroundColor Yellow
    Write-Host "  vagy rendszer-szintu CUDA 13.x. Reszletek: README.md." -ForegroundColor Yellow
    Log "cublas64_13.dll NOT FOUND on search path (warning)"
}
$env:PATH = "$ProdBin;$env:PATH"
Log ("child PATH prepended with: " + $ProdBin)

# --- idempotent start: an ALREADY-RUNNING 8081 server must IDENTIFY as b11073 ------
if (Test-Health $Host_ $Port) {
    Write-Host "MAR FUT egy szerver a ${Host_}:${Port} cimen - azonositom, mielott megbiznek benne..." -ForegroundColor Yellow
    Log "health already OK before start; verifying the existing listener's identity"
    $AlreadyOk = $true
    $Listeners = @(Get-ListenerPids -P $Port)
    Log ("listener pids on " + $Port + ": " + ($Listeners -join ", "))
    if ($Listeners.Count -eq 0) {
        Write-Host "  HIBA: a listener PID nem hatarozhato meg (Get-NetTCPConnection es netstat is)" -ForegroundColor Red
        $AlreadyOk = $false
    }
    foreach ($Lp in $Listeners) {
        $LPath = Get-ProcPath -ProcId $Lp
        if ($LPath -eq "") {
            Write-Host ("  FIGYELMEZTETES: a(z) " + $Port + "-on figyelo PID " + $Lp + " utvonala nem olvashato - a fingerprint-proba dont.") -ForegroundColor Yellow
            Log ("listener pid " + $Lp + ": executable path unreadable")
        } elseif ($LPath -ne $Exe) {
            Write-Host ("  HIBA: a(z) " + $Port + "-on figyelo PID " + $Lp + " NEM a pin-elt b11073 binaris:") -ForegroundColor Red
            Write-Host ("        " + $LPath) -ForegroundColor Red
            Log ("listener pid " + $Lp + " is NOT the pinned binary: " + $LPath)
            $AlreadyOk = $false
        } else {
            Write-Host ("  [OK] listener PID " + $Lp + " = a pin-elt b11073 binaris")
            Log ("listener pid " + $Lp + " = pinned binary")
        }
    }
    if ($AlreadyOk) {
        $Fingerprint = Get-SystemFingerprint -H $Host_ -P $Port
        if ($Fingerprint -match "b11073") {
            Write-Host ("  VERZIO-ELLENORZES OK: a mar futtato szerver rendszerujjlenyomat = " + $Fingerprint) -ForegroundColor Green
            Log ("existing server verified: system_fingerprint = " + $Fingerprint)
            Write-Host "Nem inditok masodik szervert. Visszaallas: configure_b11073.ps1 -Disable." -ForegroundColor Gray
            exit 0
        }
        Fail ("a mar futto 8081-es szerver NEM azonosithato b11073-kent (fingerprint = '" + $Fingerprint + "')") `
             "allitsd le es nezd meg, mi fut a 8081-en: configure_b11073.ps1 -Disable (csak az experimentalis binaris utvonalarol inditott folyamatot allitja le) vagy netstat -ano | findstr :8081."
    }
    Fail ("a(z) " + $Port + "-on mar egy ISMERETLEN szerver fut, ami NEM a pin-elt b11073") `
         "ez a szkript semmit nem allt le es nem is indit. Nezd meg: netstat -ano | findstr :8081, majd allitsd le a megfelelo folyamatot, vagy hasznald a configure_b11073.ps1 -Disable-t."
}

$ErrLog = Join-Path $LogsDir "llama-server-experimental.err.log"
$OutLog = Join-Path $LogsDir "llama-server-experimental.out.log"

$LlArgs = @(
    "--model", $Model,
    "--host", $Host_, "--port", "$Port",
    "-ngl", $Ngl, "-c", $Ctx, "--parallel", "1", "-t", $Threads,
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--temp", "0.7", "--reasoning", "off", "--metrics", "--no-webui", "--verbose"
)
Write-Host ("Inditas: {0} {1}" -f $Exe, ($LlArgs -join " "))
Write-Host ("Logok  : {0} / {1}" -f $ErrLog, $OutLog)
Write-Host ("Konfig : {0}" -f $CfgLog)
Log ("effective command: " + $Exe + " " + ($LlArgs -join " "))
Log ("effective env overrides: NGL=" + $Ngl + " CTX=" + $Ctx + " THREADS=" + $Threads + " PORT=" + $Port)

$Proc = Start-Process -FilePath $Exe -ArgumentList $LlArgs -WorkingDirectory $Root `
    -RedirectStandardError $ErrLog -RedirectStandardOutput $OutLog -PassThru
Log ("started pid " + $Proc.Id)

# --- /health wait + LIVE identity verification ---------------------------------------
$Deadline = (Get-Date).AddSeconds($WaitSec)
while ((Get-Date) -lt $Deadline) {
    if ($Proc.HasExited) {
        Write-Host ("HIBA: az experimentalis llama-server kilepett (kod {0}). Log: {1}" -f $Proc.ExitCode, $ErrLog) -ForegroundColor Red
        Log ("server exited early, code " + $Proc.ExitCode)
        if (Test-Path -LiteralPath $ErrLog) {
            Get-Content -LiteralPath $ErrLog -Tail 8 | ForEach-Object { Write-Host ("          " + $_) -ForegroundColor DarkYellow }
        }
        Write-Host "Gyakori ok: hianyzo cublas64_13.dll (l. fent) vagy serult modellfajl." -ForegroundColor Yellow
        exit 1
    }
    if (Test-Health $Host_ $Port) {
        Write-Host "OK: experimentalis szerver ELO a ${Host_}:${Port} cimen." -ForegroundColor Green
        Log "health OK"

        # --- [V5] the listener on the port must be the child we launched ------
        $Listeners = @(Get-ListenerPids -P $Port)
        Log ("listener pids on " + $Port + ": " + ($Listeners -join ", ") + "; our child pid = " + $Proc.Id)
        if ($Listeners.Count -eq 0 -or -not ($Listeners -contains $Proc.Id)) {
            Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
            Fail ("a(z) " + $Port + "-on figyelo folyamat NEM az altalunk inditott gyermek (listener pids: '" + ($Listeners -join ", ") + "', gyermek: " + $Proc.Id + ")") `
                 ("valami mas foglalta el a portot; nezd meg: netstat -ano | findstr :" + $Port + ".")
        }
        $LPath = Get-ProcPath -ProcId $Proc.Id
        if ($LPath -ne "" -and $LPath -ne $Exe) {
            Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
            Fail ("a gyermek folyamat utvonala elter a verify-olt binaristol: " + $LPath) "nem szabadna elofordulnia - ertesitsd a pack kesztojet."
        }
        if ($LPath -ne "") {
            Write-Host ("  [OK] a {0}-es listener (PID {1}) utvonala a verify-olt pin-elt binaris." -f $Port, $Proc.Id)
            Log ("listener executable path: " + $LPath)
        } else {
            Write-Host ("  [OK] a {0}-es listener PID {1} = az inditott gyermek (utvonal nem olvashato)." -f $Port, $Proc.Id)
        }

        # --- [V6] the RUNNING server must self-identify as b11073 ------------
        $Fingerprint = Get-SystemFingerprint -H $Host_ -P $Port
        if ($Fingerprint -match "b11073") {
            Write-Host ("  VERZIO-ELLENORZES OK: a futtato szerver rendszerujjlenyomat = " + $Fingerprint) -ForegroundColor Green
            Log ("system_fingerprint = " + $Fingerprint + " (b11073 CONFIRMED LIVE)")
        } else {
            Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
            Fail ("a 8081-es szerver NEM azonosithato b11073-kent (fingerprint = '" + $Fingerprint + "')") `
                 "a verify-olt binarisbol inditott szerver nem b11073-kent jelentkezett - ertesitsd a pack kesztojet; kozben allitsd le: configure_b11073.ps1 -Disable."
        }

        Write-Host "Visszaallas (rollback): configure_b11073.ps1 -Disable" -ForegroundColor Gray
        Write-Host "Ez az ablak nyitva tartja a folyamatot (Ctrl+C = leallitas)." -ForegroundColor Gray
        Wait-Process -Id $Proc.Id
        exit 0
    }
    Start-Sleep -Milliseconds 500
}
Write-Host "HIBA: az /health nem valt elerhetove ${WaitSec} mp alatt. Log: $ErrLog" -ForegroundColor Red
Log "health wait timed out"
Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
exit 1
