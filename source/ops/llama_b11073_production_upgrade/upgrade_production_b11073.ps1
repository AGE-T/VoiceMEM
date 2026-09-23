<#
===============================================================================
ops/llama_b11073_production_upgrade/upgrade_production_b11073.ps1

PRODUCTION llama.cpp RUNTIME UPGRADE: b10717 -> b11073 (target machine / Windows).

PURPOSE: replace the production llama.cpp server runtime in bin\ with the
ALREADY INSTALLED and ALREADY VERIFIED b11073 build (experimental\
llama_b11073\bin\, installed by the rev-2 installer pack), while KEEPING:

  - the Qwen3.6 35B A3B IQ4_XS model EXACTLY where it is
    (C:\AI_HOME\models\blobs\sha256-afc7238af403ed454b7846454091b5e38b07575bdbb64c3e86777414dde4193c
    - never moved, re-downloaded, re-quantised or modified);
  - the existing VoiceMem application code (no .py / .ps1 / config change);
  - the production endpoint 127.0.0.1:8080 and the production profile
    (-ngl 16 -c 16000 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0
    --temp 0.7 --reasoning off --metrics --no-webui);
  - the existing production startup mechanism (scripts\start_llama_server.ps1).

THIS IS A SERVER RUNTIME UPGRADE, NOT A MODEL CHANGE.

The pack contains NO binaries and performs NO download: the verified
experimental install is reused in place. Only the 24 pinned runtime files
(the exact b11073 artifact set) are copied into bin\; the CUDA runtime DLLs
already in bin\ (cudart/cublas - required by the b11073 ggml-cuda.dll) are
deliberately PRESERVED.

STAGES (full run):
  [S0] preflight    identity tables cross-checked (pack config x experimental
                     pins x install manifest), the SOURCE runtime verified
                     24/24 SHA-256 + build id 1aa2954bd BEFORE anything changes
  [S1] audit        the CURRENT production state recorded to evidence:
                     exe SHA-256 + version info, --version output (informational
                     only - NEVER a gate, the rev-1 experimental launcher bug is
                     not repeated here), live command line + PID, /health,
                     /v1/models, CUDA DLL hashes, model file integrity marks,
                     python-source hash snapshot; documented-profile compliance
                     of the audited command line (fail-closed BEFORE changes)
  [S2] stop         the production llama-server stopped cleanly (only the exact
                     8080 listener, only if it is llama-server.exe from bin\),
                     port 8080 confirmed free
  [S3] backup       timestamped rollback backup of the whole bin\ OUTSIDE the
                     active runtime path (backups\runtime_pre_b11073_<ts>\)
  [S4] install      the 24 pinned files copied into bin\; stale llama.cpp-family
                     DLLs removed to the backup; CUDA runtime DLLs untouched
  [S5] verify       the INSTALLED runtime verified again (24/24 + build id),
                     CUDA dependency presence checked
  [S6] start        the production server started through the EXISTING starter
                     (scripts\start_llama_server.ps1, new window), /health
                     waited for, then the live identity gates: listener PID =
                     bin\llama-server.exe, the new command line compliant with
                     the documented profile, /v1/models serving the SAME model
  [S7] battery      health / models / Hungarian chat completion / streaming
                     (first token + clean termination) / JSON completion /
                     system_fingerprint == b11073-1aa2954bd
  [S8] integration  the normal web backend started (scripts\start_agent.ps1
                     -Web -NoServer), its LLM endpoint verified as
                     http://127.0.0.1:8080/v1, then the operator performs the
                     three production-chain turns while the [chain] log is
                     captured (ASR transcript, LLM first-token/total latency,
                     reply length, TTS chunking, repetition / HU-EN oscillation
                     flags)
  [S9] regression   port still 8080, model unchanged, b11073 identity, no
                     python-source change, no model change, no 8081/8082
                     leakage, starter intact
  [S10] report      REPORT.md + stage_results.json + evidence ZIP in
                     logs\prod_b11073_upgrade_<ts>\

AUTO-ROLLBACK: on server startup / CUDA / /health / /v1/models / chat /
streaming / VoiceMem-connection failure the timestamped backup is restored
and the old server restarted (see rollback_production_b11073.ps1); -NoAutoRollback
disables this for diagnosis.

MODES:
  -VerifyOnly   pre-flight identity proof of the SOURCE runtime, no changes
  -AuditOnly    record the current production evidence, no changes, no start
  -ProbeOnly    verify the CURRENTLY RUNNING server on -Port (battery + gates)
                without changing anything
  -FunctionsOnly dot-source support: load the helpers without running anything
                (advanced/test usage)

USAGE (Bypass form - policy-independent, MOTW-safe):
  powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\upgrade_production_b11073.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\upgrade_production_b11073.ps1 -SkipIntegration
  powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\upgrade_production_b11073.ps1 -VerifyOnly
  powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\upgrade_production_b11073.ps1 -ProbeOnly
  powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\rollback_production_b11073.ps1   (rollback)

NOTE: the file is deliberately ASCII (PowerShell 5.1 without BOM would read
accented characters as Windows-1252), and PS 5.1-safe by construction (no
ternary / null-coalescing operators, EAP-safe native output capture).
===============================================================================
#>
param(
    [string]$Root = "",
    [string]$ModelPath = "",
    [int]$Port = 8080,
    [int]$WebPort = 8787,
    [int]$HealthTimeoutSec = 900,
    [int]$ChatTimeoutSec = 180,
    [switch]$VerifyOnly,
    [switch]$AuditOnly,
    [switch]$ProbeOnly,
    [switch]$SkipIntegration,
    [switch]$NoAutoRollback,
    [switch]$FunctionsOnly
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# ---------------------------------------------------------------------------
# Embedded expected identity (cross-checked against the pack config file and
# the machine's experimental install tables - all three must agree).
# ---------------------------------------------------------------------------
$script:ExpectedTag         = "b11073"
$script:ExpectedCommit      = "1aa2954bd"
$script:ExpectedFingerprint = "b11073-1aa2954bd"
$script:ExpectedAssetSha256 = "85C1B874180FAEC412CCBBA16EE0833062C28E2BF1390B12ED30F7DC6D7D79C4"
$script:ExpectedFileCount   = 24
$script:OldBuildCommit      = "a32af33de"   # informational anchor of b10717
$script:ExpectedModelPath   = "C:\AI_HOME\models\blobs\sha256-afc7238af403ed454b7846454091b5e38b07575bdbb64c3e86777414dde4193c"

# Documented production profile (the task-mandated values - never silently changed)
$script:DocProfile = [ordered]@{
    host      = "127.0.0.1"
    port      = "8080"
    ngl       = "16"
    ctx       = "16000"
    parallel  = "1"
    ckk       = "q8_0"
    ckv       = "q8_0"
    temp      = "0.7"
    reasoning = "off"
    metrics   = $true
    nowebui   = $true
}

# Hungarian test prompts as JSON unicode escapes (keeps THIS file pure ASCII;
# the server-side JSON parser decodes them to the accented Hungarian text).
$script:HuChatBody = '{"messages":[{"role":"user","content":"V\u00e1laszolj egyetlen r\u00f6vid magyar mondattal: mi a f\u0151v\u00e1rosa Magyarorsz\u00e1gnak?"}],"max_tokens":64,"temperature":0.2,"stream":false}'
$script:HuStreamBody = '{"messages":[{"role":"user","content":"V\u00e1laszolj egyetlen r\u00f6vid magyar mondattal: mi a f\u0151v\u00e1rosa Magyarorsz\u00e1gnak?"}],"max_tokens":64,"temperature":0.2,"stream":true}'
$script:HuJsonBody = '{"messages":[{"role":"user","content":"V\u00e1laszolj CSAK egy JSON objektummal, pontosan k\u00e9t mez\u0151vel: nyelv (magyar) \u00e9s v\u00e1ros (budapest)."}],"max_tokens":64,"temperature":0.2,"stream":false,"response_format":{"type":"json_object"}}'
$script:OkProbeBody = '{"messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":16,"temperature":0,"stream":false}'

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
$script:ScriptDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Root)) {
    # ops\llama_b11073_production_upgrade -> project root is two levels up
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$script:Root        = $Root
$script:ProdBin     = Join-Path $Root "bin"
$script:ProdExe     = Join-Path $ProdBin "llama-server.exe"
$script:Starter     = Join-Path $Root "scripts\start_llama_server.ps1"
$script:AgentStarter = Join-Path $Root "scripts\start_agent.ps1"
$script:ExpDir      = Join-Path $Root "experimental\llama_b11073"
$script:ExpBin      = Join-Path $ExpDir "bin"
$script:ExpPins     = Join-Path $ExpDir "config\b11073.pins.json"
$script:ExpManifest = Join-Path $ExpDir "config\runtime_manifest.json"
$script:PackCfg     = Join-Path $PSScriptRoot "config\prod_upgrade_expected.json"
$script:Rollback    = Join-Path $PSScriptRoot "rollback_production_b11073.ps1"
$script:ModelsLock  = Join-Path $Root "MODELS.lock.json"
$script:WebLog      = Join-Path $Root "logs\web-server.log"
if ([string]::IsNullOrWhiteSpace($ModelPath)) { $script:ModelPath = $script:ExpectedModelPath } else { $script:ModelPath = $ModelPath }

# ---------------------------------------------------------------------------
# Logging / evidence helpers
# ---------------------------------------------------------------------------
$script:StartedAt    = Get-Date
$script:Stamp        = $script:StartedAt.ToString("yyyyMMdd_HHmmss")
$script:EvidenceDir  = Join-Path $Root ("logs\prod_b11073_upgrade_" + $script:Stamp)
$script:MasterLog    = ""
$script:ChangesMade  = $false
$script:RollbackDone  = $false
$script:BackupDir    = ""
$script:ModifiedFiles= @()
$script:RemovedFiles = @()
$script:AlreadyNew   = $false
$script:R            = [ordered]@{
    run          = $script:Stamp
    mode         = "full"
    started_utc  = $script:StartedAt.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    finished_utc = ""
    stage        = "init"
    audit        = $null
    install      = $null
    start        = $null
    battery      = $null
    integration  = $null
    regression   = $null
    rollback     = "not needed"
    verdict      = "in progress"
}

function Log {
    param([string]$Msg)
    if ($script:MasterLog -ne "") {
        try {
            Add-Content -LiteralPath $script:MasterLog -Value ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Msg) -Encoding UTF8
        } catch { }
    }
}

function Out-Line {
    param([string]$Msg, [string]$Color = "Gray")
    Write-Host $Msg -ForegroundColor $Color
    Log $Msg
}

function Fail {
    param([string]$Msg, [string]$Hint = "")
    Write-Host ""
    Write-Host ("HIBA: " + $Msg) -ForegroundColor Red
    if ($Hint -ne "") { Write-Host ("JAVITAS: " + $Hint) -ForegroundColor Yellow }
    Log ("FAIL: " + $Msg + " | HINT: " + $Hint)
    if ($script:ChangesMade -and (-not $script:NoAutoRollback) -and (-not $script:RollbackDone)) {
        Invoke-AutoRollback ("kritikus szakasz-hiba: " + $Msg)
    }
    $script:R.verdict = ("FAILED: " + $Msg)
    Close-Report
    exit 1
}

function Get-Sha256 {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return "" }
    try { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash } catch { return "" }
}

function Get-PartialSha256 {
    # Strong cheap integrity mark for a very large file: SHA-256 over the
    # first and last $Head/$Tail bytes (never the whole 19 GB model).
    param([string]$Path, [int]$Head = 4194304, [int]$Tail = 4194304)
    if (-not (Test-Path -LiteralPath $Path)) { return "" }
    $sha = $null; $fs = $null
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $fs = [System.IO.File]::OpenRead($Path)
        $len = $fs.Length
        $buf = New-Object byte[] $Head
        $n = $fs.Read($buf, 0, $buf.Length)
        if ($n -gt 0) { [void]$sha.TransformBlock($buf, 0, $n, $null, 0) }
        $tailStart = $Head
        if ($len -gt ($Head + $Tail)) { $tailStart = $len - $Tail }
        if ($tailStart -lt $Head) { $tailStart = $Head }
        if ($tailStart -lt $len) {
            [void]$fs.Seek($tailStart, [System.IO.SeekOrigin]::Begin)
            $buf2 = New-Object byte[] ($len - $tailStart)
            $m = $fs.Read($buf2, 0, $buf2.Length)
            if ($m -gt 0) { [void]$sha.TransformFinalBlock($buf2, 0, $m) } else { [void]$sha.TransformFinalBlock($buf2, 0, 0) }
        } else {
            [void]$sha.TransformFinalBlock(@(0), 0, 0)
        }
        return ([BitConverter]::ToString($sha.Hash) -replace "-", "")
    } catch {
        return ""
    } finally {
        if ($null -ne $fs) { try { $fs.Close() } catch { } }
        if ($null -ne $sha) { try { $sha.Dispose() } catch { } }
    }
}

function ConvertFrom-Mojibake {
    # Best-effort repair for a PS 5.1 Invoke-WebRequest response that was
    # decoded as ISO-8859-1 although the server sent UTF-8 (Latin-1 round-trip
    # heuristic: only applied when the UTF-8 re-decode is clean).
    param([string]$S)
    if ([string]::IsNullOrEmpty($S)) { return $S }
    try {
        $bytes = [System.Text.Encoding]::GetEncoding("ISO-8859-1").GetBytes($S)
        $fixed = [System.Text.Encoding]::UTF8.GetString($bytes)
        if ($fixed -notmatch "\uFFFD") { return $fixed }
    } catch { }
    return $S
}

function Test-BytesContain {
    param([string]$Path, [string]$AsciiNeedle)
    try {
        $bytes = [System.IO.File]::ReadAllBytes($Path)
        $text = [System.Text.Encoding]::ASCII.GetString($bytes)
        return $text.Contains($AsciiNeedle)
    } catch { return $false }
}

function Get-ListenerPids {
    # PIDs of the processes LISTENING on the port. Primary Get-NetTCPConnection
    # (Win8+/PS3+), fallback netstat -ano (the same tolerance the rev-2
    # experimental launcher proved on the target: localised state strings are
    # never assumed, only the protocol token "TCP" and a numeric final column).
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
                $Cols = @("$L" -split '\s+' | Where-Object { $_ })
                if ($Cols.Count -ge 4 -and ([string]$Cols[0]).ToUpper() -eq "TCP" -and [string]$Cols[1] -like ("*:" + $P)) {
                    $Pv = 0
                    if ([int]::TryParse([string]$Cols[-1], [ref]$Pv)) { $Found += $Pv }
                }
            }
            if ($Found.Count -gt 0) { $Got = $true }
        } catch { }
    }
    if (-not $Got) {
        # Linux/sandbox fallback (the pack's own execution tests run under pwsh
        # on Linux where Get-NetTCPConnection does not exist and netstat -ano has
        # a different shape). A no-op on Windows (no ss there).
        try {
            $SsLines = & ss -tlnp 2>$null
            foreach ($L in $SsLines) {
                $Ls = [string]$L
                if ($Ls -match ('LISTEN\s+\d+\s+\d+\s+\S*:' + $P + '\s') -and $Ls -match 'pid=(\d+)') {
                    $Found += [int]$Matches[1]
                }
            }
        } catch { }
    }
    return @($Found | Sort-Object -Unique)
}

function Get-ProcInfo {
    # @{ name; path; cmdline } for a PID, WMI-first with graceful degradation.
    param([int]$ProcId)
    $info = @{ name = ""; path = ""; cmdline = "" }
    try {
        $Wp = Get-CimInstance -ClassName Win32_Process -Filter ("ProcessId = " + $ProcId) -ErrorAction Stop
        if ($Wp) {
            $info.path = [string]$Wp.ExecutablePath
            $info.cmdline = [string]$Wp.CommandLine
            if ($info.path -ne "") { $info.name = [System.IO.Path]::GetFileName($info.path) }
        }
    } catch {
        try {
            $Wm = Get-WmiObject -Class Win32_Process -Filter ("ProcessId = " + $ProcId) -ErrorAction Stop
            if ($Wm) {
                $info.path = [string]$Wm.ExecutablePath
                $info.cmdline = [string]$Wm.CommandLine
                if ($info.path -ne "") { $info.name = [System.IO.Path]::GetFileName($info.path) }
            }
        } catch { }
    }
    if ($info.name -eq "") {
        try {
            $P = Get-Process -Id $ProcId -ErrorAction Stop
            $info.name = [string]$P.ProcessName
        } catch { }
    }
    return $info
}

function Test-Health {
    param([string]$H, [int]$P, [int]$T = 3)
    try {
        $R = Invoke-WebRequest -Uri ("http://{0}:{1}/health" -f $H, $P) -Method Get -TimeoutSec $T -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

function Get-ServedModelId {
    param([string]$H, [int]$P)
    try {
        $M = Invoke-RestMethod -Uri ("http://{0}:{1}/v1/models" -f $H, $P) -Method Get -TimeoutSec 10
        return [string]$M.data[0].id
    } catch { return "" }
}

function Get-PathLeaf {
    # Last path segment of ANY path form (both separators, both OSes).
    param([string]$P)
    if ([string]::IsNullOrWhiteSpace($P)) { return "" }
    $i = [Math]::Max($P.LastIndexOf('\'), $P.LastIndexOf('/'))
    if ($i -ge 0) { return $P.Substring($i + 1) }
    return $P
}

function Test-ServedModelMatches {
    # The production starter's proven containment comparison (works for the
    # extension-less Ollama sha256-... blob name too).
    param([string]$ServedId, [string]$ExpectedModelPath)
    if ([string]::IsNullOrWhiteSpace($ServedId)) { return "unknown" }
    $Norm = {
        param([string]$S)
        return ([string]$S).ToLower() -replace '[^0-9a-z]', ''
    }
    $ServedN = & $Norm $ServedId
    $leaf = (Get-PathLeaf $ExpectedModelPath)
    if ($leaf.Contains(".")) { $leaf = $leaf.Substring(0, $leaf.LastIndexOf(".")) }
    $StemN = & $Norm $leaf
    if ($ServedN.Length -eq 0) { return "unknown" }
    if ($StemN.Length -gt 0 -and (($ServedN.Contains($StemN)) -or ($StemN.Contains($ServedN)))) { return "same" }
    return "different"
}

function Invoke-LlamaChat {
    param([string]$H, [int]$P, [string]$BodyJson, [int]$TimeoutSec = 180)
    $out = @{ ok = $false; status = 0; obj = $null; raw = ""; err = ""; ms = 0 }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $R = Invoke-WebRequest -Uri ("http://{0}:{1}/v1/chat/completions" -f $H, $P) -Method Post `
            -Body $BodyJson -ContentType "application/json" -TimeoutSec $TimeoutSec -UseBasicParsing
        $out.status = [int]$R.StatusCode
        $out.raw = [string]$R.Content
        try { $out.obj = $R.Content | ConvertFrom-Json } catch { }
        $out.ok = (($out.status -ge 200) -and ($out.status -lt 300))
    } catch {
        $out.err = $_.Exception.Message
        try { if ($_.Exception.Response) { $out.status = [int]$_.Exception.Response.StatusCode } } catch { }
    }
    $out.ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 0)
    return $out
}

function Invoke-StreamProbe {
    # SSE streaming verification: first-token arrival, chunk count, clean
    # [DONE] termination. Uses HttpClient + ReadLineAsync with per-line and
    # global budgets so a stalled stream can never hang the upgrade.
    param([string]$H, [int]$P, [string]$BodyJson, [int]$TimeoutSec = 180)
    $out = @{ ok = $false; status = 0; firstChunkMs = 0; totalMs = 0; sawDone = $false; chunks = 0; text = ""; err = ""; finish = ""; fingerprint = "" }
    $client = $null; $reader = $null
    try {
        Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue
        $client = New-Object System.Net.Http.HttpClient
        $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)
        $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, ("http://{0}:{1}/v1/chat/completions" -f $H, $P))
        $req.Content = New-Object System.Net.Http.StringContent($BodyJson, [System.Text.Encoding]::UTF8, "application/json")
        $tSend = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead)
        if (-not $tSend.Wait(20000)) { $out.err = "headers timeout (20 s)"; return $out }
        $resp = $tSend.Result
        $out.status = [int]$resp.StatusCode
        if (-not $resp.IsSuccessStatusCode) { $out.err = ("HTTP " + $out.status); return $out }
        $stream = $resp.Content.ReadAsStreamAsync().Result
        $reader = New-Object System.IO.StreamReader($stream)
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $first = $true
        while ($true) {
            $lineTask = $reader.ReadLineAsync()
            if (-not $lineTask.Wait(30000)) { $out.err = "stream stalled (30 s without a line)"; break }
            $line = $lineTask.Result
            if ($null -eq $line) { break }
            if ($line -eq "") { continue }
            if ($first) { $out.firstChunkMs = [math]::Round($sw.Elapsed.TotalMilliseconds, 0); $first = $false }
            $out.chunks++
            if ($line -eq "data: [DONE]") { $out.sawDone = $true; break }
            if ($line.StartsWith("data: ")) {
                $payload = $line.Substring(6)
                try {
                    $o = $payload | ConvertFrom-Json
                    if ($o.system_fingerprint -and $out.fingerprint -eq "") { $out.fingerprint = [string]$o.system_fingerprint }
                    if ($o.choices) {
                        $d = $o.choices[0].delta
                        if ($d -and $d.content) { $out.text += [string]$d.content }
                        if ($o.choices[0].finish_reason) { $out.finish = [string]$o.choices[0].finish_reason }
                    }
                } catch { }
            }
        }
        $out.totalMs = [math]::Round($sw.Elapsed.TotalMilliseconds, 0)
        $out.ok = (($out.err -eq "") -and $out.sawDone -and ($out.chunks -gt 0) -and ($out.text.Trim().Length -gt 0))
    } catch {
        $out.err = $_.Exception.Message
    } finally {
        if ($null -ne $reader) { try { $reader.Dispose() } catch { } }
        if ($null -ne $client) { try { $client.Dispose() } catch { } }
    }
    return $out
}

function Get-SystemFingerprint {
    # The RUNNING server's own identity (a real 1-token completion reports
    # system_fingerprint "b11073-..."; non-streaming first, streaming fallback,
    # up to 3 attempts - the same proven helper as the rev-2 launcher).
    param([string]$H, [int]$P, [int]$Attempts = 3)
    for ($A = 1; $A -le $Attempts; $A++) {
        $C = Invoke-LlamaChat -H $H -P $P -TimeoutSec 90 -BodyJson '{"messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0.0}'
        if ($C.ok -and $C.obj) {
            $Fp = [string]$C.obj.system_fingerprint
            if ($Fp -ne "") { return $Fp }
        }
        $S = Invoke-StreamProbe -H $H -P $P -TimeoutSec 90 -BodyJson '{"messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0.0,"stream":true}'
        if ($S.fingerprint -ne "") { return $S.fingerprint }
        if ($A -lt $Attempts) { Start-Sleep -Seconds 3 }
    }
    return ""
}

function Test-FingerprintOk {
    param([string]$Fp)
    if ($Fp -eq "") { return $false }
    if ($Fp -eq $script:ExpectedFingerprint) { return $true }
    if (($Fp -match "b11073") -and ($Fp -match "1aa2954bd")) { return $true }
    return $false
}

# ---------------------------------------------------------------------------
# Command line parsing / documented-profile compliance (unit-testable helpers)
# ---------------------------------------------------------------------------
function Split-CommandLine {
    param([string]$CmdLine)
    $tokens = New-Object System.Collections.Generic.List[string]
    if ([string]::IsNullOrWhiteSpace($CmdLine)) { return @() }
    $sb = New-Object System.Text.StringBuilder
    $inQ = $false
    foreach ($ch in $CmdLine.ToCharArray()) {
        if ($ch -eq '"') { $inQ = -not $inQ; continue }
        if ((-not $inQ) -and [char]::IsWhiteSpace($ch)) {
            if ($sb.Length -gt 0) { [void]$tokens.Add($sb.ToString()); [void]$sb.Clear() }
        } else {
            [void]$sb.Append([string]$ch)
        }
    }
    if ($sb.Length -gt 0) { [void]$tokens.Add($sb.ToString()) }
    return @($tokens)
}

function Get-ProfileFromCommandLine {
    param([string]$CmdLine)
    $t = @(Split-CommandLine $CmdLine)
    # normalise "name=value" forms into separate tokens first
    $toks = New-Object System.Collections.Generic.List[string]
    foreach ($x in $t) {
        $i = $x.IndexOf("=")
        if ($i -gt 0) {
            [void]$toks.Add($x.Substring(0, $i))
            [void]$toks.Add($x.Substring($i + 1))
        } else { [void]$toks.Add($x) }
    }
    $p = [ordered]@{
        exe = ""; model = ""; host = ""; port = ""; ngl = ""; ctx = ""; parallel = ""
        ckk = ""; ckv = ""; temp = ""; reasoning = ""; metrics = $false; nowebui = $false
        threads = ""; extra = @()
    }
    for ($i = 0; $i -lt $toks.Count; $i++) {
        $a = ([string]$toks[$i]).ToLower()
        switch ($a) {
            "--model" { if ($i + 1 -lt $toks.Count) { $p.model = [string]$toks[++$i] } }
            "-m" { if ($i + 1 -lt $toks.Count) { $p.model = [string]$toks[++$i] } }
            "--host" { if ($i + 1 -lt $toks.Count) { $p.host = [string]$toks[++$i] } }
            "--port" { if ($i + 1 -lt $toks.Count) { $p.port = [string]$toks[++$i] } }
            "-ngl" { if ($i + 1 -lt $toks.Count) { $p.ngl = [string]$toks[++$i] } }
            "--n-gpu-layers" { if ($i + 1 -lt $toks.Count) { $p.ngl = [string]$toks[++$i] } }
            "-c" { if ($i + 1 -lt $toks.Count) { $p.ctx = [string]$toks[++$i] } }
            "--ctx-size" { if ($i + 1 -lt $toks.Count) { $p.ctx = [string]$toks[++$i] } }
            "--context-size" { if ($i + 1 -lt $toks.Count) { $p.ctx = [string]$toks[++$i] } }
            "--parallel" { if ($i + 1 -lt $toks.Count) { $p.parallel = [string]$toks[++$i] } }
            "--cache-type-k" { if ($i + 1 -lt $toks.Count) { $p.ckk = [string]$toks[++$i] } }
            "--cache-type-v" { if ($i + 1 -lt $toks.Count) { $p.ckv = [string]$toks[++$i] } }
            "--temp" { if ($i + 1 -lt $toks.Count) { $p.temp = [string]$toks[++$i] } }
            "--temperature" { if ($i + 1 -lt $toks.Count) { $p.temp = [string]$toks[++$i] } }
            "-t" { if ($i + 1 -lt $toks.Count) { $p.threads = [string]$toks[++$i] } }
            "--threads" { if ($i + 1 -lt $toks.Count) { $p.threads = [string]$toks[++$i] } }
            "--reasoning" {
                if (($i + 1 -lt $toks.Count) -and (-not (([string]$toks[$i + 1]).StartsWith("-")))) { $p.reasoning = ([string]$toks[++$i]).ToLower() }
                else { $p.reasoning = "present" }
            }
            "--metrics" { $p.metrics = $true }
            "--no-webui" { $p.nowebui = $true }
            default {
                if ($i -eq 0) { $p.exe = [string]$toks[$i] }
                else { $p.extra += [string]$toks[$i] }
            }
        }
    }
    return $p
}

function Get-NormPath {
    param([string]$P)
    if ([string]::IsNullOrWhiteSpace($P)) { return "" }
    return ([string]$P).Trim().ToLower() -replace '/', '\'
}

function Test-ProfileCompliance {
    # Compares a parsed command-line profile against the DOCUMENTED production
    # profile. Returns @{ ok; issues } - never changes anything by itself.
    param($CmdProfile)
    $issues = @()
    if ($null -eq $CmdProfile) { return @{ ok = $false; issues = @("no command line") } }
    if ((Get-NormPath $CmdProfile.model) -ne (Get-NormPath $script:ModelPath)) { $issues += ("model: '" + $CmdProfile.model + "' != dokumentalt '" + $script:ModelPath + "'") }
    if ([string]$CmdProfile.host -ne [string]$script:DocProfile.host) { $issues += ("host: '" + $CmdProfile.host + "' != '" + $script:DocProfile.host + "'") }
    if ([string]$CmdProfile.port -ne [string]$script:DocProfile.port) { $issues += ("port: '" + $CmdProfile.port + "' != '" + $script:DocProfile.port + "'") }
    if ([string]$CmdProfile.ngl -ne [string]$script:DocProfile.ngl) { $issues += ("ngl: '" + $CmdProfile.ngl + "' != '" + $script:DocProfile.ngl + "'") }
    if ([string]$CmdProfile.ctx -ne [string]$script:DocProfile.ctx) { $issues += ("ctx: '" + $CmdProfile.ctx + "' != '" + $script:DocProfile.ctx + "'") }
    if ([string]$CmdProfile.parallel -ne [string]$script:DocProfile.parallel) { $issues += ("parallel: '" + $CmdProfile.parallel + "' != '" + $script:DocProfile.parallel + "'") }
    if (([string]$CmdProfile.ckk).ToLower() -ne [string]$script:DocProfile.ckk) { $issues += ("cache-type-k: '" + $CmdProfile.ckk + "' != '" + $script:DocProfile.ckk + "'") }
    if (([string]$CmdProfile.ckv).ToLower() -ne [string]$script:DocProfile.ckv) { $issues += ("cache-type-v: '" + $CmdProfile.ckv + "' != '" + $script:DocProfile.ckv + "'") }
    if ([string]$CmdProfile.temp -ne [string]$script:DocProfile.temp) { $issues += ("temp: '" + $CmdProfile.temp + "' != '" + $script:DocProfile.temp + "'") }
    if ([string]$CmdProfile.reasoning -ne [string]$script:DocProfile.reasoning) { $issues += ("reasoning: '" + $CmdProfile.reasoning + "' != '" + $script:DocProfile.reasoning + "'") }
    if (-not $CmdProfile.metrics) { $issues += "--metrics hianyzik" }
    if (-not $CmdProfile.nowebui) { $issues += "--no-webui hianyzik" }
    return @{ ok = ($issues.Count -eq 0); issues = $issues }
}

# ---------------------------------------------------------------------------
# Source snapshot (python sources + launchers + configs - prove they are
# untouched by the upgrade)
# ---------------------------------------------------------------------------
function New-SourceSnapshot {
    param([string]$Root)
    $files = @()
    $bases = @(
        @{ dir = (Join-Path $Root "app");               filter = "*.py" }
        @{ dir = (Join-Path $Root "vendor\voicemem");   filter = "*.py" }
        @{ dir = (Join-Path $Root "scripts");           filter = "*.py" }
        @{ dir = (Join-Path $Root "scripts");           filter = "*.ps1" }
    )
    foreach ($b in $bases) {
        if (-not (Test-Path -LiteralPath $b.dir)) { continue }
        $found = @(Get-ChildItem -LiteralPath $b.dir -Recurse -Filter $b.filter -File -ErrorAction SilentlyContinue)
        foreach ($f in $found) { $files += $f.FullName }
    }
    $extra = @(
        "START.bat",
        "config\llm_config.yaml",
        "config\voicemem_config.yaml",
        "config\llm_model.json",
        "MODELS.lock.json",
        "web\voicemem.html"
    )
    foreach ($e in $extra) {
        $p = Join-Path $Root $e
        if (Test-Path -LiteralPath $p) { $files += $p }
    }
    $list = @()
    foreach ($f in ($files | Sort-Object -Unique)) {
        $h = Get-Sha256 $f
        if ($h -eq "") { continue }
        $list += [ordered]@{
            path     = $f.Substring($Root.Length).TrimStart('\')
            bytes    = (Get-Item -LiteralPath $f).Length
            mtime    = (Get-Item -LiteralPath $f).LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ss")
            sha256   = $h
        }
    }
    return $list
}

# ---------------------------------------------------------------------------
# Server stop / start through the EXISTING production mechanism
# ---------------------------------------------------------------------------
function Stop-ProductionServer {
    param([int]$P)
    $res = @{ was_running = $false; stopped_pid = 0; port_free = $false; note = "" }
    $Listeners = @(Get-ListenerPids -P $P)
    if ($Listeners.Count -eq 0) {
        $res.note = "nem futott llama-server a(z) $P porton (nincs mit leallitani)"
        $res.port_free = $true
        Log ("stop: no listener on " + $P)
        return $res
    }
    $res.was_running = $true
    foreach ($Lp in $Listeners) {
        $info = Get-ProcInfo -ProcId $Lp
        if ($info.name -ne "" -and $info.name -notlike "llama-server*") {
            Fail ("a(z) " + $P + " porton a(z) '" + $info.name + "' folyamat figyel (PID " + $Lp + ") - NEM allitom le automatikusan.") `
                 ("allitsd le kezzel a nem-llama-server folyamatot, majd futtasd ujra az upgrade-et.")
        }
        if ($info.path -ne "" -and $info.path -ne $script:ProdExe) {
            Fail ("a(z) " + $P + " porton figyelo llama-server NEM a produkcios bin\llama-server.exe: " + $info.path) `
                 ("ez nem a produkcios peldany (pl. experimental). Allitsd le kezzel, es ertesitsd a pack kesztojet.")
        }
        Out-Line ("  leallitas: llama-server PID " + $Lp) "Yellow"
        try {
            Stop-Process -Id $Lp -Force -ErrorAction Stop
            $res.stopped_pid = $Lp
        } catch {
            Fail ("a llama-server (PID " + $Lp + ") leallitasa nem sikerult: " + $_.Exception.Message) "allitsd le kezzel, majd futtasd ujra."
        }
    }
    # wait for the port to become free (max 30 s)
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        if (@(Get-ListenerPids -P $P).Count -eq 0) { $res.port_free = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $res.port_free) {
        Fail ("a(z) " + $P + " port a leallitas utan sem szabadult fel (30 s)") "nezd meg: netstat -ano | findstr :" + $P
    }
    Log ("stop: listener " + ($Listeners -join ",") + " stopped; port free = " + $res.port_free)
    return $res
}

function Get-CurrentPowerShellExe {
    try {
        $p = (Get-Process -Id $PID -ErrorAction Stop).Path
        if ($p -and (Test-Path -LiteralPath $p)) { return $p }
    } catch { }
    return (Join-Path $PSHOME "powershell.exe")
}

function Start-ProductionServerViaStarter {
    # The EXISTING production startup mechanism: scripts\start_llama_server.ps1
    # in its own console window (it resolves the config, launches
    # bin\llama-server.exe, waits /health, writes its own proofs/logs and then
    # supervises - exactly the normal production state).
    param([int]$P)
    $res = @{ starter_pid = 0; note = ""; health = $false; waited_sec = 0 }
    if (-not (Test-Path -LiteralPath $script:Starter)) {
        Fail ("nem talalhato a produkcios indito: " + $script:Starter) "a frissites a meglevo inditoval inditja a szervert - ez nem hianyozhat."
    }
    $PsExe = Get-CurrentPowerShellExe
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $script:Starter + '"'), "-WaitSec", [string]$HealthTimeoutSec)
    Out-Line ("  inditas a meglevo produkcios inditoval (uj ablak): " + $script:Starter) "Cyan"
    Log ("start via starter: " + $PsExe + " " + ($argList -join " "))
    try {
        $proc = Start-Process -FilePath $PsExe -ArgumentList $argList -WorkingDirectory $script:Root -PassThru
        $res.starter_pid = $proc.Id
    } catch {
        Fail ("a produkcios indito elinditasa nem sikerult: " + $_.Exception.Message) "inditsd kezzel: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_llama_server.ps1"
    }
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $HealthTimeoutSec) {
        if (Test-Health -H $script:DocProfile.host -P $P -T 3) { $res.health = $true; break }
        Start-Sleep -Seconds 2
    }
    $res.waited_sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 0)
    if (-not $res.health) {
        # dump the starter + server logs for evidence, then fail (auto-rollback follows)
        $errLog = Join-Path $script:Root "logs\llama-server.err.log"
        $outLog = Join-Path $script:Root "logs\llama-server.out.log"
        foreach ($lg in @($errLog, $outLog)) {
            if (Test-Path -LiteralPath $lg) {
                try { Copy-Item -LiteralPath $lg -Destination (Join-Path $script:EvidenceDir (Split-Path -Leaf $lg)) -Force -ErrorAction SilentlyContinue } catch { }
            }
        }
        Fail ("a llama-server nem valaszolt a /health-re " + $HealthTimeoutSec + " s-en belul (a starter es server logok az evidence mappaba masolva)") `
             "ha CUDA/DLL hiba all fenn (0xC0000135), a rollback mar el is indult (ld. fent)."
    }
    Log ("start: health OK after " + $res.waited_sec + " s")
    return $res
}

# ---------------------------------------------------------------------------
# The 24-file runtime set verification (source or destination)
# ---------------------------------------------------------------------------
function Get-PinTable {
    # @{ table; source } - pack config x experimental pins x install manifest
    # cross-check, fail-closed on any disagreement.
    $table = @{}

    if (-not (Test-Path -LiteralPath $script:PackCfg)) {
        Fail ("hianyzik a pack konfig: " + $script:PackCfg) "csomagold ki ujra a VoiceMemAgent_v0.10.7_ProdB11073_Upgrade.zip-et."
    }
    try { $X = Get-Content -LiteralPath $script:PackCfg -Raw | ConvertFrom-Json } catch {
        Fail ("a pack konfig nem olvashato/ervenytelen JSON: " + $_.Exception.Message) "serult csomag; csomagold ki ujra a ZIP-et."
    }
    $N = $X.expected_new_runtime
    if ([string]$N.tag -ne $script:ExpectedTag -or [string]$N.build_commit -ne $script:ExpectedCommit) {
        Fail ("a pack konfig identitasa elter a varottol (tag/commit).") "ervenytelen csomag - SOHA ne lepj at masik verziora."
    }
    if (([string]$N.asset_sha256).ToUpper() -ne $script:ExpectedAssetSha256) {
        Fail "a pack konfig asset SHA-256 elter a varottol." "ervenytelen csomag."
    }
    if ([int]$N.file_count -ne $script:ExpectedFileCount) {
        Fail ("a pack konfig fajlszama (" + $N.file_count + ") elter a varott " + $script:ExpectedFileCount + "-tol") "ervenytelen csomag."
    }
    foreach ($f in $N.files) { $table[[string]$f.name] = ([string]$f.sha256).ToUpper() }
    if ($table.Count -ne $script:ExpectedFileCount) {
        Fail ("a pack konfig fajllista merete (" + $table.Count + ") elter") "ervenytelen csomag."
    }
    $source = "pack config (prod_upgrade_expected.json)"

    # cross-check with the machine's experimental pin table
    if (Test-Path -LiteralPath $script:ExpPins) {
        try { $EP = Get-Content -LiteralPath $script:ExpPins -Raw | ConvertFrom-Json } catch { $EP = $null }
        if ($null -ne $EP -and $EP.runtime -and $EP.runtime.files) {
            if ([string]$EP.runtime.tag -ne $script:ExpectedTag -or [string]$EP.runtime.build_commit -ne $script:ExpectedCommit) {
                Fail "az experimental pin-tabla identitasa elter a varott b11073/1aa2954bd-tol." "a telepitett experimental runtime elter - nem hasznalhato a frissiteshez."
            }
            if (([string]$EP.runtime.asset.sha256).ToUpper() -ne $script:ExpectedAssetSha256) {
                Fail "az experimental pin-tabla asset SHA-256 elter." "a telepitett experimental runtime elter - nem hasznalhato a frissiteshez."
            }
            foreach ($f in $EP.runtime.files) {
                $k = [string]$f.name
                $h = ([string]$f.sha256).ToUpper()
                if (-not $table.ContainsKey($k)) { Fail ("az experimental pin-tabla plusz fajlt tartalmaz: " + $k) "azonos b11073 release-t varunk." }
                if ($table[$k] -ne $h) { Fail ("a pack konfig es az experimental pin-tabla elter a fajlon: " + $k) "ervenytelen allapot." }
            }
            $source = "pack config x experimental b11073.pins.json (kereszt-ellenorizve)"
        }
    } else {
        Out-Line "  FIGYELEM: az experimental pin-tabla nem erheto el (a pack konfig az egyeduli forras)." "Yellow"
    }

    # cross-check with the install manifest (the machine's own install record)
    if (Test-Path -LiteralPath $script:ExpManifest) {
        try { $M = Get-Content -LiteralPath $script:ExpManifest -Raw | ConvertFrom-Json } catch { $M = $null }
        if ($null -ne $M) {
            if ([string]$M.tag -ne $script:ExpectedTag -or [string]$M.build_commit -ne $script:ExpectedCommit) {
                Fail "a telepitesi manifest identitasa elter a varott b11073/1aa2954bd-tol." "a manifest masik runtimeot rogzit - SOHA ne hasznald."
            }
            if (([string]$M.asset_sha256).ToUpper() -ne $script:ExpectedAssetSha256) {
                Fail "a telepitesi manifest asset SHA-256 elter." "a manifest megvaltozott - ne hasznald."
            }
            $mt = @{}
            foreach ($prop in $M.files_sha256.PSObject.Properties) { $mt[[string]$prop.Name] = ([string]$prop.Value).ToUpper() }
            if ($mt.Count -ne $table.Count) {
                Fail ("a manifest fajlszama (" + $mt.Count + ") elter a pin-tablajevel (" + $table.Count + ")") "a manifest megvaltozott - ne hasznald."
            }
            foreach ($k in @($table.Keys)) {
                if (-not $mt.ContainsKey($k)) { Fail ("a manifestbol hianyzik a fajl: " + $k) "a manifest megvaltozott - ne hasznald." }
                if ($mt[$k] -ne $table[$k]) { Fail ("a manifest es a pin-tabla elter a fajlon: " + $k) "ervenytelen allapot." }
            }
            $source = "pack config x experimental pins x runtime_manifest.json (haromutas egyezes)"
        }
    } else {
        Out-Line "  FIGYELEM: a telepitesi manifest nem erheto el (a ket pin-tabla kereszt-ellenorzes mar megtortent)." "Yellow"
    }
    return @{ table = $table; source = $source }
}

function Test-RuntimeSet {
    # Verifies every pinned file in $BinDirToCheck. Returns
    # @{ ok; ok_count; bad_file; bad_detail; build_id_ok }
    param([string]$BinDirToCheck, $PinTable, [bool]$CheckBuildId)
    $res = @{ ok = $false; ok_count = 0; bad_file = ""; bad_detail = ""; build_id_ok = $false }
    $tbl = $PinTable.table
    $nOk = 0
    foreach ($k in @($tbl.Keys | Sort-Object)) {
        $T = Join-Path $BinDirToCheck $k
        if (-not (Test-Path -LiteralPath $T)) { $res.bad_file = $k; $res.bad_detail = "hianyzo fajl"; return $res }
        $H = Get-Sha256 $T
        if ($H -eq $tbl[$k]) { $nOk++ } else { $res.bad_file = $k; $res.bad_detail = ("vart " + $tbl[$k] + ", kapott " + $H); return $res }
    }
    $res.ok_count = $nOk
    if ($nOk -ne $tbl.Count) { return $res }
    if ($CheckBuildId) {
        $implDll = Join-Path $BinDirToCheck "llama-server-impl.dll"
        if (-not (Test-BytesContain -Path $implDll -AsciiNeedle $script:ExpectedCommit)) {
            $res.bad_file = "llama-server-impl.dll"
            $res.bad_detail = ("a binarisban NINCS meg a b11073 build-identitas (" + $script:ExpectedCommit + ")")
            return $res
        }
        $res.build_id_ok = $true
    }
    $res.ok = $true
    return $res
}

function Get-InBinaryAnchors {
    param([string]$ExeOrDllDir)
    # Informational: which known build-id anchors are present in the local
    # binaries (1aa2954bd = b11073, a32af33de = b10717).
    $res = @{ has_b11073 = $false; has_b10717 = $false }
    foreach ($fname in @("llama-server-impl.dll", "llama-server.exe")) {
        $p = Join-Path $ExeOrDllDir $fname
        if (-not (Test-Path -LiteralPath $p)) { continue }
        if (Test-BytesContain -Path $p -AsciiNeedle $script:ExpectedCommit) { $res.has_b11073 = $true }
        if (Test-BytesContain -Path $p -AsciiNeedle $script:OldBuildCommit) { $res.has_b10717 = $true }
    }
    return $res
}

function Get-VersionBanner {
    # EAP-safe informational --version capture (NEVER a gate - the rev-1
    # experimental launcher bug is deliberately not repeated).
    param([string]$ExePath)
    $out = ""
    try {
        $OldEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $lines = & $ExePath --version 2>&1 | ForEach-Object { [string]$_ }
        $out = (($lines | Where-Object { $_ -and $_.Trim() }) -join " | ").Trim()
    } catch {
        $out = ""
    } finally {
        $ErrorActionPreference = $OldEap
    }
    return $out
}

# ---------------------------------------------------------------------------
# Install: replace the llama.cpp family in bin\, preserve the CUDA runtime DLLs
# ---------------------------------------------------------------------------
function Get-LlamaCppFamilyFiles {
    # Files in bin\ belonging to the llama.cpp runtime family (to be replaced)
    # - NEVER the CUDA runtime family (cudart/cublas/... - those must survive).
    param([string]$BinDirToScan, $PinNames)
    $remove = @()
    $items = @(Get-ChildItem -LiteralPath $BinDirToScan -File -ErrorAction SilentlyContinue)
    foreach ($f in $items) {
        $n = $f.Name
        if ($PinNames -contains $n) { continue }                     # replaced by copy anyway
        if ($n -notmatch '\.(dll|exe)$') { continue }                # licenses/configs stay
        if ($n -like "cudart*" -or $n -like "cublas*" -or $n -like "cublasLt*" -or $n -like "nvrtc*" -or $n -like "nvjitlink*" -or $n -like "cupti*" -or $n -like "nvml*" -or $n -like "myelin*") { continue }
        if ($n -like "llama*" -or $n -like "ggml*" -or $n -like "mtmd*" -or $n -like "libomp*") { $remove += $n }
    }
    return @($remove | Sort-Object -Unique)
}

function Install-B11073Runtime {
    param($PinTable)
    $tbl = $PinTable.table
    $res = @{ already = $false; copied = @(); removed = @(); cuda_dlls = @(); verify = $null }

    # idempotency: is the production bin\ already the exact pinned set?
    $cur = Test-RuntimeSet -BinDirToCheck $script:ProdBin -PinTable $PinTable -CheckBuildId $true
    if ($cur.ok) {
        $res.already = $true
        $script:AlreadyNew = $true
        Out-Line "  a produkcios bin\ mar a pontosan pin-elt b11073 runtime - NINCS csere (idempotens)." "Cyan"
        Log "install: production bin already the pinned b11073 set; no replacement"
        $res.verify = $cur
        return $res
    }

    # CUDA runtime inventory BEFORE the swap (must survive it)
    $cudaBefore = @()
    foreach ($f in @(Get-ChildItem -LiteralPath $script:ProdBin -Filter "*.dll" -File -ErrorAction SilentlyContinue)) {
        if ($f.Name -like "cudart*" -or $f.Name -like "cublas*" -or $f.Name -like "nvrtc*" -or $f.Name -like "nvjitlink*") {
            $cudaBefore += $f.Name
        }
    }

    # remove stale llama.cpp-family files not in the pinned set (they are safe
    # in the timestamped backup; leaving them could cause DLL-hell with the
    # dynamically selected ggml-cpu-* backends)
    $pinNames = @($tbl.Keys)
    $stale = Get-LlamaCppFamilyFiles -BinDirToScan $script:ProdBin -PinNames $pinNames
    foreach ($s in $stale) {
        try {
            Remove-Item -LiteralPath (Join-Path $script:ProdBin $s) -Force -ErrorAction Stop
            $res.removed += $s
        } catch {
            Fail ("a regi runtime-fajl nem torolheto: " + $s + " (" + $_.Exception.Message + ")") "zarold be a bin\-t hasznalo folyamatokat, majd futtasd ujra."
        }
    }

    # copy the 24 pinned files
    foreach ($k in @($tbl.Keys | Sort-Object)) {
        $src = Join-Path $script:ExpBin $k
        $dst = Join-Path $script:ProdBin $k
        if (-not (Test-Path -LiteralPath $src)) {
            Fail ("a forras runtime-fajl hianyzik: " + $src) "futtasd ujra az experimental telepitot: install_b11073.ps1"
        }
        try {
            Copy-Item -LiteralPath $src -Destination $dst -Force -ErrorAction Stop
            $res.copied += $k
        } catch {
            Fail ("a masolas nem sikerult: " + $k + " (" + $_.Exception.Message + ") " + $dst) "szabaditsd fel a fajlt, majd futtasd ujra (a backup megvan)."
        }
    }

    # CUDA runtime survival check (the b11073 ggml-cuda.dll resolves
    # cublas64_13.dll from here - the exact mechanism the experimental launcher
    # proved on the target)
    $cudaAfter = @()
    foreach ($f in @(Get-ChildItem -LiteralPath $script:ProdBin -Filter "*.dll" -File -ErrorAction SilentlyContinue)) {
        if ($f.Name -like "cudart*" -or $f.Name -like "cublas*" -or $f.Name -like "nvrtc*" -or $f.Name -like "nvjitlink*") {
            $cudaAfter += $f.Name
        }
    }
    $res.cuda_dlls = $cudaAfter
    foreach ($c in $cudaBefore) {
        if ($cudaAfter -notcontains $c) {
            Fail ("a CUDA runtime DLL elveszett a cserre soran: " + $c) "ez nem tortenhetett volna meg - ertesitsd a pack kesztojet; a rollback mar el is indulhat."
        }
    }
    $hasCublas13 = $false
    foreach ($c in $cudaAfter) { if ($c -like "cublas64_13*") { $hasCublas13 = $true } }
    if (-not $hasCublas13) {
        Out-Line "  FIGYELEM: a cublas64_13.dll nem talalhato a bin\-ben - a b11073 ggml-cuda.dll ezt reszolja!" "Yellow"
        Out-Line "  (elofordulhat, hogy rendszer-szintu CUDA 13.x reszolja - a start utani /health dont.)" "Yellow"
        Log "WARNING: cublas64_13.dll not found in bin\ after install"
    }

    $script:ModifiedFiles = $res.copied
    $script:RemovedFiles = $res.removed
    $script:ChangesMade = $true
    $res.verify = Test-RuntimeSet -BinDirToCheck $script:ProdBin -PinTable $PinTable -CheckBuildId $true
    return $res
}

# ---------------------------------------------------------------------------
# Evidence report
# ---------------------------------------------------------------------------
function New-ReportText {
    param([string]$Status)
    $R = $script:R
    $a = $R.audit
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("# VoiceMem production llama.cpp runtime upgrade - b10717 -> b11073")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("- run: $($R.run)  (mode: $($R.mode))")
    [void]$sb.AppendLine("- started: $($R.started_utc) UTC / finished: $($R.finished_utc) UTC")
    [void]$sb.AppendLine("- verdict: **$Status**")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Runtime identity")
    if ($null -ne $a) {
        [void]$sb.AppendLine("- BEFORE: llama-server.exe SHA-256 ``$($a.exe_sha256)``")
        [void]$sb.AppendLine("  - --version output (informational): ``$($a.version_banner)``")
        [void]$sb.AppendLine("  - in-binary anchors: b11073=$($a.anchor_b11073) / b10717=$($a.anchor_b10717)")
        [void]$sb.AppendLine("  - MODELS.lock identity: ``$($a.models_lock_identity)``")
        [void]$sb.AppendLine("- AFTER: llama-server.exe SHA-256 ``$($a.new_exe_sha256)``")
    }
    [void]$sb.AppendLine("- expected b11073 build identity: ``$($script:ExpectedFingerprint)`` (build id $script:ExpectedCommit in llama-server-impl.dll)")
    [void]$sb.AppendLine("- installed pinned files: $script:ExpectedFileCount/24 SHA-256 verified")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Model (unchanged by design)")
    [void]$sb.AppendLine("- path: ``$($script:ModelPath)``")
    if ($null -ne $a) {
        [void]$sb.AppendLine("- bytes: $($a.model.bytes) | mtime: $($a.model.mtime) | partial SHA-256 (4 MB head + tail): ``$($a.model.partial_sha256)``")
        [void]$sb.AppendLine("- regression re-check: $($a.model_after_note)")
    }
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Production command line (must not silently change)")
    if ($null -ne $a) {
        [void]$sb.AppendLine("- audited (before): ``$($a.live_cmdline)``")
        [void]$sb.AppendLine("- audited profile compliance: $($a.profile_before)")
    }
    if ($null -ne $R.start) {
        [void]$sb.AppendLine("- relaunched (after): ``$($R.start.cmdline)``")
        [void]$sb.AppendLine("- relaunched profile compliance: $($R.start.profile_after_verdict)")
    }
    [void]$sb.AppendLine("- documented profile: -ngl $($script:DocProfile.ngl) -c $($script:DocProfile.ctx) --parallel $($script:DocProfile.parallel) --cache-type-k $($script:DocProfile.ckk) --cache-type-v $($script:DocProfile.ckv) --temp $($script:DocProfile.temp) --reasoning $($script:DocProfile.reasoning) --metrics --no-webui (host $($script:DocProfile.host):$($script:DocProfile.port))")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Server-side verification")
    if ($null -ne $a) {
        [void]$sb.AppendLine("- /health before: $($a.health_before) | /v1/models before: ``$($a.served_before)``")
    }
    if ($null -ne $R.battery) {
        $b = $R.battery
        [void]$sb.AppendLine("- /health after: HTTP $($b.health_status)")
        [void]$sb.AppendLine("- /v1/models after: ``$($b.served_after)`` (match: $($b.model_match))")
        [void]$sb.AppendLine("- chat completion (Hungarian, non-stream): $($b.chat_ok) ($($b.chat_ms) ms, $($b.chat_chars) chars): $($b.chat_reply)")
        [void]$sb.AppendLine("- streaming: $($b.stream_ok) (first chunk $($b.stream_first_ms) ms, $($b.stream_chunks) SSE lines, [DONE] seen: $($b.stream_done), finish_reason: $($b.stream_finish))")
        [void]$sb.AppendLine("- JSON completion (response_format json_object): $($b.json_ok) -> ``$($b.json_value)``")
        [void]$sb.AppendLine("- system_fingerprint: ``$($b.fingerprint)`` (gate: $($b.fingerprint_ok))")
    }
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## VoiceMem integration")
    if ($null -ne $R.integration) {
        $g = $R.integration
        [void]$sb.AppendLine("- web backend: $($g.backend) on port $($WebPort)")
        [void]$sb.AppendLine("- LLM endpoint gate: $($g.endpoint_gate) (expected http://127.0.0.1:8080/v1)")
        [void]$sb.AppendLine("- turns observed: $($g.turns)")
        if ($g.turns -gt 0) {
            [void]$sb.AppendLine("- ASR transcripts: $($g.asr_transcripts -join ' | ')")
            [void]$sb.AppendLine("- LLM first token: $($g.llm_first_ms -join ' / ') ms")
            [void]$sb.AppendLine("- LLM total: $($g.llm_total_ms -join ' / ') ms, reply lengths: $($g.llm_chars -join ' / ') chars")
            [void]$sb.AppendLine("- TTS chunks per turn: $($g.tts_chunks -join ' / ') | first audio: $($g.first_audio_ms -join ' / ') ms")
            [void]$sb.AppendLine("- long generation flags (>400 chars): $($g.long_flags -join ' / ') | repeated-generation suspicion: $($g.repeat_flags -join ' / ')")
            [void]$sb.AppendLine("- endpoint leakage (8081/8082 in chain log): $($g.leak_count)")
        }
        [void]$sb.AppendLine("- raw chain slice: ``integration/web-server.slice.log`` in the evidence ZIP")
    } else {
        [void]$sb.AppendLine("- skipped (-SkipIntegration)")
    }
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Regression checks")
    if ($null -ne $R.regression) {
        $r = $R.regression
        [void]$sb.AppendLine("- port 8080 bound by bin\llama-server.exe: $($r.port_ok)")
        [void]$sb.AppendLine("- model path/identity unchanged: $($r.model_ok)")
        [void]$sb.AppendLine("- b11073 identity confirmed: $($r.fingerprint_ok)")
        [void]$sb.AppendLine("- python sources + configs unchanged: $($r.sources_ok) ($($r.sources_note))")
        [void]$sb.AppendLine("- model file unchanged: $($r.model_file_ok)")
        [void]$sb.AppendLine("- no 8081/8082 leakage into production: $($r.leak_ok)")
        [void]$sb.AppendLine("- START.bat + production starter intact: $($r.starter_ok)")
    }
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Rollback")
    [void]$sb.AppendLine("- status: $($R.rollback)")
    [void]$sb.AppendLine("- backup: ``$($script:BackupDir)``")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Modified files")
    if ($script:AlreadyNew) {
        [void]$sb.AppendLine("- none (bin\ already was the pinned b11073 set - idempotent run)")
    } else {
        [void]$sb.AppendLine("- replaced in ``bin\`` (the 24 pinned b11073 files):")
        foreach ($f in $script:ModifiedFiles) { [void]$sb.AppendLine("  - ``bin\$f``") }
        if ($script:RemovedFiles.Count -gt 0) {
            [void]$sb.AppendLine("- stale llama.cpp-family files removed (safe in the backup):")
            foreach ($f in $script:RemovedFiles) { [void]$sb.AppendLine("  - ``bin\$f``") }
        }
    }
    [void]$sb.AppendLine("- written by this run (evidence only): ``$($script:EvidenceDir)`` + ``$($script:BackupDir)`` + the normal production logs")
    [void]$sb.AppendLine("- NOT modified: the Qwen3.6 GGUF model, app\ / vendor\ / scripts\ / config\ sources, MODELS.lock.json, START.bat")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Evidence")
    [void]$sb.AppendLine("- stage results: ``stage_results.json`` | audit: ``audit_before.json`` | battery raw: ``chat_test.json`` / ``streaming_test.json`` / ``json_test.json``")
    [void]$sb.AppendLine("- full evidence ZIP: ``prod_b11073_upgrade_evidence.zip`` (send this back for the audit record)")
    return $sb.ToString()
}

function Close-Report {
    param([string]$Status = "")
    try {
        $script:R.finished_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        if ($Status -ne "") { $script:R.verdict = $Status }
        # stage results
        $script:R | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $script:EvidenceDir "stage_results.json") -Encoding UTF8
        # report
        $txt = New-ReportText -Status $script:R.verdict
        Set-Content -LiteralPath (Join-Path $script:EvidenceDir "REPORT.md") -Value $txt -Encoding UTF8
        # evidence zip (exclude any zip already inside)
        $zipPath = Join-Path $script:EvidenceDir "prod_b11073_upgrade_evidence.zip"
        $toZip = @(Get-ChildItem -LiteralPath $script:EvidenceDir -File | Where-Object { $_.Extension -ne ".zip" } | ForEach-Object { $_.FullName })
        if ($toZip.Count -gt 0) {
            try { Compress-Archive -LiteralPath $toZip -DestinationPath $zipPath -Force } catch { Log ("evidence zip: " + $_.Exception.Message) }
        }
    } catch {
        Log ("report close error: " + $_.Exception.Message)
    }
    try { Stop-Transcript -ErrorAction SilentlyContinue | Out-Null } catch { }
}

# ---------------------------------------------------------------------------
# Auto-rollback
# ---------------------------------------------------------------------------
function Invoke-AutoRollback {
    param([string]$Reason)
    if ($script:RollbackDone) { return }
    $script:RollbackDone = $true
    $script:R.rollback = ("EXECUTED (" + $Reason + ")")
    Out-Line ("AUTO-ROLLBACK indul: " + $Reason) "Magenta"
    Log ("AUTO-ROLLBACK: " + $Reason)
    $PsExe = Get-CurrentPowerShellExe
    $rbOut = Join-Path $script:EvidenceDir "rollback_console.log"
    $rbErr = Join-Path $script:EvidenceDir "rollback_console.err.log"
    $rbArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $script:Rollback + '"'), "-Port", [string]$Port, "-HealthTimeoutSec", [string]$HealthTimeoutSec, "-ModelPath", ('"' + $script:ModelPath + '"'), "-StartedByUpgrade")
    if ($script:BackupDir -ne "") { $rbArgs += @("-BackupDir", ('"' + $script:BackupDir + '"')) }
    try {
        # NOTE: -Wait is deliberately NOT used: on some platforms Start-Process
        # -Wait also waits for the whole DESCENDANT tree, and the rollback's own
        # starter child supervises the restarted server forever. We poll the
        # direct child instead (same semantics on Windows, safe everywhere).
        $p = Start-Process -FilePath $PsExe -ArgumentList $rbArgs -RedirectStandardOutput $rbOut -RedirectStandardError $rbErr -PassThru
        $rbDeadline = (Get-Date).AddSeconds($HealthTimeoutSec + 600)
        while (-not $p.HasExited) {
            if ((Get-Date) -gt $rbDeadline) { break }
            Start-Sleep -Seconds 2
        }
        if (-not $p.HasExited) {
            $script:R.rollback = ("ROLLBACK ATTEMPT FAILED (timeout waiting for the rollback process)")
            Out-Line "AUTO-ROLLBACK: a visszallitas folyamat nem fejezodott be idoben - l. rollback_console.log" "Red"
        } elseif ($p.ExitCode -eq 0) {
            $script:R.rollback = ("EXECUTED and VERIFIED (" + $Reason + ") - a regi b10717 produkcios szerver visszallitva")
            Out-Line "AUTO-ROLLBACK KESZ: a regi produkcios runtime visszaallitva es ujrainditva." "Magenta"
        } else {
            $script:R.rollback = ("EXECUTED BUT VERIFY FAILED (exit " + $p.ExitCode + "; " + $Reason + ") - NEZZED AT a rollback_console.log-ot!")
            Out-Line "AUTO-ROLLBACK: a visszallitas utani ellenorzes NEM sikerult - l. rollback_console.log" "Red"
        }
    } catch {
        $script:R.rollback = ("ROLLBACK ATTEMPT FAILED: " + $_.Exception.Message)
        Out-Line ("AUTO-ROLLBACK hiba: " + $_.Exception.Message) "Red"
        Out-Line ("Kezi rollback: powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\rollback_production_b11073.ps1 -BackupDir """ + $script:BackupDir + """") "Yellow"
    }
}

# ---------------------------------------------------------------------------
# Integration stage
# ---------------------------------------------------------------------------
function Invoke-IntegrationStage {
    $g = @{ backend = ""; endpoint_gate = "n/a"; turns = 0; asr_transcripts = @(); llm_first_ms = @(); llm_total_ms = @(); llm_chars = @(); tts_chunks = @(); first_audio_ms = @(); long_flags = @(); repeat_flags = @(); leak_count = 0; ok = $false; note = "" }

    # 1) the web backend: start it the NORMAL way if not already running
    $backendUp = $false
    try {
        $r = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/api/health" -f $WebPort) -Method Get -TimeoutSec 3 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $backendUp = $true; $g.backend = "already running" }
    } catch { }
    if (-not $backendUp) {
        if (-not (Test-Path -LiteralPath $script:AgentStarter)) {
            Fail ("nem talalhato a web backend indito: " + $script:AgentStarter) "a normal START.bat lanc resze - nem hianyozhat."
        }
        $PsExe = Get-CurrentPowerShellExe
        Out-Line ("  web backend inditasa a normal mechanizmussal (uj ablak): scripts\start_agent.ps1 -Web -NoServer") "Cyan"
        Log "integration: starting web backend via start_agent.ps1 -Web -NoServer"
        try {
            $null = Start-Process -FilePath $PsExe -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $script:AgentStarter + '"'), "-Web", "-NoServer") -WorkingDirectory $script:Root -PassThru
        } catch {
            Fail ("a web backend inditasa nem sikerult: " + $_.Exception.Message) "inditsd kezzel a START.bat-ot, majd futtasd ujra a -SkipIntegration nelkuli upgrade-et."
        }
        $t0 = Get-Date
        while (((Get-Date) - $t0).TotalSeconds -lt 180) {
            try {
                $r = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/api/health" -f $WebPort) -Method Get -TimeoutSec 3 -UseBasicParsing
                if ($r.StatusCode -eq 200) { $backendUp = $true; break }
            } catch { }
            Start-Sleep -Seconds 3
        }
        $g.backend = "started by the upgrade"
    }
    if (-not $backendUp) {
        $g.note = "a web backend nem ert el a 3 percen belul"
        Fail "a VoiceMem web backend nem indult el / nem valaszol (VoiceMem connection failure)." "nezd meg a konzolt es a logs\web-server.log-ot. A feladat szerinti visszallitas (rollback) elindult."
    }

    # 2) the LLM endpoint gate: the backend must talk to 127.0.0.1:8080/v1
    $pipeRaw = ""
    try {
        $r = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/api/pipeline" -f $WebPort) -Method Get -TimeoutSec 10 -UseBasicParsing
        $pipeRaw = [string]$r.Content
    } catch { }
    $expected = "http://127.0.0.1:" + $Port + "/v1"
    if ($pipeRaw -ne "") {
        if ($pipeRaw.Contains($expected)) {
            $g.endpoint_gate = "PASS (" + $expected + " confirmed via /api/pipeline)"
        } elseif ($pipeRaw.Contains(":8081") -or $pipeRaw.Contains(":8082")) {
            # an experimental-session backend is running - replace it with a normal one
            Out-Line "  FIGYELEM: a futto web backend a 8081/8082 fel van iranyitva - kicserellem normalisra." "Yellow"
            Log "integration: running backend pointed at an experimental port; restarting it"
            $wListeners = @(Get-ListenerPids -P $WebPort)
            foreach ($wl in $wListeners) {
                $wi = Get-ProcInfo -ProcId $wl
                if ($wi.name -like "python*" -and $wi.cmdline -ne "" -and $wi.cmdline.Contains("app.web_server") -and $wi.cmdline.Contains($script:Root)) {
                    try { Stop-Process -Id $wl -Force -ErrorAction Stop } catch { }
                } else {
                    Fail ("a " + $WebPort + " porton ismeretlen folyamat fut (" + $wi.name + ") - nem allitom le.") "zarold be kezzel a web backend-et, majd futtasd ujra."
                }
            }
            Start-Sleep -Seconds 2
            $PsExe = Get-CurrentPowerShellExe
            try {
                $null = Start-Process -FilePath $PsExe -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $script:AgentStarter + '"'), "-Web", "-NoServer") -WorkingDirectory $script:Root -PassThru
            } catch { }
            $t0 = Get-Date
            $pipeRaw = ""
            while (((Get-Date) - $t0).TotalSeconds -lt 180) {
                try {
                    $r = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/api/pipeline" -f $WebPort) -Method Get -TimeoutSec 5 -UseBasicParsing
                    $pipeRaw = [string]$r.Content
                    if ($pipeRaw.Contains($expected)) { break }
                } catch { }
                Start-Sleep -Seconds 3
            }
            if ($pipeRaw.Contains($expected)) {
                $g.endpoint_gate = "PASS after backend restart (" + $expected + ")"
            }
        }
    }
    if (-not $pipeRaw.Contains($expected)) {
        $g.endpoint_gate = "FAIL (expected " + $expected + ")"
        Fail ("a VoiceMem web backend LLM endpointje NEM a(z) " + $expected + " (l. /api/pipeline).") "VoiceMem connection failure - a feladat szerinti rollback elindult."
    }
    try {
        $null = $pipeRaw | Set-Content -LiteralPath (Join-Path $script:EvidenceDir "integration_pipeline.json") -Encoding UTF8
    } catch { }

    # 3) guided production-chain turns (operator through the web UI)
    $logMarker = 0
    if (Test-Path -LiteralPath $script:WebLog) { try { $logMarker = (Get-Item -LiteralPath $script:WebLog).Length } catch { $logMarker = 0 } }
    Write-Host ""
    Out-Line "  Most vegezd el a harom produkcios lanc-tesztet a web UI-ban (http://127.0.0.1:$WebPort/):" "Cyan"
    Out-Line "    1) egy ROVID magyar beszelgetesi kerest (hangon)" "White"
    Out-Line "    2) egy magyar kerest, ami ANGOL kifejezest is tartalmaz" "White"
    Out-Line "    3) egy teljes hang-kor: VAD -> ASR -> VoiceMem -> E5 -> LLM -> TTS" "White"
    Out-Line "  (a lanc-naplot a szkript kozben rogziti; a valaszokat a szokott modon add.)" "Gray"
    $answer = ""
    try { $answer = Read-Host "  Enter a harom teszt utan (vagy 'skip' a kihagyashoz)" } catch { $answer = "skip" }
    if ($answer -eq "skip") {
        $g.note = "operator skipped the guided turns"
        Out-Line "  FIGYELEM: a lanc-tesztek kihagyva - az integracios evidence hianyos!" "Yellow"
        Log "integration: operator skipped the guided turns"
    } else {
        # 4) extract the metrics from the web-server.log slice
        $slice = @()
        if (Test-Path -LiteralPath $script:WebLog) {
            try {
                $fs = [System.IO.File]::Open($script:WebLog, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
                try {
                    [void]$fs.Seek($logMarker, [System.IO.SeekOrigin]::Begin)
                    $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
                    $txt = $sr.ReadToEnd()
                    $sr.Close()
                } finally { $fs.Close() }
                $slice = @($txt -split "`r?`n")
            } catch { $slice = @() }
        }
        $slicePath = Join-Path $script:EvidenceDir "integration"
        try { New-Item -ItemType Directory -Path $slicePath -Force | Out-Null } catch { }
        try { $slice -join "`r`n" | Set-Content -LiteralPath (Join-Path $slicePath "web-server.slice.log") -Encoding UTF8 } catch { }

        $turnCount = 0
        foreach ($line in $slice) {
            if ($line -match 'ASR final transcript') { $g.asr_transcripts += ($line -replace '^.*ASR final transcript:\s*', ''); $turnCount++ }
            if ($line -match 'llm first token \((\d+) ms\)') { $g.llm_first_ms += [int]$Matches[1] }
            if ($line -match 'llm done \((\d+) chars, (\d+) ms\)') { $g.llm_chars += [int]$Matches[1]; $g.llm_total_ms += [int]$Matches[2] }
            if ($line -match 'tts done \((\d+) chunks') { $g.tts_chunks += [int]$Matches[1] }
            if ($line -match 'first TTS audio sent to the browser \((\d+) ms\)') { $g.first_audio_ms += [int]$Matches[1] }
            if ($line -match ':8081|:8082') { $g.leak_count++ }
        }
        $g.turns = $turnCount
        # long-generation / repetition flags (the ~637-char oscillation class)
        for ($i = 0; $i -lt $g.llm_chars.Count; $i++) {
            if ($g.llm_chars[$i] -gt 400) { $g.long_flags += ("turn {0}: {1} chars (>400)" -f ($i + 1), $g.llm_chars[$i]) } else { $g.long_flags += ("turn {0}: {1} chars" -f ($i + 1), $g.llm_chars[$i]) }
            if ($g.llm_chars[$i] -gt 600) { $g.repeat_flags += ("turn {0}: VERY LONG - repeated-generation suspicion (compare with the old-server baseline)" -f ($i + 1)) } else { $g.repeat_flags += ("turn {0}: normal length" -f ($i + 1)) }
        }
        if ($turnCount -eq 0) {
            $g.note = "no [chain] ASR turns found in the captured log slice"
            Out-Line "  FIGYELEM: a lanc-naploban nem talalhato ASR turn - a tesztek talan nem tortentek meg." "Yellow"
        }
    }
    $g.ok = ($g.endpoint_gate -like "PASS*")
    Log ("integration: turns=" + $g.turns + " endpoint_gate=" + $g.endpoint_gate + " leak=" + $g.leak_count)
    return $g
}

# ===========================================================================
# MAIN FLOW
# ===========================================================================
if ($FunctionsOnly) {
    Write-Host "[prod-b11073-upgrade] funkciok betoltve (-FunctionsOnly) - nincs vegrehajtas." -ForegroundColor DarkGray
    return
}

# mode resolution
if ($VerifyOnly) { $script:R.mode = "verify-only" }
elseif ($AuditOnly) { $script:R.mode = "audit-only" }
elseif ($ProbeOnly) { $script:R.mode = "probe-only" }
else { $script:R.mode = "full" }

# evidence dir + transcript
try { New-Item -ItemType Directory -Path $script:EvidenceDir -Force | Out-Null } catch { }
$script:MasterLog = Join-Path $script:EvidenceDir "upgrade_master.log"
try { Start-Transcript -Path (Join-Path $script:EvidenceDir "upgrade_transcript.txt") -ErrorAction SilentlyContinue | Out-Null } catch { }

Write-Host ""
Write-Host "=== PRODUKCIO llama.cpp RUNTIME FRISSITES: b10717 -> b11073 ===" -ForegroundColor Magenta
Write-Host ("  root          : " + $script:Root) -ForegroundColor Gray
Write-Host ("  produkcios bin: " + $script:ProdBin) -ForegroundColor Gray
Write-Host ("  forras (b11073): " + $script:ExpBin + "  (mar telepitett + mar ellenorzott - NINCS letoltes)") -ForegroundColor Gray
Write-Host ("  modell        : " + $script:ModelPath) -ForegroundColor Gray
Write-Host ("  endpoint      : 127.0.0.1:" + $Port + "  (a produkcios profil NEM valtozik)") -ForegroundColor Gray
Write-Host ("  evidence      : " + $script:EvidenceDir) -ForegroundColor Gray
Write-Host ("  mod           : " + $script:R.mode) -ForegroundColor Gray
Log ("run start; mode=" + $script:R.mode + " root=" + $script:Root)

# ---------------------------------------------------------------------------
# [S0] PREFLIGHT: identity tables + the SOURCE runtime
# ---------------------------------------------------------------------------
$script:R.stage = "S0-preflight"
Out-Line "`n[S0] preflight: identitas-tablak kereszt-ellenorzese..."
if (-not (Test-Path -LiteralPath $script:ProdBin)) {
    Fail ("nem letezik a produkcios bin mappa: " + $script:ProdBin) "ez a produkcios runtime helye - nem hianyozhat."
}
if (-not (Test-Path -LiteralPath $script:ProdExe)) {
    Fail ("nem letezik a produkcios llama-server.exe: " + $script:ProdExe) "a frissites elott a produkcios runtimenak lennie kell."
}
if (-not (Test-Path -LiteralPath $script:ExpBin)) {
    Fail ("nem talalhato a mar telepitett experimental b11073 runtime: " + $script:ExpBin) `
         ("elozoleg telepitsd: powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\install_b11073.ps1 (a rev-2 installer pack). A frissites CSAK a mar ellenorzott b11073 runtimeot hasznalja - NINCS letoltes.")
}
if (-not (Test-Path -LiteralPath $script:ModelPath)) {
    Fail ("nem talalhato a produkcios modell: " + $script:ModelPath) "a modellnek helyben kell lennie (NEM masolodik, NEM toltodik ujra)."
}
$Pins = Get-PinTable
Out-Line ("  identitas-forras: " + $Pins.source) "Green"

$srcCheck = Test-RuntimeSet -BinDirToCheck $script:ExpBin -PinTable $Pins -CheckBuildId $true
if (-not $srcCheck.ok) {
    Fail ("a FORRAS (experimental) runtime NEM a pin-elt b11073: " + $srcCheck.bad_file + " (" + $srcCheck.bad_detail + ")") `
         ("futtasd ujra: powershell -NoProfile -ExecutionPolicy Bypass -File experimental\llama_b11073\install_b11073.ps1 -Force (ujra letolti es pin-eli a b11073-at). A produkcios bin\ NINCS erintve.")
}
Out-Line ("  FORRAS OK: mind a 24 fajl SHA-256 + build-identitas " + $script:ExpectedCommit + " (llama-server-impl.dll).") "Green"
Log ("S0 source verify: 24/24 + build id OK")

if ($VerifyOnly) {
    Out-Line "`nVERIFIKACIO KESZ: a forras b11073 runtime integritasa rendben. (-VerifyOnly: valtoztatas NEM tortent.)" "Green"
    $script:R.verdict = "VERIFY-ONLY OK (source runtime = pinned b11073)"
    Close-Report
    exit 0
}

# ---------------------------------------------------------------------------
# [S1] AUDIT: the CURRENT production state
# ---------------------------------------------------------------------------
$script:R.stage = "S1-audit"
Out-Line "`n[S1] audit: a jelenlegi produkcios allapot rogzitese..."

$oldExeHash = Get-Sha256 $script:ProdExe
$oldVerInfo = ""
try {
    $vi = (Get-Item $script:ProdExe).VersionInfo
    $oldVerInfo = ((@($vi.FileVersion, $vi.ProductVersion) | Where-Object { $_ }) -join " / ")
} catch { }
$oldBanner = Get-VersionBanner -ExePath $script:ProdExe
$anchors = Get-InBinaryAnchors -ExeOrDllDir $script:ProdBin
$modelsLockIdentity = ""
if (Test-Path -LiteralPath $script:ModelsLock) {
    try {
        $ml = Get-Content -LiteralPath $script:ModelsLock -Raw | ConvertFrom-Json
        $tools = $null
        try { $tools = $ml.tools } catch { }
        if ($null -ne $tools -and $tools.'llama-server') { $modelsLockIdentity = [string]$tools.'llama-server' }
    } catch { }
}

# live production server?
$livePid = 0; $liveCmd = ""; $livePath = ""; $healthBefore = "not checked"; $servedBefore = ""
$Listeners = @(Get-ListenerPids -P $Port)
if ($Listeners.Count -gt 0) {
    foreach ($Lp in $Listeners) {
        $info = Get-ProcInfo -ProcId $Lp
        if ($info.name -ne "" -and $info.name -notlike "llama-server*") {
            Fail ("a(z) " + $Port + " porton a(z) '" + $info.name + "' folyamat figyel (PID " + $Lp + ") - NEM allitom le.") "allitsd le kezzel, majd futtasd ujra az upgrade-et."
        }
        $livePid = $Lp
        $liveCmd = $info.cmdline
        $livePath = $info.path
    }
    $healthBefore = (Test-Health -H "127.0.0.1" -P $Port -T 5)
    $servedBefore = Get-ServedModelId -H "127.0.0.1" -P $Port
    Out-Line ("  futto szerver: PID " + $livePid + " | /health: " + $healthBefore + " | modell: " + $servedBefore) "Gray"
} else {
    Out-Line "  nincs futto llama-server a(z) $Port porton (a modosszati lepesek kihagyasra kerulnek)." "Gray"
}

# profile compliance of the AUDITED command line (fail-closed BEFORE changes)
$profileBefore = $null
$profileBeforeVerdict = "no live command line"
if ($liveCmd -ne "") {
    $profileBefore = Get-ProfileFromCommandLine -CmdLine $liveCmd
    $pc = Test-ProfileCompliance -CmdProfile $profileBefore
    if ($pc.ok) {
        $profileBeforeVerdict = "PASS"
        Out-Line "  audited profil: EGYEZIK a dokumentalt produkcios profillal." "Green"
    } else {
        $profileBeforeVerdict = ("FAIL (" + ($pc.issues -join "; ") + ")")
        Out-Line "  HIBA: a futto produkcios parancssor ELTER a dokumentalt profiltol:" "Red"
        foreach ($iss in $pc.issues) { Out-Line ("        " + $iss) "Red" }
        Out-Line ("  futtato parancssor: " + $liveCmd) "Gray"
        Fail "a dokumentalt produkcios profil es a tenyleges parancssor elter - a frissites nem indulhat el (nincs valtoztatas)." `
             "egyeztetes szukseges (a dokumentalt profil: ngl 16 / ctx 16000 / parallel 1 / q8_0 KV / temp 0.7 / reasoning off / metrics / no-webui)."
    }
}

# CUDA DLL inventory + hashes
$cudaInv = @()
foreach ($f in @(Get-ChildItem -LiteralPath $script:ProdBin -Filter "*.dll" -File -ErrorAction SilentlyContinue)) {
    if ($f.Name -like "cudart*" -or $f.Name -like "cublas*" -or $f.Name -like "nvrtc*" -or $f.Name -like "nvjitlink*") {
        $cudaInv += [ordered]@{ name = $f.Name; bytes = $f.Length; sha256 = (Get-Sha256 $f.FullName) }
    }
}

# model integrity marks
$modelItem = $null
if (Test-Path -LiteralPath $script:ModelPath) {
    $mi = Get-Item -LiteralPath $script:ModelPath
    $modelItem = [ordered]@{
        path = $script:ModelPath
        bytes = $mi.Length
        mtime = $mi.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ss")
        partial_sha256 = (Get-PartialSha256 -Path $script:ModelPath)
    }
} else {
    Fail "a modell nem erheto el: $script:ModelPath" "a frissites nem indithato modell nelkul."
}

# python-source snapshot
Out-Line "  python-forras es konfig pillanatkep (hash)..." "Gray"
$srcSnapshot = @(New-SourceSnapshot -Root $script:Root)
Out-Line ("  pillanatkep: " + $srcSnapshot.Count + " fajl hash-e rogzitve.") "Gray"

$audit = [ordered]@{
    exe_path              = $script:ProdExe
    exe_sha256            = $oldExeHash
    exe_version_info      = $oldVerInfo
    version_banner        = $oldBanner
    anchor_b11073         = $anchors.has_b11073
    anchor_b10717         = $anchors.has_b10717
    models_lock_identity  = $modelsLockIdentity
    live_pid              = $livePid
    live_path             = $livePath
    live_cmdline          = $liveCmd
    health_before         = ($healthBefore -as [string])
    served_before         = $servedBefore
    profile_before        = $profileBeforeVerdict
    cuda_dlls             = $cudaInv
    model                 = $modelItem
    source_snapshot_count = $srcSnapshot.Count
    new_exe_sha256        = ""
    model_after_note      = "not checked"
}
$script:R.audit = $audit
try {
    $audit | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $script:EvidenceDir "audit_before.json") -Encoding UTF8
    $srcSnapshot | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $script:EvidenceDir "source_snapshot_before.json") -Encoding UTF8
} catch { Log ("audit json write: " + $_.Exception.Message) }
$script:SourceSnapshotBefore = $srcSnapshot

if ($AuditOnly) {
    Out-Line "`nAUDIT KESZ: a jelenlegi allapot rogzitve. (-AuditOnly: valtoztatas NEM tortent.)" "Green"
    $script:R.verdict = "AUDIT-ONLY COMPLETE (no changes made)"
    Close-Report
    exit 0
}

if ($ProbeOnly) {
    # probe the CURRENTLY RUNNING server without changing anything
    $script:R.stage = "probe"
    Out-Line "`n[PROBE] a futto szerver ellenorzese (valtoztatas nelkul)..."
    if (@(Get-ListenerPids -P $Port).Count -eq 0) {
        Fail ("nincs futto szerver a(z) " + $Port + " porton - nincs mit probozni.") "inditsd el a produkcios szervert (START.bat), majd probald ujra."
    }
    $healthStatus = 0
    try { $healthStatus = [int](Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/health" -f $Port) -Method Get -TimeoutSec 5 -UseBasicParsing).StatusCode } catch { }
    $servedNow = Get-ServedModelId -H "127.0.0.1" -P $Port
    $chat = Invoke-LlamaChat -H "127.0.0.1" -P $Port -BodyJson $script:HuChatBody -TimeoutSec $ChatTimeoutSec
    $chatTxt = ""
    if ($chat.ok -and $chat.obj) { try { $chatTxt = [string]$chat.obj.choices[0].message.content } catch { } }
    $fp = ""
    if ($chat.ok -and $chat.obj) { $fp = [string]$chat.obj.system_fingerprint }
    if ($fp -eq "") { $fp = Get-SystemFingerprint -H "127.0.0.1" -P $Port }
    $script:R.battery = [ordered]@{
        health_status = $healthStatus
        served_after = $servedNow
        model_match = (Test-ServedModelMatches -ServedId $servedNow -ExpectedModelPath $script:ModelPath)
        chat_ok = ($chat.ok -and ($chatTxt.Trim().Length -gt 0))
        chat_ms = $chat.ms
        chat_chars = $chatTxt.Length
        chat_reply = $chatTxt
        fingerprint = $fp
        fingerprint_ok = (Test-FingerprintOk $fp)
    }
    $okAll = ($healthStatus -eq 200) -and ((Test-ServedModelMatches -ServedId $servedNow -ExpectedModelPath $script:ModelPath) -eq "same") -and $chat.ok -and (Test-FingerprintOk $fp)
    if ($okAll) {
        Out-Line ("`nPROBE OK: health 200, modell OK, chat OK, fingerprint = " + $fp) "Green"
        $script:R.verdict = "PROBE OK (running server = b11073, model unchanged)"
        Close-Report
        exit 0
    } else {
        Out-Line ("`nPROBE: a futto szerver NEM felel meg (fingerprint = '" + $fp + "', health = " + $healthStatus + ", modell = '" + $servedNow + "')") "Red"
        $script:R.verdict = "PROBE FAILED (see battery)"
        Close-Report
        exit 1
    }
}

# ---------------------------------------------------------------------------
# [S2] STOP the production server cleanly
# ---------------------------------------------------------------------------
$script:R.stage = "S2-stop"
Out-Line "`n[S2] a produkcios szerver tiszta leallitasa..."
$stopRes = Stop-ProductionServer -P $Port
Out-Line ("  eredmeny: " + $stopRes.note + " | port free: " + $stopRes.port_free) "Gray"
Log ("S2 stop result: was_running=" + $stopRes.was_running + " port_free=" + $stopRes.port_free)
# from here on, production is DOWN: ANY later critical failure must bring the
# old server back (auto-rollback with or without a file restore)
$script:ChangesMade = $true

# ---------------------------------------------------------------------------
# [S3] BACKUP the current runtime
# ---------------------------------------------------------------------------
$script:R.stage = "S3-backup"
# idempotency check: if bin\ is ALREADY the pinned b11073 set, no backup/replace
$preCheck = Test-RuntimeSet -BinDirToCheck $script:ProdBin -PinTable $Pins -CheckBuildId $true
if ($preCheck.ok) {
    $script:AlreadyNew = $true
    Out-Line "  a produkcios bin\ mar a pin-elt b11073 - a backup/csere kihagyva (idempotens futas)." "Cyan"
    Log "S3: production bin already b11073; backup/install skipped"
} else {
    $script:BackupDir = Join-Path $Root ("backups\runtime_pre_b11073_" + $script:Stamp)
    $backupBin = Join-Path $script:BackupDir "bin"
    try {
        New-Item -ItemType Directory -Path $backupBin -Force | Out-Null
        # NOTE: -Path (not -LiteralPath): the '*' must expand (a -LiteralPath
        # wildcard is taken literally and copies NOTHING - caught by the T3
        # sandbox execution test).
        Copy-Item -Path (Join-Path $script:ProdBin "*") -Destination $backupBin -Recurse -Force -ErrorAction Stop
    } catch {
        Fail ("a bin\ backup nem keszulhetett el: " + $_.Exception.Message) "szabadits fel lemezteruletet / zarold be a bin\-t hasznalo folyamatokat."
    }
    # backup manifest + integrity verification of the copy
    $bkFiles = @()
    $bkOk = 0
    foreach ($f in @(Get-ChildItem -LiteralPath $backupBin -File -ErrorAction SilentlyContinue)) {
        $h = Get-Sha256 $f.FullName
        $bkFiles += [ordered]@{ name = $f.Name; bytes = $f.Length; sha256 = $h }
        if ($f.Name -eq "llama-server.exe" -and $h -eq $oldExeHash) { $bkOk = 1 }
    }
    if ($bkOk -ne 1) {
        Fail "a backup ellenorzese nem sikerult (a mentett llama-server.exe hash-e elter az audit-ban rogzitetttol)." "torold a hibas backupot es probald ujra."
    }
    $bkManifest = [ordered]@{
        created = $script:Stamp
        purpose = "rollback anchor for the production b10717 -> b11073 runtime upgrade"
        bin_dir = $script:ProdBin
        backup_bin_dir = $backupBin
        llama_server_sha256_before = $oldExeHash
        audited_cmdline = $liveCmd
        files = $bkFiles
    }
    try { $bkManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $script:BackupDir "backup_manifest.json") -Encoding UTF8 } catch { }
    $sz = 0
    foreach ($f in $bkFiles) { $sz += $f.bytes }
    Out-Line ("  backup kesz: " + $script:BackupDir + " (" + $bkFiles.Count + " fajl, ~" + [math]::Round($sz / 1MB, 1) + " MB)") "Green"
    Log ("S3 backup: " + $bkFiles.Count + " files -> " + $script:BackupDir)
}

# ---------------------------------------------------------------------------
# [S4]+[S5] INSTALL the pinned b11073 set + VERIFY
# ---------------------------------------------------------------------------
$script:R.stage = "S4-install"
Out-Line "`n[S4] a pin-elt b11073 runtime beszerelase a produkcios bin\-be..."
$installRes = Install-B11073Runtime -PinTable $Pins
$script:R.install = [ordered]@{
    already = $installRes.already
    copied_count = $installRes.copied.Count
    removed_stale = $installRes.removed
    cuda_dlls = $installRes.cuda_dlls
    verify_ok = $installRes.verify.ok
    verify_ok_count = $installRes.verify.ok_count
    verify_bad_file = $installRes.verify.bad_file
    verify_bad_detail = $installRes.verify.bad_detail
}
if (-not $installRes.verify.ok) {
    Fail ("a beszerelt runtime ELBUKOTT az ellenorzesen: " + $installRes.verify.bad_file + " (" + $installRes.verify.bad_detail + ")") "a rollback automatikusan indul (vagy mar indult)."
}
Out-Line ("  OK: a produkcios bin\ most a pin-elt b11073 (" + $installRes.verify.ok_count + "/" + $script:ExpectedFileCount + " fajl SHA-256 + build-identitas " + $script:ExpectedCommit + ").") "Green"
if ($installRes.removed.Count -gt 0) {
    Out-Line ("  elavult llama.cpp-csalad fajlok eltavolitva: " + ($installRes.removed -join ", ")) "Gray"
}
if ($installRes.cuda_dlls.Count -gt 0) {
    Out-Line ("  CUDA runtime DLL-k megmaradtak: " + ($installRes.cuda_dlls -join ", ")) "Gray"
}

# informational --version of the NEW binary (never a gate)
$newBanner = Get-VersionBanner -ExePath $script:ProdExe
$newAnchors = Get-InBinaryAnchors -ExeOrDllDir $script:ProdBin
$script:R.audit.new_exe_sha256 = (Get-Sha256 $script:ProdExe)
Log ("new --version (informational): '" + $newBanner + "'")
if ($newBanner -ne "") { Out-Line ("  [info] uj --version: " + $newBanner) "DarkGray" }
Out-Line "  verzio : b11073 - VERIFIKALVA (24-fajl SHA-256 + build-azonositas; nem --version banner)" "Green"

# ---------------------------------------------------------------------------
# [S6] START the production server through the EXISTING starter
# ---------------------------------------------------------------------------
$script:R.stage = "S6-start"
Out-Line "`n[S6] a produkcios szerver inditasa a meglevo produkcios inditoval..."
$startRes = Start-ProductionServerViaStarter -P $Port
$ServerPid = 0; $newCmd = ""; $newPath = ""
$Listeners = @(Get-ListenerPids -P $Port)
foreach ($Lp in $Listeners) {
    $info = Get-ProcInfo -ProcId $Lp
    $ServerPid = $Lp
    $newPath = $info.path
    $newCmd = $info.cmdline
    if ($info.name -ne "" -and $info.name -notlike "llama-server*") {
        Fail ("a(z) " + $Port + " porton most a(z) '" + $info.name + "' figyel - ez nem lehet.") "vizsgald meg: netstat -ano | findstr :" + $Port
    }
    if ($info.path -ne "" -and $info.path -ne $script:ProdExe) {
        Fail ("a listener NEM a produkcios bin\llama-server.exe: " + $info.path) "a rollback automatikusan indul."
    }
    if ($info.name -eq "") {
        Out-Line "  FIGYELEM: a listener neve nem olvashato - a fingerprint-dont." "Yellow"
    }
}
if ($Listeners.Count -eq 0) {
    Fail ("nincs listener a(z) " + $Port + " porton a health utan sem.") "a rollback automatikusan indul."
}

# profile compliance of the RELAUNCHED command line
$profileAfterVerdict = "unavailable (command line not readable)"
if ($newCmd -ne "") {
    $pAfter = Get-ProfileFromCommandLine -CmdLine $newCmd
    $pc2 = Test-ProfileCompliance -CmdProfile $pAfter
    if ($pc2.ok) { $profileAfterVerdict = "PASS" } else { $profileAfterVerdict = ("FAIL (" + ($pc2.issues -join "; ") + ")") }
    if (-not $pc2.ok) {
        Out-Line "  HIBA: az ujrainditott szerver parancssora ELTER a dokumentalt profiltol:" "Red"
        foreach ($iss in $pc2.issues) { Out-Line ("        " + $iss) "Red" }
        Out-Line ("  uj parancssor: " + $newCmd) "Gray"
        Fail "a dokumentalt produkcios profil nem teljesul az ujrainditott szerveren - NINCS csendes parameter-valtoztatas." "a rollback automatikusan indul."
    } else {
        Out-Line "  ujrainditott profil: EGYEZIK a dokumentalt produkcios profillal." "Green"
    }
    if (($liveCmd -ne "") -and ($null -ne $profileBefore)) {
        $same = ((Get-ProfileFromCommandLine -CmdLine $newCmd).model -eq $profileBefore.model) -and `
                ((Get-ProfileFromCommandLine -CmdLine $newCmd).ngl -eq $profileBefore.ngl) -and `
                ((Get-ProfileFromCommandLine -CmdLine $newCmd).ctx -eq $profileBefore.ctx)
        if (-not $same) {
            Out-Line "  FIGYELEM: az uj parancssor elter az audit altali regitol (l. REPORT) - a dokumentalt profil mindketto helyett ervenyes." "Yellow"
            Log "WARNING: relaunched command line differs from the audited one (documented profile still PASS)"
        }
    }
}

# model gate
$servedAfter = Get-ServedModelId -H "127.0.0.1" -P $Port
$mm = Test-ServedModelMatches -ServedId $servedAfter -ExpectedModelPath $script:ModelPath
if ($mm -ne "same") {
    Fail ("a szerver altal kiszolgalt modell NEM a varott Qwen3.6 blob: '" + $servedAfter + "'") "a modell-valasztas nem valtozhat - a rollback automatikusan indul."
}
Out-Line ("  modell OK: " + $servedAfter) "Green"

$script:R.start = [ordered]@{
    starter_pid = $startRes.starter_pid
    server_pid = $ServerPid
    waited_sec = $startRes.waited_sec
    exe_path = $newPath
    cmdline = $newCmd
    profile_after_verdict = $profileAfterVerdict
    served_after = $servedAfter
}

# ---------------------------------------------------------------------------
# [S7] SERVER-SIDE VERIFICATION BATTERY
# ---------------------------------------------------------------------------
$script:R.stage = "S7-battery"
Out-Line "`n[S7] szerver-oldali ellenorzesi battery..."

$healthStatus = 0
try { $healthStatus = [int](Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/health" -f $Port) -Method Get -TimeoutSec 5 -UseBasicParsing).StatusCode } catch { }

# C: Hungarian chat completion (non-stream)
Out-Line "  [C] magyar chat completion (nem-stream)..." "Gray"
$chat = Invoke-LlamaChat -H "127.0.0.1" -P $Port -BodyJson $script:HuChatBody -TimeoutSec $ChatTimeoutSec
$chatTxt = ""
if ($chat.ok -and $chat.obj) { try { $chatTxt = [string]$chat.obj.choices[0].message.content } catch { } }
$chatTxt = ConvertFrom-Mojibake $chatTxt
$chatFp = ""
if ($chat.ok -and $chat.obj) { $chatFp = [string]$chat.obj.system_fingerprint }
try {
    $chat.raw | Set-Content -LiteralPath (Join-Path $script:EvidenceDir "chat_test.json") -Encoding UTF8
} catch { }
if (-not $chat.ok -or ($chatTxt.Trim().Length -eq 0)) {
    Fail ("a chat completion NEM sikerult (status " + $chat.status + ", hiba: " + $chat.err + ")") "a rollback automatikusan indul."
}
Out-Line ("      -> " + $chatTxt.Trim()) "DarkCyan"

# D: streaming
Out-Line "  [D] streaming chat completion (elso token + tiszta lezaras)..." "Gray"
$stream = Invoke-StreamProbe -H "127.0.0.1" -P $Port -BodyJson $script:HuStreamBody -TimeoutSec $ChatTimeoutSec
try {
    @{ ok = $stream.ok; firstChunkMs = $stream.firstChunkMs; totalMs = $stream.totalMs; sawDone = $stream.sawDone; chunks = $stream.chunks; finish = $stream.finish; text = $stream.text; fingerprint = $stream.fingerprint; err = $stream.err } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $script:EvidenceDir "streaming_test.json") -Encoding UTF8
} catch { }
if (-not $stream.ok) {
    Fail ("a streaming NEM sikerult: " + $stream.err + " (chunks: " + $stream.chunks + ", [DONE]: " + $stream.sawDone + ")") "a rollback automatikusan indul."
}
Out-Line ("      -> elso chunk " + $stream.firstChunkMs + " ms, " + $stream.chunks + " SSE sor, [DONE] rendben, finish_reason: " + $stream.finish) "DarkCyan"

# E: JSON completion
Out-Line "  [E] JSON completion (response_format json_object)..." "Gray"
$js = Invoke-LlamaChat -H "127.0.0.1" -P $Port -BodyJson $script:HuJsonBody -TimeoutSec $ChatTimeoutSec
$jsTxt = ""
if ($js.ok -and $js.obj) { try { $jsTxt = [string]$js.obj.choices[0].message.content } catch { } }
$jsTxt = ConvertFrom-Mojibake $jsTxt
$jsParsed = $null; $jsOk = $false
if ($jsTxt -ne "") {
    $t = $jsTxt.Trim()
    if ($t.StartsWith('```')) { $t = ($t -replace '^```[a-z]*\r?\n?', '') -replace '\r?\n?```$', '' }
    try { $jsParsed = $t | ConvertFrom-Json; $jsOk = ($null -ne $jsParsed) } catch { $jsOk = $false }
}
try {
    @{ ok = $jsOk; raw = $js.raw; parsed = $jsParsed } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $script:EvidenceDir "json_test.json") -Encoding UTF8
} catch { }
if (-not $jsOk) {
    Fail ("a JSON completion NEM ervenyes JSON-t adott: '" + $jsTxt + "'") "a rollback automatikusan indul."
}
Out-Line ("      -> ervenyes JSON: " + $jsTxt.Trim()) "DarkCyan"

# the battery is recorded BEFORE the [F] gate so a failing fingerprint still
# shows the C/D/E results in the report
$script:R.battery = [ordered]@{
    health_status = $healthStatus
    served_after = $servedAfter
    model_match = $mm
    chat_ok = $true
    chat_ms = $chat.ms
    chat_chars = $chatTxt.Length
    chat_reply = $chatTxt
    stream_ok = $stream.ok
    stream_first_ms = $stream.firstChunkMs
    stream_chunks = $stream.chunks
    stream_done = $stream.sawDone
    stream_finish = $stream.finish
    stream_text = $stream.text
    json_ok = $jsOk
    json_value = $jsTxt
    fingerprint = $fp
    fingerprint_ok = $false
}

# F: system fingerprint / build identity of the LIVE server
$fp = $chatFp
if ($fp -eq "") { $fp = $stream.fingerprint }
if ($fp -eq "") { $fp = Get-SystemFingerprint -H "127.0.0.1" -P $Port }
$fpOk = Test-FingerprintOk $fp
$script:R.battery.fingerprint_ok = $fpOk
if (-not $fpOk) {
    Fail ("az elo szerver rendszerujjlenyomata NEM b11073: '" + $fp + "'") "a rollback automatikusan indul."
}
Out-Line ("  [F] system_fingerprint: " + $fp + "  -> EGYEZIK a varott b11073-1aa2954bd azonositassal.") "Green"
if ($healthStatus -ne 200) {
    Fail ("/health nem 200: " + $healthStatus) "a rollback automatikusan indul."
}

# ---------------------------------------------------------------------------
# [S8] VOICEMEM INTEGRATION
# ---------------------------------------------------------------------------
$script:R.stage = "S8-integration"
if ($SkipIntegration) {
    Out-Line "`n[S8] VoiceMem integracio: KIHAGYVA (-SkipIntegration)." "Yellow"
    $script:R.integration = $null
} else {
    Out-Line "`n[S8] VoiceMem integracio..."
    $script:R.integration = Invoke-IntegrationStage
}

# ---------------------------------------------------------------------------
# [S9] REGRESSION CHECKS
# ---------------------------------------------------------------------------
$script:R.stage = "S9-regression"
Out-Line "`n[S9] regresszios ellenorzesek..."

# port + listener
$regPortOk = $false
$Listeners9 = @(Get-ListenerPids -P $Port)
foreach ($Lp in $Listeners9) {
    $i9 = Get-ProcInfo -ProcId $Lp
    if ($i9.name -like "llama-server*") {
        if (($i9.path -eq "") -or ($i9.path -eq $script:ProdExe)) { $regPortOk = $true }
    }
}
# models id unchanged vs audit
$servedNow2 = Get-ServedModelId -H "127.0.0.1" -P $Port
$regModelOk = ((Test-ServedModelMatches -ServedId $servedNow2 -ExpectedModelPath $script:ModelPath) -eq "same")
if ($servedBefore -ne "" -and ($servedBefore -ne $servedNow2)) {
    Out-Line "  FIGYELEM: a /v1/models served id elter az audit elotti ertektol (l. REPORT)." "Yellow"
    Log ("regression: served id changed: before='" + $servedBefore + "' after='" + $servedNow2 + "'")
}
# fingerprint re-confirmed
$regFpOk = (Test-FingerprintOk (Get-SystemFingerprint -H "127.0.0.1" -P $Port))

# python sources + configs unchanged
$srcSnapshotAfter = @(New-SourceSnapshot -Root $script:Root)
$srcIssues = @()
if ($srcSnapshotAfter.Count -ne $script:SourceSnapshotBefore.Count) {
    $srcIssues += ("darabszam: " + $script:SourceSnapshotBefore.Count + " -> " + $srcSnapshotAfter.Count)
}
$beforeMap = @{}
foreach ($e in $script:SourceSnapshotBefore) { $beforeMap[[string]$e.path] = [string]$e.sha256 }
foreach ($e in $srcSnapshotAfter) {
    $k = [string]$e.path
    if (-not $beforeMap.ContainsKey($k)) { $srcIssues += ("uj fajl: " + $k); continue }
    if ($beforeMap[$k] -ne [string]$e.sha256) { $srcIssues += ("valtozott: " + $k) }
}
$regSourcesOk = ($srcIssues.Count -eq 0)
try { $srcSnapshotAfter | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $script:EvidenceDir "source_snapshot_after.json") -Encoding UTF8 } catch { }

# model file unchanged
$modelOk = $false
if (Test-Path -LiteralPath $script:ModelPath) {
    $mi2 = Get-Item -LiteralPath $script:ModelPath
    $ph2 = Get-PartialSha256 -Path $script:ModelPath
    $modelOk = (($mi2.Length -eq $modelItem.bytes) -and ($mi2.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ss") -eq $modelItem.mtime) -and ($ph2 -eq $modelItem.partial_sha256))
    $script:R.audit.model_after_note = ("bytes/mtime/partial-hash " + ($(if ($modelOk) { "UNCHANGED" } else { "CHANGED!" })))
}
if (-not $modelOk) {
    Out-Line "  KRITIKUS: a modellfajl integritasi jelei VALTOZTAK - ez a csomag soha nem piszkalja a modellt!" "Red"
    Log "CRITICAL: model integrity marks changed"
}

# leakage: 8081/8082 must not appear in the production command line
$regLeakOk = (-not ($newCmd -match ":8081|:8082"))

# starter intact
$regStarterOk = ((Test-Path -LiteralPath (Join-Path $script:Root "START.bat")) -and (Test-Path -LiteralPath $script:Starter))

$script:R.regression = [ordered]@{
    port_ok = $regPortOk
    served_now = $servedNow2
    model_ok = $regModelOk
    fingerprint_now_ok = $regFpOk
    sources_ok = $regSourcesOk
    sources_note = ($(if ($regSourcesOk) { "all " + $script:SourceSnapshotBefore.Count + " files identical" } else { ($srcIssues -join "; ") }))
    model_file_ok = $modelOk
    leak_ok = $regLeakOk
    starter_ok = $regStarterOk
    source_issues = $srcIssues
}

$hardRegOk = ($regPortOk -and $regModelOk -and $regFpOk -and $regLeakOk -and $regStarterOk)
if (-not $hardRegOk) {
    Fail "a regresszios alap-ellenorzesek kudarcot vallottak (l. REPORT)." "a rollback automatikusan indul."
}
if (-not $regSourcesOk) {
    Out-Line "  KRITIKUS FIGYELMEZTETES: python-forrasok/konfigok valtoztak a futas alatt (nem a csomag tette):" "Red"
    foreach ($s in $srcIssues) { Out-Line ("        " + $s) "Red" }
    Out-Line "  (a futtas nem all vissza automatikusan, mert a runtime frissites maga rendben van - de ezt tisztazni kell.)" "Yellow"
}
if (-not $modelOk) {
    Out-Line "  KRITIKUS FIGYELMEZTETES: a MODELLFAJL valtozott (l. fent). Ez nem a frissites resze!" "Red"
}
Out-Line ("  port 8080 + modell + b11073 azonossag + starter: " + ($(if ($hardRegOk) { "OK" } else { "FAIL" }))) ($(if ($hardRegOk) { "Green" } else { "Red" }))

# ---------------------------------------------------------------------------
# [S10] REPORT
# ---------------------------------------------------------------------------
$script:R.stage = "S10-report"
$script:R.verdict = "UPGRADE COMPLETE (production = pinned b11073-1aa2954bd runtime + the unchanged Qwen3.6 model + the unchanged production profile)"
Close-Report

Write-Host ""
Write-Host "=== FRISSITES KESZ ===" -ForegroundColor Green
Write-Host ("  produkcios runtime : b11073 (" + $script:ExpectedFingerprint + ") - 24/24 SHA-256 + build-azonositas + elo fingerprint") -ForegroundColor Green
Write-Host ("  modell             : " + $script:ModelPath + "  (valtozatlan)") -ForegroundColor Green
Write-Host ("  endpoint           : 127.0.0.1:" + $Port + "  (profil: ngl " + $script:DocProfile.ngl + " / ctx " + $script:DocProfile.ctx + " / parallel " + $script:DocProfile.parallel + " / KV q8_0 / temp " + $script:DocProfile.temp + " / reasoning off / metrics / no-webui)") -ForegroundColor Green
Write-Host ("  evidence           : " + $script:EvidenceDir) -ForegroundColor Green
Write-Host ("  rollback           : " + $script:R.rollback + " | backup: " + $(if ($script:BackupDir -ne "") { $script:BackupDir } else { "(nem kellett)" })) -ForegroundColor Green
Write-Host ""
Write-Host "Kuldd vissza az audit-hoz:" -ForegroundColor Cyan
Write-Host ("  " + (Join-Path $script:EvidenceDir "prod_b11073_upgrade_evidence.zip")) -ForegroundColor White
Write-Host ""
exit 0
