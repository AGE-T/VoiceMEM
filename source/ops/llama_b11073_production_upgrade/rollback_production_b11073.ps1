<#
===============================================================================
ops/llama_b11073_production_upgrade/rollback_production_b11073.ps1

PRODUCTION ROLLBACK: restore the previous production llama.cpp runtime
(the timestamped backup created by upgrade_production_b11073.ps1, default
backups\runtime_pre_b11073_<ts>\bin\) and restart the production server
through the EXISTING production starter (scripts\start_llama_server.ps1).

WHAT IT DOES (in order):
  1. stops the llama-server listening on the production port (ONLY if it is
     llama-server.exe; unrelated processes are NEVER touched);
  2. restores bin\ from the backup (when -BackupDir is given):
       - removes the llama.cpp-family files currently in bin\ that are not
         present in the backup (the b11073 set),
       - copies EVERY file from the backup bin\ back (including the CUDA
         runtime DLLs, which were never modified anyway),
       - verifies the restored llama-server.exe SHA-256 equals the hash
         recorded in backup_manifest.json at backup time;
  3. restarts the production server via the existing starter (new window),
     waits for /health;
  4. verifies: /health 200, /v1/models serving the same Qwen3.6 model, a real
     chat completion round-trip, and the live system_fingerprint is NOT
     b11073 (the rollback proof);
  5. writes a rollback log next to the backup (rollback_<ts>.log).

WITHOUT -BackupDir: no file restore is attempted (bin\ is assumed unchanged);
the script only stops + restarts the production server and verifies it.

USAGE (Bypass form):
  powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\rollback_production_b11073.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\rollback_production_b11073.ps1 -BackupDir "F:\Voicemem\VoiceMemAgent\backups\runtime_pre_b11073_20260924_120000"

The file is deliberately ASCII and PS 5.1-safe (no ternary / null-coalescing,
EAP-safe native output capture).
===============================================================================
#>
param(
    [string]$Root = "",
    [string]$BackupDir = "",
    [string]$ModelPath = "",
    [int]$Port = 8080,
    [int]$HealthTimeoutSec = 900,
    [int]$ChatTimeoutSec = 180,
    [switch]$StartedByUpgrade
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$script:ExpectedCommit = "1aa2954bd"     # the b11073 build id (must be GONE after rollback)
if ([string]::IsNullOrWhiteSpace($ModelPath)) { $script:ModelPath = "C:\AI_HOME\models\blobs\sha256-afc7238af403ed454b7846454091b5e38b07575bdbb64c3e86777414dde4193c" } else { $script:ModelPath = $ModelPath }

$script:ScriptDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$script:Root      = $Root
$script:ProdBin   = Join-Path $Root "bin"
$script:ProdExe   = Join-Path $ProdBin "llama-server.exe"
$script:Starter   = Join-Path $Root "scripts\start_llama_server.ps1"
$script:Stamp     = (Get-Date).ToString("yyyyMMdd_HHmmss")

# resolve the newest backup when -BackupDir is not given
if ([string]::IsNullOrWhiteSpace($BackupDir)) {
    $backupsRoot = Join-Path $Root "backups"
    if (Test-Path -LiteralPath $backupsRoot) {
        $cands = @(Get-ChildItem -LiteralPath $backupsRoot -Directory -Filter "runtime_pre_b11073_*" -ErrorAction SilentlyContinue | Sort-Object Name -Descending)
        if ($cands.Count -gt 0) {
            $BackupDir = $cands[0].FullName
            Write-Host ("[rollback] backup mappa automatikus feloldasa: " + $BackupDir) -ForegroundColor Cyan
        }
    }
}
$script:BackupDir = $BackupDir
$script:RollbackLog = ""
if ($script:BackupDir -ne "") {
    $script:RollbackLog = Join-Path $script:BackupDir ("rollback_" + $script:Stamp + ".log")
}

function Log {
    param([string]$Msg)
    $line = ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Msg)
    Write-Host $line -ForegroundColor Gray
    if ($script:RollbackLog -ne "") {
        try { Add-Content -LiteralPath $script:RollbackLog -Value $line -Encoding UTF8 } catch { }
    }
}

function Fail {
    param([string]$Msg, [string]$Hint = "")
    Write-Host ("HIBA: " + $Msg) -ForegroundColor Red
    if ($Hint -ne "") { Write-Host ("JAVITAS: " + $Hint) -ForegroundColor Yellow }
    Log ("FAIL: " + $Msg + " | HINT: " + $Hint)
    exit 1
}

function Get-Sha256 {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return "" }
    try { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash } catch { return "" }
}

function Get-ListenerPids {
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
        # Linux/sandbox fallback (no-op on Windows; see the orchestrator note)
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
        try { $P = Get-Process -Id $ProcId -ErrorAction Stop; $info.name = [string]$P.ProcessName } catch { }
    }
    return $info
}

function Test-Health {
    param([int]$P, [int]$T = 3)
    try {
        $R = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/health" -f $P) -Method Get -TimeoutSec $T -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

function Get-ServedModelId {
    param([int]$P)
    try {
        $M = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/v1/models" -f $P) -Method Get -TimeoutSec 10
        return [string]$M.data[0].id
    } catch { return "" }
}

function Get-CurrentPowerShellExe {
    try {
        $p = (Get-Process -Id $PID -ErrorAction Stop).Path
        if ($p -and (Test-Path -LiteralPath $p)) { return $p }
    } catch { }
    return (Join-Path $PSHOME "powershell.exe")
}

Write-Host ""
Write-Host "=== PRODUKCIO ROLLBACK: b11073 -> az elozo produkcios runtime ===" -ForegroundColor Magenta
Write-Host ("  root    : " + $script:Root) -ForegroundColor Gray
Write-Host ("  bin     : " + $script:ProdBin) -ForegroundColor Gray
Write-Host ("  backup  : " + $(if ($script:BackupDir -ne "") { $script:BackupDir } else { "(nincs megadva - csak ujrainditas)" })) -ForegroundColor Gray
Log ("rollback start; backup=" + $script:BackupDir + "; startedByUpgrade=" + $StartedByUpgrade)

# --- 1) stop the production llama-server ------------------------------------
$Listeners = @(Get-ListenerPids -P $Port)
foreach ($Lp in $Listeners) {
    $info = Get-ProcInfo -ProcId $Lp
    if ($info.name -ne "" -and $info.name -notlike "llama-server*") {
        Fail ("a(z) " + $Port + " porton a(z) '" + $info.name + "' folyamat figyel - NEM allitom le.") "kezi beavatkozas szukseges."
    }
    Log ("stopping llama-server pid " + $Lp + " (path: " + $info.path + ")")
    try { Stop-Process -Id $Lp -Force -ErrorAction Stop } catch { }
}
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    if (@(Get-ListenerPids -P $Port).Count -eq 0) { break }
    Start-Sleep -Milliseconds 500
}
if (@(Get-ListenerPids -P $Port).Count -gt 0) {
    Fail ("a(z) " + $Port + " port nem szabadult fel a leallitas utan sem.") "netstat -ano | findstr :" + $Port
}

# --- 2) restore bin\ from the backup ----------------------------------------
if ($script:BackupDir -ne "") {
    $backupBin = Join-Path $script:BackupDir "bin"
    $manifestPath = Join-Path $script:BackupDir "backup_manifest.json"
    if (-not (Test-Path -LiteralPath $backupBin)) {
        Fail ("a backup bin mappa nem letezik: " + $backupBin) "add meg a helyes -BackupDir-t."
    }
    $expectedExeHash = ""
    $backupNames = @()
    if (Test-Path -LiteralPath $manifestPath) {
        try {
            $bm = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            $expectedExeHash = ([string]$bm.llama_server_sha256_before).ToUpper()
            foreach ($f in $bm.files) { $backupNames += [string]$f.name }
        } catch { }
    }
    if ($backupNames.Count -eq 0) {
        $backupNames = @(Get-ChildItem -LiteralPath $backupBin -File | ForEach-Object { $_.Name })
    }
    if (-not (Test-Path -LiteralPath (Join-Path $backupBin "llama-server.exe"))) {
        Fail ("a backup nem tartalmaz llama-server.exe-t: " + $backupBin) "a backup hibas/ures - add meg a helyes -BackupDir-t."
    }
    if ($expectedExeHash -eq "") { $expectedExeHash = (Get-Sha256 (Join-Path $backupBin "llama-server.exe")).ToUpper() }
    if ($expectedExeHash -eq "") {
        Fail "a backup manifest es a backup exe hash-e sem olvashato." "serult backup - nezd meg a backup_manifest.json-t."
    }

    # remove the llama.cpp-family files currently in bin\ that are NOT in the
    # backup (the b11073 set) - CUDA runtime DLLs are NEVER touched
    foreach ($f in @(Get-ChildItem -LiteralPath $script:ProdBin -File -ErrorAction SilentlyContinue)) {
        $n = $f.Name
        if ($backupNames -contains $n) { continue }
        if ($n -notmatch '\.(dll|exe)$') { continue }
        if ($n -like "cudart*" -or $n -like "cublas*" -or $n -like "cublasLt*" -or $n -like "nvrtc*" -or $n -like "nvjitlink*" -or $n -like "cupti*" -or $n -like "nvml*" -or $n -like "myelin*") { continue }
        if ($n -like "llama*" -or $n -like "ggml*" -or $n -like "mtmd*" -or $n -like "libomp*") {
            try {
                Remove-Item -LiteralPath $f.FullName -Force -ErrorAction Stop
                Log ("removed (not in backup): " + $n)
            } catch {
                Fail ("a(z) " + $n + " nem torolheto: " + $_.Exception.Message) "zarold be a bin\-t hasznalo folyamatokat."
            }
        }
    }
    # copy EVERY backup file back
    foreach ($f in @(Get-ChildItem -LiteralPath $backupBin -File -ErrorAction SilentlyContinue)) {
        try {
            Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $script:ProdBin $f.Name) -Force -ErrorAction Stop
        } catch {
            Fail ("a visszamasolas nem sikerult: " + $f.Name + " (" + $_.Exception.Message + ")") "szabaditsd fel a fajlt es probald ujra."
        }
    }
    # verify the restored exe hash
    $restoredHash = (Get-Sha256 $script:ProdExe).ToUpper()
    if ($restoredHash -ne $expectedExeHash.ToUpper()) {
        Fail ("a visszallitott llama-server.exe hash-e elter a backup manifest rogzitetttol: " + $restoredHash + " != " + $expectedExeHash) "a backup serult - nezd meg a backup_manifest.json-t."
    }
    Write-Host "  OK: bin\ visszallitva a backupbol (llama-server.exe SHA-256 egyezik a backup manifesttel)." -ForegroundColor Green
    Log ("restore complete; exe sha256 = " + $restoredHash)
}

# --- 3) restart via the EXISTING production starter --------------------------
if (-not (Test-Path -LiteralPath $script:Starter)) {
    Fail ("nem talalhato a produkcios indito: " + $script:Starter) "a visszallitas megtortent, de az ujrainditashoz a starter szukseges."
}
$PsExe = Get-CurrentPowerShellExe
$argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $script:Starter + '"'), "-WaitSec", [string]$HealthTimeoutSec)
Log ("restart via starter: " + $PsExe + " " + ($argList -join " "))
try {
    $null = Start-Process -FilePath $PsExe -ArgumentList $argList -WorkingDirectory $script:Root -PassThru
} catch {
    Fail ("a produkcios indito elinditasa nem sikerult: " + $_.Exception.Message) "inditsd kezzel: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_llama_server.ps1"
}

$t0 = Get-Date
$healthy = $false
while (((Get-Date) - $t0).TotalSeconds -lt $HealthTimeoutSec) {
    if (Test-Health -P $Port -T 3) { $healthy = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $healthy) {
    Fail ("a visszallitott szerver nem valaszolt a /health-re " + $HealthTimeoutSec + " s-en belul.") "nezd meg a logs\llama-server.err.log-ot es a starter ablakat."
}
Write-Host ("  OK: /health 200 (" + [math]::Round(((Get-Date) - $t0).TotalSeconds, 0) + " s).") -ForegroundColor Green

# --- 4) verification ----------------------------------------------------------
$served = Get-ServedModelId -P $Port
$servedN = ([string]$served).ToLower() -replace '[^0-9a-z]', ''
# last path segment of ANY path form (both separators, both OSes)
$leaf = [string]$script:ModelPath
$lastIdx = [Math]::Max($leaf.LastIndexOf('\'), $leaf.LastIndexOf('/'))
if ($lastIdx -ge 0) { $leaf = $leaf.Substring($lastIdx + 1) }
if ($leaf.Contains(".")) { $leaf = $leaf.Substring(0, $leaf.LastIndexOf(".")) }
$stemN = $leaf.ToLower() -replace '[^0-9a-z]', ''
$modelOk = ($servedN.Length -gt 0 -and $stemN.Length -gt 0 -and (($servedN.Contains($stemN)) -or ($stemN.Contains($servedN))))
if (-not $modelOk) {
    Fail ("a visszallitott szerver mas modellt szolgal ki: '" + $served + "'") "a modell-valasztas nem valtozhat - kezi ellenorzes szukseges."
}
Write-Host ("  OK: /v1/models -> " + $served) -ForegroundColor Green

# a real completion round-trip (the production starter's own probe pattern)
$probeOk = $false
try {
    $body = '{"messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":16,"temperature":0,"stream":false}'
    $R = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/v1/chat/completions" -f $Port) -Method Post -Body $body -ContentType "application/json" -TimeoutSec $ChatTimeoutSec
    $txt = ""
    try { $txt = [string]$R.choices[0].message.content } catch { }
    if ($txt.Trim().Length -gt 0) { $probeOk = $true }
} catch { }
if (-not $probeOk) {
    Fail "a chat completion round-trip nem sikerult a visszallitott szerveren." "nezd meg a logs\llama-server.err.log-ot."
}
Write-Host "  OK: chat completion round-trip." -ForegroundColor Green

# the rollback proof: the live fingerprint must NOT be b11073
$fp = ""
try {
    $body2 = '{"messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0.0}'
    $R2 = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/v1/chat/completions" -f $Port) -Method Post -Body $body2 -ContentType "application/json" -TimeoutSec 90
    $fp = [string]$R2.system_fingerprint
} catch { }
$isB11073 = (($fp -match "b11073") -or ($fp -match $script:ExpectedCommit))
if ($fp -ne "") {
    if ($isB11073) {
        Fail ("a visszallitott szerver meg mindig b11073-kent azonositja magat (fingerprint = '" + $fp + "')") "a visszallitas nem teljesult - nezd meg a backup tartalmat."
    }
    Write-Host ("  OK: a visszallitott szerver fingerprint-je NEM b11073: '" + $fp + "'") -ForegroundColor Green
} else {
    Write-Host "  FIGYELEM: a fingerprint nem olvashato ki - az exe hash + health + completion bizonyitek ervenyesek." -ForegroundColor Yellow
}

Log ("rollback COMPLETE; served='" + $served + "' fingerprint='" + $fp + "'")
Write-Host ""
Write-Host "=== ROLLBACK KESZ: a regi produkcios runtime visszaallitva es ujrainditva. ===" -ForegroundColor Green
if (-not $StartedByUpgrade) {
    Write-Host "A web backend ujrainditasa a normal modon: START.bat (a szerver mar fut)." -ForegroundColor Cyan
}
exit 0
