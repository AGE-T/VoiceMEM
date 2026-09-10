<#
===============================================================================
scripts/find_qwen_gguf.ps1 - EXACT Qwen3.6 35B A3B IQ4_XS GGUF locator (v0.4.15)

v0.4.18 FIX (field report 2026-09): the GGUF metadata VALUE TYPE table was
wrong in every shipped walker (8=BOOL, 9=STRING, 10=ARRAY, 7=FLOAT64,
13=FLOAT16). The real GGUF spec (llama.cpp gguf-py constants.py
GGUFValueType) is 7=BOOL(1B), 8=STRING, 9=ARRAY, 10=UINT64(8B), 11=INT64(8B),
12=FLOAT64(8B) - no type 13. Real GGUFs desynchronised at the first STRING
KV; the walk below now follows the authoritative table.

WHY (v0.4.15 field instruction): before the llama-server runtime
configuration may be trusted, the ACTUAL local GGUF file of the EXACT model
must be located on disk - the filename must NOT be assumed, and no other
model or quantisation may ever be substituted. This script searches the
project and the local model directories for the file, validates the GGUF
header (model identity + quantisation), prints the exact absolute Windows
path, and can point the existing env configuration at the discovered file.

TARGET (the ONLY accepted model):
    Model        : Qwen3.6 35B A3B IQ4_XS (MoE, ~3B active, ~19-20 GB file)
    Tested via   : Ollama tag hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS
    Runtime      : local llama.cpp llama-server (127.0.0.1:8080/v1) - Ollama
                   is NOT the production runtime. Ollama is only a possible
                   STORAGE location: its model blob IS a plain GGUF file that
                   llama-server can read directly.

SEARCHED LOCATIONS (in order):
    1. the repo tree     : $Root\**\*.gguf (covers models\llm\... and the
                           project-local HF cache models\hf\hub\...)
    2. the Ollama store  : manifests matching Qwen*3.6*35B*A3B under
                           OLLAMA_MODELS or %USERPROFILE%\.ollama\models
                           (the IQ4_XS tag resolves to its model blob)
    3. the user HF cache : models--bartowski--Qwen_Qwen3.6-35B-A3B-GGUF
                           (HF_HOME / HF_HUB_CACHE / .cache\huggingface)
    4. LM Studio         : %USERPROFILE%\.lmstudio\models\**\*.gguf
    5. user folders      : Downloads / Desktop / Documents (*.gguf)
    6. custom root       : -Path <dir> recursive *.gguf scan

USAGE:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1 -Path F:\
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1 -CopyToRepo
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1 -SetEnv

    -Path <dir> : additionally scan a custom drive/folder recursively
    -CopyToRepo : with EXACTLY ONE exact match: copy it to the canonical
                  models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf
                  (zero config change afterwards; needs ~20 GB free)
    -SetEnv     : with EXACTLY ONE exact match: rewrite ONLY the
                  $env:LLAMA_MODEL_PATH line in config\env.local.ps1 to the
                  discovered path (a .bak backup is written first; restart
                  llama-server afterwards - START.bat)

RESULT: FOUND (exact path + quant proof + next steps) or MISSING (the exact
    download source is printed; NO substitution is performed - a missing
    active GGUF stays a loud LLM ERROR by design; there is NO fallback
    profile to switch to, never automatic).

EXIT CODES: 0 = exact match found (and copied/set when requested);
            1 = missing / not a GGUF / ambiguous / copy or env write failed.
===============================================================================
#>
param(
    [string]$Path = "",
    [switch]$CopyToRepo,
    [switch]$SetEnv
)

$ErrorActionPreference = "Stop"

# --- target identity ---------------------------------------------------------
$TargetModel    = "Qwen3.6 35B A3B IQ4_XS"
$TargetRepoRef  = "hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS"
$TargetHfUrl    = "https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF"
$TargetHfFile   = "Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
$TargetFile     = "Qwen3.6-35B-A3B-IQ4_XS.gguf"   # canonical local name
$Iq4XsFileType  = 30                               # GGUF general.file_type (llama_ftype
$MinTargetBytes = 10GB                             # 35B IQ4_XS is ~19-20 GB

$Root          = Split-Path -Parent $PSScriptRoot
$CanonicalDir  = Join-Path $Root "models\llm\qwen3.6-35b-a3b"
$CanonicalPath = Join-Path $CanonicalDir $TargetFile

# GGUF file_type -> label (common quants; unknown types print as numbers)
# llama_ftype (gguf-py LlamaFileType) - v0.4.16 CORRECTED table (the v0.4.15
# table had IQ4_XS=29, which is IQ2_M; IQ4_XS is 30):
$QuantLabels = @{
    0 = "F32";   1 = "F16";   2 = "Q4_0";  3 = "Q4_1";  7 = "Q8_0";
    8 = "Q5_0";  9 = "Q5_1";  10 = "Q2_K"; 11 = "Q3_K_S"; 12 = "Q3_K_M";
    13 = "Q3_K_L"; 14 = "Q4_K_S"; 15 = "Q4_K_M"; 16 = "Q5_K_S"; 17 = "Q5_K_M";
    18 = "Q6_K";  19 = "IQ2_XXS"; 20 = "IQ2_XS"; 21 = "Q2_K_S"; 22 = "IQ3_XS";
    23 = "IQ3_XXS"; 24 = "IQ1_S"; 25 = "IQ4_NL"; 26 = "IQ3_S"; 27 = "IQ3_M";
    28 = "IQ2_S"; 29 = "IQ2_M"; 30 = "IQ4_XS"; 31 = "IQ1_M"; 32 = "BF16"
}

function Get-Norm([string]$s) {
    if ($null -eq $s) { return "" }
    return ($s.ToLower() -replace "[._\- ]", "")
}

function Test-TargetModelName([string]$s) {
    $n = Get-Norm $s
    return ($n.Contains("qwen36") -and $n.Contains("35b") -and $n.Contains("a3b"))
}

function Test-TargetQuantName([string]$s) {
    return ((Get-Norm $s).Contains("iq4xs"))
}

# --- GGUF header identity: magic + general.name + general.file_type ----------
# GGUF v2/v3 layout: magic("GGUF") + u32 version + u64 tensor_count +
# u64 kv_count, then kv_count pairs of (u64 key_len + key bytes + u32 value
# type + value). We walk the KVs (arrays included) just far enough to read
# general.name (string) and general.file_type (u32); unknown types abort the
# walk gracefully and the caller falls back to filename+size classification.
# VALUE TYPES - the ACTUAL GGUF spec table (llama.cpp gguf-py constants.py
# GGUFValueType). v0.4.18 FIX: the v0.4.16 "corrected" table below was STILL
# WRONG (7=FLOAT64 8=BOOL 9=STRING 10=ARRAY 13=FLOAT16) - real GGUF files
# desynchronised at the first STRING KV (type 8 read as 1 byte) and every
# identity came back EMPTY. Correct table:
#   0=UINT8(1B) 1=INT8(1B) 2=UINT16(2B) 3=INT16(2B) 4=UINT32(4B)
#   5=INT32(4B) 6=FLOAT32(4B) 7=BOOL(1B) 8=STRING(u64 len + bytes)
#   9=ARRAY(u32 elem type + u64 count + items) 10=UINT64(8B) 11=INT64(8B)
#   12=FLOAT64(8B) - there is NO type 13.
$GgufScalar = @{
    0 = 1; 1 = 1; 2 = 2; 3 = 2; 4 = 4; 5 = 4; 6 = 4; 7 = 1
    10 = 8; 11 = 8; 12 = 8
}
function Read-GgufString([System.IO.BinaryReader]$br) {
    $len = $br.ReadUInt64()
    if ($len -gt 16777216) { throw ("absurd GGUF string length {0}" -f $len) }
    return [System.Text.Encoding]::UTF8.GetString($br.ReadBytes([int]$len))
}
function Skip-GgufValue([System.IO.BinaryReader]$br, [int]$vt, [int]$depth = 0) {
    # STRING (type 8): u64 length + UTF-8 bytes
    if ($vt -eq 8) { $null = Read-GgufString $br; return }
    # ARRAY (type 9): u32 elem type + u64 count + items
    if ($vt -eq 9) {
        $et = [int]$br.ReadUInt32()     # CAST: hashtable keys are Int32 -
        $cnt = $br.ReadUInt64()          # UInt32 lookups would MISS
        if ($cnt -gt 16777216) { throw ("absurd GGUF array count {0}" -f $cnt) }
        if ($et -eq 8) {
            # string array: item-wise (u64 len + bytes per item). Real token
            # arrays hold ~150-260k items - inline loop, no per-item call.
            for ($j = 0; $j -lt $cnt; $j++) {
                $l = $br.ReadUInt64()
                if ($l -gt 16777216) { throw ("absurd GGUF string length {0}" -f $l) }
                if ($l -gt 0) { $null = $br.ReadBytes([int]$l) }
            }
        } elseif ($et -eq 9) {
            # nested array (spec-legal, unused by real converters)
            if ($depth -ge 4) { throw "GGUF arrays nested too deeply" }
            for ($j = 0; $j -lt $cnt; $j++) { Skip-GgufValue $br 9 ($depth + 1) }
        } elseif ($GgufScalar.ContainsKey($et)) {
            $bytes = [UInt64]$cnt * [UInt64]$GgufScalar[$et]
            if ($bytes -gt 268435456) { throw ("absurd GGUF array byte length {0}" -f $bytes) }
            $null = $br.ReadBytes([int]$bytes)
        } else {
            throw ("unsupported GGUF array element type {0}" -f $et)
        }
        return
    }
    if ($GgufScalar.ContainsKey($vt)) {
        $null = $br.ReadBytes($GgufScalar[$vt])
        return
    }
    throw ("unknown GGUF value type {0}" -f $vt)
}
function Read-GgufIdentity {
    param([string]$FilePath)
    $out = @{ magic = $false; version = 0; name = $null; file_type = -1; error = $false }
    $fs = $null
    try {
        $fs = [System.IO.File]::OpenRead($FilePath)
        $br = New-Object System.IO.BinaryReader($fs)
        $magic = $br.ReadBytes(4)
        if (($magic.Length -ne 4) -or ($magic[0] -ne 0x47) -or ($magic[1] -ne 0x47) -or
            ($magic[2] -ne 0x55) -or ($magic[3] -ne 0x46)) {
            $out.error = $true
            return $out
        }
        $out.magic = $true
        $out.version = $br.ReadUInt32()
        $null = $br.ReadUInt64()          # tensor count
        $kvCount = $br.ReadUInt64()       # metadata KV count (u64)
        if ($kvCount -gt 65536) { throw ("absurd GGUF kv count {0}" -f $kvCount) }
        for ($i = 0; $i -lt [int]$kvCount; $i++) {
            $klen = $br.ReadUInt64()      # KEY LENGTH IS u64
            if ($klen -gt 4096) { throw ("absurd GGUF key length {0}" -f $klen) }
            $key = [System.Text.Encoding]::UTF8.GetString($br.ReadBytes([int]$klen))
            $vt = [int]$br.ReadUInt32()
            $strVal = $null
            $u32Val = -1
            if ($vt -eq 8) {             # STRING: capture for identity keys
                $strVal = Read-GgufString $br
            } elseif ($vt -eq 4) {       # UINT32: file_type key
                $u32Val = [int]$br.ReadUInt32()
            } else {
                Skip-GgufValue $br $vt
            }
            if ($null -ne $strVal) {
                if ($key -eq "general.name") { $out.name = $strVal }
            }
            if ($u32Val -ge 0) {
                if ($key -eq "general.file_type") { $out.file_type = $u32Val }
            }
            if (($null -ne $out.name) -and ($out.file_type -ge 0)) { break }
        }
        return $out
    } catch {
        $out.error = $true
        return $out
    } finally {
        if ($null -ne $fs) { $fs.Close() }
    }
}

# --- candidate collection ----------------------------------------------------
$Candidates = New-Object System.Collections.ArrayList
$Searched   = New-Object System.Collections.ArrayList

function Add-GgufCandidate([string]$FilePath, [string]$Source) {
    if (-not $FilePath) { return }
    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) { return }
    foreach ($c in $Candidates) {
        if ($c.path -eq $FilePath) { return }
    }
    [void]$Candidates.Add(@{ path = $FilePath; source = $Source })
}

function Add-ScanDir([string]$Dir, [string]$Label, [int]$Depth = -1) {
    if (-not $Dir) { return }
    if (-not (Test-Path -LiteralPath $Dir)) {
        [void]$Searched.Add("$Label : (absent) $Dir")
        return
    }
    [void]$Searched.Add("$Label : $Dir")
    $files = $null
    if ($Depth -ge 0) {
        $files = @(Get-ChildItem -LiteralPath $Dir -Recurse -File -Depth $Depth -Filter "*.gguf" -ErrorAction SilentlyContinue)
    } else {
        $files = @(Get-ChildItem -LiteralPath $Dir -Recurse -File -Filter "*.gguf" -ErrorAction SilentlyContinue)
    }
    foreach ($f in $files) { Add-GgufCandidate $f.FullName $Label }
}

# 1. the whole repo tree (canonical models\llm + the project-local HF cache)
Add-ScanDir $Root "repo tree"

# 2. the Ollama store - resolve the manifest of the TESTED model reference
$ollamaRoot = $env:OLLAMA_MODELS
if (-not $ollamaRoot) { $ollamaRoot = Join-Path $env:USERPROFILE ".ollama\models" }
$manifestRoot = Join-Path $ollamaRoot "manifests"
if (Test-Path -LiteralPath $manifestRoot) {
    [void]$Searched.Add("ollama store : $manifestRoot")
    $mans = @(Get-ChildItem -LiteralPath $manifestRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "(?i)qwen.*3\.6.*35b.*a3b" })
    foreach ($m in $mans) {
        $tag = $m.Name
        $repoDir = Split-Path -Parent $m.FullName
        if ($repoDir.Length -le ($manifestRoot.Length + 1)) { continue }
        $repoName = $repoDir.Substring($manifestRoot.Length + 1)
        $json = $null
        try { $json = Get-Content -LiteralPath $m.FullName -Raw | ConvertFrom-Json } catch { $json = $null }
        if ($null -eq $json) {
            Write-Host ("  [ollama] manifest UNREADABLE: {0}" -f $m.FullName) -ForegroundColor Yellow
            continue
        }
        $modelLayers = @($json.layers | Where-Object { $_.mediaType -eq "application/vnd.ollama.image.model" })
        if ($modelLayers.Count -eq 0) { continue }
        if ($modelLayers.Count -gt 1) {
            Write-Host ("  [ollama] {0}:{1} is SPLIT into {2} blobs - a split blob set" -f $repoName, $tag, $modelLayers.Count) -ForegroundColor Yellow
            Write-Host "           cannot be loaded by llama-server directly: download the single-file GGUF instead."
            continue
        }
        $digest = [string]$modelLayers[0].digest
        if ($digest -notmatch "^sha256:[0-9a-f]{64}$") { continue }
        $blobPath = Join-Path (Join-Path $ollamaRoot "blobs") ("sha256-" + $digest.Substring(7))
        if (-not (Test-Path -LiteralPath $blobPath -PathType Leaf)) {
            Write-Host ("  [ollama] manifest {0}:{1} found but the BLOB IS MISSING: {2}" -f $repoName, $tag, $blobPath) -ForegroundColor Yellow
            continue
        }
        Add-GgufCandidate $blobPath ("ollama {0}:{1}" -f $repoName, $tag)
    }
} else {
    [void]$Searched.Add("ollama store : (absent) $manifestRoot")
}

# 3. the user-level HF cache (default + any env overrides)
$hfRoots = @(
    (Join-Path $env:USERPROFILE ".cache\huggingface\hub"),
    (Join-Path $env:USERPROFILE ".cache\huggingface")
)
if ($env:HF_HOME)      { $hfRoots += (Join-Path $env:HF_HOME "hub"); $hfRoots += $env:HF_HOME }
if ($env:HF_HUB_CACHE) { $hfRoots += $env:HF_HUB_CACHE }
foreach ($h in $hfRoots) {
    if (-not (Test-Path -LiteralPath $h)) { continue }
    Add-ScanDir $h "hf cache" 6
}

# 4. LM Studio models
Add-ScanDir (Join-Path $env:USERPROFILE ".lmstudio\models") "lm studio models" 8

# 5. common user folders
Add-ScanDir (Join-Path $env:USERPROFILE "Downloads") "downloads" 5
Add-ScanDir (Join-Path $env:USERPROFILE "Desktop")    "desktop"  4
Add-ScanDir (Join-Path $env:USERPROFILE "Documents")  "documents" 5

# 6. custom root (-Path)
if ($Path -ne "") {
    if (Test-Path -LiteralPath $Path) {
        Add-ScanDir $Path "custom (-Path)" 8
    } else {
        Write-Host "WARNING: -Path does not exist: $Path" -ForegroundColor Yellow
    }
}

# --- header + classification -------------------------------------------------
Write-Host ""
Write-Host "================================================================="
Write-Host " Qwen3.6 35B A3B IQ4_XS GGUF locator"
Write-Host "================================================================="
Write-Host " Target model     : $TargetModel"
Write-Host " Tested reference : $TargetRepoRef"
Write-Host " Policy           : EXACT model only - no other quantisation or"
Write-Host "                    model is ever substituted. A missing file stays"
Write-Host "                    a loud LLM ERROR (no fallback profile exists)."
Write-Host " Repo root        : $Root"
Write-Host " Canonical path   : $CanonicalPath"
Write-Host "================================================================="
Write-Host ""
Write-Host "Searched locations:"
foreach ($s in $Searched) { Write-Host "  - $s" }
Write-Host ""

$Exact = New-Object System.Collections.ArrayList
$NotExact = 0

if ($Candidates.Count -eq 0) {
    Write-Host "GGUF files found : NONE (no *.gguf in any searched location)" -ForegroundColor Yellow
} else {
    Write-Host ("GGUF files found : {0}" -f $Candidates.Count)
    Write-Host ""
    foreach ($c in $Candidates) {
        $fi = Get-Item -LiteralPath $c.path
        $sizeBytes = $fi.Length
        $sizeGB = [math]::Round($sizeBytes / 1GB, 2)
        $id = Read-GgufIdentity -FilePath $c.path
        # identity = the Qwen3.6 35B A3B MODEL; quant = IQ4_XS - the two are
        # checked SEPARATELY so that an IQ4_XS of another model can never be
        # classified as an exact match of the target.
        $fileModelOk  = Test-TargetModelName $fi.Name
        $fileQuantOk  = Test-TargetQuantName $fi.Name
        $metaModelOk  = $false
        if (($null -ne $id.name) -and ($id.magic)) { $metaModelOk = Test-TargetModelName $id.name }
        $ollamaTagOk  = ($c.source -like "ollama*") -and ($c.source -like "*:IQ4_XS")
        $modelOk = ($fileModelOk -or $metaModelOk -or $ollamaTagOk)
        $headerQuantOk = ($id.file_type -eq $Iq4XsFileType)
        $quantOk = ($fileQuantOk -or $ollamaTagOk -or $headerQuantOk)
        $class = ""
        $note = ""
        if (-not $id.magic) {
            $class = "NOT a GGUF"
            $note = "the file does not start with the GGUF magic bytes"
        } elseif ($sizeBytes -lt $MinTargetBytes) {
            $class = "TOO SMALL"
            $note = ("only {0} GB - the 35B IQ4_XS file is ~19-20 GB" -f $sizeGB)
        } elseif ($modelOk -and $quantOk) {
            if ($headerQuantOk) {
                $class = "EXACT MATCH"
                $note = "GGUF header says IQ4_XS (general.file_type 30)"
            } else {
                $class = "EXACT MATCH (by name)"
                $note = "quant not stated in the header; filename/Ollama tag says IQ4_XS"
            }
        } elseif ($modelOk) {
            $class = "DIFFERENT QUANTISATION"
            if ($id.file_type -ge 0) {
                $q = $QuantLabels[[int]$id.file_type]
                if (-not $q) { $q = ("file_type {0}" -f $id.file_type) }
                $note = "same model but $q - NOT selected"
            } else {
                $note = "same model, quant unknown - NOT selected without proof"
            }
        } elseif ($quantOk) {
            $class = "DIFFERENT MODEL"
            $note = "IQ4_XS quant but NOT the Qwen3.6 35B A3B model"
            if ($null -ne $id.name) { $note = $note + " (general.name = " + $id.name + ")" }
        } else {
            $class = "DIFFERENT MODEL"
            if ($null -ne $id.name) { $note = ("general.name = {0}" -f $id.name) }
            else { $note = "name and size do not match the target model" }
        }
        if ($id.error -and ($class -ne "NOT a GGUF") -and ($class -ne "EXACT MATCH")) {
            $note = $note + " (partial header read)"
        }
        $ggufName = "(unreadable)"
        if ($null -ne $id.name) { $ggufName = $id.name }
        Write-Host ("  [{0}]" -f $class)
        Write-Host ("     path    : {0}" -f $c.path)
        Write-Host ("     size    : {0} GB   source: {1}" -f $sizeGB, $c.source)
        Write-Host ("     gguf    : general.name = {0}" -f $ggufName)
        Write-Host ("     note    : {0}" -f $note)
        Write-Host ""
        if ($class -like "EXACT MATCH*") {
            [void]$Exact.Add(@{ path = $c.path; source = $c.source; size = $sizeBytes; name = $ggufName; verified = ($id.file_type -eq $Iq4XsFileType) })
        } else {
            $NotExact++
        }
    }
}

# --- result: FOUND / MISSING -------------------------------------------------
if ($Exact.Count -ge 1) {
    Write-Host "================================================================="
    Write-Host (" RESULT: MODEL FILE FOUND - {0} exact match(es)" -f $Exact.Count)
    Write-Host "================================================================="
    foreach ($e in $Exact) {
        Write-Host ("  Path         : {0}" -f $e.path)
        Write-Host ("  Size         : {0} GB" -f [math]::Round($e.size / 1GB, 2))
        Write-Host ("  Source       : {0}" -f $e.source)
        if ($e.verified) {
            Write-Host "  Quant proof  : GGUF header general.file_type = 30 (IQ4_XS)"
        } else {
            Write-Host "  Quant proof  : filename/Ollama-tag (header does not state it)"
        }
        Write-Host ""
    }
    if ($Exact.Count -gt 1) {
        Write-Host " MULTIPLE exact matches - decide which one to use, then either copy it"
        Write-Host " manually to the canonical path or set LLAMA_MODEL_PATH by hand."
        exit 1
    }
    $p = $Exact[0].path
    if ($p -eq $CanonicalPath) {
        Write-Host " The file is ALREADY at the canonical location - the active profile"
        Write-Host " (config\env.local.ps1 LLAMA_MODEL_PATH) is already correct."
    } else {
        Write-Host " Option A (zero disk use): point the profile at the discovered file:"
        Write-Host "     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1 -SetEnv"
        Write-Host "   (edits ONLY the LLAMA_MODEL_PATH line in config\env.local.ps1, with a"
        Write-Host "    .bak backup; llama-server reads the file directly - Ollama is NOT"
        Write-Host "    part of the runtime, its blob is just a stored GGUF)"
        Write-Host "   by hand, the line looks like:"
        Write-Host ('     $env:LLAMA_MODEL_PATH = "{0}"' -f $p)
        Write-Host ""
        Write-Host " Option B (safer; ~20 GB free needed): copy to the canonical place:"
        Write-Host "     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1 -CopyToRepo"
        Write-Host "   (a copy survives Ollama garbage collection / re-pull)"
    }
    Write-Host ""
    Write-Host " Then START the server and VERIFY the exact file is loaded:"
    Write-Host "     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_llama_server.ps1"
    Write-Host "   - the starter echoes the full llama-server command line: the exact"
    Write-Host "     GGUF path must appear in it; logs\llama-server.out.log shows the load"
    Write-Host "     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify_m1.ps1"
    Write-Host "   - 'LLM GGUF (Qwen3.6 35B A3B IQ4_XS ...)' must PASS"
    Write-Host "   - 'LLM model loaded (GET /v1/models)' must show qwen3.6-35b-a3b"
} else {
    Write-Host "================================================================="
    Write-Host " RESULT: MODEL FILE MISSING - the integration STOPS here"
    Write-Host "================================================================="
    Write-Host " The exact Qwen3.6 35B A3B IQ4_XS GGUF was NOT found in any searched"
    Write-Host " location. Per the v0.4.15 policy NO other model and NO other"
    Write-Host " quantisation is substituted: the active profile keeps pointing at"
    Write-Host ("   {0}" -f $CanonicalPath)
    Write-Host " and a missing file remains a loud LLM ERROR (there is no"
    Write-Host " fallback profile - the Qwen3.6 is the ONE AND ONLY LLM)."
    Write-Host ""
    Write-Host " Download the EXACT file (~19-20 GB) from the exact repository the"
    Write-Host " model was tested with:"
    Write-Host ("   {0}" -f $TargetHfUrl)
    Write-Host ("   file: {0}   (quant tag: IQ4_XS)" -f $TargetHfFile)
    Write-Host " and place it at:"
    Write-Host ("   {0}" -f $CanonicalPath)
    Write-Host " (or anywhere on disk, then re-run with -SetEnv / set LLAMA_MODEL_PATH)"
    Write-Host ""
    Write-Host " To scan another drive/folder as well, re-run with -Path, e.g.:"
    Write-Host "     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1 -Path F:\"
    if ($NotExact -gt 0) {
        Write-Host ""
        Write-Host (" NOTE: {0} other GGUF file(s) were found but classified as different" -f $NotExact)
        Write-Host " model(s)/quantisation(s) - they are NOT selected."
    }
    exit 1
}

# --- optional actions --------------------------------------------------------
if ($CopyToRepo) {
    if ($p -eq $CanonicalPath) {
        Write-Host " -CopyToRepo: the file is already at the canonical path - nothing to copy."
        exit 0
    }
    $srcLen = $Exact[0].size
    $null = New-Item -ItemType Directory -Force -Path $CanonicalDir
    $free = -1
    try {
        $rootResolved = (Resolve-Path -LiteralPath $CanonicalDir).Path
        $drive = New-Object System.IO.DriveInfo([System.IO.Path]::GetPathRoot($rootResolved))
        $free = $drive.AvailableFreeSpace
    } catch { $free = -1 }
    if (($free -ge 0) -and ($free -lt ($srcLen + 1GB))) {
        Write-Host "COPY FAILED: not enough free space on the repo drive." -ForegroundColor Red
        Write-Host ("  needed ~{0} GB, available {1} GB" -f [math]::Round(($srcLen + 1GB) / 1GB, 1), [math]::Round($free / 1GB, 1))
        Write-Host "  -> use -SetEnv instead (Option A, zero disk use)."
        exit 1
    }
    Write-Host ("Copying {0} GB - this may take several minutes..." -f [math]::Round($srcLen / 1GB, 1))
    Copy-Item -LiteralPath $p -Destination $CanonicalPath -Force
    $okLen = ((Get-Item -LiteralPath $CanonicalPath).Length -eq $srcLen)
    $id2 = Read-GgufIdentity -FilePath $CanonicalPath
    if ($okLen -and $id2.magic) {
        Write-Host "COPY PASS: the exact GGUF is now at the canonical location:" -ForegroundColor Green
        Write-Host ("  {0}" -f $CanonicalPath)
        Write-Host " Start llama-server and verify with scripts\verify_m1.ps1."
        exit 0
    }
    Write-Host "COPY FAILED: the copied file does not match the source (size or GGUF magic)." -ForegroundColor Red
    exit 1
}

if ($SetEnv) {
    if ($p -eq $CanonicalPath) {
        Write-Host " -SetEnv: the discovered file is the canonical path - env.local.ps1 is"
        Write-Host "  already correct, nothing to change."
        exit 0
    }
    $envFile = Join-Path $Root "config\env.local.ps1"
    if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
        Write-Host "ENV WRITE FAILED: config\env.local.ps1 not found." -ForegroundColor Red
        exit 1
    }
    $lines = [System.IO.File]::ReadAllLines($envFile)
    # v0.4.17: the phantom project-relative default line is SHIPPED COMMENTED
    # OUT, so 0 active LLAMA_MODEL_PATH lines is now the NORMAL state - the
    # writer replaces the single active line when present, and INSERTS one
    # right after $env:OPENAI_MODEL when absent (never fails on 0).
    $replaced = 0
    $insertAt = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s*\$env:LLAMA_MODEL_PATH\s*=') {
            $lines[$i] = '$env:LLAMA_MODEL_PATH = "' + $p + '"'
            $replaced++
        }
        if (($insertAt -lt 0) -and ($lines[$i] -match '^\s*\$env:OPENAI_MODEL\s*=')) {
            $insertAt = $i + 1
        }
    }
    if ($replaced -eq 0) {
        if ($insertAt -lt 0) { $insertAt = $lines.Count }
        $newLine = '$env:LLAMA_MODEL_PATH = "' + $p + '"'
        $list = New-Object System.Collections.Generic.List[string]
        $list.AddRange([string[]]$lines)
        $list.Insert($insertAt, $newLine)
        $lines = @($list)
        $replaced = 1
        Write-Host " -SetEnv: no active LLAMA_MODEL_PATH line was present (v0.4.17 default)"
        Write-Host "  - the line is INSERTED after OPENAI_MODEL."
    }
    if ($replaced -ne 1) {
        Write-Host ("ENV WRITE FAILED: expected 0 or 1 active LLAMA_MODEL_PATH line, found {0}." -f $replaced) -ForegroundColor Red
        Write-Host "  Edit config\env.local.ps1 by hand (LLM-MODELPROFIL block)."
        exit 1
    }
    Copy-Item -LiteralPath $envFile -Destination ($envFile + ".bak") -Force
    [System.IO.File]::WriteAllLines($envFile, $lines, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "ENV SET: config\env.local.ps1 LLAMA_MODEL_PATH now points at the exact file" -ForegroundColor Green
    Write-Host ('  $env:LLAMA_MODEL_PATH = "{0}"' -f $p)
    Write-Host ("  (backup written: {0}.bak)" -f $envFile)
    Write-Host "  NOTE: a user selection in config\llm_model.json takes precedence over"
    Write-Host "  the env profile (resolution: llm_model.json > env > operator dir)."
    Write-Host " Restart the llama-server (START.bat) for the change to take effect,"
    Write-Host " then verify the load with scripts\verify_m1.ps1."
    exit 0
}

exit 0
