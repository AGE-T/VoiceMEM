<#
===============================================================================
scripts/identify_ollama_blob.ps1 - Ollama blob identification + direct
llama-server load verdict (v0.4.18)

v0.4.18 FIX (field report 2026-09): the GGUF metadata VALUE TYPE table was
wrong in every shipped walker (PowerShell + Python): they numbered
8=BOOL(1B), 9=STRING, 10=ARRAY, 7=FLOAT64(8B), 13=FLOAT16(2B). The real
GGUF spec (llama.cpp gguf-py constants.py GGUFValueType) is 7=BOOL(1B),
8=STRING(u64 len+bytes), 9=ARRAY(u32 elem type+u64 count+items),
10=UINT64(8B), 11=INT64(8B), 12=FLOAT64(8B) - there is NO type 13. On the
REAL blob the walk desynchronised at the first STRING KV (type 8 was read
as a 1-byte scalar), the next key length decoded as garbage, the exception
blanked the identity and the script stopped with IDENTITY MISMATCH even
though magic/version/manifest were all fine. The walker now follows the
authoritative table; the selection GATES (manifest identity, GGUF metadata
identity, sha256 digest) are UNCHANGED - no substitution, no blob copy.

SITUATION (field report, 2026-09): the tested model
    Qwen3.6 35B A3B IQ4_XS   (source: hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS)
was DOWNLOADED BY OLLAMA, so on this machine it is NOT a normally named
Qwen3.6-35B-A3B-IQ4_XS.gguf file. It is an OLLAMA CONTENT-ADDRESSED BLOB:

    C:\AI_HOME\models\blobs\sha256-afc7238af403ed454b7846454091b5e38b07575bdbb64c3e86777414dde4193c

The file NAME is a digest - it says NOTHING about the model inside. This
script therefore NEVER trusts the file name: it identifies the blob via the
OLLAMA MANIFEST (tag -> layer digests) and verifies the ACTUAL CONTENT via
the GGUF header (magic, general.name, general.architecture, general.file_type,
general.split.*) plus an independent sha256 digest check.

WHAT IT DECIDES:
  1. which exact blob is the tested model (identity + quantisation verified
     SEPARATELY - an IQ4_XS of a different model can never pass);
  2. whether llama-server can load that exact file DIRECTLY:
       - ONE model layer, complete single-file GGUF -> YES. llama.cpp checks
         the GGUF MAGIC, never the file name/extension: a sha256-... blob
         loads exactly like a named .gguf, IN PLACE, on any drive - no copy,
         no rename, no duplicate 20 GB file;
       - MULTIPLE model layers (Ollama split the GGUF into shards), or a
         single blob whose header carries general.split.no/count -> NO:
         llama.cpp discovers sibling shards by the
         <prefix>-NNNNN-of-NNNNN.gguf FILE-NAME pattern, which a
         sha256-... blob never matches. It STOPS and reports why - it does
         NOT copy/rename/duplicate the shards.
  3. Ollama itself is NOT a runtime dependency: the production architecture
     stays  VoiceMemAgent -> local llama-server -> Qwen3.6 35B A3B IQ4_XS.

PORTABILITY: C:\AI_HOME is only THIS machine's Ollama store (a search HINT).
The root resolves as  -OllamaRoot param > OLLAMA_MODELS env var >
C:\AI_HOME\models, and the PERSISTED configuration is the absolute blob path
(config/llm_model.json user selection - the same thing the web UI "LLM model"
Browse picker writes). The web UI picker remains the PREFERRED configuration
method; -Select here is its scripted equivalent (handy right after
identifying the blob). Any drive works - nothing is ever copied.

PARAMETERS
  -OllamaRoot <dir>  Ollama store root (default: OLLAMA_MODELS env, else
                     C:\AI_HOME\models - the location on THIS machine only)
  -Repo <ref>        repo filter (default: hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF;
                     matched case-insensitively by tokens qwen36/35b/a3b)
  -Tag <tag>         manifest tag to resolve (default: IQ4_XS)
  -SkipHash          skip the sha256 verification of the ~20 GB blob
  -Select            persist the identified path as the user selection
                     (config/llm_model.json - wins over the env profile)
  -SetEnv            rewrite ONLY the LLAMA_MODEL_PATH line in
                     config/env.local.ps1 (.bak backup; NOTE: a user
                     selection in llm_model.json takes precedence)

EXIT CODES
  0 = identified, verified, llama-server CAN load it directly
  1 = not found (no matching manifest / missing blob)
  2 = identified as a SPLIT shard set - llama-server CANNOT load it directly
  3 = digest mismatch (the blob is corrupt / incomplete)
  4 = identity mismatch (found, but NOT the exact tested model/quant - NO
      substitution is ever made)
===============================================================================
#>
param(
    [string]$OllamaRoot = "",
    [string]$Repo = "hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
    [string]$Tag = "IQ4_XS",
    [switch]$SkipHash,
    [switch]$Select,
    [switch]$SetEnv
)

$ErrorActionPreference = "Stop"

# --- target identity ---------------------------------------------------------
$TargetModel   = "Qwen3.6 35B A3B IQ4_XS"
$TargetRepoRef = "hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS"
$TargetHfUrl   = "https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF"
$Iq4XsFileType = 30   # GGUF general.file_type (llama_ftype; 29 would be IQ2_M)
$MinTargetBytes = 10GB  # the exact 35B A3B IQ4_XS is ~19-20 GB

$Root = Split-Path -Parent $PSScriptRoot

# GGUF file_type -> label (llama_ftype, same table as app/gguf.py)
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
    return ($s.ToLower() -replace "[^0-9a-z]", "")
}

Write-Host "==================================================================="
Write-Host " Ollama blob identification - $TargetModel"
Write-Host " tested source: $TargetRepoRef"
Write-Host " Ollama is NOT the runtime - llama-server loads the blob itself."
Write-Host "==================================================================="

# --- 1. resolve the Ollama store root ----------------------------------------
if (-not $OllamaRoot) { $OllamaRoot = $env:OLLAMA_MODELS }
if (-not $OllamaRoot) { $OllamaRoot = "C:\AI_HOME\models" }   # THIS machine's hint only
$BlobDir     = Join-Path $OllamaRoot "blobs"
$ManifestDir = Join-Path $OllamaRoot "manifests"
Write-Host ""
Write-Host ("Ollama store : {0}" -f $OllamaRoot)
Write-Host ("  resolved as: -OllamaRoot param > OLLAMA_MODELS env > C:\AI_HOME\models (hint)")
if (-not (Test-Path -LiteralPath $BlobDir -PathType Container)) {
    Write-Host "STOP: the Ollama blob directory does not exist:" -ForegroundColor Red
    Write-Host ("      {0}" -f $BlobDir)
    Write-Host "      Pass the real store root with -OllamaRoot (or set OLLAMA_MODELS)."
    exit 1
}
if (-not (Test-Path -LiteralPath $ManifestDir -PathType Container)) {
    Write-Host "STOP: the Ollama manifests directory does not exist:" -ForegroundColor Red
    Write-Host ("      {0}" -f $ManifestDir)
    exit 1
}

# --- 2. find the manifest of the tested model --------------------------------
# Manifests mirror the model reference as directories:
#   <root>\manifests\hf.co\bartowski\qwen_qwen3.6-35b-a3b-gguf\<tag>
# (case-insensitive matching - Ollama normalises repo paths to lowercase).
$TagNorm = Get-Norm $Tag
$Matches2 = New-Object System.Collections.ArrayList
$AllRepoTags = New-Object System.Collections.ArrayList
$mans = @(Get-ChildItem -LiteralPath $ManifestDir -Recurse -File -ErrorAction SilentlyContinue)
foreach ($m in $mans) {
    $repoDir = $m.DirectoryName
    if ($repoDir.Length -le ($ManifestDir.Length + 1)) { continue }
    $repoName = $repoDir.Substring($ManifestDir.Length + 1)
    $tagName = $m.Name
    $repoNorm = Get-Norm $repoName
    if (($repoNorm.Contains("qwen36")) -and ($repoNorm.Contains("35b")) -and ($repoNorm.Contains("a3b"))) {
        [void]$AllRepoTags.Add(("        {0}:{1}" -f $repoName, $tagName))
        if ((Get-Norm $tagName) -eq $TagNorm) {
            [void]$Matches2.Add(@{ repo = $repoName; tag = $tagName; file = $m.FullName })
        }
    }
}
Write-Host ""
if ($AllRepoTags.Count -gt 0) {
    Write-Host ("Qwen3.6 35B A3B manifest tag(s) found in the store ({0}):" -f $AllRepoTags.Count)
    foreach ($t in $AllRepoTags) { Write-Host $t }
} else {
    Write-Host "No Qwen*3.6*35B*A3B manifest in this Ollama store."
}
if ($Matches2.Count -eq 0) {
    Write-Host ""
    Write-Host ("STOP: no manifest matches the tested reference ($Repo tag $Tag).") -ForegroundColor Red
    Write-Host "      The model was either pulled under another Ollama store, or with"
    Write-Host "      a different tag. Per the policy NO other model and NO other"
    Write-Host "      quantisation is substituted - the integration STOPS here."
    Write-Host ("      Expected source: {0}" -f $TargetRepoRef)
    Write-Host "      Try:  -OllamaRoot <the real Ollama store>   -Tag <the real tag>"
    Write-Host "      Or use the general finder: scripts\find_qwen_gguf.ps1"
    exit 1
}
if ($Matches2.Count -gt 1) {
    Write-Host ""
    Write-Host "STOP: MULTIPLE manifests match the same repo+tag:" -ForegroundColor Red
    foreach ($x in $Matches2) { Write-Host ("        {0}:{1}" -f $x.repo, $x.tag) }
    Write-Host "      Resolve the duplicate store first (only one may stay active)."
    exit 1
}
$Man = $Matches2[0]
Write-Host ("Selected manifest: {0}:{1}" -f $Man.repo, $Man.tag)

# --- 3. manifest -> model-layer blobs -----------------------------------------
try {
    $MJson = Get-Content -LiteralPath $Man.file -Raw -Encoding UTF8 | ConvertFrom-Json
} catch {
    Write-Host ("STOP: the manifest JSON is unreadable: {0}" -f $Man.file) -ForegroundColor Red
    exit 1
}
$ModelLayers = @($MJson.layers | Where-Object { $_.mediaType -like "*ollama.image.model" })
if ($ModelLayers.Count -eq 0) {
    Write-Host "STOP: the manifest has NO model layer (unexpected Ollama layout)." -ForegroundColor Red
    exit 1
}
$BlobPaths = New-Object System.Collections.ArrayList
$Missing = New-Object System.Collections.ArrayList
$i = 0
foreach ($layer in $ModelLayers) {
    $digest = [string]$layer.digest
    if (-not $digest.StartsWith("sha256:")) {
        Write-Host ("STOP: unexpected layer digest '{0}'" -f $digest) -ForegroundColor Red
        exit 1
    }
    $hex = $digest.Substring(7)
    $bp = Join-Path $BlobDir ("sha256-" + $hex)
    $size = [long]$layer.size
    if (Test-Path -LiteralPath $bp -PathType Leaf) {
        [void]$BlobPaths.Add(@{ path = $bp; digest = $digest; hex = $hex; declaredSize = $size; index = $i })
    } else {
        [void]$Missing.Add($bp)
    }
    $i++
}
if ($Missing.Count -gt 0) {
    Write-Host ""
    Write-Host "STOP: the manifest references a blob that is MISSING on disk:" -ForegroundColor Red
    foreach ($x in $Missing) { Write-Host ("        {0}" -f $x) }
    Write-Host "      Re-pull the model with Ollama (or re-download the GGUF from"
    Write-Host ("      {0}) and re-run this script." -f $TargetHfUrl)
    exit 1
}

# --- 4. read the GGUF header of every model-layer blob -------------------------
# GGUF v2/v3: magic("GGUF") + u32 version + u64 tensor_count + u64 kv_count,
# then kv_count pairs of (u64 key_len + key + u32 value_type + value).
# VALUE TYPES - the ACTUAL GGUF spec table (llama.cpp gguf-py constants.py
# GGUFValueType; the v0.4.16/17 walkers had it WRONG, which is exactly why
# the real blob's identity came back EMPTY):
#   0=UINT8(1B) 1=INT8(1B) 2=UINT16(2B) 3=INT16(2B) 4=UINT32(4B) 5=INT32(4B)
#   6=FLOAT32(4B) 7=BOOL(1B) 8=STRING(u64 len + bytes) 9=ARRAY(u32 elem type
#   + u64 count + items) 10=UINT64(8B) 11=INT64(8B) 12=FLOAT64(8B)
#   There is NO type 13.
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
    $out = @{
        magic = $false; version = 0; name = $null; architecture = $null
        file_type = -1; split_no = -1; split_count = -1; error = $false
    }
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
        $null = $br.ReadUInt64()             # tensor count
        $kvCount = $br.ReadUInt64()          # metadata KV count (u64)
        if ($kvCount -gt 65536) { throw ("absurd GGUF kv count {0}" -f $kvCount) }
        for ($k = 0; $k -lt [int]$kvCount; $k++) {
            $klen = $br.ReadUInt64()         # KEY LENGTH IS u64
            if ($klen -gt 4096) { throw ("absurd GGUF key length {0}" -f $klen) }
            $key = [System.Text.Encoding]::UTF8.GetString($br.ReadBytes([int]$klen))
            $vt = [int]$br.ReadUInt32()
            $strVal = $null
            $u32Val = -1
            if ($vt -eq 8) {                 # STRING: capture for identity keys
                $strVal = Read-GgufString $br
            } elseif ($vt -eq 4) {           # UINT32: file_type / split keys
                $u32Val = [int]$br.ReadUInt32()
            } else {
                Skip-GgufValue $br $vt
            }
            if ($null -ne $strVal) {
                if ($key -eq "general.name") { $out.name = $strVal }
                if ($key -eq "general.architecture") { $out.architecture = $strVal }
            }
            if ($u32Val -ge 0) {
                if ($key -eq "general.file_type")  { $out.file_type = $u32Val }
                if ($key -eq "general.split.no")    { $out.split_no = $u32Val }
                if ($key -eq "general.split.count") { $out.split_count = $u32Val }
            }
            if (($null -ne $out.name) -and ($null -ne $out.architecture) -and
                ($out.file_type -ge 0) -and ($out.split_no -ge 0) -and
                ($out.split_count -ge 0)) { break }
        }
        return $out
    } catch {
        $out.error = $true
        return $out
    } finally {
        if ($null -ne $fs) { $fs.Close() }
    }
}

Write-Host ""
Write-Host ("Model layers in the manifest: {0}" -f $BlobPaths.Count)
foreach ($b in $BlobPaths) {
    $id = Read-GgufIdentity -FilePath $b.path
    $sizeBytes = 0
    try { $sizeBytes = (Get-Item -LiteralPath $b.path).Length } catch { }
    $b.ggu = $id
    $b.size = $sizeBytes
    $isShard = (($id.split_no -ge 0) -or ($id.split_count -gt 1))
    if ($isShard) {
        Write-Host ("  layer {0}: {1}  (shard part {2} of {3})" -f $b.index, (Split-Path -Leaf $b.path), ($id.split_no + 1), $id.split_count) -ForegroundColor Yellow
    } else {
        Write-Host ("  layer {0}: {1}" -f $b.index, (Split-Path -Leaf $b.path))
    }
    Write-Host ("     size         : {0} GB ({1} bytes)" -f [math]::Round($sizeBytes / 1GB, 2), $sizeBytes)
    if ($id.magic) {
        Write-Host ("     GGUF header  : magic OK, version {0}" -f $id.version)
        if ($id.error) {
            Write-Host "     GGUF header  : metadata walk FAILED - identity unreadable" -ForegroundColor Red
        }
        Write-Host ("     general.name : {0}" -f $id.name)
        Write-Host ("     general.architecture : {0}" -f $id.architecture)
        $ql = "unknown"
        if ($id.file_type -ge 0) {
            if ($QuantLabels.ContainsKey($id.file_type)) { $ql = $QuantLabels[$id.file_type] } else { $ql = ("file_type {0}" -f $id.file_type) }
        }
        Write-Host ("     quantisation : {0} (general.file_type = {1})" -f $ql, $id.file_type)
        if ($id.split_no -ge 0) {
            Write-Host ("     split        : shard index {0} of {1} (general.split.no/count)" -f $id.split_no, $id.split_count)
        } else {
            Write-Host "     split        : none (complete single-file GGUF)"
        }
    } else {
        Write-Host "     GGUF header  : NOT a valid GGUF (magic missing or walk failed)" -ForegroundColor Red
    }
}

# --- 5. identity from the GGUF HEADER (never the file name) --------------------
$Main = $BlobPaths[0]
$MainId = $Main.ggu
if (-not $MainId.magic) {
    Write-Host ""
    Write-Host "STOP: the blob is not a readable GGUF file." -ForegroundColor Red
    Write-Host ("      {0}" -f $Main.path)
    exit 4
}
# MODEL identity (tokens in general.name/architecture - the sha256 file name
# is opaque and deliberately ignored):
$nameNorm = Get-Norm ("{0} {1}" -f $MainId.name, $MainId.architecture)
$ModelOk = ($nameNorm.Contains("qwen36")) -and ($nameNorm.Contains("35b")) -and ($nameNorm.Contains("a3b"))
# QUANT identity (checked SEPARATELY - IQ4_XS of a different model is not a match):
$QuantOk = ($MainId.file_type -eq $Iq4XsFileType)
if (-not ($ModelOk -and $QuantOk)) {
    Write-Host ""
    Write-Host "STOP: IDENTITY MISMATCH - the identified blob is NOT the exact tested model." -ForegroundColor Red
    Write-Host ("      header says : general.name={0}, architecture={1}, file_type={2}" -f $MainId.name, $MainId.architecture, $MainId.file_type)
    Write-Host ("      expected    : {0} (IQ4_XS, general.file_type = {1})" -f $TargetModel, $Iq4XsFileType)
    Write-Host "      NO other model and NO other quantisation is substituted."
    exit 4
}
if ($Main.size -lt $MinTargetBytes) {
    Write-Host ""
    Write-Host ("WARNING: the blob is only {0} GB - the exact 35B A3B IQ4_XS is ~19-20 GB." -f [math]::Round($Main.size / 1GB, 2)) -ForegroundColor Yellow
    Write-Host "         The header identity matches, so it is reported as found - verify the size."
}

# --- 6. digest verification (independent, unless -SkipHash) --------------------
$DigestOk = $null
if ($SkipHash) {
    Write-Host ""
    Write-Host "Digest check : SKIPPED (-SkipHash)"
} else {
    Write-Host ""
    Write-Host ("Digest check : hashing {0} GB (streaming, ~1-2 min)..." -f [math]::Round($Main.size / 1GB, 1))
    $actual = (Get-FileHash -LiteralPath $Main.path -Algorithm SHA256).Hash.ToLower() -replace "-", ""
    $DigestOk = ($actual -eq $Main.hex)
    if ($DigestOk) {
        Write-Host "  PASS: sha256(blob content) == manifest digest == blob file name" -ForegroundColor Green
    } else {
        Write-Host "  FAIL: the blob content does NOT match its digest!" -ForegroundColor Red
        Write-Host ("        expected (manifest) : {0}" -f $Main.hex)
        Write-Host ("        actual (content)    : {0}" -f $actual)
        Write-Host "  The blob is corrupt or incomplete - re-pull it with Ollama or"
        Write-Host ("  re-download the GGUF from {0}" -f $TargetHfUrl)
        exit 3
    }
}

# --- 7. llama-server direct-load verdict ---------------------------------------
$SplitInHeader = (($MainId.split_no -ge 0) -or ($MainId.split_count -gt 1))
$BlobCount = $BlobPaths.Count
$CanLoadDirect = (($BlobCount -eq 1) -and (-not $SplitInHeader) -and $MainId.magic)

Write-Host ""
Write-Host "==================================================================="
if ($CanLoadDirect) {
    Write-Host " VERDICT: llama-server CAN load this exact file DIRECTLY." -ForegroundColor Green
    Write-Host "==================================================================="
    Write-Host (" Selected LLM : {0}" -f $TargetModel)
    Write-Host (" GGUF         : {0}" -f (Split-Path -Leaf $Main.path))
    Write-Host (" Path         : {0}" -f $Main.path)
    Write-Host  " Status       : File found"
    Write-Host (" Size         : {0} GB ({1} bytes)" -f [math]::Round($Main.size / 1GB, 2), $Main.size)
    Write-Host (" Identity     : general.name = {0} (from the GGUF header - NOT the file name)" -f $MainId.name)
    Write-Host ("                general.architecture = {0}" -f $MainId.architecture)
    Write-Host (" Quantisation : IQ4_XS (general.file_type = {0})" -f $Iq4XsFileType)
    if ($null -ne $DigestOk) {
        Write-Host  " Digest       : sha256 matches the Ollama manifest AND the blob file name"
    }
    Write-Host  " Loadability  : complete single-file GGUF - llama.cpp checks the GGUF"
    Write-Host  "                magic, NEVER the file name: the sha256-... blob loads"
    Write-Host  "                in place, on any drive, with NO copy and NO rename."
    Write-Host  " Ollama       : NOT a runtime dependency - llama-server reads the blob"
    Write-Host  "                itself (production: VoiceMem -> llama-server -> Qwen3.6)"
    Write-Host  ""
    Write-Host  " The blob is never copied, renamed or uploaded - only its absolute"
    Write-Host  " path is persisted."
    if (-not ($Select -or $SetEnv)) {
        Write-Host ""
        Write-Host  " HOW TO CONFIGURE VoiceMem (the web UI picker stays the preferred way):"
        Write-Host  "   (a) Web UI -> 'LLM model' -> Browse... -> 'Ollama blobs (sha256-*)'"
        Write-Host  "       filter -> select this blob -> Use this model  (the path is"
        Write-Host  "       persisted to config\llm_model.json; survives restart+upgrade)"
        Write-Host  "   (b) scripted equivalent of (a):"
        Write-Host  "       powershell -NoProfile -ExecutionPolicy Bypass -File scripts\identify_ollama_blob.ps1 -Select"
        Write-Host  "   (c) point the env profile at it (a user selection overrides this):"
        Write-Host  "       powershell -NoProfile -ExecutionPolicy Bypass -File scripts\identify_ollama_blob.ps1 -SetEnv"
    }
} else {
    Write-Host " VERDICT: llama-server CANNOT load the Ollama blobs directly - STOP." -ForegroundColor Red
    Write-Host "==================================================================="
    if ($BlobCount -gt 1) {
        Write-Host (" The manifest splits the model into {0} SEPARATE blobs:" -f $BlobCount)
        foreach ($b in $BlobPaths) { Write-Host ("   {0}" -f $b.path) }
    } else {
        Write-Host (" The single blob is itself ONE SHARD (general.split.no={0}, count={1}):" -f $MainId.split_no, $MainId.split_count)
        Write-Host ("   {0}" -f $Main.path)
    }
    Write-Host ""
    Write-Host " WHY direct loading is impossible:"
    Write-Host "   llama.cpp's split-GGUF support discovers sibling shards by the"
    Write-Host "   FILE-NAME pattern  <prefix>-NNNNN-of-NNNNN.gguf  in the same"
    Write-Host "   directory. Ollama blobs are content-addressed sha256-... files"
    Write-Host "   with NO extension - the loader can never find the siblings, and"
    Write-Host "   passing a single shard alone fails (its tensor data is partial)."
    Write-Host ""
    Write-Host " WHAT WAS NOT DONE (per the directive):"
    Write-Host "   - no 20 GB copy was created merely to rename the shards"
    Write-Host "   - no duplicate GGUF was placed inside the project"
    Write-Host "   - no other model/quantisation was substituted"
    Write-Host ""
    Write-Host " REMEDIES (all need a one-time ~19-20 GB download; pick ONE):"
    Write-Host ("   1. download the COMPLETE single-file GGUF from the exact tested source:" -f "")
    Write-Host ("      {0}" -f $TargetHfUrl)
    Write-Host  "      file Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf (tag IQ4_XS), place it"
    Write-Host  "      anywhere (any drive), then select it with the Web UI picker"
    Write-Host  "      or:  scripts\find_qwen_gguf.ps1 -SetEnv"
    Write-Host  "   2. merge the shards into one file with llama.cpp's own tool"
    Write-Host  "      (llama-gguf-split --merge) - this DOES create a second ~20 GB"
    Write-Host  "      file, so it is only acceptable when the source blobs can be"
    Write-Host  "      deleted afterwards; the merge must be run by the operator."
    exit 2
}

# --- 8. optional persistence ---------------------------------------------------
$Persisted = $false
if ($Select) {
    $cfgDir = Join-Path $Root "config"
    if (-not (Test-Path -LiteralPath $cfgDir)) { $null = New-Item -ItemType Directory -Path $cfgDir -Force }
    $SelFile = Join-Path $cfgDir "llm_model.json"
    $payload = @{
        llm_model_path = $Main.path
        selected_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
        model_name = [string]$MainId.name
        quant = "IQ4_XS"
        file_size = $Main.size
    }
    $json = $payload | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText($SelFile, $json + "`r`n", (New-Object System.Text.UTF8Encoding($false)))
    $Persisted = $true
    Write-Host ""
    Write-Host "SELECTED: the exact blob path is persisted as the user selection:" -ForegroundColor Green
    Write-Host ("  {0}" -f $SelFile)
    Write-Host ("  llm_model_path = {0}" -f $Main.path)
    Write-Host "  (the web UI 'LLM model' section reads the same file; a user"
    Write-Host "   selection WINS over the LLAMA_MODEL_PATH env profile; it survives"
    Write-Host "   restart AND extract-over upgrades; the GGUF is never copied)"
}
if ($SetEnv) {
    $envFile = Join-Path $Root "config\env.local.ps1"
    if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
        Write-Host "ENV WRITE FAILED: config\env.local.ps1 not found." -ForegroundColor Red
        exit 1
    }
    $lines = [System.IO.File]::ReadAllLines($envFile)
    # v0.4.17: the shipped env.local.ps1 has NO active LLAMA_MODEL_PATH line
    # (the phantom project-relative default is commented out) - the writer
    # replaces the single active line when present, and INSERTS one right
    # after $env:OPENAI_MODEL when absent.
    $replaced = 0
    $insertAt = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s*\$env:LLAMA_MODEL_PATH\s*=') {
            $lines[$i] = '$env:LLAMA_MODEL_PATH = "' + $Main.path + '"'
            $replaced++
        }
        if (($insertAt -lt 0) -and ($lines[$i] -match '^\s*\$env:OPENAI_MODEL\s*=')) {
            $insertAt = $i + 1
        }
    }
    if ($replaced -eq 0) {
        if ($insertAt -lt 0) { $insertAt = $lines.Count }
        $newLine = '$env:LLAMA_MODEL_PATH = "' + $Main.path + '"'
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
        exit 1
    }
    Copy-Item -LiteralPath $envFile -Destination ($envFile + ".bak") -Force
    [System.IO.File]::WriteAllLines($envFile, $lines, (New-Object System.Text.UTF8Encoding($false)))
    $Persisted = $true
    Write-Host ""
    Write-Host "ENV SET: config\env.local.ps1 LLAMA_MODEL_PATH now points at the exact blob" -ForegroundColor Green
    Write-Host ('  $env:LLAMA_MODEL_PATH = "{0}"' -f $Main.path)
    Write-Host ("  (backup written: {0}.bak)" -f $envFile)
    Write-Host "  NOTE: a user selection in config\llm_model.json takes precedence over"
    Write-Host "  the env profile (resolution: llm_model.json > env > operator dir)."
}
if ($Persisted) {
    Write-Host ""
    Write-Host " NEXT: restart the stack and verify the ACTUAL RUNNING SERVER:"
    Write-Host "   1. START.bat (scripts\start_llama_server.ps1 restarts llama-server"
    Write-Host "      with the configured path; a stale server with a different model"
    Write-Host "      is stopped automatically)"
    Write-Host "   2. the starter PROVES the load and writes it to"
    Write-Host "      logs\llama-server.resolved-model.json:"
    Write-Host "        - the served model id from GET /v1/models"
    Write-Host "        - the server's own log line containing the exact file path"
    Write-Host "        - a real /v1/chat/completions round-trip"
    Write-Host "   3. the Web UI 'LLM model' section shows the same proof lines"
    Write-Host "      ('llama-server loads the same file')"
    Write-Host "   4. powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify_m1.ps1"
    Write-Host "      -> 'LLM GGUF (Qwen3.6 35B A3B IQ4_XS ...)' must PASS"
}
exit 0
