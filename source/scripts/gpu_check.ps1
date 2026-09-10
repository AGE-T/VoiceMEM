# ===========================================================================
# gpu_check.ps1 - robust NVIDIA GPU detection library (v0.1.5)
#
# Dot-sourced by bootstrap.ps1 (log probe) and install_m1.ps1 (step 4).
# NEVER a user entry point (one-click rule: START.bat is the only entry).
#
# Why this library exists (root cause of the v0.1.4 false failure):
#   The old check relied on a SINGLE strategy: Get-Command nvidia-smi
#   (a PATH lookup inside the START.bat child process). The bootstrap log
#   proved that this lookup can fail in the START.bat chain environment
#   while the same command succeeds in an interactive PowerShell
#   (different PATH / PATHEXT / process bitness between the two). A
#   PATH-only probe is therefore NOT proof that the GPU or the driver is
#   missing - yet the old error message claimed exactly that.
#
# Resolution chain (each step is a fallback for the previous one):
#   1. Get-Command nvidia-smi          (PATH, incl. PATHEXT resolution)
#   2. standard absolute locations      (System32, NVSMI dirs, SysWOW64)
#   3. -SearchDirs parameter            (test hook only, empty in prod)
#   4. run nvidia-smi                   (exec + query validation)
#   5. GPU name, driver, VRAM           (nvidia-smi; NEVER WMI AdapterRAM
#                                        - uint32, wraps above 4 GiB)
#   6. Windows WMI fallback             (Win32_VideoController, name only)
#
# Status codes (distinct failures, never overclaiming):
#   OK                      GPU validated via nvidia-smi
#   GPU_NOT_FOUND           WMI ran and sees NO NVIDIA adapter
#   DRIVER_NOT_FOUND        WMI sees an NVIDIA adapter, driver unhealthy
#   NVIDIA_SMI_NOT_FOUND    WMI sees NVIDIA GPU, but nvidia-smi not found
#   NVIDIA_SMI_EXEC_FAILED  nvidia-smi found but could not be executed
#   NVIDIA_SMI_QUERY_FAILED nvidia-smi ran but its output is unusable
#   VRAM_QUERY_FAILED       name/driver OK, VRAM not parseable
#   GPU_VALIDATION_FAILED   cannot determine (e.g. WMI unavailable)
# ===========================================================================

# ---------------------------------------------------------------------------
# Find-NvidiaSmi
# Returns @{ Found; Path; Origin } - Origin: PATH | STANDARD_PATH | SEARCH_DIR
# ---------------------------------------------------------------------------
function Find-NvidiaSmi {
    param([string[]]$SearchDirs = @())

    $Cmd = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if ($Cmd -and $Cmd.Source) {
        return @{ Found = $true; Path = [string]$Cmd.Source; Origin = 'PATH' }
    }

    # Standard absolute locations (NEVER rely on PATH alone: the child
    # process environment of the START.bat chain can differ from an
    # interactive shell; C:\Windows\System32\nvidia-smi.exe is the canonical
    # location of a healthy 64-bit NVIDIA driver install).
    # VM_GPU_TEST_NO_STD_PATHS is a TEST-ONLY hook (set by the validation
    # suite) that suppresses the standard-path fallback so the failure
    # scenarios stay hermetic on machines that DO have a working nvidia-smi.
    if (-not $env:VM_GPU_TEST_NO_STD_PATHS) {
        $StdPaths = @()
        if ($env:SystemRoot) {
            $StdPaths += (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe')
            $StdPaths += (Join-Path $env:SystemRoot 'SysWOW64\nvidia-smi.exe')
        }
        if ($env:ProgramFiles) {
            $StdPaths += (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
        }
        $Pf86 = ${env:ProgramFiles(x86)}
        if ($Pf86) {
            $StdPaths += (Join-Path $Pf86 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
        }
        foreach ($P in $StdPaths) {
            if (Test-Path -LiteralPath $P) {
                return @{ Found = $true; Path = $P; Origin = 'STANDARD_PATH' }
            }
        }
    }

    # Extra search dirs (test hook; empty in production).
    foreach ($D in $SearchDirs) {
        if ([string]::IsNullOrWhiteSpace($D)) { continue }
        $P = Join-Path $D 'nvidia-smi.exe'
        if (-not (Test-Path -LiteralPath $P)) { $P = Join-Path $D 'nvidia-smi' }
        if (Test-Path -LiteralPath $P) {
            return @{ Found = $true; Path = $P; Origin = 'SEARCH_DIR' }
        }
    }
    return @{ Found = $false; Path = ''; Origin = '' }
}

# ---------------------------------------------------------------------------
# Invoke-NvidiaSmiQuery
# Runs nvidia-smi and parses name / driver / VRAM.
# Returns @{ Code; Ok; ExitCode; Name; Driver; VramMiB; Stdout; Detail }
# ---------------------------------------------------------------------------
function Invoke-NvidiaSmiQuery {
    param([string]$SmiPath)

    $QueryArgs = @('--query-gpu=name,driver_version,memory.total', '--format=csv,noheader')
    $Raw = ''
    $Threw = ''
    $Exit = -1
    # Local EAP override: bootstrap.ps1 runs with ErrorActionPreference=Stop;
    # in PS 5.1 a native command writing to a REDIRECTED stderr under
    # EAP=Stop raises NativeCommandError. 'Continue' + 2>&1 merge + try/catch
    # is deterministic in both PS 5.1 and PS 7.
    $PrevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $Raw = (& $SmiPath @QueryArgs 2>&1 | Out-String)
        $Exit = $LASTEXITCODE
    } catch {
        $Threw = $_.Exception.Message
    } finally {
        $ErrorActionPreference = $PrevEap
    }
    $Text = ([string]$Raw).Trim()

    if ($Threw -ne '') {
        return @{ Code = 'NVIDIA_SMI_EXEC_FAILED'; Ok = $false; ExitCode = -1; Name = ''; Driver = ''; VramMiB = 0; Stdout = $Text; Detail = ('execution threw: ' + $Threw) }
    }
    if ($Exit -ne 0) {
        return @{ Code = 'NVIDIA_SMI_EXEC_FAILED'; Ok = $false; ExitCode = $Exit; Name = ''; Driver = ''; VramMiB = 0; Stdout = $Text; Detail = ('nvidia-smi exit code ' + $Exit) }
    }
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return @{ Code = 'NVIDIA_SMI_QUERY_FAILED'; Ok = $false; ExitCode = $Exit; Name = ''; Driver = ''; VramMiB = 0; Stdout = ''; Detail = 'nvidia-smi produced no output' }
    }

    $Parts = $Text -split ','
    if ($Parts.Count -lt 3) {
        return @{ Code = 'NVIDIA_SMI_QUERY_FAILED'; Ok = $false; ExitCode = $Exit; Name = ''; Driver = ''; VramMiB = 0; Stdout = $Text; Detail = 'output does not have the name,driver_version,memory.total CSV structure' }
    }
    $Name = ([string]$Parts[0]).Trim()
    $Driver = ([string]$Parts[1]).Trim()
    $VramText = ([string]$Parts[2]).Trim()
    if ([string]::IsNullOrWhiteSpace($Name)) {
        return @{ Code = 'NVIDIA_SMI_QUERY_FAILED'; Ok = $false; ExitCode = $Exit; Name = ''; Driver = ''; VramMiB = 0; Stdout = $Text; Detail = 'no GPU name in output' }
    }
    $VramMiB = 0
    if ($VramText -match '(\d+)\s*MiB') {
        $VramMiB = [int]$Matches[1]
    } else {
        return @{ Code = 'VRAM_QUERY_FAILED'; Ok = $false; ExitCode = $Exit; Name = $Name; Driver = $Driver; VramMiB = 0; Stdout = $Text; Detail = ('VRAM column not parseable: ' + $VramText) }
    }
    return @{ Code = 'OK'; Ok = $true; ExitCode = $Exit; Name = $Name; Driver = $Driver; VramMiB = $VramMiB; Stdout = $Text; Detail = '' }
}

# ---------------------------------------------------------------------------
# Get-WmiNvidiaGpus
# Returns an array of @{ Name; DriverVersion; ConfigManagerErrorCode }
# records for NVIDIA video controllers, or $null when WMI/CIM is
# unavailable (non-Windows / CIM error).
# NOTE: AdapterRAM is deliberately NOT read (uint32, wraps above 4 GiB).
# ---------------------------------------------------------------------------
function Get-WmiNvidiaGpus {
    try {
        $All = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop
        $Nv = @($All | Where-Object { ($_.Name -ne $null) -and ([string]$_.Name -match 'NVIDIA') })
        $Out = @()
        foreach ($C in $Nv) {
            $CmErr = 0
            try { $CmErr = [int]$C.ConfigManagerErrorCode } catch { $CmErr = -1 }
            $Out += @{
                Name = [string]$C.Name
                DriverVersion = [string]$C.DriverVersion
                ConfigManagerErrorCode = $CmErr
            }
        }
        return ,@($Out)
    } catch {
        return $null
    }
}

# ---------------------------------------------------------------------------
# Resolve-NvidiaGpuStatus
# Orchestrates the full chain and returns a taxonomy-coded status:
# @{ Code; Ok; GpuName; Driver; VramMiB; SmiPath; SmiOrigin; SmiExitCode;
#    SmiOutput; WmiSeen; WmiName; Detail }
# $WmiGpus: injectable WMI records for tests. $null (default) = query WMI
# here (production); an empty array = "WMI ran and found nothing".
# ---------------------------------------------------------------------------
function Resolve-NvidiaGpuStatus {
    param([object]$WmiGpus = $null, [string[]]$SearchDirs = @())

    $Status = @{
        Code = 'GPU_VALIDATION_FAILED'; Ok = $false
        GpuName = ''; Driver = ''; VramMiB = 0
        SmiPath = ''; SmiOrigin = ''; SmiExitCode = 0; SmiOutput = ''
        WmiSeen = $false; WmiName = ''; Detail = ''
    }

    $Smi = Find-NvidiaSmi -SearchDirs $SearchDirs
    if ($Smi.Found) {
        $Status.SmiPath = [string]$Smi.Path
        $Status.SmiOrigin = [string]$Smi.Origin
        $Q = Invoke-NvidiaSmiQuery -SmiPath ([string]$Smi.Path)
        $Status.SmiExitCode = $Q.ExitCode
        $Status.SmiOutput = [string]$Q.Stdout
        if ($Q.Ok) {
            $Status.Code = 'OK'
            $Status.Ok = $true
            $Status.GpuName = [string]$Q.Name
            $Status.Driver = [string]$Q.Driver
            $Status.VramMiB = [int]$Q.VramMiB
            return $Status
        }
        # Specific nvidia-smi failure (exec / query / vram). WMI context
        # enriches the detail but never overrides the specific code.
        $Status.Code = [string]$Q.Code
        $Status.Detail = [string]$Q.Detail
        # Keep whatever parsed (e.g. name+driver on VRAM_QUERY_FAILED).
        $Status.GpuName = [string]$Q.Name
        $Status.Driver = [string]$Q.Driver
        $Wmi = $WmiGpus
        if ($Wmi -eq $null) { $Wmi = Get-WmiNvidiaGpus }
        if ($Wmi -ne $null) {
            $Arr = @($Wmi)
            if ($Arr.Count -gt 0) {
                $Status.WmiSeen = $true
                $Status.WmiName = [string]$Arr[0].Name
            }
        }
        return $Status
    }

    # nvidia-smi NOT FOUND (PATH + standard locations + search dirs).
    $Wmi = $WmiGpus
    if ($Wmi -eq $null) { $Wmi = Get-WmiNvidiaGpus }
    if ($Wmi -eq $null) {
        $Status.Code = 'GPU_VALIDATION_FAILED'
        $Status.Detail = 'nvidia-smi not found and WMI is unavailable - cannot determine GPU presence'
        return $Status
    }
    $Arr = @($Wmi)
    if ($Arr.Count -eq 0) {
        $Status.Code = 'GPU_NOT_FOUND'
        $Status.Detail = 'WMI (Win32_VideoController) ran and reports no NVIDIA adapter'
        return $Status
    }
    $First = $Arr[0]
    $Status.WmiSeen = $true
    $Status.WmiName = [string]$First.Name
    $HasDriver = $false
    if (([string]::IsNullOrWhiteSpace([string]$First.DriverVersion) -eq $false) -and (([int]$First.ConfigManagerErrorCode) -eq 0)) {
        $HasDriver = $true
    }
    if (-not $HasDriver) {
        $Status.Code = 'DRIVER_NOT_FOUND'
        $Status.Detail = ('WMI sees NVIDIA adapter but the driver is not healthy (driver: ' + [string]$First.DriverVersion + ', CM error code: ' + [string]$First.ConfigManagerErrorCode + ')')
    } else {
        $Status.Code = 'NVIDIA_SMI_NOT_FOUND'
        $Status.Detail = 'GPU and driver are visible in WMI, but nvidia-smi was not found on PATH or in standard locations'
    }
    return $Status
}

# ---------------------------------------------------------------------------
# New-GpuDiagnostics
# Collects environment facts from THE SAME process that runs the check (the
# START.bat -> bootstrap -> installer child PowerShell), so the log proves
# what this exact environment can and cannot see (interactive shell vs
# START.bat chain comparison).
# ---------------------------------------------------------------------------
function New-GpuDiagnostics {
    param([object]$Status = $null)

    $L = @()
    $L += '--- GPU DIAGNOSTICS (installer process environment) ---'
    try { $L += ('PowerShell version : ' + $PSVersionTable.PSVersion.ToString()) } catch { $L += 'PowerShell version : unknown' }
    try { $L += ('PowerShell 64-bit  : ' + [Environment]::Is64BitProcess) } catch { $L += 'PowerShell 64-bit  : unknown' }
    try { $L += ('OS 64-bit          : ' + [Environment]::Is64BitOperatingSystem) } catch { }
    $L += ('PATH               : ' + [string]$env:PATH)
    $PathEntries = @([string]$env:PATH -split ';')
    $Sys32Hits = @($PathEntries | Where-Object { $_ -match '(?i)system32' })
    $L += ('PATH has System32  : ' + ($Sys32Hits.Count -gt 0))

    $Gc = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if ($Gc) {
        $L += ('Get-Command nvidia-smi : ' + [string]$Gc.Source)
    } else {
        $L += 'Get-Command nvidia-smi : NOT FOUND (in this process environment)'
    }
    $WhereExe = Get-Command 'where.exe' -ErrorAction SilentlyContinue
    if ($WhereExe) {
        try {
            $PrevEap = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try {
                $WhereOut = (& where.exe nvidia-smi 2>&1 | Out-String)
                $L += ('where.exe nvidia-smi  : exit=' + $LASTEXITCODE + ' output=' + ([string]$WhereOut).Trim())
            } catch {
                $L += ('where.exe nvidia-smi  : failed: ' + $_.Exception.Message)
            } finally {
                $ErrorActionPreference = $PrevEap
            }
        } catch { }
    }

    $StdProbes = @()
    if ($env:SystemRoot) {
        $StdProbes += (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe')
        $StdProbes += (Join-Path $env:SystemRoot 'SysWOW64\nvidia-smi.exe')
    }
    if ($env:ProgramFiles) {
        $StdProbes += (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    }
    foreach ($P in $StdProbes) {
        $L += ('Test-Path ' + $P + ' : ' + (Test-Path -LiteralPath $P))
    }

    if ($Status -ne $null) {
        $L += ('Status code        : ' + [string]$Status.Code)
        if ($Status.SmiPath) {
            $L += ('nvidia-smi path    : ' + [string]$Status.SmiPath + ' [' + [string]$Status.SmiOrigin + ']')
            $L += ('nvidia-smi exit    : ' + [string]$Status.SmiExitCode)
            $L += ('nvidia-smi output  : ' + [string]$Status.SmiOutput)
        }
    }

    try {
        $WmiAll = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop
        $Names = @($WmiAll | ForEach-Object { [string]$_.Name })
        $L += ('WMI video controllers : ' + ($Names -join ' | '))
    } catch {
        $L += 'WMI video controllers : unavailable in this environment'
    }
    return ($L -join [Environment]::NewLine)
}
