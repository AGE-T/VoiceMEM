<#
===============================================================================
scripts/offline_check.ps1 - OS-szintu offline audit (netstatus capture)

Mi ez: a tests/integration/offline_test.py (Python-szintu socket/DNS audit) TARSKARSAZOJA
OS-szinten - spec 21.2.4 (21.2 4. pont): heti offline audit, hogy a zero
runtime network dependency biztositott legyen. A Python-teszt a sajat
folyamata halozati hivasait meri; ez a szkript az egesz gep letesitett
TCP kapcsolatait nezni.

VARHATO MUKODES: az M1 agent pontosan EGY loopback kapcsolatot tart fenn -
a 127.0.0.1:8080-t (llama-server). Minden loopback (127.x, ::1) cim
engedelyezett/varhato; minden NEM-loopback tavoli cim VIOLACIO
(kilepesi kod 1 + a lista).

Ket fazis:
  A) Alap (nincs -Capture): pillanatfelvetel - kiirja a letesitett
     (Established) TCP kapcsolatokat es megnezi, van-e nem-loopback tavoli
     cim. Kilepes: 0 = tiszta, 1 = van violacio.
  B) -Capture -Seconds N : hatter-munka (background job) 2 masodpercenkent
     mintavetel N masodpercig, amig te a MASIK ablakban beszelgetsz az
     agenttel (start_agent.ps1). A vegen az egyedi nem-loopback tavoli
     cimeket jelenti. Kilepes ugyanaz.

Pelda (a tests/integration/offline_test.py is ezt irja ki):
    scripts/offline_check.ps1 -Capture -Seconds 30

Alternativ hasznalat:
    .\scripts\offline_check.ps1                       # gyors pillanatfelvetel
    .\scripts\offline_check.ps1 -Capture -Seconds 60  # 1 perces capture

Megjegyzes: regebbi Windows eseten (nincs Get-NetTCPConnection) a netstat
fallback: netstat -an | findstr ESTABLISHED.
Megjegyzes: a fajl szandekosan ASCII - PowerShell 5.1 BOM nelkul
Windows-1252-kent olvasna az ekezetes karaktereket.
#>
param(
    [switch]$Capture,
    [int]$Seconds = 30
)

# ---------------------------------------------------------------------------
# Test-IsLoopback - "cim:port" tavoli vegpont loopback-e?
#   127.x.x.x  -> loopback (IPv4)
#   ::1        -> loopback (IPv6, "0:0:0:0:0:0:0:1" alakban is)
# ---------------------------------------------------------------------------
function Test-IsLoopback {
    param([string]$RemoteEndPoint)
    if (-not $RemoteEndPoint) { return $true }
    $Addr = $RemoteEndPoint
    if ($Addr.StartsWith("[")) {
        # IPv6 zarojeles alak: "[::1]:8080"
        $Close = $Addr.IndexOf("]")
        if ($Close -gt 1) { $Addr = $Addr.Substring(1, $Close - 1) }
    }
    else {
        # "127.0.0.1:8080" vagy "::1:8080" - a legutolso kettospont utan a port
        $Colon = $Addr.LastIndexOf(":")
        if ($Colon -gt 0) { $Addr = $Addr.Substring(0, $Colon) }
    }
    if ($Addr -eq "::1" -or $Addr -eq "0:0:0:0:0:0:0:1") { return $true }
    return $Addr.StartsWith("127.")
}

# ---------------------------------------------------------------------------
# Get-EstablishedEndpoints - letesitett TCP kapcsolatok tavoli vegpontjai
# (Get-NetTCPConnection ha van, kulonben netstat -an | findstr ESTABLISHED)
# ---------------------------------------------------------------------------
function Get-EstablishedEndpoints {
    $Endpoints = @()
    $HasCmdlet = [bool](Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)
    if ($HasCmdlet) {
        $Conns = Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue
        foreach ($Conn in $Conns) {
            $Endpoints += ("{0}:{1}" -f $Conn.RemoteAddress, $Conn.RemotePort)
        }
    }
    else {
        # regebbi Windows fallback
        $Lines = netstat -an | findstr ESTABLISHED
        foreach ($Line in $Lines) {
            $Parts = $Line.Trim() -split '\s+'
            if ($Parts.Length -ge 4) { $Endpoints += $Parts[2] }
        }
    }
    return , $Endpoints
}

function Show-Report {
    param([string[]]$Endpoints, [string]$Title)
    Write-Host ""
    Write-Host "================================================================="
    Write-Host "$Title"
    Write-Host "================================================================="
    if ($Endpoints.Count -eq 0) {
        Write-Host "Nincs letesitett (Established) TCP kapcsolat."
        Write-Host "EREDMENY: PASS - nincs nem-loopback kapcsolat."
        return 0
    }
    $Violations = @()
    foreach ($Ep in $Endpoints) {
        if (Test-IsLoopback $Ep) {
            Write-Host "  OK (loopback, varhato): $Ep"
        }
        else {
            $Violations += $Ep
            Write-Host "  VIOLACIO (nem loopback): $Ep" -ForegroundColor Red
        }
    }
    Write-Host "Megjegyzes: a 127.0.0.1:8080 / [::1]:8080 (llama-server) loopback"
    Write-Host "kapcsolat az agenttol VARHATO es engedelyezett."
    if ($Violations.Count -gt 0) {
        Write-Host ""
        Write-Host "EREDMENY: FAIL - nem-loopback tavoli cimek ($($Violations.Count) db):" -ForegroundColor Red
        foreach ($V in $Violations) { Write-Host "  $V" }
        return 1
    }
    Write-Host ""
    Write-Host "EREDMENY: PASS - minden megfigyelt tavoli cim loopback."
    return 0
}

if ($Capture) {
    # -----------------------------------------------------------------------
    # Fazis B: capture N masodpercig, 2 s-onkenti mintavetel (hatter-munka)
    # -----------------------------------------------------------------------
    Write-Host ""
    Write-Host "Capture mod: $Seconds masodpercig 2 s-onkent mintavetel."
    Write-Host "Most beszelj az agenttel a MASIK ablakban (scripts/start_agent.ps1)."
    Write-Host "A capture magatol leall $Seconds masodperc mulva."
    $Job = Start-Job -ScriptBlock {
        param([int]$Sec)
        $End = (Get-Date).AddSeconds($Sec)
        while ((Get-Date) -lt $End) {
            $HasCmdlet = [bool](Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)
            if ($HasCmdlet) {
                $Conns = Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue
                foreach ($Conn in $Conns) {
                    "{0}:{1}" -f $Conn.RemoteAddress, $Conn.RemotePort
                }
            }
            else {
                $Lines = netstat -an | findstr ESTABLISHED
                foreach ($Line in $Lines) {
                    $Parts = $Line.Trim() -split '\s+'
                    if ($Parts.Length -ge 4) { $Parts[2] }
                }
            }
            Start-Sleep -Seconds 2
        }
    } -ArgumentList $Seconds
    Wait-Job -Job $Job | Out-Null
    $Observed = @(Receive-Job -Job $Job | Sort-Object -Unique)
    Remove-Job -Job $Job -Force
    $Code = Show-Report -Endpoints $Observed -Title "Capture eredmenye ($Seconds s, egyedi tavoli vegpontok)"
    exit $Code
}
else {
    # -----------------------------------------------------------------------
    # Fazis A: pillanatfelvetel
    # -----------------------------------------------------------------------
    $Endpoints = @(Get-EstablishedEndpoints)
    $Code = Show-Report -Endpoints $Endpoints -Title "Pillanatfelvetel - letesitett TCP kapcsolatok"
    exit $Code
}
