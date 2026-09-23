@echo off
setlocal enableextensions
REM ===========================================================================
REM  RUN_PRODUCTION_UPGRADE.bat
REM
REM  One-click entry point for the PRODUCTION llama.cpp runtime upgrade
REM  (b10717 -> b11073) of VoiceMem Agent.
REM
REM  What it does:
REM    1. runs ops\llama_b11073_production_upgrade\upgrade_production_b11073.ps1
REM       with the Bypass execution policy (MOTW-safe, like the START.bat
REM       chain): audit -> stop -> backup -> install -> verify -> start ->
REM       server battery -> VoiceMem integration -> regression -> report;
REM    2. on any critical failure the timestamped backup is restored
REM       automatically (rollback_production_b11073.ps1) and the old
REM       production server is restarted;
REM    3. prints the evidence ZIP path at the end (send it back for the
REM       audit record).
REM
REM  The Qwen3.6 model, the VoiceMem application code, the production
REM  configuration and the port 8080 are NEVER changed by this upgrade.
REM  The pack contains NO binaries: it reuses the already-installed,
REM  already-verified experimental\llama_b11073 runtime (no download).
REM
REM  Optional arguments are passed through to the PowerShell script, e.g.:
REM     RUN_PRODUCTION_UPGRADE.bat -SkipIntegration
REM     RUN_PRODUCTION_UPGRADE.bat -VerifyOnly
REM     RUN_PRODUCTION_UPGRADE.bat -AuditOnly
REM     RUN_PRODUCTION_UPGRADE.bat -ProbeOnly
REM
REM  This file is shipped with Windows CRLF line endings (pack convention).
REM ===========================================================================

REM The project root is two levels above this BAT file's folder.
set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"

REM --- 64-bit PowerShell resolution (mirrors START.bat; 32-bit is forbidden)
set "PS_EXE="
set "PS_NATIVE_DIR=%SystemRoot%\System32"
if defined PROCESSOR_ARCHITEW6432 set "PS_NATIVE_DIR=%SystemRoot%\Sysnative"

if defined ProgramW6432 if exist "%ProgramW6432%\PowerShell\7\pwsh.exe" set "PS_EXE=%ProgramW6432%\PowerShell\7\pwsh.exe"

if not defined PS_EXE (
    where pwsh >nul 2>nul
    if not errorlevel 1 (
        for /f "usebackq delims=" %%I in (`where pwsh`) do (
            if not defined PS_EXE (
                echo "%%I" | findstr /I /C:"SysWOW64" /C:"Program Files (x86)" >nul 2>nul
                if errorlevel 1 set "PS_EXE=%%I"
            )
        )
    )
)

if not defined PS_EXE if exist "%PS_NATIVE_DIR%\WindowsPowerShell\v1.0\powershell.exe" set "PS_EXE=%PS_NATIVE_DIR%\WindowsPowerShell\v1.0\powershell.exe"

if not defined PS_EXE (
    echo.
    echo [ERROR] No 64-bit PowerShell was found on this machine.
    echo 64-bit Windows 10/11 is required - 32-bit PowerShell is not supported.
    echo.
    pause
    exit /b 1
)

echo === VoiceMem production llama.cpp runtime upgrade: b10717 - b11073 ===
echo root: %PROJECT_ROOT%
echo.

"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\ops\llama_b11073_production_upgrade\upgrade_production_b11073.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
    echo [OK] Upgrade finished successfully.
    echo Evidence ZIP: see the path printed above ^(logs\prod_b11073_upgrade_*^).
) else (
    echo [ERROR] The upgrade failed - exit code %EXITCODE%.
    echo If the automatic rollback ran, the old production server is back.
    echo Manual rollback ^(if needed^):
    echo   powershell -NoProfile -ExecutionPolicy Bypass -File ops\llama_b11073_production_upgrade\rollback_production_b11073.ps1
)
echo.
pause
endlocal & exit /b %EXITCODE%
