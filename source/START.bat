@echo off
setlocal enableextensions
REM ===========================================================================
REM  START.bat - the ONLY user-facing entry point of VoiceMem Agent.
REM
REM  Double-click this file. Everything else (Python check, .venv creation,
REM  dependency install, Hugging Face tooling, model download, config,
REM  verification, smoke tests, agent start) happens AUTOMATICALLY.
REM
REM  Double-click flow (mode "run", the default):
REM    1. validate the installation (venv, dependencies, assets, config)
REM    2. start llama-server (127.0.0.1:8080) if it is not already running
REM    3. start the LOCAL web backend  ->  http://127.0.0.1:8787
REM    4. open the browser automatically at the VoiceMem web UI
REM  The exact URL is printed in the console; the user never starts a
REM  server manually. The browser talks ONLY to the local backend and all
REM  LLM inference goes through the local llama-server (never OpenAI).
REM
REM  Optional modes (typed after the file name in a console):
REM     START.bat             open the VoiceMem web UI (auto-setup first)
REM     START.bat web         same as the default - an explicit alias
REM     START.bat cli         console voice agent (python -m app.main)
REM     START.bat mock        scripted 3-turn demo (no mic, no GPU models)
REM     START.bat check       full verification: assets + test suite
REM     START.bat benchmark   latency benchmark suite
REM     START.bat repair      automatic environment repair
REM     START.bat build       versioned release ZIP (tests gate the build)
REM
REM  v0.3.6 robustness - field report: an UNKNOWN first argument can reach
REM  this script without the user intending it (drag-and-drop onto the
REM  file, a custom .bat file association, a path with spaces), and v0.3.5
REM  then dead-ended at a usage menu instead of the web UI. Since v0.3.6 an
REM  unrecognized mode prints a short note and CONTINUES in the default
REM  web-UI mode, so a double-click always reaches the web UI. This file
REM  is also shipped with Windows CRLF line endings.
REM
REM  The project is relocatable: everything is resolved relative to this
REM  file's folder, so the project can live in any folder on any drive.
REM  All internal PowerShell scripts under scripts\ are implementation
REM  details - the user never needs to run them directly.
REM ===========================================================================

REM The project root is this BAT file's folder (works from any drive/path).
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

REM Resolve the mode (default: run = the web UI).
set "MODE=%~1"
if not defined MODE set "MODE=run"
REM "web" is an explicit alias of the default run mode.
if /I "%MODE%"=="web" set "MODE=run"

REM v0.3.6: an unknown mode NEVER dead-ends the user. Windows can deliver
REM an unexpected first argument (drag-and-drop, a file association, a
REM typed typo) - any unrecognized mode now falls back to the default
REM web-UI flow with a visible note. NOTE: %MODE% is deliberately NEVER
REM echoed inside this block: a dropped path could contain parentheses
REM that would break a parenthesized cmd block at parse time.
if /I not "%MODE%"=="run" if /I not "%MODE%"=="cli" if /I not "%MODE%"=="mock" if /I not "%MODE%"=="check" if /I not "%MODE%"=="benchmark" if /I not "%MODE%"=="repair" if /I not "%MODE%"=="build" (
    echo.
    echo [NOTE] Unknown start mode - continuing in the default WEB UI mode.
    echo [NOTE] Known modes: web, cli, mock, check, benchmark, repair, build.
    echo [NOTE] Tip: double-click START.bat with no arguments at all.
    echo.
    set "MODE=run"
)

REM ===========================================================================
REM 64-bit PowerShell resolution - 32-bit PowerShell is EXPLICITLY FORBIDDEN.
REM The generic "powershell" PATH lookup is never used here: a 32-bit cmd.exe
REM (or a SysWOW64-first PATH) could silently resolve the 32-bit engine, and
REM WOW64 file redirection would then corrupt the whole install chain.
REM Resolution order:
REM   1. PRIMARY  : 64-bit PowerShell 7 (pwsh.exe) from the standard 64-bit
REM                 install location %ProgramW6432%\PowerShell\7\pwsh.exe,
REM                 or a PATH hit that is NOT a known 32-bit location.
REM   2. SECONDARY: 64-bit Windows PowerShell from the NATIVE System32 path.
REM                 If this cmd.exe is itself 32-bit, Windows file redirection
REM                 maps System32 to SysWOW64 - Sysnative bypasses that and
REM                 always points at the REAL 64-bit System32.
REM scripts\bootstrap.ps1 enforces this again at runtime with an
REM [Environment]::Is64BitProcess guard and refuses to run in 32-bit.
REM ===========================================================================
set "PS_EXE="
set "PS_NATIVE_DIR=%SystemRoot%\System32"
if defined PROCESSOR_ARCHITEW6432 set "PS_NATIVE_DIR=%SystemRoot%\Sysnative"

REM 1) PRIMARY: 64-bit PowerShell 7 (pwsh) - standard 64-bit install path.
if defined ProgramW6432 if exist "%ProgramW6432%\PowerShell\7\pwsh.exe" set "PS_EXE=%ProgramW6432%\PowerShell\7\pwsh.exe"

REM 1b) pwsh resolved from PATH - accepted only outside 32-bit locations.
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

REM 2) SECONDARY: 64-bit Windows PowerShell from the native System32 path.
if not defined PS_EXE if exist "%PS_NATIVE_DIR%\WindowsPowerShell\v1.0\powershell.exe" set "PS_EXE=%PS_NATIVE_DIR%\WindowsPowerShell\v1.0\powershell.exe"

if not defined PS_EXE (
    echo.
    echo [ERROR] No 64-bit PowerShell was found on this machine.
    echo 64-bit Windows 10/11 is required - 32-bit PowerShell is not supported.
    echo.
    pause
    exit /b 1
)

REM Hand over to the bootstrap orchestrator (internal implementation detail).
REM bootstrap.ps1 double-checks the 64-bit process and refuses to run 32-bit.
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%scripts\bootstrap.ps1" -Mode "%MODE%"
set "EXITCODE=%ERRORLEVEL%"

REM Keep the console window open so double-click users can read the outcome.
if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] VoiceMem Agent could not start ^(mode: %MODE%^).
    echo Technical details: logs\bootstrap.log
    echo Fix the problem reported above, then double-click START.bat again.
)
echo.
pause
endlocal & exit /b %EXITCODE%
