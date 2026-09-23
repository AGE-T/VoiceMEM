@echo off
setlocal
REM ============================================================================
REM  experimental\llama_b11073\RUN_EXPERIMENTAL.bat - ONE-CLICK experimental
REM  session (llama.cpp b11073 @ 127.0.0.1:8081).
REM
REM  What it does, in order:
REM    1. installs the pinned b11073 runtime if it is not installed yet
REM       (download at INSTALL time, SHA-256-gated; ~150 MB, one-off);
REM    2. enables the experimental profile if needed (operator-local override
REM       only - NO production config file is ever modified);
REM    3. starts the experimental llama-server in its own window - the launcher
REM       verifies the pinned runtime BEFORE start (24-file SHA-256 manifest
REM       gate + in-binary build identity) and the LIVE server AFTER start
REM       (listener PID identity + system_fingerprint);
REM    4. waits for http://127.0.0.1:8081/health to turn OK;
REM    5. starts the VoiceMem app session in its own window (targets 8081 for
REM       THIS session only).
REM
REM  The production runtime (b10717 @ 127.0.0.1:8080) is NEVER touched.
REM  Rollback: close both windows + configure_b11073.ps1 -Disable.
REM  This file is deliberately ASCII with CRLF line endings.
REM ============================================================================

cd /d "%~dp0.."

echo [RUN_EXPERIMENTAL] Experimental session (b11073 @ 127.0.0.1:8081)

REM --- 1) the pinned runtime installed? --------------------------------------
if not exist "experimental\llama_b11073\bin\llama-server.exe" (
    echo [RUN_EXPERIMENTAL] The b11073 runtime is not installed yet.
    echo [RUN_EXPERIMENTAL] Starting the installer: downloads the PINNED b11073
    echo [RUN_EXPERIMENTAL] release asset and verifies its SHA-256 (~150 MB, one-off)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "experimental\llama_b11073\install_b11073.ps1"
    if errorlevel 1 goto :fail
) else (
    echo [RUN_EXPERIMENTAL] b11073 runtime: installed ^(re-verified by the launcher before start^)
)

REM --- 2) the operator-local session configured? ------------------------------
if not exist "experimental\llama_b11073\start_voicemem_experimental.ps1" (
    echo [RUN_EXPERIMENTAL] Enabling the experimental profile ^(operator-local override only^)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "experimental\llama_b11073\configure_b11073.ps1" -Enable
    if errorlevel 1 goto :fail
)

REM --- 3) the experimental llama-server in its own window -----------------------
echo [RUN_EXPERIMENTAL] Starting the experimental llama-server ^(window 1^)...
start "b11073 llama-server 127.0.0.1:8081 (EXPERIMENTAL)" powershell -NoProfile -ExecutionPolicy Bypass -File "experimental\llama_b11073\start_llama_server_experimental.ps1"

REM --- 4) wait for /health (the launcher window prints the full verification) ---
echo [RUN_EXPERIMENTAL] Waiting for http://127.0.0.1:8081/health ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=(Get-Date).AddSeconds(240);while((Get-Date) -lt $d){try{$r=Invoke-WebRequest 'http://127.0.0.1:8081/health' -UseBasicParsing -TimeoutSec 2;if($r.StatusCode -eq 200){exit 0}}catch{};Start-Sleep -Milliseconds 500};exit 1"
if errorlevel 1 (
    echo [RUN_EXPERIMENTAL] The experimental server did not become healthy within 240 s.
    echo [RUN_EXPERIMENTAL] Check window 1 - the launcher prints actionable diagnostics
    echo [RUN_EXPERIMENTAL] ^(and writes logs under experimental\llama_b11073\logs^\).
    goto :fail
)
echo [RUN_EXPERIMENTAL] /health OK on 8081.

REM --- 5) the app session in its own window -------------------------------------
echo [RUN_EXPERIMENTAL] Starting the VoiceMem app session ^(window 2, targets 8081^)...
start "VoiceMem EXPERIMENTAL session (b11073 @ 8081)" powershell -NoProfile -ExecutionPolicy Bypass -File "experimental\llama_b11073\start_voicemem_experimental.ps1"

echo.
echo [RUN_EXPERIMENTAL] Experimental session started:
echo   window 1 : b11073 llama-server   ^(127.0.0.1:8081 - NOT the production 8080^)
echo   window 2 : VoiceMem app session   ^(targets 8081 for THIS session only^)
echo   verify   : powershell -NoProfile -ExecutionPolicy Bypass -File "experimental\llama_b11073\verify_b11073.ps1"
echo   rollback : close both windows, then configure_b11073.ps1 -Disable
echo.
echo The normal START.bat flow still targets the production b10717 @ 8080.
echo.
pause
exit /b 0

:fail
echo.
echo [RUN_EXPERIMENTAL] FAILED - see the messages above. Production was NOT touched.
echo [RUN_EXPERIMENTAL] Rollback / cleanup: configure_b11073.ps1 -Disable
pause
exit /b 1
