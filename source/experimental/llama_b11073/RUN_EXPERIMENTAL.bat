@echo off
REM ============================================================================
REM experimental\llama_b11073\RUN_EXPERIMENTAL.bat - DELTA RUNTIME PACK starter
REM ============================================================================
REM PURPOSE: start the VALIDATED experimental llama.cpp b11073 llama-server with
REM ONE double-click, resolving the CUDA runtime DLL (cublas64_13.dll) that the
REM target machine ALREADY has (production bin\ from the b10717 install and/or
REM a system CUDA 13.x on PATH). This wrapper does NOT package, copy or modify
REM any CUDA component - it only extends the child process PATH so the existing
REM cublas64_13.dll becomes visible to bin\llama-server-b11073\llama-server.exe.
REM
REM WHAT IT DOES (in order):
REM   1. read-only prerequisite + integrity probe (check_environment.ps1);
REM   2. prepend ^<root^>\bin\ to the child PATH (existing CUDA DLLs only);
REM   3. hand over to the VALIDATED launcher start_llama_server_experimental.ps1
REM      (byte-identical to the audited one: ngl 99 / ctx 16000 / parallel 1 /
REM      threads 12 / reasoning off / KV q8_0 / port 127.0.0.1:8081).
REM
REM ROLLBACK (full revert of the experiment):
REM   1. close this window (stops the experimental server);
REM   2. remove LLAMA_SERVER_HOST / LLAMA_SERVER_PORT from the agent session
REM      (or open a fresh terminal) - the agent targets 127.0.0.1:8080 again;
REM   3. optionally delete bin\llama-server-b11073\ and the unpacked pack files.
REM
REM PRODUCTION IS NOT TOUCHED: bin\llama-server.exe (b10717), config\, scripts\,
REM VERSION - nothing in this flow reads or writes them.
REM ============================================================================
setlocal
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"

echo === VoiceMem b11073 DELTA RUNTIME PACK ===
echo   root : %ROOT%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%check_environment.ps1" -Quiet
if errorlevel 2 (
    echo.
    echo [RUN_EXPERIMENTAL] A korfeltetel-proba HIBAT talalt.
    echo Futtasd teljes kimenettel a reszletekert:
    echo   powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%check_environment.ps1"
    echo.
    pause
    exit /b 2
)

set "PATH=%ROOT%\bin;%PATH%"

echo A gyartasi bin\ CUDA DLL-jei most lathatok a gyermekfolyamat szamara.
echo Atadas a validalt inditonak (port 8081, a gyartasi 8080 erintetlen)...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%start_llama_server_experimental.ps1"
endlocal
