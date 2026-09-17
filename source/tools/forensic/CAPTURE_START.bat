@echo off
setlocal enableextensions
REM ============================================================================
REM  CAPTURE_START.bat - STAGE 3 forensic capture session. TEMPORARY tool.
REM
REM  Starts the PRODUCTION VoiceMem web backend with LLM-call tracing enabled
REM  via tools\forensic\run_web_with_llm_trace.py. NO production file is
REM  modified; the llama-server keeps running exactly as START.bat started it.
REM
REM  Use VoiceMem NORMALLY. Every LLM request is captured with its exact
REM  payload + timing markers into logs\llm_production_trace.jsonl.
REM  When a slow 30-50 s turn happens, run REPLAY_CAPTURED_TURN.bat.
REM
REM  To return to plain production: stop this with Ctrl+C and start normally
REM  with START.bat. Nothing to uninstall.
REM
REM  SYNTAX RULE for cmd.exe, enforced in this file: NO unescaped round
REM  brackets inside if-blocks. A raw closing bracket inside an echo line
REM  terminates the block early and cmd.exe fails with
REM  "X was unexpected at this time." This file is plain ASCII with CRLF
REM  line endings on purpose, so it parses identically from PowerShell
REM  and from cmd.exe under any OEM codepage.
REM
REM  PRE-FLIGHT: exactly ONE llama-server.exe may run. Two simultaneous
REM  instances (double START.bat during the model load) corrupt every
REM  latency measurement. Check in Task Manager or with:
REM    Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'"
REM    ^| Select-Object ProcessId,CreationDate,CommandLine
REM ============================================================================

REM -- project root = two levels above this BAT --------------------------------
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
if not defined ROOT (
    echo HIBA: nem sikerult meghatarozni a projekt gyokeret.
    pause
    exit /b 1
)
cd /d "%ROOT%"
if errorlevel 1 (
    echo HIBA: nem sikerult atvaltani a projekt gyokerbe: "%ROOT%"
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo HIBA: .venv\Scripts\python.exe nem talalhato a projekt gyokerben.
    echo Eloszor futtass START.bat-et.
    pause
    exit /b 1
)

if not exist "tools\forensic\run_web_with_llm_trace.py" (
    echo HIBA: tools\forensic\run_web_with_llm_trace.py nem talalhato.
    echo A Stage-3 forensic ZIP-et a F:\Voicemem\VoiceMemAgent gyokerbe kell
    echo kicsomagolni, hogy a fajlok a tools\forensic mappaba keruljenek.
    pause
    exit /b 1
)

REM -- the llama-server must already be running - START.bat starts it ---------
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8080/health' -Method Get -TimeoutSec 3 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo HIBA: a llama-server nem erheto el a 127.0.0.1:8080 cimen.
    echo Teendok:
    echo  1. Futtass START.bat-et, varj meg, mignem a konzolon megjelenik:
    echo     PASS: a llama-server felfutott ... /health 200
    echo  2. Zarod be CSAK a web backend konzolt. A llama-server futva marad.
    echo  3. Futtass ujra ezt a fajlt: CAPTURE_START.bat
    pause
    exit /b 1
)

REM -- the plain web backend must not hold the 8787 port -----------------------
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8787/api/health' -Method Get -TimeoutSec 2 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo HIBA: a 8787-es porton mar fut egy web backend.
    echo Zarod be azt a konzolt, majd futtass ujra ezt a fajlt.
    pause
    exit /b 1
)

echo ==========================================================================
echo  VOICEMEM FORENSIC CAPTURE SESSION - stage 3 - LLM trace ACTIVE
echo  Minden LLM keres pontos payload + idozitesi jelekkel loggolodik:
echo    logs\llm_production_trace.jsonl
echo  Lassu 30-50 s turn utan futtass: REPLAY_CAPTURED_TURN.bat
echo  Leallitas: Ctrl+C, utana START.bat = normal production mukodes.
echo ==========================================================================
echo.

REM -- the forensic web backend itself -----------------------------------------
".venv\Scripts\python.exe" -u tools\forensic\run_web_with_llm_trace.py --host 127.0.0.1 --port 8787
set "VMEC=%errorlevel%"
echo.
if not "%VMEC%"=="0" (
    echo FIGYELMEZTETES: a forensic backend hibaval allt le. Hibakod: %VMEC%
    echo A reszleteket a konzol kimenet es a logs\ mappa tartalmazza.
)
echo A forensic capture session befejezodott.
echo Vissza a normal production uzemmodhoz: Ctrl+C, majd START.bat.
echo.
pause
exit /b %VMEC%
