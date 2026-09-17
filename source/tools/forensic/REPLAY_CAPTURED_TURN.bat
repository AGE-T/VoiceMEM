@echo off
setlocal enableextensions
REM ============================================================================
REM  REPLAY_CAPTURED_TURN.bat - STAGE 3 forensic replay (TEMPORARY tool).
REM
REM  Replays the EXACT captured production LLM request (default: the SLOWEST
REM  turn in logs\llm_production_trace.jsonl) DIRECTLY, TWICE:
REM    1. WARM  - against the RUNNING production llama-server (never restarted)
REM    2. FRESH - against a NEW llama-server with the SAME production command
REM               line (only the port differs; the production server keeps
REM               running; the fresh process is stopped afterwards)
REM  Then writes the full comparison + verdict:
REM    logs\llm_production_capture_result.md / .json
REM
REM  A friss (2.) szerver nem fer el a production mellett a 12 GB VRAM-on?
REM  A tool kiirja a teendot: zar be a production llama-server ablakot (Ctrl+C),
REM  majd futtass:
REM     REPLAY_CAPTURED_TURN.bat --replay fresh-only
REM  (a korabbi warm eredmenyt osszefeszi). Utana START.bat = normal mukodes.
REM ============================================================================

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%"

if not exist ".venv\Scripts\python.exe" (
    echo HIBA: .venv\Scripts\python.exe nem talalhato. Elaszor futtass START.bat-et.
    pause
    exit /b 1
)

if not exist "logs\llm_production_trace.jsonl" (
    echo HIBA: logs\llm_production_trace.jsonl nem talalhato.
    echo Elaszor futtass CAPTURE_START.bat-et es hasznald a VoiceMem-et.
    pause
    exit /b 1
)

echo ==========================================================================
echo  VOICEMEM STAGE 3 - REPLAY: a pontosan rogzitett lassu keres ketszer:
echo    [1] WARM  - a futo production llama-server ellen (nem nyul hozza)
echo    [2] FRESH - uj process, ugyanaz a production konfiguracio (cold KV)
echo  Eredmeny: logs\llm_production_capture_result.md / .json
echo  Ez percekig tarthat (a 19 GB modell ujrabetoltese a [2] lepesben).
echo ==========================================================================
echo.

".venv\Scripts\python.exe" tools\forensic\replay_captured_request.py %*
echo.
echo Keszen all: logs\llm_production_capture_result.md
pause
