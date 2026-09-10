#!/usr/bin/env python3
"""Manual smoke test for the VoiceMem LOCAL web UI (target machine).

Run on the machine with the models installed (GPU, llama-server, Piper):

    .venv\\Scripts\\python.exe scripts\\smoke_test_web.py
    (or: START.bat check runs it as part of the full verification)

What it does
------------
PHASE A (automatic):
  1. llama-server /health + a tiny /v1/chat/completions round-trip
  2. start the LOCAL web backend (or attach to a running one at :8787)
  3. component checklist via /api/pipeline (VAD/ASR/Memory/Embedding/LLM/TTS)
  4. open the browser at the VoiceMem UI

PHASE B (guided, you speak Hungarian):
  5. pick a microphone in the UI's Microphone selector (+ Start mic test)
  6. click "Start talking", then SAY something in Hungarian, e.g.
       "Szia! Peter vagyok, Budapesten lakom, es szeretek futni a Varosligetben."
  7. verify, in the UI: live partial transcript -> final transcript,
     memory recall panel, assistant reply, Piper audio playback, emotion tag
  8. type one message in the text box (ASR-bypassed badge must appear)
  9. check the Memory Space tab: the graph grew; create a new space; switch back

PHASE C: the script polls /api/memories to auto-confirm the memory entry and
prints the final PASS/FAIL checklist.

No external network is used at any point (only 127.0.0.1).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WEB_PORT = int(os.environ.get("VOICEMEM_WEB_PORT", "8787"))
WEB_BASE = f"http://127.0.0.1:{WEB_PORT}"
LLAMA_HEALTH = "http://127.0.0.1:8080/health"
LLAMA_CHAT = "http://127.0.0.1:8080/v1/chat/completions"

PASS = "\033[92mPASS\033[0m" if sys.stdout.isatty() else "PASS"
FAIL = "\033[91mFAIL\033[0m" if sys.stdout.isatty() else "FAIL"


def http_json(url: str, timeout: float = 4.0, data: dict | None = None) -> dict | None:
    try:
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_http(url: str, timeout_s: float = 45.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if http_json(url, timeout=2.0) is not None:
            return True
        time.sleep(0.7)
    return False


def open_browser(url: str) -> None:
    for cmd in (["start", "", url] if os.name == "nt" else ["xdg-open", url]):
        try:
            if os.name == "nt":
                subprocess.Popen(cmd, shell=True)
            else:
                subprocess.Popen(cmd)
            return
        except OSError:
            continue


def phase_a() -> bool:
    print("=" * 72)
    print("PHASE A - automatic checks")
    print("=" * 72)

    # 1. llama-server -------------------------------------------------------
    print("\n[1/4] llama-server (127.0.0.1:8080)...")
    health = http_json(LLAMA_HEALTH)
    print(f"      /health -> {'ok' if health else 'unreachable'}")
    ok_llama = health is not None
    reply = None
    if ok_llama:
        reply = http_json(
            LLAMA_CHAT,
            timeout=60.0,
            data={
                "model": "qwen3.6-35b-a3b",
                "messages": [{"role": "user", "content": "Say OK."}],
                "max_tokens": 8,
                "temperature": 0.0,
                # v0.4.21: the smoke round-trip now mirrors the PRODUCTION
                # request construction (app/llm.py _thinking_control_kwargs):
                # explicit request-level thinking suppression, aligned with
                # the server-side --reasoning off. A reply that is empty
                # (thinking ate the 8-token budget) fails the smoke test.
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none",
            },
        )
        content = (((reply or {}).get("choices") or [{}])[0].get("message") or {}).get("content", "")
        reasoning = (((reply or {}).get("choices") or [{}])[0].get("message") or {}).get("reasoning_content", "")
        print(f"      chat completion -> {content.strip()[:40]!r}")
        if reasoning:
            print(f"      WARNING: reasoning_content present ({len(reasoning)} chars) - "
                  "thinking NOT disabled for this request")
        ok_llama = bool(content.strip()) and not reasoning
    print(f"      llama-server: {PASS if ok_llama else FAIL}")

    # 2. web backend --------------------------------------------------------
    print(f"\n[2/4] LOCAL web backend (127.0.0.1:{WEB_PORT})...")
    proc = None
    if http_json(f"{WEB_BASE}/api/health", timeout=2.0) is None:
        print("      not running - starting it now (python -m app.web_server)...")
        venv_py = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python.exe" if os.name == "nt" else ROOT / ".venv" / "bin" / "python"
        py = str(venv_py) if venv_py.exists() else sys.executable
        proc = subprocess.Popen(
            [py, "-m", "app.web_server", "--host", "127.0.0.1", "--port", str(WEB_PORT)],
            cwd=str(ROOT),
        )
    ok_web = wait_http(f"{WEB_BASE}/api/health")
    print(f"      /api/health -> {PASS if ok_web else FAIL}")
    if not ok_web:
        print("      The backend did not come up. Run: START.bat  (then this script again.)")
        return False

    # 3. component statuses -------------------------------------------------
    print("\n[3/4] pipeline components (/api/pipeline)...")
    pipe = http_json(f"{WEB_BASE}/api/pipeline", timeout=8.0) or {}
    comps = pipe.get("components") or {}
    ok_comps = True
    for key in ("vad", "asr", "memory", "embedding", "llm", "tts"):
        c = comps.get(key) or {}
        state = c.get("state", "?")
        good = state in ("ready", "mocked", "processing")
        ok_comps = ok_comps and good
        print(f"      {key:<10} {state:<10} {c.get('detail', '')[:48]}"
              f"   [{PASS if good else FAIL}]")
    llama_ok = (pipe.get("llama") or {}).get("healthy")
    print(f"      llama-server healthy: {llama_ok}   [{PASS if llama_ok else FAIL}]")

    # 4. open the UI ----------------------------------------------------------
    print(f"\n[4/4] opening the browser at {WEB_BASE}/ ...")
    open_browser(f"{WEB_BASE}/")
    print("      If nothing opened, open it manually.")
    return ok_llama and ok_comps and bool(llama_ok)


MANUAL_STEPS = [
    (
        "select microphone",
        "In the Live input panel: click the Microphone selector, pick your device "
        "(click Refresh if names are missing), then 'Start mic test' and SPEAK - "
        "the level indicator must move. 'Stop mic test' afterwards.",
    ),
    (
        "speak Hungarian (ASR + VAD)",
        "Click 'Start talking', allow the microphone, then say slowly:\n"
        "      'Szia! Peter vagyok, Budapesten lakom, es szeretek futni a Varosligetben.'\n"
        "The live partial transcript should appear while you speak; after you stop, "
        "the final Hungarian transcript appears in the chat.",
    ),
    (
        "memory recall + LLM response",
        "Right after the transcript, the 'Top-K recall' panel lists memories, and the "
        "assistant replies in Hungarian in the chat + the reply panel.",
    ),
    (
        "Piper playback",
        "While the reply streams, you must HEAR the Hungarian voice (Piper). "
        "The voice label shows 'Replying'.",
    ),
    (
        "emotion display",
        "The 'emotion' chip in the tags row shows the fused emotion for your turn "
        "(e.g. neutral/happy); with expressive prosody it updates via tag_update.",
    ),
    (
        "text test mode (ASR bypassed)",
        "Type 'Emlkszem a budapesti parkra.' in the text box and press Enter: the turn "
        "appears with an orange 'Text input - ASR bypassed' badge.",
    ),
    (
        "memory entry + graph",
        "Open the 'Memory Space' tab: the graph grew (new node for what you said). "
        "Click a node to inspect it.",
    ),
    (
        "Memory Space create/switch/export",
        "Click '+' next to the space selector, create 'smoke', say one sentence, "
        "then switch back to 'demo' - the graph switches with it. "
        "Download -> All memories in this space.",
    ),
]


def phase_b() -> dict[str, bool]:
    print()
    print("=" * 72)
    print("PHASE B - guided manual checks (speak Hungarian into the microphone)")
    print("=" * 72)
    results: dict[str, bool] = {}
    for i, (name, how) in enumerate(MANUAL_STEPS, 1):
        print(f"\n[{i}/{len(MANUAL_STEPS)}] {name}")
        for line in how.split("\n"):
            print(f"      {line}")
        while True:
            ans = input("      Did it work? (y/n/s=skip): ").strip().lower()
            if ans in ("y", "n", "s"):
                break
            print("      Please answer y, n or s.")
        results[name] = {"y": True, "n": False, "s": False}[ans]
    return results


def phase_c(results: dict[str, bool]) -> int:
    print()
    print("=" * 72)
    print("PHASE C - automatic confirmation + summary")
    print("=" * 72)
    print("\nPolling /api/memories for up to 20 s (async fact extraction)...")
    before = http_json(f"{WEB_BASE}/api/memories")
    n_before = len((before or {}).get("left", [])) + len((before or {}).get("right", []))
    grew = n_before > 0
    deadline = time.time() + 20
    while not grew and time.time() < deadline:
        time.sleep(2.0)
        now = http_json(f"{WEB_BASE}/api/memories") or {}
        if len(now.get("left", [])) + len(now.get("right", [])) > 0:
            grew = True
    print(f"      memory entries via /api/memories: {PASS if grew else 'WARNING: none yet'}")

    print("\n" + "=" * 72)
    print("SMOKE TEST SUMMARY")
    print("=" * 72)
    fails = 0
    for name, ok in results.items():
        print(f"  {name:<34} [{PASS if ok else FAIL}]")
        fails += 0 if ok else 1
    print(f"  {'memory entry (auto)':<34} [{PASS if grew else FAIL}]")
    if grew is False:
        fails += 1
    total = len(results) + 1
    print(f"\n  {total - fails}/{total} checks passed.")
    print(f"  UI URL: {WEB_BASE}/")
    if fails:
        print("\n  Failed checks to re-verify: reopen the UI and repeat those steps;")
        print("  check the pipeline debug panel (bottom-left) for component errors.")
    return 0 if fails == 0 else 1


def main() -> int:
    print("VoiceMem web UI - manual smoke test")
    print(f"Python {sys.version.split()[0]} | backend port {WEB_PORT}\n")
    try:
        if not phase_a():
            print("\nPHASE A failed - fix the reported component before the manual phase.")
            return 2
        results = phase_b()
        return phase_c(results)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
