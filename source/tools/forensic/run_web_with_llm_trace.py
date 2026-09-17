"""TEMPORARY forensic launcher: the PRODUCTION web backend + LLM tracing.

Stage-3 forensic tool. Run INSTEAD of the plain web backend for the capture
session (the llama-server keeps running exactly as START.bat started it):

    F:\\Voicemem\\VoiceMemAgent> .venv\\Scripts\\python.exe -u ^
            tools\\forensic\\run_web_with_llm_trace.py --host 127.0.0.1 --port 8787

or just double-click ``tools\\forensic\\CAPTURE_START.bat``.

It performs EXACTLY the production start (same module, same entry
``app.web_server.main()``, same arguments/environment), but first installs
``tools/forensic/llm_trace_instrumentation.py`` at runtime. NO production file
is modified; stopping this process and starting normally (START.bat) removes
every trace of the instrumentation.

Fail-open: if the instrumentation cannot be installed, the backend starts
UNINSTRUMENTED (identical to the normal production run) and says so.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_instrumentation():
    spec = importlib.util.spec_from_file_location(
        "llm_trace_instrumentation",
        REPO_ROOT / "tools" / "forensic" / "llm_trace_instrumentation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    # Production-equivalent process setup (start_agent.ps1 launches the venv
    # python with -u, cwd = repo root, then ``-m app.web_server``).
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Load the production config the same way app.web_server.main() will
    # (same env + cwd) so the session record describes the real run.
    try:
        from app import web_server as _ws

        cfg = _ws.load_config("")
    except Exception as exc:  # instrumentation must never block production
        print(f"[llm-trace] config pre-load failed ({exc!r}); tracing without it")
        cfg = None

    tracer = _load_instrumentation()
    ok = tracer.install(cfg)
    print("=" * 70)
    print("[llm-trace] STAGE 3 forensic capture session")
    print(f"[llm-trace] instrumentation: {'ACTIVE' if ok else 'INACTIVE (fail-open plain run)'}")
    print(f"[llm-trace] trace file     : {tracer.TRACE_PATH}")
    print("[llm-trace] every LLM request is captured with exact payload +")
    print("[llm-trace] timing markers + server-side metrics. Use VoiceMem")
    print("[llm-trace] normally; when a 30-50 s slow turn happens, run:")
    print("[llm-trace]   tools\\forensic\\REPLAY_CAPTURED_TURN.bat")
    print("[llm-trace] to replay the exact captured request directly.")
    print("=" * 70)

    from app import web_server

    return web_server.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
