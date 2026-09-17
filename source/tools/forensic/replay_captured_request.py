"""TEMPORARY forensic tool: replay the EXACT captured production request.

Stage-3 forensic tool (field-capture spec). Reads
``logs/llm_production_trace.jsonl`` (written by
``tools/forensic/run_web_with_llm_trace.py``), picks ONE real production turn
(default: the SLOWEST to the first token), then sends THE EXACT SAME REQUEST
— the literal JSON bytes captured on the wire — DIRECTLY against
llama-server, TWICE:

  1. WARM replay  - against the RUNNING production llama-server
                    (production endpoint, production config, NOTHING changed,
                    NEVER restarted by this tool);
  2. FRESH replay - against a NEW llama-server process started by this tool
                    with the SAME production configuration (command line
                    rebuilt from the production process identity in the
                    trace, only the port changes; the production server
                    keeps running). A fresh process has an EMPTY KV cache,
                    so this is a true COLD measurement — a warm-cache hit on
                    replay #1 can never masquerade as a fast cold request.

Both replays use the same marker set as the production trace
(llm_request_start -> http handoff -> headers -> first SSE line -> first
content token -> stream end) plus server-side evidence: /metrics deltas, the
server's own ``slot print_timing`` lines and (fresh) the startup effective
config lines from the fresh server's own log.

The result separates MEASURED FACTS (every number above) from INFERENCES
(the A-G decision-tree classification, ngl26/context evidence) from
UNRESOLVED QUESTIONS, and answers the 16 verdict questions A-P explicitly.

If the fresh server cannot start while the production server holds the GPU
(VRAM contention with the 19 GB model), the failure is recorded honestly
with operator guidance: close the production llama-server window and re-run
with ``--replay fresh-only`` — the previous warm result is MERGED, never
lost.

Writes ``logs/llm_production_capture_result.json`` + ``.md``. NO production
file is modified; only ``logs/`` is written.

Usage (field machine):
    .venv\\Scripts\\python.exe tools\\forensic\\replay_captured_request.py
    ... --trace-id <id> | --mode slowest (default) | --mode last
    ... --url http://127.0.0.1:8080/v1        (warm replay endpoint)
    ... --replay both (default) | warm | fresh-only
    ... --fresh-port 8181 --fresh-timeout 900
    ... --server-exe PATH --model PATH        (fresh-start overrides)
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = REPO_ROOT / "logs" / "llm_production_trace.jsonl"
RESULT_PATH = REPO_ROOT / "logs" / "llm_production_capture_result.json"
REPORT_PATH = REPO_ROOT / "logs" / "llm_production_capture_result.md"
FRESH_LOG_PATH = REPO_ROOT / "logs" / "llama-forensic-fresh-server.log"
WEB_SERVER_LOG_PATH = REPO_ROOT / "logs" / "web-server.log"
RESOLVED_MODEL_PATH = REPO_ROOT / "logs" / "llama-server.resolved-model.json"

# llama-server b10717 /metrics counters (verified against the live build)
METRIC_PTOK = "llamacpp:prompt_tokens_total"
METRIC_PSEC = "llamacpp:prompt_seconds_total"
METRIC_GTOK = "llamacpp:tokens_predicted_total"
METRIC_GSEC = "llamacpp:tokens_predicted_seconds_total"

DEFAULT_FRESH_PORT = 8181
DEFAULT_SLOW_THRESHOLD_S = 30.0

# instrumentation module (reuses the metrics parser / log tail helpers)
_spec = importlib.util.spec_from_file_location(
    "llm_trace_instrumentation",
    REPO_ROOT / "tools" / "forensic" / "llm_trace_instrumentation.py",
)
_tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tr)

# production SSE parser (same code path the app uses)
sys.path.insert(0, str(REPO_ROOT))
from app.llm import parse_sse_content_delta, parse_sse_reasoning_delta  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _fmt(v: Any, unit: str = "s") -> str:
    if v is None:
        return "NOT MEASURED"
    try:
        return f"{float(v):.3f} {unit}"
    except Exception:
        return str(v)


def _pick(d: Optional[dict], *keys: str) -> Optional[float]:
    for k in keys:
        if d and k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return None


# --------------------------------------------------------------------------- #
# trace loading
# --------------------------------------------------------------------------- #


def load_turns(trace_path: Path) -> tuple[Optional[dict], dict[str, dict]]:
    """Return (session_event, {trace_id: turn}) from the JSONL trace."""
    session: Optional[dict] = None
    turns: dict[str, dict] = {}
    if not trace_path.exists():
        return None, turns
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        ev = rec.get("event")
        tid = rec.get("trace_id")
        if ev == "session":
            session = rec
        elif ev and tid:
            turn = turns.setdefault(tid, {"trace_id": tid})
            if ev in ("start", "wire", "headers", "end"):
                turn[ev] = rec
    return session, turns


def select_turn(turns: dict[str, dict], mode: str, trace_id: Optional[str]):
    """Pick the target production turn (default: slowest first content token)."""
    if trace_id:
        if trace_id not in turns:
            raise SystemExit(f"trace_id {trace_id!r} not found in {TRACE_PATH}")
        return turns[trace_id]
    completed = [
        t
        for t in turns.values()
        if "end" in t and (t["end"].get("markers") or {}).get("first_content_token_rel")
    ]
    if not completed:
        raise SystemExit(
            "no completed LLM turn (with a first-token marker) found in the "
            "trace file. Run the instrumented backend (CAPTURE_START.bat) and "
            "use VoiceMem until a turn completes."
        )
    if mode == "last":
        return completed[-1]

    def ttft(t: dict) -> float:
        try:
            return t["end"]["markers"].get("first_content_token_rel", 0.0) or 0.0
        except Exception:
            return 0.0

    return max(completed, key=ttft)


# --------------------------------------------------------------------------- #
# server-side print_timing parsing (the server's own numbers)
# --------------------------------------------------------------------------- #

_RE_PROMPT_TIMING = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", re.I
)
_RE_EVAL_TIMING = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", re.I
)
_RE_TOTAL_TIMING = re.compile(
    r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", re.I
)
_RE_NEW_PROMPT_TOKENS = re.compile(r"new prompt.*task\.n_tokens\s*=\s*(\d+)", re.I)


def _parse_print_timing(lines: list[str]) -> dict[str, Any]:
    """Extract the server's own prompt/eval/total timing numbers."""
    out: dict[str, Any] = {}
    try:
        for ln in lines or []:
            m = _RE_PROMPT_TIMING.search(ln)
            if m and "prompt_ms" not in out:
                out["prompt_ms"] = float(m.group(1))
                out["prompt_tokens"] = int(m.group(2))
            m = _RE_EVAL_TIMING.search(ln)
            if m and "eval_ms" not in out:
                out["eval_ms"] = float(m.group(1))
                out["eval_tokens"] = int(m.group(2))
            m = _RE_TOTAL_TIMING.search(ln)
            if m and "total_ms" not in out:
                out["total_ms"] = float(m.group(1))
                out["total_tokens"] = int(m.group(2))
            m = _RE_NEW_PROMPT_TOKENS.search(ln)
            if m and "new_prompt_tokens" not in out:
                out["new_prompt_tokens"] = int(m.group(1))
        if out.get("prompt_ms") and out.get("prompt_tokens"):
            out["prompt_tok_s"] = round(out["prompt_tokens"] / (out["prompt_ms"] / 1000.0), 2)
        if out.get("eval_ms") and out.get("eval_tokens"):
            out["eval_tok_s"] = round(out["eval_tokens"] / (out["eval_ms"] / 1000.0), 2)
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# production web-server [chain] log correlation
# --------------------------------------------------------------------------- #

_CHAIN_TS = "%Y-%m-%d %H:%M:%S,%f"
_RE_CHAIN_LINE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"\[chain\] (llm start|llm first token \((\d+) ms\)|"
    r"llm done \((\d+) chars, (\d+) ms\))"
)


def _chain_correlation(turn: dict) -> dict[str, Any]:
    """Match the production [chain] log trail to the instrumented turn.

    The chain log's own timestamps give an INDEPENDENT LLM first-token and
    total measurement for the same turn (log-line clock, written by the
    production web server itself). Note: the ``llm first token (X ms)`` X is
    measured from TURN start (includes ASR/memory), so the line-to-line time
    delta is used instead, which measures exactly llm start -> first token.
    """
    out: dict[str, Any] = {"attempted": True, "matched": False}
    end = turn.get("end", {}) or {}
    content_chars = end.get("content_chars")
    inst_ttft = _pick(end.get("markers"), "first_content_token_rel")
    inst_total = _pick(end.get("markers"), "llm_request_end_rel", "stream_end_rel")
    try:
        text = WEB_SERVER_LOG_PATH.read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        out["note"] = f"{WEB_SERVER_LOG_PATH.name} not readable"
        return out

    events: list[dict] = []  # {ts, kind, chars?, ms_from_turn0?, ms_from_llm0?}
    for line in text.splitlines():
        m = _RE_CHAIN_LINE.match(line.strip())
        if not m:
            continue
        ts = datetime.strptime(m.group(1), _CHAIN_TS)
        if m.group(2) == "llm start":
            events.append({"ts": ts, "kind": "start"})
        elif m.group(2).startswith("llm first token"):
            events.append({"ts": ts, "kind": "first_token", "ms_from_turn0": int(m.group(3))})
        else:
            events.append(
                {
                    "ts": ts,
                    "kind": "done",
                    "chars": int(m.group(4)),
                    "ms_from_llm0": int(m.group(5)),
                }
            )

    # candidate turns: every 'start' with its following first_token + done
    cands: list[dict] = []
    for i, ev in enumerate(events):
        if ev["kind"] != "start":
            continue
        ft = next((e for e in events[i + 1 :] if e["kind"] == "first_token"), None)
        dn = next((e for e in events[i + 1 :] if e["kind"] == "done"), None)
        if ft and dn:
            cands.append(
                {
                    "start": ev,
                    "first_token": ft,
                    "done": dn,
                    "ttft_s": (ft["ts"] - ev["ts"]).total_seconds(),
                    "total_s": (dn["ts"] - ev["ts"]).total_seconds(),
                }
            )
    if not cands:
        out["note"] = "no complete [chain] llm turn found in web-server.log"
        return out

    # choose the candidate whose done-chars match this turn's content_chars;
    # fall back to the one whose total best matches the instrumented total.
    chosen = None
    if content_chars:
        for c in cands:
            if c["done"].get("chars") == content_chars:
                chosen = c
                break
    if chosen is None and inst_total is not None:
        chosen = min(cands, key=lambda c: abs(c["total_s"] - inst_total))
    if chosen is None:
        chosen = cands[-1]
    out["matched"] = True
    out["match_by"] = "content_chars" if content_chars and chosen["done"].get("chars") == content_chars else "closest total"
    out["chain_llm_first_token_s"] = round(chosen["ttft_s"], 3)
    out["chain_llm_total_s"] = round(chosen["total_s"], 3)
    out["chain_llm_done_ms_from_llm_start"] = chosen["done"].get("ms_from_llm0")
    out["chain_first_token_ms_from_turn_start"] = chosen["first_token"].get("ms_from_turn0")
    out["instrumented_ttft_s"] = inst_ttft
    out["instrumented_total_s"] = inst_total
    if inst_ttft is not None:
        out["ttft_delta_s"] = round(chosen["ttft_s"] - inst_ttft, 3)
        out["agrees"] = abs(chosen["ttft_s"] - inst_ttft) <= max(2.0, 0.15 * inst_ttft)
    else:
        out["agrees"] = None
    out["note"] = (
        "chain 'llm first token (X ms)' X counts from TURN start (includes "
        "ASR/memory before the LLM); the correlation here uses the log-line "
        "time delta llm start -> first token, which measures the LLM stage "
        "exactly like the instrumentation."
    )
    return out


# --------------------------------------------------------------------------- #
# wire equivalence proof (payload identity between capture and replay)
# --------------------------------------------------------------------------- #


def _wire_equivalence(turn: dict) -> dict[str, Any]:
    wire = turn.get("wire", {}) or {}
    start = turn.get("start", {}) or {}
    raw = wire.get("exact_wire_body_raw")
    parsed = wire.get("exact_wire_body")
    reconstructed = start.get("payload")
    out: dict[str, Any] = {
        "wire_raw_available": bool(raw),
        "reconstructed_available": bool(reconstructed),
        "wire_equals_reconstruction": (
            parsed == reconstructed if (parsed is not None and reconstructed is not None) else None
        ),
    }
    if raw and parsed is not None:
        try:
            # byte-exact reserialization with httpx's own encoder parameters
            reser = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            out["sha256_wire_text"] = _sha256_text(raw)
            out["sha256_reserialized"] = _sha256_text(reser)
            out["reserialized_equals_wire"] = reser == raw
        except Exception as exc:
            out["reserialize_error"] = repr(exc)[:200]
    return out


# --------------------------------------------------------------------------- #
# fresh llama-server management (same production configuration, new process)
# --------------------------------------------------------------------------- #


def _production_server_cmdline(session: Optional[dict]) -> Optional[str]:
    """The RUNNING production llama-server command line, from the trace."""
    try:
        proc_info = (session or {}).get("server", {}).get("llama_server_process")
    except Exception:
        return None
    if isinstance(proc_info, dict):
        cmd = proc_info.get("CommandLine")
        if isinstance(cmd, str) and "llama-server" in cmd.lower():
            return cmd
        procs = proc_info.get("processes")
        if isinstance(procs, list) and procs:
            first = str(procs[0])
            if "llama-server" in first.lower():
                return first
        return None
    if isinstance(proc_info, list):
        for item in proc_info:
            if isinstance(item, dict):
                cmd = item.get("CommandLine")
                if isinstance(cmd, str) and "llama-server" in cmd.lower():
                    return cmd
    return None


def _build_fresh_argv(
    production_cmd: Optional[str],
    port: int,
    session: Optional[dict],
    exe_override: Optional[str],
    model_override: Optional[str],
) -> tuple[Optional[str], list[str], list[str]]:
    """Rebuild the production command line with ONLY the port changed.

    Priority: the real production process command line from the trace
    session (byte-for-byte flags, port/host swapped); fallback: the flags
    from the session config + the resolved-model marker.
    """
    notes: list[str] = []
    if production_cmd:
        raw = production_cmd.strip()
        if os.name != "nt":
            m = re.match(r"^\s*(\d+)\s+(.*)$", raw, re.S)
            if m and "llama-server" in m.group(2):
                raw = m.group(2)
            tokens = shlex.split(raw)
        else:
            tokens = [t.strip('"') for t in shlex.split(raw, posix=False)]
        if not tokens:
            return None, [], ["production command line parsed to nothing"]
        exe = tokens[0]
        rest = tokens[1:]
        drop_value_flags = {"--port", "-port", "--host", "-host"}
        out: list[str] = []
        i = 0
        while i < len(rest):
            t = rest[i]
            tl = t.lower()
            if tl in drop_value_flags:
                i += 2  # drop flag + value (we set our own host/port at the end)
                continue
            if tl.startswith("--port=") or tl.startswith("-port=") or tl.startswith("--host=") or tl.startswith("-host="):
                i += 1
                continue
            if tl in ("--model", "-m") and i + 1 < len(rest):
                if model_override:
                    out += ["--model", model_override]
                    notes.append(f"model overridden -> {model_override}")
                else:
                    out += ["--model", rest[i + 1]]
                i += 2
                continue
            if tl.startswith("--model="):
                out.append(f"--model={model_override}" if model_override else t)
                if model_override:
                    notes.append(f"model overridden -> {model_override}")
                i += 1
                continue
            out.append(t)
            i += 1
        out += ["--host", "127.0.0.1", "--port", str(port)]
        argv = [exe_override or exe] + out
        notes.append(
            "command source: the production llama-server process command line "
            "captured in the trace session (only --host/--port replaced"
            + (", --model overridden" if model_override else "")
            + ")"
        )
        return argv[0], argv[1:], notes

    # ---- fallback: session config + resolved-model marker ------------------
    cfg = (session or {}).get("config", {}) or {}
    model = model_override
    if model is None:
        try:
            rm = json.loads(RESOLVED_MODEL_PATH.read_text(encoding="utf-8"))
            model = (
                rm.get("model_path")
                or rm.get("path")
                or rm.get("model")
                or rm.get("resolved_model")
            )
        except Exception:
            model = None
    if model is None:
        return None, [], [
            "no production command line in the trace and no resolved-model "
            "marker: pass --server-exe/--model explicitly for the fresh replay"
        ]
    exe = exe_override
    if exe is None:
        for cand in ("bin/llama-server.exe", "bin/llama-server"):
            if (REPO_ROOT / cand).exists():
                exe = str(REPO_ROOT / cand)
                break
    if exe is None:
        return None, [], ["llama-server executable not found: pass --server-exe"]
    ngl = cfg.get("llm_n_gpu_layers") or 26
    ctx = cfg.get("llm_context_size") or 8192
    argv = [
        "--model", model,
        "--host", "127.0.0.1", "--port", str(port),
        "-ngl", str(ngl),
        "-c", str(ctx),
        "--parallel", str(cfg.get("llm_parallel") or 1),
        "--cache-type-k", cfg.get("llm_cache_type_k") or "q8_0",
        "--cache-type-v", cfg.get("llm_cache_type_v") or "q8_0",
        "--temp", str(cfg.get("llm_temperature") or 0.7),
        "--metrics", "--no-webui", "--reasoning", "off",
    ]
    notes.append(
        "command source: session config + resolved-model marker (fallback; "
        "the production process command line was not captured)"
    )
    return exe, argv, notes


class FreshServer:
    """One fresh llama-server process with the production configuration."""

    def __init__(self, exe: str, args: list[str], port: int, log_path: Path):
        self.exe = exe
        self.args = args
        self.port = port
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.command_line = f'"{exe}" ' + " ".join(args)
        self.load_s: Optional[float] = None
        self.start_error: Optional[dict] = None
        self._log_h = None

    @property
    def base_root(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> bool:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._log_h = open(self.log_path, "w", encoding="utf-8", errors="replace")
            env = dict(os.environ)
            # POSIX: the b10717 build ships its shared libs next to the binary
            if os.name != "nt":
                env["LD_LIBRARY_PATH"] = str(Path(self.exe).parent)
            creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.proc = subprocess.Popen(
                [self.exe] + self.args,
                stdout=self._log_h,
                stderr=subprocess.STDOUT,
                cwd=str(Path(self.exe).parent),
                env=env,
                creationflags=creation,
            )
            return True
        except Exception as exc:
            self.start_error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            return False

    def wait_health(self, timeout: float) -> bool:
        import httpx

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=5.0) as client:
                while time.perf_counter() - t0 < timeout:
                    if self.proc is not None and self.proc.poll() is not None:
                        self.start_error = {
                            "type": "exited_early",
                            "returncode": self.proc.returncode,
                            "log_tail": self._read_log()[-2000:],
                        }
                        return False
                    try:
                        r = client.get(f"{self.base_root}/health")
                        if r.status_code == 200:
                            self.load_s = round(time.perf_counter() - t0, 2)
                            return True
                    except httpx.HTTPError:
                        pass
                    time.sleep(2.0)
        except Exception as exc:
            self.start_error = {"type": type(exc).__name__, "message": str(exc)[:300]}
        return False

    def _read_log(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def startup_lines(self) -> list[str]:
        """Effective-config evidence lines from the server's own startup log."""
        try:
            return _tr._log_tail_lines({str(self.log_path): 0})
        except Exception:
            return []

    def identity(self) -> dict[str, Any]:
        import httpx

        out: dict[str, Any] = {}
        try:
            with httpx.Client(base_url=self.base_root, timeout=5.0) as c:
                r = c.get("/v1/models")
                out["v1_models"] = r.json() if r.status_code == 200 else {"status": r.status_code}
                r = c.get("/props")
                out["props"] = r.json() if r.status_code == 200 else {"status": r.status_code}
        except Exception as exc:
            out["error"] = repr(exc)[:200]
        return out

    def stop(self) -> bool:
        ok = False
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
                ok = True
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
                    ok = True
                except Exception:
                    pass
        elif self.proc is not None:
            ok = True
        if self._log_h is not None:
            try:
                self._log_h.close()
            except OSError:
                pass
        time.sleep(2.0)  # let the driver release the model
        return ok


# --------------------------------------------------------------------------- #
# the direct replay (independent client, exact request bytes)
# --------------------------------------------------------------------------- #


async def replay_exact_request(
    base_url: str,
    body: dict,
    raw_text: Optional[str] = None,
    log_files: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Send THE exact captured request; measure every boundary marker.

    ``raw_text`` (the literal wire text from the trace) is sent as raw bytes
    -> the replay is byte-identical to the captured wire payload. Without it
    the parsed body is re-serialized with httpx's own encoder parameters.
    """
    import httpx

    out: dict[str, Any] = {
        "replayed_at": _utc_now(),
        "url": base_url + "/chat/completions",
        "sent_raw_wire_bytes": bool(raw_text),
    }
    sent_bytes: bytes
    if raw_text:
        sent_bytes = raw_text.encode("utf-8")
        out["sent_sha256"] = hashlib.sha256(sent_bytes).hexdigest()
        if raw_text:
            out["wire_sha256"] = _sha256_text(raw_text)
        content_kw = {"content": sent_bytes}
        headers_kw = {"Content-Type": "application/json"}
    else:
        sent_bytes = json.dumps(
            body, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        out["sent_sha256"] = hashlib.sha256(sent_bytes).hexdigest()
        content_kw = {"json": body}
        headers_kw = {}

    root = _server_root(base_url)
    log_offsets = log_files if log_files is not None else _tr._log_offsets()
    metrics_before = await async_metrics(root)

    markers: dict[str, Any] = {}
    sse_lines = 0
    reasoning_chars = 0
    content_chars = 0
    content_deltas = 0
    first_line = None
    error: Optional[dict] = None
    done_received = False
    t0 = time.perf_counter()
    try:
        timeout = httpx.Timeout(600.0, connect=10.0)
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            markers["http_request_start_rel"] = 0.0
            t0 = time.perf_counter()
            async with client.stream(
                "POST", "chat/completions", headers=headers_kw, **content_kw
            ) as response:
                markers["http_headers_received_rel"] = round(
                    time.perf_counter() - t0, 4
                )
                out["http_status"] = response.status_code
                async for line in response.aiter_lines():
                    if first_line is None:
                        first_line = line
                        markers["first_sse_line_rel"] = round(
                            time.perf_counter() - t0, 4
                        )
                    sse_lines += 1
                    if line.strip() == "data: [DONE]":
                        done_received = True
                    delta = parse_sse_content_delta(line)
                    if delta:
                        if "first_content_token_rel" not in markers:
                            markers["first_content_token_rel"] = round(
                                time.perf_counter() - t0, 4
                            )
                        content_chars += len(delta)
                        content_deltas += 1
                        continue
                    reasoning_chars += len(parse_sse_reasoning_delta(line))
                markers["stream_end_rel"] = round(time.perf_counter() - t0, 4)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
        try:
            markers["error_rel"] = round(time.perf_counter() - t0, 4)
        except Exception:
            pass

    metrics_after = await async_metrics(root)
    metrics_delta: dict[str, float] = {}
    if metrics_before and metrics_after:
        for k, v_after in metrics_after.items():
            v_before = metrics_before.get(k, 0.0)
            if v_after != v_before:
                metrics_delta[k] = round(v_after - v_before, 4)
    out["markers"] = markers
    out["sse_lines"] = sse_lines
    out["content_deltas"] = content_deltas
    out["content_chars"] = content_chars
    out["reasoning_chars"] = reasoning_chars
    out["done_marker_received"] = done_received
    out["first_sse_line_preview"] = (first_line or "")[:200]
    out["error"] = error
    out["metrics_before"] = metrics_before
    out["metrics_after"] = metrics_after
    out["metrics_delta"] = metrics_delta
    out["llama_server_log_lines"] = _tr._log_tail_lines(log_offsets)
    out["server_print_timing"] = _parse_print_timing(out["llama_server_log_lines"])
    out["kv_cache_metrics"] = {
        k: {"before": metrics_before.get(k) if metrics_before else None,
            "after": metrics_after.get(k) if metrics_after else None}
        for k in list((metrics_before or {}).keys()) + list((metrics_after or {}).keys())
        if "kv" in k.lower()
    } if (metrics_before or metrics_after) else {}
    return out


# --------------------------------------------------------------------------- #
# server-side state capture (warm endpoint)
# --------------------------------------------------------------------------- #


def _server_root(url: str) -> str:
    root = url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def sync_server_info(base_root: str) -> dict[str, Any]:
    import httpx

    info: dict[str, Any] = {"root_url": base_root}
    try:
        with httpx.Client(base_url=base_root, timeout=httpx.Timeout(3.0)) as c:
            try:
                r = c.get("/health")
                info["health"] = {"status": r.status_code}
            except Exception as exc:
                info["health"] = {"error": repr(exc)[:200]}
            try:
                r = c.get("/metrics")
                info["metrics_cumulative"] = (
                    _tr._parse_prometheus(r.text) if r.status_code == 200 else None
                )
            except Exception as exc:
                info["metrics_cumulative"] = {"error": repr(exc)[:200]}
            try:
                r = c.get("/props")
                info["props"] = r.json() if r.status_code == 200 else None
            except Exception as exc:
                info["props"] = {"error": repr(exc)[:200]}
    except Exception as exc:
        info["http_error"] = repr(exc)[:200]
    info["llama_server_process"] = _tr._llama_server_process_cmdline()
    return info


async def async_metrics(base_root: str) -> Optional[dict[str, float]]:
    import httpx

    try:
        async with httpx.AsyncClient(base_url=base_root, timeout=3.0) as c:
            r = await c.get("/metrics")
            if r.status_code == 200:
                return _tr._parse_prometheus(r.text)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# comparison: the 22 measured points + the A-G decision tree + verdicts
# --------------------------------------------------------------------------- #


def build_result(
    session: Optional[dict],
    turn: dict,
    warm_replay: Optional[dict],
    fresh_block: Optional[dict],
    args,
    chain_corr: dict,
    wire_eq: dict,
    merge_info: Optional[dict],
) -> dict:
    start = turn.get("start", {})
    end = turn.get("end", {})
    wire = turn.get("wire", {})
    headers = turn.get("headers", {})
    p_markers = end.get("markers", {}) or {}
    p_delta = end.get("metrics_delta", {}) or {}

    prod_ttft = _pick(p_markers, "first_content_token_rel")
    prod_total = _pick(p_markers, "llm_request_end_rel", "stream_end_rel")
    pre_http = _pick(p_markers, "body_start_rel", "http_request_start_rel")
    http_start = _pick(p_markers, "http_request_start_rel")
    headers_rel = _pick(p_markers, "http_headers_received_rel")
    first_line_rel = _pick(p_markers, "first_sse_line_rel")
    stream_end = _pick(p_markers, "stream_end_rel")
    llm_end = _pick(p_markers, "llm_request_end_rel")

    prompt_tokens = _pick(p_delta, METRIC_PTOK)
    prompt_seconds = _pick(p_delta, METRIC_PSEC)
    gen_tokens = _pick(p_delta, METRIC_GTOK)
    gen_seconds = _pick(p_delta, METRIC_GSEC)

    prod_print_timing = _parse_print_timing(end.get("llama_server_log_lines") or [])
    if prompt_tokens is None:
        prompt_tokens = float(prod_print_timing["prompt_tokens"]) if "prompt_tokens" in prod_print_timing else None
        prompt_seconds = (
            prod_print_timing["prompt_ms"] / 1000.0 if "prompt_ms" in prod_print_timing else None
        )
    if gen_tokens is None and "eval_tokens" in prod_print_timing:
        gen_tokens = float(prod_print_timing["eval_tokens"])
        gen_seconds = prod_print_timing["eval_ms"] / 1000.0

    prompt_tok_s = (
        round(prompt_tokens / prompt_seconds, 2) if prompt_tokens and prompt_seconds else None
    )
    gen_tok_s = round(gen_tokens / gen_seconds, 2) if gen_tokens and gen_seconds else None

    # ---- warm replay numbers -----------------------------------------------
    w: dict = warm_replay or {}
    w_markers = w.get("markers", {}) or {}
    w_delta = w.get("metrics_delta", {}) or {}
    w_ttft = _pick(w_markers, "first_content_token_rel")
    w_total = _pick(w_markers, "stream_end_rel")
    w_ptok = _pick(w_delta, METRIC_PTOK)
    w_psec = _pick(w_delta, METRIC_PSEC)
    w_gtok = _pick(w_delta, METRIC_GTOK)
    w_gsec = _pick(w_delta, METRIC_GSEC)
    warm_cache_hit = bool(
        w_ptok is not None and prompt_tokens is not None and w_ptok <= max(4, 0.02 * prompt_tokens)
    )
    warm_prompt_eval_s = w_psec
    w_tok_s = round(w_ptok / w_psec, 2) if w_ptok and w_psec else None

    # ---- fresh replay numbers ----------------------------------------------
    f: dict = {}
    f_server: dict = {}
    f_delta: dict = {}
    f_ttft = None
    f_total = None
    f_ptok = None
    f_psec = None
    f_tok_s = None
    f_gtok = None
    f_gsec = None
    fresh_ok = False
    if fresh_block:
        f_server = {k: v for k, v in fresh_block.items() if k != "replay"}
        f = fresh_block.get("replay") or {}
        f_markers = f.get("markers", {}) or {}
        f_delta = f.get("metrics_delta", {}) or {}
        f_ttft = _pick(f_markers, "first_content_token_rel")
        f_total = _pick(f_markers, "stream_end_rel")
        f_ptok = _pick(f_delta, METRIC_PTOK)
        f_psec = _pick(f_delta, METRIC_PSEC)
        f_gtok = _pick(f_delta, METRIC_GTOK)
        f_gsec = _pick(f_delta, METRIC_GSEC)
        f_ptiming = f.get("server_print_timing") or {}
        if f_ptok is None and "prompt_tokens" in f_ptiming:
            f_ptok = float(f_ptiming["prompt_tokens"])
        if f_psec is None and "prompt_ms" in f_ptiming:
            f_psec = f_ptiming["prompt_ms"] / 1000.0
        f_tok_s = round(f_ptok / f_psec, 2) if f_ptok and f_psec else None
        fresh_ok = f_ttft is not None

    # ---- production phase decomposition (where the TTFT was spent) ---------
    def _gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return round(b - a, 3)

    phases = {
        "pre_http_llm_call_to_http_request_start_s": pre_http,
        "http_request_start_to_headers_s": _gap(http_start, headers_rel),
        "headers_to_first_sse_line_s": _gap(headers_rel, first_line_rel),
        "first_sse_line_to_first_content_token_s": _gap(first_line_rel, prod_ttft),
        "first_content_token_to_stream_end_s": _gap(prod_ttft, stream_end),
        "stream_end_to_llm_completion_s": _gap(stream_end, llm_end),
    }

    # ---- slow-turn qualification -------------------------------------------
    slow_threshold = float(args.slow_threshold)
    slow_turn = bool(prod_ttft is not None and prod_ttft >= slow_threshold)

    # ---- decision tree (A-G), measured numbers only -------------------------
    dt = _classify_case(
        prod_ttft=prod_ttft,
        phases=phases,
        prompt_seconds=prompt_seconds,
        fresh_ttft=f_ttft,
        warm_ttft=w_ttft,
        warm_cache_hit=warm_cache_hit,
        slow_threshold=slow_threshold,
        slow_turn=slow_turn,
        fresh_attempted=bool(fresh_block and fresh_block.get("attempted")),
    )

    # ---- ngl26 / context evidence -------------------------------------------
    server_lines = (end.get("llama_server_log_lines", []) or []) + (
        (f.get("llama_server_log_lines") or []) if f else []
    )
    fit_fail = [ln for ln in server_lines if "failed to fit" in ln.lower()]
    overflow_lines = [
        ln for ln in server_lines
        if "n_ctx" in ln.lower() and ("trunc" in ln.lower() or "overflow" in ln.lower() or ">" in ln)
    ][:3]
    session_cfg = (session or {}).get("config", {}) or {}
    prod_ctx = session_cfg.get("llm_context_size")
    ctx_overflow_ruled_out = bool(
        prompt_tokens is not None
        and prod_ctx
        and (prompt_tokens + float((wire.get("exact_wire_body") or {}).get("max_tokens") or 512)) < float(prod_ctx)
        and not overflow_lines
    )
    inside_server = dt.get("inside_server") or False
    fresh_reproduces = dt.get("fresh_reproduces")
    ngl26_proven = bool(
        inside_server
        and fresh_reproduces
        and prompt_tok_s is not None
        and prompt_tok_s < 10.0
        and (
            (f_tok_s is not None and f_tok_s < 10.0)
            or (w_tok_s is not None and not warm_cache_hit and w_tok_s < 10.0)
        )
    )
    ngl_evidence = {
        "server_fit_failure_lines": fit_fail[:5],
        "production_prefill_tok_s_this_request": prompt_tok_s,
        "fresh_replay_prefill_tok_s": f_tok_s,
        "warm_replay_prefill_tok_s": w_tok_s,
        "warm_replay_cache_hit": warm_cache_hit,
        "ngl_from_session_config": session_cfg.get("llm_n_gpu_layers"),
        "note": (
            "ngl26 is proven responsible for THIS exact request ONLY when the "
            "delay is inside llama-server, the fresh replay reproduces it, and "
            "the measured prefill rate for the exact request collapsed "
            "(~4-10 tok/s class). Otherwise: NOT PROVEN for this request."
        ),
    }
    ctx_evidence = {
        "prompt_tokens_this_request": prompt_tokens,
        "context_size_from_session_config": prod_ctx,
        "truncation_or_overflow_lines": overflow_lines,
        "ruled_out_with_high_confidence": ctx_overflow_ruled_out,
        "note": (
            "Context size is proven responsible ONLY if truncation/overflow "
            "evidence exists for THIS request; tokens+max_tokens comfortably "
            "below n_ctx with no truncation lines rules it out for THIS request."
        ),
    }

    # ---- the 16 verdict answers (A-P) ---------------------------------------
    verdict = _verdict_16(
        slow_turn=slow_turn,
        slow_threshold=slow_threshold,
        prod_ttft=prod_ttft,
        prod_total=prod_total,
        w_ttft=w_ttft,
        f_ttft=f_ttft,
        warm_cache_hit=warm_cache_hit,
        prompt_tokens=prompt_tokens,
        prompt_tok_s=prompt_tok_s,
        prompt_seconds=prompt_seconds,
        f_tok_s=f_tok_s,
        gen_tokens=gen_tokens,
        gen_tok_s=gen_tok_s,
        gen_seconds=gen_seconds,
        phases=phases,
        dt=dt,
        ngl26_proven=ngl26_proven,
        ctx_overflow_ruled_out=ctx_overflow_ruled_out,
        fresh_block=fresh_block,
    )

    # ---- measured facts / inferences / unresolved ---------------------------
    measured: list[str] = []
    inferred: list[str] = []
    unresolved: list[str] = []
    measured.append(
        f"production TTFT (llm_request_start -> first content token, instrumented): {_fmt(prod_ttft)}"
    )
    measured.append(f"production LLM total: {_fmt(prod_total)}")
    if prompt_tokens is not None:
        measured.append(
            f"exact prompt tokens for this request (server /metrics delta + print_timing): {prompt_tokens:.0f}"
        )
    if prompt_seconds is not None:
        measured.append(f"server-side prompt processing seconds for this request: {_fmt(prompt_seconds)}")
    if gen_tokens is not None:
        measured.append(
            f"generated tokens / seconds / rate: {gen_tokens:.0f} tok / {_fmt(gen_seconds)} / {_fmt(gen_tok_s, 'tok/s')}"
        )
    for k, v in phases.items():
        if v is not None:
            measured.append(f"production phase {k}: {v}")
    if w_ttft is not None:
        measured.append(
            f"warm direct replay (running production server): TTFT {_fmt(w_ttft)}, total {_fmt(w_total)}"
            + (", KV/prompt CACHE HIT (prompt tokens delta ~1)" if warm_cache_hit else "")
        )
    if fresh_ok:
        measured.append(
            f"fresh direct replay (new process, same config, cold KV): TTFT {_fmt(f_ttft)}, total {_fmt(f_total)}"
        )
        if f_psec is not None:
            measured.append(f"fresh replay server-side prefill seconds: {_fmt(f_psec)}")
    if chain_corr.get("matched"):
        measured.append(
            f"production web-server [chain] log for the same turn: llm start -> first token "
            f"{_fmt(chain_corr.get('chain_llm_first_token_s'))}, llm total "
            f"{_fmt(chain_corr.get('chain_llm_total_s'))} (independent log-line clock)"
        )
    measured.append(
        f"wire payload identity: raw wire text available={wire_eq.get('wire_raw_available')}, "
        f"re-serialized == wire: {wire_eq.get('reserialized_equals_wire')}, "
        f"wire == reconstructed payload: {wire_eq.get('wire_equals_reconstruction')}"
    )

    inferred.append(
        f"decision-tree classification: CASE {dt.get('case')} - {dt.get('case_label')} "
        f"(confidence: {dt.get('confidence')})"
    )
    inferred.append("ngl26 responsible for THIS request: " + ("YES (see measured prefill rates)" if ngl26_proven else "NOT PROVEN"))
    inferred.append(
        "context size responsible for THIS request: "
        + ("NO (ruled out with high confidence)" if ctx_overflow_ruled_out else "NOT PROVEN")
    )
    if not slow_turn and prod_ttft is not None:
        inferred.append(
            "the selected turn is BELOW the slow threshold - this is a diagnostic "
            "capture, NOT a reproduction of the production problem"
        )

    if prod_ttft is None:
        unresolved.append("the selected turn has no first-token marker")
    if w_ttft is None and warm_replay_requested(args):
        unresolved.append("warm direct replay not measured")
    if fresh_block and not fresh_ok:
        unresolved.append(
            "fresh direct replay not measured: " + json.dumps(fresh_block.get("start_error") or "no result")[:300]
        )
    if not chain_corr.get("matched"):
        unresolved.append("production [chain] log correlation not available")
    if not unresolved:
        unresolved.append("none - all planned measurements completed for this turn")

    messages = (start.get("payload") or {}).get("messages") or []
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), None)

    result = {
        "schema": "voicemem-llm-prod-capture/2",
        "generated_at": _utc_now(),
        "change_policy": (
            "forensic only - no production/config change; warm replay targets "
            "the RUNNING production server (never restarted/reconfigured); the "
            "fresh replay starts a SEPARATE process with the production command "
            "line (only the port differs) and stops it afterwards"
        ),
        "selected_turn": {
            "trace_id": turn.get("trace_id"),
            "selection": "slowest first content token (default)",
            "start_wall": start.get("ts_wall"),
            "first_token_wall": _add_wall(start.get("ts_wall"), prod_ttft),
            "message_stats": start.get("message_stats"),
            "last_user_utterance": last_user,
            "http_client_existed_before": start.get("http_client_existed_before"),
            "first_request_after_server_start": (
                (end.get("metrics_before") or {}).get(METRIC_PTOK) in (0.0, 0)
            ),
            "payload_source": "wire (exact bytes captured at send time)"
            if wire.get("exact_wire_body_raw") or wire.get("exact_wire_body")
            else "start-event reconstruction",
            "request_params": {
                k: v
                for k, v in (wire.get("exact_wire_body") or start.get("payload") or {}).items()
                if k != "messages"
            },
            "slow_turn_qualified": slow_turn,
            "slow_threshold_s": slow_threshold,
            "server_url_from_trace": start.get("server_url"),
        },
        "production_observation": {
            "llm_request_start_rel": 0.0,
            "body_start_rel": p_markers.get("body_start_rel"),
            "http_request_start_rel": p_markers.get("http_request_start_rel"),
            "http_headers_received_rel": p_markers.get("http_headers_received_rel"),
            "http_send_elapsed_s": headers.get("http_send_elapsed_s"),
            "http_status": headers.get("http_status"),
            "first_sse_line_rel": p_markers.get("first_sse_line_rel"),
            "first_content_token_rel": p_markers.get("first_content_token_rel"),
            "stream_end_rel": p_markers.get("stream_end_rel"),
            "llm_request_end_rel": p_markers.get("llm_request_end_rel"),
            "forensic_end_rel": p_markers.get("forensic_end_rel"),
            "sse_lines": end.get("sse_lines"),
            "content_delta_count": end.get("content_delta_count"),
            "content_chars": end.get("content_chars"),
            "error": end.get("error"),
            "metrics_delta": p_delta,
            "server_print_timing": prod_print_timing,
            "llama_server_log_lines": end.get("llama_server_log_lines"),
        },
        "warm_replay": w,
        "fresh_replay": fresh_block,
        "comparison": {
            # the 22 required comparison points
            "01_production_request_start_wall": start.get("ts_wall"),
            "02_production_first_token_wall": _add_wall(start.get("ts_wall"), prod_ttft),
            "03_production_ttft_s": prod_ttft,
            "04_production_total_latency_s": prod_total,
            "05_direct_warm_replay_ttft_s": w_ttft,
            "06_direct_warm_replay_total_s": w_total,
            "07_direct_fresh_replay_ttft_s": f_ttft,
            "08_direct_fresh_replay_total_s": f_total,
            "09_exact_prompt_token_count": None if prompt_tokens is None else int(prompt_tokens),
            "10_prompt_tokens_per_second_production": prompt_tok_s,
            "10b_prompt_tokens_per_second_fresh": f_tok_s,
            "11_generated_token_count": None if gen_tokens is None else int(gen_tokens),
            "12_generation_tokens_per_second": gen_tok_s,
            "13_kv_cache_state": {
                "warm_replay_cache_hit": warm_cache_hit,
                "warm_replay_prompt_tokens_evaluated": None if w_ptok is None else int(w_ptok),
                "warm_replay_prompt_seconds": warm_prompt_eval_s,
                "fresh_replay_cold_by_construction": True if fresh_block else None,
                "kv_metrics": (w.get("kv_cache_metrics") or {}),
            },
            "14_server_side_prompt_evaluation_time_s": prompt_seconds,
            "15_server_side_generation_time_s": gen_seconds,
            "16_http_transport_delay_s": phases.get("http_request_start_to_headers_s"),
            "17_production_llm_call_to_http_request_start_s": phases.get("pre_http_llm_call_to_http_request_start_s"),
            "18_http_request_start_to_headers_s": phases.get("http_request_start_to_headers_s"),
            "19_headers_to_first_sse_line_s": phases.get("headers_to_first_sse_line_s"),
            "20_first_sse_line_to_first_content_token_s": phases.get("first_sse_line_to_first_content_token_s"),
            "21_first_content_token_to_stream_end_s": phases.get("first_content_token_to_stream_end_s"),
            "22_stream_end_to_llm_completion_s": phases.get("stream_end_to_llm_completion_s"),
            "difference_first_token_s_production_minus_fresh": (
                round(prod_ttft - f_ttft, 3) if prod_ttft is not None and f_ttft is not None else None
            ),
            "difference_first_token_s_production_minus_warm": (
                round(prod_ttft - w_ttft, 3) if prod_ttft is not None and w_ttft is not None else None
            ),
        },
        "phase_decomposition_of_production_ttft": phases,
        "decision_tree": dt,
        "ngl26_evidence": ngl_evidence,
        "context_evidence": ctx_evidence,
        "chain_log_correlation": chain_corr,
        "wire_equivalence": wire_eq,
        "quality_control": {
            "captured_turn_genuinely_slow": slow_turn,
            "chain_log_agrees_with_instrumentation": chain_corr.get("agrees"),
            "captured_request_contains_real_user_utterance": bool(last_user),
            "messages_array_preserved": bool(messages),
            "wire_body_captured": bool(wire.get("exact_wire_body")),
            "wire_body_and_reconstruction_proven_equivalent": wire_eq.get("wire_equals_reconstruction"),
            "replay_byte_identical_to_wire_payload": bool(w.get("sent_raw_wire_bytes") and w.get("sent_sha256") == w.get("wire_sha256")),
            "direct_replay_uses_same_model": _same_model_check(session, f_server, w),
            "direct_replay_uses_same_server_for_warm": True,
            "warm_and_fresh_distinguished": True,
            "fresh_server_configuration": f_server.get("command_line"),
            "fresh_server_effective_config_lines": f_server.get("startup_log_lines"),
            "server_side_metrics_included": bool(p_delta or f_delta),
            "no_production_source_file_modified": True,
            "no_production_configuration_changed": True,
        },
        "measured_facts": measured,
        "inferences": inferred,
        "unresolved_questions": unresolved,
        "verdict_16": verdict,
        "session": session,
        "merge": merge_info,
    }
    return result


def warm_replay_requested(args) -> bool:
    return args.replay in ("both", "warm")


def _same_model_check(session: Optional[dict], f_server: dict, w: dict) -> Optional[bool]:
    """Did the fresh server serve the same model id the app configures?"""
    try:
        cfg_model = ((session or {}).get("config") or {}).get("llm_model_name")
        models = (f_server.get("identity") or {}).get("v1_models") or {}
        data = models.get("data") if isinstance(models, dict) else None
        served = None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            served = data[0].get("id")
        if served is None or cfg_model is None:
            return None
        return served == cfg_model
    except Exception:
        return None


def _add_wall(wall: Optional[str], rel: Optional[float]) -> Optional[str]:
    if not wall or rel is None:
        return None
    try:
        t = datetime.strptime(wall.replace("Z", "+00:00"), "%Y-%m-%dT%H:%M:%S.%f%z") + timedelta(seconds=rel)
        return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# decision-tree classifier (cases A-G)
# --------------------------------------------------------------------------- #


def _classify_case(
    prod_ttft: Optional[float],
    phases: dict,
    prompt_seconds: Optional[float],
    fresh_ttft: Optional[float],
    warm_ttft: Optional[float],
    warm_cache_hit: bool,
    slow_threshold: float,
    slow_turn: bool,
    fresh_attempted: bool,
) -> dict:
    """Classify the delay location from measured numbers only.

    Precedence (first match wins):
      E  delay before the HTTP request leaves the app
      F  delay inside llama-server before the first SSE line (prefill/handling)
      D/G streaming or app-side consumption (line arrives, content token late)
      C  fresh replay slow + warm replay fast (cache hit) -> KV cache state
      B  fresh replay slow -> the request itself / server configuration
      A  fresh replay fast -> application path / server state, not the request
    """
    reasoning: list[str] = []
    if prod_ttft is None:
        return {
            "case": "INCONCLUSIVE",
            "case_label": "no first-token marker for the production turn",
            "confidence": "none",
            "reasoning": ["capture a completed turn first"],
            "inside_server": None,
            "fresh_reproduces": None,
            "next_measurement": "re-run the instrumented backend and capture a completed turn",
        }
    if not slow_turn:
        return {
            "case": "NOT_SLOW",
            "case_label": (
                f"the selected turn is below the slow threshold "
                f"({_fmt(prod_ttft)} < {slow_threshold:.0f} s)"
            ),
            "confidence": "high",
            "reasoning": [
                "keep using VoiceMem under the instrumented backend until a real "
                "30-50 s turn occurs; the numbers in this result are diagnostic only"
            ],
            "inside_server": None,
            "fresh_reproduces": None,
            "next_measurement": "continue monitoring for a real slow turn (TTFT >= 30 s)",
        }

    pre = phases.get("pre_http_llm_call_to_http_request_start_s")
    to_headers = phases.get("http_request_start_to_headers_s")
    to_line = phases.get("headers_to_first_sse_line_s")
    to_token = phases.get("first_sse_line_to_first_content_token_s")

    fresh_reproduces: Optional[bool] = None
    if fresh_ttft is not None:
        # 'slow on fresh' = at least half the production TTFT AND inside the
        # slow class (0.75x the slow threshold) - the floor scales with the
        # threshold instead of being an absolute constant
        fresh_reproduces = fresh_ttft >= max(0.5 * prod_ttft, 0.75 * slow_threshold)
    warm_fast = warm_ttft is not None and warm_ttft <= max(0.10 * prod_ttft, 2.0)

    reasoning.append(f"production TTFT {_fmt(prod_ttft)} (slow turn: YES)")
    reasoning.append(f"phase decomposition: {json.dumps(phases)}")
    if prompt_seconds is not None:
        reasoning.append(f"server-side prompt seconds for this request: {_fmt(prompt_seconds)} "
                         f"({(prompt_seconds / prod_ttft) * 100:.0f}% of TTFT)")

    # CASE E: before the HTTP request
    if pre is not None and pre >= 5.0 and pre >= 0.25 * prod_ttft:
        return {
            "case": "E",
            "case_label": "the delay occurs BEFORE the HTTP request reaches llama-server",
            "confidence": "high",
            "reasoning": reasoning + [
                f"llm_request_start -> http_request_start = {_fmt(pre)} dominates: the request "
                "had not even left the application when the delay elapsed. Investigate "
                "application-side blocking, synchronous work, locking, event-loop "
                "scheduling or resource contention between chat_stream entry and the "
                "httpx send."
            ],
            "inside_server": False,
            "fresh_reproduces": fresh_reproduces,
            "next_measurement": "profile the application between LlmClient.chat_stream entry and httpx send",
        }

    # CASE F-family: delay before the first SSE line (inside the server)
    line_side = (to_line if to_line is not None else 0.0) + (to_headers if to_headers is not None else 0.0)
    server_seconds_share = (prompt_seconds / prod_ttft) if prompt_seconds else None
    inside_server = bool(
        (line_side >= 0.5 * prod_ttft and line_side >= 5.0)
        or (server_seconds_share is not None and server_seconds_share >= 0.5)
    )

    # CASE D/G: streaming / app-side consumption
    stream_side = bool(
        to_token is not None and to_token >= 5.0 and to_token >= 0.25 * prod_ttft
        and (prompt_seconds is None or prompt_seconds < 0.3 * prod_ttft)
    )

    if stream_side and not inside_server:
        return {
            "case": "G",
            "case_label": "server streams quickly but the application receives the first content token late "
                          "(transport / SSE parsing / app-side streaming)",
            "confidence": "high",
            "reasoning": reasoning + [
                f"headers arrived, then the first SSE line, but the first content token "
                f"only {to_token:.1f} s after the first line (first line "
                f"{_fmt(phases.get('headers_to_first_sse_line_s'))} s after headers); "
                f"server-side prompt seconds were only {_fmt(prompt_seconds)}. "
                "Investigate SSE parsing, buffering, the async event loop, proxying and "
                "stream consumption on the application side."
            ],
            "inside_server": False,
            "fresh_reproduces": fresh_reproduces,
            "next_measurement": "instrument aiter_lines vs first content delta inside the app SSE loop",
        }

    if inside_server:
        base = {
            "inside_server": True,
            "fresh_reproduces": fresh_reproduces,
        }
        if fresh_reproduces is True and warm_fast and warm_cache_hit:
            return {
                **base,
                "case": "C",
                "case_label": "KV cache state is decisive: the exact request is slow on a fresh server, "
                              "fast on the warm one (cache hit)",
                "confidence": "high",
                "reasoning": reasoning + [
                    f"fresh replay of the exact request: TTFT {_fmt(fresh_ttft)} (slow, reproduces); "
                    f"warm replay: TTFT {_fmt(warm_ttft)} (CACHE HIT - prompt tokens delta ~1). "
                    "The production turn performed a full cold prefill; measure why the "
                    "production cache is cold/ineffective (slot state, cache_reuse, prompt changes)."
                ],
                "next_measurement": "llama-server cache_reuse / slot state lines around the production turn",
            }
        if fresh_reproduces is True:
            return {
                **base,
                "case": "B",
                "case_label": "the exact request itself produces the delay inside llama-server "
                              "(prompt processing / offload / server configuration)",
                "confidence": "high",
                "reasoning": reasoning + [
                    f"fresh replay of the exact request: TTFT {_fmt(fresh_ttft)} - the request is slow "
                    "against a freshly started server with the production configuration too. "
                    "Investigate llama-server prompt processing, GPU offload, context, KV cache, "
                    "request shape and server configuration for THIS request."
                ],
                "next_measurement": "controlled configuration experiments with THIS exact payload (only now justified)",
            }
        if fresh_reproduces is False:
            return {
                **base,
                "case": "A",
                "case_label": "the model/server does not reproduce the production delay on a fresh "
                              "server with the same configuration - the delay was a one-time "
                              "server state (or application path) event",
                "confidence": "medium",
                "reasoning": reasoning + [
                    f"fresh replay of the exact request: TTFT {_fmt(fresh_ttft)} (fast); the same request "
                    "against a fresh server with the production configuration is fast. The production "
                    "turn was slow inside the server (prefill share "
                    f"{'' if server_seconds_share is None else f'{server_seconds_share*100:.0f}%'}) but does not "
                    "reproduce: investigate server state at the time (queueing, other requests, "
                    "thermal/power state) or application-side queueing."
                ],
                "next_measurement": "re-capture the next slow turn and compare the /metrics queue state",
            }
        # fresh not measured
        return {
            **base,
            "case": "F",
            "case_label": "the delay is inside llama-server (prompt processing before the first output)",
            "confidence": "medium",
            "reasoning": reasoning + [
                "the production turn's delay sits between request handoff and the first "
                "SSE line, and the server-side prompt seconds account for it; the fresh "
                "replay measurement is missing, so the A/B/C refinement is pending."
            ],
            "next_measurement": "complete the fresh replay (--replay fresh-only after freeing VRAM)",
        }

    # gaps do not dominate (unusual) - fall back to the replay comparison
    if fresh_reproduces is True:
        return {
            "case": "B",
            "case_label": "the exact request itself produces the delay (replay reproduces)",
            "confidence": "medium",
            "reasoning": reasoning + [
                f"fresh replay TTFT {_fmt(fresh_ttft)} reproduces the production latency."
            ],
            "inside_server": True,
            "fresh_reproduces": True,
            "next_measurement": "controlled configuration experiments with THIS exact payload",
        }
    if fresh_reproduces is False:
        return {
            "case": "A",
            "case_label": "application path / server state: the direct replays are fast while production was slow",
            "confidence": "medium",
            "reasoning": reasoning + [
                f"fresh replay TTFT {_fmt(fresh_ttft)}, warm replay TTFT {_fmt(warm_ttft)} - both fast; "
                "the request itself is fine. Investigate application runtime, request "
                "handling, HTTP transport, event loop, locking, queueing, streaming."
            ],
            "inside_server": False,
            "fresh_reproduces": False,
            "next_measurement": "application-side profiling between ASR done and first answer_delta",
        }
    return {
        "case": "INCONCLUSIVE",
        "case_label": "the measured numbers do not cleanly separate the cases",
        "confidence": "low",
        "reasoning": reasoning
        + [
            "fresh replay missing and no dominant phase; capture the next slow turn."
        ],
        "inside_server": False,
        "fresh_reproduces": None,
        "next_measurement": "re-run on the next slow turn with warm + fresh replays",
    }


# --------------------------------------------------------------------------- #
# the 16 verdict answers
# --------------------------------------------------------------------------- #


def _verdict_16(
    slow_turn: bool,
    slow_threshold: float,
    prod_ttft: Optional[float],
    prod_total: Optional[float],
    w_ttft: Optional[float],
    f_ttft: Optional[float],
    warm_cache_hit: bool,
    prompt_tokens: Optional[float],
    prompt_tok_s: Optional[float],
    prompt_seconds: Optional[float],
    f_tok_s: Optional[float],
    gen_tokens: Optional[float],
    gen_tok_s: Optional[float],
    gen_seconds: Optional[float],
    phases: dict,
    dt: dict,
    ngl26_proven: bool,
    ctx_overflow_ruled_out: bool,
    fresh_block: Optional[dict],
) -> dict:
    pre = phases.get("pre_http_llm_call_to_http_request_start_s")
    to_line = (phases.get("http_request_start_to_headers_s") or 0) + (
        phases.get("headers_to_first_sse_line_s") or 0
    )
    to_token = phases.get("first_sse_line_to_first_content_token_s")
    direct_desc = []
    if w_ttft is not None:
        direct_desc.append(f"warm (running production server): {_fmt(w_ttft)}"
                           + (" [CACHE HIT - measures the warm slot]" if warm_cache_hit else ""))
    if f_ttft is not None:
        direct_desc.append(f"fresh process, same production config, cold KV: {_fmt(f_ttft)}")
    return {
        "A_real_slow_production_turn_captured": (
            f"{'YES' if slow_turn else 'NO'} - production TTFT {_fmt(prod_ttft)} "
            f"(threshold {slow_threshold:.0f} s)"
            + ("" if slow_turn else "; KEEP MONITORING - this result is diagnostic only")
        ),
        "B_exact_production_ttft": _fmt(prod_ttft),
        "C_exact_production_total_latency": _fmt(prod_total),
        "D_direct_replay_ttft_same_exact_request": "; ".join(direct_desc) or "NOT MEASURED",
        "E_was_the_direct_test_warm_or_cold": (
            "warm replay = the RUNNING production server (state: "
            + ("KV cache hit" if warm_cache_hit else "prompt re-evaluated")
            + "); fresh replay = a NEW process with the production command line "
              "(cold by construction)"
        ),
        "F_prompt_tokens_processed": (
            f"{prompt_tokens:.0f}" if prompt_tokens is not None else "NOT MEASURED"
        ),
        "G_prefill_rate": (
            f"production {_fmt(prompt_tok_s, 'tok/s')}"
            + (f"; fresh replay {_fmt(f_tok_s, 'tok/s')}" if f_tok_s is not None else "")
            if (prompt_tok_s is not None or f_tok_s is not None)
            else "NOT MEASURED"
        ),
        "H_was_generation_itself_slow": (
            f"generated {gen_tokens:.0f} tokens in {_fmt(gen_seconds)} "
            f"({_fmt(gen_tok_s, 'tok/s')})"
            if gen_tokens is not None
            else "NOT MEASURED (server-side generation seconds unavailable)"
        ),
        "I_was_kv_cache_involved": (
            f"warm replay cache hit: {'YES' if warm_cache_hit else 'NO'}; "
            "fresh replay: cold by construction; production turn prefill "
            + ("re-evaluated (cold)" if prompt_tokens is not None and prompt_tokens > 10 else "unknown")
        ),
        "J_delay_before_http_request": (
            f"{_fmt(pre)} - {'YES' if (pre or 0) >= 5.0 else 'NO'}"
        ),
        "K_delay_inside_llama_server": (
            f"server-side prompt seconds {_fmt(prompt_seconds)}; request->first-line window "
            f"{_fmt(to_line if to_line else None)}; prompt share of TTFT "
            + (
                f"{(prompt_seconds / prod_ttft) * 100:.0f}%"
                if (prompt_seconds is not None and prod_ttft)
                else "UNKNOWN"
            )
            + " - "
            + ("YES" if dt.get("inside_server") else ("NO" if dt.get("inside_server") is False else "NOT CLASSIFIED"))
        ),
        "L_delay_during_streaming": (
            f"first SSE line -> first content token gap {_fmt(to_token)} - "
            + ("YES" if (to_token or 0) >= 5.0 else "NO")
        ),
        "M_ngl26_proven_responsible_for_this_request": (
            "YES - measured prefill collapse on the exact request (see ngl26_evidence)"
            if ngl26_proven
            else "NOT PROVEN for this exact request (no ngl change is justified by this capture alone)"
        ),
        "N_context_size_proven_responsible_for_this_request": (
            "NO - ruled out with high confidence for this request"
            if ctx_overflow_ruled_out
            else "NOT PROVEN for this exact request"
        ),
        "O_single_highest_confidence_root_cause": (
            f"CASE {dt.get('case')}: {dt.get('case_label')}"
        ),
        "P_next_single_measurement_if_not_proven": dt.get("next_measurement"),
    }


# --------------------------------------------------------------------------- #
# report rendering
# --------------------------------------------------------------------------- #


def render_report(result: dict) -> str:
    c = result["comparison"]
    sel = result["selected_turn"]
    prod = result["production_observation"]
    dt = result["decision_tree"]
    qc = result["quality_control"]
    ms = sel.get("message_stats", {}) or {}
    fresh = result.get("fresh_replay") or {}
    w = result.get("warm_replay") or {}
    chain = result.get("chain_log_correlation") or {}
    wire_eq = result.get("wire_equivalence") or {}

    def _row(num: str, label: str, key: str) -> str:
        return f"| {num} | {label} | {_fmt(c.get(key))} |"

    lines: list[str] = [
        "# LLM production-turn capture + direct replay - Stage 3 field capture result",
        "",
        f"- generated: {result['generated_at']} by tools/forensic/replay_captured_request.py",
        f"- selected production turn: {sel.get('trace_id')} (start {sel.get('start_wall')}, "
        f"payload: {sel.get('payload_source')})",
        f"- real slow turn captured: {'YES' if sel.get('slow_turn_qualified') else 'NO'} "
        f"(threshold {sel.get('slow_threshold_s')} s, production TTFT {_fmt(c.get('03_production_ttft_s'))})",
    ]
    if not sel.get("slow_turn_qualified"):
        lines += [
            "",
            "> **NOT A REPRODUCTION**: the slowest captured turn is below the 30 s slow",
            "> threshold. Keep using VoiceMem under the instrumented backend",
            "> (CAPTURE_START.bat) until a real 30-50 s turn occurs, then re-run",
            "> REPLAY_CAPTURED_TURN.bat. The numbers below are diagnostic only.",
        ]
    lines += [
        "",
        "## MEASURED FACTS",
        "",
        "### The 22-point comparison",
        "",
        "| # | measurement | value |",
        "|---|---|---|",
        f"| 1 | production request start (wall) | {c.get('01_production_request_start_wall')} |",
        f"| 2 | production first token (wall) | {c.get('02_production_first_token_wall')} |",
        f"| 3 | production TTFT | {_fmt(c.get('03_production_ttft_s'))} |",
        f"| 4 | production total latency | {_fmt(c.get('04_production_total_latency_s'))} |",
        f"| 5 | direct WARM replay TTFT | {_fmt(c.get('05_direct_warm_replay_ttft_s'))} |",
        f"| 6 | direct WARM replay total | {_fmt(c.get('06_direct_warm_replay_total_s'))} |",
        f"| 7 | direct FRESH replay TTFT | {_fmt(c.get('07_direct_fresh_replay_ttft_s'))} |",
        f"| 8 | direct FRESH replay total | {_fmt(c.get('08_direct_fresh_replay_total_s'))} |",
        f"| 9 | exact prompt token count | {c.get('09_exact_prompt_token_count') if c.get('09_exact_prompt_token_count') is not None else 'NOT MEASURED'} |",
        f"| 10 | prompt tokens/s (production / fresh) | {_fmt(c.get('10_prompt_tokens_per_second_production'), 'tok/s')} / {_fmt(c.get('10b_prompt_tokens_per_second_fresh'), 'tok/s')} |",
        f"| 11 | generated token count | {c.get('11_generated_token_count') if c.get('11_generated_token_count') is not None else 'NOT MEASURED'} |",
        f"| 12 | generation tokens/s | {_fmt(c.get('12_generation_tokens_per_second'), 'tok/s')} |",
        f"| 13 | KV cache state | hit={'YES' if (c.get('13_kv_cache_state') or {}).get('warm_replay_cache_hit') else 'NO'}; fresh=cold by construction |",
        f"| 14 | server-side prompt evaluation time | {_fmt(c.get('14_server_side_prompt_evaluation_time_s'))} |",
        f"| 15 | server-side generation time | {_fmt(c.get('15_server_side_generation_time_s'))} |",
        f"| 16 | HTTP transport delay | {_fmt(c.get('16_http_transport_delay_s'))} |",
        f"| 17 | production LLM call -> HTTP request start | {_fmt(c.get('17_production_llm_call_to_http_request_start_s'))} |",
        f"| 18 | HTTP request start -> response headers | {_fmt(c.get('18_http_request_start_to_headers_s'))} |",
        f"| 19 | headers -> first SSE line | {_fmt(c.get('19_headers_to_first_sse_line_s'))} |",
        f"| 20 | first SSE line -> first content token | {_fmt(c.get('20_first_sse_line_to_first_content_token_s'))} |",
        f"| 21 | first content token -> stream end | {_fmt(c.get('21_first_content_token_to_stream_end_s'))} |",
        f"| 22 | stream end -> application LLM completion | {_fmt(c.get('22_stream_end_to_llm_completion_s'))} |",
        "",
        "### Production TTFT phase decomposition (where the 30-50 s went)",
        "",
        "```json",
        json.dumps(result.get("phase_decomposition_of_production_ttft"), indent=2),
        "```",
        "",
        "### Independent production [chain] log correlation",
        "",
        "```json",
        json.dumps(chain, indent=2, ensure_ascii=False),
        "```",
        "",
        "### Wire payload identity (byte-exact replay proof)",
        "",
        "```json",
        json.dumps(wire_eq, indent=2, ensure_ascii=False),
        "```",
        "",
        "### Fresh replay server (same production configuration, new process)",
        "",
    ]
    if fresh:
        lines += [
            f"- command line: `{fresh.get('command_line')}`",
            f"- command source: {fresh.get('source_of_command')}",
            f"- load time: {_fmt(fresh.get('load_s'))}; log: `{fresh.get('log_path')}`",
            f"- started: {fresh.get('started')}; stopped: {fresh.get('stopped')}",
        ]
        if fresh.get("start_error"):
            lines += [
                "- START FAILED:",
                "```json",
                json.dumps(fresh.get("start_error"), indent=2)[:1500],
                "```",
                "",
                f"- OPERATOR GUIDANCE: {fresh.get('operator_guidance')}",
            ]
        if fresh.get("startup_log_lines"):
            lines += ["", "Fresh server effective-config lines (from its own startup log):", "", "```text"]
            lines += [ln[:200] for ln in fresh["startup_log_lines"][:25]]
            lines += ["```"]
    else:
        lines += ["- fresh replay not attempted in this run (`--replay warm`)"]

    lines += [
        "",
        "## INFERENCES",
        "",
    ]
    lines += [f"- {s}" for s in result.get("inferences", [])]
    lines += [
        "",
        "Decision tree classification:",
        "",
        f"- **CASE {dt.get('case')}: {dt.get('case_label')}** (confidence: {dt.get('confidence')})",
        "",
        "Reasoning (measured numbers only):",
        "",
    ]
    lines += [f"  - {s}" for s in dt.get("reasoning", [])]
    lines += [
        "",
        "ngl26 evidence:",
        "",
        "```json",
        json.dumps(result.get("ngl26_evidence"), indent=2, ensure_ascii=False),
        "```",
        "",
        "Context-size evidence:",
        "",
        "```json",
        json.dumps(result.get("context_evidence"), indent=2, ensure_ascii=False),
        "```",
        "",
        "## UNRESOLVED QUESTIONS",
        "",
    ]
    lines += [f"- {s}" for s in result.get("unresolved_questions", [])]
    lines += [
        "",
        "## VERDICT (A-P)",
        "",
    ]
    for k, v in (result.get("verdict_16") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Quality control",
        "",
        "```json",
        json.dumps(qc, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Evidence: production observation (from the trace)",
        "",
        "```json",
        json.dumps(
            {k: v for k, v in prod.items() if k != "llama_server_log_lines"},
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        "Production-turn llama-server log lines:",
        "",
        "```text",
        *[ln[:200] for ln in (prod.get("llama_server_log_lines") or [])[:40]],
        "```",
        "",
        "## Evidence: WARM direct replay (independent client, exact bytes)",
        "",
        "```json",
        json.dumps(
            {k: v for k, v in w.items() if k not in ("metrics_before", "metrics_after", "llama_server_log_lines")},
            indent=2,
            ensure_ascii=False,
        ),
        "```",
    ]
    if fresh.get("replay"):
        lines += [
            "",
            "## Evidence: FRESH direct replay (new process, same production config)",
            "",
            "```json",
            json.dumps(
                {
                    k: v
                    for k, v in fresh["replay"].items()
                    if k not in ("metrics_before", "metrics_after", "llama_server_log_lines")
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "Fresh replay llama-server log lines (the request window):",
            "",
            "```text",
            *[ln[:200] for ln in (fresh["replay"].get("llama_server_log_lines") or [])[:40]],
            "```",
        ]
    if result.get("merge"):
        lines += [
            "",
            "## Merge note",
            "",
            "```json",
            json.dumps(result["merge"], indent=2, ensure_ascii=False),
            "```",
        ]
    lines += [
        "",
        "---",
        "",
        "The exact captured request (full messages array, wire bytes) is preserved in "
        "`logs/llm_production_trace.jsonl` (start + wire events of this trace_id). "
        "The production llama-server was NEVER restarted or reconfigured by the warm "
        "replay; the fresh replay used a separate process that has been stopped. "
        "No production source file was modified by any stage-3 tool.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


async def _run(args) -> int:
    trace_path = Path(args.trace) if args.trace else TRACE_PATH
    session, turns = load_turns(trace_path)
    if not turns:
        print(f"No turns found in {trace_path}. Run the instrumented backend first.")
        return 1
    turn = select_turn(turns, args.mode, args.trace_id)
    print(f"Selected production turn: {turn['trace_id']}")
    start_ev = turn.get("start", {})
    wire_ev = turn.get("wire", {})
    body = wire_ev.get("exact_wire_body") or start_ev.get("payload")
    raw_text = wire_ev.get("exact_wire_body_raw")
    if not body:
        print("The selected turn has no captured payload (no start/wire event).")
        return 1
    prod_ttft = _pick((turn.get("end", {}).get("markers") or {}), "first_content_token_rel")
    print(
        f"  production TTFT: {_fmt(prod_ttft)} | slow threshold: {args.slow_threshold} s "
        f"({'SLOW TURN' if prod_ttft and prod_ttft >= args.slow_threshold else 'below threshold - diagnostic only'})"
    )

    chain_corr = _chain_correlation(turn)
    wire_eq = _wire_equivalence(turn)
    print(f"[chain] log correlation: {chain_corr.get('matched')}")
    print(
        f"[wire] raw available: {wire_eq.get('wire_raw_available')} | "
        f"reserialized == wire: {wire_eq.get('reserialized_equals_wire')} | "
        f"wire == reconstruction: {wire_eq.get('wire_equals_reconstruction')}"
    )

    merge_info: Optional[dict] = None
    warm_replay: Optional[dict] = None
    fresh_block: Optional[dict] = None

    # ---- WARM replay: the RUNNING production server ------------------------
    if args.replay in ("both", "warm"):
        base_url = args.url or start_ev.get("server_url") or "http://127.0.0.1:8080/v1"
        print(f"\n=== WARM replay: the RUNNING production server ({base_url}) ===")
        print(f"  messages: {len(body.get('messages', []))}, params: "
              f"{ {k: v for k, v in body.items() if k != 'messages'} }")
        server_info = sync_server_info(_server_root(base_url))
        print(f"  server health: {server_info.get('health')}")
        if server_info.get("health", {}).get("error"):
            print(
                "  WARNING: the production llama-server is not reachable; if you "
                "stopped it for VRAM reasons, run with --replay fresh-only "
                "(the previous warm result will be merged)."
            )
        warm_replay = await replay_exact_request(base_url, body, raw_text=raw_text)
        w_ttft = _pick(warm_replay.get("markers"), "first_content_token_rel")
        print(f"  warm replay TTFT: {_fmt(w_ttft)} (status {warm_replay.get('http_status')})")
        warm_replay["server_info_before_replay"] = server_info

    # ---- FRESH replay: new process, same production configuration ----------
    if args.replay in ("both", "fresh-only"):
        if args.replay == "fresh-only":
            # merge a previous warm result if it belongs to the same turn
            try:
                prev = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
                if (
                    prev.get("schema") == "voicemem-llm-prod-capture/2"
                    and prev.get("selected_turn", {}).get("trace_id") == turn["trace_id"]
                    and prev.get("warm_replay")
                ):
                    warm_replay = prev["warm_replay"]
                    merge_info = {
                        "merged_previous_result": str(RESULT_PATH),
                        "kept": "warm_replay (previous run)",
                        "same_trace_id": turn["trace_id"],
                    }
                    print("\nMerged the previous WARM replay result for this turn.")
            except Exception:
                pass

        print(f"\n=== FRESH replay: new llama-server, same production config, port {args.fresh_port} ===")
        production_cmd = _production_server_cmdline(session)
        if production_cmd:
            print(f"  production command line: {production_cmd[:160]}...")
        else:
            print("  production command line not captured in the trace - using config fallback")
        exe, fargv, notes = _build_fresh_argv(
            production_cmd,
            args.fresh_port,
            session,
            args.server_exe,
            args.model,
        )
        fresh_block = {
            "attempted": True,
            "port": args.fresh_port,
            "log_path": str(FRESH_LOG_PATH),
            "notes": notes,
        }
        if exe is None:
            fresh_block.update(
                {
                    "started": False,
                    "start_error": {"message": "; ".join(notes)},
                    "source_of_command": "unavailable",
                    "replay": None,
                    "stopped": False,
                    "operator_guidance": (
                        "Start the fresh replay with explicit paths: "
                        "--server-exe <llama-server path> --model <GGUF path>"
                    ),
                }
            )
            print("  FRESH replay unavailable: " + "; ".join(notes))
        else:
            fresh_block["command_line"] = f'"{exe}" ' + " ".join(fargv)
            fresh_block["source_of_command"] = notes[0] if notes else "production process cmdline"
            server = FreshServer(exe, fargv, args.fresh_port, FRESH_LOG_PATH)
            fresh_block["exe"] = exe
            fresh_block["args"] = fargv
            print(f"  starting: {fresh_block['command_line'][:160]}...")
            t0 = time.perf_counter()
            if server.start() and server.wait_health(args.fresh_timeout):
                fresh_block["started"] = True
                fresh_block["load_s"] = server.load_s
                print(f"  fresh server ready in {_fmt(server.load_s)} (pid {server.proc.pid})")
                fresh_block["startup_log_lines"] = server.startup_lines()
                fresh_block["identity"] = server.identity()
                try:
                    offset = FRESH_LOG_PATH.stat().st_size
                except OSError:
                    offset = 0
                frep = await replay_exact_request(
                    f"http://127.0.0.1:{args.fresh_port}/v1",
                    body,
                    raw_text=raw_text,
                    log_files={str(FRESH_LOG_PATH): offset},
                )
                fresh_block["replay"] = frep
                f_ttft = _pick(frep.get("markers"), "first_content_token_rel")
                print(f"  fresh replay TTFT: {_fmt(f_ttft)} (cold KV cache by construction)")
            else:
                fresh_block["started"] = False
                err = server.start_error or {"message": "health timeout"}
                fresh_block["start_error"] = err
                tail = ""
                try:
                    tail = FRESH_LOG_PATH.read_text(encoding="utf-8", errors="replace")[-600:]
                except OSError:
                    pass
                if tail:
                    fresh_block["start_error"]["log_tail"] = tail
                fresh_block["operator_guidance"] = (
                    "The fresh llama-server could not start (common cause on a 12 GB GPU: "
                    "the 19 GB model + ngl 26 cannot fit TWICE while the production server "
                    "is still running). TO COMPLETE THE FRESH REPLAY: close the production "
                    "llama-server console (Ctrl+C; the web backend can keep running), then "
                    "re-run this tool with --replay fresh-only - it MERGES the warm result "
                    "already captured. Afterwards restart the production server with "
                    "START.bat."
                )
                print("  fresh server FAILED to start - see operator guidance in the result.")
            fresh_block["stopped"] = server.stop()
            print(f"  fresh server stopped: {fresh_block['stopped']}")

    if warm_replay is None and fresh_block is None:
        print("Nothing to do: neither warm nor fresh replay was selected.")
        return 1

    result = build_result(
        session, turn, warm_replay, fresh_block, args, chain_corr, wire_eq, merge_info
    )
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = render_report(result)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print()
    print("=" * 74)
    print("VERDICT (A-P):")
    for k, v in (result.get("verdict_16") or {}).items():
        print(f"  {k}: {v}")
    print("=" * 74)
    print()
    print(f"Result JSON : {RESULT_PATH}")
    print(f"Result MD   : {REPORT_PATH}")
    return 0


def _parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Replay the exact captured VoiceMem production LLM request: "
            "warm (running server) + fresh (new process, same production config)"
        )
    )
    p.add_argument("--trace", default=str(TRACE_PATH), help="trace JSONL path")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--trace-id", default=None, help="specific trace_id")
    g.add_argument("--mode", choices=["slowest", "last"], default="slowest")
    p.add_argument(
        "--url",
        default=None,
        help="llama-server base URL for the WARM replay (default: from the trace / 127.0.0.1:8080/v1)",
    )
    p.add_argument(
        "--replay",
        choices=["both", "warm", "fresh-only"],
        default="both",
        help="both = warm + fresh (default); fresh-only merges a previous warm result",
    )
    p.add_argument("--fresh-port", type=int, default=DEFAULT_FRESH_PORT)
    p.add_argument(
        "--fresh-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for the fresh server to load the model (default 900)",
    )
    p.add_argument("--server-exe", default=None, help="llama-server executable override for the fresh start")
    p.add_argument("--model", default=None, help="model path override for the fresh start")
    p.add_argument(
        "--slow-threshold",
        type=float,
        default=DEFAULT_SLOW_THRESHOLD_S,
        help="TTFT threshold that qualifies a turn as the real production problem (default 30 s)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
