"""TEMPORARY forensic instrumentation: capture the EXACT production LLM request.

Stage-3 forensic tool (real-turn capture). This module is NOT production code
and is NOT wired into the installer/tests/release. It is loaded ONLY by
``tools/forensic/run_web_with_llm_trace.py`` (the forensic launcher) and
attaches itself at RUNTIME:

* wraps ``app.llm.LlmClient.chat_stream`` (the exact production LLM call),
* patches ``httpx.AsyncClient.send`` + ``httpx.Response.aiter_lines`` at class
  level to timestamp the transport boundary (request handed to httpx, response
  headers received, first SSE line on the wire),
* snapshots the llama-server ``/metrics`` counters before/after every LLM
  request (server-side prompt/eval seconds + exact token counts),
* tails the llama-server log files across the request window (the server's own
  ``slot print_timing`` lines for the exact production turn),
* writes one append-only JSONL event stream to
  ``logs/llm_production_trace.jsonl``.

NO production file is modified; the only writes are forensic artifacts under
``logs/``. EVERY instrumentation step is fail-open: if anything here raises,
the wrapper delegates to the original implementation unchanged, so the
application's semantics (payload, streaming, error paths) are byte-identical.

Event record types (one JSON object per line):
  session   once at install time: process + server identity + effective config
  start     LLM call entered: exact reconstructed payload + config params
  wire      request handed to httpx: the EXACT JSON body sent on the wire
            (exact_wire_body parsed + exact_wire_body_raw = the literal
            wire text, so the replay can resend byte-identical bytes)
  headers   response headers received (HTTP status)
  end       stream finished/failed: all timing markers + metrics deltas
            + the llama-server log lines produced during the turn
            (llm_request_end_rel = app-side completion BEFORE forensic
            bookkeeping; forensic_end_rel = after it)

All *_rel timings are seconds relative to llm_request_start of that turn.
"""

from __future__ import annotations

import contextvars
import copy
import json
import os
import sys
import threading
import time
import traceback
import uuid
from asyncio import CancelledError as _CancelledError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = REPO_ROOT / "logs" / "llm_production_trace.jsonl"

# llama-server side log files watched across each request window (best-effort).
_SERVER_LOG_FILES = [
    "llama-server.out.log",
    "llama-server.err.log",
    "llama-server.log",
    "server.log",
]

_TURN_SEQ = 0
_TURN_SEQ_LOCK = threading.Lock()
_FILE_LOCK = threading.Lock()
_FILE_H = None

# ContextVar: active trace record for THIS task's LLM request (async
# generators resume in the resumer's context, so the httpx patches called
# inside the original chat_stream body see the value set by the wrapper).
_ACTIVE_TRACE: contextvars.ContextVar = contextvars.ContextVar(
    "voicemem_llm_trace", default=None
)

_INSTALLED = {"chat_stream": False, "send": False, "aiter_lines": False}
_ORIGINALS: dict[str, Any] = {}
_SESSION_SERVER_INFO: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# JSONL event writer (append-only, flush-per-event, fail-open)
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _write_event(record: dict) -> None:
    """Append one event to the trace JSONL. NEVER raises."""
    try:
        global _FILE_H
        with _FILE_LOCK:
            if _FILE_H is None:
                TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
                _FILE_H = open(TRACE_PATH, "a", encoding="utf-8")
            _FILE_H.write(json.dumps(record, ensure_ascii=False) + "\n")
            _FILE_H.flush()
    except Exception:
        try:
            traceback.print_exc()
        except Exception:
            pass


def _next_turn_id() -> str:
    global _TURN_SEQ
    with _TURN_SEQ_LOCK:
        _TURN_SEQ += 1
        seq = _TURN_SEQ
    return f"turn-{seq:04d}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# llama-server /metrics snapshots (server-side evidence)
# --------------------------------------------------------------------------- #


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse a Prometheus exposition body; keep llamacpp* counters."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, value = parts[0], parts[1]
        if not name.startswith("llamacpp"):
            continue
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


async def _metrics_snapshot(url: str) -> Optional[dict[str, float]]:
    """GET /metrics with a short timeout. Returns None on any failure.

    A FRESH client per call: the snapshot must work on whatever event loop
    the caller runs on (the web backend has one loop, but a forensic driver
    may use several; a cached client bound to a closed loop returns None).
    """
    try:
        import httpx

        async with httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(2.0)) as c:
            resp = await c.get("/metrics")
            if resp.status_code == 200:
                return _parse_prometheus(resp.text)
            return None
    except Exception:
        return None


def _server_root_from_config(config: Any) -> str:
    """http://host:port from the production config (strip the /v1 base)."""
    try:
        base = config.llama_server_url  # http://127.0.0.1:8080/v1
        root = base.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return root
    except Exception:
        return "http://127.0.0.1:8080"


# --------------------------------------------------------------------------- #
# llama-server log tailing (the server's own lines for the exact turn)
# --------------------------------------------------------------------------- #


def _log_offsets() -> dict[str, int]:
    offsets: dict[str, int] = {}
    for name in _SERVER_LOG_FILES:
        p = REPO_ROOT / "logs" / name
        try:
            offsets[str(p)] = p.stat().st_size if p.exists() else -1
        except OSError:
            offsets[str(p)] = -1
    return offsets


# Priority: the server's own per-request timing/config evidence lines.
_LOG_PRIORITY_KEYS = (
    "print_timing",
    "prompt eval time",
    "eval time",
    "total time",
    "new prompt",
    "n_ctx",
    "cache_reuse",
    "n_past",
    "truncated",
    "need to evaluate",
)
# Secondary: fit/offload/config warnings (rare, high value).
_LOG_SECONDARY_KEYS = (
    "failed to fit",
    "offload",
    "n_gpu_layers",
    "flash_attn",
    "n_batch",
    "n_ubatch",
    "error",
    "warn",
    "listening",
    "model loaded",
)
# Verbose-mode debug spam that would otherwise match the keys above.
_LOG_DROP_SUBSTRINGS = (
    "slot decode token",
    "n_batch (effective)",
    "launching slot",
    "update slots",
    "all slots are idle",
    "waiting for new tasks",
    "processing new tasks",
    "processing task",
    "add task",
    "remove task",
    "post: new task",
    "add_waiting",
    "remove_waiti",
)


def _log_tail_lines(
    offsets: dict[str, int], max_priority: int = 60, max_secondary: int = 25
) -> list[str]:
    """Read the bytes appended since ``offsets``; keep the server's own
    per-request evidence lines, priority-ranked (print_timing / new prompt /
    n_ctx / cache / truncation first; fit-offload warnings second). Verbose
    queue/slot decode spam is dropped."""
    priority: list[str] = []
    secondary: list[str] = []
    for path_str, off in offsets.items():
        if off < 0:
            continue
        try:
            p = Path(path_str)
            if not p.exists():
                continue
            size = p.stat().st_size
            if size <= off:
                continue
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(off)
                new_text = fh.read(size - off)
            for line in new_text.splitlines():
                s = line.strip()
                if not s or len(s) > 400:
                    continue
                if any(d in s for d in _LOG_DROP_SUBSTRINGS):
                    continue
                low = s.lower()
                if any(key in low for key in _LOG_PRIORITY_KEYS):
                    priority.append(f"[{p.name}] {s}")
                elif any(key in low for key in _LOG_SECONDARY_KEYS):
                    secondary.append(f"[{p.name}] {s}")
        except Exception:
            continue
    return priority[:max_priority] + secondary[:max_secondary]


# --------------------------------------------------------------------------- #
# httpx transport patches (class-level; guarded by the active-trace ContextVar)
# --------------------------------------------------------------------------- #


def _install_httpx_patches() -> None:
    if _INSTALLED["send"] and _INSTALLED["aiter_lines"]:
        return
    try:
        import httpx

        if not _INSTALLED["send"]:
            orig_send = httpx.AsyncClient.send
            _ORIGINALS["send"] = orig_send

            async def _traced_send(self, request, *args, **kwargs):
                trace = None
                try:
                    trace = _ACTIVE_TRACE.get()
                except Exception:
                    trace = None
                if trace is None or not str(request.url.path).endswith(
                    "chat/completions"
                ):
                    return await orig_send(self, request, *args, **kwargs)
                t_req = time.perf_counter()
                markers = trace.setdefault("markers", {})
                markers["http_request_start_rel"] = round(
                    t_req - trace["t_request_start"], 4
                )
                try:
                    body_text = request.content.decode("utf-8", "replace")
                    trace["wire_body_raw"] = body_text
                    trace["wire_body"] = json.loads(body_text)
                except Exception:
                    trace["wire_body"] = None
                    trace["wire_body_raw"] = None
                _write_event(
                    {
                        "schema": "voicemem-llm-prod-trace/1",
                        "event": "wire",
                        "trace_id": trace["trace_id"],
                        "ts_wall": _utc_now(),
                        "http_request_start_rel": markers[
                            "http_request_start_rel"
                        ],
                        "exact_wire_body": trace["wire_body"],
                        "exact_wire_body_raw": trace["wire_body_raw"],
                        "request_url": str(request.url),
                    }
                )
                t0 = time.perf_counter()
                response = await orig_send(self, request, *args, **kwargs)
                elapsed = time.perf_counter() - t0
                markers["http_headers_received_rel"] = round(
                    time.perf_counter() - trace["t_request_start"], 4
                )
                _write_event(
                    {
                        "schema": "voicemem-llm-prod-trace/1",
                        "event": "headers",
                        "trace_id": trace["trace_id"],
                        "ts_wall": _utc_now(),
                        "http_send_elapsed_s": round(elapsed, 4),
                        "http_headers_received_rel": markers[
                            "http_headers_received_rel"
                        ],
                        "http_status": getattr(response, "status_code", None),
                    }
                )
                return response

            httpx.AsyncClient.send = _traced_send
            _INSTALLED["send"] = True

        if not _INSTALLED["aiter_lines"]:
            orig_aiter_lines = httpx.Response.aiter_lines
            _ORIGINALS["aiter_lines"] = orig_aiter_lines

            async def _traced_aiter_lines(self, *args, **kwargs):
                trace = None
                try:
                    trace = _ACTIVE_TRACE.get()
                except Exception:
                    trace = None
                if trace is None:
                    async for line in orig_aiter_lines(self, *args, **kwargs):
                        yield line
                    return
                first = True
                markers = trace.setdefault("markers", {})
                async for line in orig_aiter_lines(self, *args, **kwargs):
                    if first:
                        first = False
                        markers["first_sse_line_rel"] = round(
                            time.perf_counter() - trace["t_request_start"], 4
                        )
                    trace["sse_lines"] = trace.get("sse_lines", 0) + 1
                    yield line

            httpx.Response.aiter_lines = _traced_aiter_lines
            _INSTALLED["aiter_lines"] = True
    except Exception:
        try:
            traceback.print_exc()
        except Exception:
            pass


def uninstall_httpx_patches() -> None:
    """Restore httpx class attributes (used by tests / cleanup)."""
    try:
        import httpx

        if _INSTALLED["send"] and "send" in _ORIGINALS:
            httpx.AsyncClient.send = _ORIGINALS["send"]
            _INSTALLED["send"] = False
        if _INSTALLED["aiter_lines"] and "aiter_lines" in _ORIGINALS:
            httpx.Response.aiter_lines = _ORIGINALS["aiter_lines"]
            _INSTALLED["aiter_lines"] = False
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Session record (process + server identity), gathered in a background thread
# --------------------------------------------------------------------------- #


def _gather_server_info_sync(config: Any) -> dict[str, Any]:
    """Best-effort synchronous server identity capture (own thread)."""
    info: dict[str, Any] = {"root_url": _server_root_from_config(config)}
    try:
        import httpx

        with httpx.Client(base_url=info["root_url"], timeout=httpx.Timeout(2.0)) as c:
            try:
                r = c.get("/health")
                info["health"] = {"status": r.status_code, "body": r.text[:200]}
            except Exception as exc:
                info["health"] = {"error": repr(exc)[:200]}
            try:
                r = c.get("/props")
                info["props"] = r.json() if r.status_code == 200 else None
            except Exception as exc:
                info["props"] = {"error": repr(exc)[:200]}
            try:
                r = c.get("/metrics")
                info["metrics_cumulative_at_session"] = (
                    _parse_prometheus(r.text) if r.status_code == 200 else None
                )
            except Exception as exc:
                info["metrics_cumulative_at_session"] = {"error": repr(exc)[:200]}
    except Exception as exc:
        info["http_error"] = repr(exc)[:200]
    info["llama_server_log"] = _effective_config_from_logs()
    info["llama_server_process"] = _llama_server_process_cmdline()
    return info


def _effective_config_from_logs(max_lines: int = 60) -> dict[str, Any]:
    """Extract effective-config evidence lines from the llama-server logs."""
    found: dict[str, Any] = {}
    for name in _SERVER_LOG_FILES:
        p = REPO_ROOT / "logs" / name
        try:
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        keys = (
            "n_ctx",
            "n_ctx_slot",
            "offloaded",
            "offload",
            "flash_attn",
            "n_batch",
            "n_ubatch",
            "n_threads",
            "failed to fit",
            "mmap",
            "load",
            "listening",
            "model",
            "n_gpu_layers",
            "KV",
            "kv",
        )
        lines = [
            ln.strip()
            for ln in text.splitlines()
            if any(k in ln for k in keys) and len(ln.strip()) < 300
        ]
        if lines:
            found[name] = {
                "startup_lines": lines[:max_lines],
                "tail_lines": lines[-10:],
            }
    return found


def _llama_server_process_cmdline() -> Any:
    """Windows: llama-server.exe command line via CIM; POSIX: ps."""
    try:
        if os.name == "nt":
            import subprocess

            ps = (
                "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" "
                "| Select-Object ProcessId,CreationDate,CommandLine "
                "| ConvertTo-Json -Compress"
            )
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip():
                return json.loads(out.stdout)
            return {"error": (out.stderr or "")[:300]}
        import subprocess

        out = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10
        )
        lines = [
            ln.strip()
            for ln in out.stdout.splitlines()
            if "llama-server" in ln and "ps -eo" not in ln
        ]
        return {"processes": lines[:5]}
    except Exception as exc:
        return {"error": repr(exc)[:300]}


def _write_session_record(config: Any) -> None:
    """Session event: process identity + server identity (background thread)."""

    def worker() -> None:
        try:
            record = {
                "schema": "voicemem-llm-prod-trace/1",
                "event": "session",
                "ts_wall": _utc_now(),
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "process_cmdline": " ".join(sys.argv)[:2000],
                "trace_path": str(TRACE_PATH),
                "config": {
                    "llama_server_url": getattr(config, "llama_server_url", None),
                    "llama_server_health_url": getattr(
                        config, "llama_server_health_url", None
                    ),
                    "llm_model_name": getattr(config, "llm_model_name", None),
                    "llm_temperature": getattr(config, "llm_temperature", None),
                    "llm_max_tokens": getattr(config, "llm_max_tokens", None),
                    "llm_disable_thinking": getattr(
                        config, "llm_disable_thinking", None
                    ),
                    "llm_n_gpu_layers": getattr(config, "llm_n_gpu_layers", None),
                    "llm_context_size": getattr(config, "llm_context_size", None),
                    "llm_parallel": getattr(config, "llm_parallel", None),
                    "llm_cache_type_k": getattr(config, "llm_cache_type_k", None),
                    "llm_cache_type_v": getattr(config, "llm_cache_type_v", None),
                },
                "httpx_version": _httpx_version(),
                "server": _gather_server_info_sync(config),
            }
            global _SESSION_SERVER_INFO
            _SESSION_SERVER_INFO = record
            _write_event(record)
        except Exception:
            try:
                traceback.print_exc()
            except Exception:
                pass

    threading.Thread(target=worker, name="llm-trace-session", daemon=True).start()


def _httpx_version() -> Optional[str]:
    try:
        import httpx

        return httpx.__version__
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# chat_stream wrapper: THE production boundary capture
# --------------------------------------------------------------------------- #


def _install_chat_stream_wrapper() -> None:
    if _INSTALLED["chat_stream"]:
        return
    try:
        from app import llm as llm_mod

        orig_chat_stream = llm_mod.LlmClient.chat_stream
        _ORIGINALS["chat_stream"] = orig_chat_stream
        thinking_kwargs = llm_mod._thinking_control_kwargs

        async def _traced_chat_stream(
            self,
            messages: list[dict],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
        ):
            """Fail-open wrapper: delegates to the ORIGINAL chat_stream.

            The only observable difference is forensic bookkeeping (a few
            localhost /metrics GETs and file appends); payload, streaming
            order, and every error path are the original's.
            """
            trace_id = _next_turn_id()
            t_start = time.perf_counter()
            trace: dict[str, Any] = {
                "trace_id": trace_id,
                "t_request_start": t_start,
                "markers": {},
                "sse_lines": 0,
            }
            client_warm = self._client is not None
            # --- exact payload reconstruction (identical to production) -----
            payload: Optional[dict[str, Any]] = None
            try:
                payload = {
                    "model": self._config.llm_model_name,
                    "messages": copy.deepcopy(messages),
                    "stream": True,
                    "temperature": (
                        self._config.llm_temperature
                        if temperature is None
                        else temperature
                    ),
                    "max_tokens": (
                        self._config.llm_max_tokens
                        if max_tokens is None
                        else max_tokens
                    ),
                    **thinking_kwargs(self._config),
                }
            except Exception:
                payload = None
            msg_stats: dict[str, Any] = {}
            try:
                roles = [m.get("role") for m in (messages or [])]
                msg_stats = {
                    "message_count": len(messages or []),
                    "roles": roles,
                    "chars_per_message": [
                        len(str(m.get("content", ""))) for m in (messages or [])
                    ],
                    "total_chars": sum(
                        len(str(m.get("content", ""))) for m in (messages or [])
                    ),
                    "system_chars": len(str((messages or [{}])[0].get("content", ""))),
                    "last_user_chars": (
                        len(str((messages or [{}])[-1].get("content", "")))
                        if messages
                        else 0
                    ),
                    "history_entries": max(0, len(messages or []) - 2),
                }
            except Exception:
                msg_stats = {}
            try:
                _write_event(
                    {
                        "schema": "voicemem-llm-prod-trace/1",
                        "event": "start",
                        "trace_id": trace_id,
                        "ts_wall": _utc_now(),
                        "llm_request_start_rel": 0.0,
                        "payload": payload,
                        "message_stats": msg_stats,
                        "http_client_existed_before": client_warm,
                        "server_url": getattr(
                            self._config, "llama_server_url", None
                        ),
                    }
                )
            except Exception:
                pass

            # --- pre-request server state (metrics + log offsets) -----------
            metrics_before = None
            log_offsets: dict[str, int] = {}
            try:
                metrics_before = await _metrics_snapshot(
                    _server_root_from_config(self._config)
                )
            except Exception:
                metrics_before = None
            try:
                log_offsets = _log_offsets()
            except Exception:
                log_offsets = {}
            trace["log_offsets"] = log_offsets

            token = None
            first_content_recorded = False
            content_chars = 0
            delta_count = 0
            error_info: Optional[dict] = None
            markers = trace["markers"]
            try:
                token = _ACTIVE_TRACE.set(trace)
                underlying = orig_chat_stream(
                    self, messages, temperature=temperature, max_tokens=max_tokens
                )
                # The production body (payload build -> httpx send -> SSE loop)
                # starts running at the first __anext__ below; the send patch
                # timestamps the actual HTTP handoff (http_request_start).
                markers["body_start_rel"] = round(time.perf_counter() - t_start, 4)
                while True:
                    try:
                        delta = await underlying.__anext__()
                    except StopAsyncIteration:
                        break
                    if not first_content_recorded:
                        first_content_recorded = True
                        markers["first_content_token_rel"] = round(
                            time.perf_counter() - t_start, 4
                        )
                    delta_count += 1
                    content_chars += len(delta)
                    yield delta
                markers["stream_end_rel"] = round(time.perf_counter() - t_start, 4)
            except _CancelledError:
                error_info = {"type": "CancelledError", "stage": "stream"}
                markers["cancelled_rel"] = round(time.perf_counter() - t_start, 4)
                raise
            except BaseException as exc:
                error_info = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "stage": "stream",
                }
                markers["error_rel"] = round(time.perf_counter() - t_start, 4)
                raise
            finally:
                # App-side completion boundary: stamped BEFORE any forensic
                # bookkeeping below (the /metrics snapshot + log tail are
                # instrumentation-only overhead, reported separately as
                # forensic_end_rel so llm_request_end_rel stays the true
                # application-side completion timestamp).
                markers["llm_request_end_rel"] = round(
                    time.perf_counter() - t_start, 4
                )
                if token is not None:
                    try:
                        _ACTIVE_TRACE.reset(token)
                    except Exception:
                        pass
                # --- post-request server state (forensic only) -----------------
                metrics_after = None
                try:
                    metrics_after = await _metrics_snapshot(
                        _server_root_from_config(self._config)
                    )
                except Exception:
                    metrics_after = None
                metrics_delta: dict[str, float] = {}
                try:
                    if metrics_before and metrics_after:
                        for k, v_after in metrics_after.items():
                            v_before = metrics_before.get(k, 0.0)
                            if v_after != v_before:
                                metrics_delta[k] = round(v_after - v_before, 4)
                except Exception:
                    metrics_delta = {}
                server_lines: list[str] = []
                try:
                    server_lines = _log_tail_lines(trace.get("log_offsets", {}))
                except Exception:
                    server_lines = []
                markers["forensic_end_rel"] = round(
                    time.perf_counter() - t_start, 4
                )
                try:
                    _write_event(
                        {
                            "schema": "voicemem-llm-prod-trace/1",
                            "event": "end",
                            "trace_id": trace_id,
                            "ts_wall": _utc_now(),
                            "markers": markers,
                            "sse_lines": trace.get("sse_lines", 0),
                            "content_delta_count": delta_count,
                            "content_chars": content_chars,
                            "error": error_info,
                            "metrics_before": metrics_before,
                            "metrics_after": metrics_after,
                            "metrics_delta": metrics_delta,
                            "llama_server_log_lines": server_lines,
                        }
                    )
                except Exception:
                    pass

        llm_mod.LlmClient.chat_stream = _traced_chat_stream
        _INSTALLED["chat_stream"] = True
    except Exception:
        try:
            traceback.print_exc()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def install(config: Any = None) -> bool:
    """Install all patches. Returns True when the chat_stream wrapper is on."""
    _install_httpx_patches()
    _install_chat_stream_wrapper()
    if config is not None:
        try:
            _write_session_record(config)
        except Exception:
            pass
    return bool(_INSTALLED["chat_stream"])


def uninstall() -> None:
    """Remove all patches (best-effort; for validation/tests)."""
    try:
        if _INSTALLED["chat_stream"] and "chat_stream" in _ORIGINALS:
            from app import llm as llm_mod

            llm_mod.LlmClient.chat_stream = _ORIGINALS["chat_stream"]
            _INSTALLED["chat_stream"] = False
    except Exception:
        pass
    uninstall_httpx_patches()


def status() -> dict[str, Any]:
    return {
        "installed": dict(_INSTALLED),
        "trace_path": str(TRACE_PATH),
        "turn_count": _TURN_SEQ,
    }
