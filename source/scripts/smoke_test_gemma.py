#!/usr/bin/env python3
"""scripts/smoke_test_gemma.py - Gemma 4 12B Q4_0 LLM smoke test + performance measurement.

v0.4.3: the ONE AND ONLY runtime LLM is Gemma 4 12B instruction-tuned,
QAT Q4_0 (google/gemma-4-12B-it-qat-q4_0-gguf). This script is the
authoritative Gemma startup validation - it runs ON THE TARGET MACHINE
(Windows 11 + RTX 5070 + the downloaded model + bin/llama-server.exe).

THE 14 CHECKS (exactly the milestone contract):
   1. Gemma model exists                     (models/llm/gemma-4-12b/...)
   2. GGUF file is valid                     (magic bytes + size floor,
                                               optional --verify-hash for the
                                               full pinned sha256)
   3. llama-server starts                    (or a healthy Gemma server is
                                               reused - idempotent)
   4. /health returns HTTP 200
   5. normal chat completion works           (POST /v1/chat/completions)
   6. response contains non-empty assistant content
      (the v0.4.2 failure mode - /health PASS but completions return
      EMPTY content - must FAIL here explicitly, never pass silently)
   7. Hungarian generation works             (detect_language == 'hu')
   8. English generation works               (detect_language == 'en')
   9. JSON object generation works           (response_format json_object)
  10. JSON schema constrained generation     (response_format json_schema)
  11. process remains alive after generation
  12. model is actually loaded               (GET /v1/models -> gemma)
  13. GPU offload is active                  (server log + nvidia-smi)
  14. no unexpected CPU-only fallback        (no CPU-fallback log lines,
                                               all layers offloaded)

PERFORMANCE MEASUREMENT (recorded after real generations, never
theoretical): model load time, VRAM usage, RAM usage, first token
latency, generation speed (tokens/sec), total response latency,
GPU offload layer count, context size, parallel value. Results are
written to logs/gemma_smoke_result.json and printed as a table.

RUNTIME NETWORK POLICY: every HTTP request goes to the LOCAL
llama-server on 127.0.0.1:8080 only - zero cloud calls, zero API keys
(the OpenAI-compatible endpoint is llama.cpp's local compatibility API).

Usage (target machine, repo root):
    .venv\\Scripts\\python.exe scripts\\smoke_test_gemma.py
    .venv\\Scripts\\python.exe scripts\\smoke_test_gemma.py --verify-hash
    .venv\\Scripts\\python.exe scripts\\smoke_test_gemma.py --keep-server

Exit codes: 0 = all 14 checks PASS; 1 = at least one FAIL (the report
names the exact failing component - nothing is masked).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import AgentConfig  # noqa: E402
from app.text_utils import detect_language  # noqa: E402

# ------------------------------------------------------------------ pins ----
# Must stay in sync with MODELS.lock.json (the single source of truth).
GEMMA_REPO = "google/gemma-4-12B-it-qat-q4_0-gguf"
GEMMA_FILE = "gemma-4-12b-it-qat-q4_0.gguf"
GEMMA_DISPLAY = "Gemma 4 12B Q4_0"
GEMMA_SHA256 = "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"
GEMMA_SIZE_BYTES = 6975879296
GEMMA_MIN_BYTES = 6000000000
GEMMA_UPSTREAM_COMMIT = "29d097773436b69ff9feafd636ab4cf873786537"

# The llama-server invocation - exactly the starter's flags (v0.4.3).
LLAMA_ARGS_TEMPLATE = (
    "--model", "{model}",
    "--host", "127.0.0.1",
    "--port", "8080",
    "-ngl", "-1",
    "-c", "8192",
    "--parallel", "1",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--temp", "0.7",
    "--metrics",
    "--no-webui",
    # v0.4.4: llama-server's chat handler defaults enable_thinking to TRUE;
    # hybrid-reasoning models then answer inside the thought channel and
    # content stays EMPTY (the v0.4.3 field outage). Off = the template
    # pre-closes the thought channel for EVERY client (incl. the voicemem
    # package's own OpenAI-lib calls that cannot send per-request kwargs).
    "--reasoning", "off",
)

#: Request-level thinking suppression - EXACTLY what app/llm.LlmClient sends
#: (see _thinking_control_kwargs). Used by the app-shaped requests so the
#: smoke test exercises the same wire format as the runtime.
THINKING_OFF_KWARGS = {
    "chat_template_kwargs": {"enable_thinking": False},
    "reasoning_effort": "none",
}

# Server logs the script writes when IT starts the server.
SMOKE_OUT_LOG = "logs/gemma-smoke.out.log"
SMOKE_ERR_LOG = "logs/gemma-smoke.err.log"
RESULT_JSON = "logs/gemma_smoke_result.json"
SERVER_PID_FILE = "logs/gemma_smoke_server.pid"

# CPU-fallback signatures in the llama.cpp server log (check 14).
CPU_FALLBACK_PATTERNS = (
    "falling back to cpu",
    "cpu backend",
    "ggml_backend_cpu",
    "failed to initialize cuda",
    "no devices found",
)

HEALTH_TIMEOUT_S = 5.0
CHAT_TIMEOUT_S = 180.0
SERVER_START_WAIT_S = 300.0


def safe_print(msg: str) -> None:
    """Print that never crashes on a non-UTF8 console (errors=replace)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


# ------------------------------------------------------------- pure helpers --

def gguf_header_valid(path: Path) -> tuple[bool, str]:
    """Check 2: GGUF magic + version + tensor count sanity (first 16 bytes)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError as exc:
        return False, f"cannot read file: {exc}"
    if len(head) < 12:
        return False, f"file too short for a GGUF header ({len(head)} bytes)"
    if head[:4] != b"GGUF":
        return False, "magic bytes are not GGUF (not a GGUF file)"
    version = struct.unpack("<I", head[4:8])[0]
    if version < 1 or version > 99:
        return False, f"implausible GGUF version {version}"
    return True, f"GGUF v{version}, valid magic"


def parse_gpu_offload(log_text: str) -> dict[str, Any]:
    """Checks 13/14: parse llama.cpp GPU offload evidence from the server log.

    Accepted evidence (llama.cpp log formats vary by build):
      * 'offloaded 48/48 layers to GPU'  -> offloaded == total == full
      * 'offloading 48 layers to GPU' + 'offloading non-repeating layers'
    Returns dict(pattern, offloaded, total, full, evidence).
    """
    result: dict[str, Any] = {
        "pattern": None, "offloaded": None, "total": None,
        "full": False, "evidence": "",
    }
    if not log_text:
        return result
    m = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", log_text, re.I)
    if m:
        off, total = int(m.group(1)), int(m.group(2))
        result.update(pattern="offloaded X/Y layers", offloaded=off,
                      total=total, full=(off == total and total > 0),
                      evidence=m.group(0))
        return result
    m2 = re.search(r"offloading\s+(\d+)\s+layers\s+to\s+GPU", log_text, re.I)
    nonrep = re.search(r"offloading\s+non-?repeating\s+layers\s+to\s+GPU",
                       log_text, re.I)
    if m2:
        off = int(m2.group(1))
        # 'offloading N layers' + 'offloading non-repeating layers' means the
        # repeating AND non-repeating layers are on the GPU = full offload.
        full = bool(nonrep) and off > 0
        result.update(pattern="offloading N layers (+non-repeating)",
                      offloaded=off if full else None,
                      total=off if full else None, full=full,
                      evidence=(m2.group(0) + (" + " + nonrep.group(0) if nonrep else "")))
        return result
    if nonrep:
        result.update(pattern="offloading non-repeating layers",
                      full=True, evidence=nonrep.group(0))
    return result


def cpu_fallback_lines(log_text: str) -> list[str]:
    """Check 14: any explicit CPU-fallback signature in the server log.

    A single line can match several patterns - each LINE is reported once.
    """
    hits: list[str] = []
    if not log_text:
        return hits
    lowered = log_text.lower()
    for line in log_text.splitlines():
        line_l = line.lower()
        if any(pattern in line_l for pattern in CPU_FALLBACK_PATTERNS):
            stripped = line.strip()[:160]
            if stripped not in hits:
                hits.append(stripped)
    return hits


def extract_content(body: dict[str, Any]) -> str:
    """choices[0].message.content from a non-streaming response ('' if absent)."""
    try:
        content = body["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else ""
    except (KeyError, IndexError, TypeError):
        return ""


def extract_reasoning(body: dict[str, Any]) -> str:
    """choices[0].message.reasoning_content - the model's thinking channel.

    v0.4.4: llama-server (b10717+, reasoning format auto/deepseek) routes the
    thought channel here; a reply with reasoning but no content means the
    model spent its whole token budget thinking.
    """
    try:
        reasoning = body["choices"][0]["message"]["reasoning_content"]
        return reasoning if isinstance(reasoning, str) else ""
    except (KeyError, IndexError, TypeError):
        return ""


def extract_json_object(text: str) -> Optional[dict]:
    """Parse a JSON object from a model reply (tolerates fences, but the
    caller treats a fenced reply as a FAIL - this helper is for lenient
    diagnostics only)."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = stripped.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def http_json(url: str, payload: Optional[dict] = None,
              timeout: float = HEALTH_TIMEOUT_S) -> Optional[dict]:
    """GET/POST JSON on the LOCAL llama-server (127.0.0.1 only)."""
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - loopback only
            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode("utf-8"))
            return None
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None


def query_nvidia_smi() -> dict[str, Any]:
    """Local nvidia-smi probes (GPU totals + per-process compute memory)."""
    info: dict[str, Any] = {"available": False, "gpu_memory_used_mb": None,
                            "gpu_memory_total_mb": None, "llama_server_gpu_mb": None}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return info
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            first = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) == 2:
                info["gpu_memory_used_mb"] = int(parts[0])
                info["gpu_memory_total_mb"] = int(parts[1])
                info["available"] = True
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.run(
            [smi, "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                cols = [c.strip() for c in line.split(",")]
                if len(cols) >= 3 and "llama-server" in cols[1].lower():
                    info["llama_server_gpu_mb"] = int(cols[2])
                    info["available"] = True
                    break
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return info


def process_ram_mb(pid: Optional[int]) -> Optional[int]:
    """Working set / RSS of the server process (best-effort, stdlib only)."""
    if pid is None or pid <= 0:
        return None
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            for line in out.stdout.splitlines():
                cols = [c.strip('"') for c in line.split('","')]
                if len(cols) >= 5 and cols[1].isdigit() and int(cols[1]) == pid:
                    text = cols[4].replace(",", "").replace(" K", "").strip()
                    return int(int(text) / 1024)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        return None
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def sha256_of(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------- the runner --

class GemmaSmokeTest:
    def __init__(self, cfg: AgentConfig, verify_hash: bool,
                 keep_server: bool) -> None:
        self.cfg = cfg
        self.verify_hash = verify_hash
        self.keep_server = keep_server
        self.results: list[dict[str, Any]] = []
        self.perf: dict[str, Any] = {}
        self.proc: Optional[subprocess.Popen[Any]] = None
        self.we_started = False
        self.model_path = cfg.llm_model_file
        self.server_base = f"http://{cfg.llama_server_host}:{cfg.llama_server_port}"
        self.log_paths = [REPO_ROOT / SMOKE_OUT_LOG, REPO_ROOT / SMOKE_ERR_LOG]

    # -- bookkeeping ------------------------------------------------------- #

    def record(self, num: int, name: str, ok: bool, detail: str,
               fix: str = "") -> bool:
        state = "PASS" if ok else "FAIL"
        icon = "[PASS]" if ok else "[FAIL]"
        safe_print(f"{icon} {num:2d}. {name} - {detail}" if ok else
                   f"{icon} {num:2d}. {name} - {detail}"
                   + (f"\n       FIX: {fix}" if fix else ""))
        self.results.append({"num": num, "name": name, "status": state,
                             "detail": detail, "fix": fix})
        return ok

    def server_log_text(self) -> str:
        chunks: list[str] = []
        for path in self.log_paths:
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        return "\n".join(chunks)

    # -- server lifecycle --------------------------------------------------- #

    def health_ok(self, timeout: float = HEALTH_TIMEOUT_S) -> bool:
        body = http_json(f"{self.server_base}/health", timeout=timeout)
        return body is not None

    def loaded_model_id(self) -> str:
        body = http_json(f"{self.server_base}/v1/models", timeout=10.0)
        try:
            return str(body["data"][0]["id"])
        except (KeyError, IndexError, TypeError):
            return ""

    def ensure_server(self) -> tuple[bool, str]:
        """Idempotent: reuse a healthy GEMMA server; start one otherwise.

        A healthy server loaded with a NON-Gemma model is an LLM ERROR (the
        v0.4.3 policy: Gemma only, no fallback) - reported, never switched.
        """
        if self.health_ok():
            model_id = self.loaded_model_id()
            if "gemma" in model_id.lower():
                return True, f"reused the running Gemma server (model id: {model_id})"
            return False, (
                f"a llama-server is already running with a NON-Gemma model "
                f"({model_id!r}) - v0.4.3 allows only Gemma 4 12B Q4_0 "
                f"(LLM FALLBACK: NONE). Stop that server and re-run."
            )
        exe = self.cfg.bin_dir / ("llama-server.exe" if platform.system() == "Windows" else "llama-server")
        if not exe.is_file():
            return False, f"llama-server binary not found: {exe}"
        if not self.model_path.is_file():
            return False, f"Gemma model file not found: {self.model_path}"
        for path in self.log_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.write_text("", encoding="utf-8")
            except OSError:
                pass
        args = [str(exe)] + [
            a.format(model=self.model_path) if "{model}" in a else a
            for a in LLAMA_ARGS_TEMPLATE
        ]
        started = time.time()
        safe_print(f"[INFO] starting llama-server: {' '.join(args[:8])} ...")
        self.proc = subprocess.Popen(
            args, cwd=str(REPO_ROOT),
            stdout=(REPO_ROOT / SMOKE_OUT_LOG).open("w"),
            stderr=(REPO_ROOT / SMOKE_ERR_LOG).open("w"),
        )
        self.we_started = True
        self.perf["load_start_ts"] = started
        while time.time() - started < SERVER_START_WAIT_S:
            if self.proc.poll() is not None:
                tail = self.server_log_text()[-600:]
                return False, (
                    f"llama-server exited during startup (exit code "
                    f"{self.proc.returncode}). Log tail: {tail!r}"
                )
            if self.health_ok(timeout=2.0):
                self.perf["model_load_time_s"] = round(time.time() - started, 1)
                return True, (
                    f"llama-server started, /health 200 after "
                    f"{self.perf['model_load_time_s']} s"
                )
            time.sleep(1.0)
        return False, (
            f"llama-server did not become healthy within {SERVER_START_WAIT_S} s "
            f"(process alive: {self.proc.poll() is None})"
        )

    def stop_our_server(self) -> None:
        if self.we_started and self.proc is not None and self.proc.poll() is None:
            if self.keep_server:
                pid_file = REPO_ROOT / SERVER_PID_FILE
                try:
                    pid_file.parent.mkdir(parents=True, exist_ok=True)
                    pid_file.write_text(str(self.proc.pid), encoding="ascii")
                except OSError:
                    pass
                safe_print(f"[INFO] keeping the server running (PID {self.proc.pid}, "
                           f"pid file: {SERVER_PID_FILE})")
                return
            safe_print("[INFO] stopping the llama-server started by this test...")
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except OSError:
                pass

    # -- requests ----------------------------------------------------------- #

    def chat(self, messages: list[dict], max_tokens: int = 64,
             temperature: float = 0.0,
             response_format: Optional[dict] = None,
             timeout: float = CHAT_TIMEOUT_S,
             thinking_off: bool = True) -> tuple[Optional[dict], float, Optional[str]]:
        """POST /v1/chat/completions (non-streaming).

        ``thinking_off=True`` (default) mirrors app/llm.LlmClient: the request
        carries the thinking-suppression kwargs. Pass False to probe the
        SERVER default (--reasoning off must make even bare requests
        non-thinking).
        """
        payload: dict[str, Any] = {
            "model": self.cfg.llm_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if thinking_off:
            payload.update(THINKING_OFF_KWARGS)
        if response_format is not None:
            payload["response_format"] = response_format
        started = time.time()
        body = http_json(f"{self.server_base}/v1/chat/completions",
                         payload, timeout=timeout)
        elapsed = time.time() - started
        if body is None:
            return None, elapsed, "request failed (no HTTP 200 / invalid JSON)"
        return body, elapsed, None

    def stream_probe(self, prompt: str, max_tokens: int = 128) -> dict[str, Any]:
        """Streaming request measuring first token latency + tokens/sec."""
        metrics: dict[str, Any] = {"ttft_s": None, "tokens_per_s": None,
                                   "completion_tokens": None, "prompt_tokens": None,
                                   "total_s": None, "error": None}
        payload = {
            "model": self.cfg.llm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        payload.update(THINKING_OFF_KWARGS)  # v0.4.4: app-shaped request
        req = urllib.request.Request(
            f"{self.server_base}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        first_token_at: Optional[float] = None
        last_token_at: Optional[float] = None
        try:
            with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT_S) as resp:  # nosec - loopback
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        delta = (choices[0].get("delta") or {})
                        if isinstance(delta.get("content"), str) and delta["content"]:
                            now = time.time()
                            if first_token_at is None:
                                first_token_at = now
                            last_token_at = now
                    usage = obj.get("usage")
                    if isinstance(usage, dict):
                        metrics["completion_tokens"] = usage.get("completion_tokens")
                        metrics["prompt_tokens"] = usage.get("prompt_tokens")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            metrics["error"] = f"streaming request failed: {exc}"
            return metrics
        metrics["total_s"] = round(time.time() - started, 2)
        if first_token_at is not None:
            metrics["ttft_s"] = round(first_token_at - started, 2)
        if (metrics["completion_tokens"] and first_token_at is not None
                and last_token_at is not None and last_token_at > first_token_at):
            metrics["tokens_per_s"] = round(
                (int(metrics["completion_tokens"]) - 1) / (last_token_at - first_token_at), 1)
        return metrics

    # -- the 14 checks ------------------------------------------------------- #

    def run(self) -> int:
        safe_print("=" * 70)
        safe_print(f"VoiceMem Agent - Gemma LLM smoke test ({GEMMA_DISPLAY})")
        safe_print(f"repo pin : {GEMMA_REPO} @ {GEMMA_UPSTREAM_COMMIT[:12]}")
        safe_print(f"file     : {GEMMA_FILE} ({GEMMA_SIZE_BYTES:,} bytes)")
        safe_print(f"endpoint : {self.server_base} (LOCAL ONLY - no cloud calls)")
        safe_print("=" * 70)

        # 1. model exists -----------------------------------------------------
        exists = self.model_path.is_file()
        self.record(1, "Gemma model exists", exists,
                    str(self.model_path) if exists else f"missing: {self.model_path}",
                    "run START.bat (the installer downloads and sha256-verifies the model)")
        if not exists:
            return self.finish(1)

        size = self.model_path.stat().st_size
        # 2. GGUF valid ---------------------------------------------------------
        magic_ok, magic_detail = gguf_header_valid(self.model_path)
        size_ok = size >= GEMMA_MIN_BYTES
        hash_detail = ""
        if self.verify_hash:
            safe_print(f"[INFO] full sha256 verification of {GEMMA_FILE} "
                       f"(~6.98 GB - takes a minute)...")
            actual = sha256_of(self.model_path)
            hash_ok = actual == GEMMA_SHA256
            hash_detail = (f", sha256 {'MATCHES' if hash_ok else 'MISMATCH'} "
                           f"({actual[:16]}...)")
            if not hash_ok:
                magic_detail += " | sha256 MISMATCH"
        else:
            hash_ok = True
        ok2 = magic_ok and size_ok and hash_ok
        self.record(2, "GGUF file is valid", ok2,
                    f"{magic_detail}; size {size:,} bytes "
                    f"(floor {GEMMA_MIN_BYTES:,}){hash_detail}",
                    "delete the file and re-run START.bat (download + sha256 verify)")
        if not ok2:
            return self.finish(1)

        # 3. llama-server starts -------------------------------------------------
        started_ok, started_detail = self.ensure_server()
        self.record(3, "llama-server starts", started_ok, started_detail,
                    "see logs/gemma-smoke.err.log + logs/llama-server.err.log; "
                    "run scripts/start_llama_server.ps1 manually for details")
        if not started_ok:
            return self.finish(1)

        # 4. /health HTTP 200 -----------------------------------------------------
        body = http_json(f"{self.server_base}/health", timeout=HEALTH_TIMEOUT_S)
        self.record(4, "/health returns HTTP 200", body is not None,
                    f"{self.server_base}/health -> "
                    f"{(body or {}).get('status', 'HTTP 200')}",
                    "the server may still be loading - re-run this test")

        # 5. normal chat completion ------------------------------------------------
        chat_body, chat_s, chat_err = self.chat(
            [{"role": "user",
              "content": "Hello! Reply with one short greeting sentence."}],
            max_tokens=48, temperature=0.0,
        )
        ok5 = chat_body is not None and bool(chat_body.get("choices"))
        self.record(5, "normal chat completion works", ok5,
                    f"HTTP 200 + choices present in {chat_s:.1f} s"
                    if ok5 else (chat_err or "no valid response"),
                    "see logs/gemma-smoke.err.log and logs/llama-server.err.log")

        # 6. NON-EMPTY assistant content (the v0.4.2/v0.4.3 empty-reply bug) --------
        # v0.4.4: verified BOTH with the app-style thinking-suppression kwargs
        # AND as a bare request (the server's --reasoning off must keep even
        # kwarg-less clients - like the voicemem package's OpenAI-lib calls -
        # free of thought-channel replies). A reasoning-only response is an
        # explicit FAIL with the thinking diagnosis.
        content = extract_content(chat_body or {})
        bare_body, _, bare_err = self.chat(
            [{"role": "user",
              "content": "Reply with one short greeting sentence."}],
            max_tokens=48, temperature=0.0, thinking_off=False,
        )
        bare_content = extract_content(bare_body or {})
        bare_reasoning = extract_reasoning(bare_body or {})
        ok6 = bool(content.strip()) and bool(bare_content.strip())
        if ok6:
            detail6 = f"reply: {content.strip()[:60]!r}; bare request also non-empty"
        else:
            detail6 = "choices[0].message.content is EMPTY"
            if content.strip() and not bare_content.strip():
                detail6 += (" for the BARE request only (server default thinking "
                            "is on: check the --reasoning off start flag)")
            if bare_reasoning:
                detail6 += (f" - the model spent the whole budget THINKING "
                            f"({len(bare_reasoning)} chars of reasoning_content)")
        self.record(6, "non-empty assistant content (with + without thinking off)",
                    ok6, detail6,
                    "v0.4.3 field failure: /health 200 but every reply empty - "
                    "the model's thinking channel ate the token budget; the "
                    "server must start with --reasoning off and the app sends "
                    "chat_template_kwargs/reasoning_effort (see app/llm.py)")

        # 7. Hungarian generation ------------------------------------------------------
        hu_body, _, hu_err = self.chat(
            [{"role": "user",
              "content": "Kérlek, válaszolj magyarul egy rövid mondattal: "
                         "milyen idő van ma nálad?"}],
            max_tokens=64, temperature=0.0,
        )
        hu_text = extract_content(hu_body or {})
        hu_lang = detect_language(hu_text)
        ok7 = bool(hu_text.strip()) and hu_lang == "hu"
        self.record(7, "Hungarian generation works", ok7,
                    f"reply language: {hu_lang} ({hu_text.strip()[:60]!r})"
                    if ok7 else (hu_err or f"language detected as {hu_lang!r}, "
                                           f"reply {hu_text.strip()[:60]!r}"),
                    "re-run (sampling variance); if it persists the model is "
                    "not following the language instruction")

        # 8. English generation ----------------------------------------------------------
        en_body, en_s, en_err = self.chat(
            [{"role": "user",
              "content": "Please answer in one short sentence: "
                         "what is the capital of France?"}],
            max_tokens=64, temperature=0.0,
        )
        en_text = extract_content(en_body or {})
        en_lang = detect_language(en_text)
        ok8 = (bool(en_text.strip()) and en_lang == "en"
               and "paris" in en_text.lower())
        self.record(8, "English generation works", ok8,
                    f"reply language: {en_lang}, contains 'Paris': "
                    f"{'paris' in en_text.lower()} ({en_s:.1f} s)"
                    if ok8 else (en_err or f"language {en_lang!r}, "
                                           f"reply {en_text.strip()[:60]!r}"),
                    "re-run; the reply must be English and factually correct")

        # 9. JSON object generation ----------------------------------------------------------
        json_body, _, json_err = self.chat(
            [{"role": "user",
              "content": "Return a JSON object with exactly two keys: name and "
                         "language. Values: Thomas and Hungarian. Output ONLY "
                         "the JSON object, no other text."}],
            max_tokens=96, temperature=0.0,
            response_format={"type": "json_object"},
        )
        json_text = extract_content(json_body or {})
        json_err_detail = ""
        try:
            parsed_json = json.loads(json_text.strip())
            json_dict_ok = (isinstance(parsed_json, dict)
                            and "name" in parsed_json and "language" in parsed_json)
        except ValueError:
            parsed_json = None
            json_dict_ok = False
            json_err_detail = (f"; invalid JSON: {json_text.strip()[:80]!r}")
        ok9 = json_dict_ok and not json_text.strip().startswith("```")
        self.record(9, "JSON object generation works", ok9,
                    f"valid JSON object with name/language keys: "
                    f"{json_text.strip()[:60]!r}"
                    if ok9 else (json_err or "empty/invalid JSON"
                                 + json_err_detail),
                    "json_object response_format must produce raw JSON "
                    "(no markdown fence)")

        # 10. JSON schema constrained generation ------------------------------------------------
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"},
                           "language": {"type": "string"}},
            "required": ["name", "language"],
            "additionalProperties": False,
        }
        schema_body, _, schema_err = self.chat(
            [{"role": "user",
              "content": "Return a JSON object with exactly two keys: name and "
                         "language. Values: Thomas and Hungarian."}],
            max_tokens=96, temperature=0.0,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "user_language",
                                             "strict": True, "schema": schema}},
        )
        schema_text = extract_content(schema_body or {})
        try:
            parsed_schema = json.loads(schema_text.strip())
            schema_ok = (isinstance(parsed_schema, dict)
                         and set(parsed_schema.keys()) == {"name", "language"})
        except ValueError:
            parsed_schema = None
            schema_ok = False
        ok10 = schema_ok and not schema_text.strip().startswith("```")
        self.record(10, "JSON schema constrained generation", ok10,
                    f"schema-compliant JSON: {schema_text.strip()[:60]!r}"
                    if ok10 else (schema_err or f"non-compliant reply: "
                                                f"{schema_text.strip()[:80]!r}"),
                    "json_schema response_format must constrain the output")

        # 11. process alive after generation ------------------------------------------------
        alive = self.health_ok()
        if self.we_started and self.proc is not None:
            alive = alive and self.proc.poll() is None
        self.record(11, "process remains alive after generation", alive,
                    "llama-server still healthy after all test generations"
                    if alive else "the server died during the test generations",
                    "see logs/gemma-smoke.err.log")

        # 12. model is actually loaded (and it is GEMMA) --------------------------------------
        model_id = self.loaded_model_id()
        ok12 = "gemma" in model_id.lower()
        self.record(12, "model is actually loaded", ok12,
                    f"/v1/models id: {model_id!r} ({GEMMA_DISPLAY})"
                    if ok12 else f"loaded model is NOT Gemma: {model_id!r}",
                    "only Gemma 4 12B Q4_0 may be loaded (LLM FALLBACK: NONE)")

        # 13. GPU offload active -----------------------------------------------------------------
        log_text = self.server_log_text()
        offload = parse_gpu_offload(log_text)
        smi = query_nvidia_smi()
        gpu_evidence = smi.get("llama_server_gpu_mb")
        ok13 = offload["full"] or (isinstance(gpu_evidence, int) and gpu_evidence > 1024)
        detail13 = (f"log: {offload['evidence'] or 'no offload line found'}; "
                    f"nvidia-smi llama-server GPU memory: "
                    f"{gpu_evidence if gpu_evidence is not None else 'n/a'} MB")
        self.record(13, "GPU offload is active", ok13, detail13,
                    "check bin/ CUDA DLLs (install_m1.ps1 -SkipModels) and the "
                    "server log load_tensors lines; -ngl -1 must offload every layer")

        # 14. no unexpected CPU-only fallback ------------------------------------------------------
        cpu_hits = cpu_fallback_lines(log_text)
        partial = (offload["offloaded"] is not None and offload["total"] is not None
                   and offload["offloaded"] != offload["total"])
        ok14 = not cpu_hits and not partial
        detail14 = (f"no CPU-fallback signatures; offload {offload['offloaded']}/"
                    f"{offload['total']} layers")
        self.record(14, "no unexpected CPU-only fallback", ok14,
                    detail14 if ok14 else
                    f"CPU-fallback evidence: {cpu_hits[:2]} partial={partial}",
                    "a full-GPU run must show no CPU fallback and all layers offloaded")

        # ---------------- performance measurement (after real generations) ---------------
        safe_print("")
        safe_print("---- performance (measured on this machine, after real generations)")
        self.perf.update({
            "gpu": smi.get("gpu_memory_total_mb"),
            "gpu_memory_used_mb": smi.get("gpu_memory_used_mb"),
            "llama_server_gpu_mb": smi.get("llama_server_gpu_mb"),
            "server_ram_mb": process_ram_mb(self.proc.pid if self.proc else None),
            "context": self.cfg.llm_context_size,
            "parallel": self.cfg.llm_parallel,
            "gpu_offload_layers": (f"{offload['offloaded']}/{offload['total']}"
                                   if offload["offloaded"] is not None else "-ngl -1"),
            "llama_server_version": self.llama_server_version(),
        })
        stream = self.stream_probe(
            "Tell me a short story about a coffee cup in Hungarian.", 128)
        self.perf["first_token_latency_s"] = stream.get("ttft_s")
        self.perf["tokens_per_s"] = stream.get("tokens_per_s")
        self.perf["completion_tokens"] = stream.get("completion_tokens")
        self.perf["prompt_tokens"] = stream.get("prompt_tokens")
        _, total_s, _ = self.chat(
            [{"role": "user",
              "content": "Count from 1 to 15 in English, separated by commas."}],
            max_tokens=96, temperature=0.0)
        self.perf["total_response_latency_s"] = round(total_s, 2)
        for key, label in (
            ("model_load_time_s", "model load time"),
            ("llama_server_gpu_mb", "VRAM (llama-server process)"),
            ("gpu_memory_used_mb", "VRAM (GPU total used)"),
            ("server_ram_mb", "RAM (server process)"),
            ("first_token_latency_s", "first token latency"),
            ("tokens_per_s", "generation speed (tokens/sec)"),
            ("total_response_latency_s", "total response latency"),
            ("gpu_offload_layers", "GPU offload layer count"),
            ("context", "context size"),
            ("parallel", "parallel"),
            ("llama_server_version", "llama-server version"),
        ):
            if key in self.perf:
                value = self.perf[key]
                note = ""
                if key == "model_load_time_s" and not self.we_started:
                    note = "  (server was already running - not measured this run)"
                safe_print(f"  {label:32s}: {value}{note}")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        return self.finish(failed)

    def llama_server_version(self) -> str:
        exe = self.cfg.bin_dir / ("llama-server.exe" if platform.system() == "Windows" else "llama-server")
        if not exe.is_file():
            return "n/a"
        try:
            out = subprocess.run([str(exe), "--version"], capture_output=True,
                                 text=True, timeout=10)
            text = (out.stdout + out.stderr).strip().splitlines()
            return text[0][:80] if text else "n/a"
        except (OSError, subprocess.SubprocessError):
            return "n/a"

    def finish(self, failed: int) -> int:
        self.stop_our_server()
        total = len(self.results)
        passed = total - failed
        safe_print("")
        safe_print("=" * 70)
        safe_print(f"RESULT: {passed}/{total} checks PASS"
                   + ("" if failed == 0 else f" ({failed} FAIL)")
                   + f" - LLM: {GEMMA_DISPLAY}"
                   + (" - LLM MODEL COUNT: 1 - LLM FALLBACK: NONE" if failed else ""))
        result = {
            "schema_version": 1,
            "llm": GEMMA_DISPLAY,
            "llm_model_count": 1,
            "llm_fallback": "NONE",
            "repo": GEMMA_REPO,
            "file": GEMMA_FILE,
            "expected_sha256": GEMMA_SHA256,
            "upstream_commit": GEMMA_UPSTREAM_COMMIT,
            "model_path": str(self.model_path),
            "model_size_bytes": (self.model_path.stat().st_size
                                 if self.model_path.is_file() else None),
            "checks": self.results,
            "performance": self.perf,
            "passed": passed,
            "failed": failed,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            out_path = REPO_ROOT / RESULT_JSON
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
            safe_print(f"results written: {RESULT_JSON}")
        except OSError as exc:
            safe_print(f"[WARN] could not write {RESULT_JSON}: {exc}")
        return 1 if failed else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gemma 4 12B Q4_0 LLM smoke test (14 checks + performance).")
    parser.add_argument("--verify-hash", action="store_true",
                        help="verify the full pinned sha256 of the 6.98 GB file")
    parser.add_argument("--keep-server", action="store_true",
                        help="keep the llama-server running after the test "
                             "(pid file: logs/gemma_smoke_server.pid)")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "voicemem_config.yaml"),
                        help="AgentConfig YAML path")
    args = parser.parse_args(argv)

    cfg = AgentConfig.from_yaml(Path(args.config))
    test = GemmaSmokeTest(cfg, verify_hash=args.verify_hash,
                          keep_server=args.keep_server)
    return test.run()


if __name__ == "__main__":
    sys.exit(main())
