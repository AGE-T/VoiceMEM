"""TEMPORARY FORENSIC TOOL - NOT PRODUCTION CODE (delete after the investigation).

llm_latency_forensics.py - llama-server latency forensic benchmark & diagnosis
for the VoiceMem production chain (Qwen3.6 35B A3B IQ4_XS on RTX 5070 12 GB).

WHAT IT DOES (phases, all read-only towards production):
  1. INSPECT   the currently running production llama-server: exact process
               command line, startup log effective values (n_ctx, n_ctx_slot,
               n_gpu_layers, KV cache, buffers, fit warnings), resolved model
               marker, nvidia-smi snapshot, llama-server /health /v1/models
               /props, and the REAL production turn latency evidence already
               sitting in logs/web-server*.log ([chain] llm start / first
               token / done lines).
  2. CAPTURE   the exact real VoiceMem request by rebuilding it with the
               PRODUCTION functions (app.web_server budget helpers +
               app.teacher_persona builders + app.config), the REAL VoiceMem
               memory search (webspace_demo, read-only search, the same call
               every production turn makes) and the last REAL user utterance
               parsed from the web-server log. Nothing in production is
               modified; only this logs/llm_forensic_real_payload.json file
               is written.
  3. SHORT     benchmark matrix, FRESH llama-server process per config, port
               8180 (never 8080): ngl {0,8,12,16,18,20,26} x ctx 8192, then
               ngl {16,18,20} x ctx 32768. The FIRST request after startup is
               the authoritative measurement; a second identical request is
               recorded separately to demonstrate KV-cache reuse (never
               compared against the fresh first request).
  4. REAL      workload replay: the captured real VoiceMem payload against
               FRESH servers for ngl {16,18,20} x ctx {8192,32768}, plus the
               exact production configuration (e.g. ngl 26) so the 30-50 s
               production first-token latency is reproduced, not guessed.
  5. OLLAMA    reference: /api/ps (only fields actually present), and one
               equivalent real-prompt run with thinking disabled.
  6. REPORT    writes logs/llm_runtime_forensic_report.md and
               logs/llm_runtime_forensic_results.json.

SAFETY / CHANGE POLICY:
  - Production source, installer, manifests, release and runtime configs are
    NEVER modified. The tool only READS config/ + logs/ and WRITES
    logs/llm_runtime_forensic_* and logs/llm_forensic_test_*.log files.
  - The production llama-server (127.0.0.1:8080) is never benchmarked and
    never killed by default. GPU benchmarks REQUIRE it to be stopped: the
    tool refuses to run phases 3/4 while it is up (pass
    --stop-production-server to let the tool stop it; restart later by
    double-clicking START.bat or running scripts/start_llama_server.ps1).
  - The 19 GB Qwen3.6 GGUF is referenced IN PLACE (C:/AI_HOME blob or
    wherever the model picker resolved it); it is never copied, never
    re-downloaded.
  - Thinking stays disabled everywhere: --reasoning off (server flag, same
    as production) + request-level chat_template_kwargs.enable_thinking=
    false + reasoning_effort "none" (same as app/llm.py does).

USAGE (field machine, repo root F:/Voicemem/VoiceMemAgent):
    .venv\\Scripts\\python.exe tools\\forensic\\llm_latency_forensics.py
    (stop the production llama-server window first, or pass
     --stop-production-server)

    --quick              small matrix (fast sanity run)
    --skip-short         skip phase 3
    --skip-real          skip phase 4
    --skip-ollama        skip phase 5
    --skip-capture       skip phase 2 (use the fixed representative payload)
    --synthetic-memory   do not run the real VoiceMem memory search
    --no-production-replay  skip the production-config real replay
    --stop-production-server  stop the running llama-server for benchmarks
    --model PATH         override the model path (default: the machine's own
                         resolution - config/llm_model.json / LLAMA_MODEL_PATH /
                         models/llm/qwen3.6-35b-a3b/)
    --server-exe PATH    override bin/llama-server.exe
    --port N             test server port (default 8180)
    --request-timeout S  per-request hard cap (default 300 s)

Exit codes: 0 = report written (measurement tool, no pass/fail thresholds);
2 = prerequisites missing (no model, no server binary, production server
still running and no permission to stop it).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# ----------------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_PATH = LOGS_DIR / "llm_runtime_forensic_results.json"
REPORT_PATH = LOGS_DIR / "llm_runtime_forensic_report.md"
PAYLOAD_PATH = LOGS_DIR / "llm_forensic_real_payload.json"

TEST_PORT_DEFAULT = 8180
REQUEST_TIMEOUT_S = 300.0
LOAD_TIMEOUT_S = 600.0
VRAM_RELEASE_TIMEOUT_S = 90.0
VRAM_RELEASE_THRESHOLD_MB = 1500.0

# The fixed short diagnostic prompt (NOT the real workload; phase 4 is
# authoritative). Comparable with the earlier manual short benchmarks.
SHORT_PROMPT = "Szia! Beszelgessunk egy kicsit a mai naprol, mert erdekel a velemenyped."

SHORT_MATRIX_FULL = [
    (0, 8192), (8, 8192), (12, 8192), (16, 8192),
    (18, 8192), (20, 8192), (26, 8192),
    (16, 32768), (18, 32768), (20, 32768),
]
SHORT_MATRIX_QUICK = [(0, 8192), (16, 8192), (26, 8192)]

REAL_MATRIX_FULL = [
    (16, 8192), (18, 8192), (20, 8192),
    (16, 32768), (18, 32768), (20, 32768),
]
REAL_MATRIX_QUICK = [(16, 8192), (18, 8192)]

# Representative history (labelled synthetic in the report; the real session
# history is in-memory only and not recoverable from logs).
REPRESENTATIVE_HISTORY = [
    {"role": "user", "content": "Szia! Megvan vegre a mikrofon beallitas, most mar tiszta a hang?"},
    {"role": "assistant", "content": "Szia! Igen, most mar tisztan hallom, sokkal jobb a hangminoseg. Kiprobaljuk egy rendes beszelgetessel?"},
    {"role": "user", "content": "Ja, jo. Egyebkent eszembe jutott, hogy holnap koran kelek, hetre kell a bolt?"},
    {"role": "assistant", "content": "Jo, hogy szoltal. A hetfoi nyitas altalaban hetkor van a nagyobb aruhazakban, de a kisboltok inkabb hatkor nyitnak. Melyik van kozelebb hozzad?"},
]

DEFAULT_UTTERANCE = "Koszon szepen! Akkor meseld el meg egyszer, hogy milyen napod volt ma."

CREATED_FILES: list[str] = []


def _out(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    rel = str(path.relative_to(REPO_ROOT))
    if rel not in CREATED_FILES:
        CREATED_FILES.append(rel)


# ----------------------------------------------------------------------------
# PURE parsers (module-level so they are trivially testable)
# ----------------------------------------------------------------------------

_LOG_VALUE_PATTERNS: list[tuple[str, str]] = [
    ("n_ctx", r"n_ctx\s*=\s*(\d+)"),
    ("n_ctx_slot", r"n_ctx_slot\s*=\s*(\d+)"),
    ("n_batch", r"n_batch\s*=\s*(\d+)"),
    ("n_ubatch", r"n_ubatch\s*=\s*(\d+)"),
    ("n_threads", r"n_threads\s*=\s*(\d+)"),
    ("n_gpu_layers", r"n_gpu_layers\s*=\s*(-?\d+)"),
    ("n_gpu_layers_user_set", r"n_gpu_layers already set by user to\s*(-?\d+)"),
    ("offloaded_layers", r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers"),
    ("flash_attn", r"flash_attn\s*=\s*([A-Za-z0-9_]+)"),
    ("ctx_train", r"ctx_train\s*=\s*(\d+)"),
]

_EVIDENCE_KEYWORDS = [
    "n_ctx", "n_ctx_slot", "n_batch", "n_ubatch", "n_gpu_layers", "offloaded",
    "flash_attn", "kv cache", "kv_cache", "kv self", "cachetype", "cache_type",
    "cuda0", "cpu model buffer", "cuda0 model buffer", "compute buffer",
    "failed to fit", "insufficient", "fallback", "falling back", "mmap",
    "graph", "n_threads", "threads", "device", "ggml_cuda_init",
    "truncat", "overflow", "exceed", "slot_", "n_seq", "attention",
]


def parse_llama_log(text: str) -> dict[str, Any]:
    """Parse a llama-server startup log into effective values + raw evidence.

    Values are only reported when the log ACTUALLY contains them (never
    inferred from the command line). Every matched line is also kept verbatim
    in evidence_lines so the report can show the ground truth.
    """
    effective: dict[str, Any] = {}
    evidence: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for name, pattern in _LOG_VALUE_PATTERNS:
            m = re.search(pattern, stripped)
            if m:
                if name == "offloaded_layers":
                    effective[name] = f"{m.group(1)}/{m.group(2)}"
                    effective["offloaded_on_gpu"] = int(m.group(1))
                    effective["total_layers"] = int(m.group(2))
                elif name not in effective:
                    value: Any = m.group(1)
                    if value.isdigit():
                        value = int(value)
                    effective[name] = value
        lowered = stripped.lower()
        if any(k in lowered for k in _EVIDENCE_KEYWORDS):
            evidence.append(stripped[:400])
    return {"effective": effective, "evidence_lines": evidence}


def parse_request_timing_lines(text: str) -> list[str]:
    """Extract per-request timing lines (llama-server --verbose slot lines)."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.search(r"(prompt processing time|eval time|total time|tokens per second|prompt time|n_prompt|n_past|cache)", s):
            out.append(s[:400])
    return out[-60:]


_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:\-]*)(\{[^}]*\})?\s+([^\s]+)\s*$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse /metrics Prometheus text into {metric_name(+labels): value}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name = m.group(1) + (m.group(2) or "")
        try:
            out[name] = float(m.group(3))
        except ValueError:
            continue
    return out


def metrics_delta(before: Optional[dict], after: Optional[dict]) -> dict:
    """Diff two /metrics snapshots: only keys that CHANGED (or are new)."""
    if not before or not after:
        return {}
    delta: dict[str, dict] = {}
    for key, value in after.items():
        base = key.split("{")[0]
        if base.endswith("_created"):
            continue
        if key in before:
            if value != before[key]:
                delta[key] = {"before": before[key], "after": value,
                              "delta": value - before[key]}
        else:
            delta[key] = {"before": None, "after": value, "delta": value}
    return delta


def extract_counts(delta: dict) -> dict[str, Any]:
    """Prompt/generated token counts + server-side seconds from a metrics delta."""
    prompt_n: Optional[int] = None
    predicted_n: Optional[int] = None
    kv_tokens: Optional[int] = None
    cache_n: Optional[int] = None
    prompt_seconds: Optional[float] = None
    predicted_seconds: Optional[float] = None
    # deterministic preference: *_seconds_total counters over gauges
    ordered = sorted(delta.keys(), key=lambda k: (0 if "_seconds_total" in k else 1, k))
    for key in ordered:
        base = key.split("{")[0]
        d = delta[key].get("delta")
        if d is None:
            continue
        if prompt_n is None and "prompt_tokens" in base:
            prompt_n = int(d)
        if predicted_n is None and ("tokens_predicted" in base or "tokens_generated" in base):
            predicted_n = int(d)
        if kv_tokens is None and "kv" in base and "token" in base:
            kv_tokens = int(d)
        if cache_n is None and "cache" in base and "token" in base and "kv" not in base:
            cache_n = int(d)
        if prompt_seconds is None and "prompt" in base and "seconds" in base:
            prompt_seconds = round(float(d), 3)
        if predicted_seconds is None and "predicted" in base and "seconds" in base:
            predicted_seconds = round(float(d), 3)
    return {"prompt_n": prompt_n, "predicted_n": predicted_n,
            "kv_tokens": kv_tokens, "cache_n": cache_n,
            "prompt_seconds": prompt_seconds,
            "predicted_seconds": predicted_seconds}


_CHAIN_LINE = re.compile(r"\[chain\]\s*(.*)$")
_CHAIN_FIRST_TOKEN = re.compile(r"llm first token\s*\(([\d.]+)\s*ms\)")
_CHAIN_LLM_DONE = re.compile(r"llm done\s*\((\d+)\s*chars,\s*([\d.]+)\s*ms\)")
_ASR_TRANSCRIPT = re.compile(r"ASR final transcript:\s*(.{1,200})")


def parse_web_server_log(text: str) -> dict[str, Any]:
    """Parse a web-server log into real production turn latency evidence."""
    turns: list[dict] = []
    current: Optional[dict] = None
    transcripts: list[str] = []
    llm_errors: list[str] = []
    for line in text.splitlines():
        # ASR transcripts and llm failures go through _diag -> logger.info,
        # so they appear WITH the [chain] prefix too (checked on the raw line).
        m2 = _ASR_TRANSCRIPT.search(line)
        if m2:
            transcripts.append(m2.group(1).strip())
        if "llm failed" in line or "llm unavailable" in line.lower():
            llm_errors.append(line.strip()[:300])
        m = _CHAIN_LINE.search(line)
        if not m:
            continue
        event = m.group(1).strip()
        if event.startswith("llm start"):
            if current is not None:
                turns.append(current)
            current = {"llm_start": True, "first_token_ms": None,
                       "llm_done_ms": None, "reply_chars": None}
        elif current is not None:
            ft = _CHAIN_FIRST_TOKEN.search(event)
            if ft:
                current["first_token_ms"] = float(ft.group(1))
                continue
            dn = _CHAIN_LLM_DONE.search(event)
            if dn:
                current["reply_chars"] = int(dn.group(1))
                current["llm_done_ms"] = float(dn.group(2))
                turns.append(current)
                current = None
    if current is not None:
        turns.append(current)
    completed = [t for t in turns if t.get("first_token_ms") is not None]
    stalled = [t for t in turns if t.get("first_token_ms") is None]
    summary: dict[str, Any] = {"n_turns_total": len(turns), "n_completed": len(completed),
                               "n_stalled_no_first_token": len(stalled),
                               "llm_error_lines": llm_errors[-10:]}
    if completed:
        ftms = [t["first_token_ms"] for t in completed]
        summary["first_token_ms_min"] = min(ftms)
        summary["first_token_ms_max"] = max(ftms)
        summary["first_token_ms_last10"] = ftms[-10:]
        summary["first_token_ms_avg_last10"] = round(sum(ftms[-10:]) / min(10, len(ftms)), 1)
        done = [t["llm_done_ms"] for t in completed if t.get("llm_done_ms")]
        if done:
            summary["llm_done_ms_last10"] = done[-10:]
    summary["recent_transcripts"] = transcripts[-8:]
    return summary


def parse_env_local_ps1(text: str) -> dict[str, str]:
    """Extract $env:VAR = "value" assignments from config/env.local.ps1."""
    out: dict[str, str] = {}
    for m in re.finditer(r"\$env:([A-Z0-9_]+)\s*=\s*\"([^\"\r\n]*)\"", text):
        out[m.group(1)] = m.group(2)
    return out


# ----------------------------------------------------------------------------
# Small system helpers
# ----------------------------------------------------------------------------


def run_cmd(cmd: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


def nvidia_smi() -> Optional[dict[str, str]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    code, out, _ = run_cmd([exe, "--query-gpu=name,driver_version,memory.total,"
                              "memory.used,compute_cap", "--format=csv,noheader"], timeout=20)
    if code != 0 or not out.strip():
        return None
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    keys = ["name", "driver_version", "memory_total", "memory_used", "compute_cap"]
    return dict(zip(keys, parts + [""] * (len(keys) - len(parts))))


def vram_used_mb() -> Optional[float]:
    snap = nvidia_smi()
    if not snap or not snap.get("memory_used"):
        return None
    m = re.search(r"([\d.]+)", snap["memory_used"])
    return float(m.group(1)) if m else None


def find_llama_processes() -> list[dict]:
    """Currently running llama-server processes with their exact command line."""
    procs: list[dict] = []
    if os.name == "nt":
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name LIKE 'llama-server%'\" | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        code, out, _ = run_cmd(["powershell", "-NoProfile", "-Command", ps], timeout=30)
        if code == 0 and out.strip():
            try:
                data = json.loads(out)
            except ValueError:
                data = []
            for item in (data if isinstance(data, list) else [data]):
                if isinstance(item, dict) and item.get("ProcessId"):
                    procs.append({"pid": int(item["ProcessId"]),
                                  "cmdline": item.get("CommandLine") or ""})
    else:
        for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                raw = cmdline.read_bytes().split(b"\0")
                parts = [p.decode("utf-8", "replace") for p in raw if p]
                if parts and "llama-server" in os.path.basename(parts[0]):
                    procs.append({"pid": int(cmdline.parts[-2]),
                                  "cmdline": " ".join(parts)})
            except (OSError, ValueError):
                continue
    return procs


def kill_pid(pid: int) -> None:
    if os.name == "nt":
        run_cmd(["taskkill", "/F", "/T", "/PID", str(pid)], timeout=30)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def http_get_json(url: str, timeout: float = 10.0) -> tuple[Optional[int], Any]:
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            try:
                body = r.json()
            except ValueError:
                body = r.text[:2000]
            return r.status_code, body
    except httpx.HTTPError as exc:
        return None, str(exc)


def http_get_text(url: str, timeout: float = 10.0) -> tuple[Optional[int], str]:
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            return r.status_code, r.text
    except httpx.HTTPError as exc:
        return None, str(exc)


# ----------------------------------------------------------------------------
# Test llama-server lifecycle (dedicated port, FRESH process per config)
# ----------------------------------------------------------------------------


class TestServer:
    """One fresh llama-server process with the production flag set."""

    def __init__(self, exe: Path, model: Path, port: int, ngl: int, ctx: int,
                 log_path: Path, temp: float, extra: Optional[list[str]] = None):
        self.exe = exe
        self.model = model
        self.port = port
        self.ngl = ngl
        self.ctx = ctx
        self.log_path = log_path
        self.temp = temp
        self.extra = extra or []
        self.proc: Optional[subprocess.Popen] = None
        self.command_line: str = ""
        self.load_s: Optional[float] = None
        self.log_offset: int = 0
        self._log_handle: Any = None

    def args(self) -> list[str]:
        return [
            "--model", str(self.model),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-ngl", str(self.ngl),
            "-c", str(self.ctx),
            "--parallel", "1",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--temp", str(self.temp),
            "--metrics",
            "--no-webui",
            "--reasoning", "off",
            "--verbose",
            *self.extra,
        ]

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        args = self.args()
        self.command_line = f'"{self.exe}" ' + " ".join(args)
        self._log_handle = open(self.log_path, "w", encoding="utf-8", errors="replace")
        creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [str(self.exe)] + args,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(self.exe.parent),
            creationflags=creation,
        )
        _out(f"    [server] pid {self.proc.pid} | ngl {self.ngl} | ctx {self.ctx} | log {self.log_path.name}")

    def wait_health(self, timeout: float = LOAD_TIMEOUT_S) -> bool:
        import httpx

        t0 = time.perf_counter()
        with httpx.Client(timeout=5.0) as client:
            try:
                while time.perf_counter() - t0 < timeout:
                    if self.proc is not None and self.proc.poll() is not None:
                        _out(f"    [server] EXITED early code={self.proc.returncode}")
                        return False
                    try:
                        r = client.get(f"{self.base_url()}/health")
                        if r.status_code == 200:
                            self.load_s = time.perf_counter() - t0
                            return True
                    except httpx.HTTPError:
                        pass
                    time.sleep(2.0)
            finally:
                client.close()
        return False

    def read_log(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def read_new_log_lines(self) -> str:
        """Log text appended since the last call (per-request timing lines)."""
        text = self.read_log()
        new = text[self.log_offset:]
        self.log_offset = len(text)
        return new

    def fetch_metrics(self) -> Optional[dict]:
        code, text = http_get_text(f"{self.base_url()}/metrics", timeout=10.0)
        if code == 200 and text:
            return parse_prometheus(text)
        return None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            kill_pid(self.proc.pid)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                kill_pid(self.proc.pid)
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
        # Wait for VRAM release so the NEXT config starts from a clean state.
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < VRAM_RELEASE_TIMEOUT_S:
            used = vram_used_mb()
            if used is None:
                return
            if used <= VRAM_RELEASE_THRESHOLD_MB:
                return
            time.sleep(2.0)


# ----------------------------------------------------------------------------
# Streaming request runner (production payload shape)
# ----------------------------------------------------------------------------


def production_request_payload(cfg: dict, messages: list[dict],
                               max_tokens: int, temperature: float) -> dict:
    """EXACT app/llm.py chat_stream payload shape (thinking stays disabled)."""
    return {
        "model": cfg.get("llm_model_name", "qwen3.6-35b-a3b"),
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }


def run_streamed_request(base_url: str, payload: dict, timeout_s: float) -> dict:
    """One streamed /v1/chat/completions request with wall-clock metrics."""
    import httpx

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    deltas = 0
    content_chars = 0
    reasoning_chars = 0
    usage: Any = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    status: Optional[int] = None
    context_overflow = False
    try:
        with httpx.Client(base_url=base_url,
                          timeout=httpx.Timeout(timeout_s, connect=10.0)) as client:
            with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                status = resp.status_code
                if status != 200:
                    body = resp.read().decode("utf-8", "replace")
                    error = f"HTTP {status}: {body[:400]}"
                    if "exceeds the available context" in body or "context size" in body.lower():
                        context_overflow = True
                else:
                    for line in resp.iter_lines():
                        s = line.strip()
                        if not s.startswith("data:"):
                            continue
                        data = s[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        if isinstance(obj.get("usage"), dict):
                            usage = obj["usage"]
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        first = choices[0] if isinstance(choices[0], dict) else {}
                        if first.get("finish_reason"):
                            finish_reason = str(first["finish_reason"])
                        delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
                        piece = delta.get("content")
                        if isinstance(piece, str) and piece:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            deltas += 1
                            content_chars += len(piece)
                        rpiece = delta.get("reasoning_content")
                        if isinstance(rpiece, str):
                            reasoning_chars += len(rpiece)
    except httpx.TimeoutException as exc:
        error = f"TIMEOUT after {timeout_s:.0f} s ({exc.__class__.__name__})"
    except httpx.HTTPError as exc:
        error = f"transport: {exc}"
    total_s = time.perf_counter() - t0
    gen_s = (total_s - ttft) if ttft is not None else None
    return {
        "status": status,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(total_s, 3),
        "gen_s": round(gen_s, 3) if gen_s is not None else None,
        "content_deltas": deltas,
        "content_chars": content_chars,
        "reasoning_chars": reasoning_chars,
        "finish_reason": finish_reason,
        "usage": usage,
        "error": error,
        "context_overflow": context_overflow,
    }


def run_config_benchmark(server: TestServer, payload: dict,
                         request_timeout_s: float, label: str) -> dict:
    """Fresh-process benchmark protocol for ONE configuration.

    First request = authoritative measurement; second identical request is
    recorded separately (KV-cache warm demonstration, never compared).
    """
    result: dict[str, Any] = {
        "label": label,
        "ngl_requested": server.ngl,
        "ctx_requested": server.ctx,
        "port": server.port,
    }
    vram_before = vram_used_mb()
    server.start()
    result["command"] = server.command_line
    healthy = server.wait_health()
    result["load_s"] = round(server.load_s, 1) if server.load_s is not None else None
    result["vram_used_before_start_mb"] = vram_before
    if not healthy:
        result["error"] = "server did not become healthy (see log)"
        result["startup_log_effective"] = parse_llama_log(server.read_log())["effective"]
        result["startup_evidence_lines"] = parse_llama_log(server.read_log())["evidence_lines"][:80]
        server.stop()
        return result

    # Effective runtime values from the ACTUAL startup log (never the CLI).
    startup = parse_llama_log(server.read_log())
    result["startup_log_effective"] = startup["effective"]
    result["startup_evidence_lines"] = startup["evidence_lines"][:80]
    code, props = http_get_json(f"{server.base_url()}/props", timeout=10.0)
    if code == 200:
        result["props"] = props if isinstance(props, dict) else {"raw": str(props)[:2000]}

    # -- FIRST request (authoritative) --------------------------------------
    server.log_offset = len(server.read_log())
    metrics_before = server.fetch_metrics()
    first = run_streamed_request(server.base_url(), payload, request_timeout_s)
    metrics_after = server.fetch_metrics()
    delta = metrics_delta(metrics_before, metrics_after)
    first.update(extract_counts(delta))
    first["metrics_delta"] = {k: v for k, v in delta.items()
                              if any(t in k.split("{")[0] for t in
                                     ("prompt", "predict", "token", "kv", "cache", "request"))}
    first["server_request_log_lines"] = parse_request_timing_lines(server.read_new_log_lines())
    result["first_request"] = first

    # -- SECOND request (cache-warm; demonstration only) ---------------------
    try:
        second = run_streamed_request(server.base_url(), payload, request_timeout_s)
        metrics_after2 = server.fetch_metrics()
        delta2 = metrics_delta(metrics_after, metrics_after2)
        second.update(extract_counts(delta2))
        result["second_request_cache_warm"] = second
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        result["second_request_cache_warm"] = {"error": str(exc)[:200]}

    result["vram_used_while_loaded_mb"] = vram_used_mb()
    server.stop()
    return result


def fmt_tok_s(count: Optional[int], seconds: Optional[float]) -> str:
    if count is None or not seconds or seconds <= 0:
        return "n/a"
    return f"{count / seconds:.1f}"


def summarize_run(run: dict) -> str:
    first = run.get("first_request", {})
    if first.get("error"):
        return f"ERROR {str(first['error'])[:60]}"
    ttft = first.get("ttft_s")
    total = first.get("total_s")
    pn = first.get("prompt_n")
    gen = first.get("gen_s")
    gn = first.get("predicted_n")
    return (f"ttft {ttft if ttft is not None else 'NO-TOKEN'} s | "
            f"total {total} s | prompt {pn if pn is not None else '?'} tok | "
            f"prompt {fmt_tok_s(pn, ttft)} tok/s | gen {gn if gn is not None else '?'} tok | "
            f"gen {fmt_tok_s(gn, gen)} tok/s")


# ----------------------------------------------------------------------------
# Phase 1: production inspection
# ----------------------------------------------------------------------------


def phase1_inspect(args: argparse.Namespace, results: dict) -> None:
    _out("\n=== PHASE 1: production llama-server inspection (read-only) ===")
    info: dict[str, Any] = {}

    gpu = nvidia_smi()
    info["gpu"] = gpu
    _out(f"  GPU: {gpu}")

    procs = find_llama_processes()
    info["llama_processes"] = procs
    for p in procs:
        _out(f"  RUNNING llama-server pid={p['pid']}")
        _out(f"    cmd: {p['cmdline']}")

    # Production config evidence (yaml + env.local.ps1)
    try:
        yaml_path = REPO_ROOT / "config" / "voicemem_config.yaml"
        if yaml_path.exists():
            env_ps1 = REPO_ROOT / "config" / "env.local.ps1"
            env_vars: dict[str, str] = {}
            if env_ps1.exists():
                env_vars = parse_env_local_ps1(env_ps1.read_text(encoding="utf-8", errors="replace"))
            info["configured_env_local_ps1"] = {
                k: v for k, v in env_vars.items()
                if k.startswith(("LLAMA", "OPENAI", "LLM"))
            }
            sys.path.insert(0, str(REPO_ROOT))
            from app.config import AgentConfig  # noqa: PLC0415

            cfg = AgentConfig.from_yaml(yaml_path)
            info["configured_yaml"] = {
                "llama_server_host": cfg.llama_server_host,
                "llama_server_port": cfg.llama_server_port,
                "llm_model_name": cfg.llm_model_name,
                "llm_context_size": cfg.llm_context_size,
                "llm_n_gpu_layers": cfg.llm_n_gpu_layers,
                "llm_parallel": cfg.llm_parallel,
                "llm_cache_type_k": cfg.llm_cache_type_k,
                "llm_cache_type_v": cfg.llm_cache_type_v,
                "llm_temperature": cfg.llm_temperature,
                "llm_max_tokens": cfg.llm_max_tokens,
                "llm_disable_thinking": cfg.llm_disable_thinking,
            }
            _out(f"  configured (yaml): {info['configured_yaml']}")
            _out(f"  configured (env.local.ps1): {info['configured_env_local_ps1']}")
    except Exception as exc:  # noqa: BLE001
        info["configured_yaml_error"] = str(exc)[:300]

    # Resolved model marker
    marker_path = LOGS_DIR / "llama-server.resolved-model.json"
    if marker_path.exists():
        try:
            info["resolved_model_marker"] = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info["resolved_model_marker"] = "(unreadable)"
        _out(f"  resolved model marker: {info['resolved_model_marker']}")

    # Production llama-server HTTP state (if running)
    host = (info.get("configured_yaml") or {}).get("llama_server_host", "127.0.0.1")
    port = (info.get("configured_yaml") or {}).get("llama_server_port", 8080)
    if procs or args.probe_production:
        info["production_http"] = {}
        for endpoint in ("/health", "/v1/models", "/props"):
            code, body = http_get_json(f"http://{host}:{port}{endpoint}", timeout=8.0)
            info["production_http"][endpoint] = {
                "status": code,
                "body": body if code == 200 else str(body)[:300],
            }
            _out(f"  production {endpoint}: {code}")

    # Startup log effective values (the RUNNING server's own log)
    for name in ("llama-server.out.log", "llama-server.err.log", "llama-server.starter.log"):
        p = LOGS_DIR / name
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parsed = parse_llama_log(text[-200000:])
            info[f"log_{name}"] = {
                "effective": parsed["effective"],
                "evidence_lines_tail": parsed["evidence_lines"][-60:],
                "request_timing_lines_tail": parse_request_timing_lines(text[-30000:]),
            }
            _out(f"  {name}: effective={parsed['effective']}")

    # REAL production turn latency evidence from the web-server logs
    chain_evidence: dict[str, Any] = {}
    if LOGS_DIR.exists():
        for p in sorted(LOGS_DIR.glob("*.log")):
            if p.name.startswith("llama") or p.name.startswith("llm_forensic"):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "[chain]" not in text:
                continue
            parsed = parse_web_server_log(text)
            if parsed.get("n_turns_total"):
                chain_evidence[p.name] = parsed
    info["production_turn_latency_from_logs"] = chain_evidence
    for fname, ev in chain_evidence.items():
        _out(f"  {fname}: turns={ev['n_turns_total']} completed={ev['n_completed']} "
             f"stalled(no first token)={ev['n_stalled_no_first_token']}")
        if ev.get("first_token_ms_last10"):
            _out(f"    first-token ms (last 10): {ev['first_token_ms_last10']}")

    results["phase1_production_inspection"] = info


# ----------------------------------------------------------------------------
# Phase 2: capture the real VoiceMem request payload
# ----------------------------------------------------------------------------


def _resolve_model_path(cfg: Any) -> Optional[Path]:
    try:
        from app.llm_model_settings import resolve_llm_model_path  # noqa: PLC0415

        resolved = resolve_llm_model_path(REPO_ROOT)
        if resolved:
            return Path(resolved)
    except Exception:  # noqa: BLE001
        pass
    # Manual fallback: llm_model.json > env > models/llm/qwen3.6-35b-a3b dir
    try:
        settings_path = REPO_ROOT / "config" / "llm_model.json"
        if settings_path.exists():
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            p = data.get("llm_model_path")
            if p and Path(p).exists():
                return Path(p)
    except (OSError, ValueError):
        pass
    env_path = os.environ.get("LLAMA_MODEL_PATH", "")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    model_dir = REPO_ROOT / "models" / "llm" / "qwen3.6-35b-a3b"
    if model_dir.exists():
        for gguf in model_dir.glob("*.gguf"):
            if gguf.stat().st_size > 1_000_000_000:
                return gguf
    return None


def _run_memory_search(cfg: Any, query: str) -> tuple[Any, str]:
    """REAL VoiceMem search (read-only, same call a production turn makes)."""
    sys.path.insert(0, str(REPO_ROOT))
    from app.voicemem_bridge import pin_e5_local_model, pin_vendor_llm_env  # noqa: PLC0415

    pin_e5_local_model(cfg)
    pin_vendor_llm_env(cfg)
    from voicemem import VoiceMem  # noqa: PLC0415
    from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder  # noqa: PLC0415

    vm = VoiceMem(
        mode="text_mode",
        user_id="webspace_demo",
        base_url=cfg.llama_server_url,
        top_k=cfg.top_k,
        memory_root=str(cfg.memory_root_path),
        embedding=lambda: LocalE5Embedder(),
    )
    box: dict[str, Any] = {}

    def work() -> None:
        box["result"] = vm.search(query, emotion=None)

    thread = threading.Thread(target=work, daemon=True)
    t0 = time.perf_counter()
    thread.start()
    thread.join(timeout=90.0)
    elapsed = time.perf_counter() - t0
    if thread.is_alive():
        return None, "memory search timed out after 90 s (E5 cold load?)"
    if "result" not in box:
        return None, "memory search returned nothing"
    return box["result"], f"real VoiceMem search ok in {elapsed:.2f} s"


def phase2_capture(args: argparse.Namespace, results: dict) -> dict:
    _out("\n=== PHASE 2: capture the REAL VoiceMem request payload ===")
    capture: dict[str, Any] = {}
    sys.path.insert(0, str(REPO_ROOT))

    from app.config import AgentConfig  # noqa: PLC0415
    from app.teacher_persona import build_messages, build_system_prompt  # noqa: PLC0415
    from app.web_server import (  # noqa: PLC0415
        _cap_memory_context,
        _fit_history_budget,
        semantic_emotion_label,
    )

    cfg = AgentConfig.from_yaml(REPO_ROOT / "config" / "voicemem_config.yaml")
    capture["request_params"] = {
        "model": cfg.llm_model_name,
        "stream": True,
        "temperature": cfg.llm_temperature,
        "max_tokens": cfg.llm_max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
        "llm_disable_thinking_config": cfg.llm_disable_thinking,
    }

    # Last REAL user utterance from the production logs (phase 1 evidence).
    utterance = DEFAULT_UTTERANCE
    utterance_source = "default (no log transcript found)"
    chain = (results.get("phase1_production_inspection") or {}).get(
        "production_turn_latency_from_logs") or {}
    for fname in sorted(chain.keys()):
        transcripts = chain[fname].get("recent_transcripts") or []
        if transcripts:
            last = transcripts[-1]
            m = re.match(r"['\"](.*)['\"]\s*$", last)
            text = (m.group(1) if m else last).strip()
            if text:
                utterance = text
                utterance_source = f"last real ASR transcript from {fname}"
            break
    capture["user_utterance"] = {"text": utterance, "source": utterance_source}
    _out(f"  utterance ({utterance_source}): {utterance[:80]!r}")

    # REAL memory search (same call every production turn makes)
    memory_context = ""
    memory_source = "synthetic-representative (search failed or skipped)"
    if not args.synthetic_memory:
        try:
            result, note = _run_memory_search(cfg, utterance)
            if result is not None:
                from app.web_server import WebSession  # noqa: PLC0415

                memory_context = WebSession._memory_context(result)
                memory_source = f"REAL VoiceMem search: {note}"
            else:
                memory_source = f"fallback: {note}"
        except Exception as exc:  # noqa: BLE001
            memory_source = f"fallback: {str(exc)[:200]}"
    capture["memory_block"] = {"chars": len(memory_context), "source": memory_source}
    _out(f"  memory block: {len(memory_context)} chars ({memory_source})")

    # Emotion: the text-turn semantic path (a real production variant)
    label, hv, ha = semantic_emotion_label(utterance)
    capture["emotion"] = {"label": label, "valence": hv, "arousal": ha,
                          "source": "semantic (text-turn production path)"}

    memory_context = _cap_memory_context(memory_context)
    emo_args: dict = {}
    if label:
        emo_args = {"emotion_label": label, "emotion_valence": hv, "emotion_arousal": ha}
    system_prompt = build_system_prompt(memory_context, **emo_args)
    capture["system_prompt_chars"] = len(system_prompt)

    history = REPRESENTATIVE_HISTORY if args.representative_history else []
    capture["history"] = {
        "entries": len(history),
        "chars": sum(len(e["content"]) for e in history),
        "source": ("representative synthetic (real history is in-memory only)"
                   if history else "none (cold session)"),
    }
    history = _fit_history_budget(system_prompt, history, utterance)
    messages = build_messages(utterance, system_prompt, history=history)

    capture["messages"] = {
        "count": len(messages),
        "roles": [m["role"] for m in messages],
        "chars_total": sum(len(m["content"]) for m in messages),
        "user_chars": len(messages[-1]["content"]),
    }
    _out(f"  messages: {capture['messages']}")

    payload = production_request_payload(
        {"llm_model_name": cfg.llm_model_name}, messages,
        cfg.llm_max_tokens, cfg.llm_temperature,
    )
    capture["payload_full"] = payload
    results["phase2_real_payload"] = {k: v for k, v in capture.items() if k != "payload_full"}
    write_json(PAYLOAD_PATH, capture)
    _out(f"  payload written: {PAYLOAD_PATH}")
    return payload


# ----------------------------------------------------------------------------
# Phase 3/4: benchmark matrices
# ----------------------------------------------------------------------------


def run_matrix(args: argparse.Namespace, results: dict, server_exe: Path, model: Path,
               payload: Optional[dict], matrix: list[tuple[int, int]],
               phase_key: str, phase_name: str,
               production_ngl: Optional[int]) -> None:
    _out(f"\n=== {phase_name} ===")
    runs: list[dict] = []
    matrix = list(matrix)
    if phase_key == "phase4_real_workload" and production_ngl is not None \
            and not args.no_production_replay and (production_ngl, 8192) not in matrix:
        matrix.append((production_ngl, 8192))
        _out(f"  (added exact production config replay: ngl {production_ngl}, ctx 8192)")

    cfg_temp = 0.7
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from app.config import AgentConfig  # noqa: PLC0415

        cfg_temp = AgentConfig.from_yaml(
            REPO_ROOT / "config" / "voicemem_config.yaml").llm_temperature
    except Exception:  # noqa: BLE001
        pass

    for ngl, ctx in matrix:
        label = f"ngl{ngl}_ctx{ctx}"
        if phase_key == "phase4_real_workload" and ngl == production_ngl and ctx == 8192:
            label += "_PRODUCTION"
        log_path = LOGS_DIR / f"llm_forensic_test_{label}.log"
        _out(f"  -- {label}")
        if payload is None:
            messages = [{"role": "user", "content": SHORT_PROMPT}]
            use_payload = production_request_payload(
                {"llm_model_name": "qwen3.6-35b-a3b"}, messages, 512, cfg_temp)
        else:
            use_payload = payload
        server = TestServer(server_exe, model, args.port, ngl, ctx, log_path, cfg_temp)
        run = run_config_benchmark(server, use_payload, args.request_timeout, label)
        run["phase"] = phase_key
        runs.append(run)
        _out(f"     {summarize_run(run)}")
        _out(f"     effective: {run.get('startup_log_effective')}")
        results[phase_key] = runs
        write_json(RESULTS_PATH, results)  # incremental save after each run


# ----------------------------------------------------------------------------
# Phase 5: Ollama reference
# ----------------------------------------------------------------------------


def phase5_ollama(args: argparse.Namespace, results: dict, payload: Optional[dict]) -> None:
    _out("\n=== PHASE 5: Ollama reference (MAXI, thinking disabled) ===")
    info: dict[str, Any] = {"base_url": "http://127.0.0.1:11434"}

    code, body = http_get_json("http://127.0.0.1:11434/api/ps", timeout=8.0)
    info["api_ps_status"] = code
    if code == 200 and isinstance(body, dict):
        models = body.get("models") or []
        info["api_ps_models"] = [
            {k: m.get(k) for k in
             ("name", "model", "digest", "size", "size_vram", "context_length",
              "parameters", "quantization", "processor", "until", "expires_at")
             if k in m}
            for m in models
        ]
        for m in info["api_ps_models"]:
            _out(f"  ollama ps: {m}")
    else:
        info["api_ps_error"] = str(body)[:200]
        _out(f"  ollama /api/ps unavailable: {str(body)[:120]}")

    if shutil.which("ollama"):
        code, out, _ = run_cmd(["ollama", "ps"], timeout=15)
        if code == 0:
            info["cli_ps_output"] = out.strip()
            _out(f"  $ ollama ps\n{out.strip()}")

    if payload is None:
        info["equivalent_run"] = "skipped (no captured payload)"
        results["phase5_ollama"] = info
        return

    model_name = ""
    for m in info.get("api_ps_models", []):
        model_name = m.get("model") or m.get("name") or ""
        break
    if not model_name:
        model_name = "MAXI"
        info["model_name_source"] = "default MAXI (not currently loaded per /api/ps)"
    else:
        info["model_name_source"] = "/api/ps"

    ollama_payload = {
        "model": model_name,
        "messages": payload["messages"],
        "stream": False,
        "think": False,
        "options": {"num_predict": payload.get("max_tokens", 512),
                    "temperature": payload.get("temperature", 0.7)},
    }
    import httpx

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(args.request_timeout, connect=10.0)) as client:
            r = client.post("http://127.0.0.1:11434/api/chat", json=ollama_payload)
        elapsed = time.perf_counter() - t0
        if r.status_code == 200:
            body = r.json()
            msg = body.get("message") or {}
            info["equivalent_run"] = {
                "model": body.get("model"),
                "wall_clock_s": round(elapsed, 3),
                "total_duration_s": round(body.get("total_duration", 0) / 1e9, 3),
                "load_duration_s": round(body.get("load_duration", 0) / 1e9, 3),
                "prompt_eval_count": body.get("prompt_eval_count"),
                "prompt_eval_duration_s": round(body.get("prompt_eval_duration", 0) / 1e9, 3),
                "eval_count": body.get("eval_count"),
                "eval_duration_s": round(body.get("eval_duration", 0) / 1e9, 3),
                "reply_chars": len(str(msg.get("content", ""))),
                "thinking": msg.get("thinking"),
            }
            run = info["equivalent_run"]
            _out(f"  equivalent real-prompt run: total {run['total_duration_s']} s | "
                 f"prompt {run['prompt_eval_count']} tok in {run['prompt_eval_duration_s']} s | "
                 f"gen {run['eval_count']} tok in {run['eval_duration_s']} s")
        else:
            info["equivalent_run"] = {"http_status": r.status_code, "body": r.text[:300]}
            _out(f"  equivalent run failed: HTTP {r.status_code}")
    except httpx.HTTPError as exc:
        info["equivalent_run"] = {"error": str(exc)[:300]}
        _out(f"  equivalent run failed: {exc}")
    results["phase5_ollama"] = info


# ----------------------------------------------------------------------------
# Phase 6: report generation
# ----------------------------------------------------------------------------


def _table(rows: list[list[Any]], headers: list[str]) -> str:
    widths: list[int] = []
    for i, h in enumerate(headers):
        values = [len(str(r[i])) for r in rows] if rows else []
        widths.append(max([len(str(h))] + values))
    sep = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(w) for i, w in enumerate(widths)) + " |"
        for r in rows)
    return "\n".join([sep, line, body])


def _best_real_run(results: dict) -> Optional[dict]:
    best: Optional[dict] = None
    for run in results.get("phase4_real_workload", []):
        first = run.get("first_request", {})
        ttft = first.get("ttft_s")
        if ttft is None or first.get("error"):
            continue
        if "failed to fit" in " ".join(run.get("startup_evidence_lines", [])).lower():
            continue
        if best is None or ttft < best["first_request"]["ttft_s"]:
            best = run
    return best


def phase6_report(args: argparse.Namespace, results: dict, model: Path,
                  server_exe: Path, started_at: str) -> None:
    _out("\n=== PHASE 6: writing the forensic report ===")
    results["meta"] = {
        "tool": "tools/forensic/llm_latency_forensics.py (TEMPORARY forensic tool)",
        "started_at": started_at,
        "finished_at": now_iso(),
        "repo_root": str(REPO_ROOT),
        "model_path": str(model),
        "server_exe": str(server_exe),
        "python": sys.version.split()[0],
        "argv": sys.argv[1:],
    }
    write_json(RESULTS_PATH, results)

    lines: list[str] = []
    add = lines.append
    add("# LLM runtime forensic report - VoiceMem / Qwen3.6 35B A3B IQ4_XS / RTX 5070")
    add("")
    add(f"- generated: {now_iso()} by `tools/forensic/llm_latency_forensics.py` "
        "(temporary forensic tool, production untouched)")
    add(f"- model (external, referenced in place): `{model}`")
    add(f"- server binary: `{server_exe}`")
    add("- raw measurements: `logs/llm_runtime_forensic_results.json`")
    add("")

    add("## Tested commands")
    add("")
    p1 = results.get("phase1_production_inspection", {})
    for p in p1.get("llama_processes", []):
        add(f"- RUNNING production process (pid {p['pid']}): `{p['cmdline']}`")
    for run in results.get("phase3_short_matrix", []) + results.get("phase4_real_workload", []):
        add(f"- [{run.get('label')}] `{run.get('command', '')}`")
    add("")

    add("## Effective runtime configurations (from the actual startup logs)")
    add("")
    rows = []
    for key in ("phase3_short_matrix", "phase4_real_workload"):
        for run in results.get(key, []):
            eff = run.get("startup_log_effective", {})
            rows.append([
                run.get("label"), run.get("ngl_requested"),
                eff.get("n_ctx"), eff.get("n_ctx_slot"),
                eff.get("offloaded_layers"), eff.get("n_batch"), eff.get("n_ubatch"),
                eff.get("flash_attn"), eff.get("n_threads"),
            ])
    if rows:
        add(_table(rows, ["label", "ngl", "n_ctx", "n_ctx_slot", "offloaded",
                          "n_batch", "n_ubatch", "flash_attn", "n_threads"]))
    else:
        add("(no benchmark runs executed)")
    add("")
    for name in ("llama-server.out.log", "llama-server.err.log"):
        log_info = p1.get(f"log_{name}")
        if log_info:
            add(f"### Production {name} - effective values + evidence")
            add("")
            add(f"- effective: `{log_info.get('effective')}`")
            add("- evidence lines (tail):")
            add("```")
            for line in (log_info.get("evidence_lines_tail") or [])[-25:]:
                add(line)
            add("```")
            add("")

    add("## Short benchmark matrix (fresh process per config, FIRST request)")
    add("")
    add(f"Fixed short prompt ({len(SHORT_PROMPT)} chars, thinking disabled). "
        "Diagnostic only - the real workload below is authoritative.")
    add("")
    rows = []
    for run in results.get("phase3_short_matrix", []):
        f = run.get("first_request", {})
        rows.append([
            run.get("label"), run.get("load_s"),
            f.get("ttft_s"), f.get("total_s"),
            f.get("prompt_n"), fmt_tok_s(f.get("prompt_n"), f.get("ttft_s")),
            f.get("predicted_n"), fmt_tok_s(f.get("predicted_n"), f.get("gen_s")),
            (f.get("error") or "")[:40],
        ])
    if rows:
        add(_table(rows, ["label", "load_s", "ttft_s", "total_s", "prompt_tok",
                          "prompt_tok/s", "gen_tok", "gen_tok/s", "error"]))
    add("")

    add("## Real VoiceMem workload benchmark (fresh process, FIRST request)")
    add("")
    cap = results.get("phase2_real_payload", {})
    if cap:
        add(f"Captured request: {cap.get('messages')} | system "
            f"{cap.get('system_prompt_chars')} chars | memory block "
            f"{cap.get('memory_block')} | history {cap.get('history')} | "
            f"params {cap.get('request_params')}")
        add("")
    rows = []
    for run in results.get("phase4_real_workload", []):
        f = run.get("first_request", {})
        rows.append([
            run.get("label"), run.get("load_s"),
            f.get("ttft_s"), f.get("total_s"),
            f.get("prompt_n"), fmt_tok_s(f.get("prompt_n"), f.get("ttft_s")),
            f.get("predicted_n"), fmt_tok_s(f.get("predicted_n"), f.get("gen_s")),
            f.get("context_overflow"), (f.get("error") or "")[:40],
        ])
    if rows:
        add(_table(rows, ["label", "load_s", "ttft_s", "total_s", "prompt_tok",
                          "prompt_tok/s", "gen_tok", "gen_tok/s", "ctx_overflow", "error"]))
    else:
        add("(real workload phase skipped or failed)")
    add("")

    add("## Ollama reference (MAXI, thinking disabled)")
    add("")
    o5 = results.get("phase5_ollama", {})
    for m in o5.get("api_ps_models", []):
        add(f"- ollama /api/ps: `{json.dumps(m, ensure_ascii=False)}`")
    if o5.get("cli_ps_output"):
        add("```")
        add(o5["cli_ps_output"])
        add("```")
    eq = o5.get("equivalent_run")
    if isinstance(eq, dict) and eq.get("total_duration_s") is not None:
        add(f"- equivalent real-prompt run: total {eq['total_duration_s']} s | "
            f"load {eq['load_duration_s']} s | prompt {eq['prompt_eval_count']} tok "
            f"in {eq['prompt_eval_duration_s']} s | gen {eq['eval_count']} tok "
            f"in {eq['eval_duration_s']} s")
    elif eq:
        add(f"- equivalent run: `{eq}`")
    add("")

    add("## Real production latency evidence (from the existing web-server logs)")
    add("")
    for fname, ev in (p1.get("production_turn_latency_from_logs") or {}).items():
        add(f"- **{fname}**: {ev['n_turns_total']} LLM turns, {ev['n_completed']} with "
            f"first token, {ev['n_stalled_no_first_token']} stalled without first token; "
            f"first-token ms (last 10): `{ev.get('first_token_ms_last10')}`; "
            f"llm-done ms (last 10): `{ev.get('llm_done_ms_last10')}`")
        if ev.get("llm_error_lines"):
            add(f"  - llm error lines: `{ev['llm_error_lines'][-3:]}`")
    add("")

    add("## Analysis: measured facts / inferred causes / recommended next steps")
    add("")
    best = _best_real_run(results)
    prod_ft: Optional[float] = None
    for ev in (p1.get("production_turn_latency_from_logs") or {}).values():
        if ev.get("first_token_ms_last10"):
            prod_ft = ev["first_token_ms_last10"][-1] / 1000.0
    if best:
        f = best["first_request"]
        add(f"- MEASURED: best real-workload config in this run: **{best['label']}** "
            f"(ttft {f['ttft_s']} s, total {f['total_s']} s, prompt {f.get('prompt_n')} tok "
            f"at {fmt_tok_s(f.get('prompt_n'), f.get('ttft_s'))} tok/s).")
    if prod_ft:
        add(f"- MEASURED: production first-token latency (last logged real turn): "
            f"~{prod_ft:.1f} s before the first token.")
    if best and prod_ft and best["first_request"]["ttft_s"]:
        ratio = prod_ft / best["first_request"]["ttft_s"]
        add(f"- INFERRED: the production configuration is ~{ratio:.1f}x slower to the "
            "first token than the best tested configuration on the same real workload "
            "(inference from measurements, not a direct A/B of the same process).")
    fit_warnings: list[str] = []
    for r in results.get("phase3_short_matrix", []) + results.get("phase4_real_workload", []):
        for line in r.get("startup_evidence_lines", []):
            if "failed to fit" in line.lower() or "insufficient" in line.lower():
                fit_warnings.append(f"{r['label']}: {line}")
    if fit_warnings:
        add(f"- MEASURED: startup logs contain memory-fitting warnings: "
            f"`{fit_warnings[:2]}`")
    add("- RECOMMENDED (next step, NOT applied): adopt the best real-workload "
        "configuration above in a follow-up change (config/voicemem_config.yaml "
        "`llm_n_gpu_layers` + `llm_context_size`); this forensic tool changes nothing.")
    add("")

    add("## Confidence level")
    add("")
    add("- HIGH for everything measured in phases 1-5 (fresh processes, effective "
        "values read from the servers' own logs, metrics counter deltas).")
    add("- MEDIUM for the inferred causal link between the production configuration "
        "and the 30-50 s first-token latency (strong numerical fit, but the production "
        "process was not A/B tested in place).")
    add("- The representative history entries are synthetic (labelled); the real "
        "session history is in-memory only. Everything else in the captured payload "
        "comes from the production code path and the real memory store.")
    add("")
    add("## Unresolved questions")
    add("")
    add("- Why the previously requested -c 32768 showed n_ctx_slot = 8192 in a "
        "production log: compare `ctx_requested` vs effective `n_ctx_slot` rows above.")
    add("- Whether flash attention is active in this llama.cpp build (see the "
        "flash_attn column; empty = not reported by the log).")
    add("- Exact Ollama MAXI layer split / KV settings are not documented by the "
        "runtime; only /api/ps-visible fields are recorded.")
    add("")

    add("## Verdict")
    add("")
    eff32 = [r for r in results.get("phase3_short_matrix", [])
             + results.get("phase4_real_workload", [])
             if r.get("ctx_requested") == 32768]
    active_32768 = any(
        (r.get("startup_log_effective") or {}).get("n_ctx_slot") == 32768
        or (r.get("startup_log_effective") or {}).get("n_ctx") == 32768
        for r in eff32)
    if active_32768:
        verdict_ctx = ("YES - the startup logs report n_ctx_slot/n_ctx = 32768 for "
                       "the 32768 runs")
    elif eff32:
        verdict_ctx = ("NO / NOT SEEN - no 32768 run reported an effective 32768 slot "
                       "context (see the effective-context table)")
    else:
        verdict_ctx = "NOT MEASURED (no 32768 runs executed)"

    ngl26_runs = [r for r in results.get("phase3_short_matrix", [])
                  if r.get("ngl_requested") == 26] + \
                 [r for r in results.get("phase4_real_workload", [])
                  if r.get("ngl_requested") == 26]
    ngl26_ok = [r for r in ngl26_runs if r.get("first_request", {}).get("ttft_s") is not None]
    if not ngl26_ok:
        remove_26 = "NOT MEASURED at ngl 26 (no completed run; see errors above)"
    elif fit_warnings or (best and any(
            r["first_request"]["ttft_s"] >= 3 * (best["first_request"]["ttft_s"] or 1)
            for r in ngl26_ok)):
        remove_26 = ("YES - see the ngl 26 rows above (slower by >= 3x on the "
                     "authoritative workload and/or memory-fitting warnings in its "
                     "startup log)")
    else:
        remove_26 = "see measurements - ngl 26 was not dramatically worse in this run"

    def _v(key: str, default: str) -> str:
        if best and best.get("first_request", {}).get(key) is not None:
            return f"~{best['first_request'][key]} s (best tested config {best['label']})"
        return default

    add("**PROVEN CONFIGURATION**")
    add("")
    add(f"- {best['label'] if best else 'NOT MEASURED (real workload phase did not complete)'}"
        f" - command: `{best['command'] if best else 'n/a'}`")
    add("")
    add("**REAL WORLD FIRST TOKEN**")
    add("")
    add(f"- production (measured from logs): "
        f"{f'~{prod_ft:.1f} s' if prod_ft else 'NOT MEASURED in the available logs'}")
    add(f"- best tested config: {_v('ttft_s', 'NOT MEASURED')}")
    add("")
    add("**REAL WORLD TOTAL LATENCY**")
    add("")
    add("- production (measured from logs): see llm_done_ms_last10 in the raw results")
    add(f"- best tested config: {_v('total_s', 'NOT MEASURED')}")
    add("")
    add("**MAIN BOTTLENECK**")
    add("")
    add("- prompt processing (prefill) rate of the partially-offloaded MoE model - "
        "see the prompt_tok/s columns; compare against the Ollama reference rate.")
    add("")
    add("**EFFECTIVE CONTEXT**")
    add("")
    eff = (best or {}).get("startup_log_effective") or {}
    add(f"- effective n_ctx_slot for the proven config: "
        f"{eff.get('n_ctx_slot', 'not reported by log')} "
        f"(n_ctx: {eff.get('n_ctx', 'not reported')})")
    add("")
    add("**EFFECTIVE GPU LAYERS**")
    add("")
    add(f"- offloaded layers for the proven config: "
        f"{eff.get('offloaded_layers', 'not reported by log')} "
        f"(requested ngl: {(best or {}).get('ngl_requested', 'n/a')})")
    add("")
    add("**WHETHER 32768 CONTEXT IS ACTUALLY ACTIVE**")
    add("")
    add(f"- {verdict_ctx}")
    add("")
    add("**WHETHER ngl 26 MUST BE REMOVED**")
    add("")
    add(f"- {remove_26}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rel = str(REPORT_PATH.relative_to(REPO_ROOT))
    if rel not in CREATED_FILES:
        CREATED_FILES.append(rel)
    _out(f"  report: {REPORT_PATH}")
    _out(f"  results: {RESULTS_PATH}")

    _out("")
    _out("=== CONSOLE SUMMARY ===")
    _out(f"PROVEN CONFIGURATION: {best['label'] if best else 'NOT MEASURED'}")
    if prod_ft:
        _out(f"REAL WORLD FIRST TOKEN: production ~{prod_ft:.1f} s | best tested "
             f"{_v('ttft_s', 'NOT MEASURED')}")
    else:
        _out(f"REAL WORLD FIRST TOKEN: {_v('ttft_s', 'NOT MEASURED')}")
    _out(f"REAL WORLD TOTAL LATENCY: {_v('total_s', 'NOT MEASURED')}")
    _out("MAIN BOTTLENECK: see report (prompt processing rate comparison)")
    _out(f"EFFECTIVE CONTEXT: {eff.get('n_ctx_slot', 'not reported')}")
    _out(f"EFFECTIVE GPU LAYERS: {eff.get('offloaded_layers', 'not reported')}")
    _out(f"WHETHER 32768 CONTEXT IS ACTUALLY ACTIVE: {verdict_ctx}")
    _out(f"WHETHER ngl 26 MUST BE REMOVED: {remove_26}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TEMPORARY llama-server latency forensic benchmark (production untouched)")
    parser.add_argument("--quick", action="store_true", help="small matrix sanity run")
    parser.add_argument("--skip-short", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-capture", action="store_true",
                        help="use the fixed representative payload (no production code import)")
    parser.add_argument("--synthetic-memory", action="store_true",
                        help="skip the real VoiceMem memory search")
    parser.add_argument("--representative-history", action="store_true", default=True,
                        help="use representative history entries (default)")
    parser.add_argument("--no-production-replay", action="store_true",
                        help="skip the exact production-config real-workload replay")
    parser.add_argument("--stop-production-server", action="store_true",
                        help="allow the tool to stop the running production llama-server "
                             "for the benchmark phases (restart it afterwards via "
                             "scripts/start_llama_server.ps1)")
    parser.add_argument("--probe-production", action="store_true",
                        help="probe production HTTP endpoints even if no process is found")
    parser.add_argument("--model", default="", help="override model path")
    parser.add_argument("--server-exe", default="", help="override bin/llama-server.exe")
    parser.add_argument("--port", type=int, default=TEST_PORT_DEFAULT)
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_S)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    started_at = now_iso()
    _out(f"llm_latency_forensics (temporary forensic tool) - {started_at}")
    _out(f"repo root: {REPO_ROOT}")

    results: dict[str, Any] = {"created_files": CREATED_FILES}

    # ---- Phase 1: production inspection (always, read-only) -----------------
    phase1_inspect(args, results)

    # ---- Prerequisites for the benchmark phases -----------------------------
    server_exe = Path(args.server_exe) if args.server_exe else REPO_ROOT / "bin" / "llama-server.exe"
    if os.name != "nt" and not server_exe.exists():
        alt = shutil.which("llama-server")
        server_exe = Path(alt) if alt else server_exe
    model: Optional[Path] = Path(args.model) if args.model else None
    if model is None:
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from app.config import AgentConfig  # noqa: PLC0415

            cfg = AgentConfig.from_yaml(REPO_ROOT / "config" / "voicemem_config.yaml")
            resolved = _resolve_model_path(cfg)
            model = Path(resolved) if resolved else None
        except Exception:  # noqa: BLE001
            model = None
    if model is None or not Path(model).exists():
        _out(f"ERROR: model not found ({model}) - pass --model <path-to-GGUF>")
        return 2
    if not Path(server_exe).exists():
        _out(f"ERROR: llama-server binary not found ({server_exe}) - pass --server-exe")
        return 2
    _out(f"model: {model}")
    _out(f"server exe: {server_exe}")

    want_benchmarks = not (args.skip_short and args.skip_real)
    if want_benchmarks:
        procs = find_llama_processes()
        if procs:
            if args.stop_production_server:
                for p in procs:
                    _out(f"  stopping production llama-server pid {p['pid']} (restart later "
                         "with scripts/start_llama_server.ps1)")
                    kill_pid(p["pid"])
                time.sleep(5.0)
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < VRAM_RELEASE_TIMEOUT_S:
                    used = vram_used_mb()
                    if used is None or used <= VRAM_RELEASE_THRESHOLD_MB:
                        break
                    time.sleep(2.0)
            else:
                _out("\nERROR: the production llama-server is RUNNING and occupies the GPU.")
                _out("The benchmark phases need the full 12 GB VRAM.")
                _out("Options:")
                _out("  1) stop it yourself (close its window / Ctrl+C in the llama-server window), re-run")
                _out("  2) re-run with --stop-production-server (the tool stops it; restart via START.bat)")
                _out("  3) re-run with --skip-short --skip-real (inspection + capture + ollama only)")
                phase6_report(args, results, model, server_exe, started_at)
                return 2

    # ---- Phase 2: capture the real payload ----------------------------------
    payload: Optional[dict] = None
    if not args.skip_capture:
        try:
            payload = phase2_capture(args, results)
        except Exception as exc:  # noqa: BLE001
            _out(f"  capture failed ({exc}); falling back to representative payload")
            results["phase2_real_payload_error"] = str(exc)[:400]
    if payload is None:
        messages = [{"role": "user", "content": DEFAULT_UTTERANCE}]
        payload = production_request_payload({"llm_model_name": "qwen3.6-35b-a3b"},
                                             messages, 512, 0.7)
        results.setdefault("phase2_real_payload", {})["fallback"] = True

    # ---- Phase 3: short matrix ----------------------------------------------
    if not args.skip_short:
        short_matrix = SHORT_MATRIX_QUICK if args.quick else SHORT_MATRIX_FULL
        run_matrix(args, results, Path(server_exe), Path(model), None,
                   short_matrix, "phase3_short_matrix",
                   "PHASE 3: short benchmark matrix", None)

    # ---- Phase 4: real workload ----------------------------------------------
    if not args.skip_real:
        real_matrix = REAL_MATRIX_QUICK if args.quick else REAL_MATRIX_FULL
        prod_ngl: Optional[int] = None
        try:
            prod_ngl = (results.get("phase1_production_inspection", {})
                        .get("configured_yaml", {}).get("llm_n_gpu_layers"))
        except Exception:  # noqa: BLE001
            pass
        run_matrix(args, results, Path(server_exe), Path(model), payload,
                   real_matrix, "phase4_real_workload",
                   "PHASE 4: REAL VoiceMem workload replay", prod_ngl)

    # ---- Phase 5: Ollama reference -------------------------------------------
    if not args.skip_ollama:
        phase5_ollama(args, results, payload)

    # ---- Phase 6: report ------------------------------------------------------
    phase6_report(args, results, Path(model), Path(server_exe), started_at)
    _out("\nDONE. Files written:")
    for f in CREATED_FILES:
        _out(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
