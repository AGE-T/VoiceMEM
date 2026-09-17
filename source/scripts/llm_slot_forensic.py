"""LLM slot / priority / cache forensic (v0.10.3 forensic validation prep).

Runs measurements against a live llama.cpp llama-server (the pinned b10717
build, the production startup flags) AND correlates the product's own
web-server.log, recording with timings, token counts and timestamps:

  S1  server identity      /health, /props, /slots, /metrics availability
  S2  baseline TTFT         one streaming chat request, first-token latency
  S3  FIFO occupancy       user chat TTFT while a background-shaped request
                           runs (non-streaming), then with the background
                           request STREAMING and DISCONNECTED mid-generation
                           — the delta is the slot-yield behaviour of the
                           server (the PART 8 design input)
  S4  JSON streaming       response_format=json_object + stream=True: the
                           assembled text must parse as JSON (the vendor
                           cancellation transport path, llm_bg_gate)
  S5  LCP cache reuse      identical-prefix prompts vs disjoint prompts,
                           ACTUAL re-evaluated prompt-token counts measured
                           from /metrics counters (never inferred from log
                           text; the raw llamacpp counter deltas are stored
                           so nothing is inferred)
  S6  cache after cancel   prompt A cancelled mid-generation, then re-sent
                           — how much of its prefix survives (the requeue
                           cost model)
  S7  production pattern   user chat -> extraction-shaped prompt (large
                           static prefix) -> user chat again: per-request
                           prompt work, showing how background work evicts
                           the user's chat prefix from the single slot
  S8  product turn trace   OFFLINE correlation phase: parses the product's
                           logs/web-server.log (plus rotations .1-.3) and
                           rebuilds EVERY user turn with absolute timestamps
                           for: ASR final, memory retrieval, emotion
                           processing, LLM request start, first token,
                           stream completion, background cancellation and
                           gate acquire/release — then attributes each
                           slow turn (see THE FIVE-WAY SPLIT below).

TIMING SEMANTICS (v2, aligned with the product):
  ``ttft_ms``   = first NON-EMPTY content delta — exactly what the product
                  counts as its first token (pipeline._stream_and_speak /
                  web_server count only non-empty content deltas).
  ``first_data_ms`` = the first SSE data line of any kind (role chunk,
                  keep-alive): the raw server TTFB. A first_data_ms far
                  below ttft_ms is normal (role chunk); a ttft_ms of -1
                  with first_data_ms > 0 means the stream produced data
                  but never a content token (thinking-only / stall).

THE FIVE-WAY SPLIT — what distinguishes the five candidate causes of a
slow turn, and the evidence this report carries for each:

  (a) REAL FOREGROUND LLM LATENCY — the request itself is slow (prompt
      prefill, generation), nothing else on the slot.
      Evidence: s2 (idle baseline), s5/s7 (prefill cost of prompt sizes),
      and in s8: a slow turn whose llm_wait_ms is large while
      preprocessing_ms is small, NO background events overlap the turn,
      and no cancel fired. If a background chain ran shortly BEFORE the
      turn (bg_before non-empty) the likely mechanism is slot-cache
      eviction — quantified by s7 (the user prompt is then re-prefilled
      from scratch).

  (b) BACKGROUND SLOT CONTENTION — an in-flight background chain holds
      the single slot while the user waits.
      Evidence: s3 Run A quantifies the cost; in s8: a slow turn with a
      gate-cancel / vendor-cancel-requested / mid-chain-requeue event
      INSIDE the turn window ([user speech start .. first token]) — the
      chain WAS on the slot and was cancelled to yield it.

  (c) GATE WAITING — by design the gate never blocks a USER turn (arm()
      cancels background work; it does not wait). Gate waiting delays
      BACKGROUND starts only, so it can never directly explain a slow
      user turn. Its failure MODES are visible: background starting too
      early (contention, (b)) or background starved forever (s8 gate
      events: repeated "deferred" / "re-queued" with no "ingest
      started" — a memory-consolidation problem, not a latency one).

  (d) LLAMA SERVER QUEUEING — requests waiting inside llama-server's
      internal queue (parallel=1). Any in-flight request causes this —
      background chains are the (b) case; s8 shows whether vendor
      retrieval/other calls were plausible queue occupants via the
      preprocessing timeline; s3 Run B proves the disconnect transport
      actually frees the slot (if it did NOT, cancels would not help
      and (b) would persist after v0.10.2 — Run B is exactly that
      regression test).

  (e) MEMORY/EMOTION PREPROCESSING DELAY — time BEFORE the LLM request
      is issued (E5 retrieval, prosody analysis, emotion init, prompt
      build).
      Evidence: s8 per-turn preprocessing_ms (turn start -> llm start),
      memory_ms, the emotion markers; a turn where preprocessing_ms
      dominates while llm_wait_ms is normal is a preprocessing outlier.

USAGE (target machine, product venv, server RUNNING, product IDLE for
s2-s7 — the probes deliberately occupy the single slot):
    .venv\\Scripts\\python.exe scripts\\llm_slot_forensic.py
Offline correlation only (server/product state irrelevant):
    .venv\\Scripts\\python.exe scripts\\llm_slot_forensic.py   with
    LSF_PHASES=s8 (PowerShell: $env:LSF_PHASES="s8")
Optional env:
    LLAMA_SERVER_URL   base url without /v1 (default http://127.0.0.1:8080)
    LSF_MAX_TOKENS     generation cap for the probes (default 384; the
                       production extraction uses 1536 — a smaller cap
                       keeps the forensic quick; ratios are what matter)
    LSF_OUT            report directory (default logs/llm_slot_forensic_<ts>)
    LSF_PHASES         comma list (default s1,s2,s3,s4,s5,s6,s7,s8)
    LSF_MERGE          path to a previous report.json to continue
    LSF_PRODUCT_LOG    product log (default logs/web-server.log; the
                       rotations web-server.log.1/.2/.3 are picked up
                       automatically)
    LSF_SINCE          epoch seconds or ISO datetime: drop older events
    LSF_TURNS          detail the last N turns in s8 (default 30; 0 = all)
    LSF_OUTLIER_MS     first-token threshold for s8 outlier verdicts
                       (default 10000)
    LSF_PREPROC_MS     preprocessing threshold for s8 verdicts
                       (default 3000)
    LSF_BG_WINDOW_S    how long before a turn a background chain start
                       still counts as "recent" (default 45)

READ-ONLY with respect to the product: sends only its own probe requests
and GETs; writes only inside logs/. It never changes server state (no
/slots erase, no cache clear).

The verdict JSON deliberately reports FACTS (numbers), not conclusions —
the s8 per-turn "verdict" strings are documented, deterministic
attributions over those facts (priority: no-first-token > contention
> preprocessing > real-LLM-latency), and every raw event that fed the
verdict is included so the interpretation can be redone by hand.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

DEFAULT_BASE = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
MAX_TOKENS = int(os.environ.get("LSF_MAX_TOKENS", "384"))
# crash-safe forensics: write the report after EVERY phase, and allow
# running selected phases (LSF_PHASES="s3,s5") continuing an earlier
# report (LSF_MERGE=<path to report.json>) — a full run on slow hardware
# may not fit one console session.
PHASES = [p.strip() for p in os.environ.get(
    "LSF_PHASES", "s1,s2,s3,s4,s5,s6,s7,s8").split(",") if p.strip()]
MERGE_FROM = os.environ.get("LSF_MERGE", "").strip()
PRODUCT_LOG = os.environ.get("LSF_PRODUCT_LOG", "logs/web-server.log").strip()
DETAIL_TURNS = int(os.environ.get("LSF_TURNS", "30"))
OUTLIER_MS = float(os.environ.get("LSF_OUTLIER_MS", "10000"))
PREPROC_MS = float(os.environ.get("LSF_PREPROC_MS", "3000"))
BG_WINDOW_S = float(os.environ.get("LSF_BG_WINDOW_S", "45"))
SCHEMA = "llm-slot-forensic/2"
REPORT: dict[str, Any] = {
    "schema": SCHEMA,
    "base_url": DEFAULT_BASE,
    "max_tokens_probe": MAX_TOKENS,
}


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _out_dir() -> Path:
    out = Path(os.environ.get("LSF_OUT", f"logs/llm_slot_forensic_{_ts()}"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save(out_dir: Path) -> Path:
    REPORT["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = out_dir / "report.json"
    path.write_text(json.dumps(REPORT, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------ helpers


def _chat_payload(messages: list[dict], *, stream: bool, max_tokens: int,
                   json_mode: bool = False, temperature: float = 0.1,
                   extra: dict | None = None) -> dict:
    payload: dict[str, Any] = {
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if extra:
        payload.update(extra)
    return payload


def _stream_first_token_ms(client: httpx.Client, payload: dict) -> dict:
    """POST streaming; measure first-token latency + full wall time.

    Returns {"ttft_ms", "first_data_ms", "total_ms", "text_len",
    "finish_reason"}. ``ttft_ms`` is the first NON-EMPTY content delta —
    production parity: the pipeline counts its first token the same way
    (a bare role/keep-alive chunk must not masquerade as a token).
    ``first_data_ms`` is the first SSE data line of any kind (server TTFB);
    together they separate "server slow to answer" from "stream produced
    no content" (reasoning-only / stalled stream).
    """
    t0 = time.perf_counter()
    ttft_ms = None
    first_data_ms = None
    parts: list[str] = []
    finish = ""
    with client.stream("POST", "/v1/chat/completions", json=payload,
                       timeout=httpx.Timeout(600.0, connect=10.0)) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            if first_data_ms is None:
                first_data_ms = (time.perf_counter() - t0) * 1000.0
            try:
                obj = json.loads(body)
            except ValueError:
                continue
            choices = obj.get("choices") or []
            if choices:
                fr = choices[0].get("finish_reason")
                if fr:
                    finish = fr
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    parts.append(piece)
    return {
        "ttft_ms": round(ttft_ms or -1.0, 1),
        "first_data_ms": round(first_data_ms or -1.0, 1),
        "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "text_len": sum(len(p) for p in parts),
        "finish_reason": finish,
    }


def _nonstreaming(client: httpx.Client, payload: dict) -> dict:
    """POST non-streaming; wall time + content length."""
    t0 = time.perf_counter()
    resp = client.post("/v1/chat/completions", json=payload,
                        timeout=httpx.Timeout(600.0, connect=10.0))
    resp.raise_for_status()
    body = resp.json()
    choices = body.get("choices") or [{}]
    content = ((choices[0].get("message") or {}).get("content")) or ""
    usage = body.get("usage") or {}
    return {
        "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "text_len": len(content),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def _metrics_snapshot(client: httpx.Client) -> dict[str, float]:
    """GET /metrics -> {name: value} for the counter-style metrics."""
    try:
        resp = client.get("/metrics", timeout=10.0)
        if resp.status_code != 200:
            return {}
    except httpx.HTTPError:
        return {}
    out: dict[str, float] = {}
    for line in resp.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.rpartition(" ")
        try:
            out[name] = float(rest.strip())
        except ValueError:
            continue
    return out


def _metrics_delta(before: dict, after: dict) -> dict[str, float]:
    """Non-negative deltas of plain (non-bucket) counters between snapshots.

    Histogram BUCKET lines carry a ``{le="..."}` label and are dropped:
    they are per-bucket observations, not counters of interest.
    """
    out: dict[str, float] = {}
    for name, value in after.items():
        if "{" in name or name not in before:
            continue
        d = value - before[name]
        if d >= 0:
            out[name] = d
    return out


#: Explicit PROMPT-TOKEN-TOTAL counters only. The v1 script also accepted
#: ``llamacpp:prompt_tokens_seconds_count`` — that is the observation COUNT
#: of the prompt-eval-duration histogram (how many evals happened), NOT a
#: token count: using it would report "1" as the re-processed token count
#: of every request. v2 records the RAW counter deltas (facts) and derives
#: tokens only from these unambiguous names.
_TOKEN_TOTAL_NAMES = (
    "llamacpp:n_prompt_tokens_processed_total",
    "llamacpp:prompt_tokens_total",
    "llamacpp:n_prompt_tokens_processed",
    "prompt_tokens_total",
)


def _reprocessed_tokens(m_before: dict, m_after: dict) -> dict:
    """Derived re-processed prompt tokens + the raw llamacpp counter deltas.

    Returns {"counter": name|null, "tokens": int|null,
    "raw_llamacpp_counters": {...}} — the raw block is the evidence; the
    derived field names the counter it came from so nothing is inferred.
    """
    raw = _metrics_delta(m_before, m_after)
    name: Optional[str] = None
    tokens: Optional[int] = None
    for n in _TOKEN_TOTAL_NAMES:
        if n in raw:
            name, tokens = n, int(raw[n])
            break
    return {
        "counter": name,
        "tokens": tokens,
        "raw_llamacpp_counters": {
            k: raw[k] for k in sorted(raw) if k.startswith("llamacpp:")
        },
    }


def _slots_snapshot(client: httpx.Client) -> Any:
    """GET /slots (absent on some builds -> None)."""
    try:
        resp = client.get("/slots", timeout=10.0)
        if resp.status_code != 200:
            return {"http_status": resp.status_code}
        return resp.json()
    except httpx.HTTPError as exc:
        return {"error": str(exc)[:200]}


def _static_block(n_words: int, seed: str) -> str:
    """Deterministic filler text (~1 token per word) with a stable prefix."""
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliett", "kilo", "lima", "mike", "november",
        "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    ]
    out: list[str] = []
    for i in range(n_words):
        out.append(words[(i * 7 + 3) % len(words)])
    return f"{seed} " + " ".join(out)


# --------------------------------------------------- product log parsing (S8)

#: The logging format of app/web_server.py::_setup_logging is
#: ``%(asctime)s %(levelname)s %(name)s: %(message)s`` — asctime carries
#: millisecond precision (``2025-06-12 10:23:45,123``). All events come
#: from the same machine clock, so only RELATIVE time is meaningful; the
#: parser keeps both the epoch (for arithmetic) and the original string.
_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]\d{3})\s+([A-Z]+)\s+"
    r"([\w.]+):\s+(.*)$"
)

#: [chain] marker of the web session diagnostics (app.web_server._diag).
_CHAIN_PREFIX = "[chain] "

#: kind -> message prefix, FIRST MATCH WINS (order matters: specific
#: before general). These are the stable prefixes of the product's own
#: INFO logging — the S8 correlation backbone. Sources:
#:   app/web_server.py      _diag()/logger lines (ASR, turn stages)
#:   app/background_memory.py gate INFO lines (v0.10.3 forensic prep)
#:   voicemem/.../llm_bg_gate.py vendor cancel INFO lines
_EVENT_KINDS: tuple[tuple[str, str], ...] = (
    # -- background gate / chain ------------------------------------------ #
    ("gate_armed",           "background memory gate armed"),
    ("gate_cancel",          "background memory CANCELLED"),
    ("bg_cancel_requested",  "background LLM cancel requested"),
    ("bg_cancel_cleared",    "background LLM cancel cleared"),
    ("gate_released",        "background memory gate released"),
    ("gate_open",            "background memory gate open"),
    ("bg_deferred",          "background memory deferred"),
    ("bg_ingest_started",    "background memory ingest started"),
    ("bg_requeued",          "background memory cancelled mid-chain"),
    ("bg_requeue_pressure",  "background memory pair re-queued"),
    ("bg_ingest_failed",     "background memory ingest failed"),
    # -- microphone / ASR --------------------------------------------------- #
    ("speech_start",         "speech start (level"),
    ("speech_end",           "speech end ("),
    ("asr_flush_done",       "ASR flush done ("),
    ("asr_final",            "ASR final transcript:"),
    ("asr_dispatch",         "ASR turn dispatch"),
    # -- turn stages --------------------------------------------------------- #
    ("turn_start",           "turn received (source="),
    ("turn_crashed",         "turn CRASHED:"),
    ("memory_start",         "memory start"),
    ("memory_done",          "memory done ("),
    ("emotion_init_start",   "emotion start (init)"),
    ("emotion_init_done",    "emotion done ("),
    ("emotion_prosody_done", "emotion prosody done"),
    ("emotion_prosody_late", "emotion prosody pending"),
    ("llm_start",            "llm start ("),
    ("llm_first_token",      "llm first token ("),
    ("llm_done",             "llm done ("),
    ("llm_failed",           "llm failed ("),
    ("tts_start",            "tts start ("),
    ("first_audio",          "first TTS audio sent"),
    ("tts_done",             "tts done"),
    ("answer_done",          "answer done:"),
)

#: Gate/background kinds — the events that can relate to slot occupancy.
_BG_KINDS = frozenset({
    "gate_armed", "gate_cancel", "bg_cancel_requested", "bg_cancel_cleared",
    "gate_released", "gate_open", "bg_deferred", "bg_ingest_started",
    "bg_requeued", "bg_requeue_pressure", "bg_ingest_failed",
})


def parse_product_log_line(line: str) -> Optional[dict]:
    """One web-server.log line -> {"ts", "ts_str", "level", "logger", "msg"}.

    Pure function; returns None for anything that does not match the
    product's logging format (uvicorn noise, tracebacks, foreign lines).
    """
    m = _LOG_LINE_RE.match(line.rstrip("\r\n"))
    if m is None:
        return None
    ts_str, level, logger_name, msg = m.groups()
    try:
        ts = datetime.strptime(ts_str.replace(",", "."),
                               "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        return None
    return {"ts": ts, "ts_str": ts_str, "level": level,
            "logger": logger_name, "msg": msg}


def classify_event(msg: str) -> str:
    """Map a log message to its event kind ("" when unclassified)."""
    if msg.startswith(_CHAIN_PREFIX):
        msg = msg[len(_CHAIN_PREFIX):]
    for kind, prefix in _EVENT_KINDS:
        if msg.startswith(prefix):
            return kind
    return ""


def collect_product_log_files(base: str | Path) -> list[Path]:
    """The product log plus its rotations, OLDEST FIRST.

    RotatingFileHandler(backupCount=3) yields web-server.log.3 (oldest)
    ... web-server.log (newest); events are also re-sorted by timestamp,
    so this ordering only breaks ties deterministically.
    """
    base_path = Path(base)
    candidates = [base_path.with_name(base_path.name + f".{i}")
                  for i in (3, 2, 1)]
    candidates.append(base_path)
    return [p for p in candidates if p.is_file()]


def load_product_events(paths: list[Path],
                        since_epoch: Optional[float] = None) -> list[dict]:
    """Parse + classify every product log line into a time-sorted list.

    Each event: {"ts", "ts_str", "kind", "msg", "file", "line"}. Lines
    that match the log format but no event kind are dropped from the
    event list (counted by the caller if desired).
    """
    events: list[dict] = []
    for order, path in enumerate(paths):
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    parsed = parse_product_log_line(line)
                    if parsed is None:
                        continue
                    kind = classify_event(parsed["msg"])
                    if not kind:
                        continue
                    if since_epoch is not None and parsed["ts"] < since_epoch:
                        continue
                    events.append({
                        "ts": parsed["ts"], "ts_str": parsed["ts_str"],
                        "kind": kind, "msg": parsed["msg"][:160],
                        "file": str(path), "file_order": order,
                        "line": lineno,
                    })
        except OSError:
            continue
    events.sort(key=lambda e: (e["ts"], e["file_order"], e["line"]))
    return events


def _pct(values_sorted: list[float], q: float) -> Optional[float]:
    """Percentile of a pre-sorted list (None when empty)."""
    if not values_sorted:
        return None
    idx = min(len(values_sorted) - 1, max(0, int(round(q * (len(values_sorted) - 1)))))
    return values_sorted[idx]


def _nearest_before(events: list[dict], upto: int, kinds: tuple[str, ...],
                    stop_kinds: tuple[str, ...] = ("turn_start",)
                    ) -> Optional[dict]:
    """Scan backwards from index ``upto``-1 for the last event of ``kinds``.

    Stops at the first event of ``stop_kinds`` (e.g. the previous turn's
    start) so a typed turn never inherits the previous utterance's ASR
    trail. Returns None when nothing matched.
    """
    for i in range(upto - 1, -1, -1):
        ev = events[i]
        if ev["kind"] in stop_kinds and ev["kind"] not in kinds:
            return None
        if ev["kind"] in kinds:
            return ev
    return None


def _fmt_event(ev: dict) -> dict:
    """Compact JSON-friendly event for the report."""
    return {"ts": ev["ts_str"], "kind": ev["kind"], "msg": ev["msg"]}


def _turn_verdict(gaps: dict, bg_during: list[dict], bg_before: list[dict],
                  llm_started: bool) -> str:
    """Deterministic attribution of one turn (documented priority order).

    NO_LLM_REQUEST / NO_FIRST_TOKEN beat everything (the turn died
    before or inside the LLM stage); CONTENTION beats PREPROCESSING (a
    cancelled chain delays the request even when the preprocessing
    numbers are also large); REAL_LLM_LATENCY is the residual — with the
    bg_before evidence attached so slot-cache eviction (see s7) can be
    distinguished from a plain slow prompt.
    """
    llm_wait = gaps.get("llm_wait_ms")
    first_token = gaps.get("first_token_ms")
    if llm_wait is None:
        return "NO_FIRST_TOKEN" if llm_started else "NO_LLM_REQUEST"
    slow = (llm_wait >= OUTLIER_MS) or \
           (first_token is not None and first_token >= OUTLIER_MS)
    if not slow:
        return "normal"
    cancel_kinds = ("gate_cancel", "bg_cancel_requested", "bg_requeued")
    if any(e["kind"] in cancel_kinds for e in bg_during):
        return "CONTENTION_CONFIRMED"
    if bg_before:
        return "CONTENTION_SUSPECT_OR_CACHE_EVICTED"
    pre = gaps.get("preprocessing_ms")
    if pre is not None and pre >= PREPROC_MS:
        return "PREPROCESSING_DOMINATED"
    return "REAL_LLM_LATENCY"


def rebuild_turns(events: list[dict], *, keep_turns: int = 30) -> dict:
    """Reconstruct per-turn timelines + the whole-session gate timeline.

    Turn boundaries are ``turn received`` lines; the product cancels any
    in-flight turn BEFORE starting a new one (web_server._start_turn),
    so turn trails never interleave and an abandoned turn simply ends
    without ``answer done``.
    """
    starts = [i for i, ev in enumerate(events) if ev["kind"] == "turn_start"]
    turns: list[dict] = []
    for n, si in enumerate(starts):
        ei = starts[n + 1] if n + 1 < len(starts) else len(events)
        window = events[si:ei]
        start_ev = events[si]

        def first(kind: str) -> Optional[dict]:
            for ev in window:
                if ev["kind"] == kind:
                    return ev
            return None

        stages = {kind: first(kind) for kind in (
            "memory_start", "memory_done", "emotion_init_start",
            "emotion_init_done", "emotion_prosody_done",
            "emotion_prosody_late", "llm_start", "llm_first_token",
            "llm_done", "llm_failed", "tts_start", "first_audio", "tts_done",
            "answer_done", "turn_crashed",
        )}
        # the turn's own utterance trail: nearest ASR final before the
        # turn start, not crossing the previous turn's start (typed turns
        # have none). speech_start bounds the contention window (arm()
        # fires on the FIRST VAD speech frame, before the turn exists).
        asr_final = _nearest_before(events, si, ("asr_final",))
        speech_start = _nearest_before(events, si, ("speech_start",))
        source = "asr"
        m = re.search(r"source=(\w+)", start_ev["msg"])
        if m:
            source = m.group(1)

        def gap(a: Optional[dict], b: Optional[dict]) -> Optional[float]:
            if a is None or b is None:
                return None
            return round((b["ts"] - a["ts"]) * 1000.0, 1)

        turn_ts = start_ev["ts"]
        ctx_start = speech_start["ts"] if speech_start is not None else turn_ts
        llm_ft = stages["llm_first_token"]
        window_end = llm_ft["ts"] if llm_ft is not None else \
            (window[-1]["ts"] if window else turn_ts)
        # bg_during is TIME-bounded over the WHOLE event list, not the
        # turn's index slice: arm() fires on the FIRST VAD speech frame —
        # the cancel lines land in the PREVIOUS turn's slice (between its
        # answer_done and this turn's start) but belong to THIS turn.
        bg_during = [_fmt_event(e) for e in events
                     if e["kind"] in _BG_KINDS
                     and ctx_start <= e["ts"] <= window_end]
        bg_before = [_fmt_event(e) for e in events
                     if e["kind"] in ("bg_ingest_started", "gate_open")
                     and turn_ts - BG_WINDOW_S <= e["ts"] < ctx_start]

        gaps = {
            "asr_to_turn_ms": gap(asr_final, start_ev),
            "memory_ms": gap(stages["memory_start"], stages["memory_done"]),
            "preprocessing_ms": gap(start_ev, stages["llm_start"]),
            "llm_wait_ms": gap(stages["llm_start"], llm_ft),
            "first_token_ms": gap(start_ev, llm_ft),
            "llm_stream_ms": gap(llm_ft, stages["llm_done"]),
            "total_ms": gap(start_ev, stages["answer_done"]),
        }
        turns.append({
            "idx": n + 1,
            "ts": start_ev["ts_str"],
            "source": source,
            "stages": {
                **({"speech_start": speech_start["ts_str"]}
                   if speech_start is not None else {}),
                **({"asr_final": asr_final["ts_str"]}
                   if asr_final is not None else {}),
                **{k: (v["ts_str"] if v is not None else None)
                   for k, v in stages.items() if v is not None},
            },
            "gaps_ms": gaps,
            "bg_during": bg_during,
            "bg_before": bg_before,
            "complete": stages["answer_done"] is not None,
            "verdict": _turn_verdict(
                gaps, bg_during, bg_before,
                llm_started=stages["llm_start"] is not None,
            ),
        })

    # session-level gate timeline (sparse: a handful of lines per turn)
    gate_events = [_fmt_event(e) for e in events if e["kind"] in _BG_KINDS]

    def values(key: str) -> list[float]:
        vs = [t["gaps_ms"][key] for t in turns
              if t["gaps_ms"].get(key) is not None]
        vs.sort()
        return vs

    def dist(key: str) -> dict:
        vs = values(key)
        return {
            "n": len(vs),
            "min": vs[0] if vs else None,
            "median": _pct(vs, 0.5),
            "p90": _pct(vs, 0.9),
            "max": vs[-1] if vs else None,
        }

    outliers = [t for t in turns if t["verdict"] != "normal"]
    detail = turns[-keep_turns:] if keep_turns > 0 else turns
    return {
        "turns_total": len(turns),
        "turns_complete": sum(1 for t in turns if t["complete"]),
        "turns_abandoned": sum(1 for t in turns if not t["complete"]),
        "gate_deferred_total": sum(1 for e in gate_events
                                   if e["kind"] == "bg_deferred"),
        "gate_requeue_total": sum(1 for e in gate_events
                                  if e["kind"] in ("bg_requeued",
                                                   "bg_requeue_pressure")),
        "distributions_ms": {
            "llm_wait": dist("llm_wait_ms"),
            "preprocessing": dist("preprocessing_ms"),
            "first_token": dist("first_token_ms"),
            "memory": dist("memory_ms"),
            "total": dist("total_ms"),
        },
        "outliers": [{
            "idx": t["idx"], "ts": t["ts"], "source": t["source"],
            "gaps_ms": t["gaps_ms"], "verdict": t["verdict"],
            "bg_during": t["bg_during"], "bg_before": t["bg_before"],
        } for t in outliers[:200]],
        "turns_detail": detail,
        "gate_events": gate_events[:2000],
    }


def _parse_since(raw: str) -> Optional[float]:
    """LSF_SINCE -> epoch seconds (float epoch or ISO datetime)."""
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def s8_product_trace(client: httpx.Client) -> None:
    """OFFLINE: correlate the product's own log into per-turn timelines.

    No probe requests are sent; works whether or not the server (or even
    the product) is running — it reads logs/web-server.log* only.
    """
    files = collect_product_log_files(PRODUCT_LOG)
    payload: dict[str, Any] = {
        "product_log": PRODUCT_LOG,
        "note": (
            "timestamps are the product machine's LOCAL clock; only "
            "relative arithmetic is meaningful. verdict priority: "
            "NO_LLM_REQUEST / NO_FIRST_TOKEN > CONTENTION_CONFIRMED > "
            "CONTENTION_SUSPECT_OR_CACHE_EVICTED > PREPROCESSING_DOMINATED "
            "> REAL_LLM_LATENCY"
        ),
    }
    if not files:
        payload["error"] = (
            f"no product log found at {PRODUCT_LOG} (run from the product "
            "root, or set LSF_PRODUCT_LOG)"
        )
        REPORT["s8_product_trace"] = payload
        return
    since = _parse_since(os.environ.get("LSF_SINCE", "").strip())
    events = load_product_events(files, since_epoch=since)
    files_meta = []
    for p in files:
        try:
            n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
        except OSError:
            n = 0
        files_meta.append({"path": str(p), "lines": n})
    payload["files"] = files_meta
    payload["since_epoch"] = since
    payload["events_matched"] = len(events)
    payload["window"] = (
        {"first": events[0]["ts_str"], "last": events[-1]["ts_str"]}
        if events else None
    )
    payload.update(rebuild_turns(events, keep_turns=DETAIL_TURNS))
    REPORT["s8_product_trace"] = payload


# ------------------------------------------------------------------ phases


def s1_identity(client: httpx.Client) -> None:
    out: dict[str, Any] = {}
    for path in ("/health", "/props", "/slots", "/metrics"):
        try:
            r = client.get(path, timeout=10.0)
            body = r.text[:2000] if path != "/metrics" else f"({len(r.text)} bytes)"
            out[path] = {"status": r.status_code, "body_head": body}
        except httpx.HTTPError as exc:
            out[path] = {"error": str(exc)[:200]}
    REPORT["s1_server"] = out


def s2_baseline(client: httpx.Client) -> None:
    msgs = [{"role": "user", "content": "Say exactly: ok"}]
    r = _stream_first_token_ms(
        client, _chat_payload(msgs, stream=True, max_tokens=16))
    REPORT["s2_baseline"] = r


def s3_fifo_and_disconnect(client: httpx.Client) -> None:
    """The PART 7/8 core: what does a user chat wait behind?

    Run A (non-streaming background): launch a long non-streaming request
    in a thread; 400 ms later start the user streaming chat. The user
    TTFT includes whatever the server makes it wait behind the background
    request.

    Run B (streaming background + disconnect): same, but the background
    request is streaming and we close it after ~1.5 s (exactly what
    llm_bg_gate does on cancel). If llama-server frees the slot on client
    disconnect, the user TTFT collapses to its own prefill. The
    disconnect fires on the first SSE line at/after 1.5 s — when the
    background prompt's prefill alone exceeds 1.5 s the first line (and
    so the disconnect) arrives later; first_line_at_s records it so the
    run is never misread.
    """
    import threading

    bg_msgs = [{"role": "user", "content":
                _static_block(600, "background-shaped extraction prompt")
                + "\n\nList the words above, one per line, numbered."}]

    def _bg_nonstreaming() -> None:
        try:
            _nonstreaming(client, _chat_payload(bg_msgs, stream=False,
                                                max_tokens=MAX_TOKENS))
        except Exception:  # noqa: BLE001 - background probe only
            pass

    # --- Run A: non-streaming background, no cancel
    slots_before = _slots_snapshot(client)
    t = threading.Thread(target=_bg_nonstreaming, daemon=True)
    t.start()
    time.sleep(0.4)
    user = _stream_first_token_ms(
        client, _chat_payload([{"role": "user", "content": "Say exactly: ready"}],
                              stream=True, max_tokens=16))
    t.join(timeout=600.0)
    REPORT["s3_fifo"] = {
        "background": "non-streaming, max_tokens=%d" % MAX_TOKENS,
        "user_chat": user,
        "slots_before": slots_before,
    }

    # cooldown: let the server drain
    time.sleep(1.0)

    # --- Run B: streaming background + client disconnect mid-generation
    slots_before_b = _slots_snapshot(client)
    t_disconnect_at = {}

    def _bg_stream_disconnect() -> None:
        try:
            t0 = time.perf_counter()
            with client.stream(
                "POST", "/v1/chat/completions",
                json=_chat_payload(bg_msgs, stream=True, max_tokens=MAX_TOKENS),
                timeout=httpx.Timeout(600.0, connect=10.0),
            ) as resp:
                resp.raise_for_status()
                got = 0
                for line in resp.iter_lines():
                    got += 1
                    if got == 1:
                        t_disconnect_at["first_line_at_s"] = round(
                            time.perf_counter() - t0, 2)
                    if (time.perf_counter() - t0) >= 1.5:
                        # hard disconnect mid-stream (what llm_bg_gate does)
                        t_disconnect_at["at_s"] = round(
                            time.perf_counter() - t0, 2)
                        t_disconnect_at["sse_lines"] = got
                        return  # leaving the with-block closes the response
        except Exception:  # noqa: BLE001
            pass

    t = threading.Thread(target=_bg_stream_disconnect, daemon=True)
    t.start()
    time.sleep(2.0)  # background is generating, already disconnected at ~1.5 s
    user_b = _stream_first_token_ms(
        client, _chat_payload([{"role": "user", "content": "Say exactly: go"}],
                              stream=True, max_tokens=16))
    t.join(timeout=600.0)
    time.sleep(1.0)
    slots_after_b = _slots_snapshot(client)
    REPORT["s3_disconnect_yield"] = {
        "background": "streaming, disconnected at ~1.5 s",
        "disconnect": t_disconnect_at,
        "user_chat": user_b,
        "slots_after": slots_after_b,
        "verdict_hint": (
            "user_ttft_after_disconnect << user_ttft_behind_nonstreaming "
            "=> the server FREES the slot on streaming client disconnect "
            "(the llm_bg_gate cancellation transport is sound)"
        ),
    }


def s4_json_streaming(client: httpx.Client) -> None:
    """response_format=json_object + stream=True assembles to valid JSON."""
    msgs = [{"role": "user", "content":
             'Return a JSON object {"ok": true, "n": 3} and nothing else.'}]
    t0 = time.perf_counter()
    parts: list[str] = []
    with client.stream("POST", "/v1/chat/completions",
                       json=_chat_payload(msgs, stream=True, max_tokens=64,
                                          json_mode=True),
                       timeout=httpx.Timeout(120.0, connect=10.0)) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            try:
                obj = json.loads(body)
            except ValueError:
                continue
            choices = obj.get("choices") or []
            if choices:
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    parts.append(piece)
    text = "".join(parts)
    ok = False
    try:
        ok = isinstance(json.loads(text), dict)
    except ValueError:
        ok = False
    REPORT["s4_json_streaming"] = {
        "assembles_to_json_object": ok,
        "text": text[:300],
        "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


def s5_lcp_cache(client: httpx.Client) -> None:
    """PART 10: ACTUAL re-evaluated prompt tokens, from /metrics counters."""
    prefix = _static_block(1500, "stable shared prefix for LCP measurement")

    def _msgs(tail: str) -> list[dict]:
        return [{"role": "user",
                 "content": f"{prefix}\n\nQuestion {tail}: answer with the single word: {tail}"}]

    rows: dict[str, Any] = {}
    # A: cold prompt (may reuse whatever the slot already had)
    m0 = _metrics_snapshot(client)
    r_a = _nonstreaming(client, _chat_payload(_msgs("alpha"), stream=False,
                                               max_tokens=8))
    m1 = _metrics_snapshot(client)
    # A': SAME prefix, different tail -> only the tail should be evaluated
    r_a2 = _nonstreaming(client, _chat_payload(_msgs("bravo"), stream=False,
                                                max_tokens=8))
    m2 = _metrics_snapshot(client)
    # B: disjoint prompt -> full evaluation
    r_b = _nonstreaming(
        client,
        _chat_payload([{"role": "user",
                        "content": _static_block(1500, "disjoint other topic")
                        + "\n\nAnswer with the single word: zulu"}],
                      stream=False, max_tokens=8))
    m3 = _metrics_snapshot(client)
    rows["same_prefix_again"] = {
        "usage_prompt_tokens": r_a2.get("prompt_tokens"),
        "reprocessed": _reprocessed_tokens(m1, m2),
        "wall_ms": r_a2.get("total_ms"),
    }
    rows["disjoint_prompt"] = {
        "usage_prompt_tokens": r_b.get("prompt_tokens"),
        "reprocessed": _reprocessed_tokens(m2, m3),
        "wall_ms": r_b.get("total_ms"),
    }
    rows["first_prompt"] = {
        "usage_prompt_tokens": r_a.get("prompt_tokens"),
        "reprocessed": _reprocessed_tokens(m0, m1),
        "wall_ms": r_a.get("total_ms"),
    }
    REPORT["s5_lcp_cache"] = rows


def s6_cache_after_cancel(client: httpx.Client) -> None:
    """Cancel a streaming generation mid-way, then re-send the same prompt."""
    prefix = _static_block(1200, "cache survival probe prefix")
    msgs = [{"role": "user", "content":
             f"{prefix}\n\nCount from 1 to 50, one number per line."}]

    # start streaming, cancel after ~1.2 s
    t0 = time.perf_counter()
    with client.stream("POST", "/v1/chat/completions",
                       json=_chat_payload(msgs, stream=True, max_tokens=256),
                       timeout=httpx.Timeout(600.0, connect=10.0)) as resp:
        resp.raise_for_status()
        for _line in resp.iter_lines():
            if (time.perf_counter() - t0) >= 1.2:
                break  # leaving the with-block closes mid-generation
    cancelled_at_s = round(time.perf_counter() - t0, 2)

    m0 = _metrics_snapshot(client)
    r = _nonstreaming(client, _chat_payload(msgs, stream=False, max_tokens=8))
    m1 = _metrics_snapshot(client)
    REPORT["s6_cache_after_cancel"] = {
        "cancelled_generation_after_s": cancelled_at_s,
        "resend_reprocessed": _reprocessed_tokens(m0, m1),
        "resend_usage_prompt_tokens": r.get("prompt_tokens"),
        "resend_wall_ms": r.get("total_ms"),
    }


def s7_production_interleave(client: httpx.Client) -> None:
    """user chat -> extraction-shaped -> user chat (same prefix family per
    phase), measuring per-request reprocessed tokens: shows the cache
    eviction cost of interleaving background and user prompts."""
    user_prefix = _static_block(400, "user conversation system context")
    extract_prefix = _static_block(1500, "extraction static schema block")

    def _user_msg(i: int) -> list[dict]:
        return [{"role": "user", "content":
                 f"{user_prefix}\n\nUser turn {i}. Reply with the single word: fine{i}"}]

    def _extract_msg() -> list[dict]:
        return [{"role": "user", "content":
                 f"{extract_prefix}\n\nExtract facts from: the sky is blue."}]

    rows: list[dict] = []
    m = _metrics_snapshot(client)
    r1 = _nonstreaming(client, _chat_payload(_user_msg(1), stream=False,
                                             max_tokens=8))
    m_ = _metrics_snapshot(client)
    rows.append({"phase": "user-1", "reprocessed": _reprocessed_tokens(m, m_),
                 "wall_ms": r1.get("total_ms")})
    m = m_
    r2 = _nonstreaming(client, _chat_payload(_extract_msg(), stream=False,
                                              max_tokens=32))
    m_ = _metrics_snapshot(client)
    rows.append({"phase": "extraction", "reprocessed": _reprocessed_tokens(m, m_),
                 "wall_ms": r2.get("total_ms")})
    m = m_
    r3 = _nonstreaming(client, _chat_payload(_user_msg(2), stream=False,
                                             max_tokens=8))
    m_ = _metrics_snapshot(client)
    rows.append({"phase": "user-2-after-extraction",
                 "reprocessed": _reprocessed_tokens(m, m_),
                 "wall_ms": r3.get("total_ms")})
    REPORT["s7_production_interleave"] = rows


def main() -> None:
    out_dir = _out_dir()
    print(f"[llm-slot-forensic] server: {DEFAULT_BASE}  out: {out_dir}")
    print(f"[llm-slot-forensic] phases: {','.join(PHASES)}")
    if MERGE_FROM and Path(MERGE_FROM).is_file():
        try:
            REPORT.update(json.loads(Path(MERGE_FROM).read_text(encoding="utf-8")))
            # the merge must never downgrade the schema of a fresh run
            REPORT["schema"] = SCHEMA
            print(f"[llm-slot-forensic] merged previous report: {MERGE_FROM}")
        except (ValueError, OSError):
            print("[llm-slot-forensic] LSF_MERGE unreadable - starting fresh")
    runners = {
        "s1": (s1_identity, "identity"),
        "s2": (s2_baseline, "baseline TTFT"),
        "s3": (s3_fifo_and_disconnect, "FIFO vs disconnect-yield"),
        "s4": (s4_json_streaming, "JSON+stream"),
        "s5": (s5_lcp_cache, "LCP cache"),
        "s6": (s6_cache_after_cancel, "cache-after-cancel"),
        "s7": (s7_production_interleave, "production interleave"),
        "s8": (s8_product_trace, "product turn trace (offline)"),
    }
    with httpx.Client(base_url=DEFAULT_BASE) as client:
        m_run_start = _metrics_snapshot(client)
        for phase, (fn, label) in runners.items():
            if phase not in PHASES:
                continue
            t_start = time.time()
            try:
                fn(client)
            except Exception as exc:  # noqa: BLE001 - crash-safe forensics
                # one phase failing (e.g. server down) must not lose the
                # others; the failure itself is forensic evidence.
                REPORT[f"{phase}_error"] = f"{type(exc).__name__}: {exc}"[:500]
            REPORT.setdefault("phase_windows", {})[phase] = {
                "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                "start_epoch": round(t_start, 3),
                "end_epoch": round(time.time(), 3),
            }
            path = _save(out_dir)
            print(f"[{phase}] {label} done -> {path.name}")
        # what ELSE hit the server while this script ran (sanity check:
        # probe counts should account for the whole llamacpp delta)
        m_run_end = _metrics_snapshot(client)
        REPORT["run_window"] = {
            "llamacpp_counter_delta": {
                k: v for k, v in _metrics_delta(m_run_start, m_run_end).items()
                if k.startswith("llamacpp:")
            },
        }
    print(f"[llm-slot-forensic] report written: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
