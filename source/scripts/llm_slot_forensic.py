"""LLM slot / priority / cache forensic (v0.10.2, operator PART 7 + PART 10).

Runs a REAL behavioural measurement against a live llama.cpp llama-server
(the pinned b10717 build, the production startup flags) and records, with
timings and token counts:

  S1  server identity      /health, /props, /slots, /metrics availability
  S2  baseline TTFT        one streaming chat request, first-token latency
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
                           text)
  S6  cache after cancel    prompt A cancelled mid-generation, then re-sent
                           — how much of its prefix survives (the requeue
                           cost model)
  S7  production pattern    user chat -> extraction-shaped prompt (large
                           static prefix) -> user chat again: per-request
                           prompt work, showing how background work evicts
                           the user's chat prefix from the single slot

USAGE (target machine, product venv, server RUNNING):
    .venv\\Scripts\\python.exe scripts\\llm_slot_forensic.py
Optional env:
    LLAMA_SERVER_URL   base url without /v1 (default http://127.0.0.1:8080)
    LSF_MAX_TOKENS     generation cap for the probes (default 384; the
                       production extraction uses 1536 — a smaller cap
                       keeps the forensic quick; ratios are what matter)
    LSF_OUT            report directory (default logs/llm_slot_forensic_<ts>)

READ-ONLY with respect to the product: sends only its own probe requests
and GETs; writes only inside logs/. It never changes server state (no
/slots erase, no cache clear).

The verdict JSON deliberately reports FACTS (numbers), not conclusions —
the report.md cross-references these numbers against the fix design.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
MAX_TOKENS = int(os.environ.get("LSF_MAX_TOKENS", "384"))
# crash-safe forensics: write the report after EVERY phase, and allow
# running selected phases (LSF_PHASES="s3,s5") continuing an earlier
# report (LSF_MERGE=<path to report.json>) — a full run on slow hardware
# may not fit one console session.
PHASES = [p.strip() for p in os.environ.get(
    "LSF_PHASES", "s1,s2,s3,s4,s5,s6,s7").split(",") if p.strip()]
MERGE_FROM = os.environ.get("LSF_MERGE", "").strip()
REPORT: dict[str, Any] = {
    "schema": "llm-slot-forensic/1",
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

    Returns {"ttft_ms", "total_ms", "text_len", "finish_reason"}.
    """
    t0 = time.perf_counter()
    ttft_ms = None
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
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
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
                    parts.append(piece)
    return {
        "ttft_ms": round(ttft_ms or -1.0, 1),
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


def _prompt_tokens_processed(m_before: dict, m_after: dict) -> int | None:
    """Delta of the server's processed-prompt-token counter, if exposed.

    llama.cpp --metrics publishes `llamacpp:prompt_tokens_seconds_count`
    (total prompt tokens processed) and `prompt_tokens_total`-style
    counters depending on the build; accept any of the known names and
    return the delta, or None when the build exposes none (the timing
    evidence in the report still stands on its own).
    """
    names = (
        "llamacpp:prompt_tokens_seconds_count",
        "llamacpp:prompt_tokens_total",
        "prompt_tokens_total",
        "llamacpp:prompt_tokens",
    )
    for n in names:
        if n in m_before and n in m_after:
            d = m_after[n] - m_before[n]
            return int(d) if d >= 0 else None
    return None


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
    disconnect, the user TTFT collapses to its own prefill.
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
        "server_reprocessed_tokens": _prompt_tokens_processed(m1, m2),
        "wall_ms": r_a2.get("total_ms"),
    }
    rows["disjoint_prompt"] = {
        "usage_prompt_tokens": r_b.get("prompt_tokens"),
        "server_reprocessed_tokens": _prompt_tokens_processed(m2, m3),
        "wall_ms": r_b.get("total_ms"),
    }
    rows["first_prompt"] = {
        "usage_prompt_tokens": r_a.get("prompt_tokens"),
        "server_reprocessed_tokens": _prompt_tokens_processed(m0, m1),
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
        "resend_reprocessed_tokens": _prompt_tokens_processed(m0, m1),
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
    rows.append({"phase": "user-1", "reprocessed": _prompt_tokens_processed(m, m_),
                 "wall_ms": r1.get("total_ms")})
    m = m_
    r2 = _nonstreaming(client, _chat_payload(_extract_msg(), stream=False,
                                              max_tokens=32))
    m_ = _metrics_snapshot(client)
    rows.append({"phase": "extraction", "reprocessed": _prompt_tokens_processed(m, m_),
                 "wall_ms": r2.get("total_ms")})
    m = m_
    r3 = _nonstreaming(client, _chat_payload(_user_msg(2), stream=False,
                                             max_tokens=8))
    m_ = _metrics_snapshot(client)
    rows.append({"phase": "user-2-after-extraction",
                 "reprocessed": _prompt_tokens_processed(m, m_),
                 "wall_ms": r3.get("total_ms")})
    REPORT["s7_production_interleave"] = rows


def main() -> None:
    out_dir = _out_dir()
    print(f"[llm-slot-forensic] server: {DEFAULT_BASE}  out: {out_dir}")
    print(f"[llm-slot-forensic] phases: {','.join(PHASES)}")
    if MERGE_FROM and Path(MERGE_FROM).is_file():
        try:
            REPORT.update(json.loads(Path(MERGE_FROM).read_text(encoding="utf-8")))
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
    }
    with httpx.Client(base_url=DEFAULT_BASE) as client:
        for phase, (fn, label) in runners.items():
            if phase not in PHASES:
                continue
            fn(client)
            path = _save(out_dir)
            print(f"[{phase}] {label} done -> {path.name}")
    print(f"[llm-slot-forensic] report written: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
