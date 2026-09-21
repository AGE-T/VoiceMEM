#!/usr/bin/env python3
"""A/B benchmark harness — llama.cpp b10717 (pinned production) vs b11073
(experimental) for the VoiceMem v0.10.7 workload. Sandbox side.

Runs ONE cell per invocation (the 4 GB sandbox can hold exactly ONE
llama-server at a time — proven by an OOM-kill during setup), boots the
cell's server as a direct subprocess, waits for /health, runs the probe
set through the REAL production client (app/llm.py LlmClient + the REAL
system prompt from app/teacher_persona.py), then kills the server and
writes a JSON result.

Cells (the 2x2 version x configuration matrix):
  A1: b10717, sandbox production-proxy flags (ngl 0, ctx 8192)
  B1: b11073, IDENTICAL flags to A1            -> pure llama.cpp version effect
  B2: b11073, operator baseline adapted (ngl 99, ctx 16000->16128, explicit threads)
  A2: b10717, operator baseline adapted         -> config effect on the old build

Probes per cell (fixed VoiceMem-shaped workload on every cell):
  warmup   1 streaming request (not measured)
  normal   8 streaming user-turn probes   -> first SSE line, TTFT, tok/s,
             total, streaming integrity ([DONE] + assembled content)
  extract  4 non-streaming JSON probes    -> prefill (prompt tokens +
             cached_tokens), total, JSON validity, gen tok/s
  contend  4 queue-behind-background      -> user TTFT while a non-streaming
             background extraction leg holds the single slot (the 30-42 s
             outlier mechanism class); on alternating trials the background
             connection is DISCONNECTED mid-generation through the async
             httpx cancel path (= the production v0.10.2 cancellation
             transport) and the slot-free latency is measured
  cancel   3 barge-in disconnects         -> user streaming turn dropped
             right after the first content token; slot-free latency +
             immediate follow-up usability
  cache    2 identical-prompt runs        -> prefix re-evaluation cost
             (prompt_tokens_details.cached_tokens)
  sse      1 raw-SSE integrity capture    -> line-kind census of a full
             stream (role chunk / content chunks / finish chunk / [DONE])

Everything is measured through the SAME client code path the product uses.
No application file is modified by this harness.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/z/my-project/voicemem-agent")
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import AgentConfig  # noqa: E402
from app.llm import LlmClient  # noqa: E402
from app.teacher_persona import build_system_prompt  # noqa: E402

BIN = {"b10717": "/tmp/llama/llama-b10717", "b11073": "/tmp/llama/llama-b11073"}
MODEL = "/tmp/llama/qwen3-1.7b-q4km.gguf"

CELLS = {
    "A1": {"bin": "b10717", "port": 8080, "ngl": 0, "ctx": 8192, "threads": os.cpu_count() or 2,
           "label": "b10717 @ sandbox production-proxy flags (ngl 0, ctx 8192)"},
    "B1": {"bin": "b11073", "port": 8080, "ngl": 0, "ctx": 8192, "threads": os.cpu_count() or 2,
           "label": "b11073 @ IDENTICAL flags to A1 (pure version effect)"},
    "B2": {"bin": "b11073", "port": 8081, "ngl": 99, "ctx": 16000, "threads": os.cpu_count() or 2,
           "label": "b11073 @ operator baseline adapted (ngl 99, ctx 16000, -t explicit)"},
    "A2": {"bin": "b10717", "port": 8081, "ngl": 99, "ctx": 16000, "threads": os.cpu_count() or 2,
           "label": "b10717 @ operator baseline adapted"},
}

# --- the fixed VoiceMem-shaped workload -------------------------------------
MEMORY_BLOCK = (
    "A felhasználó neve Imre. Budapesten él, egy kis csapattal dolgozik egy "
    "hangalapú jegyzetprojekten. Kedvenc kávéja a hosszú kávé tejjel. Hetente "
    "kétszer futni szokott a Margit-szigeten. A lánya Zsófia jövő héten vizsgázik "
    "biológiából, ami miatt most kicsit feszült. Múlt kedden beszéltek a béna "
    "meetingről, ahol a projektfázisok összekeveredtek. Szereti a rövid, őszinte "
    "válaszokat és nem kedveli a körülményeskedést."
)
USER_TURNS = [
    "Holnap lesz a meeting a csapattal, mit vigyek magammal?",
    "Emlékszel, mit mondtam a múlt heti projektfázisokról?",
    "Hogy áll Zsófi vizsgájával kapcsolatban a tervem?",
    "Mit ajánlasz reggelire futás előtt?",
    "A hosszú kávé tejjel pont jó lesz most, nem?",
    "Mi volt a legfontosabb tegnapi döntés a projektben?",
    "Kell ma még valamit előkészítenem a hétvégére?",
    "Hogyan köszönjek majd el a csapattól a meeting végén?",
]
HISTORY = [
    {"role": "user", "content": "Sziasztok, itt Imre vagyok."},
    {"role": "assistant", "content": "Szia Imre! Hallgatom, mi újság?"},
]
EXTRACTION_INSTRUCTION = (
    "Extract the memory-worthy facts from the following conversation turn as a "
    "JSON object with keys \"facts\" (array of short strings), \"language\" "
    "(\"hu\" or \"en\"), \"sentiment\" (one word). Conversation turn: "
    "User said: \"Holnap lesz a meeting a csapattal, és Zsófi vizsgájára is "
    "kell készülni, szóval elfoglalt hét lesz.\" Assistant answered briefly. "
    "Context: the user is Imre from Budapest, works on a voice-notes project, "
    "daughter Zsofia has a biology exam next week. Return only JSON."
)
THINKING_KWARGS = {"chat_template_kwargs": {"enable_thinking": False},
                   "reasoning_effort": "none"}


def build_messages(user_text: str) -> list[dict]:
    system_prompt = build_system_prompt(MEMORY_BLOCK)
    return [{"role": "system", "content": system_prompt}, *HISTORY,
            {"role": "user", "content": user_text}]


def metrics_snapshot(port: int) -> dict:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5)
        out = {}
        for line in r.text.splitlines():
            if line.startswith("llamacpp:") and " " in line and not line.startswith("#"):
                k, v = line.rsplit(" ", 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
        return out
    except Exception:
        return {}


def slot_busy(port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/slots", timeout=3)
        slots = r.json()
        return any(s.get("is_processing") for s in slots)
    except Exception:
        return False


async def boot_server(cell: dict, log_path: Path):
    bin_dir = BIN[cell["bin"]]
    args = [f"{bin_dir}/llama-server", "--model", MODEL,
            "--host", "127.0.0.1", "--port", str(cell["port"]),
            "-ngl", str(cell["ngl"]), "-c", str(cell["ctx"]),
            "--parallel", "1", "-t", str(cell["threads"]),
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--temp", "0.7", "--reasoning", "off", "--metrics", "--no-webui",
            "--verbose"]
    env = {**os.environ, "LD_LIBRARY_PATH": bin_dir}
    log = open(log_path, "w")
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=log)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 120:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode} — see {log_path}")
        try:
            if httpx.get(f"http://127.0.0.1:{cell['port']}/health", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        await asyncio.sleep(0.25)
    else:
        proc.kill()
        raise RuntimeError("server did not become healthy in 120 s")
    return proc


def make_client(port: int) -> LlmClient:
    cfg = AgentConfig()
    cfg.llama_server_host = "127.0.0.1"
    cfg.llama_server_port = port
    cfg.llm_model_name = "qwen3.6-35b-a3b"
    cfg.llm_temperature = 0.7
    cfg.llm_max_tokens = 128
    cfg.llm_disable_thinking = True  # request-level suppression exactly as production
    return LlmClient(cfg)


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    return {
        "median": round(statistics.median(s), 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
        "p90": round(s[max(0, int(round(0.9 * len(s))) - 1)], 3),
        "n": len(s),
    }


async def consume(client: LlmClient, user_text: str, max_tokens: int = 100):
    """One streaming user turn through the REAL client. Returns
    (ttft_s, total_s, chars, n_deltas)."""
    t0 = time.perf_counter()
    first_tok = None
    n_chars = 0
    n_deltas = 0
    async for delta in client.chat_stream(build_messages(user_text), max_tokens=max_tokens):
        if first_tok is None:
            first_tok = time.perf_counter() - t0
        n_chars += len(delta)
        n_deltas += 1
    return first_tok, time.perf_counter() - t0, n_chars, n_deltas


async def probe_normal(client: LlmClient, user_text: str, max_tokens=100) -> dict:
    ttft, total, n_chars, n_deltas = await consume(client, user_text, max_tokens)
    return {
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_s": round(total, 4),
        "reply_chars": n_chars,
        "n_deltas": n_deltas,
        "ok": n_chars > 0,
    }


async def probe_extract_raw(port: int) -> dict:
    """Non-streaming JSON probe with a raw POST (captures usage/prefill detail)."""
    payload = {
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "system", "content": "You extract memories as JSON."},
                     {"role": "user", "content": EXTRACTION_INSTRUCTION}],
        "stream": False, "temperature": 0.7, "max_tokens": 200,
        **THINKING_KWARGS,
        "response_format": {"type": "json_object"},
    }
    t0 = time.perf_counter()
    r = await asyncio.to_thread(httpx.post, f"http://127.0.0.1:{port}/v1/chat/completions",
                                json=payload, timeout=600)
    total = time.perf_counter() - t0
    body = r.json()
    usage = body.get("usage", {})
    content = body["choices"][0]["message"].get("content", "")
    try:
        json.loads(content)
        valid = True
    except Exception:
        valid = False
    comp = usage.get("completion_tokens") or 0
    return {
        "http": r.status_code, "total_s": round(total, 4),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": comp,
        "gen_tok_s": round(comp / total, 3) if comp and total else None,
        "json_valid": valid,
        "content_head": content[:60],
    }


async def probe_contention(port: int, client: LlmClient, disconnect_bg: bool) -> dict:
    """User streaming turn started 0.4 s AFTER a background non-streaming leg.

    Reproduces the production contention class (the S8 30-42 s outliers):
    the user's TTFT includes the queue wait behind the background leg. The
    background leg runs on its OWN async httpx client so that task.cancel()
    closes the TCP connection — the production v0.10.2 cancellation
    transport — and the slot-free latency is measured.
    """
    bg_payload = {
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "system", "content": "You extract memories as JSON."},
                     {"role": "user", "content": EXTRACTION_INSTRUCTION}],
        "stream": False, "temperature": 0.7, "max_tokens": 220,
        **THINKING_KWARGS,
        "response_format": {"type": "json_object"},
    }
    t_bg_start = time.perf_counter()

    async def bg_leg():
        async with httpx.AsyncClient(timeout=600) as ac:
            await ac.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=bg_payload)
        return time.perf_counter() - t_bg_start

    bg_task = asyncio.create_task(bg_leg())
    await asyncio.sleep(0.4)
    if not bg_task.done():
        assert slot_busy(port), "background leg did NOT occupy the slot"

    ttft, total, n_chars, _ = await consume(client, USER_TURNS[0])
    res = {"user_ttft_s": round(ttft, 4), "user_total_s": round(total, 4),
           "user_chars": n_chars}

    if disconnect_bg:
        cancel_t0 = time.perf_counter()
        bg_task.cancel()
        try:
            await bg_task
        except (asyncio.CancelledError, Exception):
            pass
        while slot_busy(port):
            await asyncio.sleep(0.02)
        res["slot_free_ms"] = round((time.perf_counter() - cancel_t0) * 1000, 1)
        res["disconnect_transport"] = "async-httpx-cancel (production v0.10.2 class)"
    else:
        bg_dur = await bg_task
        res["bg_total_s"] = round(bg_dur, 4)
    return res


async def probe_cancel(port: int, client: LlmClient) -> dict:
    """Streaming request disconnected right after the first content token
    (the barge-in pattern); slot-free latency + follow-up usability."""
    gen = client.chat_stream(build_messages(USER_TURNS[1]), max_tokens=300)
    t0 = time.perf_counter()
    first_tok = None
    async for delta in gen:
        first_tok = time.perf_counter() - t0
        break  # barge-in: stop consuming
    await gen.aclose()  # deterministic: closes the stream context -> TCP disconnect
    cancel_t0 = time.perf_counter()
    while slot_busy(port):
        await asyncio.sleep(0.02)
    slot_free_ms = (time.perf_counter() - cancel_t0) * 1000
    client2 = make_client(port)
    ttft2, total2, n2, _ = await consume(client2, "Rendben, köszönöm!", max_tokens=40)
    await client2.aclose()
    return {
        "first_token_before_disconnect_s": round(first_tok, 4) if first_tok else None,
        "slot_free_ms": round(slot_free_ms, 1),
        "followup_ttft_s": round(ttft2, 4),
        "followup_total_s": round(total2, 4),
        "followup_chars": n2,
    }


async def probe_cache(port: int) -> dict:
    """Identical prompt twice through the real client, then a usage probe."""
    client = make_client(port)
    runs = []
    for i in range(2):
        ttft, total, n_chars, _ = await consume(client, USER_TURNS[2], max_tokens=60)
        runs.append({"run": i, "ttft_s": round(ttft, 4), "total_s": round(total, 4)})
    payload = {
        "model": "m", "stream": False, "temperature": 0.7, "max_tokens": 16,
        **THINKING_KWARGS,
        "messages": build_messages(USER_TURNS[2]),
    }
    r = await asyncio.to_thread(httpx.post, f"http://127.0.0.1:{port}/v1/chat/completions",
                                json=payload, timeout=300)
    usage = r.json().get("usage", {})
    await client.aclose()
    runs.append({"prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")})
    return {"runs": runs}


async def probe_sse_integrity(port: int) -> dict:
    """Raw SSE line-kind census of one full stream (same payload shape the
    production client sends)."""
    payload = {
        "model": "qwen3.6-35b-a3b",
        "messages": build_messages(USER_TURNS[3]),
        "stream": True, "temperature": 0.7, "max_tokens": 80,
        **THINKING_KWARGS,
    }
    census = {"role_chunks": 0, "content_chunks": 0, "reasoning_chunks": 0,
              "finish_chunks": 0, "done_marker": False, "other_lines": 0,
              "total_lines": 0, "assembled_chars": 0, "first_line_kind": None}
    async with httpx.AsyncClient(timeout=300) as ac:
        async with ac.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                             json=payload) as resp:
            census["http_status"] = resp.status_code
            async for line in resp.aiter_lines():
                s = line.strip()
                if not s:
                    continue
                census["total_lines"] += 1
                if s == "data: [DONE]":
                    census["done_marker"] = True
                    continue
                if s.startswith("data:"):
                    try:
                        obj = json.loads(s[5:].strip())
                        delta = (obj.get("choices") or [{}])[0].get("delta", {})
                        if census["first_line_kind"] is None:
                            census["first_line_kind"] = (
                                "role" if delta.get("role") else
                                "reasoning" if delta.get("reasoning_content") else
                                "content" if delta.get("content") else "other")
                        if delta.get("reasoning_content"):
                            census["reasoning_chunks"] += 1
                        if delta.get("content"):
                            census["content_chunks"] += 1
                            census["assembled_chars"] += len(delta["content"])
                        if (obj.get("choices") or [{}])[0].get("finish_reason"):
                            census["finish_chunks"] += 1
                        if delta.get("role"):
                            census["role_chunks"] += 1
                    except Exception:
                        census["other_lines"] += 1
                else:
                    census["other_lines"] += 1
    return census


async def run_cell(cell_id: str, out_dir: Path) -> dict:
    cell = CELLS[cell_id]
    log_path = out_dir / f"server_{cell_id}.log"
    print(f"[{cell_id}] booting {cell['label']} ...", flush=True)
    proc = await boot_server(cell, log_path)
    try:
        _v = subprocess.run(
            [f"{BIN[cell['bin']]}/llama-server", "--version"],
            env={**os.environ, "LD_LIBRARY_PATH": BIN[cell["bin"]]},
            capture_output=True, text=True)
        version_line = (_v.stdout or _v.stderr).strip().splitlines()[0]

        client = make_client(cell["port"])
        await consume(client, "Egy rövid bemelegítő válasz most.", max_tokens=30)
        m0 = metrics_snapshot(cell["port"])
        try:
            props = httpx.get(f"http://127.0.0.1:{cell['port']}/props", timeout=10).json()
        except Exception:
            props = {}
        t_cell0 = time.perf_counter()

        results: dict = {
            "cell": cell_id, "label": cell["label"], "version": version_line,
            "flags": {k: cell[k] for k in ("bin", "port", "ngl", "ctx", "threads")},
            "machine": {"cpu_count": os.cpu_count()},
            "props_head": {"total_slots": props.get("total_slots"),
                           "model_path": props.get("model_path"),
                           "mode": props.get("mode")},
        }

        # --- normal streaming user turns --------------------------------------
        normals = []
        for i, turn in enumerate(USER_TURNS):
            r = await probe_normal(client, turn)
            r["i"] = i
            normals.append(r)
            print(f"[{cell_id}] normal {i}: ttft={r['ttft_s']}s total={r['total_s']}s ok={r['ok']}", flush=True)
        results["normal"] = normals
        results["normal_stats"] = {
            "ttft_s": percentiles([r["ttft_s"] for r in normals if r["ttft_s"]]),
            "total_s": percentiles([r["total_s"] for r in normals]),
            "streaming_ok": sum(1 for r in normals if r["ok"]),
        }

        # --- non-streaming JSON extraction legs --------------------------------
        extracts = []
        for i in range(4):
            r = await probe_extract_raw(cell["port"])
            r["i"] = i
            extracts.append(r)
            print(f"[{cell_id}] extract {i}: total={r['total_s']}s json_valid={r['json_valid']} cached={r['cached_tokens']}", flush=True)
        results["extract"] = extracts
        results["extract_stats"] = {
            "total_s": percentiles([r["total_s"] for r in extracts]),
            "prompt_tokens": extracts[0]["prompt_tokens"],
            "json_valid_count": sum(1 for r in extracts if r["json_valid"]),
        }

        # --- contention (queue behind background leg) --------------------------
        contends = []
        for i in range(4):
            r = await probe_contention(cell["port"], client, disconnect_bg=(i % 2 == 0))
            r["i"] = i
            contends.append(r)
            print(f"[{cell_id}] contend {i}: user_ttft={r['user_ttft_s']}s slot_free_ms={r.get('slot_free_ms')}", flush=True)
            await asyncio.sleep(1.0)
        results["contention"] = contends
        results["contention_stats"] = {
            "user_ttft_behind_bg_s": percentiles([r["user_ttft_s"] for r in contends]),
            "slot_free_ms_on_disconnect": percentiles(
                [r["slot_free_ms"] for r in contends if "slot_free_ms" in r]),
        }

        # --- cancellation / barge-in -------------------------------------------
        cancels = []
        for i in range(3):
            r = await probe_cancel(cell["port"], client)
            r["i"] = i
            cancels.append(r)
            print(f"[{cell_id}] cancel {i}: slot_free_ms={r['slot_free_ms']} followup_ttft={r['followup_ttft_s']}s", flush=True)
        results["cancel"] = cancels
        results["cancel_stats"] = {
            "slot_free_ms": percentiles([r["slot_free_ms"] for r in cancels]),
            "followup_ttft_s": percentiles([r["followup_ttft_s"] for r in cancels]),
        }

        # --- prefix cache reuse --------------------------------------------------
        results["cache"] = await probe_cache(cell["port"])

        # --- raw SSE integrity ----------------------------------------------------
        results["sse"] = await probe_sse_integrity(cell["port"])
        print(f"[{cell_id}] sse: {results['sse']}", flush=True)

        m1 = metrics_snapshot(cell["port"])
        results["metrics_delta"] = {k: round(m1.get(k, 0) - m0.get(k, 0), 1)
                                   for k in sorted(set(m0) & set(m1))}
        results["cell_wall_s"] = round(time.perf_counter() - t_cell0, 1)
        await client.aclose()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = asyncio.run(run_cell(args.cell, out_dir))
    path = out_dir / f"{args.cell}.json"
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[{args.cell}] written {path}", flush=True)


if __name__ == "__main__":
    main()
