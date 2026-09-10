#!/usr/bin/env python3
"""TASK 1 forensic driver: trace the ASR-transcript → LLM first-token chain.

Drives the REAL application components (VoiceMemBridge, LlmClient,
teacher_persona) with a synthetic transcript against the forensic wiretap
mock (tests/forensic_llm_mock.py) standing in for llama-server, and records
a concrete proof at every stage boundary:

  S1  transcript received (synthetic; mic→VAD→ASR is user-verified)
  S2  VoiceMem turn: feed + retrieval (vendor LLM calls observed on the wire)
  S3  prompt assembled (system + memory context + user text)
  S4  LLM request configuration (model, endpoint, stream)
  S5  HTTP request left the application (wiretap record)
  S6  first response token received (latency)
  S7  reply complete (char count) / explicit failure reason

Vendor-env modes (--vendor-env) simulate how the process was started:
  full    — like config/env.local.ps1 sourced (web path via start_agent.ps1)
  partial — base_url + api_key exported, OPENAI_MODEL forgotten
  none    — raw shell: no OPENAI_* variables at all (CLI started directly)

Output: JSON report on stdout (and --out file); exit 0 only when S1..S7 all
pass. See tests/forensic_llm_mock.py for the server modes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TRANSCRIPT = "Nehezen használom a present perfectet."
_MODEL = "qwen3.6-35b-a3b"

_OPENAI_VARS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_CHAT_MODEL",
)


def _stage(idx: str, name: str, ok: bool, **detail) -> dict:
    return {"id": idx, "name": name, "ok": bool(ok), "detail": detail}


def _read_wiretap(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        return []


def _apply_env(vendor_env: str, port: int, memory_root: Path, e5_dir: Path) -> None:
    for var in _OPENAI_VARS:
        os.environ.pop(var, None)
    os.environ["LLAMA_SERVER_HOST"] = "127.0.0.1"
    os.environ["LLAMA_SERVER_PORT"] = str(port)
    os.environ["VOICEMEM_MEMORY_ROOT"] = str(memory_root)
    os.environ["VOICEMEM_E5_MODEL"] = str(e5_dir)
    os.environ.setdefault("VOICEMEM_EMBED_DIM", "384")
    base = f"http://127.0.0.1:{port}/v1"
    if vendor_env == "full":
        os.environ["OPENAI_BASE_URL"] = base
        os.environ["OPENAI_API_KEY"] = "not-needed-but-required-by-openai-lib"
        os.environ["OPENAI_MODEL"] = _MODEL
        os.environ["OPENAI_CHAT_MODEL"] = _MODEL
    elif vendor_env == "partial":
        os.environ["OPENAI_BASE_URL"] = base
        os.environ["OPENAI_API_KEY"] = "not-needed-but-required-by-openai-lib"
    # "none": leave unset


async def _wait_health(port: int, timeout_s: float = 15.0) -> bool:
    import httpx

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            # NOTE: httpx.get is SYNCHRONOUS — do not await it. It blocks the
            # loop for at most its 1 s timeout, which is fine during startup.
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001 - retry until timeout
            pass
        await asyncio.sleep(0.15)
    return False


async def run_trace(args: argparse.Namespace, mock: subprocess.Popen) -> dict:
    from app.config import AgentConfig
    from app.llm import LlmClient, LlmUnavailableError
    from app.teacher_persona import build_messages, build_system_prompt, fit_prompt_budget
    from app.voicemem_bridge import VoiceMemBridge, pin_e5_local_model

    os.chdir(REPO)
    config = AgentConfig()
    config.apply_env()  # same as app/main.py load_config — env overrides
    pin_e5_local_model(config)

    llm = LlmClient(config)
    bridge = VoiceMemBridge(config, llm)

    stages: list[dict] = []
    stages.append(_stage("S1", "transcript", True, text=TRANSCRIPT, chars=len(TRANSCRIPT)))

    # S1b — seed memory so the retrieval leg has something to find (a fresh
    # store legitimately returns 0 hits and runs no classifier LLM call).
    # store_fact() also exercises the v0.5.1 ingest feed adapter.
    seed = [
        "Thomas prefers warmer lighting.",
        "Nehezen használja a present perfectet, gyakran összekeveri a past simple-lal.",
    ]
    seed_ok = True
    seed_err = ""
    try:
        for fact in seed:
            await bridge.store_fact(fact)
        # ingest(async_facts=True) writes in a background thread that needs
        # ~10 s (E5 encode + extraction LLM round trip + store writes) —
        # measured: the thread finishes in 9.7 s on this sandbox
        await asyncio.sleep(12.0)
    except Exception as exc:  # noqa: BLE001 - evidence, not a crash
        seed_ok = False
        seed_err = f"{type(exc).__name__}: {exc}"
    stages.append(
        _stage("S1b", "seed_memory", seed_ok, facts=len(seed), error=seed_err)
    )

    # S2 — VoiceMem turn (feed + retrieval; vendor LLM calls hit the mock)
    t0 = time.perf_counter()
    turn_ctx = await bridge.process_turn(TRANSCRIPT)
    s2_ms = (time.perf_counter() - t0) * 1000.0
    stages.append(
        _stage(
            "S2",
            "voicemem.process_turn",
            True,  # the bridge degrades instead of raising; wiretap tells the truth
            memory_context_chars=len(turn_ctx.memory_context or ""),
            latency_ms=round(s2_ms, 1),
            note="ok=True means no crash; vendor LLM activity is judged from the wiretap",
        )
    )

    # S3 — prompt assembly
    system_prompt = build_system_prompt(turn_ctx.memory_context or "")
    messages = fit_prompt_budget(build_messages(TRANSCRIPT, system_prompt))
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    stages.append(
        _stage(
            "S3",
            "prompt_built",
            len(messages) >= 2 and total_chars > 0,
            n_messages=len(messages),
            total_chars=total_chars,
        )
    )

    # S4 — request configuration (what the app WOULD put on the wire)
    stages.append(
        _stage(
            "S4",
            "llm_request_config",
            bool(config.llama_server_url) and bool(config.llm_model_name),
            model=config.llm_model_name,
            base_url=config.llama_server_url,
            stream=True,
        )
    )

    # S5–S7 — drive the real streaming call
    parts: list[str] = []
    t_first: float | None = None
    err: str | None = None
    t0 = time.perf_counter()
    try:
        async for delta in llm.chat_stream(messages):
            if t_first is None:
                t_first = (time.perf_counter() - t0) * 1000.0
            if delta:
                parts.append(delta)
    except LlmUnavailableError as exc:
        err = str(exc)
    except Exception as exc:  # noqa: BLE001 - any failure is evidence
        err = f"{type(exc).__name__}: {exc}"
    llm_ms = (time.perf_counter() - t0) * 1000.0
    reply = "".join(parts)

    # early wiretap read: proves the app request actually left (S5)
    early_wiretap = _read_wiretap(args.log)
    early_app_reqs = [r for r in early_wiretap if r.get("stream")]

    stages.append(
        _stage(
            "S5",
            "http_request_left_app",
            bool(early_app_reqs),
            app_request_count=len(early_app_reqs),
            app_request_model=(early_app_reqs[-1].get("model") if early_app_reqs else None),
        )
    )
    stages.append(
        _stage(
            "S6",
            "first_response_token",
            t_first is not None,
            latency_ms=(round(t_first, 1) if t_first is not None else None),
        )
    )
    stages.append(
        _stage(
            "S7",
            "reply_complete",
            err is None and len(reply) > 0,
            chars=len(reply),
            head=reply[:80],
            error=(err or ""),
        )
    )

    # S7b — consolidate the turn (user text + reply) into long-term memory,
    # exactly like the pipeline does after a successful reply. Exercises the
    # v0.5.1 ingest(text, agent_reply=...) commit path; the wiretap should
    # show the vendor extraction call for the PAIR.
    commit_status = "(not attempted)"
    if reply:
        try:
            commit_status = await bridge.commit_reply(reply)
        except Exception as exc:  # noqa: BLE001 - evidence, not a crash
            commit_status = f"EXCEPTION {type(exc).__name__}: {exc}"
    stages.append(
        _stage(
            "S7b",
            "commit_reply",
            commit_status == "committed",
            status=commit_status,
        )
    )

    # wiretap breakdown (read AFTER the commit so the pair-ingest vendor
    # call is included)
    wiretap = _read_wiretap(args.log)
    app_reqs = [r for r in wiretap if r.get("stream")]
    vendor_reqs = [r for r in wiretap if not r.get("stream")]
    vendor_models = sorted({str(r.get("model")) for r in vendor_reqs})
    app_models = sorted({str(r.get("model")) for r in app_reqs})

    last_ok = next((s["id"] for s in reversed(stages) if s["ok"]), None)
    first_bad = next((s["id"] for s in stages if not s["ok"]), None)
    verdict = {
        "last_confirmed_success": last_ok,
        "first_unconfirmed_or_failed_stage": first_bad,
        "root_cause_hint": "",
        "vendor_llm_requests": len(vendor_reqs),
        "vendor_models_on_the_wire": vendor_models,
        "app_llm_requests": len(app_reqs),
        "app_models_on_the_wire": app_models,
    }
    if err:
        verdict["root_cause_hint"] = err[:300]
    elif vendor_env_note(args):
        verdict["root_cause_hint"] = vendor_env_note(args)

    report = {
        "schema": "voicemem-agent/forensic_trace@1",
        "run": {
            "vendor_env": args.vendor_env,
            "llm_mode": args.llm_mode,
            "port": args.port,
            "memory_root": args.memory_root,
            "wiretap": args.log,
        },
        "stages": stages,
        "wiretap_records": wiretap,
        "verdict": verdict,
    }
    await llm.aclose()
    return report


def vendor_env_note(args: argparse.Namespace) -> str:
    if args.vendor_env == "none":
        return (
            "raw-shell start simulated (no OPENAI_* env): vendor LLM legs are "
            "expected to fail silently — check vendor_llm_requests and the "
            "stderr log for swallowed exceptions"
        )
    if args.vendor_env == "partial":
        return (
            "partial env simulated (OPENAI_MODEL unset): vendor requests are "
            "expected to carry the gpt-4o-mini fallback model name"
        )
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vendor-env", choices=["full", "partial", "none"], default="full")
    ap.add_argument("--llm-mode", choices=["normal", "thinking", "empty"], default="normal")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--memory-root", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--keep-mock", action="store_true", help="leave the mock running (debug)")
    args = ap.parse_args()

    if not args.memory_root:
        args.memory_root = f"/tmp/forensic_vm_{args.vendor_env}_{args.llm_mode}"
    if not args.log:
        args.log = f"/tmp/forensic_wiretap_{args.vendor_env}_{args.llm_mode}.jsonl"
    if not args.out:
        args.out = f"/tmp/forensic_report_{args.vendor_env}_{args.llm_mode}.json"

    memory_root = Path(args.memory_root)
    if memory_root.exists():
        shutil.rmtree(memory_root)
    memory_root.mkdir(parents=True)

    for f in (args.log, args.out):
        p = Path(f)
        if p.exists():
            p.unlink()

    e5_dir = REPO / "models" / "embedding" / "multilingual-e5-small"
    _apply_env(args.vendor_env, args.port, memory_root, e5_dir)

    mock = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "tests" / "forensic_llm_mock.py"),
            "--port",
            str(args.port),
            "--mode",
            args.llm_mode,
            "--log",
            args.log,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not asyncio.run(_wait_health(args.port)):
            print(json.dumps({"fatal": f"mock server did not come up on {args.port}"}))
            return 2
        report = asyncio.run(run_trace(args, mock))
    finally:
        if not args.keep_mock:
            mock.terminate()
            try:
                mock.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mock.kill()

    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    ok = all(s["ok"] for s in report["stages"])
    v = report["verdict"]
    print(
        f"\n=== VERDICT: last OK={v['last_confirmed_success']} "
        f"first BAD={v['first_unconfirmed_or_failed_stage']} "
        f"vendor_requests={v['vendor_llm_requests']} "
        f"vendor_models={v['vendor_models_on_the_wire']} "
        f"app_models={v['app_models_on_the_wire']} ===",
        file=sys.stderr,
    )
    # The vendor's embedded qdrant + daemon threads abort the C++ runtime at
    # interpreter teardown ("terminate called without an active exception",
    # SIGABRT) AFTER everything completed and the report is on disk. Bypass
    # the noisy teardown: the report file is the source of truth.
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
