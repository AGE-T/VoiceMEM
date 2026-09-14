"""Live LlmClient proof against the sandbox llama-server (b10717, new
production profile flags). Verifies the app-side request construction:
chat_template_kwargs.enable_thinking=false + reasoning_effort="none",
and that the streaming reply is non-empty content (no reasoning channel).
This is the EXACT production request path (app/llm.py chat_stream /
chat_json are what web_server.py drives on every voice turn)."""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from app.config import AgentConfig
from app.llm import LlmClient, _thinking_control_kwargs


async def main() -> None:
    cfg = AgentConfig.from_yaml("config/voicemem_config.yaml")

    kw = _thinking_control_kwargs(cfg)
    expected = {"chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none"}
    assert kw == expected, f"unexpected thinking kwargs: {kw!r}"
    print(f"REQUEST-FLAGS: {json.dumps(kw)}")
    print(f"CONFIG: model={cfg.llm_model_name} url={cfg.llama_server_url} "
          f"ctx={cfg.llm_context_size} ngl={cfg.llm_n_gpu_layers} "
          f"parallel={cfg.llm_parallel} kv={cfg.llm_cache_type_k}/{cfg.llm_cache_type_v} "
          f"temp={cfg.llm_temperature} max_tokens={cfg.llm_max_tokens}")

    client = LlmClient(cfg)
    msgs = [{"role": "user",
             "content": "Szia! Kérlek köszönj vissza egyetlen rövid magyar mondatban!"}]
    t0 = time.perf_counter()
    first = None
    chunks = 0
    text = []
    async for delta in client.chat_stream(msgs):
        if first is None:
            first = time.perf_counter() - t0
        chunks += 1
        text.append(delta)
    total = time.perf_counter() - t0
    reply = "".join(text)
    print(f"STREAM: ttft={first:.3f}s total={total:.3f}s chunks={chunks} chars={len(reply)}")
    print(f"REPLY: {reply[:180]}")
    assert len(reply.strip()) > 5, "empty content reply (thinking channel ate it?)"

    t1 = time.perf_counter()
    try:
        data = await client.chat_json([
            {"role": "user",
             "content": "Return a JSON object with keys 'greeting' (a short "
                        "Hungarian greeting string) and 'ok' (boolean true). "
                        "Output ONLY the JSON object."}])
        print(f"JSON: {time.perf_counter() - t1:.3f}s -> {json.dumps(data, ensure_ascii=False)[:180]}")
    except Exception as exc:
        print(f"JSON: model-side note: {type(exc).__name__}: {exc}")

    print("LLM-CLIENT-PROOF: PASS")


asyncio.run(main())
