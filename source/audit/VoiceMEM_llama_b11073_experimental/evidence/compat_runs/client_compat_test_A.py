"""Environment-only override proof: the REAL production client against the
production-proxy b10717 runtime. Zero code changes — only env vars."""
import asyncio, os, sys, time

os.environ["LLAMA_SERVER_HOST"] = "127.0.0.1"
os.environ["LLAMA_SERVER_PORT"] = "8080"

sys.path.insert(0, "/home/z/my-project/voicemem-agent")
from app.config import AgentConfig
from app.llm import LlmClient, parse_sse_content_delta

async def main():
    cfg = AgentConfig()
    cfg.apply_env()
    print(f"[override proof] llama_server_url = {cfg.llama_server_url}")
    print(f"[override proof] health_url      = {cfg.llama_server_health_url}")
    assert cfg.llama_server_url == "http://127.0.0.1:8080/v1", "env override failed"

    client = LlmClient(cfg)
    print(f"[health] {await client.health_check()}")

    # 1) Streaming chat (the production user-turn path)
    t0 = time.perf_counter()
    chunks, first_tok = [], None
    async for delta in client.chat_stream([
        {"role": "user", "content": "Say exactly: experimental runtime compatible. Then stop."}
    ], max_tokens=40):
        if first_tok is None:
            first_tok = time.perf_counter() - t0
        chunks.append(delta)
    reply = "".join(chunks)
    print(f"[stream] ttft={first_tok:.3f}s reply={reply[:90]!r}")

    # 2) JSON mode (the production background-extraction path)
    j = await client.chat_json([
        {"role": "user", "content": 'Return a JSON object {"ok": true, "n": 7} exactly.'}
    ])
    print(f"[json]   parsed={j}")

    # 3) reasoning suppression visible in the payload shape?
    kw = cfg.llm_runtime.thinking_control_kwargs() if cfg.llm_runtime else {}
    print(f"[thinking kwargs active] {kw}")
    await client.aclose()

asyncio.run(main())
