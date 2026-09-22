"""Full production config path (from_yaml -> materialise_llm_runtime ->
apply_env) against the experimental runtime; wire payload capture."""
import asyncio, json, os, sys

os.environ["LLAMA_SERVER_PORT"] = "8081"
sys.path.insert(0, "/home/z/my-project/voicemem-agent")
from app.config import AgentConfig
from app.llm import LlmClient
from app.llm_config import load_llm_config

cfg = AgentConfig.from_yaml("/home/z/my-project/voicemem-agent/config/voicemem_config.yaml")
print("config file loaded; llm_runtime materialised:", cfg.llm_runtime is not None)
print("llm_model_name =", cfg.llm_model_name)
print("thinking_control_kwargs =", cfg.llm_runtime.thinking_control_kwargs())
print("llama_server_url =", cfg.llama_server_url)

async def main():
    c = LlmClient(cfg)
    n = 0
    async for d in c.chat_stream([{"role":"user","content":"Reply with the single word: ok"}], max_tokens=20):
        n += len(d)
    print(f"[stream via full config] {n} content chars received")
    await c.aclose()
asyncio.run(main())

# wire-shape check: raw request to be 100% sure what the client sends
import httpx
payload = {
    "model": cfg.llm_model_name,
    "messages": [{"role":"user","content":"Reply: ok"}],
    "stream": False,
    "temperature": cfg.llm_temperature,
    "max_tokens": 30,
    **cfg.llm_runtime.thinking_control_kwargs(),
}
r = httpx.post("http://127.0.0.1:8081/v1/chat/completions", json=payload, timeout=60)
body = r.json()
msg = body["choices"][0]["message"]
print("[wire] http", r.status_code, "| finish:", body["choices"][0].get("finish_reason"),
      "| content:", repr(msg.get("content","")[:40]),
      "| reasoning_content present:", "reasoning_content" in msg and bool(msg.get("reasoning_content")))
print("[wire] usage:", body.get("usage"))
