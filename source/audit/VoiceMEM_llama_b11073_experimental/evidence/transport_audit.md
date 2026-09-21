# Transport audit — VoiceMem v0.10.7 LLM client (READ-ONLY, 2026-09-21)

Baseline audited BEFORE any experimental change, per the workstream gate
("VERIFY BEFORE CHANGING ANYTHING"). Every row is file:line verifiable on
the v0.10.7 tree (git HEAD ca93e19-era working tree, VERSION 0.10.7).

| Aspect | Mechanism (evidence) |
| --- | --- |
| Endpoint | `POST {base}/chat/completions` — `base = config.llama_server_url` = `http://{host}:{port}/v1` (app/llm.py:237 `base_url=`, app/llm.py:308 `client.stream("POST", "chat/completions", ...)`; app/config.py:246-247) |
| Health | `GET http://{host}:{port}/health`, 2 s timeout, True on HTTP 200 (app/llm.py:256-267; url property app/config.py:249-251) |
| Request format | OpenAI-compatible JSON: `model`, `messages`, `stream`, `temperature`, `max_tokens` (app/llm.py:283-294); JSON mode adds `response_format={"type":"json_object"}` (app/llm.py:378-387) |
| Streaming / SSE | `stream=True`; `aiter_lines()`; `parse_sse_content_delta` reads `choices[0].delta.content` (fallback `message.content`); tolerates `:` keep-alives, `[DONE]`, malformed JSON (app/llm.py:27-67, 308-321). First-token diagnostic counters: `first_line_s`, `sse_lines` (app/llm.py:304-312) |
| Reasoning channel | `parse_sse_reasoning_delta` reads `choices[0].delta.reasoning_content` — thinking-only streams are detected and reported, never silently empty (app/llm.py:108-132, 319-321, 328-355) |
| Reasoning flags | THREE single-source mechanisms from `config/llm_config.yaml` `reasoning.enabled: false`: server `--reasoning off` (generated, app/llm_config.py:185-212), request `chat_template_kwargs.enable_thinking=false` + `reasoning_effort="none"` (app/llm_config.py:214-230, consumed app/llm.py:174-214, 293, 386) |
| Model name | `config.llm_model_name` ← canonical `llm.model` via `materialise_llm_runtime` (app/config.py:464) → payload `model` field (app/llm.py:284, 379). NOT a file path. |
| Timeouts | `httpx.Timeout(120.0, connect=3.0)` on the shared client (app/llm.py:235); health 2 s (app/llm.py:262) |
| Cancellation | Consumer breaks out of `async for` → the `async with client.stream(...)` context releases the connection (TCP disconnect aborts the server-side slot) (app/llm.py:308; the production consumer loop app/pipeline.py:653; turn cancel app/pipeline.py:259-266; web disconnect cancels the turn task app/web_server.py:2652) |
| Empty-reply failure | HARD `LlmUnavailableError` with four differentiated causes: thinking-only / no data / >10 s stall before first line / keep-alive-only (app/llm.py:328-355) |
| Host/port config | `AgentConfig.llama_server_host/port` defaults `127.0.0.1:8080` (app/config.py:167-168); **existing env overrides** `LLAMA_SERVER_HOST` / `LLAMA_SERVER_PORT` (app/config.py:502-505, applied in `apply_env`); `OPENAI_BASE_URL` alternative (app/config.py:535-536) |
| Vendor bridge legs | `OPENAI_BASE_URL` pinned from `config.llama_server_url` when unset (app/voicemem_bridge.py:238-241) — every vendor (mem0) LLM leg targets the same server |
| Canonical server config | `config/llm_config.yaml` → `LlmRuntimeConfig.llama_server_args()` = `-ngl 20 -c 32768 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --temp 0.7 --reasoning off` (the pinned llama.cpp **b10717**; MODELS.lock.json component "llama" tag b10717) |

## Consequence for the experimental runtime

**The required environment-only override ALREADY EXISTS and needs no code
change:** with the experimental server bound on `127.0.0.1:8081`,

```
LLAMA_SERVER_HOST=127.0.0.1
LLAMA_SERVER_PORT=8081
```

the whole application (Python client `app/llm.py` AND every vendor bridge
leg through `OPENAI_BASE_URL`) targets the experimental server, because
every path derives from `config.llama_server_url`. No application file is
modified by the experiment; unsetting the two variables restores the
production endpoint (port 8080) byte-for-byte.
