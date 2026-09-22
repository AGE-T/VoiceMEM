import httpx, json, sys
port = sys.argv[1]
for trial in range(3):
    r = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
        "model": "m", "stream": False, "temperature": 0.7,
        "max_tokens": 60,
        "response_format": {"type": "json_object"},
        "messages": [{"role":"user","content":'Return a JSON object {"ok": true, "n": 7} exactly.'}],
    }, timeout=120)
    b = r.json()
    c = b["choices"][0]["message"].get("content", "")
    try:
        json.loads(c); verdict = "VALID-JSON"
    except Exception:
        verdict = "NOT-JSON"
    print(f"trial{trial}: http={r.status_code} {verdict} content={c[:80]!r}")
