#!/usr/bin/env python3
"""Forensic wiretap mock of the llama-server OpenAI-compatible API.

Purpose: TASK 1 (ASR → LLM forensic trace). This server stands in for the
real llama-server (127.0.0.1:8080) inside the sandbox so the FULL app chain
can be traced end-to-end, and so every HTTP request that leaves the
application is RECORDED (the "wiretap") — model names, stream flags, and
message heads become concrete evidence instead of assumptions.

Modes (--mode):
  normal    — streaming requests get real SSE content deltas; non-streaming
              requests get a minimal JSON-object completion (vendor
              extraction calls use response_format=json_object).
  thinking  — streaming requests get reasoning_content deltas ONLY (no
              content): reproduces the "thinking-only, 0-char reply"
              field failure (v0.4.4/v0.4.9) without a GPU.
  empty     — streaming requests get keep-alive comments only, no content
              deltas and no [DONE]: reproduces the "server stalled /
              keep-alive-only" failure mode.

Every request is appended to the wiretap log (--log, JSONL):
  {"ts": ..., "path": ..., "model": ..., "stream": bool, "n_messages": int,
   "last_role": ..., "last_head": <first 120 chars of the last message>,
   "response_format": ...}

Run:  python tests/forensic_llm_mock.py --port 18081 --mode normal \
          --log /tmp/forensic_wiretap.jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_REPLY_CHUNKS = [
    "Rendben, ",
    "gyakoroljuk a ",
    "present perfectet. ",
    "Példa: I have just finished my homework.",
]
_THINKING_CHUNKS = [
    "The user struggles with present perfect. ",
    "I should give a short example and encourage them. ",
    "Let me keep the answer brief and friendly.",

]

# -- non-streaming vendor shapes (mirrors tests/e2e_mock_llm.py routing) ----
#: Extraction scenes: driver seed facts + transcript → merged-extraction
#: payloads the vendor's additive parser accepts ("memory" array, ADD events).
_SCENES: dict[str, dict] = {
    "Thomas prefers warmer lighting.": {
        "memory": [
            {
                "id": "th-light-1",
                "text": "Thomas prefers warmer lighting",
                "event": "ADD",
                "slot": "daily_life",
                "confidence": 0.95,
                "entities": [
                    {"name": "warmer lighting", "entity_type": "preference", "role": "object"}
                ],
            }
        ],
        "emotion": "",
        "traits": [],
    },
    "Nehezen használja a present perfectet, gyakran összekeveri a past simple-lal.": {
        "memory": [
            {
                "id": "hu-pp-1",
                "text": "The user finds the present perfect difficult and often confuses it with past simple",
                "event": "ADD",
                "slot": "daily_life",
                "confidence": 0.95,
                "entities": [
                    {"name": "present perfect", "entity_type": "skill", "role": "object"},
                    {"name": "past simple", "entity_type": "skill", "role": "object"},
                ],
            }
        ],
        "emotion": "",
        "traits": [],
    },
    "Nehezen használom a present perfectet.": {
        "memory": [
            {
                "id": "hu-pp-2",
                "text": "The user finds the present perfect difficult",
                "event": "ADD",
                "slot": "daily_life",
                "confidence": 0.9,
                "entities": [
                    {"name": "present perfect", "entity_type": "skill", "role": "object"}
                ],
            }
        ],
        "emotion": "",
        "traits": [],
    },
}

#: Query-classifier answers (system prompt "memory router").
_CLASSIFIER: dict[str, dict] = {
    "Nehezen használom a present perfectet.": {
        "slots": ["daily_life"],
        "entities": ["present perfect"],
    }
}


def _route_nonstream(messages: list) -> dict:
    """Keyword-routed canned reply (same logic as tests/e2e_mock_llm.py)."""
    system = " ".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "system"
    )
    user = " ".join(
        str(m.get("content", "")) for m in messages if m.get("role") != "system"
    )
    everything = system + "\n" + user

    if "memory manager" in system:
        # the conflict resolver: make NO change decisions (data safety)
        return {"memory": []}

    if "memory router" in system:
        for query, answer in _CLASSIFIER.items():
            if query.strip() in user.strip():
                return answer
        return {"slots": ["daily_life"], "entities": []}

    for sentence, payload in _SCENES.items():
        if sentence in user or sentence in everything:
            return payload

    if '"items"' in everything and "喜好与厌恶" in everything:
        # standalone right-brain trait extraction
        return {"items": []}

    return {"memory": [], "emotion": "", "traits": []}

_LOCK = threading.Lock()


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def make_handler(mode: str, log_path: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # silence default stderr access log; the wiretap is the record
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            pass

        # -- helpers -------------------------------------------------------
        def _wiretap(self, path: str, body: dict) -> None:
            msgs = body.get("messages") or []
            last = msgs[-1] if msgs else {}
            rec = {
                "ts": _now_iso(),
                "path": path,
                "model": body.get("model"),
                "stream": bool(body.get("stream")),
                "n_messages": len(msgs),
                "last_role": last.get("role"),
                "last_head": str(last.get("content", ""))[:120],
                "response_format": body.get("response_format"),
            }
            with _LOCK:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        def _send_json(self, obj: dict, status: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        # -- GET -----------------------------------------------------------
        def do_GET(self) -> None:
            if self.path in ("/health", "/v1/health", "/healthz"):
                self._send_json({"status": "ok", "mode": mode})
            elif self.path == "/v1/models":
                self._send_json(
                    {"object": "list", "data": [{"id": "qwen3.6-35b-a3b", "object": "model"}]}
                )
            else:
                self._send_json({"error": f"not found: {self.path}"}, status=404)

        # -- POST ----------------------------------------------------------
        def do_POST(self) -> None:
            if not self.path.endswith("/chat/completions"):
                self._send_json({"error": f"not found: {self.path}"}, status=404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._send_json({"error": "invalid json body"}, status=400)
                return
            self._wiretap(self.path, body)
            stream = bool(body.get("stream"))
            if stream:
                self._send_sse()
            else:
                reply_content = _route_nonstream(body.get("messages") or [])
                self._send_json(
                    {
                        "id": "forensic-mock",
                        "object": "chat.completion",
                        "model": body.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(reply_content, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )

        def _send_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if mode == "thinking":
                for chunk in _THINKING_CHUNKS:
                    self._sse_chunk({"reasoning_content": chunk})
                self._sse_done()
            elif mode == "empty":
                for _ in range(3):
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                # stream ends WITHOUT any content delta and WITHOUT [DONE]
            else:  # normal
                for chunk in _REPLY_CHUNKS:
                    self._sse_chunk({"content": chunk})
                self._sse_done()

        def _sse_chunk(self, delta: dict) -> None:
            payload = json.dumps({"choices": [{"index": 0, "delta": delta}]}, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _sse_done(self) -> None:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--mode", choices=["normal", "thinking", "empty"], default="normal")
    ap.add_argument("--log", default="/tmp/forensic_wiretap.jsonl")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.mode, args.log))
    print(
        f"forensic mock llama-server: {args.host}:{args.port} mode={args.mode} "
        f"wiretap={args.log}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
