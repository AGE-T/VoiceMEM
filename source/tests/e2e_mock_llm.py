#!/usr/bin/env python3
"""Deterministic OpenAI-compatible mock LLM for VoiceMem runtime E2E tests.

Serves ONLY ``POST /v1/chat/completions`` (non-streaming, the exact surface
voicemem's text-mode ingest/retrieval uses) with canned, keyword-routed
responses so the full memory pipeline (extraction -> mem0/Qdrant store +
local E5 embedding -> cognitive graph -> right-brain traits -> search) can
run end-to-end WITHOUT a GPU and with fully deterministic content.

Routing (by content sniffing the request messages):
  1. conflict-resolver system prompt ("memory manager")
        -> {"memory": []} (no ADD/UPDATE/DELETE decisions - data safety)
  2. a known E2E sentence in the user turn
        -> the merged-extraction JSON for that sentence (memory items with
           slot + entities, emotion, traits)
  3. the standalone right-brain trait prompt ('"items": [{"slot"' hint)
        -> {"items": []}
  4. anything else -> the empty merged shape {"memory": [], "emotion": "",
     "traits": []}

The E2E sentence table mirrors scripts/verify_voicemem_runtime.py.

Usage:
    python tests/e2e_mock_llm.py [--port 18080]
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: E2E sentences -> merged-extraction payload. The memory TEXT is the
#: sentence's normalised fact; entities feed the cognitive graph; traits
#: feed the right-brain judgment table (rb_traits) - the embedding path we
#: are proving.
SCENES: dict[str, dict[str, Any]] = {
    # -- Hungarian fact -----------------------------------------------------
    "A kedvenc kávém a hosszú kávé, minden reggel iszom egyet Budapesten": {
        "memory": [
            {
                "id": "hu-coffee-1",
                "text": "The user's favourite coffee is a long coffee and they drink one every morning in Budapest",
                "event": "ADD",
                "slot": "daily_life",
                "confidence": 0.95,
                "entities": [
                    {"name": "long coffee", "entity_type": "preference",
                     "role": "object"},
                    {"name": "Budapest", "entity_type": "place",
                     "role": "context"},
                ],
            }
        ],
        "emotion": "",
        "traits": [],
    },
    # -- English fact (same topic, cross-language retrieval target) ---------
    "My favourite coffee is a long black, I drink one every morning": {
        "memory": [
            {
                "id": "en-coffee-1",
                "text": "The user's favourite coffee is a long black and they drink one every morning",
                "event": "ADD",
                "slot": "daily_life",
                "confidence": 0.95,
                "entities": [
                    {"name": "long black", "entity_type": "preference",
                     "role": "object"},
                ],
            }
        ],
        "emotion": "",
        "traits": [],
    },
    # -- Hungarian trait-bearing utterance ----------------------------------
    "Mindig ideges leszek prezentáció előtt": {
        "memory": [
            {
                "id": "hu-trait-1",
                "text": "The user always gets nervous before presentations",
                "event": "ADD",
                "slot": "work",
                "confidence": 0.9,
                "entities": [
                    {"name": "presentations", "entity_type": "event",
                     "role": "context"},
                ],
            }
        ],
        "emotion": "anxious",
        "traits": [
            {"slot": "情绪", "label": "gets nervous before presentations"},
        ],
    },
    # -- English trait-bearing utterance ------------------------------------
    "I dislike long meetings": {
        "memory": [
            {
                "id": "en-trait-1",
                "text": "The user dislikes long meetings",
                "event": "ADD",
                "slot": "work",
                "confidence": 0.9,
                "entities": [
                    {"name": "long meetings", "entity_type": "event",
                     "role": "object"},
                ],
            }
        ],
        "emotion": "",
        "traits": [
            {"slot": "喜好与厌恶", "label": "dislikes long meetings"},
        ],
    },
}


#: Query-classifier routing (system prompt: "You are a memory router").
#: The user message IS the query; the answer shape is
#: {"slots": [...], "entities": [...]} (1-2 base-7 slots, named entities).
CLASSIFIER_ANSWERS: dict[str, dict[str, Any]] = {
    "Milyen kávét iszom reggel?": {"slots": ["daily_life"], "entities": ["long coffee"]},
    "What coffee do I drink every morning?": {"slots": ["daily_life"], "entities": ["long black"]},
    "kedvenc kávé minden reggel": {"slots": ["daily_life"], "entities": ["long coffee"]},
    "my favourite morning coffee Budapest": {"slots": ["daily_life"], "entities": ["Budapest"]},
    "hogyan érzem magam prezentáció előtt?": {"slots": ["work"], "entities": ["presentations"]},
    "how do I feel before presentations?": {"slots": ["work"], "entities": ["presentations"]},
    "what do I dislike at work?": {"slots": ["work"], "entities": ["long meetings"]},
}


def _route(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the canned response for a request."""
    system = " ".join(str(m.get("content", "")) for m in messages
                      if m.get("role") == "system")
    user = " ".join(str(m.get("content", "")) for m in messages
                    if m.get("role") != "system")
    everything = system + "\n" + user

    if "memory manager" in system:
        # the conflict resolver: make NO change decisions
        return {"memory": []}

    if "memory router" in system:
        # the query classifier: the user message is the query itself
        for query, answer in CLASSIFIER_ANSWERS.items():
            if query.strip() in user.strip():
                return answer
        return {"slots": ["daily_life"], "entities": []}

    for sentence, payload in SCENES.items():
        if sentence in user or sentence in everything:
            return payload

    if '"items"' in everything and "喜好与厌恶" in everything:
        # standalone right-brain trait extraction fallback
        return {"items": []}

    return {"memory": [], "emotion": "", "traits": []}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        print(f"[mock-llm] {self.path} {fmt % args}", flush=True)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = _route(body.get("messages") or [])
        completion = {
            "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-llm"),
            "system_fingerprint": "e2e-mock",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }
        self._json(200, completion)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] in ("/health", "/v1/models"):
            self._json(200, {"status": "ok", "models": ["mock-llm"]})
            return
        self._json(404, {"error": "not found"})

    def _json(self, code: int, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mock-llm] listening on http://{args.host}:{args.port}/v1 "
          f"({len(SCENES)} canned scenes)", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
