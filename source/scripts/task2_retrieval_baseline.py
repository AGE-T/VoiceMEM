#!/usr/bin/env python3
"""TASK 2 (PHASE 1) — authoritative retrieval baseline from the v0.5.1 state.

Measures, WITHOUT modifying either implementation:

  PATH A  vm.search(query)                      — the primitive the app calls
          (CLI bridge + web MemoryLayer, post-v0.5.1 TASK 1 fix)
  PATH B  classification = vm.classify(query);
          vm.search(query, slots=…, entities=…) — the rich path
          (the pattern Memory.recall()/PrimeSubgraphFromQuery use)

on a controlled corpus (HU + EN facts, temporal ground truth, traits) with
the REAL pinned vendor (e8384e0), REAL local multilingual-E5 embeddings and
a deterministic keyword-routed mock LLM standing in for llama-server —
the same proven methodology as the v0.5.0 verification / v0.5.1 forensics.

Query matrix (16): HU, EN, HU→EN and EN→HU cross-language, simple factual,
multi-condition, temporal (ma/tegnap/holnap/tegnap este/ma reggel/múlt héten/
jövő héten + EN equivalents), and one no-hit control.

Per (query, path): latency (warmup + 5 runs: mean/median/min/max), hit count,
rank of the expected fact, full top-k ordering, search_mode, slot/entity
classification, candidate-set sizes, per-hit field audit (score, base_score,
time_boost, observed_at, attributed_to, metadata keys/values incl. confidence
+ provenance) and rb_hits fields.

Dedicated legs (evidence for later phases):
  * speaker   — does any speaker:<id> tag exist in text mode? does
                speaker_filter narrow anything? (measured, not assumed)
  * temporal  — what expand_relative_dates / time_question_kind /
                query_dates actually do with the HU/EN temporal words
                (recognition measurement; no code changed)
  * renderers — vendor build_memory_context vs the app's two renderers
                (bridge _extract_memory_context / web _memory_context):
                exactly which fields survive into LLM context

Output: JSON report (stdout + --out). Exit 0 always — this is a measurement,
pass/fail assertions live in the gate (TASK2_GATE.md), not here.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "voicemem"))

E5_DIR = (
    REPO / "models" / "hf" / "models--intfloat--multilingual-e5-small"
    / "snapshots" / "614241f622f53c4eeff9890bdc4f31cfecc418b3"
)
MOCK_PORT = 18081
_USER_ID = "task2_baseline"
TODAY = date.today()  # sandbox local = Europe/Budapest per session config
YESTERDAY = TODAY - timedelta(days=1)
LAST_MONDAY = TODAY - timedelta(days=TODAY.weekday() + 7)
NEXT_MONDAY = TODAY + timedelta(days=(7 - TODAY.weekday()) % 7 or 7)

# ── controlled corpus ─────────────────────────────────────────────────────────
# 8 utterances → 8 facts, deterministic mock-extraction payloads. HU utterances
# keep HU fact text (the vendor's own extraction prompt: "Write each memory in
# the SAME language the user spoke it in"), so cross-language retrieval is real
# at the vector level. observed_at = the temporal ground truth.
CORPUS: list[dict[str, Any]] = [
    {
        "utterance": "A kedvenc kávém a hosszú kávé, minden reggel iszom egyet Budapesten.",
        "observed_at": str(TODAY),
        "payload": {
            "memory": [{"id": "t2-coffee", "text": "A felhasználó kedvenc kávéja a hosszú kávé, minden reggel iszik egyet Budapesten.",
                        "event": "ADD", "slot": "daily_life", "confidence": 0.95,
                        "entities": [{"name": "hosszú kávé", "entity_type": "preference", "role": "object"},
                                     {"name": "Budapest", "entity_type": "place", "role": "context"}]}],
            "emotion": "", "traits": []},
    },
    {
        "utterance": "I always get nervous before presentations",
        "observed_at": str(TODAY),
        "payload": {
            "memory": [{"id": "t2-nervous", "text": "The user always gets nervous before presentations",
                        "event": "ADD", "slot": "work", "confidence": 0.9,
                        "entities": [{"name": "presentations", "entity_type": "event", "role": "context"}]}],
            "emotion": "anxious", "traits": [{"slot": "情绪", "label": "gets nervous before presentations"}]},
    },
    {
        "utterance": "Tegnap este moziba mentem Annával.",
        "observed_at": str(YESTERDAY),
        "payload": {
            "memory": [{"id": "t2-cinema", "text": "A felhasználó tegnap este moziba ment Annával.",
                        "event": "ADD", "slot": "relationships", "confidence": 0.95,
                        "entities": [{"name": "Anna", "entity_type": "person", "role": "companion"},
                                     {"name": "mozi", "entity_type": "place", "role": "object"}]}],
            "emotion": "happy", "traits": []},
    },
    {
        "utterance": "Holnap reggel 8-kor fogorvoshoz megyek.",
        "observed_at": str(TODAY),
        "payload": {
            "memory": [{"id": "t2-dentist", "text": "A felhasználó holnap reggel 8-kor fogorvoshoz megy.",
                        "event": "ADD", "slot": "health", "confidence": 0.95,
                        "entities": [{"name": "fogorvos", "entity_type": "person", "role": "object"}]}],
            "emotion": "", "traits": []},
    },
    {
        "utterance": "Múlt héten 10 kilométert futottam.",
        "observed_at": str(LAST_MONDAY),
        "payload": {
            "memory": [{"id": "t2-running", "text": "A felhasználó múlt héten 10 kilométert futott.",
                        "event": "ADD", "slot": "health", "confidence": 0.9,
                        "entities": [{"name": "10 kilométer", "entity_type": "measure", "role": "object"}]}],
            "emotion": "", "traits": []},
    },
    {
        "utterance": "Last week I finished the quarterly report for the Q3 audit.",
        "observed_at": str(LAST_MONDAY),
        "payload": {
            "memory": [{"id": "t2-report", "text": "The user finished the quarterly report for the Q3 audit last week",
                        "event": "ADD", "slot": "work", "confidence": 0.95,
                        "entities": [{"name": "quarterly report", "entity_type": "document", "role": "object"},
                                     {"name": "Q3 audit", "entity_type": "event", "role": "context"}]}],
            "emotion": "relieved", "traits": []},
    },
    {
        "utterance": "Jövő héten Párizsba utazom a családdal.",
        "observed_at": str(TODAY),
        "payload": {
            "memory": [{"id": "t2-paris", "text": "A felhasználó jövő héten Párizsba utazik a családjával.",
                        "event": "ADD", "slot": "goals", "confidence": 0.95,
                        "entities": [{"name": "Párizs", "entity_type": "place", "role": "destination"},
                                     {"name": "család", "entity_type": "person", "role": "companion"}]}],
            "emotion": "excited", "traits": []},
    },
    {
        "utterance": "I dislike long meetings",
        "observed_at": str(TODAY),
        "payload": {
            "memory": [{"id": "t2-dislike", "text": "The user dislikes long meetings",
                        "event": "ADD", "slot": "work", "confidence": 0.9,
                        "entities": [{"name": "long meetings", "entity_type": "event", "role": "object"}]}],
            "emotion": "", "traits": [{"slot": "喜好与厌恶", "label": "dislikes long meetings"}]},
    },
]

# (query, tag, expected fact substring or None, classifier answer)
QUERIES: list[tuple[str, str, str | None, dict[str, Any]]] = [
    ("Milyen kávét iszom reggel?", "HU-simple", "kedvenc kávéja",
     {"slots": ["daily_life"], "entities": ["hosszú kávé"]}),
    ("how do I feel before presentations?", "EN-simple", "nervous before presentations",
     {"slots": ["work"], "entities": ["presentations"]}),
    ("what do I dislike at work?", "EN-simple", "dislikes long meetings",
     {"slots": ["work"], "entities": ["long meetings"]}),
    ("kedvenc kávé minden reggel", "HU-simple-keywords", "kedvenc kávéja",
     {"slots": ["daily_life"], "entities": ["hosszú kávé"]}),
    ("Mit csináltam tegnap este?", "HU-temporal", "moziba ment Annával",
     {"slots": ["relationships"], "entities": []}),
    ("What did I do yesterday evening?", "EN->HU-temporal", "moziba ment Annával",
     {"slots": ["relationships"], "entities": []}),
    ("Mi a programom holnap reggel?", "HU-temporal", "fogorvoshoz",
     {"slots": ["health"], "entities": ["fogorvos"]}),
    ("What are my plans for tomorrow morning?", "EN->HU-temporal", "fogorvoshoz",
     {"slots": ["health"], "entities": ["fogorvos"]}),
    ("Múlt héten mit futottam?", "HU-temporal", "10 kilométert futott",
     {"slots": ["health"], "entities": []}),
    ("Mit csináltam múlt héten a Q3 audit kapcsán?", "HU->EN-multi", "quarterly report",
     {"slots": ["work"], "entities": ["Q3 audit"]}),
    ("Which report did I finish last week?", "EN-multi", "quarterly report",
     {"slots": ["work"], "entities": ["quarterly report"]}),
    ("Jövő héten hová utazom?", "HU-temporal", "Párizsba utazik",
     {"slots": ["goals"], "entities": ["Párizs"]}),
    ("Mit csináltam ma reggel?", "HU-temporal-recognition", None,
     {"slots": ["daily_life"], "entities": []}),
    ("Milyen hosszú értekezleteket nem szeretek?", "HU->EN-simple", "dislikes long meetings",
     {"slots": ["work"], "entities": ["long meetings"]}),
    ("Mit csináltam tegnap este Annával?", "HU-multi-entity", "moziba ment Annával",
     {"slots": ["relationships"], "entities": ["Anna"]}),
    ("qwertyuiop asdfgh zxcvbn", "EMPTY-control", None,
     {"slots": ["daily_life"], "entities": []}),
]

TEMPORAL_WORDS = [
    "ma", "tegnap", "holnap", "tegnap este", "ma reggel", "múlt héten",
    "jövő héten", "yesterday", "today", "tomorrow", "yesterday evening",
    "last week", "next week", "Mit csináltam tegnap este?",
    "What did I do yesterday evening?",
]

_INNER_OS_REPLY = ("[curious] A felhasználó most osztott meg valamit, ami "
                   "láthatóan fontos neki.")


def _route(messages: list[dict[str, Any]]) -> dict[str, Any]:
    system = " ".join(str(m.get("content", "")) for m in messages
                      if m.get("role") == "system")
    user = " ".join(str(m.get("content", "")) for m in messages
                    if m.get("role") != "system")
    everything = system + "\n" + user

    if "memory manager" in system:            # conflict resolver: no changes
        return {"memory": []}
    if "memory router" in system:              # query classifier
        for query, _tag, _exp, ans in QUERIES:
            if query.strip() in user.strip():
                return ans
        return {"slots": ["daily_life"], "entities": []}
    if "empathetic AI assistant" in system:    # right-brain inner-OS
        return {"text": _INNER_OS_REPLY}
    if "用户说了这句话" in user:                # left-brain LLM slot re-tagging
        # A real LLM answers this with the 1-2 correct base slots; without
        # this route the embedding-written all-7-slot tags survive and the
        # slot filter degenerates to a no-op (measured: every memory tagged
        # into every slot at ~0.75). Route it to the corpus payload's slot.
        for entry in CORPUS:
            if entry["utterance"] in user:
                return {"slots": [entry["payload"]["memory"][0]["slot"]]}
        return {"slots": []}
    for entry in CORPUS:                       # fact extraction by utterance
        if entry["utterance"] in user or entry["utterance"] in everything:
            return entry["payload"]
    if '"items"' in everything and "喜好与厌恶" in everything:
        return {"items": []}                   # standalone trait extraction
    # anything else: empty merged shape
    return {"memory": [], "emotion": "", "traits": []}


class _MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        pass  # silent — the wiretap value is in the report, not stdout noise

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = _route(body.get("messages") or [])
        self._json(200, {
            "id": f"chatcmpl-t2-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-llm"),
            "system_fingerprint": "task2-baseline-mock",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": json.dumps(payload, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

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


def _start_mock(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), _MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ── field audit ───────────────────────────────────────────────────────────────

def _hit_audit(h: Any) -> dict[str, Any]:
    md = dict(getattr(h, "metadata", {}) or {})
    return {
        "memory_id": getattr(h, "memory_id", ""),
        "text": getattr(h, "text", ""),
        "score": round(float(getattr(h, "score", 0.0) or 0.0), 4),
        "base_score": round(float(getattr(h, "base_score", 0.0) or 0.0), 4),
        "time_boost": bool(getattr(h, "time_boost", False)),
        "observed_at": getattr(h, "observed_at", "") or "",
        "attributed_to": getattr(h, "attributed_to", "") or "",
        "metadata_keys": sorted(md.keys()),
        "metadata": {k: (str(v)[:120] if not isinstance(v, (int, float, bool))
                         else v) for k, v in md.items()},
    }


def _rb_audit(r: Any) -> dict[str, Any]:
    md = dict(getattr(r, "metadata", {}) or {})
    return {
        "source": getattr(r, "source", ""),
        "priority": round(float(getattr(r, "priority", 0.0) or 0.0), 4),
        "content": str(getattr(r, "content", ""))[:160],
        "metadata_keys": sorted(md.keys()),
    }


def _measure(fn, runs: int = 5) -> tuple[Any, list[float]]:
    fn()  # warmup (loads models/stores on the first call)
    lat, result = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        lat.append((time.perf_counter() - t0) * 1000.0)
    return result, lat


def _lat_stats(lat: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(lat), 1),
        "median_ms": round(statistics.median(lat), 1),
        "min_ms": round(min(lat), 1),
        "max_ms": round(max(lat), 1),
    }


def _result_summary(result: Any, expected: str | None) -> dict[str, Any]:
    hits = list(getattr(result, "hits", []) or [])
    texts = [getattr(h, "text", "") for h in hits]
    rank = None
    if expected:
        for i, t in enumerate(texts, 1):
            if expected in t:
                rank = i
                break
    cls = getattr(result, "classification", None)
    timing = dict(getattr(result, "timing", {}) or {})
    return {
        "hits": len(hits),
        "rb_hits": len(getattr(result, "rb_hits", []) or []),
        "rank_of_expected": rank,
        "hit_texts": texts,
        "search_mode": getattr(result, "search_mode", ""),
        "slots": list(getattr(cls, "slots", []) or []),
        "entities": list(getattr(cls, "entities", []) or []),
        "slot_mem_ids": len(getattr(result, "slot_mem_ids", []) or []),
        "final_candidate_ids": len(getattr(result, "final_candidate_ids", []) or []),
        "timing": {k: round(v, 1) for k, v in timing.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/verification_evidence/task2_retrieval_baseline.json")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--port", type=int, default=MOCK_PORT)
    args = ap.parse_args()

    os.environ.update({
        "OPENAI_BASE_URL": f"http://127.0.0.1:{args.port}/v1",
        "OPENAI_API_KEY": "task2-baseline-not-needed",
        "OPENAI_MODEL": "mock-llm",
        "OPENAI_CHAT_MODEL": "mock-llm",
        "VOICEMEM_E5_MODEL": str(E5_DIR),
        "VOICEMEM_EMBED_DIM": "384",
        "VOICEMEM_VERBOSE": "0",
    })

    srv = _start_mock(args.port)
    report: dict[str, Any] = {
        "schema": "task2-retrieval-baseline/1",
        "repo_state": {"version": "0.5.1", "vendor_pin": "e8384e087bd2f44eb05fc7ae1a3c525ea8244179"},
        "today": str(TODAY), "yesterday": str(YESTERDAY),
        "last_monday": str(LAST_MONDAY), "next_monday": str(NEXT_MONDAY),
        "runs_per_measurement": args.runs,
        "corpus": [{"utterance": c["utterance"], "observed_at": c["observed_at"],
                    "fact": c["payload"]["memory"][0]["text"],
                    "slot": c["payload"]["memory"][0]["slot"]}
                   for c in CORPUS],
        "path_a": [], "path_b": [], "classify_only": [],
        "speaker_leg": {}, "temporal_leg": {}, "renderer_leg": {},
    }

    from voicemem import VoiceMem
    from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder

    root = tempfile.mkdtemp(prefix="task2_baseline_")
    vm = VoiceMem(mode="text_mode", memory_root=root, user_id=_USER_ID,
                  embedding=lambda: LocalE5Embedder())

    # ── ingest the controlled corpus (synchronous; deterministic) ─────────
    t0 = time.perf_counter()
    for entry in CORPUS:
        res = vm.ingest(entry["utterance"], observed_at=entry["observed_at"])
        time.sleep(0.05)
    report["ingest"] = {
        "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
        "facts_expected": len(CORPUS),
    }

    # ── per-query, both paths ──────────────────────────────────────────────
    for query, tag, expected, _ans in QUERIES:
        row: dict[str, Any] = {"query": query, "tag": tag, "expected": expected}

        # PATH A — the primitive the app calls
        res_a, lat_a = _measure(lambda: vm.search(query), args.runs)
        row["path_a"] = {"latency": _lat_stats(lat_a), **_result_summary(res_a, expected)}
        row["path_a"]["hit_fields"] = [_hit_audit(h) for h in (res_a.hits or [])[:3]]
        row["path_a"]["rb_fields"] = [_rb_audit(r) for r in (res_a.rb_hits or [])[:2]]

        # PATH B — classify → search(slots, entities)
        def _rich():
            cls = vm.classify(query)
            return cls, vm.search(query, slots=cls.slots, entities=cls.entities)

        (cls_b, res_b), lat_b = _measure(_rich, args.runs)
        row["path_b"] = {
            "latency": _lat_stats(lat_b),
            "classified_slots": list(cls_b.slots or []),
            "classified_entities": list(cls_b.entities or []),
            **_result_summary(res_b, expected),
        }
        row["path_b"]["hit_fields"] = [_hit_audit(h) for h in (res_b.hits or [])[:3]]
        row["path_b"]["rb_fields"] = [_rb_audit(r) for r in (res_b.rb_hits or [])[:2]]

        # classify latency on its own (one call per query, warm)
        t1 = time.perf_counter()
        vm.classify(query)
        report["classify_only"].append(
            {"query": query, "classify_ms": round((time.perf_counter() - t1) * 1000, 1)})

        report["path_a"].append(row["path_a"] | {"query": query, "tag": tag})
        report["path_b"].append(row["path_b"] | {"query": query, "tag": tag})
        print(f"[t2] {tag:26} {query[:44]:46} A: {row['path_a']['hits']} hits "
              f"rank={row['path_a']['rank_of_expected']} | B: {row['path_b']['hits']} hits "
              f"rank={row['path_b']['rank_of_expected']}", flush=True)

    # ── speaker leg ────────────────────────────────────────────────────────
    try:
        store = vm._o._get_repo()._cognitive_store
        tags: dict[str, int] = {}
        for probe in ("speaker:user", "speaker:Speaker 0", "speaker:person_probe",
                      "scene:unknown", "daily_life", "work", "health",
                      "relationships", "goals"):
            ids = store.memory_ids_for_slots_v2(_USER_ID, [probe]) if probe != "scene:unknown" else []
            tags[probe] = len(set(ids)) if ids else 0
        base = vm.search("Milyen kávét iszom reggel?")
        filt = vm.search("Milyen kávét iszom reggel?", speaker_filter="user")
        report["speaker_leg"] = {
            "cognitive_slot_tag_counts": tags,
            "search_no_filter_hits": [h.text for h in base.hits],
            "search_speaker_filter_user_hits": [h.text for h in filt.hits],
            "filter_narrowed": len(filt.hits) != len(base.hits),
            "api_surface": "vm.search(query, speaker_filter=...) exists; "
                           "tags are written by the AUDIO path (voiceprint) only",
        }
    except Exception as exc:  # noqa: BLE001 - measurement must not crash
        report["speaker_leg"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ── temporal recognition leg (no code changed) ─────────────────────────
    from voicemem.leftbrain.time_expand import expand_relative_dates
    from voicemem.leftbrain.local_memory_store import (
        time_question_kind, query_dates as q_dates,
    )
    report["temporal_leg"] = {
        "expand_relative_dates": {w: expand_relative_dates(w) for w in TEMPORAL_WORDS},
        "time_question_kind": {w: time_question_kind(w) for w in TEMPORAL_WORDS},
        "query_dates": {w: sorted(q_dates(w)) for w in TEMPORAL_WORDS},
        "note": "expand_relative_dates is called INSIDE Search() on every query; "
                "time_question_kind gates _widen_for_time_question",
    }

    # ── renderer leg (what actually reaches the LLM context) ───────────────
    # The three REAL renderers are: vendor memory_api.build_memory_context,
    # app.voicemem_bridge._extract_memory_context (CLI path) and
    # app.web_server._memory_context (web path). The two app renderers are
    # byte-identical in behaviour (top-5 facts + top-3 cleaned rb notes, 1200
    # chars), so the bridge one is imported and the web one is reproduced
    # verbatim (importing web_server would pull FastAPI/uvicorn into the
    # measurement env).
    from voicemem.memory_api import build_memory_context
    from app.voicemem_bridge import _extract_memory_context

    def _app_web_render(result: Any) -> str:  # app/web_server.py _memory_context
        if result is None:
            return ""
        parts: list[str] = []
        hits = getattr(result, "hits", None) or []
        for h in hits[:5]:
            t = (getattr(h, "text", "") or "").strip()
            if t:
                parts.append(f"- {t}")
        rb = getattr(result, "rb_hits", None) or []
        for h in rb[:3]:
            from app.voicemem_bridge import clean_rb_content
            t = clean_rb_content(getattr(h, "content", "") or "").strip()
            if t:
                parts.append(f"- {t}")
        return "\n".join(parts)[:1200]

    for query, tag in (("Mit csináltam tegnap este?", "HU-temporal"),
                       ("Which report did I finish last week?", "EN-multi"),
                       ("how do I feel before presentations?", "EN-simple-trait")):
        res = vm.search(query)
        report["renderer_leg"][tag] = {
            "query": query,
            "vendor_build_memory_context": build_memory_context(res),
            "app_bridge_render": _extract_memory_context(res),
            "app_web_render": _app_web_render(res),
        }

    srv.shutdown()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[t2] report -> {out} ({out.stat().st_size} B)", flush=True)

    # keep the console honest about the headline numbers
    targeted = sum(1 for _q, _t, exp, _a in QUERIES if exp)
    a_ok = sum(1 for r, (_q, _t, exp, _a) in zip(report["path_a"], QUERIES)
               if exp and r["rank_of_expected"] == 1)
    b_ok = sum(1 for r, (_q, _t, exp, _a) in zip(report["path_b"], QUERIES)
               if exp and r["rank_of_expected"] == 1)
    print(f"[t2] rank-1: path A {a_ok}/{targeted}, path B {b_ok}/{targeted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
