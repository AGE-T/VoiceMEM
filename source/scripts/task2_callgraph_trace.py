#!/usr/bin/env python3
"""TASK 2 (PHASE 2) — runtime call-graph trace of both retrieval paths.

Follows the ACTUAL runtime call graph (sys.setprofile call/return events), not
class names, for:

  PATH A  vm.search(query)
  PATH B  vm.classify(query) + vm.search(query, slots=…, entities=…)

on the controlled TASK 2 corpus. Produces the ordered vendor+app function-call
sequence for each path, the exact divergence point, and per-stage annotations
(classification / slot extraction / embedding / vector search / filters /
ranking / boosts / speaker / language / dedup / rendering / final object).

Measurement-only: no implementation is modified. Output: JSON (--out).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "voicemem"))

E5_DIR = (
    REPO / "models" / "hf" / "models--intfloat--multilingual-e5-small"
    / "snapshots" / "614241f622f53c4eeff9890bdc4f31cfecc418b3"
)
_PORT = 18083

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("t2base", REPO / "scripts" / "task2_retrieval_baseline.py")
t2base = importlib.util.module_from_spec(spec)
# The baseline module only executes main() under __main__, import is safe:
spec.loader.exec_module(t2base)
from http.server import ThreadingHTTPServer  # noqa: E402

TRACE_QUERY = "Mit csináltam tegnap este Annával?"


def _start_mock(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), t2base._MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class _CallGraph:
    """sys.setprofile-based call recorder for the main thread only."""

    def __init__(self) -> None:
        self.events: list[dict[str, str]] = []
        self._tid = threading.get_ident()

    def _on_event(self, frame, event: str, arg: Any) -> None:
        if threading.get_ident() != self._tid or event != "call":
            return
        code = frame.f_code
        fn = code.co_filename
        if "/vendor/voicemem/" not in fn and "/app/" not in fn:
            return
        try:
            rel = fn.split("/vendor/voicemem/voicemem/")[-1] if "/vendor/voicemem/voicemem/" in fn else fn.split("/app/")[-1]
        except Exception:
            rel = fn
        self.events.append({"fn": f"{rel}:{code.co_name}"})

    def __enter__(self) -> "_CallGraph":
        sys.setprofile(self._on_event)
        return self

    def __exit__(self, *exc: Any) -> None:
        sys.setprofile(None)

    def collapsed(self) -> list[str]:
        out: list[str] = []
        for e in self.events:
            if not out or out[-1] != e["fn"]:
                out.append(e["fn"])
        return out


_STAGE_HINTS = [
    ("facade/core", ["core.py:search", "core.py:classify"]),
    ("orchestration", ["orchestrator.py:Search", "orchestrator.py:Classify"]),
    ("temporal-expansion", ["time_expand.py:expand_relative_dates"]),
    ("scene-inference", ["scene_classifier.py:infer_scene_from_text"]),
    ("leftbrain-search", ["brain.py:search"]),
    ("slot-filter", ["brain.py:SearchCogGraph", "store_v2.py:memory_ids_for_slots_v2"]),
    ("entity-narrow", ["brain.py:_search_data_impl", "store_v2.py:find_entities_by_name_fuzzy",
                       "store_v2.py:memory_ids_for_entities", "store_v2.py:neighbor_entity_ids"]),
    ("time-widen", ["brain.py:_widen_for_time_question", "local_memory_store.py:time_question_kind",
                    "mem0_backend_store.py:memory_ids_with_time_expr"]),
    ("classifier", ["brain.py:Classify", "query_slot_classifier.py:classify",
                    "query_slot_classifier.py:resolved_model"]),
    ("rank", ["brain.py:Rank", "memory_repository.py:search",
              "mem0_backend_store.py:search", "local_memory_store.py:_lexical_time_bonus",
              "local_memory_store.py:date_overlap_bonus", "local_memory_store.py:query_dates"]),
    ("dedupe", ["brain.py:_dedupe_near"]),
    ("heat", ["store_v2.py:record_memory_hits", "store.py:record_memory_hits"]),
    ("rightbrain", ["rightbrain/brain.py:search", "rightbrain/store.py:search_by_anchors",
                    "rightbrain/store.py:search_global", "rightbrain/traits_store.py:search"]),
    ("subgraph-accounting", ["brain.py:_record_subgraph_activation"]),
    ("result-assembly", ["orchestrator.py:Search"]),
    ("rendering", ["memory_api.py:build_memory_context", "voicemem_bridge.py:_extract_memory_context"]),
]


def _stages(seq: list[str]) -> list[dict[str, Any]]:
    """Annotate which sequence entries belong to which pipeline stage."""
    by_name: dict[str, set[str]] = {}
    for stage, markers in _STAGE_HINTS:
        names = by_name.setdefault(stage, set())
        names.update(m.split(":")[-1] for m in markers)
    hits = []
    for i, fn in enumerate(seq):
        name = fn.split(":")[-1]
        for stage, names in by_name.items():
            if name in names:
                hits.append({"idx": i, "stage": stage, "fn": fn})
                break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/verification_evidence/task2_callgraph.json")
    args = ap.parse_args()

    os.environ.update({
        "OPENAI_BASE_URL": f"http://127.0.0.1:{_PORT}/v1",
        "OPENAI_API_KEY": "task2-trace-not-needed",
        "OPENAI_MODEL": "mock-llm",
        "OPENAI_CHAT_MODEL": "mock-llm",
        "VOICEMEM_E5_MODEL": str(E5_DIR),
        "VOICEMEM_EMBED_DIM": "384",
        "VOICEMEM_VERBOSE": "0",
    })
    srv = _start_mock(_PORT)

    from voicemem import VoiceMem
    from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder
    from voicemem.memory_api import build_memory_context

    vm = VoiceMem(mode="text_mode", memory_root=tempfile.mkdtemp(prefix="task2_cg_"),
                  user_id="task2_callgraph", embedding=lambda: LocalE5Embedder())
    for entry in t2base.CORPUS:
        vm.ingest(entry["utterance"], observed_at=entry["observed_at"])
        time.sleep(0.05)

    # warm all lazy singletons so the traced call is the steady-state path
    vm.search(TRACE_QUERY)
    vm.classify(TRACE_QUERY)

    report: dict[str, Any] = {"schema": "task2-callgraph/1", "query": TRACE_QUERY}

    with _CallGraph() as cg:
        result_a = vm.search(TRACE_QUERY)
    seq_a = cg.collapsed()
    report["path_a"] = {"call_count": len(cg.events), "sequence": seq_a,
                        "stages": _stages(seq_a)}

    with _CallGraph() as cg:
        cls = vm.classify(TRACE_QUERY)
        result_b = vm.search(TRACE_QUERY, slots=cls.slots, entities=cls.entities)
    seq_b = cg.collapsed()
    report["path_b"] = {"call_count": len(cg.events), "sequence": seq_b,
                        "stages": _stages(seq_b)}

    # exact divergence point
    div = next((i for i, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y),
               min(len(seq_a), len(seq_b)))
    common = seq_a[:div]
    report["divergence"] = {
        "common_prefix_len": div,
        "common_prefix": common,
        "a_only": seq_a[div:],
        "b_only": seq_b[div:],
        "note": "path A = vm.search(q); path B = classify(q) + search(q, slots, entities)",
    }
    report["classified"] = {"slots": list(cls.slots), "entities": list(cls.entities)}
    report["result_shape"] = {
        "fields": [f for f in dir(result_a) if not f.startswith("_")],
        "hit_fields": [f for f in dir((result_a.hits or [None])[0]) if not f.startswith("_")] if result_a.hits else [],
        "mode_a": result_a.search_mode, "mode_b": result_b.search_mode,
        "rendered_vendor_chars": len(build_memory_context(result_a)),
    }

    srv.shutdown()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cg] A: {report['path_a']['call_count']} calls, B: {report['path_b']['call_count']} calls; "
          f"divergence at seq index {div}; report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
