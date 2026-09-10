#!/usr/bin/env python3
"""VoiceMem runtime sanity + retrieval baseline (v0.5.0 Phases 6/9).

Runs the REAL controlled VoiceMem pipeline end-to-end against whatever
OpenAI-compatible endpoint ``OPENAI_BASE_URL`` points at (llama-server on
the target machine, tests/e2e_mock_llm.py in the sandbox) with the REAL
local multilingual E5 and REAL mem0/Qdrant storage:

  Phase 6 sanity:
    1. ingest a simple factual memory (HU)
    2. retrieve it (HU query)
    3. ingest a repeated related memory (EN, same topic)
    4. retrieve both (cross-language query)
    5. create traits (HU + EN)
    6. verify trait embeddings EXIST (non-NULL)
    7. verify the embedding dimension (384)
    8. Hungarian query
    9. English query
    10. cross-language related query

  Phase 9 baseline (measurement, no default changes):
    - per-query: search_mode, slot classification, entity narrowing
      (slot_mem_ids / final_candidate_ids), hit count, ranks of the
      expected memories, rb trait hits, latency ms
    - the renderer information-loss measurement (Phase 11): what
      SearchResult carries vs what build_memory_context + the app's
      renderer expose

Usage (sandbox, from the agent root with .venv installed):
    .venv/bin/python tests/e2e_mock_llm.py --port 18080 &   # mock LLM
    OPENAI_BASE_URL=http://127.0.0.1:18080/v1 OPENAI_API_KEY=dummy \
    OPENAI_MODEL=mock-llm VOICEMEM_E5_MODEL=models/embedding/multilingual-e5-small \
    .venv/bin/python scripts/verify_voicemem_runtime.py --memory-root /tmp/vmrt

Exit 0 = all checks pass; 1 = failure (printed with [FAIL]).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "voicemem"))

EXPECTED_DIM = 384

HU_FACT = "A kedvenc kávém a hosszú kávé, minden reggel iszom egyet Budapesten."
EN_FACT = "My favourite coffee is a long black, I drink one every morning."
HU_TRAIT_SENTENCE = "Mindig ideges leszek prezentáció előtt."
EN_TRAIT_SENTENCE = "I dislike long meetings."

QUERIES = [
    ("HU", "Milyen kávét iszom reggel?"),
    ("EN", "What coffee do I drink every morning?"),
    ("X-LANG", "kedvenc kávé minden reggel"),
    ("X-LANG", "my favourite morning coffee Budapest"),
    ("TRAIT-HU", "hogyan érzem magam prezentáció előtt?"),
    ("TRAIT-EN", "how do I feel before presentations?"),
    ("TRAIT-EN", "what do I dislike at work?"),
]


def _p(ok: bool, msg: str, failures: list[str]) -> bool:
    print(("  [ok]   " if ok else "  [FAIL] ") + msg)
    if not ok:
        failures.append(msg)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="VoiceMem runtime sanity + baseline")
    ap.add_argument("--memory-root", default="",
                    help="memory root (default: a fresh temp dir)")
    ap.add_argument("--user", default="e2e_user")
    ap.add_argument("--keep", action="store_true",
                    help="keep the memory root (default: temp dir removed)")
    args = ap.parse_args()

    failures: list[str] = []

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    print("VoiceMem runtime sanity + retrieval baseline (controlled fork)")
    print(f"  base_url   : {base_url or '<unset>'}")
    print(f"  e5 model   : {os.environ.get('VOICEMEM_E5_MODEL', '<resolver default>')}")
    mem_root = Path(args.memory_root) if args.memory_root else Path(
        tempfile.mkdtemp(prefix="vm_runtime_"))
    print(f"  memory root: {mem_root}")

    # -- construction --------------------------------------------------------
    import voicemem
    from voicemem.core import VoiceMem
    from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder

    print(f"  identity   : CONTROLLED_FORK={voicemem.CONTROLLED_FORK} "
          f"commit={voicemem.CONTROLLED_UPSTREAM_COMMIT}")
    _p(voicemem.CONTROLLED_FORK
       and voicemem.CONTROLLED_UPSTREAM_COMMIT
       == "e8384e087bd2f44eb05fc7ae1a3c525ea8244179",
       "runtime identity = controlled fork @ e8384e0", failures)

    t0 = time.perf_counter()
    vm = VoiceMem(
        mode="text_mode",
        user_id=args.user,
        base_url=base_url or None,
        memory_root=str(mem_root),
        embedding=lambda: LocalE5Embedder(),
    )
    print(f"  constructed in {time.perf_counter() - t0:.2f}s")

    # -- 1..4: ingest + retrieve facts --------------------------------------
    print("\n[ingest]")
    for label, sentence in (
        ("HU fact", HU_FACT),
        ("EN related fact", EN_FACT),
        ("HU trait sentence", HU_TRAIT_SENTENCE),
        ("EN trait sentence", EN_TRAIT_SENTENCE),
    ):
        t = time.perf_counter()
        vm.ingest(sentence)
        print(f"  ingest {label:<20} {time.perf_counter() - t:.2f}s")

    # -- 5..7: trait embeddings ---------------------------------------------
    print("\n[trait embeddings]")
    db_path = _find_db(mem_root)
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, slot, claim, embedding FROM rb_traits WHERE user_id=?",
        (args.user,)).fetchall()
    total = len(rows)
    nulls = [r for r in rows if r["embedding"] is None]
    dims = {}
    for r in rows:
        if r["embedding"] is not None:
            dims[len(r["embedding"]) // 4] = dims.get(len(r["embedding"]) // 4, 0) + 1
    _p(total >= 2, f"traits created ({total} >= 2)", failures)
    _p(len(nulls) == 0,
       f"NULL trait embeddings: {len(nulls)} (the v0.4.x defect is gone)", failures)
    _p(list(dims.keys()) == [EXPECTED_DIM],
       f"trait embedding dimensions {dims} == 384", failures)
    for r in rows:
        claim = r["claim"]
        print(f"    trait [{r['slot']}] {claim} "
              f"(dim {len(r['embedding']) // 4 if r['embedding'] else 'NULL'})")
    conn.close()

    # -- 8..10: queries (also the Phase 9 baseline) ---------------------------
    print("\n[retrieval baseline]")
    baseline: list[dict[str, Any]] = []
    expected_fragments = {
        "Milyen kávét iszom reggel?": ("coffee", "kávé"),
        "What coffee do I drink every morning?": ("coffee", "kávé"),
        "kedvenc kávé minden reggel": ("coffee", "kávé"),
        "my favourite morning coffee Budapest": ("coffee", "kávé"),
        "hogyan érzem magam prezentáció előtt?": ("nervous",),
        "how do I feel before presentations?": ("nervous",),
        "what do I dislike at work?": ("dislikes long meetings", "meetings"),
    }
    for tag, query in QUERIES:
        t = time.perf_counter()
        result = vm.search(query)
        dt_ms = (time.perf_counter() - t) * 1000.0
        hit_texts = [h.text for h in (result.hits or [])]
        rb_texts = [h.content for h in (result.rb_hits or [])]
        rank = None
        frags = expected_fragments.get(query)
        if frags:
            for i, text in enumerate(hit_texts):
                if any(f.lower() in text.lower() for f in frags):
                    rank = i + 1
                    break
        classification = getattr(result, "classification", None)
        entry = {
            "query": query, "tag": tag, "latency_ms": round(dt_ms, 1),
            "search_mode": getattr(result, "search_mode", ""),
            "slots": list(getattr(classification, "slots", []) or []),
            "classified_entities": list(getattr(classification, "entities", []) or []),
            "hits": len(hit_texts), "rank_of_expected": rank,
            "rb_hits": rb_texts[:5],
            "slot_mem_ids": len(getattr(result, "slot_mem_ids", None) or []),
            "final_candidate_ids": len(getattr(result, "final_candidate_ids", None) or []),
            "timing": getattr(result, "timing", None),
            "hit_texts": hit_texts[:5],
        }
        baseline.append(entry)
        print(f"  [{tag}] {query!r}")
        print(f"      {dt_ms:7.1f} ms  mode={entry['search_mode']} "
              f"hits={entry['hits']} rank={rank} "
              f"slots={entry['slots']} slot_ids={entry['slot_mem_ids']} "
              f"final_ids={entry['final_candidate_ids']}")
        if rb_texts:
            print(f"      rb: {rb_texts[:3]}")
        if hit_texts:
            print(f"      top: {hit_texts[0][:80]}")

    # sanity assertions on retrieval
    coffee_queries = [q for q in baseline if q["tag"] in ("HU", "EN", "X-LANG")]
    _p(all(q["hits"] > 0 for q in coffee_queries),
       "every HU/EN/cross-language coffee query returned hits", failures)
    _p(any(q["rank_of_expected"] == 1 for q in coffee_queries),
       "the expected coffee memory is rank-1 in at least one query", failures)
    trait_queries = [q for q in baseline if q["tag"].startswith("TRAIT")]
    _p(any(q["rb_hits"] for q in trait_queries),
       "right-brain trait hits returned for a trait query", failures)

    # -- Phase 9: PATH B (rich: Classify -> search with slots+entities) -------
    print("\n[retrieval baseline - PATH B: rich (Classify -> slots/entities)]")
    rich_baseline: list[dict[str, Any]] = []
    for tag, query in QUERIES:
        t = time.perf_counter()
        c = vm.classify(query)
        t_cls = (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        result = vm.search(query, slots=c.slots, entities=c.entities)
        dt_ms = (time.perf_counter() - t) * 1000.0
        hit_texts = [h.text for h in (result.hits or [])]
        rank = None
        frags = expected_fragments.get(query)
        if frags:
            for i, text in enumerate(hit_texts):
                if any(f.lower() in text.lower() for f in frags):
                    rank = i + 1
                    break
        entry = {
            "query": query, "tag": tag,
            "classify_ms": round(t_cls, 1), "search_ms": round(dt_ms, 1),
            "search_mode": getattr(result, "search_mode", ""),
            "slots": list(getattr(c, "slots", []) or []),
            "classified_entities": list(getattr(c, "entities", []) or []),
            "hits": len(hit_texts), "rank_of_expected": rank,
            "rb_hits": [h.content for h in (result.rb_hits or [])][:5],
            "slot_mem_ids": len(getattr(result, "slot_mem_ids", None) or []),
            "final_candidate_ids": len(getattr(result, "final_candidate_ids", None) or []),
            "hit_texts": hit_texts[:5],
        }
        rich_baseline.append(entry)
        print(f"  [{tag}] {query!r}")
        print(f"      cls={t_cls:6.1f} ms (slots={entry['slots']} ents={entry['classified_entities']}) "
              f"+ search={dt_ms:6.1f} ms  mode={entry['search_mode']} "
              f"hits={entry['hits']} rank={rank} "
              f"slot_ids={entry['slot_mem_ids']} final_ids={entry['final_candidate_ids']}")
    # the audit question, answered with numbers: does the PLAIN public
    # search() engage the graph narrowing? (baseline: it does NOT - the
    # caller must classify first; vm.search() alone = pure vector fallback)
    plain_modes = {q["search_mode"] for q in baseline}
    rich_modes = {q["search_mode"] for q in rich_baseline}
    print(f"  PATH A (plain vm.search) modes : {plain_modes} "
          f"(slot_ids {[q['slot_mem_ids'] for q in baseline]})")
    print(f"  PATH B (classify+search) modes: {rich_modes} "
          f"(slot_ids {[q['slot_mem_ids'] for q in rich_baseline]})")

    # -- renderer information loss (Phase 11 measurement) ---------------------
    print("\n[renderer information loss]")
    from voicemem.memory_api import build_memory_context
    sample = next((q for q in baseline if q["hits"]), None)
    if sample is None:
        _p(False, "no search result available for the renderer measurement",
           failures)
    else:
        # re-run the same query to keep the objects fresh
        result = vm.search(sample["query"])
        hit = (result.hits or [None])[0]
        available = {}
        if hit is not None:
            for field in ("memory_id", "text", "score", "base_score",
                          "time_boost", "observed_at", "attributed_to",
                          "metadata"):
                available[field] = getattr(hit, field, None)
        rendered = build_memory_context(result)
        lost = [f for f, v in available.items()
                if v not in (None, "", {}) and f not in ("text",)
                and (f != "observed_at" or not rendered)]
        print(f"    SearchResult.hit fields: "
              + ", ".join(f"{k}={'set' if v not in (None, '', {}) else '-'}"
                          for k, v in available.items()))
        print(f"    rendered context ({len(rendered)} chars):")
        for line in rendered.splitlines()[:8]:
            print(f"      | {line}")
        print(f"    note: score/base_score/time_boost/attributed_to/metadata "
              f"are NOT rendered (information loss to measure)")

    # -- dump the baseline as JSON -------------------------------------------
    out = Path(tempfile.gettempdir()) / "voicemem_retrieval_baseline.json"
    out.write_text(json.dumps(
        {"path_a_plain_search": baseline, "path_b_rich": rich_baseline},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nbaseline JSON (paths A+B): {out}")

    if not args.keep and not args.memory_root:
        import shutil
        shutil.rmtree(mem_root, ignore_errors=True)

    if failures:
        print(f"\nRUNTIME SANITY FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nRUNTIME SANITY OK: ingest + retrieve + traits (embedded, 384-d) "
          "+ HU/EN/cross-language retrieval all work on the controlled fork")
    return 0


def _find_db(memory_root: Path) -> Path:
    want = memory_root / f"{memory_root.name}.sqlite"
    if want.exists():
        return want
    candidates = sorted(memory_root.glob("*.sqlite"))
    if not candidates:
        raise SystemExit(f"no sqlite found under {memory_root}")
    return candidates[0]


if __name__ == "__main__":
    raise SystemExit(main())
