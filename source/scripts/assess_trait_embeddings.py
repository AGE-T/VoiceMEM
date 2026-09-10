#!/usr/bin/env python3
"""Trait embedding status assessment (v0.5.0, Phase 8 tool). READ-ONLY.

Reports the health of the right-brain trait table (``rb_traits``) in the
memory sqlite:

  * total traits (per user, per slot)
  * NULL embeddings      — rows written while the embedder was failing
                           (the v0.4.x defect: silent NULL writes)
  * non-NULL embeddings  — usable for semantic retrieval/merge
  * dimension histogram  — float32 vector lengths (len(blob)/4)
  * unexpected dimensions — anything != the expected local-E5 dimension
                           (default 384; ``VOICEMEM_EMBED_DIM`` overrides)

``--dry-run-backfill`` additionally lists exactly which trait rows WOULD be
re-embedded (NULL or wrong dimension) — a PLAN, never a mutation. No mass
data change is ever performed by this tool. Actual backfill is a separate,
explicitly approved operation.

Read-only guarantees: the sqlite is opened with ``file:...?mode=ro``; no
write transaction is ever started; the tool works even when the Qdrant
store is locked by a running agent (it never touches the vector store).

Usage:
    python scripts/assess_trait_embeddings.py                     # human report
    python scripts/assess_trait_embeddings.py --user webspace_demo
    python scripts/assess_trait_embeddings.py --json              # machine report
    python scripts/assess_trait_embeddings.py --dry-run-backfill  # plan only
    python scripts/assess_trait_embeddings.py --strict            # exit 1 if NULLs

Exit codes: 0 = assessment completed (see --strict for NULL gating);
            2 = memory database not found; 1 = unexpected error.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMORY_ROOT = ROOT / "memory"
DEFAULT_DIM = 384


def _expected_dim() -> int:
    try:
        return int(os.environ.get("VOICEMEM_EMBED_DIM", "") or DEFAULT_DIM)
    except ValueError:
        return DEFAULT_DIM


def _find_db(memory_root: Path) -> Path | None:
    """Mirror the vendor's space._pick logic for the single sqlite."""
    if not memory_root.is_dir():
        return None
    want = memory_root / f"{memory_root.name}.sqlite"
    if want.exists():
        return want
    candidates = sorted(p for p in memory_root.glob("*.sqlite") if p.is_file())
    return candidates[0] if candidates else None


def _open_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def assess(db_path: Path, user: str | None, expected_dim: int) -> dict[str, Any]:
    conn = _open_ro(db_path)
    try:
        has_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rb_traits'"
        ).fetchone()
        if not has_table:
            return {"database": str(db_path), "rb_traits_table": False, "note":
                    "rb_traits table does not exist (no right-brain traits written yet)"}

        where = "WHERE user_id=?" if user else ""
        params: tuple[Any, ...] = (user,) if user else ()
        rows = conn.execute(
            f"SELECT id, user_id, slot, claim, embedding, updated_at "
            f"FROM rb_traits {where}", params
        ).fetchall()

        per_user: dict[str, dict[str, int]] = {}
        per_slot: dict[str, dict[str, int]] = {}
        dim_counter: Counter[int] = Counter()
        null_rows: list[dict[str, Any]] = []
        stale_rows: list[dict[str, Any]] = []  # non-NULL but wrong dim
        ok_rows = 0

        for r in rows:
            blob = r["embedding"]
            uid = r["user_id"]
            slot = r["slot"]
            if blob is None:
                dim = 0
            else:
                dim = len(blob) // 4  # float32
            for bucket, key in ((per_user, uid), (per_slot, slot)):
                entry = bucket.setdefault(key, {"total": 0, "null": 0, "ok": 0, "stale": 0})
                entry["total"] += 1
                if blob is None:
                    entry["null"] += 1
                elif dim == expected_dim:
                    entry["ok"] += 1
                else:
                    entry["stale"] += 1
            if blob is None:
                null_rows.append({"id": r["id"], "user_id": uid, "slot": slot,
                                  "claim": r["claim"]})
            elif dim != expected_dim:
                stale_rows.append({"id": r["id"], "user_id": uid, "slot": slot,
                                   "claim": r["claim"], "dim": dim})
            else:
                ok_rows += 1
            dim_counter[dim if blob is not None else -1] += 1

        return {
            "database": str(db_path),
            "rb_traits_table": True,
            "filter_user": user,
            "expected_dim": expected_dim,
            "total_traits": len(rows),
            "null_embeddings": len(null_rows),
            "non_null_embeddings": len(rows) - len(null_rows),
            "healthy_embeddings": ok_rows,
            "stale_dimension_embeddings": len(stale_rows),
            "dimension_histogram": {str(k): v for k, v in sorted(dim_counter.items())},
            "unexpected_dimensions": sorted(
                int(k) for k in dim_counter if k not in (expected_dim, -1)
            ),
            "per_user": per_user,
            "per_slot": per_slot,
            "backfill_plan": {
                "would_reembed_null": null_rows,
                "would_reembed_stale_dim": stale_rows,
                "note": ("DRY RUN ONLY - no data was modified. Rows with NULL "
                         "or wrong-dimension embeddings would be re-embedded "
                         "with the local multilingual E5 in an explicitly "
                         "approved backfill operation."),
            },
        }
    finally:
        conn.close()


def _print_human(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("Trait embedding status (rb_traits) - READ-ONLY assessment")
    print("=" * 72)
    print(f"database        : {report['database']}")
    if not report.get("rb_traits_table"):
        print(f"status          : {report.get('note')}")
        return
    print(f"filter (user)   : {report.get('filter_user') or '(all users)'}")
    print(f"expected dim    : {report['expected_dim']} (local multilingual E5)")
    print("-" * 72)
    print(f"total traits    : {report['total_traits']}")
    print(f"NULL embeddings : {report['null_embeddings']}"
          + ("   <-- silent-defect rows" if report["null_embeddings"] else ""))
    print(f"non-NULL        : {report['non_null_embeddings']}")
    print(f"  healthy (dim) : {report['healthy_embeddings']}")
    print(f"  stale (dim)   : {report['stale_dimension_embeddings']}")
    print(f"dimension hist  : {report['dimension_histogram']}")
    print(f"unexpected dims : {report['unexpected_dimensions'] or 'none'}")
    print("-" * 72)
    for uid, stats in sorted(report["per_user"].items()):
        print(f"user {uid:<18} total={stats['total']:<5} null={stats['null']:<5} "
              f"ok={stats['ok']:<5} stale={stats['stale']}")
    for slot, stats in sorted(report["per_slot"].items()):
        print(f"slot {slot:<12} total={stats['total']:<5} null={stats['null']:<5} "
              f"ok={stats['ok']:<5} stale={stats['stale']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only rb_traits embedding assessment")
    ap.add_argument("--root", default=str(ROOT), help="agent repo root")
    ap.add_argument("--memory-root", default="",
                    help="memory root (default <root>/memory)")
    ap.add_argument("--user", default="", help="restrict to one user_id")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON report")
    ap.add_argument("--dry-run-backfill", action="store_true",
                    help="include the backfill PLAN (never mutates anything)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when NULL embeddings exist (for gates)")
    args = ap.parse_args()

    root = Path(args.root)
    memory_root = Path(args.memory_root) if args.memory_root else root / "memory"
    db_path = _find_db(memory_root)
    if db_path is None:
        msg = f"no memory sqlite found under {memory_root}"
        if args.json:
            print(json.dumps({"error": msg, "memory_root": str(memory_root)}, indent=2))
        else:
            print(msg)
        return 2

    try:
        report = assess(db_path, args.user or None, _expected_dim())
    except sqlite3.Error as exc:
        print(f"sqlite error: {exc}", file=sys.stderr)
        return 1

    # the plan is always computed (cheap); it is only PRINTED on request or
    # with --json (where it is part of the report anyway)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report)
        if args.dry_run_backfill:
            plan = report["backfill_plan"]
            print("-" * 72)
            print(f"BACKFILL DRY RUN (no mutation): {len(plan['would_reembed_null'])} NULL + "
                  f"{len(plan['would_reembed_stale_dim'])} stale-dim rows would be re-embedded")
            for row in plan["would_reembed_null"][:20]:
                print(f"  [null]  {row['user_id']}/{row['slot']}: {row['claim'][:60]}")
            for row in plan["would_reembed_stale_dim"][:20]:
                print(f"  [dim {row['dim']:>4}] {row['user_id']}/{row['slot']}: {row['claim'][:60]}")
            if len(plan["would_reembed_null"]) + len(plan["would_reembed_stale_dim"]) > 40:
                print("  ... (truncated; use --json for the full list)")
            print("  NOTE: actual backfill is a separate, explicitly approved operation.")

    if args.strict and report.get("null_embeddings"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
