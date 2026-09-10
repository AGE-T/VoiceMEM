#!/usr/bin/env python3
"""One-off migration: translate Chinese memory claims/notes to British English.

Background (v0.4.6 field report): the vendored xzf-thu/VoiceMem package used to
prompt the LLM in Chinese for right-brain trait labels ("5-15 Chinese
characters") and emotion words, so a Hungarian/English conversation produced
memory cards like 纠错时直截了当 / Profile · 应对方式 ×1. The v0.4.7 installer
patches the vendor PROMPT (new extractions are English) and the web UI
translates the slot enum values for display — but the ALREADY-STORED claims
stay Chinese until this migration runs.

What it does, per ``<memory_root>/*.sqlite`` (all Memory Spaces share the
space DB with different user_ids):

* ``rb_traits.claim``       — Chinese → English via the local llama-server
                              (the local llama-server LLM, temperature 0). The trait's
                              embedding is set to NULL: it was computed from
                              the Chinese text and no longer matches — a NULL
                              embedding only disables merge-similarity, the
                              store handles it natively.
* ``right_brain_memories.content`` — same translation (inner-OS narratives /
                              situation patterns that were generated in
                              Chinese for Chinese input).
* ``rb_evidence.quote`` is NEVER touched: quotes are the user's own words.

Idempotent: rows without Han characters are skipped, so re-running after a
partial migration (or with the LLM down) only retries what is left. The
script never deletes anything and prints a per-table report.

Usage (from the repo root, inside .venv on the target machine):
    python scripts/localise_memories.py [--dry-run] [--memory-root DIR] [--url URL]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

#: CJK Unified Ideographs (plus extensions): the "is this Chinese?" test.
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_ROOT = Path(__file__).resolve().parents[1]


def _has_han(text: str) -> bool:
    return bool(_HAN_RE.search(text or ""))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LlamaTranslator:
    """Minimal OpenAI-compatible chat client for llama-server (stdlib only)."""

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self._url = url.rstrip("/") + "/v1/chat/completions"
        self._timeout = timeout

    def translate(self, text: str, kind: str) -> str:
        if kind == "trait":
            instruction = (
                "Translate this Chinese personality-trait description into "
                "natural British English. Keep it a SHORT PATTERN: 3 to 8 words, "
                "no subject, no full stop, no quotes. Reply with ONLY the "
                "translation."
            )
        else:
            instruction = (
                "Translate this Chinese note about a person into natural "
                "British English. Keep the [emotion] bracket at the start when "
                "one is present. Reply with ONLY the translation."
            )
        payload = {
            "model": "qwen3.6-35b-a3b",
            "temperature": 0.0,
            "max_tokens": 160,
            # v0.4.21: explicit request-level thinking suppression — this tool
            # builds its own raw payload (stdlib urllib, not app/llm.py's
            # LlmClient), so the switches must be added HERE to keep the
            # application layer aligned with the server-side --reasoning off:
            # chat_template_kwargs.enable_thinking=false is the llama.cpp
            # native mechanism (the Qwen template pre-closes the thought
            # channel), reasoning_effort "none" is the OpenAI-compatible
            # field (b10717 server-common.cpp maps "none" -> disable
            # reasoning). Verified live against llama-server b10717.
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
        }
        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = ((body.get("choices") or [{}])[0].get("message") or {}).get(
            "content", ""
        )
        out = str(content or "").strip().strip('"“”').strip()
        if not out or _has_han(out):
            raise ValueError(f"unusable translation: {out[:60]!r}")
        return out


def _migrate_traits(conn: sqlite3.Connection, tr: LlamaTranslator, dry: bool) -> tuple[int, int]:
    """rb_traits.claim: translate Han rows. Returns (translated, failed)."""
    rows = conn.execute(
        "SELECT id, claim FROM rb_traits WHERE claim IS NOT NULL"
    ).fetchall()
    done = failed = 0
    for row_id, claim in rows:
        if not _has_han(claim):
            continue
        try:
            claim_en = tr.translate(claim, "trait")
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the run
            print(f"    [trait] SKIP {row_id}: {exc}")
            failed += 1
            continue
        if dry:
            print(f"    [trait] DRY {row_id}: {claim!r} -> {claim_en!r}")
        else:
            conn.execute(
                "UPDATE rb_traits SET claim=?, embedding=NULL, updated_at=? WHERE id=?",
                (claim_en, _now(), row_id),
            )
        done += 1
    conn.commit()
    return done, failed


def _migrate_right_memories(
    conn: sqlite3.Connection, tr: LlamaTranslator, dry: bool
) -> tuple[int, int]:
    """right_brain_memories.content: translate Han rows. Returns (translated, failed)."""
    try:
        rows = conn.execute(
            "SELECT id, content FROM right_brain_memories WHERE content IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return 0, 0  # table absent in this DB
    done = failed = 0
    for row_id, content in rows:
        if not _has_han(content):
            continue
        try:
            content_en = tr.translate(content, "note")
        except Exception as exc:  # noqa: BLE001
            print(f"    [note]  SKIP {row_id}: {exc}")
            failed += 1
            continue
        if dry:
            print(f"    [note]  DRY {row_id}: {content!r} -> {content_en!r}")
        else:
            conn.execute(
                "UPDATE right_brain_memories SET content=? WHERE id=?",
                (content_en, row_id),
            )
        done += 1
    conn.commit()
    return done, failed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--memory-root", default="", help="override memory root dir")
    ap.add_argument("--url", default="", help="override llama-server base URL")
    args = ap.parse_args(argv)

    # Resolve config (paths + llama URL) without importing the app package:
    # this script may run in a bare venv where numpy & co are present, but
    # keep the footprint minimal anyway.
    memory_root = Path(args.memory_root) if args.memory_root else _ROOT / "memory"
    llama_url = args.url
    if not llama_url:
        llama_url = "http://127.0.0.1:8080"
        yaml_path = _ROOT / "config" / "voicemem_config.yaml"
        if yaml_path.is_file():
            m = re.search(
                r"llama_server_host:\s*(\S+).*?llama_server_port:\s*(\d+)",
                yaml_path.read_text(encoding="utf-8"),
                re.S,
            )
            if m:
                llama_url = f"http://{m.group(1)}:{m.group(2)}"

    if not memory_root.is_dir():
        print(f"memory root not found: {memory_root} — nothing to migrate.")
        return 0

    dbs = sorted(p for p in memory_root.glob("*.sqlite") if p.is_file())
    dbs += sorted(
        p
        for sub in memory_root.iterdir()
        if sub.is_dir()
        for p in sub.glob("*.sqlite")
        if p.is_file()
    )
    # de-duplicate, keep order
    seen: set[Path] = set()
    dbs = [p for p in dbs if not (p in seen or seen.add(p))]
    if not dbs:
        print(f"no *.sqlite under {memory_root} — nothing to migrate.")
        return 0

    tr = LlamaTranslator(llama_url)
    total_done = total_failed = 0
    print(f"localise_memories: {len(dbs)} database(s) under {memory_root}")
    print(f"llama-server: {llama_url}{' (dry run)' if args.dry_run else ''}")
    for db_path in dbs:
        print(f"  DB {db_path}")
        try:
            conn = sqlite3.connect(str(db_path))
        except sqlite3.Error as exc:
            print(f"    cannot open: {exc}")
            continue
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "rb_traits" not in tables and "right_brain_memories" not in tables:
                print("    no voicemem right-brain tables — skipped.")
                continue
            if "rb_traits" in tables:
                done, failed = _migrate_traits(conn, tr, args.dry_run)
                total_done += done
                total_failed += failed
                print(f"    rb_traits: {done} translated, {failed} failed")
            if "right_brain_memories" in tables:
                done, failed = _migrate_right_memories(conn, tr, args.dry_run)
                total_done += done
                total_failed += failed
                print(f"    right_brain_memories: {done} translated, {failed} failed")
        finally:
            conn.close()

    print(
        f"done: {total_done} row(s) migrated, {total_failed} failed"
        + (" — re-run this script later to retry the failures." if total_failed else "")
    )
    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
