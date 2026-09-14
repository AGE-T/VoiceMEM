#!/usr/bin/env python3
"""scripts/sync_page_version.py — regenerate the page's PAGE_VERSION literal.

v0.8.1 (post-implementation-audit P1-1): ``web/voicemem.html`` carries a
``const PAGE_VERSION='X.Y.Z';`` literal that the browser compares against
``/api/health`` (stale-page detection, v0.4.12). Up to v0.8.0 the literal was
bumped BY HAND at release time — and v0.8.0 shipped it stale (page 0.7.2 vs
VERSION 0.8.0, a false "stale page" toast on every session).

The single source of truth is the ``VERSION`` file (the same value
``/api/health`` serves). This script regenerates the FILE literal from it, so
the git-visible tree, the ZIP self-check markers and the release gate all see
the same value the runtime serves. The web server additionally stamps the
served page at request time (app/web_server.py ``_inject_page_version``) —
belt and braces: even a bypassed injection cannot ship a false version, and
even a stale file literal cannot reach the browser.

Byte-level replacement (the literal is pure ASCII) so line endings and the
rest of the file are never touched. Idempotent; exits non-zero if the literal
is missing (the page JS requires it — a missing literal is a defect).

Usage (release prep, BEFORE the release gate):
    python scripts/sync_page_version.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HTML = REPO / "web" / "voicemem.html"
VERSION = REPO / "VERSION"
_LITERAL = re.compile(rb"const PAGE_VERSION='[^']*';")


def main() -> int:
    version = VERSION.read_text(encoding="utf-8").strip()
    if not version:
        print("sync_page_version FAILED: VERSION file is empty")
        return 1
    data = HTML.read_bytes()
    m = _LITERAL.search(data)
    if m is None:
        print(
            "sync_page_version FAILED: no "
            "const PAGE_VERSION='X.Y.Z'; literal in web/voicemem.html"
        )
        return 1
    injected = f"const PAGE_VERSION='{version}';".encode("ascii")
    if m.group(0) == injected:
        print(f"page version already in sync: {version}")
        return 0
    HTML.write_bytes(data[: m.start()] + injected + data[m.end():])
    print(
        "page version synced: "
        f"{m.group(0).decode('ascii')} -> {injected.decode('ascii')} "
        "(source: VERSION file)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
