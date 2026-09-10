#!/usr/bin/env python3
"""Controlled VoiceMem source + pin verification (v0.5.0 ownership model).

Three verification layers, each independently runnable:

  1. SOURCE  (default, no Python package install needed):
     - vendor/voicemem tree present (voicemem/__init__.py + pyproject.toml)
     - VOICEMEM_PIN.json present and records the pinned upstream commit
     - the runtime identity block in voicemem/__init__.py matches the pin
     - the local patch markers exist in the patched files
     - NO OpenAI embeddings call remains on the memory-embedding paths
       (orchestrator _embed_uncached; the OpenAILocalEmbedder default is
       gone from defaults.py / leftbrain/brain.py)
     - the DELETE safety guard is present

  2. RUNTIME (--runtime; requires voicemem importable, e.g. pip install -e
     vendor/voicemem into the active environment):
     - import voicemem; voicemem.__file__ resolves UNDER vendor/voicemem
     - voicemem.CONTROLLED_FORK is True
     - voicemem.CONTROLLED_UPSTREAM_COMMIT equals the pin commit
     - the installed distribution still maps to the controlled source

  3. EMBEDDING REGRESSION (--embedding; fully self-contained subprocess
     instrumentation — no network, no real model):
     - a fake `openai` module that EXPLODES on any OpenAI() construction
     - a fake local E5 encoder (384-dim deterministic vectors)
     - proves Orchestrator._embed_text routes to the injected embedder /
       the local E5 and NEVER constructs an OpenAI client

Exit code 0 = all requested checks pass; 1 = any failure (printed).
Used by tests/unit/test_voicemem_controlled.py (subprocess), verify_m1.ps1
(Runtime mode) and the release self-check (Source mode).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "voicemem"
PKG = VENDOR / "voicemem"
PIN_FILE = ROOT / "VOICEMEM_PIN.json"

#: The authoritative upstream identity this repository pins.
EXPECTED_COMMIT = "e8384e087bd2f44eb05fc7ae1a3c525ea8244179"
EXPECTED_TAG = "v0.0.1"

_EMBEDDING_PROBE = r'''
import sys, os, tempfile
sys.path.insert(0, {vendor!r})
os.environ["VOICEMEM_MEMORY_ROOT"] = tempfile.mkdtemp(prefix="vmpin_")

# --- fake openai: ANY construction must explode loudly --------------------
class _OpenAICalledError(RuntimeError):
    pass

def _boom(*a, **k):
    raise _OpenAICalledError("OpenAI client constructed on the memory path!")

class _FakeOpenAIModule:
    class OpenAI:
        def __init__(self, *a, **k):
            _boom()
    class AsyncOpenAI:
        def __init__(self, *a, **k):
            _boom()
    def __getattr__(self, name):
        _boom()

sys.modules["openai"] = _FakeOpenAIModule()

import voicemem
from voicemem.orchestrator import Orchestrator

DIM = 384
def _fake_vec(text: str):
    # deterministic pseudo-embedding from the text bytes
    raw = text.encode("utf-8") or b"\x00"
    vals = []
    h = 2166136261
    for i in range(DIM):
        h = (h * 16777619 + raw[i % len(raw)] + i) & 0xFFFFFFFF
        vals.append(((h % 2000) / 1000.0) - 1.0)
    return vals

class _FakeST:
    def encode(self, texts, normalize_embeddings=False, **kw):
        import numpy as np
        return np.asarray([_fake_vec(t) for t in texts], dtype="float32")

import voicemem.leftbrain.local_e5_embedder as le5
le5.shared_e5 = lambda: _FakeST()   # bypass the real model entirely

failures = []

# --- Case A: injected embedder is used (DI architecture) -------------------
class _Injected:
    def embed_query_text(self, text):
        return _fake_vec("Q:" + text)
    def embed_texts(self, texts):
        return [_fake_vec("D:" + t) for t in texts]

o = Orchestrator.__new__(Orchestrator)
o._embedder = _Injected()
o._base_url = "http://127.0.0.1:8080/v1"
v = o._embed_text("hello world")
if v != _fake_vec("Q:hello world"):
    failures.append("injected embedder NOT used by _embed_text")

# --- Case B: no embedder -> local E5 via embed_cache (never OpenAI) --------
o2 = Orchestrator.__new__(Orchestrator)
o2._embedder = None
o2._base_url = "http://127.0.0.1:8080/v1"
from voicemem.utils.common import embed_cache
embed_cache._cache.clear()
v2 = o2._embed_text("hello world")
if len(v2) != DIM:
    failures.append("no-injection default returned wrong dimension: %r" % (len(v2),))
# cache hit proves the local key is used (and shared)
v3 = o2._embed_text("hello world")
if v3 != v2:
    failures.append("embed_cache miss for the local key")
if not any(k[0].startswith("local-multilingual-e5-small") for k in embed_cache._cache):
    failures.append("embed_cache keys do not contain the local E5 key: %r" % (list(embed_cache._cache.keys())[:3],))

# --- Case C: _embed_uncached directly (the removed OpenAI body) ------------
v4 = o2._embed_uncached(["a", "b"])
if len(v4) != 2 or any(len(x) != DIM for x in v4):
    failures.append("_embed_uncached wrong shape")

if failures:
    print("EMBEDDING-REGRESSION-FAIL:")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("EMBEDDING-REGRESSION-OK: injected embedder used; no-injection default = "
      "local E5 (384-d, embed_cache local key); _embed_uncached local-only; "
      "OpenAI client NEVER constructed")
'''


def source_checks(print_ok: bool = True) -> list[str]:
    """Layer 1: static verification of the controlled tree. Returns failures."""
    failures: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if ok:
            if print_ok:
                print(f"  [ok]   {msg}")
        else:
            failures.append(msg)
            print(f"  [FAIL] {msg}")

    check((PKG / "__init__.py").is_file(), "vendor/voicemem/voicemem package present")
    check((VENDOR / "pyproject.toml").is_file(), "vendor/voicemem/pyproject.toml present")
    check(PIN_FILE.is_file(), "VOICEMEM_PIN.json present at the agent root")

    pin: dict[str, Any] = {}
    if PIN_FILE.is_file():
        try:
            pin = json.loads(PIN_FILE.read_text(encoding="utf-8"))
        except ValueError as exc:
            failures.append(f"VOICEMEM_PIN.json is not valid JSON: {exc}")
    prov = pin.get("provenance") or {}
    check(
        prov.get("upstream_commit") == EXPECTED_COMMIT,
        f"pin upstream_commit == {EXPECTED_COMMIT}",
    )
    check(prov.get("upstream_tag") == EXPECTED_TAG, "pin upstream_tag == v0.0.1")

    init_py = (PKG / "__init__.py").read_text(encoding="utf-8")
    check("CONTROLLED_FORK = True" in init_py, "runtime identity block: CONTROLLED_FORK")
    check(f'CONTROLLED_UPSTREAM_COMMIT = "{EXPECTED_COMMIT}"' in init_py,
          "runtime identity block: CONTROLLED_UPSTREAM_COMMIT matches the pin")
    for patch_id in ("VM-LOCAL-001", "VM-LOCAL-002", "VM-LOCAL-003",
                     "VM-LOCAL-004", "VM-LOCAL-005", "VM-LOCAL-006"):
        check(f'"{patch_id}"' in init_py, f"patch ledger lists {patch_id}")

    orch = (PKG / "orchestrator.py").read_text(encoding="utf-8")
    check("if self._embedder is not None:" in orch,
          "VM-LOCAL-001: _embed_text routes through the injected embedder")
    check("_LOCAL_E5_CACHE_KEY" in orch, "VM-LOCAL-001: local E5 cache key used")
    m = re.search(r"def _embed_uncached.*?(?=\n    def |\nclass |\Z)", orch, re.S)
    body = m.group(0) if m else ""
    check("local_e5_embedder" in body and "LocalE5Embedder" in body,
          "VM-LOCAL-001: _embed_uncached is local-E5 only")
    check("client.embeddings.create(" not in body,
          "VM-LOCAL-001: no OpenAI embeddings call in _embed_uncached")

    defaults = (PKG / "utils" / "defaults.py").read_text(encoding="utf-8")
    m = re.search(r"def embedding\(\).*?(?=\n    def )", defaults, re.S)
    emb_body = m.group(0) if m else ""
    check("LocalE5Embedder()" in emb_body and "OpenAILocalEmbedder(" not in emb_body,
          "VM-LOCAL-003: default embedding factory is LocalE5Embedder")

    brain = (PKG / "leftbrain" / "brain.py").read_text(encoding="utf-8")
    m = re.search(r"def _get_repo\(self\).*?(?=\n    def )", brain, re.S)
    repo_body = m.group(0) if m else ""
    check("LocalE5Embedder()" in repo_body and "OpenAILocalEmbedder(" not in repo_body,
          "VM-LOCAL-002: left-brain repo default embedder is LocalE5Embedder")

    ts = (PKG / "rightbrain" / "traits_store.py").read_text(encoding="utf-8")
    check("rb_traits embedding FAILED" in ts,
          "VM-LOCAL-004: trait embedding failure is loud (logged, not silent NULL)")
    check("trait_min_sim" in (PKG / "rightbrain" / "brain.py").read_text(encoding="utf-8"),
          "UPSTREAM-f535f9d: trait threshold bound to embedder dimension")
    gc = (PKG / "utils" / "common" / "_graph_common.py").read_text(encoding="utf-8")
    check("if len(a) != len(b):" in gc,
          "UPSTREAM-91d2e42: cosine dimension guard")

    mem0 = (PKG / "leftbrain" / "mem0_backend_store.py").read_text(encoding="utf-8")
    check("VOICEMEM_ALLOW_MEMORY_DELETE" in mem0,
          "VM-LOCAL-005: delete_memory opt-in safety guard")
    check("_as_date" in mem0, "UPSTREAM-e3cc965: observed_at date validation")

    merged = (PKG / "leftbrain" / "merged_extraction.py").read_text(encoding="utf-8")
    check("IN ENGLISH (British spelling)" in merged,
          "VM-LOCAL-EN: English trait/emotion extraction localisation marker")

    # no upstream git machinery may be required by the vendor tree itself
    check(not (VENDOR / ".git").exists(),
          "vendor tree is NOT a git clone (controlled source, no fetch surface)")
    check(not (VENDOR / ".clone_ref").exists(), "no stale .clone_ref file")

    return failures


def runtime_checks(print_ok: bool = True) -> list[str]:
    """Layer 2: import-resolution verification (voicemem must be installed)."""
    failures: list[str] = []
    try:
        import voicemem  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] import voicemem: {exc}")
        return [f"import voicemem failed: {exc}"]

    path = str(Path(voicemem.__file__).resolve())  # type: ignore[union-attr]
    norm = path.replace("\\", "/")
    if print_ok:
        print(f"  [info] voicemem.__file__ = {path}")
    if "/vendor/voicemem/" not in norm:
        failures.append(f"import does NOT resolve under vendor/voicemem: {path}")
        print(f"  [FAIL] import path not under vendor/voicemem: {path}")
    else:
        print("  [ok]   import resolves under vendor/voicemem")
    if not bool(getattr(voicemem, "CONTROLLED_FORK", False)):
        failures.append("voicemem.CONTROLLED_FORK is not True")
        print("  [FAIL] CONTROLLED_FORK is not True")
    else:
        print("  [ok]   CONTROLLED_FORK is True")
    commit = getattr(voicemem, "CONTROLLED_UPSTREAM_COMMIT", "")
    if commit != EXPECTED_COMMIT:
        failures.append(f"runtime commit {commit!r} != pin {EXPECTED_COMMIT!r}")
        print(f"  [FAIL] runtime upstream commit {commit!r} != pin")
    else:
        print("  [ok]   runtime CONTROLLED_UPSTREAM_COMMIT matches the pin")
    return failures


def embedding_regression(print_ok: bool = True) -> list[str]:
    """Layer 3: instrumented subprocess proof (fake openai + fake E5)."""
    probe = _EMBEDDING_PROBE.format(vendor=str(VENDOR))
    with tempfile.NamedTemporaryFile(
        "w", suffix="_vm_embed_probe.py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(probe)
        probe_path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, probe_path],
            capture_output=True, text=True, timeout=120,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if print_ok and out:
            print("  " + out.replace("\n", "\n  "))
        if proc.returncode != 0:
            if err and print_ok:
                print("  [stderr] " + err[:2000])
            return ["embedding regression subprocess failed"]
        if "EMBEDDING-REGRESSION-OK" not in out:
            return ["embedding regression subprocess produced no OK marker"]
        return []
    finally:
        Path(probe_path).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runtime", action="store_true",
                    help="also run the import-resolution checks (voicemem must be installed)")
    ap.add_argument("--embedding", action="store_true",
                    help="also run the instrumented embedding regression (subprocess)")
    ap.add_argument("--quiet", action="store_true", help="only print failures")
    args = ap.parse_args()

    print("VoiceMem controlled-source verification")
    print(f"  root   : {ROOT}")
    print(f"  vendor : {VENDOR}")
    print(f"  pin    : {PIN_FILE}")
    print("  commit : " + EXPECTED_COMMIT)
    print()

    failures: list[str] = []
    print("[1/3] SOURCE checks")
    failures += source_checks(print_ok=not args.quiet)
    print()
    if args.runtime:
        print("[2/3] RUNTIME checks")
        failures += runtime_checks(print_ok=not args.quiet)
        print()
    if args.embedding:
        print("[3/3] EMBEDDING regression (instrumented)")
        failures += embedding_regression(print_ok=not args.quiet)
        print()

    if failures:
        print(f"VERIFICATION FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERIFICATION OK"
          + (" (source)" if not args.runtime else " (source+runtime)")
          + (" + embedding regression" if args.embedding else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
