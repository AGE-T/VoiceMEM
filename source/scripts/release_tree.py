"""scripts/release_tree.py — the shared release-tree fingerprint (v0.8.1).

WHY: the v0.8.0 release ran its full test gate BEFORE the VERSION bump, so the
tree that was packaged was never the tree that was gated (the shipped
PAGE_VERSION='0.7.2' defect passed undetected). From v0.8.1 the rule is:

    THE EXACT TREE THAT IS PACKAGED MUST BE THE TREE THAT PASSED THE GATE.

Both sides of that contract need to hash the SAME file set the same way:

* scripts/run_release_gate.py fingerprints the tree it is about to gate and
  writes the fingerprint into releases/gate_record.json together with the
  test outcome;
* scripts/build_release_sandbox.py recomputes the fingerprint at build time
  and REFUSES to build unless it matches a GREEN record (and the record's
  VERSION equals the builder's NEW_VERSION and the tree's VERSION file).

The file set is exactly what the release ZIP stages (same directories, same
root files, same ignore patterns as the builder's copy_tree), EXCLUDING the
release ledgers under releases/ (RELEASE_INDEX.json / BUILD_HISTORY.json are
build outputs the builder itself appends to after the ZIP is closed; their
pre-update state is staged deterministically) and BUILD_INFO.json (generated
into the stage from the gate record). This keeps the fingerprint stable and
the guarantee honest: any change to product source between gate and build
invalidates the record and forces a re-gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: Directories staged into the release ZIP (docs/ is added when present,
#: exactly like the builder's stage_list).
STAGED_DIRS = ("app", "config", "scripts", "tests", "web", "vendor")

#: Root files staged into the release ZIP (mirrors the builder's ROOT_FILES).
ROOT_FILES = (
    "README.md", "CHANGELOG.md", "LICENSES.md", "CONTRACT.md", "VERSION",
    "INSTALL_MANIFEST.example.json", "requirements.txt", "requirements.lock",
    "pyproject.toml", ".gitignore", "START.bat", "MODELS.lock.json",
    "VOICEMEM_PIN.json", "UPSTREAM_POLICY.md",
)

#: The same ignore patterns the builder's copy_tree applies — a file that
#: never ships must not influence the fingerprint, and a file that ships must.
IGNORE_PATTERNS = (
    "__pycache__", ".pytest_cache", "*.pyc", "*.pyo", ".env",
    "voice_settings.json", "llm_model.json", "*.egg-info", "build", "dist",
    "voicemem_memory*", "results", ".mypy_cache",
)


def _ignored(name: str) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(name, pat) for pat in IGNORE_PATTERNS)


def staged_files(repo: Path) -> list[Path]:
    """Every file that would ship in the release ZIP, sorted by repo-relative
    posix path (docs/ included when present; releases/ ledgers excluded)."""
    out: list[Path] = []
    dirs = list(STAGED_DIRS)
    if (repo / "docs").is_dir():
        dirs.insert(1, "docs")
    for d in dirs:
        base = repo / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not _ignored(p.name):
                out.append(p)
    for name in ROOT_FILES:
        p = repo / name
        if p.is_file():
            out.append(p)
    return sorted(out, key=lambda p: p.relative_to(repo).as_posix())


def tree_fingerprint(repo: Path) -> tuple[str, int]:
    """sha256 over (sorted relpath, content-hash) of the staged file set.

    Returns (fingerprint_hex, file_count). Deterministic; a single byte
    change in any shipped file changes the fingerprint.
    """
    h = hashlib.sha256()
    files = staged_files(repo)
    for p in files:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(p.relative_to(repo).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest(), len(files)


def tree_version(repo: Path) -> str:
    """The tree's VERSION file content (the single authoritative version)."""
    try:
        return (repo / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
