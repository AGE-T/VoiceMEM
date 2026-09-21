"""VoiceMem controlled-vendor pin: identity + fingerprint verification.

v0.4.22 vendor move: ``vendor/voicemem`` is a SOURCE-OWNED controlled tree
(base = upstream xzf-thu/VoiceMem tag v0.0.1, commit e8384e087bd2…, plus
first-party local fixes documented in ``vendor/voicemem/PROVENANCE.md``).
There is no nested git repository any more, so the immutable identity needs
a machine-verifiable equivalent of "git commit + git status --clean":

  * ``VOICEMEM_PIN.json`` inside the vendor tree records the upstream base
    (repo / tag / commit), the local-fix count and a FINGERPRINT;
  * the fingerprint is a deterministic sha256 over every file of the tree
    (sorted relative paths, each file's own sha256) - byte-level tamper
    detection: any post-freeze edit (or a partial copy / stale extraction)
    changes it and verification FAILS LOUDLY.

Who verifies:
  * ``scripts/install_m1.ps1`` step 12 - BEFORE ``pip install -e`` (a wrong
    or corrupted vendor tree is a hard install failure, never a silent
    fallback to some other VoiceMem);
  * ``scripts/verify_m1.ps1`` gate 6 - the installed import must resolve
    into THIS tree and the pin must verify;
  * ``scripts/write_install_manifest.py`` - records the pin facts into
    INSTALL_MANIFEST.json;
  * ``scripts/build_release_sandbox.py`` / ``build_release.ps1`` - the
    staged (ZIP) copy of the vendor tree must still verify.

Intentional changes to the vendor tree REQUIRE refreshing the pin:

    python scripts/voicemem_vendor_pin.py --write

Pure stdlib; runs on the target Windows machine and in the dev sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

#: The pin file itself is excluded from the fingerprint (it carries the
#: fingerprint - hashing it would be circular).
PIN_FILENAME = "VOICEMEM_PIN.json"

#: Runtime/clone-era artifacts that must never contribute to identity.
_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git", ".idea", ".vscode"}
_EXCLUDE_NAMES = {PIN_FILENAME, ".clone_ref", ".DS_Store"}
_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

#: v1 recipe id recorded in the pin (change the recipe -> bump this).
FINGERPRINT_ALGORITHM = "sha256-over-sorted-relpath-digests-v1"

_SHA_CHUNK = 1 << 20  # 1 MiB chunks: vendor files are prompt-sized, but stay safe

#: Required pin fields (schema 1).
_REQUIRED_FIELDS = ("schema", "source", "upstream_repo", "upstream_tag",
                    "upstream_commit", "fingerprint")


def _iter_files(vendor_root: Path):
    """Deterministic file list: sorted by POSIX relpath, junk excluded."""
    for path in sorted(vendor_root.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(vendor_root)
        if rel.parts[0] in _EXCLUDE_DIRS:
            continue
        if path.name in _EXCLUDE_NAMES:
            continue
        if path.suffix in _EXCLUDE_SUFFIXES:
            continue
        yield path, rel


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_SHA_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(vendor_root: Path) -> str:
    """Deterministic content fingerprint of the controlled vendor tree.

    recipe v1: sha256 over the concatenation of
    ``"<posix-relpath>\\0<file-sha256>\\n"`` lines in sorted path order.
    """
    vendor_root = Path(vendor_root)
    h = hashlib.sha256()
    files = 0
    for path, rel in _iter_files(vendor_root):
        h.update(rel.as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(_file_sha256(path).encode("ascii"))
        h.update(b"\n")
        files += 1
    return h.hexdigest()


def file_count(vendor_root: Path) -> int:
    return sum(1 for _ in _iter_files(Path(vendor_root)))


def load_pin(vendor_root: Path) -> Optional[dict[str, Any]]:
    """Parse VOICEMEM_PIN.json (None when absent/invalid JSON)."""
    pin_file = Path(vendor_root) / PIN_FILENAME
    if not pin_file.is_file():
        return None
    try:
        data = json.loads(pin_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_pin(vendor_root: Path, *, upstream_repo: str, upstream_tag: str,
              upstream_commit: str, base_taken: str, local_fixes: int,
              source: str = "controlled-vendor") -> dict[str, Any]:
    """(Re)write the pin with a FRESH fingerprint of the current tree.

    Call ONLY for intentional vendor-tree changes, then commit the updated
    pin together with the source change (the repo's git history is the
    audit trail; PROVENANCE.md documents WHAT changed and why).
    """
    vendor_root = Path(vendor_root)
    pin: dict[str, Any] = {
        "schema": 1,
        "source": source,
        "upstream_repo": upstream_repo,
        "upstream_tag": upstream_tag,
        "upstream_commit": upstream_commit,
        "base_taken": base_taken,
        "local_fixes": local_fixes,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "fingerprint": fingerprint(vendor_root),
        "files": file_count(vendor_root),
    }
    (vendor_root / PIN_FILENAME).write_text(
        json.dumps(pin, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pin


def verify(vendor_root: Path) -> tuple[bool, list[str], Optional[dict[str, Any]]]:
    """Verify the vendor tree against its pin.

    Returns (ok, problems, pin). Every problem is a human-readable line;
    verification NEVER repairs anything (detection only).
    """
    vendor_root = Path(vendor_root)
    problems: list[str] = []
    if not vendor_root.is_dir():
        return False, [f"vendor tree missing: {vendor_root}"], None

    # a VoiceMem package tree must carry its own packaging + package dir
    if not (vendor_root / "pyproject.toml").is_file():
        problems.append("pyproject.toml missing (not a pip-installable vendor tree)")
    if not (vendor_root / "voicemem" / "__init__.py").is_file():
        problems.append("voicemem/__init__.py missing (not the VoiceMem package)")
    if not (vendor_root / "PROVENANCE.md").is_file():
        problems.append("PROVENANCE.md missing (controlled-tree documentation)")

    pin = load_pin(vendor_root)
    if pin is None:
        problems.append(f"{PIN_FILENAME} missing or invalid JSON")
        return False, problems, None

    for field in _REQUIRED_FIELDS:
        if field not in pin:
            problems.append(f"pin field missing: {field}")
    if problems:
        return False, problems, pin

    if pin.get("fingerprint_algorithm") not in (None, FINGERPRINT_ALGORITHM):
        problems.append(
            f"unknown fingerprint algorithm {pin.get('fingerprint_algorithm')!r} "
            f"(expected {FINGERPRINT_ALGORITHM!r})")

    actual = fingerprint(vendor_root)
    expected = str(pin.get("fingerprint", ""))
    if not expected:
        problems.append("pin carries no fingerprint")
    elif actual != expected:
        problems.append(
            f"FINGERPRINT MISMATCH - the vendor tree differs from the pinned "
            f"state (pinned {expected[:12]}…, actual {actual[:12]}…). Either "
            f"the tree was modified outside the repo, or the extraction is "
            f"partial/stale. Intentional change? refresh with "
            f"'python scripts/voicemem_vendor_pin.py --write' and commit.")

    return (not problems), problems, pin


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify (or refresh) the controlled VoiceMem vendor pin.")
    parser.add_argument("--root", default=None,
                        help="vendor/voicemem root (default: <repo>/vendor/voicemem)")
    parser.add_argument("--write", action="store_true",
                        help="REFRESH the pin fingerprint (intentional changes only)")
    parser.add_argument("--upstream-repo",
                        default="https://github.com/xzf-thu/VoiceMem")
    parser.add_argument("--upstream-tag", default="v0.0.1")
    parser.add_argument("--upstream-commit",
                        default="e8384e087bd2f44eb05fc7ae1a3c525ea8244179")
    parser.add_argument("--base-taken", default="2026-09")
    parser.add_argument("--local-fixes", type=int, default=6)
    args = parser.parse_args(argv)

    vendor_root = Path(args.root) if args.root else \
        _repo_root_from_script() / "vendor" / "voicemem"

    if not vendor_root.is_dir():
        print(f"VOICEMEM PIN FAIL: vendor tree not found: {vendor_root}")
        return 2

    if args.write:
        pin = write_pin(vendor_root,
                        upstream_repo=args.upstream_repo,
                        upstream_tag=args.upstream_tag,
                        upstream_commit=args.upstream_commit,
                        base_taken=args.base_taken,
                        local_fixes=args.local_fixes)
        print(f"[voicemem-pin] pin written: {PIN_FILENAME}")
        print(f"[voicemem-pin] upstream {pin['upstream_tag']} "
              f"{pin['upstream_commit'][:12]}… + "
              f"{pin['local_fixes']} local fixes")
        print(f"[voicemem-pin] fingerprint {pin['fingerprint']}")
        print(f"[voicemem-pin] files {pin['files']}")
        return 0

    ok, problems, pin = verify(vendor_root)
    if ok:
        print(f"[voicemem-pin] OK: controlled-vendor tree verified "
              f"({pin.get('files', file_count(vendor_root))} files, "
              f"upstream {pin.get('upstream_tag')} "
              f"{str(pin.get('upstream_commit'))[:12]}…, "
              f"fingerprint {str(pin.get('fingerprint'))[:12]}…)")
        return 0
    print("VOICEMEM PIN FAIL:")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
