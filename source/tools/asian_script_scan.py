#!/usr/bin/env python3
"""Deterministic Asian-script regression scanner (2026-09-18 architecture prep).

Companion to docs/BRITISH_ENGLISH_AND_LOCALISATION.md. Scans the VoiceMemAgent
tree for Asian-script characters and classifies every occurrence by ownership,
so that NEW product-owned Asian-script text cannot be introduced silently
while approved vendor / model / historical / runtime content stays untouched.

Zones (docs/BRITISH_ENGLISH_AND_LOCALISATION.md section 7):

  HARD-FAIL  app/ web/ config/ + root product files.
             Every Asian-script run must appear in the per-file allowlist
             (tools/asian_script_baseline.json "hard_fail") with a count cap;
             anything else is a violation. Root product files have no
             allowlist: any occurrence is a violation.
  BASELINE   tests/ scripts/ docs/ tools/ (+ any unknown top-level path).
             File-level occurrence counts must not exceed the committed
             baseline; a file with occurrences that is not in the baseline
             is a violation.
  REPORT-ONLY  memory/ data/  (runtime user/model-generated content).
  EXEMPT     vendor/ models/ logs/ audit/ releases/ + docs historical capture
             subdirs + binaries + files over 2 MB + CHANGELOG.md + this
             scanner's own baseline file.

Exit code 0 = clean, 1 = violations found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO / "tools" / "asian_script_baseline.json"

#: One contiguous run of Asian-script characters (the occurrence unit).
ASIAN_RUN = re.compile(
    "[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf"
    "\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]+"
)

HARD_FAIL_DIRS = ("app", "web", "config")
HARD_FAIL_ROOT_FILES = (
    "BUILD_INFO.json", "CONTRACT.md", "INSTALL_MANIFEST.example.json",
    "LICENSES.md", "MODELS.lock.json", "README.md", "START.bat",
    "UPSTREAM_POLICY.md", "VERSION", "pyproject.toml",
    "requirements.txt", "requirements.lock", "requirements.lock.json",
    "VOICEMEM_PIN.json",
)
BASELINE_DIRS = ("tests", "scripts", "docs", "tools")
REPORT_ONLY_DIRS = ("memory", "data")
EXEMPT_DIRS = (
    "vendor", "models", "logs", "audit", "releases", ".git",
    "__pycache__", "node_modules", ".pytest_cache", ".idea", ".vscode",
)
EXEMPT_DOCS_SUBDIRS = (
    "docs/verification_evidence", "docs/recovery", "docs/audits",
)
EXEMPT_FILES = ("CHANGELOG.md", "tools/asian_script_baseline.json")
SKIP_SUFFIXES = (
    ".zip", ".onnx", ".gguf", ".bin", ".wav", ".png", ".webp", ".pyc",
    ".ico", ".jpg", ".jpeg", ".pdf", ".sqlite", ".db", ".npy", ".npz",
    ".log", ".exe", ".dll", ".pt", ".safetensors",
)
MAX_FILE_BYTES = 2_000_000


def load_baseline(path: Path | None = None) -> dict:
    path = path or BASELINE_PATH
    if not path.exists():
        return {"hard_fail": {}, "baseline": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "hard_fail": data.get("hard_fail", {}),
        "baseline": data.get("baseline", {}),
    }


def _zone_of(rel: str) -> str:
    top = rel.split("/", 1)[0]
    if rel in HARD_FAIL_ROOT_FILES or top in HARD_FAIL_DIRS:
        return "hard_fail"
    if top in REPORT_ONLY_DIRS:
        return "report_only"
    # baseline zone is the default, including unknown top-level paths:
    # a new directory with Asian text must be consciously baselined.
    return "baseline"


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        top = rel.split("/", 1)[0]
        if top in EXEMPT_DIRS or rel in EXEMPT_DIRS:
            continue
        if rel in EXEMPT_FILES:
            continue
        if any(rel.startswith(sub + "/") for sub in EXEMPT_DOCS_SUBDIRS):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path, rel


def scan(root: Path | str | None = None, baseline: dict | None = None) -> dict:
    """Scan ``root`` (default: the repository) and classify all occurrences.

    Returns a dict with:
      violations  - list of {file, line?, run?, reason} policy breaches
      hard_fail   - {relative_path: {run_string: count}} for the hard-fail zone
      baseline    - {relative_path: total_runs} for the baseline zone
      report_only - {relative_path: total_runs} for the report-only zone
    """
    root = Path(root) if root is not None else REPO
    if baseline is None:
        baseline = load_baseline() if root.resolve() == REPO.resolve() else {"hard_fail": {}, "baseline": {}}
    allow = baseline.get("hard_fail", {})
    caps = baseline.get("baseline", {})
    violations: list[dict] = []
    hard_counts: dict[str, dict[str, int]] = {}
    baseline_counts: dict[str, int] = {}
    report_counts: dict[str, int] = {}
    for path, rel in _iter_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        runs = [m for m in ASIAN_RUN.finditer(text)]
        if not runs:
            continue
        zone = _zone_of(rel)
        per_run = Counter(m.group() for m in runs)
        total = sum(per_run.values())
        if zone == "hard_fail":
            hard_counts[rel] = dict(per_run)
            allowed_here = allow.get(rel, {})
            for run, count in per_run.items():
                if run not in allowed_here:
                    violations.append({
                        "file": rel, "run": run,
                        "reason": "unapproved-asian-text-in-product-code",
                    })
                elif count > allowed_here[run]:
                    violations.append({
                        "file": rel, "run": run, "count": count,
                        "cap": allowed_here[run],
                        "reason": "occurrence-count-exceeds-allowlist-cap",
                    })
        elif zone == "baseline":
            baseline_counts[rel] = total
            cap = caps.get(rel)
            if cap is None:
                violations.append({
                    "file": rel, "count": total,
                    "reason": "new-file-with-asian-text-needs-baseline-review",
                })
            elif total > cap:
                violations.append({
                    "file": rel, "count": total, "cap": cap,
                    "reason": "baseline-count-exceeded",
                })
        else:
            report_counts[rel] = total
    return {
        "violations": violations,
        "hard_fail": hard_counts,
        "baseline": baseline_counts,
        "report_only": report_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", metavar="PATH", help="write the full report as JSON to PATH")
    parser.add_argument("--update-baseline", action="store_true",
                        help="regenerate tools/asian_script_baseline.json from the current tree "
                             "(REVIEW the diff before committing - it is the ratchet)")
    args = parser.parse_args(argv)

    result = scan()
    if args.update_baseline:
        BASELINE_PATH.write_text(
            json.dumps(
                {"hard_fail": result["hard_fail"], "baseline": result["baseline"]},
                indent=1, sort_keys=True, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"baseline updated: {BASELINE_PATH}")

    violations = result["violations"]
    print(f"hard-fail files with occurrences: {len(result['hard_fail'])}")
    print(f"baseline files with occurrences:   {len(result['baseline'])}")
    print(f"report-only files with occurrences: {len(result['report_only'])}")
    print(f"violations: {len(violations)}")
    for violation in violations[:50]:
        print(f"  VIOLATION {violation['file']}: {violation['reason']} {violation.get('run', '')}")
    if len(violations) > 50:
        print(f"  ... and {len(violations) - 50} more")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"report written: {args.json}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
