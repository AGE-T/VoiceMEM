"""Release & versioning feature validation (Task 12 - M0.1).

The user-facing requirement validated here: "a versioned ZIP file after
every update, and the full validation test suite must run before every ZIP
build".

Validated contracts:
* version sync rule: VERSION == pyproject.toml [project] version ==
  config/voicemem_config.yaml project.version (scripts/build_release.ps1
  keeps all three in sync on every release);
* CHANGELOG.md exists, follows the "## [X.Y.Z] - YYYY-MM-DD" heading format
  and contains an entry for the current version;
* releases/RELEASE_INDEX.json schema + on-disk ZIP integrity (sha256 +
  the BUILD_INFO.json inside every built ZIP);
* scripts/build_release.ps1 exists and enforces the MANDATORY pre-build
  test gate (run_tests.ps1 invocation before ZIP creation, checksum,
  index/changelog updates);
* every path of the staging allowlist exists in the repo (the ZIP content
  contract);
* the release ZIP ships the M0 placeholder directory skeleton (ZIP
  archives do not store empty directories - the builder stages .gitkeep
  placeholders) and its own CHANGELOG entry (written into the STAGED
  changelog BEFORE the ZIP is created) - v0.1.6 release-blocker fix;
* the validation report writer (tests/validation/_report.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
CHANGELOG_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})", re.MULTILINE)

#: Must mirror the staging allowlist of scripts/build_release.ps1 (step 4/8).
#: START.bat + MODELS.lock.json are M0.2 one-click requirements: the release
#: ZIP must be installable by double-clicking START.bat (user spec 25/13).
#: "web" joined in v0.3.5: the release ZIP must ship EVERY element of the
#: system, including the local VoiceMem web UI.
STAGED_DIRS = ("app", "config", "scripts", "tests", "web", "vendor")
STAGED_FILES = (
    "README.md", "CHANGELOG.md", "LICENSES.md", "CONTRACT.md", "VERSION",
    "INSTALL_MANIFEST.example.json", "requirements.txt", "requirements.lock",
    "pyproject.toml", ".gitignore", "config/.env.example",
    "START.bat", "MODELS.lock.json",
    # v0.5.0: the controlled VoiceMem ownership artifacts
    "VOICEMEM_PIN.json", "UPSTREAM_POLICY.md",
)

#: The M0 placeholder directory skeleton the release ZIP must ship (the
#: ZIP format does not reliably store empty directories, so the builder
#: stages a .gitkeep placeholder + the repo README.md in each). Must mirror
#: the placeholder list of scripts/build_release.ps1 (step 4a/8).
PLACEHOLDER_DIRS = (
    "models/asr/qwen3-asr-0.6b",
    "models/llm/qwen3.6-35b-a3b",
    "models/tts/piper",
    "models/vad/silero-vad",
    "models/embedding/multilingual-e5-small",
    "models/hf",
    "models/emotion",
    "models/speaker",
    "memory/sqlite",
    "memory/qdrant",
    "memory/backups",
)

#: The release that first shipped the M0 directory skeleton + its own
#: CHANGELOG entry inside the ZIP (v0.1.6 release-blocker fix). Deep checks
#: skip zips that legitimately predate the fix.
SKELETON_FIX_VERSION = (0, 1, 6)


def _latest_entry(entries):
    """The highest-version index entry, whatever order the index uses.

    v0.4.6 bug fix: the deep tests used to take ``entries[-1]`` as "the
    newest release", but build_release_sandbox.py writes the index
    NEWEST-FIRST (a cosmetic re-sort), so ``entries[-1]`` was the OLDEST
    zip - with a fully populated releases/ folder the M0-skeleton deep
    check then tested a v0.3.0 zip and failed. Selecting by version tuple
    makes the checks order-independent (append order or newest-first).
    """
    return max(entries, key=lambda e: tuple(int(p) for p in e["version"].split(".")))

#: The first release whose ZIP ships the LOCAL web UI (web/voicemem.html,
#: app/web_server.py, scripts/smoke_test_web.py, the web tests) - the
#: v0.3.5 completeness fix: the ZIP must contain EVERY element of the
#: system. Deep checks skip zips that legitimately predate the web UI.
WEB_UI_FIX_VERSION = (0, 3, 5)

#: The web UI payload the v0.3.5+ ZIP must carry (paths relative to the
#: VoiceMemAgent_vX.Y.Z/ prefix inside the archive).
WEB_UI_ZIP_PAYLOAD = (
    "web/voicemem.html",
    "web/images/background.webp",
    "app/web_server.py",
    "scripts/smoke_test_web.py",
    "tests/unit/test_web_server.py",
    "tests/integration/test_web_e2e.py",
)

#: The first release whose START.bat ships with Windows CRLF line endings
#: and the unknown-mode fallback to the web UI (v0.3.6 field report: the
#: user's START.bat dead-ended at the usage menu because an unexpected
#: first argument reached the batch). Deep checks skip zips that
#: legitimately predate the fix.
START_BAT_CRLF_VERSION = (0, 3, 6)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class ReleaseVersionFeatureTest(FeatureValidationTest):
    """Version bookkeeping: VERSION / pyproject / YAML / CHANGELOG sync."""

    FEATURE = "release"

    def test_logic_version_files_are_in_sync(self):
        version = _read_text(REPO_ROOT / "VERSION").strip()
        self.assertTrue(
            SEMVER.match(version), f"VERSION ('{version}') is not X.Y.Z semver"
        )

        pyproject = _read_text(REPO_ROOT / "pyproject.toml")
        m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
        self.assertIsNotNone(m, "pyproject.toml [project] version not found")
        self.assertEqual(m.group(1), version, "pyproject.toml version != VERSION file")

        yaml_text = _read_text(REPO_ROOT / "config" / "voicemem_config.yaml")
        m = re.search(r'(?m)^(\s*)version:\s*"([^"]+)"', yaml_text)
        self.assertIsNotNone(m, "config YAML project.version not found")
        self.assertEqual(
            m.group(2), version, "config YAML project.version != VERSION file"
        )

    def test_logic_changelog_has_entry_for_current_version(self):
        version = _read_text(REPO_ROOT / "VERSION").strip()
        changelog = _read_text(REPO_ROOT / "CHANGELOG.md")
        headings = CHANGELOG_HEADING.findall(changelog)
        self.assertTrue(
            headings, "CHANGELOG.md has no '## [X.Y.Z] - YYYY-MM-DD' headings"
        )
        self.assertIn(
            version,
            [h[0] for h in headings],
            f"CHANGELOG.md has no entry for the current version {version}",
        )

    def test_logic_changelog_versions_are_unique_and_sorted(self):
        changelog = _read_text(REPO_ROOT / "CHANGELOG.md")
        versions = [h[0] for h in CHANGELOG_HEADING.findall(changelog)]
        self.assertEqual(
            len(versions), len(set(versions)), "duplicate CHANGELOG versions"
        )

    def test_logic_release_index_schema_and_zip_integrity(self):
        index_path = REPO_ROOT / "releases" / "RELEASE_INDEX.json"
        zips = sorted((REPO_ROOT / "releases").glob("*.zip"))
        if not index_path.is_file():
            # Fresh clone: index is created by the first build; but then no
            # ZIP may exist either.
            self.assertEqual(
                zips, [], "ZIPs exist in releases/ without RELEASE_INDEX.json"
            )
            return
        data = json.loads(_read_text(index_path))
        self.assertEqual(data.get("schema_version"), 1)
        self.assertIsInstance(data.get("releases"), list)
        for entry in data["releases"]:
            for key in ("version", "built_at_utc", "zip", "sha256", "size_bytes"):
                self.assertIn(key, entry, f"index entry missing '{key}'")
            ver = entry["version"]
            self.assertTrue(SEMVER.match(ver), f"index version '{ver}' invalid")
            self.assertEqual(entry["zip"], f"VoiceMemAgent_v{ver}.zip")
            zip_path = REPO_ROOT / "releases" / entry["zip"]
            sha_path = zip_path.with_name(entry["zip"] + ".sha256")
            self.assertTrue(zip_path.is_file(), f"indexed ZIP missing: {entry['zip']}")
            self.assertTrue(
                sha_path.is_file(), f"checksum file missing: {entry['zip']}.sha256"
            )
            if zip_path.is_file():
                digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
                self.assertEqual(
                    digest, entry["sha256"].lower(),
                    f"sha256 mismatch for {entry['zip']}",
                )
                recorded = sha_path.read_text(encoding="utf-8").split()[0].lower()
                self.assertEqual(recorded, digest, ".sha256 file content mismatch")

    def test_logic_build_release_gate_contract(self):
        """build_release.ps1 must: run run_tests.ps1 BEFORE creating the ZIP."""
        script = REPO_ROOT / "scripts" / "build_release.ps1"
        self.assertTrue(script.is_file(), "scripts/build_release.ps1 missing")
        raw = script.read_bytes()
        self.assertLessEqual(
            max(raw), 0x7F, "build_release.ps1 must be pure ASCII (PS 5.1)"
        )
        text = raw.decode("ascii")
        for marker in (
            "run_tests.ps1",          # the mandatory test gate
            "CreateFromDirectory",    # ZIP creation
            "Get-FileHash",           # sha256 checksum
            "RELEASE_INDEX.json",     # release ledger
            "CHANGELOG.md",           # changelog update
            "BUILD_INFO.json",        # metadata inside the ZIP
            "validation_report.json",  # deep-validation report consumption
            "StrictValidation",
            "START.bat",              # M0.2: the ZIP ships the one-click entry
            "MODELS.lock.json",       # M0.2: the ZIP ships the model lock
        ):
            self.assertIn(marker, text, f"build_release.ps1 missing '{marker}'")
        self.assertIn('[string]$Bump = "patch"', text, "default bump must be patch")

    def test_logic_build_release_stages_web_ui(self):
        """v0.3.5 completeness: the builder's staging allowlist must include
        the web/ directory (the ZIP ships every element of the system, the
        web UI included) and the models/speaker placeholder."""
        script = REPO_ROOT / "scripts" / "build_release.ps1"
        self.assertTrue(script.is_file(), "scripts/build_release.ps1 missing")
        text = script.read_bytes().decode("ascii")
        self.assertIn(
            '@("app", "config", "scripts", "tests", "web", "vendor")', text,
            "build_release.ps1 step 4/8 must stage the web/ directory",
        )
        self.assertIn(
            '"models/speaker"', text,
            "build_release.ps1 placeholder list must include models/speaker",
        )

    def test_logic_staging_allowlist_exists(self):
        """Every path the ZIP staging allowlist references must exist."""
        for d in STAGED_DIRS:
            self.assertTrue(
                (REPO_ROOT / d).is_dir(), f"staged directory missing: {d}/"
            )
        for f in STAGED_FILES:
            self.assertTrue(
                (REPO_ROOT / f).is_file(), f"staged file missing: {f}"
            )

    def test_logic_build_release_stages_m0_skeleton(self):
        """Release-blocker regression (v0.1.6): the ZIP must ship the M0
        placeholder directory tree and its own CHANGELOG entry.

        Static contract on scripts/build_release.ps1: the placeholder
        directory list, the .gitkeep markers, the Add-ChangelogEntry
        mechanism (staged copy gets the entry BEFORE the ZIP is created)
        and the in-ZIP self-check that hard-fails the build otherwise.
        """
        script = REPO_ROOT / "scripts" / "build_release.ps1"
        self.assertTrue(script.is_file(), "scripts/build_release.ps1 missing")
        text = script.read_bytes().decode("ascii")
        for d in PLACEHOLDER_DIRS:
            self.assertIn(
                d, text,
                f"build_release.ps1 does not stage the M0 directory '{d}'",
            )
        self.assertIn(".gitkeep", text, "the builder must stage .gitkeep placeholders")
        self.assertIn(
            "Add-ChangelogEntry", text,
            "the builder must write the own-version CHANGELOG entry into the "
            "staged copy before creating the ZIP",
        )
        self.assertIn(
            "ZIP self-check", text,
            "the builder must self-check the finished ZIP (dirs + changelog)",
        )

    def test_logic_validation_report_writer(self):
        """_report.py must write the JSON report when the env var is set."""
        from tests.validation import _report

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            saved = dict(_report._FEATURES)
            try:
                with patch.dict(
                    os.environ, {"VMA_VALIDATION_REPORT": str(out)}
                ):
                    _report.record_deep_pass("release-selftest", "detail")
                    _report.record_deep_skip("release-selftest-2", "reason")
                    _report._write_report()
                data = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(data["schema_version"], 1)
                self.assertIn("generated_at", data)
                self.assertEqual(
                    data["features"]["release-selftest"]["status"], "pass"
                )
                self.assertEqual(
                    data["features"]["release-selftest-2"]["status"], "skipped"
                )
            finally:
                _report._FEATURES.clear()
                _report._FEATURES.update(saved)

    def test_deep_zip_inner_build_info(self):
        """When a release ZIP exists, BUILD_INFO.json inside must match it."""
        index_path = REPO_ROOT / "releases" / "RELEASE_INDEX.json"
        if not index_path.is_file():
            self.deep_skip("no RELEASE_INDEX.json yet (no build ran)")
        data = json.loads(_read_text(index_path))
        entries = [e for e in data.get("releases", []) if "zip" in e]
        if not entries:
            self.deep_skip("no release ZIP built yet")
        checked = 0
        for entry in entries:
            zip_path = REPO_ROOT / "releases" / entry["zip"]
            if not zip_path.is_file():
                continue
            with zipfile.ZipFile(zip_path) as zf:
                names = [n.replace("\\", "/") for n in zf.namelist()]
                expected = f"VoiceMemAgent_v{entry['version']}/BUILD_INFO.json"
                self.assertIn(
                    expected, names, f"BUILD_INFO.json missing inside {entry['zip']}"
                )
                info = json.loads(zf.read(expected).decode("utf-8"))
                self.assertEqual(info["version"], entry["version"])
                self.assertIn("built_at_utc", info)
                self.assertIn("test_gate", info)
            checked += 1
        self.assertGreater(checked, 0)
        self.deep_pass(f"BUILD_INFO.json verified inside {checked} ZIP(s)")

    def test_deep_latest_zip_m0_skeleton_and_changelog(self):
        """Release-blocker regression (v0.1.6): the NEWEST release ZIP must
        ship the M0 placeholder directory skeleton (with .gitkeep) and its
        own CHANGELOG entry (extractable content, not just the builder
        contract). Zips that predate the fix are skipped explicitly."""
        index_path = REPO_ROOT / "releases" / "RELEASE_INDEX.json"
        if not index_path.is_file():
            self.deep_skip("no RELEASE_INDEX.json yet (no build ran)")
        data = json.loads(_read_text(index_path))
        entries = [e for e in data.get("releases", []) if e.get("zip")]
        if not entries:
            self.deep_skip("no release ZIP built yet")
        latest = _latest_entry(entries)
        ver = tuple(int(p) for p in latest["version"].split("."))
        if ver < SKELETON_FIX_VERSION:
            self.deep_skip(
                f"latest ZIP {latest['version']} predates the M0 skeleton fix"
            )
        zip_path = REPO_ROOT / "releases" / latest["zip"]
        if not zip_path.is_file():
            self.deep_skip(f"latest ZIP file missing on disk: {latest['zip']}")
        prefix = f"VoiceMemAgent_v{latest['version']}/"
        with zipfile.ZipFile(zip_path) as zf:
            names = [n.replace("\\", "/") for n in zf.namelist()]
            for d in PLACEHOLDER_DIRS:
                self.assertTrue(
                    any(n.startswith(prefix + d + "/") for n in names),
                    f"{latest['zip']} does not ship the M0 directory '{d}/'",
                )
            changelog_entry = prefix + "CHANGELOG.md"
            self.assertIn(
                changelog_entry, names, "CHANGELOG.md missing inside the ZIP"
            )
            changelog_text = zf.read(changelog_entry).decode("utf-8")
            headings = [h[0] for h in CHANGELOG_HEADING.findall(changelog_text)]
            self.assertIn(
                latest["version"], headings,
                f"the ZIP's own CHANGELOG.md has no entry for "
                f"{latest['version']}",
            )
        self.deep_pass(
            f"{latest['zip']}: {len(PLACEHOLDER_DIRS)} M0 dirs + own "
            "CHANGELOG entry verified"
        )

    def test_deep_latest_zip_ships_web_ui(self):
        """v0.3.5 completeness fix: the NEWEST release ZIP must ship the
        LOCAL web UI payload (web/voicemem.html + images, app/web_server.py,
        scripts/smoke_test_web.py, the web tests). Zips that predate the
        web-UI milestone are skipped explicitly."""
        index_path = REPO_ROOT / "releases" / "RELEASE_INDEX.json"
        if not index_path.is_file():
            self.deep_skip("no RELEASE_INDEX.json yet (no build ran)")
        data = json.loads(_read_text(index_path))
        entries = [e for e in data.get("releases", []) if e.get("zip")]
        if not entries:
            self.deep_skip("no release ZIP built yet")
        latest = _latest_entry(entries)
        ver = tuple(int(p) for p in latest["version"].split("."))
        if ver < WEB_UI_FIX_VERSION:
            self.deep_skip(
                f"latest ZIP {latest['version']} predates the web-UI milestone"
            )
        zip_path = REPO_ROOT / "releases" / latest["zip"]
        if not zip_path.is_file():
            self.deep_skip(f"latest ZIP file missing on disk: {latest['zip']}")
        prefix = f"VoiceMemAgent_v{latest['version']}/"
        with zipfile.ZipFile(zip_path) as zf:
            names = [n.replace("\\", "/") for n in zf.namelist()]
            for rel in WEB_UI_ZIP_PAYLOAD:
                self.assertIn(
                    prefix + rel, names,
                    f"{latest['zip']} does not ship the web UI element '{rel}'",
                )
            ui = zf.read(prefix + "web/voicemem.html").decode("utf-8")
            for marker in ("'/ws'", "micSel", "pipeStrip", "XTransformPort",
                           "asrBypassed", "SAMPLE_RATE=24000"):
                self.assertIn(
                    marker, ui,
                    f"web/voicemem.html inside {latest['zip']} lost the "
                    f"marker {marker!r}",
                )
        self.deep_pass(
            f"{latest['zip']}: web UI payload verified "
            f"({len(WEB_UI_ZIP_PAYLOAD)} files + UI markers)"
        )

    def test_deep_latest_zip_start_bat_crlf(self):
        """v0.3.6 field report: the NEWEST release ZIP must ship START.bat
        with Windows CRLF line endings AND the unknown-mode fallback that
        continues in the default web-UI mode (an unexpected %1 - drag-and-
        drop, a custom association - must never dead-end the user at a
        usage menu). Zips that predate the fix are skipped explicitly."""
        index_path = REPO_ROOT / "releases" / "RELEASE_INDEX.json"
        if not index_path.is_file():
            self.deep_skip("no RELEASE_INDEX.json yet (no build ran)")
        data = json.loads(_read_text(index_path))
        entries = [e for e in data.get("releases", []) if e.get("zip")]
        if not entries:
            self.deep_skip("no release ZIP built yet")
        latest = _latest_entry(entries)
        ver = tuple(int(p) for p in latest["version"].split("."))
        if ver < START_BAT_CRLF_VERSION:
            self.deep_skip(
                f"latest ZIP {latest['version']} predates the START.bat "
                "CRLF + fallback fix"
            )
        zip_path = REPO_ROOT / "releases" / latest["zip"]
        if not zip_path.is_file():
            self.deep_skip(f"latest ZIP file missing on disk: {latest['zip']}")
        prefix = f"VoiceMemAgent_v{latest['version']}/"
        with zipfile.ZipFile(zip_path) as zf:
            names = [n.replace("\\", "/") for n in zf.namelist()]
            self.assertIn(prefix + "START.bat", names,
                          f"{latest['zip']} does not ship START.bat")
            raw = zf.read(prefix + "START.bat")
        lone_lf = raw.count(b"\n") - raw.count(b"\r\n")
        self.assertEqual(
            lone_lf, 0,
            f"START.bat inside {latest['zip']} has {lone_lf} LF-only lines "
            "- a Windows .bat must be CRLF (cmd.exe parser hazard)",
        )
        self.assertGreater(raw.count(b"\r\n"), 100,
                           f"START.bat inside {latest['zip']} looks truncated")
        bat = raw.decode("ascii")
        self.assertIn("[NOTE] Unknown start mode", bat,
                      f"START.bat inside {latest['zip']} lost the v0.3.6 "
                      "unknown-mode fallback note")
        self.assertIn('set "MODE=run"', bat,
                      f"START.bat inside {latest['zip']} does not fall back "
                      "to the web-UI run mode")
        self.assertIn("http://127.0.0.1:8787", bat,
                      f"START.bat inside {latest['zip']} lost the web UI URL")
        self.assertIn('"%MODE%"=="web" set "MODE=run"', bat,
                      f"START.bat inside {latest['zip']} lost the web alias")
        self.assertNotIn('Unknown mode: "%MODE%"', bat,
                         f"START.bat inside {latest['zip']} still has the "
                         "v0.3.5 dead-end unknown-mode error block")
        self.deep_pass(
            f"{latest['zip']}: START.bat is CRLF "
            f"({raw.count(b'\r\n')} lines) + unknown-mode fallback present"
        )


if __name__ == "__main__":
    unittest.main()
