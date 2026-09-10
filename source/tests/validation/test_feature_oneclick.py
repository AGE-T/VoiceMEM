r"""One-click UX feature validation (Task 13 - M0.2).

The user-facing requirement validated here: "START.bat is the ONLY entry
point the user ever needs - double-click and everything (Python check, venv,
dependencies, Hugging Face tooling, model download, config, verification,
smoke tests, the agent itself) happens automatically. No user-facing
message may ever defer fixable work back to the user (install the CLI
manually / run the installer first / pip install ...)."

Validated contracts:
* START.bat exists, is pure ASCII, resolves the project root from its own
  location (relocatable project), maps all six modes (run, mock, check,
  benchmark, repair, build), hands over to scripts\bootstrap.ps1 and keeps
  the console open (pause) for double-click users;
* scripts\bootstrap.ps1 is the internal orchestrator: param block first,
  PS 5.1-safe, drives the existing idempotent scripts as child processes,
  reads/writes .install_state.json with the six mandatory flags, checks
  models via MODELS.lock.json, logs to logs\bootstrap.log, and implements
  the "environment already ready" fast path (idempotency);
* scripts\download_models.ps1 SELF-HEALS a missing huggingface_hub
  LIBRARY (auto-installs it into the .venv) and never prints the old
  manual-install instructions; the download itself runs through the
  huggingface_hub PYTHON API (scripts/download_models_hf.py via the .venv
  interpreter) - the Hugging Face CLI is NOT a runtime dependency
  (v0.1.6 release blocker: the CLI's emoji deprecation warning crashed
  the child Python with UnicodeEncodeError under a CP1252 console);
* MODELS.lock.json ships with the repo (schema 2: the M1 strict model set
  + the llama.cpp/piper tool pins) and matches the M0 path layout;
* no user-facing script instructs the user to run manual pip / installer /
  CLI commands for things the system can fix itself.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
START_BAT = REPO_ROOT / "START.bat"
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap.ps1"
DOWNLOAD = REPO_ROOT / "scripts" / "download_models.ps1"
START_AGENT = REPO_ROOT / "scripts" / "start_agent.ps1"
MODELS_LOCK = REPO_ROOT / "MODELS.lock.json"
GITIGNORE = REPO_ROOT / ".gitignore"
STATE_FILE = REPO_ROOT / ".install_state.json"

#: The six START.bat modes (user spec 18) + the default.
MODES = ("run", "mock", "check", "benchmark", "repair", "build")

#: User spec 15: the mandatory .install_state.json flags.
STATE_KEYS = (
    "python_ready", "venv_ready", "dependencies_ready",
    "models_ready", "config_ready", "smoke_tests_passed",
)

#: PS 5.1 does not support these operators (PowerShell 7+ syntax).
PS7_ONLY_TOKENS = ("??", "?.", "&&", "||")

#: Messages that must NEVER be user-facing (user spec 3): manual work the
#: system can do itself.
FORBIDDEN_PHRASES = (
    "Telepitsd a .venv-be",
    "Telepitsd a huggingface",
    "Futtasd elobb",
    "install manually",
    "vagy kezzel:",
    "vagy potold kezzel",
)

#: The M0 section-7 canonical model layout the lock must pin.
CANONICAL_TARGETS = {
    "llm": "models/llm/qwen3.6-35b-a3b",
    "asr": "models/asr/qwen3-asr-0.6b",
    "embedding": "models/embedding/multilingual-e5-small",
    "vad": "models/vad/silero-vad",
    "tts": "models/tts/piper",
    "emotion": "models/emotion/emotion2vec-plus-base",
    "speaker": "models/speaker/ecapa-voxceleb",
}


def _strip_ps_comments(text: str) -> str:
    """Remove <# ... #> block comments and full-line # comments."""
    without_blocks = re.sub(r"<#.*?#>", "", text, flags=re.DOTALL)
    lines = [ln for ln in without_blocks.splitlines() if not ln.strip().startswith("#")]
    return "\n".join(lines)


def _read_ascii(path: Path) -> str:
    raw = path.read_bytes()
    assert max(raw) <= 0x7F, f"{path.name} must be pure ASCII"
    return raw.decode("ascii")


class OneClickFeatureTest(FeatureValidationTest):
    """START.bat + bootstrap + self-healing downloader + shipped lock."""

    FEATURE = "oneclick"

    # ---------------------------------------------------------- START.bat ----

    def test_logic_start_bat_exists_and_ascii(self):
        self.assertTrue(START_BAT.is_file(), "START.bat missing at the repo root")
        self.assertGreater(START_BAT.stat().st_size, 0, "START.bat is empty")
        raw = START_BAT.read_bytes()
        self.assertLessEqual(max(raw), 0x7F, "START.bat must be pure ASCII")

    def test_logic_start_bat_contract(self):
        """START.bat: relocatable root, all modes, bootstrap handover, pause."""
        text = _read_ascii(START_BAT)
        self.assertIn("%~dp0", text, "START.bat must resolve the root from its own location")
        self.assertIn("cd /d", text, "START.bat must cd to the project root")
        self.assertIn("scripts\\bootstrap.ps1", text, "START.bat must call scripts\\bootstrap.ps1")
        self.assertIn("-ExecutionPolicy Bypass", text, "START.bat must bypass the execution policy")
        self.assertIn('-Mode "%MODE%"', text, "START.bat must forward the mode to the bootstrap")
        for mode in MODES:
            self.assertIn(mode, text, f"START.bat does not know the mode '{mode}'")
        self.assertIn("pause", text, "START.bat must pause so double-click users can read output")

    def test_logic_start_bat_windows_crlf(self):
        """v0.3.6 field report: START.bat must carry Windows CRLF line
        endings. The v0.3.5 ZIP shipped an LF-only .bat (0 CR bytes) - a
        known cmd.exe parser hazard. Every \\n must be preceded by \\r."""
        raw = START_BAT.read_bytes()
        lone_lf = raw.count(b"\n") - raw.count(b"\r\n")
        self.assertEqual(lone_lf, 0, f"START.bat has {lone_lf} LF-only lines - "
                                    "a Windows .bat must be CRLF")
        self.assertGreater(raw.count(b"\r\n"), 100, "START.bat looks truncated")

    def test_logic_start_bat_unknown_mode_falls_back(self):
        """v0.3.6 field report: an unknown first argument (drag-and-drop,
        a custom .bat association) must never dead-end at a usage menu -
        the batch prints a note and CONTINUES in the default web-UI mode.
        The v0.3.5 error block exited with code 1 after 'Unknown mode'."""
        text = _read_ascii(START_BAT)
        self.assertIn("[NOTE] Unknown start mode", text,
                      "START.bat lost the v0.3.6 unknown-mode fallback note")
        # the fallback must RESET the mode to the web-UI default
        self.assertGreaterEqual(
            text.count('set "MODE=run"'), 2,
            "the unknown-mode fallback must set MODE=run (web UI)",
        )
        self.assertIn("http://127.0.0.1:8787", text,
                      "START.bat must keep documenting the web UI URL")
        # the v0.3.5 dead-end is gone: no 'Unknown mode' error + exit block
        self.assertNotIn('Unknown mode: "%MODE%"', text,
                         "the v0.3.5 dead-end unknown-mode error is still present")
        # parse safety: %MODE% must never be echoed inside the fallback
        # block (a dropped path with parentheses would break the block)
        lines = text.splitlines()
        in_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("if /I not") and stripped.endswith("("):
                in_block = True
                continue
            if in_block and stripped == ")":
                in_block = False
                continue
            if in_block:
                self.assertFalse(
                    "echo" in stripped and "%MODE%" in stripped,
                    f"%MODE% echoed inside the fallback block: {stripped!r}",
                )

    def test_logic_start_bat_resolves_64bit_powershell(self):
        """Release-blocker regression (v0.1.6): START.bat must launch 64-bit
        PowerShell. The generic 'powershell' PATH lookup is forbidden (a
        32-bit cmd.exe or a SysWOW64-first PATH could resolve the 32-bit
        engine): primary is 64-bit PowerShell 7, secondary is the NATIVE
        System32 Windows PowerShell, 32-bit is explicitly excluded.
        """
        text = _read_ascii(START_BAT)
        # 1) never the generic PATH lookup for powershell.exe
        self.assertNotIn(
            "where powershell", text,
            "START.bat must not resolve powershell via the generic PATH lookup",
        )
        # 2) launch happens through the resolved variable, fully quoted
        self.assertIn(
            '"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass', text,
            "START.bat must launch the resolved 64-bit engine via %PS_EXE%",
        )
        # 3) primary: 64-bit PowerShell 7 standard install location
        self.assertIn(
            "%ProgramW6432%\\PowerShell\\7\\pwsh.exe", text,
            "START.bat primary engine must be 64-bit PowerShell 7 "
            "(%ProgramW6432%\\PowerShell\\7\\pwsh.exe)",
        )
        # 4) secondary: native System32 Windows PowerShell (Sysnative-aware:
        #    a 32-bit cmd.exe gets System32 redirected to SysWOW64, so the
        #    native dir flips to Sysnative to reach the REAL 64-bit System32)
        self.assertIn(
            "%PS_NATIVE_DIR%\\WindowsPowerShell\\v1.0\\powershell.exe", text,
            "START.bat secondary engine must be the native System32 "
            "Windows PowerShell",
        )
        self.assertIn(
            'set "PS_NATIVE_DIR=%SystemRoot%\\System32"', text,
            "64-bit cmd.exe resolves the native dir from the real System32",
        )
        self.assertIn(
            'set "PS_NATIVE_DIR=%SystemRoot%\\Sysnative"', text,
            "32-bit cmd.exe must use Sysnative (WOW64 redirection bypass)",
        )
        self.assertIn("PROCESSOR_ARCHITEW6432", text, "32-bit cmd.exe detection missing")
        # 5) explicit 32-bit exclusion while scanning PATH for pwsh
        self.assertIn("SysWOW64", text, "SysWOW64 32-bit location must be excluded")
        self.assertIn("Program Files (x86)", text, "32-bit Program Files must be excluded")

    def test_logic_bootstrap_refuses_32bit_process(self):
        """bootstrap.ps1 must enforce the 64-bit engine at runtime."""
        text = _read_ascii(BOOTSTRAP)
        self.assertIn(
            "[Environment]::Is64BitProcess", text,
            "bootstrap.ps1 must guard against 32-bit PowerShell processes",
        )

    # --------------------------------------------------------- bootstrap -----

    def test_logic_bootstrap_contract(self):
        """bootstrap.ps1: modes, state, lock, log, orchestration, fast path."""
        text = _read_ascii(BOOTSTRAP)
        code = _strip_ps_comments(text)

        for marker in (
            "install_m1.ps1",       # heavy idempotent install engine
            "start_agent.ps1",      # run / mock dispatch
            "verify_m1.ps1",        # check mode asset verification
            "run_tests.ps1",        # check mode test gate
            "build_release.ps1",    # build mode
            ".install_state.json",  # resumable install state (spec 15)
            "MODELS.lock.json",     # model presence check (spec 13)
            "logs\\bootstrap.log",  # technical log (spec 21/22)
            "environment already ready",  # idempotent fast path (spec 16)
        ):
            self.assertIn(marker, text, f"bootstrap.ps1 missing '{marker}'")

        self.assertIn(
            "[ValidateSet('run', 'cli', 'mock', 'check', 'benchmark', 'repair', 'build')]",
            text, "bootstrap.ps1 must support the six START.bat modes",
        )
        self.assertIn("$Mode -eq 'repair'", text, "bootstrap.ps1 must handle repair mode")
        for key in STATE_KEYS:
            self.assertIn(key, text, f"bootstrap.ps1 must track the state flag '{key}'")

        for token in PS7_ONLY_TOKENS:
            self.assertNotIn(
                token, code,
                f"bootstrap.ps1 uses the PowerShell 7-only operator {token!r}",
            )

    def test_logic_bootstrap_param_first_statement(self):
        code = _strip_ps_comments(_read_ascii(BOOTSTRAP))
        nonempty = [ln.strip() for ln in code.splitlines() if ln.strip()]
        self.assertTrue(nonempty, "bootstrap.ps1 has no code after comment strip")
        self.assertTrue(
            nonempty[0].startswith("param("),
            "bootstrap.ps1: param( must be the first executable statement",
        )

    def test_logic_bootstrap_clean_error_language(self):
        """Primary errors must be clean; detail goes to the log (spec 21)."""
        text = _read_ascii(BOOTSTRAP)
        self.assertIn("Python 3.11 is required.", text)
        self.assertIn("python.org", text)
        self.assertIn("Technical details: logs\\bootstrap.log", text)

    # ------------------------------------------------- self-healing HF lib --

    def test_logic_downloader_self_heals_hf_library(self):
        """download_models.ps1 must auto-install the huggingface_hub LIBRARY.

        v0.1.6 release blocker regression guard: the download used to shell
        out to huggingface-cli.exe, whose deprecation warning printed an
        EMOJI and crashed the child Python with UnicodeEncodeError under a
        CP1252 Windows console - before a single byte arrived. The wrapper
        must auto-install the huggingface_hub LIBRARY (never the [cli]
        extra) and invoke the venv interpreter with the Python API helper.
        """
        text = _read_ascii(DOWNLOAD)
        self.assertIn(
            '-m pip install "huggingface_hub"',
            text,
            "download_models.ps1 must auto-install the huggingface_hub "
            "library into the .venv (one-click rule)",
        )
        self.assertNotIn(
            'huggingface_hub[cli]',
            text,
            "the [cli] extra must NOT be installed - the Hugging Face CLI "
            "is not a runtime dependency",
        )
        self.assertIn(
            "AUTOMATIKUS telepites", text,
            "the auto-install must be logged as automatic",
        )
        self.assertIn(
            "download_models_hf.py", text,
            "the wrapper must invoke the Python API helper script",
        )
        self.assertIn(
            ".venv\\Scripts\\python.exe", text,
            "the download must always run with the project venv interpreter",
        )

    def test_logic_downloader_uses_hf_python_api(self):
        """The helper must drive hf_hub_download/snapshot_download in-process."""
        helper = REPO_ROOT / "scripts" / "download_models_hf.py"
        self.assertTrue(helper.is_file(), "scripts/download_models_hf.py missing")
        raw = helper.read_bytes()
        self.assertLessEqual(
            max(raw), 0x7F,
            "download_models_hf.py must be pure ASCII (any Windows codepage)",
        )
        code = raw.decode("ascii")
        self.assertIn(
            "hf_hub_download(", code,
            "the helper must call hf_hub_download() directly",
        )
        self.assertIn(
            "snapshot_download(", code,
            "the helper must call snapshot_download() directly",
        )
        self.assertIn(
            "MODELS.lock.json", code,
            "the helper must be driven by MODELS.lock.json",
        )
        self.assertIn(
            "with_retries", code,
            "the helper must implement download retry",
        )
        self.assertIn(
            "pinned_revision", code,
            "the helper must honour pinned_revision (revision support)",
        )
        # v0.4.16: the LLM is NOT auto-downloaded - the Qwen3.6 35B A3B
        # IQ4_XS GGUF is OPERATOR-PLACED (any drive; web UI model picker /
        # find_qwen_gguf.ps1), NO mirror chain, NO fallback profile. The VAD
        # mirror stays (not an LLM).
        self.assertIn("OPERATOR-PLACED", code)
        self.assertIn("bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", code)
        for banned in ("unsloth/Qwen3-8B-GGUF", "fallback_repos"):
            self.assertNotIn(banned, code,
                             f"the LLM must have no mirror chain/fallback ({banned})")
        self.assertIn("tphakala/silero-vad", code, "the VAD mirror chain must be present")

    def test_logic_no_hf_cli_subprocess_regression(self):
        """No orchestration script may invoke or install the HF CLI.

        Regression guard for the v0.1.6 model-download blocker: the
        huggingface-cli.exe subprocess crashed with UnicodeEncodeError
        (emoji + CP1252 console) before downloading anything.
        """
        scripts = REPO_ROOT / "scripts"
        for name in ("download_models.ps1", "install_m1.ps1", "bootstrap.ps1"):
            code = _strip_ps_comments(_read_ascii(scripts / name))
            for token in ("huggingface-cli", "huggingface_hub[cli]", "hf.exe"):
                self.assertNotIn(
                    token, code,
                    f"{name} still references the HF CLI artifact {token!r}",
                )

    def test_logic_no_manual_install_instructions(self):
        """User-facing scripts must not defer fixable work to the user."""
        for path in (START_BAT, BOOTSTRAP, DOWNLOAD, START_AGENT):
            code = _strip_ps_comments(_read_ascii(path))
            for phrase in FORBIDDEN_PHRASES:
                self.assertNotIn(
                    phrase, code,
                    f"{path.name} contains the forbidden manual-install "
                    f"instruction {phrase!r}",
                )

    def test_logic_start_agent_points_to_start_bat(self):
        """start_agent.ps1 repair hints must point to START.bat."""
        code = _strip_ps_comments(_read_ascii(START_AGENT))
        self.assertIn("START.bat", code, "start_agent.ps1 must point fixes at START.bat")

    # -------------------------------------------------------- MODELS.lock ----

    def test_logic_models_lock_schema(self):
        self.assertTrue(MODELS_LOCK.is_file(), "MODELS.lock.json must ship with the repo")
        raw = MODELS_LOCK.read_bytes()
        self.assertLessEqual(max(raw), 0x7F, "MODELS.lock.json must be pure ASCII")
        lock = json.loads(raw.decode("ascii"))

        self.assertEqual(lock.get("schema_version"), 2)
        models = lock.get("models")
        self.assertIsInstance(models, list)
        self.assertEqual(
            len(models), 6,
            "v0.4.16: 6 auto-download entries (the LLM is operator-placed, "
            "removed from the set in v0.4.16 - no Gemma fallback)",
        )
        components = set()
        for entry in models:
            for field in ("component", "repo", "target_dir", "files", "pinned_revision"):
                self.assertIn(field, entry, f"lock entry missing '{field}'")
            self.assertIsInstance(entry["files"], list)
            self.assertGreater(len(entry["files"]), 0)
            target = entry["target_dir"]
            self.assertTrue(target.startswith("models/"), f"target '{target}' must live under models/")
            self.assertNotIn("..", target, f"target '{target}' must not escape the repo")
            components.add(entry["component"])
        self.assertEqual(
            components, set(CANONICAL_TARGETS) - {"llm"},
            "v0.4.16: the llm is operator-placed (not in the lock)",
        )
        for component, target in CANONICAL_TARGETS.items():
            if component == "llm":
                continue  # v0.4.16: operator-placed, not in the lock
            entry = next(e for e in models if e["component"] == component)
            self.assertEqual(entry["target_dir"], target, f"{component} target mismatch")

        # Snapshot entries need a probe + a total-size floor.
        for entry in models:
            if entry.get("snapshot"):
                self.assertIn("snapshot_probe", entry)
                self.assertGreaterEqual(int(entry["min_total_mb"]), 100)

        # v0.4.16: NO llm entry at all - the Qwen3.6 35B A3B IQ4_XS GGUF is
        # operator-placed (never auto-downloaded, no fallback profile).
        llm_entries = [e for e in models if e["component"] == "llm"]
        self.assertEqual(
            llm_entries, [],
            "the LLM must NOT be in the auto-download set (operator-placed GGUF)",
        )
        components = {e["component"] for e in models}
        self.assertIn("asr", components)
        self.assertIn("vad", components)

        # The llama.cpp + piper tool pins (must match install_m1.ps1).
        tools = lock.get("tools")
        self.assertIsInstance(tools, list)
        tool_components = {t.get("component") for t in tools}
        self.assertEqual(tool_components, {"llama-server", "piper"})
        for tool in tools:
            self.assertEqual(tool.get("target_dir"), "bin")
            self.assertIn("tag", tool)
            self.assertIn("assets", tool)
        tool_files = {f for t in tools for f in t["files"]}
        self.assertEqual(tool_files, {"llama-server.exe", "piper.exe"})

    def test_logic_gitignore_tracks_lock_and_state(self):
        text = GITIGNORE.read_text(encoding="utf-8")
        lines = {ln.strip() for ln in text.splitlines()}
        self.assertIn(".install_state.json", lines, "runtime install state must be ignored")
        self.assertNotIn(
            "MODELS.lock.json", lines,
            "MODELS.lock.json is a shipped repo file since M0.2 and must NOT be ignored",
        )


class OneClickStateFeatureTest(FeatureValidationTest):
    """Deep validation: the real .install_state.json after START.bat ran."""

    FEATURE = "oneclick-state"

    def test_deep_install_state_file_schema(self):
        if not STATE_FILE.is_file():
            self.deep_skip("START.bat has not run on this machine yet")
        raw = STATE_FILE.read_text(encoding="utf-8")
        state = json.loads(raw)
        for key in STATE_KEYS:
            self.assertIn(key, state, f".install_state.json missing '{key}'")
            self.assertIsInstance(state[key], bool, f"state flag '{key}' must be a boolean")
        self.assertEqual(state.get("schema_version"), 1)
        if not state["smoke_tests_passed"]:
            # START.bat ran, but the install did not complete on THIS machine
            # (e.g. wrong Python, no GPU) - an intermediate state, not a
            # validation failure. A completed install is validated below.
            self.deep_skip(
                f"install not completed here (last_result: {state.get('last_result', '?')})"
            )
        self.assertGreaterEqual(int(state.get("runs_count", 0)), 1)
        self.assertTrue(state["python_ready"], "python_ready must be True after a successful run")
        self.assertTrue(state["venv_ready"], "venv_ready must be True after a successful run")
        self.deep_pass(
            f"install state OK (run #{state['runs_count']}, python {state.get('python_version', '?')})"
        )


if __name__ == "__main__":
    unittest.main()
