"""Python invocation contract (Task 15 - v0.1.4 hotfix).

Regression tests for the Windows PowerShell call-operator trap that broke
the user's first run of v0.1.3 (log: "The term 'py -3.11' is not
recognized"):

    $PythonCmd = @("py", "-3.11")      # multi-element COMMAND array
    & $PythonCmd -m venv $VenvDir      # PS stringifies the array into ONE
                                       # (nonexistent) command name
                                       "py -3.11" -> CommandNotFoundException
                                       -> broken .venv creation

Contract under test:

* scripts NEVER pass a variable assigned a multi-element string-literal
  array directly to the call operator (`& $Var`);
* install_m1.ps1 resolves Python as an EXECUTABLE PATH + ARGUMENT ARRAY
  (`$PythonExe` + `$PythonArgs`) and creates the venv via
  `& $PythonExe @PythonArgs -m venv $VenvDir` at BOTH creation sites
  (first creation and the automatic rebuild branch);
* bootstrap.ps1 keeps its `& $Found.Exe @($Found.Extra)` probe pattern;
* deep: the invocation pattern is executed LIVE through a real PowerShell
  engine (powershell.exe / pwsh) against the machine's Python, proving
  `& $exe @args` works end-to-end (skipped when no PowerShell engine is
  on PATH - e.g. in the offline sandbox).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: install_m1.ps1 must create the venv through the fixed pattern at exactly
#: the two creation sites (first creation + automatic rebuild branch).
VENV_PATTERN = "& $PythonExe @PythonArgs -m venv $VenvDir"

#: A variable assigned a MULTI-ELEMENT string-literal array, e.g.
#: $X = @("py", "-3.11") or $X = @("a", "b", "c").
_ARRAY_ASSIGN_RE = re.compile(
    r"^\s*\$(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*@\(\s*"
    r"(?P<elems>\"[^\"\n]*\"\s*(?:,\s*\"[^\"\n]*\"\s*)*)"
    r"\)\s*$",
    re.MULTILINE,
)


def _strip_comments(text: str) -> str:
    """Drop block (<# #>) and full-line (#) comments before scanning."""
    text = re.sub(r"<#.*?#>", "", text, flags=re.DOTALL)
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _array_command_violations(text: str) -> list:
    """Return [(var, line_no)] where `& $Var` is invoked with $Var being a
    multi-element string-literal array (the PS 5.1 call-operator trap).

    Limitation (documented): only literal array assignments are detected;
    arrays built dynamically (e.g. `$a = @(); $a += "x"`) are covered by the
    install_m1-specific contract test below, not by this generic scan.
    """
    code = _strip_comments(text)
    violations = []
    for m in _ARRAY_ASSIGN_RE.finditer(code):
        elements = re.findall(r'"([^"\n]*)"', m.group("elems"))
        if len(elements) < 2:
            continue  # single-element arrays stringify to a valid command
        var = m.group("var")
        call_re = re.compile(r"&\s+\$" + re.escape(var) + r"\b")
        for cm in call_re.finditer(code):
            line_no = code[: cm.start()].count("\n") + 1
            violations.append((var, line_no))
    return violations


class PythonExecLogicTest(FeatureValidationTest):
    FEATURE = "pyexec"

    def test_logic_no_array_command_invocation(self):
        """No script may pass a multi-element array to the call operator."""
        scripts = sorted(SCRIPTS_DIR.glob("*.ps1"))
        self.assertTrue(scripts, "scripts/ must contain .ps1 files")
        problems = []
        for script in scripts:
            violations = _array_command_violations(script.read_text("utf-8"))
            for var, line_no in violations:
                problems.append(f"{script.name}:{line_no} (& ${var})")
        self.assertEqual(
            problems,
            [],
            "call-operator array trap found (PS 5.1 stringifies the array "
            "into one command name): " + "; ".join(problems),
        )

    def test_logic_install_m1_venv_pattern(self):
        """install_m1.ps1 must use the $PythonExe @PythonArgs pattern."""
        text = (SCRIPTS_DIR / "install_m1.ps1").read_text("utf-8")
        code = _strip_comments(text)
        self.assertEqual(
            code.count(VENV_PATTERN),
            2,
            "expected the fixed venv invocation at both creation sites "
            "(first creation + automatic rebuild)",
        )
        self.assertNotIn(
            "$PythonCmd ",
            code.replace("$PythonCmd =", ""),
            "the legacy array command variable must not exist",
        )
        self.assertIn("$PythonExe = ", code)
        self.assertIn("$PythonArgs = ", code)
        self.assertIn("& $PythonExe @PythonArgs", code)

    def test_logic_install_m1_rebuild_guard(self):
        """The rebuild branch must guard Remove-Item with Test-Path."""
        text = _strip_comments(
            (SCRIPTS_DIR / "install_m1.ps1").read_text("utf-8")
        )
        self.assertIn(
            "if (Test-Path $VenvDir) { Remove-Item -Path $VenvDir",
            text,
            "rebuild Remove-Item must be guarded (a missing .venv must not "
            "produce a raw PathNotFound error)",
        )

    def test_logic_bootstrap_probe_pattern(self):
        """bootstrap.ps1 keeps the exe + splatted-args probe pattern."""
        text = (SCRIPTS_DIR / "bootstrap.ps1").read_text("utf-8")
        code = _strip_comments(text)
        self.assertIn("& $Found.Exe @($Found.Extra)", code)
        self.assertIn("& $VenvPython", code)

    def test_logic_all_scripts_ascii(self):
        """PowerShell 5.1 without BOM reads non-ASCII as Windows-1252."""
        for script in sorted(SCRIPTS_DIR.glob("*.ps1")):
            data = script.read_bytes()
            non_ascii = [b for b in data if b >= 128]
            self.assertEqual(
                non_ascii,
                [],
                f"{script.name} must stay pure ASCII",
            )


class PythonExecDeepTest(FeatureValidationTest):
    FEATURE = "pyexec"

    def _find_powershell(self):
        for name in ("powershell", "powershell.exe", "pwsh"):
            exe = shutil.which(name)
            if exe:
                return exe
        return None

    def test_deep_live_python_invocation(self):
        """Run `& $exe @args -c` through a real PowerShell engine."""
        ps = self._find_powershell()
        if ps is None:
            self.deep_skip("no powershell.exe / pwsh on PATH")
        # Mirrors install_m1.ps1 discovery, with a python3 fallback so the
        # probe also works on non-Windows dev machines.
        snippet = r"""
$ErrorActionPreference = 'Stop'
$cands = @(
    @{ Exe = 'py';         Extra = @('-3.11') },
    @{ Exe = 'python3.11'; Extra = @() },
    @{ Exe = 'python3';    Extra = @() },
    @{ Exe = 'python';     Extra = @() }
)
$Found = $null
foreach ($c in $cands) {
    if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
    $Out = & $c.Exe @($c.Extra) --version 2>$null
    $OutText = (@($Out) -join ' ').Trim()
    if (($LASTEXITCODE -eq 0) -and ($OutText -ne '')) { $Found = $c; break }
}
if (-not $Found) { Write-Output 'NO_PYTHON'; exit 0 }
$Src = (Get-Command $Found.Exe -ErrorAction SilentlyContinue).Source
if ([string]::IsNullOrWhiteSpace($Src)) { $Src = $Found.Exe }
$Out = & $Src @($Found.Extra) -c 'print(2+3)'
if ($LASTEXITCODE -ne 0) { Write-Output ('INVOKE_FAIL exit=' + $LASTEXITCODE); exit 0 }
Write-Output ('PY_OK ' + ((@($Out) -join ' ').Trim()))
"""
        try:
            proc = subprocess.run(
                [ps, "-NoProfile", "-Command", snippet],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            self.fail("live PowerShell invocation probe timed out (60 s)")
        out = (proc.stdout or "").strip()
        if "NO_PYTHON" in out:
            self.deep_skip("no Python found on this machine")
        self.assertIn(
            "PY_OK 5",
            out,
            f"fixed invocation pattern failed: rc={proc.returncode} "
            f"stdout={out!r} stderr={(proc.stderr or '').strip()!r}",
        )
        self.deep_pass("& $exe @args live invocation OK through " + ps)


if __name__ == "__main__":
    unittest.main()
