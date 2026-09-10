"""GPU check validation (Task 17 - v0.1.5).

Regression tests for the installer step 4 NVIDIA false failure (v0.1.4,
user log: "Az nvidia-smi nem erheto el - nincs NVIDIA GPU vagy nincs driver
telepitve" on a machine with a healthy RTX 5070 + driver +
C:\\Windows\\System32\\nvidia-smi.exe; bootstrap log line: "GPU:
nvidia-smi not available").

Root cause: the old check used ONE strategy (Get-Command nvidia-smi = PATH
lookup in the START.bat child process). The child environment can differ
from an interactive PowerShell (PATH / PATHEXT / bitness), so a PATH-only
miss is NOT proof of a missing GPU/driver - yet the old error overclaimed.

Contract under test (scripts/gpu_check.ps1, dot-sourced by install_m1.ps1
step 4 and bootstrap.ps1):

* resolution chain: Get-Command (PATH) -> standard absolute locations
  (System32, NVSMI, SysWOW64) -> -SearchDirs (test hook) -> execute ->
  parse name/driver/VRAM -> WMI fallback (Win32_VideoController);
* distinct error taxonomy: OK, GPU_NOT_FOUND, DRIVER_NOT_FOUND,
  NVIDIA_SMI_NOT_FOUND, NVIDIA_SMI_EXEC_FAILED, NVIDIA_SMI_QUERY_FAILED,
  VRAM_QUERY_FAILED, GPU_VALIDATION_FAILED - never an overclaiming
  "no GPU or driver" message;
* VRAM comes from nvidia-smi (memory.total), NEVER from
  Win32_VideoController.AdapterRAM (uint32, wraps above 4 GiB);
* on failure the installer collects a diagnostics block in the SAME
  process environment (PS version, bitness, PATH, Get-Command result,
  where.exe, standard-path probes, smi exec output, WMI) into
  logs/bootstrap.log.

Scenario matrix (user spec section 10), exercised LIVE through a real
PowerShell engine using the exact START.bat child invocation pattern
(powershell -NoProfile -ExecutionPolicy Bypass -File <driver>):

  A: nvidia-smi on PATH                       -> OK (name/VRAM parsed)
  B: not on PATH, found via search dirs       -> OK (fallback origin)
  C: WMI sees NVIDIA GPU, no nvidia-smi       -> NVIDIA_SMI_NOT_FOUND
  D: WMI ran, no NVIDIA adapter               -> GPU_NOT_FOUND
  E: nvidia-smi exists but fails to execute   -> NVIDIA_SMI_EXEC_FAILED
  F: nvidia-smi runs, output unusable         -> NVIDIA_SMI_QUERY_FAILED
  + VRAM column unparseable                   -> VRAM_QUERY_FAILED

Deep tests skip (with reason) when no PowerShell engine is on PATH.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
GPU_CHECK = SCRIPTS_DIR / "gpu_check.ps1"
INSTALL_M1 = SCRIPTS_DIR / "install_m1.ps1"
BOOTSTRAP = SCRIPTS_DIR / "bootstrap.ps1"
START_BAT = REPO_ROOT / "START.bat"

#: The full status taxonomy the library must implement.
TAXONOMY = (
    "OK",
    "GPU_NOT_FOUND",
    "DRIVER_NOT_FOUND",
    "NVIDIA_SMI_NOT_FOUND",
    "NVIDIA_SMI_EXEC_FAILED",
    "NVIDIA_SMI_QUERY_FAILED",
    "VRAM_QUERY_FAILED",
    "GPU_VALIDATION_FAILED",
)

#: The old overclaiming message that must be GONE from the installer.
OLD_OVERCLAIM = "nincs NVIDIA GPU vagy nincs driver telepitve"


def _find_powershell():
    for name in ("powershell", "powershell.exe", "pwsh"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


class GpuCheckLogicTest(FeatureValidationTest):
    FEATURE = "gpu-check"

    def test_logic_library_exists(self):
        self.assertTrue(GPU_CHECK.is_file(), "scripts/gpu_check.ps1 missing")
        self.assertGreater(GPU_CHECK.stat().st_size, 1000)
        data = GPU_CHECK.read_bytes()
        self.assertEqual(
            [b for b in data if b >= 128],
            [],
            "gpu_check.ps1 must be pure ASCII (PS 5.1 no-BOM)",
        )

    def test_logic_resolution_chain_order(self):
        """PATH lookup FIRST, then standard absolute locations."""
        text = _strip_comments(GPU_CHECK.read_text("utf-8"))
        i_getcmd = text.find("Get-Command 'nvidia-smi'")
        i_std = text.find("System32")
        self.assertGreater(
            i_getcmd, -1, "Find-NvidiaSmi must probe PATH via Get-Command"
        )
        self.assertGreater(i_std, i_getcmd, "standard paths must come after PATH")
        # All the standard locations the user requires.
        for needle in (
            "System32",
            "NVIDIA Corporation",
            "NVSMI",
        ):
            self.assertIn(needle, text)

    def test_logic_taxonomy_complete(self):
        """All distinct status codes exist in the library."""
        text = GPU_CHECK.read_text("utf-8")
        for code in TAXONOMY:
            self.assertIn(
                code, text, f"taxonomy code {code} missing from gpu_check.ps1"
            )
        # The installer must map every failure code to a distinct message.
        inst = INSTALL_M1.read_text("utf-8")
        for code in TAXONOMY:
            if code == "OK":
                continue
            self.assertIn(
                code,
                inst,
                f"install_m1.ps1 must handle status code {code}",
            )

    def test_logic_install_m1_uses_library(self):
        """Step 4 delegates to Resolve-NvidiaGpuStatus + diagnostics."""
        inst = _strip_comments(INSTALL_M1.read_text("utf-8"))
        self.assertIn("Resolve-NvidiaGpuStatus", inst)
        self.assertIn("New-GpuDiagnostics", inst)
        self.assertIn(
            ". (Join-Path $ScriptsDir 'gpu_check.ps1')",
            inst,
            "install_m1 must dot-source the GPU library",
        )
        self.assertNotIn(
            OLD_OVERCLAIM,
            inst,
            "the old overclaiming single-probe failure message must be gone",
        )
        # The old single-strategy check must be gone as the gate.
        self.assertNotIn(
            "if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue))",
            inst,
        )

    def test_logic_vram_not_from_wmi(self):
        """VRAM comes from nvidia-smi, never from WMI AdapterRAM."""
        gpu = _strip_comments(GPU_CHECK.read_text("utf-8"))
        self.assertNotIn("AdapterRAM", gpu)
        self.assertIn("memory.total", gpu)
        # The library reads the driver name via WMI only for the fallback.
        self.assertIn("Win32_VideoController", gpu)

    def test_logic_bootstrap_uses_library(self):
        """bootstrap.ps1 probes via the library (no raw PATH-only probe)."""
        boot = _strip_comments(BOOTSTRAP.read_text("utf-8"))
        self.assertIn("Find-NvidiaSmi", boot)
        self.assertIn("Invoke-NvidiaSmiQuery", boot)
        self.assertIn(
            ". (Join-Path $Root 'scripts\\gpu_check.ps1')",
            boot,
            "bootstrap must dot-source the GPU library",
        )
        self.assertNotIn(
            "if (Get-Command nvidia-smi -ErrorAction SilentlyContinue)",
            boot,
        )

    def test_logic_no_path_modification_before_gpu_check(self):
        """The START.bat chain must not touch PATH before the GPU check.

        START.bat must contain no PATH assignment at all; the only PATH
        modification in scripts/ is start_llama_server.ps1 (bin dir prepend,
        runs AFTER the install, long after step 4).
        """
        bat = START_BAT.read_text("utf-8", errors="replace")
        self.assertNotIn("set PATH", bat, "START.bat must not modify PATH")
        for script in SCRIPTS_DIR.glob("*.ps1"):
            text = script.read_text("utf-8")
            if "$env:PATH = " in _strip_comments(text):
                self.assertEqual(
                    script.name,
                    "start_llama_server.ps1",
                    f"unexpected PATH modification in {script.name}",
                )

    def test_logic_ps_version_safety(self):
        """No PowerShell 7-only operators in the new library."""
        text = _strip_comments(GPU_CHECK.read_text("utf-8"))
        for token in ("??", "?.", "&&", "||"):
            self.assertNotIn(token, text, f"PS7-only token {token} in gpu_check.ps1")


def _strip_comments(text: str) -> str:
    import re

    text = re.sub(r"<#.*?#>", "", text, flags=re.DOTALL)
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


class GpuCheckDeepTest(FeatureValidationTest):
    FEATURE = "gpu-check"

    # ------------------------------------------------------------------ utils

    def _ps(self):
        ps = _find_powershell()
        if ps is None:
            self.deep_skip("no powershell.exe / pwsh on PATH")
        return ps

    def _make_fake_smi(self, tmpdir: Path, body: str) -> Path:
        """Create a fake nvidia-smi executable (bash script on POSIX)."""
        smi = tmpdir / "nvidia-smi"
        smi.write_text("#!/bin/bash\n" + body, "utf-8")
        smi.chmod(smi.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return smi

    def _run_driver(self, ps: str, driver_body: str, path_dirs=(), no_std=False):
        """Run a generated driver through the EXACT START.bat child pattern:
        powershell -NoProfile -ExecutionPolicy Bypass -File <driver>.
        Returns (returncode, stdout, stderr)."""
        driver = self._tmp / "driver.ps1"
        sep = ":" if os.name != "nt" and os.environ.get("OS") != "Windows_NT" else ";"
        if path_dirs:
            path_value = sep.join(str(d) for d in path_dirs)
        else:
            # hermetic, GPU-less PATH: an empty dedicated directory
            path_value = str(self._empty)
        hook = ""
        if no_std:
            hook = "$env:VM_GPU_TEST_NO_STD_PATHS = '1'\n"
        driver.write_text(
            textwrap.dedent(
                f"""
                $env:PATH = '{path_value}'
                {hook}. '{GPU_CHECK.as_posix()}'
                {driver_body}
                """
            ).lstrip(),
            "utf-8",
        )
        proc = subprocess.run(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(driver)],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(self._tmp),
        )
        return proc.returncode, proc.stdout, proc.stderr

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="vmgpu_"))
        self._empty = self._tmp / "empty"
        self._empty.mkdir()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _parse(self, out: str):
        """Parse CODE=... NAME=... VRAM=... ORIGIN=... marker lines."""
        vals = {}
        for line in out.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                if key in ("CODE", "NAME", "DRIVER", "VRAM", "ORIGIN"):
                    vals[key] = value.strip()
        return vals

    # --------------------------------------------------------- scenarios A-F

    def _scenario_body(self, wmi_expr="$null", search_expr="@()"):
        return textwrap.dedent(
            f"""
            $S = Resolve-NvidiaGpuStatus -WmiGpus {wmi_expr} -SearchDirs {search_expr}
            Write-Output ('CODE=' + $S.Code)
            Write-Output ('NAME=' + $S.GpuName)
            Write-Output ('DRIVER=' + $S.Driver)
            Write-Output ('VRAM=' + $S.VramMiB)
            Write-Output ('ORIGIN=' + $S.SmiOrigin)
            exit 0
            """
        ).strip()

    def test_deep_scenario_a_smi_on_path(self):
        """A: nvidia-smi on PATH -> OK with parsed name/driver/VRAM."""
        ps = self._ps()
        fake = self._make_fake_smi(
            self._tmp,
            'echo "NVIDIA GeForce RTX 5070, 580.16, 12282 MiB"\n',
        )
        rc, out, err = self._run_driver(
            ps, self._scenario_body(), path_dirs=[self._tmp]
        )
        vals = self._parse(out)
        self.assertEqual(
            vals.get("CODE"), "OK", f"rc={rc} out={out!r} err={err!r}"
        )
        self.assertEqual(vals.get("NAME"), "NVIDIA GeForce RTX 5070")
        self.assertEqual(vals.get("DRIVER"), "580.16")
        self.assertEqual(vals.get("VRAM"), "12282")
        self.assertEqual(vals.get("ORIGIN"), "PATH")
        self.deep_pass("scenario A: PATH-found nvidia-smi -> OK (RTX 5070 parsed)")

    def test_deep_scenario_b_smi_via_search_dirs(self):
        """B: not on PATH, found in extra search dir -> OK via fallback."""
        ps = self._ps()
        self._make_fake_smi(
            self._tmp, 'echo "NVIDIA GeForce RTX 5070, 32.0.15.9186, 12282 MiB"\n'
        )
        body = self._scenario_body(search_expr=f"@('{self._tmp.as_posix()}')")
        rc, out, err = self._run_driver(
            ps, body, path_dirs=[], no_std=True
        )
        vals = self._parse(out)
        self.assertEqual(
            vals.get("CODE"), "OK", f"rc={rc} out={out!r} err={err!r}"
        )
        self.assertEqual(vals.get("ORIGIN"), "SEARCH_DIR")
        self.deep_pass("scenario B: nvidia-smi found via search-dir fallback -> OK")

    def test_deep_scenario_c_wmi_sees_gpu_no_smi(self):
        """C: WMI sees NVIDIA GPU, nvidia-smi missing -> NVIDIA_SMI_NOT_FOUND."""
        ps = self._ps()
        body = self._scenario_body(
            wmi_expr="@( @{ Name = 'NVIDIA GeForce RTX 5070'; DriverVersion = '32.0.15.9186'; ConfigManagerErrorCode = 0 } )"
        )
        rc, out, err = self._run_driver(
            ps, body, path_dirs=[], no_std=True
        )
        vals = self._parse(out)
        self.assertEqual(
            vals.get("CODE"), "NVIDIA_SMI_NOT_FOUND", f"out={out!r} err={err!r}"
        )
        self.deep_pass("scenario C: WMI GPU + missing smi -> NVIDIA_SMI_NOT_FOUND")

    def test_deep_scenario_d_no_nvidia_gpu(self):
        """D: WMI ran, no NVIDIA adapter -> GPU_NOT_FOUND."""
        ps = self._ps()
        body = self._scenario_body(wmi_expr="@()")
        rc, out, err = self._run_driver(
            ps, body, path_dirs=[], no_std=True
        )
        vals = self._parse(out)
        self.assertEqual(vals.get("CODE"), "GPU_NOT_FOUND", f"out={out!r}")
        self.deep_pass("scenario D: no NVIDIA adapter anywhere -> GPU_NOT_FOUND")

    def test_deep_scenario_e_smi_not_executable(self):
        """E: nvidia-smi exists but fails (exit 3) -> NVIDIA_SMI_EXEC_FAILED."""
        ps = self._ps()
        self._make_fake_smi(self._tmp, 'echo "driver crashed" >&2\nexit 3\n')
        rc, out, err = self._run_driver(
            ps, self._scenario_body(), path_dirs=[self._tmp]
        )
        vals = self._parse(out)
        self.assertEqual(vals.get("CODE"), "NVIDIA_SMI_EXEC_FAILED", f"out={out!r}")
        self.deep_pass("scenario E: failing nvidia-smi -> NVIDIA_SMI_EXEC_FAILED")

    def test_deep_scenario_f_smi_bad_output(self):
        """F: nvidia-smi runs but output unusable -> NVIDIA_SMI_QUERY_FAILED."""
        ps = self._ps()
        self._make_fake_smi(self._tmp, 'echo "banana"\n')
        rc, out, err = self._run_driver(
            ps, self._scenario_body(), path_dirs=[self._tmp]
        )
        vals = self._parse(out)
        self.assertEqual(
            vals.get("CODE"), "NVIDIA_SMI_QUERY_FAILED", f"out={out!r}"
        )
        self.deep_pass("scenario F: unparseable smi output -> QUERY_FAILED")

    def test_deep_scenario_vram_unparseable(self):
        """VRAM column unparseable (name/driver fine) -> VRAM_QUERY_FAILED."""
        ps = self._ps()
        self._make_fake_smi(self._tmp, 'echo "NVIDIA GeForce RTX 5070, 580.16, [N/A]"\n')
        rc, out, err = self._run_driver(
            ps, self._scenario_body(), path_dirs=[self._tmp]
        )
        vals = self._parse(out)
        self.assertEqual(vals.get("CODE"), "VRAM_QUERY_FAILED", f"out={out!r}")
        self.assertEqual(vals.get("NAME"), "NVIDIA GeForce RTX 5070")
        self.deep_pass("VRAM unparseable -> VRAM_QUERY_FAILED (name kept)")

    def test_deep_real_machine_graceful(self):
        """No fakes: the REAL machine status must be a valid taxonomy code
        (OK on the RTX 5070 target; a graceful coded failure elsewhere)."""
        ps = self._ps()
        proc = subprocess.run(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
             f". '{GPU_CHECK.as_posix()}'; $S = Resolve-NvidiaGpuStatus; "
             "Write-Output ('CODE=' + $S.Code)"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        vals = self._parse(proc.stdout)
        self.assertIn(vals.get("CODE"), TAXONOMY, f"out={proc.stdout!r}")
        self.deep_pass(f"real-machine status resolves gracefully: {vals.get('CODE')}")


if __name__ == "__main__":
    unittest.main()
