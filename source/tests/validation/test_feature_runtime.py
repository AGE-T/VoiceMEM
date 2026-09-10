"""Runtime environment feature validation (Task 12 - M0.1).

Validates the runtime layer of the M0 bootstrap: the declared Python version
contract (3.11+), the core dependency set of ``pyproject.toml`` (numpy, PyYAML,
httpx - installable both in the sandbox and on the target machine) and the
target-only dependency documentation in ``requirements.txt``.

Deep checks (run for real on the target machine, skipped with a recorded
reason in the sandbox):
* ``runtime-venv``  - the repo-local ``.venv`` created by ``install_m1.ps1``;
* ``runtime-deps``  - the core deps are actually importable;
* ``runtime-gpu``   - RTX 5070 / CUDA 12.8 / sm_120 smoke test (M0 Section 6).
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest, has_module

REPO_ROOT = Path(__file__).resolve().parents[2]


class RuntimeDepsFeatureTest(FeatureValidationTest):
    """Dependency contract: pyproject core deps + requirements.txt docs."""

    FEATURE = "runtime-deps"

    def test_logic_python_version_contract(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r"(?m)^requires-python\s*=\s*[\"']([^\"']+)[\"']", pyproject)
        self.assertIsNotNone(match, "pyproject.toml requires-python missing")
        self.assertIn("3.11", match.group(1), "requires-python must mention 3.11")

    def test_logic_core_deps_declared(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        block = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)\]", pyproject)
        self.assertIsNotNone(block, "pyproject.toml [project] dependencies missing")
        deps_text = block.group(1)
        for dep in ("numpy", "PyYAML", "httpx"):
            self.assertIn(dep, deps_text, f"core dependency '{dep}' not declared")

    def test_logic_target_only_deps_documented(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        for dep in ("transformers", "onnxruntime", "sounddevice", "soundfile",
                    "sentence-transformers"):
            pattern = re.compile(rf"(?m)^\s*{re.escape(dep)}\b")
            self.assertRegex(
                requirements, pattern,
                f"requirements.txt must document the target-only dep '{dep}'",
            )

    def test_deep_core_deps_importable(self):
        """numpy, PyYAML and httpx must import (an ImportError is a real FAIL)."""
        import httpx  # noqa: F401 - plain import is the check
        import numpy  # noqa: F401
        import yaml  # noqa: F401

        self.deep_pass("core deps importable (numpy, PyYAML, httpx)")


class RuntimeTorchPinFeatureTest(FeatureValidationTest):
    """PyTorch trio pinning: torch 2.7.0 + torchvision 0.22.0 + torchaudio 2.7.0.

    v0.1.6 release blocker regression guard: the installer pinned only
    torch==2.7.0 and left torchvision/torchaudio unpinned, so pip resolved
    torchaudio 2.11.0+cu128 from the cu128 index next to torch 2.7.0+cu128 -
    an incompatible pairing. The installer must pin all three officially
    paired versions from the same CUDA 12.8 index, verify them by import
    AFTER installation, and re-verify (with automatic re-pin) after the
    later VoiceMem dependency install so pip can never leave a broken trio
    behind.
    """

    FEATURE = "runtime-torch-pin"

    #: The officially paired trio (must mirror install_m1.ps1 variables).
    PINNED_TRIO = {
        "torch": "2.7.0",
        "torchvision": "0.22.0",
        "torchaudio": "2.7.0",
    }

    @classmethod
    def _installer_code(cls) -> str:
        raw = (REPO_ROOT / "scripts" / "install_m1.ps1").read_bytes()
        assert max(raw) <= 0x7F, "install_m1.ps1 must be pure ASCII"
        return raw.decode("ascii")

    def test_logic_installer_pins_paired_trio(self):
        code = self._installer_code()
        # The pins live in the script's version variables (single source).
        for var, version in (
            ("$TorchVersion = \"2.7.0\"", "torch"),
            ("$TorchvisionVersion = \"0.22.0\"", "torchvision"),
            ("$TorchaudioVersion = \"2.7.0\"", "torchaudio"),
        ):
            self.assertIn(
                var, code,
                f"install_m1.ps1 must define the paired pin {var}",
            )
        # ... and every install command must pass all three pins through
        # the variables (both the fresh install and the re-pin guard).
        self.assertIn(
            '"torch==$TorchVersion" "torchvision==$TorchvisionVersion" '
            '"torchaudio==$TorchaudioVersion"',
            code,
            "the pip install must pin all three packages from the variables",
        )
        self.assertEqual(
            code.count("--index-url $TorchCudaIndex"),
            code.count('"torch==$TorchVersion" "torchvision==$TorchvisionVersion" '
                       '"torchaudio==$TorchaudioVersion"'),
            "every pinned trio install must use the cu128 index",
        )
        self.assertIn(
            "https://download.pytorch.org/whl/cu128", code,
            "the trio must come from the CUDA 12.8 (cu128) index",
        )
        # The OLD broken command (unpinned torchvision/torchaudio) must be
        # gone: the v0.1.6 bug pip-resolved torchaudio 2.11.0+cu128 from it.
        self.assertNotIn(
            '"torch==$TorchVersion" torchvision torchaudio', code,
            "the v0.1.6 unpinned torchvision/torchaudio install is back",
        )

    def test_logic_installer_verifies_trio_by_import(self):
        code = self._installer_code()
        # The probe imports all three and compares versions after install.
        for marker in (
            "import torch", "import torchvision", "import torchaudio",
            "TRIO_MISMATCH", "TORCHVISION_VERSION=", "TORCHAUDIO_VERSION=",
            "TORCH_CUDA=",
        ):
            self.assertIn(
                marker, code,
                f"install_m1.ps1 trio probe missing '{marker}'",
            )
        # The GPU smoke test must import and print all three versions too.
        smoke = re.search(
            r"\$GpuSmokeCode = @'\r?\n(.*?)\r?\n'@", code, re.DOTALL,
        )
        self.assertIsNotNone(smoke, "the $GpuSmokeCode here-string was not found")
        for pkg in ("torch", "torchvision", "torchaudio"):
            self.assertIn(f"import {pkg}", smoke.group(1))

    def test_logic_installer_guards_post_voicemem_overwrite(self):
        code = self._installer_code()
        self.assertIn(
            "--force-reinstall --no-deps", code,
            "the installer must re-pin the trio after later dependency "
            "installs could overwrite it (torch/vision/audio only)",
        )
        self.assertIn(
            "feluliras-ellenorzes", code,
            "the post-VoiceMem re-verification must be present",
        )
        self.assertIn(
            "torchaudio 2.11.0", code,
            "the comment must document the v0.1.6 torchaudio 2.11.0+cu128 "
            "root cause",
        )

    def test_logic_requirements_document_paired_trio(self):
        for path in ("requirements.txt", "requirements.lock"):
            text = (REPO_ROOT / path).read_text(encoding="utf-8")
            self.assertIn(
                "torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0", text,
                f"{path} must document the pinned paired trio",
            )

    def test_deep_torch_trio_versions(self):
        """On the target machine all three versions must be the pinned pair."""
        if not has_module("torch"):
            self.deep_skip("torch not installed")
        if not has_module("torchvision"):
            self.deep_skip("torchvision not installed")
        if not has_module("torchaudio"):
            self.deep_skip("torchaudio not installed")
        import torch
        import torchaudio
        import torchvision

        observed = {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torchaudio": torchaudio.__version__,
        }
        for pkg, version in observed.items():
            base = version.split("+")[0]
            local = version.split("+", 1)[1] if "+" in version else ""
            self.assertEqual(
                base, self.PINNED_TRIO[pkg],
                f"{pkg} is {version!r}, want {self.PINNED_TRIO[pkg]}+cu128 "
                "(officially paired trio)",
            )
            self.assertFalse(
                local and "cu128" not in local,
                f"{pkg} local tag {local!r} is not the cu128 build",
            )
        self.deep_pass(
            "paired trio: torch {} + torchvision {} + torchaudio {}".format(
                observed["torch"], observed["torchvision"],
                observed["torchaudio"],
            )
        )


class RuntimeVenvFeatureTest(FeatureValidationTest):
    """The repo-local virtual environment created by install_m1.ps1."""

    FEATURE = "runtime-venv"

    def test_deep_repo_venv_exists(self):
        venv_win = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
        venv_posix = REPO_ROOT / ".venv" / "bin" / "python"
        if venv_win.is_file():
            self.deep_pass(f".venv python found: {venv_win}")
        elif venv_posix.is_file():
            self.deep_pass(f".venv python found: {venv_posix}")
        else:
            self.deep_skip("no .venv on this machine (installer not run)")


class RuntimeGpuFeatureTest(FeatureValidationTest):
    """GPU smoke test: sm_120 (Blackwell), CUDA 12.8, >= 10 GB VRAM, math."""

    FEATURE = "runtime-gpu"

    def test_deep_gpu_smoke(self):
        if not has_module("torch"):
            self.deep_skip("torch not installed")
        import torch

        if not torch.cuda.is_available():
            self.deep_skip("torch without CUDA device")
        capability = torch.cuda.get_device_capability(0)
        self.assertGreaterEqual(
            capability, (12, 0),
            f"compute capability {capability} < (12, 0): sm_120/Blackwell required",
        )
        self.assertTrue(
            str(torch.version.cuda).startswith("12.8"),
            f"torch.version.cuda is {torch.version.cuda!r}, expected 12.8.x",
        )
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        self.assertGreaterEqual(
            vram_gb, 10.0, f"only {vram_gb:.1f} GB VRAM (need >= 10 GB)"
        )
        # Matmul correctness: GPU result must match the CPU reference.
        torch.manual_seed(42)
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        gpu_result = (a @ b).to("cpu")
        cpu_result = a.to("cpu") @ b.to("cpu")
        self.assertTrue(
            torch.allclose(gpu_result, cpu_result, atol=1e-3),
            "GPU matmul differs from the CPU reference",
        )
        self.deep_pass(
            f"GPU smoke OK ({props.name}, sm_{capability[0]}{capability[1]}, "
            f"CUDA {torch.version.cuda}, {vram_gb:.1f} GB VRAM)"
        )


if __name__ == "__main__":
    unittest.main()
