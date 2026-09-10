"""VRAM measurement feature validation (Task 2-a - llama-server VRAM audit).

Static validation of ``scripts/measure_vram.ps1``: the sandbox has no
PowerShell engine and no NVIDIA GPU, so the actual six-phase measurement
(nvidia-smi sampling around the llama-server lifecycle) runs on the target
Windows machine; everything statically decidable is asserted here. No
network, no PowerShell execution - deterministic by construction.

Contract under test (user-facing VRAM measurement script):

* PS 5.1-safe deliverable: pure ASCII (PS 5.1 reads a BOM-less file as
  Windows-1252), CRLF line endings with a trailing newline, ``param()``
  as the first executable statement (``$Root`` default = parent of
  ``$PSScriptRoot``, ``$Port`` = 8080, ``$Samples`` = 5), and NO
  PowerShell 7-only operators (``??``, ``?.``, ``&&``, ``||``);
* six phases: idle / model_loaded / first_request / normal_generation /
  json_generation / peak;
* BOTH memory views per phase: total GPU memory via
  ``nvidia-smi --query-gpu=memory.used,memory.total`` AND per-process
  usage via ``nvidia-smi --query-compute-apps=pid,process_name,used_memory``
  (CSV, noheader, nounits) with llama-server.exe grouped SEPARATELY from
  every other GPU process (python/PyTorch ASR etc.);
* ``/health`` gate on ``http://127.0.0.1:<port>`` BEFORE starting a server
  (an already-running server is noted, never doubled), the documented
  child invocation pattern (powershell -ExecutionPolicy Bypass -File
  scripts\\start_llama_server.ps1, hidden window, -PassThru, 90 s health
  poll, fast-fail dumping the logs\\llama-server.err.log / .out.log
  tails), and POST /v1/chat/completions requests: non-streaming "Hello"
  (max_tokens 16, temperature 0.0) plus two streaming generation phases
  (max_tokens 200), the JSON one with
  ``response_format={"type":"json_object"}``;
* sampling DURING generation: the streaming request runs as a background
  job while the foreground loops nvidia-smi with 0.5 s sleeps;
* JSON report written to ``logs\\vram_report.json`` via ConvertTo-Json
  with per-sample phase / timestamp / total_used_mb / total_mb /
  per_process records;
* cleanup mirrors verify_m1.ps1: llama-server processes are stopped ONLY
  when this script started them (an already-running server is left
  running);
* exit codes 0 (PASS) / 1 (FAIL) and a final ``VRAM REPORT: ...`` line
  carrying the JSON report path; robustness flags (-UseBasicParsing,
  -TimeoutSec, $ProgressPreference) and try/catch around nvidia-smi.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "measure_vram.ps1"

#: PS 5.1 does not support these operators (PowerShell 7+ syntax).
PS7_ONLY_TOKENS = ("??", "?.", "&&", "||")

#: The six measurement phases the script must implement.
PHASES = (
    "idle",
    "model_loaded",
    "first_request",
    "normal_generation",
    "json_generation",
    "peak",
)


def _strip_ps_comments(text: str) -> str:
    """Remove <# ... #> block comments and full-line # comments."""
    without_blocks = re.sub(r"<#.*?#>", "", text, flags=re.DOTALL)
    lines = [ln for ln in without_blocks.splitlines() if not ln.strip().startswith("#")]
    return "\n".join(lines)


class VramMeasureFeatureTest(FeatureValidationTest):
    """Static contract of scripts/measure_vram.ps1 (target-machine tool)."""

    FEATURE = "vram"

    # ------------------------------------------------------------- helpers

    def _text(self) -> str:
        self.assertTrue(SCRIPT.is_file(), "scripts/measure_vram.ps1 missing")
        return SCRIPT.read_text(encoding="ascii")

    def _code(self) -> str:
        return _strip_ps_comments(self._text())

    # ------------------------------------------------------------ encoding

    def test_logic_script_exists(self):
        self.assertTrue(SCRIPT.is_file(), "scripts/measure_vram.ps1 missing")
        self.assertGreater(
            SCRIPT.stat().st_size, 1000, "measure_vram.ps1 suspiciously small"
        )

    def test_logic_pure_ascii(self):
        raw = SCRIPT.read_bytes()
        self.assertEqual(
            [b for b in raw if b >= 0x80],
            [],
            "measure_vram.ps1 must be pure ASCII (PS 5.1 no-BOM)",
        )

    def test_logic_crlf_line_endings(self):
        raw = SCRIPT.read_bytes()
        lone = raw.replace(b"\r\n", b"")
        self.assertNotIn(
            b"\n", lone, "measure_vram.ps1 must use CRLF line endings"
        )
        self.assertTrue(
            raw.endswith(b"\r\n"), "measure_vram.ps1 must end with a newline"
        )

    def test_logic_param_block_first_with_documented_defaults(self):
        code = self._code()
        nonempty = [ln.strip() for ln in code.splitlines() if ln.strip()]
        self.assertTrue(nonempty, "no code after comment strip")
        self.assertTrue(
            nonempty[0].startswith("param("),
            "param( must be the first executable statement",
        )
        window = "\n".join(nonempty[:10])
        for needle in ("[string]$Root", "[int]$Port = 8080", "[int]$Samples = 5"):
            self.assertIn(needle, window, f"param default {needle} missing")
        # $Root falls back to the parent of $PSScriptRoot (relocatable repo).
        self.assertIn("$PSScriptRoot", self._text())

    def test_logic_no_ps7_only_operators(self):
        text = self._text()
        for token in PS7_ONLY_TOKENS:
            self.assertNotIn(
                token, text, f"PS7-only operator {token!r} in measure_vram.ps1"
            )

    # -------------------------------------------------------------- phases

    def test_logic_six_phases_present(self):
        code = self._code()
        for phase in PHASES:
            self.assertIn(
                '"{0}"'.format(phase),
                code,
                f"phase literal {phase!r} missing from measure_vram.ps1",
            )

    # --------------------------------------------------------- nvidia-smi

    def test_logic_nvidia_smi_queries(self):
        code = self._code()
        self.assertIn(
            "--query-gpu=memory.used,memory.total", code,
            "total/used must come from nvidia-smi --query-gpu",
        )
        self.assertIn(
            "--query-compute-apps=pid,process_name,used_memory", code,
            "per-process breakdown via --query-compute-apps missing",
        )
        self.assertIn("--format=csv,noheader,nounits", code)

    def test_logic_llama_server_separated_from_other_processes(self):
        code = self._code()
        self.assertIn('"llama-server"', code, "llama-server group missing")
        self.assertIn('"other"', code, "other-process group missing")
        self.assertIn("*llama-server*", code, "process-name matching missing")
        # Distinct reporting lines for the two groups.
        self.assertIn("llama-server (pid", code)
        self.assertIn("egyeb:", code)

    # ------------------------------------------------------- server phases

    def test_logic_health_gate_before_starting_server(self):
        code = self._code()
        self.assertIn("/health", code, "http health check missing")
        self.assertIn("127.0.0.1", code, "loopback health endpoint missing")
        i_check = code.find("if (Test-LlamaHealth")
        i_start = code.find("Start-Process -FilePath")
        self.assertGreater(i_check, -1, "no health-first branch found")
        self.assertGreater(i_start, -1, "no Start-Process invocation found")
        self.assertLess(
            i_check, i_start,
            "/health must be checked BEFORE starting a second server",
        )
        # The documented child invocation pattern.
        self.assertIn('"-ExecutionPolicy", "Bypass", "-File"', code)
        self.assertIn("start_llama_server.ps1", code)
        self.assertIn("-WindowStyle Hidden", code)
        self.assertIn("-PassThru", code)

    def test_logic_fast_fail_dumps_server_log_tails(self):
        text = self._text()
        for needle in (
            "llama-server.err.log",
            "llama-server.out.log",
            "Get-Content",
            "-Tail",
        ):
            self.assertIn(
                needle, text, f"fast-fail log-tail element {needle!r} missing"
            )

    def test_logic_chat_completions_request_bodies(self):
        code = self._code()
        self.assertIn("/v1/chat/completions", code)
        self.assertIn('"model":"qwen3.6-35b-a3b"', code)
        self.assertIn("Hello", code)
        self.assertIn('"max_tokens":16', code)
        self.assertIn('"temperature":0.0', code)
        self.assertIn('"stream":false', code)
        self.assertIn('"stream":true', code)
        self.assertIn('"max_tokens":200', code)
        self.assertIn('"response_format":{"type":"json_object"}', code)
        self.assertIn("coffee cup", code, "longer generation prompt missing")
        self.assertIn("Thomas", code, "JSON generation prompt missing")

    def test_logic_sampling_during_generation(self):
        code = self._code()
        self.assertIn("Start-Job", code, "generation must run as a background job")
        self.assertIn(
            "Start-Sleep -Milliseconds 500", code, "0.5 s sampling cadence missing"
        )
        self.assertIn("Receive-Job", code)
        self.assertIn("Wait-Job", code)

    # -------------------------------------------------------------- report

    def test_logic_json_report_written_to_logs(self):
        code = self._code()
        self.assertIn("vram_report.json", code, "JSON report file missing")
        self.assertIn('"logs"', code, "logs directory join missing")
        self.assertIn("ConvertTo-Json", code)
        for field in ("total_used_mb", "total_mb", "per_process", "timestamp", "phase"):
            self.assertIn(field, code, f"report field {field!r} missing")

    # ------------------------------------------------------------- cleanup

    def test_logic_cleanup_only_own_server(self):
        code = self._code()
        self.assertIn("Get-Process -Name", code)
        self.assertIn('"llama-server"', code)
        self.assertIn("Stop-Process", code)
        self.assertIn("StartedServer", code, "started-server tracking missing")
        gated = re.search(r"if \(-not \$Script:StartedServer\) \{ return \}", code)
        self.assertIsNotNone(
            gated,
            "the llama-server stop must be gated on $Script:StartedServer "
            "(mirror of the verify_m1.ps1 cleanup pattern)",
        )
        # The starter powershell pid is also tracked and stopped.
        self.assertIn("$Script:StarterProc", code)

    # ------------------------------------------------------- exit contract

    def test_logic_exit_codes_and_final_report_line(self):
        code = self._code()
        self.assertIn("exit 0", code, "PASS exit code 0 missing")
        self.assertIn("exit 1", code, "FAIL exit code 1 missing")
        self.assertIn("VRAM REPORT:", code, "final VRAM REPORT line missing")

    def test_logic_robustness_flags(self):
        code = self._code()
        self.assertIn("-UseBasicParsing", code)
        self.assertIn("-TimeoutSec", code)
        self.assertIn('$ProgressPreference = "SilentlyContinue"', code)
        self.assertIn("Resolve-NvidiaSmi", code, "nvidia-smi resolution missing")
        self.assertIn(
            "System32", code, "standard-path fallback for nvidia-smi missing"
        )


if __name__ == "__main__":
    unittest.main()
