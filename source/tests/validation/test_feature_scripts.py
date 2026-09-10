"""Scripts feature validation (Task 12 - M0.1).

Validates the ``scripts`` layer of the repo:

* every M0 script (installer, model downloader, server/agent starters,
  verifier, offline audit, test gate, release builder, setup checker,
  manifest writer) exists and is non-empty;
* every ``*.ps1`` under ``scripts/`` is pure ASCII (PowerShell 5.1 without a
  BOM reads non-ASCII bytes as Windows-1252);
* the Task-12-owned gate scripts (``run_tests.ps1``, ``build_release.ps1``)
  avoid PowerShell 7-only operators (``??``, ``?.``, ``&&``, ``||``) in code
  text (comments stripped);
* the parameterized scripts declare their ``param(`` block at the top, before
  any executable statement.

NOTE on the param check: the raw "first 60 non-empty lines" count of
``download_models.ps1`` lands at exactly 61 because its ASCII header comment
block alone spans 60 non-empty lines; the check therefore operates on the
COMMENT-STRIPPED text (block comments and full-line comments removed), which
is a strictly stronger position assertion: ``param(`` must be the FIRST
executable statement of the script.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Every M0 script that must exist and be non-empty.
REQUIRED_SCRIPTS = (
    "install_m1.ps1", "download_models.ps1", "start_llama_server.ps1",
    "start_agent.ps1", "verify_m1.ps1", "offline_check.ps1",
    "run_tests.ps1", "build_release.ps1", "verify_setup.py",
    "write_install_manifest.py", "bootstrap.ps1", "gpu_check.ps1",
    "download_models_hf.py",
)

#: Task-12-owned gate scripts scanned for PowerShell 7-only operators.
GATE_SCRIPTS = ("build_release.ps1", "run_tests.ps1")

#: PS 5.1 does not support these operators (PowerShell 7+ syntax).
PS7_ONLY_TOKENS = ("??", "?.", "&&", "||")

#: A real PowerShell ``-File`` argument (word boundary). ``-FilePath`` of a
#: NATIVE exe launch (e.g. Start-Process -FilePath $LlamaExe) must NOT count
#: as a PowerShell engine launch.
PS_FILE_LAUNCH_RE = re.compile(r'(^|[\s"\'()])-File([\s,"\']|$)')


def _is_ps_file_launch(line: str) -> bool:
    return bool(PS_FILE_LAUNCH_RE.search(line))

#: Parameterized scripts: param( must be the first executable statement.
#: (start_agent.ps1 and run_tests.ps1 are intentionally param-less.)
PARAM_SCRIPTS = (
    "install_m1.ps1", "download_models.ps1", "start_llama_server.ps1",
    "verify_m1.ps1", "offline_check.ps1", "build_release.ps1",
    "bootstrap.ps1",
)


def _strip_ps_comments(text: str) -> str:
    """Remove <# ... #> block comments and full-line # comments."""
    without_blocks = re.sub(r"<#.*?#>", "", text, flags=re.DOTALL)
    lines = [ln for ln in without_blocks.splitlines() if not ln.strip().startswith("#")]
    return "\n".join(lines)


class ScriptsFeatureTest(FeatureValidationTest):
    """Repo script inventory, encoding and PowerShell 5.1 safety."""

    FEATURE = "scripts"

    def test_logic_required_scripts_exist(self):
        for name in REQUIRED_SCRIPTS:
            path = SCRIPTS_DIR / name
            self.assertTrue(path.is_file(), f"scripts/{name} missing")
            self.assertGreater(
                path.stat().st_size, 0, f"scripts/{name} is empty"
            )

    def test_logic_all_ps1_are_pure_ascii(self):
        ps1_files = sorted(SCRIPTS_DIR.glob("*.ps1"))
        self.assertGreater(len(ps1_files), 0, "no .ps1 files found under scripts/")
        for path in ps1_files:
            raw = path.read_bytes()
            self.assertLessEqual(
                max(raw), 0x7F,
                f"{path.name} must be pure ASCII for PowerShell 5.1",
            )

    def test_logic_gate_scripts_avoid_ps7_syntax(self):
        """run_tests.ps1 / build_release.ps1 must run on PowerShell 5.1.

        Comment lines and block comments are stripped first. The current
        scripts contain none of these tokens even inside string literals, so
        the whole-code scan is exact (no string-context false positives).
        """
        for name in GATE_SCRIPTS:
            text = (SCRIPTS_DIR / name).read_text(encoding="ascii")
            code = _strip_ps_comments(text)
            self.assertTrue(code.strip(), f"{name} has no code after comment strip")
            for token in PS7_ONLY_TOKENS:
                self.assertNotIn(
                    token, code,
                    f"{name} uses the PowerShell 7-only operator {token!r}",
                )

    def test_logic_param_block_is_first_statement(self):
        for name in PARAM_SCRIPTS:
            text = (SCRIPTS_DIR / name).read_text(encoding="ascii")
            code = _strip_ps_comments(text)
            nonempty = [ln.strip() for ln in code.splitlines() if ln.strip()]
            self.assertTrue(nonempty, f"{name} has no code after comment strip")
            window = "\n".join(nonempty[:60])
            self.assertIn(
                "param(", window,
                f"{name}: param( not within the first 60 code lines",
            )
            self.assertTrue(
                nonempty[0].startswith("param("),
                f"{name}: param( must be the first executable statement",
            )


class LlamaServerCliContractTests(FeatureValidationTest):
    """v0.3.1 regression: the llama.cpp b10717 field report (--grammar-json).

    The v0.2.0 start_llama_server.ps1 passed ``--grammar-json`` - a CLI
    argument that does NOT exist in the pinned llama.cpp b10717 build
    (llama-server exits instantly with ``error: invalid argument:
    --grammar-json``). JSON constrained generation is REQUEST-LEVEL
    (``response_format`` on POST /v1/chat/completions - verified live
    against the b10717 binary with Gemma 4 12B QAT Q4_0: json_object and
    json_schema both produce valid, fence-free, schema-compliant JSON).
    """

    FEATURE = "scripts"

    def _code(self, name: str) -> str:
        text = (SCRIPTS_DIR / name).read_text(encoding="ascii")
        return _strip_ps_comments(text)

    def test_logic_start_script_never_passes_grammar_json(self):
        """The exact v0.2.0 field-report bug must never come back (code)."""
        code = self._code("start_llama_server.ps1")
        self.assertNotIn(
            "--grammar-json", code,
            "--grammar-json is not a llama.cpp b10717 CLI argument "
            "(the server exits instantly) - JSON mode is request-level",
        )

    def test_logic_verify_script_never_passes_grammar_json(self):
        code = self._code("verify_m1.ps1")
        self.assertNotIn("--grammar-json", code)

    def test_logic_start_script_passes_supported_flags(self):
        """Every server flag must exist in llama-server --help (b10717)."""
        code = self._code("start_llama_server.ps1")
        for flag in (
            '"--model"', '"--host"', '"--port"', '"-ngl"', '"-c"',
            '"--parallel"', '"--cache-type-k"', '"--cache-type-v"',
            '"--temp"', '"--metrics"', '"--no-webui"',
        ):
            self.assertIn(flag, code, f"start_llama_server.ps1 must pass {flag}")

    def test_logic_start_script_redirects_logs_and_checks_health(self):
        """Robust start: log redirect, health poll, fast fail, PASS/FAIL."""
        code = self._code("start_llama_server.ps1")
        self.assertIn("RedirectStandardOutput", code)
        self.assertIn("RedirectStandardError", code)
        self.assertIn("llama-server.out.log", code)
        self.assertIn("llama-server.err.log", code)
        self.assertIn("/health", code)
        self.assertIn("HasExited", code)      # fast fail when the child dies
        self.assertIn("PASS:", code)
        self.assertIn("FAIL:", code)

    def test_logic_verify_script_tests_json_completions(self):
        """verify_m1 must exercise BOTH request-level JSON modes + logs."""
        code = self._code("verify_m1.ps1")
        self.assertIn('{"type":"json_object"}', code)
        self.assertIn('"json_schema"', code)
        # v0.4.3: the plain-text completion uses a REAL generation request
        # (max_tokens 48) - the old max_tokens=8 request could return EMPTY
        # content (token budget eaten before any visible token), which is
        # exactly the v0.4.2 field failure this check must catch.
        self.assertIn('"Hello! Reply with one short greeting sentence."', code)
        self.assertIn("Show-LlamaLogTail", code)   # fast-fail evidence dump
        self.assertIn("llama-server.err.log", code)

    def test_logic_start_agent_surfaces_server_error_log(self):
        """start_agent failure path shows the real llama-server error."""
        code = self._code("start_agent.ps1")
        self.assertIn("llama-server.err.log", code)


class StartupRobustnessV032Tests(FeatureValidationTest):
    """v0.3.2 regression: target-machine field report #2.

    Two distinct failure pictures after the v0.3.1 --grammar-json fix:
    (a) manual script runs blocked with "not digitally signed" - the
    browser-downloaded release ZIP propagates the Mark-of-the-Web
    (Zone.Identifier) to the extracted .ps1 files and the default
    RemoteSigned policy blocks them in a plain session;
    (b) verify_m1 -WithServer's nested starter launch exited with code 1
    in ~2 s with EMPTY llama-server logs: a partial bin\\ (exe present,
    ggml-cuda/cudart/cublas DLLs missing) makes llama-server.exe die
    instantly and SILENTLY (STATUS_DLL_NOT_FOUND 0xC0000135, no stderr),
    while the v0.3.1 installer skipped re-download (it only checked the
    exe) and the starter's own diagnosis was lost in the minimized child
    console.
    """

    FEATURE = "scripts"

    def _code(self, name: str) -> str:
        text = (SCRIPTS_DIR / name).read_text(encoding="ascii")
        return _strip_ps_comments(text)

    def _raw(self, name: str) -> str:
        return (SCRIPTS_DIR / name).read_text(encoding="ascii")

    # --- (a) execution policy / MOTW -------------------------------------

    def test_logic_every_powershell_file_invocation_uses_bypass(self) -> None:
        """No child powershell -File launch may rely on the session policy.

        Scans every script + START.bat: any line that launches a
        PowerShell engine with -File must carry -ExecutionPolicy Bypass
        on the same line (all call sites are single-line argument
        arrays). This keeps the whole chain independent of execution
        policy AND of the Mark-of-the-Web on the .ps1 files.
        """
        engine_markers = ("powershell", "pwsh", "PS_EXE", "ChildPs",
                          "-ArgumentList")
        checked = 0
        for path in sorted(SCRIPTS_DIR.glob("*.ps1")):
            for lineno, line in enumerate(
                    self._raw(path.name).splitlines(), start=1):
                if not _is_ps_file_launch(line):
                    continue
                if not any(m in line for m in engine_markers):
                    continue  # Get-ChildItem -File / Get-FileHash etc.
                checked += 1
                self.assertIn(
                    "Bypass", line,
                    f"{path.name}:{lineno} launches a PowerShell engine "
                    "with -File without -ExecutionPolicy Bypass",
                )
        bat = REPO_ROOT / "START.bat"
        for lineno, line in enumerate(
                bat.read_text(encoding="ascii").splitlines(), start=1):
            if "-File" in line and "PS_EXE" in line:
                checked += 1
                self.assertIn("Bypass", line, f"START.bat:{lineno}")
        self.assertGreater(checked, 4, "scanner must find the known sites")

    def test_logic_entry_scripts_unblock_motw(self) -> None:
        """bootstrap / installer / verify remove the zone mark (MOTW).

        The Unblock-File block makes manual runs work after any START.bat
        entry and is the documented fix for the field report's
        "not digitally signed" error.
        """
        for name in ("bootstrap.ps1", "install_m1.ps1", "verify_m1.ps1"):
            code = self._raw(name)
            self.assertIn("Unblock-File", code,
                          f"{name} must run Unblock-File over scripts")

    def test_logic_verify_probes_zone_identifier_on_fast_fail(self) -> None:
        """verify_m1 names the MOTW cause + the Unblock-File fix."""
        code = self._code("verify_m1.ps1")
        self.assertIn("Zone.Identifier", code)
        self.assertIn("Unblock-File", code)

    # --- (b) silent DLL death: pre-flight, transcript, decoding ----------

    def test_logic_start_script_dll_family_preflight(self) -> None:
        """The starter must check the CUDA DLL families BY NAME."""
        code = self._code("start_llama_server.ps1")
        for pattern in ('"ggml-cuda*"', '"cudart*"', '"cublas*"'):
            self.assertIn(pattern, code,
                          f"starter must check the {pattern} DLL family")

    def test_logic_start_script_transcripts_own_console(self) -> None:
        """The starter's own output survives hidden/background launches.

        Start-Transcript into logs/llama-server.starter.log, wrapped in
        try/finally so EVERY exit (including fast-fail paths) closes the
        transcript - the minimized child console output was previously
        lost, which is exactly why field report #2 had no diagnosis.
        """
        raw = self._raw("start_llama_server.ps1")
        self.assertIn("Start-Transcript", raw)
        self.assertIn("llama-server.starter.log", raw)
        self.assertIn("Stop-Transcript", raw)
        self.assertIn("} finally {", raw)
        # transcript must open BEFORE the main body and close at the END
        i_open = raw.find("Start-Transcript")
        i_try = raw.find("try {", i_open)
        i_finally = raw.rfind("} finally {")
        self.assertLess(i_open, i_try)
        self.assertLess(i_try, i_finally)

    def test_logic_start_script_decodes_dll_exit_codes(self) -> None:
        """0xC0000135 (STATUS_DLL_NOT_FOUND) is decoded in the FAIL text.

        A missing CUDA DLL produces NO stderr output - the exit code is
        the only evidence, so the starter must translate it.
        """
        code = self._code("start_llama_server.ps1")
        self.assertIn("-1073741515", code)
        self.assertIn("STATUS_DLL_NOT_FOUND", code)
        self.assertIn("-1073741519", code)   # entry point not found
        self.assertIn("-1073741819", code)   # access violation

    def test_logic_installer_dll_aware_idempotency(self) -> None:
        """Partial bin\\ (exe only) must trigger re-download, not skip.

        v0.3.1 skipped the binary step whenever llama-server.exe existed;
        a missing CUDA DLL set then caused the silent startup death on
        the target machine.
        """
        code = self._code("install_m1.ps1")
        self.assertIn("$LlamaDllSetOk", code)
        self.assertIn("(-not $LlamaDllSetOk)", code)   # wired into NeedLlama
        self.assertIn("$NeedLlama", code)
        # post-extract re-validation of the flattened DLL set
        self.assertIn("$PostGgmlCuda", code)
        self.assertIn("$PostCudart", code)
        self.assertIn("$PostCublas", code)

    def test_logic_verify_fastfail_shows_starter_transcript(self) -> None:
        """verify_m1 fast-fail dumps the starter transcript FIRST.

        Show-LlamaLogTail must list llama-server.starter.log before the
        server's own logs, and the nested launch must use -NoProfile.
        """
        code = self._code("verify_m1.ps1")
        self.assertIn('@("llama-server.starter.log", "llama-server.err.log",',
                      code)
        self.assertIn('"-NoProfile", "-ExecutionPolicy", "Bypass", "-File"',
                      code)

    def test_logic_start_agent_and_measure_vram_show_transcript(self) -> None:
        """The other two starter launchers surface the transcript too."""
        for name in ("start_agent.ps1", "measure_vram.ps1"):
            code = self._code(name)
            self.assertIn("llama-server.starter.log", code,
                          f"{name} must dump the starter transcript")



class StartupHealthFirstV033Tests(FeatureValidationTest):
    """v0.3.3 regression: target-machine field report #3.

    Manual tests proved llama-server itself healthy on the target machine
    (llama-server.exe --version PASS, model load PASS, /health HTTP 200,
    127.0.0.1:8080 listening PASS); the failure was purely in the
    verify/starter STARTUP DETECTION and EXIT-CODE handling. v0.3.3
    contract:
    - primary success condition: GET /health -> HTTP 200 (the starter
      wrapper's process exit code is NOT a success criterion; with
      /health = 200 the smoke test is PASS even if the wrapper exited
      non-zero);
    - FAIL only if the llama-server process actually stopped AND /health
      did not become available within the timeout;
    - starter-wrapper failure -> direct llama-server.exe fallback launch
      with the shell-derived config (SAME flags: -ngl -1, -c 8192, ...);
    - the python-bridge config error is a separate WARNING, not fatal,
      whenever the shell-derived llama-server config works;
    - the final verify result reports SIX separate indicators: server
      startup, health, chat completion, JSON completion, model loaded,
      process alive.
    """

    FEATURE = "scripts"

    def _code(self, name: str) -> str:
        text = (SCRIPTS_DIR / name).read_text(encoding="ascii")
        return _strip_ps_comments(text)

    # --- verify: /health is the primary success condition -----------------

    def test_logic_verify_health_is_primary_not_wrapper_exit(self) -> None:
        code = self._code("verify_m1.ps1")
        self.assertNotIn("FailedFast", code,
                         "the v0.3.2 wrapper-exit gate must be gone")
        self.assertNotIn("-lt 100", code,
                         "the old 100 s blind wait must be gone")
        self.assertIn("LLM resz-eredmenyek", code)   # 6-indicator summary
        self.assertIn("ezert ez PASS", code)         # /health 200 -> PASS even with non-0 wrapper exit

    def test_logic_verify_polls_health_30s_then_fallback(self) -> None:
        code = self._code("verify_m1.ps1")
        self.assertIn("-ge 30", code)                # 30 s primary poll window
        self.assertIn("FALLBACK: a starter wrapper kilepett", code)  # direct launch fallback
        self.assertIn(r'Join-Path $Root "bin\llama-server.exe"', code)

    def test_logic_verify_fallback_args_match_starter_contract(self) -> None:
        """The direct fallback must launch llama-server with the SAME flags."""
        code = self._code("verify_m1.ps1")
        for flag in ('"-ngl", "-1"', '"-c", "8192"', '"--parallel", "1"',
                     '"--cache-type-k", "q8_0"', '"--cache-type-v", "q8_0"',
                     '"--temp", "0.7"', '"--metrics"', '"--no-webui"',
                     '"--host", "127.0.0.1"', '"--port", "8080"'):
            self.assertIn(flag, code, f"verify fallback must pass {flag}")

    def test_logic_verify_reports_six_indicators(self) -> None:
        code = self._code("verify_m1.ps1")
        for name in (
            '"LLM server startup"',
            '"LLM health (/health 200)"',
            '"LLM chat completion (POST /v1/chat/completions)"',
            '"LLM JSON completion (response_format json_object)"',
            '"LLM model loaded (GET /v1/models)"',
            '"LLM process alive"',
        ):
            self.assertIn(name, code, f"verify must report {name} separately")

    def test_logic_verify_probes_models_endpoint(self) -> None:
        code = self._code("verify_m1.ps1")
        self.assertIn("/v1/models", code)

    def test_logic_verify_bridge_warning_not_fatal(self) -> None:
        """The config-bridge error must be a WARNING when direct config works."""
        code = self._code("verify_m1.ps1")
        self.assertIn("nem toltheto be a python-bridge-en keresztul", code)
        self.assertIn("NEM fatal", code)

    def test_logic_verify_fail_rule_requires_dead_process(self) -> None:
        """FAIL text must state the rule: process stopped AND no /health."""
        code = self._code("verify_m1.ps1")
        self.assertIn("leallt ES a /health nem valt elerhetove a timeouton belul",
                      code)

    # --- starter: bridge failure is a WARNING + shell fallback -----------

    def test_logic_starter_bridge_failure_is_not_fatal(self) -> None:
        code = self._code("start_llama_server.ps1")
        self.assertIn("nem toltheto be a python-bridge-en keresztul", code)
        self.assertIn("NEM fatal", code)
        self.assertIn("$null -eq $Cfg", code)         # the fallback gate
        self.assertIn(
            r'Join-Path $Root "models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf"',
            code,
        )

    def test_logic_starter_fallback_keeps_ngl_minus_one(self) -> None:
        """v0.4.14: the shell fallback follows the ACTIVE profile's partial
        offload (35B IQ4_XS ~19 GB does not fit 12 GB VRAM)."""
        code = self._code("start_llama_server.ps1")
        self.assertIn('$FbNgl = "26"', code)

    def test_logic_starter_dll_preflight_is_warning_only(self) -> None:
        code = self._code("start_llama_server.ps1")
        self.assertIn("ez mar NEM fatal", code)

    def test_logic_measure_vram_survives_starter_exit(self) -> None:
        code = self._code("measure_vram.ps1")
        self.assertIn("a /health varakozas folytatodik", code)


class StartupV034FieldReportTests(FeatureValidationTest):
    r"""v0.3.4 regression: target-machine field report #4.

    The llama-server start itself was FIXED and confirmed healthy on the
    target machine in v0.3.3 (server up in 2 s, /health HTTP 200); the
    pasted START.bat run then exposed the NEXT blockers, all outside the
    llama-server path:
    (1) the agent died at SileroVad init - "ModuleNotFoundError: No
        module named 'onnxruntime'" - because NOTHING installed
        onnxruntime: the lock file's TARGET-ONLY lines are comments, and
        voicemem's own dependencies pull sherpa-onnx instead;
    (2) "python -m app.main --check" crashed with UnicodeEncodeError
        ('charmap' codec - cp1252 pipe vs. the Hungarian checklist text)
        whenever the launcher captured its output;
    (3) the config python-bridge ALWAYS failed with a SyntaxError because
        PowerShell 5.1's native argument passing strips the inner double
        quotes of `python -c $Bridge` (the from_yaml r"..." raw string
        arrived as rconfig\... - quotes vanished).
    """

    FEATURE = "scripts"

    def _code(self, name: str) -> str:
        text = (SCRIPTS_DIR / name).read_text(encoding="ascii")
        return _strip_ps_comments(text)

    def _py(self, rel: str) -> str:
        return (REPO_ROOT / rel).read_text(encoding="utf-8")

    # --- (1) onnxruntime is actually installed ---------------------------

    def test_logic_installer_installs_onnxruntime(self) -> None:
        """The installer must install + import-verify onnxruntime (M1)."""
        code = self._code("install_m1.ps1")
        self.assertIn('$OnnxRuntimeVersion = "1.23.0"', code)
        self.assertIn('"onnxruntime==$OnnxRuntimeVersion"', code)
        self.assertIn("import onnxruntime", code)

    def test_logic_installer_step_counts_match(self) -> None:
        """22 documented steps == 22 Write-Step calls, /22 counter (v0.4.4: +transformers guard)."""
        raw = (SCRIPTS_DIR / "install_m1.ps1").read_text(encoding="ascii")
        self.assertIn("[{0}/22]", raw)
        self.assertEqual(raw.count('\nWrite-Step "'), 22)

    def test_logic_bootstrap_probe_sees_onnxruntime(self) -> None:
        """A venv missing onnxruntime (or the web deps) must NOT pass as 'ready'."""
        code = self._code("bootstrap.ps1")
        self.assertIn(
            "'import yaml, numpy, httpx, soundfile, torch, onnxruntime, fastapi, uvicorn, websockets'",
            code,
        )

    def test_logic_bootstrap_probe_enforces_transformers_floor(self) -> None:
        """v0.4.5 (field report #5): the dependency probe is VERSION-aware for
        transformers. An in-place upgraded venv still carrying voicemem's
        4.52.3 pin passed the import-only probe, the bootstrap skipped the
        installer, and the v0.4.4 step-16 guard never ran -> qwen3_asr never
        loaded ("ASR not green", text-only mode). The probe must exit 5 for
        an importable-but-older version and route to the installer."""
        code = self._code("bootstrap.ps1")
        self.assertIn("transformers.__version__", code)
        # v0.4.7: the floor is 5.0 - the native qwen3_asr module (the
        # processor + generate call path in app/asr.py) exists from
        # transformers 5.x; 4.57 does NOT contain it.
        self.assertIn("(5, 0)", code)
        self.assertIn("else 5", code)
        # exit 5 must NOT pass as ready -> the installer (self-healing) runs
        self.assertIn("if ($LASTEXITCODE -eq 5) {", code)
        after_probe = code.split("$TfProbe = '")[1]
        self.assertIn("return $false", after_probe)
        # the probe names the concrete field symptom (ASR) in its warning
        self.assertIn("Qwen3-ASR", after_probe)

    def test_logic_bootstrap_version_probe_has_no_embedded_double_quotes(self) -> None:
        """PS 5.1 native-argument quoting: the probe code must stay free of
        embedded double quotes (the '' pairs are escaped single quotes)."""
        code = self._code("bootstrap.ps1")
        lines = [l for l in code.splitlines() if "$TfProbe = '" in l]
        self.assertEqual(len(lines), 1, "exactly one $TfProbe assignment")
        inner = lines[0].split("$TfProbe = '", 1)[1]
        self.assertTrue(inner.endswith("'"), "assignment closes the string")
        inner = inner[:-1]
        self.assertNotIn('"', inner)
        # the escaped single quotes survive as Python string literals
        self.assertIn("split(''+'')", inner)
        self.assertIn("split(''.'')", inner)
        self.assertIn("sys.exit(0 if v >= (5, 0) else 5)", inner)

    # --- (2) cp1252 crash guard ------------------------------------------

    def test_logic_main_stdio_reconfigured_to_replace(self) -> None:
        """app/main.py must never crash on console encoding."""
        main_py = self._py("app/main.py")
        self.assertIn("reconfigure(errors=", main_py)
        self.assertIn("_make_stdio_resilient()", main_py)

    def test_logic_launchers_set_utf8_stdio(self) -> None:
        """start_agent + bootstrap set PYTHONIOENCODING for python children."""
        for name in ("start_agent.ps1", "bootstrap.ps1"):
            code = self._code(name)
            self.assertIn(
                "PYTHONIOENCODING", code,
                f"{name} must set PYTHONIOENCODING for its python children",
            )

    # --- (3) the bridge runs from a temp FILE, not `python -c` -----------

    def test_logic_starter_bridge_runs_from_temp_file(self) -> None:
        """`python -c $Bridge` is gone: the bridge runs from a .py file."""
        code = self._code("start_llama_server.ps1")
        self.assertIn("Set-Content -Path $BridgePy", code)
        self.assertIn("& $VenvPython $BridgePy", code)
        self.assertIn("Remove-Item -Path $BridgePy", code)
        self.assertNotIn("-c $Bridge", code)

    def test_logic_verify_bridge_probe_runs_from_temp_file(self) -> None:
        """The verify bridge probe gets the same temp-file treatment."""
        code = self._code("verify_m1.ps1")
        self.assertIn("Set-Content -Path $BridgeProbePy", code)
        self.assertIn("& $VenvPython $BridgeProbePy", code)
        self.assertNotIn("-c $BridgeProbe", code)



    # --- v0.4.0: the LOCAL web UI startup flow ----------------------------

    def test_logic_start_agent_has_web_mode(self) -> None:
        """--web: local backend on 8787, health poll, browser opens, URL printed."""
        code = self._code("start_agent.ps1")
        self.assertIn('$WebMode = $false', code)
        self.assertIn('-eq "-Web" -or $Arg -eq "--web"', code)
        self.assertIn('app.web_server', code)
        self.assertIn('$WebPort = 8787', code)
        self.assertIn('/api/health', code)
        self.assertIn('Start-Process $WebUrl', code)
        self.assertIn('VOICEMEM WEB UI:', code)
        self.assertIn('Wait-Process -Id $Proc.Id', code)

    def test_logic_bootstrap_run_mode_is_web(self) -> None:
        """bootstrap 'run' dispatches to start_agent --web (the normal flow)."""
        code = self._code("bootstrap.ps1")
        self.assertIn("@('--web')", code)
        self.assertIn("'cli'", code.split("ValidateSet")[-1].split(")")[0])

    def test_logic_start_bat_documents_web_url(self) -> None:
        """START.bat: default flow documented as web UI at 127.0.0.1:8787."""
        text = (REPO_ROOT / "START.bat").read_text(encoding="ascii")
        self.assertIn("http://127.0.0.1:8787", text)
        self.assertIn('"cli"', text)
        self.assertIn('"%MODE%"=="web" set "MODE=run"', text)

    def test_logic_web_ui_assets_exist(self) -> None:
        """web/voicemem.html exists, speaks /ws, has mic + pipeline panels."""
        ui = REPO_ROOT / "web" / "voicemem.html"
        self.assertTrue(ui.is_file(), "web/voicemem.html missing")
        html = ui.read_text(encoding="utf-8")
        self.assertIn("'/ws'", html)
        self.assertIn("micSel", html)
        self.assertIn("pipeStrip", html)
        self.assertIn("XTransformPort", html)
        self.assertIn("asrBypassed", html)
        self.assertIn("SAMPLE_RATE=24000", html)
        img = REPO_ROOT / "web" / "images" / "background.webp"
        self.assertTrue(img.is_file(), "web/images/background.webp missing")

    def test_logic_smoke_test_script_exists(self) -> None:
        """scripts/smoke_test_web.py: the manual smoke checklist runner."""
        smoke = SCRIPTS_DIR / "smoke_test_web.py"
        self.assertTrue(smoke.is_file(), "smoke_test_web.py missing")
        text = smoke.read_text(encoding="utf-8")
        self.assertIn("/api/pipeline", text)
        self.assertIn("/api/memories", text)
        self.assertIn("open_browser", text)


class ReleaseWebUiZipV035Tests(FeatureValidationTest):
    r"""v0.3.5 regression: the release ZIP ships the whole system.

    The v0.3.4 archive was built BEFORE the web-UI milestone, so it carried
    no web element at all (web/, app/web_server.py, smoke_test_web.py).
    User rule: the ZIP must contain EVERY element of the system - so the
    builder allowlist stages ``web/`` from v0.3.5 on, and the ZIP
    self-check verifies the web UI payload.
    """

    FEATURE = "scripts"

    def _code(self, name: str) -> str:
        text = (SCRIPTS_DIR / name).read_text(encoding="ascii")
        return _strip_ps_comments(text)

    def test_logic_builder_stages_web_dir(self) -> None:
        """build_release.ps1: web/ is in the staging allowlist + skeleton.

        v0.5.0: the allowlist also stages vendor/ — the CONTROLLED VoiceMem
        source ships inside the release ZIP (the installer pip-installs from
        it; no upstream clone at install time anymore).
        """
        code = self._code("build_release.ps1")
        self.assertIn(
            '@("app", "config", "scripts", "tests", "web", "vendor")', code)
        self.assertIn('"models/speaker"', code)

    def test_logic_web_ui_source_files_exist(self) -> None:
        """Every web element the ZIP must carry exists in the repo."""
        for rel in (
            "web/voicemem.html",
            "web/images/background.webp",
            "app/web_server.py",
            "scripts/smoke_test_web.py",
            "tests/unit/test_web_server.py",
            "tests/integration/test_web_e2e.py",
        ):
            self.assertTrue(
                (REPO_ROOT / rel).is_file(), f"web system element missing: {rel}"
            )

    def test_logic_web_server_is_staged_component(self) -> None:
        """app/web_server.py stays the --web backend entry (START.bat flow)."""
        bat = (REPO_ROOT / "START.bat").read_text(encoding="ascii")
        self.assertIn("http://127.0.0.1:8787", bat)
        code = self._code("start_agent.ps1")
        self.assertIn("app.web_server", code)
        self.assertIn("$WebPort = 8787", code)


class StartBatFieldReportV036Tests(FeatureValidationTest):
    r"""v0.3.6 field report: START.bat dead-ended at the usage menu.

    The user's run printed ONLY the "Known modes" menu + "Press any key"
    - the web UI never opened. The menu is the v0.3.5 unknown-mode error
    block, so an unexpected first argument reached the batch (drag-and-
    drop, a custom .bat association, an unquoted path). Fixes: (1) an
    unknown mode now CONTINUES in the default web-UI mode, (2) the
    START.bat ships with Windows CRLF line endings (the v0.3.5 ZIP was
    LF-only), and (3) the builder normalizes every staged *.bat to CRLF.
    """

    FEATURE = "scripts"

    def _code(self, name: str) -> str:
        text = (SCRIPTS_DIR / name).read_text(encoding="ascii")
        return _strip_ps_comments(text)

    def test_logic_start_bat_is_crlf(self) -> None:
        """The repo START.bat must be CRLF (cmd.exe parser hazard)."""
        raw = (REPO_ROOT / "START.bat").read_bytes()
        lone_lf = raw.count(b"\n") - raw.count(b"\r\n")
        self.assertEqual(lone_lf, 0, "START.bat must not have LF-only lines")
        self.assertGreater(raw.count(b"\r\n"), 100)

    def test_logic_start_bat_unknown_mode_continues(self) -> None:
        """Unknown mode -> [NOTE] + set MODE=run (the web UI), no exit."""
        text = (REPO_ROOT / "START.bat").read_text(encoding="ascii")
        self.assertIn("[NOTE] Unknown start mode", text)
        self.assertIn("[NOTE] Known modes: web, cli, mock, check, "
                      "benchmark, repair, build.", text)
        self.assertGreaterEqual(text.count('set "MODE=run"'), 2)
        self.assertIn('"%MODE%"=="web" set "MODE=run"', text)
        self.assertIn("http://127.0.0.1:8787", text)
        # the v0.3.5 dead-end block is gone
        self.assertNotIn('Unknown mode: "%MODE%"', text)
        self.assertNotIn("[ERROR] Unknown mode", text)

    def test_logic_start_bat_fallback_block_is_parse_safe(self) -> None:
        """%MODE% is never echoed inside the fallback block (a dropped
        path with parentheses would break a parenthesized cmd block)."""
        lines = (REPO_ROOT / "START.bat").read_text(
            encoding="ascii").splitlines()
        in_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("if /I not") and stripped.endswith("("):
                in_block = True
                continue
            if in_block and stripped == ")":
                in_block = False
                continue
            if in_block and "echo" in stripped:
                self.assertNotIn(
                    "%MODE%", stripped,
                    f"%MODE% echoed inside the fallback block: {stripped!r}")

    def test_logic_builder_normalizes_bat_to_crlf(self) -> None:
        """build_release.ps1 step 4c: every staged *.bat becomes CRLF."""
        code = self._code("build_release.ps1")
        self.assertIn('Get-ChildItem -Path $StageRepo -Recurse -Filter "*.bat"',
                      code, "the builder does not collect staged .bat files")
        self.assertIn('[IO.File]::WriteAllBytes', code,
                      "the builder does not rewrite the staged .bat bytes")
        self.assertIn('.Replace("`n", "`r`n")', code,
                      "the builder does not normalize LF to CRLF")


if __name__ == "__main__":
    unittest.main()
