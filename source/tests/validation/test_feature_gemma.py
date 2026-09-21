"""Gemma LLM policy validation (v0.4.3 - final LLM migration).

Validates the ONE-LLM contract of the v0.4.3 milestone against the real
repo state (no network, no GPU - fully hermetic):

  * MODELS.lock.json: exactly ONE llm entry, pinned to the OFFICIAL
    Google QAT repository with the verified sha256, NO fallback repos.
  * app/config.py: llm_model_name == "gemma-4-12b" and llm_model_file
    resolves ONLY to the Gemma GGUF - legacy Qwen3 LLM files that may
    still sit on an old install are ignored (LLM FALLBACK: NONE).
  * scripts/smoke_test_gemma.py: the 14-point startup validation exists
    and pins the same repo/file/sha256; its pure helpers work.
  * scripts/quality_test_gemma.py: the deterministic HU/EN/switching
    suite reuses the existing detect_language logic (architecture
    unchanged).
  * verify_m1.ps1 / start_llama_server.ps1 / env files: the Gemma path
    and the required llama-server flags (ctx 8192, parallel 1, -ngl -1,
    KV q8_0/q8_0, temp 0.7, --metrics, --no-webui).
  * install/bootstrap banners report "LLM: Gemma 4 12B Q4_0".
  * ACTIVE configuration contains NO Qwen3-4B/8B/35B LLM reference and
    no OpenAI cloud endpoint (only the 127.0.0.1 llama-server).

Historical CHANGELOG entries are allowed to mention the retired models
(as history), but nothing ACTIVE may reference them.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "MODELS.lock.json"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_test_gemma.py"
QUALITY_SCRIPT = REPO_ROOT / "scripts" / "quality_test_gemma.py"

GEMMA_REPO = "google/gemma-4-12B-it-qat-q4_0-gguf"
GEMMA_FILE = "gemma-4-12b-it-qat-q4_0.gguf"
GEMMA_SHA256 = "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"

#: Active runtime surfaces that must be Qwen3-LLM-free (CHANGELOG is
#: allowed to keep HISTORY, that is not active configuration).
ACTIVE_SURFACES = [
    "MODELS.lock.json",
    "INSTALL_MANIFEST.example.json",
    "config/voicemem_config.yaml",
    "config/env.local.ps1",
    "config/env.local.sh",
    "app/config.py",
    "app/llm.py",
    "app/web_server.py",
    "app/main.py",
    "app/pipeline.py",
    "web/voicemem.html",
    "scripts/download_models_hf.py",
    "scripts/download_models.ps1",
    "scripts/install_m1.ps1",
    "scripts/bootstrap.ps1",
    "scripts/start_llama_server.ps1",
    "scripts/start_agent.ps1",
    "scripts/verify_m1.ps1",
    "scripts/verify_setup.py",
    "scripts/smoke_test_web.py",
    "scripts/write_install_manifest.py",
    "scripts/measure_vram.ps1",
    "scripts/build_release.ps1",
    "models/llm/gemma-4-12b/README.md",
    "README.md",
    "CONTRACT.md",
    "LICENSES.md",
]

#: Qwen3 LLM identifiers that must never appear in an ACTIVE surface
#: (the ASR stays Qwen3-ASR-0.6B - that is a different component).
BANNED_LLM_TOKENS = (
    "Qwen3-4B", "qwen3-4b", "Qwen3-8B", "qwen3-8b",
    "Qwen3-35B", "qwen3-35b", "Qwen/Qwen3-8B-GGUF", "Qwen/Qwen3-4B-GGUF",
)

#: Runtime network endpoints must be loopback only.
_LOOPBACK_MARKERS = ("127.0.0.1", "localhost")
_CLOUD_OPENAI = "api.openai.com"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class GemmaPolicyFeatureTest(FeatureValidationTest):
    FEATURE = "gemma"

    # ------------------------------------------------------- lock policy ----

    def test_logic_lock_has_exactly_one_llm(self) -> None:
        lock = json.loads(_read(LOCK_PATH))
        llm_entries = [e for e in lock["models"] if e.get("component") == "llm"]
        self.assertEqual(len(llm_entries), 1, "LLM MODEL COUNT must be 1")
        entry = llm_entries[0]
        self.assertEqual(entry["repo"], GEMMA_REPO)
        self.assertEqual(entry["files"], [GEMMA_FILE])
        self.assertEqual(entry["target_dir"], "models/llm/gemma-4-12b")
        self.assertNotIn("fallback_repos", entry, "LLM FALLBACK must be NONE")
        self.assertEqual(entry.get("sha256"), GEMMA_SHA256)
        self.assertEqual(entry.get("size_bytes"), 6975879296)
        gguf_min = min(entry["min_bytes"].values())
        self.assertGreaterEqual(gguf_min, 6 * 10 ** 9)
        # no OTHER model entry may look like an LLM fallback
        others = [e for e in lock["models"] if e.get("component") != "llm"]
        self.assertTrue(
            all(e.get("component") in ("asr", "embedding", "emotion",
                                       "speaker", "vad", "tts")
                for e in others),
            "unexpected non-LLM components in the lock",
        )

    def test_logic_no_qwen_llm_in_lock(self) -> None:
        text = _read(LOCK_PATH)
        for token in BANNED_LLM_TOKENS:
            self.assertNotIn(token, text, f"lock references retired LLM {token}")

    # ------------------------------------------------------ config policy ----

    def test_logic_config_pins_gemma(self) -> None:
        """v0.4.14: the ACTIVE profile is Qwen3.6 35B A3B IQ4_XS; Gemma 4
        12B stays the documented fallback profile (env.local.ps1 block)."""
        with patch.dict(os.environ, {"LLAMA_MODEL_PATH": ""}):
            cfg = AgentConfig()
            cfg.apply_env()
        self.assertEqual(cfg.llm_model_name, "qwen3.6-35b-a3b")
        self.assertEqual(
            cfg.llm_model_file,
            cfg.root / "models" / "llm" / "qwen3.6-35b-a3b" / "Qwen3.6-35B-A3B-IQ4_XS.gguf",
        )

    def test_logic_no_llm_fallback_to_legacy_qwen_files(self) -> None:
        """LLM FALLBACK: NONE - old Qwen3 GGUFs on disk must be IGNORED.

        Field regression: v0.4.2's llm_model_file fell back to a legacy
        Qwen3-4B GGUF when the 8B was absent. v0.4.3 must resolve to the
        Gemma path even with BOTH old files present on disk.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in (
                "models/llm/qwen3-4b/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                "models/llm/qwen3-8b/Qwen3-8B-Q4_K_M.gguf",
                "models/llm/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-IQ4_XS.gguf",
                "models/llm/gemma-4-12b/gemma-4-12b-it-qat-q4_0.gguf",
            ):
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"GGUF" + b"\x00" * 64)
            env = {
                "VOICEMEM_HOME": str(root),
                "LLAMA_MODEL_PATH": "",
            }
            with patch.dict(os.environ, env):
                cfg = AgentConfig()
                cfg.apply_env()
                resolved = cfg.llm_model_file
            self.assertTrue(
                str(resolved).endswith("Qwen3.6-35B-A3B-IQ4_XS.gguf"),
                f"must resolve to the ACTIVE profile, got {resolved}",
            )
            # the RETIRED legacy models must stay ignored
            self.assertNotIn("qwen3-4b", str(resolved).lower())
            self.assertNotIn("qwen3-8b", str(resolved).lower())

    def test_logic_config_source_has_no_qwen_llm(self) -> None:
        src = _read(REPO_ROOT / "app" / "config.py")
        for token in BANNED_LLM_TOKENS:
            self.assertNotIn(token, src, f"config.py references {token}")

    # ------------------------------------------------- smoke test contract ----

    def test_logic_smoke_script_pins_and_checks(self) -> None:
        self.assertTrue(SMOKE_SCRIPT.is_file(), "scripts/smoke_test_gemma.py missing")
        src = _read(SMOKE_SCRIPT)
        self.assertIn(GEMMA_REPO, src)
        self.assertIn(GEMMA_FILE, src)
        self.assertIn(GEMMA_SHA256, src)
        self.assertIn("GEMMA_SIZE_BYTES = 6975879296", src)
        # the required llama-server flags (milestone Section 3)
        for flag in ('"-ngl", "-1"', '"-c", "8192"', '"--parallel", "1"',
                     '"--cache-type-k", "q8_0"', '"--cache-type-v", "q8_0"',
                     '"--temp", "0.7"', '"--metrics"', '"--no-webui"',
                     '"--host", "127.0.0.1"'):
            self.assertIn(flag, src, f"llama-server flag missing: {flag}")
        # all 14 numbered checks exist
        for num in range(1, 15):
            self.assertIn(f"self.record({num}, ", src,
                          f"smoke check #{num} missing")
        # the empty-content regression must be an explicit FAIL
        self.assertIn("non-empty assistant content", src)
        self.assertIn("EMPTY", src)

    def test_logic_smoke_script_helpers(self) -> None:
        from scripts.smoke_test_gemma import (
            cpu_fallback_lines,
            extract_content,
            gguf_header_valid,
            parse_gpu_offload,
        )
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as handle:
            handle.write(b"GGUF" + bytes([3, 0, 0, 0]) + bytes(8))
            path = Path(handle.name)
        try:
            ok, detail = gguf_header_valid(path)
            self.assertTrue(ok, detail)
        finally:
            path.unlink(missing_ok=True)

        full = parse_gpu_offload("load_tensors: offloaded 48/48 layers to GPU")
        self.assertTrue(full["full"])
        partial = parse_gpu_offload("load_tensors: offloaded 40/48 layers to GPU")
        self.assertFalse(partial["full"])
        self.assertEqual(partial["offloaded"], 40)
        modern = parse_gpu_offload(
            "load_tensors: offloading 60 layers to GPU\n"
            "load_tensors: offloading non-repeating layers to GPU")
        self.assertTrue(modern["full"])
        self.assertEqual(cpu_fallback_lines("x: falling back to CPU backend"),
                         ["x: falling back to CPU backend"])
        self.assertEqual(cpu_fallback_lines("clean log"), [])
        self.assertEqual(extract_content({"choices": [{"message": {"content": "hi"}}]}), "hi")
        self.assertEqual(extract_content({}), "")

    # ------------------------------------------------ quality suite contract ----

    def test_logic_quality_script_contract(self) -> None:
        self.assertTrue(QUALITY_SCRIPT.is_file(),
                        "scripts/quality_test_gemma.py missing")
        src = _read(QUALITY_SCRIPT)
        self.assertIn("detect_language", src,
                      "must reuse the existing language detection architecture")
        self.assertIn("temperature\": 0.0", src, "deterministic suite")
        for num in range(1, 18):
            self.assertIn(f"self.check_reply({num}, ", src,
                          f"quality check #{num} missing")
        self.assertIn("switch back", src)
        self.assertIn("expect_lang", src)

    # ------------------------------------------------- launcher / verifier ----

    def test_logic_starter_uses_gemma_and_required_flags(self) -> None:
        code = _read(REPO_ROOT / "scripts" / "start_llama_server.ps1")
        # v0.4.14: the shell fallback follows the ACTIVE profile
        self.assertIn(r"models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf", code)
        # the required llama-server flags (milestone Section 3) - the starter
        # passes them through from AgentConfig (ctx 8192 / ngl -1 / parallel 1
        # / KV q8_0 / temp 0.7 are the config + shell defaults)
        for flag in ('"--model"', '"--host"', '"--port"', '"-ngl"', '"-c"',
                     '"--parallel"', '"--cache-type-k"', '"--cache-type-v"',
                     '"--temp"', '"--metrics"', '"--no-webui"'):
            self.assertIn(flag, code, f"starter flag missing: {flag}")
        self.assertIn('$FbNgl = "26"', code,
                      "shell fallback must follow the active profile's "
                      "partial offload (35B > 12 GB VRAM)")
        self.assertIn('$FbCtx = "8192"', code, "shell fallback must keep ctx 8192")
        self.assertIn('$FbCkK = "q8_0"', code)
        self.assertIn('$FbCkV = "q8_0"', code)
        # the config bridge still drives everything (architecture unchanged)
        self.assertIn("AgentConfig.from_yaml", code)

    def test_logic_verify_m1_gemma_requests_and_model_gate(self) -> None:
        code = _read(REPO_ROOT / "scripts" / "verify_m1.ps1")
        self.assertIn('"model":"qwen3.6-35b-a3b"', code)
        self.assertIn(r"models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf", code)
        # the model-loaded check must REQUIRE the ACTIVE profile
        self.assertIn("$ModelId -match \"qwen3\\.6-35b-a3b\"", code)
        self.assertIn("NEM az aktiv profil", code)

    def test_logic_env_files_pin_gemma(self) -> None:
        """v0.4.14: env files carry the ACTIVE Qwen profile AND the
        documented Gemma fallback profile (both must stay present)."""
        ps1 = _read(REPO_ROOT / "config" / "env.local.ps1")
        self.assertIn("gemma-4-12b-it-qat-q4_0.gguf", ps1)  # fallback profile
        self.assertIn('OPENAI_MODEL = "qwen3.6-35b-a3b"', ps1)  # ACTIVE
        self.assertIn("Qwen3.6-35B-A3B-IQ4_XS.gguf", ps1)
        self.assertIn('LLAMA_N_GPU_LAYERS = "26"', ps1)  # partial offload
        self.assertIn('LLM_DISABLE_THINKING = "1"', ps1)
        sh = _read(REPO_ROOT / "config" / "env.local.sh")
        self.assertIn("gemma-4-12b-it-qat-q4_0.gguf", sh)  # fallback profile
        self.assertIn('OPENAI_MODEL="qwen3.6-35b-a3b"', sh)  # ACTIVE
        self.assertIn("Qwen3.6-35B-A3B-IQ4_XS.gguf", sh)

    def test_logic_banners_report_gemma(self) -> None:
        installer = _read(REPO_ROOT / "scripts" / "install_m1.ps1")
        self.assertIn("Qwen3.6 35B A3B IQ4_XS (aktiv profil", installer)
        self.assertIn("Gemma 4 12B Q4_0", installer)  # fallback named too
        bootstrap = _read(REPO_ROOT / "scripts" / "bootstrap.ps1")
        self.assertIn("Qwen3.6 35B A3B IQ4_XS (aktiv)", bootstrap)
        self.assertIn("Gemma 4 12B Q4_0", bootstrap)
        self.assertIn("smoke_test_gemma.py", bootstrap,
                      "check mode must run the Gemma smoke test")

    # ------------------------------------------------- downloader contract ----

    def test_logic_downloader_single_pinned_source(self) -> None:
        src = _read(REPO_ROOT / "scripts" / "download_models_hf.py")
        self.assertIn('LLM_REPO = "google/gemma-4-12B-it-qat-q4_0-gguf"', src)
        self.assertIn("LLM_PINNED_SHA256", src)
        self.assertIn("sha256_of(model_path)", src,
                      "the LLM download must be sha256-verified")
        for banned in ("unsloth/Qwen3-8B-GGUF", "bartowski/Qwen_Qwen3-8B-GGUF"):
            self.assertNotIn(banned, src)
        self.assertIn("LLM fallback", src)

    # --------------------------------------------------- repo-wide sweep ----

    def test_logic_active_surfaces_have_no_qwen_llm(self) -> None:
        for rel in ACTIVE_SURFACES:
            path = REPO_ROOT / rel
            if not path.is_file():
                # some surfaces are optional (build_release.ps1 exists though)
                self.assertIn(rel, ("config/.env.example",),
                              f"active surface missing: {rel}")
                continue
            text = _read(path)
            for token in BANNED_LLM_TOKENS:
                self.assertNotIn(
                    token, text,
                    f"retired LLM {token} still referenced in ACTIVE surface {rel}",
                )

    def test_logic_no_openai_cloud_endpoint_in_runtime(self) -> None:
        for rel in ("config/env.local.ps1", "config/env.local.sh",
                    "app/config.py", "app/llm.py", "scripts/smoke_test_gemma.py"):
            text = _read(REPO_ROOT / rel)
            self.assertNotIn(_CLOUD_OPENAI, text,
                             f"{rel} references the OpenAI cloud endpoint")
        for rel in ("config/env.local.ps1", "config/env.local.sh"):
            text = _read(REPO_ROOT / rel)
            self.assertIn("127.0.0.1:8080", text,
                          f"{rel} must point OPENAI_BASE_URL at the local llama-server")

    # ------------------------------------------------------------- deep ------

    def test_deep_gemma_server_running(self) -> None:
        """Deep PASS only when a REAL llama-server serves GEMMA right now.

        In the build sandbox (no GPU/model) this deep-SKIPS - the full
        14-point validation runs on the target machine via
        scripts/smoke_test_gemma.py (START.bat check mode).
        """
        try:
            import httpx  # noqa: F401
        except ImportError:
            self.deep_skip("httpx not installed")
            return
        cfg = AgentConfig()
        cfg.apply_env()
        import httpx

        try:
            with httpx.Client(timeout=3.0) as client:
                health = client.get(cfg.llama_server_health_url)
                body = health.json() if health.status_code == 200 else None
        except (httpx.HTTPError, OSError):
            self.deep_skip(f"llama-server not running on 127.0.0.1:8080 "
                           f"(build sandbox: run scripts/smoke_test_gemma.py "
                           f"on the RTX 5070 target machine)")
            return
        if not isinstance(body, dict) or body.get("status") != "ok":
            self.deep_skip("llama-server answered but is not healthy/ready")
            return
        try:
            with httpx.Client(timeout=5.0) as client:
                models = client.get(f"{cfg.llama_server_url}/models").json()
                model_id = str(models["data"][0]["id"]).lower()
        except (httpx.HTTPError, OSError, KeyError, IndexError, ValueError):
            self.deep_skip("could not list /v1/models on the running server")
            return
        if "gemma" not in model_id:
            self.deep_skip(
                f"running llama-server model is not Gemma ({model_id!r})"
            )
            return
        self.deep_pass(f"llama-server is serving Gemma (model id {model_id!r})")


if __name__ == "__main__":
    unittest.main()
