"""Single-LLM policy validation (v0.4.16/0.4.17 — Qwen3.6 35B A3B IQ4_XS ONLY).

The v0.4.16 field directive retired the Gemma fallback profile entirely:
Qwen3.6 35B A3B IQ4_XS is the ONE AND ONLY production LLM and there is NO
fallback profile, NO switching back, NO second model. This suite pins that
contract against the real repo state (no network, no GPU - hermetic):

  * MODELS.lock.json: NO llm entry at all (the Qwen GGUF is OPERATOR-PLACED,
    ~19 GB, never auto-downloaded; select it with the web UI "LLM model"
    picker / scripts/find_qwen_gguf.ps1 / scripts/identify_ollama_blob.ps1
    / manual placement).
  * app/config.py: llm_model_name == "qwen3.6-35b-a3b"; llm_model_file is
    OPTIONAL — user-selection (config/llm_model.json) > LLAMA_MODEL_PATH env
    > a GGUF ACTUALLY PRESENT in models/llm/qwen3.6-35b-a3b/ (magic-
    validated, any file name). v0.4.17: NO phantom project-default path —
    when nothing resolves it is None (the honest "no model configured").
  * Portable model path: config/llm_model.json wins over the env profile
    (the resolution chain is pinned here).
  * start_llama_server.ps1 / verify_m1.ps1: they resolve the model through
    the same chain and never suggest a Gemma fallback.
  * v0.4.17: Ollama blob support — an Ollama content-addressed blob
    (sha256-... in C:\AI_HOME\models\blobs\ or ANY OLLAMA_MODELS root) IS
    a valid llama-server model file (magic decides, never the file name);
    scripts/identify_ollama_blob.ps1 pins the exact blob + digest + split
    verdict; split shards are rejected (not directly loadable).
  * No ACTIVE runtime surface mentions Gemma as an alternative runtime
    model (CHANGELOG keeps history, which is allowed).
  * The retired Gemsmoke/quality scripts are GONE (deleted in v0.4.16).
  * No OpenAI cloud endpoint anywhere (only the 127.0.0.1 llama-server).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "MODELS.lock.json"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_test_gemma.py"
QUALITY_SCRIPT = REPO_ROOT / "scripts" / "quality_test_gemma.py"
SETTINGS_MODULE = REPO_ROOT / "app" / "llm_model_settings.py"
GGUF_MODULE = REPO_ROOT / "app" / "gguf.py"
PICKER_MODULE = REPO_ROOT / "app" / "native_picker.py"

QWEN_NAME = "qwen3.6-35b-a3b"
QWEN_FILE = "Qwen3.6-35B-A3B-IQ4_XS.gguf"
QWEN_REPO = "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF"
IDENTIFY_SCRIPT = REPO_ROOT / "scripts" / "identify_ollama_blob.ps1"

#: Active runtime surfaces that must NOT present Gemma (or any other
#: model) as an alternative runtime model. CHANGELOG keeps history.
ACTIVE_SURFACES = [
    "MODELS.lock.json",
    "INSTALL_MANIFEST.example.json",
    "config/voicemem_config.yaml",
    "config/env.local.ps1",
    "config/env.local.sh",
    "config/.env.example",
    "app/config.py",
    "app/llm.py",
    "app/web_server.py",
    "app/teacher_persona.py",
    "app/llm_model_settings.py",
    "app/gguf.py",
    "app/native_picker.py",
    "scripts/start_llama_server.ps1",
    "scripts/verify_m1.ps1",
    "scripts/find_qwen_gguf.ps1",
    "scripts/identify_ollama_blob.ps1",
    "scripts/start_agent.ps1",
    "scripts/measure_vram.ps1",
    "scripts/bootstrap.ps1",
    "scripts/install_m1.ps1",
    "scripts/download_models_hf.py",
    "scripts/write_install_manifest.py",
    "scripts/localise_memories.py",
    "scripts/smoke_test_web.py",
    "scripts/build_release.ps1",
    "web/voicemem.html",
    "README.md",
    "models/llm/qwen3.6-35b-a3b/README.md",
]


class SingleLlmPolicyFeatureTest(FeatureValidationTest):
    """v0.4.16: Qwen3.6 35B A3B IQ4_XS is the ONE AND ONLY LLM (no fallback)."""

    feature = "llm_single_model_policy"

    # -- lock ---------------------------------------------------------------- #

    def test_logic_lock_has_no_llm_entry(self) -> None:
        """The LLM is operator-placed: not part of the auto-download set."""
        data = json.loads(LOCK_PATH.read_text("utf-8"))
        comps = [str(m.get("component", "")) for m in data["models"]]
        self.assertNotIn("llm", comps, f"llm must not be auto-downloaded: {comps}")
        self.assertIn("asr", comps)
        self.assertIn("vad", comps)

    def test_logic_lock_documents_operator_placed_qwen(self) -> None:
        """The lock note points at the exact tested Qwen source."""
        raw = LOCK_PATH.read_text("utf-8")
        self.assertIn(QWEN_REPO, raw)
        self.assertIn("IQ4_XS", raw)
        self.assertIn("operator-placed", raw.replace("OPERATOR-PLACED", "operator-placed").replace("Operator-Placed", "operator-placed").lower() or "operator-placed")
        self.assertNotIn("gemma", raw.lower())

    # -- config --------------------------------------------------------------- #

    def test_logic_config_pins_qwen(self) -> None:
        cfg = AgentConfig()
        self.assertEqual(cfg.llm_model_name, QWEN_NAME)

    def test_logic_default_model_file_is_qwen(self) -> None:
        """v0.4.17: NO phantom default. None when nothing is configured; a
        GGUF ACTUALLY PRESENT in the operator dir (ANY file name — an
        Ollama sha256-... blob qualifies, magic decides) is the default."""
        import struct

        with tempfile.TemporaryDirectory() as tmp:
            os.environ.pop("LLAMA_MODEL_PATH", None)
            try:
                cfg = AgentConfig(root=Path(tmp))
                self.assertIsNone(
                    cfg.llm_model_file,
                    "empty operator dir must resolve to None, never a "
                    "non-existent project-default path",
                )
                op_dir = Path(tmp) / "models" / "llm" / "qwen3.6-35b-a3b"
                op_dir.mkdir(parents=True)
                blob = op_dir / (
                    "sha256-afc7238af403ed454b7846454091b5e38b07575bdbb64c3e86777414dde4193c"
                )
                blob.write_bytes(
                    b"GGUF" + struct.pack("<IIQQ", 3, 0, 0, 0) + b"\x00" * 32
                )
                cfg = AgentConfig(root=Path(tmp))
                self.assertEqual(str(cfg.llm_model_file), str(blob))
            finally:
                pass

    def test_logic_env_override_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("LLAMA_MODEL_PATH")
            os.environ["LLAMA_MODEL_PATH"] = str(Path(tmp) / "custom.gguf")
            try:
                cfg = AgentConfig(root=Path(tmp))
                self.assertEqual(str(cfg.llm_model_file), str(Path(tmp) / "custom.gguf"))
            finally:
                if old is None:
                    os.environ.pop("LLAMA_MODEL_PATH", None)
                else:
                    os.environ["LLAMA_MODEL_PATH"] = old

    def test_logic_user_selection_wins_over_env(self) -> None:
        """v0.4.16 resolution chain: llm_model.json > LLAMA_MODEL_PATH."""
        from app.llm_model_settings import LlmModelSettings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["VOICEMEM_LLM_MODEL_SETTINGS"] = str(root / "llm_model.json")
            os.environ["LLAMA_MODEL_PATH"] = str(root / "env-profile.gguf")
            try:
                settings = LlmModelSettings.load(root)
                settings.apply_selection(
                    str(root / "selected.gguf"),
                    model_name="Qwen_Qwen3.6-35B-A3B",
                    quant="IQ4_XS",
                    file_size=123,
                )
                cfg = AgentConfig(root=root)
                self.assertEqual(
                    str(cfg.llm_model_file), str(root / "selected.gguf")
                )
                # persistence across a fresh load
                cfg2 = AgentConfig(root=root)
                self.assertEqual(
                    str(cfg2.llm_model_file), str(root / "selected.gguf")
                )
                # clearing returns control to the env profile
                settings.clear()
                cfg3 = AgentConfig(root=root)
                self.assertEqual(
                    str(cfg3.llm_model_file), str(root / "env-profile.gguf")
                )
            finally:
                os.environ.pop("VOICEMEM_LLM_MODEL_SETTINGS", None)
                os.environ.pop("LLAMA_MODEL_PATH", None)

    # -- resolution chain in the starter/verify -------------------------------- #

    def test_logic_starter_resolves_json_first(self) -> None:
        raw = (REPO_ROOT / "scripts" / "start_llama_server.ps1").read_text(
            "utf-8", errors="replace"
        )
        self.assertIn("llm_model.json", raw)
        self.assertIn("llama-server.resolved-model.json", raw)
        # v0.4.16: a stale server with a DIFFERENT model is auto-restarted
        self.assertIn("Get-LlamaServedModel", raw)
        self.assertIn("Stop-LlamaServerOnPort", raw)

    def test_logic_verify_resolves_json_first(self) -> None:
        raw = (REPO_ROOT / "scripts" / "verify_m1.ps1").read_text(
            "utf-8", errors="replace"
        )
        self.assertIn("llm_model.json", raw)

    # -- no Gemma fallback anywhere active ------------------------------------- #

    def test_logic_retired_scripts_are_gone(self) -> None:
        self.assertFalse(SMOKE_SCRIPT.exists(), "smoke_test_gemma.py must be deleted")
        self.assertFalse(QUALITY_SCRIPT.exists(), "quality_test_gemma.py must be deleted")

    def test_logic_no_gemma_fallback_on_active_surfaces(self) -> None:
        """No active surface may present Gemma as an alternative runtime
        model (the directive: no fallback, no switching back)."""
        offenders: list[str] = []
        for rel in ACTIVE_SURFACES:
            path = REPO_ROOT / rel
            if not path.exists():
                continue
            raw = path.read_text("utf-8", errors="replace").lower()
            if "gemma" in raw:
                offenders.append(rel)
        self.assertEqual(offenders, [], f"active surfaces still mention Gemma: {offenders}")

    def test_logic_no_gemma_model_dir_ships(self) -> None:
        self.assertFalse((REPO_ROOT / "models" / "llm" / "gemma-4-12b").exists())
        self.assertTrue((REPO_ROOT / "models" / "llm" / "qwen3.6-35b-a3b").is_dir())

    def test_logic_banners_report_single_llm(self) -> None:
        bootstrap = (REPO_ROOT / "scripts" / "bootstrap.ps1").read_text(
            "utf-8", errors="replace"
        )
        self.assertIn("az EGYETLEN LLM", bootstrap)
        install = (REPO_ROOT / "scripts" / "install_m1.ps1").read_text(
            "utf-8", errors="replace"
        )
        self.assertIn("az EGYETLEN LLM", install)

    def test_logic_no_openai_cloud_endpoint_in_runtime(self) -> None:
        for rel in ("app/config.py", "app/llm.py", "app/web_server.py",
                    "config/env.local.ps1"):
            raw = (REPO_ROOT / rel).read_text("utf-8", errors="replace")
            self.assertNotIn("api.openai.com", raw)

    # -- the v0.4.16 model picker modules exist --------------------------------- #

    def test_logic_picker_modules_exist(self) -> None:
        for mod in (SETTINGS_MODULE, GGUF_MODULE, PICKER_MODULE):
            self.assertTrue(mod.is_file(), f"missing module {mod}")

    def test_logic_gguf_reader_iq4xs_is_30(self) -> None:
        """IQ4_XS == 30 per llama.cpp llama_ftype (the v0.4.15 finder
        shipped 29 = IQ2_M — corrected in v0.4.16, never change back)."""
        from app.gguf import FILE_TYPE_LABELS, IQ4_XS_FILE_TYPE

        self.assertEqual(IQ4_XS_FILE_TYPE, 30)
        self.assertEqual(FILE_TYPE_LABELS[29], "IQ2_M")
        self.assertEqual(FILE_TYPE_LABELS[30], "IQ4_XS")

    def test_logic_finder_uses_correct_iq4xs(self) -> None:
        raw = (REPO_ROOT / "scripts" / "find_qwen_gguf.ps1").read_text(
            "utf-8", errors="replace"
        )
        self.assertIn("$Iq4XsFileType  = 30", raw)
        self.assertNotIn("$Iq4XsFileType  = 29", raw)

    def test_deep_qwen_status_endpoint_contract(self) -> None:
        """The /api/llm-model surface exists with the full payload keys."""
        source = (REPO_ROOT / "app" / "web_server.py").read_text("utf-8")
        for marker in (
            "api_llm_model",
            "api_llm_model_browse",
            "api_llm_model_inspect",
            "api_llm_model_select",
            "api_llm_model_reset",
            "apply_llm_model_selection",
            "read_llama_server_loaded_model",
        ):
            self.assertIn(marker, source)

    # -- v0.4.17: Ollama blob identification + direct llama-server load ------- #

    def test_logic_identify_script_exists_and_is_ollama_aware(self) -> None:
        """scripts/identify_ollama_blob.ps1: manifest->blob identification,
        GGUF-header identity (never the opaque file name), digest check,
        split verdict, -Select persistence — and C:\AI_HOME only as an
        overridable search HINT (never the permanent model location)."""
        self.assertTrue(IDENTIFY_SCRIPT.is_file(), "identify_ollama_blob.ps1 missing")
        raw = IDENTIFY_SCRIPT.read_text("utf-8", errors="replace")
        for marker in (
            "param(",
            "OllamaRoot",
            "OLLAMA_MODELS",
            "manifests",
            "blobs",
            "sha256-",
            "general.name",
            "general.file_type",
            "general.split.no",
            "general.split.count",
            "IQ4_XS",
            "llm_model.json",
            "LLAMA_MODEL_PATH",
            "Get-FileHash",
            "llama-server",
            "hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
        ):
            self.assertIn(marker, raw, f"identify script must contain {marker!r}")
        # the AI_HOME hint must be OVERRIDABLE (param + env), not baked in
        self.assertIn("[string]$OllamaRoot", raw)
        # IQ4_XS = 30 everywhere (the v0.4.15 lesson; "29 = IQ2_M" in the
        # label table is the CORRECT mapping and contains no "= 29")
        self.assertIn("30 = \"IQ4_XS\"", raw)
        self.assertNotIn("= 29", raw)
        # no Gemma, no cloud, no auto-download (local llama-server queries
        # use Invoke-RestMethod on 127.0.0.1 only)
        self.assertNotIn("gemma", raw.lower())
        self.assertNotIn("api.openai.com", raw)
        self.assertNotIn("Invoke-WebRequest", raw)
        self.assertNotIn("Start-BitsTransfer", raw)

    def test_logic_identify_script_never_copies_the_blob(self) -> None:
        """The 20 GB blob is NEVER copied/renamed/duplicated: selection is
        path-only (llm_model.json); there is no -CopyToRepo switch and no
        Copy-Item of any blob path (the ONLY Copy-Item is the tiny
        env.local.ps1 .bak backup in -SetEnv)."""
        raw = IDENTIFY_SCRIPT.read_text("utf-8", errors="replace")
        self.assertNotIn("CopyToRepo", raw)
        self.assertNotIn("Move-Item", raw)
        self.assertNotIn("Copy-Item -LiteralPath $Main", raw)
        self.assertNotIn("Copy-Item -LiteralPath $b", raw)
        self.assertNotIn("Copy-Item -LiteralPath $p", raw)

    def test_logic_app_code_has_no_hardcoded_ai_home(self) -> None:
        """C:\\AI_HOME may appear ONLY as documentation comments ("the Ollama
        blob store on THIS machine") and in the identify script's
        OVERRIDABLE search-hint param — never as a functional default in
        app code, never as the persisted model location."""
        doc_phrase = "C:" + chr(92) + "AI_HOME" + chr(92) + "models" + chr(92) + "blobs" + chr(92) + "sha256-... blobs"
        for rel in (
            "app/config.py",
            "app/llm.py",
            "app/llm_model_settings.py",
            "app/gguf.py",
            "app/native_picker.py",
            "app/web_server.py",
            "config/env.local.ps1",
            "config/env.local.sh",
            "config/voicemem_config.yaml",
        ):
            raw = (REPO_ROOT / rel).read_text("utf-8", errors="replace")
            raw = raw.replace(doc_phrase, "")
            self.assertNotIn("AI_HOME", raw, f"{rel} hardcodes AI_HOME")

    def test_logic_split_shards_are_rejected_on_select(self) -> None:
        """A gguf-split shard (one Ollama blob of a split set) is refused by
        the selection API — llama-server cannot load it directly."""
        source = (REPO_ROOT / "app" / "web_server.py").read_text("utf-8")
        self.assertIn("is_split_shard", source)
        self.assertIn("llama_server_load_verdict", source)
        gguf = (REPO_ROOT / "app" / "gguf.py").read_text("utf-8")
        self.assertIn("general.split.no", gguf)
        self.assertIn("general.split.count", gguf)

    def test_logic_starter_records_runtime_load_proof(self) -> None:
        """v0.4.17: the starter must PROVE the running server loaded the
        exact file: the server's own log line + /v1/models served id + a
        completion round-trip, all written into the marker the UI reads."""
        raw = (REPO_ROOT / "scripts" / "start_llama_server.ps1").read_text(
            "utf-8", errors="replace"
        )
        for marker in (
            "llama-server.resolved-model.json",
            "served_model_id",
            "log_path_found",
            "completion_ok",
            "chat/completions",
            "Select-String",
        ):
            self.assertIn(marker, raw, f"starter must record {marker!r}")

    def test_logic_picker_lists_ollama_blobs(self) -> None:
        """The native picker's filter must surface Ollama blobs (they have
        no .gguf extension)."""
        raw = (REPO_ROOT / "app" / "native_picker.py").read_text("utf-8")
        self.assertIn("Ollama blobs", raw)
        self.assertIn("sha256-*", raw)


if __name__ == "__main__":
    unittest.main()
