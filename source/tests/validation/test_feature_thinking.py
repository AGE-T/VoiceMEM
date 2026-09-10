"""v0.4.4 field-report fix contract: thinking suppression + ASR + E5.

The v0.4.3 field report ("Már a szöveges formátumra sem válaszol a rendszer.
ASR nem zöld a ui-on!") had THREE root causes, each with a pinned fix:

  1. LLM 0-char replies — llama.cpp b10717 defaults ``enable_thinking`` to
     TRUE; the Gemma 4 chat template leaves the thought channel OPEN at
     the generation prompt; the model answers INSIDE the thinking channel;
     the server routes it to ``reasoning_content``; with the 512-token
     budget consumed by thinking, ``content`` is empty on every turn.
     FIX LAYERS (all request/server level, verified live against b10717):
       a) app/llm.py sends chat_template_kwargs + reasoning_effort "none";
       b) scripts/start_llama_server.ps1 starts llama-server with
          ``--reasoning off`` (protects voicemem's own OpenAI-lib calls);
       c) the web backend verifies an actual reply at startup
          (``_verify_llm_reply``) — "/health 200 but empty content" is a
          VISIBLE LLM ERROR, never a silently green chip;
       d) chat_stream/chat_json raise on empty content (with the thinking
          diagnosis when reasoning_content was present).
  2. ASR never loads — the vendored voicemem package pins
     ``transformers==4.52.3`` which does NOT know the ``qwen3_asr``
     architecture (needs >= 5.0: the native qwen3_asr module). FIX:
     requirements/pyproject floor is 5.0 AND the installer runs a
     transformers guard step (16) that reinstalls/validates >= 5.0 as the
     LAST pip step.
  3. Memory/embedding ERROR on every turn — the runtime offline env
     (by design) plus voicemem's flat-layout-only E5 resolver fell back
     to the HF repo id. FIX: ``VOICEMEM_E5_MODEL`` pins the local
     ``models/embedding/multilingual-e5-small`` directory (env.local.ps1 +
     app-side ``pin_e5_local_model`` from both construction paths).

This file validates the whole contract against the real repo state
(fully hermetic: no network, no GPU, no models).
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "app"
SCRIPTS = REPO_ROOT / "scripts"
CONFIG = REPO_ROOT / "config"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_ascii(path: Path) -> str:
    data = path.read_bytes()
    assert max(data) <= 0x7F, f"{path} is not pure ASCII"
    assert data.replace(b"\r\n", b"").count(b"\n") == 0, f"{path} is not pure CRLF"
    return data.decode("ascii")


class ThinkingSuppressionContractTests(FeatureValidationTest):
    """Layer-by-layer: the 0-char reply outage can never ship silently."""

    def test_llm_client_sends_request_level_suppression(self) -> None:
        code = _read(APP / "llm.py")
        self.assertIn("_thinking_control_kwargs", code)
        self.assertIn('"reasoning_effort": "none"', code)
        self.assertIn('{"enable_thinking": False}', code)
        self.assertIn("parse_sse_reasoning_delta", code)

    def test_llm_client_raises_on_empty_content(self) -> None:
        code = _read(APP / "llm.py")
        self.assertIn("LLM reply was empty", code)
        self.assertIn("reasoning (thinking)", code)

    def test_config_default_disables_thinking(self) -> None:
        from app.config import AgentConfig

        self.assertIs(AgentConfig().llm_disable_thinking, True)

    def test_yaml_carries_the_flag(self) -> None:
        self.assertIn("llm_disable_thinking: true", _read(CONFIG / "voicemem_config.yaml"))

    def test_env_example_carries_the_flag(self) -> None:
        env_example = _read(CONFIG / ".env.example")
        self.assertIn("LLM_DISABLE_THINKING=1", env_example)

    def test_starter_starts_server_with_reasoning_off(self) -> None:
        """The server-side default protects clients that cannot send kwargs
        (the voicemem package's own OpenAI-lib calls)."""
        starter = _read_ascii(SCRIPTS / "start_llama_server.ps1")
        self.assertIn('"--reasoning", "off"', starter)
        # the pinned v0.4.3 flag set is still intact (config-driven values)
        for flag in ('"--parallel", [string]$Cfg.parallel',
                     '"--cache-type-k", [string]$Cfg.ck_k',
                     '"--cache-type-v", [string]$Cfg.ck_v',
                     '"--temp", [string]$Cfg.temp',
                     '"--metrics"', '"--no-webui"'):
            self.assertIn(flag, starter)

    def test_llm_client_sends_kwargs_and_handles_reasoning_replies(self) -> None:
        """v0.4.16: the Gemma smoke script was deleted (no fallback model);
        the thinking-suppression CONTRACT now lives in app/llm.py — the
        runtime client every /v1 call goes through."""
        llm = _read(APP / "llm.py")
        self.assertIn("_thinking_control_kwargs", llm)
        self.assertIn("enable_thinking", llm)
        self.assertIn("reasoning_effort", llm)
        # a reasoning-only reply (empty content) must raise with a diagnosis
        self.assertIn("parse_sse_reasoning_delta", llm)
        self.assertIn("reasoning_content", llm)
        self.assertIn("LLM reply was empty", llm)
        # the server-side default flag is pinned in the starter
        starter = _read_ascii(SCRIPTS / "start_llama_server.ps1")
        self.assertIn('"--reasoning", "off"', starter)

    def test_verify_m1_bodies_suppress_thinking(self) -> None:
        verify = _read_ascii(SCRIPTS / "verify_m1.ps1")
        self.assertIn('"chat_template_kwargs":{"enable_thinking":false}', verify)
        self.assertIn('"reasoning_effort":"none"', verify)
        self.assertEqual(verify.count('"reasoning_effort":"none"'), 3)  # chat + json + schema

    def test_web_backend_probes_an_actual_reply(self) -> None:
        code = _read(APP / "web_server.py")
        self.assertIn("_verify_llm_reply", code)
        self.assertIn("reply verified", code)
        self.assertIn("LLM startup probe", code)
        # the probe never overwrites an error with a green chip
        self.assertIn("comp.state not in (_STATE_ERROR, _STATE_PROCESSING)", code)

    def test_startup_probe_timeout_is_bounded(self) -> None:
        code = _read(APP / "web_server.py")
        self.assertIn("asyncio.wait_for(_collect(), timeout=60.0)", code)


class ThinkingAlignmentV0421ContractTests(FeatureValidationTest):
    """v0.4.21: BOTH layers explicitly disable reasoning — server flag AND
    application request payload — for the Qwen3.6 35B A3B IQ4_XS production
    model on llama-server b10717 (verified live: bare default = thinking with
    0-char content; --reasoning off OR request kwargs = immediate content,
    0 reasoning chunks; the Qwen template pre-closes the thought channel:
    'imd\\n\\nthink\\n\\n' in the server log's generation_prompt).

    Every request-construction site in the application must carry the
    switches (never rely on the system prompt, never on Ollama /set nothink
    or a Modelfile PARAMETER think false — no Ollama in the runtime)."""

    def test_localise_memories_carries_the_switches(self) -> None:
        """The stdlib-urllib translator builds its own raw payload — the
        switches must be present there too (not only in app/llm.py)."""
        code = _read(SCRIPTS / "localise_memories.py")
        self.assertIn('"chat_template_kwargs": {"enable_thinking": False}', code)
        self.assertIn('"reasoning_effort": "none"', code)

    def test_smoke_test_round_trip_mirrors_production(self) -> None:
        smoke = _read(SCRIPTS / "smoke_test_web.py")
        self.assertIn('"chat_template_kwargs": {"enable_thinking": False}', smoke)
        self.assertIn('"reasoning_effort": "none"', smoke)
        # a reply WITH reasoning_content now fails the smoke check
        self.assertIn("reasoning_content", smoke)
        self.assertIn("and not reasoning", smoke)

    def test_benchmark_verifies_the_no_thinking_profile(self) -> None:
        bench = _read(REPO_ROOT / "tests" / "benchmark" / "llm_benchmark.py")
        # per-turn thinking-block detection + the Ollama nothink comparison
        self.assertIn("THINKING_TAG_MARKERS", bench)
        self.assertIn("reply_has_thinking_block", bench)
        self.assertIn("thinking not disabled", bench)
        self.assertIn("reasoning_eaten_errors", bench)
        self.assertIn("empty_replies", bench)
        self.assertIn("first_token_s", bench)
        # the wire capture proves the request switches during the run
        self.assertIn("_capture_request", bench)
        self.assertIn("NO-THINKING VERDICT", bench)
        self.assertIn("/set nothink", bench)

    def test_llm_client_both_switches_on_both_paths(self) -> None:
        """chat_stream AND chat_json each merge _thinking_control_kwargs."""
        code = _read(APP / "llm.py")
        self.assertEqual(code.count("**_thinking_control_kwargs(self._config)"), 2)

    def test_modeles_lock_pins_the_verified_server(self) -> None:
        """The exact llama-server whose reasoning API was verified: b10717
        (build 10717, commit a32af33de) — -rea/--reasoning [on|off|auto],
        request fields chat_template_kwargs.enable_thinking (bool) and
        reasoning_effort 'none' (mapped to disable-reasoning in
        server-common.cpp)."""
        lock = _read(REPO_ROOT / "MODELS.lock.json")
        self.assertIn('"tag": "b10717"', lock)
        self.assertIn("llama-b10717-bin-win-cuda-13.3-x64.zip", lock)

    def test_no_ollama_runtime_dependency(self) -> None:
        """No Ollama /set nothink, no Modelfile PARAMETER think, no Ollama
        API/serve runtime dependency in any request-construction site.
        (Mentions of "Ollama blob" as a GGUF file format and comments that
        say there is NO Ollama runtime are FINE — llama-server loads the blob
        in place; Ollama itself never runs.)"""
        banned = (
            (re.compile(r"/set\s+nothink", re.I), "/set nothink"),
            (re.compile(r"PARAMETER\s+think", re.I), "Modelfile PARAMETER think"),
            (re.compile(r"ollama\s+(serve|pull|run|list|stop|show)\b", re.I),
             "an Ollama CLI invocation"),
            (re.compile(r":11434\b"), "the Ollama API port"),
        )
        for path in (APP / "llm.py", APP / "web_server.py",
                     SCRIPTS / "localise_memories.py",
                     SCRIPTS / "smoke_test_web.py",
                     SCRIPTS / "start_llama_server.ps1",
                     SCRIPTS / "verify_m1.ps1"):
            text = _read(path)
            for pattern, label in banned:
                self.assertIsNone(
                    pattern.search(text),
                    f"{path} must not contain {label}",
                )


class TransformersFloorContractTests(FeatureValidationTest):
    """ASR (Qwen3-ASR-0.6B) requires transformers >= 5.0 — pinned everywhere.

    v0.4.7: the floor moved from 4.57 to 5.0 — the qwen3_asr module (model +
    processor for the processor+generate call path in app/asr.py) exists
    natively from transformers 5.x; no 4.x release contains it."""

    def test_requirements_floor(self) -> None:
        self.assertIn("transformers>=5.0", _read(REPO_ROOT / "requirements.txt"))

    def test_pyproject_floor(self) -> None:
        self.assertIn("transformers>=5.0", _read(REPO_ROOT / "pyproject.toml"))

    def test_lock_note_explains_the_voicemem_pin(self) -> None:
        lock = _read(REPO_ROOT / "requirements.lock")
        self.assertIn("transformers>=5.0", lock)
        self.assertIn("4.52.3", lock)  # the voicemem pin that caused the outage

    def test_installer_guard_step(self) -> None:
        installer = _read_ascii(SCRIPTS / "install_m1.ps1")
        self.assertIn("# 16) transformers OR", installer)
        self.assertIn('pip install "transformers>=$TransformersFloor"', installer)
        self.assertIn('$TransformersFloor = "5.0"', installer)
        # tuple-compare assert (string compare would rank "4.9" < "5.0")
        self.assertIn("v >= (5, 0)", installer)

    def test_installer_guard_is_the_last_pip_step(self) -> None:
        """The guard must run AFTER voicemem/funasr/speechbrain (any of them
        can downgrade transformers); the models download follows it."""
        installer = _read_ascii(SCRIPTS / "install_m1.ps1")
        guard_pos = installer.index("# 16) transformers OR")
        speechbrain_pos = installer.index("# 15) speechbrain telepitese")
        models_pos = installer.index("# 17) Modellek letoltese")
        self.assertLess(speechbrain_pos, guard_pos)
        self.assertLess(guard_pos, models_pos)

    def test_asr_hint_names_the_floor(self) -> None:
        code = _read(APP / "asr.py")
        self.assertIn(">= 5.0", code)


class E5LocalModelContractTests(FeatureValidationTest):
    """The E5 embedder must resolve to the LOCAL offline copy."""

    def test_env_local_exports_e5_model(self) -> None:
        env_local = _read_ascii(CONFIG / "env.local.ps1")
        self.assertIn("VOICEMEM_E5_MODEL", env_local)
        self.assertIn("models\\embedding\\multilingual-e5-small", env_local)

    def test_env_example_exports_e5_model(self) -> None:
        self.assertIn("VOICEMEM_E5_MODEL", _read(CONFIG / ".env.example"))

    def test_bridge_pins_the_local_model(self) -> None:
        code = _read(APP / "voicemem_bridge.py")
        self.assertIn("def pin_e5_local_model", code)
        self.assertIn("VOICEMEM_E5_MODEL", code)

    def test_web_memory_layer_pins_the_local_model(self) -> None:
        code = _read(APP / "web_server.py")
        self.assertIn("pin_e5_local_model", code)


if __name__ == "__main__":
    unittest.main()
