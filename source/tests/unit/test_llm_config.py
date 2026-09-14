"""CENTRALISE LLM CONFIGURATION — the 8-test battery (task requirement).

Pins the contract of ``config/llm_config.yaml`` (the single authoritative
source) and ``app/llm_config.py`` (the ONE loader), against the REAL
shipped files and the REAL captured llama-server startup log
(``docs/llm_config_ngl20-c32768/evidence/server-log-startup-config.txt``):

  T1  the YAML loads correctly (structure, single ``llm:`` section);
  T2  the default production profile is 20 / 32768 (q8_0 / q8_0, 0.7,
      512, thinking off) and the built-in fallback equals the shipped file;
  T3  the llama-server command line is GENERATED from these values
      (``-ngl 20 -c 32768 --parallel 1 --cache-type-k q8_0
      --cache-type-v q8_0 --temp 0.7 --reasoning off``);
  T4  the Python LLM client path (AgentConfig -> app/llm.py) carries the
      SAME values (migration equivalence: no behaviour change);
  T5  thinking stays disabled with BOTH request switches
      (``enable_thinking=false`` + ``reasoning_effort="none"``);
  T6  environment overrides behave exactly as documented (yaml first,
      env second, every override REPORTED, never silent);
  T7  invalid values fail with a NAMED, actionable error (no silent
      fallback);
  T8  the startup-log report + the llama-server log cross-check
      (offloaded 20, n_ctx 32768, n_ctx_slot 32768, K/V q8_0,
      thinking 0) — a drifted log is a MISMATCH, never a silent pass.

The historical drift the negative controls model is real: the target
machine's pre-centralisation starter log ran ``-ngl 26 -c 8192`` from
scattered config copies (see docs/llm_config_ngl20-c32768/README.md).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import yaml

from app.llm_config import (
    DEFAULT_PROFILE,
    ENV_OVERRIDES,
    LlmConfigError,
    LlmRuntimeConfig,
    format_llm_config_summary,
    load_llm_config,
    match_llama_server_log,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
VOICEMEM_YAML = CONFIG_DIR / "voicemem_config.yaml"
LLM_YAML = CONFIG_DIR / "llm_config.yaml"
EVIDENCE_SERVER_LOG = (
    REPO_ROOT / "docs" / "llm_config_ngl20-c32768" / "evidence"
    / "server-log-startup-config.txt"
)

#: The exact production profile the task pins (migration equivalence).
PRODUCTION_PROFILE = {
    "model": "qwen3.6-35b-a3b",
    "gpu_layers": 20,
    "context_size": 32768,
    "parallel": 1,
    "cache_k": "q8_0",
    "cache_v": "q8_0",
    "temperature": 0.7,
    "max_tokens": 512,
    "reasoning_enabled": False,
}

#: The same profile in the NESTED shape config/llm_config.yaml uses
#: (llm.cache.key, llm.sampling.temperature, ...) — invalid-value tests
#: write THIS shape: the loader validates the yaml structure, not the
#: flattened runtime view.
YAML_PROFILE = {
    "model": "qwen3.6-35b-a3b",
    "gpu_layers": 20,
    "context_size": 32768,
    "parallel": 1,
    "cache": {"key": "q8_0", "value": "q8_0"},
    "sampling": {"temperature": 0.7},
    "reasoning": {"enabled": False},
    "generation": {"max_tokens": 512},
}

#: Exact generated command line for the production profile (pinned
#: llama.cpp b10717 flag set; --model/--host/--port are appended by the
#: caller and are deliberately NOT part of the generated list).
PRODUCTION_LLAMA_SERVER_ARGS = [
    "-ngl", "20",
    "-c", "32768",
    "--parallel", "1",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--temp", "0.7",
    "--reasoning", "off",
]

#: Env vars the OLD AgentConfig.apply_env path also reads — tests blank
#: them so os.environ noise can never flip a result.
_AGENTCONFIG_LLAMA_ENV_VARS = (
    "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT", "LLAMA_MODEL_PATH",
    "LLAMA_CONTEXT_SIZE", "LLAMA_N_GPU_LAYERS", "LLAMA_CACHE_TYPE_K",
    "LLAMA_CACHE_TYPE_V", "OPENAI_MODEL", "LLM_DISABLE_THINKING",
    "OPENAI_BASE_URL", "LLM_CONFIG_PATH",
)


def _blank_llm_env() -> dict[str, str]:
    """Values that make every LLM env var *unset* for AgentConfig paths."""
    return {name: "" for name in _AGENTCONFIG_LLAMA_ENV_VARS}


def _load_production() -> LlmRuntimeConfig:
    return load_llm_config(config_dir=CONFIG_DIR, env={})


def _write_llm_yaml(profile: dict) -> Path:
    """Write one llm-only YAML to a temp config dir; returns the dir."""
    tmp = tempfile.mkdtemp(prefix="llm_config_test.")
    (Path(tmp) / "llm_config.yaml").write_text(
        yaml.safe_dump({"llm": profile}, sort_keys=False), encoding="utf-8"
    )
    return Path(tmp)


class Test01_YamlLoadsCorrectly(unittest.TestCase):
    """T1 — the canonical YAML loads through the ONE loader."""

    def test_shipped_file_loads_without_error(self) -> None:
        rt = _load_production()
        self.assertIsInstance(rt, LlmRuntimeConfig)

    def test_loader_points_at_the_shipped_file(self) -> None:
        rt = _load_production()
        self.assertEqual(rt.config_path, LLM_YAML)
        self.assertTrue(rt.config_path.is_file())

    def test_single_llm_section_and_known_keys(self) -> None:
        raw = yaml.safe_load(LLM_YAML.read_text(encoding="utf-8"))
        self.assertIsInstance(raw, dict)
        self.assertEqual(set(raw), {"llm"}, "exactly one top-level key: llm:")
        self.assertEqual(
            set(raw["llm"]),
            {"model", "gpu_layers", "context_size", "parallel", "cache",
             "sampling", "reasoning", "generation"},
        )

    def test_all_values_come_from_yaml_provenance(self) -> None:
        rt = _load_production()
        for key in PRODUCTION_PROFILE:
            self.assertEqual(
                rt.provenance.get(key), "yaml",
                f"{key} must be provenance 'yaml', got {rt.provenance.get(key)!r}",
            )


class Test02_DefaultProductionProfile(unittest.TestCase):
    """T2 — production profile 20 / 32768 / 1 / q8_0 / q8_0 / 0.7 / 512 / off."""

    def test_shipped_profile_is_the_production_profile(self) -> None:
        rt = _load_production()
        for key, expected in PRODUCTION_PROFILE.items():
            self.assertEqual(
                getattr(rt, key), expected,
                f"llm.{key}: expected {expected!r}, got {getattr(rt, key)!r}",
            )

    def test_built_in_default_profile_equals_shipped_file(self) -> None:
        # missing file -> built-in fallback; MUST be identical (pinned)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs("app.llm_config", level="WARNING") as logs:
                rt = load_llm_config(config_dir=Path(tmp), env={})
        for key, expected in PRODUCTION_PROFILE.items():
            self.assertEqual(getattr(rt, key), expected, f"fallback {key}")
        self.assertTrue(any("llm_config.yaml" in m for m in logs.output),
                        "missing file must WARN, not stay silent")

    def test_default_profile_constant_matches_shipped_file(self) -> None:
        rt = _load_production()
        self.assertEqual(rt.model, DEFAULT_PROFILE["model"])
        self.assertEqual(rt.gpu_layers, DEFAULT_PROFILE["gpu_layers"])
        self.assertEqual(rt.context_size, DEFAULT_PROFILE["context_size"])
        self.assertEqual(rt.parallel, DEFAULT_PROFILE["parallel"])
        self.assertEqual(rt.cache_k, DEFAULT_PROFILE["cache"]["key"])
        self.assertEqual(rt.cache_v, DEFAULT_PROFILE["cache"]["value"])
        self.assertEqual(
            rt.temperature, DEFAULT_PROFILE["sampling"]["temperature"]
        )
        self.assertEqual(
            rt.max_tokens, DEFAULT_PROFILE["generation"]["max_tokens"]
        )
        self.assertEqual(
            rt.reasoning_enabled, DEFAULT_PROFILE["reasoning"]["enabled"]
        )

    def test_to_dict_carries_every_field(self) -> None:
        d = _load_production().to_dict()
        for key, expected in PRODUCTION_PROFILE.items():
            self.assertEqual(d[key], expected)
        self.assertEqual(d["llama_server_args"], PRODUCTION_LLAMA_SERVER_ARGS)
        self.assertEqual(d["reasoning_server_arg"], "off")
        self.assertIn("provenance", d)


class Test03_LlamaServerCommandGeneration(unittest.TestCase):
    """T3 — the command line is GENERATED from the config values."""

    def test_production_command_line(self) -> None:
        self.assertEqual(_load_production().llama_server_args(),
                         PRODUCTION_LLAMA_SERVER_ARGS)

    def test_args_follow_values_not_hardcoded(self) -> None:
        rt = load_llm_config(
            config_dir=CONFIG_DIR,
            env={
                "LLAMA_N_GPU_LAYERS": "10",
                "LLAMA_CONTEXT_SIZE": "4096",
                "LLAMA_PARALLEL": "2",
                "LLAMA_CACHE_TYPE_K": "q5_1",
                "LLAMA_CACHE_TYPE_V": "q4_0",
                "LLM_TEMPERATURE": "0.3",
            },
        )
        args = rt.llama_server_args()
        for flag, value in (("-ngl", "10"), ("-c", "4096"),
                            ("--parallel", "2"), ("--cache-type-k", "q5_1"),
                            ("--cache-type-v", "q4_0"), ("--temp", "0.3")):
            i = args.index(flag)
            self.assertEqual(args[i + 1], value)

    def test_reasoning_enabled_omits_reasoning_flag(self) -> None:
        rt = load_llm_config(
            config_dir=CONFIG_DIR, env={"LLM_DISABLE_THINKING": "0"}
        )
        self.assertEqual(rt.reasoning_server_arg, "")
        self.assertNotIn("--reasoning", rt.llama_server_args())

    def test_production_reasoning_flag_is_off(self) -> None:
        self.assertEqual(_load_production().reasoning_server_arg, "off")


class Test04_PythonClientSameValues(unittest.TestCase):
    """T4 — the Python LLM client path carries the SAME values.

    AgentConfig (consumed by app/llm.py: temperature, max_tokens and the
    thinking kwargs come from its fields) must agree with the canonical
    file on EVERY value — the migration-equivalence guarantee.
    """

    def test_agentconfig_agrees_with_canonical_file(self) -> None:
        from app.config import AgentConfig

        rt = _load_production()
        with patch.dict("os.environ", _blank_llm_env(), clear=False):
            cfg = AgentConfig.from_yaml(VOICEMEM_YAML)
        self.assertEqual(cfg.llm_model_name, rt.model)
        self.assertEqual(cfg.llm_n_gpu_layers, rt.gpu_layers)
        self.assertEqual(cfg.llm_context_size, rt.context_size)
        self.assertEqual(cfg.llm_parallel, rt.parallel)
        self.assertEqual(cfg.llm_cache_type_k, rt.cache_k)
        self.assertEqual(cfg.llm_cache_type_v, rt.cache_v)
        self.assertEqual(cfg.llm_temperature, rt.temperature)
        self.assertEqual(cfg.llm_max_tokens, rt.max_tokens)
        self.assertEqual(cfg.llm_disable_thinking, not rt.reasoning_enabled)

    def test_from_yaml_materialises_the_canonical_runtime(self) -> None:
        # v0.7.2 wiring: from_yaml does not merely AGREE with the canonical
        # file — it materialises the fields FROM it (cfg.llm_runtime is the
        # LlmRuntimeConfig produced by the ONE loader; the provenance and
        # the generated llama-server args ride along).
        from app.config import AgentConfig

        rt = _load_production()
        with patch.dict("os.environ", _blank_llm_env(), clear=False):
            cfg = AgentConfig.from_yaml(VOICEMEM_YAML)
        self.assertIsNotNone(cfg.llm_runtime)
        self.assertEqual(cfg.llm_runtime.llama_server_args(), rt.llama_server_args())
        self.assertEqual(cfg.llm_runtime.provenance, rt.provenance)
        self.assertIsNotNone(cfg.llm_runtime.config_path)
        self.assertTrue(cfg.llm_config_summary().startswith("LLM configuration:"))

    def test_canonical_values_beat_voicemem_yaml_duplicates(self) -> None:
        # migration guarantee: a voicemem_config.yaml carrying STALE llm
        # value keys CANNOT win — materialise_llm_runtime runs last and
        # re-imposes the canonical values (the duplicate keys were removed
        # from the shipped file; this pins the precedence for anyone who
        # re-adds them locally).
        import tempfile

        from app.config import AgentConfig

        rt = _load_production()
        content = "\n".join(
            [
                "app:",
                f"  llm_context_size: {rt.context_size * 2}",
                f"  llm_n_gpu_layers: {rt.gpu_layers + 7}",
                "  llm_temperature: 1.9",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "voicemem_config.yaml"
            stale.write_text(content, encoding="utf-8")
            with patch.dict("os.environ", _blank_llm_env(), clear=False):
                cfg = AgentConfig.from_yaml(stale)
        # the temp dir has no llm_config.yaml sibling -> the loader falls
        # back to <root>/config/llm_config.yaml (the repo canonical file)
        self.assertEqual(cfg.llm_context_size, rt.context_size)
        self.assertEqual(cfg.llm_n_gpu_layers, rt.gpu_layers)
        self.assertEqual(cfg.llm_temperature, rt.temperature)

    def test_load_config_helpers_materialise_runtime(self) -> None:
        # main.load_config / web_server.load_config call
        # materialise_llm_runtime on the no-yaml branch too — the canonical
        # file is the source even when voicemem_config.yaml is absent.
        from app.config import AgentConfig

        rt = _load_production()
        with patch.dict("os.environ", _blank_llm_env(), clear=False):
            cfg = AgentConfig()
            cfg.apply_env()
            cfg.materialise_llm_runtime()
        self.assertIsNotNone(cfg.llm_runtime)
        self.assertEqual(cfg.llm_temperature, rt.temperature)
        self.assertEqual(cfg.llm_n_gpu_layers, rt.gpu_layers)

    def test_request_payload_kwargs_agree(self) -> None:
        # app/llm.py builds request payloads from AgentConfig fields:
        # temperature/max_tokens (pinned above) + the thinking kwargs via
        # _thinking_control_kwargs — must equal the canonical kwargs.
        from app.config import AgentConfig
        from app.llm import _thinking_control_kwargs

        rt = _load_production()
        with patch.dict("os.environ", _blank_llm_env(), clear=False):
            cfg = AgentConfig.from_yaml(VOICEMEM_YAML)
        self.assertEqual(_thinking_control_kwargs(cfg),
                         rt.thinking_control_kwargs())

    def test_thinking_kwargs_come_from_the_runtime_single_source(self) -> None:
        # v0.7.2: _thinking_control_kwargs delegates to the canonical
        # runtime's thinking_control_kwargs() when present — the yaml flag,
        # the server-side --reasoning flag and the request kwargs are
        # generated from ONE definition and can never disagree.
        from app.config import AgentConfig
        from app.llm import _thinking_control_kwargs

        rt = _load_production()
        with patch.dict("os.environ", _blank_llm_env(), clear=False):
            cfg = AgentConfig.from_yaml(VOICEMEM_YAML)
        self.assertIsNotNone(cfg.llm_runtime)
        # the runtime IS the source of the kwargs
        self.assertEqual(_thinking_control_kwargs(cfg),
                         cfg.llm_runtime.thinking_control_kwargs())
        # enabling thinking in the runtime removes the suppression
        cfg.llm_runtime = load_llm_config(
            config_dir=CONFIG_DIR, env={"LLM_DISABLE_THINKING": "0"}
        )
        self.assertEqual(_thinking_control_kwargs(cfg), {})


class Test05_ThinkingStaysDisabled(unittest.TestCase):
    """T5 — thinking disabled keeps BOTH suppression switches."""

    def test_shipped_file_disables_thinking(self) -> None:
        rt = _load_production()
        self.assertFalse(rt.reasoning_enabled)

    def test_both_request_switches_present(self) -> None:
        kwargs = _load_production().thinking_control_kwargs()
        self.assertEqual(
            kwargs,
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none",
            },
        )

    def test_enabled_thinking_removes_suppression(self) -> None:
        rt = load_llm_config(
            config_dir=CONFIG_DIR, env={"LLM_DISABLE_THINKING": "0"}
        )
        self.assertTrue(rt.reasoning_enabled)
        self.assertEqual(rt.thinking_control_kwargs(), {})

    def test_server_and_request_flags_can_never_disagree(self) -> None:
        # one flag -> three mechanisms, generated together by the loader
        rt = _load_production()
        self.assertIn(["--reasoning", "off"], [rt.llama_server_args()[i:i + 2]
                        for i in range(len(rt.llama_server_args()))])
        self.assertFalse(
            rt.thinking_control_kwargs()["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertEqual(rt.thinking_control_kwargs()["reasoning_effort"], "none")


class Test06_EnvOverridesDocumentedBehaviour(unittest.TestCase):
    """T6 — env overrides: optional, reported, never a second truth source."""

    def test_documented_override_set_is_pinned(self) -> None:
        self.assertEqual(set(ENV_OVERRIDES), {
            "LLAMA_N_GPU_LAYERS", "LLAMA_CONTEXT_SIZE", "LLAMA_PARALLEL",
            "LLAMA_CACHE_TYPE_K", "LLAMA_CACHE_TYPE_V", "LLM_TEMPERATURE",
            "LLM_MAX_TOKENS", "LLM_DISABLE_THINKING", "OPENAI_MODEL",
        })

    def test_every_override_takes_effect_and_is_recorded(self) -> None:
        cases = [
            ("LLAMA_N_GPU_LAYERS", "10", "gpu_layers", 10),
            ("LLAMA_CONTEXT_SIZE", "4096", "context_size", 4096),
            ("LLAMA_PARALLEL", "2", "parallel", 2),
            ("LLAMA_CACHE_TYPE_K", "q5_1", "cache_k", "q5_1"),
            ("LLAMA_CACHE_TYPE_V", "q4_0", "cache_v", "q4_0"),
            ("LLM_TEMPERATURE", "0.3", "temperature", 0.3),
            ("LLM_MAX_TOKENS", "256", "max_tokens", 256),
            ("OPENAI_MODEL", "other-model", "model", "other-model"),
        ]
        for name, value, field, expected in cases:
            with self.subTest(env=name):
                rt = load_llm_config(
                    config_dir=CONFIG_DIR, env={name: value}
                )
                self.assertEqual(getattr(rt, field), expected)
                self.assertEqual(rt.provenance[field], f"env:{name}")
        for name, value, field, expected in cases:
            rt = load_llm_config(config_dir=CONFIG_DIR, env={name: value})
            # non-overridden fields keep the yaml values
            if field != "gpu_layers":
                self.assertEqual(rt.gpu_layers, 20)
            if field != "context_size":
                self.assertEqual(rt.context_size, 32768)
            if field != "max_tokens":
                self.assertEqual(rt.max_tokens, 512)

    def test_llm_disable_thinking_inverted_semantics(self) -> None:
        # the env var DISABLES thinking (v1-spec semantics); 1 -> off
        rt = load_llm_config(
            config_dir=CONFIG_DIR, env={"LLM_DISABLE_THINKING": "1"}
        )
        self.assertFalse(rt.reasoning_enabled)
        rt = load_llm_config(
            config_dir=CONFIG_DIR, env={"LLM_DISABLE_THINKING": "0"}
        )
        self.assertTrue(rt.reasoning_enabled)

    def test_empty_env_value_is_no_override(self) -> None:
        rt = load_llm_config(
            config_dir=CONFIG_DIR, env={"LLAMA_N_GPU_LAYERS": ""}
        )
        self.assertEqual(rt.gpu_layers, 20)
        self.assertEqual(rt.provenance["gpu_layers"], "yaml")

    def test_override_is_reported_at_startup_never_silent(self) -> None:
        rt = load_llm_config(
            config_dir=CONFIG_DIR, env={"LLAMA_N_GPU_LAYERS": "10"}
        )
        summary = format_llm_config_summary(rt)
        self.assertIn("env overrides active", summary)
        self.assertIn("gpu_layers <- env:LLAMA_N_GPU_LAYERS", summary)

    def test_pure_yaml_summary_has_no_override_line(self) -> None:
        summary = format_llm_config_summary(_load_production())
        self.assertNotIn("env overrides active", summary)

    def test_bad_env_value_fails_with_named_error(self) -> None:
        with self.assertRaises(LlmConfigError) as ctx:
            load_llm_config(
                config_dir=CONFIG_DIR, env={"LLAMA_N_GPU_LAYERS": "twenty"}
            )
        self.assertIn("LLAMA_N_GPU_LAYERS", str(ctx.exception))

    def test_env_cannot_disable_validation(self) -> None:
        # an override is validated exactly like a yaml value
        with self.assertRaises(LlmConfigError) as ctx:
            load_llm_config(
                config_dir=CONFIG_DIR, env={"LLAMA_CONTEXT_SIZE": "0"}
            )
        self.assertIn("context_size", str(ctx.exception))


class Test07_InvalidConfigFailsClearly(unittest.TestCase):
    """T7 — invalid values: NAMED error, no silent fallback, no start."""

    def _assert_invalid(self, profile: dict, needle: str) -> None:
        cfg_dir = _write_llm_yaml(profile)
        with self.assertRaises(LlmConfigError) as ctx:
            load_llm_config(config_dir=cfg_dir, env={})
        self.assertIn(needle, str(ctx.exception),
                      f"error must name {needle!r}: {ctx.exception}")

    def test_negative_gpu_layers(self) -> None:
        self._assert_invalid({**YAML_PROFILE, "gpu_layers": -1},
                             "gpu_layers")

    def test_non_integer_gpu_layers(self) -> None:
        self._assert_invalid({**YAML_PROFILE, "gpu_layers": "20"},
                             "gpu_layers")

    def test_zero_context_size(self) -> None:
        self._assert_invalid({**YAML_PROFILE, "context_size": 0},
                             "context_size")

    def test_zero_parallel(self) -> None:
        self._assert_invalid({**YAML_PROFILE, "parallel": 0},
                             "parallel")

    def test_unsupported_cache_type(self) -> None:
        self._assert_invalid(
            {**YAML_PROFILE, "cache": {"key": "q3_0", "value": "q8_0"}},
            "cache.key",
        )

    def test_out_of_range_temperature(self) -> None:
        self._assert_invalid(
            {**YAML_PROFILE, "sampling": {"temperature": 5.0}},
            "temperature",
        )

    def test_non_numeric_temperature(self) -> None:
        self._assert_invalid(
            {**YAML_PROFILE, "sampling": {"temperature": "hot"}},
            "temperature",
        )

    def test_zero_max_tokens(self) -> None:
        self._assert_invalid(
            {**YAML_PROFILE, "generation": {"max_tokens": 0}},
            "max_tokens",
        )

    def test_empty_model(self) -> None:
        self._assert_invalid({**YAML_PROFILE, "model": ""}, "model")

    def test_non_boolean_reasoning_flag(self) -> None:
        self._assert_invalid(
            {**YAML_PROFILE, "reasoning": {"enabled": "nope"}},
            "reasoning.enabled",
        )

    def test_unknown_llm_key_rejected(self) -> None:
        self._assert_invalid(
            {**YAML_PROFILE, "gpu_layerz": 20}, "unknown llm key"
        )

    def test_non_mapping_root_rejected(self) -> None:
        tmp = tempfile.mkdtemp(prefix="llm_config_test.")
        (Path(tmp) / "llm_config.yaml").write_text("- a\n- b\n", encoding="utf-8")
        with self.assertRaises(LlmConfigError):
            load_llm_config(config_dir=Path(tmp), env={})

    def test_error_names_the_fix_location(self) -> None:
        cfg_dir = _write_llm_yaml({**YAML_PROFILE, "parallel": 0})
        with self.assertRaises(LlmConfigError) as ctx:
            load_llm_config(config_dir=cfg_dir, env={})
        msg = str(ctx.exception)
        self.assertIn("llm_config.yaml", msg)
        self.assertIn("llm.parallel", msg)


class Test08_StartupLogReportAndServerCrossCheck(unittest.TestCase):
    """T8 — startup reporting + llama-server log cross-check."""

    def test_summary_lines_exact(self) -> None:
        self.assertEqual(_load_production().summary_lines(), [
            "model = qwen3.6-35b-a3b",
            "gpu_layers = 20",
            "context_size = 32768",
            "parallel = 1",
            "cache_k = q8_0",
            "cache_v = q8_0",
            "temperature = 0.7",
            "max_tokens = 512",
            "thinking = false",
            "reasoning_effort = none",
        ])

    def test_full_summary_block(self) -> None:
        block = format_llm_config_summary(_load_production())
        self.assertTrue(block.startswith("LLM configuration:"))
        self.assertIn(f"source = {LLM_YAML}", block)

    def test_real_server_evidence_log_matches(self) -> None:
        # the pinned llama.cpp b10717 log captured with the production
        # command line (offloaded 20, n_ctx 32768, n_ctx_slot 32768)
        self.assertTrue(EVIDENCE_SERVER_LOG.is_file(),
                        "captured server log evidence must ship with the repo")
        text = EVIDENCE_SERVER_LOG.read_text(encoding="utf-8", errors="replace")
        verdict = match_llama_server_log(text, _load_production())
        items = verdict["items"]
        self.assertEqual(items["offloaded_layers"]["actual"], 20)
        self.assertEqual(items["n_ctx"]["actual"], 32768)
        self.assertEqual(items["n_ctx_slot"]["actual"], 32768)
        self.assertEqual(items["cache_k"]["actual"], "q8_0")
        self.assertEqual(items["cache_v"]["actual"], "q8_0")
        self.assertEqual(items["thinking"]["actual"], 0)
        self.assertTrue(verdict["all_match"],
                        f"cross-check failed: { {k: v for k, v in items.items() if not v['match']} }")

    def test_drifted_log_is_a_mismatch(self) -> None:
        # models the REAL pre-centralisation drift: -ngl 26 -c 8192
        drifted = (
            "load_tensors: offloaded 26/29 layers to GPU\n"
            "llama_context: n_ctx                 = 8192\n"
            "srv    load_model: initializing, n_slots = 1, n_ctx_slot = 8192\n"
            "llama_kv_cache: K (q8_0):  952.00 MiB, V (q8_0):  952.00 MiB\n"
            "srv          init: init: chat template, thinking = 0\n"
        )
        verdict = match_llama_server_log(drifted, _load_production())
        self.assertFalse(verdict["all_match"])
        self.assertFalse(verdict["items"]["offloaded_layers"]["match"])
        self.assertFalse(verdict["items"]["n_ctx"]["match"])
        self.assertFalse(verdict["items"]["n_ctx_slot"]["match"])

    def test_missing_log_lines_are_never_a_silent_pass(self) -> None:
        # fail-closed: lines the server did not print make the overall
        # verdict FALSE (a non-pass), and each item's match is None
        # (= "not verifiable"). An unknown log can never yield a pass.
        verdict = match_llama_server_log("", _load_production())
        self.assertFalse(verdict["all_match"])
        for item in verdict["items"].values():
            self.assertIsNone(item["match"])


if __name__ == "__main__":
    unittest.main()
