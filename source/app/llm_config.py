"""Canonical LLM runtime configuration — the ONE loader (v0.7.2).

``config/llm_config.yaml`` is the single operator-editable source for
every production LLM runtime setting. This module is the ONLY code that
parses it; every subsystem consumes the typed result:

  * llama-server startup — ``scripts/start_llama_server.ps1`` and
    ``scripts/verify_m1.ps1`` call the python-bridge which imports THIS
    module (through ``AgentConfig.from_yaml``) and emits the resolved
    values as JSON; the command line (``-ngl``, ``-c``, ``--parallel``,
    ``--cache-type-k/-v``, ``--temp``, ``--reasoning``) is GENERATED from
    it — no PowerShell file carries a second copy of the values;
  * the Python LLM client — ``app/llm.py`` reads the same values through
    ``AgentConfig`` fields materialised by ``AgentConfig.from_yaml``;
  * startup diagnostics — ``app/main.py`` and ``app/web_server.py`` print
    :meth:`LlmRuntimeConfig.summary_lines`;
  * configuration validation — :func:`load_llm_config` fails with a
    named, actionable :class:`LlmConfigError` on any invalid value.

PRECEDENCE (strongest first, documented once — here):
  1. ``config/llm_config.yaml`` (the canonical file);
  2. optional environment overrides — the exact V1-spec 18.1 names
     (``LLAMA_N_GPU_LAYERS``, ``LLAMA_CONTEXT_SIZE``, ``LLAMA_PARALLEL``,
     ``LLAMA_CACHE_TYPE_K``, ``LLAMA_CACHE_TYPE_V``, ``LLM_TEMPERATURE``,
     ``LLM_MAX_TOKENS``, ``LLM_DISABLE_THINKING``, ``OPENAI_MODEL``);
     each active override is recorded in ``provenance`` and reported at
     startup, so the environment can never silently become a second
     source of truth;
  3. :data:`DEFAULT_PROFILE` built-in defaults — identical to the
     shipped file (pinned by
     ``tests/unit/test_llm_config.py::test_default_profile_matches_shipped_file``),
     used only when the file is missing entirely, with a WARNING.

The historical state (pre-v0.7.2) this replaces: production values lived
in THREE operator-reachable places at once — ``config/voicemem_config.yaml``
(``app:`` flat keys), ``config/env.local.ps1`` (active ``$env:`` assignments
that silently overrode the YAML), and hardcoded shell fallbacks in
``scripts/start_llama_server.ps1`` (``$FbNgl = "20"``, ``$FbCtx = "32768"``...).
All three duplicated copies are REMOVED; the effective values are
unchanged (20 / 32768 / 1 / q8_0 / q8_0 / 0.7 / 512 / thinking off).

Pure stdlib + PyYAML — importable anywhere ``app.config`` is.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import yaml

    _HAS_YAML = True
except ImportError:  # pragma: no cover - PyYAML missing on a bare machine
    _HAS_YAML = False

logger = logging.getLogger(__name__)

#: Canonical file name, next to voicemem_config.yaml in config/.
LLM_CONFIG_FILENAME = "llm_config.yaml"

#: Env var that redirects the canonical file (tests / exotic installs).
LLM_CONFIG_PATH_ENV = "LLM_CONFIG_PATH"

#: llama.cpp b10717-supported KV cache quant types (--cache-type-k/-v).
SUPPORTED_KV_CACHE_TYPES = frozenset(
    {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}
)

#: Built-in fallback profile — MUST equal the shipped config/llm_config.yaml
#: (pinned by tests). Used only when the file is missing entirely.
DEFAULT_PROFILE: dict[str, Any] = {
    "model": "qwen3.6-35b-a3b",
    "gpu_layers": 20,
    "context_size": 32768,
    "parallel": 1,
    "cache": {"key": "q8_0", "value": "q8_0"},
    "sampling": {"temperature": 0.7},
    "reasoning": {"enabled": False},
    "generation": {"max_tokens": 512},
}

#: Every environment variable the loader honours, with the field it
#: overrides (documented precedence; all optional).
ENV_OVERRIDES: dict[str, str] = {
    "LLAMA_N_GPU_LAYERS": "gpu_layers",
    "LLAMA_CONTEXT_SIZE": "context_size",
    "LLAMA_PARALLEL": "parallel",
    "LLAMA_CACHE_TYPE_K": "cache.key",
    "LLAMA_CACHE_TYPE_V": "cache.value",
    "LLM_TEMPERATURE": "sampling.temperature",
    "LLM_MAX_TOKENS": "generation.max_tokens",
    "OPENAI_MODEL": "model",
    "LLM_DISABLE_THINKING": "reasoning.enabled",
}

_TOP_LEVEL_KEYS = frozenset({"llm"})
_LLM_KEYS = frozenset(
    {"model", "gpu_layers", "context_size", "parallel", "cache", "sampling", "reasoning", "generation"}
)
_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


class LlmConfigError(ValueError):
    """Raised when the canonical LLM configuration is invalid.

    The message always names the field and the offending value with the
    accepted range — configuration validation must FAIL CLEARLY, never
    silently fall back (the v0.4.x silent-default incidents: a typo in one
    env var produced a whole day of wrong runtime behaviour).
    """


def _err(path: str, value: Any, expect: str) -> LlmConfigError:
    return LlmConfigError(
        f"config/llm_config.yaml: llm.{path} = {value!r} is invalid ({expect}). "
        "Fix config/llm_config.yaml (or the corresponding LLAMA_*/LLM_* "
        "environment override) and restart — the llama-server command line "
        "is generated from this value and will not start with an invalid one."
    )


def _parse_env_int(name: str, value: str) -> int:
    """Parse one integer env override with a NAMED, actionable error."""
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        raise LlmConfigError(
            f"Environment override {name}={value!r} is not an integer "
            f"(example: {name}=20). Unset it or fix the value — the "
            "canonical source is config/llm_config.yaml."
        ) from None


def _parse_env_float(name: str, value: str) -> float:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        raise LlmConfigError(
            f"Environment override {name}={value!r} is not a number "
            f"(example: {name}=0.7). Unset it or fix the value — the "
            "canonical source is config/llm_config.yaml."
        ) from None


def _parse_env_bool(name: str, value: str) -> bool:
    v = (value or "").strip().lower()
    if v in _BOOL_TRUE:
        return True
    if v in _BOOL_FALSE:
        return False
    raise LlmConfigError(
        f"Environment override {name}={value!r} is not a boolean "
        f"(use 1/true/yes/on or 0/false/no/off)."
    )


@dataclass(frozen=True)
class LlmRuntimeConfig:
    """Typed, validated LLM runtime values — the one consumer contract.

    Field provenance is recorded so startup diagnostics can show WHERE
    each effective value came from (``yaml`` / ``env:NAME`` /
    ``built-in-default``) — an environment override is never silent.
    """

    model: str
    gpu_layers: int
    context_size: int
    parallel: int
    cache_k: str
    cache_v: str
    temperature: float
    max_tokens: int
    reasoning_enabled: bool
    provenance: dict[str, str] = field(default_factory=dict)
    config_path: Optional[Path] = None

    # -- derived, single-source values ------------------------------------- #

    def llama_server_args(self) -> list[str]:
        """The llama-server CLI arguments GENERATED from this config.

        Exactly the flags the pinned llama.cpp b10717 supports (verified
        live — see docs/llm_config_ngl20-c32768/evidence/). ``--model``,
        ``--host`` and ``--port`` are appended by the caller (model path
        resolution and server binding are separate mechanisms); the
        reasoning flag is only passed when thinking is DISABLED (the
        b10717-verified value ``off`` — when reasoning is enabled the
        flag is omitted and the server default applies).
        """
        args = [
            "-ngl", str(self.gpu_layers),
            "-c", str(self.context_size),
            "--parallel", str(self.parallel),
            "--cache-type-k", self.cache_k,
            "--cache-type-v", self.cache_v,
            "--temp", str(self.temperature),
        ]
        if self.reasoning_server_arg:
            args += ["--reasoning", self.reasoning_server_arg]
        return args

    @property
    def reasoning_server_arg(self) -> str:
        """``"off"`` when thinking is disabled; ``""`` (flag omitted) when
        enabled — the server-side default then keeps the channel open."""
        return "off" if not self.reasoning_enabled else ""

    def thinking_control_kwargs(self) -> dict[str, Any]:
        """Request-level thinking suppression (the OpenAI-compat payload).

        Both switches land on ``enable_thinking = false`` server-side
        (llama.cpp-native template kwarg + the OpenAI-standard effort
        field, parsed by b10717 as "disable reasoning") — belt and
        braces, exactly the v0.4.4 fix contract. Empty when reasoning
        is enabled. ``app/llm.py`` consumes this; it is defined HERE so
        the yaml flag, the server flag and the request kwargs can never
        disagree.
        """
        if self.reasoning_enabled:
            return {}
        return {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }

    def summary_lines(self) -> list[str]:
        """The 'LLM configuration:' block for startup logging."""
        return [
            f"model = {self.model}",
            f"gpu_layers = {self.gpu_layers}",
            f"context_size = {self.context_size}",
            f"parallel = {self.parallel}",
            f"cache_k = {self.cache_k}",
            f"cache_v = {self.cache_v}",
            f"temperature = {self.temperature}",
            f"max_tokens = {self.max_tokens}",
            f"thinking = {str(self.reasoning_enabled).lower()}",
            "reasoning_effort = " + ("none" if not self.reasoning_enabled else "auto"),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "gpu_layers": self.gpu_layers,
            "context_size": self.context_size,
            "parallel": self.parallel,
            "cache_k": self.cache_k,
            "cache_v": self.cache_v,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_enabled": self.reasoning_enabled,
            "reasoning_server_arg": self.reasoning_server_arg,
            "llama_server_args": self.llama_server_args(),
            "provenance": dict(self.provenance),
            "config_path": str(self.config_path) if self.config_path else "",
        }


def resolve_llm_config_path(
    config_dir: Optional[Path] = None, root: Optional[Path] = None
) -> Optional[Path]:
    """Where the canonical file lives: LLM_CONFIG_PATH env > the config/
    directory next to the loaded voicemem_config.yaml > the repo root's
    config/ directory. None when no candidate exists."""
    env_path = os.environ.get(LLM_CONFIG_PATH_ENV, "").strip()
    if env_path:
        p = Path(env_path)
        return p
    candidates: list[Path] = []
    if config_dir is not None:
        candidates.append(Path(config_dir) / LLM_CONFIG_FILENAME)
    if root is not None:
        candidates.append(Path(root) / "config" / LLM_CONFIG_FILENAME)
    for c in candidates:
        if c.is_file():
            return c
    # No candidate exists on disk: report the primary location (the
    # sibling of the voicemem yaml when given, else the root config dir)
    # so error/warning messages point somewhere concrete.
    if config_dir is not None:
        return Path(config_dir) / LLM_CONFIG_FILENAME
    if root is not None:
        return Path(root) / "config" / LLM_CONFIG_FILENAME
    return None


def _validate_profile(profile: dict[str, Any], source: str) -> dict[str, Any]:
    """Validate one merged profile dict; returns normalised values or
    raises :class:`LlmConfigError` naming the field (fail clearly)."""
    model = profile.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _err("model", model, "a non-empty model identifier string")
    gpu_layers = profile.get("gpu_layers")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool):
        raise _err("gpu_layers", gpu_layers, "an integer layer count")
    if gpu_layers < 0:
        # llama-server accepts -1 (=all), but the canonical config
        # REQUIRES an explicit count: the production profile is a
        # measured partial offload (20/42), never an implicit "all".
        raise _err("gpu_layers", gpu_layers, ">= 0 (0 = CPU only; 20 = the measured 20/42 partial offload)")
    context_size = profile.get("context_size")
    if not isinstance(context_size, int) or isinstance(context_size, bool):
        raise _err("context_size", context_size, "an integer token count")
    if context_size <= 0:
        raise _err("context_size", context_size, "> 0 (llama-server -c)")
    if context_size > 2**20:
        raise _err("context_size", context_size, "<= 1048576 (a context this large is almost certainly a typo)")
    parallel = profile.get("parallel")
    if not isinstance(parallel, int) or isinstance(parallel, bool):
        raise _err("parallel", parallel, "an integer slot count")
    if parallel < 1:
        raise _err("parallel", parallel, ">= 1 (llama-server --parallel slots)")
    cache = profile.get("cache")
    if not isinstance(cache, dict):
        raise _err("cache", cache, "a mapping with 'key' and 'value'")
    cache_k = cache.get("key")
    cache_v = cache.get("value")
    for name, val in (("cache.key", cache_k), ("cache.value", cache_v)):
        if not isinstance(val, str) or val.strip().lower() not in SUPPORTED_KV_CACHE_TYPES:
            raise _err(
                name,
                val,
                "one of " + ", ".join(sorted(SUPPORTED_KV_CACHE_TYPES)) + " (llama.cpp KV cache types)",
            )
    sampling = profile.get("sampling")
    if not isinstance(sampling, dict):
        raise _err("sampling", sampling, "a mapping with 'temperature'")
    temperature = sampling.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise _err("sampling.temperature", temperature, "a number")
    if not (0.0 <= float(temperature) <= 2.0):
        raise _err("sampling.temperature", temperature, "within [0.0, 2.0]")
    reasoning = profile.get("reasoning")
    if not isinstance(reasoning, dict):
        raise _err("reasoning", reasoning, "a mapping with 'enabled'")
    reasoning_enabled = reasoning.get("enabled")
    if not isinstance(reasoning_enabled, bool):
        raise _err("reasoning.enabled", reasoning_enabled, "true or false (the single thinking switch)")
    generation = profile.get("generation")
    if not isinstance(generation, dict):
        raise _err("generation", generation, "a mapping with 'max_tokens'")
    max_tokens = generation.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise _err("generation.max_tokens", max_tokens, "an integer token budget")
    if max_tokens < 1:
        raise _err("generation.max_tokens", max_tokens, ">= 1")
    return {
        "model": model.strip(),
        "gpu_layers": gpu_layers,
        "context_size": context_size,
        "parallel": parallel,
        "cache_k": str(cache_k).strip().lower(),
        "cache_v": str(cache_v).strip().lower(),
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "reasoning_enabled": reasoning_enabled,
        "source": source,
    }


def load_llm_config(
    config_dir: Optional[Path] = None,
    root: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> LlmRuntimeConfig:
    """Load + validate the canonical config; apply documented env overrides.

    Args:
        config_dir: the directory of the loaded voicemem_config.yaml (the
            canonical file is its sibling).
        root: repository root (fallback: ``<root>/config/llm_config.yaml``).
        env: environment mapping (tests inject; default ``os.environ``).

    Raises:
        LlmConfigError: on any invalid value or unreadable structure —
        NEVER falls back silently (the file being MISSING falls back to
        :data:`DEFAULT_PROFILE` with a WARNING; an INVALID file is an
        error: the llama-server command line is generated from these
        values and must not start on garbage).
    """
    environ = os.environ if env is None else env
    path = resolve_llm_config_path(config_dir=config_dir, root=root)
    raw: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    if path is not None and path.is_file():
        if not _HAS_YAML:  # pragma: no cover
            raise LlmConfigError(
                "PyYAML is required to load the canonical LLM configuration "
                f"({path}): pip install pyyaml"
            )
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise LlmConfigError(
                f"{path}: the YAML root must be a mapping with a single 'llm:' key"
            )
        unknown_top = set(loaded) - _TOP_LEVEL_KEYS
        if unknown_top:
            raise LlmConfigError(
                f"{path}: unknown top-level key(s) {sorted(unknown_top)} — "
                "the canonical LLM config has exactly one section: 'llm:'"
            )
        raw = loaded.get("llm")
        if not isinstance(raw, dict):
            raise LlmConfigError(
                f"{path}: the 'llm:' section is missing or not a mapping"
            )
        unknown = set(raw) - _LLM_KEYS
        if unknown:
            raise LlmConfigError(
                f"{path}: unknown llm key(s) {sorted(unknown)} — supported: "
                + ", ".join(sorted(_LLM_KEYS))
            )
    else:
        logger.warning(
            "config/llm_config.yaml not found (looked at %s) — using the "
            "built-in DEFAULT_PROFILE (identical to the shipped file). "
            "Restore the file from the release ZIP to edit the LLM profile.",
            path or "<no candidate path>",
        )
        raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_PROFILE.items()}
        provenance = {k: "built-in-default" for k in (
            "model", "gpu_layers", "context_size", "parallel",
            "cache_k", "cache_v", "temperature", "max_tokens", "reasoning_enabled",
        )}

    merged: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
    for field_key in ("cache", "sampling", "reasoning", "generation"):
        section = merged.get(field_key)
        if not isinstance(section, dict):
            merged[field_key] = {}

    # --- documented env overrides (explicit, never silent) ----------------
    for name, field_path in ENV_OVERRIDES.items():
        value = (environ.get(name) or "").strip()
        if not value:
            continue
        section_key, _, leaf = field_path.partition(".")
        target: dict[str, Any] = merged
        if section_key in ("cache", "sampling", "reasoning", "generation"):
            target = merged.setdefault(section_key, {})
            key = leaf
        else:
            key = section_key
        if name == "LLM_DISABLE_THINKING":
            # 1 = thinking DISABLED (the env keeps the v1-spec semantics:
            # the variable disables thinking; the yaml flag enables it).
            target[key] = not _parse_env_bool(name, value)
        elif field_path in ("gpu_layers", "context_size", "parallel", "generation.max_tokens"):
            target[key] = _parse_env_int(name, value)
        elif field_path == "sampling.temperature":
            target[key] = _parse_env_float(name, value)
        else:  # model / cache types
            target[key] = value
        provenance[_provenance_key(field_path)] = f"env:{name}"
        logger.info("LLM config env override: %s -> llm.%s", name, field_path)

    normalised = _validate_profile(merged, "yaml" if (path and path.is_file()) else "defaults")
    # provenance for non-overridden yaml values
    field_to_path = {
        "model": "model", "gpu_layers": "gpu_layers", "context_size": "context_size",
        "parallel": "parallel", "cache_k": "cache.key", "cache_v": "cache.value",
        "temperature": "sampling.temperature", "max_tokens": "generation.max_tokens",
        "reasoning_enabled": "reasoning.enabled",
    }
    for fkey, fpath in field_to_path.items():
        provenance.setdefault(fkey, "yaml" if (path and path.is_file()) else "built-in-default")

    return LlmRuntimeConfig(
        model=normalised["model"],
        gpu_layers=normalised["gpu_layers"],
        context_size=normalised["context_size"],
        parallel=normalised["parallel"],
        cache_k=normalised["cache_k"],
        cache_v=normalised["cache_v"],
        temperature=normalised["temperature"],
        max_tokens=normalised["max_tokens"],
        reasoning_enabled=normalised["reasoning_enabled"],
        provenance=provenance,
        config_path=path if (path and path.is_file()) else None,
    )


def _provenance_key(field_path: str) -> str:
    mapping = {
        "model": "model", "gpu_layers": "gpu_layers", "context_size": "context_size",
        "parallel": "parallel", "cache.key": "cache_k", "cache.value": "cache_v",
        "sampling.temperature": "temperature", "generation.max_tokens": "max_tokens",
        "reasoning.enabled": "reasoning_enabled",
    }
    return mapping.get(field_path, field_path)


def format_llm_config_summary(runtime: LlmRuntimeConfig) -> str:
    """The full startup block: 'LLM configuration:' + every effective value."""
    lines = ["LLM configuration:"]
    lines += [f"  {line}" for line in runtime.summary_lines()]
    if runtime.config_path is not None:
        lines.append(f"  source = {runtime.config_path}")
    overridden = {k: v for k, v in runtime.provenance.items() if v != "yaml"}
    if overridden:
        lines.append(
            "  env overrides active: "
            + ", ".join(f"{k} <- {v}" for k, v in sorted(overridden.items()))
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# llama-server startup-log cross-check (requirement: the effective runtime
# configuration must be verifiable against the server's own startup log —
# offloaded layers = 20/42, n_ctx = 32768, n_ctx_slot = 32768).
# Line formats below are the pinned llama.cpp b10717 output, captured live:
# docs/llm_config_ngl20-c32768/evidence/server-log-startup-config.txt
# ---------------------------------------------------------------------------

import re  # noqa: E402  (kept next to the log matcher it belongs to)

_RE_OFFLOADED = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+(GPU|CPU)")
_RE_N_CTX = re.compile(r"llama_context:\s+n_ctx\s*=\s*(\d+)")
_RE_N_CTX_SLOT = re.compile(r"n_ctx_slot\s*=\s*(\d+)")
_RE_KV_K = re.compile(r"K\s*\((\w+)\)\s*:")
_RE_KV_V = re.compile(r"V\s*\((\w+)\)\s*:")
_RE_THINKING = re.compile(r"chat template,\s*thinking\s*=\s*(\d)")


def match_llama_server_log(text: str, runtime: LlmRuntimeConfig) -> dict[str, Any]:
    """Compare the llama-server startup log against the effective config.

    Returns a verdict dict — each item carries ``expected`` (the canonical
    config), ``actual`` (what the running server reported) and ``match``.
    Unknown log lines are reported as ``actual=None`` (the server did not
    print it — e.g. an older build): the per-item ``match`` is then None
    ("not verifiable") and the overall ``all_match`` verdict is FALSE —
    fail-closed, never a silent pass.
    """
    verdict: dict[str, Any] = {"items": {}, "all_match": None}

    def add(key: str, expected: Any, actual: Any) -> None:
        verdict["items"][key] = {
            "expected": expected,
            "actual": actual,
            "match": (expected == actual) if actual is not None else None,
        }

    m = _RE_OFFLOADED.search(text)
    # gpu_layers = the number of layers the server offloaded to the GPU
    add("offloaded_layers", runtime.gpu_layers,
        int(m.group(1)) if m else None)
    if m:
        verdict["items"]["offloaded_layers"]["detail"] = f"{m.group(1)}/{m.group(2)} layers to {m.group(3)}"

    m = _RE_N_CTX.search(text)
    add("n_ctx", runtime.context_size, int(m.group(1)) if m else None)
    m = _RE_N_CTX_SLOT.search(text)
    add("n_ctx_slot", runtime.context_size, int(m.group(1)) if m else None)

    m = _RE_KV_K.search(text)
    add("cache_k", runtime.cache_k, (m.group(1).lower() if m else None))
    m = _RE_KV_V.search(text)
    add("cache_v", runtime.cache_v, (m.group(1).lower() if m else None))

    m = _RE_THINKING.search(text)
    # chat template thinking flag: 0 == disabled (matches reasoning off)
    add("thinking", 0, (int(m.group(1)) if m else None))

    matches = [it["match"] for it in verdict["items"].values()]
    verdict["all_match"] = all(x is True for x in matches) and None not in matches
    return verdict
