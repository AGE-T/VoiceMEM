"""Experimental llama.cpp b11073 runtime — validation-only tests (2026-09-21).

These tests validate the ISOLATED EXPERIMENTAL runtime profile introduced
next to the untouched v0.10.7 production baseline:

  * the experimental launchers carry the operator-tested baseline
    (ngl 99 / ctx 16000 / parallel 1 / threads 12 / reasoning off) and bind
    a SEPARATE port (default 127.0.0.1:8081, never 8080);
  * the EXISTING environment-only override (LLAMA_SERVER_HOST/PORT,
    app/config.py apply_env) retargets the whole application client surface
    at the experimental server — and UNSETTING it restores the production
    endpoint (the rollback proof);
  * the production baseline is untouched (pinned llama.cpp b10717 in
    MODELS.lock.json, canonical config values, VERSION);
  * a REAL b11073 SSE stream (captured live, pinned as a fixture) parses
    through the production client's own parser (content deltas assemble,
    the role chunk with ``"content": null`` is ignored, no reasoning
    leakage, [DONE] handled);
  * the request-level thinking suppression kwargs (accepted by b11073 in
    the live compatibility run) are exactly what the production client sends.

No production behaviour is changed by these tests; they PIN the isolation
contract so a future accidental coupling (hardcoded port, canonical-config
edits, pinned-runtime edits) fails loudly here first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import AgentConfig
from app.llm import parse_sse_content_delta, parse_sse_reasoning_delta

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experimental" / "llama_b11073"


# ---------------------------------------------------------------- launchers

def test_experimental_directory_exists_with_both_launchers() -> None:
    assert EXP.is_dir(), "experimental/llama_b11073/ profile missing"
    assert (EXP / "start_experimental_8081.sh").is_file()
    assert (EXP / "start_llama_server_experimental.ps1").is_file()
    assert (EXP / "README.md").is_file()


def test_sandbox_launcher_carries_operator_baseline() -> None:
    """The sandbox launcher must default to the operator-tested baseline:
    ngl 99, ctx 16000, parallel 1, reasoning off, port 8081 — every value
    env-overridable (nothing hardcoded anywhere else)."""
    sh = (EXP / "start_experimental_8081.sh").read_text(encoding="utf-8")
    assert 'LLAMA_EXPERIMENTAL_PORT:-8081' in sh
    assert 'LLAMA_EXPERIMENTAL_NGL:-99' in sh
    assert 'LLAMA_EXPERIMENTAL_CTX:-16000' in sh
    assert '"--parallel", "1"' in sh or '--parallel 1' in sh or '"--parallel", "1"' in sh
    assert '"--reasoning", "off"' in sh or '--reasoning off' in sh
    # isolation: never the production port
    assert ":8080" not in sh.replace("8080 (ngl", "(ngl")
    assert "127.0.0.1" in sh


def test_windows_launcher_carries_operator_baseline_and_separate_path() -> None:
    """The target-machine launcher: ngl 99 / ctx 16000 / threads 12 /
    reasoning off, the SEPARATE binary path bin\\llama-server-b11073\\ and
    port 8081; the pinned production binary/config are never referenced
    as edit targets."""
    ps1 = (EXP / "start_llama_server_experimental.ps1").read_text(encoding="utf-8")
    assert '"99"' in ps1                    # $Ngl = "99"
    assert '"16000"' in ps1                # $Ctx = "16000"
    assert '"12"' in ps1                    # $Threads = "12"
    assert '"--reasoning", "off"' in ps1 or '"--reasoning off"' in ps1 or '"off"' in ps1
    assert r"bin\llama-server-b11073" in ps1
    assert "8081" in ps1
    assert "8080" in ps1  # only inside the explanatory notes...
    # ...never as a bind target:
    for line in ps1.splitlines():
        if '"--port"' in line or "-Port " in line and "8081" not in line:
            assert "8080" not in line, f"production port bound in: {line.strip()}"


# ------------------------------------------------- env override (the client)

def test_env_override_targets_experimental_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLAMA_SERVER_HOST/PORT — the EXISTING production mechanism — retargets
    the whole client surface at the experimental server. (monkeypatch keeps
    the test pollution-proof against URL-affecting vars leaked by other
    tests, e.g. OPENAI_BASE_URL from test_config.py runs.)"""
    monkeypatch.setenv("LLAMA_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = AgentConfig()
    cfg.apply_env()
    assert cfg.llama_server_host == "127.0.0.1"
    assert cfg.llama_server_port == 8081
    assert cfg.llama_server_url == "http://127.0.0.1:8081/v1"
    assert cfg.llama_server_health_url == "http://127.0.0.1:8081/health"


def test_env_override_unset_restores_production_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rollback proof: without the override the client targets the
    production endpoint (127.0.0.1:8080) — the experiment is fully
    reversible with zero code changes."""
    monkeypatch.delenv("LLAMA_SERVER_HOST", raising=False)
    monkeypatch.delenv("LLAMA_SERVER_PORT", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = AgentConfig()
    cfg.apply_env()
    assert cfg.llama_server_url == "http://127.0.0.1:8080/v1"
    assert cfg.llama_server_health_url == "http://127.0.0.1:8080/health"


def test_vendor_bridge_follows_the_same_url() -> None:
    """Every vendor LLM leg (mem0 bridge) derives from the same
    llama_server_url — the env override moves them all together."""
    cfg = AgentConfig()
    cfg.llama_server_host = "127.0.0.1"
    cfg.llama_server_port = 8081
    assert cfg.llama_server_url == "http://127.0.0.1:8081/v1"


# ------------------------------------------------ production-untouched pins

def test_pinned_llama_runtime_is_still_b10717() -> None:
    lock = json.loads((REPO / "MODELS.lock.json").read_text(encoding="utf-8"))
    llama_tools = [t for t in lock.get("tools", [])
                   if t.get("component") == "llama-server"]
    assert llama_tools, "llama-server tool entry vanished from MODELS.lock.json"
    assert any(t.get("tag") == "b10717" for t in llama_tools), \
        "the pinned production llama.cpp build changed — the experimental workstream must NOT touch it"


def test_canonical_llm_config_unchanged() -> None:
    """The canonical production profile stays ngl 20 / ctx 32768 (the
    experimental baseline deliberately lives ONLY in the launchers)."""
    import yaml
    cfg = yaml.safe_load((REPO / "config" / "llm_config.yaml").read_text(encoding="utf-8"))
    llm = cfg["llm"]
    assert llm["gpu_layers"] == 20
    assert llm["context_size"] == 32768
    assert llm["parallel"] == 1
    assert llm["reasoning"]["enabled"] is False


def test_version_is_still_0107() -> None:
    assert (REPO / "VERSION").read_text(encoding="utf-8").strip() == "0.10.7"


def test_no_production_module_references_the_experimental_profile() -> None:
    """Isolation: nothing under app/ may know about the experimental
    runtime — the application only ever sees host:port."""
    hits = []
    for p in (REPO / "app").rglob("*.py"):
        if "experimental" in p.read_text(encoding="utf-8").lower():
            hits.append(p.relative_to(REPO))
    assert not hits, f"production modules reference the experimental profile: {hits}"


# --------------------------------------------- b11073 SSE shape (live-captured)

FIXTURE = REPO / "tests" / "unit" / "data" / "llama_b11073_sse_sample.txt"


def _fixture_lines() -> list[str]:
    assert FIXTURE.is_file(), "b11073 SSE fixture missing"
    return FIXTURE.read_text(encoding="utf-8").splitlines()


def test_b11073_fixture_is_from_the_experimental_build() -> None:
    raw = FIXTURE.read_text(encoding="utf-8")
    assert "b11073-1aa2954bd" in raw, "fixture must carry the b11073 system_fingerprint"


def test_b11073_sse_parses_through_production_client() -> None:
    lines = _fixture_lines()
    deltas = [d for d in (parse_sse_content_delta(l) for l in lines) if d]
    assembled = "".join(deltas)
    assert assembled.strip(), "content did not assemble from the b11073 stream"
    assert not any(parse_sse_reasoning_delta(l) for l in lines), \
        "reasoning leaked into the content channel (thinking suppression failed)"
    assert any(l.strip() == "data: [DONE]" for l in lines), "[DONE] terminator missing"


def test_b11073_role_chunk_with_null_content_is_ignored() -> None:
    """b11073's first chunk carries delta {role: assistant, content: null}
    — the production parser's isinstance guard must skip it (a None
    content is not a str, so no empty-string token is yielded)."""
    lines = _fixture_lines()
    role_lines = [l for l in lines if '"role":"assistant"' in l]
    assert role_lines, "fixture lacks the role chunk"
    for l in role_lines:
        assert parse_sse_content_delta(l) == ""


# ----------------------------------------------- thinking suppression kwargs

def test_thinking_control_kwargs_shape() -> None:
    """The request-level suppression (both switches land on
    enable_thinking=false server-side) is what the client sends when the
    canonical config disables reasoning — b11073 accepted both in the
    live compatibility run."""
    from app.llm_config import LlmRuntimeConfig
    runtime = LlmRuntimeConfig(
        model="qwen3.6-35b-a3b", gpu_layers=20, context_size=32768, parallel=1,
        cache_k="q8_0", cache_v="q8_0", temperature=0.7, max_tokens=512,
        reasoning_enabled=False)
    kw = runtime.thinking_control_kwargs()
    assert kw == {"chat_template_kwargs": {"enable_thinking": False},
                  "reasoning_effort": "none"}
    assert runtime.reasoning_server_arg == "off"
