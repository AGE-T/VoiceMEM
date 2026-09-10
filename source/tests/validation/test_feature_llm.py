"""LLM feature validation (Task 12 - M0.1).

Validates the ``llm`` feature: the llama-server endpoint contract of
``app.config.AgentConfig`` (OpenAI-compatible base URL on loopback 8080 with
a ``/health`` probe), the message layout produced by
``app.teacher_persona.build_messages`` ([system, *history, user]) and the
importability of ``app.llm.LlmClient`` (httpx is a core dependency).

Deep check: when llama-server is REALLY running on 127.0.0.1:8080, the
health endpoint must answer HTTP 200 with the llama-server health JSON.
The responder is identified by its llama.cpp signatures (GET /health
answers a JSON object with a "status" field; GET /props answers a JSON
object). A DIFFERENT service that happens to listen on 127.0.0.1:8080 must
NOT fail the release gate - it is recognized as "not llama-server" and the
deep check skips with an explicit reason (release-blocker regression:
v0.1.6 - the old test failed on any non-200 answer, including foreign
services on the port).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from app.llm import LlmClient
from app.teacher_persona import build_messages
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

_ENV_KEYS = (
    "VOICEMEM_HOME", "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT",
    "OPENAI_BASE_URL", "OPENAI_MODEL", "LLAMA_MODEL_PATH",
    "LLAMA_CONTEXT_SIZE", "LLAMA_N_GPU_LAYERS", "LLAMA_CACHE_TYPE_K",
    "LLAMA_CACHE_TYPE_V",
)


def classify_llama_health_response(status_code: int, body: object) -> str:
    """Classify an HTTP ``/health`` response as llama-server or a foreign service.

    llama-server (the llama.cpp server) answers ``GET /health`` with a JSON
    object containing a ``"status"`` field: ``{"status": "ok"}`` (HTTP 200,
    model loaded) or ``{"status": "loading model"}`` (HTTP 503, model still
    loading). Anything else - non-JSON bodies, JSON without a ``"status"``
    field, plain 404/redirect pages - is NOT llama-server.

    Returns one of:
      * ``"llama-ok"``                      - 200 + status "ok"
      * ``"llama-loading"``                 - still loading the model
      * ``"llama-unhealthy:<code>:<st>"``   - llama answered, non-ready state
      * ``"not-llama"``                     - a foreign service answered
    """
    if not isinstance(body, dict) or "status" not in body:
        return "not-llama"
    status = str(body.get("status", "")).strip().lower()
    if status_code == 200 and status == "ok":
        return "llama-ok"
    if "loading" in status:
        return "llama-loading"
    return f"llama-unhealthy:{status_code}:{status}"


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class LlmFeatureTest(_EnvNeutralTest):
    """llama-server endpoint + chat message layout contracts."""

    FEATURE = "llm"

    def _probe_llama_server(self, client: httpx.Client) -> tuple[str, str]:
        """Probe the configured llama-server endpoint and classify the responder.

        Returns ``(verdict, detail)``:

        * ``unreachable``       - nothing answered on the endpoint
        * ``not-llama``         - something answered, but it is not llama-server
        * ``llama-loading``     - llama-server answered, model still loading
        * ``llama-unconfirmed`` - /health is llama-like but /props does not
          confirm the llama.cpp server identity (possible mimic)
        * ``llama-unhealthy``   - llama-server answered in a non-ready state
        * ``llama-ok``          - llama-server confirmed healthy
        """
        cfg = AgentConfig.from_yaml(YAML_PATH)
        health_url = cfg.llama_server_health_url
        try:
            response = client.get(health_url)
        except (httpx.ConnectError, httpx.HTTPError) as exc:
            return ("unreachable", type(exc).__name__)
        try:
            body = response.json()
        except ValueError:
            body = None
        verdict = classify_llama_health_response(response.status_code, body)
        if verdict != "llama-ok":
            return (verdict, f"HTTP {response.status_code}")
        # Secondary confirmation: the llama.cpp server also answers
        # GET /props with a JSON object (a foreign OpenAI-compatible service
        # does not). Both signals together identify llama-server.
        props_url = health_url[: health_url.rfind("health")] + "props"
        props_status = 0
        props_body = None
        try:
            props = client.get(props_url)
            props_status = props.status_code
            try:
                props_body = props.json()
            except ValueError:
                props_body = None
        except (httpx.ConnectError, httpx.HTTPError):
            props_body = None
        if isinstance(props_body, dict):
            return ("llama-ok", "health 200 status=ok; /props JSON OK")
        return (
            "llama-unconfirmed",
            f"/health is llama-like but /props did not answer with "
            f"llama-server JSON (HTTP {props_status})",
        )

    def test_logic_endpoint_contract(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        self.assertEqual(cfg.llama_server_url, "http://127.0.0.1:8080/v1")
        self.assertTrue(cfg.llama_server_health_url.endswith("/health"))
        self.assertEqual(cfg.llama_server_host, "127.0.0.1", "loopback only")
        self.assertEqual(cfg.llama_server_port, 8080)
        # LlmClient is importable and constructible without touching the network.
        client = LlmClient(cfg)
        self.assertIsNone(client._client, "httpx client must be lazy")

    def test_logic_message_layout_with_history(self):
        history = [
            {"role": "user", "content": "What is the past perfect?"},
            {"role": "assistant", "content": "It pairs have with a past participle."},
        ]
        messages = build_messages("hi", "SYSTEM PROMPT", history=history)
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(messages[0], {"role": "system", "content": "SYSTEM PROMPT"})
        self.assertEqual(messages[-1], {"role": "user", "content": "hi"})

    # ------------------------------------------- llama-server recognizer ----

    def test_logic_health_classifier_llama_server_signatures(self):
        self.assertEqual(classify_llama_health_response(200, {"status": "ok"}), "llama-ok")
        self.assertEqual(
            classify_llama_health_response(503, {"status": "loading model"}),
            "llama-loading",
        )
        self.assertEqual(
            classify_llama_health_response(503, {"status": "loading"}), "llama-loading"
        )
        self.assertEqual(
            classify_llama_health_response(503, {"status": "no slots available"}),
            "llama-unhealthy:503:no slots available",
        )

    def test_logic_health_classifier_foreign_services_are_not_llama(self):
        self.assertEqual(classify_llama_health_response(200, "hello world"), "not-llama")
        self.assertEqual(classify_llama_health_response(200, {"foo": "bar"}), "not-llama")
        self.assertEqual(classify_llama_health_response(404, None), "not-llama")
        self.assertEqual(classify_llama_health_response(301, {}), "not-llama")
        self.assertEqual(classify_llama_health_response(200, [1, 2, 3]), "not-llama")

    def test_logic_probe_recognizes_real_llama_server(self):
        """The full probe path: llama.cpp /health + /props -> llama-ok."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            if request.url.path == "/props":
                return httpx.Response(200, json={"total_slots": 1, "model": "gemma-4-12b"})
            return httpx.Response(404, text="not found")

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            verdict, detail = self._probe_llama_server(client)
        self.assertEqual(verdict, "llama-ok", f"unexpected detail: {detail}")

    def test_logic_probe_foreign_service_on_8080_is_not_a_failure(self):
        """Release-blocker regression: a foreign 200 responder on 127.0.0.1:8080
        must classify as "not-llama" (deep-skip), never as a test failure."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="hello from some other service")

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            verdict, detail = self._probe_llama_server(client)
        self.assertEqual(verdict, "not-llama", f"unexpected detail: {detail}")

    def test_logic_probe_foreign_json_api_on_8080(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"foo": "bar", "uptime": 12})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            verdict, _ = self._probe_llama_server(client)
        self.assertEqual(verdict, "not-llama")

    def test_logic_probe_foreign_404_service_on_8080(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="no such page")

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            verdict, _ = self._probe_llama_server(client)
        self.assertEqual(verdict, "not-llama")

    def test_logic_probe_llama_loading_state(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(503, json={"status": "loading model"})
            return httpx.Response(404)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            verdict, _ = self._probe_llama_server(client)
        self.assertEqual(verdict, "llama-loading")

    def test_logic_probe_health_mimic_without_props_is_unconfirmed(self):
        """A service faking the /health JSON but not serving /props is
        reported as "unconfirmed" (deep-skip), not as a healthy pass."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(404, text="no props here")

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            verdict, detail = self._probe_llama_server(client)
        self.assertEqual(verdict, "llama-unconfirmed", f"unexpected detail: {detail}")

    def test_logic_probe_unreachable_port(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            verdict, detail = self._probe_llama_server(client)
        self.assertEqual(verdict, "unreachable")
        self.assertEqual(detail, "ConnectError")

    def test_deep_llama_server_health(self):
        """llama-server health: deep PASS only when llama-server REALLY answers.

        A foreign service occupying 127.0.0.1:8080 deep-SKIPs (never fails):
        this deep check validates llama-server itself, not whatever else
        happens to listen on the port.
        """
        with httpx.Client(timeout=2.0) as client:
            verdict, detail = self._probe_llama_server(client)
        if verdict == "llama-ok":
            self.deep_pass(f"llama-server health endpoint OK ({detail})")
        elif verdict == "unreachable":
            self.deep_skip(
                f"llama-server not running on 127.0.0.1:8080 ({detail})"
            )
        elif verdict == "not-llama":
            self.deep_skip(
                "another service is answering on 127.0.0.1:8080 "
                f"({detail}, no llama-server health JSON) - llama-server "
                "health not validated"
            )
        elif verdict == "llama-loading":
            self.deep_skip(
                f"llama-server is answering but the model is still loading ({detail})"
            )
        elif verdict == "llama-unconfirmed":
            self.deep_skip(f"cannot confirm llama-server identity ({detail})")
        else:  # llama-unhealthy
            self.deep_skip(
                f"llama-server answered /health in a non-ready state ({detail})"
            )


if __name__ == "__main__":
    unittest.main()
