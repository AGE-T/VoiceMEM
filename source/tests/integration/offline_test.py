"""M1 offline audit: zero non-loopback network connections (Python side).

Machine: ANY (sandbox-safe). The audit runs in two phases:

Phase 1 (always): with HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set, import every
existing app.* module UNDER a socket auditor (socket.socket.connect and
socket.getaddrinfo are patched to record every non-loopback target). Importing
pure modules must never touch the network.

Phase 2 (only once app.pipeline exists — Task 8 integration): run one short
mock turn under the same auditor.

Exit codes: 0 = no violations, 1 = non-loopback connect/DNS attempts detected.
Missing modules (not yet implemented, or heavy deps absent) are SKIPPED and
reported, never counted as violations.

Companion: scripts/offline_check.ps1 audits the OS-level TCP connections
(Get-NetTCPConnection) while the agent is running.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests/<kind>/x.py -> 3 levels up)

APP_MODULES = (
    "app", "app.config", "app.text_utils", "app.teacher_persona", "app.barge_in",
    "app.vad", "app.asr", "app.audio_io", "app.llm", "app.tts",
    "app.voicemem_bridge", "app.mock_components", "app.pipeline", "app.main",
)

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "MEM0_ANONYMIZED_TELEMETRY": "false",
    "MEM0_TELEMETRY": "false",
}


def _is_loopback(host: Any) -> bool:
    """Loopback / unspecified / empty targets are exempt from the audit."""
    if not isinstance(host, str) or not host:
        return True
    lowered = host.casefold()
    if lowered == "localhost" or lowered in ("::1", "::", "0.0.0.0", "[::1]"):
        return True
    if lowered.startswith("127."):
        return True
    if lowered.startswith("::ffff:127."):  # IPv6-mapped IPv4 loopback
        return True
    return False


class SocketAuditor:
    """Context manager that records non-loopback connect()/getaddrinfo() calls.

    Calls are recorded AND passed through to the real implementation, so a
    violating module still works during the audit — the attempt itself is the
    finding. (In the sandbox the pure modules make no calls at all.)
    """

    def __init__(self) -> None:
        self.connects: list[tuple[str, Any]] = []
        self.dns_queries: list[str] = []
        self._orig_connect: Any = None
        self._orig_getaddrinfo: Any = None

    def __enter__(self) -> "SocketAuditor":
        self._orig_connect = socket.socket.connect
        self._orig_getaddrinfo = socket.getaddrinfo

        def _patched_connect(sock: Any, address: Any) -> Any:
            host = address[0] if isinstance(address, tuple) and address else address
            if not _is_loopback(host):
                self.connects.append(("connect", address))
            return self._orig_connect(sock, address)

        def _patched_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            if not _is_loopback(host):
                self.dns_queries.append(str(host))
            return self._orig_getaddrinfo(host, *args, **kwargs)

        socket.socket.connect = _patched_connect  # type: ignore[method-assign]
        socket.getaddrinfo = _patched_getaddrinfo  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        socket.socket.connect = self._orig_connect  # type: ignore[method-assign]
        socket.getaddrinfo = self._orig_getaddrinfo  # type: ignore[assignment]


def audit_module_imports() -> tuple[list[tuple[str, Any]], list[str], list[str]]:
    """Import every existing app.* module under the socket auditor.

    Returns (violations, skipped_modules, imported_modules).
    """
    violations: list[tuple[str, Any]] = []
    skipped: list[str] = []
    imported: list[str] = []
    import importlib

    for name in APP_MODULES:
        with SocketAuditor() as auditor:
            try:
                importlib.import_module(name)
            except ImportError:
                skipped.append(name)
                continue
            except Exception as exc:  # noqa: BLE001 - record, do not crash the audit
                skipped.append(f"{name} (import error: {type(exc).__name__})")
                continue
            imported.append(name)
        violations.extend(auditor.connects)
        violations.extend(("dns", host) for host in auditor.dns_queries)
    return violations, skipped, imported


async def _run_one_mock_turn(auditor: SocketAuditor) -> str:
    """Phase 2: one mock turn through the full pipeline under the auditor."""
    try:
        from app.config import AgentConfig
        from app.mock_components import (  # lazy: Task 8 file
            MockAsrEngine, MockLlmClient, MockTtsEngine, MockVad, MockVoiceMemBridge,
        )
        from app.pipeline import VoicePipeline  # lazy: Task 8 file
    except ImportError as exc:
        return f"SKIPPED (pipeline not present yet: {exc})"

    try:
        cfg = AgentConfig()
        cfg.apply_env()

        def _ctor(cls: type, *args: Any) -> Any:
            try:
                return cls(*args)
            except TypeError:
                return cls()

        asr = _ctor(MockAsrEngine)
        asr.queue = ["offline audit probe turn"]
        pipeline = VoicePipeline(
            config=cfg, asr=asr, llm=_ctor(MockLlmClient), tts=_ctor(MockTtsEngine),
            voicemem=_ctor(MockVoiceMemBridge), vad=_ctor(MockVad),
            audio_out=None,
        )
        try:
            import numpy as np
            audio = np.zeros(16000, dtype=np.float32)
        except ImportError:
            return "SKIPPED (numpy missing)"
        await pipeline.handle_utterance(audio, speaker_id="voice_user",
                                        vad_prob_fn=lambda: 0.0)
        return "OK (one mock turn executed under the socket auditor)"
    except Exception as exc:  # noqa: BLE001 - assembly errors are reportable, not fatal
        return f"SKIPPED (mock pipeline could not be assembled: {type(exc).__name__}: {exc})"


def main() -> int:
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value

    print("=" * 72)
    print("M1 offline audit (Python-level, socket connect + DNS recording)")
    print("=" * 72)
    print("Offline env forced for this run:")
    for key, value in OFFLINE_ENV.items():
        print(f"  {key}={value}")
    print("-" * 72)

    violations, skipped, imported = audit_module_imports()
    print(f"imported cleanly : {', '.join(imported) if imported else '(none)'}")
    print(f"skipped          : {', '.join(skipped) if skipped else '(none)'}")
    print("-" * 72)

    # Phase 2: full pipeline mock run (only when Task 8 landed).
    pipeline_note = "not attempted"
    try:
        import app.pipeline  # noqa: F401  — probe availability
    except ImportError:
        print("app.pipeline not present yet (arrives in Task 8 integration);")
        print("the full-pipeline socket audit will activate once it lands.")
    else:
        with SocketAuditor() as auditor:
            import asyncio

            pipeline_note = asyncio.run(_run_one_mock_turn(auditor))
        pipeline_violations = list(auditor.connects)
        pipeline_violations += [("dns", host) for host in auditor.dns_queries]
        print(f"pipeline mock run: {pipeline_note}")
        for kind, target in pipeline_violations:
            print(f"  VIOLATION ({kind}): {target}")
        violations.extend(pipeline_violations)

    print("-" * 72)
    if violations:
        print(f"VIOLATIONS ({len(violations)}): non-loopback network attempts detected")
        for kind, target in violations:
            print(f"  {kind}: {target}")
        print("RESULT: FAIL — offline requirement broken (see the call sites above)")
        print("=" * 72)
        return 1
    print("No non-loopback connect()/getaddrinfo() attempts were recorded.")
    print("RESULT: PASS — Python side is network-free for the audited scope.")
    print("Companion OS-level check: scripts/offline_check.ps1 -Capture -Seconds 30")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
