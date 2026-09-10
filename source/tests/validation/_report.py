"""Shared machinery for feature validation tests (Task 12 - M0.1).

Collects per-feature DEEP validation outcomes (pass / skipped + reason) and
writes them as JSON at process exit when ``VMA_VALIDATION_REPORT`` is set
(``scripts/run_tests.ps1`` sets it; ``scripts/build_release.ps1`` reads it).

Logic-level outcomes are NOT collected here: a failing logic test fails the
whole unittest run (exit code != 0), which already fails the release gate.

Usage (see ``test_feature_release.py`` for a live example)::

    class VadFeatureTest(FeatureValidationTest):
        FEATURE = "vad"

        def test_logic_frame_math(self):
            ...

        def test_deep_real_silero_inference(self):
            if not has_module("onnxruntime"):
                self.deep_skip("onnxruntime not installed")
            if not self.cfg.silero_vad_path.is_file():
                self.deep_skip("silero_vad.onnx not downloaded")
            ...  # real inference
            self.deep_pass("silero_vad.onnx session OK")

Rules for deep tests:
* check AVAILABILITY first and ``self.deep_skip(reason)`` when the component
  is simply not present on this machine;
* if the component IS present but misbehaves, let the test FAIL (that is a
  real validation failure, not a skip);
* always finish with ``self.deep_pass(detail)`` so the report can list the
  feature as validated;
* never hang: bound every network/subprocess interaction with a timeout.
"""
from __future__ import annotations

import atexit
import importlib.util
import json
import os
import platform
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

_SCHEMA_VERSION = 1

#: feature name -> {"status": "pass"|"skipped", "detail"/"reason": str}
_FEATURES: dict[str, dict[str, str]] = {}


def has_module(name: str) -> bool:
    """True when an importable top-level module exists (no import performed)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def record_deep_pass(feature: str, detail: str = "") -> None:
    """Record that a feature's deep validation ran and passed."""
    _FEATURES[feature] = {"status": "pass", "detail": detail}


def record_deep_skip(feature: str, reason: str) -> None:
    """Record that a feature's deep validation could not run (component absent)."""
    _FEATURES[feature] = {"status": "skipped", "reason": reason}


def _write_report() -> None:
    """Write the JSON report if VMA_VALIDATION_REPORT is set (atexit hook)."""
    path = os.environ.get("VMA_VALIDATION_REPORT", "").strip()
    if not path:
        return
    try:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "features": _FEATURES,
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(out)
    except Exception:  # reporting must never break the test run itself
        pass


atexit.register(_write_report)


class FeatureValidationTest(unittest.TestCase):
    """Base class for feature validation tests.

    Class attribute ``FEATURE`` names the feature in the JSON report
    (e.g. "vad", "asr", "llm"). Use ``deep_skip``/``deep_pass`` ONLY from
    ``test_deep_*`` methods.
    """

    FEATURE = "unknown"

    def deep_skip(self, reason: str) -> NoReturn:
        """Skip this deep test and record the reason in the report."""
        record_deep_skip(self.FEATURE, reason)
        self.skipTest(f"[deep:{self.FEATURE}] {reason}")

    def deep_pass(self, detail: str = "") -> None:
        """Record a successful deep validation for this feature."""
        record_deep_pass(self.FEATURE, detail)
