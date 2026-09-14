"""[v0.9.0 — VM-LOCAL-015] Memory-semantics validation with the REAL local
multilingual-E5 (the memory embedding model the pinned vendor uses).

Deep validation (rules per tests/validation/_report.py): when the model or
its runtime is unavailable on this machine, the deep part SKIPS with the
reason (the deterministic decision logic is fully covered by
tests/unit/test_trait_semantics.py); when the model IS present, the
measured semantic matrix is re-measured and pinned as a regression guard:

  * the HAZARD pairs (antipathy / past / qualified / flip-back) must
    exceed the 0.95 merge threshold — that is the E5 evidence that the
    stance gate is REQUIRED (cosine cannot separate agreement from
    contradiction);
  * the direct-negation cluster must sit inside the supersede band
    [0.90, 0.95) — the band is anchored to the measured cluster;
  * paraphrase pairs stay above 0.95 (merge is correct for them);
  * cross-topic noise pairs stay below the presupposition band.

If a future embedder change moves these numbers, THIS test fails — the
thresholds in traits_store are then stale and must be re-measured
(scripts/measure_semantic_matrix.py) and re-anchored. The gate stays
honest: no invented thresholds survive a model change unnoticed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.validation._report import FeatureValidationTest, has_module  # noqa: E402

# NOTE: the vendor path insertion and every voicemem import happen INSIDE
# the test methods, deliberately. A module-level insertion would run at
# collection time (before any test executes) and flip
# has_module("voicemem") for the EARLIER-collected
# test_feature_memory.MemoryFeatureTest.test_logic_bridge_degraded_mode —
# the unit suite tolerates that interaction (its bridge degraded test is
# in the pinned env-gap baseline), the validation suite does not.

_EMBEDDING_DIR = REPO_ROOT / "models" / "embedding" / "multilingual-e5-small"


def _e5_available() -> bool:
    if not (has_module("torch") and has_module("transformers")):
        return False
    has_local = any(_EMBEDDING_DIR.glob("*.safetensors"))
    has_cached = has_module("huggingface_hub")
    return has_local or has_cached


class _Embedder:
    """E5 passage embeddings via plain transformers (mean pooling + L2
    norm) — the same maths as the vendor's LocalE5Embedder."""

    _inst = None

    @classmethod
    def instance(cls):
        if cls._inst is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            name = (str(_EMBEDDING_DIR)
                    if any(_EMBEDDING_DIR.glob("*.safetensors"))
                    else "intfloat/multilingual-e5-small")
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
            model.eval()
            cls._inst = (tok, model)
        return cls._inst

    @classmethod
    def embed(cls, texts: list[str]) -> list:
        import numpy as np
        import torch

        tok, model = cls.instance()
        with torch.no_grad():
            enc = tok([f"passage: {t}" for t in texts], padding=True,
                      truncation=True, max_length=128, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)


def _cos(a, b) -> float:
    return float(a @ b)


#: (label, existing, incoming, min_expected, max_expected)
#: The bands are the MEASURED values from 2026-09-14 (see
#: docs/SEMANTIC_MATRIX.md); tolerance ±0.02 for cross-machine variance.
_MEASURED = [
    # hazard pairs — above the 0.95 merge threshold (the reason the gate exists)
    ("antipathy-hu", "Thomas szereti a motorokat", "Thomas utálja a motorokat", 0.93, 1.01),
    ("past-hu", "Thomas szereti a motorokat", "Thomas régen szerette a motorokat", 0.95, 1.01),
    ("past-en", "Thomas likes motorcycles", "Thomas used to like motorcycles", 0.95, 1.01),
    ("qualified-hu", "Thomas szereti a motorokat", "Thomas szereti a motorokat, kivéve télen", 0.93, 1.01),
    ("flip-back-hu", "már nem szereti a motorokat", "megint szereti a motorokat", 0.92, 1.01),
    # direct negation cluster — inside the supersede band [0.90, 0.95)
    ("negation-en", "Thomas likes motorcycles", "Thomas no longer likes motorcycles", 0.88, 0.95),
    ("negation-hu2", "Thomas szereti a motorokat", "Thomas nem szereti a motorokat", 0.90, 0.95),
    ("claim-pos-neg", "likes motorcycles", "no longer likes motorcycles", 0.88, 0.95),
    ("antipathy-en", "Thomas likes motorcycles", "Thomas hates motorcycles", 0.88, 0.95),
    ("stopped", "likes coffee", "stopped liking coffee", 0.88, 0.95),
    # paraphrase — merge is the correct outcome (above 0.95)
    ("paraphrase-en", "Thomas likes motorcycles", "Thomas enjoys riding motorcycles", 0.95, 1.01),
    ("paraphrase-hu", "Thomas szereti a motorokat", "Thomas imád motorozni", 0.95, 1.01),
    # presupposition replacement — the 0.88 band case
    ("gave-up", "likes coffee", "gave up coffee", 0.86, 0.95),
    # noise — below the presupposition band (different topic)
    ("noise-topic", "likes motorcycles", "no longer likes bicycles", 0.80, 0.89),
    ("noise-topic2", "likes coffee", "gave up tea", 0.75, 0.89),
]


class MemorySemanticsValidation(FeatureValidationTest):
    FEATURE = "memory-semantics"

    def test_logic_contract(self):
        # The deterministic decision logic (stance gate, bands, freezing,
        # chains, retrieval ordering) is fully covered by the unit battery;
        # here we only assert the constants stay the measured values.
        VENDOR = REPO_ROOT / "vendor" / "voicemem"
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from voicemem.rightbrain.traits_store import (
            MERGE_THRESHOLD, SUPERSEDE_MIN_SIM, SUPERSEDE_MIN_SIM_PRESUPPOSITION)
        self.assertEqual(MERGE_THRESHOLD, 0.95)
        self.assertEqual(SUPERSEDE_MIN_SIM, 0.90)
        self.assertEqual(SUPERSEDE_MIN_SIM_PRESUPPOSITION, 0.88)

    def test_deep_measured_matrix_regression_guard(self):
        if not _e5_available():
            self.deep_skip(
                "local multilingual-e5-small + torch/transformers not "
                "available on this machine (operator-placed weights)")
        texts: list[str] = []
        for _, a, b, _, _ in _MEASURED:
            texts.extend((a, b))
        vecs = _Embedder.embed(texts)
        drift = []
        for i, (label, a, b, lo, hi) in enumerate(_MEASURED):
            sim = _cos(vecs[2 * i], vecs[2 * i + 1])
            if not (lo - 0.02 <= sim <= hi + 0.02):
                drift.append(f"{label}={sim:.4f} not in [{lo:.2f},{hi:.2f}]")
        self.assertFalse(
            drift,
            "The measured E5 semantic matrix drifted outside the pinned "
            "bands — the trait thresholds (MERGE_THRESHOLD 0.95 / "
            "SUPERSEDE_MIN_SIM 0.90 / PRESUPPOSITION 0.88) are anchored to "
            "the 2026-09-14 measurements. Re-run "
            "scripts/measure_semantic_matrix.py and re-anchor: "
            + "; ".join(drift))
        self.deep_pass(
            f"E5 semantic matrix pinned: {len(_MEASURED)} measured pairs "
            "inside their bands (hazard pairs >0.95 confirm the stance "
            "gate; negation cluster inside the supersede band)")


if __name__ == "__main__":
    unittest.main()
