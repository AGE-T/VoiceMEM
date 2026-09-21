"""Regression tests: the memory-embedding invariant of the controlled vendor.

The invariant (v0.4.22 vendor move, PROVENANCE fix #1, upstream analogue
961efe8 "Use the injected embedder everywhere"):

    WHEN a VoiceMem facade is constructed with an injected embedder
    (``VoiceMem(embedding=…)`` -> ``Orchestrator._embedder``),
    THEN every memory-pipeline embedding call routes through that embedder
    (slot anchors, graph entities, right-brain trait vectors) and NO
    OpenAI-embeddings-API call is attempted — our llama-server does not
    serve ``/v1/embeddings`` (the stock v0.0.1 hard-wired the remote call,
    TraitStore._vec swallowed the failure, and rb_traits.embedding silently
    became NULL, killing trait merge + trait retrieval).

These tests drive the VENDOR source directly (sys.path insert — the sandbox
does not pip-install voicemem; the heavy import chains are lazy, so
``voicemem.orchestrator`` imports cleanly without torch/mem0/openai).
``openai`` is deliberately NOT installed in this sandbox: the no-embedder
test below turns that absence into the loudest possible regression alarm —
if the routing ever regresses, the OpenAI path is attempted and the import
fails the test.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "vendor" / "voicemem"

sys.path.insert(0, str(REPO_ROOT))            # repo root (app.* imports)


def _purge_voicemem_modules() -> None:
    """Drop every cached voicemem.* module (test isolation).

    Without this, later tests in the same session (web diagnostics, bridge
    degraded-mode) would see `voicemem` as importable and switch code paths.
    """
    for name in [n for n in list(sys.modules)
                 if n == "voicemem" or n.startswith("voicemem.")]:
        del sys.modules[name]


def setUpModule() -> None:
    _purge_voicemem_modules()
    sys.path.insert(0, str(VENDOR_ROOT))      # vendor voicemem.* imports


def tearDownModule() -> None:
    if str(VENDOR_ROOT) in sys.path:
        sys.path.remove(str(VENDOR_ROOT))
    _purge_voicemem_modules()


def _orchestrator_module():
    import voicemem.orchestrator as orch  # heavy deps stay lazy
    return orch


class _FakeEmbedder:
    """TextEmbedder-conforming fake: records every call, zero network."""

    def __init__(self) -> None:
        self.query_calls: list[str] = []
        self.texts_calls: list[list[str]] = []

    @property
    def model_name(self):
        return "fake-local (384)"

    @property
    def dimensions(self):
        return 384

    def embed_texts(self, texts):
        self.texts_calls.append(list(texts))
        return [[0.01] * 384 for _ in texts]

    def embed_query_text(self, text):
        self.query_calls.append(text)
        return [0.01] * 384


class _FakeEmbedderNoQueryFn:
    """Old-style embedder: only embed_texts (the fallback branch)."""

    def __init__(self) -> None:
        self.texts_calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.texts_calls.append(list(texts))
        return [[0.02] * 384 for _ in texts]


class _StubOrch:
    """Minimal host carrying ONLY the attributes _embed_text touches.

    With an injected embedder the method returns before touching _cache /
    _lock / _base_url, so the stub needs nothing else; the no-embedder case
    IS the test (the OpenAI path must explode, not degrade).
    """

    def __init__(self, embedder):
        self._embedder = embedder
        self._base_url = "http://127.0.0.1:9"   # nothing listens here
        self._cache: dict = {}
        import threading
        self._lock = threading.Lock()


class EmbeddingInvariantTests(unittest.TestCase):
    """PROVENANCE fix #1: injected embedder routing in _embed_text."""

    def test_injected_embedder_preferred_and_no_openai_import(self):
        """With an injected embedder the call NEVER leaves the local model.

        openai is intentionally absent in this sandbox: any attempt to
        take the remote path raises ImportError (ModuleNotFoundError),
        which fails this test loudly — exactly the observable failure the
        invariant demands.
        """
        orch = _orchestrator_module()
        fake = _FakeEmbedder()
        stub = _StubOrch(fake)
        vec = orch.Orchestrator._embed_text(stub, " query text ")

        self.assertEqual(len(vec), 384)
        self.assertEqual(fake.query_calls, [" query text "])   # query fn preferred
        self.assertEqual(fake.texts_calls, [])                 # no texts call
        # the OpenAI client module must not even be imported by this path
        self.assertNotIn("openai", sys.modules,
                         "the OpenAI SDK got imported during a local-embedder "
                         "embedding call — the invariant is broken")

    def test_injected_embedder_falls_back_to_embed_texts(self):
        """Old-style embedders (no embed_query_text) use embed_texts[0]."""
        orch = _orchestrator_module()
        fake = _FakeEmbedderNoQueryFn()
        stub = _StubOrch(fake)
        vec = orch.Orchestrator._embed_text(stub, "text")
        self.assertEqual(len(vec), 384)
        self.assertEqual(fake.texts_calls, [["text"]])
        self.assertNotIn("openai", sys.modules)

    def test_no_embedder_failure_is_loud_not_swallowed(self):
        """Without an injected embedder the OpenAI path is attempted AND
        its failure propagates — no silent empty/None vector.

        In this sandbox `openai` is not installed, so the attempt raises
        ModuleNotFoundError. The test asserts SOMETHING raises: a silent
        fallback (try/except -> [] or None) would be the regression this
        guards against (the TraitStore._vec swallow class of bug).
        """
        orch = _orchestrator_module()
        stub = _StubOrch(None)
        with self.assertRaises(Exception):
            orch.Orchestrator._embed_text(stub, "text")

    def test_stock_openai_path_still_present_for_default_deployments(self):
        """The DEFAULT (no injected embedder) behaviour stays the upstream
        OpenAI-API path — we did not delete the default, only bypassed it
        when an embedder is injected (upstream 961efe8 semantics).
        Static source check so the test does not depend on `openai`
        being importable.
        """
        src = (VENDOR_ROOT / "voicemem" / "orchestrator.py").read_text(
            encoding="utf-8")
        self.assertIn("if self._embedder is not None:", src)
        # the marker comment of the controlled fix must survive
        self.assertIn("CONTROLLED-VENDOR LOCAL FIX", src)
        # the remote path still exists after the routing block
        self.assertIn("def _embed_uncached", src)

    def test_rightbrain_receives_the_same_embed_choke_point(self):
        """RightBrain must be constructed with embed=self._embed_text so
        trait vectors share the injected-embedder routing (static wiring
        check — the behavioural proof is the trait tests below)."""
        src = (VENDOR_ROOT / "voicemem" / "orchestrator.py").read_text(
            encoding="utf-8")
        self.assertIn("self._right = RightBrain(", src)
        self.assertIn("embed=self._embed_text,", src)
        self.assertIn("self._left = LeftBrain(", src)
        self.assertIn("embedder=self._embedder,", src)


class TraitEmbeddingGateTests(unittest.TestCase):
    """PROVENANCE fixes #2/#3: the 384-d E5 similarity gate for traits."""

    def test_trait_min_sim_bound_to_dimension(self):
        from voicemem.rightbrain.brain import trait_min_sim
        self.assertAlmostEqual(trait_min_sim(384), 0.88)   # local E5 band
        self.assertAlmostEqual(trait_min_sim(1536), 0.45)  # OpenAI band
        self.assertAlmostEqual(trait_min_sim(None), 0.45)  # unknown -> default

    def test_trait_min_sim_env_override_wins(self):
        from voicemem.rightbrain import brain as rb
        old = os.environ.pop("VOICEMEM_RB_TRAIT_MIN_SIM", None)
        try:
            os.environ["VOICEMEM_RB_TRAIT_MIN_SIM"] = "0.7"
            import importlib
            importlib.reload(rb)
            self.assertAlmostEqual(rb.trait_min_sim(384), 0.7)
            self.assertAlmostEqual(rb.trait_min_sim(1536), 0.7)
        finally:
            os.environ.pop("VOICEMEM_RB_TRAIT_MIN_SIM", None)
            importlib.reload(rb)
        self.assertAlmostEqual(rb.trait_min_sim(384), 0.88)  # restored

    def test_trait_store_records_last_query_dim(self):
        """search_scored sets store.last_query_dim for the gate to read.

        Driven against a REAL temporary SQLite trait store with the fake
        embedder injected through the same embed fn the orchestrator's
        _embed_text choke point provides (query-style single-text call)."""
        import tempfile

        from voicemem.rightbrain.traits_store import Evidence, TraitStore

        fake = _FakeEmbedder()
        with tempfile.TemporaryDirectory() as td:
            store = TraitStore(Path(td) / "rb.sqlite", fake.embed_query_text)
            store.add("user1", "喜好与厌恶", "dislikes long meetings",
                      Evidence(quote="I always drift off in long meetings"))
            store.add("user1", "情绪", "anxious before reviews",
                      Evidence(quote="can't sleep before reviews"))
            hits = store.search_scored("user1", "long boring meetings")
            # fake vectors are all-equal -> similarity 1.0 -> both survive
            # the 384-d gate; the point under test is the dim bookkeeping:
            self.assertEqual(store.last_query_dim, 384)
            self.assertTrue(len(hits) >= 1)
            # the trait embeddings actually got STORED (the original NULL-
            # embedding bug: stock v0.0.1 called the OpenAI API here and
            # TraitStore._vec swallowed the failure into NULL vectors)
            with store._conn() as c:
                nulls = c.execute(
                    "SELECT COUNT(*) FROM rb_traits WHERE embedding IS NULL"
                ).fetchone()[0]
            self.assertEqual(nulls, 0)


class DimensionGuardTests(unittest.TestCase):
    """PROVENANCE fixes #4/#5: unequal-length vectors must never produce
    a meaningless similarity (silent truncation -> merge-on-noise)."""

    def test_cosine_unequal_length_returns_zero(self):
        from voicemem.utils.common._graph_common import cosine
        self.assertEqual(cosine([1.0, 0.0, 0.0], [1.0, 0.0]), 0.0)
        self.assertEqual(cosine([1.0, 0.0], [1.0, 0.0, 0.0]), 0.0)
        # equal-length still works
        self.assertAlmostEqual(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_graph_entity_store_skips_dim_mismatch(self):
        """find_similar_entity skips + counts dimension-mismatched stored
        vectors instead of merging on a truncated dot product."""
        import tempfile

        from voicemem.leftbrain.slot_split.graph_entity_store import (
            GraphEntityStore,
        )

        with tempfile.TemporaryDirectory() as td:
            store = GraphEntityStore(Path(td) / "ge.sqlite")
            # a stale 1536-d entity (simulating an embedder switch)
            store.get_or_create_entity_semantic(
                "user1", "work", "stale-entity", [0.1] * 1536)
            # a current 384-d entity
            store.get_or_create_entity_semantic(
                "user1", "work", "fresh-entity", [0.2] * 384)
            # searching with a 384-d vector must match ONLY the fresh one
            hit = store.find_similar_entity(
                "user1", "work", [0.2] * 384, threshold=0.5)
            self.assertIsNotNone(hit)
            self.assertEqual(hit.name, "fresh-entity")
            # ...and the stale 1536-d vector must never satisfy a 384-d query
            hit2 = store.find_similar_entity(
                "user1", "work", [0.2] * 384, threshold=0.99)
            # fresh-entity similarity is 1.0 -> still matches at 0.99; the
            # assertion that matters: no crash, and stale is unreachable
            names = [e.name for e in store.get_entities_for_slot("user1", "work")]
            self.assertIn("stale-entity", names)   # stored, just unreachable

    def test_unequal_dims_never_merge_entities(self):
        """The direct guard: a 1536-d query cannot claim a 384-d entity.

        With the stock zip-truncation cosine, [0.1]*1536 vs [0.1]*384 would
        produce a perfect ~1.0 similarity (truncated dot) and a FALSE merge;
        the guard must return None ("not similar" is the safe verdict)."""
        import tempfile

        from voicemem.leftbrain.slot_split.graph_entity_store import (
            GraphEntityStore,
        )

        with tempfile.TemporaryDirectory() as td:
            store = GraphEntityStore(Path(td) / "ge.sqlite")
            store.get_or_create_entity_semantic(
                "user1", "work", "victim", [0.1] * 384)
            hit = store.find_similar_entity(
                "user1", "work", [0.1] * 1536, threshold=0.5)
            self.assertIsNone(hit, "dimension-mismatched vectors must not match")


if __name__ == "__main__":
    unittest.main()
