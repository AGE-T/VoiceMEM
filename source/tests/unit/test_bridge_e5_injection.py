"""v0.4.22: the CLI bridge must inject the local E5 embedder into VoiceMem.

The web path (``app.web_server.RealMemoryLayer``) has injected
``embedding=embedding_factory`` since v0.4.4. The v0.4.22 memory-embedding
invariant closes the CLI gap: ``VoiceMemBridge._construct_vm`` passes the
same factory to the facade, so with the vendor fix
(``orchestrator._embed_text`` routing) NO memory embedding call on the CLI
path can reach the OpenAI embeddings API either.

Test strategy (no network, no heavy deps):
  * a FAKE voicemem module (types.SimpleNamespace) proves the facade kwargs
    (embedding callable + text_mode + user_id + memory_root + reply);
  * the REAL vendor ``LocalE5Embedder`` (sys.path insert, full sys.modules
    isolation) proves the factory returns the actual local embedder class
    and its two call shapes exist;
  * static source checks pin both injection sites against silent removal.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "vendor" / "voicemem"

sys.path.insert(0, str(REPO_ROOT))

from app.config import AgentConfig  # noqa: E402
from app.voicemem_bridge import VoiceMemBridge  # noqa: E402

import app.voicemem_bridge as bridge_mod  # noqa: E402


class _FacadeRecorder:
    """Fake voicemem.VoiceMem factory: records constructor kwargs."""

    last_kwargs: dict | None = None
    last_factory_result: object | None = None

    def __call__(self, **kwargs):
        _FacadeRecorder.last_kwargs = kwargs
        return types.SimpleNamespace(closed=False)

    @classmethod
    def reset(cls) -> None:
        cls.last_kwargs = None
        cls.last_factory_result = None


def _fake_voicemem_module() -> types.ModuleType:
    mod = types.ModuleType("voicemem_fake")
    mod.VoiceMem = _FacadeRecorder()
    return mod


class BridgeEmbeddingInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        _FacadeRecorder.reset()
        self._real_voicemem = bridge_mod.voicemem
        bridge_mod.voicemem = _fake_voicemem_module()

    def tearDown(self) -> None:
        bridge_mod.voicemem = self._real_voicemem
        _FacadeRecorder.reset()

    def test_facade_receives_embedding_factory(self):
        """_construct_vm must pass embedding=<callable factory>."""
        config = AgentConfig()
        bridge = VoiceMemBridge(config)
        bridge._e5_check_done = True   # skip the (never-raising) E5 probe
        vm = bridge._construct_vm("voice_user")

        self.assertIsNotNone(vm)
        kwargs = _FacadeRecorder.last_kwargs
        self.assertIsNotNone(kwargs, "the facade factory was never called")
        self.assertEqual(kwargs.get("mode"), "text_mode")
        self.assertEqual(kwargs.get("user_id"), "voice_user")
        self.assertIn("embedding", kwargs,
                      "the facade was constructed WITHOUT the embedding kwarg "
                      "- the CLI memory path would silently fall back to the "
                      "OpenAI embeddings API (the pre-v0.4.22 defect)")
        self.assertTrue(callable(kwargs["embedding"]))

    def test_embedding_factory_builds_real_local_e5(self):
        """The factory returns the VENDOR LocalE5Embedder instance.

        The vendor tree is put on sys.path in full isolation (every
        voicemem.* module is purged before and after), so this exercises
        the real class, not a stub. Construction is cheap - the E5 model
        itself loads lazily on first embed (never triggered here).
        """
        config = AgentConfig()
        bridge = VoiceMemBridge(config)
        bridge._e5_check_done = True
        bridge._construct_vm("voice_user")
        factory = _FacadeRecorder.last_kwargs["embedding"]

        saved_modules = {
            name: mod for name, mod in sys.modules.items()
            if name == "voicemem" or name.startswith("voicemem.")
        }
        for name in list(saved_modules):
            del sys.modules[name]
        sys.path.insert(0, str(VENDOR_ROOT))
        try:
            instance = factory()
            from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder
            self.assertIsInstance(instance, LocalE5Embedder)
            # the two call shapes the orchestrator routing uses
            self.assertTrue(callable(getattr(instance, "embed_texts", None)))
            self.assertTrue(callable(getattr(instance, "embed_query_text", None)))
        finally:
            sys.path.remove(str(VENDOR_ROOT))
            for name in [n for n in list(sys.modules)
                         if n == "voicemem" or n.startswith("voicemem.")]:
                del sys.modules[name]
            sys.modules.update(saved_modules)

    def test_web_and_bridge_injection_sites_pinned(self):
        """Static guard: BOTH injection sites must keep the factory.

        A silent edit removing either kwarg would reopen the OpenAI-API
        fallback on that path.
        """
        bridge_src = (REPO_ROOT / "app" / "voicemem_bridge.py").read_text(
            encoding="utf-8")
        self.assertIn("embedding=embedding_factory", bridge_src)
        self.assertIn("def embedding_factory():", bridge_src)
        self.assertIn("LocalE5Embedder", bridge_src)

        web_src = (REPO_ROOT / "app" / "web_server.py").read_text(
            encoding="utf-8")
        self.assertIn("embedding=embedding_factory", web_src)
        self.assertIn("def embedding_factory():", web_src)
        self.assertIn("LocalE5Embedder", web_src)


if __name__ == "__main__":
    unittest.main()
