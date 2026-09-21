"""Regression tests: the write-path decision guards (PROVENANCE fix #7).

v0.4.22 controlled-vendor fix (upstream analogue ab97cf5 for the UPDATE
half; the DELETE guard has no upstream analogue - upstream applies LLM
DELETEs unguarded on every revision):

  * UPDATE   -> downgraded to ADD by default (history preserved; the
                answering model arbitrates by date). Opt-in to the old
                overwrite behaviour with VOICEMEM_APPLY_UPDATE=1.
  * DELETE   -> NOT executed by default; the suppression is logged; text,
                embedding and provenance stay intact. Opt-in to real
                deletion with VOICEMEM_APPLY_DELETE=1.

Both guards are deterministic protection against a single misjudged LLM
conflict-resolver decision irreversibly destroying stored memory.

Driven behaviourally against the REAL ingest_voice_input with fakes for
the registry/messages/extractor/resolver/repo (the module's heavy deps
are lazy; the vendor tree is on sys.path with full isolation).
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "vendor" / "voicemem"

sys.path.insert(0, str(REPO_ROOT))


def _purge_voicemem_modules() -> None:
    for name in [n for n in list(sys.modules)
                 if n == "voicemem" or n.startswith("voicemem.")]:
        del sys.modules[name]


def setUpModule() -> None:
    _purge_voicemem_modules()
    sys.path.insert(0, str(VENDOR_ROOT))


def tearDownModule() -> None:
    if str(VENDOR_ROOT) in sys.path:
        sys.path.remove(str(VENDOR_ROOT))
    _purge_voicemem_modules()


class _Repo:
    """Fake left-brain repository: records every destructive call."""

    def __init__(self) -> None:
        self.updated: list[tuple] = []
        self.deleted: list[str] = []
        self.appended: list = []
        self.existing = [{"id": "mem-old-1", "text": "favourite restaurant is A"}]

    def search(self, query, user_id=None, top_k=5):
        class _Hit:
            memory_id = "mem-old-1"
            text = "favourite restaurant is A"
        return [_Hit()]

    def append_extracted(self, extracted, user_id=None, extra_metadata=None):
        self.appended = list(extracted)
        return [f"mem-new-{i}" for i in range(len(extracted))]

    def update_memory(self, memory_id, text, **kw):
        self.updated.append((memory_id, text))

    def delete_memory(self, memory_id):
        self.deleted.append(memory_id)


class _Extractor:
    """Fake extractor: one atomic fact per call."""

    def extract(self, new_messages=None, existing_memories=None,
                observation_date="", current_date="", **kw):
        from voicemem.leftbrain.extract_facts_openai import (
            ExtractedAdditiveMemory,
        )
        return [ExtractedAdditiveMemory(
            local_id="f1", text="favourite restaurant is B",
            attributed_to="user")]


class _Registry:
    """Fake voiceprint registry: no bindings."""

    def display_name(self, voiceprint_id):
        return None

    def entity_id(self, voiceprint_id):
        return None

    def all_display_names(self):
        return []


def _make_vi():
    from voicemem.utils.common.voice_input import VoiceInput
    return VoiceInput.from_dict({
        "id": "turn-1",
        "time_stamp": {"begin": "2026-09-09 10:00:00",
                       "end": "2026-09-09 10:00:05"},
        "slots": [],
        "contents": [{
            "sub_id": "s1",
            "time_start": "2026-09-09 10:00:00",
            "time_end": "2026-09-09 10:00:05",
            "sentence": "my favourite restaurant is B now",
            "voiceprint_id": "vp-1",
        }],
    })


class _FakeResolver:
    """Scripted ConflictResolver stand-in (the LLM call is not under test)."""

    events: list[types.SimpleNamespace] = []

    def __init__(self, single_valued_rule=True):
        pass

    def resolve(self, facts, existing, speaker_name=None):
        return list(_FakeResolver.events)


_GUARD_ENV = {
    "VOICEMEM_APPLY_UPDATE": "",
    "VOICEMEM_APPLY_DELETE": "",
    "VOICEMEM_ALWAYS_ADD": "",
    "VOICEMEM_INGEST_RAW_FALLBACK": "",
}


class WritePathGuardTests(unittest.TestCase):

    def _ingest(self, event, memory_id="mem-old-1", text=None):
        import voicemem.utils.common.voice_input as vi_mod
        import voicemem.leftbrain.extract_facts_openai as ef_mod
        repo = _Repo()
        _FakeResolver.events = [types.SimpleNamespace(
            event=event, memory_id=memory_id, text=text)]
        with patch.dict(os.environ, _GUARD_ENV), \
             patch.object(vi_mod, "voice_input_to_messages",
                          return_value=[{"role": "user",
                                         "content": "my favourite restaurant is B now"}]), \
             patch.object(ef_mod, "ConflictResolver", _FakeResolver):
            result = vi_mod.ingest_voice_input(
                _make_vi(), "user1",
                registry=_Registry(), repo=repo, extractor=_Extractor())
        return repo, result

    # ---- UPDATE ----------------------------------------------------------

    def test_update_downgraded_to_add_by_default(self):
        """PROVENANCE #7a: the resolver's UPDATE never overwrites stored
        memory unless VOICEMEM_APPLY_UPDATE=1. The new fact is appended
        instead - both dated rows stay, the answering model arbitrates."""
        repo, result = self._ingest("UPDATE", "mem-old-1",
                                    "favourite restaurant is B")
        self.assertEqual(repo.updated, [],
                         "update_memory was called without VOICEMEM_APPLY_UPDATE=1")
        self.assertEqual(repo.deleted, [])
        # the new fact was appended (ADD downgraded), not silently dropped
        self.assertEqual(len(repo.appended), 1)
        self.assertEqual(repo.appended[0].text, "favourite restaurant is B")
        self.assertEqual(result.facts_count, 1)

    def test_update_applies_with_explicit_env_opt_in(self):
        """The old overwrite behaviour stays reachable: explicit opt-in."""
        import voicemem.utils.common.voice_input as vi_mod
        import voicemem.leftbrain.extract_facts_openai as ef_mod
        repo = _Repo()
        _FakeResolver.events = [types.SimpleNamespace(
            event="UPDATE", memory_id="mem-old-1",
            text="favourite restaurant is B")]
        env = dict(_GUARD_ENV, VOICEMEM_APPLY_UPDATE="1")
        with patch.dict(os.environ, env), \
             patch.object(vi_mod, "voice_input_to_messages",
                          return_value=[{"role": "user", "content": "x"}]), \
             patch.object(ef_mod, "ConflictResolver", _FakeResolver):
            vi_mod.ingest_voice_input(_make_vi(), "user1", registry=_Registry(),
                                      repo=repo, extractor=_Extractor())
        self.assertEqual(repo.updated,
                         [("mem-old-1", "favourite restaurant is B")])
        self.assertEqual(repo.appended, [])   # nothing duplicated

    # ---- DELETE ----------------------------------------------------------

    def test_delete_not_applied_by_default(self):
        """PROVENANCE #7b: the resolver's DELETE is NOT executed by default -
        the stored memory (text, embedding, provenance) survives; the
        suppression is observable in the log."""
        import voicemem.utils.common.voice_input as vi_mod
        import voicemem.leftbrain.extract_facts_openai as ef_mod
        import logging
        repo = _Repo()
        _FakeResolver.events = [types.SimpleNamespace(
            event="DELETE", memory_id="mem-old-1", text=None)]
        with patch.dict(os.environ, _GUARD_ENV), \
             patch.object(vi_mod, "voice_input_to_messages",
                          return_value=[{"role": "user", "content": "x"}]), \
             patch.object(ef_mod, "ConflictResolver", _FakeResolver), \
             self.assertLogs("voicemem.utils.common.voice_input",
                             level="WARNING") as logs:
            result = vi_mod.ingest_voice_input(
                _make_vi(), "user1", registry=_Registry(), repo=repo,
                extractor=_Extractor())
        self.assertEqual(repo.deleted, [],
                         "delete_memory was called without VOICEMEM_APPLY_DELETE=1")
        self.assertEqual(repo.updated, [])
        self.assertTrue(
            any("DELETE decision(s) NOT applied" in rec.getMessage()
                for rec in logs.records),
            "the suppressed DELETE must be logged (observable failure)")
        self.assertIsNotNone(result)

    def test_delete_applies_with_explicit_env_opt_in(self):
        """Real deletion stays reachable for explicit operator intent."""
        import voicemem.utils.common.voice_input as vi_mod
        import voicemem.leftbrain.extract_facts_openai as ef_mod
        repo = _Repo()
        _FakeResolver.events = [types.SimpleNamespace(
            event="DELETE", memory_id="mem-old-1", text=None)]
        env = dict(_GUARD_ENV, VOICEMEM_APPLY_DELETE="1")
        with patch.dict(os.environ, env), \
             patch.object(vi_mod, "voice_input_to_messages",
                          return_value=[{"role": "user", "content": "x"}]), \
             patch.object(ef_mod, "ConflictResolver", _FakeResolver):
            vi_mod.ingest_voice_input(_make_vi(), "user1", registry=_Registry(),
                                      repo=repo, extractor=_Extractor())
        self.assertEqual(repo.deleted, ["mem-old-1"])

    def test_add_still_appends_normally(self):
        """The guard changed nothing about the plain ADD path."""
        repo, result = self._ingest("ADD", "f1", "favourite restaurant is B")
        self.assertEqual(repo.updated, [])
        self.assertEqual(repo.deleted, [])
        self.assertEqual(len(repo.appended), 1)

    def test_guards_are_pinned_in_source(self):
        """Static guard: the env-gated routing must survive future edits."""
        src = (VENDOR_ROOT / "voicemem" / "utils" / "common" /
               "voice_input.py").read_text(encoding="utf-8")
        self.assertIn('VOICEMEM_APPLY_UPDATE', src)
        self.assertIn('VOICEMEM_APPLY_DELETE', src)
        self.assertIn("CONTROLLED-VENDOR LOCAL FIX", src)
        self.assertIn("suppressed_deletes", src)


if __name__ == "__main__":
    unittest.main()
