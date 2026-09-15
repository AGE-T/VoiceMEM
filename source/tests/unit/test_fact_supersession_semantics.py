"""[VM-LOCAL-008 / v0.9.0 Phase 12 scenarios 4+5] Behavioural tests: the
fact-side supersession semantics that the trait side now mirrors.

The v0.8.1 fact layer already implements append + explicit supersession
(VM-LOCAL-008). These tests DEMONSTRATE the behaviour the v0.9.0 semantic
matrix requires of it (Phase 1 cases D and E):

  * explicit replacement: "lives in Budapest" → "moved to Vienna" —
    the new state is current, the old state stays recoverable, the
    relationship between the two is explicit (supersedes/superseded_by);
  * historical state: retrieval returns the current value first, the
    historical value remains queryable and carries superseded_by;
  * occurrence counting never fires on an UPDATE (the correction is not
    a confirmation).

A deterministic in-memory fake stands in for ``mem0.Memory`` ONLY at the
storage-SDK boundary (get/add/update/search) — the exact boundary the
integration suite mocks for the LLM decision. The supersession LOGIC under
test (Mem0BackendStore.update_memory + the search ordering) is the real
pinned vendor code.
"""

from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
for p in (str(REPO), str(VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore  # noqa: E402


_UNSET = object()          # distinguishes "kwarg absent" from None


class _FakeMem0:
    """In-memory stand-in for the mem0.Memory SDK surface the store uses:
    get / add / update / search / get_all. Deterministic, no network, no
    vectors — search returns rows in insertion order with fixed scores."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self._seq = 0

    def _new_id(self) -> str:
        return f"mem-{self._seq:04d}"

    # -- SDK surface ------------------------------------------------------

    def get(self, memory_id: str):
        return self.rows.get(memory_id)

    def add(self, *, messages, user_id, infer=False, metadata=None):
        text = str(messages[0].get("content", ""))
        mid = self._new_id()
        self._seq += 1
        row = {"id": mid, "memory": text, "user_id": user_id,
               "metadata": dict(metadata or {}), "created_at": "2026-09-14",
               "score": 1.0, "role": "user"}
        self.rows[mid] = row
        self.order.append(mid)
        return {"results": [{"id": mid, "memory": text, "event": "ADD"}]}

    def update(self, memory_id: str, *, data=None, metadata=None,
               expiration_date=_UNSET):
        row = self.rows.get(memory_id)
        if row is None:
            return False
        if metadata:
            row["metadata"].update(metadata)
        if data:
            row["memory"] = str(data)
        if expiration_date is not _UNSET:
            # mem0 semantics: update(expiration_date=None) CLEARS the field
            # (unarchive); a date string sets it (archive).
            row["expiration_date"] = expiration_date
        return True

    def _visible(self, row) -> bool:
        """mem0's real _payload_is_expired behaviour: search()/get_all()
        hide rows whose expiration_date is in the past (archive)."""
        exp = row.get("expiration_date")
        if not exp:
            return True
        from datetime import date
        return str(exp) >= date.today().isoformat()

    def search(self, query, *, filters=None, top_k=10, threshold=0.0):
        uid = (filters or {}).get("user_id")
        out = []
        for mid in self.order:
            row = self.rows[mid]
            if uid and row["user_id"] != uid:
                continue
            if not self._visible(row):
                continue
            out.append(dict(row))
        return {"results": out[:top_k]}

    def get_all(self, *, filters=None, top_k=10_000):
        return self.search("", filters=filters, top_k=top_k)["results"]


def _make_store():
    """Mem0BackendStore with the fake SDK — __init__ is bypassed (it would
    build a real mem0/Qdrant stack); only the attributes the methods under
    test touch are set."""
    store = object.__new__(Mem0BackendStore)
    store._mem0 = _FakeMem0()
    store._path = Path("/tmp/voicemem_semantics_fake")
    store._embedder = None
    return store


class FactSupersessionSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        self.mem = self.store._mem0

    def _seed(self, text, mid=None, metadata=None):
        self.mem.add(messages=[{"role": "user", "content": text}],
                     user_id="u1", infer=False, metadata=metadata or {})
        if mid is None:
            mid = self.mem.order[-1]
        return mid

    # ── scenario 4: explicit replacement ──────────────────────────────────

    def test_explicit_replacement_budapest_to_vienna(self):
        old = self._seed("Thomas lives in Budapest.")
        new = self.store.update_memory(
            old, "Thomas moved to Vienna.", user_id="u1",
            session_id=7, observed_at="2026-09-14")
        self.assertTrue(new)
        self.assertNotEqual(old, new, "UPDATE appends a NEW row (no in-place rewrite)")
        old_row, new_row = self.mem.rows[old], self.mem.rows[new]
        # the old observation: text intact, only the supersession mark added
        self.assertEqual(old_row["memory"], "Thomas lives in Budapest.")
        self.assertEqual(old_row["metadata"].get("superseded_by"), new)
        self.assertTrue(old_row["metadata"].get("superseded_at"))
        # the new observation: current value + explicit provenance
        self.assertEqual(new_row["memory"], "Thomas moved to Vienna.")
        self.assertEqual(new_row["metadata"].get("supersedes"), old)
        self.assertEqual(new_row["metadata"].get("session_id"), 7)
        self.assertEqual(new_row["metadata"].get("created_at"), "2026-09-14")

    def test_replacement_refused_on_missing_row(self):
        self.assertFalse(self.store.update_memory(
            "does-not-exist", "new text", user_id="u1"))

    def test_replacement_refused_blind_without_owner(self):
        old = self._seed("orphan fact")
        self.mem.rows[old]["user_id"] = ""
        self.assertFalse(self.store.update_memory(
            old, "new text", user_id=""))

    # ── scenario 5: historical state through retrieval ────────────────────

    def test_retrieval_prefers_current_state(self):
        old = self._seed("Thomas lives in Budapest.")
        new = self.store.update_memory(old, "Thomas moved to Vienna.",
                                       user_id="u1")
        hits = self.store.search("Where does Thomas live?", user_id="u1")
        self.assertEqual(hits[0].memory_id, new,
                         "the current value (Vienna) ranks first")
        self.assertEqual(hits[0].text, "Thomas moved to Vienna.")
        self.assertFalse(hits[0].superseded_by)
        budapest = [h for h in hits if h.memory_id == old]
        self.assertEqual(len(budapest), 1,
                         "the historical value remains queryable")
        self.assertEqual(budapest[0].superseded_by, new,
                         "the historical hit carries the supersession link")
        self.assertEqual(budapest[0].text, "Thomas lives in Budapest.")

    def test_temporal_coexistence_is_not_supersession(self):
        # Phase 1 case E: "previously owned a BMW" is a separate fact, not
        # an UPDATE of "owns a Renault" — nothing here calls update_memory,
        # and the store simply holds both rows as current.
        self._seed("Thomas owns a Renault.")
        self._seed("Thomas previously owned a BMW.")
        hits = self.store.search("cars", user_id="u1")
        for h in hits:
            self.assertFalse(h.superseded_by,
                             "temporal coexistence: both facts stay current")

    # ── Phase 15: archive / TTL interaction ──────────────────────────────

    def test_active_superseded_archived_chain_is_consistent(self):
        # active → superseded → archived: archiving the SUPERSEDED row
        # hides it from search (historical recoverability via search ends,
        # the data itself is retained and unarchive_memory restores it) —
        # the current row is unaffected, nothing is silently corrupted.
        old = self._seed("Thomas lives in Budapest.")
        new = self.store.update_memory(old, "Thomas moved to Vienna.",
                                       user_id="u1")
        self.assertTrue(self.store.archive_memory(old))
        hits = self.store.search("Where does Thomas live?", user_id="u1")
        ids = [h.memory_id for h in hits]
        self.assertIn(new, ids)
        self.assertNotIn(old, ids, "archived row is hidden by search")
        self.assertEqual(hits[0].memory_id, new)
        # unarchive restores the historical row (non-destructive)
        self.assertTrue(self.store.unarchive_memory(old))
        hits = self.store.search("Where does Thomas live?", user_id="u1")
        self.assertIn(old, [h.memory_id for h in hits])

    def test_archived_current_then_new_observation_starts_fresh(self):
        # archived → new observation: the cold row is invisible to the
        # resolver (mem0 hides expired rows from get_all too), so a new
        # observation of the same fact arrives as a fresh ADD — no
        # supersession link to the archived row, no silent resurrection.
        # This is the DOCUMENTED current behaviour (reactivation of an
        # archived row is not implemented; unarchive_memory is the manual
        # path). Nothing here corrupts the archived row.
        old = self._seed("Thomas lives in Budapest.")
        self.assertTrue(self.store.archive_memory(old))
        visible = self.store.existing_for_extractor("u1")
        self.assertEqual(visible, [],
                         "the archived row is invisible to the resolver")
        fresh = self._seed("Thomas lives in Budapest again.")
        hits = self.store.search("Where does Thomas live?", user_id="u1")
        self.assertEqual([h.memory_id for h in hits], [fresh])
        self.assertFalse(hits[0].superseded_by)
        # the archived row itself is untouched by the new observation
        self.assertNotIn("superseded_by", self.mem.rows[old]["metadata"])

    # ── occurrence semantics on UPDATE ────────────────────────────────────

    def test_update_is_not_a_confirmation(self):
        old = self._seed("Thomas lives in Budapest.")
        self.store.update_memory(old, "Thomas moved to Vienna.", user_id="u1")
        # the OLD row's occurrence metadata is untouched (no OCC-1 keys)
        self.assertNotIn("occurrence_count", self.mem.rows[old]["metadata"])
        # the NEW row starts without confirmation bookkeeping as well
        new = self.mem.rows[self.mem.order[-1]]
        self.assertNotIn("occurrence_count", new["metadata"])


if __name__ == "__main__":
    unittest.main()
