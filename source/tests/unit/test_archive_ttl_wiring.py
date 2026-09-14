"""[VM-LOCAL-014] Behavioural tests: cold-memory archive wiring + TTL read.

Covers (external audit finding F-G — "built but never wired / written but
never read"):
  * ArchiveColdMemories is invoked from the post-Ingest maintenance path,
    daily-throttled via the kv state (20h cooldown), non-fatal on failure
  * heartnote TTL is enforced at READ time (session: 1 day,
    short_term: 14 days, long_term: never) — rows skipped, never deleted
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
for p in (str(REPO), str(VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import unittest

from voicemem.utils.common import space as _space
from voicemem.rightbrain.store import RightBrainStore, ttl_expired
from voicemem.rightbrain.types import MemoryAnchor


def _days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
        timespec="seconds")


class FakeLeftBrain:
    """Records ArchiveColdMemories invocations; returns a canned result."""

    def __init__(self, result=None, raise_exc: Exception | None = None):
        self.calls: list[dict] = []
        self._result = result or {"status": "nothing_to_archive", "archived": []}
        self._raise = raise_exc

    def ArchiveColdMemories(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise:
            raise self._raise
        return self._result


class ArchiveWiringTests(unittest.TestCase):
    """The orchestrator maintenance path (unbound method on a minimal self)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="archive_wire_")
        self.root = self._tmp.name
        # ensure a clean kv state for the throttle key
        _space.kv_set(self.root, "archive_cold_last_run", "")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, fake_left):
        from voicemem.orchestrator import Orchestrator
        fake_self = SimpleNamespace(_memory_root=self.root, _left=fake_left)
        Orchestrator._maybe_archive_cold_memories(fake_self)

    def test_first_run_invokes_archive(self):
        left = FakeLeftBrain(result={"status": "archived", "archived": ["m1", "m2"]})
        self._run(left)
        self.assertEqual(len(left.calls), 1)
        self.assertEqual(left.calls[0].get("min_age_days"), 30.0)
        # throttle state was persisted
        self.assertTrue(_space.kv_get(self.root, "archive_cold_last_run", ""))

    def test_throttle_blocks_second_run_within_20h(self):
        left = FakeLeftBrain()
        self._run(left)
        self.assertEqual(len(left.calls), 1)
        self._run(left)                       # immediate second run
        self.assertEqual(len(left.calls), 1, "kv throttle must suppress re-run")

    def test_run_after_cooldown_invokes_again(self):
        left = FakeLeftBrain()
        self._run(left)
        # backdate the throttle state by 25 hours
        stale = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        _space.kv_set(self.root, "archive_cold_last_run", stale)
        self._run(left)
        self.assertEqual(len(left.calls), 2)

    def test_archive_failure_is_non_fatal(self):
        left = FakeLeftBrain(raise_exc=RuntimeError("boom"))
        try:
            self._run(left)                  # must not raise
        except Exception as exc:             # pragma: no cover
            self.fail(f"archive failure must not propagate: {exc}")
        self.assertEqual(len(left.calls), 1)

    def test_maintenance_thread_wired_into_ingest(self):
        """The post-Ingest block spawns the archive maintenance thread."""
        import inspect
        from voicemem import orchestrator as orch_mod
        src = inspect.getsource(orch_mod.Orchestrator)
        self.assertIn("_maybe_archive_cold_memories, daemon=True", src,
                     "the Ingest maintenance block must start the archive thread")


class TtlExpiryTests(unittest.TestCase):
    def test_horizons(self):
        self.assertTrue(ttl_expired("session", _days_ago(1.5)))
        self.assertFalse(ttl_expired("session", _days_ago(0.5)))
        self.assertTrue(ttl_expired("short_term", _days_ago(15.0)))
        self.assertFalse(ttl_expired("short_term", _days_ago(13.0)))
        self.assertFalse(ttl_expired("long_term", _days_ago(3650.0)))
        self.assertFalse(ttl_expired("unknown_ttl", _days_ago(3650.0)))
        self.assertFalse(ttl_expired("session", ""))
        self.assertFalse(ttl_expired("session", "not-a-date"))


class TtlReadFilterTests(unittest.TestCase):
    """Expired heartnotes are skipped at retrieval; rows survive in the DB."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="rb_ttl_")
        self.store = RightBrainStore(
            os.path.join(self._tmp.name, "space.sqlite"))
        anchor = MemoryAnchor(anchor_type="user", anchor_id="u1",
                              role="topic", weight=1.0, confidence=1.0)
        # fresh long-term note (must be retrievable)
        m_keep = self.store.upsert_memory(
            "u1", "heartnote", "long-term note", ttl="long_term")
        # session note 2 days old (must be skipped at read)
        m_session = self.store.upsert_memory(
            "u1", "heartnote", "expired session note", ttl="session",
            created_at=_days_ago(2.0))
        # short_term note 20 days old (must be skipped at read)
        m_short = self.store.upsert_memory(
            "u1", "heartnote", "expired short note", ttl="short_term",
            created_at=_days_ago(20.0))
        for m in (m_keep, m_session, m_short):
            self.store.link_anchor(m.id, "u1", anchor)

    def tearDown(self):
        self._tmp.cleanup()

    def _ids(self, rows):
        return {r.id for r in rows}

    def test_search_by_anchors_skips_expired(self):
        rows = self.store.search_by_anchors("u1", [
            MemoryAnchor(anchor_type="user", anchor_id="u1", role="topic")])
        contents = {r.content for r in rows}
        self.assertIn("long-term note", contents)
        self.assertNotIn("expired session note", contents)
        self.assertNotIn("expired short note", contents)

    def test_rows_survive_non_destructively(self):
        """Expired rows are filtered at retrieval, NOT deleted from the DB."""
        with self.store._conn() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM right_brain_memories WHERE user_id='u1'"
            ).fetchone()[0]
        self.assertEqual(n, 3, "all three rows must physically exist")
        # get_all (brain-map / maintenance view) still sees everything
        self.assertEqual(len(self.store.get_all("u1")), 3)

    def test_fresh_session_note_still_retrievable(self):
        m = self.store.upsert_memory(
            "u1", "heartnote", "fresh session note", ttl="session")
        self.store.link_anchor(m.id, "u1", MemoryAnchor(
            anchor_type="user", anchor_id="u1", role="topic"))
        rows = self.store.search_by_anchors("u1", [
            MemoryAnchor(anchor_type="user", anchor_id="u1", role="topic")])
        contents = {r.content for r in rows}
        self.assertIn("fresh session note", contents)


if __name__ == "__main__":
    unittest.main()
