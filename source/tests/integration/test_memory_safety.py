"""TASK 1.5 — MEMORY SAFETY GATE: real behavioural regression tests.

The independent Claude audit (v0.5.0, 57fb6a7) found that the memory engine
could silently destroy information:

* CD-1 (P0): an UNGATED LLM decision hard-deletes right-brain emotional
  memory on the Ingest hot path (run_cleanup).
* CD-2 (P0): UPDATE is a destructive in-place overwrite — a previous fact
  becomes unrecoverable the moment an LLM ConflictResolver says UPDATE.
* F-H (P2): a permitted left-brain delete leaves stale graph truth behind.
* F-F (P1): none of these had real behavioural tests.

These tests prove the TASK 1.5 safety patches against the REAL pinned vendor
(e8384e0 + VM-LOCAL-007/008/009) with the REAL mem0/Qdrant store, the REAL
local E5 embeddings, the REAL SQLite graph layers — and a deterministic
mock LLM that stands in ONLY for the LLM decision (the exact boundary the
audit demanded: mock the LLM decision, never the storage behaviour being
proven).

Coverage (Phase 8 minimum):

  1. right-brain delete gate — default deny / =0 deny / =1 can delete
     (plus the anchor-link cascade when deletion is permitted)
  2. UPDATE history preservation — the restaurant A -> B scenario:
     B = current value, A recoverable, relationship explicit,
     no destructive overwrite, provenance preserved
  3. supersession visible through search (current-first ranking; the
     historical hit survives and carries superseded_by)
  4. left-brain delete gate on the real store — default deny, then
     permitted delete cascades every graph row (no orphans)
  5. production import resolution — voicemem resolves to the controlled
     vendor tree, pin + patch ledger verified, shadowing detected
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor" / "voicemem"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(VENDOR))

E5_DIR = (
    REPO / "models" / "hf" / "models--intfloat--multilingual-e5-small"
    / "snapshots" / "614241f622f53c4eeff9890bdc4f31cfecc418b3"
)

MOCK_PORT = 8791
USER_ID = "safety_gate_user"

# ---------------------------------------------------------------------------
# Controlled scenario data
# ---------------------------------------------------------------------------

# The Phase 4 restaurant scenario (Hungarian, matching the product language).
OBS_A = "A kedvenc éttermem az Arany Kanna vendéglő."
OBS_B = "A kedvenc éttermem megváltozott, most a Bistro Buda a kedvencem."
FACT_A = "A felhasználó kedvenc étterme az Arany Kanna vendéglő."
FACT_B = "A felhasználó kedvenc étterme mostantól a Bistro Buda."

# Emotional utterances -> heartnotes (each carries an emotion so the
# right-brain write() path fires for every one of them).
RB_UTTERANCES = [
    ("Ma igazán jó napom volt, sikerült a vizsgám.", "joy"),
    ("Nagyon büszke vagyok a fiamra, megnyerte a versenyt.", "joy"),
    ("Kicsit szomorú vagyok, elköltözött a legjobb barátom.", "sad"),
    ("Friss feltöltéssel indítottam a reggelem, nagyon jó érzés.", "joy"),
    ("Aggódom a anyukám egészsége miatt.", "anxiety"),
    ("Ma dühös voltam a főnökömre a megbeszélés miatt.", "anger"),
    ("Meglepetésajándékot kaptam a páromtól, nagyon örülök.", "joy"),
    ("Fáradt vagyok, de elégedett a heti munkámmal.", "calm"),
    ("Kicsit ideges vagyok a holnapi prezentáció miatt.", "anxiety"),
    ("Nagyon jó volt a családi hétvége a Balatonnál.", "joy"),
    ("Új hobbit kezdtem, kerámiázni tanulok, imádom.", "joy"),
    ("Ma meditáltam először, nyugodtabbnak érzem magam.", "calm"),
]

_FACT_SLOT = "daily_life"


def _extraction_payload(fact: str, emotion: str = "",
                        entities: list[str] | None = None) -> dict[str, Any]:
    """Deterministic merged-extraction response for one utterance."""
    item: dict[str, Any] = {"text": fact, "slot": _FACT_SLOT}
    if entities:
        item["entities"] = entities
    return {"memory": [item], "emotion": emotion, "traits": []}


# ---------------------------------------------------------------------------
# Deterministic keyword-routed mock LLM (the ONLY thing mocked: the LLM
# decision. Storage, embedding, graph, ranking: all real.)
# ---------------------------------------------------------------------------

def _route(messages: list[dict[str, Any]]) -> dict[str, Any]:
    system = " ".join(str(m.get("content", "")) for m in messages
                      if m.get("role") == "system")
    user = " ".join(str(m.get("content", "")) for m in messages
                    if m.get("role") != "system")
    everything = system + "\n" + user

    # ── right-brain cleanup decision (CD-1 test surface) ──────────────────
    # The cleanup prompt lists heartnotes as "[i] ID:xxxxxxxx | 情感:… | …".
    # Deterministic decision: request deletion of EVERY listed heartnote.
    if "记忆清洁助手" in system or "记忆清洁助手" in everything:
        ids: list[str] = []
        for tok in everything.replace(" ", "\n").split("\n"):
            if "ID:" in tok:
                short = tok.split("ID:")[1].split("|")[0].strip()
                if len(short) >= 8:
                    ids.append(short[:8])
        return {"delete_ids": ids, "supersede": []}

    # ── conflict resolution (CD-2 test surface) ────────────────────────────
    # The resolver prompt embeds "You are a smart memory manager..." in the
    # single user message. Only the OBS_B turn requests an UPDATE (id "0" is
    # the anti-hallucination index of the single existing memory, mapped by
    # the vendor back to the real UUID); every other turn resolves to no-op
    # so the ADD path stays exercised.
    if "smart memory manager" in user:
        if "Bistro Buda" in user and FACT_B not in user:
            # new facts section carries the raw utterance; respond UPDATE
            return {"memory": [{"event": "UPDATE", "id": "0", "text": FACT_B}]}
        if "Bistro Buda" in user:
            return {"memory": [{"event": "UPDATE", "id": "0", "text": FACT_B}]}
        return {"memory": []}

    # ── merged fact extraction (utterance-keyed) ───────────────────────────
    if OBS_A in everything:
        return _extraction_payload(FACT_A, entities=["Arany Kanna"])
    if OBS_B in everything:
        return _extraction_payload(FACT_B, entities=["Bistro Buda"])
    for utt, emo in RB_UTTERANCES:
        if utt in everything:
            return _extraction_payload(
                f"A felhasználó mondta: {utt}", emotion=emo)
    # LB delete-gate scenario facts
    if "A legjobb barátom Bence" in everything:
        return _extraction_payload(
            "A felhasználó legjobb barátja Bence.",
            entities=["Bence"])
    if "Bence németországi" in everything:
        return _extraction_payload(
            "A felhasználó legjobb barátja Bence Németországban él.",
            entities=["Bence", "Németország"])

    # ── left-brain LLM slot re-tagging ─────────────────────────────────────
    if "用户说了这句话" in user:
        return {"slots": [_FACT_SLOT]}

    # ── right-brain inner-OS generation ────────────────────────────────────
    if "empathetic AI assistant" in system:
        return {"text": "user is processing feelings"}

    # ── standalone trait extraction ────────────────────────────────────────
    if '"items"' in everything and "喜好与厌恶" in everything:
        return {"items": []}

    # default: additive empty shape
    return {"memory": [], "emotion": "", "traits": []}


class _MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = _route(body.get("messages") or [])
        self._json(200, {
            "id": f"chatcmpl-t15-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-llm"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": json.dumps(payload, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] in ("/health", "/v1/models"):
            self._json(200, {"status": "ok", "models": ["mock-llm"]})
            return
        self._json(404, {"error": "not found"})

    def _json(self, code: int, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


_MOCK_SRV: ThreadingHTTPServer | None = None


def _mock_server() -> ThreadingHTTPServer:
    global _MOCK_SRV
    if _MOCK_SRV is None:
        _MOCK_SRV = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        threading.Thread(target=_MOCK_SRV.serve_forever, daemon=True).start()
    return _MOCK_SRV


# ---------------------------------------------------------------------------
# Module-level env (must precede any voicemem import: local_e5_embedder
# resolves VOICEMEM_E5_MODEL at import time)
# ---------------------------------------------------------------------------

_SAVED_ENV: dict[str, str | None] = {}


def setUpModule() -> None:  # noqa: N802
    global _SAVED_ENV
    srv = _mock_server()
    port = srv.server_address[1]
    _touched = (
        "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
        "OPENAI_CHAT_MODEL", "VOICEMEM_E5_MODEL", "VOICEMEM_EMBED_DIM",
        "VOICEMEM_VERBOSE", "VOICEMEM_ALLOW_MEMORY_DELETE",
    )
    _SAVED_ENV = {k: os.environ.get(k) for k in _touched}
    os.environ.update({
        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "OPENAI_API_KEY": "safety-gate-not-needed",
        "OPENAI_MODEL": "mock-llm",
        "OPENAI_CHAT_MODEL": "mock-llm",
        "VOICEMEM_E5_MODEL": str(E5_DIR),
        "VOICEMEM_EMBED_DIM": "384",
        "VOICEMEM_VERBOSE": "0",
    })
    os.environ.pop("VOICEMEM_ALLOW_MEMORY_DELETE", None)


def tearDownModule() -> None:  # noqa: N802
    # restore the environment EXACTLY: other test modules in the same
    # discovery run read these at runtime (e.g. the streaming E2E asserts
    # the model name on the wire) — this module must not leak its mock env
    for k, v in _SAVED_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _release_vm(vm: Any) -> None:
    """Free the heavy resources of a test VoiceMem (embedded qdrant +
    cached clients) so later test packages in the same gate process start
    from a low memory baseline (the sandbox RSS cap is hard)."""
    import gc
    try:
        store = vm._o._get_repo()._vector_store
        mem0 = getattr(store, "_mem0", None)
        for closer_name in ("close", "aclose", "_close"):
            closer = getattr(mem0, closer_name, None)
            if callable(closer):
                try:
                    closer()
                    break
                except Exception:
                    pass
    except Exception:
        pass
    try:
        vm.flush()
    except Exception:
        pass
    gc.collect()


def _new_vm() -> Any:
    """A fresh VoiceMem on a fresh temp memory_root (real E5 + real store).

    The single model instance is shared (lru_cache in the vendor) so only
    the first construction pays the E5 load.
    """
    from voicemem import VoiceMem
    from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder
    root = tempfile.mkdtemp(prefix="t15_safety_")
    return VoiceMem(mode="text_mode", memory_root=root, user_id=USER_ID,
                    embedding=lambda: LocalE5Embedder())


def _mem0(vm: Any) -> Any:
    return vm._o._get_repo()._vector_store._mem0


def _repo(vm: Any) -> Any:
    return vm._o._get_repo()


def _rb_store(vm: Any) -> Any:
    return vm._o._right._rb_repo()._store


def _heartnotes(vm: Any) -> list[Any]:
    return [m for m in _rb_store(vm).get_all(USER_ID)
            if m.memory_class == "heartnote"]


def _anchor_count(vm: Any) -> int:
    import sqlite3
    with sqlite3.connect(_rb_store(vm)._path) as conn:
        return int(conn.execute(
            "SELECT COUNT(*) FROM right_brain_anchor_links").fetchone()[0])


def _graph_refs(db_path: Path, memory_id: str) -> dict[str, int]:
    """Count every graph row that references the memory id (orphan audit)."""
    import sqlite3
    out: dict[str, int] = {}
    with sqlite3.connect(str(db_path)) as conn:
        for table in ("memories", "entity_memory_links", "memory_tags",
                      "graph_entity_memories"):
            try:
                col = "id" if table == "memories" else "memory_id"
                out[table] = int(conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {col}=?",
                    (memory_id,)).fetchone()[0])
            except Exception:
                out[table] = -1  # table not present in this schema
    return out


def _space_db(vm: Any) -> Path:
    # the space sqlite is named after the memory_root dir (space semantics);
    # the cognitive store holds the authoritative resolved path
    return Path(vm._o._get_repo()._cognitive_store._path)


# ===========================================================================
# 1. Right-brain delete gate (CD-1 / VM-LOCAL-007)
# ===========================================================================

class RightBrainDeleteGateTests(unittest.TestCase):
    """run_cleanup with an LLM that DETERMINISTICALLY requests deletion.

    The vendor's real SQLite right-brain store, real anchor links, real
    cleanup code path — only the LLM decision is the mock.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        for utt, emo in RB_UTTERANCES:
            cls.vm.ingest(utt, observed_at="2026-09-10")
        cls.notes_before = len(_heartnotes(cls.vm))
        cls.anchors_before = _anchor_count(cls.vm)
        assert cls.notes_before >= 10, (
            f"harness setup failed: only {cls.notes_before} heartnotes "
            "(run_cleanup requires >= 10)")

    def _env(self, value: str | None):
        if value is None:
            os.environ.pop("VOICEMEM_ALLOW_MEMORY_DELETE", None)
        else:
            os.environ["VOICEMEM_ALLOW_MEMORY_DELETE"] = value

    def test_01_default_env_blocks_delete(self):
        """Default (env absent): the LLM delete decision is DENIED."""
        self._env(None)
        before = len(_heartnotes(self.vm))
        anchors = _anchor_count(self.vm)
        self.vm._o._right.run_cleanup()
        after = len(_heartnotes(self.vm))
        self.assertEqual(after, before,
                         "CD-1 REGRESSION: right-brain heartnotes were deleted "
                         "with VOICEMEM_ALLOW_MEMORY_DELETE unset (default must deny)")
        self.assertEqual(_anchor_count(self.vm), anchors)

    def test_02_explicit_zero_blocks_delete(self):
        """Explicit VOICEMEM_ALLOW_MEMORY_DELETE=0: still denied."""
        self._env("0")
        before = len(_heartnotes(self.vm))
        self.vm._o._right.run_cleanup()
        self.assertEqual(len(_heartnotes(self.vm)), before,
                         "CD-1 REGRESSION: =0 must deny the right-brain delete")

    def test_03_explicit_one_deletes_with_cascade(self):
        """Explicit opt-in (=1): deletion works AND removes anchor links
        (no graph/reference orphans are left behind)."""
        self._env("1")
        before = len(_heartnotes(self.vm))
        target = _heartnotes(self.vm)[0]
        import sqlite3
        with sqlite3.connect(_rb_store(self.vm)._path) as conn:
            target_anchors = int(conn.execute(
                "SELECT COUNT(*) FROM right_brain_anchor_links WHERE right_memory_id=?",
                (target.id,)).fetchone()[0])
        self.vm._o._right.run_cleanup()
        after = len(_heartnotes(self.vm))
        self.assertLess(after, before,
                        "explicit opt-in failed to delete (harness or gate bug)")
        remaining_ids = {m.id for m in _heartnotes(self.vm)}
        self.assertNotIn(target.id, remaining_ids)
        # the deleted heartnote must leave NO anchor rows behind
        with sqlite3.connect(_rb_store(self.vm)._path) as conn:
            orphans = int(conn.execute(
                "SELECT COUNT(*) FROM right_brain_anchor_links WHERE right_memory_id=?",
                (target.id,)).fetchone()[0])
        self.assertEqual(orphans, 0,
                         "permitted RB delete left orphaned anchor links")
        self.assertGreaterEqual(target_anchors, 1,
                                "harness setup: the target heartnote should "
                                "have anchor links (so the cascade proof is "
                                "meaningful)")
        self._env(None)

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("VOICEMEM_ALLOW_MEMORY_DELETE", None)

        _release_vm(cls.vm)

# ===========================================================================
# 2/3. Non-destructive UPDATE (CD-2 / VM-LOCAL-008) — the restaurant scenario
# ===========================================================================

class UpdateHistoryPreservationTests(unittest.TestCase):
    """Phase 4 scenario:

    Observation 1: the user's favourite restaurant is A (Arany Kanna).
    Observation 2: the favourite restaurant changed to B (Bistro Buda).

    The deterministic mock LLM resolver emits a real UPDATE decision on the
    REAL ingest path. Proofs:

      * B is represented as the current value (rank 1 in search)
      * A remains recoverable (row exists, text unchanged, findable in search)
      * the A -> B relationship is explicit (superseded_by / supersedes)
      * no destructive overwrite occurred anywhere (mem0 row, JSON mirror)
      * provenance remains available on both observations
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        cls.res1 = cls.vm.ingest(OBS_A, observed_at="2026-09-01")
        cls.res2 = cls.vm.ingest(OBS_B, observed_at="2026-09-10")
        # the old id: the fact created by observation 1
        cls.old_id = (cls.res1.get("memory_ids") or [None])[0]
        # the new id: the UPDATE turn appends the new observation inside
        # repo.update_memory (VM-LOCAL-008) — Ingest's memory_ids only lists
        # plain ADDs, so the new id is read from the old row's superseded_by
        # (which is exactly the explicit relationship under test).
        old_row = _mem0(cls.vm).get(cls.old_id) if cls.old_id else None
        old_meta = (old_row or {}).get("metadata") or {}
        cls.new_id = str(old_meta.get("superseded_by") or "").strip() or None

    def test_01_update_decision_actually_fired(self):
        """Harness check: the resolver DID emit UPDATE (else these proofs
        would be vacuous). The UPDATE path must have produced a new id."""
        self.assertTrue(self.old_id, "observation 1 produced no fact (harness)")
        self.assertTrue(self.new_id, "observation 2 produced no fact (harness)")
        self.assertNotEqual(self.old_id, self.new_id,
                            "UPDATE must not reuse the old memory id")

    def test_02_old_observation_text_survives(self):
        """No destructive overwrite: the old row keeps its ORIGINAL text."""
        old = _mem0(self.vm).get(self.old_id)
        self.assertIsNotNone(old, "old observation row is GONE (CD-2)")
        self.assertEqual(str(old.get("memory", "")).strip(), FACT_A,
                         "old observation text was overwritten in place")

    def test_03_supersession_is_explicit(self):
        """Both directions of the relationship are first-class metadata."""
        old = _mem0(self.vm).get(self.old_id)
        new = _mem0(self.vm).get(self.new_id)
        old_meta = old.get("metadata") or {}
        new_meta = new.get("metadata") or {}
        self.assertEqual(str(old_meta.get("superseded_by", "")).strip(),
                         self.new_id,
                         "old row lacks superseded_by = new id")
        self.assertTrue(str(old_meta.get("superseded_at", "")),
                        "old row lacks superseded_at timestamp")
        self.assertEqual(str(new_meta.get("supersedes", "")).strip(),
                         self.old_id,
                         "new row lacks supersedes = old id")

    def test_04_provenance_preserved_on_both_rows(self):
        """Provenance survives: the new observation carries the event date
        of observation 2; the old row keeps its original event date."""
        old = _mem0(self.vm).get(self.old_id)
        new = _mem0(self.vm).get(self.new_id)
        self.assertEqual(str(old.get("created_at", "")).strip(), "2026-09-01",
                         "old row lost its original event date")
        # mem0 stores add-time metadata FLAT in the payload: created_at is a
        # core payload key (top level of the get() result), not nested.
        self.assertEqual(str(new.get("created_at", "")).strip(), "2026-09-10",
                         "new row lacks the observation-2 event date")

    def test_05_json_mirror_is_lossless(self):
        """The JSON mirror keeps BOTH entries: old annotated, new appended."""
        store = _repo(self.vm).load_json_store()
        entries = {str(o.get("id", "")): o for o in store["results"]
                   if isinstance(o, dict)}
        self.assertIn(self.old_id, entries)
        self.assertIn(self.new_id, entries)
        self.assertEqual(str(entries[self.old_id].get("memory", "")).strip(),
                         FACT_A, "mirror overwrote the old entry (CD-2)")
        self.assertEqual(
            str(entries[self.old_id].get("superseded_by", "")).strip(),
            self.new_id, "mirror old entry lacks the supersede link")
        self.assertEqual(
            str(entries[self.new_id].get("supersedes", "")).strip(),
            self.old_id, "mirror new entry lacks the supersedes link")

    def test_06_search_prefers_current_recovers_history(self):
        """Current retrieval ranks B first; the historical A survives in
        the hit list carrying superseded_by (recoverable through search)."""
        res = self.vm.search("kedvenc étterem", top_k=5)
        hits = res.hits or []
        self.assertTrue(hits, "no hits for the scenario query")
        top = hits[0]
        self.assertIn("Bistro Buda", top.text,
                      "current value must rank first "
                      f"(got: {top.text!r})")
        old_hits = [h for h in hits if h.memory_id == self.old_id]
        self.assertTrue(old_hits,
                        "historical observation is not reachable through "
                        "search (it must remain recoverable)")
        self.assertEqual(old_hits[0].superseded_by, self.new_id,
                         "the historical hit does not carry superseded_by")
        # a query aimed at the old value must also still find it
        res2 = self.vm.search("Arany Kanna", top_k=5)
        self.assertTrue(any(h.memory_id == self.old_id
                            for h in (res2.hits or [])),
                        "query naming the old value lost the old observation")

    def test_07_cognitive_graph_has_both_rows(self):
        """The graph layer holds a wrapper row for the NEW fact id (the
        current value); the old wrapper stays (consistent with the old row)."""
        import sqlite3
        db = _space_db(self.vm)
        with sqlite3.connect(str(db)) as conn:
            n_new = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE id=?",
                (self.new_id,)).fetchone()[0]
            n_old = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE id=?",
                (self.old_id,)).fetchone()[0]
        self.assertEqual(n_new, 1,
                         "the current value has no cognitive-graph wrapper")
        self.assertEqual(n_old, 1,
                         "the old observation lost its graph wrapper")

    @classmethod
    def tearDownClass(cls) -> None:
        _release_vm(cls.vm)


# ===========================================================================
# 4. Left-brain delete gate on the REAL store (VM-LOCAL-005 + VM-LOCAL-009)
# ===========================================================================

class LeftBrainDeleteGateTests(unittest.TestCase):
    """Default deny, then a permitted delete must cascade every graph row —
    no stale authoritative-looking data anywhere (audit F-H)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        cls.res = cls.vm.ingest(
            "A legjobb barátom Bence, együtt futni szoktunk reggelente.",
            observed_at="2026-09-10")
        cls.fact_id = (cls.res.get("memory_ids") or [None])[0]
        assert cls.fact_id, "harness: fact not created"

    def test_01_default_env_blocks_delete_and_harms_nothing(self):
        os.environ.pop("VOICEMEM_ALLOW_MEMORY_DELETE", None)
        mem0_row = _mem0(self.vm).get(self.fact_id)
        self.assertIsNotNone(mem0_row)
        ok = _repo(self.vm).delete_memory(self.fact_id)
        self.assertFalse(ok, "default must refuse the left-brain delete")
        # deny is fully non-destructive: row AND graph rows intact
        self.assertIsNotNone(_mem0(self.vm).get(self.fact_id))
        refs = _graph_refs(_space_db(self.vm), self.fact_id)
        self.assertGreaterEqual(refs["memories"], 1,
                                "denied delete must not remove graph rows")

    def test_02_enabled_delete_cascades_every_graph_row(self):
        os.environ["VOICEMEM_ALLOW_MEMORY_DELETE"] = "1"
        try:
            ok = _repo(self.vm).delete_memory(self.fact_id)
            self.assertTrue(ok, "explicit opt-in failed to delete")
        finally:
            os.environ.pop("VOICEMEM_ALLOW_MEMORY_DELETE", None)
        # vector row gone
        self.assertIsNone(_mem0(self.vm).get(self.fact_id),
                          "mem0 row survived a permitted delete")
        # JSON mirror gone
        store = _repo(self.vm).load_json_store()
        ids = {str(o.get("id", "")) for o in store["results"]
               if isinstance(o, dict)}
        self.assertNotIn(self.fact_id, ids,
                         "JSON mirror row survived a permitted delete")
        # every graph reference gone — NO ORPHANS
        refs = _graph_refs(_space_db(self.vm), self.fact_id)
        for table, count in refs.items():
            self.assertEqual(
                count, 0,
                f"permitted delete left {count} orphan row(s) in {table} "
                "(stale graph truth — audit F-H)")

    def test_03_deleting_unknown_id_is_safe(self):
        os.environ.pop("VOICEMEM_ALLOW_MEMORY_DELETE", None)
        ok = _repo(self.vm).delete_memory("definitely-not-an-id")
        self.assertFalse(ok)

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("VOICEMEM_ALLOW_MEMORY_DELETE", None)

        _release_vm(cls.vm)

# ===========================================================================
# 5. Production import resolution (audit F-E behavioural gap)
# ===========================================================================

class VendorImportResolutionTests(unittest.TestCase):
    """Real import resolution semantics, in a subprocess with the SAME
    resolution mechanism production uses (an editable-install-equivalent
    path entry — the installer's ``pip install -e vendor\\voicemem``).

    * resolves to the controlled vendor tree (not a shadow)
    * the pin (commit + fork flag + patch ledger incl. 007/008/009) holds
    * a PYTHONPATH shadow placed BEFORE the vendor is DETECTED by the pin
      check (the check itself works; what it does on failure is policy)
    """

    _PY = sys.executable

    def test_01_resolves_to_controlled_vendor_with_full_ledger(self):
        code = (
            "import voicemem, pathlib\n"
            "p = pathlib.Path(voicemem.__file__).resolve()\n"
            "assert voicemem.CONTROLLED_FORK, 'not the controlled fork'\n"
            f"assert voicemem.CONTROLLED_UPSTREAM_COMMIT == 'e8384e087bd2f44eb05fc7ae1a3c525ea8244179', 'pin mismatch'\n"
            "assert 'vendor' + chr(92) + 'voicemem' in str(p) or 'vendor/voicemem' in str(p), f'resolved to {p}'\n"
            "for pid in ('VM-LOCAL-007','VM-LOCAL-008','VM-LOCAL-009'):\n"
            "    assert pid in voicemem.CONTROLLED_PATCHES, f'missing patch {pid}'\n"
            "print('VENDOR-PIN-OK', p)\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(VENDOR)
        proc = subprocess.run([self._PY, "-c", code], capture_output=True,
                              text=True, timeout=120, cwd="/",
                              env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("VENDOR-PIN-OK", proc.stdout)

    def test_02_shadowing_is_detectable(self):
        """A fake ``voicemem`` placed EARLIER on the path wins the import
        (Python semantics) — and the pin check detects exactly that: the
        shadow has no CONTROLLED_FORK. Bootstrap's identity probe (the same
        check, wired into START.bat) fails loud on this condition."""
        with tempfile.TemporaryDirectory(prefix="t15_shadow_") as td:
            shadow = Path(td) / "voicemem"
            shadow.mkdir()
            (shadow / "__init__.py").write_text(
                "CONTROLLED_FORK = False\n"
                "CONTROLLED_UPSTREAM_COMMIT = '0000'\n"
                "CONTROLLED_PATCHES = ()\n")
            code = (
                "import voicemem, pathlib\n"
                "p = pathlib.Path(voicemem.__file__).resolve()\n"
                "pin_ok = bool(getattr(voicemem, 'CONTROLLED_FORK', False))\n"
                "print('SHADOW-RESOLVED', p, 'PIN-OK' if pin_ok else 'PIN-FAIL')\n"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{td}{os.pathsep}{VENDOR}"
            proc = subprocess.run([self._PY, "-c", code], capture_output=True,
                                  text=True, timeout=120, cwd="/",
                                  env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertIn("SHADOW-RESOLVED", proc.stdout)
            self.assertIn("PIN-FAIL", proc.stdout,
                          "the pin check failed to DETECT the shadow "
                          "(shadowing would be silently accepted)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
