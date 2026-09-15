"""v0.10 — TEMPORAL MEMORY SEMANTICS (integration, real stack).

Real E5 embeddings, real mem0/Qdrant store, real cognitive graph, real
voice_input ingest chain — ONLY the LLM decisions are mocked (the same
boundary the v0.9.0 memory-safety gate established: mock the LLM decision,
never the storage behaviour being proven).

Scenario map (PHASE 2 classes A–G + PHASE 3 observation-time + PHASE 4
chains + PHASE 6 retrieval + PHASE 7 cross-language/replay):

  A current .......... ClassACurrentTests
  B past ............. ClassBPastTests (current- vs past-intent ranking)
  C cessation ......... ClassCCessationTests (UPDATE closes the old interval)
  D uncertain ........ ClassDUncertainTests (facet, never replaces)
  E future ........... ClassEFutureTests (incl. the quote-rescue path: the
                        extractor normalises "fogja"→"szereti", the QUOTE
                        keeps the future signal)
  F bounded .......... (folded into C: the old row's January event date is
                        preserved and the interval closes at the observation)
  G repeated+change .. ClassGReinforcementThenChangeTests (occurrence frozen)
  chains ............. SupersessionChainTests (A→B→A→B, all rows recoverable)
  cross-language ..... CrossLanguageSupersessionTests (HU fact, EN cessation)
  observation time ... ObservationTimeTests (missing ts → real ISO; explicit
                        ts passes; out-of-order = write-order currency — the
                        documented fact-path boundary, proven honest)
  traits future ..... TraitFutureStanceTests (real sqlite TraitStore; future
                        never merges into / flips a current row)
"""
from __future__ import annotations

import json
import os
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
for p in (str(REPO), str(VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

E5_DIR = REPO / "models" / "embedding" / "multilingual-e5-small"

USER_ID = "temporal_v10_user"
_FACT_SLOT = "日常生活"

# ---------------------------------------------------------------------------
# Scenario data (Hungarian product language + English cross-language leg)
# ---------------------------------------------------------------------------

UTT_A = "Tamás szereti a hegyi kerékpározást."
FACT_A = "Tamás szereti a hegyi kerékpározást."

UTT_B_PAST = "Tamás régen szerette a síelést."
FACT_B_PAST = "Tamás régen szerette a síelést."

UTT_C1 = "Tamás szereti a kávét."
FACT_C1 = "Tamás szereti a kávét."
UTT_C2 = "Tamás már nem szereti a kávét."
FACT_C2 = "Tamás már nem szereti a kávét."

UTT_D = "Talán Tamás már nem szereti a teát."
FACT_D = "Talán Tamás már nem szereti a teát."

UTT_E = "Tamás szeretni fogja a horgászatot."
FACT_E_NORMALISED = "Tamás szereti a horgászatot."  # what a normalising
# extractor emits — the future signal survives only in the quote (ground
# truth); the classifier must recover it from the utterance.
UTT_E_CUR = "Tamás szereti az úszást."
FACT_E_CUR = "Tamás szereti az úszást."

UTT_X = "Tamás szeretettel foglalkozik a kertészkedéssel."
FACT_X = "Tamás szeretettel foglalkozik a kertészkedéssel."

UTT_EN_CEASE = "Thomas no longer likes motorcycles."
FACT_HU_MOTO = "Tamás szereti a motorozást."
FACT_EN_CEASE = "Tamás már nem szereti a motorozást."

UTT_REST_A1 = "Tamás kedvenc étterme az Arany Kanna."
FACT_REST_A1 = "Tamás kedvenc étterme az Arany Kanna."


def _extraction_payload(fact: str) -> dict[str, Any]:
    return {"memory": [{"text": fact, "slot": _FACT_SLOT}],
            "emotion": "", "traits": []}


def _resolver_rows(user: str) -> list[tuple[str, str]]:
    """(index, text) pairs of the old-memory list as rendered into the
    resolver prompt (Python repr: {'id': '0', 'text': '…'})."""
    import re
    return [(m.group(1), m.group(2)) for m in re.finditer(
        r"\{'id': '(\d+)', 'text': '([^']*)'\}", user)]


def _id_of(rows: list[tuple[str, str]], keyword: str) -> str | None:
    for i, t in rows:
        if keyword in t:
            return i
    return None


def _route(messages: list[dict[str, Any]]) -> dict[str, Any]:
    system = " ".join(str(m.get("content", "")) for m in messages
                      if m.get("role") == "system")
    user = " ".join(str(m.get("content", "")) for m in messages
                    if m.get("role") != "system")
    everything = system + "\n" + user

    # ── right-brain inner-OS (stand-in reply) ─────────────────────────────
    if "empathetic AI assistant" in system:
        return {"text": "user is processing feelings"}

    # ── conflict resolution ─────────────────────────────────────────────────
    if "smart memory manager" in user:
        rows = _resolver_rows(user)
        # The prompt is: ```old``` lead-in ```new facts``` tail — the NEW
        # FACTS live in the SECOND-TO-LAST ``` segment (routing on the whole
        # user content would also match the OLD memories' texts).
        fences = user.split("```")
        new_facts = fences[-2] if len(fences) >= 3 else user
        if "már nem szereti a kávét" in new_facts:
            return {"memory": [{"event": "UPDATE",
                                "id": _id_of(rows, "szereti a kávét") or "0",
                                "text": FACT_C2,
                                "old_memory": FACT_C1}]}
        if "már nem szereti a motorozást" in new_facts:
            return {"memory": [{"event": "UPDATE",
                                "id": _id_of(rows, "motorozást") or "0",
                                "text": FACT_EN_CEASE,
                                "old_memory": FACT_HU_MOTO}]}
        if "szereti a kávét" in new_facts:
            tid = _id_of(rows, "szereti a kávét")
            if tid is not None:
                return {"memory": [{"event": "NONE", "id": tid}]}
            # no coffee row yet → let it fall through to the plain ADD path
            return {"memory": []}
        return {"memory": []}

    # ── merged fact extraction (utterance-keyed) ──────────────────────────
    # Route on the NEW MESSAGES section ONLY: the prompt also embeds the
    # existing memories (dedupe reference) — in these scenarios the fact
    # text equals the utterance, so a whole-prompt match would fire on old
    # rows and "re-extract" them on every later turn.
    if "## New Messages" in everything:
        new_msgs = everything.split("## New Messages")[-1].split(
            "## Observation Date")[0]
    else:
        new_msgs = everything
    if UTT_C2 in new_msgs:
        return _extraction_payload(FACT_C2)
    if UTT_C1 in new_msgs:
        return _extraction_payload(FACT_C1)
    if UTT_EN_CEASE in new_msgs:
        return _extraction_payload(FACT_EN_CEASE)
    if "motorozást" in new_msgs and "szereti" in new_msgs:
        return _extraction_payload(FACT_HU_MOTO)
    if UTT_REST_A1 in new_msgs:
        return _extraction_payload(FACT_REST_A1)
    if UTT_B_PAST in new_msgs:
        return _extraction_payload(FACT_B_PAST)
    if UTT_D in new_msgs:
        return _extraction_payload(FACT_D)
    if UTT_E in new_msgs:  # the normalising extractor (quote keeps future)
        return _extraction_payload(FACT_E_NORMALISED)
    if UTT_E_CUR in new_msgs:
        return _extraction_payload(FACT_E_CUR)
    if UTT_A in new_msgs:
        return _extraction_payload(FACT_A)
    if UTT_X in new_msgs:
        return _extraction_payload(FACT_X)

    # ── LLM slot re-tagging / trait extraction / cleanup ──────────────────
    if "用户说了这句话" in user:
        return {"slots": [_FACT_SLOT]}
    if '"items"' in everything and "喜好与厌恶" in everything:
        return {"items": []}
    if "记忆清洁助手" in everything:
        return {"delete_ids": [], "supersede": []}

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
            "id": f"chatcmpl-v10-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-llm"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": json.dumps(payload,
                                                           ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        })

    def do_GET(self):  # noqa: N802
        self._json(200, {"status": "ok"})

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


_SAVED_ENV: dict[str, str | None] = {}


def setUpModule() -> None:  # noqa: N802
    global _SAVED_ENV
    srv = _mock_server()
    port = srv.server_address[1]
    keys = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL",
            "OPENAI_CHAT_MODEL", "VOICEMEM_E5_MODEL", "VOICEMEM_EMBED_DIM",
            "VOICEMEM_VERBOSE", "VOICEMEM_ALLOW_MEMORY_DELETE")
    _SAVED_ENV = {k: os.environ.get(k) for k in keys}
    os.environ.update({
        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "OPENAI_API_KEY": "v10-temporal-tests",
        "OPENAI_MODEL": "mock-llm",
        "OPENAI_CHAT_MODEL": "mock-llm",
        "VOICEMEM_E5_MODEL": str(E5_DIR),
        "VOICEMEM_E5_DIM": "384",
        "VOICEMEM_VERBOSE": "0",
    })
    os.environ.pop("VOICEMEM_ALLOW_MEMORY_DELETE", None)


def tearDownModule() -> None:  # noqa: N802
    for k, v in _SAVED_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _new_vm(user_id: str = USER_ID) -> Any:
    from voicemem import VoiceMem
    from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder
    root = tempfile.mkdtemp(prefix="t_v10_")
    return VoiceMem(mode="text_mode", memory_root=root, user_id=user_id,
                    embedding=lambda: LocalE5Embedder())


def _release_vm(vm: Any) -> None:
    import gc
    try:
        store = vm._o._get_repo()._vector_store
        mem0 = getattr(store, "_mem0", None)
        for name in ("close", "aclose", "_close"):
            closer = getattr(mem0, name, None)
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


def _mem0(vm: Any) -> Any:
    return vm._o._get_repo()._vector_store._mem0


def _repo(vm: Any) -> Any:
    return vm._o._get_repo()


def _row(vm: Any, mid: str) -> dict[str, Any]:
    return _mem0(vm).get(mid) or {}


def _meta(vm: Any, mid: str) -> dict[str, Any]:
    return (_row(vm, mid).get("metadata") or {})


def _hits(vm: Any, query: str, top_k: int = 5) -> list[Any]:
    return _repo(vm).search(query, user_id=USER_ID, top_k=top_k)


def _add_id(vm: Any, res: dict[str, Any], idx: int = 0) -> str:
    return str((res.get("memory_ids") or [None] * (idx + 1))[idx])


# ===========================================================================
# A — current state
# ===========================================================================

class ClassACurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        cls.res = cls.vm.ingest(UTT_A, observed_at="2026-09-10")
        cls.mid = _add_id(cls.vm, cls.res)

    @classmethod
    def tearDownClass(cls) -> None:
        _release_vm(cls.vm)

    def test_row_carries_positive_stance(self):
        self.assertEqual(_meta(self.vm, self.mid).get("stance"), "pos")

    def test_status_current_and_retrievable(self):
        from voicemem.leftbrain.temporal import fact_status
        self.assertEqual(fact_status(_meta(self.vm, self.mid)), "current")
        hits = _hits(self.vm, "Mit szeret Tamás?")
        self.assertTrue(any(h.memory_id == self.mid for h in hits),
                        "the current fact must be retrievable")

    def test_observed_event_time_stored(self):
        self.assertEqual(_meta(self.vm, self.mid).get("time_start"),
                         "2026-09-10")


# ===========================================================================
# B — past state (self-declared historical, never claims currency)
# ===========================================================================

class ClassBPastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        cls.vm.ingest(UTT_A, observed_at="2026-09-10")   # current fact
        cls.vm.ingest(UTT_B_PAST, observed_at="2026-09-12")  # past fact
        ids = [str(o.get("id")) for o in
               _repo(cls.vm).load_json_store()["results"]]
        cls.cur_id, cls.past_id = ids[0], ids[1]

    @classmethod
    def tearDownClass(cls) -> None:
        _release_vm(cls.vm)

    def test_past_stance_stored(self):
        self.assertEqual(_meta(self.vm, self.past_id).get("stance"), "past")

    def test_past_row_is_historical_not_superseded(self):
        from voicemem.leftbrain.temporal import fact_status
        self.assertEqual(fact_status(_meta(self.vm, self.past_id)),
                         "historical")
        self.assertEqual(_meta(self.vm, self.past_id).get("superseded_by", ""),
                         "")

    def test_current_intent_ranks_current_first(self):
        hits = _hits(self.vm, "Mit szeret Tamás?")
        self.assertEqual(hits[0].memory_id, self.cur_id,
                         "current fact must outrank the self-declared past "
                         "one for a present-tense question")

    def test_past_intent_ranks_history_first(self):
        hits = _hits(self.vm, "Mit szeretett régen Tamás?")
        self.assertEqual(hits[0].memory_id, self.past_id,
                         "a past-tense question must prefer the historical "
                         "row — without excluding the current one")
        self.assertTrue(any(h.memory_id == self.cur_id for h in hits))


# ===========================================================================
# C — cessation closes the old row's interval (UPDATE path, full chain)
# ===========================================================================

class ClassCCessationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        r1 = cls.vm.ingest(UTT_C1, observed_at="2026-01-10")
        cls.old_id = _add_id(cls.vm, r1)
        cls.vm.ingest(UTT_C2, observed_at="2026-09-15")
        cls.new_id = str(_meta(cls.vm, cls.old_id).get("superseded_by") or "")

    @classmethod
    def tearDownClass(cls) -> None:
        _release_vm(cls.vm)

    def test_harness_update_fired(self):
        self.assertTrue(self.new_id, "resolver UPDATE must have fired")

    def test_old_row_interval_closed_at_observation(self):
        md = _meta(self.vm, self.old_id)
        self.assertEqual(md.get("valid_until"), "2026-09-15",
                         "the cessation must close the old row's interval at "
                         "the observation date (WHEN it ended, not just THAT)")

    def test_old_row_keeps_its_january_event_time(self):
        # PHASE 2 class F: "liked X in January but not now" — the January
        # half survives as the old row's event time; the cessation is current.
        self.assertEqual(_meta(self.vm, self.old_id).get("time_start"),
                         "2026-01-10")

    def test_new_row_carries_neg_stance_and_is_current(self):
        from voicemem.leftbrain.temporal import fact_status
        md = _meta(self.vm, self.new_id)
        self.assertEqual(md.get("stance"), "neg")
        self.assertEqual(md.get("supersedes"), self.old_id)
        self.assertEqual(fact_status(md), "current")

    def test_current_query_prefers_cessation_and_marks_history(self):
        hits = _hits(self.vm, "Mit szeret most Tamás?")
        self.assertEqual(hits[0].memory_id, self.new_id)
        old_hit = next((h for h in hits if h.memory_id == self.old_id), None)
        if old_hit is not None:  # still retrievable = historical, ranked after
            self.assertEqual(old_hit.superseded_by, self.new_id)

    def test_past_query_recovers_the_january_state(self):
        hits = _hits(self.vm, "Mit szeretett Tamás korábban?", top_k=5)
        self.assertTrue(any(h.memory_id == self.old_id for h in hits),
                        "the superseded January observation must stay "
                        "retrievable for a past-intent question")


# ===========================================================================
# D — uncertain stays a distinguishable facet, never replaces
# ===========================================================================

class ClassDUncertainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        cls.vm.ingest(UTT_C1, observed_at="2026-09-01")
        cls.vm.ingest(UTT_D, observed_at="2026-09-14")
        ids = [str(o.get("id")) for o in
               _repo(cls.vm).load_json_store()["results"]]
        cls.coffee_id, cls.uncertain_id = ids[0], ids[1]

    @classmethod
    def tearDownClass(cls) -> None:
        _release_vm(cls.vm)

    def test_uncertain_stance_stored_and_current(self):
        from voicemem.leftbrain.temporal import fact_status
        md = _meta(self.vm, self.uncertain_id)
        self.assertEqual(md.get("stance"), "uncertain")
        self.assertEqual(fact_status(md), "current")  # facet, not status

    def test_uncertain_did_not_touch_the_confirmed_row(self):
        md = _meta(self.vm, self.coffee_id)
        self.assertEqual(md.get("superseded_by", ""), "",
                         "an uncertain observation must never supersede")
        self.assertEqual(md.get("stance", ""), "pos",
                         "the confirmed row must not be re-labelled")

    def test_render_marks_uncertain(self):
        sys.path.insert(0, str(REPO))
        from app.web_server import hit_provenance_suffix
        hit = next(h for h in _hits(self.vm, "teát")
                   if h.memory_id == self.uncertain_id)
        self.assertIn("uncertain", hit_provenance_suffix(hit))


# ===========================================================================
# E — future is not present state (incl. the quote-rescue path)
# ===========================================================================

class ClassEFutureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        cls.vm.ingest(UTT_E_CUR, observed_at="2026-09-10")
        cls.vm.ingest(UTT_E, observed_at="2026-09-12")
        ids = [str(o.get("id")) for o in
               _repo(cls.vm).load_json_store()["results"]]
        cls.cur_id, cls.fut_id = ids[0], ids[1]

    @classmethod
    def tearDownClass(cls) -> None:
        _release_vm(cls.vm)

    def test_future_stance_recovered_from_quote(self):
        """The stored fact text is the NORMALISED present-tense claim
        ("szereti"); the future class must come from the utterance quote —
        exactly the production failure mode the quote-rescue prevents."""
        md = _meta(self.vm, self.fut_id)
        self.assertEqual(_row(self.vm, self.fut_id).get("memory"),
                         FACT_E_NORMALISED)
        self.assertEqual(md.get("stance"), "future")

    def test_future_status_and_ranking(self):
        from voicemem.leftbrain.temporal import fact_status
        self.assertEqual(fact_status(_meta(self.vm, self.fut_id)), "future")
        hits = _hits(self.vm, "Mit szeret Tamás mostanában?")
        self.assertEqual(hits[0].memory_id, self.cur_id,
                         "a future statement must not appear as present "
                         "state for a present-tense question")

    def test_future_intent_prefers_future(self):
        hits = _hits(self.vm, "Mit fog szeretni Tamás?")
        self.assertEqual(hits[0].memory_id, self.fut_id)

    def test_render_marks_future(self):
        sys.path.insert(0, str(REPO))
        from app.web_server import hit_provenance_suffix
        hit = next(h for h in _hits(self.vm, "horgászat")
                   if h.memory_id == self.fut_id)
        self.assertIn("future", hit_provenance_suffix(hit))


# ===========================================================================
# G — repeated confirmation then change (occurrence frozen at flip)
# ===========================================================================

class ClassGReinforcementThenChangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        r1 = cls.vm.ingest(UTT_C1, observed_at="2026-09-01")
        cls.old_id = _add_id(cls.vm, r1)
        cls.vm.ingest(UTT_C1, observed_at="2026-09-05")  # NONE → occurrence 1
        cls.vm.ingest(UTT_C1, observed_at="2026-09-08")  # NONE → occurrence 2
        cls.vm.ingest(UTT_C2, observed_at="2026-09-15")  # UPDATE → flip
        cls.new_id = str(_meta(cls.vm, cls.old_id).get("superseded_by") or "")
        cls.flip_id = cls.new_id

    @classmethod
    def tearDownClass(cls) -> None:
        _release_vm(cls.vm)

    def test_occurrence_frozen_at_flip(self):
        self.assertEqual(_meta(self.vm, self.old_id).get("occurrence_count"),
                         2, "the historical row's count must freeze at the "
                            "pre-flip confirmations")

    def test_chain_after_reinforcement(self):
        md_old, md_new = _meta(self.vm, self.old_id), _meta(self.vm, self.flip_id)
        self.assertEqual(md_old.get("superseded_by"), self.flip_id)
        self.assertEqual(md_new.get("supersedes"), self.old_id)
        self.assertEqual(md_new.get("stance"), "neg")

    def test_replay_of_the_old_statement_does_not_flip_back(self):
        """Delayed replay (the pre-flip statement re-observed later): the
        fact path counts it as confirmation of the CURRENT row's... no —
        the replayed old text matches the superseded row's text exactly;
        the omit-NONE counter only counts non-superseded matches. Proven
        below: currency must NOT flip back."""
        self.vm.ingest(UTT_C1, observed_at="2026-09-20")
        md_old = _meta(self.vm, self.old_id)
        self.assertEqual(md_old.get("superseded_by"), self.flip_id,
                         "replaying the old statement must not resurrect it")
        # and the current negation row keeps its stance
        self.assertEqual(_meta(self.vm, self.flip_id).get("stance"), "neg")


# ===========================================================================
# PHASE 4 — supersession chains (repo-level: real mem0 UPDATE machinery)
# ===========================================================================

class SupersessionChainTests(unittest.TestCase):
    """A→B→A→B rapid alternation through the REAL update_memory chain."""
    # (texts chosen so E5 keeps them near-identical topic-wise)

    def setUp(self):
        self.vm = _new_vm()
        repo = _repo(self.vm)
        r = self.vm.ingest("Tamás kedvenc étterme az Arany Kanna.",
                           observed_at="2026-09-01")
        self.a1 = _add_id(self.vm, r)
        self.b1 = repo.update_memory(self.a1, "Tamás kedvence most a Bistro Buda.",
                                     observed_at="2026-09-02", user_id=USER_ID,
                                     valid_until="2026-09-02",
                                     new_meta={"stance": "pos"})
        self.a2 = repo.update_memory(self.b1, "Tamás kedvenc étterme megint az Arany Kanna.",
                                     observed_at="2026-09-03", user_id=USER_ID,
                                     valid_until="2026-09-03",
                                     new_meta={"stance": "pos"})
        self.b2 = repo.update_memory(self.a2, "Tamás kedvence végül a Bistro Buda.",
                                     observed_at="2026-09-04", user_id=USER_ID,
                                     valid_until="2026-09-04",
                                     new_meta={"stance": "pos"})

    def tearDown(self):
        _release_vm(self.vm)

    def test_chain_structure_complete(self):
        self.assertEqual(_meta(self.vm, self.a1).get("superseded_by"), self.b1)
        self.assertEqual(_meta(self.vm, self.b1).get("superseded_by"), self.a2)
        self.assertEqual(_meta(self.vm, self.a2).get("superseded_by"), self.b2)
        self.assertEqual(_meta(self.vm, self.b2).get("superseded_by", ""), "")
        self.assertEqual(_meta(self.vm, self.b2).get("supersedes"), self.a2)

    def test_exactly_one_current_row(self):
        rows = [_meta(self.vm, i) for i in (self.a1, self.b1, self.a2, self.b2)]
        current = [m for m in rows if not m.get("superseded_by")]
        self.assertEqual(len(current), 1,
                         "rapid alternation must never leave two "
                         "contradictory rows both current")

    def test_intervals_closed_in_order(self):
        self.assertEqual(_meta(self.vm, self.a1).get("valid_until"), "2026-09-02")
        self.assertEqual(_meta(self.vm, self.b1).get("valid_until"), "2026-09-03")
        self.assertEqual(_meta(self.vm, self.a2).get("valid_until"), "2026-09-04")

    def test_all_four_states_retrievable(self):
        for mid in (self.a1, self.b1, self.a2, self.b2):
            found = any(h.memory_id == mid
                        for h in _hits(self.vm, "Tamás kedvenc étterme", top_k=8))
            self.assertTrue(found, f"chain row {mid} lost from retrieval")

    def test_current_intent_prefers_b2(self):
        hits = _hits(self.vm, "Mi Tamás kedvenc étterme most?")
        self.assertEqual(hits[0].memory_id, self.b2)

    def test_memory_history_walks_the_whole_chain(self):
        """PHASE 9 — the consolidation hook: the chain read OLDEST→CURRENT
        with the temporal bookkeeping on every entry."""
        hist = _repo(self.vm).memory_history(self.a1)
        self.assertEqual([h["id"] for h in hist],
                         [self.a1, self.b1, self.a2, self.b2])
        self.assertEqual([h["valid_until"] for h in hist],
                         ["2026-09-02", "2026-09-03", "2026-09-04", ""])
        self.assertEqual(hist[-1]["superseded_by"], "")

    def test_past_intent_prefers_history_over_current(self):
        hits = _hits(self.vm, "Mi volt korábban Tamás kedvenc étterme?", top_k=8)
        self.assertIn(hits[0].memory_id, (self.a1, self.b1, self.a2),
                      "a past-intent query must be topped by a historical "
                      "row (tier 0), not the current one")
        self.assertNotEqual(hits[0].memory_id, self.b2)


# ===========================================================================
# PHASE 7 — cross-language contradiction
# ===========================================================================

class CrossLanguageSupersessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        r = cls.vm.ingest("Tamás szereti a motorozást.",
                          observed_at="2026-09-01")
        cls.old_id = _add_id(cls.vm, r)
        cls.vm.ingest(UTT_EN_CEASE, observed_at="2026-09-14")
        cls.new_id = str(_meta(cls.vm, cls.old_id).get("superseded_by") or "")

    @classmethod
    def tearDownClass(cls) -> None:
        _release_vm(cls.vm)

    def test_english_cessation_supersedes_hungarian_fact(self):
        self.assertTrue(self.new_id)
        md_old = _meta(self.vm, self.old_id)
        self.assertEqual(md_old.get("valid_until"), "2026-09-14")
        self.assertEqual(_meta(self.vm, self.new_id).get("stance"), "neg")

    def test_hungarian_current_query_gets_cessation_first(self):
        hits = _hits(self.vm, "Szereti Tamás a motorozást?")
        self.assertEqual(hits[0].memory_id, self.new_id)


# ===========================================================================
# PHASE 3 — observation time vs write time vs missing timestamps
# ===========================================================================

class ObservationTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vm = _new_vm()
        # timestamp MISSING → the v0.10 anchor fix: real wall-clock ISO
        # (pre-v0.10 this was a time-of-day string that every parser
        # rejected — the whole event-time chain was dead in the app path).
        r1 = cls.vm.ingest(UTT_X)  # no observed_at
        cls.mid_missing = _add_id(cls.vm, r1)
        # timestamp EXPLICIT → passes through verbatim
        r2 = cls.vm.ingest(UTT_A, observed_at="2026-08-20")
        cls.mid_explicit = _add_id(cls.vm, r2)

    @classmethod
    def tearDownClass(cls) -> None:
        _release_vm(cls.vm)

    def test_missing_timestamp_falls_back_to_real_iso_datetime(self):
        ts = _meta(self.vm, self.mid_missing).get("time_start", "")
        from voicemem.leftbrain.temporal import parse_iso_date
        self.assertIsNotNone(parse_iso_date(ts),
                             f"time_start must now be ISO-shaped, got {ts!r}")

    def test_missing_timestamp_row_has_working_event_date(self):
        hits = _hits(self.vm, "kertészkedés")
        hit = next(h for h in hits if h.memory_id == self.mid_missing)
        self.assertTrue(hit.observed_at,
                        "observed_at must now parse (the recency weight and "
                        "the [date] provenance prefix finally bind)")

    def test_explicit_timestamp_passes_through(self):
        self.assertEqual(_meta(self.vm, self.mid_explicit).get("time_start"),
                         "2026-08-20")

    def test_out_of_order_observation_is_write_order_currency(self):
        """DOCUMENTED boundary (PHASE 0 #10): the fact path has no
        event-time guard — currency follows the WRITE order. Proven here
        honestly: a backdated observation still supersedes, and BOTH rows
        keep their true event times (the structure needed for a future
        guard, or for offline consolidation)."""
        repo = _repo(self.vm)
        r3 = self.vm.ingest(UTT_REST_A1, observed_at="2026-09-05",
                            agent_reply="Rendben, megjegyeztem.")
        cur_id = _add_id(self.vm, r3)
        self.assertTrue(cur_id, "harness: the ADD turn must have stored")
        new_id = repo.update_memory(
            cur_id, "Tamás kedvence már nem az Arany Kanna.",
            observed_at="2026-03-01",  # backdated
            user_id=USER_ID,
            valid_until="2026-03-01",
            new_meta={"stance": "neg"})
        self.assertTrue(new_id)
        self.assertEqual(_meta(self.vm, cur_id).get("superseded_by"), new_id)
        self.assertEqual(_meta(self.vm, cur_id).get("time_start"), "2026-09-05")
        # mem0 lifts metadata created_at to the TOP-LEVEL payload key
        self.assertEqual(_row(self.vm, new_id).get("created_at"), "2026-03-01")


# ===========================================================================
# Traits — the future stance never merges into / flips a current row
# ===========================================================================

class TraitFutureStanceTests(unittest.TestCase):
    """Real sqlite TraitStore, deterministic degenerate embedder (the
    v0.9.0 hard-regime convention: cosine 1.0 within a topic)."""

    @classmethod
    def setUpClass(cls) -> None:
        from voicemem.rightbrain.traits_store import Evidence, TraitStore

        class _Topic:
            DIM = 8

            def __call__(self, text):
                low = (text or "").lower()
                v = [0.0] * self.DIM
                if "káv" in low or "coffee" in low:
                    v[0] = 1.0
                else:
                    v[2] = 1.0
                return v

        cls._tmp = tempfile.TemporaryDirectory(prefix="t_v10_traits_")
        store = TraitStore(Path(cls._tmp.name) / "t.sqlite", _Topic())
        # current positive row
        cls.pos_id = store.add(USER_ID, "喜好与厌恶",
                               "Tamás szereti a kávét.",
                               Evidence(quote="Tamás szereti a kávét.",
                                        emotion="", cause="", cause_id="",
                                        at="2026-09-01T10:00:00"))
        # future row on the SAME topic (cosine 1.0 — hardest regime)
        cls.fut_id = store.add(USER_ID, "喜好与厌恶",
                               "Tamás szeretni fogja a kávét.",
                               Evidence(quote="Tamás szeretni fogja a kávét.",
                                        emotion="", cause="", cause_id="",
                                        at="2026-09-02T10:00:00"))
        cls._store = store

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _row(self, tid):
        with self._store._conn() as c:
            return c.execute("SELECT stance, superseded_by, occurrence_count "
                             "FROM rb_traits WHERE id=?", (tid,)).fetchone()

    def test_future_stance_recorded(self):
        self.assertEqual(self._row(self.fut_id)["stance"], "future")

    def test_future_did_not_merge_into_the_current_row(self):
        pos = self._row(self.pos_id)
        self.assertEqual(pos["occurrence_count"], 1,
                         "a future observation must not reinforce the "
                         "current row")
        self.assertEqual(pos["superseded_by"], "",
                          "a future observation must never supersede")

    def test_future_did_not_flip_or_get_flipped(self):
        fut = self._row(self.fut_id)
        self.assertEqual(fut["superseded_by"], "")
        self.assertEqual(self._row(self.pos_id)["superseded_by"], "")


if __name__ == "__main__":
    unittest.main()
