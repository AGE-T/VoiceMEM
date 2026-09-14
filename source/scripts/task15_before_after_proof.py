#!/usr/bin/env python3
"""TASK 1.5 before/after proof — the CD-1 gate comparison (evidence only).

Runs the SAME right-brain scenario against TWO vendor trees:

  BEFORE = the pre-patch vendor (extracted from git HEAD = v0.5.1 state)
  AFTER  = the current working tree (VM-LOCAL-007/008/009 applied)

Scenario: >=10 heartnotes exist; the LLM cleanup decision deterministically
requests deletion of every heartnote; VOICEMEM_ALLOW_MEMORY_DELETE is ABSENT.

Expected:
  BEFORE (v0.5.1): heartnotes are DELETED (ungated LLM hard-delete — the P0)
  AFTER  (v0.5.2): nothing is deleted (default deny)

Output: JSON report on stdout. Exit 0 always (measurement; the verdict is
data, not a process code).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

E5_DIR = (
    REPO / "models" / "hf" / "models--intfloat--multilingual-e5-small"
    / "snapshots" / "614241f622f53c4eeff9890bdc4f31cfecc418b3"
)
USER_ID = "t15_before_after"

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


def _route(messages):
    system = " ".join(str(m.get("content", "")) for m in messages
                      if m.get("role") == "system")
    user = " ".join(str(m.get("content", "")) for m in messages
                    if m.get("role") != "system")
    everything = system + "\n" + user
    if "记忆清洁助手" in everything:
        ids = []
        for tok in everything.replace(" ", "\n").split("\n"):
            if "ID:" in tok:
                short = tok.split("ID:")[1].split("|")[0].strip()
                if len(short) >= 8:
                    ids.append(short[:8])
        return {"delete_ids": ids, "supersede": []}
    for utt, emo in RB_UTTERANCES:
        if utt in everything:
            return {"memory": [{"text": f"A felhasználó mondta: {utt}",
                                "slot": "daily_life"}],
                    "emotion": emo, "traits": []}
    if "用户说了这句话" in user:
        return {"slots": ["daily_life"]}
    if "empathetic AI assistant" in system:
        return {"text": "user is processing feelings"}
    if '"items"' in everything and "喜好与厌恶" in everything:
        return {"items": []}
    return {"memory": [], "emotion": "", "traits": []}


def _start_mock():
    import threading, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            if self.path.split("?")[0] != "/v1/chat/completions":
                self._json(404, {"error": "no route"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            payload = _route(body.get("messages") or [])
            self._json(200, {
                "id": f"chatcmpl-ba-{int(time.time() * 1000)}",
                "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", "mock-llm"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": json.dumps(payload, ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            })

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _run_scenario(vendor_path: Path) -> dict:
    """Import voicemem from vendor_path, run the CD-1 scenario, report."""
    import contextlib
    import io
    for mod in list(sys.modules):
        if mod == "voicemem" or mod.startswith("voicemem."):
            del sys.modules[mod]
    sys.path.insert(0, str(vendor_path))
    stdout_capture = io.StringIO()
    try:
        # the vendor prints its [Cleanup] lines to stdout — capture them so
        # the report JSON on stdout stays parseable
        with contextlib.redirect_stdout(stdout_capture):
            import voicemem
            from voicemem import VoiceMem
            from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder
            root = tempfile.mkdtemp(prefix="t15_ba_")
            vm = VoiceMem(mode="text_mode", memory_root=root, user_id=USER_ID,
                          embedding=lambda: LocalE5Embedder())
            for utt, _emo in RB_UTTERANCES:
                vm.ingest(utt, observed_at="2026-09-10")
            rb_store = vm._o._right._rb_repo()._store
            notes_before = sum(
                1 for m in rb_store.get_all(USER_ID)
                if m.memory_class == "heartnote")
            # the LLM mock requests deletion of EVERY heartnote; env ABSENT
            os.environ.pop("VOICEMEM_ALLOW_MEMORY_DELETE", None)
            vm._o._right.run_cleanup()
            notes_after = sum(
                1 for m in rb_store.get_all(USER_ID)
                if m.memory_class == "heartnote")
            patches = list(getattr(voicemem, "CONTROLLED_PATCHES", ()) or [])
            captured = stdout_capture.getvalue()
        return {
            "vendor_resolved": str(Path(voicemem.__file__).resolve()),
            "controlled_patches": patches,
            "has_vm_local_007": "VM-LOCAL-007" in patches,
            "heartnotes_before": notes_before,
            "heartnotes_after_default_env": notes_after,
            "deleted_with_env_absent": notes_before - notes_after,
            "vendor_stdout_tail": captured.strip()[-400:],
            "verdict": (
                "UNGATED DELETE (CD-1 P0 present — v0.5.1 behaviour)"
                if notes_after < notes_before
                else "DEFAULT DENY (gated — v0.5.2 behaviour)"),
        }
    finally:
        sys.path.remove(str(vendor_path))


def main() -> int:
    srv = _start_mock()
    port = srv.server_address[1]
    os.environ.update({
        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "OPENAI_API_KEY": "before-after-not-needed",
        "OPENAI_MODEL": "mock-llm",
        "OPENAI_CHAT_MODEL": "mock-llm",
        "VOICEMEM_E5_MODEL": str(E5_DIR),
        "VOICEMEM_EMBED_DIM": "384",
        "VOICEMEM_VERBOSE": "0",
    })
    report = {
        "schema": "task15-before-after/1",
        "scenario": ("12 emotional utterances ingested; run_cleanup with a "
                     "deterministic LLM decision requesting deletion of ALL "
                     "heartnotes; VOICEMEM_ALLOW_MEMORY_DELETE ABSENT"),
        "before": _run_scenario(Path("/tmp/vendor_before")),
        "after": _run_scenario(REPO / "vendor" / "voicemem"),
    }
    report["comparison"] = {
        "cd1_fixed": (report["before"]["deleted_with_env_absent"] > 0
                      and report["after"]["deleted_with_env_absent"] == 0),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
