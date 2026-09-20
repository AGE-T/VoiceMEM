"""READ-ONLY-forensic companion: KV micro-benchmark on a SEPARATE SCRATCH DB.
Vendor module loaded standalone (space.py is dependency-free). No LLM, no
embedding, no SessionTracker activation, no production DB touched."""
import importlib.util, json, shutil, statistics, time
from pathlib import Path

VENDOR_SPACE = "/home/z/my-project/voicemem-agent/vendor/voicemem/voicemem/utils/common/space.py"
spec = importlib.util.spec_from_file_location("vmspace", VENDOR_SPACE)
space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(space)

ROOT = Path("/tmp/vm_kbench/scratch_root")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)

# Representative bounded checkpoint (the 100-500 token class from the study)
checkpoint = {
    "schema": 1,
    "session_id": "webspace_demo#2026-09-20T09:15:00Z",
    "topic": "VoiceMEM session-continuity tervezes",
    "state": "A dormant pipeline audit lezarult; a KV-lane checkpoint irany dontes elott.",
    "decisions": ["Lane B (app-level KV checkpoint) elvi irany", "Digest: write-then-upgrade minta"],
    "open_items": ["Backlog meres production DB-n", "TTL/injection window operator dontes"],
    "last_user_goal": "Mekkora a dormant backlog es mit erintene az elso Flush?",
    "next_step": "Read-only SQL csomag futtatasa a production space masolatan",
    "turn_count": 14,
    "started_at": "2026-09-20T08:41:12Z",
    "ended_at": "2026-09-20T09:14:47Z",
    "digest_method": "deterministic",
}
KEY = "session_checkpoint:webspace_demo"

# static size evidence
blob = json.dumps(checkpoint, ensure_ascii=False)
print(f"[static] checkpoint JSON: {len(blob.encode('utf-8'))} bytes, {len(blob)} chars")

space.kv_set(ROOT, KEY, checkpoint)   # scratch write (permitted: kulon scratch DB)

def bench(fn, n=300):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "n": n, "min_ms": round(ts[0], 4), "p50_ms": round(statistics.median(ts), 4),
        "p95_ms": round(ts[int(n * 0.95) - 1], 4), "max_ms": round(ts[-1], 4),
    }

print("[kv_get HIT  ]", bench(lambda: space.kv_get(ROOT, KEY)))
print("[kv_get MISS ]", bench(lambda: space.kv_get(ROOT, "session_checkpoint:nonexistent")))
print("[kv_set      ]", bench(lambda: space.kv_set(ROOT, KEY, checkpoint)))
print("[json.loads  ]", bench(lambda: json.loads(blob)))
got = space.kv_get(ROOT, KEY)
assert got == checkpoint, "roundtrip mismatch!"
print("[roundtrip   ] OK (kv_set -> kv_get -> object equality)")
import sqlite3
db = space.db(ROOT)
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
rows = con.execute("SELECT k, length(v) FROM kv").fetchall()
print("[scratch db kv table]", rows)
con.close()
print("[env] python:", __import__("sys").version.split()[0])
