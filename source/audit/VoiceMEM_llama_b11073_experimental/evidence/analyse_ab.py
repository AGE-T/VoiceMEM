import json, statistics
from pathlib import Path

cells = {}
for cid in ("A1","B1","A2","B2"):
    cells[cid] = json.loads(Path(f"ab_results/{cid}.json").read_text())

def col(cid, *keys):
    d = cells[cid]
    for k in keys: d = d[k]
    return d

print(f"{'metric':<38} {'A1 (b10717 prod)':>18} {'B1 (b11073 same)':>18} {'A2 (b10717 op)':>18} {'B2 (b11073 op)':>18}")
def row(name, path_a, path_b, path_c=None, path_d=None, fmt="{:>.3f}"):
    vals = []
    for cid, p in (("A1",path_a),("B1",path_b),("A2",path_c),("B2",path_d)):
        if p is None: vals.append(""); continue
        d = cells[cid]
        for k in p: d = d[k]
        vals.append(fmt.format(d) if isinstance(d,(int,float)) else str(d))
    print(f"{name:<38} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18} {vals[3]:>18}")

print("== version ==", )
row("version line", ("version",), ("version",), ("version",), ("version",), "{}")
row("normal TTFT median s", ("normal_stats","ttft_s","median"), ("normal_stats","ttft_s","median"), ("normal_stats","ttft_s","median"), ("normal_stats","ttft_s","median"))
row("normal TTFT p90 s", ("normal_stats","ttft_s","p90"), ("normal_stats","ttft_s","p90"), ("normal_stats","ttft_s","p90"), ("normal_stats","ttft_s","p90"))
row("normal total median s", ("normal_stats","total_s","median"), ("normal_stats","total_s","median"), ("normal_stats","total_s","median"), ("normal_stats","total_s","median"))
row("normal total max s", ("normal_stats","total_s","max"), ("normal_stats","total_s","max"), ("normal_stats","total_s","max"), ("normal_stats","total_s","max"))
row("streaming ok (of 8)", ("normal_stats","streaming_ok"), ("normal_stats","streaming_ok"), ("normal_stats","streaming_ok"), ("normal_stats","streaming_ok"), "{:>2}")
print()
row("extract total median s", ("extract_stats","total_s","median"), ("extract_stats","total_s","median"), ("extract_stats","total_s","median"), ("extract_stats","total_s","median"))
row("extract total max s", ("extract_stats","total_s","max"), ("extract_stats","total_s","max"), ("extract_stats","total_s","max"), ("extract_stats","total_s","max"))
row("extract JSON valid (of 4)", ("extract_stats","json_valid_count"), ("extract_stats","json_valid_count"), ("extract_stats","json_valid_count"), ("extract_stats","json_valid_count"), "{:>2}")
row("extract prompt tokens", ("extract_stats","prompt_tokens"), ("extract_stats","prompt_tokens"), ("extract_stats","prompt_tokens"), ("extract_stats","prompt_tokens"), "{:>4}")
print()
row("contend user TTFT median s", ("contention_stats","user_ttft_behind_bg_s","median"), ("contention_stats","user_ttft_behind_bg_s","median"), ("contention_stats","user_ttft_behind_bg_s","median"), ("contention_stats","user_ttft_behind_bg_s","median"))
row("contend user TTFT max s", ("contention_stats","user_ttft_behind_bg_s","max"), ("contention_stats","user_ttft_behind_bg_s","max"), ("contention_stats","user_ttft_behind_bg_s","max"), ("contention_stats","user_ttft_behind_bg_s","max"))
row("bg-disconnect slot-free ms", ("contention_stats","slot_free_ms_on_disconnect","median"), ("contention_stats","slot_free_ms_on_disconnect","median"), ("contention_stats","slot_free_ms_on_disconnect","median"), ("contention_stats","slot_free_ms_on_disconnect","median"), "{:>.1f}")
print()
row("barge-in slot-free median ms", ("cancel_stats","slot_free_ms","median"), ("cancel_stats","slot_free_ms","median"), ("cancel_stats","slot_free_ms","median"), ("cancel_stats","slot_free_ms","median"), "{:>.1f}")
row("barge-in slot-free max ms", ("cancel_stats","slot_free_ms","max"), ("cancel_stats","slot_free_ms","max"), ("cancel_stats","slot_free_ms","max"), ("cancel_stats","slot_free_ms","max"), "{:>.1f}")
row("follow-up TTFT median s", ("cancel_stats","followup_ttft_s","median"), ("cancel_stats","followup_ttft_s","median"), ("cancel_stats","followup_ttft_s","median"), ("cancel_stats","followup_ttft_s","median"))
print()
print("== SSE integrity ==")
for cid in ("A1","B1","A2","B2"):
    s = cells[cid]["sse"]
    print(f"  {cid}: role={s['role_chunks']} content={s['content_chunks']} reasoning={s['reasoning_chunks']} finish={s['finish_chunks']} done={s['done_marker']} other={s['other_lines']} first={s['first_line_kind']} http={s['http_status']}")
print()
print("== cache reuse (identical prompt 2x) ==")
for cid in ("A1","B1","A2","B2"):
    c = cells[cid]["cache"]["runs"]
    print(f"  {cid}: run0 ttft={c[0]['ttft_s']}s run1 ttft={c[1]['ttft_s']}s prompt_tokens={c[2]['prompt_tokens']} cached={c[2]['cached_tokens']}")
print()
print("== metrics delta (cell totals) ==")
for cid in ("A1","B1","A2","B2"):
    md = cells[cid]["metrics_delta"]
    interesting = {k:v for k,v in md.items() if any(x in k for x in ("prompt_tokens_total","tokens_predicted_total","prompt_per_second","predicted_per_second","prompt_tokens_seconds","predicted_tokens_seconds"))}
    print(f"  {cid}: {interesting}")
print()
print("== cell wall time ==")
for cid in ("A1","B1","A2","B2"):
    print(f"  {cid}: {cells[cid]['cell_wall_s']}s")
