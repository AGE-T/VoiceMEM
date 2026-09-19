import json, sys
sys.path.insert(0, '.')
base = "audit/VoiceMEM_tts_codeswitch_v0105_forensic/evidence/"
before = json.load(open(base + "forensic_trace_before_fix.json"))
after = json.load(open(base + "forensic_trace_after_fix.json"))
print(f"{'case':4} {'d':>3} {'calls B/A':>10} {'EN spans B':>34} {'EN spans A':>34}")
tot_b = tot_a = 0
for b, a in zip(before, after):
    assert b["case"] == a["case"] and b["delta_size"] == a["delta_size"]
    tot_b += b["n_synth_calls"]; tot_a += a["n_synth_calls"]
    enb = " | ".join(s for c in b["chunks"] for s in c["DETECTED_ENGLISH_SPANS"]) or "-"
    ena = " | ".join(s for c in a["chunks"] for s in c["DETECTED_ENGLISH_SPANS"]) or "-"
    mark = "  <-- CHANGED" if (enb != ena or b["n_synth_calls"] != a["n_synth_calls"]) else ""
    print(f"{b['case']:4} {b['delta_size']:>3} {b['n_synth_calls']:>4}/{a['n_synth_calls']:<4}   {enb[:34]:34} {ena[:34]:34}{mark}")
print(f"\nTOTAL synth calls: before={tot_b}  after={tot_a}  delta={tot_a-tot_b}")

# First-span text (first-audio cost driver) before/after for every case:
print("\nFirst-call text (first-audio latency driver), before -> after:")
for b, a in zip(before, after):
    fb = b["chunks"][0]["SYNTH_CALLS"][0] if b["chunks"] else None
    fa = a["chunks"][0]["SYNTH_CALLS"][0] if a["chunks"] else None
    if fb and fa:
        changed = "  CHANGED" if fb["FINAL_TTS_TEXT"] != fa["FINAL_TTS_TEXT"] else ""
        print(f"  {b['case']}/{b['delta_size']:>3}: {fb['FINAL_TTS_TEXT']!r} -> {fa['FINAL_TTS_TEXT']!r}{changed}")
