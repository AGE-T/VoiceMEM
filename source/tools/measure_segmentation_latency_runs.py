"""Segmentation latency benchmark: phrase runs (v0.10.5 phase 2).

Mirrors the v0.10.4 evidence format (1000 calls per case) so the numbers
are directly comparable with audit/VoiceMEM_tts_codeswitch_v0104.

Run:  python tools/measure_segmentation_latency_runs.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.text_utils import detect_language, segment_language_spans  # noqa: E402

OUT = (
    _ROOT / "audit" / "VoiceMEM_tts_codeswitch_v0105_unquoted"
    / "evidence" / "segmentation_latency_20260918.txt"
)

CASES = [
    ("pure_hu_53ch", "Szeretnék röviden beszélni veled az új projektről."),
    ("pure_en_46ch", "I'd like to touch base with you."),
    ("unquoted_run_43ch", "Szeretnék egy gyors touch base-t veled."),
    ("unquoted_run_36ch", "Szerintem később catch up-olhatunk."),
    ("quoted_plus_run_55ch", 'A "touch base" kifejezés, de ma catch up-olhatunk.'),
    (
        "worst_multiregion_360ch",
        "A „touch base” és a „catch up” és egy (check this out) meg egy "
        "hosszabb magyar szövegrész, hogy legyen mit szegmentálni. "
        "Rendben, see you later, aztán touch base, majd catch up. ",
    ),
]


def main() -> int:
    lines = ["segment_language_spans latency (sandbox, 1000 calls per case)"]
    for name, text in CASES:
        for host in ("hu", "en"):
            n = 1000
            t0 = time.perf_counter()
            for _ in range(n):
                segment_language_spans(text, host)
            per_call_us = (time.perf_counter() - t0) / n * 1e6
            spans = segment_language_spans(text, host)
            lines.append(
                f"{name} host={host} chars={len(text)} spans={len(spans)} "
                f"per_call={per_call_us:.1f}us"
            )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
