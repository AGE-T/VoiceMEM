"""Call-latency variance supplement for the synthesis evidence file."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.config import AgentConfig  # noqa: E402
from app.text_utils import LANG_EN, LANG_HU  # noqa: E402
from app.tts_supertonic import SupertonicTtsEngine  # noqa: E402

OUT = (
    _ROOT / "audit" / "VoiceMEM_tts_codeswitch_v0105_unquoted"
    / "evidence" / "synthesis_latency_20260918.txt"
)


def main() -> int:
    engine = SupertonicTtsEngine(AgentConfig(root=_ROOT))
    engine.synthesize("Bemelegítés.", LANG_HU)
    samples = {"Szeretnék egy gyors (HU prefix)": [], "touch base (EN run)": []}
    for _ in range(5):
        for text, lang, key in (
            ("Szeretnék egy gyors ", LANG_HU, "Szeretnék egy gyors (HU prefix)"),
            ("touch base", LANG_EN, "touch base (EN run)"),
        ):
            t = time.perf_counter()
            engine.synthesize(text, lang, None, "F1")
            samples[key].append((time.perf_counter() - t) * 1000.0)
    lines = [
        "",
        "call latency variance (5 repeats per span, same warm engine):",
    ]
    for key, vals in samples.items():
        lines.append(
            f"  {key}: median {statistics.median(vals):.0f} ms, "
            f"min {min(vals):.0f}, max {max(vals):.0f}, "
            f"stdev {statistics.stdev(vals):.0f}"
        )
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
