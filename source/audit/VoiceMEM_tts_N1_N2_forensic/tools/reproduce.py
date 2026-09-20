#!/usr/bin/env python3
"""N1/N2 forensic reproduction on the CURRENT tree (pre-fix).

Exercises the exact production path the TTS call sites use:
normalize_for_speech -> detect_language (host) -> segment_language_spans.

N1: an English-looking loan with orthographic EN evidence ('meeting', 'ee')
    incorrectly absorbs a FOLLOWING Hungarian word into the EN span
    ("A 2026-os meeting fontos..." -> en 'meeting fontos').

N2: an accent-free Hungarian host context incorrectly FLIPS the detected
    host language to English when an English loan is present
    ("Holnap lesz a meeting..." -> host=en, whole HU sentence on EN voice).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from app.text_utils import (  # noqa: E402
    LANG_EN,
    LANG_HU,
    detect_language,
    normalize_for_speech,
    segment_language_spans,
)


def trace(label: str, text: str) -> None:
    norm = normalize_for_speech(text)
    host = detect_language(norm)
    spans = segment_language_spans(norm, host)
    print(f"--- {label}")
    print(f"    text : {text}")
    print(f"    host : {host}")
    for chunk, lang in spans:
        print(f"    {lang:>3} | {chunk!r}")
    print()


def main() -> int:
    print("=== N1: EN-evidence loan absorbs following HU word ===")
    trace("N1-a operator case", "A 2026-os meeting fontos lesz mindenkinnek.")
    trace("N1-b loan + HU adjective", "A meeting fontos volta meglepett.")
    trace("N1-c loan + HU verb", "Holnap meeting kezdődik.")
    trace("N1-d loan + HU noun", "A meeting előtt még beszélünk.")
    trace("N1-e multiple loans", "A meeting afterparty fontos volt.")
    trace("N1-f control: touch base (correct EN)", "Holnap touch base-t tartunk.")

    print("=== N2: accent-free HU host flips to EN ===")
    trace("N2-a operator case", "Holnap lesz a meeting a csapattal.")
    trace("N2-b meeting + plain HU", "Holnap meeting van.")
    trace("N2-c accented control", "Holnap lesz a meeting a csapatával.")
    trace("N2-d pure accent-free HU", "Holnap lesz a stands a csapattal.")
    trace("N2-e touch base control", "Holnap touch base.")

    print("=== protections that must NOT move ===")
    for t in (
        "A hazai a leghasznosabb megoldás.",
        "Az autó gyors.",
        "Ezt a projektet szeretem.",
        "This only really makes sense in the city.",
        "Nagyon sok a munka, de rendben van.",
    ):
        trace("guard", t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
