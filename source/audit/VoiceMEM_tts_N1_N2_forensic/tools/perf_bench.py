#!/usr/bin/env python3
"""Hot-path performance of the N1/N2-relevant detector functions.

Measures detect_language + segment_language_spans over the N1/N2 matrix
corpus (deterministic, repeated) — median / p90 / max in microseconds.
Run BEFORE and AFTER the deterministic fix; the fix must stay
microsecond-class on the same scale as the existing detector.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from app.text_utils import (  # noqa: E402
    detect_language,
    normalize_for_speech,
    segment_language_spans,
)

CORPUS = [
    "A 2026-os meeting fontos lesz mindenkinnek.",
    "Az új meeting fontos lett.",
    "A meeting marad, ahogy van.",
    "A meeting afterparty fontos volt.",
    "Fontos see you later, rendben?",
    "Holnap lesz a meeting.",
    "Holnap lesz a meeting a csapattal.",
    "Reggel lesz a meeting a csapattal.",
    "Holnap lesz a status report a főnökkel.",
    "Holnap touch base.",
    "Holnap touch base veled lesz.",
    "Rendben, catch up után beszélünk.",
    "Holnap lesz a meeting, utána pedig catch up.",
    "Touch base után megbeszéljük a projektet.",
    "Ez a projekt fontos, de most nem akarok meetinget.",
    "Ma először checkeljük a resultot, aztán megnézzük a statuszt.",
    "Most már tényleg cut to the chase, és menjünk tovább.",
    "Holnap kell hit the ground running-gyel indulni.",
    "Muszáj volt burn the midnight oil, de sikerült.",
    "A csapat gyorsan elindult szombaton reggel",
    "This only really makes sense in the city.",
    "I will go to the store tomorrow",
]

REPEATS = 200


def bench() -> dict:
    # prepare: each corpus entry through the production path
    prepared = [(normalize_for_speech(t),) for t in CORPUS]
    samples: list[float] = []
    for _ in range(REPEATS):
        for (norm,) in prepared:
            t0 = time.perf_counter_ns()
            host = detect_language(norm)
            segment_language_spans(norm, host)
            samples.append((time.perf_counter_ns() - t0) / 1000.0)
    samples.sort()
    n = len(samples)
    return {
        "samples": n,
        "median_us": round(statistics.median(samples), 2),
        "p90_us": round(samples[int(n * 0.90) - 1], 2),
        "max_us": round(samples[-1], 2),
        "mean_us": round(statistics.fmean(samples), 2),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench(), indent=2))
