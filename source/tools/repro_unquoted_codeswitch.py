"""Forensic reproduction: unquoted code-switching on v0.10.5 HEAD.

Production field report (2026-09-18, sessions ~19:31-19:35 local):
"SEE YOU LATER" and other UNQUOTED English phrases inside Hungarian
assistant replies are read with the Hungarian Supertonic model
(observed phonetic output similar to "se ju...").

This harness drives the exact production decision path
(detect_language + segment_language_spans, auto voice mode — the same
functions web_server._synthesize_chunk and pipeline._speak_chunk call)
with the reported and required sentences and records the language
decision for every span.

Run:  python tools/repro_unquoted_codeswitch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.text_utils import detect_language, segment_language_spans  # noqa: E402

#: (case id, sentence, expected English substrings — phrases that MUST
#:  reach the EN model; empty list = the whole sentence must stay HU/EN host)
CASES = [
    ("D1-hu-pure",      "Szeretnék röviden beszélni veled az új projektről.", []),
    ("D2-hu-loan-meeting", "A meeting után átbeszéljük a projektet.", []),
    ("D3-hu-hazai",     "A hazai megoldás olcsóbb, mint a tea és az autó.", []),
    ("F1-quoted",       'A "touch base" egy gyakori angol kifejezés.', ["touch base"]),
    ("F2-unquoted-tb",  "Szeretnék egy gyors touch base-t veled.", ["touch base"]),
    ("F3-unquoted-cu", "Szerintem később catch up-olhatunk.", ["catch up"]),
    ("F4-unquoted-syl", "Rendben, see you later.", ["see you later"]),
    ("F5-caps-syl",     "Rendben, SEE YOU LATER!", ["SEE YOU LATER"]),
    ("E1-en-pure",      "I'd like to touch base with you.", []),
    ("E2-en-pure",      "Let's catch up later.", []),
    ("E3-en-pure",      "See you later.", []),
]


def main() -> int:
    print(f"{'case':22s} {'host':5s} spans  decisions")
    print("-" * 88)
    failures: list[str] = []
    for case_id, sentence, expected_en in CASES:
        host = detect_language(sentence)
        spans = segment_language_spans(sentence, host)
        decisions = [(t.strip() or t, lang) for t, lang in spans]
        joined = " | ".join(f"[{lang}] {t[:44]!r}" for t, lang in decisions)
        print(f"{case_id:22s} {host:5s} {len(spans):<5d} {joined}")
        for phrase in expected_en:
            hit = any(lang == "en" and phrase.lower() in t.lower() for t, lang in decisions)
            if not hit:
                failures.append(f"{case_id}: expected EN routing for {phrase!r}")
        if not expected_en and len(spans) != 1:
            failures.append(f"{case_id}: expected a single host span, got {len(spans)}")
    print("-" * 88)
    if failures:
        print("FAILURES (current v0.10.5 behaviour):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("no failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
