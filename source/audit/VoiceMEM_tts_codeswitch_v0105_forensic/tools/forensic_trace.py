#!/usr/bin/env python3
"""FORENSIC TRACE: TTS code-switching pipeline, v0.10.5 (audit-only).

Drives the REAL production functions of the web speak path — no production
code is modified, no audio is synthesized (a recording TTS stub stands in
for Supertonic). The glue below replicates web_server._synthesize_chunk
(web_server.py:3331-3411) and the speak_worker stream (web_server.py:3098,
3210-3213) line-for-line; every imported function is the live one.

Per-chunk trace fields (the operator's required minimum):
    TEXT                 — the chunk leaving SentenceStream (pre-normalize)
    HOST_LANGUAGE        — detect_language(chunk)            [chunk-level]
    DETECTED_ENGLISH_SPANS — segment_language_spans output, EN spans only
    SPAN_LANGUAGE        — per-span language from segmentation
    ROUTED_LANGUAGE      — language argument of the tts.synthesize call
    SELECTED_VOICE       — VoiceSettings.resolve() preset for the chunk
    FINAL_TTS_TEXT       — text actually handed to the engine per span
                           (post normalize_for_speech)

Also records, per case: the exact ``mode auto`` diag line the web server
logs (web_server.py:3377-3380) and the SentenceStream chunk boundaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from app.text_utils import (  # noqa: E402
    LANG_EN,
    LANG_HU,
    SentenceStream,
    detect_language,
    normalize_for_speech,
    segment_language_spans,
)
from app.tts_supertonic import _LANG_MAP, _DEFAULT_LANG  # noqa: E402
from app.voice_settings import VoiceSettings, voice_label  # noqa: E402

# Production SentenceStream parameters (app/config.py:197-198).
FIRST_CHUNK_CHARS = 24
CHUNK_CHARS = 80

# The 10 mandatory test sentences (operator order, numbered as given).
CASES = [
    ("T1", "Let's cut to the chase."),
    ("T2", "Sometimes you just have to hit the ground running."),
    ("T3", "We had to burn the midnight oil."),
    ("T4", "Before the show, everyone said break a leg."),
    ("T5", "Most már tényleg cut to the chase, és menjünk tovább."),
    ("T6", "Ma először checkeljük a resultot, aztán megnézzük a statuszt, "
           "végül pedig beszélünk a next step-ről."),
    ("T7", "A mai test során először checkeljük a resultot, aztán megnézzük "
           "a statuszt, végül pedig beszélünk a next step-ről."),
    ("T8", "Holnap lesz a meeting, utána pedig catch up."),
    ("T9", "Touch base után megbeszéljük a projektet."),
    ("T10", "Ez a projekt fontos, de most nem akarok meetinget."),
]


class RecordingTts:
    """Stands in for SupertonicTtsEngine; records every synthesize call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def synthesize(self, text, language, length_scale=None, voice=None):
        self.calls.append(
            {
                "text": text,
                "language": language,          # app-level hu/en
                "supertonic_lang": _LANG_MAP.get(language, _DEFAULT_LANG),
                "voice": voice or "",
                "length_scale": length_scale,
            }
        )
        return b"PCM"  # non-empty marker


def feed_sentence_stream(sentence: str, delta_size: int):
    """SentenceStream over *sentence* fed in *delta_size*-char deltas.

    Returns (chunks, delta_count) — the exact chunks the speak_worker would
    put on speak_queue (web_server.py:3210-3213 incl. the final flush).
    """
    stream = SentenceStream(
        first_chunk_chars=FIRST_CHUNK_CHARS, chunk_chars=CHUNK_CHARS
    )
    chunks: list[str] = []
    deltas = 0
    for i in range(0, len(sentence), delta_size):
        for chunk in stream.add_delta(sentence[i : i + delta_size]):
            chunks.append(chunk)
        deltas += 1
    for chunk in stream.flush():
        chunks.append(chunk)
    return chunks, deltas


def trace_case(case_id: str, sentence: str, delta_size: int) -> dict:
    """Replicates web_server._synthesize_chunk for every streamed chunk."""
    voice_settings = VoiceSettings(mode="auto")  # production default mode
    tts = RecordingTts()
    chunks, n_deltas = feed_sentence_stream(sentence, delta_size)
    per_chunk: list[dict] = []

    for chunk in chunks:
        # --- web_server.py:3364 (normalize BEFORE detection) --------------- #
        norm = normalize_for_speech(chunk)
        # --- web_server.py:3367 (chunk-level host detection) --------------- #
        language = detect_language(norm) if norm.strip() else LANG_HU
        # --- web_server.py:3371-3380 (mode-aware voice resolution) --------- #
        voice_id = ""
        if True:  # c.voice is not None on the production web server
            voice_id, language = voice_settings.resolve(language)
        diag_line = (
            f"TTS voice: {voice_label(voice_id)} "
            f"({'Hungarian' if language == LANG_HU else 'English'}, "
            f"mode {voice_settings.mode})"
        )
        # --- web_server.py:3383-3385 (span segmentation, auto mode only) --- #
        spans: list[tuple[str, str]] = [(norm, language)]
        if voice_settings.mode == "auto":
            spans = segment_language_spans(norm, language)
        # --- web_server.py:3389-3394 (per-span synthesis) ------------------- #
        first_call = len(tts.calls)
        for span_text, span_lang in spans:
            tts.synthesize(span_text, span_lang, None, voice_id or None)
        per_chunk.append(
            {
                "TEXT": chunk,
                "FINAL_INPUT_AFTER_NORMALIZE": norm,
                "HOST_LANGUAGE": language,
                "DETECTED_ENGLISH_SPANS": [
                    t for t, l in spans if l == LANG_EN
                ],
                "SPANS": [
                    {"SPAN_TEXT": t, "SPAN_LANGUAGE": l} for t, l in spans
                ],
                "SELECTED_VOICE": voice_id,
                "DIAG_LINE": diag_line,
                "SYNTH_CALLS": [
                    {
                        "ROUTED_LANGUAGE": c["language"],
                        "SUPERTONIC_LANG": c["supertonic_lang"],
                        "FINAL_TTS_TEXT": c["text"],
                        "VOICE": c["voice"],
                    }
                    for c in tts.calls[first_call:]
                ],
            }
        )

    idioms = ["cut to the chase", "hit the ground running",
              "burn the midnight oil", "break a leg"]
    idiom_verdicts = {}
    joined = json.dumps(per_chunk, ensure_ascii=False)
    for idiom in idioms:
        if idiom in sentence.lower():
            # What language(s) did the idiom's words actually get?
            langs = set()
            for pc in per_chunk:
                for call in pc["SYNTH_CALLS"]:
                    for word in idiom.split():
                        if word in call["FINAL_TTS_TEXT"].lower().split():
                            langs.add(call["ROUTED_LANGUAGE"])
            idiom_verdicts[idiom] = sorted(langs)

    return {
        "case": case_id,
        "sentence": sentence,
        "delta_size": delta_size,
        "n_deltas": n_deltas,
        "n_chunks": len(chunks),
        "n_synth_calls": len(tts.calls),
        "chunks": per_chunk,
        "idiom_word_routing": idiom_verdicts,
    }


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "evidence" / "forensic_trace.json"
    results = []
    # Two realistic streaming regimes + the whole-sentence control:
    #   3-char deltas  ~ fast local LLM token bursts
    #   8-char deltas  ~ typical streamed fragments
    #   whole-sentence ~ the no-chunking control (single delta)
    for case_id, sentence in CASES:
        for delta_size in (3, 8, len(sentence) or 1):
            results.append(trace_case(case_id, sentence, delta_size))
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")

    # Human-readable summary
    lines = []
    for r in results:
        lines.append(
            f"[{r['case']}] delta={r['delta_size']:>3}  "
            f"chunks={r['n_chunks']}  calls={r['n_synth_calls']}"
        )
        for pc in r["chunks"]:
            en_spans = " | ".join(pc["DETECTED_ENGLISH_SPANS"]) or "(none)"
            calls = " || ".join(
                f"{c['ROUTED_LANGUAGE']}:{c['FINAL_TTS_TEXT']!r}"
                for c in pc["SYNTH_CALLS"]
            )
            lines.append(f"    TEXT={pc['TEXT']!r}")
            lines.append(f"      HOST={pc['HOST_LANGUAGE']}  VOICE={pc['SELECTED_VOICE']}")
            lines.append(f"      EN_SPANS={en_spans}")
            lines.append(f"      CALLS={calls}")
        for idiom, langs in r["idiom_word_routing"].items():
            lines.append(f"    IDIOM {idiom!r} -> word languages {langs}")
        lines.append("")
    summary = Path(__file__).resolve().parents[1] / "evidence" / "forensic_trace.txt"
    summary.write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))
    print(f"[written] {out}")
    print(f"[written] {summary}")


if __name__ == "__main__":
    main()
