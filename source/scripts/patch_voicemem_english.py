#!/usr/bin/env python3
"""Localise the vendored VoiceMem package's trait/emotion extraction to English.

The vendored VoiceMem package (controlled fork, vendor/voicemem - the
localisation is folded into the fork as patch VM-LOCAL-EN; this script is
now the idempotent VERIFIER the installer runs to prove the marker is
present) originally prompted the LLM in Chinese for right-brain trait labels
("5-15 Chinese characters") and emotion tags ("a single Chinese word") — on a
Hungarian/English conversation that lands as Chinese nodes on the memory
graph, which the web UI then shows verbatim (the v0.4.6 field report:
memory cards with 纠错时直截了当 / 应对方式).

This script patches ONE file, ``voicemem/leftbrain/merged_extraction.py``:
the PROMPT_ADDENDUM block between the anchors "And add ONE top-level field"
and "OUTPUT SHAPE" is replaced with an English-localised version. The five
slot VALUES (情绪/应对方式/表达风格/思维模式/喜好与厌恶) stay verbatim —
they are the enum the package's TraitStore.add() validates against — but the
labels, the emotion word and every example become British English.

The LEFT brain is untouched: its extraction prompt (additive_extraction_
prompt.txt) is already English. The slot values remain machine-facing enums;
the web UI displays them through app/web_server.py's SLOT translation map.

Idempotent: skips when the English marker is already present (exit 0). The
controlled fork ships the localisation applied; the installer runs this
script as a guard - a MISSING marker means the vendor tree is not the
controlled fork (exit 1 -> install aborts).

Usage (from the repo root or with -v):
    python scripts/patch_voicemem_english.py [vendor_dir]

Exit codes: 0 = patched or already patched; 1 = vendor file missing / anchor
not found (the pinned ref changed upstream — needs a manual review).
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Marker proving the English localisation is already in place.
_MARKER = "IN ENGLISH (British spelling)"

#: Start/end anchors of the block to replace (exact, from the pinned v0.0.1).
_START_ANCHOR = 'And add ONE top-level field "traits"'
_END_ANCHOR = "OUTPUT SHAPE"

#: The English replacement block. Format-string safe: the only literal braces
#: are the doubled {{ }} pairs (PROMPT_ADDENDUM is .format(slots=...)'ed).
_ENGLISH_BLOCK = '''And add ONE top-level field "traits": subjective things this utterance reveals about
the speaker. Each item {{"slot": "...", "label": "<3-8 words>"}}, slot is one of
EXACTLY these five values (copy the value verbatim - it is an enum):

  情绪        WHEN they feel WHAT - the situation plus the feeling it triggers.
              e.g. "gets nervous before reviews", "irritated when interrupted"
  应对方式     what they DO about a feeling, or how they want to be treated.
              e.g. "wants comfort when stressed", "prefers to be alone when upset"
  表达风格     habits of speaking and communicating.
              e.g. "gives examples before conclusions"
  思维模式     how they think, weigh things, decide.
              e.g. "weighs every option before deciding"
  喜好与厌恶   what they like or dislike.
              e.g. "dislikes long meetings"

情绪 vs 应对方式 is the one people get wrong: "irritated when interrupted" is
情绪 (a feeling appearing), "walks away when interrupted" is 应对方式 (an
action taken). If the label has no verb of doing or wanting in it, it is 情绪.

For "情绪" the label must read as **a pattern, not a bare feeling word**:
"gets nervous before reviews" - NOT "anxious" / "happy".
It becomes the title of a node on a graph; a bare word tells the user nothing.

When the utterance states a RECURRING tendency about the speaker - "always",
"every time", "never", "I am the kind of person who ...", or any habit or
reaction that clearly holds beyond this one moment - a trait is REQUIRED. "I
always drift off in long meetings" is 喜好与厌恶 "dislikes long meetings";
"I never sleep before a presentation" is 情绪 "sleepless before presentations".
Do not skip it just because the same content also went into "memory":
"memory" records WHAT HAPPENED, "traits" records WHAT THIS PERSON IS LIKE,
and one sentence very often carries both.

Outside that case, only include a category the utterance clearly shows -
for a one-off event or a plain question, "traits": [] is the right answer.

Each label becomes the TITLE of a node on a graph, so write it as a short
pattern IN ENGLISH (British spelling) - 3 to 8 words, no subject, no full
stop:
  good: hates being interrupted / wants comfort when stressed / conclusions before explanations
  bad: The user tends to plan in detail. (a full sentence with a subject)
  bad: I study computer science (copied from the utterance)

Also add ONE top-level field "emotion": how the speaker feels, as a single
English word (happy / calm / anxious / sad / wronged / angry / surprised /
tired / disappointed ...).
**Judge from what they actually say.** "I love strawberries" is happy, not
sad; "I am so angry" is angry, not anxious. If the utterance carries no clear
feeling (a plain fact, a question), return "" - an empty string is the right
answer far more often than a guess. A wrong label is worse than none: it gets
shown to the user as [label] next to their own words.

Never invent entities, traits or feelings that are not in the text.

Keep one-off requests OUT of "memory": asking for a recommendation, asking
what you remember about them, asking you to do something right now. When the
same sentence ALSO states a lasting fact, write only the lasting half - never
both in one item. "I have a GRE exam next week, any book recommendations?"
gives exactly one memory, "the user takes the GRE exam next week", and
nothing about the book request.

'''


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    vendor = Path(argv[1]) if len(argv) > 1 else root / "vendor" / "voicemem"
    target = vendor / "voicemem" / "leftbrain" / "merged_extraction.py"
    if not target.is_file():
        print(f"[patch-voicemem] vendor file not found: {target}")
        print("[patch-voicemem] run install_m1.ps1 step 12 (vendor clone) first.")
        return 1

    src = target.read_text(encoding="utf-8")
    if _MARKER in src:
        print("[patch-voicemem] already localised (English trait prompt) - nothing to do.")
        return 0
    if _START_ANCHOR not in src or _END_ANCHOR not in src:
        print(
            "[patch-voicemem] ANCHOR NOT FOUND in merged_extraction.py - the pinned "
            "vendor ref changed upstream; review the diff manually."
        )
        return 1

    start = src.index(_START_ANCHOR)
    end = src.index(_END_ANCHOR)
    patched = src[:start] + _ENGLISH_BLOCK + src[end:]

    # The PROMPT_ADDENDUM is a .format(slots=...) template: single braces would
    # crash prompt_addendum() at runtime. Guard before writing.
    block_only = src[start:end]
    if block_only.count("{") != block_only.count("{{") * 2 or block_only.count(
        "}"
    ) != block_only.count("}}") * 2:
        # The original uses only doubled braces; keep the same property.
        print("[patch-voicemem] unexpected brace layout in the original block - aborting.")
        return 1

    target.write_text(patched, encoding="utf-8")

    # Syntax check: a broken patch must never silently kill the vendor import.
    import py_compile

    try:
        py_compile.compile(str(target), doraise=True)
    except py_compile.PyCompileError as exc:  # pragma: no cover - safety net
        print(f"[patch-voicemem] syntax check FAILED, restoring original: {exc}")
        target.write_text(src, encoding="utf-8")
        return 1

    print(
        "[patch-voicemem] merged_extraction.py localised: trait labels and emotion "
        "tags are now extracted in English (slot values stay the package enum)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
