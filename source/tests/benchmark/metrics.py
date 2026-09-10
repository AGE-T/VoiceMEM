"""Pure-stdlib ASR/TTS evaluation metrics (WER, CER, token accuracy, accent preservation).

Machine: ANY (sandbox-safe). Imports nothing beyond the Python standard library,
so it can be unit-tested and imported by every benchmark script in every
environment (the sandbox has no numpy/torch, the target machine has everything).

Definitions (documented, deterministic):
- wer(): word error rate over case-insensitive tokens. Tokenization splits on
  any non-alphanumeric character (whitespace + punctuation, hyphen included),
  keeping Hungarian accented letters as part of words. WER = (S+D+I)/N.
- cer(): character error rate over the raw case-sensitive character sequence
  (whitespace counted as a character).
- token_accuracy(): case-insensitive presence check per requested token; a
  multi-word token (e.g. "Hojsz Tamas") must appear as a CONTIGUOUS word
  sequence in the hypothesis.
- accent_preservation(): ratio of Hungarian accented characters (lowercase
  a-e-i-o-o-o-u-u-u with acute/diaeresis/double-acute, both cases) present in
  the hypothesis, counted as a per-character-type multiset.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

# Alphanumeric runs (unicode-aware): Hungarian letters (incl. a-e-i-o-u accents)
# stay intact, punctuation/whitespace/hyphen act as separators, digits kept.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

_ACCENTED_LOWER = frozenset("aáeéiíoóöőuúüű") - frozenset("aeiou")
# Full sets (lower- and uppercase forms handled via str.casefold).
_ACCENTED_ALL = frozenset("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, split on whitespace and punctuation."""
    return _WORD_RE.findall(text.casefold())


def levenshtein(a: str, b: str) -> int:
    """Plain Levenshtein edit distance between two strings (case-sensitive)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _align_counts(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """Levenshtein alignment counts: (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return (0, 0, m)
    if m == 0:
        return (0, n, 0)
    # Full DP matrix (sizes here are modest: words of an utterance / chars of a
    # sentence), then backtrack counting the three operation types.
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(dist[i - 1][j - 1] + cost, dist[i - 1][j] + 1, dist[i][j - 1] + 1)
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return (subs, dels, ins)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: (substitutions + deletions + insertions) / reference words.

    Case-insensitive; tokenization splits on whitespace AND punctuation
    (Hungarian-friendly: accented letters stay inside tokens, hyphens split).
    Empty reference with a non-empty hypothesis yields 1.0 (100% insertion).
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    n = len(ref_tokens)
    if n == 0:
        return 0.0 if not hyp_tokens else 1.0
    subs, dels, ins = _align_counts(ref_tokens, hyp_tokens)
    return (subs + dels + ins) / n


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate over the raw, case-sensitive character sequence.

    Whitespace counts as a character. Relevant for Hungarian, where accented
    character confusions (o vs. o-acute etc.) must be visible in the metric.
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    n = len(ref_chars)
    if n == 0:
        return 0.0 if not hyp_chars else 1.0
    subs, dels, ins = _align_counts(ref_chars, hyp_chars)
    return (subs + dels + ins) / n


def token_accuracy(reference: str, hypothesis: str, tokens: list[str]) -> dict[str, bool]:
    """Per-token presence check in the hypothesis (case-insensitive).

    - A single-word token passes iff the word occurs among hypothesis tokens.
    - A multi-word token (e.g. "Hojsz Tamas", "flat white") passes iff its
      words occur as a CONTIGUOUS sequence in the hypothesis token list.
    - ``reference`` is kept for interface symmetry only (the caller supplies
      the reference text alongside the hypothesis); it is not used.

    Returns a mapping of the requested token (original spelling) to a bool.
    """
    hyp_words = tokenize(hypothesis)
    result: dict[str, bool] = {}
    for token in tokens:
        wanted = tokenize(token)
        if not wanted:
            result[token] = True
            continue
        if len(wanted) == 1:
            result[token] = wanted[0] in hyp_words
        else:
            result[token] = _contains_contiguous(hyp_words, wanted)
    return result


def _contains_contiguous(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """True iff ``needle`` appears as a contiguous run inside ``haystack``."""
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start] != first:
            continue
        if list(haystack[start : start + len(needle)]) == list(needle):
            return True
    return False


def accent_preservation(reference: str, hypothesis: str) -> float:
    """Ratio of Hungarian accented characters preserved in the hypothesis.

    Counts accented characters (both cases, compared case-insensitively) as a
    per-character multiset: for every accented character type, at most as many
    occurrences as the reference contains count as preserved. A reference
    without accented characters vacuously yields 1.0.
    """
    ref_counts = Counter(c for c in reference.casefold() if c in _ACCENTED_ALL)
    total = sum(ref_counts.values())
    if total == 0:
        return 1.0
    hyp_counts = Counter(c for c in hypothesis.casefold() if c in _ACCENTED_ALL)
    kept = sum(min(count, hyp_counts.get(char, 0)) for char, count in ref_counts.items())
    return kept / total
