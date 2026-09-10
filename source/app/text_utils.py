"""Text utilities: HU/EN language detection, sentence splitting, stream chunking.

Pure standard library only (importable anywhere, no heavy dependencies).
Used by the pipeline to pick TTS voices per language and to convert streaming
LLM deltas into speakable chunks for low-latency synthesis.
"""

from __future__ import annotations

from typing import Optional

LANG_HU = "hu"
LANG_EN = "en"

# --------------------------------------------------------------------------- #
# Language detection tables
# --------------------------------------------------------------------------- #

#: Hungarian-specific diacritic letters (lowercase; tokens are lowercased).
HU_DIACRITICS = frozenset("áéíóöőúüű")

#: Common Hungarian stopwords (weight 2 per token).
HU_STOPWORDS = frozenset({
    "az", "és", "hogy", "egy", "ez", "ezt", "azt", "de", "van", "volt",
    "nem", "is", "meg", "már", "csak", "még", "akkor", "ott", "majd",
    "vagy", "szereted", "szeretem", "szeretnék", "kell", "lesz", "lett",
    "nincs", "mert", "mint", "olyan", "ilyen", "itt", "nagyon", "mindent",
    "valami", "sem", "semmi", "nekem", "neki", "én", "te", "ő", "mi", "ti",
    "ők", "hát",
})

#: Common English stopwords (weight 1 per token).
EN_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "and", "but", "or",
    "of", "to", "in", "on", "at", "for", "with", "as", "that", "this",
    "it", "i", "you", "he", "she", "we", "they", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "am", "be",
    "been", "my", "me", "not", "what", "when", "how", "there", "here",
})

#: Punctuation stripped from token edges before matching (ASCII + Hungarian usage).
_TOKEN_STRIP = (
    " \t\r\n"
    ".,!?;:()[]{}<>\"'`~@#$%^&*_+=/\\|-"
    "–—―„”“’‘…¡¿"
)


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace-separated tokens with edge punctuation stripped."""
    tokens: list[str] = []
    for raw in text.split():
        token = raw.strip(_TOKEN_STRIP).lower()
        if token:
            tokens.append(token)
    return tokens


def detect_language(text: str) -> str:
    """Heuristic HU/EN detection.

    Hungarian diacritic-bearing tokens weigh 3, Hungarian stopwords weigh 2,
    English stopwords weigh 1. Ties go to ``"hu"`` iff any Hungarian diacritic
    is present in the text, otherwise ``"en"``. Empty/whitespace text is
    detected as ``"en"``.
    """
    if not text or not text.strip():
        return LANG_EN

    has_hu_diacritic = any(ch in HU_DIACRITICS for ch in text.lower())

    hu_score = 0
    en_score = 0
    for token in _tokenize(text):
        if any(ch in HU_DIACRITICS for ch in token):
            hu_score += 3
        if token in HU_STOPWORDS:
            hu_score += 2
        if token in EN_STOPWORDS:
            en_score += 1

    if hu_score > en_score:
        return LANG_HU
    if en_score > hu_score:
        return LANG_EN
    return LANG_HU if has_hu_diacritic else LANG_EN


# --------------------------------------------------------------------------- #
# Sentence splitting
# --------------------------------------------------------------------------- #

#: Known abbreviations whose trailing dot does NOT end a sentence.
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "stb", "pl", "ill", "vs", "kb",
    "ún", "st", "jr", "nr",
})

#: Characters that terminate a sentence (runs are kept together).
_TERMINATORS = "!?…"


def _word_before(text: str, dot_index: int) -> str:
    """Lowercase word immediately preceding the dot at *dot_index* (or "")."""
    start = dot_index
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    return text[start:dot_index].lower()


def _sentence_end_positions(text: str) -> list[int]:
    """Cut positions just after each true sentence terminator.

    Honors: abbreviations (``Dr.``, ``stb.`` ...), decimal numbers
    (``3.14``) and dot runs (``...``). Newline runs separate sentences;
    the cut is placed before the run. Positions may repeat; callers filter.
    """
    positions: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\r\n":
            positions.append(i)
            while i < n and text[i] in "\r\n":
                i += 1
            continue
        if ch == ".":
            prev_digit = i > 0 and text[i - 1].isdigit()
            next_digit = i + 1 < n and text[i + 1].isdigit()
            if prev_digit and next_digit:
                i += 1                      # decimal separator (3.14)
                continue
            if i + 1 < n and text[i + 1] == ".":
                i += 1                      # dot run: defer to the last dot
                continue
            if _word_before(text, i) in _ABBREVIATIONS:
                i += 1                      # abbreviation dot (Dr., stb.)
                continue
            positions.append(i + 1)
            i += 1
            continue
        if ch in _TERMINATORS:
            j = i
            while j < n and text[j] in _TERMINATORS:
                j += 1
            positions.append(j)
            i = j
            continue
        i += 1
    return positions


def split_sentences(text: str) -> list[str]:
    """Sentence split keeping abbreviations (``Dr.``, ``Mr.``, ``stb.``, ``vs.``)
    and decimal numbers intact. Terminators stay attached to their sentence;
    newline runs act as separators. Whitespace-only results are stripped.
    """
    if not text:
        return []
    positions = _sentence_end_positions(text)
    sentences: list[str] = []
    start = 0
    for pos in positions:
        if pos <= start:
            continue
        sentences.append(text[start:pos])
        start = pos
    if start < len(text):
        sentences.append(text[start:])
    return [s.strip() for s in sentences if s.strip()]


# --------------------------------------------------------------------------- #
# Speakability helper
# --------------------------------------------------------------------------- #


def is_speakable(chunk: str, min_chars: int = 2) -> bool:
    """Return True when *chunk* is worth synthesizing.

    A chunk is speakable when its stripped length is at least *min_chars*
    and it contains at least one alphanumeric character (pure punctuation
    like ``"..."`` is not speakable).
    """
    if not isinstance(chunk, str):
        return False
    stripped = chunk.strip()
    if len(stripped) < min_chars:
        return False
    return any(ch.isalnum() for ch in stripped)


# --------------------------------------------------------------------------- #
# Streaming chunker
# --------------------------------------------------------------------------- #


class SentenceStream:
    """Buffers streaming LLM deltas and emits speakable text chunks.

    Emission rules (checked after every delta):

    1. A sentence terminator arrives -> the completed sentence is emitted
       immediately (aggressive, low time-to-first-audio). A trailing dot
       preceded by a digit is held back until the next delta because it may
       still turn out to be a decimal separator (``3.14``).
    2. The buffer grows past the current limit without a terminator -> emit
       at the comma/space boundary nearest before the limit (never mid-word).
       The first emission targets *first_chunk_chars* (low TTFB), later ones
       target *chunk_chars*.

    Chunks are stripped and non-empty; text is never emitted twice.
    ``flush()`` returns the remaining buffer when it is at least 4 characters.
    """

    _MIN_REMAINDER_CHARS = 4

    def __init__(self, first_chunk_chars: int = 24, chunk_chars: int = 80) -> None:
        self._first_chunk_chars = max(1, first_chunk_chars)
        self._chunk_chars = max(1, chunk_chars)
        self._buf = ""
        self._emitted_any = False

    def add_delta(self, delta: str) -> list[str]:
        """Append *delta* and return 0+ complete speakable chunks."""
        if not delta:
            return []
        self._buf += delta
        emitted: list[str] = []
        while True:
            limit = self._first_chunk_chars if not self._emitted_any else self._chunk_chars
            cut = self._find_sentence_cut()
            if cut is None and len(self._buf) > limit:
                cut = self._find_boundary_cut(limit)
            if cut is None or cut <= 0:
                break
            chunk, self._buf = self._buf[:cut], self._buf[cut:]
            text = chunk.strip()
            if text:
                emitted.append(text)
                self._emitted_any = True
            if not self._buf:
                break
        return emitted

    def flush(self) -> list[str]:
        """Return the remaining buffer if it is >= 4 characters (stripped).

        Resets the first-chunk window so a reused stream starts aggressively.
        Never emits the same text twice (the buffer is consumed).
        """
        remainder = self._buf.strip()
        self._buf = ""
        self._emitted_any = False
        if len(remainder) >= self._MIN_REMAINDER_CHARS:
            return [remainder]
        return []

    # -- internals ---------------------------------------------------------- #

    def _find_sentence_cut(self) -> Optional[int]:
        """Position after the first complete sentence in the buffer, if any."""
        buf = self._buf
        for pos in _sentence_end_positions(buf):
            if pos <= 0:
                continue
            # A buffer-final dot preceded by a digit may still become a
            # decimal separator once more deltas arrive; hold it back.
            if (
                pos == len(buf)
                and buf.endswith(".")
                and len(buf) >= 2
                and buf[-2].isdigit()
            ):
                break
            return pos
        return None

    def _find_boundary_cut(self, limit: int) -> Optional[int]:
        """Cut at the comma/space boundary nearest before *limit*, if any."""
        window = self._buf[:limit]
        cut = -1
        for i, ch in enumerate(window):
            if ch == "," or ch.isspace():
                cut = i
        if cut <= 0:
            return None
        return cut + 1
