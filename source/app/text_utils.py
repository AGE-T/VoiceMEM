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

#: [external audit v1.0 F-O fix] Hungarian function words that carry NO
#: diacritics — they survive informal typing without accents ("szerintem",
#: "persze", "mikor"...). Without these, an accent-free Hungarian sentence
#: with few listed stopwords could lose to English and get the EN voice.
HU_PLAIN_WORDS = frozenset({
    "szia", "sziasztok", "persze", "koszi", "koszonom", "szivesen",
    "mikor", "hol", "ki", "mit", "miert", "rendben", "igen",
    "vagyok", "voltam", "leszek", "tudom", "tudok", "tudsz",
    "akarom", "akarok", "gondolom", "gondoltam", "szerintem", "na",
    "ugye", "talalkozunk", "meseld",
})

#: [F-O fix] Hungarian digraphs — they appear in most Hungarian words even
#: when typed without diacritics, and are rare in English text.
HU_DIGRAPHS = ("sz", "cs", "gy", "ny", "ly", "ty", "zs")

#: [F-O fix] q is essentially absent from native Hungarian orthography (and
#: from its diacritic-free typing); in English it appears in content words
#: ("question", "quick"). w is deliberately NOT used: too many English
#: stopwords carry w ("was/what/when/...") and a w-bonus double-counts them,
#: misrouting mixed text like "The tejeskávé was excellent".
EN_ONLY_LETTERS = frozenset("q")

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

    Hungarian diacritic-bearing tokens weigh 3, Hungarian stopwords weigh 2
    (plus diacritic-free Hungarian function words, weight 2 — F-O fix),
    English stopwords weigh 1. A Hungarian digraph-density signal (sz/cs/gy/
    ny/ly/ty/zs in at least half the tokens) adds +2 to the Hungarian score;
    any q in the text adds +2 to English (native Hungarian lacks it). Ties go to ``"hu"`` iff any Hungarian diacritic is present in
    the text, otherwise ``"en"``. Empty/whitespace text is detected as
    ``"en"``.
    """
    if not text or not text.strip():
        return LANG_EN

    lowered = text.lower()
    has_hu_diacritic = any(ch in HU_DIACRITICS for ch in lowered)

    tokens = _tokenize(text)
    hu_score = 0
    en_score = 0
    for token in tokens:
        if any(ch in HU_DIACRITICS for ch in token):
            hu_score += 3
        if token in HU_STOPWORDS:
            hu_score += 2
        if token in HU_PLAIN_WORDS:
            hu_score += 2
        if token in EN_STOPWORDS:
            en_score += 1

    # [F-O fix] digraph density: Hungarian words carry sz/cs/gy/ny/... even
    # when typed without diacritics; English rarely does. Only fires when
    # there are at least 3 tokens, so short mixed fragments stay unbiased.
    if len(tokens) >= 3:
        digraph_tokens = sum(1 for t in tokens if any(d in t for d in HU_DIGRAPHS))
        if digraph_tokens / len(tokens) >= 0.5:
            hu_score += 2

    # [F-O fix] q is essentially absent from native Hungarian orthography.
    if any(ch in EN_ONLY_LETTERS for ch in lowered):
        en_score += 2

    if hu_score > en_score:
        return LANG_HU
    if en_score > hu_score:
        return LANG_EN
    return LANG_HU if has_hu_diacritic else LANG_EN


# --------------------------------------------------------------------------- #
# Span-level language segmentation (TTS code-switching, v0.10.4 field fix)
# --------------------------------------------------------------------------- #
#
# Field report (v0.10.4): language was selected ONCE PER TTS CHUNK, so an
# English phrase embedded inside a Hungarian sentence ("A "touch base" egy
# gyakori angol kifejezés.") was synthesized with the Hungarian Supertonic
# model — Hungarian grapheme-to-phoneme rules read the English graphemes and
# the phrase became unintelligible. ``detect_language`` cannot provide span
# boundaries (it returns exactly one label for the whole input), so this
# section adds a SEPARATE span segmentation used by the TTS call sites. The
# policy is deliberately narrow — see ``segment_language_spans``.

#: English contractions (ASCII apostrophes; ``normalize_for_speech`` maps
#: U+2019 first). Hungarian orthography never uses apostrophes, so these are
#: unambiguous English evidence, and the apostrophe must never be treated as
#: a quote delimiter or a word boundary ("I'd", "don't", "we're").
EN_CONTRACTIONS = frozenset({
    "i'd", "i'll", "i'm", "i've",
    "you'd", "you'll", "you're", "you've",
    "he'd", "he'll", "he's", "she'd", "she'll", "she's",
    "it'd", "it'll", "it's",
    "we'd", "we'll", "we're", "we've",
    "they'd", "they'll", "they're", "they've",
    "that's", "there's", "what's", "who's", "let's",
    "ain't", "can't", "couldn't", "didn't", "doesn't", "don't",
    "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
    "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't",
    "wouldn't", "y'all",
})

#: Letter pairs Hungarian orthography never writes natively (Hungarian uses
#: cs/k/s/t/f/v/g/kv for these sounds) but English words carry constantly.
#: Any of these inside a word is strong English evidence. "ch" is included
#: even though Hungarian loans exist ("technika", "pech") — only the QUOTED
#: span detector consumes it, so the worst case is one loanword in quotes
#: read with English phonetics (documented, intelligible).
EN_CONSONANT_CLUSTERS = ("ch", "ck", "sh", "th", "ph", "wh", "gh", "qu")

#: Vowel pairs impossible in native Hungarian words ("ou" in "touch", "ee" in
#: "need", "oo" in "good"). "au"/"ea"/"eu" are deliberately absent ("autó",
#: "tea", "euro" are everyday Hungarian loans), and so is "ai": Hungarian
#: forms adjectives in "-ai" ("hazai" — domestic) constantly, and that word
#: class matters far more than the English words it would have caught.
EN_VOWEL_CLUSTERS = ("ou", "ee", "oo", "oa", "ue", "ui", "ei")

#: Hungarian digraphs that do NOT collide with common English words. The
#: other four digraphs are excluded on purpose: "only"/"really" carry "ly",
#: "city"/"party" carry "ty", "many"/"money" carry "ny", and using them
#: would shred ordinary English sentences.
HU_STRONG_DIGRAPHS = ("sz", "cs", "gy", "zs")

#: Word-level English stopwords: the chunk-level table minus the Hungarian
#: function words "a" (the HU definite article) and "is" (HU "also") — at
#: word level those two are Hungarian first and would shred every Hungarian
#: sentence into alternating spans.
EN_WORD_STOPWORDS = frozenset(EN_STOPWORDS) - {"a", "is"}


def _classify_word(token: str) -> str:
    """One classification for one (lowercased, edge-stripped) token.

    Returns ``LANG_HU`` / ``LANG_EN`` or ``""`` (neutral). Priority order
    matters:

    1. digits / URLs / e-mails are neutral — they never carry language;
    2. a Hungarian diacritic beats every English signal ("chatelünk" is
       Hungarian despite the "ch");
    3. English letter clusters beat the stopword tables, so quoted CONTENT
       words with positive English evidence ("touch", "catch") classify as
       English even though no stopword table lists them;
    4. everything else (names, signal-free loans like "projekt", "base")
       is neutral and stays with the host language.
    """
    if not token:
        return ""
    if any(ch.isdigit() for ch in token) or "://" in token or "@" in token:
        return ""
    if any(ch in HU_DIACRITICS for ch in token):
        return LANG_HU
    if token in EN_CONTRACTIONS:
        return LANG_EN
    if any(c in token for c in EN_CONSONANT_CLUSTERS) or "q" in token:
        return LANG_EN
    if any(c in token for c in EN_VOWEL_CLUSTERS):
        return LANG_EN
    if token in HU_STOPWORDS or token in HU_PLAIN_WORDS:
        return LANG_HU
    if any(d in token for d in HU_STRONG_DIGRAPHS):
        return LANG_HU
    if token in EN_WORD_STOPWORDS:
        return LANG_EN
    return ""


def _scan_regions(text: str) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` ranges of paired set-off delimiters.

    * Double quotes pair greedily (``normalize_for_speech`` already mapped
      every typographic double-quote form to ASCII ``"``).
    * Parentheses pair at nesting level one.
    * Single quotes pair only when alnum-boundary rules prove an opening
      (preceded by a non-alphanumeric, followed by an alphanumeric) or a
      closing (preceded by an alphanumeric, followed by a non-alphanumeric)
      — so the apostrophes of contractions ("I'd", "don't", "base's") are
      never treated as delimiters.

    Unclosed regions are dropped and a region overlapping an already
    accepted one is skipped — both degrade to plain host text, never crash.
    """
    regions: list[tuple[int, int]] = []
    n = len(text)
    dq = par = sq = -1
    for i, ch in enumerate(text):
        if ch == '"':
            if dq < 0:
                dq = i
            else:
                regions.append((dq, i))
                dq = -1
        elif ch == "(":
            if par < 0:
                par = i
        elif ch == ")":
            if par >= 0:
                regions.append((par, i))
                par = -1
        elif ch == "'":
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < n else ""
            if prev.isalnum() and nxt.isalnum():
                continue  # internal apostrophe (contraction) — not a delimiter
            if sq >= 0 and prev.isalnum():
                regions.append((sq, i))
                sq = -1
            elif sq < 0 and nxt.isalnum():
                sq = i
            # anything else: a stray apostrophe — ignore
    accepted: list[tuple[int, int]] = []
    last_end = -1
    for start, end in regions:
        if start > last_end:
            accepted.append((start, end))
            last_end = end
    return accepted


def _region_language(content: str, host: str, foreign: str) -> str:
    """Language of one set-off region's content — evidence-gated.

    * no foreign evidence → host: a quoted native, signal-free phrase never
      splits (``A "hazai" megoldás`` stays one Hungarian chunk);
    * foreign evidence, no host evidence → foreign: ``"touch base"`` inside
      Hungarian (the reported production bug);
    * evidence on both sides → the existing weighted ``detect_language``
      arbitrates, and only a FOREIGN verdict splits (conservative).
    """
    ev = {LANG_HU: 0, LANG_EN: 0}
    for token in _tokenize(content):
        cls = _classify_word(token)
        if cls:
            ev[cls] += 1
    if ev[foreign] == 0:
        return host
    if ev[host] == 0:
        return foreign
    detected = detect_language(content)
    return detected if detected == foreign else host


def segment_language_spans(text: str, host_language: str) -> list[tuple[str, str]]:
    """Split *text* into ``(span_text, span_lang)`` segments for TTS.

    The production TTS path selects one Supertonic language per synthesis
    call; before this fix the unit of selection was the whole CHUNK, which
    made embedded foreign phrases unintelligible. This function segments a
    chunk at EXPLICIT set-off boundaries only — paired quotes (double or
    single) and parentheses — so:

    * pure Hungarian / pure English text returns ONE span and the caller
      synthesizes exactly as before (byte-identical behaviour);
    * a quoted foreign phrase becomes its own span with its own language,
      spoken by the SAME voice (voice continuity is the caller's job);
    * unquoted code-switching ("touch base-elni" without quotes) stays with
      the host language: word-level runs cannot place phrase boundaries
      without a lexicon ("touch" is English-signal-bearing, "base" is
      signal-neutral) and Hungarian loans ("technika") would false-positive.
      Documented policy — the field-reported cases are all quoted;
    * forced voice modes are handled by the CALLER (they skip this
      function entirely and keep today's single-language behaviour).

    Hard invariant: ``"".join(t for t, _ in result) == text`` — no character
    is lost, added or reordered, so playback ordering cannot change.
    """
    if not text or not text.strip():
        return [(text, host_language)]
    host = LANG_EN if host_language == LANG_EN else LANG_HU
    foreign = LANG_HU if host == LANG_EN else LANG_EN
    spans: list[tuple[str, str]] = []
    pos = 0
    for start, end in _scan_regions(text):
        if start > pos:
            spans.append((text[pos:start], host))
        spans.append(
            (
                text[start : end + 1],
                _region_language(text[start + 1 : end], host, foreign),
            )
        )
        pos = end + 1
    if pos < len(text):
        spans.append((text[pos:], host))
    merged: list[tuple[str, str]] = []
    for span_text, span_lang in spans:
        if not span_text:
            continue
        if merged and merged[-1][1] == span_lang:
            merged[-1] = (merged[-1][0] + span_text, span_lang)
        else:
            merged.append((span_text, span_lang))
    return merged or [(text, host)]


# --------------------------------------------------------------------------- #
# Speech normalisation (TTS input)
# --------------------------------------------------------------------------- #

#: v0.10.3 (upstream audit PART 10): Supertonic 3 rejects unsupported Unicode
#: characters — the field-reported case was the Hungarian opening quote „
#: (U+201E) killing the whole synthesis. The map below replaces typographic
#: and Unicode punctuation with ASCII equivalents that every TTS engine
#: accepts, WITHOUT touching the spoken content: quotes become straight
#: quotes (still punctuation, never spoken), dashes become hyphens, the
#: ellipsis becomes three dots, invisible/formatting code points are
#: dropped or turned into plain spaces.
_SPEECH_CHAR_MAP = {
    "\u201E": '"',   # „ LOW DOUBLE QUOTE (the reported failure)
    "\u201C": '"',   # “ LEFT DOUBLE QUOTATION MARK
    "\u201D": '"',   # ” RIGHT DOUBLE QUOTATION MARK
    "\u2018": "'",   # ‘ LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # ’ RIGHT SINGLE QUOTATION MARK
    "\u201A": ",",   # ‚ LOW SINGLE QUOTE (Hungarian/German decimal comma use)
    "\u00AB": '"',   # «
    "\u00BB": '"',   # »
    "\u2013": "-",   # – EN DASH
    "\u2014": "-",   # — EM DASH
    "\u2015": "-",   # ― HORIZONTAL BAR
    "\u2212": "-",   # − MINUS SIGN
    "\u2026": "...",  # … HORIZONTAL ELLIPSIS
    "\u00A0": " ",   # NO-BREAK SPACE
    "\u202F": " ",   # NARROW NO-BREAK SPACE
    "\u2007": " ",   # FIGURE SPACE
    "\u2009": " ",   # THIN SPACE
    "\u200A": " ",   # HAIR SPACE
    "\u200B": "",    # ZERO WIDTH SPACE (drop)
    "\u200C": "",    # ZWNJ (drop)
    "\u200D": "",    # ZWJ (drop)
    "\uFEFF": "",    # BOM / zero-width no-break space (drop)
    "\u3000": " ",   # IDEOGRAPHIC SPACE
    "\u02DC": "~",   # ˜ SMALL TILDE
    "\u00A1": "!",   # ¡
    "\u00BF": "?",   # ¿
    "\u2032": "'",   # ′ PRIME
    "\u2033": '"',   # ″ DOUBLE PRIME
    "\u2044": "/",   # ⁄ FRACTION SLASH
}

#: Characters dropped outright (invisible/unsupported, zero spoken content).
_SPEECH_DROP_CHARS = "\u200B\u200C\u200D\uFEFF"


def normalize_for_speech(text: str) -> str:
    """Return *text* with unsupported/typographic characters made speakable.

    v0.10.3: applied at the single TTS choke point before every synthesis
    call. Guarantees:
      * letters, digits and ASCII punctuation pass through untouched;
      * the replacement NEVER changes spoken content (quotes stay quotes,
        dashes stay dashes — only the code point changes);
      * runs of introduced whitespace are collapsed so mapping does not
        shift sentence rhythm;
      * the result is never longer than a small constant factor of the
        input (the ellipsis expands 1:3).
    """
    if not text:
        return text
    out = []
    for ch in text:
        mapped = _SPEECH_CHAR_MAP.get(ch)
        if mapped is not None:
            out.append(mapped)
        elif ch in _SPEECH_DROP_CHARS:
            continue
        else:
            out.append(ch)
    joined = "".join(out)
    # Collapse only whitespace we introduced/expanded around; keep single
    # spaces exactly as the writer used them.
    if "  " in joined:
        collapsed: list[str] = []
        prev_space = False
        for ch in joined:
            is_space = ch == " "
            if is_space and prev_space:
                continue
            collapsed.append(ch)
            prev_space = is_space
        joined = "".join(collapsed)
    return joined.strip()


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
