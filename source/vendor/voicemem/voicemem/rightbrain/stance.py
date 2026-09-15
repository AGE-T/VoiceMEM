"""[CONTROLLED FORK - patch VM-LOCAL-015] Stance cues for trait contradiction safety.

WHY THIS MODULE EXISTS (v0.9.0 memory semantics, measured evidence):
the v0.8.1 trait merge is cosine-only (``MERGE_THRESHOLD = 0.95``). Real
measurements with the local multilingual-e5-small on this machine
(``scripts/measure_semantic_matrix.py``, docs/SEMANTIC_MATRIX.md) show that
semantic OPPOSITES embed inside the merge band:

    "Thomas szereti a motorokat"  ↔ "Thomas utálja a motorokat"      0.9685
    "likes motorcycles"            ↔ "used to like motorcycles"      0.9865
    "Thomas szereti…"              ↔ "…régen szerette…"              0.9821
    "Thomas likes motorcycles"     ↔ "…except during winter" (HU)    0.9552
    "már nem szereti"              ↔ "megint szereti"                0.9521

So a "hates/used to/except" observation can merge with — and REINFORCE —
the positive trait it contradicts. Similarity is retrieval evidence; it is
not semantic agreement. This module is the deterministic, 0-LLM signal
layer that classifies the *stance* of an observation (current-state
polarity + temporal/modal qualifiers) from surface cues in English,
Hungarian and the Chinese fallback labels, so the merge decision can
refuse to convert a known contradiction into reinforcement.

DELIBERATE LIMITS (documented, not hidden):

* surface cues only — explicit markers ("no longer", "már nem", "utálja").
  Deep semantic opposition with no marker is out of scope (documented gap);
* Hungarian past-tense detection uses the frequent verb-suffix endings plus
  the explicit adverbs; agglutinative edge cases can be missed;
* a mixed sentence is classified by the FIRST matching class in the scan
  order ``uncertain > qualified > past > neg > pos`` — e.g. "talán már
  nem szeretem" classifies as ``uncertain`` (uncertainty must never become
  a hard replacement) and "kivéve télen" as ``qualified``; a predominantly
  historical sentence with a trailing correction clause ("régen szerettem,
  de már nem") classifies as ``past`` → the separate-state outcome (never
  a false reinforcement, never a flip) — position-aware mixed-sentence
  parsing is a documented limitation;
* the affect lexicon covers like/love/enjoy/prefer vs hate/dislike/avoid
  (+ Hungarian equivalents). Pairs outside the lexicon fall back to ""
  (neutral) and NEVER trigger a flip.

The scanner is pure (no I/O, no state) so it is directly unit-testable.
"""
from __future__ import annotations

import re

#: Short Hungarian cue words that ALSO occur inside English words
#: ("volt" ⊂ "revolt", "este" ⊂ "esteem", "csak"…). These must match as
#: real Hungarian words (word-start boundary, limited suffix), never as
#: English substrings. Longer/unambiguous cues stay plain substring matches.
_CUE_BOUNDARY_RE = {
    "volt": re.compile(r"\bvolt\w*"),          # volt / voltam / voltak
    "este": re.compile(r"\beste\b|\bestén\b|\bestefelé\b"),
    "lehet": re.compile(r"\blehet\b|\blehet, hogy|\blehet hogy"),
    "csak": re.compile(r"\bcsak\b|\bcsaknem\b"),
}

#: Stance values. ``""`` = no cue found (neutral / unknown) — the safe
#: default that preserves the pre-v0.9.0 merge behaviour for affect-neutral
#: claims ("gives examples before conclusions").
#:
#: * ``pos``  — current-state positive affect
#: * ``neg``  — current-state negative: explicit negation or antipathy
#: * ``past`` — a statement about a PAST state (temporal coexistence,
#:              not a contradiction: "used to", "régen", past suffixes)
#: * ``qualified``  — a scoped/conditional statement ("except during
#:              winter", "kivéve télen", "not always")
#: * ``uncertain`` — hedged change of state ("might no longer",
#:              "talán már nem") — never a hard replacement
#: * ``future`` — [v0.10] a NOT-YET-VALID state ("will like",
#:              "szeretni fogom", "jövő hónapban") — never merges into
#:              a current row and never flips one; it is its own
#:              (forward-looking) observation. Deliberately narrow: a
#:              present-tense fact about a future EVENT ("flight on
#:              Dec 1") carries no future cue and stays current.
STANCE_VALUES = ("pos", "neg", "past", "qualified", "uncertain", "future", "")

#: Scan order: the FIRST class that matches wins. Uncertainty beats a hard
#: negation ("talán már nem szeretem" → uncertain, not neg); a qualifier
#: beats plain polarity ("kivéve télen" → qualified); a current-state
#: negation beats a past marker when both are present ("régen szerettem,
#: de már nem" → neg: the "no longer" is the current truth, the "régen"
#: half is historical colour); a future marker beats only plain positive
#: ("will like" must never read as current pos).
_SCAN_ORDER = ("uncertain", "qualified", "past", "neg", "future", "pos")

# ── cue tables (lowercase; matched on lowercased text) ──────────────────────

_UNCERTAIN_CUES = (
    # English
    "maybe", "might", "may no longer", "may not", "perhaps", "possibly",
    "probably", "not sure", "not certain", "i think", "i guess", "i wonder",
    "seems to", "not convinced",
    # Hungarian
    "talán", "talan", "lehet, hogy", "lehet hogy", "lehet", "gondolom",
    "nem biztos", "esetleg", "véleményem szerint", "velemenyem szerint",
    "kellene", "talán már", "talan mar",
)

_QUALIFIED_CUES = (
    # English
    "except", "apart from", "but not", "but only", "not always", "not every",
    "only when", "only in", "only during", "only if", "unless", "sometimes",
    "occasionally", "in winter", "at night", "on weekends", "mostly",
    "usually", "depends",
    # Hungarian
    "kivéve", "kiveve", "nem mindig", "nem minden", "csak", "néha", "neha",
    "időnként", "idönkent", "általában", "altalaban", "rendszerint",
    "függ", "attól függ", "téli", "teli", "nyáron", "nyaron",
    "hétvégén", "hetvegen", "este", "reggel",
)

_PAST_CUES = (
    # English — explicit past-state markers (a bare "-ed" is NOT enough:
    # "need"/"indeed" false-positive and event narration is out of scope)
    "used to", "previously", "formerly", "once ", "back then", "as a kid",
    "as a child", "when i was", "in the past", "before, ",
    # Hungarian
    "régen", "regen", "korábban", "korabban", "azelőtt", "azelott",
    "valaha", "egykor", "gyerekkor", "volt", "akkor",
)

#: Hungarian past-tense verb endings. ONLY the unambiguous double-t /
#: vowel-conjugation endings — the single-t endings ("tem"/"tam") also
#: terminate PRESENT-tense definite forms ("szeretem" = I like, present!),
#: so matching them would misclassify every 1st-person present statement
#: as past. The explicit adverbs above carry the primary signal; these
#: endings catch "szerette"/"lakott"-style 3rd-person past without an adverb.
_HU_PAST_ENDINGS = (
    "ttem", "ttam", "ttük", "ttuk", "ttunk", "ttok", "ttek",
    "tta", "tte", "ott", "ett", "ött", "ottam", "ettem",
)
_HU_PAST_MIN_LEN = 6

#: Strong current-state negation — presuppositional cues FIRST (they
#: explicitly reference a prior state, which makes the wider similarity
#: band safe: "no longer likes motorcycles" presupposes the liking).
_NEG_PRESUPPOSITION_CUES = (
    "no longer", "not anymore", "any more", "no more", "stopped", "quit", "gave up",
    "gave it up", "lost interest", "got over", "moved on from", "left behind",
    "swore off", "grew out of", "outgrew",
    "már nem", "mar nem", "nem többé", "nem tobbe", "nem már", "nem mar",
    "abbahagyta", "abbahagytam", "felhagyott", "felhagytam", "megunta",
    "megundorodott", "kinőtte", "kinotte", "túltette", "tultette",
    "lefektette", "kiszállt", "kizallt",
)

#: Current-state negation WITHOUT presupposition ("does not like").
_NEG_PLAIN_CUES = (
    "don't like", "dont like", "doesn't like", "doesnt like", "do not like",
    "does not like", "don't enjoy", "dont enjoy", "doesn't enjoy",
    "doesnt enjoy", "don't love", "dont love", "doesn't love", "doesnt love",
    "never liked", "never enjoyed", "never loved", "not a fan of",
    "no fan of",
    # Hungarian "nem <affect-verb>" handled by regex below
)

#: Antipathy — opposite affect with no negation word. Checked BEFORE the
#: positive lexicon: "dislikes" contains "likes" as a substring.
_ANTIPATHY_CUES = (
    "hate", "hates", "hated", "hating", "dislike", "dislikes", "disliked",
    "can't stand", "cannot stand", "can not stand", "cant stand",
    "detest", "detests", "loathe", "loathes", "avoids", "avoided",
    "fed up", "sick of", "tired of", "turned off by",
    "utál", "utal", "utálja", "utálom", "utáltam", "rühelli", "ruhelli",
    "rühel", "undorodik", "undorodott", "nem állhatja", "nem allhatja",
    "nem szenvedheti", "kerüli", "keruli", "kerülöm", "kerulom",
)

_POS_CUES = (
    "likes", "loves", "enjoys", "prefers", "is fond of", "fond of",
    "is into", "adores", "adores it", "big fan of", "fan of",
    "szereti", "szeretem", "szereted", "szeretjük", "szeretjuk",
    "kedveli", "kedvelem", "imádja", "imadja", "imádom", "imadom",
    "rajong", "őrül", "orul", "tetszik", "kedvel",
)

#: [v0.10] Future-affect cues — the statement asserts a NOT-YET-VALID
#: preference or state change. Hungarian has no future tense; the
#: periphrastic "fog/fogom/fogjuk + infinitive" and explicit future
#: adverbs carry the signal. Present-tense knowledge about future events
#: ("flight on 2026-12-01") deliberately does NOT match (design boundary:
#: current knowledge, not future validity).
_FUTURE_CUES = (
    # English
    "will like", "will love", "will enjoy", "will prefer", "will start",
    "will begin", "will probably like", "will get into", "going to like",
    "going to love", "plans to like", "wants to try",
    # Hungarian (fog-periphrasis on affect verbs + future adverbs)
    "szeretni fog", "kedvelni fog", "imádni fog", "szeretni fogom",
    "kedvelni fogom", "szeretni fogjuk", "kedvelni fogjuk",
    "meg fogja szeretni", "meg fogom szeretni", "majd megszereti",
    "jövő hónapban", "jövő héten", "jövő évben", "jovo honapban",
    "jovo heten", "jovo evben", "hamarosan kedveli", "hamarosan szereti",
)

#: Hungarian negated affect: "nem" + an affect verb within a couple of
#: words ("nem szeretem", "nem kedvelem", "nem is szeretem már").
_HU_NEG_AFFECT_RE = re.compile(
    r"\bnem\b[^.!?]{0,24}\b(szeret|kedvel|imád|imad|rajong|őrül|orul|tetszik|kapott)\w*",
    re.IGNORECASE,
)

#: Affect + negation cue tokens are NOT topic words — the topic-overlap
#: guard for supersession ignores them (plus generic stopwords).
_NON_TOPIC_TOKENS = {
    "likes", "like", "loves", "love", "enjoys", "enjoy", "enjoyed",
    "prefers", "prefer", "hates", "hate", "hated", "dislikes", "dislike",
    "avoid", "avoids", "stopped", "stop", "quit", "gave", "up", "gaveup",
    "used", "previously", "formerly", "once", "back", "then", "no",
    "longer", "not", "never", "always", "sometimes", "except", "only",
    "when", "during", "unless", "might", "may", "maybe", "perhaps",
    "possibly", "probably", "már", "mar", "nem", "többé", "tobbe",
    "szereti", "szeretem", "szeret", "kedveli", "kedvelem", "kedvel",
    "imádja", "imadja", "imád", "imad", "utálja", "utalja", "utál", "utal",
    "régen", "regen", "korábban", "korabban", "valaha", " volt", "volt",
    "abbahagyta", "felhagyott", "kinőtte", "kivéve", "kiveve", "talán",
    "talan", "lehet", "thomas", "the", "user", "a", "an", "of", "in", "on",
    "at", "to", "is", "are", "was", "were", "his", "her", "their", "my",
    "he", "she", "they", "i", "and", "but", "or", "for", "with", "very",
}

#: Hungarian suffixes stripped for topic-token stemming (best-effort
#: inflection tolerance: motor/motorok/motorokat/motorja…).
_HU_STRIP_SUFFIXES = (
    "okat", "eket", "ait", "eit", "jait", "jeit",
    "ban", "ben", "ra", "re", "ba", "be", "val", "vel",
    "nak", "nek", "kok", "kek", "kat", "ket", "ot", "et", "at",
    "t", "a", "e", "ok", "ek", "ja", "ji",
)


def _has_cue(low: str, cues: tuple[str, ...]) -> bool:
    for c in cues:
        rx = _CUE_BOUNDARY_RE.get(c)
        if rx is not None:
            if rx.search(low):
                return True
        elif c in low:
            return True
    return False


def _hu_past_word(low: str) -> bool:
    for tok in re.findall(r"[a-záéíóöőúüű]+", low):
        if len(tok) < _HU_PAST_MIN_LEN:
            continue
        for end in _HU_PAST_ENDINGS:
            if tok.endswith(end):
                return True
    return False


def classify_stance(text: str) -> str:
    """Classify the stance of ONE text (claim or quote).

    Returns a :data:`STANCE_VALUES` member. Pure function; deterministic;
    case-insensitive; language-agnostic across EN/HU (Chinese fallback
    labels contain the same patterns through the slot enum only, which is
    not a stance signal).

    Scan order: ``uncertain > qualified > past > neg > pos`` (see the
    module docstring for why).
    """
    low = (text or "").lower()
    if not low.strip():
        return ""
    if _has_cue(low, _UNCERTAIN_CUES):
        return "uncertain"
    if _has_cue(low, _QUALIFIED_CUES):
        return "qualified"
    if _has_cue(low, _PAST_CUES) or _hu_past_word(low):
        return "past"
    # neg: antipathy first ("dislikes" contains "likes"), then explicit
    # negation cues, then the Hungarian nem+affect pattern.
    if _has_cue(low, _ANTIPATHY_CUES):
        return "neg"
    if _HU_NEG_AFFECT_RE.search(low):
        return "neg"
    if _has_cue(low, _NEG_PLAIN_CUES) or _has_cue(low, _NEG_PRESUPPOSITION_CUES):
        return "neg"
    # [v0.10] future before plain pos: "will like" must never read as a
    # current positive. (After neg so "don't like it now but will" → neg.)
    if _has_cue(low, _FUTURE_CUES):
        return "future"
    if _has_cue(low, _POS_CUES):
        return "pos"
    return ""


#: Cues that explicitly reference a prior state ("no longer", "gave up",
#: "már nem") — these license the WIDER supersede similarity band (0.88)
#: because the statement pragmatically presupposes the very trait it
#: replaces.
def is_presuppositional(text: str) -> bool:
    low = (text or "").lower()
    return _has_cue(low, _NEG_PRESUPPOSITION_CUES)


def observation_stance(claim: str, quote: str = "") -> str:
    """Stance of an OBSERVATION = cues from the claim OR the original quote.

    The quote is the user's actual words (ground truth); the claim is the
    distilled label. If the extractor normalised the negation away from the
    label, the quote still carries it — scanning both means a negation is
    never missed because of label rewriting. If EITHER signals a stance,
    that stance wins (never miss a negation).

    When both carry different non-empty stances, the QUOTE's class wins for
    ``uncertain``/``qualified`` (they qualify the whole utterance), and the
    strongest current-state signal (``neg``) otherwise. [v0.10] ``future``
    follows the same either-side rule as ``neg``: the extractor normalises
    "szeretni fogom" to a present-tense claim ("likes"), so the future
    signal usually survives ONLY in the quote — never miss it.
    """
    s_claim = classify_stance(claim)
    s_quote = classify_stance(quote) if quote else ""
    if not s_quote:
        return s_claim
    if not s_claim:
        return s_quote
    if s_claim == s_quote:
        return s_claim
    for cls in ("uncertain", "qualified"):
        if cls in (s_claim, s_quote):
            return cls
    if "neg" in (s_claim, s_quote):
        return "neg"
    if "future" in (s_claim, s_quote):
        return "future"
    if "past" in (s_claim, s_quote):
        return "past"
    return s_claim


# ── topic-overlap guard ─────────────────────────────────────────────────────

def _stem(tok: str) -> str:
    """Best-effort topic stem: strip Hungarian inflection suffixes."""
    best = tok
    for suf in _HU_STRIP_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            cand = tok[: len(tok) - len(suf)]
            if len(cand) < len(best):
                best = cand
    return best


def topic_tokens(text: str) -> set[str]:
    """Content-word stems for the topic-overlap guard (affect words,
    negation cues and stopwords excluded — a supersede decision must agree
    on the TOPIC, not on the function words)."""
    out: set[str] = set()
    for tok in re.findall(r"[a-záéíóöőúüű]{4,}", (text or "").lower()):
        if tok in _NON_TOPIC_TOKENS:
            continue
        out.add(_stem(tok))
    return out


def topic_overlaps(a: str, b: str) -> bool:
    """True when two texts share at least one topic content stem.

    Stems match on equality OR one being a ≥4-char prefix of the other
    (motor/motorcycles, kávé/coffee-style Latin roots share prefixes;
    different topics like motorcycles/bicycles share nothing ≥4).
    This guard keeps the wide supersede band from crossing topics: measured
    noise pairs ("likes motorcycles" ↔ "no longer likes bicycles" 0.8876)
    must not supersede each other.
    """
    ta, tb = topic_tokens(a), topic_tokens(b)
    if not ta or not tb:
        return False
    for x in ta:
        for y in tb:
            if x == y:
                return True
            if len(x) >= 4 and len(y) >= 4 and (x.startswith(y) or y.startswith(x)):
                return True
    return False
