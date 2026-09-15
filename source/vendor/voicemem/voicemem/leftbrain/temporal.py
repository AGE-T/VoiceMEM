"""leftbrain/temporal.py — v0.10 temporal semantics for the FACT path.

WHY: the v0.9.x system represents currency structurally (supersession
chain, current-first ranking) but the FACT path has no notion of WHEN a
statement applies. A future statement ("Thomas will like X") is stored as
an ordinary current fact; a hedged one ("may not like X") can overwrite a
confirmed one; a past one ("used to like X") is only text. PHASE 0 of the
v0.10 audit (audit/VoiceMEM_v010/PHASE0_BASELINE_AND_SEMANTIC_MAP.md)
listed these as impossible-with-current-model; this module closes that gap
DETERMINISTICALLY — zero LLM calls, zero prompt changes (the v0.9.2 prompt
reduction and latency budget stay untouched).

What lives here (all pure functions, EN+HU cues, no I/O):

* :func:`fact_stance` — stance of a fact observation, reusing
  ``rightbrain.stance.observation_stance`` (claim OR quote; the quote is
  ground truth) EXTENDED with a ``future`` class (see :data:`FUTURE_CUES`).
* :func:`fact_validity` — ``valid_from`` / ``valid_until`` detected from
  the fact text when the statement itself bounds it (a future-affect
  statement with a date → valid_from; an explicit "until Y" → valid_until).
* :func:`query_temporal_intent` — is the QUERY about now / the past / the
  future (default: now). Used once per search to pick the ranking tiers.
* :func:`fact_status` — DERIVED row status from the row's metadata:
  ``current`` | ``historical`` | ``superseded`` | ``future``. NOT stored —
  computed from the structural fields (supersession chain + validity
  interval + stance), per the v0.10 design (no redundant state enum).
* :func:`temporal_rank_tier` — the ranking tier of a hit under a query
  intent. Legacy rows (no temporal keys) get EXACTLY the v0.9.2 ordering
  (see the tier table and the equivalence proof in
  :func:`temporal_rank_tier`).

DESIGN BOUNDARIES (documented, deliberate):

* A future DATED EVENT ("flight to Paris on 2026-12-01") is NOT future
  status — it is CURRENT knowledge about a future event. Only future-AFFECT
  cues (will like / szeretni fogom …) mark a fact future. Bare dates never
  imply future validity (conservative: no invented semantics).
* ``stance == "past"`` rows are historical BY SELF-DECLARATION even when
  not superseded ("used to like X" stands alone — it never claimed to be
  current). ``stance == "uncertain"`` is a FACET, not a status: an
  uncertain observation is current knowledge, rendered as uncertain —
  never silently equal to a confirmed one.
* ``valid_from`` in the future with no end date is "indefinite future":
  ranked below current, rendered as future, NEVER auto-promoted (no date
  to promote on — promotion happens only by a later explicit observation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date_cls

from voicemem.rightbrain.stance import (
    _has_cue,
    classify_stance,
    observation_stance,
)

__all__ = [
    "FACT_STANCE_VALUES",
    "FactTemporal",
    "fact_stance",
    "fact_validity",
    "fact_status",
    "row_status",
    "query_temporal_intent",
    "temporal_rank_tier",
    "parse_iso_date",
]


#: Stance vocabulary for facts = the trait vocabulary + ``future``
#: (stance.py STANCE_VALUES — one vocabulary across both memory layers).
FACT_STANCE_VALUES = ("pos", "neg", "past", "qualified", "uncertain", "future", "")

#: Explicit boundedness on the END side: "until Y" / "through Y" / "-ig".
#: These license a ``valid_until`` even on present-tense statements
#: ("stays until Friday", "péntekig").
_VALID_UNTIL_CUES = (
    "until ", " through ", "till ", " up to ",
    "-ig", "ig.", " -ig ", "végéig",
)

_VALID_UNTIL_RE = re.compile(
    r"(?:until|through|till|up to)\s+"
    r"((?:19|20)\d{2}[-.]\s?\d{1,2}[-.]\s?\d{1,2}"
    r"|(?:19|20)\d{2}\s?\.\s?(?:január|február|március|április|május|"
    r"június|július|augusztus|szeptember|október|november|december)\s?\d{1,2}"
    r"|(?:január|február|március|április|május|június|július|augusztus|"
    r"szeptember|október|november|december)\s?\d{1,2}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2})",
    re.I,
)
_HU_UNTIL_RE = re.compile(
    r"((?:19|20)\d{2}[-.]\s?\d{1,2}[-.]\s?\d{1,2}"
    r"|(?:19|20)\d{2}\s?\.\s?(?:január|február|március|április|május|"
    r"június|július|augusztus|szeptember|október|november|december)\s?\d{1,2}"
    r"|(?:január|február|március|április|május|június|július|augusztus|"
    r"szeptember|október|november|december)\s?\d{1,2})"
    r"\s?-?(?:ig|végéig|áig)",
    re.I,
)

#: Past-INTENT query cues (asking ABOUT the past). Distinct from stance
#: past cues: "mit szeretett" asks about history; the answer rows are the
#: historical ones. Keep narrow — a false "past" intent only reorders
#: (never drops) hits, but a missed one loses the temporal preference.
_PAST_INTENT_CUES = (
    # English
    "used to", "before,", "previously", "back then", "in the past",
    "what did", "did he", "did she", "did they", "used to like",
    "no longer", "any more", "anymore", "what was",
    # Hungarian
    "régen", "regen", "korábban", "korabban", "azelőtt", "azelott",
    "valaha", "mit szeretett", "mit kedvelt", "szeretett valamit",
    "már nem", "mar nem", "nem többé", "nem tobbe", "akkor",
)

#: Future-INTENT query cues (asking ABOUT the future / plans).
_FUTURE_INTENT_CUES = (
    # English
    "will he", "will she", "will they", "going to", "what will",
    "in the future", "planned", "plans to",
    # Hungarian
    "fog ", "fogom", "fogja", "fogjuk", "jövő", "jovo", "lesz",
    "tervezi", "tervez ", "mit fog", "mi lesz", "majd",
)


def parse_iso_date(s: str) -> _date_cls | None:
    """Parse a strict ``YYYY-MM-DD`` prefix of *s* (None when not date-shaped).

    Reused everywhere a stored temporal key must be compared to today —
    one parser, one behaviour (no per-callsite date guessing).
    """
    t = str(s or "").strip()[:10]
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", t)
    if not m:
        return None
    try:
        return _date_cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


@dataclass(frozen=True)
class FactTemporal:
    """The temporal classification of ONE fact observation.

    ``stance``      — pos/neg/past/qualified/uncertain/future/"" (the fact
                      side now carries the same stance vocabulary as traits
                      + ``future``);
    ``valid_from``  — YYYY-MM-DD when the statement itself says it starts
                      in the future ("" = open/unknown);
    ``valid_until`` — YYYY-MM-DD when the statement itself bounds its end
                      ("" = open). NOT set by cessation here — a
                      superseding cessation sets it on the OLD row at
                      update time (see voice_input/mem0_backend_store).
    """

    stance: str = ""
    valid_from: str = ""
    valid_until: str = ""


def fact_stance(fact_text: str, utterance_quote: str = "") -> str:
    """Stance of a fact observation (claim OR quote — quote is ground truth).

    Thin wrapper over :func:`rightbrain.stance.observation_stance`: the
    ``future`` class lives in the shared classifier (one vocabulary, both
    memory layers). Priority: uncertain > qualified > past > neg >
    future > pos (see stance.py — a hedged future is uncertain first; a
    current negation beats a future marker; "will like" never reads as
    current pos).
    """
    return observation_stance(fact_text, utterance_quote)


def fact_validity(fact_text: str, stance: str, today: _date_cls | None = None) -> tuple[str, str]:
    """``(valid_from, valid_until)`` detected from the fact's own words.

    * valid_from: ONLY for future-stance statements — the earliest FUTURE
      date in the text ("will like X from October" → 2026-10-01). A future
      statement with no date keeps "" (indefinite future — still ranked
      below current, never auto-promoted).
    * valid_until: an explicit boundedness cue ("until Y", "péntekig") on
      ANY stance — the FIRST dated bound found after the cue.
    """
    today = today or _date_cls.today()
    text = str(fact_text or "")
    if not text:
        return "", ""

    from voicemem.leftbrain.local_memory_store import parse_date_values

    valid_from = ""
    if stance == "future":
        future_dates = [d for d in parse_date_values(text, today) if d > today]
        if future_dates:
            valid_from = min(future_dates).isoformat()

    valid_until = ""
    m = _VALID_UNTIL_RE.search(text) or _HU_UNTIL_RE.search(text)
    if m:
        dates = parse_date_values(m.group(0), today)
        if dates:
            valid_until = sorted(dates)[0].isoformat()
    return valid_from, valid_until


def classify_fact_temporal(fact_text: str, utterance_quote: str = "",
                          today: _date_cls | None = None) -> FactTemporal:
    """Full temporal classification of one extracted fact (stance + bounds)."""
    st = fact_stance(fact_text, utterance_quote)
    vf, vu = fact_validity(fact_text, st, today)
    return FactTemporal(stance=st, valid_from=vf, valid_until=vu)


def fact_status(meta: dict | None, today: _date_cls | None = None) -> str:
    """DERIVED status of a stored fact row from its metadata dict.

    Convenience wrapper over :func:`row_status` (dict → fields). See
    :func:`row_status` for the derivation rules and the legacy-row
    equivalence proof.
    """
    meta = meta or {}
    return row_status(
        superseded_by=str(meta.get("superseded_by") or ""),
        stance=str(meta.get("stance") or ""),
        valid_from=str(meta.get("valid_from") or ""),
        valid_until=str(meta.get("valid_until") or ""),
        today=today,
    )


def row_status(*, superseded_by: str = "", stance: str = "",
               valid_from: str = "", valid_until: str = "",
               today: _date_cls | None = None) -> str:
    """DERIVED status of a fact row from its structural fields.

    Returns ``current`` | ``historical`` | ``superseded`` | ``future``:

    * ``superseded`` — ``superseded_by`` set (the chain closed it);
    * ``future``     — ``valid_from`` parses AND is after today, or the
                       stance is ``future`` (indefinite future);
    * ``historical`` — ``valid_until`` parses AND is before today, or the
                       stance is ``past`` (self-declared past observation);
    * ``current``    — everything else (incl. uncertain — that is a FACET).

    Legacy rows (no temporal keys): ``superseded_by`` decides, else
    ``current`` — exactly the v0.9.2 behaviour. Uncertainty NEVER changes
    status (an uncertain row is current knowledge rendered as uncertain).
    """
    today = today or _date_cls.today()
    if str(superseded_by or "").strip():
        return "superseded"
    vf = parse_iso_date(str(valid_from or ""))
    if vf is not None and vf > today:
        return "future"
    if str(stance or "") == "future":
        return "future"
    vu = parse_iso_date(str(valid_until or ""))
    if vu is not None and vu < today:
        return "historical"
    if str(stance or "") == "past":
        return "historical"
    return "current"


def query_temporal_intent(query: str) -> str:
    """Is the query asking about now / the past / the future?

    Default ``current``. Past beats future when both appear ("what did he
    plan before" → past). A past/future intent NEVER excludes rows — it
    only reorders tiers (temporal preference, not a filter).
    """
    low = (query or "").lower()
    if not low.strip():
        return "current"
    if _has_cue(low, _PAST_INTENT_CUES):
        return "past"
    if _has_cue(low, _FUTURE_INTENT_CUES):
        return "future"
    return "current"


#: Ranking tiers per intent. Lower tier = preferred for that intent.
#:
#: Equivalence with v0.9.2 for legacy rows (proof): a legacy row has no
#: stance/valid keys, so fact_status() returns exactly
#: ``"superseded" if superseded_by else "current"``. Under the default
#: ("current") intent both maps give current→0 and superseded→2, so the
#: composed sort key (tier, not superseded_by, base+recency) is identical
#: in ORDER to the old (not superseded_by, base+recency) — tier and the
#: superseded flag are the SAME partition for legacy rows. Old data, old
#: order, byte-for-byte behaviour.
_INTENT_TIERS = {
    "current": {"current": 0, "historical": 1, "future": 1, "superseded": 2},
    # past-intent: the historical record first — superseded rows ARE that
    # record (tier 0, same as self-declared past); current rows follow as
    # context; future last.
    "past": {"current": 1, "historical": 0, "future": 2, "superseded": 0},
    # future-intent: future statements first, current knowledge of plans
    # next, history last.
    "future": {"current": 1, "historical": 2, "future": 0, "superseded": 2},
}


def temporal_rank_tier(status: str, intent: str) -> int:
    """Ranking tier of a hit with *status* under *intent* (lower = first)."""
    return _INTENT_TIERS.get(intent, _INTENT_TIERS["current"]).get(status, 0)
