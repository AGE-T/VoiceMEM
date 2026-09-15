"""app/retrieval_contract.py — the v0.8.1 minimal retrieval result contract.

WHY (post-implementation-audit P1-2): the trait bookkeeping computed by the
v0.8.0 memory layer (VM-LOCAL-013: confidence, effective confidence,
occurrence_count, first_seen, last_seen — produced in
vendor/voicemem/voicemem/rightbrain/brain.py ``_rb_trait_hits`` and attached
to every trait ``RightBrainHit.metadata``) was DISCARDED at every app
consumer: the web payload read only content/slot/source/priority, the LLM
context rendered a bare ``- text`` line, the CLI bridge did the same. The
information was computed, then dropped on the floor.

This module is the smallest useful canonical contract for information that
is ALREADY CALCULATED — it adds NO retrieval logic, NO scoring, NO new
semantics, NO second search API (per the v0.8.0 audit: one canonical
pipeline, improve the result contract and consumers):

* :class:`TraitObservationInfo` — the typed view of the trait bookkeeping
  (identity + confirmation state + observation dates);
* :func:`extract_trait_info` — pull it off any ``RightBrainHit``-like object
  (returns None for non-trait hits — the discriminator is the ``trait_id``
  metadata key, which only ``_rb_trait_hits`` sets);
* :func:`trait_fields_payload` — the canonical transport dict for the web
  payload / machine consumers (uniform keys, neutral defaults for non-trait
  hits so the payload schema stays stable);
* :func:`trait_prompt_suffix` — the LLM-context provenance suffix for trait
  hits (``[last heard <date> | <N>x heard]``), mirroring the v0.6.3 F-B
  fact-side ``hit_provenance_suffix`` semantics.

CONSUMPTION BOUNDARIES (documented in CONTRACT.md):

* Web payload: the full typed field set (confidence values included — the
  UI/clients get the numbers the memory layer computed);
* LLM prompt: observation date + confirmation count ONLY. The raw
  confidence/eff_confidence floats are deliberately NOT rendered into the
  prompt: the v0.8.0 audit established that trait confidence is a
  reinforcement counter's asymptotic residue (c' = c + (1-c)*0.30 per merge,
  starting at 0.9), NOT a calibrated probability — printing
  ``confidence = 0.93`` would inject misleading pseudo-precision ("93%
  certain") into the persona's context. The values stay available to the
  application/UI layer, where they can be presented with their true meaning
  (confirmation strength).
* rb_directive (SearchResult.rb_directive — the vendor's right-brain
  situational guidance + the orchestrator's low-evidence abstention note):
  PRESERVED in the web payload (transport-only) and documented as currently
  UNCONSUMED by design — it is not injected into the LLM prompt (the vendor
  text is Chinese-centric guidance that duplicates the app's own cleaned
  rb render; the abstention hint would change reply behaviour, which is a
  semantics change, out of scope for a stabilization release).

Canonical terminology for the fields touched by this contract (Phase 9,
no broad rename): ``occurrence_count`` (traits: 1-based number of stored
observations — 1 add = 1 evidence row + 1 increment; facts: 0-based
re-confirmations of the same value, the v0.6.3 OCC-1 convention — the units
differ and the renders say so: traits render "``Nx heard``", facts render
"``Nx confirmed``"); ``first_seen``/``last_seen`` (trait bookkeeping dates,
distinct from the fact-side event-time ``observed_at``/
``last_observed_at``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TraitObservationInfo:
    """The typed trait bookkeeping attached to a trait ``RightBrainHit``.

    Every field is produced by the vendor brain layer (VM-LOCAL-013) and
    attached to ``RightBrainHit.metadata``; this dataclass is the app-side
    typed view of that metadata — the canonical retrieval result contract
    for trait observations. Semantics (per the v0.8.0 audit):

    * ``confidence`` — raw confirmation strength (asymptotic reinforcement
      c' = c + (1-c)*0.30 per merge, starting 0.9; NOT a probability);
    * ``eff_confidence`` — the read-side decayed variant (90-day grace,
      180-day halflife, computed at retrieval);
    * ``occurrence_count`` — 1-based count of stored observations;
    * ``first_seen``/``last_seen`` — bookkeeping dates (YYYY-MM-DD).
    """

    trait_id: str
    slot_name: str
    claim: str
    confidence: float
    eff_confidence: float
    occurrence_count: int
    first_seen: str
    last_seen: str
    # [v0.9.0 — VM-LOCAL-015] trait semantic state: the stance recorded at
    # write time (pos/neg/past/qualified/uncertain/future, "" = neutral/
    # legacy) and the supersession chain (this row superseded <id> / was
    # superseded by <id> at <ts>). superseded_by non-empty => the trait is
    # HISTORICAL, not current: consumers must not present it as current
    # truth. [v0.10] ``future`` = not-yet-valid forward-looking observation
    # (never merges into / flips a current row — same class as the fact side,
    # leftbrain/temporal.py).
    stance: str = ""
    supersedes: str = ""
    superseded_by: str = ""
    superseded_at: str = ""


def _meta(hit: Any) -> dict:
    """The hit's metadata dict ({} when absent — robust to any hit shape)."""
    meta = getattr(hit, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def extract_trait_info(hit: Any) -> Optional[TraitObservationInfo]:
    """Typed trait bookkeeping from a ``RightBrainHit``-like object.

    Returns None for non-trait hits (response_experience / situation_pattern
    / relation carry their own metadata WITHOUT ``trait_id`` — that key is
    set only by the vendor's ``_rb_trait_hits``, which is exactly where the
    VM-LOCAL-013 bookkeeping lives).
    """
    meta = _meta(hit)
    trait_id = str(meta.get("trait_id") or "").strip()
    if not trait_id:
        return None
    try:
        confidence = float(meta.get("confidence", 0.9))
    except (TypeError, ValueError):
        confidence = 0.9
    try:
        eff = float(meta.get("eff_confidence", 0.9))
    except (TypeError, ValueError):
        eff = 0.9
    try:
        occ = int(meta.get("occurrence_count", 1))
    except (TypeError, ValueError):
        occ = 1
    return TraitObservationInfo(
        trait_id=trait_id,
        slot_name=str(meta.get("slot_name") or ""),
        claim=str(meta.get("claim") or ""),
        confidence=confidence,
        eff_confidence=eff,
        occurrence_count=occ,
        first_seen=str(meta.get("first_seen") or ""),
        last_seen=str(meta.get("last_seen") or ""),
        stance=str(meta.get("stance") or ""),
        supersedes=str(meta.get("supersedes") or ""),
        superseded_by=str(meta.get("superseded_by") or ""),
        superseded_at=str(meta.get("superseded_at") or ""),
    )


#: Uniform payload keys (stable schema — non-trait hits get neutral values).
#: [v0.9.0 — VM-LOCAL-015] the trait semantic state keys ride the same
#: uniform schema: ``stance`` ("" = neutral/legacy), ``supersedes`` /
#: ``superseded_by`` / ``superseded_at`` ("" = no chain membership).
TRAIT_PAYLOAD_KEYS = (
    "is_trait", "trait_id", "confidence", "eff_confidence",
    "occurrence_count", "first_seen", "last_seen",
    "stance", "supersedes", "superseded_by", "superseded_at",
)


def trait_fields_payload(hit: Any) -> dict:
    """Canonical transport dict for the web payload / machine consumers.

    Uniform keys for every right-brain hit (``is_trait`` discriminates);
    non-trait hits carry neutral defaults so consumers never branch on key
    presence. ``slot_name``/``claim`` stay inside the existing payload's
    ``slot``/content fields — only the bookkeeping fields are new.
    """
    info = extract_trait_info(hit)
    if info is None:
        return {
            "is_trait": False,
            "trait_id": "",
            "confidence": 0.0,
            "eff_confidence": 0.0,
            "occurrence_count": 0,
            "first_seen": "",
            "last_seen": "",
            "stance": "",
            "supersedes": "",
            "superseded_by": "",
            "superseded_at": "",
        }
    return {
        "is_trait": True,
        "trait_id": info.trait_id,
        "confidence": round(info.confidence, 4),
        "eff_confidence": round(info.eff_confidence, 4),
        "occurrence_count": info.occurrence_count,
        "first_seen": info.first_seen,
        "last_seen": info.last_seen,
        "stance": info.stance,
        "supersedes": info.supersedes,
        "superseded_by": info.superseded_by,
        "superseded_at": info.superseded_at,
    }


def trait_prompt_suffix(hit: Any) -> str:
    """LLM-context provenance suffix for a trait hit (mirrors the fact-side
    v0.6.3 ``hit_provenance_suffix``).

    ``[last heard 2026-06-01 | 3x heard]`` — the observation date and the
    confirmation count are the two things that materially change how the
    persona should interpret a remembered trait. The confidence floats are
    deliberately NOT rendered: trait confidence is a confirmation-strength
    measure (asymptotic reinforcement), not a calibrated probability —
    ``confidence = 0.93`` in the prompt would read as "93% certain"
    (misleading pseudo-precision). The values remain available to the
    application/UI via :func:`trait_fields_payload`.

    Units: traits count STORED OBSERVATIONS (1-based; 1 add = 1 evidence
    row + 1 increment) → the render says ``Nx heard``; facts count
    re-confirmations (0-based, OCC-1) → the fact-side render says
    ``Nx confirmed``. Distinct wording for distinct semantics.
    """
    info = extract_trait_info(hit)
    if info is None:
        return ""
    bits: list[str] = []
    last = info.last_seen[:10]
    if len(last) == 10 and last[4] == "-":
        bits.append(f"last heard {last}")
    if info.occurrence_count > 1:
        bits.append(f"{info.occurrence_count}x heard")
    # [v0.9.0 — VM-LOCAL-015] currency status: a superseded trait is
    # HISTORICAL knowledge. The marker uses the fact-side wording
    # (``superseded``, hit_provenance_suffix) — one term, one meaning,
    # across both memory layers. Absence of the marker = current (the
    # existing convention; the LLM is never told an unmarked trait is
    # current — it is simply not told otherwise). The superseded row's
    # occurrence/confidence are frozen at flip time, so "Nx heard" on a
    # superseded line counts only PRE-flip observations.
    if info.superseded_by:
        bits.append("superseded")
    return f" [{' | '.join(bits)}]" if bits else ""
