"""M2 emotion intelligence: prosody analysis + semantic fusion.

Pipeline position (milestones chapter 8): the prosody analyzer runs
IN PARALLEL with the VoiceMem retrieval, both started from the
utterance transcript+audio; the fused result is injected into the LLM
system prompt and (when the user sounds frustrated) slows the TTS down.

Three layers, cleanly separated for testability:

1. PURE mapping tables + math (no heavy imports, sandbox-testable):
   - :data:`EMOTION2VEC_LABELS` - the 9 raw emotion2vec+ categories as
     they appear in the model ``tokens.txt`` (bilingual "zh/en" forms).
   - :func:`map_raw_label` - raw category -> teacher-facing category.
   - :func:`prosody_valence_arousal` - softmax distribution -> weighted
     (valence, arousal) point.
   - :func:`fuse_emotion` - the spec 8.3.3 linear blend
     ``final = prosody * 0.6 + semantic * 0.4`` where the semantic side
     is a deterministic transcript heuristic (frustration / positive
     markers in HU and EN).
   - :func:`tail_window` - last N seconds of the utterance audio.
   - :class:`EmotionResult` - the result record.
   - :class:`EmotionMemory` - JSONL per-turn emotion log + threshold.

2. :class:`EmotionAnalyzer` - the real prosody analyzer (emotion2vec+
   base via the FunASR ``AutoModel``, CPU, FP32 torch). funasr/torch are
   imported LAZILY inside :meth:`EmotionAnalyzer._load` so this module
   stays importable without them (CONTRACT hard rule 2). Every failure
   path degrades to ``None`` - the M1 pipeline must survive a dead
   emotion analyzer (modularity contract 8.6 item 5).

3. Swappable by design (modularity contract 8.6 item 3): the pipeline
   only needs ``analyze(audio) -> Optional[EmotionResult]``; replacing
   emotion2vec+ with a future MIT SER model keeps the interface.

License note (LICENSES.md #17): emotion2vec_plus_base weights are under
the FunASR Model Open Source License Agreement v1.1 (attribution + keep
the model name). The funasr pip package code is MIT.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Raw emotion2vec+ categories (tokens.txt order; index == class index).
#: The tokens.txt entries are bilingual "zh/en" strings - both accepted.
EMOTION2VEC_LABELS: Tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

#: Bilingual forms as they literally appear in tokens.txt (index-aligned
#: with EMOTION2VEC_LABELS).
EMOTION2VEC_LABELS_ZH_EN: Tuple[str, ...] = (
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
)

#: Teacher-facing emotion categories injected into the LLM prompt.
TEACHER_EMOTIONS: Tuple[str, ...] = ("frustrated", "sad", "neutral", "happy")

#: Raw category -> teacher category (M2 mapping decision: for a language
#: learner, negative high-arousal audio states are handled as frustration -
#: the pedagogically relevant adaptation - while sad stays sad).
_RAW_TO_TEACHER: Dict[str, str] = {
    "angry": "frustrated",
    "disgusted": "frustrated",
    "fearful": "frustrated",
    "happy": "happy",
    "neutral": "neutral",
    "other": "neutral",
    "sad": "sad",
    "surprised": "happy",
    "unknown": "neutral",
    "<unk>": "neutral",
}

#: (valence, arousal) anchor per raw category (spec 8.3.1: label plus
#: valence -1..1 and arousal -1..1). Used both for the argmax label and
#: as per-class anchors for the distribution-weighted estimate.
_VALENCE_AROUSAL: Dict[str, Tuple[float, float]] = {
    "angry": (-0.8, 0.8),
    "disgusted": (-0.7, 0.4),
    "fearful": (-0.7, 0.6),
    "happy": (0.8, 0.5),
    "neutral": (0.0, 0.0),
    "other": (0.0, 0.0),
    "sad": (-0.7, -0.4),
    "surprised": (0.4, 0.7),
    "unknown": (0.0, 0.0),
    "<unk>": (0.0, 0.0),
}

#: Default prosody weight of the 0.6/0.4 blend (spec 8.3.3).
DEFAULT_PROSODY_WEIGHT = 0.6

#: Frustration / struggle markers in the transcript (semantic heuristic).
#: Matched case-insensitively as substrings; Hungarian first (teacher
#: persona: the student is a native Hungarian speaker).
_SEMANTIC_NEGATIVE_MARKERS: Tuple[str, ...] = (
    "nem ertem",        # "nem értem" without accents
    "nem értem",
    "nem értem meg",
    "nem megy",
    "nem tudom",
    "rosszul mondtam",
    "nehez",            # "nehéz"
    "nehéz",
    "bonyolult",
    "mindig hibazom",   # "mindig hibázom"
    "hibázom",
    "hibazom",
    "elrontottam",
    "why is",
    "i do not understand",
    "i don't understand",
    "i dont understand",
    "don't get it",
    "dont get it",
    "so difficult",
    "too difficult",
    "confus",
    "wrong again",
    "mistake again",
)

#: Positive / confident markers in the transcript.
_SEMANTIC_POSITIVE_MARKERS: Tuple[str, ...] = (
    "koszonom",         # "köszönöm"
    "köszönöm",
    "köszi",
    "koszi",
    "ertem",            # "értem"
    "értem",
    "erem mar",         # rare without accents - skip, covered above
    "igen",
    "oke",
    "oké",
    "super",
    "szuper",
    "jo lett",
    "jó lett",
    "nagyon jo",
    "nagyon jó",
    "thanks",
    "thank you",
    "great",
    "nice",
    "perfect",
    "got it",
    "i understand",
    "makes sense",
)

#: Semantic anchor when negative markers dominate the transcript.
_SEMANTIC_NEGATIVE_VA = (-0.5, 0.4)
#: Semantic anchor when positive markers dominate the transcript.
_SEMANTIC_POSITIVE_VA = (0.6, 0.3)
#: Semantic anchor when the transcript carries no signal.
_SEMANTIC_NEUTRAL_VA = (0.0, 0.0)

#: Default threshold for persisting a turn emotion (|valence| or arousal).
DEFAULT_STORE_THRESHOLD = 0.5


# --------------------------------------------------------------------- pure #


@dataclass
class EmotionResult:
    """One prosody analysis outcome (after optional fusion).

    ``label`` is the teacher-facing category (``TEACHER_EMOTIONS``);
    ``raw_label`` the emotion2vec+ category. ``valence``/``arousal`` are
    the FUSED values when the result was produced by :func:`fuse_emotion`
    (``fused=True``), otherwise the pure prosody estimates.
    ``scores`` holds the softmax distribution aligned with
    ``EMOTION2VEC_LABELS`` (when provided by the analyzer).
    """

    raw_label: str
    label: str
    valence: float
    arousal: float
    confidence: float = 0.0
    fused: bool = False
    scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict (all fields, rounded for readability)."""
        return {
            "raw_label": self.raw_label,
            "label": self.label,
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "confidence": round(self.confidence, 3),
            "fused": self.fused,
            "latency_ms": round(self.latency_ms, 1),
        }


def map_raw_label(raw_label: str) -> str:
    """Map an emotion2vec+ raw category to a teacher-facing category.

    Accepts both the plain English form (``"angry"``) and the bilingual
    tokens.txt form (``"生气/angry"``); unknown values map to
    ``"neutral"`` (never raises - the pipeline must survive surprises).
    """
    label = (raw_label or "").strip().lower()
    if label in _RAW_TO_TEACHER:
        return _RAW_TO_TEACHER[label]
    # bilingual "zh/en" form: take the part after the slash
    if "/" in label:
        tail = label.split("/")[-1].strip()
        if tail in _RAW_TO_TEACHER:
            return _RAW_TO_TEACHER[tail]
    return "neutral"


def raw_label_valence_arousal(raw_label: str) -> Tuple[float, float]:
    """The (valence, arousal) anchor of one raw category (neutral on miss)."""
    label = (raw_label or "").strip().lower()
    if "/" in label:
        label = label.split("/")[-1].strip()
    return _VALENCE_AROUSAL.get(label, (0.0, 0.0))


def prosody_valence_arousal(
    labels: Sequence[str], scores: Sequence[float]
) -> Tuple[float, float]:
    """Distribution-weighted (valence, arousal) from a softmax output.

    Anchors every raw category at its :data:`_VALENCE_AROUSAL` point and
    averages them with the softmax probabilities - smoother than the
    argmax-only estimate and still cheap. ``labels`` may contain the
    bilingual tokens.txt forms. An empty/shape-mismatched input returns
    (0.0, 0.0).
    """
    if not labels or not scores or len(labels) != len(scores):
        return (0.0, 0.0)
    total = 0.0
    valence = 0.0
    arousal = 0.0
    for raw_label, score in zip(labels, scores):
        try:
            weight = float(score)
        except (TypeError, ValueError):
            continue
        if weight <= 0.0 or weight != weight:  # NaN guard
            continue
        v, a = raw_label_valence_arousal(raw_label)
        valence += v * weight
        arousal += a * weight
        total += weight
    if total <= 0.0:
        return (0.0, 0.0)
    return (valence / total, arousal / total)


def semantic_valence_arousal(transcript: str) -> Tuple[float, float]:
    """Deterministic transcript heuristic -> (valence, arousal).

    Counts frustration/struggle vs positive/confident markers
    (case-insensitive substring match, both HU and EN). Negative
    markers are matched FIRST and blanked out of the text before the
    positive pass, so overlaps like "nem értem" (negative) vs "értem"
    (positive) cannot double-count. The winner sets the semantic
    anchor; ties or no signal -> neutral anchor. This is the cheap
    stand-in for the semantic side of the 8.3.3 blend - the LLM itself
    performs the full content verification inside the reply prompt
    (teacher_persona emotion block).
    """
    text = (transcript or "").lower()
    if not text:
        return _SEMANTIC_NEUTRAL_VA
    neg_hits = 0
    for marker in _SEMANTIC_NEGATIVE_MARKERS:
        if marker in text:
            neg_hits += 1
            text = text.replace(marker, " ")
    pos_hits = sum(1 for marker in _SEMANTIC_POSITIVE_MARKERS if marker in text)
    if neg_hits > pos_hits:
        return _SEMANTIC_NEGATIVE_VA
    if pos_hits > neg_hits:
        return _SEMANTIC_POSITIVE_VA
    return _SEMANTIC_NEUTRAL_VA


def teacher_label_from_va(valence: float, arousal: float) -> str:
    """Decision thresholds: (valence, arousal) -> teacher category.

    - very negative + aroused (>= 0.2)   -> frustrated
    - very negative + calm               -> sad
    - positive (>= 0.25)                 -> happy
    - otherwise                          -> neutral
    """
    if valence <= -0.25:
        if arousal >= 0.2:
            return "frustrated"
        return "sad"
    if valence >= 0.25:
        return "happy"
    return "neutral"


def fuse_emotion(
    prosody: Optional[EmotionResult],
    transcript: str,
    prosody_weight: float = DEFAULT_PROSODY_WEIGHT,
) -> Optional[EmotionResult]:
    """Spec 8.3.3 blend: final = prosody * w + semantic * (1 - w).

    Returns a NEW :class:`EmotionResult` with ``fused=True`` carrying the
    blended valence/arousal and the teacher label derived from them.
    ``prosody=None`` (analyzer dead / disabled) short-circuits to ``None``
    - a missing prosody side must NEVER fabricate an emotion.
    """
    if prosody is None:
        return None
    weight = min(max(float(prosody_weight), 0.0), 1.0)
    sem_v, sem_a = semantic_valence_arousal(transcript)
    final_v = weight * prosody.valence + (1.0 - weight) * sem_v
    final_a = weight * prosody.arousal + (1.0 - weight) * sem_a
    return EmotionResult(
        raw_label=prosody.raw_label,
        label=teacher_label_from_va(final_v, final_a),
        valence=final_v,
        arousal=final_a,
        confidence=prosody.confidence,
        fused=True,
        scores=list(prosody.scores),
        latency_ms=prosody.latency_ms,
    )


def tail_window(
    audio: Any, sample_rate: int, seconds: float
) -> Any:
    """The last ``seconds`` of a 1-D float audio array (spec 8.3.1: the
    analyzer sees the last ~5 s of the utterance).

    Returns the input unchanged when it is shorter than the window or
    when it has no ``size`` attribute (duck-typed mocks).
    """
    size = int(getattr(audio, "size", 0) or 0)
    limit = int(sample_rate * seconds)
    if size <= 0 or limit <= 0 or size <= limit:
        return audio
    return audio[size - limit :]


# ------------------------------------------------------------- emotion log #


class EmotionMemory:
    """Per-turn emotion persistence (spec 8.3.4 / exit criterion 3).

    Two layers:
    - ALWAYS: one JSON line per analyzed turn in ``data/emotion_log.jsonl``
      (machine-readable, offline, greppable - the "per-turn
      valence-arousal" record).
    - STRONG emotions only (``should_store``): the caller (pipeline) also
      pushes a short human-readable fact into VoiceMem via the bridge -
      that is where long-term aggregation (e.g. repeated frustration about
      one grammar mistake) happens.

    The writer is thread-safe (a lock guards the append) and never raises:
    a failing emotion log must not kill the turn (modularity 8.6).
    """

    def __init__(self, log_path: Path) -> None:
        self._path = Path(log_path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def should_store(result: EmotionResult, threshold: float = DEFAULT_STORE_THRESHOLD) -> bool:
        """True when the emotion is strong enough to persist as a fact.

        Spec 8.3.4: frustrated with arousal >= 0.5, or happy/sad with
        |valence| >= 0.5.
        """
        if result.label == "frustrated":
            return abs(result.arousal) >= threshold or abs(result.valence) >= threshold
        return abs(result.valence) >= threshold

    def record_turn(
        self, result: EmotionResult, transcript: str, stored_as_fact: bool = False
    ) -> bool:
        """Append one turn record to the JSONL log (best effort, never raises).

        Returns True when a line was written.
        """
        entry: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "transcript": (transcript or "")[:400],
            "stored_as_fact": bool(stored_as_fact),
        }
        entry.update(result.to_dict())
        line = json.dumps(entry, ensure_ascii=True)
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            return True
        except OSError as exc:
            logger.warning("Emotion log write failed (non-fatal): %s", exc)
            return False

    @staticmethod
    def fact_text(result: EmotionResult, transcript: str) -> str:
        """The short human-readable fact text pushed into VoiceMem."""
        quote = (transcript or "").strip().replace("\n", " ")[:120]
        text = (
            "Emotional state: the user sounded %s (valence=%.2f, arousal=%.2f)"
            % (result.label, result.valence, result.arousal)
        )
        if quote:
            text += " while saying: %s" % quote
        return text


# ------------------------------------------------------------------ analyzer #


class EmotionAnalyzer:
    """Prosody analyzer backed by emotion2vec+ base (FunASR, CPU).

    Contract (pipeline duck-typing):
    - ``is_available() -> bool``  - funasr importable AND model files present.
    - ``analyze(audio) -> Optional[EmotionResult]`` - sync, CPU, ~tens of
      ms per 5 s window; ``None`` on ANY failure (graceful degradation).

    funasr/torch are imported lazily inside :meth:`_load` (the module
    itself must stay importable in the dependency-free sandbox). The
    model is loaded ONCE (guarded by ``self._loaded``); a failed load is
    remembered so a broken install does not retry every turn.
    """

    #: Files the local model directory must contain (lock-downloaded).
    REQUIRED_FILES: Tuple[str, ...] = ("model.pt", "config.yaml", "tokens.txt")

    def __init__(
        self,
        model_dir: Path,
        window_s: float = 5.0,
        sample_rate: int = 16000,
        model_factory: Optional[Callable[[Path], Any]] = None,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._window_s = float(window_s)
        self._sample_rate = int(sample_rate)
        self._model: Optional[Any] = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._model_factory = model_factory  # test seam: fake AutoModel
        self._torch: Optional[Any] = None
        # v0.4.13: the web warm-up thread and a turn's analyze() can race
        # into _load() concurrently (turn arrives while the startup warm-up
        # is still constructing the AutoModel). Without the lock BOTH threads
        # constructed the model — double the CPU/RAM for minutes. With it,
        # the second caller waits for the first load and then reuses the
        # model; the turn-side caller is still bounded externally by the
        # turn's asyncio.wait timeout (emotion degrades, never blocks).
        self._load_lock = threading.Lock()

    # -- availability -------------------------------------------------------- #

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def files_present(self) -> bool:
        """True when every REQUIRED_FILES entry exists in the model dir."""
        try:
            return all(
                (self._model_dir / name).is_file() for name in self.REQUIRED_FILES
            )
        except OSError:
            return False

    def is_available(self) -> bool:
        """funasr importable AND model files on disk (never raises)."""
        if not self.files_present():
            return False
        try:
            import funasr  # noqa: F401 - pure presence check
        except Exception:  # noqa: BLE001 - ImportError and friends
            return False
        return True

    def warm_up(self) -> bool:
        """Preload the model NOW (at startup) - the first turn must not pay
        the multi-second lazy load (live-measured ~2 s on the target class
        hardware). Returns True when the model is ready; a failed warm-up
        leaves the analyzer in the degraded (None) state.
        """
        return self._load() is not None

    # -- inference ----------------------------------------------------------- #

    def _load(self) -> Optional[Any]:
        """Load the FunASR AutoModel once (lazily, from the local dir).

        Returns the model or None (the failure is logged ONCE and
        remembered; subsequent calls return None immediately). v0.4.13:
        serialised by ``_load_lock`` — a concurrent caller (web turn during
        the startup warm-up) waits for the in-flight load instead of
        constructing a SECOND AutoModel; callers on the turn path are
        bounded by an external asyncio timeout, so this wait can never
        block a turn indefinitely.
        """
        with self._load_lock:
            if self._loaded:
                return self._model
            if self._load_error is not None:
                return None
            self._loaded = True  # one attempt only - no per-turn retry storm
            try:
                if self._model_factory is not None:
                    self._model = self._model_factory(self._model_dir)
                    return self._model
                from funasr import AutoModel

                self._model = AutoModel(
                    model=str(self._model_dir),
                    device="cpu",
                    disable_update=True,
                    log_level="ERROR",
                )
                return self._model
            except Exception as exc:  # noqa: BLE001 - degradation, never raise
                self._load_error = str(exc)
                logger.warning(
                    "Emotion analyzer unavailable (M1 pipeline continues without "
                    "emotion): %s",
                    exc,
                )
                return None

    def analyze(self, audio: Any) -> Optional[EmotionResult]:
        """Analyze the tail window of one utterance; None on any failure.

        Runs the model with ``granularity="utterance"``,
        ``extract_embedding=False`` (classification only), under
        ``torch.no_grad`` when torch is importable. The softmax output is
        mapped to the teacher label + (valence, arousal).
        """
        window = tail_window(audio, self._sample_rate, self._window_s)
        size = int(getattr(window, "size", 0) or 0)
        if size <= 0:
            return None
        start = time.perf_counter()
        try:
            model = self._load()
            if model is None:
                return None
            tensor = self._to_tensor(window)
            raw = model.generate(
                tensor,
                fs=self._sample_rate,
                granularity="utterance",
                extract_embedding=False,
            )
        except Exception as exc:  # noqa: BLE001 - never kill a turn
            logger.warning("Emotion analysis failed (non-fatal): %s", exc)
            return None
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._parse_result(raw, latency_ms)

    # -- internals ------------------------------------------------------------ #

    def _to_tensor(self, window: Any) -> Any:
        """float32 torch tensor on CPU from a numpy array / sequence."""
        import torch

        self._torch = torch
        if hasattr(window, "dtype"):
            array = window
        else:
            array = self._numpy(window)
        if str(getattr(array, "dtype", "")).endswith("float32"):
            tensor = torch.from_numpy(array)
        else:
            tensor = torch.from_numpy(self._numpy(array, dtype="float32"))
        return tensor.to(torch.float32)

    @staticmethod
    def _numpy(array: Any, dtype: Optional[str] = None) -> Any:
        """numpy array from a sequence (dtype='float32' optional)."""
        import numpy as np

        result = np.asarray(array)
        if dtype:
            result = result.astype(np.dtype(dtype))
        return result

    def _parse_result(
        self, raw: Any, latency_ms: float
    ) -> Optional[EmotionResult]:
        """funasr generate() output -> EmotionResult.

        Expected shape: ``[{"key": ..., "labels": [...], "scores": [...]}]``
        (funasr models/emotion2vec/model.py). Missing/odd shapes degrade
        to None.
        """
        try:
            first = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
            if not isinstance(first, dict):
                return None
            labels = list(first.get("labels") or [])
            scores = list(first.get("scores") or [])
            if not labels or not scores:
                return None
            pairs = [
                (str(label), float(score))
                for label, score in zip(labels, scores)
            ]
            best_label, best_score = max(pairs, key=lambda pair: pair[1])
            valence, arousal = prosody_valence_arousal(labels, scores)
            total = sum(score for _, score in pairs)
            confidence = best_score / total if total > 0.0 else 0.0
            return EmotionResult(
                raw_label=best_label,
                label=map_raw_label(best_label),
                valence=valence,
                arousal=arousal,
                confidence=confidence,
                fused=False,
                scores=[score for _, score in pairs],
                latency_ms=latency_ms,
            )
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            logger.warning("Emotion result parsing failed (non-fatal): %s", exc)
            return None
