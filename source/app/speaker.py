"""M3 speaker recognition: ECAPA embedding + identification + registry.

Pipeline position (milestones chapter 9): the speaker embedder runs on the
COMPLETED utterance audio right after ASR (spec 9.3: "parallel with the
VAD" - in the turn pipeline the embedding window is the last ``speaker_window_s``
seconds of the utterance). The resolved ``user_id`` routes the VoiceMem
memory space BEFORE retrieval (spec 9.3.4): ``user_id="thomas"`` ->
separate SQLite + Qdrant collection, so memories never mix across speakers.

Four layers, cleanly separated for testability:

1. PURE math + records (no heavy imports, sandbox-testable):
   - :func:`cosine_similarity` - cosine of two embedding sequences.
   - :func:`identify_speaker` - spec 9.3.2 identification: best cosine over
     the registered reference embeddings, ``known`` only when
     ``best_sim >= threshold`` (default 0.50).
   - :func:`average_embeddings` - registration averaging (spec 9.3.3).
   - :class:`SpeakerIdentification` - the result record.
   - :class:`SpeakerRegistry` - JSON persistence of reference embeddings
     (``data/speaker_registry.json``), thread-safe, never raises.

2. :class:`SpeakerEmbedder` - the real embedding extractor (SpeechBrain
   ``spkrec-ecapa-voxceleb`` - ECAPA-TDNN, 192-dim, ~23M params, CPU).
   speechbrain/torch are imported LAZILY inside :meth:`SpeakerEmbedder._load`
   so this module stays importable without them (CONTRACT hard rule 2).
   The model is loaded from the LOCAL lock-downloaded directory
   (``models/speaker/ecapa-voxceleb``) - NO network call at runtime
   (``from_hparams(source=<local path>)``). Every failure path degrades
   to ``None`` - the pipeline must survive a dead speaker recognizer
   (modularity contract 9.5 item 5/6: ``user_id="voice_user"`` fallback).

3. :class:`SpeakerRecognizer` - the pipeline-facing facade:
   ``identify(audio) -> Optional[SpeakerIdentification]`` combining the
   embedder + registry + threshold. Swappable by design (9.5 item 3):
   replacing SpeechBrain ECAPA with 3D-Speaker ERes2Net or wespeaker keeps
   the interface (string speaker id).

4. Registration helpers (spec 9.3.3): :func:`register_speaker` averages
   ~10 s of clean speech (multiple chunks) into one reference embedding.

Threshold note: the spec pins 0.50 as the ECAPA identification threshold
(conservative: unknown -> fallback user, memories NEVER mix - measured
cross-speaker cosine ~0.13-0.18, same-speaker segments 0.37-0.75 on short
windows, higher with the spec's 3-5 s window + enrollment averaging).
SpeechBrain's own pairwise verification default is 0.25; the threshold is
configurable (``speaker_match_threshold``) for tuning.

License note (LICENSES.md): speechbrain/spkrec-ecapa-voxceleb weights are
Apache-2.0 (not gated, official speechbrain org re-host); the speechbrain
pip package is Apache-2.0.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Embedding dimension of the pinned ECAPA-TDNN model (hyperparams.yaml
#: ``lin_neurons: 192`` - spec 9.3.1 output).
ECAPA_EMBEDDING_DIM = 192

#: Default identification threshold (spec 9.3.2: ``if best_sim >= 0.50``).
DEFAULT_MATCH_THRESHOLD = 0.50

#: Default analysis window: the last N seconds of the utterance (spec 9.3.1:
#: "last 3-5 seconds"; 5.0 is the stable upper bound for short turns).
DEFAULT_WINDOW_S = 5.0

#: Default registration minimum (spec 9.3.3: ~10 s of clean speech).
DEFAULT_REGISTRATION_MIN_S = 10.0

#: Registry JSON schema version.
REGISTRY_SCHEMA_VERSION = 1

#: Fallback user id when the speaker is unknown or recognition is off
#: (spec 9.5 item 6: the M1 pipeline must never stall on an unknown voice).
UNKNOWN_USER_ID = "voice_user"


# --------------------------------------------------------------------- pure #


@dataclass
class SpeakerIdentification:
    """One identification outcome.

    ``id`` is the recognized speaker id or ``None`` (unknown speaker);
    ``similarity`` the best cosine score (0.0 when nothing matched);
    ``registered`` the number of reference speakers compared against.
    """

    id: Optional[str]
    similarity: float = 0.0
    best_id: Optional[str] = None  # best candidate even when below threshold
    registered: int = 0
    latency_ms: float = 0.0

    @property
    def known(self) -> bool:
        return self.id is not None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict (all fields, rounded for readability)."""
        return {
            "id": self.id,
            "known": self.known,
            "similarity": round(self.similarity, 3),
            "best_id": self.best_id,
            "registered": self.registered,
            "latency_ms": round(self.latency_ms, 1),
        }


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two embedding sequences (pure Python, no numpy).

    Internally normalizes both vectors (identical semantics to torch's
    ``CosineSimilarity`` on raw embeddings). Zero-length or shape-mismatched
    inputs return 0.0 (never raises - identification must survive surprises).
    """
    try:
        if len(a) == 0 or len(a) != len(b):
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            xf = float(x)
            yf = float(y)
            dot += xf * yf
            norm_a += xf * xf
            norm_b += yf * yf
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    except (TypeError, ValueError):
        return 0.0


def identify_speaker(
    embedding: Optional[Sequence[float]],
    references: Dict[str, Sequence[float]],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Optional[SpeakerIdentification]:
    """Spec 9.3.2 identification: best cosine over the reference set.

    Returns ``None`` when the embedding itself is missing/empty (a dead
    analyzer - the caller falls back to the default user id); otherwise a
    :class:`SpeakerIdentification` whose ``id`` is the best-matching
    registered speaker with ``similarity >= threshold`` or ``None``
    (unknown speaker - graceful degradation, never an exception).
    """
    if embedding is None or len(embedding) == 0:
        return None
    best_id: Optional[str] = None
    best_sim = 0.0
    for spk_id, ref in references.items():
        sim = cosine_similarity(embedding, ref)
        if sim > best_sim:
            best_sim = sim
            best_id = spk_id
    matched: Optional[str] = best_id if best_sim >= threshold else None
    return SpeakerIdentification(
        id=matched,
        similarity=best_sim,
        best_id=best_id,
        registered=len(references),
    )


def average_embeddings(embeddings: Sequence[Sequence[float]]) -> List[float]:
    """Element-wise mean of the embeddings (spec 9.3.3 registration).

    Empty input returns ``[]``; shape-mismatched entries are skipped.
    """
    valid = [list(map(float, e)) for e in embeddings if e and len(e) > 0]
    if not valid:
        return []
    dim = len(valid[0])
    rows = [row for row in valid if len(row) == dim]
    if not rows:
        return []
    return [sum(row[i] for row in rows) / len(rows) for i in range(dim)]


def tail_window(audio: Any, sample_rate: int, seconds: float) -> Any:
    """The last ``seconds`` of a 1-D float audio array (spec 9.3.1).

    Returns the input unchanged when it is shorter than the window or has
    no ``size`` attribute (duck-typed mocks).
    """
    size = int(getattr(audio, "size", 0) or 0)
    limit = int(sample_rate * seconds)
    if size <= 0 or limit <= 0 or size <= limit:
        return audio
    return audio[size - limit :]


# ---------------------------------------------------------------- registry #


class SpeakerRegistry:
    """JSON persistence of registered speaker reference embeddings.

    Layout (``data/speaker_registry.json``)::

        {
          "schema_version": 1,
          "speakers": {
            "thomas": {
              "embedding": [...192 floats...],
              "created_at": "2026-09-01T12:00:00+00:00",
              "updated_at": "...",
              "samples": 3,
              "audio_seconds": 10.4
            }
          }
        }

    Thread-safe (a lock guards every mutation), best-effort: a failing
    registry never raises (the turn falls back to the unknown user id).
    Re-registering an existing id OVERWRITES the reference (idempotent).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = self._load()

    # -- paths / state ------------------------------------------------------ #

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> Dict[str, Any]:
        """Load the registry file (tolerant: missing/corrupt -> empty)."""
        try:
            if not self._path.is_file():
                return {"schema_version": REGISTRY_SCHEMA_VERSION, "speakers": {}}
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(
                data.get("speakers"), dict
            ):
                return {"schema_version": REGISTRY_SCHEMA_VERSION, "speakers": {}}
            return data
        except (OSError, ValueError) as exc:
            logger.warning("Speaker registry unreadable (non-fatal): %s", exc)
            return {"schema_version": REGISTRY_SCHEMA_VERSION, "speakers": {}}

    def _save(self) -> bool:
        """Write the registry file (best effort, never raises).

        v0.4.4: atomic replace (temp file + os.replace). A crash mid-write
        used to leave a truncated JSON file behind and the tolerant loader
        then silently reset the registry to empty - every enrolled speaker
        was lost. os.replace is atomic on POSIX and Windows alike.
        """
        import os
        import tempfile

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self._path.parent),
                prefix=self._path.name + ".",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(self._data, ensure_ascii=True, indent=1))
                os.replace(tmp_name, self._path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
            return True
        except OSError as exc:
            logger.warning("Speaker registry write failed (non-fatal): %s", exc)
            return False

    # -- queries ------------------------------------------------------------ #

    def speaker_ids(self) -> List[str]:
        """Registered ids (sorted for deterministic identification order)."""
        try:
            with self._lock:
                return sorted(self._data["speakers"].keys())
        except (KeyError, TypeError):
            return []

    def count(self) -> int:
        return len(self.speaker_ids())

    def references(self) -> Dict[str, List[float]]:
        """Reference embeddings as ``{speaker_id: [floats]}`` (a copy)."""
        result: Dict[str, List[float]] = {}
        with self._lock:
            speakers = self._data.get("speakers") or {}
            for spk_id, entry in speakers.items():
                if not isinstance(entry, dict):
                    continue
                emb = entry.get("embedding")
                if isinstance(emb, list) and emb:
                    try:
                        result[str(spk_id)] = [float(x) for x in emb]
                    except (TypeError, ValueError):
                        continue
        return result

    # -- mutation ----------------------------------------------------------- #

    def register(
        self,
        speaker_id: str,
        embedding: Sequence[float],
        samples: int = 1,
        audio_seconds: float = 0.0,
    ) -> bool:
        """Store one reference embedding (overwrite on re-registration).

        Returns True when persisted; an empty embedding is rejected.
        """
        spk = (speaker_id or "").strip()
        if not spk or not embedding or len(embedding) == 0:
            return False
        emb = [float(x) for x in embedding]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self._lock:
                speakers = self._data.setdefault("speakers", {})
                existing = speakers.get(spk) or {}
                speakers[spk] = {
                    "embedding": emb,
                    "created_at": existing.get("created_at") or now,
                    "updated_at": now,
                    "samples": int(samples),
                    "audio_seconds": round(float(audio_seconds), 1),
                }
                return self._save()
        except (TypeError, ValueError) as exc:
            logger.warning("Speaker registration failed (non-fatal): %s", exc)
            return False

    def remove(self, speaker_id: str) -> bool:
        """Remove one registered speaker (best effort)."""
        spk = (speaker_id or "").strip()
        try:
            with self._lock:
                speakers = self._data.get("speakers") or {}
                if spk not in speakers:
                    return False
                del speakers[spk]
                return self._save()
        except KeyError:
            return False


# ---------------------------------------------------------------- embedder #


class SpeakerEmbedder:
    """ECAPA-TDNN embedder backed by SpeechBrain (local model dir, CPU).

    Contract (pipeline duck-typing):
    - ``is_available() -> bool`` - speechbrain importable AND model files
      present in the LOCAL directory (no network at runtime).
    - ``embed(audio) -> Optional[List[float]]`` - sync, CPU, ~30-50 ms per
      5 s window on the target machine; ``None`` on ANY failure (graceful
      degradation - the pipeline falls back to ``user_id="voice_user"``).

    speechbrain/torch are imported lazily inside :meth:`_load` (the module
    itself must stay importable in the dependency-free sandbox). The model
    is loaded ONCE (guarded by ``self._loaded``); a failed load is
    remembered so a broken install does not retry every turn.
    """

    #: Files the local model directory must contain (lock-downloaded;
    #: hyperparams.yaml's pretrainer loads embedding_model.ckpt,
    #: mean_var_norm_emb.ckpt, classifier.ckpt, label_encoder.txt).
    REQUIRED_FILES: Tuple[str, ...] = (
        "embedding_model.ckpt",
        "hyperparams.yaml",
        "mean_var_norm_emb.ckpt",
        "classifier.ckpt",
        "label_encoder.txt",
    )

    def __init__(
        self,
        model_dir: Path,
        window_s: float = DEFAULT_WINDOW_S,
        sample_rate: int = 16000,
        model_factory: Optional[Callable[[Path], Any]] = None,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._window_s = float(window_s)
        self._sample_rate = int(sample_rate)
        self._model: Optional[Any] = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._model_factory = model_factory  # test seam: fake from_hparams
        self._torch: Optional[Any] = None

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
        """speechbrain importable AND model files on disk (never raises)."""
        if not self.files_present():
            return False
        try:
            import speechbrain  # noqa: F401 - pure presence check
        except Exception:  # noqa: BLE001 - ImportError and friends
            return False
        return True

    def warm_up(self) -> bool:
        """Preload the model NOW (at startup) - the first turn must not pay
        the ~1 s lazy load. Returns True when ready; a failed warm-up leaves
        the embedder in the degraded (None) state.
        """
        return self._load() is not None

    # -- inference ----------------------------------------------------------- #

    def _load(self) -> Optional[Any]:
        """Load the SpeechBrain interface once from the LOCAL directory.

        ``SpeakerRecognition.from_hparams(source=<local dir>)`` resolves a
        local directory directly (speechbrain ``guess_source``: an existing
        directory is LOCAL - no HuggingFace call). Returns the interface or
        None (failure logged ONCE and remembered).
        """
        if self._loaded:
            return self._model
        if self._load_error is not None:
            return None
        self._loaded = True  # one attempt only - no per-turn retry storm
        try:
            if self._model_factory is not None:
                self._model = self._model_factory(self._model_dir)
                return self._model
            from speechbrain.inference.speaker import SpeakerRecognition

            self._model = SpeakerRecognition.from_hparams(
                source=str(self._model_dir), savedir=str(self._model_dir)
            )
            return self._model
        except Exception as exc:  # noqa: BLE001 - degradation, never raise
            self._load_error = str(exc)
            logger.warning(
                "Speaker embedder unavailable (pipeline falls back to the "
                "default user id): %s",
                exc,
            )
            return None

    def embed(self, audio: Any) -> Optional[List[float]]:
        """Embed the tail window of one utterance; None on any failure.

        The input is a 16 kHz mono float array (numpy or duck-typed). The
        output is the flattened 192-dim embedding as plain floats (no torch
        leak into the pipeline). Runs under ``torch.no_grad`` when torch is
        importable.
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
            if self._torch is not None:
                with self._torch.no_grad():
                    raw = model.encode_batch(tensor)
            else:
                raw = model.encode_batch(tensor)
        except Exception as exc:  # noqa: BLE001 - never kill a turn
            logger.warning("Speaker embedding failed (non-fatal): %s", exc)
            return None
        return self._flatten(raw)

    def embed_windows(self, audio: Any) -> List[List[float]]:
        """Embed EVERY consecutive ``window_s`` slice of *audio*.

        v0.4.4 fix: :meth:`embed` keeps its utterance semantics (the TAIL
        window - right for live turns), but REGISTRATION must cover all the
        collected speech. Passing one concatenated array through ``embed``
        (what the CLI registration used to do) silently discarded everything
        except the last ~5 s: 10.4 s collected, 5.0 s actually embedded.
        Splitting into consecutive windows and averaging the embeddings (the
        caller) restores spec 9.3.3 (~10 s averaged enrollment). Windows that
        fail to embed are skipped, not fatal.
        """
        import numpy as np

        try:
            x = np.asarray(audio)
        except Exception:  # noqa: BLE001 - duck-typed input
            return []
        total = int(getattr(x, "size", 0) or 0)
        win = int(self._sample_rate * float(self._window_s))
        if win <= 0 or total <= 0:
            return []
        out: List[List[float]] = []
        start = 0
        while start < total:
            segment = x[start : start + win]
            emb = self.embed(segment)
            if emb:
                out.append(emb)
            start += win
        return out

    # -- internals ------------------------------------------------------------ #

    def _to_tensor(self, window: Any) -> Any:
        """float32 (batch=1, time) input for ``encode_batch``.

        Prefers a torch tensor (the REAL speechbrain interface takes torch
        tensors; speechbrain requires torch, so the real path always has
        it). When torch is NOT importable (sandbox tests with a fake model
        factory), falls back to a (1, N) float32 numpy array - duck-typed
        fakes accept it, and the real model is never reachable without
        torch anyway (``is_available`` guards the speechbrain import).
        """
        try:
            import torch

            self._torch = torch
        except ImportError:  # sandbox/test path: numpy fallback
            self._torch = None
            return self._as_batched_numpy(window)
        if hasattr(window, "dtype"):
            array = window
        else:
            array = self._numpy(window)
        if str(getattr(array, "dtype", "")).endswith("float32"):
            tensor = torch.from_numpy(array)
        else:
            tensor = torch.from_numpy(self._numpy(array, dtype="float32"))
        tensor = tensor.to(torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _as_batched_numpy(window: Any) -> Any:
        """(1, N) float32 numpy array (torch-free input shape)."""
        import numpy as np

        array = np.asarray(window)
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    @staticmethod
    def _numpy(array: Any, dtype: Optional[str] = None) -> Any:
        """numpy array from a sequence (dtype='float32' optional)."""
        import numpy as np

        result = np.asarray(array)
        if dtype:
            result = result.astype(np.dtype(dtype))
        return result

    @staticmethod
    def _flatten(raw: Any) -> Optional[List[float]]:
        """Flatten a (1, 1, 192)/(1, 192)/[192] embedding to plain floats."""
        try:
            if hasattr(raw, "detach"):  # torch tensor
                raw = raw.detach().cpu().tolist()
            elif hasattr(raw, "tolist"):  # numpy array
                raw = raw.tolist()
            if not isinstance(raw, (list, tuple)):
                return None
            flat: List[float] = []
            stack: List[Any] = [raw]
            while stack:
                item = stack.pop()
                if isinstance(item, (list, tuple)):
                    stack.extend(reversed(list(item)))
                else:
                    try:
                        flat.append(float(item))
                    except (TypeError, ValueError):
                        return None
            if not flat:
                return None
            return flat
        except (TypeError, ValueError, IndexError):
            return None


# --------------------------------------------------------------- recognizer #


class SpeakerRecognizer:
    """Pipeline-facing facade: embedder + registry + threshold.

    Contract (pipeline duck-typing):
    - ``is_available() -> bool`` - embedder usable (model + speechbrain).
    - ``identify(audio) -> Optional[SpeakerIdentification]`` - sync, CPU;
      ``None`` when the analyzer itself is dead (embedding failed), an
      unknown-speaker record (``id=None``) when nothing matched the
      threshold. Either way the pipeline falls back to the default user
      id (modularity 9.5 item 6).
    - ``warm_up() -> bool`` - preload at startup.
    - ``register(speaker_id, audio_chunks) -> bool`` - registration flow.
    """

    def __init__(
        self,
        embedder: SpeakerEmbedder,
        registry: SpeakerRegistry,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        registration_min_s: float = DEFAULT_REGISTRATION_MIN_S,
        sample_rate: int = 16000,
    ) -> None:
        self._embedder = embedder
        self._registry = registry
        self._threshold = float(threshold)
        self._registration_min_s = float(registration_min_s)
        self._sample_rate = int(sample_rate)

    # -- availability -------------------------------------------------------- #

    @property
    def registry(self) -> SpeakerRegistry:
        return self._registry

    @property
    def threshold(self) -> float:
        return self._threshold

    def is_available(self) -> bool:
        return self._embedder.is_available()

    def warm_up(self) -> bool:
        return self._embedder.warm_up()

    # -- identification ------------------------------------------------------ #

    def identify(self, audio: Any) -> Optional[SpeakerIdentification]:
        """Identify the speaker of one utterance (None = analyzer dead).

        Runs the embedder on the utterance tail window and matches against
        the registry references with the configured threshold.
        """
        start = time.perf_counter()
        embedding = self._embedder.embed(audio)
        if embedding is None:
            return None
        result = identify_speaker(embedding, self._registry.references(), self._threshold)
        if result is None:
            return None
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        return result

    # -- registration -------------------------------------------------------- #

    def register(
        self, speaker_id: str, audio_chunks: Sequence[Any]
    ) -> Tuple[bool, float]:
        """Register a speaker from ~``registration_min_s`` of clean speech.

        Embeds every chunk, averages the embeddings (spec 9.3.3) and stores
        the reference in the registry. Returns ``(ok, audio_seconds)``; the
        seconds value reports how much speech was actually used so callers
        can warn about too-short registrations.
        """
        chunks = [c for c in audio_chunks if int(getattr(c, "size", 0) or 0) > 0]
        if not chunks:
            return False, 0.0
        embeddings = []
        for chunk in chunks:
            # v0.4.4: embed every consecutive window of each chunk (not just
            # the tail 5 s) so the full collected speech contributes to the
            # averaged reference (spec 9.3.3). Duck-typed embedders without
            # embed_windows fall back to the single embed call.
            embed_windows = getattr(self._embedder, "embed_windows", None)
            if callable(embed_windows):
                for emb in embed_windows(chunk):
                    if emb:
                        embeddings.append(emb)
            else:
                emb = self._embedder.embed(chunk)
                if emb:
                    embeddings.append(emb)
        if not embeddings:
            return False, 0.0
        total_samples = sum(int(getattr(c, "size", 0) or 0) for c in chunks)
        seconds = total_samples / float(self._sample_rate)
        reference = average_embeddings(embeddings)
        if not reference:
            return False, seconds
        ok = self._registry.register(
            speaker_id, reference, samples=len(embeddings), audio_seconds=seconds
        )
        if seconds < self._registration_min_s:
            logger.warning(
                "Speaker %r registered from only %.1f s of speech (spec 9.3.3 "
                "recommends ~%.0f s - identification accuracy may be lower)",
                speaker_id,
                seconds,
                self._registration_min_s,
            )
        return ok, seconds


# ------------------------------------------------------------ registration #


def register_speaker(
    recognizer: SpeakerRecognizer,
    speaker_id: str,
    audio_chunks: Sequence[Any],
) -> Tuple[bool, float]:
    """Convenience wrapper (CLI registration mode): ``recognizer.register``."""
    return recognizer.register(speaker_id, audio_chunks)
