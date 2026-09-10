"""Quasi-streaming ASR engine for Qwen3-ASR-0.6B (torch/transformers, lazy imports).

Milestone 6.5 strategy: Qwen3-ASR has no true streaming partials, so the engine
batches audio. ``feed`` accumulates ``config.asr_chunk_samples`` (600 ms @
16 kHz = 9600 samples) and transcribes each completed chunk for a running
partial; ``flush`` re-transcribes the WHOLE utterance buffer in one call (the
accurate approach) and returns the final transcript, then resets the engine.

v0.4.7 rewrite — the call path. The transformers ``automatic-speech-
recognition`` PIPELINE never supported the Qwen3-ASR chat-template input:
``pipeline(...)`` calls only the feature extractor (no ``input_ids`` from the
chat template) and then ``generate(input_features=...)``, which cannot work
for this decoder-based model. The v0.4.6 field report showed exactly that
("ASR call-form probe: no form returned a transcript dict" — every call form
raised, so speech produced no transcript while typed text worked). The engine
now drives processor + ``model.generate`` directly, matching the official
usage (Qwen3-ASR model card / qwen-asr package / transformers >= 5.0):

  1. transformers >= 5.0 native: ``processor.apply_transcription_request(
     audio, language, prompt)`` builds the chat template internally.
  2. official qwen-asr package (pins transformers 4.57): manual
     ``apply_chat_template`` + ``processor(text=[prompt], audio=[wav])``.

Both end in ``model.generate(**inputs)`` and decode only the GENERATED tokens
(the prompt is stripped by ``input_ids`` length), then parse the
``language <LANG><asr_text>...`` output format into plain text.

Heavy dependencies: torch and transformers are imported via a guarded
module-level try/except (CONTRACT.md convention 2), so importing this module
never fails — but constructing/using the engine without the stack raises
RuntimeError with install hints. The model itself loads lazily on first use.

v0.4.9 thread safety: the SHARED backend is loaded once per
``(model_source, device)`` and every session's AsrEngine (plus the warm-up
and the /api/asr-test one-shot) calls ``model.generate`` on that single
PyTorch instance. PyTorch ``generate`` is NOT safe for unsynchronised
concurrent calls from multiple threads (CUDA/CPU race → corrupt transcripts,
CUDA OOM or crashes — the v0.4.7 code-analysis report's issue #2). The
module lock therefore guards BOTH the load AND every inference call, so
sessions serialize on the model instead of racing inside it.

v0.4.10 field fix ("hangfelvétel elkészül, de az ASR nem csinál vele
semmit"): the ORIGINAL Qwen/Qwen3-ASR-0.6B repo — the one our downloader
fetches — declares ``"feature_extractor_type": "WhisperFeatureExtractor"``
in preprocessor_config.json. Whisper's extractor never right-pads the mel
time axis to a multiple of ``2 * n_window`` (100) and computes features the
Qwen3ASREncoder cannot chunk, so every inference raises
``ValueError: Qwen3ASREncoder expects padded_feature_length to be a
multiple of n_window * 2 (100), but got 938`` — the model loads, the
recording is made, the transcript is never produced. The -hf repos ship the
correct ``Qwen3ASRFeatureExtractor``; ``_ensure_qwen3_feature_extractor``
now detects the Whisper extractor and swaps in a properly configured
``Qwen3ASRFeatureExtractor`` (n_window resolved from the model config), so
BOTH repo worlds feed the encoder correctly. (The same field log's
``apply_transcription_request ... continue_final_message`` warning is the
original repo's chat template dropping assistant turns — the manual
Layer-2 path already handles that world, and with the extractor fixed it
now completes.)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import AgentConfig

try:  # pragma: no cover — sandbox has no torch; target machine has it
    import torch  # type: ignore[import-untyped]
    import transformers  # type: ignore[import-untyped]  # noqa: F401 — availability flag only

    _HAS_ASR_DEPS = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    transformers = None  # type: ignore[assignment]
    _HAS_ASR_DEPS = False

logger = logging.getLogger(__name__)

#: Shared model cache: ONE loaded backend per (model_source, device) — the
#: web server creates an AsrEngine per browser session, and without this cache
#: every tab would load its own 0.6B copy into VRAM. The warm-up at server
#: start populates the same entry, so the first session is warm.
_BACKEND_CACHE: "dict[tuple, Any]" = {}
#: v0.4.9: guards the cache AND serializes every ``model.generate`` call.
#: Loading and inference share ONE lock on purpose: an inference that starts
#: while the model is still loading would run against a half-initialised
#: backend, and two concurrent generates on the same PyTorch model corrupt
#: state (report issue #2). Sessions queue here instead of racing.
_BACKEND_CACHE_LOCK = __import__("threading").Lock()

#: Marker the model uses between the (auto-detected) language line and the
#: transcription: output looks like "language Hungarian<asr_text>hello ...".
_ASR_TEXT_TAG = "<asr_text>"


def _import_model_class() -> Any:
    """Return the Qwen3-ASR model class from whichever provider is installed.

    Order: transformers >= 5.0 ships native ``qwen3_asr`` support; the
    official ``qwen-asr`` PyPI package (which pins transformers 4.57) bundles
    the same class and registers it with AutoModel. Raises ImportError with
    an actionable message when neither world can provide the class.
    """
    try:
        from transformers import Qwen3ASRForConditionalGeneration  # type: ignore

        return Qwen3ASRForConditionalGeneration
    except ImportError:
        pass
    try:
        from qwen_asr.core.transformers_backend import (  # type: ignore
            Qwen3ASRForConditionalGeneration,
        )

        return Qwen3ASRForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "No Qwen3-ASR model class available: install transformers >= 5.0 "
            '(pip install -U "transformers>=5.0") or the official package '
            "(pip install qwen-asr)."
        ) from exc


def _parse_thinker_layout(model_source: str) -> "Optional[dict[str, Any]]":
    """Parse the ORIGINAL Qwen repo config layout; None for anything else.

    Pure stdlib (json only) so the layout detection is unit-testable on any
    machine. Returns ``{"text": {...}, "audio": {...}, "tokens": {...}}``
    with the sub-config dicts cleaned for Qwen3ASRConfig consumption, or
    None when the directory is not the original layout (-hf repos and hub
    ids — they load natively, nothing to repair) or the JSON is unreadable.
    """
    import json
    from pathlib import Path

    cfg_path = Path(model_source) / "config.json"
    if not cfg_path.is_file():
        return None  # hub id or missing file — nothing to repair
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    thinker = data.get("thinker_config")
    if not isinstance(thinker, dict):
        return None  # -hf layout (top-level audio_config/text_config)
    text_cfg = dict(thinker.get("text_config") or {})
    audio_cfg = dict(thinker.get("audio_config") or {})
    for sub in (text_cfg, audio_cfg):
        sub.pop("_name_or_path", None)
        sub.pop("architectures", None)
    text_cfg.setdefault("model_type", "qwen3")
    # the original repo names the encoder "qwen3_asr_audio_encoder";
    # transformers registers the class as "qwen3_asr_encoder"
    audio_cfg["model_type"] = "qwen3_asr_encoder"
    tokens = {
        key: thinker[key]
        for key in ("audio_token_id", "audio_start_token_id", "audio_end_token_id")
        if thinker.get(key) is not None
    }
    return {"text": text_cfg, "audio": audio_cfg, "tokens": tokens}


def _repaired_qwen_config(model_source: str) -> Any:
    """Build a correct Qwen3ASRConfig for the ORIGINAL Qwen repo layout, or None.

    The second half of the v0.4.10 field diagnosis: Qwen/Qwen3-ASR-0.6B's
    config.json nests everything under ``thinker_config`` (the original
    qwen-asr library layout). transformers' ``AutoConfig`` does not map that
    layout for qwen3_asr, so the sub-configs fall back to DEFAULTS (text
    hidden 2048 vs the real 1024, audio d_model 1024 vs the real 896) — and
    the checkpoint keys carry the ``thinker.`` prefix, so ``from_pretrained``
    matches ZERO weights and silently runs a randomly initialised model
    (its "some weights were not used" warnings go to stderr via the
    transformers logger, which does NOT propagate into web-server.log —
    invisible in the field log; the model "loads" and every inference
    produces garbage).

    Returns a Qwen3ASRConfig built from the repo's own thinker values, or
    None when the layout is not the original one / transformers cannot
    provide the class — the caller then falls back to the plain
    from_pretrained path instead of dying.
    """
    layout = _parse_thinker_layout(model_source)
    if layout is None:
        return None
    try:
        from transformers import Qwen3ASRConfig  # type: ignore
    except ImportError:
        logger.warning(
            "Qwen3ASRConfig unavailable on this transformers version — the "
            "original-repo config cannot be repaired; ASR may run with "
            "default shapes"
        )
        return None
    try:
        config = Qwen3ASRConfig(
            audio_config=layout["audio"], text_config=layout["text"], **layout["tokens"]
        )
        logger.info(
            "Original Qwen repo layout detected (thinker_config) — repaired "
            "config: text hidden %s x %s layers, audio d_model %s x %s layers, "
            "n_window %s",
            config.text_config.hidden_size,
            config.text_config.num_hidden_layers,
            config.audio_config.d_model,
            config.audio_config.encoder_layers,
            config.audio_config.n_window,
        )
        return config
    except Exception as exc:  # noqa: BLE001 — fall back to the plain path
        logger.warning("Qwen config repair skipped (%s); using the plain load path", exc)
        return None


#: Key remap for the ORIGINAL Qwen repo checkpoint (``thinker.`` prefix) to
#: the transformers class layout. Regex source patterns (specific rules FIRST
#: — transforms apply in order): the audio tower's proj1/proj2 become the
#: model's multi_modal_projector, the inner text model becomes
#: model.language_model, and the tied lm_head drops the thinker prefix.
_ORIGINAL_QWEN_KEY_MAPPING: "dict[str, str]" = {
    r"^thinker\.audio_tower\.proj1\.": "model.multi_modal_projector.linear_1.",
    r"^thinker\.audio_tower\.proj2\.": "model.multi_modal_projector.linear_2.",
    r"^thinker\.audio_tower\.": "model.audio_tower.",
    r"^thinker\.model\.": "model.language_model.",
    r"^thinker\.lm_head\.": "lm_head.",
}


def _shared_backend(model_source: str, device: str) -> Any:
    """Return the cached ASR backend for (model_source, device), loading once.

    The backend is a small dict with ``processor``/``model`` keys (the model
    class + AutoProcessor, both placed on *device*). Raises RuntimeError
    (with an actionable hint) when the stack is missing or the model cannot
    be loaded. Concurrency: guarded by the module lock — the warm-up thread
    and the first session feed may race; both block until the single load
    finishes. The same lock is re-acquired per inference in ``_transcribe``
    (see the v0.4.9 note on ``_BACKEND_CACHE_LOCK``), so a load in progress
    also blocks inference, never the other way round (no deadlock: loading
    never calls ``_transcribe``).
    """
    key = (str(model_source), str(device))
    with _BACKEND_CACHE_LOCK:
        if key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        if not _HAS_ASR_DEPS:
            raise RuntimeError(
                "torch and transformers are required for ASR but not installed. "
                "On the target machine (Windows + CUDA): "
                "pip install torch --index-url https://download.pytorch.org/whl/cu128 "
                '&& pip install -U "transformers>=5.0"'
            )
        logger.info("Loading ASR model %s on device %s (shared, once)", model_source, device)
        try:
            from transformers import AutoProcessor

            model_cls = _import_model_class()
            processor = AutoProcessor.from_pretrained(model_source)
            # v0.4.10 field fix, part 1: the ORIGINAL Qwen repo layout
            # (thinker_config + thinker. checkpoint keys) leaves
            # from_pretrained with default shapes and ZERO weights loaded —
            # repair the config and remap the checkpoint keys.
            repaired_config = _repaired_qwen_config(model_source)
            if repaired_config is not None:
                model = model_cls.from_pretrained(
                    model_source,
                    config=repaired_config,
                    key_mapping=dict(_ORIGINAL_QWEN_KEY_MAPPING),
                    torch_dtype="auto",
                    low_cpu_mem_usage=True,
                )
            else:
                model = model_cls.from_pretrained(
                    model_source, torch_dtype="auto", low_cpu_mem_usage=True
                )
            model = model.to(device)
            model.eval()
            # v0.4.10 field fix, part 2: the ORIGINAL Qwen repo ships a
            # Whisper feature extractor that cannot feed the Qwen3ASREncoder
            # (no n_window mel padding) — swap in the correct extractor.
            _ensure_qwen3_feature_extractor(processor, model)
            backend = {"processor": processor, "model": model}
            _BACKEND_CACHE[key] = backend
            logger.info(
                "ASR model loaded (%s on %s, dtype %s)",
                model_source,
                device,
                str(getattr(model, "dtype", "auto")),
            )
            return backend
        except Exception as exc:
            hint = _load_error_hint(exc)
            raise RuntimeError(hint) from exc


def _qwen3_feature_extractor_class() -> Any:
    """Return the Qwen3ASRFeatureExtractor class, or None when unavailable.

    transformers >= 5.0 exports it at the top level; older versions (the
    official qwen-asr package pins 4.57) do not have it at all — in that
    world the repair is skipped with a warning and the caller keeps the
    processor as loaded (the qwen-asr package drives its own feature
    extraction).
    """
    try:
        from transformers import Qwen3ASRFeatureExtractor  # type: ignore

        return Qwen3ASRFeatureExtractor
    except ImportError:
        pass
    try:
        from transformers.models.qwen3_asr import (  # type: ignore
            Qwen3ASRFeatureExtractor,
        )

        return Qwen3ASRFeatureExtractor
    except ImportError:
        return None


def _resolve_n_window(model: Any) -> int:
    """Read the audio encoder's n_window from the model config (default 50).

    The value lives on the encoder sub-config: ``config.audio_config`` for
    the -hf repos, ``config.thinker_config.audio_config`` for the original
    Qwen repo layout (dicts in pre-from_pretrained form are handled too).
    50 is the value both worlds ship — 100 mel-frame chunks, 1 s of audio.
    """
    cfg = getattr(model, "config", None)
    nodes: list[Any] = []
    if cfg is not None:
        nodes.append(cfg)
        for attr in ("audio_config", "thinker_config"):
            node = getattr(cfg, attr, None)
            if isinstance(node, dict):
                node = node.get("audio_config") or node
            if node is not None:
                nodes.append(node)
                sub = node.get("audio_config") if isinstance(node, dict) else getattr(node, "audio_config", None)
                if sub is not None:
                    nodes.append(sub)
    for node in nodes:
        value = node.get("n_window") if isinstance(node, dict) else getattr(node, "n_window", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    return 50


def _ensure_qwen3_feature_extractor(processor: Any, model: Any) -> None:
    """Swap the original Qwen repo's Whisper feature extractor for the real one.

    Root cause of the v0.4.9 field report (recording made, ASR produced
    nothing): ``Qwen/Qwen3-ASR-0.6B/preprocessor_config.json`` declares
    ``"feature_extractor_type": "WhisperFeatureExtractor"`` — an extractor
    that never right-pads the mel time axis to a multiple of ``2 * n_window``
    (100) and whose attention mask the processor cannot map to mel frames,
    so ``Qwen3ASREncoder.forward`` raises ``ValueError`` on every call. The
    -hf repos declare the correct ``Qwen3ASRFeatureExtractor``. This repair
    builds that extractor from the loaded audio parameters + the model
    config's ``n_window`` and installs it on the processor (both the
    ``feature_extractor`` and ``audio_processor`` slots, whichever exist),
    making the ORIGINAL repo fully usable with transformers >= 5.0.

    Idempotent (already-correct extractors are left alone) and safe: any
    failure is logged and skipped, never propagated — a load must not die
    because the repair could not run.
    """
    try:
        fe_cls = _qwen3_feature_extractor_class()
        if fe_cls is None:
            logger.warning(
                "Qwen3ASRFeatureExtractor is unavailable on this transformers "
                "version — the original Qwen repo's Whisper feature extractor "
                "cannot be repaired here; if ASR fails with a "
                "padded_feature_length error, upgrade transformers (>= 5.0)"
            )
            return
        current = getattr(processor, "_audio_processor", None) or getattr(processor, "feature_extractor", None)
        if current is None:
            return
        if isinstance(current, fe_cls):
            return  # -hf repo (or already repaired) — nothing to do
        params = {
            "feature_size": int(getattr(current, "feature_size", None) or 128),
            "sampling_rate": int(getattr(current, "sampling_rate", None) or 16000),
            "hop_length": int(getattr(current, "hop_length", None) or 160),
            "n_fft": int(getattr(current, "n_fft", None) or 400),
            "chunk_length": int(getattr(current, "chunk_length", None) or 30),
            "padding_value": float(getattr(current, "padding_value", 0.0) or 0.0),
            "dither": float(getattr(current, "dither", 0.0) or 0.0),
            "return_attention_mask": bool(getattr(current, "return_attention_mask", True)),
            "n_window": _resolve_n_window(model),
            "min_length": 8000,  # original Qwen3-ASR library default (0.5 s)
        }
        fe = fe_cls(**params)
        # ProcessorMixin resolves the audio FE through the _audio_processor
        # property: ``audio_processor`` first, then ``feature_extractor``.
        # Set every slot that exists so the swap always takes effect.
        if hasattr(processor, "audio_processor"):
            processor.audio_processor = fe
        if hasattr(processor, "feature_extractor"):
            processor.feature_extractor = fe
        logger.info(
            "ASR feature extractor swapped: %s -> Qwen3ASRFeatureExtractor "
            "(n_window=%d, hop=%d, rate=%d) — the original Qwen repo declares "
            "a Whisper extractor that cannot feed this encoder",
            type(current).__name__,
            params["n_window"],
            params["hop_length"],
            params["sampling_rate"],
        )
    except Exception as exc:  # noqa: BLE001 — never kill the load for the repair
        logger.warning("ASR feature-extractor repair skipped: %s", exc)


def _load_error_hint(exc: Exception) -> str:
    """Turn a model load failure into an actionable message.

    Qwen3-ASR model support landed in transformers 5.0 (native ``qwen3_asr``
    module); on older versions the load rejects the config with 'model type
    qwen3_asr' / 'Unrecognized configuration'. The hint names the exact fix
    instead of a bare traceback, because the web UI surfaces this string in
    the Pipeline panel.
    """
    version = "unknown"
    try:
        from importlib.metadata import version as _v  # stdlib

        version = _v("transformers")
    except Exception:  # noqa: BLE001
        pass
    msg = f"Failed to load the ASR model: {exc}"
    low = str(exc).lower()
    if (
        "unrecognized configuration" in low
        or "qwen3" in low
        or "not a valid model identifier" in low
        or "does not recognize this architecture" in low
    ):
        msg += (
            f" — Qwen3-ASR needs transformers >= 5.0 (installed: {version}) "
            "or the official qwen-asr package; upgrade with: "
            '.venv\\Scripts\\python.exe -m pip install -U "transformers>=5.0" '
            "- or close this window and double-click START.bat again: the "
            "bootstrap detects the old version and repairs it automatically"
        )
    return msg


def _parse_asr_output(raw: str, forced: bool) -> str:
    """Parse one decoded ASR string into the plain transcript.

    Auto-detect output format: ``language Hungarian<asr_text>hello there``.
    Forced-language output is plain text (the ``language X<asr_text>``
    prefill is part of the prompt, not the generation). No tag at all → the
    whole string is treated as transcription (mirrors the official
    ``parse_asr_output`` behaviour).
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    if forced:
        return s
    if _ASR_TEXT_TAG in s:
        s = s.split(_ASR_TEXT_TAG, 1)[1]
    else:
        # Model omitted the tag: drop a leading bare "language X" line if
        # present so it never leaks into the transcript.
        first_break = s.find("\n")
        head = s if first_break < 0 else s[:first_break]
        if head.strip().lower().startswith("language "):
            s = "" if first_break < 0 else s[first_break + 1 :]
    return s.strip()


class AsrEngine:
    """ASR over raw 16 kHz float32 mono PCM (Qwen3-ASR-0.6B via transformers).

    Quasi-streaming contract:

    * ``feed(samples)`` appends audio to the utterance buffer. Whenever at least
      ``config.asr_chunk_samples`` NEW samples accumulated, the new chunk is
      transcribed and the running partial is returned ("" before the first chunk
      boundary). A failed chunk transcription is logged and skipped so a single
      inference hiccup never kills the conversation loop.
    * ``flush()`` transcribes the full utterance buffer in ONE call and returns
      the final transcript (falling back to the joined partials if that call
      fails), then resets the audio state.
    * ``reset()`` drops buffered audio and partials without unloading the model.
    * ``last_partial`` exposes the running partial for speculative retrieval.
    * ``last_error`` exposes the most recent inference failure for the Pipeline
      panel (empty string when healthy).
    """

    def __init__(self, config: AgentConfig) -> None:
        """Store config; the model loads lazily on the first feed/flush call."""
        self._config = config
        self._backend: Optional[dict] = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._buffer: "np.ndarray" = np.zeros(0, dtype=np.float32)
        self._chunk_cursor = 0
        self._partials: list[str] = []
        self._last_partial = ""
        self._last_error = ""
        self._use_native_request: Optional[bool] = None  # set by the probe

    # ---------------------------------------------------------------- public

    def is_available(self) -> bool:
        """True when torch + transformers are importable (model may still need loading)."""
        return _HAS_ASR_DEPS

    @property
    def last_error(self) -> str:
        """Most recent transcription failure ("" when the engine is healthy)."""
        return self._last_error

    def warm_up(self) -> bool:
        """Load the model NOW and verify one REAL inference (server warm-up).

        Returns True only when the model loads AND the probe transcription
        call completes without error — a model that loads but cannot
        transcribe must surface as a red ASR chip, not a green one (the
        v0.4.6 field report: everything green, speech produced nothing).
        Never raises.
        """
        try:
            self._ensure_loaded()
            return self._probe()
        except Exception as exc:  # noqa: BLE001 - warm-up must never kill the server
            logger.warning("ASR warm-up failed: %s", exc)
            self._last_error = str(exc)
            return False

    def feed(self, samples: "np.ndarray") -> str:
        """Append samples and run a partial transcription per accumulated chunk.

        Returns the running partial transcript of the current utterance — ""
        while no new chunk boundary has been reached yet.

        Raises:
            RuntimeError: the ASR stack (torch/transformers/model) is unavailable.
        """
        self._ensure_loaded()
        x = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        if x.size == 0:
            return self._last_partial
        self._buffer = np.concatenate((self._buffer, x))
        chunk = int(self._config.asr_chunk_samples)
        while self._buffer.size - self._chunk_cursor >= chunk:
            window = self._buffer[self._chunk_cursor : self._chunk_cursor + chunk]
            self._chunk_cursor += chunk
            try:
                text = self._transcribe(window)
            except Exception:
                self._last_error = "partial transcription failed - see logs/web-server.log"
                logger.exception("Partial ASR transcription failed; chunk skipped")
                continue
            if text:
                self._partials.append(text)
                self._last_partial = self._join(self._partials)
        return self._last_partial

    def flush(self) -> str:
        """Transcribe the whole utterance buffer in one call and return the final text.

        Falls back to the joined chunk partials when the full-utterance call
        fails. Resets the audio state afterwards (the loaded model is kept, so
        the next utterance starts immediately).

        Raises:
            RuntimeError: the ASR stack (torch/transformers/model) is unavailable.
        """
        self._ensure_loaded()
        final = ""
        if self._buffer.size > 0:
            try:
                final = self._transcribe(self._buffer)
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"{str(exc)[:200]} - see logs/web-server.log"
                logger.exception("Full-utterance ASR transcription failed; using partials")
                final = ""
        if not final.strip():
            final = self._join(self._partials)
        self.reset()
        return final.strip()

    def reset(self) -> None:
        """Drop buffered audio, partials and last partial (the model stays loaded)."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._chunk_cursor = 0
        self._partials = []
        self._last_partial = ""

    def transcribe_utterance(self, samples: "np.ndarray") -> str:
        """One-shot transcription of a COMPLETE utterance (v0.4.8 ASR test).

        The /api/asr-test endpoint feeds a whole recorded clip through the
        same call path the live chain uses (``_ensure_loaded`` →
        ``_transcribe`` → parse), bypassing VAD/WS/turn state entirely —
        "what does the mic + ASR actually hear" becomes testable even when
        the chain never reaches ASR. The model is the SHARED backend (the
        warm-up already loaded it), so this costs one inference, not a load.
        Raises RuntimeError with the load/inference reason on failure and
        leaves ``last_error`` set for the Pipeline panel.
        """
        x = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        if x.size == 0:
            return ""  # nothing to transcribe - do not even load the model
        t0 = time.perf_counter()
        try:
            self._ensure_loaded()
            self.reset()
            text = self._transcribe(x)
            logger.info(
                "ASR one-shot: %.1f s audio -> %d chars in %.0f ms",
                x.size / float(int(self._config.sample_rate)),
                len(text),
                (time.perf_counter() - t0) * 1000.0,
            )
            return text
        except Exception as exc:  # noqa: BLE001 - surfaced via HTTP verdict
            self._last_error = str(exc)[:200]
            logger.exception("ASR one-shot transcription failed")
            raise

    @property
    def last_partial(self) -> str:
        """Running partial transcript of the current utterance ("" before the first chunk)."""
        return self._last_partial

    # --------------------------------------------------------------- internals

    def _ensure_loaded(self) -> None:
        """Lazily resolve the SHARED backend on first use (caches failures).

        The heavy model lives in the module-level cache (one copy for all
        sessions + the warm-up); a failed load is remembered per engine
        instance so a broken install does not retry every frame.
        """
        if self._loaded:
            return
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        model_source = self._model_source()
        device = self._resolve_device()
        try:
            self._backend = _shared_backend(model_source, device)
            self._loaded = True
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            raise

    def _model_source(self) -> str:
        """Local model directory when present, else the HuggingFace model name."""
        candidates = [Path(self._config.asr_model_dir)]
        if self._config.asr_model_path:
            candidates.append(Path(self._config.asr_model_path))
        for local in candidates:
            if (local / "config.json").is_file():
                return str(local)
        return self._config.asr_model_name

    def _resolve_device(self) -> str:
        """Honor config.asr_device, falling back to CPU when CUDA is unavailable."""
        device = str(self._config.asr_device)
        if device.startswith("cuda") and torch is not None and not torch.cuda.is_available():
            logger.warning("cuda requested for ASR but unavailable — falling back to CPU")
            return "cpu"
        return device

    @staticmethod
    def _join(parts: list[str]) -> str:
        """Join chunk texts into one transcript (whitespace-normalized)."""
        return " ".join(p.strip() for p in parts if p.strip())

    # -- the official processor + generate call path (v0.4.7) -----------------

    def _probe(self) -> bool:
        """Verify ONE real transcription (0.6 s of quiet noise) and pick the call layer.

        Success = the call completes without an exception (the transcript of
        quiet noise may legitimately be empty). When the transformers >= 5.0
        native ``apply_transcription_request`` helper is available it is
        locked in; otherwise the manual chat-template path is used. Every
        failure is logged with its full traceback — the old probe swallowed
        the exceptions, which hid the broken pipeline call for a whole
        release.
        """
        rate = int(self._config.sample_rate)
        rng = np.random.default_rng(0)
        probe = (rng.standard_normal(int(rate * 0.6)) * 0.01).astype(np.float32)
        try:
            self._transcribe(probe)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"probe failed: {str(exc)[:200]}"
            logger.error("ASR probe failed: %s", exc, exc_info=True)
            return False
        layer = (
            "native apply_transcription_request"
            if self._use_native_request
            else "manual chat template"
        )
        logger.info("ASR call path verified: %s (probe inference OK)", layer)
        return True

    def _transcribe(self, samples: "np.ndarray") -> str:
        """Run the official Qwen3-ASR inference on raw float32 16 kHz mono samples.

        Drives processor + ``model.generate`` directly (the transformers ASR
        pipeline cannot build the chat-template ``input_ids`` this
        decoder-based model requires). Two compatible layers, probed once:

        * transformers >= 5.0: ``processor.apply_transcription_request(audio,
          language, prompt)`` (builds the chat template internally).
        * otherwise (official qwen-asr package / older transformers):
          ``apply_chat_template`` + ``processor(text=[prompt], audio=[wav])``.

        The generated tokens are decoded with the prompt stripped
        (``sequences[:, input_ids.shape[1]:]``) and the
        ``language <LANG><asr_text>...`` output format parsed into plain
        text. This sandbox has no GPU/torch, so the call is validated against
        the official model card and the qwen-asr reference implementation.
        """
        if not self._loaded or self._backend is None:
            raise RuntimeError("ASR model is not loaded (feed/flush load it lazily)")
        x = np.ascontiguousarray(np.asarray(samples, dtype=np.float32).reshape(-1))
        if x.size == 0:
            return ""
        processor = self._backend["processor"]
        model = self._backend["model"]
        language = (self._config.asr_language or "").strip() or None
        inputs = self._build_inputs(processor, x, language)
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is not None:
            inputs = inputs.to(model.device, dtype=model_dtype)
        else:
            inputs = inputs.to(model.device)
        max_new_tokens = int(getattr(self._config, "asr_max_new_tokens", 256) or 256)
        import contextlib

        guard = torch.inference_mode() if torch is not None else contextlib.nullcontext()
        # v0.4.9 (report issue #2): the backend is SHARED across sessions, and
        # PyTorch ``generate`` is not safe under concurrent calls — hold the
        # module lock for the whole inference+decode so two browser tabs (or a
        # live feed racing the /api/asr-test one-shot) queue up instead of
        # corrupting model state. The lock is also the load lock; a load in
        # flight simply delays this call.
        with _BACKEND_CACHE_LOCK, guard:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            sequences = getattr(out, "sequences", None)
            if sequences is None:
                sequences = out  # plain tensor return
            prompt_len = int(inputs["input_ids"].shape[1])
            decoded = processor.batch_decode(
                sequences[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        raw = str(decoded[0]) if decoded else ""
        return _parse_asr_output(raw, forced=bool(language))

    def _build_inputs(self, processor: Any, x: "np.ndarray", language: Optional[str]) -> Any:
        """Build the model inputs (chat template + audio features) for one waveform.

        ``_use_native_request`` caches the layer choice after the first call:
        None = not probed yet (try native, fall back), True/False = locked.
        """
        # Layer 1: transformers >= 5.0 native helper.
        if self._use_native_request is not False and hasattr(
            processor, "apply_transcription_request"
        ):
            try:
                inputs = processor.apply_transcription_request(
                    audio=[x], language=language if language else None, prompt=None
                )
                self._use_native_request = True
                return inputs
            except Exception as exc:  # noqa: BLE001 - fall through to layer 2
                if self._use_native_request is None:
                    # v0.4.8: first failure only, message trimmed — the
                    # transformers error carries the whole rendered chat
                    # template (multi-line dump), which buried the useful
                    # part in the field log. The layer choice is locked to
                    # the manual path right after, so this logs once per
                    # engine instance, never per call.
                    logger.warning(
                        "apply_transcription_request failed (%s); "
                        "falling back to the manual chat-template path",
                        str(exc).split("\n")[0][:200],
                    )

        # Layer 2: manual chat template (official qwen-asr package path).
        messages = [{"role": "user", "content": [{"type": "audio", "audio": x}]}]
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if language:
            # Force text-only output: prefill the assistant turn.
            prompt = prompt + f"language {language}{_ASR_TEXT_TAG}"
        inputs = processor(text=[prompt], audio=[x], return_tensors="pt", padding=True)
        if self._use_native_request is None:
            self._use_native_request = False
        return inputs
