"""Voice Activity Detection: Silero ONNX wrapper + pure state machine.

Two layers:

* :class:`VadStateMachine` — pure-Python frame-level speech start/end logic with
  a configurable hangover window. Floats in, :class:`VadEvent` out; no numpy and
  no external dependencies (unit-testable on any machine).
* :class:`SileroVad` — Silero VAD v5/v6 ONNX model executed through onnxruntime
  on the CPU. ``onnxruntime`` is imported lazily inside ``__init__``; every
  failure raises :class:`RuntimeError` with an actionable install/path hint.

v0.6.0 ROOT-CAUSE FIX (TASK-A Phase 3 — the first objectively broken boundary
of the voice chain, proven in scripts/asr_forensics.py): the OFFICIAL
silero-vad OnnxWrapper contract prepends a 64-sample rolling CONTEXT to every
512-sample frame (576-sample model windows, context carried between frames —
verified against snakers4/silero-vad ``utils_vad.py``). The v0.4.x-v0.5.2
wrapper fed bare 512-sample windows, which the model scores at ~0.003 on REAL,
perfectly ASR-transcribable speech (the field's "Silero never fires" reports):
measured on the real Windows capture, bare 512-sample feed → prob_max 0.0031 /
0 speech segments, official 64-sample-context feed → prob_max 1.000 /
235-of-312 frames above threshold. The energy/gain fallback gate (FusedVad,
v0.4.14) that was added to compensate for this bug — and which fragmented
continuous speech into ~320-384 ms utterances through its noise-floor gate —
is REMOVED: Silero with the correct feed contract is the SOLE production VAD
decision path (AudioBuffer -> Silero -> speech state; no heuristic gate).

The audio pipeline feeds 512-sample 16 kHz float32 frames (32 ms) into
:meth:`SileroVad.prob` and routes the returned probability into
:meth:`VadStateMachine.update`.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from app.config import AgentConfig

if TYPE_CHECKING:  # pragma: no cover — typing only, never imported at runtime
    import onnxruntime as ort

logger = logging.getLogger(__name__)

# Input names seen across Silero VAD ONNX releases (v3 .. v6).
_STATE_INPUT_NAMES = ("state", "h", "c", "rnn_state")
_SR_INPUT_NAMES = ("sr", "sample_rate", "sampling_rate")
_AUDIO_INPUT_PREFERRED = "input"
_PROB_OUTPUT_PREFERRED = "output"

#: The official silero-vad OnnxWrapper context window (samples @ 16 kHz):
#: every model call receives [previous 64 samples | new 512 samples].
_SILERO_CONTEXT_SAMPLES = 64


class VadEvent(Enum):
    """Transition event emitted by :class:`VadStateMachine` for one frame."""

    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


class VadStateMachine:
    """Frame-level speech start/end detector with hangover logic (pure Python).

    Semantics (``prob`` = speech probability of the current 32 ms frame):

    * ``SPEECH_START`` — exactly once, on the first frame with
      ``prob >= threshold`` while not already in speech (threshold is inclusive).
    * ``SPEECH_END`` — once, after ``ceil(hangover_ms / frame_ms)`` CONSECUTIVE
      frames with ``prob < threshold`` while in speech (hangover). Any
      above-threshold frame re-arms the hangover counter.
    * ``NONE`` — every other frame. ``in_speech`` stays True during the hangover
      window (speech has not ended yet).
    """

    def __init__(self, threshold: float, hangover_ms: int, frame_ms: int) -> None:
        """Configure the detector from raw millisecond units.

        Args:
            threshold: speech probability threshold (inclusive, 0..1).
            hangover_ms: silence required after speech before SPEECH_END.
            frame_ms: duration of one frame in milliseconds.
        """
        self._threshold = float(threshold)
        self._frame_ms = max(1, int(frame_ms))
        self._hangover_frames = max(0, math.ceil(int(hangover_ms) / self._frame_ms))
        self._in_speech = False
        self._silence_frames = 0

    def update(self, prob: float) -> VadEvent:
        """Process one frame's speech probability and return the transition event."""
        p = float(prob)
        if not self._in_speech:
            if p >= self._threshold:
                self._in_speech = True
                self._silence_frames = 0
                return VadEvent.SPEECH_START
            return VadEvent.NONE
        if p >= self._threshold:
            self._silence_frames = 0
            return VadEvent.NONE
        # In speech, below threshold: run the hangover countdown.
        self._silence_frames += 1
        if self._silence_frames >= self._hangover_frames:
            self._in_speech = False
            self._silence_frames = 0
            return VadEvent.SPEECH_END
        return VadEvent.NONE

    def reset(self) -> None:
        """Reset to the initial (out-of-speech) state; drops hangover progress."""
        self._in_speech = False
        self._silence_frames = 0

    @property
    def in_speech(self) -> bool:
        """True between SPEECH_START and the subsequent SPEECH_END (hangover included)."""
        return self._in_speech


class SileroVad:
    """Silero VAD v5/v6 ONNX inference wrapper (CPU execution provider).

    Implements the OFFICIAL feed contract (v0.6.0 root-cause fix): every
    512-sample frame is scored together with a 64-sample rolling CONTEXT
    (the previous window's tail), exactly like the reference
    ``silero_vad.utils_vad.OnnxWrapper``: model input =
    ``[context (64) | frame (512)]`` = 576 samples, and the context for
    the next call is the LAST 64 samples of the current input. Streams
    start with a zero context; :meth:`reset` re-zeroes it between
    utterances.

    Handles the recurrent model signature (inputs ``input`` / ``state`` /
    ``sr``) defensively: the declared session inputs are inspected at init
    and probed with a silent dummy frame; the first working feed plan is
    locked in. If the signature does not expose a usable recurrent state,
    the class falls back to a stateless call and logs a warning. The hidden
    state (v5/v6: shape ``[1, 128]`` or ``[2, 1, 128]`` depending on the
    export) is zeroed by :meth:`reset` — call it between utterances.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Create the onnxruntime CPU session for ``config.silero_vad_path``.

        Raises:
            RuntimeError: onnxruntime is not installed (with pip hint), or the
                model file does not exist (with the expected path), or the model
                signature is not supported by this wrapper.
        """
        ort = self._import_onnxruntime()
        model_path = config.silero_vad_path
        if not model_path.is_file():
            raise RuntimeError(
                "Silero VAD model file not found. Expected at: "
                f"{model_path} — set SILERO_VAD_PATH or place silero_vad.onnx "
                "under <root>/bin (see scripts/download_models.ps1)"
            )
        self._config = config
        self._sample_rate = int(config.sample_rate)
        self._frame_samples = int(config.vad_frame_samples)
        # v0.6.0 official feed contract: rolling 64-sample context state.
        self._ctx = np.zeros(_SILERO_CONTEXT_SAMPLES, dtype=np.float32)
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        self._session: Any = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._bind_model()
        self.reset()
        # v0.4.9: frame-buffer state (sub-window leftovers + last probability).
        self._frame_buf: Optional["np.ndarray"] = None
        self._last_prob = 0.0
        logger.info(
            "Silero VAD loaded from %s (%d-sample frames + %d-sample rolling "
            "context @ %d Hz, %s inference)",
            model_path,
            self._frame_samples,
            _SILERO_CONTEXT_SAMPLES,
            self._sample_rate,
            "stateful" if self._state_template is not None else "stateless",
        )

    # ---------------------------------------------------------------- public

    def is_available(self) -> bool:
        """True once the ONNX session exists and a feed plan was bound.

        A successfully constructed SileroVad is always available (the
        constructor raises on every failure), so this is a plain sanity
        probe for callers that hold an Optional[SileroVad].
        """
        return getattr(self, "_session", None) is not None

    def prob(self, frame: "np.ndarray") -> float:
        """Return the speech probability (0..1) of the frame's audio.

        v0.4.9 (report issue #8): frames are BUFFERED, never truncated. The
        incoming samples are appended to an internal buffer and consumed in
        exact ``vad_frame_samples`` windows, each window advancing the
        recurrent state in order — a caller handing in a longer block (e.g. a
        100 ms resampler chunk) used to lose everything past 512 samples,
        silently biasing the VAD state machine. The return value is the
        probability of the LAST completed window; when the call completed no
        new window (short remainder), the previous probability is returned
        (0.0 before the first window). Callers feeding exactly 512-sample
        frames — the MicStream blocksize and the web uplink framing — see
        identical behaviour to the old implementation.
        """
        x = np.ascontiguousarray(np.asarray(frame, dtype=np.float32).reshape(-1))
        if x.size == 0:
            return self._last_prob
        self._frame_buf = (
            x
            if self._frame_buf is None
            else np.concatenate((self._frame_buf, x))
        )
        n = int(self._frame_samples)
        prob = self._last_prob
        while self._frame_buf.size >= n:
            window = self._frame_buf[:n]
            self._frame_buf = self._frame_buf[n:]
            prob = self._infer(window)
        if self._frame_buf.size > 0:
            logger.debug(
                "VAD frame buffered: %d leftover samples < %d window",
                int(self._frame_buf.size),
                n,
            )
        self._last_prob = prob
        return prob

    def flush_prob(self) -> float:
        """Zero-pad and score the buffered remainder (call at utterance end).

        Scores any sub-window leftovers with zero padding so the tail of an
        utterance is not silently dropped; returns the new probability (the
        previous one when the buffer was empty). The buffer is consumed.
        """
        n = int(self._frame_samples)
        if self._frame_buf is None or self._frame_buf.size == 0:
            return self._last_prob
        window = self._frame_buf[:n]
        if window.size < n:
            window = np.concatenate(
                (window, np.zeros(n - window.size, dtype=np.float32))
            )
        self._frame_buf = None
        self._last_prob = self._infer(window)
        return self._last_prob

    def _infer(self, window: "np.ndarray") -> float:
        """Run ONE exact-length window through the locked ONNX feed plan.

        v0.6.0: the window is scored WITH the official 64-sample rolling
        context — ``model input = [ctx | window]`` — and the context becomes
        the last 64 samples of that concatenated input (the reference
        OnnxWrapper semantics). This is the root-cause fix for the v0.4.x
        "Silero deaf on real speech" cascade: without the context the model
        returns ~0.003 on valid speech.
        """
        ctx = self._ctx
        if int(ctx.size) != _SILERO_CONTEXT_SAMPLES:  # defensive: exact shape
            ctx = np.zeros(_SILERO_CONTEXT_SAMPLES, dtype=np.float32)
        model_in = np.concatenate(
            (ctx, np.asarray(window, dtype=np.float32))
        )
        feed = self._build_feed(model_in)
        outputs = self._session.run(None, feed)
        if self._state_output_idx is not None and self._state_name is not None:
            self._state = np.asarray(outputs[self._state_output_idx], dtype=np.float32)
        value = float(np.asarray(outputs[self._prob_output_idx]).reshape(-1)[0])
        self._ctx = model_in[-_SILERO_CONTEXT_SAMPLES:].copy()
        return min(1.0, max(0.0, value))

    def reset(self) -> None:
        """Zero the hidden recurrent state (call between utterances/speakers).

        v0.4.9: also drops any sub-window leftover in the frame buffer so a
        new utterance never starts on the tail samples of the previous one.
        v0.6.0: re-zeroes the 64-sample rolling CONTEXT — a new utterance
        starts from the official silent-context state.
        """
        if self._state_template is not None:
            self._state: Optional["np.ndarray"] = np.zeros_like(self._state_template)
        else:
            self._state = None
        self._frame_buf = None
        self._last_prob = 0.0
        self._ctx = np.zeros(_SILERO_CONTEXT_SAMPLES, dtype=np.float32)

    # --------------------------------------------------------------- internals

    @staticmethod
    def _import_onnxruntime() -> "ort":
        """Import onnxruntime lazily with an actionable install hint."""
        try:
            import onnxruntime  # local, lazy — heavy dependency
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with: "
                "pip install onnxruntime (Silero VAD runs on CPU)"
            ) from exc
        return onnxruntime

    def _bind_model(self) -> None:
        """Inspect the ONNX signature and lock in a working inference plan.

        Probes candidate feed plans with a silent frame: stateful plans (declared
        state shape, then the known [1,128] / [2,1,128] shapes) are tried before
        a stateless fallback; audio is tried as [1, N] then [N]; the sample-rate
        input dtype is derived from the declared type with generic fallbacks.
        """
        inputs = {inp.name: inp for inp in self._session.get_inputs()}
        output_names = [out.name for out in self._session.get_outputs()]
        if not inputs or not output_names:
            raise RuntimeError("Silero VAD model has no inputs or no outputs")

        audio_name = self._pick_audio_input(inputs)
        sr_name: Optional[str] = next((n for n in inputs if n in _SR_INPUT_NAMES), None)
        state_name: Optional[str] = next((n for n in inputs if n in _STATE_INPUT_NAMES), None)

        self._prob_output_idx = (
            output_names.index(_PROB_OUTPUT_PREFERRED)
            if _PROB_OUTPUT_PREFERRED in output_names
            else 0
        )
        state_idx: Optional[int] = next(
            (i for i, n in enumerate(output_names) if i != self._prob_output_idx and n in _STATE_INPUT_NAMES),
            None,
        )
        if state_idx is None and state_name is not None and len(output_names) > 1:
            state_idx = 1  # v5 emits (prob, state)
        self._state_output_idx = state_idx

        state_plan: list[Optional[tuple[int, ...]]] = []
        if state_name is not None:
            for shape in (self._declared_shape(inputs[state_name]), (1, 128), (2, 1, 128)):
                if shape is not None and shape not in state_plan:
                    state_plan.append(shape)
            state_plan.append(None)  # stateless fallback (last resort)
        else:
            state_plan = [None]  # model has no recurrent state input

        audio_shapes: list[tuple[int, ...]] = [
            (1, self._frame_samples + _SILERO_CONTEXT_SAMPLES),
            (self._frame_samples + _SILERO_CONTEXT_SAMPLES,),
        ]
        sr_candidates = self._sr_value_candidates(inputs.get(sr_name))

        last_error: Optional[Exception] = None
        for state_shape in state_plan:
            for audio_shape in audio_shapes:
                for sr_value in sr_candidates:
                    try:
                        self._try_plan(audio_name, audio_shape, state_name, state_shape, sr_name, sr_value)
                    except Exception as exc:  # onnxruntime raises several types
                        last_error = exc
                        continue
                    # Plan works — lock it in.
                    self._audio_name = audio_name
                    self._audio_shape = audio_shape
                    self._state_name = state_name if state_shape is not None else None
                    self._state_template = (
                        np.zeros(state_shape, dtype=np.float32) if state_shape is not None else None
                    )
                    self._sr_name = sr_name if sr_value is not None else None
                    self._sr_value = sr_value
                    if state_name is not None and state_shape is None:
                        logger.warning(
                            "Silero model declares a state input %r but no stateful call "
                            "worked; using stateless inference (per-frame accuracy may "
                            "degrade)",
                            state_name,
                        )
                    return
        raise RuntimeError(
            "Unsupported Silero VAD model signature (inputs: "
            f"{sorted(inputs)}, outputs: {output_names}); last probe error: {last_error}"
        )

    @staticmethod
    def _pick_audio_input(inputs: dict[str, Any]) -> str:
        """Choose the audio input name (prefer 'input', else the non-state/non-sr one)."""
        if _AUDIO_INPUT_PREFERRED in inputs:
            return _AUDIO_INPUT_PREFERRED
        for name in inputs:
            if name not in _STATE_INPUT_NAMES and name not in _SR_INPUT_NAMES:
                return name
        raise RuntimeError(f"Silero VAD model exposes no audio input (inputs: {sorted(inputs)})")

    @staticmethod
    def _declared_shape(inp: Any) -> Optional[tuple[int, ...]]:
        """Concrete state shape from the ONNX metadata (dynamic dims become 1)."""
        shape = getattr(inp, "shape", None)
        if not shape:
            return None
        dims: list[int] = []
        for dim in shape:
            dims.append(dim if isinstance(dim, int) and dim > 0 else 1)
        return tuple(dims)

    def _sr_value_candidates(self, meta: Any) -> list[Any]:
        """Sample-rate feed values to probe, derived from the declared tensor type."""
        if meta is None:
            return [None]
        type_str = str(getattr(meta, "type", "") or "")
        dtype_map: dict[str, Any] = {
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
        }
        for key, dtype in dtype_map.items():
            if key in type_str:
                return [dtype(self._sample_rate), np.array([self._sample_rate], dtype=dtype)]
        return [
            np.int64(self._sample_rate),
            np.int32(self._sample_rate),
            np.float32(self._sample_rate),
            np.array([self._sample_rate], dtype=np.int64),
        ]

    def _try_plan(
        self,
        audio_name: str,
        audio_shape: tuple[int, ...],
        state_name: Optional[str],
        state_shape: Optional[tuple[int, ...]],
        sr_name: Optional[str],
        sr_value: Any,
    ) -> None:
        """Run one dummy inference to validate a candidate feed (raises on mismatch).

        v0.6.0: probes use the FULL official window (frame + 64-sample
        context) so the locked plan is validated against exactly the shape
        production feeds.
        """
        feed: dict[str, Any] = {
            audio_name: np.zeros(
                self._frame_samples + _SILERO_CONTEXT_SAMPLES, dtype=np.float32
            ).reshape(audio_shape)
        }
        if state_name is not None and state_shape is not None:
            feed[state_name] = np.zeros(state_shape, dtype=np.float32)
        if sr_name is not None and sr_value is not None:
            feed[sr_name] = sr_value
        outputs = self._session.run(None, feed)
        float(np.asarray(outputs[self._prob_output_idx]).reshape(-1)[0])

    def _build_feed(self, x: "np.ndarray") -> dict[str, Any]:
        """Build the inference feed dict for one frame using the locked plan."""
        feed: dict[str, Any] = {self._audio_name: np.ascontiguousarray(x.reshape(self._audio_shape))}
        if self._state_name is not None and self._state is not None:
            feed[self._state_name] = self._state
        if self._sr_name is not None:
            feed[self._sr_name] = self._sr_value
        return feed
