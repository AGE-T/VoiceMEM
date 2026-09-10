"""Unit tests for the pure VAD state machine (app.vad.VadStateMachine).

Numbers used throughout: threshold 0.5, hangover 300 ms, frame 32 ms —
the hangover is ceil(300/32) = 10 consecutive below-threshold frames.
Also covers the SileroVad constructor failure paths (no onnxruntime / no model
file in this sandbox — both must raise RuntimeError with a clear hint).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import unittest

from app.config import AgentConfig
from app.vad import SileroVad, VadEvent, VadStateMachine


def make_vad() -> VadStateMachine:
    """Fresh state machine with the default M1 VAD parameters."""
    return VadStateMachine(threshold=0.5, hangover_ms=300, frame_ms=32)


class VadStateMachineTests(unittest.TestCase):
    """Frame-level state machine behaviour."""

    def test_speech_start_on_first_above_threshold_frame(self) -> None:
        """SPEECH_START fires on the FIRST above-threshold frame, not later."""
        vad = make_vad()
        self.assertFalse(vad.in_speech)
        self.assertIs(vad.update(0.10), VadEvent.NONE)
        self.assertIs(vad.update(0.49), VadEvent.NONE)
        self.assertIs(vad.update(0.60), VadEvent.SPEECH_START)
        self.assertTrue(vad.in_speech)
        # subsequent above-threshold frames do not re-fire
        self.assertIs(vad.update(0.70), VadEvent.NONE)
        self.assertTrue(vad.in_speech)

    def test_no_speech_end_until_hangover_elapsed(self) -> None:
        """9 below-threshold frames (288 ms < 300 ms) keep the state in speech."""
        vad = make_vad()
        vad.update(0.9)
        for i in range(9):
            self.assertIs(vad.update(0.1), VadEvent.NONE, f"silent frame {i}")
        self.assertTrue(vad.in_speech)

    def test_speech_end_exactly_after_hangover_frames(self) -> None:
        """SPEECH_END fires on the 10th consecutive below-threshold frame."""
        vad = make_vad()
        vad.update(0.9)
        for _ in range(9):
            self.assertIs(vad.update(0.1), VadEvent.NONE)
        self.assertIs(vad.update(0.1), VadEvent.SPEECH_END)
        self.assertFalse(vad.in_speech)
        # further silence produces no further events
        self.assertIs(vad.update(0.1), VadEvent.NONE)

    def test_repeated_events_fire_only_once(self) -> None:
        """One SPEECH_START per utterance and one SPEECH_END per silence stretch."""
        vad = make_vad()
        events = [vad.update(0.9) for _ in range(20)]
        self.assertIs(events[0], VadEvent.SPEECH_START)
        self.assertTrue(all(e is VadEvent.NONE for e in events[1:]))
        ends = [vad.update(0.1) for _ in range(30)]
        self.assertEqual(ends.count(VadEvent.SPEECH_END), 1)
        # a new utterance fires SPEECH_START again afterwards
        self.assertIs(vad.update(0.9), VadEvent.SPEECH_START)
        self.assertTrue(vad.in_speech)

    def test_reset_works(self) -> None:
        """reset() returns to the initial state with no stale events."""
        vad = make_vad()
        vad.update(0.9)
        vad.reset()
        self.assertFalse(vad.in_speech)
        # no stale SPEECH_END after reset
        self.assertIs(vad.update(0.1), VadEvent.NONE)
        self.assertFalse(vad.in_speech)
        # speech restarts cleanly after reset
        self.assertIs(vad.update(0.9), VadEvent.SPEECH_START)

    def test_reset_during_hangover_prevents_speech_end(self) -> None:
        """reset() mid-hangover must not produce a delayed SPEECH_END."""
        vad = make_vad()
        vad.update(0.9)
        for _ in range(5):
            vad.update(0.1)
        vad.reset()
        self.assertIs(vad.update(0.1), VadEvent.NONE)

    def test_threshold_is_inclusive(self) -> None:
        """prob == threshold counts as speech (>= comparison)."""
        vad = make_vad()
        self.assertIs(vad.update(0.5), VadEvent.SPEECH_START)
        vad2 = make_vad()
        self.assertIs(vad2.update(0.4999), VadEvent.NONE)
        # inclusive on the hangover side too: exactly 0.5 keeps speech alive
        self.assertIs(vad.update(0.5), VadEvent.NONE)

    def test_hangover_counter_resets_on_speech(self) -> None:
        """An above-threshold frame re-arms the hangover counter."""
        vad = make_vad()
        vad.update(0.9)
        for _ in range(9):
            vad.update(0.1)
        vad.update(0.8)  # speech resumes: hangover countdown restarts
        for _ in range(9):
            self.assertIs(vad.update(0.1), VadEvent.NONE)
        self.assertTrue(vad.in_speech)
        self.assertIs(vad.update(0.1), VadEvent.SPEECH_END)  # 10th consecutive silent frame


class SileroVadConstructorTests(unittest.TestCase):
    """SileroVad must fail loudly (RuntimeError + hint) without onnxruntime/model."""

    def test_missing_runtime_or_model_raises_runtime_error(self) -> None:
        """No onnxruntime and no model file in the sandbox -> RuntimeError with hint."""
        cfg = AgentConfig()  # root points at <repo>/runtime (does not exist here)
        with self.assertRaises(RuntimeError) as ctx:
            SileroVad(cfg)
        message = str(ctx.exception).lower()
        # Either the onnxruntime install hint or the missing-model path hint.
        self.assertTrue("onnxruntime" in message or "silero" in message, message)


if __name__ == "__main__":
    unittest.main()
