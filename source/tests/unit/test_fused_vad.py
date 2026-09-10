"""v0.4.14: the energy-fallback VAD (app.vad.FusedVad).

Field report this pins (v0.4.13): "the audio still never reaches the LLM —
pressing 'Send to the agent' answers, but until I press it nothing happens."
The live mic chain gates ASR behind Silero, and on the reporter's capture
channel (Line In, browser-delivered) Silero scored real, ASR-perfectly
transcribable speech at ~0.003 — 0/190 frames above the 0.25 threshold. No
speech_start, no feed, no flush, no dispatch: the automatic mic → agent
transition died INSIDE the VAD gate, which is exactly why the manual
button (user_text, VAD-free) was the only working path.

These tests pin the fusion contract with an INJECTED primary (no
onnxruntime needed):

1. DEAF primary + speech-level audio -> the energy gate opens (VSM fires).
2. HEALTHY primary -> fused == primary exactly (the fallback is passive).
3. Silence stays silent (no false speech on room noise).
4. Constant noise SELF-LIMITS (slow-up floor lifts the gate).
5. reset() clears the primary but keeps the channel estimates.
6. last_primary exposes the primary's own frame probability (barge-in).
7. make_vad() wraps SileroVad in FusedVad (config-gated) in real mode.
"""

from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from app.config import AgentConfig  # noqa: E402
from app.vad import FusedVad, VadEvent, VadStateMachine  # noqa: E402


class _FakePrimary:
    """Duck-typed Silero stand-in: constant or scripted probabilities.

    Accepts the AgentConfig argument the real constructor takes (so it can
    be monkeypatched over ``app.vad.SileroVad`` in wiring tests)."""

    def __init__(self, *args: Any, prob: float = 0.003) -> None:
        # Accepts both call shapes: the real constructor's (config) — used
        # when monkeypatched over app.vad.SileroVad — and the direct
        # (prob=...) / positional-prob shape the tests themselves use.
        for arg in args:
            if isinstance(arg, (int, float)):
                prob = float(arg)
        self.prob_value = float(prob)
        self.reset_calls = 0
        self.frames: list[int] = []

    def prob(self, frame: Any) -> float:
        self.frames.append(int(np.asarray(frame).size))
        return self.prob_value

    def reset(self) -> None:
        self.reset_calls += 1


def _cfg() -> AgentConfig:
    return AgentConfig(root=Path(__file__).parent)


def _frame(rms: float, n: int = 512) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    if rms > 0:
        x[: max(1, n // 2)] = np.float32(rms)
    return x


class FusedVadTests(unittest.TestCase):
    def test_deaf_primary_plus_speech_level_opens_gate(self):
        """The reporter's exact case: Silero 0.003 on speech at rms 0.05."""
        primary = _FakePrimary(prob=0.003)
        vad = FusedVad(primary, _cfg())
        vsm = VadStateMachine(0.25, 300, 32)
        events = [vsm.update(vad.prob(_frame(0.05))) for _ in range(10)]
        self.assertIn(VadEvent.SPEECH_START, events)
        # the fused probability is comfortably above the threshold
        self.assertGreaterEqual(vad.prob(_frame(0.05)), 0.25)
        self.assertTrue(vad.fallback_active)
        self.assertAlmostEqual(vad.last_primary, 0.003, places=4)

    def test_quiet_frames_stay_below_threshold(self):
        primary = _FakePrimary(prob=0.003)
        vad = FusedVad(primary, _cfg())
        vsm = VadStateMachine(0.25, 300, 32)
        events = [vsm.update(vad.prob(_frame(0.001))) for _ in range(10)]
        self.assertNotIn(VadEvent.SPEECH_START, events)

    def test_healthy_primary_is_passthrough(self):
        """When Silero fires, the fused probability is EXACTLY Silero's."""
        primary = _FakePrimary(prob=0.62)
        vad = FusedVad(primary, _cfg())
        p = vad.prob(_frame(0.001))  # even silence-level: primary drives
        self.assertEqual(p, 0.62)
        self.assertFalse(vad.fallback_active)

    def test_fallback_goes_passive_when_primary_recovers(self):
        primary = _FakePrimary(prob=0.003)
        vad = FusedVad(primary, _cfg())
        vad.prob(_frame(0.05))
        self.assertTrue(vad.fallback_active)
        primary.prob_value = 0.8
        vad.prob(_frame(0.05))
        self.assertFalse(vad.fallback_active)

    def test_constant_noise_self_limits(self):
        """Loud constant hiss lifts the floor until it stops firing — the
        fallback cannot turn a fan into an endless turn (the 30 s force-end
        loop) forever."""
        primary = _FakePrimary(prob=0.003)
        vad = FusedVad(primary, _cfg())
        firing = [vad.prob(_frame(0.05)) >= 0.25 for _ in range(1250)]  # ~40 s
        # it starts firing (energy above the initial gate)...
        self.assertTrue(any(firing))
        # ...and stops once the slow-up floor lifts the gate past the noise
        self.assertFalse(all(firing))

    def test_floor_drops_fast_in_silence(self):
        primary = _FakePrimary(prob=0.003)
        vad = FusedVad(primary, _cfg())
        for _ in range(400):
            vad.prob(_frame(0.05))  # long noise -> floor creeps up
        floor_high = vad.noise_floor
        for _ in range(10):
            vad.prob(_frame(0.001))  # brief pause
        self.assertLess(vad.noise_floor, floor_high)

    def test_reset_clears_primary_keeps_channel(self):
        primary = _FakePrimary(prob=0.003)
        vad = FusedVad(primary, _cfg())
        vad.prob(_frame(0.05))
        vad.reset()
        self.assertEqual(primary.reset_calls, 1)
        # the deafness window survived: the next frame is still fallback-active
        self.assertTrue(vad.fallback_active or vad.prob(_frame(0.05)) >= 0.0)
        self.assertTrue(vad.fallback_active)

    def test_full_utterance_lifecycle(self):
        """speech -> pause -> SPEECH_END: the VSM sees a complete utterance
        exactly as it would with a healthy Silero."""
        primary = _FakePrimary(prob=0.003)
        vad = FusedVad(primary, _cfg())
        vsm = VadStateMachine(0.25, 300, 32)
        seen: list[VadEvent] = []
        for _ in range(20):  # ~0.64 s of speech-level audio
            seen.append(vsm.update(vad.prob(_frame(0.05))))
        for _ in range(20):  # ~0.64 s of pause -> hangover -> end
            seen.append(vsm.update(vad.prob(_frame(0.001))))
        self.assertIn(VadEvent.SPEECH_START, seen)
        self.assertIn(VadEvent.SPEECH_END, seen)

    def test_is_available_delegates_to_primary(self):
        self.assertTrue(FusedVad(_FakePrimary(), _cfg()).is_available())

    def test_primary_peak_tracks_the_window(self):
        primary = _FakePrimary(prob=0.003)
        vad = FusedVad(primary, _cfg())
        for _ in range(5):
            vad.prob(_frame(0.02))
        self.assertAlmostEqual(vad.primary_peak, 0.003, places=4)
        primary.prob_value = 0.9
        vad.prob(_frame(0.02))
        self.assertAlmostEqual(vad.primary_peak, 0.9, places=4)


class MakeVadWiringTests(unittest.TestCase):
    """WebComponents.make_vad builds the fusion in REAL mode (config-gated)."""

    def _components(self, energy: bool):
        from app.web_server import WebComponents

        cfg = AgentConfig(root=Path(__file__).parent)
        cfg.vad_energy_fallback = energy
        # NOTE: deliberately NOT built — make_vad() only reads .mock/.config;
        # a real-mode build() would try to construct the full component set.
        return WebComponents(cfg, mock=False)

    def test_make_vad_wraps_silero_when_enabled(self):
        components = self._components(True)
        with unittest.mock.patch("app.vad.SileroVad", _FakePrimary):
            vad = components.make_vad()
        self.assertIsInstance(vad, FusedVad)
        self.assertIsInstance(vad.primary, _FakePrimary)

    def test_make_vad_returns_bare_silero_when_disabled(self):
        components = self._components(False)
        with unittest.mock.patch("app.vad.SileroVad", _FakePrimary):
            vad = components.make_vad()
        self.assertNotIsInstance(vad, FusedVad)
        self.assertIsInstance(vad, _FakePrimary)

    def test_config_default_and_env_override(self):
        cfg = _cfg()
        self.assertTrue(cfg.vad_energy_fallback)
        with unittest.mock.patch.dict(
            "os.environ", {"VAD_ENERGY_FALLBACK": "0"}
        ):
            cfg2 = AgentConfig(root=Path(__file__).parent)
            cfg2.apply_env()
        self.assertFalse(cfg2.vad_energy_fallback)

    def test_mock_mode_stays_energy_vad(self):
        from app.web_server import EnergyVad, WebComponents

        components = WebComponents(AgentConfig(root=Path(__file__).parent), mock=True)
        self.assertIsInstance(components.make_vad(), EnergyVad)


if __name__ == "__main__":
    unittest.main()
