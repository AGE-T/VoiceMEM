"""v0.9.1 regression pin: the web path must CAPTURE the SPEECH_START trigger
frame — CLI parity for speech onset.

Pre-v0.9.1 ``app/web_server.py``'s ``speech_start`` branch returned early,
so the 32 ms frame that crossed the threshold was dropped from the capture;
the CLI path (``app/main.py``) appends it because ``state.in_speech`` is
already true when SPEECH_START is returned. On sibilant onsets ("Szia")
Silero needs ~86 ms to cross the threshold, and losing the trigger frame on
top of that was measured to turn "Szia" into "Fia" (v0.9.1 forensic
evidence: /home/z/vmforensic/case_szia_float32*.json — pre-fix capture
started one frame after speech_start, post-fix it starts ON it and
"Szia, hogyan vagy!" survives the cut).

The test drives the REAL ``WebSession._on_vad_frame`` with a scripted VAD
and a real ``VadStateMachine``, then compares the captured utterance
frame-by-frame against a replica of the CLI collection loop fed the same
probabilities. Sandbox-safe: no torch/onnx/voicemem.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from app.config import AgentConfig  # noqa: E402
from app.vad import VadStateMachine  # noqa: E402
from app.web_server import WebComponents, WebSession  # noqa: E402


class _FakeSock:
    """Records everything a WebSession would send over the WebSocket."""

    def __init__(self) -> None:
        self.json: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.json.append(payload)

    async def send_bytes(self, raw: bytes) -> None:
        self.json.append({"type": "__binary__", "bytes": len(raw)})


class _ScriptedVad:
    """``prob(frame)`` returns one scripted value per call, content-blind."""

    def __init__(self, probs: list[float]) -> None:
        self.probs = list(probs)
        self.calls = 0

    def prob(self, frame: object) -> float:
        value = self.probs[self.calls] if self.calls < len(self.probs) else 0.0
        self.calls += 1
        return float(value)

    def reset(self) -> None:  # pragma: no cover - stateless stub
        pass


def _make_frames(n: int) -> list[np.ndarray]:
    """Distinct 512-sample frames: frame i is filled with the value i+1."""
    return [np.full(512, float(i + 1), dtype=np.float32) for i in range(n)]


# 4 silence + 6 speech + 14 silence frames: speech_start at frame 4, the
# 10th consecutive below-threshold frame lands on frame 19 -> speech_end.
_PROBS = [0.1] * 4 + [0.6] * 6 + [0.1] * 14
_START, _END = 4, 19  # SPEECH_START frame, SPEECH_END frame (0-based)


class TestTriggerFrameRetention(unittest.TestCase):
    """The trigger frame is the FIRST captured frame (web = CLI + 1 tail)."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="vm_vadtrig_test_")

    def _build_session(self) -> tuple[WebSession, list]:
        from app.mock_components import MockAsrEngine

        cfg = AgentConfig(root=Path(self.tmp))
        cfg.enable_speaker = False
        comp = WebComponents(cfg, mock=True).build()
        comp.make_asr = lambda: MockAsrEngine(queue=[""])  # type: ignore[method-assign]
        session = WebSession(_FakeSock(), comp, vad=_ScriptedVad(_PROBS))
        # a fresh REAL state machine (production thresholds from the config)
        session._vsm = VadStateMachine(
            cfg.vad_threshold, cfg.vad_hangover_ms, cfg.vad_frame_ms
        )
        captured: list = []
        orig_transcribe = session._asr.transcribe

        def rec_transcribe(audio):
            captured.append(np.asarray(audio.samples, dtype=np.float32))
            return orig_transcribe(audio)

        session._asr.transcribe = rec_transcribe  # type: ignore[method-assign]
        return session, captured

    def test_trigger_frame_is_first_captured_frame(self) -> None:
        session, captured = self._build_session()
        frames = _make_frames(len(_PROBS))

        async def scenario() -> None:
            for f in frames:
                await session._on_vad_frame(f)

        asyncio.run(scenario())
        self.assertEqual(len(captured), 1)
        audio = captured[0]
        # frames _START.._END inclusive -> the TRIGGER frame leads the capture
        expected = np.concatenate(frames[_START : _END + 1])
        self.assertEqual(audio.size, expected.size)
        np.testing.assert_array_equal(audio, expected)
        # the first 512 samples ARE the trigger frame's content
        np.testing.assert_array_equal(audio[:512], frames[_START])
        # session counters agree (all captured frames counted as speech)
        self.assertEqual(session._c.status.session["speech_ms"], 0)

    def test_cli_collection_parity(self) -> None:
        """Replica of the app/main.py loop on the same probabilities must
        produce the SAME audio, modulo the one extra web TAIL frame (the
        speech_end frame — web appends it because its session flag is
        cleared only after the capture block; documented, intentional
        asymmetry, harmless for ASR: trailing silence)."""
        session, captured = self._build_session()
        frames = _make_frames(len(_PROBS))

        async def scenario() -> None:
            for f in frames:
                await session._on_vad_frame(f)

        asyncio.run(scenario())
        web_audio = captured[0]

        # --- CLI replica (app/main.py: event = state.update(prob);
        # if state.in_speech: utterance.append(frame)) ----------------------
        cfg = AgentConfig(root=Path(self.tmp))
        cli_vsm = VadStateMachine(
            cfg.vad_threshold, cfg.vad_hangover_ms, cfg.vad_frame_ms
        )
        cli_utterance: list[np.ndarray] = []
        for i, f in enumerate(frames):
            cli_vsm.update(_PROBS[i])
            if cli_vsm.in_speech:
                cli_utterance.append(f)
        cli_audio = np.concatenate(cli_utterance)

        # same audio, web keeps exactly ONE extra tail frame (speech_end)
        self.assertEqual(cli_audio.size, (_END - _START) * 512)
        self.assertEqual(web_audio.size, cli_audio.size + 512)
        np.testing.assert_array_equal(web_audio[: cli_audio.size], cli_audio)

    def test_single_utterance_stays_one_turn(self) -> None:
        """One continuous sentence -> exactly one utterance/turn boundary
        (the fix adds 32 ms of audio, it must not split or duplicate)."""
        session, captured = self._build_session()
        frames = _make_frames(len(_PROBS))

        async def scenario() -> None:
            for f in frames:
                await session._on_vad_frame(f)

        asyncio.run(scenario())
        self.assertEqual(len(captured), 1)
        s = session._c.status.session
        self.assertEqual(s["utterances"], 1)
        self.assertFalse(s["vad_in_speech"])
        self.assertIsNone(session._utterance)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
