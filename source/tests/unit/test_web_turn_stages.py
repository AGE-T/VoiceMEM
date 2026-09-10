"""v0.4.13: turn pipeline stage events + the emotion-init timeout fix.

Field report this pins (v0.4.12): "the ASR test transcribes real Hungarian,
'Send to the agent' sends the text, the transcript appears — then nothing
reaches the LLM." Root cause: ``_run_turn`` awaited
``asyncio.to_thread(c.emotion_analyzer)`` with NO timeout; the construction
imports funasr (-> torch, 30-180 s cold on Windows, worse under antivirus,
and a concurrent warm-up import serialises behind the same module lock), so
the first turn could freeze between the transcript and the LLM forever with
no error and no diagnostic line.

These tests pin the contract the fix introduces:

1. STAGE EVENTS — every turn writes a complete, named stage trail into the
   session diagnostics ring (turn received / memory start / memory done /
   emotion start / emotion done / llm start / llm first token / llm done /
   tts start / tts done / answer done). A blocked turn shows exactly where
   it stopped by the LAST line of the trail.
2. EMOTION INIT CAN NEVER BLOCK THE TURN — a hanging (or raising) analyzer
   construction is cut off after ``_EMOTION_INIT_TIMEOUT_S``; the turn
   continues with emotion disabled and STILL produces an LLM reply + TTS
   audio.
3. The timeout is a module constant so operators can tune it (and tests can
   patch it).
4. ``EmotionAnalyzer._load`` is serialised by a lock (warm-up + turn race
   used to construct the model twice).

Sandbox-safe: mock components + the scripted demo LLM; timing is controlled
by patching ``app.web_server._EMOTION_INIT_TIMEOUT_S``.
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
import unittest.mock
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from app.config import AgentConfig  # noqa: E402
from app.web_server import WebComponents  # noqa: E402

#: The full ordered stage trail a healthy turn must leave in the ring.
STAGES = (
    "turn received",
    "memory start",
    "emotion start",
    "emotion done",
    "memory done",
    "llm start",
    "llm first token",
    "llm done",
    "tts start",
    "tts done",
    "answer done",
)


class _RecordingSock:
    """Minimal WebSocket stand-in: records every sent JSON/bytes payload."""

    def __init__(self) -> None:
        self.json_events: list[dict] = []
        self.binary_chunks: list[bytes] = []

    async def send_json(self, payload: dict) -> None:
        self.json_events.append(payload)

    async def send_bytes(self, raw: bytes) -> None:
        self.binary_chunks.append(raw)


def _make_session(root: Optional[Path] = None) -> tuple[Any, Any, _RecordingSock]:
    """Build WebComponents + a WebSession with a recording socket."""
    from app.web_server import WebSession

    tmp = root if root is not None else Path(__file__).parent
    cfg = AgentConfig(root=tmp)
    components = WebComponents(cfg, mock=True).build()
    sock = _RecordingSock()
    session = WebSession(sock, components)
    return session, components, sock


def _stage_trail(components: WebComponents) -> list[str]:
    """The '[chain]' diag lines from the events ring (stamps stripped)."""
    out: list[str] = []
    for line in components.status.session.get("events", []):
        out.append(line.split(" ", 1)[1] if " " in line else line)
    return out


async def _drive_turn(session: Any, text: str, audio: Any, source: str) -> None:
    await session._start_turn(text, audio=audio, source=source)
    task = session._turn_task
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=15)


class TestTurnStageTrail(unittest.TestCase):
    """A healthy turn leaves the complete named stage trail."""

    def test_text_turn_emits_full_stage_trail(self):
        session, components, sock = _make_session()

        asyncio.run(
            asyncio.wait_for(
                _drive_turn(session, "Szia, hogy vagy ma?", None, "text"), timeout=25
            )
        )
        trail = " | ".join(_stage_trail(components))
        for stage in STAGES:
            self.assertIn(stage, trail, f"stage {stage!r} missing from trail: {trail}")
        # the turn reached the browser with a REAL answer + audio
        types = [e["type"] for e in sock.json_events]
        self.assertIn("answer_delta", types)
        self.assertIn("answer_done", types)
        self.assertTrue(
            any(e.get("type") == "answer_done" and not e.get("error") for e in sock.json_events),
            "answer_done must carry no error",
        )
        self.assertGreater(len(sock.binary_chunks), 0, "TTS audio bytes missing")

    def test_asr_turn_emits_full_stage_trail(self):
        """The mic path (source='asr', audio present) leaves the same trail."""
        session, components, sock = _make_session()
        sr = components.config.sample_rate
        audio = (np.sin(2 * np.pi * 200.0 * np.arange(sr) / sr) * 0.4).astype(np.float32)

        asyncio.run(
            asyncio.wait_for(
                _drive_turn(session, "Ez egy mikrofon teszt.", audio, "asr"), timeout=25
            )
        )
        trail = " | ".join(_stage_trail(components))
        for stage in STAGES:
            self.assertIn(stage, trail, f"stage {stage!r} missing from trail: {trail}")
        self.assertGreater(len(sock.binary_chunks), 0)

    def test_ring_keeps_at_most_48_events(self):
        """v0.4.13: the ring was 12; v0.4.14: 24 → 48 — the LIVE MIC trail
        (mic frame received → … → ASR turn dispatch → …turn trail… ) is
        ~20 events per turn, so two complete microphone turns survive."""
        session, components, sock = _make_session()
        events = components.status.session.setdefault("events", [])
        for i in range(80):
            events.append(f"00:00:00 filler {i}")
            del events[:-48]
        self.assertLessEqual(len(events), 48)

    def test_diag_lines_go_to_the_logger(self):
        """_diag writes '[chain] <message>' log lines — the backend log shows
        the same trail as the UI ring."""
        import logging

        session, components, sock = _make_session()
        records: list[str] = []
        handler = logging.Handler()

        def emit(record: logging.LogRecord) -> None:
            if record.getMessage().startswith("[chain]"):
                records.append(record.getMessage())

        handler.emit = emit  # type: ignore[method-assign]
        logger = logging.getLogger("app.web_server")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # INFO is the app runtime's level
        try:
            asyncio.run(
                asyncio.wait_for(
                    _drive_turn(session, "teszt log", None, "text"), timeout=25
                )
            )
        finally:
            logger.removeHandler(handler)
        joined = " | ".join(records)
        for stage in ("turn received", "memory start", "llm start", "answer done"):
            self.assertIn(f"[chain] {stage}", joined)


class TestEmotionInitCannotBlock(unittest.TestCase):
    """The fix: a hanging/raising emotion init degrades, never blocks."""

    def _run_turn_with_emotion(self, emotion_behavior: str) -> tuple[WebComponents, _RecordingSock]:
        session, components, sock = _make_session()

        if emotion_behavior == "hang":

            def slow_analyzer() -> Any:
                time.sleep(8.0)  # far beyond the patched 0.3 s budget
                return None

            components.emotion_analyzer = slow_analyzer  # type: ignore[method-assign]
        elif emotion_behavior == "raise":

            def bad_analyzer() -> Any:
                raise RuntimeError("funasr import exploded")

            components.emotion_analyzer = bad_analyzer  # type: ignore[method-assign]

        with unittest.mock.patch("app.web_server._EMOTION_INIT_TIMEOUT_S", 0.3):
            asyncio.run(
                asyncio.wait_for(
                    _drive_turn(session, "Hello, mivan?", None, "text"), timeout=20
                )
            )
        return components, sock

    def test_hanging_init_still_reaches_llm_and_tts(self):
        components, sock = self._run_turn_with_emotion("hang")
        trail = " | ".join(_stage_trail(components))
        # the cut-off is VISIBLE and names the stage
        self.assertIn("emotion done (init timed out", trail)
        # ... and the turn still completed end-to-end
        for stage in (
            "memory done",
            "llm start",
            "llm first token",
            "llm done",
            "tts done",
            "answer done",
        ):
            self.assertIn(stage, trail)
        deltas = [e for e in sock.json_events if e["type"] == "answer_delta"]
        self.assertTrue(deltas, "LLM reply deltas never arrived")
        self.assertGreater(len(sock.binary_chunks), 0, "TTS audio never arrived")

    def test_raising_init_still_reaches_llm(self):
        components, sock = self._run_turn_with_emotion("raise")
        trail = " | ".join(_stage_trail(components))
        self.assertIn("emotion done (init failed", trail)
        for stage in ("memory done", "llm start", "llm done", "answer done"):
            self.assertIn(stage, trail)
        deltas = [e for e in sock.json_events if e["type"] == "answer_delta"]
        self.assertTrue(deltas, "LLM reply deltas never arrived")

    def test_no_timeout_error_event_on_degradation(self):
        """Emotion degradation is NOT an error toast — the conversation just
        continues (the emotion module stays optional per the constraint)."""
        components, sock = self._run_turn_with_emotion("hang")
        errors = [e for e in sock.json_events if e["type"] == "error"]
        self.assertEqual(errors, [], "emotion timeout must not raise a UI error")
        done = [e for e in sock.json_events if e["type"] == "answer_done"]
        self.assertTrue(done)
        self.assertFalse(done[-1].get("error"))


class TestEmotionLoadLock(unittest.TestCase):
    """v0.4.13: EmotionAnalyzer._load is serialised (warm-up + turn race)."""

    def test_concurrent_load_calls_construct_once(self):
        import threading

        from app.emotion import EmotionAnalyzer

        constructed: list[int] = []

        def factory(model_dir: Path) -> Any:
            constructed.append(1)
            time.sleep(0.2)  # simulate the AutoModel construction cost
            return object()

        analyzer = EmotionAnalyzer(Path("/tmp/fake"), model_factory=factory)
        results: list[Optional[Any]] = [None, None]

        def worker(i: int) -> None:
            results[i] = analyzer._load()

        t0 = threading.Thread(target=worker, args=(0,))
        t1 = threading.Thread(target=worker, args=(1,))
        t0.start()
        t1.start()
        t0.join(timeout=5)
        t1.join(timeout=5)

        self.assertEqual(len(constructed), 1, "AutoModel constructed twice")
        self.assertIs(results[0], results[1], "concurrent caller did not reuse the model")

    def test_failed_load_remembered_under_lock(self):
        from app.emotion import EmotionAnalyzer

        def factory(model_dir: Path) -> Any:
            raise RuntimeError("boom")

        analyzer = EmotionAnalyzer(Path("/tmp/fake"), model_factory=factory)
        self.assertIsNone(analyzer._load())
        self.assertIsNone(analyzer._load())  # one attempt only


class TestBlockedTurnTrailShape(unittest.TestCase):
    """The contract from the field report: a blocked turn's trail must name
    the blocked stage. With the v0.4.12 bug this test would have hung
    forever (the emotion init never returned); with the fix the trail shows
    'emotion start' followed by the timed-out 'emotion done' and the turn
    completes."""

    def test_blocked_emotion_trail_is_diagnosable(self):
        session, components, sock = _make_session()

        def never_returns() -> Any:
            time.sleep(30.0)
            return None

        components.emotion_analyzer = never_returns  # type: ignore[method-assign]

        with unittest.mock.patch("app.web_server._EMOTION_INIT_TIMEOUT_S", 0.2):
            asyncio.run(
                asyncio.wait_for(
                    _drive_turn(session, "teszt", None, "text"), timeout=15
                )
            )
        trail = _stage_trail(components)
        # the last "emotion ..." line is the timed-out done — NOT a bare
        # "emotion start" with nothing after (the v0.4.12 failure shape)
        emo_lines = [l for l in trail if l.startswith("emotion ")]
        self.assertTrue(emo_lines, "no emotion stage lines at all")
        self.assertTrue(emo_lines[-1].startswith("emotion done"), emo_lines[-1])
        # and the turn still produced a spoken answer
        self.assertGreater(len(sock.binary_chunks), 0)


if __name__ == "__main__":
    unittest.main()
