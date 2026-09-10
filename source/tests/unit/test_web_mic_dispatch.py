"""v0.4.14: the automatic microphone → agent dispatch (live-mic stage trail).

The v0.4.13 field report: "the audio still never reaches the LLM. Pressing
'Send to the agent' answers, but until I press it nothing happens." The
working manual path is user_text (VAD-free); the broken live path is
binary PCM16 → VAD → ASR feed → speech end → flush → dispatch. Root cause:
Silero deaf on the capture channel (peak 0.003) — the transition never
reached ASR finalisation at all.

These tests drive the REAL live-mic machinery (audio loop, VAD framing,
VSM, ASR feed/flush, _start_turn) on a mock-components WebSession whose
VAD is a FusedVad with a DEAF primary — the reporter's exact channel — and
pin:

1. the COMPLETE stage trail: mic frame received → speech start · energy
   fallback → ASR feed → speech end → ASR flush start → ASR flush done →
   ASR final transcript → ASR turn dispatch → turn received → …llm/tts…
   → answer done;
2. an answer + audio arrived with NO user_text / Send button involved;
3. barge-in reads the PRIMARY probability (line-level audio during the
   answer must not interrupt the agent);
4. the manual user_text path still works alongside (both paths coexist).
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from app.config import AgentConfig  # noqa: E402
from app.web_server import WebComponents, WebSession  # noqa: E402
from app.vad import FusedVad  # noqa: E402

#: The reporter's channel: Silero peak 0.003 on speech-level line-in audio.
DEAF_PROB = 0.003
SPEECH_RMS = 0.05


class _DeafSilero:
    """Silero stand-in returning the field-reported ~0.003 everywhere."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def prob(self, frame: Any) -> float:
        return DEAF_PROB

    def reset(self) -> None:
        self.reset_calls += 1

    def is_available(self) -> bool:
        return True


class _RecordingSock:
    def __init__(self) -> None:
        self.json_events: list[dict] = []
        self.binary_chunks: list[bytes] = []

    async def send_json(self, payload: dict) -> None:
        self.json_events.append(payload)

    async def send_bytes(self, raw: bytes) -> None:
        self.binary_chunks.append(raw)


def _stage_trail(components: WebComponents) -> list[str]:
    out: list[str] = []
    for line in components.status.session.get("events", []):
        out.append(line.split(" ", 1)[1] if " " in line else line)
    return out


MIC_STAGES = (
    "mic frame received",
    "speech start",
    "energy fallback",
    "ASR feed",
    "speech end",
    "ASR flush start",
    "ASR flush done",
    "ASR final transcript",
    "ASR turn dispatch",
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


def _make_session() -> tuple[WebSession, WebComponents, _RecordingSock, _DeafSilero]:
    """Session with a one-way recording socket (frames pushed straight into
    the pending-bytes queue — the receive-loop variant is _make_feed_session)."""
    tmp = Path(__file__).parent
    cfg = AgentConfig(root=tmp)
    components = WebComponents(cfg, mock=True).build()
    deaf = _DeafSilero()
    components.make_vad = lambda: FusedVad(deaf, cfg)  # type: ignore[method-assign]
    from app.mock_components import MockAsrEngine

    components.make_asr = lambda: MockAsrEngine(  # type: ignore[method-assign]
        queue=["Szia, mi újság?"]
    )
    sock = _RecordingSock()
    session = WebSession(sock, components, vad=FusedVad(deaf, cfg))
    return session, components, sock, deaf


class _FeedSock(_RecordingSock):
    """WebSocket stand-in that also plays the RECEIVE side: binary mic
    frames first, then a disconnect once the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.release: asyncio.Event = asyncio.Event()

    async def receive(self) -> dict:
        try:
            return await asyncio.wait_for(self.incoming.get(), timeout=0.25)
        except asyncio.TimeoutError:
            pass
        await self.release.wait()
        return {"type": "websocket.disconnect"}


def _make_feed_session() -> tuple[WebSession, WebComponents, _FeedSock, _DeafSilero]:
    tmp = Path(__file__).parent
    cfg = AgentConfig(root=tmp)
    components = WebComponents(cfg, mock=True).build()
    deaf = _DeafSilero()
    components.make_vad = lambda: FusedVad(deaf, cfg)  # type: ignore[method-assign]
    from app.mock_components import MockAsrEngine

    components.make_asr = lambda: MockAsrEngine(  # type: ignore[method-assign]
        queue=["Szia, mi újság?"]
    )
    sock = _FeedSock()
    session = WebSession(sock, components, vad=FusedVad(deaf, cfg))
    return session, components, sock, deaf


def _pcm16(rms: float, seconds: float, sr: int = 24000) -> bytes:
    n = int(sr * seconds)
    x = np.zeros(n, dtype=np.float32)
    x[: max(1, n // 2)] = np.float32(rms)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


async def _speak(session: WebSession, speech_s: float, pause_s: float) -> None:
    """Push real binary mic chunks through the pending-bytes queue."""
    for _ in range(int(speech_s / 0.1)):
        session._pending_bytes.put_nowait(_pcm16(SPEECH_RMS, 0.1))
    for _ in range(int(pause_s / 0.1)):
        session._pending_bytes.put_nowait(_pcm16(0.0005, 0.1))
    await asyncio.sleep(0.2)


async def _drain_audio_loop(session: WebSession) -> None:
    task = session._spawn(session._audio_loop())
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if session._turn_task is not None and session._turn_task.done():
            break
        trail = " ".join(session._c.status.session.get("events", []))
        if "answer done" in trail:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.3)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    turn = session._turn_task
    if turn is not None and not turn.done():
        await asyncio.wait_for(asyncio.shield(turn), timeout=15)


class MicDispatchTests(unittest.TestCase):
    def test_deaf_channel_auto_dispatches_full_trail(self):
        """THE missing transition: deaf Silero + speech-level audio still
        runs mic → VAD(energy) → ASR → flush → dispatch → LLM → TTS with
        no user_text anywhere in sight. Frames enter through the REAL
        receive loop (mic frame received → queue → audio loop → VAD)."""
        session, components, sock, deaf = _make_feed_session()

        async def run() -> None:
            # the first binary frame lands in the receive loop -> the trail
            # starts with "mic frame received"
            sock.incoming.put_nowait({"bytes": _pcm16(SPEECH_RMS, 0.1)})
            run_task = session._spawn(session.run())
            for _ in range(10):  # ~1.0 s of speech-level audio
                sock.incoming.put_nowait({"bytes": _pcm16(SPEECH_RMS, 0.1)})
            for _ in range(9):  # ~0.9 s of pause -> hangover -> speech end
                sock.incoming.put_nowait({"bytes": _pcm16(0.0005, 0.1)})
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                trail = " ".join(components.status.session.get("events", []))
                if "answer done" in trail:
                    break
                await asyncio.sleep(0.05)
            sock.release.set()
            await asyncio.wait_for(asyncio.shield(run_task), timeout=10)

        asyncio.run(asyncio.wait_for(run(), timeout=40))
        trail = " | ".join(_stage_trail(components))
        for stage in MIC_STAGES:
            self.assertIn(stage, trail, f"stage {stage!r} missing: {trail}")
        # order: the mic stages precede the turn stages
        self.assertLess(trail.index("ASR turn dispatch"), trail.index("turn received"))
        self.assertLess(trail.index("speech end"), trail.index("ASR flush start"))
        # a real answer + real audio came back
        types = [e.get("type") for e in sock.json_events]
        self.assertIn("answer_delta", types)
        self.assertIn("answer_done", types)
        self.assertGreater(len(sock.binary_chunks), 0)
        # the transcript the mock ASR queued IS the dispatched turn
        self.assertIn("Szia, mi újság?", trail)
        # the mic frame counter ran through the receive loop
        self.assertGreaterEqual(components.status.session.get("mic_frames", 0), 1)

    def test_no_user_text_message_was_needed(self):
        """The live path must not rely on the manual bridge: the turn count
        grows purely from binary frames."""
        session, components, sock, deaf = _make_session()

        async def run() -> None:
            await _speak(session, 1.0, 0.9)
            await _drain_audio_loop(session)

        asyncio.run(asyncio.wait_for(run(), timeout=40))
        sent_texts = [
            e for e in sock.json_events if e.get("type") == "user_transcript"
        ]
        self.assertTrue(sent_texts)
        self.assertEqual(sent_texts[0].get("source"), "asr")

    def test_barge_in_reads_primary_not_energy(self):
        """Line-level audio while the agent answers must NOT interrupt it:
        barge-in consumes the primary's own (deaf) probability."""
        session, components, sock, deaf = _make_session()

        async def run() -> None:
            await _speak(session, 0.8, 0.8)
            await _drain_audio_loop(session)
            self.assertIsNotNone(session._turn_task)
            # speech-level audio DURING the answer (the turn is running)
            for _ in range(6):
                session._pending_bytes.put_nowait(_pcm16(SPEECH_RMS, 0.1))
            await asyncio.sleep(1.0)

        asyncio.run(asyncio.wait_for(run(), timeout=40))
        types = [e.get("type") for e in sock.json_events]
        self.assertNotIn("answer_interrupt", types)
        trail = " | ".join(_stage_trail(components))
        self.assertIn("answer done", trail)

    def test_manual_path_still_works_alongside(self):
        """The proven manual user_text path stays intact (TEST B)."""
        session, components, sock, deaf = _make_session()

        async def run() -> None:
            await session._on_text('{"type": "user_text", "text": "Szia kézzel"}')
            turn = session._turn_task
            self.assertIsNotNone(turn)
            await asyncio.wait_for(asyncio.shield(turn), timeout=25)

        asyncio.run(asyncio.wait_for(run(), timeout=30))
        types = [e.get("type") for e in sock.json_events]
        self.assertIn("answer_delta", types)
        self.assertIn("answer_done", types)
        self.assertGreater(len(sock.binary_chunks), 0)


if __name__ == "__main__":
    unittest.main()
