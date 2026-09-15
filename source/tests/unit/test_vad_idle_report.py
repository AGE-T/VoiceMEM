"""v0.10.1 (P0 ASR forensic) — the idle-VAD report fix + dump policy tests.

Pinned behaviours:

1. THE FALSE "peak 0.97 < 0.25" LINE IS GONE. The v0.10.0 field log showed
   "VAD never reached the speech threshold: peak 0.97 < 0.25" immediately
   before "speech start" — a logging-only defect proven by code order: the
   rolling report ran BEFORE the state-machine update, so the window could
   close on the very frame that starts speech (its probability already
   accumulated into the peak while ``in_speech`` was still False), and the
   message text asserted "peak < threshold" without ever testing it.
   After the v0.10.1 fix the report runs BELOW the update: the trigger
   frame is excluded, every accumulated frame is strictly below the
   threshold, and the message carries an explicit condition.

2. A genuinely-idle window still reports (v0.4.7/v0.4.8 contract kept).

3. ``asr_dump_utterances`` defaults to False; Cyrillic detection (the
   always-dump trigger) is exact.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from app.config import AgentConfig  # noqa: E402
from app.web_server import WebComponents, _has_cyrillic  # noqa: E402

REPORT_FRAMES = 150  # 150 x 32 ms ~ 5 s report window


class _ScriptedVad:
    """Fake Silero wrapper returning scripted probabilities, in frame order."""

    def __init__(self, probs: list[float]) -> None:
        self._probes = list(probs)
        self.calls = 0

    def prob(self, frame) -> float:  # noqa: ANN001 - numpy array
        self.calls += 1
        return self._probes.pop(0) if self._probes else 0.0

    def reset(self) -> None:  # the speech-end path calls it best-effort
        pass


def _make_session():
    from app.web_server import WebSession

    tmp = Path(sys.prefix)  # any existing root; the ASR path is never started
    cfg = AgentConfig(root=tmp)
    cfg.asr_dump_utterances = False
    components = WebComponents(cfg, mock=True).build()

    class _Sock:
        async def send_json(self, payload: dict) -> None:
            pass

        async def send_bytes(self, raw: bytes) -> None:
            pass

    session = WebSession(_Sock(), components)
    return session, components


def _frames(n: int) -> list[np.ndarray]:
    return [np.zeros(512, dtype=np.float32) for _ in range(n)]


class _IdleReportBase(unittest.TestCase):
    """Drives N frames through _on_vad_frame with a scripted VAD."""

    def _run(self, probs: list[float]) -> None:
        session, _ = _make_session()
        session._vad = _ScriptedVad(probs)
        with unittest.mock.patch("app.web_server.logger") as log:
            asyncio.run(self._drive(session, len(probs)))
            self._log = log
        self._session = session

    @staticmethod
    async def _drive(session, n: int) -> None:
        for frame in _frames(n):
            await session._on_vad_frame(frame)

    def _info_lines(self) -> list[str]:
        lines: list[str] = []
        for call in self._log.info.call_args_list:
            lines.append(call.args[0] % call.args[1:] if len(call.args) > 1
                         else call.args[0])
        return lines

    def _warn_lines(self) -> list[str]:
        lines: list[str] = []
        for call in self._log.warning.call_args_list:
            lines.append(call.args[0] % call.args[1:] if len(call.args) > 1
                         else call.args[0])
        return lines


class IdleVadReportTests(_IdleReportBase):
    """The v0.10.1 fix: the report may never claim a false peak<threshold."""

    def test_trigger_frame_closing_the_window_is_not_reported_below(self) -> None:
        """149 quiet frames + the speech-start trigger frame as #150.

        OLD (buggy) behaviour: the window closed ON the trigger frame and
        printed "VAD never reached the speech threshold: peak 0.97 < 0.25"
        right before speech started — the exact v0.10.0 field log.
        """
        probs = [0.10] * (REPORT_FRAMES - 1) + [0.97]
        self._run(probs)
        never = [ln for ln in self._info_lines()
                 if "never reached the speech threshold" in ln]
        self.assertEqual(
            never, [],
            f"false below-threshold report emitted: {never}",
        )
        self.assertTrue(self._session._in_speech,
                        "the trigger frame must still START speech")

    def test_genuinely_idle_window_still_reports(self) -> None:
        """150 quiet frames below the threshold DO report (v0.4.7 pact)."""
        self._run([0.10] * REPORT_FRAMES)
        never = [ln for ln in self._info_lines()
                 if "never reached the speech threshold" in ln]
        self.assertEqual(len(never), 1, f"expected one idle report: {never}")
        self.assertIn("peak 0.10 < 0.25", never[0])

    def test_silence_window_reports_silence(self) -> None:
        """The near-zero branch keeps its own actionable message."""
        self._run([0.003] * REPORT_FRAMES)
        warn = [ln for ln in self._warn_lines()
                if "mic uplink carries silence" in ln]
        self.assertEqual(len(warn), 1)


class DumpPolicyTests(unittest.TestCase):
    """The evidence-preservation additions (v0.10.1)."""

    def test_asr_dump_utterances_defaults_off(self) -> None:
        cfg = AgentConfig()
        self.assertFalse(cfg.asr_dump_utterances,
                         "opt-in diagnostic must default to False")

    def test_yaml_roundtrip_loads_the_flag(self) -> None:
        import tempfile

        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            # config needs the llm_config sibling — point at the real one
            real_cfg_dir = Path(__file__).resolve().parents[2] / "config"
            data = {"app": {"asr_dump_utterances": True,
                            "root": tmp}}
            p = Path(tmp) / "voicemem_config.yaml"
            p.write_text(yaml.safe_dump(data), encoding="utf-8")
            import shutil

            (Path(tmp) / "config").mkdir()
            shutil.copy(real_cfg_dir / "llm_config.yaml",
                        Path(tmp) / "config" / "llm_config.yaml")
            cfg = AgentConfig.from_yaml(p)
            self.assertTrue(cfg.asr_dump_utterances)

    def test_has_cyrillic_exact(self) -> None:
        self.assertTrue(_has_cyrillic("Я им работаю, валь."))
        self.assertTrue(_has_cyrillic("abc Я def"))
        self.assertFalse(_has_cyrillic("Szia, Imre! Mi újság?"))
        self.assertFalse(_has_cyrillic("store school"))
        self.assertFalse(_has_cyrillic("Árvíztűrő tükörfúrógép"))
        self.assertFalse(_has_cyrillic(""))


if __name__ == "__main__":
    unittest.main()
