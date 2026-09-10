"""Unit tests for the v0.4.12 UI capture/diagnosis features.

Field report (session web-7b05b5bc, 2026-09-04 18:13): "A szöveget már érti a
rendszer, de valami még gátolja, hogy átkerüljön az LLM-hez." — the ASR test
produced a real Hungarian transcript (rms 0.0546, peak 0.470) but the LLM
never answered, because the live VAD gate never fired (Silero peak 0.003 <
threshold 0.25; the live uplink reported VAD peak 0.00 all session).

Root cause analysis: the capture numbers were IDENTICAL to the v0.4.9 report
(rms 0.0489/VAD 0.003) even though v0.4.10+ shipped noiseSuppression:false +
the anti-aliased downsampler — the strongest explanation is that the browser
tab was left open across the upgrade and silently kept running the OLD
(voicemem.html v0.4.9) JS. v0.4.12 therefore adds:

1. PAGE_VERSION in the page, checked against /api/health on load — a stale
   page shows a warning line + toast (and the page reports itself to
   /api/lang, which the backend logs: field logs now always name the page
   build the browser actually ran);
2. a "Raw" capture toggle (Chrome AEC/AGC/NS ALL off) for clean line-in /
   pro chains, persisted in localStorage, feeding all three capture paths
   through selectedMicConstraint();
3. a "Send to the agent" button in the ASR test panel — the heard transcript
   can be sent as a real user_text turn, so the speech→LLM chain works even
   while VAD/capture is being debugged;
4. a smarter ASR-test verdict: real level (rms > 0.02) but Silero deaf
   (peak < 0.05) is NOT a threshold problem — it names the stale page / Raw
   toggle / send-button instead of suggesting a lower threshold.

Pure stdlib + the repo files (HTML text pinned the same way
tests/validation/test_feature_scripts.py pins the UI).
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UI = REPO_ROOT / "web" / "voicemem.html"
SERVER = REPO_ROOT / "app" / "web_server.py"
VERSION_FILE = REPO_ROOT / "VERSION"


def _ui_text() -> str:
    return UI.read_text(encoding="utf-8")


class PageVersionTests(unittest.TestCase):
    """The page's own build version + the staleness check against /api/health."""

    def test_page_version_matches_repo_version(self) -> None:
        html = _ui_text()
        self.assertIn("PAGE_VERSION", html)
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
        self.assertIn(
            f"const PAGE_VERSION='{version}';", html,
            "PAGE_VERSION in voicemem.html must be bumped together with VERSION",
        )

    def test_staleness_check_on_load(self) -> None:
        """The page compares its version to /api/health and warns on mismatch."""
        html = _ui_text()
        self.assertIn("/api/health", html)
        self.assertIn("uiVerStale", html)
        self.assertIn("uiVerOk", html)
        self.assertIn("Ctrl+F5", html)  # the reload instruction
        self.assertIn('id="uiVer"', html)  # the visible version line

    def test_page_reports_version_to_lang_endpoint(self) -> None:
        """/api/lang POST carries the page version for field-log diagnostics."""
        html = _ui_text()
        self.assertIn("{lang:'en',page:PAGE_VERSION}", html)


class ServerLangPageVersionTests(unittest.TestCase):
    """The /api/lang handler logs the reported page version (+ stale flag)."""

    def test_lang_handler_logs_page_version(self) -> None:
        src = SERVER.read_text(encoding="utf-8")
        self.assertIn('data.get("page")', src)
        self.assertIn("web UI page version", src)
        self.assertIn("STALE PAGE, reload needed", src)

    def test_lang_handler_roundtrip(self) -> None:
        """POST /api/lang with a page version returns the language (no error)."""
        import tempfile

        from fastapi.testclient import TestClient

        from app.config import AgentConfig
        from app.web_server import WebComponents, build_web_app

        with tempfile.TemporaryDirectory(prefix="vm_v0412_") as tmp:
            cfg = AgentConfig(root=Path(tmp))
            components = WebComponents(cfg, mock=True).build()
            app = build_web_app(components)
            client = TestClient(app)
            r = client.post(
                "/api/lang", json={"lang": "en", "page": "0.4.12"}
            )
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json().get("lang"), "en")


class RawCaptureToggleTests(unittest.TestCase):
    """The Raw toggle: all-Chrome-processing-off constraints, persisted."""

    def test_toggle_markup_and_id(self) -> None:
        html = _ui_text()
        self.assertIn('id="rawCapChk"', html)
        self.assertIn('id="rawCapLbl"', html)
        self.assertIn("rawCaptureTitle", html)

    def test_constraint_reads_raw_state(self) -> None:
        """selectedMicConstraint() branches on the Raw checkbox."""
        html = _ui_text()
        self.assertIn("rawCapChk&&rawCapChk.checked", html)
        # raw branch: all three Chrome processing flags false
        self.assertIn(
            "?{echoCancellation:false,noiseSuppression:false,autoGainControl:false}",
            html,
        )
        # default branch unchanged (v0.4.10 behaviour: NS off, AEC+AGC on)
        self.assertIn(
            ":{echoCancellation:true,noiseSuppression:false,autoGainControl:true}",
            html,
        )

    def test_toggle_persisted_in_localstorage(self) -> None:
        html = _ui_text()
        self.assertIn("vm-rawcap", html)
        self.assertIn("localStorage.setItem('vm-rawcap','1')", html)
        self.assertIn("localStorage.removeItem('vm-rawcap')", html)

    def test_toggle_has_on_off_toasts(self) -> None:
        html = _ui_text()
        self.assertIn("rawCapOn", html)
        self.assertIn("rawCapOff", html)


class AsrTestSendTurnTests(unittest.TestCase):
    """The VAD-independent escape hatch: transcript → real user_text turn."""

    def test_send_button_markup(self) -> None:
        html = _ui_text()
        self.assertIn('id="asrTestSend"', html)
        self.assertIn('id="asrTestSendRow"', html)
        self.assertIn("asrTestSend", html)  # i18n label
        self.assertIn("Send to the agent", html)

    def test_send_sends_user_text_over_ws(self) -> None:
        """The button reuses the typed-turn message: {type:'user_text'}."""
        html = _ui_text()
        self.assertIn(
            "ws.send(JSON.stringify({type:'user_text',text:t}));", html
        )
        self.assertIn("lastAsrTestText", html)

    def test_send_row_visibility_follows_transcript(self) -> None:
        html = _ui_text()
        self.assertIn("asrTestSendRow.hidden=!txt", html)
        self.assertIn("asrTestSend.disabled=true", html)

    def test_sent_ack_text_is_set(self) -> None:
        """The 'sent' acknowledgement span must actually carry text."""
        html = _ui_text()
        self.assertIn("asrTestSent.textContent=i18n('asrTestSentOk')", html)


class SmarterVerdictTests(unittest.TestCase):
    """Level-but-deaf (rms>0.02, VAD peak<0.05) — v0.4.14: the message now
    explains the energy-fallback gate instead of blaming the capture chain."""

    def test_deaf_verdict_key_and_branch(self) -> None:
        html = _ui_text()
        self.assertIn("asrTestVerdictDeaf", html)
        self.assertIn("rms>0.02", html)
        self.assertIn("peak<0.05", html)

    def test_deaf_verdict_names_the_energy_fallback(self) -> None:
        """v0.4.14: the deaf verdict must say the live path works via the
        energy fallback (the automatic turn dispatch no longer needs the
        Raw toggle / reload first) and keep the manual button mention."""
        html = _ui_text()
        self.assertIn("energy-fallback gate", html)
        self.assertIn("speaking will start turns automatically", html)
        self.assertIn("speech start · energy fallback", html)
        self.assertIn("button below stays available", html)

    def test_vad_row_reports_silero_peak_and_gate(self) -> None:
        """The VAD stat row names the primary's own peak and the gate that
        opened (silero vs energy fallback)."""
        html = _ui_text()
        self.assertIn("asrTestVadSilero", html)
        self.assertIn("asrTestVadGate", html)
        self.assertIn("vad.gate||'silero'", html)

    def test_gate_opened_through_fallback_branch(self) -> None:
        """v0.4.14: vad_would_fire TRUE + deaf silero peak → explanatory
        deaf verdict (the user must know the live path works)."""
        html = _ui_text()
        self.assertIn("silero<(vad.threshold!=null?vad.threshold:0.25)", html)


if __name__ == "__main__":
    unittest.main()
