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


class PageVersionInjectionTests(unittest.TestCase):
    """v0.8.1 (post-audit P1-1): the SERVED page version comes from VERSION.

    The repo literal is regenerated at release prep (scripts/sync_page_version.py)
    and gate-checked by test_page_version_matches_repo_version above; the SERVED
    page is additionally stamped at request time from the VERSION file — the
    single authoritative source (the same value /api/health serves). These
    tests fail if a future release changes VERSION and the served page stops
    carrying it (injection removed, bypassed, or broken).
    """

    def _served_page(self, html_text: str):
        """Context manager: a TestClient whose index() serves a COPY of the
        given page text from a temp web dir (the repo's own
        web/voicemem.html is untouched; _WEB_DIR is always restored)."""
        import tempfile
        from contextlib import contextmanager

        from fastapi.testclient import TestClient

        import app.web_server as web_server
        from app.config import AgentConfig
        from app.web_server import WebComponents, build_web_app

        @contextmanager
        def _ctx():
            tmp_obj = tempfile.TemporaryDirectory(prefix="vm_v081_page_")
            web_dir = Path(tmp_obj.name) / "web"
            web_dir.mkdir()
            (web_dir / "voicemem.html").write_text(html_text, encoding="utf-8")
            original_web_dir = web_server._WEB_DIR
            web_server._WEB_DIR = web_dir
            try:
                cfg = AgentConfig(root=Path(tmp_obj.name))
                components = WebComponents(cfg, mock=True).build()
                app = build_web_app(components)
                with TestClient(app) as client:
                    yield client
            finally:
                web_server._WEB_DIR = original_web_dir
                tmp_obj.cleanup()

        return _ctx()

    def test_served_page_carries_repo_version(self) -> None:
        """GET / returns the page with the CURRENT VERSION stamped in —
        the effective PAGE_VERSION always equals VERSION."""
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
        with self._served_page(_ui_text()) as client:
            r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(f"const PAGE_VERSION='{version}';", r.text)
        self.assertTrue(
            r.headers.get("content-type", "").startswith("text/html"),
            f"content-type must be text/html, got {r.headers.get('content-type')}",
        )

    def test_served_page_injects_over_a_stale_literal(self) -> None:
        """The v0.8.0 defect shape (stale file literal) must NOT reach the
        browser: the served page carries the CURRENT VERSION even when the
        file on disk still holds an old literal."""
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
        stale = _ui_text().replace(
            "const PAGE_VERSION='", "const PAGE_VERSION='0.0.0-", 1
        )  # a deliberately wrong literal (0.0.0-…)
        self.assertIn("const PAGE_VERSION='0.0.0-", stale)  # precondition
        with self._served_page(stale) as client:
            r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(f"const PAGE_VERSION='{version}';", r.text)
        self.assertNotIn("const PAGE_VERSION='0.0.0-", r.text)

    def test_injection_helper_is_pure_and_idempotent(self) -> None:
        """_inject_page_version: stamps the current VERSION, leaves an
        already-synced page untouched, and never touches anything else
        (synthetic snippets — independent of the repo's current literal)."""
        from app.web_server import _inject_page_version

        version = VERSION_FILE.read_text(encoding="utf-8").strip()
        body = "…<script>const PAGE_VERSION='{}';\nrest of the page ünïcode</script>…"
        stale = body.format("0.0.0")
        fixed = _inject_page_version(stale)
        self.assertEqual(fixed, body.format(version))
        # idempotent: a second pass changes nothing
        self.assertEqual(_inject_page_version(fixed), fixed)
        # a page without the literal is served untouched
        no_literal = "<html><body>no version here</body></html>"
        self.assertEqual(_inject_page_version(no_literal), no_literal)


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
