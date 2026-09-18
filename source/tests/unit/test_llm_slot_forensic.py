"""v0.10.3 forensic validation prep — llm_slot_forensic.py unit tests.

Pins the S8 product-turn-trace contract of scripts/llm_slot_forensic.py:

  * the web-server.log line parser (format, timestamps, garbage rejection)
  * the event classifier ([chain] prefix, gate/vendor/turn-stage kinds)
  * rotation-aware log collection order
  * per-turn reconstruction: ASR final association (a typed turn must NOT
    inherit the previous voice turn's transcript), stage timestamps,
    derived gaps, background-during / background-before correlation
  * the deterministic 5-way verdict (NO_LLM_REQUEST / NO_FIRST_TOKEN /
    CONTENTION_CONFIRMED / CONTENTION_SUSPECT_OR_CACHE_EVICTED /
    PREPROCESSING_DOMINATED / REAL_LLM_LATENCY / normal)
  * the /metrics counter logic: raw deltas recorded, prompt-token totals
    preferred by name, the v1 histogram-count misinterpretation fixed
  * the streaming probe helper: ttft_ms counts the first NON-EMPTY content
    delta (production parity) while first_data_ms marks the first SSE line

All fixtures are synthetic log text / mocked HTTP — no llama-server, no
product runtime, sandbox-safe.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "llm_slot_forensic", REPO / "scripts" / "llm_slot_forensic.py")
forensic = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(forensic)  # type: ignore[union-attr]


def _line(h: int, m: int, s: float, msg: str,
          logger: str = "app.web_server") -> str:
    """One product-log-shaped line (logging format of _setup_logging)."""
    base = datetime(2025, 6, 12, h, m, int(s))
    ms = int(round((s - int(s)) * 1000))
    stamp = base.strftime("%Y-%m-%d %H:%M:%S") + f",{ms:03d}"
    return f"{stamp} INFO {logger}: {msg}"


def _chain(h: int, m: int, s: float, msg: str) -> str:
    """A [chain] diagnostics line (app.web_server._diag output)."""
    return _line(h, m, s, f"[chain] {msg}")


def _gate(h: int, m: int, s: float, msg: str) -> str:
    return _line(h, m, s, msg, logger="app.background_memory")


def _vendor(h: int, m: int, s: float, msg: str) -> str:
    return _line(h, m, s, msg, logger="voicemem.utils.common.llm_bg_gate")


def _fixture_lines() -> list[str]:
    """A synthetic session with six turns, one per verdict class.

    turn 1 (10:00) healthy voice turn — normal
    turn 2 (10:05) voice turn behind an in-flight background chain that is
                   cancelled mid-turn — CONTENTION_CONFIRMED
    turn 3 (10:10) typed turn, nothing on the slot, slow request —
                   REAL_LLM_LATENCY (and NO inherited ASR)
    turn 4 (10:15) typed turn, 6 s memory retrieval — PREPROCESSING_DOMINATED
    turn 5 (10:20) voice turn whose LLM stream dies before any token —
                   NO_FIRST_TOKEN, incomplete
    turn 6 (10:25) typed turn right after a background chain started (no
                   cancel observed) — CONTENTION_SUSPECT_OR_CACHE_EVICTED
    """
    return [
        # ---- turn 1: healthy ------------------------------------------------
        _chain(10, 0, 0.000, "speech start (level 0.80)"),
        _gate(10, 0, 0.010, "background memory gate armed (reason=speech)"),
        _chain(10, 0, 1.500, "ASR flush done (800 ms, engine=nemotron)"),
        _chain(10, 0, 1.600, "ASR final transcript: 'szia, hogy vagy?'"),
        _chain(10, 0, 1.650, "ASR turn dispatch (source=asr)"),
        _chain(10, 0, 1.700, "turn received (source=asr, 15 chars: 'szia, hogy vagy?')"),
        _chain(10, 0, 1.750, "memory start"),
        _chain(10, 0, 1.760, "emotion start (init)"),
        _chain(10, 0, 2.000, "emotion done (analyzer ready · prosody task started)"),
        _chain(10, 0, 3.000, "memory done (3 hits)"),
        _chain(10, 0, 3.010, "emotion prosody done (waited)"),
        _chain(10, 0, 3.100, "llm start (qwen3.6-35b-a3b @ http://127.0.0.1:8080/v1)"),
        _chain(10, 0, 5.500, "llm first token (3800 ms)"),
        _chain(10, 0, 8.000, "llm done (200 chars, 4900 ms)"),
        _chain(10, 0, 8.050, "tts start (chunk 1, 60 chars)"),
        _chain(10, 0, 9.000, "first TTS audio sent to the browser (7300 ms)"),
        _chain(10, 0, 12.000, "tts done (2 chunks, 10300 ms)"),
        _chain(10, 0, 12.100, "answer done: 200 chars, total 10400 ms"),
        _gate(10, 0, 12.110, "background memory gate released (turn ended; grace=2.0s idle=6.0s)"),
        _gate(10, 0, 18.200, "background memory gate open (idle window held; background may start)"),
        _gate(10, 0, 18.210, "background memory ingest started (turn_no=1, 1 queued, start_wait=6.1s)"),
        # ---- turn 2: contention outlier -------------------------------------
        _chain(10, 5, 0.000, "speech start (level 0.75)"),
        _gate(10, 5, 0.050, "background memory CANCELLED (user needs the slot: speech)"),
        _vendor(10, 5, 0.060, "background LLM cancel requested (trigger=speech, #2)"),
        _gate(10, 5, 1.000, "background memory cancelled mid-chain, re-queued (0 queued): leg aborted"),
        _chain(10, 5, 2.500, "ASR final transcript: 'mesélek a nyaralásról'"),
        _chain(10, 5, 2.550, "ASR turn dispatch (source=asr)"),
        _chain(10, 5, 2.600, "turn received (source=asr, 23 chars: 'mesélek a nyaralásról')"),
        _chain(10, 5, 2.700, "memory start"),
        _chain(10, 5, 4.000, "memory done (2 hits)"),
        _chain(10, 5, 4.100, "llm start (qwen3.6-35b-a3b @ http://127.0.0.1:8080/v1)"),
        _chain(10, 5, 40.000, "llm first token (37400 ms)"),
        _chain(10, 5, 44.000, "llm done (150 chars, 39900 ms)"),
        _chain(10, 5, 44.500, "answer done: 150 chars, total 41900 ms"),
        # ---- turn 3: typed, real LLM latency --------------------------------
        _chain(10, 10, 0.000, "turn received (source=text, 10 chars: 'hogy vagy?')"),
        _chain(10, 10, 0.100, "memory start"),
        _chain(10, 10, 1.000, "memory done (0 hits)"),
        _chain(10, 10, 1.050, "llm start (qwen3.6-35b-a3b @ http://127.0.0.1:8080/v1)"),
        _chain(10, 10, 32.000, "llm first token (31000 ms)"),
        _chain(10, 10, 35.000, "llm done (80 chars, 34000 ms)"),
        _chain(10, 10, 35.100, "answer done: 80 chars, total 35100 ms"),
        # ---- turn 4: preprocessing dominated ---------------------------------
        _chain(10, 15, 0.000, "turn received (source=text, 18 chars: 'emlékszel a hétre?')"),
        _chain(10, 15, 0.100, "memory start"),
        _chain(10, 15, 6.000, "memory done (5 hits)"),
        _chain(10, 15, 6.050, "llm start (qwen3.6-35b-a3b @ http://127.0.0.1:8080/v1)"),
        _chain(10, 15, 11.000, "llm first token (11000 ms)"),
        _chain(10, 15, 13.000, "llm done (60 chars, 6950 ms)"),
        _chain(10, 15, 13.100, "answer done: 60 chars, total 13100 ms"),
        # ---- turn 5: stream dies before the first token ----------------------
        _chain(10, 19, 54.000, "speech start (level 0.66)"),
        _chain(10, 19, 58.000, "ASR final transcript: 'teszt'"),
        _chain(10, 19, 58.050, "ASR turn dispatch (source=asr)"),
        _chain(10, 20, 0.000, "turn received (source=asr, 5 chars: 'teszt')"),
        _chain(10, 20, 0.100, "memory start"),
        _chain(10, 20, 1.000, "memory done (1 hits)"),
        _chain(10, 20, 1.100, "llm start (qwen3.6-35b-a3b @ http://127.0.0.1:8080/v1)"),
        _chain(10, 20, 30.000, "llm failed (LLM streaming request failed: [Errno 111] refused)"),
        # ---- turn 6: background started just before the turn, no cancel ------
        _gate(10, 25, 0.000, "background memory ingest started (turn_no=2, 1 queued, start_wait=7.0s)"),
        _chain(10, 25, 10.000, "turn received (source=text, 12 chars: 'mi a hírek?')"),
        _chain(10, 25, 10.100, "memory start"),
        _chain(10, 25, 11.000, "memory done (0 hits)"),
        _chain(10, 25, 11.200, "llm start (qwen3.6-35b-a3b @ http://127.0.0.1:8080/v1)"),
        _chain(10, 25, 35.000, "llm first token (23800 ms)"),
        _chain(10, 25, 38.000, "llm done (70 chars, 26800 ms)"),
        _chain(10, 25, 38.100, "answer done: 70 chars, total 28100 ms"),
        # noise that must be ignored entirely
        "INFO:     127.0.0.1:54321 - \"GET /api/health HTTP/1.1\" 200 OK",
        "  File \"app/web_server.py\", line 1, in <module>",
        "2025-06-12 10:00:00 INFO app.web_server: some other line",
    ]


class ParseTests(unittest.TestCase):
    def test_parse_valid_line(self):
        ev = forensic.parse_product_log_line(
            "2025-06-12 10:23:45,123 INFO app.web_server: [chain] memory start")
        self.assertIsNotNone(ev)
        assert ev is not None
        self.assertEqual(ev["ts_str"], "2025-06-12 10:23:45,123")
        self.assertEqual(ev["level"], "INFO")
        self.assertEqual(ev["logger"], "app.web_server")
        self.assertEqual(ev["msg"], "[chain] memory start")

    def test_parse_rejects_garbage(self):
        for line in ("", "random text", "2025-06-12 10:23:45 INFO no colon here"):
            self.assertIsNone(forensic.parse_product_log_line(line))
        # millisecond precision is required (a whole-second stamp cannot
        # order sub-second turn stages)
        self.assertIsNone(forensic.parse_product_log_line(
            "2025-06-12 10:23:45 INFO app.web_server: x"))

    def test_parse_epoch_is_usable(self):
        a = forensic.parse_product_log_line(
            "2025-06-12 10:23:45,000 INFO app.web_server: a")
        b = forensic.parse_product_log_line(
            "2025-06-12 10:23:46,500 INFO app.web_server: b")
        assert a and b
        self.assertAlmostEqual(b["ts"] - a["ts"], 1.5, places=6)

    def test_classify_strips_chain_prefix(self):
        self.assertEqual(forensic.classify_event("[chain] memory start"),
                         "memory_start")
        self.assertEqual(forensic.classify_event("memory start"), "memory_start")

    def test_classify_kinds(self):
        cases = {
            "background memory gate armed (reason=speech)": "gate_armed",
            "background memory CANCELLED (user needs the slot: speech)": "gate_cancel",
            "background LLM cancel requested (trigger=speech, #2)": "bg_cancel_requested",
            "background LLM cancel cleared (trigger=idle)": "bg_cancel_cleared",
            "background memory gate released (turn ended; grace=2.0s idle=6.0s)": "gate_released",
            "background memory gate open (idle window held; background may start)": "gate_open",
            "background memory deferred (conversation active; 1 queued)": "bg_deferred",
            "background memory ingest started (turn_no=1, 1 queued, start_wait=6.1s)": "bg_ingest_started",
            "background memory cancelled mid-chain, re-queued (0 queued): x": "bg_requeued",
            "background memory ingest failed: boom": "bg_ingest_failed",
            "speech start (level 0.80)": "speech_start",
            "speech end (1200 ms of speech)": "speech_end",
            "ASR flush done (800 ms, engine=nemotron)": "asr_flush_done",
            "ASR final transcript: 'szia'": "asr_final",
            "ASR turn dispatch (source=asr)": "asr_dispatch",
            "turn received (source=asr, 15 chars: 'x')": "turn_start",
            "turn CRASHED: ValueError": "turn_crashed",
            "memory done (3 hits)": "memory_done",
            "emotion start (init)": "emotion_init_start",
            "emotion done (analyzer ready · prosody task started)": "emotion_init_done",
            "emotion prosody done (waited)": "emotion_prosody_done",
            "emotion prosody pending (late — ran past the wait window)": "emotion_prosody_late",
            "llm start (qwen @ http://127.0.0.1:8080/v1)": "llm_start",
            "llm first token (3800 ms)": "llm_first_token",
            "llm done (200 chars, 4900 ms)": "llm_done",
            "llm failed (timeout)": "llm_failed",
            "tts start (chunk 1, 60 chars)": "tts_start",
            "first TTS audio sent to the browser (7300 ms)": "first_audio",
            "tts done (2 chunks, 10300 ms)": "tts_done",
            "answer done: 200 chars, total 10400 ms": "answer_done",
        }
        for msg, kind in cases.items():
            self.assertEqual(forensic.classify_event(msg), kind,
                             f"classify({msg!r}) != {kind}")
        self.assertEqual(forensic.classify_event("stage llm started: qwen"), "")


class FileCollectionTests(unittest.TestCase):
    def test_collect_rotation_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "web-server.log.3").write_text("old3", encoding="utf-8")
            (root / "web-server.log.1").write_text("old1", encoding="utf-8")
            (root / "web-server.log").write_text("now", encoding="utf-8")
            files = forensic.collect_product_log_files(root / "web-server.log")
            self.assertEqual([p.name for p in files],
                             ["web-server.log.3", "web-server.log.1",
                              "web-server.log"])

    def test_collect_missing_is_empty(self):
        self.assertEqual(
            forensic.collect_product_log_files("no/such/web-server.log"), [])

    def test_parse_since(self):
        self.assertIsNone(forensic._parse_since(""))
        self.assertEqual(forensic._parse_since("1750000000"), 1750000000.0)
        iso = forensic._parse_since("2025-06-12T10:05:00")
        assert iso is not None
        self.assertEqual(datetime.fromtimestamp(iso).hour, 10)
        self.assertIsNone(forensic._parse_since("not-a-time"))


class RebuildTests(unittest.TestCase):
    """Turn reconstruction + the deterministic 5-way verdict."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log = Path(self._tmp.name) / "web-server.log"
        self.log.write_text("\n".join(_fixture_lines()) + "\n",
                            encoding="utf-8")
        self._saved = (forensic.OUTLIER_MS, forensic.PREPROC_MS,
                       forensic.BG_WINDOW_S)
        forensic.OUTLIER_MS = 10000.0
        forensic.PREPROC_MS = 3000.0
        forensic.BG_WINDOW_S = 45.0

    def tearDown(self):
        (forensic.OUTLIER_MS, forensic.PREPROC_MS,
         forensic.BG_WINDOW_S) = self._saved

    def _payload(self, since_epoch=None):
        events = forensic.load_product_events(
            [self.log], since_epoch=since_epoch)
        return forensic.rebuild_turns(events, keep_turns=0)

    def test_six_turns_reconstructed_with_expected_verdicts(self):
        p = self._payload()
        self.assertEqual(p["turns_total"], 6)
        self.assertEqual(p["turns_complete"], 5)
        self.assertEqual(p["turns_abandoned"], 1)
        verdicts = [t["verdict"] for t in p["turns_detail"]]
        self.assertEqual(verdicts, [
            "normal",
            "CONTENTION_CONFIRMED",
            "REAL_LLM_LATENCY",
            "PREPROCESSING_DOMINATED",
            "NO_FIRST_TOKEN",
            "CONTENTION_SUSPECT_OR_CACHE_EVICTED",
        ])

    def test_healthy_turn_gaps(self):
        p = self._payload()
        t1 = p["turns_detail"][0]
        self.assertEqual(t1["source"], "asr")
        self.assertEqual(t1["gaps_ms"]["asr_to_turn_ms"], 100.0)
        self.assertEqual(t1["gaps_ms"]["memory_ms"], 1250.0)
        self.assertEqual(t1["gaps_ms"]["preprocessing_ms"], 1400.0)
        self.assertEqual(t1["gaps_ms"]["llm_wait_ms"], 2400.0)
        self.assertEqual(t1["gaps_ms"]["first_token_ms"], 3800.0)
        self.assertEqual(t1["gaps_ms"]["llm_stream_ms"], 2500.0)
        self.assertEqual(t1["gaps_ms"]["total_ms"], 10400.0)
        self.assertTrue(t1["complete"])
        self.assertIn("llm_first_token", t1["stages"])
        self.assertIn("asr_final", t1["stages"])  # via the pre-turn trail

    def test_typed_turn_does_not_inherit_previous_asr(self):
        p = self._payload()
        t3 = p["turns_detail"][2]  # source=text, follows a voice turn
        self.assertEqual(t3["source"], "text")
        self.assertIsNone(t3["gaps_ms"]["asr_to_turn_ms"])

    def test_contention_turn_carries_cancel_evidence(self):
        p = self._payload()
        t2 = p["turns_detail"][1]
        kinds = [e["kind"] for e in t2["bg_during"]]
        self.assertIn("gate_cancel", kinds)
        self.assertIn("bg_cancel_requested", kinds)
        self.assertIn("bg_requeued", kinds)
        self.assertEqual(t2["gaps_ms"]["llm_wait_ms"], 35900.0)

    def test_suspect_turn_carries_bg_before_evidence(self):
        p = self._payload()
        t6 = p["turns_detail"][5]
        kinds = [e["kind"] for e in t6["bg_before"]]
        self.assertIn("bg_ingest_started", kinds)
        self.assertEqual([e["kind"] for e in t6["bg_during"]
                          if e["kind"] in ("gate_cancel",
                                           "bg_cancel_requested")], [])

    def test_failed_turn_is_incomplete_and_flagged(self):
        p = self._payload()
        t5 = p["turns_detail"][4]
        self.assertFalse(t5["complete"])
        self.assertIsNone(t5["gaps_ms"]["llm_wait_ms"])
        self.assertIn("llm_failed", t5["stages"])

    def test_distributions(self):
        p = self._payload()
        d = p["distributions_ms"]["llm_wait"]
        self.assertEqual(d["n"], 5)  # turn 5 never produced a first token
        self.assertEqual(d["min"], 2400.0)
        self.assertEqual(d["median"], 23800.0)
        self.assertEqual(d["max"], 35900.0)
        self.assertEqual(p["distributions_ms"]["preprocessing"]["n"], 6)

    def test_outliers_listed_with_evidence(self):
        p = self._payload()
        self.assertEqual(len(p["outliers"]), 5)  # everything but turn 1
        by_idx = {o["idx"]: o for o in p["outliers"]}
        self.assertEqual(by_idx[2]["verdict"], "CONTENTION_CONFIRMED")

    def test_gate_events_collected(self):
        p = self._payload()
        kinds = [e["kind"] for e in p["gate_events"]]
        for expected in ("gate_armed", "gate_cancel", "bg_cancel_requested",
                         "bg_requeued", "gate_released", "gate_open",
                         "bg_ingest_started"):
            self.assertIn(expected, kinds)

    def test_since_filter_drops_old_turns(self):
        events = forensic.load_product_events([self.log])
        first_kept = events[0]["ts"]
        p = forensic.rebuild_turns(
            forensic.load_product_events(
                [self.log], since_epoch=first_kept + 300.0),  # +5 min
            keep_turns=0)
        self.assertEqual(p["turns_total"], 5)
        self.assertEqual(p["turns_detail"][0]["gaps_ms"]["llm_wait_ms"],
                         35900.0)  # the old turn 2 is now first

    def test_verdict_unit_no_llm_request(self):
        self.assertEqual(
            forensic._turn_verdict({}, [], [], llm_started=False),
            "NO_LLM_REQUEST")
        self.assertEqual(
            forensic._turn_verdict({}, [], [], llm_started=True),
            "NO_FIRST_TOKEN")

    def test_verdict_unit_contention_beats_preprocessing(self):
        gaps = {"llm_wait_ms": 30000.0, "first_token_ms": 31000.0,
                "preprocessing_ms": 5000.0}
        bg_during = [{"ts": "x", "kind": "bg_cancel_requested", "msg": "y"}]
        self.assertEqual(
            forensic._turn_verdict(gaps, bg_during, [], llm_started=True),
            "CONTENTION_CONFIRMED")

    def test_verdict_unit_normal_under_thresholds(self):
        gaps = {"llm_wait_ms": 4000.0, "first_token_ms": 5000.0,
                "preprocessing_ms": 1000.0}
        self.assertEqual(
            forensic._turn_verdict(gaps, [], [], llm_started=True),
            "normal")

    def test_verdict_unit_first_token_only_slowness(self):
        # llm_wait below the bar but the user still waited >10 s for the
        # first token (preprocessing) — must be PREPROCESSING_DOMINATED
        gaps = {"llm_wait_ms": 4000.0, "first_token_ms": 12000.0,
                "preprocessing_ms": 8000.0}
        self.assertEqual(
            forensic._turn_verdict(gaps, [], [], llm_started=True),
            "PREPROCESSING_DOMINATED")


class MetricsTests(unittest.TestCase):
    """The /metrics token-counter logic (v1 histogram-count bug fixed)."""

    BEFORE = {
        "llamacpp:n_prompt_tokens_processed_total": 100.0,
        "llamacpp:prompt_tokens_seconds_count": 5.0,
        'llamacpp:prompt_tokens_seconds_bucket{le="1.0"}': 3.0,
        "llamacpp:n_tokens_generated_total": 40.0,
    }
    AFTER = {
        "llamacpp:n_prompt_tokens_processed_total": 1600.0,
        "llamacpp:prompt_tokens_seconds_count": 6.0,
        'llamacpp:prompt_tokens_seconds_bucket{le="1.0"}': 4.0,
        "llamacpp:n_tokens_generated_total": 90.0,
    }

    def test_delta_filters_buckets_and_negatives(self):
        d = forensic._metrics_delta(self.BEFORE, self.AFTER)
        self.assertEqual(d["llamacpp:n_prompt_tokens_processed_total"], 1500.0)
        self.assertEqual(d["llamacpp:n_tokens_generated_total"], 50.0)
        self.assertNotIn('llamacpp:prompt_tokens_seconds_bucket{le="1.0"}', d)

    def test_reprocessed_prefers_explicit_token_totals(self):
        out = forensic._reprocessed_tokens(self.BEFORE, self.AFTER)
        self.assertEqual(out["counter"],
                         "llamacpp:n_prompt_tokens_processed_total")
        self.assertEqual(out["tokens"], 1500)
        # the histogram observation count stays as RAW evidence only
        self.assertEqual(
            out["raw_llamacpp_counters"]["llamacpp:prompt_tokens_seconds_count"],
            1.0)

    def test_reprocessed_never_misreads_histogram_count(self):
        """v1 accepted llamacpp:prompt_tokens_seconds_count as a TOKEN count.

        That counter counts prompt-eval OPERATIONS (the histogram's
        observation count): using it would have reported 1 re-processed
        token per request. v2 must return None rather than lie.
        """
        before = {"llamacpp:prompt_tokens_seconds_count": 5.0}
        after = {"llamacpp:prompt_tokens_seconds_count": 6.0}
        out = forensic._reprocessed_tokens(before, after)
        self.assertIsNone(out["tokens"])
        self.assertIsNone(out["counter"])

    def test_reprocessed_handles_empty_snapshots(self):
        out = forensic._reprocessed_tokens({}, {})
        self.assertIsNone(out["tokens"])
        self.assertEqual(out["raw_llamacpp_counters"], {})


class StreamProbeTests(unittest.TestCase):
    """ttft_ms = first NON-EMPTY content delta (production parity)."""

    def test_first_content_delta_semantics(self):
        import httpx

        def gen():
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            time.sleep(0.15)
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"

        def handler(request):
            return httpx.Response(200, content=gen())

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport, base_url="http://mock") as client:
            r = forensic._stream_first_token_ms(
                client, {"messages": [{"role": "user", "content": "hi"}],
                          "stream": True, "max_tokens": 8})
        # the role-only chunk arrives immediately: it is the first DATA
        # line but NOT a token (production counts content deltas only)
        self.assertGreaterEqual(r["first_data_ms"], 0.0)
        self.assertLess(r["first_data_ms"], 140.0)
        self.assertGreaterEqual(r["ttft_ms"], 140.0)
        self.assertEqual(r["text_len"], 2)
        self.assertEqual(r["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
