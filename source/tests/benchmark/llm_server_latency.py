"""llama-server latency probe (server component only, TARGET machine).

Machine: TARGET with llama-server running on 127.0.0.1:8080
(scripts/start_llama_server.ps1). Importing this file is safe anywhere:
the module is pure stdlib and app.config / app.llm are imported lazily
inside run().

Purpose: measure the llama-server's OWN latency contribution - NOT the
end-to-end voice pipeline (that is tests/benchmark/latency_benchmark.py,
which needs the full app.pipeline stack). Per streaming request
(LlmClient.chat_stream):
  - ttft_s            request start -> first content delta (time-to-first-token)
  - first_sentence_s  request start -> first sentence boundary ("." "!" "?"
                      followed by whitespace or end) in the accumulated text
  - full_s            request start -> end of the SSE stream
  - tokens            number of non-empty content deltas (approx. token count)
  - tokens_per_s      tokens / gen_s with gen_s = full_s - ttft_s
                      (0.0 when gen_s <= 0)
JSON mode (default ON, --no-json skips): LlmClient.chat_json with
response_format={"type": "json_object"} (request-level JSON mode, supported
by llama.cpp b10717 on /v1/chat/completions): full request latency plus a
validity check of the returned object (must contain both "name" and
"language"); an invalid response is reported, never fatal.

Protocol: one warmup streaming request (not counted), then N measured
streaming requests (default 5, --runs) with a fixed prompt that elicits a
multi-sentence reply, then N JSON-mode requests.

Exit codes: 0 = measurement completed (this is a measurement tool: no
pass/fail thresholds); 2 = prerequisites missing (llama-server unreachable,
app modules not importable). The M1 end-to-end target (milestones 7.6.4) is
printed as an informational line for context only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests/<kind>/x.py -> 3 levels up)

DEFAULT_JSON_OUT = "data/benchmarks/llm_server_latency.json"

# Fixed prompts (module constants so the probe is reproducible).
STREAM_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Hello! Please tell me a few sentences about coffee."},
]

JSON_MESSAGES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            'Return a JSON object with exactly the keys "name" and "language", '
            'values "Thomas" and "Hungarian".'
        ),
    },
]

JSON_REQUIRED_KEYS = ("name", "language")

_SENTENCE_END_CHARS = ".!?"


# ---------------------------------------------------------------------------
# PURE helpers (unit-tested in tests/unit/test_llm_server_latency.py)
# ---------------------------------------------------------------------------

def percentile(values: list[float], pct: float) -> float:
    """PURE: linear-interpolation percentile; empty list -> 0.0.

    ``pct`` is in the 0..100 range. The sorted values are interpolated
    between the closest ranks (numpy's default "linear" method):
    rank = (n - 1) * pct / 100. Documented examples:
    percentile([1, 2, 3, 4, 5], 50) == 3.0 and
    percentile([1, 2, 3, 4, 5], 95) == 4.8 (rank 3.8 -> 4 + 0.8 * 1).
    """
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def find_first_sentence_end(text: str) -> int:
    """PURE: index (exclusive) of the first sentence boundary in ``text``.

    A boundary is "." "!" or "?" followed by whitespace or the end of the
    string; returns -1 when no boundary exists (also for non-str input).
    Deliberately simple: decimal points ("3.14") and inner ellipsis dots are
    NOT boundaries (no whitespace follows them), while a trailing ellipsis
    counts (its last dot is followed by the end of the string).
    """
    if not isinstance(text, str) or not text:
        return -1
    for index, char in enumerate(text):
        if char in _SENTENCE_END_CHARS:
            if index + 1 == len(text) or text[index + 1].isspace():
                return index + 1
    return -1


def summarize(samples: list[dict]) -> dict:
    """PURE: percentile summary over per-request metric dicts.

    Each sample dict holds {"ttft_s", "first_sentence_s", "full_s", "tokens"}
    plus an optional "gen_s" (generation time); when "gen_s" is absent it is
    derived as full_s - ttft_s. tokens/s per request = tokens / gen_s, 0.0
    when gen_s <= 0. Missing keys default to 0.0.

    Returns {"ttft_p50_s", "ttft_p95_s", "first_sentence_p50_s", "full_p50_s",
    "tokens_per_s_p50", "tokens_per_s_p95", "n"}; an empty list gives n = 0
    and zeroed metrics.
    """
    summary: dict[str, Any] = {
        "ttft_p50_s": 0.0,
        "ttft_p95_s": 0.0,
        "first_sentence_p50_s": 0.0,
        "full_p50_s": 0.0,
        "tokens_per_s_p50": 0.0,
        "tokens_per_s_p95": 0.0,
        "n": 0,
    }
    if not samples:
        return summary
    ttfts: list[float] = []
    first_sentences: list[float] = []
    fulls: list[float] = []
    tokens_per_s: list[float] = []
    for sample in samples:
        ttft = float(sample.get("ttft_s", 0.0))
        first = float(sample.get("first_sentence_s", 0.0))
        full = float(sample.get("full_s", 0.0))
        tokens = float(sample.get("tokens", 0))
        gen_s = sample.get("gen_s")
        gen_s = float(gen_s) if gen_s is not None else full - ttft
        ttfts.append(ttft)
        first_sentences.append(first)
        fulls.append(full)
        tokens_per_s.append(tokens / gen_s if gen_s > 0 else 0.0)
    summary["ttft_p50_s"] = percentile(ttfts, 50)
    summary["ttft_p95_s"] = percentile(ttfts, 95)
    summary["first_sentence_p50_s"] = percentile(first_sentences, 50)
    summary["full_p50_s"] = percentile(fulls, 50)
    summary["tokens_per_s_p50"] = percentile(tokens_per_s, 50)
    summary["tokens_per_s_p95"] = percentile(tokens_per_s, 95)
    summary["n"] = len(samples)
    return summary


# ---------------------------------------------------------------------------
# Async run phase (llama-server must be RUNNING)
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> tuple[list[dict], list[dict], bool]:
    """Measure streaming + JSON-mode latency.

    Returns (stream_samples, json_samples, server_ok); json_samples is empty
    when --no-json is set, server_ok is False when llama-server was
    unreachable (main() turns that into exit code 2).
    """
    from app.config import AgentConfig
    from app.llm import LlmClient

    cfg = AgentConfig()
    cfg.apply_env()
    if args.config and Path(args.config).exists():
        try:
            cfg = AgentConfig.from_yaml(Path(args.config))
        except (TypeError, ValueError, RuntimeError) as exc:
            print(f"WARN: YAML config load failed ({exc}); using env + defaults")

    llm = LlmClient(cfg)
    try:
        if not await llm.health_check():
            print(f"llama-server unreachable at {cfg.llama_server_health_url}")
            print("Start it first: scripts/start_llama_server.ps1")
            return ([], [], False)

        print(f"llama-server: {cfg.llama_server_url} (model: {cfg.llm_model_name})")
        print(f"streaming probe: {args.runs} measured requests (1 warmup, not counted)")

        # Warmup (not counted): primes the connection and the server cache.
        try:
            async for _ in llm.chat_stream(STREAM_MESSAGES):
                pass
        except Exception as exc:  # noqa: BLE001 - warmup failure is not fatal
            print(f"WARN: warmup request failed ({exc})")

        samples: list[dict[str, Any]] = []
        for i in range(1, args.runs + 1):
            t0 = time.perf_counter()
            t_first: float | None = None
            first_sentence_s: float | None = None
            accumulated = ""
            tokens = 0
            async for delta in llm.chat_stream(STREAM_MESSAGES):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                tokens += 1
                accumulated += delta
                if first_sentence_s is None and find_first_sentence_end(accumulated) >= 0:
                    first_sentence_s = now - t0
            t_end = time.perf_counter()
            ttft_s = (t_first - t0) if t_first is not None else (t_end - t0)
            full_s = t_end - t0
            if first_sentence_s is None:
                # No boundary in the whole reply: fall back to the full time.
                first_sentence_s = full_s
                note = "  [no sentence boundary detected]"
            else:
                note = ""
            gen_s = full_s - ttft_s
            tps = tokens / gen_s if gen_s > 0 else 0.0
            samples.append(
                {
                    "ttft_s": ttft_s,
                    "first_sentence_s": first_sentence_s,
                    "full_s": full_s,
                    "tokens": tokens,
                    "gen_s": gen_s,
                }
            )
            print(f"  run {i:2d}/{args.runs}: ttft {ttft_s:7.3f} s | "
                  f"first sentence {first_sentence_s:7.3f} s | full {full_s:7.3f} s | "
                  f"{tokens:4d} tokens | {tps:7.1f} tokens/s{note}")

        json_samples: list[dict[str, Any]] = []
        if args.no_json:
            print("json mode: skipped (--no-json)")
            return (samples, json_samples, True)

        print(f"json mode probe: {args.runs} requests (chat_json, "
              "response_format json_object)")
        for i in range(1, args.runs + 1):
            t0 = time.perf_counter()
            json_valid = False
            error = ""
            try:
                obj = await llm.chat_json(JSON_MESSAGES)
                json_valid = (
                    isinstance(obj, dict)
                    and all(key in obj for key in JSON_REQUIRED_KEYS)
                )
            except (ValueError, RuntimeError) as exc:  # LlmUnavailableError is a RuntimeError
                error = str(exc)
            full_s = time.perf_counter() - t0
            json_samples.append({"full_s": full_s, "json_valid": json_valid})
            status = "valid" if json_valid else "INVALID"
            suffix = f"  [{error[:80]}]" if error else ""
            print(f"  json run {i:2d}/{args.runs}: full {full_s:7.3f} s | "
                  f"{status}{suffix}")
        return (samples, json_samples, True)
    finally:
        await llm.aclose()


def _resolve_output_path(path_str: str) -> Path:
    """Absolute output path; relative paths resolve against the repo root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="llama-server latency probe (server component, not end-to-end)")
    parser.add_argument("--config", help="optional YAML config (config/voicemem_config.yaml)")
    parser.add_argument("--runs", type=int, default=5,
                        help="measured requests per mode (default 5)")
    parser.add_argument("--no-json", action="store_true",
                        help="skip the JSON-mode (chat_json) latency probe")
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT,
                        help="write the summary dict as JSON here (default: %(default)s)")
    args = parser.parse_args()

    try:
        import app.config  # noqa: F401  (import guard with informative message)
        import app.llm  # noqa: F401
    except ImportError:
        print("app.config / app.llm not importable - run from the repo root.")
        return 2

    if args.runs < 1:
        print("--runs must be >= 1")
        return 2

    try:
        stream_samples, json_samples, server_ok = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - transport errors abort the probe
        print(f"Measurement failed: {exc}")
        return 2
    if not server_ok:
        return 2

    summary = summarize(stream_samples)
    print("=" * 72)
    print(f"llama-server latency summary ({summary['n']} measured streaming requests)")
    print("=" * 72)
    print(f"{'metric':20s} {'p50':>10s} {'p95':>10s}")
    print(f"{'ttft_s':20s} {summary['ttft_p50_s']:>10.3f} {summary['ttft_p95_s']:>10.3f}")
    print(f"{'first_sentence_s':20s} {summary['first_sentence_p50_s']:>10.3f} {'-':>10s}")
    print(f"{'full_s':20s} {summary['full_p50_s']:>10.3f} {'-':>10s}")
    print(f"{'tokens_per_s':20s} {summary['tokens_per_s_p50']:>10.1f} "
          f"{summary['tokens_per_s_p95']:>10.1f}")
    print("-" * 72)

    if json_samples:
        json_fulls = [float(sample.get("full_s", 0.0)) for sample in json_samples]
        valid = sum(1 for sample in json_samples if sample.get("json_valid"))
        print(f"json mode (chat_json): {len(json_samples)} requests, "
              f"full p50 {percentile(json_fulls, 50):.3f} s")
        print(f"  valid JSON objects: {valid}/{len(json_samples)} "
              f"(required keys: {', '.join(JSON_REQUIRED_KEYS)})")
        print("-" * 72)

    out_path = _resolve_output_path(args.json_out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"summary written: {out_path}")
    except OSError as exc:
        print(f"WARN: could not write {out_path} ({exc})")

    print("M1 end-to-end target (speech end -> first audio): p50 < 2 s, p95 < 2.5 s")
    print("- this probe measures only the server component of that budget")
    print("=" * 72)
    print("RESULT: MEASURED (no thresholds - latency probe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
