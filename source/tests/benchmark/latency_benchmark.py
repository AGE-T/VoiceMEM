"""M1 end-to-end latency benchmark (mock-input turns through the FULL pipeline).

Machine: runs on ANY machine once app.pipeline / app.mock_components exist
(written in Task 8 integration); the latency NUMBERS are only meaningful on
the TARGET machine. Importing this file in the sandbox is safe: the pipeline
import is lazy and guarded.

Timing contract with Task 8 (app/pipeline.py):
  - VoicePipeline.handle_utterance(...) returns a TurnResult whose
    TurnTimings.to_dict() contains perf_counter-style timestamps:
      speech_end_s, asr_final_s, first_token_s, first_audio_s, full_response_s
    plus memory_s (VoiceMem retrieval DURATION, not a timestamp).
  - Latency per turn = first_audio_s - speech_end_s (spec Section 19.4:
    "speech end -> first audio"); other stage deltas are reported too.
  - If the pipeline changes these semantics, only _extract_latencies() needs
    updating here.

Exit criteria (milestones 7.6.4): p50 < 2.0 s and p95 < 2.5 s
  -> exit 0 on success, 1 on failure, 2 when the pipeline is not present yet.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests/<kind>/x.py -> 3 levels up)

P50_LIMIT = 2.0
P95_LIMIT = 2.5
TIMESTAMP_KEYS = ("asr_final_s", "first_token_s", "first_audio_s", "full_response_s")


class _NoopSpeaker:
    """Duck-typed SpeakerOutput stand-in: latency benchmark makes no sound."""

    def play(self, pcm: Any, sample_rate: int | None = None) -> None:
        return None

    async def play_async(self, pcm: Any, sample_rate: int | None = None) -> None:
        return None

    def stop(self) -> None:
        return None


def _build_mock_pipeline(cfg: Any) -> Any:
    """Assemble a mock-component pipeline (Task 8 provides app.mock_components).

    Construction is defensive: the exact mock constructors may still change
    during integration; on any mismatch we bail out with exit-code guidance.
    """
    from app.mock_components import (  # lazy: Task 8 file
        MockAsrEngine,
        MockLlmClient,
        MockTtsEngine,
        MockVad,
        MockVoiceMemBridge,
    )
    from app.pipeline import VoicePipeline  # lazy: Task 8 file

    def _ctor(cls: type, *args: Any) -> Any:
        try:
            return cls(*args)
        except TypeError:
            return cls()

    asr = _ctor(MockAsrEngine)
    try:
        asr.queue = [f"latency benchmark turn {i}" for i in range(64)]
    except AttributeError as exc:  # .queue is the contract in CONTRACT.md
        raise RuntimeError(f"MockAsrEngine has no settable .queue: {exc}") from exc
    pipeline = VoicePipeline(
        config=cfg,
        asr=asr,
        llm=_ctor(MockLlmClient),
        tts=_ctor(MockTtsEngine),
        voicemem=_ctor(MockVoiceMemBridge),
        vad=_ctor(MockVad),
        audio_out=_NoopSpeaker(),
    )
    return pipeline


def _extract_latencies(timings: dict[str, float | None]) -> dict[str, float | None]:
    """TurnTimings.to_dict() -> per-stage latencies relative to speech_end_s."""
    speech_end = timings.get("speech_end_s")
    if speech_end is None:
        return {key: None for key in TIMESTAMP_KEYS}
    out: dict[str, float | None] = {}
    for key in TIMESTAMP_KEYS:
        value = timings.get(key)
        out[key] = None if value is None else round(value - speech_end, 4)
    out["memory_duration_s"] = timings.get("memory_s")  # already a duration
    return out


def _percentiles(values: list[float]) -> tuple[float, float]:
    """p50 and p95 via statistics.quantiles (inclusive percentile method)."""
    if not values:
        raise ValueError("no samples")
    if len(values) == 1:
        return (values[0], values[0])
    # n=100 -> cut points; [49] is the 50th, [94] the 95th percentile.
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    return (cuts[49], cuts[94])


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run N mock turns; return per-metric sample lists + percentile summary."""
    from app.config import AgentConfig

    cfg = AgentConfig()
    cfg.apply_env()
    if args.config and Path(args.config).exists():
        try:
            cfg = AgentConfig.from_yaml(Path(args.config))
        except (TypeError, ValueError, RuntimeError) as exc:
            print(f"WARN: YAML config load failed ({exc}); using env + defaults")

    pipeline = _build_mock_pipeline(cfg)
    try:
        import numpy as np  # lazy: only for the mock utterance buffer
    except ImportError as exc:
        raise RuntimeError("numpy is required to build mock utterance buffers") from exc
    audio = np.zeros(int(cfg.sample_rate * 1.0), dtype=np.float32)
    no_speech: Callable[[], float] = lambda: 0.0  # barge-in probe: silence

    samples: dict[str, list[float]] = {key: [] for key in TIMESTAMP_KEYS}
    samples["memory_duration_s"] = []
    for turn in range(args.turns):
        result = await pipeline.handle_utterance(audio, speaker_id="voice_user",
                                                 vad_prob_fn=no_speech)
        timings = result.timings.to_dict()
        latencies = _extract_latencies(timings)
        for key, value in latencies.items():
            if value is not None:
                samples[key].append(value)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 end-to-end latency benchmark")
    parser.add_argument("--turns", type=int, default=20, help="number of mock turns (default 20)")
    parser.add_argument("--config", help="optional YAML config (config/voicemem_config.yaml)")
    args = parser.parse_args()

    missing: list[str] = []
    for module in ("app.pipeline", "app.mock_components"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        print("The full pipeline is not available yet: " + ", ".join(missing))
        print("app.pipeline / app.mock_components arrive in Task 8 (integration).")
        print("Re-run this benchmark after the integration step.")
        return 2

    try:
        samples = asyncio.run(run(args))
    except (RuntimeError, TypeError) as exc:
        print(f"Could not assemble or run the mock pipeline: {exc}")
        print("This usually means the mock component constructors changed —")
        print("update _build_mock_pipeline() to match app/mock_components.py.")
        return 2

    print("=" * 72)
    print(f"M1 end-to-end latency benchmark ({args.turns} mock turns, speech end -> first audio)")
    print("=" * 72)
    print(f"{'stage':22s} {'n':>4s} {'p50 s':>8s} {'p95 s':>8s} {'max s':>8s}")
    first_audio = samples.get("first_audio_s", [])
    for key in ("first_audio_s", "asr_final_s", "memory_duration_s", "first_token_s",
                "full_response_s"):
        values = samples.get(key, [])
        if not values:
            print(f"{key:22s} {'0':>4s} {'n/a':>8s} {'n/a':>8s} {'n/a':>8s}")
            continue
        p50, p95 = _percentiles(values)
        print(f"{key:22s} {len(values):>4d} {p50:>8.3f} {p95:>8.3f} {max(values):>8.3f}")
    print("-" * 72)

    if len(first_audio) < 2:
        print("Not enough valid first_audio samples — check TurnTimings fields "
              "(speech_end_s / first_audio_s must be set on the same clock).")
        return 2
    p50, p95 = _percentiles(first_audio)
    passed = p50 < P50_LIMIT and p95 < P95_LIMIT
    print(f"first audio p50 = {p50:.3f} s (limit {P50_LIMIT})  "
          f"p95 = {p95:.3f} s (limit {P95_LIMIT})")
    print("=" * 72)
    print(f"RESULT: {'PASS' if passed else 'FAIL'} (milestones 7.6.4: p50 < 2.0 s, p95 < 2.5 s)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
