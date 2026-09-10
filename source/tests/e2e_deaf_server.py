#!/usr/bin/env python3
"""v0.4.14 E2E launcher: the DEMO web backend with the REPORTER'S channel.

The v0.4.13 field report channel, replayed deterministically:

* the VAD primary is DEAF (Silero probability 0.003 on every frame — the
  reporter's line-in through the browser; the v0.4.13 screenshot's
  "VAD peak 0.003 / 0.25, would fire: no, 0/190 frames"), wrapped in the
  v0.4.14 FusedVad (the energy fallback gate) exactly as make_vad() does
  in real mode;
* the ASR engine is a MockAsrEngine with a seeded Hungarian transcript
  queue, so a live utterance produces a REAL final transcript (in real
  mode Qwen3-ASR does this);
* everything else is the standard mock backend (demo LLM + demo TTS).

Run:  python3 tests/e2e_deaf_server.py [port]
Drive: tests/e2e_fake_mic.js-style main-world injection (see the browser
E2E): Start talking -> speak-level audio -> the turn must dispatch
WITHOUT the "Send to the agent" button.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import AgentConfig  # noqa: E402
from app.vad import FusedVad  # noqa: E402
from app.web_server import WebComponents, build_web_app  # noqa: E402


class _DeafSilero:
    """The field-reported channel: Silero says 0.003 on speech-level audio."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def prob(self, frame) -> float:  # noqa: ANN001 - numpy array
        return 0.003

    def reset(self) -> None:
        self.reset_calls += 1

    def is_available(self) -> bool:
        return True


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    cfg = AgentConfig(root=ROOT)
    components = WebComponents(cfg, mock=True).build()
    deaf = _DeafSilero()
    # the WS route builds the per-session VAD through components.make_vad()
    components.make_vad = lambda: FusedVad(deaf, cfg)  # type: ignore[method-assign]
    from app.mock_components import MockAsrEngine

    components.make_asr = lambda: MockAsrEngine(  # type: ignore[method-assign]
        queue=[
            "Szia! Hogy telt a napod?",
            "Most válaszolj angolul.",
        ]
    )
    app = build_web_app(components)
    import uvicorn

    print(f"[e2e] deaf-Silero demo backend on http://127.0.0.1:{port}/", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
