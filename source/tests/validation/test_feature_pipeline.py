"""Pipeline feature validation (Task 12 - M0.1).

Validates the ``pipeline`` feature at the feature level (the full deep matrix
lives in ``tests/integration/test_pipeline_mock.py``): ONE focused mock turn
through ``app.pipeline.VoicePipeline`` assembled from the mock components of
``app/mock_components.py`` must produce a ``TurnResult`` with a non-empty
transcript and reply, a valid language, recorded timings, and must commit the
complete reply to the memory mock.

Also validates the CLI asset checklist: ``python -m app.main --check`` runs
as a subprocess from the repo root with a neutralized environment. NOTE: the
CLI's documented contract (app/main.py docstring) is "exit 0/1" - exit 1
means missing runtime assets, which is the expected fresh-sandbox state; the
checklist output itself is asserted in full.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from app.mock_components import (
    MockAsrEngine,
    MockLlmClient,
    MockSpeaker,
    MockTtsEngine,
    MockVad,
    MockVoiceMemBridge,
)
from app.pipeline import TurnResult, TurnTimings, VoicePipeline
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_KEYS = (
    "VOICEMEM_HOME", "VOICEMEM_MEMORY_ROOT", "VOICEMEM_EMBED_DIM",
    "PIPER_VOICES_PATH", "PIPER_EXECUTABLE", "TTS_HU_VOICE", "TTS_EN_VOICE",
    "SILERO_VAD_PATH", "QWEN3_ASR_MODEL_PATH", "LLAMA_MODEL_PATH",
    "LLAMA_SERVER_HOST", "LLAMA_SERVER_PORT", "OPENAI_BASE_URL", "OPENAI_MODEL",
)

REPLY = (
    "Rovid valasz a kerdesre. "
    "Ez a masodik mondat, hogy tobb darabra tortenjen a valasz. "
    "Es ez a harmadik zaro mondat."
)

#: Every asset key the --check checklist must report.
CHECK_ASSET_KEYS = (
    "llama_model", "piper_executable", "silero_vad", "hu_voice",
    "en_voice", "asr_model",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class PipelineFeatureTest(_EnvNeutralTest):
    """One full mock turn end to end (ASR -> memory -> LLM -> TTS -> commit)."""

    FEATURE = "pipeline"

    def test_logic_full_mock_turn(self):
        cfg = AgentConfig()
        asr = MockAsrEngine(["Szia, hogy vagy ma?"])
        llm = MockLlmClient([REPLY])
        tts = MockTtsEngine()
        voicemem = MockVoiceMemBridge(["- USER FACT: likes coffee"])
        speaker = MockSpeaker()
        pipeline = VoicePipeline(
            cfg,
            asr=asr,
            llm=llm,
            tts=tts,
            voicemem=voicemem,
            vad=MockVad(),
            audio_out=speaker,
        )
        audio = np.zeros(int(cfg.sample_rate * 1.0), dtype=np.float32)
        result = asyncio.run(
            pipeline.handle_utterance(audio, vad_prob_fn=lambda: 0.0)
        )
        self.assertIsInstance(result, TurnResult)
        self.assertEqual(result.transcript, "Szia, hogy vagy ma?")
        self.assertGreater(len(result.reply), 0, "reply must not be empty")
        self.assertEqual(result.reply.strip(), REPLY.strip())
        self.assertIn(result.language, ("hu", "en"))
        self.assertIsInstance(result.timings, TurnTimings)
        self.assertIsNotNone(result.timings.full_response_s)
        self.assertFalse(result.cancelled)
        self.assertFalse(result.barge_in)
        # The complete reply was committed to long-term memory.
        self.assertEqual(voicemem.committed, [result.reply])

    def test_logic_cli_asset_check(self):
        """python -m app.main --check: full checklist, exit code 0/1."""
        env = dict(os.environ)
        env.update({key: "" for key in _ENV_KEYS})
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, "-m", "app.main", "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        # Exit contract (app/main.py): 0 = all assets present, 1 = missing
        # assets (the normal fresh-sandbox state; a crash would be 2+).
        self.assertIn(
            proc.returncode, (0, 1),
            f"--check crashed (rc={proc.returncode}): {proc.stderr[-400:]}",
        )
        self.assertIn("root", proc.stdout, "checklist must print the config root")
        for key in CHECK_ASSET_KEYS:
            self.assertIn(key, proc.stdout, f"checklist must list the asset '{key}'")


if __name__ == "__main__":
    unittest.main()
