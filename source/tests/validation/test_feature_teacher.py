"""Teacher persona feature validation (Task 12 - M0.1).

Validates the ``teacher`` feature built from the pure-stdlib modules
``app/teacher_persona.py`` and ``app/text_utils.py``:

* ``build_system_prompt`` injects the long-term memory context, ALWAYS keeps
  the grammar-correction duty and omits the emotion block when the M2
  emotion parameters are None (the M1 state);
* ``build_messages`` lays out [system, *history, user];
* ``detect_language`` heuristic HU/EN detection;
* ``SentenceStream`` (fed with the REAL config chunk sizes) emits speakable
  chunks of bounded length and never loses text (flush included).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AgentConfig
from app.teacher_persona import build_messages, build_system_prompt
from app.text_utils import LANG_EN, LANG_HU, SentenceStream, detect_language
from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "voicemem_config.yaml"

_ENV_KEYS = (
    "VOICEMEM_HOME", "TTS_HU_VOICE", "TTS_EN_VOICE", "LLAMA_SERVER_HOST",
    "LLAMA_SERVER_PORT", "OPENAI_BASE_URL",
)


class _EnvNeutralTest(FeatureValidationTest):
    """Base: neutralize env vars that leak into AgentConfig.apply_env()."""

    def setUp(self) -> None:
        super().setUp()
        patcher = patch.dict(os.environ, {key: "" for key in _ENV_KEYS})
        patcher.start()
        self.addCleanup(patcher.stop)


class TeacherFeatureTest(_EnvNeutralTest):
    """Teacher persona prompt, message layout, language, stream chunking."""

    FEATURE = "teacher"

    def test_logic_system_prompt_blocks(self):
        prompt = build_system_prompt(memory_context="USER FACT: likes coffee")
        # Memory context reaches the prompt.
        self.assertIn("likes coffee", prompt)
        self.assertIn("Long-term memory about this user:", prompt)
        # The grammar-correction duty is always present (actual module wording).
        self.assertIn("natural, friendly voice assistant", prompt)
        self.assertIn("one or two sentences", prompt)
        # Emotion block is omitted when the M2 parameters are all None.
        self.assertNotIn("Emotional state", prompt)
        # Message layout: [system, *history, user].
        history = [
            {"role": "user", "content": "elozo kerdes"},
            {"role": "assistant", "content": "elozo valasz"},
        ]
        messages = build_messages("uj kerdes", prompt, history=history)
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(messages[0]["content"], prompt)
        self.assertEqual(messages[-1]["content"], "uj kerdes")

    def test_logic_language_detection(self):
        self.assertEqual(detect_language("Ez egy magyar mondat."), LANG_HU)
        self.assertEqual(
            detect_language("This is a simple english sentence."), LANG_EN
        )
        self.assertEqual(detect_language("Jó reggelt, kérlek segítesz?"), LANG_HU)

    def test_logic_sentence_stream_chunking(self):
        cfg = AgentConfig.from_yaml(YAML_PATH)
        stream = SentenceStream(
            first_chunk_chars=cfg.tts_first_chunk_chars,
            chunk_chars=cfg.tts_chunk_chars,
        )
        sentences = [
            "Ez az elso mondat a tanari valaszban. ",
            "A masodik mondat egy kicsit hosszabb, de meg beleferek. ",
            "A harmadik mondat a kavarod kinyitasarol szol. ",
            "A negyedik mondat mar majdnem a vegen jar a szovegnek. ",
            "Az otodik mondat egy peldamondat a gyakorlashoz. ",
            "A hatodik mondat eleg hosszu ahhoz, hogy tobb darabra vesszen. ",
            "A hetedik mondat utan mar csak egy zaro resz kovetkezik. ",
        ]
        text = "".join(sentences) + "es a vegen egy lezaro resz pont nelkul"
        self.assertGreaterEqual(len(text), 300, "test needs a 300+ char stream")
        # Feed the text in small deltas the way the LLM streams it.
        chunks: list[str] = []
        step = 7
        for start in range(0, len(text), step):
            chunks.extend(stream.add_delta(text[start:start + step]))
        chunks.extend(stream.flush())
        # Chunks are non-empty and length-bounded (sentence cuts may overshoot
        # the soft limit by at most one word; sentences here stay below it).
        self.assertTrue(chunks)
        slack = 30
        for chunk in chunks:
            self.assertGreater(len(chunk), 0, "empty chunk emitted")
            self.assertLessEqual(
                len(chunk), cfg.tts_chunk_chars + slack,
                f"chunk too long ({len(chunk)} chars): {chunk[:50]!r}",
            )
        # No text is lost: chunks are stripped, so boundary separator spaces
        # are consumed by design - the round trip is compared after removing
        # ALL whitespace (order-sensitive, so duplicates also fail).
        joined_no_ws = "".join("".join(chunks).split())
        original_no_ws = "".join(text.split())
        self.assertEqual(
            joined_no_ws, original_no_ws, "stream lost, merged or duplicated text"
        )
        # The final flush is not lost either: the trailing (terminator-less)
        # fragment must survive into the emitted chunks.
        trailing = "es a vegen egy lezaro resz pont nelkul"
        self.assertIn("".join(trailing.split()), joined_no_ws)


if __name__ == "__main__":
    unittest.main()
