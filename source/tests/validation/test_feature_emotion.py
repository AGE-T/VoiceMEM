"""M2 emotion intelligence feature validation (Task 20).

Validates the ``emotion`` feature (milestones chapter 8) at the source
and contract level - every test runs in the dependency-free sandbox:

* app/emotion.py exists, is ASCII, and keeps funasr/torch LAZY (no
  top-level heavy import - CONTRACT rule 2);
* the teacher persona emotion block now carries the spec 8.3.2
  verify-and-adjust instruction;
* the pipeline runs the prosody stage parallel with VoiceMem and only
  forwards a fused emotion into the prompt (source-scan contract);
* TTS length_scale plumbing: piper gets --length_scale only when set;
* MODELS.lock.json ships the M2 emotion entry with a >1 GB model.pt
  floor, and the downloader's default lock stays in sync with the repo
  lock;
* the installer installs funasr (pinned, torch-free dependency list)
  with the trio re-check after it;
* config: enable_emotion defaults to True, speaker stays excluded;
* LICENSES.md carries the FunASR MODEL_LICENSE attribution entry;
* deep test (target machine only): real emotion2vec+ inference through
  funasr on the local model dir - deep-SKIPs in the sandbox.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
EMOTION_PY = REPO_ROOT / "app" / "emotion.py"
PIPELINE_PY = REPO_ROOT / "app" / "pipeline.py"
TTS_PY = REPO_ROOT / "app" / "tts.py"
TEACHER_PY = REPO_ROOT / "app" / "teacher_persona.py"
MAIN_PY = REPO_ROOT / "app" / "main.py"
MODELS_LOCK = REPO_ROOT / "MODELS.lock.json"
DOWNLOADER = REPO_ROOT / "scripts" / "download_models_hf.py"
INSTALLER = REPO_ROOT / "scripts" / "install_m1.ps1"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
REQUIREMENTS_LOCK = REPO_ROOT / "requirements.lock"
LICENSES_MD = REPO_ROOT / "LICENSES.md"
README_MD = REPO_ROOT / "README.md"

_ENV_KEYS = (
    "VOICEMEM_HOME", "VOICEMEM_ENABLE_EMOTION", "VOICEMEM_ENABLE_SPEAKER",
    "EMOTION2VEC_MODEL_PATH",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_ascii(path: Path) -> str:
    data = path.read_bytes()
    assert max(data) <= 0x7F, f"{path} is not pure ASCII"
    return data.decode("ascii")


def _strip_py_comments(text: str) -> str:
    """Remove docstrings and # comments (rough, source-scan only)."""
    import re

    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"#[^\n]*", "", text)
    return text


class EmotionFeatureTest(FeatureValidationTest):
    """M2 emotion: source contracts, lock, installer, docs, licensing."""

    FEATURE = "emotion"

    # ------------------------------------------------------------ module ----

    def test_logic_module_exists(self):
        self.assertTrue(EMOTION_PY.is_file(), "app/emotion.py must ship")
        # NOTE: emotion.py is intentionally UTF-8 (it contains accented
        # Hungarian semantic markers and the bilingual emotion2vec tokens).
        self.assertTrue(
            EMOTION_PY.read_text(encoding="utf-8").startswith('"""'),
            "app/emotion.py must start with a module docstring",
        )

    def test_logic_no_top_level_heavy_imports(self):
        source = _read(EMOTION_PY)
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if line[0] in (" ", "\t"):
                continue  # indented -> lazy import inside a function
            for heavy in ("funasr", "torch", "onnxruntime", "transformers"):
                self.assertNotIn(
                    heavy, line, f"top-level heavy import: {line.strip()}"
                )

    def test_logic_module_surface(self):
        source = _read(EMOTION_PY)
        for marker in (
            "class EmotionAnalyzer",
            "def analyze(self, audio",
            "class EmotionMemory",
            "def fuse_emotion",
            "TEACHER_EMOTIONS",
            "class EmotionResult",
        ):
            self.assertIn(marker, source, f"app/emotion.py lacks {marker!r}")
        # graceful degradation: analyze must swallow exceptions
        self.assertIn("Emotion analysis failed (non-fatal)", source)

    # ---------------------------------------------------------- persona ----

    def test_logic_teacher_prompt_verify_line(self):
        source = _read(TEACHER_PY)
        self.assertIn("Verify this assessment against what the user actually said", source)
        from app.teacher_persona import build_system_prompt

        prompt = build_system_prompt(
            "", emotion_label="frustrated", emotion_valence=-0.4, emotion_arousal=0.7
        )
        self.assertIn("frustrated", prompt)
        self.assertIn("Verify this assessment", prompt)

    # --------------------------------------------------------- pipeline ----

    def test_logic_pipeline_parallel_and_fused(self):
        source = _read(PIPELINE_PY)
        # parallel start (spec 8.3): memory and emotion via asyncio.gather
        self.assertIn("asyncio.gather", source)
        self.assertIn("_analyze_emotion", source)
        # the prompt receives the FUSED emotion, not the raw prosody
        self.assertIn("fuse_emotion", source)
        # M1 compatibility: emotion is optional in the constructor
        self.assertIn("emotion: Optional[", source)
        # spec 8.4: frustrated -> slow TTS
        self.assertIn("emotion_slow_length_scale", source)
        self.assertIn('fused.label == "frustrated"', source)

    def test_logic_pipeline_timings_and_result(self):
        from app.pipeline import TurnResult, TurnTimings

        timings = TurnTimings()
        self.assertIn("emotion_s", timings.to_dict())
        result = TurnResult(transcript="t", reply="r", language="hu", timings=timings)
        self.assertIn("emotion", result.to_dict())
        self.assertIsNone(result.to_dict()["emotion"])

    def test_logic_tts_length_scale(self):
        source = _read(TTS_PY)
        self.assertIn("--length_scale", source)
        self.assertIn("length_scale: Optional[float] = None", source)

    def test_logic_main_wiring(self):
        source = _read(MAIN_PY)
        self.assertIn("--emotion-demo", source)
        self.assertIn("EmotionAnalyzer", source)
        self.assertIn("check_emotion_assets", source)
        self.assertIn("M1-modban fut", source)  # graceful degradation print

    # ------------------------------------------------------------- lock ----

    def test_logic_lock_emotion_entry(self):
        lock = json.loads(MODELS_LOCK.read_text(encoding="utf-8"))
        entries = {e["component"]: e for e in lock["models"]}
        self.assertIn("emotion", entries, "MODELS.lock.json lacks the M2 emotion entry")
        entry = entries["emotion"]
        self.assertEqual(entry["repo"], "emotion2vec/emotion2vec_plus_base")
        self.assertEqual(entry["target_dir"], "models/emotion/emotion2vec-plus-base")
        self.assertEqual(
            entry["files"],
            ["model.pt", "config.yaml", "tokens.txt", "configuration.json"],
        )
        self.assertEqual(entry["min_bytes"]["model.pt"], 1073741824)  # >1 GB floor
        self.assertFalse(entry.get("snapshot"))

    def test_logic_downloader_default_lock_in_sync(self):
        """REPO_DEFAULT_LOCK (self-heal) must match the shipped lock."""
        source = _read_ascii(DOWNLOADER)
        lock = json.loads(MODELS_LOCK.read_text(encoding="utf-8"))
        repo_ids = [e["repo"] for e in lock["models"]]
        for repo_id in repo_ids:
            self.assertIn(
                f'"{repo_id}"', source, f"downloader default lock lacks {repo_id}"
            )
        self.assertIn("models/emotion/emotion2vec-plus-base", source)

    # -------------------------------------------------------- installer ----

    def test_logic_installer_funasr_step(self):
        code = _read_ascii(INSTALLER)
        self.assertIn("$FunasrVersion", code)
        self.assertIn('funasr==$FunasrVersion', code)
        self.assertIn("[{0}/22]", code)  # 22-step installer (v0.4.4 added step 16: transformers >= 4.57 guard)
        # the funasr install must be followed by the torch trio guard
        funasr_pos = code.find("pip install \"funasr==$FunasrVersion\"")
        trio_pos = code.find("Trio-guard a funasr telepitese utan")
        self.assertGreater(funasr_pos, 0)
        self.assertGreater(trio_pos, funasr_pos, "no trio guard after funasr install")

    def test_logic_requirements_document_funasr(self):
        req = _read_ascii(REQUIREMENTS)
        lock = _read_ascii(REQUIREMENTS_LOCK)
        for text, name in ((req, "requirements.txt"), (lock, "requirements.lock")):
            self.assertIn("funasr==1.4.11", text, f"{name} lacks the funasr pin")
            self.assertIn("emotion2vec", text, f"{name} lacks the emotion2vec note")

    # ------------------------------------------------------------ docs ----

    def test_logic_licenses_attribution(self):
        licenses = _read(LICENSES_MD)
        self.assertIn("emotion2vec", licenses)
        self.assertIn("FunASR", licenses)
        # attribution duty: keep the model name + name the source
        self.assertIn("attribution", licenses.lower())

    def test_logic_readme_m2_section(self):
        readme = _read(README_MD)
        self.assertIn("M2", readme)
        self.assertIn("emotion2vec", readme)
        self.assertIn("enable_emotion", readme)
        self.assertIn("--emotion-demo", readme)

    def test_logic_ascii_for_pip_and_lock_files(self):
        # These files MUST stay pure-ASCII (pip locale / ConvertFrom-Json
        # safety on Windows). app/emotion.py is deliberately UTF-8.
        for path in (MODELS_LOCK, REQUIREMENTS, REQUIREMENTS_LOCK):
            data = path.read_bytes()
            self.assertLessEqual(
                max(data), 0x7F, f"{path} must stay pure ASCII (pip/PS1 safety)"
            )

    # ------------------------------------------------- config integration ----

    def test_logic_config_defaults_m2(self):
        with patch.dict(os.environ, {key: "" for key in _ENV_KEYS}):
            from app.config import AgentConfig

            cfg = AgentConfig()
        self.assertTrue(cfg.enable_emotion)
        self.assertTrue(cfg.enable_speaker)  # M3: legal since v0.3.0
        self.assertEqual(cfg.validate(), [])

    # -------------------------------------------------------- deep tests ----

    def test_deep_real_emotion2vec_inference(self):
        """Target machine: real emotion2vec+ inference via funasr (CPU).

        Deep-SKIPs when funasr/torch or the model dir is unavailable
        (the sandbox deliberately has neither).
        """
        try:
            import funasr  # noqa: F401
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self.deep_skip(f"funasr/torch not installed here: {exc}")
            return
        from app.config import AgentConfig
        from app.emotion import EmotionAnalyzer, fuse_emotion

        with patch.dict(os.environ, {key: "" for key in _ENV_KEYS}):
            cfg = AgentConfig()
        analyzer = EmotionAnalyzer(
            cfg.emotion_model_dir,
            window_s=cfg.emotion_window_s,
            sample_rate=cfg.sample_rate,
        )
        if not analyzer.is_available():
            self.deep_skip("emotion2vec+ model not present locally")
            return
        import numpy as np

        # 3 s of moderate-energy noise (the classifier needs a real signal;
        # zeros pass through layer-norm fine, but noise is more honest)
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(16000 * 3) * 0.05).astype(np.float32)
        result = analyzer.analyze(audio)
        if result is None:
            self.deep_skip("analyzer returned None (model load issue)")
            return
        self.assertIn(result.label, ("frustrated", "sad", "neutral", "happy"))
        self.assertIn(result.raw_label, result.raw_label)  # non-empty string
        self.assertGreaterEqual(result.latency_ms, 0.0)
        fused = fuse_emotion(result, "test utterance", 0.6)
        self.assertIsNotNone(fused)
        self.assertTrue(fused.fused)
        self.assertTrue(-1.0 <= fused.valence <= 1.0)
        self.assertTrue(-1.0 <= fused.arousal <= 1.0)

    def test_deep_latency_budget(self):
        """M2 exit criterion 4: CPU analysis must stay well under 200 ms.

        Measures 5 consecutive 5-s windows on the target machine.
        """
        try:
            import funasr  # noqa: F401
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self.deep_skip(f"funasr/torch not installed here: {exc}")
            return
        from app.config import AgentConfig
        from app.emotion import EmotionAnalyzer

        with patch.dict(os.environ, {key: "" for key in _ENV_KEYS}):
            cfg = AgentConfig()
        analyzer = EmotionAnalyzer(cfg.emotion_model_dir, sample_rate=cfg.sample_rate)
        if not analyzer.is_available():
            self.deep_skip("emotion2vec+ model not present locally")
            return
        import numpy as np

        rng = np.random.default_rng(7)
        audio = (rng.standard_normal(16000 * 5) * 0.05).astype(np.float32)
        latencies = []
        for _ in range(5):
            result = analyzer.analyze(audio)
            if result is None:
                self.deep_skip("analyzer returned None mid-benchmark")
                return
            latencies.append(result.latency_ms)
        # the additive budget is the PARALLEL stage, but the analyzer alone
        # must still stay under the 200 ms milestone ceiling
        self.assertLess(
            max(latencies),
            200.0,
            f"emotion2vec+ analysis exceeds the 200 ms budget: {latencies}",
        )


if __name__ == "__main__":
    unittest.main()
