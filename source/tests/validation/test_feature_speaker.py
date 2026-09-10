"""M3 speaker recognition feature validation (Task 21).

Validates the ``speaker`` feature (milestones chapter 9) at the source
and contract level - every test runs in the dependency-free sandbox:

* app/speaker.py exists, is ASCII, and keeps speechbrain/torch LAZY (no
  top-level heavy import - CONTRACT rule 2);
* the pipeline resolves the speaker BEFORE the VoiceMem retrieval
  (source-scan contract) and runs the embedding parallel with the
  emotion start; the resolved user_id routes retrieval + commit +
  emotion facts;
* the bridge keeps ONE VoiceMem facade per user_id (9.3.4 memory-space
  separation);
* MODELS.lock.json ships the M3 speaker entry (5 runtime files, 80 MB
  embedding floor) and the downloader's default lock stays in sync;
* the installer installs speechbrain (pinned, trio-safe) as step 14 and
  the doc header is 20-step;
* config: enable_speaker defaults to True (9.5 flag), the threshold /
  window / registration fields exist with spec defaults;
* main.py: --speaker-demo + --register-speaker CLI contract;
* LICENSES.md carries the Apache-2.0 speechbrain rows;
* deep test (target machine / sandbox with model): real ECAPA inference
  from the LOCAL model dir - 192-dim, self-similarity 1.0, CPU latency
  budget; deep-SKIPs when speechbrain or the model is unavailable.
"""
from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEAKER_PY = REPO_ROOT / "app" / "speaker.py"
PIPELINE_PY = REPO_ROOT / "app" / "pipeline.py"
BRIDGE_PY = REPO_ROOT / "app" / "voicemem_bridge.py"
MAIN_PY = REPO_ROOT / "app" / "main.py"
CONFIG_PY = REPO_ROOT / "app" / "config.py"
MODELS_LOCK = REPO_ROOT / "MODELS.lock.json"
DOWNLOADER = REPO_ROOT / "scripts" / "download_models_hf.py"
INSTALLER = REPO_ROOT / "scripts" / "install_m1.ps1"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
REQUIREMENTS_LOCK = REPO_ROOT / "requirements.lock"
LICENSES_MD = REPO_ROOT / "LICENSES.md"
README_MD = REPO_ROOT / "README.md"

_ENV_KEYS = (
    "VOICEMEM_HOME", "VOICEMEM_ENABLE_EMOTION", "VOICEMEM_ENABLE_SPEAKER",
    "SPEAKER_MODEL_PATH", "VOICEMEM_SPEAKER_THRESHOLD",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _lock_entry(component: str) -> dict:
    lock = json.loads(_read(MODELS_LOCK))
    for entry in lock["models"]:
        if entry.get("component") == component:
            return entry
    raise AssertionError(f"no {component} entry in MODELS.lock.json")


class SpeakerFeatureTest(FeatureValidationTest):
    FEATURE = "speaker"

    # ------------------------------------------------------- source level --

    def test_logic_module_exists_ascii_lazy(self):
        self.assertTrue(SPEAKER_PY.is_file(), "app/speaker.py missing")
        raw = SPEAKER_PY.read_bytes()
        self.assertLessEqual(max(raw), 0x7F, "app/speaker.py must be pure ASCII")
        for line in _read(SPEAKER_PY).splitlines():
            if line.startswith(("import torch", "import speechbrain",
                                "from torch", "from speechbrain")):
                self.fail(f"module-level heavy import in app/speaker.py: {line}")

    def test_logic_speaker_stage_precedes_memory_in_pipeline(self):
        """Spec 9.3: user_id resolution happens BEFORE the VoiceMem call."""
        src = _read(PIPELINE_PY)
        self.assertIn("_identify_speaker", src)
        identify_pos = src.find("identification = await self._identify_speaker")
        retrieve_pos = src.find("await self._retrieve_memory(transcript, speaker_id")
        self.assertGreater(
            identify_pos, 0, "pipeline does not call _identify_speaker"
        )
        self.assertGreater(
            retrieve_pos, 0, "pipeline does not call _retrieve_memory with speaker_id"
        )
        self.assertLess(
            identify_pos, retrieve_pos,
            "speaker identification must precede the VoiceMem retrieval",
        )
        # the resolved id overrides the passed speaker_id (9.5.2)
        self.assertIn("speaker_id = identification.id", src)
        # emotion + speaker start in PARALLEL (create_task before the await)
        create_task_pos = src.find("asyncio.create_task(self._analyze_emotion")
        self.assertGreater(create_task_pos, 0, "emotion task creation missing")
        self.assertLess(
            create_task_pos, identify_pos,
            "the emotion task must START before the speaker await (parallel)",
        )
        # commit + emotion facts route with the RESOLVED speaker_id
        # (v0.4.9: the commit goes through the retry wrapper, which calls
        # commit_reply(reply, speaker_id=speaker_id) with the SAME resolved id)
        self.assertIn("commit_reply(reply, speaker_id=speaker_id)", src)

    def test_logic_bridge_keeps_per_user_facades(self):
        """9.3.4: one VoiceMem facade per user_id."""
        src = _read(BRIDGE_PY)
        self.assertIn("_vm_by_user", src)
        self.assertIn("_construct_vm, key", src)
        self.assertIn('user_id=user_id', src)
        # every public routing point passes the speaker_id through
        for anchor in (
            "vm = await self._ensure_vm(speaker_id)",
        ):
            count = src.count(anchor)
            self.assertGreaterEqual(
                count, 3,
                f"expected >=3 per-speaker _ensure_vm routing points "
                f"(process_turn/commit/store_fact), found {count}",
            )

    def test_logic_bridge_cancel_iterates_all_facades(self):
        src = _read(BRIDGE_PY)
        self.assertIn("for vm in self._vm_by_user.values():", src)

    # ------------------------------------------------------------- config --

    def test_logic_config_defaults_m3(self):
        with patch.dict(os.environ, {key: "" for key in _ENV_KEYS}):
            from app.config import AgentConfig

            cfg = AgentConfig()
        self.assertTrue(cfg.enable_speaker, "enable_speaker must default True in M3")
        self.assertEqual(cfg.speaker_match_threshold, 0.5)
        self.assertEqual(cfg.speaker_window_s, 5.0)
        self.assertEqual(cfg.speaker_registration_min_s, 10.0)
        self.assertEqual(
            cfg.speaker_model_name, "speechbrain/spkrec-ecapa-voxceleb"
        )
        self.assertEqual(cfg.validate(), [])
        self.assertTrue(
            cfg.speaker_model_dir.as_posix().endswith("models/speaker/ecapa-voxceleb")
        )
        self.assertTrue(
            cfg.speaker_registry_file.as_posix().endswith(
                "data/speaker_registry.json"
            )
        )
        self.assertIn("speaker_model", cfg.check_speaker_assets())

    def test_logic_config_env_off_switch(self):
        with patch.dict(
            os.environ, {**{key: "" for key in _ENV_KEYS},
                         "VOICEMEM_ENABLE_SPEAKER": "0"}
        ):
            from app.config import AgentConfig

            cfg = AgentConfig()
            cfg.apply_env()
        self.assertFalse(cfg.enable_speaker)

    # --------------------------------------------------------------- lock --

    def test_logic_models_lock_speaker_entry(self):
        entry = _lock_entry("speaker")
        self.assertEqual(entry["repo"], "speechbrain/spkrec-ecapa-voxceleb")
        self.assertEqual(entry["target_dir"], "models/speaker/ecapa-voxceleb")
        self.assertEqual(
            entry["files"],
            [
                "embedding_model.ckpt",
                "hyperparams.yaml",
                "mean_var_norm_emb.ckpt",
                "classifier.ckpt",
                "label_encoder.txt",
            ],
        )
        self.assertFalse(entry.get("snapshot", False))
        self.assertGreaterEqual(
            entry["min_bytes"]["embedding_model.ckpt"], 80_000_000,
            "the ECAPA embedding floor must guard a thin/failed download",
        )

    def test_logic_downloader_default_lock_in_sync(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from download_models_hf import REPO_DEFAULT_LOCK

        repo = json.loads(_read(MODELS_LOCK))
        a = {
            m["component"]: (m["repo"], m["target_dir"], sorted(m["files"]))
            for m in repo["models"]
        }
        b = {
            m["component"]: (m["repo"], m["target_dir"], sorted(m.get("files", [])))
            for m in REPO_DEFAULT_LOCK["models"]
        }
        self.assertEqual(a, b, "REPO_DEFAULT_LOCK drifted from MODELS.lock.json")

    # ---------------------------------------------------------- installer --

    def test_logic_installer_step14_speechbrain(self):
        src = _read(INSTALLER)
        self.assertIn('$SpeechbrainVersion = "1.1.1"', src)
        self.assertIn("# 15) speechbrain telepitese", src)
        self.assertIn('pip install "speechbrain==$SpeechbrainVersion"', src)
        self.assertIn(
            "speechbrain telepitese utan is (feluliras-ellenorzes)",
            src,
            "the trio-guard must re-run after the speechbrain install",
        )
        # 22 steps now (v0.4.4 added the transformers >= 4.57 guard; was 21)
        self.assertIn("# 22) Vegso install report", src)
        self.assertIn("# 16) transformers OR", src)
        self.assertIn('pip install "transformers>=$TransformersFloor"', src)

    def test_logic_requirements_mention_speechbrain(self):
        self.assertIn("speechbrain", _read(REQUIREMENTS))
        self.assertIn("speechbrain", _read(REQUIREMENTS_LOCK))
        # the requirement text documents the torch-safety property
        self.assertIn("torch>=2.1.0", _read(REQUIREMENTS_LOCK))

    # --------------------------------------------------------------- CLI --

    def test_logic_main_cli_surface(self):
        src = _read(MAIN_PY)
        self.assertIn("--speaker-demo", src)
        self.assertIn("--register-speaker", src)
        self.assertIn("run_register_speaker", src)
        # the real mode wires the recognizer + registry
        self.assertIn("SpeakerRecognizer(", src)
        self.assertIn("SpeakerRegistry(config.speaker_registry_file)", src)
        # degraded paths exist (9.5 items 5/6)
        self.assertIn("beszélő-felismerő nem elérhető", src)

    # ------------------------------------------------------------ licences --

    def test_logic_licenses_covers_speechbrain(self):
        src = _read(LICENSES_MD)
        self.assertIn("spkrec-ecapa-voxceleb", src)
        self.assertIn("speechbrain", src.lower())
        self.assertIn("Apache-2.0", src)

    def test_logic_readme_has_m3_chapter(self):
        src = _read(README_MD)
        self.assertIn("M3", src)
        self.assertIn("beszélő", src.lower())

    # ------------------------------------------------------- deep (target) --

    def test_deep_real_ecapa_inference(self):
        """Target machine: real ECAPA embedding from the LOCAL model dir.

        Deep-SKIPs when speechbrain or the model dir is unavailable (the
        development sandbox has neither in the DEFAULT test environment;
        the live sandbox proof ran in a throwaway venv - see worklog).
        """
        try:
            import speechbrain  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self.deep_skip(f"speechbrain not importable: {exc}")
            return

        with patch.dict(os.environ, {key: "" for key in _ENV_KEYS}):
            from app.config import AgentConfig
            from app.speaker import SpeakerEmbedder

            cfg = AgentConfig()

        embedder = SpeakerEmbedder(
            cfg.speaker_model_dir,
            window_s=cfg.speaker_window_s,
            sample_rate=cfg.sample_rate,
        )
        if not embedder.files_present():
            self.deep_skip(
                f"ECAPA model not downloaded: {cfg.speaker_model_dir}"
            )
            return
        if not embedder.warm_up():
            self.fail("ECAPA model files present but the load failed")

        import numpy as np

        audio = np.random.default_rng(42).standard_normal(
            16000 * 5
        ).astype(np.float32)
        start = time.perf_counter()
        first = embedder.embed(audio)
        embed_ms = (time.perf_counter() - start) * 1000.0
        self.assertIsNotNone(first, "embedding returned None with a warm model")
        assert first is not None
        self.assertEqual(len(first), 192, "ECAPA embedding must be 192-dim")

        second = embedder.embed(audio)
        assert second is not None
        from app.speaker import cosine_similarity

        self.assertAlmostEqual(
            cosine_similarity(first, second), 1.0, places=5,
            msg="the same audio must embed identically",
        )
        # Sandbox CPU is slow (measured ~850 ms on 2 cores); the TARGET
        # budget is the 100 ms additive M3 rule (latency benchmark owns
        # the end-to-end measurement) - this bound catches pathological
        # regressions (wrong device, thread thrashing).
        self.assertLess(embed_ms, 5000.0, embed_ms)
        self.deep_pass(
            f"ECAPA 192-dim self-consistency OK ({embed_ms:.0f} ms CPU)"
        )


if __name__ == "__main__":
    unittest.main()
