"""Model asset feature validation (Task 12 - M0.1).

Validates the on-disk M0 Section 7 model layout under ``models/``:

* the expected component directories exist (asr/llm/tts/vad/embedding/hf and
  the M2 emotion placeholder);
* the M1 exclusion policy is enforced: ``models/emotion`` must stay EMPTY
  (repo docs only) and no banned M2/M3/V2 component (emotion2vec, SpeechBrain
  ECAPA, MOSS TTS, scene recognition, Gemma audio, Qwen Omni) may appear as a
  directory or file name under the component dirs (the ``models/hf`` cache
  root is excluded from the ban scan - it may legally hold HF cache dirs).

Deep check: when the models are downloaded (target machine after
``download_models.ps1``), the actual files are verified (existence, size,
safetensors presence, piper voice pairs).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

#: M0 Section 7 component layout (repo-side, always required).
EXPECTED_DIRS = (
    "asr/qwen3-asr-0.6b",
    "llm/qwen3.6-35b-a3b",
    "tts/piper",
    "vad/silero-vad",
    "embedding/multilingual-e5-small",
    "hf",
    "emotion",
    "speaker",
)

#: Only repo documentation files may live in the M2 placeholder dir.
EMOTION_ALLOWED_FILES = frozenset({"README.md", ".gitkeep"})

#: The M2 model directory (emotion2vec+ base) - the ONLY legal payload of
#: models/emotion after the M2 milestone (v0.2.0).
EMOTION_MODEL_DIR = "emotion2vec-plus-base"

#: The M3 model directory (SpeechBrain ECAPA) - the ONLY legal payload of
#: models/speaker after the M3 milestone (v0.3.0).
SPEAKER_MODEL_DIR = "ecapa-voxceleb"

#: The 5 runtime files the M3 lock downloads into models/speaker/ecapa-voxceleb
#: (hyperparams.yaml pretrainer references all of them).
SPEAKER_MODEL_FILES = (
    "embedding_model.ckpt",
    "hyperparams.yaml",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
)

#: V2 components that must NEVER appear under models/ (M0 Section 25,
#: M3-alapjan frissitve: az emotion2vec es a speechbrain ECAPA mar LEGALIS).
#: v0.4.16: a "gemma" token VISZSAKERULT a tiltott tokenek koze - a Gemma
#: fallback profilt ELTAVOLITOTTUK (a Qwen3.6 35B A3B IQ4_XS az EGYETLEN
#: LLM; a gepen marade regi Gemma fajlok mar nem hasznalhatok futtasra).
BANNED_NAME_TOKENS = (
    "moss", "scene", "omni", "gemma",
)

#: HF cache root: excluded from the ban scan (may hold cached model dirs).
HF_CACHE_DIR = "hf"


class ModelsLayoutFeatureTest(FeatureValidationTest):
    """Directory layout of the M0 Section 7 model tree."""

    FEATURE = "models"

    def test_logic_component_directories_exist(self):
        for rel in EXPECTED_DIRS:
            path = MODELS_DIR / rel
            self.assertTrue(path.is_dir(), f"models/{rel} directory missing")

    def test_deep_downloaded_model_files(self):
        """Verify the downloaded model files when they are present."""
        asr_dir = MODELS_DIR / "asr" / "qwen3-asr-0.6b"
        if not (asr_dir / "config.json").is_file():
            self.deep_skip("ASR model not downloaded")
        verified: list[str] = []
        # ASR: config + at least one safetensors shard.
        self.assertTrue((asr_dir / "config.json").is_file(), "ASR config.json missing")
        safetensors = sorted(asr_dir.glob("*.safetensors"))
        self.assertGreater(
            len(safetensors), 0, "no *.safetensors file in the ASR model dir"
        )
        verified.append(f"asr config+{len(safetensors)} safetensors")
        # LLM (v0.4.16): the operator-placed Qwen3.6 35B A3B IQ4_XS GGUF
        # (~19 GB). The file is OPTIONAL on a dev machine (the web UI model
        # picker / verify_m1 gate handle a missing file with guidance);
        # the canonical LAYOUT is pinned here.
        llm_dir = MODELS_DIR / "llm" / "qwen3.6-35b-a3b"
        self.assertTrue(llm_dir.is_dir(), "models/llm/qwen3.6-35b-a3b dir missing")
        llm_file = llm_dir / "Qwen3.6-35B-A3B-IQ4_XS.gguf"
        if llm_file.is_file():
            self.assertGreater(
                llm_file.stat().st_size, 6 * 10 ** 9,
                "LLM GGUF must be > 6 GB (Qwen3.6 35B IQ4_XS is ~19 GB)",
            )
            verified.append("llm qwen gguf > 6 GB")
        # VAD: silero ONNX, > 1 MB.
        vad_file = MODELS_DIR / "vad" / "silero-vad" / "silero_vad.onnx"
        self.assertTrue(vad_file.is_file(), "silero_vad.onnx missing")
        self.assertGreater(vad_file.stat().st_size, 1024 ** 2, "silero_vad.onnx < 1 MB")
        verified.append("vad onnx > 1 MB")
        # Embedding: config.json present.
        embed_dir = MODELS_DIR / "embedding" / "multilingual-e5-small"
        self.assertTrue((embed_dir / "config.json").is_file(), "e5 config.json missing")
        verified.append("embedding config.json")
        # Piper voices: the two selected voices + their .onnx.json configs.
        piper_dir = MODELS_DIR / "tts" / "piper"
        for voice in ("hu_HU-anna-medium", "en_US-lessac-medium"):
            onnx = piper_dir / f"{voice}.onnx"
            conf = piper_dir / f"{voice}.onnx.json"
            self.assertTrue(onnx.is_file(), f"piper voice missing: {voice}.onnx")
            self.assertTrue(conf.is_file(), f"piper voice config missing: {voice}.onnx.json")
        verified.append("piper hu+en voices")
        # Optional extra HU voices: validated only when present.
        for voice in ("hu_HU-berta-medium", "hu_HU-imre-medium"):
            onnx = piper_dir / f"{voice}.onnx"
            if onnx.is_file():
                self.assertTrue(
                    (piper_dir / f"{voice}.onnx.json").is_file(),
                    f"piper voice {voice} exists without its .onnx.json",
                )
                verified.append(f"piper {voice}")
        self.deep_pass("model files verified: " + ", ".join(verified))


class ModelsBanFeatureTest(FeatureValidationTest):
    """M2-era exclusion policy: only emotion2vec+ may live in models/emotion.

    M0/M1 kept models/emotion EMPTY; since M2 (v0.2.0) the emotion2vec+
    base model is a LEGAL, lock-downloaded payload there (graceful
    degradation keeps an empty clone runnable). M3/V2 components stay
    banned everywhere.
    """

    FEATURE = "models-ban"

    def test_logic_emotion_dir_only_holds_the_m2_model(self):
        emotion_dir = MODELS_DIR / "emotion"
        self.assertTrue(emotion_dir.is_dir(), "models/emotion placeholder missing")
        offenders = sorted(
            p.name
            for p in emotion_dir.iterdir()
            if p.name not in EMOTION_ALLOWED_FILES and p.name != EMOTION_MODEL_DIR
        )
        self.assertEqual(
            offenders, [],
            f"models/emotion may only hold the M2 model dir (found: {offenders})",
        )
        model_dir = emotion_dir / EMOTION_MODEL_DIR
        if model_dir.is_dir():
            # downloaded state: the 4 runtime files from MODELS.lock.json
            for name in ("model.pt", "config.yaml", "tokens.txt"):
                self.assertTrue(
                    (model_dir / name).is_file() or not any(model_dir.iterdir()),
                    f"partial emotion model: {name} missing (re-run download_models)",
                )

    def test_logic_no_banned_component_names(self):
        """No banned model name in any dir/file under the component roots.

        The ``models/hf`` cache root is skipped (HF cache dirs are legal);
        everything below the other depth-1 component dirs is scanned
        recursively, both directory and file names.
        """
        for component_dir in sorted(MODELS_DIR.iterdir()):
            if not component_dir.is_dir() or component_dir.name == HF_CACHE_DIR:
                continue
            for path in component_dir.rglob("*"):
                name = path.name.lower()
                for token in BANNED_NAME_TOKENS:
                    self.assertNotIn(
                        token, name,
                        f"banned M3/V2 component '{token}' found: {path}",
                    )


class ModelDownloadFeatureTest(FeatureValidationTest):
    """Model downloader contract (v0.1.7: huggingface_hub Python API).

    The download step must be lock-driven (MODELS.lock.json), retry and
    resume transfers, honour pinned_revision, download straight into the
    project models/ tree, and be successful ONLY when the mandatory files
    actually exist with their size floors. No Hugging Face CLI anywhere
    (v0.1.6 blocker: the CLI's emoji deprecation warning crashed the child
    Python with UnicodeEncodeError under a CP1252 Windows console).
    """

    FEATURE = "models-download"

    HELPER = REPO_ROOT / "scripts" / "download_models_hf.py"

    def test_logic_helper_contract(self):
        raw = self.HELPER.read_bytes()
        self.assertLessEqual(
            max(raw), 0x7F,
            "download_models_hf.py must be pure ASCII (any Windows codepage)",
        )
        code = raw.decode("ascii")
        for marker in (
            "hf_hub_download(",           # per-file Python API download
            "snapshot_download(",         # full-repo Python API download
            "local_dir",                  # direct local destination
            "with_retries",               # retry with backoff
            "pinned_revision",            # revision support
            "min_bytes",                  # per-file size floor
            "min_total_mb",               # snapshot payload floor
            "MODELS.lock.json",           # lock-driven
            "ALL_MODELS_OK",              # unambiguous success marker
            "MODELS_FAILED",              # unambiguous failure marker
        ):
            self.assertIn(
                marker, code,
                f"download_models_hf.py missing contract marker {marker!r}",
            )
        # No CLI artifacts inside the helper either (the download must run
        # in-process - no subprocess module, no shell-out to any CLI).
        for token in ("huggingface-cli", "import subprocess", "os.system",
                      "os.popen"):
            self.assertNotIn(
                token, code,
                f"download_models_hf.py must not use {token!r} (CLI-free)",
            )

    def test_deep_verify_only_matches_presence(self):
        """--verify-only must exit 0 iff the locked model set is on disk."""
        import subprocess

        if not (MODELS_DIR / "asr" / "qwen3-asr-0.6b" / "config.json").is_file():
            self.deep_skip("models not downloaded on this machine")
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            self.deep_skip("huggingface_hub not installed")
        proc = subprocess.run(
            [sys.executable, str(self.HELPER), "--root", str(REPO_ROOT),
             "--verify-only"],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(
            proc.returncode, 0,
            f"--verify-only failed (models are on disk): {proc.stdout}",
        )
        self.assertIn("ALL_MODELS_OK", proc.stdout)
        self.deep_pass("download_models_hf.py --verify-only: ALL_MODELS_OK")


if __name__ == "__main__":
    unittest.main()
