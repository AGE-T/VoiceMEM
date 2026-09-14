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
safetensors presence, supertonic onnx + voice styles).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation._report import FeatureValidationTest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def _deep_model_set_ready() -> bool:
    """v0.6.1: the DEEP checks run only when the locked model set is present.

    The guard follows the components the deep tests actually assert (asr +
    embedding + vad + the default supertonic presets). The dev sandbox
    intentionally hosts E5 under the HF_HOME cache root (models/hf), not at
    the lock's embedding target - there the deep checks skip, exactly like
    the v0.6.0 gate skipped on the then-absent qwen dir. A machine that
    downloaded the lock set (the field/one-click flow) runs them for real.
    """
    checks = (
        MODELS_DIR / "asr" / "parakeet-tdt-0.6b-v3" / "config.json",
        MODELS_DIR / "embedding" / "multilingual-e5-small" / "config.json",
        MODELS_DIR / "vad" / "silero-vad" / "silero_vad.onnx",
        MODELS_DIR / "tts" / "supertonic-3" / "onnx" / "vocoder.onnx",
        MODELS_DIR / "tts" / "supertonic-3" / "voice_styles" / "F1.json",
    )
    return all(p.is_file() for p in checks)

#: M0 Section 7 component layout (repo-side, always required).
#: v0.6.1: parakeet (production) + nemotron (selectable) joined; the retired
#: qwen dir still SHIPS as the legacy migration-module placeholder.
EXPECTED_DIRS = (
    "asr/parakeet-tdt-0.6b-v3",
    "asr/nemotron-3.5-asr-streaming-0.6b",
    "asr/qwen3-asr-0.6b",
    "llm/qwen3.6-35b-a3b",
    "tts/supertonic-3",
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
        if not _deep_model_set_ready():
            self.deep_skip("model set not fully downloaded on this machine")
        # v0.6.1: the ASR deep check follows the PRODUCTION engine dir
        # (parakeet; the retired qwen dir is never auto-downloaded).
        asr_dir = MODELS_DIR / "asr" / "parakeet-tdt-0.6b-v3"
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
        # Supertonic 3 (v0.7.0): the 6 onnx modules + the default voice styles.
        st_onnx = MODELS_DIR / "tts" / "supertonic-3" / "onnx"
        for module in (
            "tts.json", "unicode_indexer.json", "duration_predictor.onnx",
            "text_encoder.onnx", "vector_estimator.onnx", "vocoder.onnx",
        ):
            f = st_onnx / module
            self.assertTrue(f.is_file(), f"supertonic onnx module missing: {module}")
            self.assertGreater(f.stat().st_size, 1000, f"supertonic {module} suspiciously small")
        verified.append("supertonic 3 onnx modules (6)")
        styles = MODELS_DIR / "tts" / "supertonic-3" / "voice_styles"
        for preset in ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"):
            self.assertTrue(
                (styles / f"{preset}.json").is_file(),
                f"supertonic voice style missing: {preset}.json",
            )
        verified.append("supertonic preset voice styles (10)")
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

        if not _deep_model_set_ready():
            self.deep_skip("model set not fully downloaded on this machine")
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
