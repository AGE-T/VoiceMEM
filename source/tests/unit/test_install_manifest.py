"""Unit tests for scripts/write_install_manifest.py (pure stdlib, M0).

build_manifest() is driven against a temporary fake project tree: a repo
without git, a models/ tree with an llm GGUF, an asr config.json, tts ONNX
voices, docs (.gitkeep/README.md) and an hf cache subtree that must all be
handled deterministically. GPU/torch blocks are checked with --fake-gpu
semantics (the sandbox has no torch).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

try:
    from scripts.write_install_manifest import (  # package import
        build_manifest,
        write_manifest,
    )
except ImportError:  # direct execution: load the script by file location
    _SPEC = importlib.util.spec_from_file_location(
        "write_install_manifest",
        Path(__file__).resolve().parents[2] / "scripts" / "write_install_manifest.py",
    )
    assert _SPEC is not None and _SPEC.loader is not None
    _MODULE = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    build_manifest = _MODULE.build_manifest  # type: ignore[assignment]
    write_manifest = _MODULE.write_manifest  # type: ignore[assignment]


def _fake_repo(tmp: Path) -> Path:
    """Create a minimal M0-style project tree and return its root."""
    root = tmp / "fake-repo"
    (root / "models" / "llm" / "qwen3.6-35b-a3b").mkdir(parents=True)
    (root / "models" / "asr" / "qwen3-asr-0.6b").mkdir(parents=True)
    (root / "models" / "tts" / "piper").mkdir(parents=True)
    (root / "models" / "embedding" / "multilingual-e5-small").mkdir(parents=True)
    (root / "models" / "vad" / "silero-vad").mkdir(parents=True)
    (root / "models" / "hf" / "hub").mkdir(parents=True)
    (root / "vendor").mkdir(parents=True)

    (root / "VERSION").write_text("9.9.9\n", encoding="utf-8")
    (root / "models" / "llm" / "qwen3.6-35b-a3b" / "Qwen3.6-35B-A3B-IQ4_XS.gguf").write_bytes(
        b"GGUF-fake-weights" * 4
    )
    (root / "models" / "llm" / "qwen3.6-35b-a3b" / ".gitkeep").write_bytes(b"")
    (root / "models" / "llm" / "qwen3.6-35b-a3b" / "README.md").write_text("doc", encoding="utf-8")
    (root / "models" / "asr" / "qwen3-asr-0.6b" / "config.json").write_text(
        '{"model_type": "qwen3_asr"}', encoding="utf-8"
    )
    (root / "models" / "tts" / "piper" / "hu_HU-anna-medium.onnx").write_bytes(b"onnx" * 100)
    (root / "models" / "tts" / "piper" / "hu_HU-anna-medium.onnx.json").write_text(
        "{}", encoding="utf-8"
    )
    (root / "models" / "embedding" / "multilingual-e5-small" / "config.json").write_text(
        "{}", encoding="utf-8"
    )
    (root / "models" / "vad" / "silero-vad" / "silero_vad.onnx").write_bytes(b"vad" * 10)
    # hf cache internals + repo docs must be skipped by the walker
    (root / "models" / "hf" / "hub" / "cached-blob.bin").write_bytes(b"blob" * 1000)
    (root / "models" / "README.md").write_text("top doc", encoding="utf-8")
    return root


def _paths(manifest: dict) -> list[str]:
    return [entry["path"] for entry in manifest["models"]]


class BuildManifestTests(unittest.TestCase):
    """build_manifest() over a fake, git-less project tree."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = _fake_repo(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_top_level_schema(self) -> None:
        manifest = build_manifest(self.root)
        self.assertEqual(
            set(manifest),
            {
                "generated_at", "project", "os", "python", "gpu", "torch",
                "voicemem", "models", "dependencies", "offline",
            },
        )

    def test_project_block_uses_version_file(self) -> None:
        manifest = build_manifest(self.root)
        self.assertEqual(manifest["project"]["name"], "voicemem-agent")
        self.assertEqual(manifest["project"]["version"], "9.9.9")
        self.assertEqual(manifest["project"]["git_commit"], None)  # no git repo here

    def test_os_and_python_blocks_are_type_correct(self) -> None:
        manifest = build_manifest(self.root)
        for key in ("system", "release", "version", "machine"):
            self.assertIsInstance(manifest["os"][key], str, key)
        for key in ("version", "executable", "implementation"):
            self.assertIsInstance(manifest["python"][key], str, key)

    def test_models_list_skips_docs_and_hf_cache(self) -> None:
        manifest = build_manifest(self.root)
        paths = _paths(manifest)
        self.assertEqual(
            paths,
            [
                "models/asr/qwen3-asr-0.6b/config.json",
                "models/embedding/multilingual-e5-small/config.json",
                "models/llm/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-IQ4_XS.gguf",
                "models/tts/piper/hu_HU-anna-medium.onnx",
                "models/tts/piper/hu_HU-anna-medium.onnx.json",
                "models/vad/silero-vad/silero_vad.onnx",
            ],
        )
        self.assertNotIn("models/hf/hub/cached-blob.bin", paths)
        self.assertFalse(any(p.endswith(".gitkeep") or p.endswith("README.md") for p in paths))

    def test_component_and_hf_repo_mapping(self) -> None:
        manifest = build_manifest(self.root)
        by_path = {entry["path"]: entry for entry in manifest["models"]}
        self.assertEqual(
            by_path["models/llm/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-IQ4_XS.gguf"]["component"],
            "llm",
        )
        self.assertEqual(
            by_path["models/llm/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-IQ4_XS.gguf"]["hf_repo"],
            "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
        )
        self.assertEqual(by_path["models/asr/qwen3-asr-0.6b/config.json"]["component"], "asr")
        self.assertEqual(by_path["models/asr/qwen3-asr-0.6b/config.json"]["hf_repo"], "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(
            by_path["models/tts/piper/hu_HU-anna-medium.onnx"]["hf_repo"], "rhasspy/piper-voices"
        )
        self.assertEqual(
            by_path["models/vad/silero-vad/silero_vad.onnx"]["hf_repo"], "snakers4/silero-vad"
        )
        self.assertEqual(
            by_path["models/embedding/multilingual-e5-small/config.json"]["component"], "embedding"
        )

    def test_sha256_and_size_are_correct(self) -> None:
        manifest = build_manifest(self.root)
        gguf = (
            self.root / "models" / "llm" / "qwen3.6-35b-a3b" / "Qwen3.6-35B-A3B-IQ4_XS.gguf"
        )
        entry = next(e for e in manifest["models"] if e["path"].endswith(".gguf"))
        self.assertEqual(entry["size_bytes"], gguf.stat().st_size)
        self.assertEqual(entry["sha256"], hashlib.sha256(gguf.read_bytes()).hexdigest())

    def test_deterministic_model_ordering(self) -> None:
        first = build_manifest(self.root)
        second = build_manifest(self.root)
        self.assertEqual(_paths(first), _paths(second))
        self.assertEqual(first["models"], second["models"])

    def test_gpu_and_torch_none_without_fake_flag(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            manifest = build_manifest(self.root)
        self.assertIsNone(manifest["gpu"])
        self.assertIsNone(manifest["torch"])

    def test_fake_gpu_fills_blocks(self) -> None:
        manifest = build_manifest(self.root, fake_gpu=True)
        self.assertEqual(manifest["gpu"]["name"], "NVIDIA GeForce RTX 5070")
        self.assertTrue(manifest["gpu"]["cuda_available"])
        self.assertEqual(manifest["gpu"]["compute_capability"], "12.0")
        self.assertEqual(manifest["torch"]["version"], "2.7.0+cu128")
        self.assertEqual(manifest["torch"]["cuda"], "12.8")

    def test_voicemem_not_installed_marker(self) -> None:
        manifest = build_manifest(self.root)
        self.assertEqual(manifest["voicemem"]["version"], "not_installed")
        self.assertIsNone(manifest["voicemem"]["commit"])

    def test_voicemem_controlled_fork_identity_from_pin(self) -> None:
        """v0.5.0: the vendor tree is OUR controlled fork, identity comes from
        VOICEMEM_PIN.json (the tree is NOT a git checkout; upstream tag
        v0.0.1 = commit e8384e0 = package 0.2.3 are recorded explicitly).

        The old test simulated a git clone + annotated tag + .clone_ref —
        the controlled-fork design deliberately has none of those (no clone,
        no fetch surface, no main-branch fallback).
        """
        import json

        vendor = self.root / "vendor" / "voicemem" / "voicemem"
        vendor.mkdir(parents=True)
        (vendor / "__init__.py").write_text(
            "CONTROLLED_FORK = True\n"
            "CONTROLLED_UPSTREAM_COMMIT = "
            "'e8384e087bd2f44eb05fc7ae1a3c525ea8244179'\n",
            encoding="utf-8",
        )
        pin = {
            "status": "controlled-fork",
            "provenance": {
                "upstream_repo": "https://github.com/xzf-thu/VoiceMem.git",
                "upstream_commit": "e8384e087bd2f44eb05fc7ae1a3c525ea8244179",
                "upstream_tag": "v0.0.1",
            },
            "local_patches": [
                {"id": "VM-LOCAL-001"},
                {"id": "VM-LOCAL-005"},
            ],
        }
        (self.root / "VOICEMEM_PIN.json").write_text(
            json.dumps(pin, indent=2), encoding="utf-8")

        manifest = build_manifest(self.root)
        block = manifest["voicemem"]
        self.assertEqual(block["requested_ref"], "v0.0.1")
        self.assertEqual(block["git_tag"], "v0.0.1")
        self.assertEqual(
            block["git_commit"], "e8384e087bd2f44eb05fc7ae1a3c525ea8244179")
        self.assertEqual(
            block["commit"], "e8384e087bd2f44eb05fc7ae1a3c525ea8244179")
        self.assertEqual(block["upstream_commit"],
                         "e8384e087bd2f44eb05fc7ae1a3c525ea8244179")
        self.assertTrue(block["controlled"])
        self.assertIn("VM-LOCAL-001", block["local_patches"])
        self.assertIn("VM-LOCAL-005", block["local_patches"])
        # the voicemem dist is NOT pip-installed in this sandbox process ->
        # the runtime probe reports not-verified and the fallback version
        self.assertFalse(block["pin_verified"])
        self.assertIsNone(block["runtime_import_path"])
        self.assertEqual(block["package_version"], "0.2.3-vendor-controlled")
        self.assertEqual(block["version"], "0.2.3-vendor-controlled")

    def test_voicemem_pinless_vendor_tree_reports_null_identity(self) -> None:
        """A vendor tree without VOICEMEM_PIN.json is a BROKEN pairing —
        the manifest must record the missing identity, not invent one."""
        vendor = self.root / "vendor" / "voicemem" / "voicemem"
        vendor.mkdir(parents=True)
        (vendor / "__init__.py").write_text("", encoding="utf-8")

        manifest = build_manifest(self.root)
        block = manifest["voicemem"]
        self.assertIsNone(block["git_tag"])
        self.assertIsNone(block["upstream_commit"])
        self.assertIsNone(block["requested_ref"])
        self.assertIsNone(block["commit"])
        self.assertFalse(block.get("pin_verified"))
        self.assertFalse(block.get("controlled"))

    def test_voicemem_corrupt_pin_reports_null_identity(self) -> None:
        vendor = self.root / "vendor" / "voicemem" / "voicemem"
        vendor.mkdir(parents=True)
        (vendor / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "VOICEMEM_PIN.json").write_text("{not json",
                                                      encoding="utf-8")
        manifest = build_manifest(self.root)
        block = manifest["voicemem"]
        self.assertIsNone(block["upstream_commit"])
        self.assertFalse(block.get("pin_verified"))

    def test_source_record_enriches_llm_entry(self) -> None:
        """v0.3.1+: the ACTUAL download source (provenance).

        v0.4.16: the LLM is the operator-placed Qwen3.6 35B A3B IQ4_XS
        (no mirror chain) - models/.download_sources.json records that
        source, and the manifest must merge it into every llm entry.
        """
        record = {
            "schema_version": 1,
            "components": {
                "llm": {
                    "source_repo": "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
                    "requested_revision": "main",
                    "resolved_commit": "aa11bb22cc33dd44ee55ff66778899aabbccddeeff",
                    "target_dir": "models/llm/qwen3.6-35b-a3b",
                    "files": {"Qwen3.6-35B-A3B-IQ4_XS.gguf":
                              {"size_bytes": 6975879296}},
                },
            },
        }
        (self.root / "models" / ".download_sources.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

        manifest = build_manifest(self.root)
        paths = _paths(manifest)
        # the provenance record itself is NOT a model asset
        self.assertNotIn("models/.download_sources.json", paths)
        llm = [e for e in manifest["models"]
               if e["path"].endswith("Qwen3.6-35B-A3B-IQ4_XS.gguf")]
        self.assertEqual(len(llm), 1)
        entry = llm[0]
        self.assertEqual(entry["source_repo"], "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF")
        self.assertEqual(entry["source_revision"], "main")
        self.assertEqual(entry["source_commit"],
                         "aa11bb22cc33dd44ee55ff66778899aabbccddeeff")
        # canonical mapping stays for the OTHER (undownloaded) components
        self.assertEqual(entry["hf_repo"], "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF")
        asr = [e for e in manifest["models"] if e["component"] == "asr"][0]
        self.assertNotIn("source_repo", asr)  # no record -> no fake provenance

    def test_offline_block_follows_env(self) -> None:
        with patch.dict(
            "os.environ", {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": ""}
        ):
            manifest = build_manifest(self.root)
        self.assertTrue(manifest["offline"]["hf_hub_offline"])
        self.assertFalse(manifest["offline"]["transformers_offline"])

    def test_dependencies_are_strings(self) -> None:
        manifest = build_manifest(self.root)
        deps = manifest["dependencies"]
        self.assertIsInstance(deps, dict)
        self.assertTrue(all(isinstance(k, str) and isinstance(v, str) for k, v in deps.items()))
        self.assertIn("PyYAML", deps)  # installed in every test environment


class WriteManifestTests(unittest.TestCase):
    """The thin file-writing layer."""

    def test_write_manifest_produces_sorted_valid_json(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_repo(Path(tmp))
            out = Path(tmp) / "sub" / "INSTALL_MANIFEST.json"
            manifest = build_manifest(root)
            returned = write_manifest(manifest, out)
            self.assertEqual(returned, out)
            text = out.read_text(encoding="utf-8")
            loaded = json.loads(text)
            self.assertEqual(loaded["project"]["version"], "9.9.9")
            # determinism: same manifest -> identical bytes (except timestamp)
            self.assertEqual(text, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
