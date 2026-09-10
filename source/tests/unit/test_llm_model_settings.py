"""v0.4.16 model-picker tests (portable LLM GGUF path).

Covers the full contract of the new "LLM model" feature:
* app/gguf.py: the stdlib GGUF v2/v3 header reader (magic, version,
  tensor/kv counts, general.name / general.architecture / general.file_type,
  array/string/fixed-type skipping, early identity break, truncation and
  non-GGUF rejection) + the expected-model identity check (model and quant
  separately; IQ4_XS == 30 per llama.cpp llama_ftype).
* app/llm_model_settings.py: persistence (atomic, lenient, env override),
  the resolution chain (user-selection > env > default), the resolved-model
  marker reader.
* app/native_picker.py: platform guard (non-Windows = unsupported).
* REST surface (DEMO components + TestClient): GET /api/llm-model payload,
  /inspect validation, /select + persistence + reset, 400 on bad paths.
* web UI: the "LLM model" section markers (Browse…, manual path, reset).

Sandbox-safe: synthetic GGUF files built in-memory; the settings file is
redirected via VOICEMEM_LLM_MODEL_SETTINGS so tests never touch the real
config/llm_model.json.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import AgentConfig
from app.gguf import (
    EXPECTED_MODEL_LABEL,
    FILE_TYPE_LABELS,
    IQ4_XS_FILE_TYPE,
    GgufInfo,
    expected_model_check,
    llama_server_load_verdict,
    quant_label,
    read_gguf_metadata,
)
from app.llm_model_settings import (
    SOURCE_DEFAULT,
    SOURCE_ENV,
    SOURCE_NONE,
    SOURCE_USER,
    LlmModelSettings,
    operator_default_candidates,
    resolve_llm_model_path,
    read_llama_server_loaded_model,
)
from app.native_picker import is_supported, pick_file
from app.web_server import WebComponents, build_web_app

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# synthetic GGUF construction (mirrors the real v2/v3 layout)
#
# v0.4.18: the type ids below follow the ACTUAL GGUF spec table (llama.cpp
# gguf-py constants.py GGUFValueType): 8=STRING, 9=ARRAY, 7=BOOL, 12=FLOAT64,
# 10=UINT64, 4=UINT32, 2=UINT16, 1=INT8, 6=FLOAT32. The v0.4.16/17 fixtures
# encoded the same WRONG numbering as the shipped parsers (9=string, 10=array,
# 8=bool, 7=f64, 13=f16), so the tests stayed green while real GGUF files
# failed - both are corrected against the authoritative table now.
# --------------------------------------------------------------------------- #

def kv_str(key: str, value: str) -> bytes:
    k, v = key.encode(), value.encode()
    return (
        struct.pack("<Q", len(k)) + k
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(v)) + v
    )


def kv_u32(key: str, value: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_u64(key: str, value: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 10) + struct.pack("<Q", value)


def kv_f64(key: str, value: float) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 12) + struct.pack("<d", value)


def kv_f32(key: str, value: float) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 6) + struct.pack("<f", value)


def kv_u16(key: str, value: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 2) + struct.pack("<H", value)


def kv_bool(key: str, value: bool) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 7) + struct.pack("<B", int(value))


def kv_arr_u32(key: str, items: list[int]) -> bytes:
    k = key.encode()
    return (
        struct.pack("<Q", len(k)) + k
        + struct.pack("<I", 9)
        + struct.pack("<I", 4)
        + struct.pack("<Q", len(items))
        + b"".join(struct.pack("<I", i) for i in items)
    )


def kv_arr_str(key: str, items: list[str]) -> bytes:
    k = key.encode()
    body = b"".join(
        struct.pack("<Q", len(s.encode())) + s.encode() for s in items
    )
    return (
        struct.pack("<Q", len(k)) + k
        + struct.pack("<I", 9)
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(items))
        + body
    )


def build_gguf(
    version: int = 3,
    tensor_count: int = 2,
    kvs: bytes = b"",
    kv_count: int | None = None,
    trailer: int = 64,
) -> bytes:
    if kv_count is None:
        kv_count = kvs.count(b"\x00") if False else _count_kvs(kvs)
    return (
        b"GGUF"
        + struct.pack("<IQQ", version, tensor_count, kv_count)
        + kvs
        + b"\x00" * trailer
    )


def _count_kvs(kvs: bytes) -> int:
    # each kv starts with u64 key length; count by walking (test helper only)
    # - the type table is the REAL GGUF spec table (see kv_str note)
    _SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
              10: 8, 11: 8, 12: 8}
    count, pos = 0, 0
    while pos < len(kvs):
        klen = struct.unpack_from("<Q", kvs, pos)[0]
        pos += 8 + klen
        vtype = struct.unpack_from("<I", kvs, pos)[0]
        pos += 4
        if vtype == 8:
            slen = struct.unpack_from("<Q", kvs, pos)[0]
            pos += 8 + slen
        elif vtype == 9:
            etype = struct.unpack_from("<I", kvs, pos)[0]
            ecount = struct.unpack_from("<Q", kvs, pos + 4)[0]
            pos += 12
            for _ in range(ecount):
                if etype == 8:
                    slen = struct.unpack_from("<Q", kvs, pos)[0]
                    pos += 8 + slen
                elif etype in _SIZES:
                    pos += _SIZES[etype]
                else:
                    raise ValueError(f"bad etype {etype}")
        elif vtype in _SIZES:
            pos += _SIZES[vtype]
        else:
            raise ValueError(f"bad vtype {vtype}")
        count += 1
    return count


QWEN_KVS = (
    kv_str("general.architecture", "qwen3moe")
    + kv_f64("general.some.float", 3.5)
    + kv_str("general.name", "Qwen_Qwen3.6-35B-A3B")
    + kv_u32("general.file_type", 30)
    + kv_arr_u32("test.array", [1, 2, 3])
    + kv_arr_str("test.strings", ["a", "bb", "ccc"])
    + kv_bool("test.bool", True)
    + kv_u64("test.u64", 1234567890123)
    + kv_f32("test.f32", 1.5)
    + kv_u16("test.u16", 7)
    + kv_str("general.finetune", "iq4xs")
)


class TestGgufReader(unittest.TestCase):
    """app/gguf.py: header walk + identity check."""

    def test_magic_and_identity(self):
        with tempfile.TemporaryDirectory() as td:
            q = Path(td) / "Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
            q.write_bytes(build_gguf(version=3, tensor_count=2, kvs=QWEN_KVS))
            info = read_gguf_metadata(q)
            self.assertTrue(info.ok, info.error)
            self.assertEqual(info.version, 3)
            self.assertEqual(info.tensor_count, 2)
            self.assertEqual(info.name, "Qwen_Qwen3.6-35B-A3B")
            self.assertEqual(info.architecture, "qwen3moe")
            self.assertEqual(info.file_type, 30)
            self.assertEqual(info.quant, "IQ4_XS")

    def test_expected_check_full_match(self):
        with tempfile.TemporaryDirectory() as td:
            q = Path(td) / "Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
            q.write_bytes(build_gguf(kvs=QWEN_KVS))
            info = read_gguf_metadata(q)
            check = expected_model_check(info)
            self.assertTrue(check["is_expected"])
            self.assertTrue(check["model_match"])
            self.assertTrue(check["quant_match"])
            self.assertEqual(check["reasons"], [])
            self.assertIn("IQ4_XS", check["detected"])

    def test_wrong_quant_is_not_expected(self):
        kvs = kv_str("general.name", "Qwen_Qwen3.6-35B-A3B") + kv_u32("general.file_type", 15)
        with tempfile.TemporaryDirectory() as td:
            q = Path(td) / "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
            q.write_bytes(build_gguf(kvs=kvs))
            check = expected_model_check(read_gguf_metadata(q))
            self.assertTrue(check["model_match"])
            self.assertFalse(check["quant_match"])
            self.assertFalse(check["is_expected"])

    def test_wrong_model_is_not_expected(self):
        kvs = (
            kv_str("general.name", "gemma-4-12b-it")
            + kv_str("general.architecture", "gemma")
            + kv_u32("general.file_type", 30)
        )
        with tempfile.TemporaryDirectory() as td:
            g = Path(td) / "some-model.gguf"
            g.write_bytes(build_gguf(kvs=kvs))
            check = expected_model_check(read_gguf_metadata(g))
            # IQ4_XS quant of a DIFFERENT model: quant matches, model not
            self.assertTrue(check["quant_match"])
            self.assertFalse(check["model_match"])
            self.assertFalse(check["is_expected"])
            self.assertTrue(check["reasons"])

    def test_missing_file(self):
        info = read_gguf_metadata("/nonexistent/x.gguf")
        self.assertFalse(info.exists)
        self.assertFalse(info.ok)

    def test_not_a_gguf(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td) / "bad.gguf"
            b.write_bytes(b"JUNKJUNKJUNK" * 8)
            info = read_gguf_metadata(b)
            self.assertFalse(info.is_gguf)
            self.assertIn("GGUF", info.error)

    def test_truncated_header(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td) / "trunc.gguf"
            t.write_bytes(build_gguf(kvs=QWEN_KVS)[:20])
            info = read_gguf_metadata(t)
            self.assertFalse(info.ok)
            self.assertTrue(info.error)

    def test_unsupported_version(self):
        with tempfile.TemporaryDirectory() as td:
            v = Path(td) / "old.gguf"
            v.write_bytes(build_gguf(version=1, kvs=QWEN_KVS))
            info = read_gguf_metadata(v)
            self.assertFalse(info.ok)
            self.assertIn("version", info.error)

    def test_iq4_xs_is_30(self):
        """llama_ftype: IQ4_XS = 30; 29 is IQ2_M (the v0.4.15 bug)."""
        self.assertEqual(IQ4_XS_FILE_TYPE, 30)
        self.assertEqual(FILE_TYPE_LABELS[30], "IQ4_XS")
        self.assertEqual(FILE_TYPE_LABELS[29], "IQ2_M")
        self.assertEqual(quant_label(30), "IQ4_XS")

    def test_expected_labels(self):
        self.assertEqual(EXPECTED_MODEL_LABEL, "Qwen3.6 35B A3B IQ4_XS")


class TestGgufSplitShardDetection(unittest.TestCase):
    """v0.4.17: an Ollama blob may be ONE SHARD of a gguf-split set.

    llama.cpp's split loading discovers sibling shards by the
    ``-NNNNN-of-NNNNN.gguf`` file-name pattern — a ``sha256-...`` blob can
    never match it, so a shard blob is NOT directly loadable and must be
    reported (and rejected on select), never silently accepted.
    """

    def _shard(self, td: str, no: int, count: int, name: str) -> Path:
        kvs = QWEN_KVS + kv_u32("general.split.no", no) + kv_u32(
            "general.split.count", count
        )
        p = Path(td) / name
        p.write_bytes(build_gguf(kvs=kvs))
        return p

    def test_shard_zero_is_detected(self):
        """general.split.no is ZERO-indexed: shard 1 of 2 has no=0."""
        with tempfile.TemporaryDirectory() as td:
            p = self._shard(td, 0, 2, "sha256-afc7238af403ed45")
            info = read_gguf_metadata(p)
            self.assertTrue(info.ok)
            self.assertEqual((info.split_no, info.split_count), (0, 2))
            self.assertTrue(info.is_split_shard)

    def test_shard_identity_still_matches_but_not_loadable(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._shard(
                td, 1, 3, "sha256-afc7238af403ed454b7846454091b5e38b07575bdbb64c3e86777414dde4193c"
            )
            info = read_gguf_metadata(p)
            check = expected_model_check(info)
            # identity (model + quant) matches — it IS the right model —
            self.assertTrue(check["model_match"])
            self.assertTrue(check["quant_match"])
            # ...but it is a shard: not loadable, reasons say so
            self.assertTrue(check.get("split_shard"))
            self.assertFalse(check.get("loadable"))
            self.assertTrue(
                any("shard" in r for r in check["reasons"]),
                check["reasons"],
            )
            verdict = llama_server_load_verdict(info)
            self.assertFalse(verdict["loadable"])
            self.assertIn("cannot load", verdict["reason"])

    def test_complete_file_is_loadable(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sha256-blob-without-extension"
            p.write_bytes(build_gguf(kvs=QWEN_KVS))
            info = read_gguf_metadata(p)
            self.assertFalse(info.is_split_shard)
            self.assertTrue(llama_server_load_verdict(info)["loadable"])

    def test_select_rejects_split_shard(self):
        """The REST select refuses a shard — accepting it would silently
        produce a dead LLM (llama-server cannot load one shard alone)."""
        with tempfile.TemporaryDirectory() as td:
            env = {"VOICEMEM_LLM_MODEL_SETTINGS": str(Path(td) / "llm_model.json")}
            with unittest.mock.patch.dict(os.environ, env):
                os.environ.pop("LLAMA_MODEL_PATH", None)
                cfg = AgentConfig(root=Path(td))
                components = WebComponents(cfg, mock=True).build()
                from fastapi.testclient import TestClient

                client = TestClient(build_web_app(components))
                p = self._shard(td, 0, 2, "sha256-abc123")
                r = client.post("/api/llm-model/select", json={"path": str(p)})
                self.assertEqual(r.status_code, 400)
                self.assertIn("shard", r.json()["detail"])
                # nothing persisted
                self.assertEqual(LlmModelSettings.load(Path(td)).llm_model_path, "")

    def test_inspect_reports_shard_facts(self):
        with tempfile.TemporaryDirectory() as td:
            env = {"VOICEMEM_LLM_MODEL_SETTINGS": str(Path(td) / "llm_model.json")}
            with unittest.mock.patch.dict(os.environ, env):
                os.environ.pop("LLAMA_MODEL_PATH", None)
                cfg = AgentConfig(root=Path(td))
                components = WebComponents(cfg, mock=True).build()
                from fastapi.testclient import TestClient

                client = TestClient(build_web_app(components))
                p = self._shard(td, 0, 2, "sha256-abc123")
                r = client.post("/api/llm-model/inspect", json={"path": str(p)})
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertTrue(body["split_shard"])
                self.assertEqual(body["split_count"], 2)
                self.assertFalse(body["load"]["loadable"])
                self.assertIn("shard", body["load"]["reason"])
                # inspect does not persist anything either
                self.assertEqual(LlmModelSettings.load(Path(td)).llm_model_path, "")


class TestLlmModelSettings(unittest.TestCase):
    """app/llm_model_settings.py: persistence + resolution chain."""

    def _env(self, tmp: str) -> dict[str, str]:
        return {"VOICEMEM_LLM_MODEL_SETTINGS": str(Path(tmp) / "llm_model.json")}

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            with unittest.mock.patch.dict(os.environ, self._env(td)):
                settings = LlmModelSettings.load(Path(td))
                self.assertEqual(settings.llm_model_path, "")
                settings.apply_selection(
                    str(Path(td) / "model.gguf"),
                    model_name="Qwen_Qwen3.6-35B-A3B",
                    quant="IQ4_XS",
                    file_size=42,
                )
                reloaded = LlmModelSettings.load(Path(td))
                self.assertEqual(reloaded.llm_model_path, str(Path(td) / "model.gguf"))
                self.assertEqual(reloaded.model_name, "Qwen_Qwen3.6-35B-A3B")
                self.assertEqual(reloaded.quant, "IQ4_XS")
                self.assertEqual(reloaded.file_size, 42)
                self.assertTrue(reloaded.selected_at)

    def test_corrupt_file_is_lenient(self):
        with tempfile.TemporaryDirectory() as td:
            env = self._env(td)
            path = Path(env["VOICEMEM_LLM_MODEL_SETTINGS"])
            path.write_text("{ this is not json", encoding="utf-8")
            with unittest.mock.patch.dict(os.environ, env):
                settings = LlmModelSettings.load(Path(td))
                self.assertEqual(settings.llm_model_path, "")

    def test_clear(self):
        with tempfile.TemporaryDirectory() as td:
            with unittest.mock.patch.dict(os.environ, self._env(td)):
                settings = LlmModelSettings.load(Path(td))
                settings.apply_selection("/x/y.gguf", "m", "q", 1)
                settings.clear()
                reloaded = LlmModelSettings.load(Path(td))
                self.assertEqual(reloaded.llm_model_path, "")

    def test_resolution_chain(self):
        with tempfile.TemporaryDirectory() as td:
            env = self._env(td)
            with unittest.mock.patch.dict(os.environ, env):
                # 1. v0.4.17: NO phantom default — when nothing is set and
                # no GGUF sits in the operator dir, the honest empty state
                os.environ.pop("LLAMA_MODEL_PATH", None)
                path, source = resolve_llm_model_path(Path(td), None)
                self.assertEqual((path, source), ("", SOURCE_NONE))
                # 1b. a GGUF ACTUALLY PRESENT in models/llm/qwen3.6-35b-a3b/
                # IS the project default (magic-validated, any file name)
                from app.llm_model_settings import operator_default_candidates

                op_dir = Path(td) / "models" / "llm" / "qwen3.6-35b-a3b"
                op_dir.mkdir(parents=True)
                op_file = op_dir / "Qwen3.6-35B-A3B-IQ4_XS.gguf"
                op_file.write_bytes(build_gguf(kvs=QWEN_KVS))
                self.assertEqual(
                    operator_default_candidates(Path(td)), [str(op_file)]
                )
                path, source = resolve_llm_model_path(Path(td), None)
                self.assertEqual((path, source), (str(op_file), SOURCE_DEFAULT))
                # 1c. a SPLIT SHARD in the operator dir is NOT a default
                shard = op_dir / "sha256-abc"
                shard.write_bytes(
                    build_gguf(
                        kvs=QWEN_KVS
                        + kv_u32("general.split.no", 0)
                        + kv_u32("general.split.count", 2)
                    )
                )
                op_file.unlink()
                self.assertEqual(
                    operator_default_candidates(Path(td)), []
                )
                path, source = resolve_llm_model_path(Path(td), None)
                self.assertEqual((path, source), ("", SOURCE_NONE))
                # 2. env wins over default
                os.environ["LLAMA_MODEL_PATH"] = str(Path(td) / "env.gguf")
                path, source = resolve_llm_model_path(Path(td), None)
                self.assertEqual((path, source), (str(Path(td) / "env.gguf"), SOURCE_ENV))
                # 3. user selection wins over env
                settings = LlmModelSettings.load(Path(td))
                settings.apply_selection(str(Path(td) / "user.gguf"), "m", "q", 1)
                path, source = resolve_llm_model_path(Path(td), None)
                self.assertEqual((path, source), (str(Path(td) / "user.gguf"), SOURCE_USER))
                # 4. clear returns control to env
                settings.clear()
                path, source = resolve_llm_model_path(Path(td), None)
                self.assertEqual((path, source), (str(Path(td) / "env.gguf"), SOURCE_ENV))

    def test_operator_dir_ambiguity_is_not_a_default(self):
        """Two GGUFs in the operator dir -> NO silent pick (pick in the UI)."""
        with tempfile.TemporaryDirectory() as td:
            with unittest.mock.patch.dict(os.environ, self._env(td)):
                os.environ.pop("LLAMA_MODEL_PATH", None)
                op_dir = Path(td) / "models" / "llm" / "qwen3.6-35b-a3b"
                op_dir.mkdir(parents=True)
                (op_dir / "a.gguf").write_bytes(build_gguf(kvs=QWEN_KVS))
                (op_dir / "b.gguf").write_bytes(build_gguf(kvs=QWEN_KVS))
                self.assertEqual(len(operator_default_candidates(Path(td))), 2)
                path, source = resolve_llm_model_path(Path(td), None)
                self.assertEqual((path, source), ("", SOURCE_NONE))

    def test_config_llm_model_file_follows_the_chain(self):
        with tempfile.TemporaryDirectory() as td:
            env = self._env(td)
            with unittest.mock.patch.dict(os.environ, env):
                os.environ.pop("LLAMA_MODEL_PATH", None)
                # v0.4.17: None when nothing is configured (never a phantom
                # project-default path that does not exist on disk)
                cfg = AgentConfig(root=Path(td))
                self.assertIsNone(cfg.llm_model_file)
                self.assertFalse(cfg.check_runtime_assets()["llama_model"])
                settings = LlmModelSettings.load(Path(td))
                settings.apply_selection(str(Path(td) / "picked.gguf"), "m", "q", 1)
                cfg = AgentConfig(root=Path(td))
                self.assertEqual(str(cfg.llm_model_file), str(Path(td) / "picked.gguf"))

    def test_config_llm_model_file_operator_default(self):
        with tempfile.TemporaryDirectory() as td:
            with unittest.mock.patch.dict(os.environ, self._env(td)):
                os.environ.pop("LLAMA_MODEL_PATH", None)
                op_dir = Path(td) / "models" / "llm" / "qwen3.6-35b-a3b"
                op_dir.mkdir(parents=True)
                # ANY file name with a valid GGUF magic is the default
                blob = op_dir / "sha256-afc7238af403ed454b7846454091b5e38b07575bdbb64c3e86777414dde4193c"
                blob.write_bytes(build_gguf(kvs=QWEN_KVS))
                cfg = AgentConfig(root=Path(td))
                self.assertEqual(str(cfg.llm_model_file), str(blob))
                self.assertTrue(cfg.check_runtime_assets()["llama_model"])

    def test_marker_reader(self):
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "logs" / "llama-server.resolved-model.json"
            marker.parent.mkdir()
            self.assertEqual(read_llama_server_loaded_model(Path(td)), {})
            payload = {"model_path": "D:\\AI\\model.gguf", "resolved_at": "x"}
            marker.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(read_llama_server_loaded_model(Path(td)), payload)
            marker.write_text("not json", encoding="utf-8")
            self.assertEqual(read_llama_server_loaded_model(Path(td)), {})


class TestNativePicker(unittest.TestCase):
    """app/native_picker.py: platform guard."""

    def test_non_windows_is_unsupported(self):
        import platform

        if platform.system() == "Windows":
            self.skipTest("Windows host: the native dialog is available")
        self.assertFalse(is_supported())
        path, status = pick_file()
        self.assertIsNone(path)
        self.assertEqual(status, "unsupported")


class TestLlmModelRest(unittest.TestCase):
    """REST surface with DEMO components (hermetic, no models)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="vm_llmmodel_")
        cls.env = {
            "VOICEMEM_LLM_MODEL_SETTINGS": str(Path(cls.tmp) / "llm_model.json"),
        }
        cls._backup = {
            k: os.environ.get(k)
            for k in ("VOICEMEM_LLM_MODEL_SETTINGS", "LLAMA_MODEL_PATH")
        }
        os.environ["VOICEMEM_LLM_MODEL_SETTINGS"] = cls.env["VOICEMEM_LLM_MODEL_SETTINGS"]
        os.environ.pop("LLAMA_MODEL_PATH", None)
        cfg = AgentConfig(root=Path(cls.tmp))
        cls.components = WebComponents(cfg, mock=True).build()
        from fastapi.testclient import TestClient

        cls.client = TestClient(build_web_app(cls.components))

    @classmethod
    def tearDownClass(cls) -> None:
        for k, v in cls._backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # -- GET status -------------------------------------------------------- #

    def test_get_status_payload(self):
        r = self.client.get("/api/llm-model")
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertEqual(payload["selected_llm"], "qwen3.6-35b-a3b")
        self.assertEqual(payload["expected_model"], "Qwen3.6 35B A3B IQ4_XS")
        self.assertIn("model_path", payload)
        # v0.4.17: "none" is a legitimate source (no model configured);
        # the sandbox test env has no GGUF anywhere, so it must be none.
        self.assertIn(
            payload["source"],
            ("user-selection", "env-profile", "project-default", "none"),
        )
        if payload["source"] == "none":
            self.assertEqual(payload["model_path"], "")
            self.assertIn("no model file configured", payload["file"]["error"])
        self.assertIn("file", payload)
        self.assertIn("identity", payload)
        self.assertIn("load", payload)
        self.assertIn("llama_server", payload)
        # v0.4.17: the llama-server block carries the runtime load PROOF
        for key in (
            "served_model_id",
            "log_path_found",
            "log_line",
            "completion_ok",
            "completion_reply",
            "verified_at",
        ):
            self.assertIn(key, payload["llama_server"])
        self.assertIn("picker_supported", payload)

    def test_get_status_reports_the_selected_file(self):
        # create a real synthetic GGUF in the temp dir and select it
        gguf = Path(self.tmp) / "Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
        gguf.write_bytes(build_gguf(kvs=QWEN_KVS))
        r = self.client.post(
            "/api/llm-model/select", json={"path": str(gguf)}
        )
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertEqual(payload["source"], "user-selection")
        self.assertEqual(payload["model_path"], str(gguf))
        self.assertTrue(payload["file"]["exists"])
        self.assertTrue(payload["file"]["is_gguf"])
        self.assertEqual(payload["file"]["quant"], "IQ4_XS")
        self.assertTrue(payload["identity"]["is_expected"])
        # the GET now serves the same selection
        r2 = self.client.get("/api/llm-model")
        self.assertEqual(r2.json()["model_path"], str(gguf))
        # AgentConfig resolution follows it too
        cfg = AgentConfig(root=Path(self.tmp))
        self.assertEqual(str(cfg.llm_model_file), str(gguf))

    # -- inspect -------------------------------------------------------------- #

    def test_inspect_validates_without_saving(self):
        gguf = Path(self.tmp) / "another.gguf"
        gguf.write_bytes(build_gguf(kvs=QWEN_KVS))
        r = self.client.post("/api/llm-model/inspect", json={"path": str(gguf)})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["is_gguf"])
        self.assertEqual(body["quant"], "IQ4_XS")
        # not persisted: status still points elsewhere
        status = self.client.get("/api/llm-model").json()
        self.assertNotEqual(status["model_path"], str(gguf))

    def test_inspect_rejects_missing_path(self):
        r = self.client.post("/api/llm-model/inspect", json={"path": ""})
        self.assertEqual(r.status_code, 400)

    def test_inspect_reports_non_gguf(self):
        junk = Path(self.tmp) / "junk.bin"
        junk.write_bytes(b"JUNKJUNK")
        r = self.client.post("/api/llm-model/inspect", json={"path": str(junk)})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["is_gguf"])
        self.assertFalse(body["exists"] is False)

    # -- select ------------------------------------------------------------------ #

    def test_select_rejects_missing_file(self):
        r = self.client.post(
            "/api/llm-model/select", json={"path": str(Path(self.tmp) / "nope.gguf")}
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("exist", r.json()["detail"])

    def test_select_rejects_non_gguf(self):
        junk = Path(self.tmp) / "notgguf.gguf"
        junk.write_bytes(b"JUNK" * 16)
        r = self.client.post("/api/llm-model/select", json={"path": str(junk)})
        self.assertEqual(r.status_code, 400)

    def test_select_accepts_other_gguf_with_warning(self):
        """A non-Qwen GGUF is allowed (llama-server serves any GGUF) but the
        identity check reports it honestly (never a silent substitution)."""
        kvs = (
            kv_str("general.name", "some-other-model")
            + kv_str("general.architecture", "llama")
            + kv_u32("general.file_type", 15)
        )
        gguf = Path(self.tmp) / "other.gguf"
        gguf.write_bytes(build_gguf(kvs=kvs))
        r = self.client.post("/api/llm-model/select", json={"path": str(gguf)})
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertFalse(payload["identity"]["is_expected"])
        self.assertTrue(payload["identity"]["reasons"])

    # -- reset --------------------------------------------------------------------- #

    def test_reset_clears_selection(self):
        gguf = Path(self.tmp) / "Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
        if not gguf.exists():
            gguf.write_bytes(build_gguf(kvs=QWEN_KVS))
        self.client.post("/api/llm-model/select", json={"path": str(gguf)})
        r = self.client.post("/api/llm-model/reset")
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertNotEqual(payload["source"], "user-selection")
        settings = LlmModelSettings.load(Path(self.tmp))
        self.assertEqual(settings.llm_model_path, "")

    # -- browse ----------------------------------------------------------------------- #

    def test_browse_reports_support_status(self):
        r = self.client.post("/api/llm-model/browse")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("supported", body)
        self.assertIn("status", body)
        import platform

        if platform.system() != "Windows":
            self.assertFalse(body["supported"])
            self.assertEqual(body["status"], "unsupported")


class TestWebUiLlmModelSection(unittest.TestCase):
    """The web UI "LLM model" section markers (Browse + manual path + reset)."""

    def test_ui_has_the_model_section(self):
        html = (REPO_ROOT / "web" / "voicemem.html").read_text("utf-8")
        for marker in (
            'id="llmBrowseBtn"',
            'id="llmManualPath"',
            'id="llmCheckBtn"',
            'id="llmUseBtn"',
            'id="llmResetBtn"',
            "llmLoadStatus",
            "/api/llm-model/browse",
            "/api/llm-model/inspect",
            "/api/llm-model/select",
            "/api/llm-model/reset",
            "llmWarnUnexpected",
            "never copied or uploaded",
        ):
            self.assertIn(marker, html)


import unittest.mock  # noqa: E402  (used inside tests)


if __name__ == "__main__":
    unittest.main()
