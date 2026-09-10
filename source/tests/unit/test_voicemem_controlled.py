"""Controlled VoiceMem foundation regression tests (v0.5.0 ownership model).

Covers the TASK's success criteria that must FAIL if the OpenAI embedding
defect is ever reintroduced or if the controlled-source deployment chain
regresses to the upstream-clone design:

  * source-level: VOICEMEM_PIN.json identity, vendor tree, patch markers,
    installer design (no upstream clone / no main-branch fallback),
    verify_m1 pin probe, bridge embedder injection, release staging
  * runtime (subprocess-isolated so the parent test process never gains an
    importable voicemem — the degraded-mode bridge tests must keep passing):
    identity attributes, _embed_text routing (injected embedder AND the
    no-injection local-E5 default), OpenAI client NEVER constructed on the
    memory-embedding path (fake openai module that explodes), DELETE guard
  * manifest: the voicemem block records the controlled-fork identity
  * assess tool: NULL / stale-dim trait detection on a synthetic DB
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR = ROOT / "vendor" / "voicemem"
PKG = VENDOR / "voicemem"
PIN_FILE = ROOT / "VOICEMEM_PIN.json"
POLICY_FILE = ROOT / "UPSTREAM_POLICY.md"
EXPECTED_COMMIT = "e8384e087bd2f44eb05fc7ae1a3c525ea8244179"

sys.path.insert(0, str(ROOT / "scripts"))


# --------------------------------------------------------------------------- #
# Source-level: pin + vendor identity
# --------------------------------------------------------------------------- #

class PinFileContractTests(unittest.TestCase):
    """VOICEMEM_PIN.json is the authoritative identity record."""

    def test_pin_file_exists_and_records_controlled_fork(self):
        self.assertTrue(PIN_FILE.is_file(), "VOICEMEM_PIN.json missing")
        pin = json.loads(PIN_FILE.read_text(encoding="utf-8"))
        self.assertEqual(pin.get("status"), "controlled-fork")
        self.assertEqual(pin.get("controlled_source"), "vendor/voicemem")
        prov = pin.get("provenance") or {}
        self.assertEqual(prov.get("upstream_commit"), EXPECTED_COMMIT)
        self.assertEqual(prov.get("upstream_tag"), "v0.0.1")
        self.assertEqual(prov.get("upstream_repo"),
                         "https://github.com/xzf-thu/VoiceMem.git")
        self.assertEqual(
            pin.get("embedding_policy", {}).get("dimensions"), 384
        )

    def test_patch_ledger_lists_every_local_patch(self):
        pin = json.loads(PIN_FILE.read_text(encoding="utf-8"))
        ids = {p.get("id") for p in pin.get("local_patches", [])}
        for required in ("VM-LOCAL-001", "VM-LOCAL-002", "VM-LOCAL-003",
                         "VM-LOCAL-004", "VM-LOCAL-005", "VM-LOCAL-006",
                         "VM-LOCAL-EN"):
            self.assertIn(required, ids, f"pin ledger missing {required}")
        ported = {p.get("id") for p in pin.get("ported_upstream_fixes", [])}
        for required in ("UPSTREAM-961efe8", "UPSTREAM-91d2e42",
                         "UPSTREAM-f535f9d", "UPSTREAM-e3cc965"):
            self.assertIn(required, ported, f"pin ledger missing {required}")

    def test_policy_file_declares_the_invariants(self):
        self.assertTrue(POLICY_FILE.is_file(), "UPSTREAM_POLICY.md missing")
        text = POLICY_FILE.read_text(encoding="utf-8")
        for needle in (
            "CONTROLLED FORK", EXPECTED_COMMIT,
            "Local E5 is the only memory embedder",
            "No new OpenAI runtime dependency",
            "HU/EN behaviour is preserved",
            "DELETE stays opt-in",
        ):
            self.assertIn(needle, text)

    def test_vendor_tree_is_not_a_git_clone(self):
        self.assertFalse((VENDOR / ".git").exists(),
                         "vendor/voicemem must not be a git checkout")
        self.assertFalse((VENDOR / ".clone_ref").exists(),
                         "stale .clone_ref from the old clone design")
        self.assertTrue((PKG / "__init__.py").is_file())
        self.assertTrue((VENDOR / "pyproject.toml").is_file())
        self.assertTrue((VENDOR / "LICENSE").is_file(),
                        "Apache-2.0 LICENSE must ship with the fork")

    def test_gitignore_tracks_the_vendor_source(self):
        gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertNotIn("vendor/voicemem/\n", gi.split("vendor/voicemem/**")[0],
                         ".gitignore still blanket-ignores vendor/voicemem")


# --------------------------------------------------------------------------- #
# Source-level: the defect fixes (all patch markers)
# --------------------------------------------------------------------------- #

class VendorPatchMarkerTests(unittest.TestCase):
    """Each controlled patch must be present in the vendored source."""

    def _read(self, rel: str) -> str:
        return (PKG / rel).read_text(encoding="utf-8")

    def test_vm_local_001_embed_text_routes_local(self):
        orch = self._read("orchestrator.py")
        self.assertIn("if self._embedder is not None:", orch)
        self.assertIn("_LOCAL_E5_CACHE_KEY", orch)
        # the OpenAI embeddings CALL must be gone from _embed_uncached
        import re
        m = re.search(r"def _embed_uncached.*?(?=\n    def )", orch, re.S)
        self.assertIsNotNone(m)
        body = m.group(0)
        self.assertIn("LocalE5Embedder", body)
        self.assertNotIn("client.embeddings.create(", body)
        self.assertNotIn("from openai import OpenAI", body)

    def test_vm_local_002_repo_default_is_local_e5(self):
        import re
        brain = self._read("leftbrain/brain.py")
        m = re.search(r"def _get_repo\(self\).*?(?=\n    def )", brain, re.S)
        self.assertIsNotNone(m)
        body = m.group(0)
        self.assertIn("LocalE5Embedder()", body)
        self.assertNotIn("OpenAILocalEmbedder(", body)

    def test_vm_local_003_defaults_factory_is_local_e5(self):
        import re
        defaults = self._read("utils/defaults.py")
        m = re.search(r"def embedding\(\).*?(?=\n    def )", defaults, re.S)
        self.assertIsNotNone(m)
        body = m.group(0)
        self.assertIn("LocalE5Embedder()", body)
        self.assertNotIn("OpenAILocalEmbedder(", body)

    def test_vm_local_004_trait_embedding_failure_is_loud(self):
        ts = self._read("rightbrain/traits_store.py")
        self.assertIn("_WARNED_VEC", ts)
        self.assertIn("rb_traits embedding FAILED", ts)
        # the search path records the query dim for the threshold binding
        self.assertIn("last_query_dim", ts)

    def test_vm_local_005_delete_guard_is_opt_in(self):
        mem0 = self._read("leftbrain/mem0_backend_store.py")
        self.assertIn("VOICEMEM_ALLOW_MEMORY_DELETE", mem0)
        self.assertIn("BLOCKED by the controlled-fork safety", mem0)

    def test_vm_local_006_runtime_identity_block(self):
        init = self._read("__init__.py")
        self.assertIn("CONTROLLED_FORK = True", init)
        self.assertIn(f'CONTROLLED_UPSTREAM_COMMIT = "{EXPECTED_COMMIT}"', init)
        self.assertIn("CONTROLLED_PATCHES", init)

    def test_upstream_ports_present(self):
        gc = self._read("utils/common/_graph_common.py")
        self.assertIn("if len(a) != len(b):", gc)
        rb = self._read("rightbrain/brain.py")
        self.assertIn("trait_min_sim", rb)
        self.assertIn("_TRAIT_MIN_SIM_BY_DIM = {384: 0.88}", rb)
        mem0 = self._read("leftbrain/mem0_backend_store.py")
        self.assertIn("_as_date", mem0)

    def test_english_localisation_marker(self):
        merged = self._read("leftbrain/merged_extraction.py")
        self.assertIn("IN ENGLISH (British spelling)", merged)


# --------------------------------------------------------------------------- #
# Source-level: deployment chain (installer / verify / manifest / bridge)
# --------------------------------------------------------------------------- #

class DeploymentChainSourceTests(unittest.TestCase):
    """No deployment path may clone upstream v0.0.1 (or main)."""

    def test_installer_has_no_upstream_clone(self):
        ps1 = (ROOT / "scripts" / "install_m1.ps1").read_text(encoding="utf-8")
        self.assertNotIn("$VoiceMemRepo", ps1,
                         "installer still defines an upstream repo variable")
        self.assertNotIn("$VoiceMemRef", ps1,
                         "installer still defines an upstream ref variable")
        self.assertNotIn("git clone $VoiceMemRepo", ps1)
        # the controlled install + pin verification must be present
        self.assertIn("pip install -e $VmDir", ps1)
        self.assertIn(EXPECTED_COMMIT, ps1)
        self.assertIn("VOICEMEM_PIN.json", ps1)
        self.assertIn("CONTROLLED_UPSTREAM_COMMIT", ps1)

    def test_verify_m1_has_pin_probe(self):
        ps1 = (ROOT / "scripts" / "verify_m1.ps1").read_text(encoding="utf-8")
        self.assertIn("CONTROLLED_UPSTREAM_COMMIT", ps1)
        self.assertIn("vezerlt forras + pin", ps1)

    def test_manifest_writer_uses_pin_not_git(self):
        py = (ROOT / "scripts" / "write_install_manifest.py").read_text(encoding="utf-8")
        self.assertIn("_voicemem_pin", py)
        self.assertIn("pin_verified", py)
        self.assertIn("runtime_import_path", py)
        self.assertNotIn("_clone_ref(vendor_dir)", py)

    def test_bridge_injects_local_e5_on_the_cli_path(self):
        py = (ROOT / "app" / "voicemem_bridge.py").read_text(encoding="utf-8")
        self.assertIn("def embedding_factory():", py)
        self.assertIn("LocalE5Embedder()", py)
        self.assertIn("embedding=embedding_factory", py)

    def test_web_layer_still_injects_local_e5(self):
        py = (ROOT / "app" / "web_server.py").read_text(encoding="utf-8")
        self.assertIn("LocalE5Embedder", py)

    def test_assess_tool_present_and_read_only(self):
        tool = ROOT / "scripts" / "assess_trait_embeddings.py"
        self.assertTrue(tool.is_file())
        text = tool.read_text(encoding="utf-8")
        self.assertIn("mode=ro", text, "assessor must open sqlite read-only")
        self.assertIn("DRY RUN ONLY", text)
        self.assertIn("--strict", text)

    def test_pin_verifier_tool_present(self):
        tool = ROOT / "scripts" / "verify_voicemem_pin.py"
        self.assertTrue(tool.is_file())


# --------------------------------------------------------------------------- #
# Runtime probes (subprocess-isolated: the parent process must stay clean so
# the degraded-mode bridge tests keep their semantics)
# --------------------------------------------------------------------------- #

class SubprocessRuntimeProbeTests(unittest.TestCase):
    """`import voicemem` from the vendor tree + the embedding regression."""

    def _run(self, code: str, timeout: int = 180) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile(
            "w", suffix="_vm_probe.py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(code)
            path = fh.name
        try:
            return subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "VOICEMEM_MEMORY_ROOT": tempfile.mkdtemp(prefix="vmprobe_")},
            )
        finally:
            os.unlink(path)

    def test_identity_attributes_from_vendor_import(self):
        proc = self._run(
            "import sys\n"
            f"sys.path.insert(0, r'{VENDOR}')\n"
            "import voicemem\n"
            "from pathlib import Path\n"
            "src = str(Path(voicemem.__file__).resolve())\n"
            "assert '/vendor/voicemem/' in src.replace(chr(92), '/'), src\n"
            "assert voicemem.CONTROLLED_FORK is True\n"
            f"assert voicemem.CONTROLLED_UPSTREAM_COMMIT == '{EXPECTED_COMMIT}'\n"
            "print('IDENTITY-OK', src)\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("IDENTITY-OK", proc.stdout)

    def test_embedding_regression_via_verify_tool(self):
        """The full instrumented proof (fake openai + fake E5)."""
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_voicemem_pin.py"),
             "--embedding", "--quiet"],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        self.assertIn("VERIFICATION OK", proc.stdout)

    def test_openai_client_never_constructed_on_embed_path(self):
        """Minimal inline proof: _embed_text with a broken local E5 must
        propagate the local failure — never silently construct OpenAI()."""
        proc = self._run(
            "import sys\n"
            f"sys.path.insert(0, r'{VENDOR}')\n"
            "marker = []\n"
            "class _Boom:\n"
            "    def __init__(self, *a, **k):\n"
            "        marker.append('openai-constructed')\n"
            "        raise RuntimeError('OpenAI constructed on memory path')\n"
            "import types\n"
            "fake = types.ModuleType('openai')\n"
            "fake.OpenAI = _Boom\n"
            "fake.AsyncOpenAI = _Boom\n"
            "sys.modules['openai'] = fake\n"
            "from voicemem.orchestrator import Orchestrator\n"
            "o = Orchestrator.__new__(Orchestrator)\n"
            "o._embedder = None\n"
            "o._base_url = 'http://127.0.0.1:8080/v1'\n"
            "try:\n"
            "    o._embed_text('probe')\n"
            "    raise AssertionError('embed unexpectedly succeeded')\n"
            "except Exception as e:\n"
            "    if 'OpenAI constructed' in str(e):\n"
            "        raise AssertionError('OPENAI PATH USED: ' + str(e))\n"
            "    # the local-E5 failure (ModuleNotFoundError for\n"
            "    # sentence_transformers, or a model-load error) is the\n"
            "    # EXPECTED outcome: the local failure propagates\n"
            "assert not marker, marker\n"
            "print('NO-OPENAI-OK: local failure propagated, no OpenAI construction')\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("NO-OPENAI-OK", proc.stdout)

    def test_delete_guard_blocks_without_opt_in(self):
        """The guard must refuse without the env opt-in (needs no mem0:
        we exercise the guard logic directly with a stub store object)."""
        proc = self._run(
            "import sys\n"
            f"sys.path.insert(0, r'{VENDOR}')\n"
            "from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore\n"
            "deleted = []\n"
            "class _StubMem0:\n"
            "    def delete(self, mid):\n"
            "        deleted.append(mid)\n"
            "store = Mem0BackendStore.__new__(Mem0BackendStore)\n"
            "store._mem0 = _StubMem0()\n"
            "store._path = None\n"
            "import os\n"
            "os.environ.pop('VOICEMEM_ALLOW_MEMORY_DELETE', None)\n"
            "ok = store.delete_memory('mem-123')\n"
            "assert ok is False, 'delete must report failure when blocked'\n"
            "assert not deleted, 'delete reached mem0 despite the guard'\n"
            "os.environ['VOICEMEM_ALLOW_MEMORY_DELETE'] = '1'\n"
            "ok2 = store.delete_memory('mem-123')\n"
            "assert ok2 is True\n"
            "assert deleted == ['mem-123']\n"
            "print('DELETE-GUARD-OK')\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("DELETE-GUARD-OK", proc.stdout)


# --------------------------------------------------------------------------- #
# Manifest block (controlled-fork identity)
# --------------------------------------------------------------------------- #

class ManifestVoicememBlockTests(unittest.TestCase):
    """write_install_manifest._voicemem_block records pin-derived identity."""

    def test_block_on_the_real_repo(self):
        from write_install_manifest import _voicemem_block  # noqa: E402
        block = _voicemem_block(ROOT)
        self.assertNotEqual(block.get("version"), "not_installed")
        self.assertTrue(block.get("controlled"))
        self.assertEqual(block.get("upstream_commit"), EXPECTED_COMMIT)
        self.assertEqual(block.get("upstream_tag"), "v0.0.1")
        self.assertEqual(block.get("git_commit"), EXPECTED_COMMIT)
        self.assertIn("VM-LOCAL-001", block.get("local_patches") or [])
        # sandbox: voicemem is not pip-installed in the parent process, so the
        # runtime resolution legitimately reports not-verified here; the
        # subprocess probes above prove the runtime identity separately.
        if block.get("runtime_import_path") is None:
            self.assertFalse(block.get("pin_verified"))

    def test_block_on_a_pinless_tree(self):
        from write_install_manifest import _voicemem_block  # noqa: E402
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "vendor" / "voicemem" / "voicemem").mkdir(parents=True)
            (root / "vendor" / "voicemem" / "voicemem" / "__init__.py").touch()
            block = _voicemem_block(root)
            self.assertIsNone(block.get("upstream_commit"))
            self.assertFalse(block.get("pin_verified"))

    def test_missing_vendor_reports_not_installed(self):
        from write_install_manifest import _voicemem_block  # noqa: E402
        with tempfile.TemporaryDirectory() as td:
            block = _voicemem_block(Path(td))
            self.assertEqual(block.get("version"), "not_installed")
            self.assertIsNone(block.get("commit"))


# --------------------------------------------------------------------------- #
# Trait embedding assessment tool
# --------------------------------------------------------------------------- #

class AssessTraitEmbeddingsTests(unittest.TestCase):
    """The Phase 8 read-only assessor detects NULL / stale-dim rows."""

    def _make_db(self, root: Path) -> Path:
        db = root / "memory" / "memory.sqlite"
        db.parent.mkdir(parents=True, exist_ok=True)
        if db.exists():
            db.unlink()
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE rb_traits (id TEXT PRIMARY KEY, user_id TEXT NOT NULL,"
            " slot TEXT NOT NULL, claim TEXT NOT NULL, embedding BLOB,"
            " confidence REAL NOT NULL DEFAULT 0.9, created_at TEXT NOT NULL,"
            " updated_at TEXT NOT NULL)"
        )

        def vec(n: int) -> bytes:
            return struct.pack(f"{n}f", *([0.1] * n))

        rows = [
            ("t1", "webspace_demo", "情绪", "gets nervous before reviews", vec(384)),
            ("t2", "webspace_demo", "喜好与厌恶", "dislikes long meetings", None),
            ("t3", "webspace_demo", "应对方式", "wants comfort when stressed", vec(1536)),
            ("t4", "voice_user", "表达风格", "gives examples first", vec(384)),
        ]
        conn.executemany(
            "INSERT INTO rb_traits (id, user_id, slot, claim, embedding, confidence,"
            " created_at, updated_at) VALUES (?,?,?,?,?,0.9,'now','now')", rows
        )
        conn.commit()
        conn.close()
        return db

    def test_assessment_counts_and_plan(self):
        from assess_trait_embeddings import assess  # noqa: E402
        with tempfile.TemporaryDirectory() as td:
            db = self._make_db(Path(td))
            report = assess(db, None, 384)
            self.assertEqual(report["total_traits"], 4)
            self.assertEqual(report["null_embeddings"], 1)
            self.assertEqual(report["non_null_embeddings"], 3)
            self.assertEqual(report["healthy_embeddings"], 2)
            self.assertEqual(report["stale_dimension_embeddings"], 1)
            self.assertEqual(report["unexpected_dimensions"], [1536])
            plan = report["backfill_plan"]
            self.assertEqual(len(plan["would_reembed_null"]), 1)
            self.assertEqual(len(plan["would_reembed_stale_dim"]), 1)
            self.assertIn("DRY RUN ONLY", plan["note"])

    def test_user_filter(self):
        from assess_trait_embeddings import assess  # noqa: E402
        with tempfile.TemporaryDirectory() as td:
            db = self._make_db(Path(td))
            report = assess(db, "voice_user", 384)
            self.assertEqual(report["total_traits"], 1)
            self.assertEqual(report["null_embeddings"], 0)

    def test_missing_table_is_not_an_error(self):
        from assess_trait_embeddings import assess  # noqa: E402
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "memory"
            root.mkdir()
            db = root / "memory.sqlite"
            sqlite3.connect(db).close()  # empty db
            report = assess(db, None, 384)
            self.assertFalse(report["rb_traits_table"])

    def test_read_only_mode_actually_blocks_writes(self):
        from assess_trait_embeddings import _open_ro  # noqa: E402
        with tempfile.TemporaryDirectory() as td:
            db = self._make_db(Path(td))
            conn = _open_ro(db)
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("UPDATE rb_traits SET claim='x' WHERE id='t1'")
            conn.close()


if __name__ == "__main__":
    unittest.main()
