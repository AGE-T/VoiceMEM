"""GGUF metadata reader + model identity validation (v0.4.16).

The LLM model file selector (web UI "LLM model" section) needs to answer,
for ANY absolute path the operator picks from any drive:

1. does the file exist?
2. is it readable?
3. is it actually a GGUF file (magic bytes)?
4. what model does it claim to be (``general.name`` / ``general.architecture``)?
5. which quantisation (``general.file_type`` -> label)?

Pure stdlib (``struct``) — importable everywhere, no torch/llama.cpp
dependency, no network. The reader walks the GGUF v2/v3 header:

    magic   "GGUF" (u32)          version u32 (2 or 3)
    tensors u64                   kv count u64
    per KV: key = u64 len + utf-8, type u32, value by type

v0.4.18 PARSER FIX (field report 2026-09): the v0.4.16/17 walkers had
memorised a WRONG value-type numbering (7=FLOAT64 8B, 8=BOOL 1B, 9=STRING,
10=ARRAY, 13=FLOAT16). The real GGUF spec (llama.cpp gguf-py ``constants.py``
``GGUFValueType``) is 7=BOOL(1B), 8=STRING(u64 len+bytes), 9=ARRAY(u32
elem type+u64 count+items), 10=UINT64(8B), 11=INT64(8B), 12=FLOAT64(8B) —
there is NO type 13. On the REAL Ollama blob the walk desynchronised at
the first STRING KV (type 8 was read as 1 byte), the next key length came
back as garbage, the exception blanked the identity and the file was
reported as "IDENTITY MISMATCH" with EMPTY name/architecture and
``general.file_type = -1`` — while the model itself was perfectly valid.
The synthetic test fixtures encoded the same wrong ids, so the old tests
stayed green; both are corrected against the authoritative table now.

It stops as soon as ``general.name`` + ``general.architecture`` +
``general.file_type`` + ``general.split.no`` are all seen (the header of a
20 GB Qwen sits in the first few KB — the file is NEVER read beyond the
metadata block).

v0.4.17 SPLIT DETECTION: a GGUF written by ``gguf-split`` carries
``general.split.no`` / ``general.split.count`` (llama.cpp LLM_KV_SPLIT_*).
Ollama stores multi-file HF pulls as SEPARATE content-addressed blobs —
each blob IS one shard. llama.cpp's split loading discovers sibling shards
by the ``<prefix>-NNNNN-of-NNNNN.gguf`` FILE NAME pattern, which a
``sha256-...`` blob can never match, so a shard blob CANNOT be loaded by
llama-server directly — this is reported explicitly (never a silent
broken selection).

The ``general.file_type`` numbering follows llama.cpp's ``llama_ftype``
(gguf-py ``constants.py`` LlamaFileType — "ALL VALUES SHOULD BE THE SAME
HERE AS THEY ARE OVER THERE"). NOTE for future maintainers: **IQ4_XS = 30**,
NOT 29 — the v0.4.15 finder script shipped 29 (= IQ2_M) and was corrected
in v0.4.16; do not "fix" it back based on stale blog tables.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

#: First four bytes of every GGUF file (b"GGUF" as little-endian u32 —
#: 0x46554747; gguf-py constants.py literally documents it as `"GGUF"`).
GGUF_MAGIC = 0x46554747

#: Max bytes we are ever willing to read while walking the header. The
#: metadata block of real models is a few hundred KB at most; a walk that
#: would exceed this is treated as a corrupt/hostile file instead of
#: reading an unbounded stream.
_MAX_HEADER_BYTES = 8 * 1024 * 1024

#: Max sane length of one metadata STRING (name/architecture URLs are short;
#: anything larger means the length prefix is garbage).
_MAX_STRING_BYTES = 4 * 1024 * 1024

#: Max sane element count of one metadata ARRAY (real token arrays are
#: ~150-260k elements; anything larger means the count prefix is garbage).
_MAX_ARRAY_COUNT = 16 * 1024 * 1024

#: general.file_type -> quant label (llama.cpp llama_ftype, gguf-py mirrors).
#: Unknown values render as "file_type N" — never guessed.
FILE_TYPE_LABELS: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1_SOME_F16",
    5: "Q4_2 (removed)",
    6: "Q4_3 (removed)",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
}

#: ``general.file_type`` of the ACTIVE production profile
#: (Qwen3.6 35B A3B IQ4_XS; bartowski IQ4_XS quant writes exactly 30).
IQ4_XS_FILE_TYPE = 30

#: Split-shard KV keys written by llama.cpp's gguf-split (LLM_KV_SPLIT_*).
SPLIT_NO_KEY = "general.split.no"
SPLIT_COUNT_KEY = "general.split.count"

#: Expected-model identity of the active production profile.
EXPECTED_MODEL_LABEL = "Qwen3.6 35B A3B IQ4_XS"
EXPECTED_MODEL_REPO = "hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS"

#: Tokens every one of which must appear in a normalised identity string
#: (GGUF general.name + file name) for the file to count as the exact
#: expected Qwen3.6 35B A3B MODEL (quant checked separately).
_EXPECTED_MODEL_TOKENS = ("qwen", "3", "6", "35b", "a3b")


@dataclass
class GgufInfo:
    """Validated identity of one candidate GGUF file (never raises)."""

    path: str = ""
    exists: bool = False
    readable: bool = False
    is_gguf: bool = False
    error: str = ""
    magic_ok: bool = False
    version: int = 0
    tensor_count: int = 0
    kv_count: int = 0
    name: str = ""           # general.name
    architecture: str = ""   # general.architecture
    file_type: int = -1      # general.file_type
    file_size: int = 0
    quant: str = ""          # FILE_TYPE_LABELS.get(file_type) or "file_type N"
    split_no: int = -1       # general.split.no (-1 = key absent)
    split_count: int = -1    # general.split.count (-1 = key absent)
    extras: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """File exists, is readable and carries a valid GGUF header."""
        return self.exists and self.readable and self.is_gguf

    @property
    def is_split_shard(self) -> bool:
        """True when this file is ONE SHARD of a gguf-split set.

        ``general.split.no`` is ZERO-indexed (shard 1 of 2 has no=0), so the
        mere PRESENCE of the key marks a shard. Such a file cannot be loaded
        by llama-server on its own (the loader discovers sibling shards by
        the ``-NNNNN-of-NNNNN.gguf`` file-name pattern). Ollama stores split
        HF pulls exactly this way: separate ``sha256-...`` blobs, each one
        shard.
        """
        return self.split_no >= 0 or self.split_count > 1


def quant_label(file_type: int) -> str:
    return FILE_TYPE_LABELS.get(file_type, f"file_type {file_type}")


def llama_server_load_verdict(info: GgufInfo) -> dict[str, Any]:
    """Can llama-server load this file directly by absolute path?

    YES for any complete single-file GGUF — llama.cpp checks the GGUF magic,
    NOT the file name/extension, so an Ollama content-addressed blob
    (``sha256-<hex>``, no extension) loads exactly like a renamed GGUF, in
    place, with no copy and no rename. NO for a split shard: the sibling
    shards cannot be discovered behind opaque blob names.
    """
    if not info.exists:
        return {"loadable": False, "reason": "file does not exist"}
    if not info.is_gguf:
        return {
            "loadable": False,
            "reason": info.error or "not a GGUF file (magic bytes are not 'GGUF')",
        }
    if info.is_split_shard:
        return {
            "loadable": False,
            "split_shard": True,
            "split_no": max(info.split_no, 0),
            "split_count": max(info.split_count, 0),
            "reason": (
                f"GGUF split shard (part {max(info.split_no, 0) + 1} of "
                f"{max(info.split_count, 0)}): llama-server cannot load a "
                "single shard directly — it discovers sibling shards by the "
                "-NNNNN-of-NNNNN.gguf file-name pattern, which a sha256-... "
                "Ollama blob never matches; the complete single-file GGUF "
                "is required (do NOT duplicate 20 GB just to rename shards)"
            ),
        }
    return {
        "loadable": True,
        "reason": (
            "complete single-file GGUF — llama-server loads it in place by "
            "absolute path (the GGUF magic decides, never the file name; an "
            "Ollama sha256-... blob works exactly like a named .gguf)"
        ),
    }


def read_gguf_metadata(path: str | os.PathLike[str]) -> GgufInfo:
    """Read the identity-relevant GGUF metadata (never raises).

    Every failure lands in ``info.error`` + the matching boolean flag so the
    REST layer can answer "why is this not usable" without a traceback.
    """
    p = Path(path)
    info = GgufInfo(path=str(p))
    try:
        info.exists = p.is_file()
    except OSError:
        info.exists = False
    if not info.exists:
        info.error = "file does not exist"
        return info
    try:
        info.file_size = p.stat().st_size
    except OSError:
        pass
    try:
        with open(p, "rb") as fh:
            head = fh.read(4)
            info.readable = True
    except OSError as exc:
        info.error = f"file is not readable: {exc}"
        return info
    if head != b"GGUF":
        info.error = "not a GGUF file (magic bytes are not 'GGUF')"
        return info
    info.magic_ok = True
    try:
        info.is_gguf, info.version, info.name, info.architecture, \
            info.file_type, info.tensor_count, info.kv_count, \
            info.split_no, info.split_count = _walk_header(
                p, info.file_size
            )
    except Exception as exc:  # noqa: BLE001 - any walk failure = invalid
        info.error = f"GGUF header walk failed: {exc}"
        info.is_gguf = False
        return info
    if info.version not in (2, 3):
        # v1 exists only in pre-release converters; llama-server refuses it.
        info.error = f"unsupported GGUF version {info.version} (need 2 or 3)"
        info.is_gguf = False
    info.quant = quant_label(info.file_type)
    return info


def _walk_header(
    p: Path, file_size: int
) -> tuple[bool, int, str, str, int, int, int, int, int]:
    """Walk magic/version/tensors/kv-count + the KV metadata block.

    Returns ``(is_gguf, version, general.name, general.architecture,
    general.file_type, tensor_count, kv_count, split_no, split_count)``.
    ``split_no``/``split_count`` stay -1 when the keys are absent (a
    complete, non-split GGUF). Raises on truncated or absurd structures
    (mapped to a validation error by the caller).
    """
    with open(p, "rb") as fh:
        def read_exact(n: int) -> bytes:
            if n < 0 or n > _MAX_HEADER_BYTES:
                raise ValueError(f"absurd block size {n}")
            buf = fh.read(n)
            if len(buf) != n:
                raise ValueError("truncated GGUF header")
            return buf

        magic, version, tensor_count, kv_count = struct.unpack(
            "<IIQQ", read_exact(4 + 4 + 8 + 8)
        )
        if magic != GGUF_MAGIC:
            return False, 0, "", "", -1, 0, 0, -1, -1

        name = architecture = ""
        file_type = -1
        split_no = -1
        split_count = -1
        # Value-type ids — the ACTUAL GGUF spec table (llama.cpp gguf-py
        # constants.py GGUFValueType; verified against the real 2026-09
        # Ollama blob field report after the v0.4.17 parser returned an
        # EMPTY identity on a valid file):
        #   0=UINT8(1B) 1=INT8(1B) 2=UINT16(2B) 3=INT16(2B) 4=UINT32(4B)
        #   5=INT32(4B) 6=FLOAT32(4B) 7=BOOL(1B) 8=STRING(u64 len+bytes)
        #   9=ARRAY(u32 elem type+u64 count+items) 10=UINT64(8B)
        #   11=INT64(8B) 12=FLOAT64(8B)  — there is NO type 13.
        _SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                         10: 8, 11: 8, 12: 8}
        _SCALAR_FMTS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                        6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        _STRING, _ARRAY = 8, 9

        def _skip_array(depth: int = 0) -> None:
            """Skip one ARRAY value (elem type u32 + count u64 + items)."""
            etype, ecount = struct.unpack("<IQ", read_exact(12))
            if ecount > _MAX_ARRAY_COUNT:
                raise ValueError(f"absurd array count {ecount}")
            if etype == _STRING:  # string array: per-item u64 len + bytes
                for _ in range(ecount):
                    slen = struct.unpack("<Q", read_exact(8))[0]
                    if slen > _MAX_STRING_BYTES:
                        raise ValueError(f"absurd string length {slen}")
                    read_exact(slen)
            elif etype == _ARRAY:  # nested array (spec-legal, unused in files)
                if depth >= 4:
                    raise ValueError("GGUF arrays nested too deeply")
                for _ in range(ecount):
                    _skip_array(depth + 1)
            elif etype in _SCALAR_SIZES:
                read_exact(ecount * _SCALAR_SIZES[etype])
            else:
                raise ValueError(
                    f"unsupported GGUF array element type {etype}"
                )

        for _ in range(kv_count):
            key_len = struct.unpack("<Q", read_exact(8))[0]
            if key_len > _MAX_STRING_BYTES:
                raise ValueError(f"absurd key length {key_len}")
            key = read_exact(key_len).decode("utf-8", errors="replace")
            vtype = struct.unpack("<I", read_exact(4))[0]
            value: Any = None
            if vtype == _STRING:  # string: u64 len + bytes
                slen = struct.unpack("<Q", read_exact(8))[0]
                if slen > _MAX_STRING_BYTES:
                    raise ValueError(f"absurd string length {slen}")
                value = read_exact(slen).decode("utf-8", errors="replace")
            elif vtype == _ARRAY:  # array: skipped (never captured)
                _skip_array()
            elif vtype in _SCALAR_SIZES:
                value = struct.unpack(
                    _SCALAR_FMTS[vtype], read_exact(_SCALAR_SIZES[vtype])
                )[0]
            else:
                raise ValueError(f"unknown GGUF value type {vtype}")
            if key == "general.name" and isinstance(value, str):
                name = value
            elif key == "general.architecture" and isinstance(value, str):
                architecture = value
            elif key == "general.file_type" and isinstance(value, int):
                file_type = value
            elif key == SPLIT_NO_KEY and isinstance(value, int):
                split_no = value
            elif key == SPLIT_COUNT_KEY and isinstance(value, int):
                split_count = value
            # v0.4.17: keep walking until the split keys are ALSO seen (or
            # the KV block ends) — a shard's identity KVs sit before the
            # split KVs, and stopping early would hide the "not loadable
            # directly" verdict. gguf-split always writes split.no/count
            # together, so requiring both avoids half-read state. The
            # metadata block is small; the walk stays bounded.
            if (
                name
                and architecture
                and file_type >= 0
                and split_no >= 0
                and split_count >= 0
            ):
                break  # identity + split info complete — stop here
        return (
            True, version, name, architecture, file_type,
            tensor_count, kv_count, split_no, split_count,
        )


def _normalise(s: str) -> str:
    """Lowercase, separators to single spaces, collapse repeats."""
    out = []
    prev_space = False
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
            prev_space = True
    return "".join(out).strip()


def expected_model_check(info: GgufInfo) -> dict[str, Any]:
    """Identity check against the ACTIVE production profile.

    The MODEL identity (Qwen3.6 35B A3B) and the QUANT identity (IQ4_XS =
    ``general.file_type`` 30) are checked SEPARATELY, so an IQ4_XS of a
    different model — or a Qwen3.6 35B A3B in another quant — is honestly
    reported as a mismatch, never silently accepted. Matching uses BOTH the
    GGUF metadata and the file name (converters occasionally omit
    ``general.name``; the file name of an Ollama blob is opaque, in which
    case metadata decides).

    Returns a JSON-ready dict: ``{"is_expected", "model_match",
    "quant_match", "reasons", "detected"}``.
    """
    reasons: list[str] = []
    haystack = _normalise(
        f"{info.name} {info.architecture} {Path(info.path).name}"
    )
    token_misses = [t for t in _EXPECTED_MODEL_TOKENS if t not in haystack]
    # "3" and "6" already appear inside "qwen3 6"/"35b"/"a3b" tokens; require
    # the version pair only as "3 6" adjacency OR "3 6" as separate tokens,
    # which the normalised haystack covers via "qwen3 6 35b a3b".
    version_ok = "3 6" in haystack or "36" in haystack
    model_match = (
        not token_misses
        and version_ok
        and "qwen" in haystack
        and "35b" in haystack
        and "a3b" in haystack
    )
    if not model_match:
        reasons.append(
            "the file does not identify as the expected Qwen3.6 35B A3B model "
            f"(general.name={info.name!r}, architecture={info.architecture!r})"
        )
    quant_match = info.file_type == IQ4_XS_FILE_TYPE
    if not quant_match:
        if info.file_type < 0:
            reasons.append("general.file_type is missing from the GGUF metadata")
        else:
            reasons.append(
                f"the file's quantisation is {quant_label(info.file_type)} "
                f"(general.file_type={info.file_type}), not IQ4_XS"
            )
    detected = info.name or Path(info.path).stem
    if info.quant:
        detected = f"{detected} · {info.quant}"
    result = {
        "is_expected": bool(model_match and quant_match),
        "model_match": bool(model_match),
        "quant_match": bool(quant_match),
        "reasons": reasons,
        "detected": detected,
        "expected": EXPECTED_MODEL_LABEL,
        "expected_repo": EXPECTED_MODEL_REPO,
    }
    # v0.4.17: a split shard is a loadability problem even when the identity
    # matches — surface it in the same check the UI renders.
    if info.is_gguf and info.is_split_shard:
        result["split_shard"] = True
        result["split_no"] = max(info.split_no, 0)
        result["split_count"] = max(info.split_count, 0)
        result["loadable"] = False
        result["reasons"].append(
            llama_server_load_verdict(info)["reason"]
        )
    elif info.is_gguf and info.exists and info.readable:
        result["loadable"] = True
    return result
