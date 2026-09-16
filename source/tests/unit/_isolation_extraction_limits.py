"""Isolated runner for test_extraction_output_limits.py (subprocess fixture).

Runs OUTSIDE the parent test process so importing voicemem here never pollutes
the parent's import state (same discipline as test_voicemem_controlled.py).
The fake `openai` module is injected into sys.modules BEFORE voicemem imports
(the vendor methods do `from openai import OpenAI` inside the call — the module
must be faked at the sys.modules level).

Usage: python _isolation_extraction_limits.py <action>
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent.parent / "vendor" / "voicemem"
sys.path.insert(0, str(VENDOR))


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason):
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, content, finish_reason):
        self.choices = [_Choice(content, finish_reason)]


class _Delta:
    def __init__(self, content):
        self.content = content


class _ChunkChoice:
    def __init__(self, content, finish_reason):
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Chunk:
    """One SSE chunk shape (the openai SDK yields these on stream=True)."""

    def __init__(self, content, finish_reason, usage=None):
        self.choices = [_ChunkChoice(content, finish_reason)]
        self.usage = usage


class _Stream:
    """Iterable + closeable stream (v0.10.2: the vendor legs issue streaming
    requests through llm_bg_gate.bg_chat_create — the cooperative
    cancellation transport)."""

    def __init__(self, chunks):
        self._it = iter(chunks)
        self.closed = False

    def close(self):
        self.closed = True

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class _Completions:
    def __init__(self, captured):
        self._captured = captured

    def create(self, **kw):
        self._captured.update(kw)
        if kw.get("max_tokens") is not None and kw["max_tokens"] <= 4:
            # emulate hard truncation: JSON cut mid-string
            text, finish = '{"memory": [{"id": "0", "text": "User trunca', "length"
        else:
            text, finish = json.dumps({
                "memory": [{"id": "0", "text": "User bought new running shoes",
                            "attributed_to": "user"}],
                "emotion": "", "traits": []}), "stop"
        # v0.10.2: the legs always stream — serve SSE-shaped chunks the way
        # llama-server does (deltas + a final finish_reason/usage chunk).
        pieces = [text[i:i + 64] for i in range(0, len(text), 64)]
        chunks = [_Chunk(p, None) for p in pieces]
        chunks.append(_Chunk("", finish, usage={"prompt_tokens": 1}))
        return _Stream(chunks)


class _Chat:
    def __init__(self, captured):
        self.completions = _Completions(captured)


class _FakeOpenAI:
    def __init__(self, **kw):
        self.chat = _Chat(_FAKE_CAPTURE)


_FAKE_CAPTURE: dict = {}


def _install_fake_openai() -> None:
    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI
    sys.modules["openai"] = mod


def main() -> None:
    action = sys.argv[1]
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    if action == "show-limits":
        from voicemem.leftbrain import extract_facts_openai as E
        print(json.dumps({"extract": E._EXTRACT_MAX_TOKENS,
                          "resolve": E._RESOLVE_MAX_TOKENS}))
        return

    _install_fake_openai()
    from voicemem.leftbrain import extract_facts_openai as E

    if action == "capture-extract":
        cfg = E.OpenAIAdditiveExtractorConfig(api_key="test-key", model="m")
        ext = E.OpenAIMem0V3AdditiveExtractor(cfg)
        try:
            out = ext.extract(new_messages=[{"role": "user", "content": "veszek edzocipot"}])
            ok, n, err = True, len(out), ""
        except Exception as exc:  # parse failures must propagate (fail closed)
            ok, n, err = False, None, f"{type(exc).__name__}: {exc}"
        print(json.dumps({**{k: v for k, v in _FAKE_CAPTURE.items()},
                          "parse_ok": ok, "n": n, "err": err[:80]}))

    elif action == "capture-resolve":
        res = E.ConflictResolver(E.OpenAIAdditiveExtractorConfig(api_key="k", model="m"))
        out = res.resolve(["User bought shoes"], [{"id": "u1", "text": "User has old shoes"}])
        print(json.dumps({**_FAKE_CAPTURE, "n_results": len(out)}))


if __name__ == "__main__":
    main()
