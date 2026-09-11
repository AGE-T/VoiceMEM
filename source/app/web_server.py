"""LOCAL web backend for the existing VoiceMem web UI (CONTRACT: web/run.py protocol).

Serves ``web/voicemem.html`` (the existing VoiceMem UI, adapted) and speaks the
SAME WebSocket / REST protocol the original demo spoke, but every model call is
LOCAL:

    browser  ->  this backend  ->  agent pipeline  ->  llama-server :8080

Pipeline (LOCAL mode), v0.6.0 modular ASR:

    microphone (24 kHz PCM16 over WS)
      -> Silero VAD            (app.vad, official 64-sample context feed)
      -> selected ASR engine   (app.asr_core registry: parakeet default,
                               nemotron selectable; ONE authoritative path,
                               no engine fallback, structured AsrError)
      -> VoiceMem              (voicemem text_mode; per-space user_id)
         embedding: multilingual E5 small, CPU (injected into VoiceMem)
      -> Qwen3.6 35B A3B IQ4_XS (llama-server /v1/chat/completions, SSE)
      -> Piper HU/EN           (app.tts, subprocess) -> 24 kHz PCM16 to browser

The legacy Qwen integration lives on only as the clearly-marked non-production
migration module ``app/asr.py`` (kept for its historical tests); it is NOT in
the engine registry and NOT imported by this server.

M2 emotion (emotion2vec+ M2) stays enabled. M3 speaker stays optional and is
DISABLED by default here (config.enable_speaker is ignored — the web layer
never builds a recognizer). Scene recognition stays permanently excluded
(text_mode only).

DEMO mode (``--mock`` / ``VOICEMEM_WEB_MOCK=1``): every component is replaced
by an offline stand-in (scripted LLM, tone TTS, scripted MockAsrEngine
implementing the same engine contract, in-memory store) so the UI can be
exercised on machines without models/GPU. The UI clearly shows "DEMO" in that
mode. No OpenAI call exists in either mode.

The browser must NEVER call OpenAI: this module does not import the openai
package at all; the only outbound HTTP goes to ``config.llama_server_url``
(llama.cpp, 127.0.0.1:8080) through :class:`app.llm.LlmClient` (httpx).

Import safety: heavy imports (torch/onnxruntime/voicemem) happen lazily inside
factories, so this module imports cleanly in the dependency-free sandbox.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# NOTE: fastapi is imported at MODULE level on purpose: `from __future__ import
# annotations` turns endpoint signatures into STRINGS, and FastAPI resolves
# them against module globals — a closure-scoped `WebSocket` import would make
# the /ws endpoint look like a query-parameter endpoint and reject every
# connection with HTTP 403 (silent, no traceback). Keeping the names here is
# the contract for every route defined in build_web_app().
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import AgentConfig
from app.gguf import (
    EXPECTED_MODEL_LABEL,
    EXPECTED_MODEL_REPO,
    GgufInfo,
    expected_model_check,
    llama_server_load_verdict,
    read_gguf_metadata,
)
from app.llm import LlmUnavailableError
from app.llm_model_settings import (
    SOURCE_DEFAULT,
    SOURCE_ENV,
    SOURCE_NONE,
    SOURCE_USER,
    LlmModelSettings,
    operator_default_candidates,
    read_llama_server_loaded_model,
    resolve_llm_model_path,
)
from app.native_picker import is_supported as picker_supported, pick_file
from app.teacher_persona import build_messages, build_system_prompt
from app.text_utils import LANG_EN, LANG_HU, SentenceStream, detect_language
from app.voice_settings import (
    PREVIEW_SENTENCES,
    VOICE_MODES,
    VoiceSettings,
    language_name,
    language_of_voice,
    voice_label,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 0. Constants
# ═══════════════════════════════════════════════════════════════════════════

#: The browser up/down-links at 24 kHz PCM16 (SAMPLE_RATE in voicemem.html).
WEB_SAMPLE_RATE = 24000
#: VAD/ASR sample rate (agent pipeline standard).
PIPE_SAMPLE_RATE = 16000
#: Default web port (START flow prints http://127.0.0.1:8787).
DEFAULT_WEB_PORT = 8787
DEFAULT_WEB_HOST = "127.0.0.1"
#: VAD frame = 512 samples @ 16 kHz (Silero v5/v6 contract).
VAD_FRAME_SAMPLES = 512

#: Pipeline components surfaced in the debug view, in display order.
#: v0.6.0: "mic" (browser uplink health) joined the chain — the UI stage
#: list is MIC -> VAD -> ASR -> VoiceMEM -> E5 -> LLM -> TTS now.
COMPONENT_KEYS = ("mic", "vad", "asr", "memory", "embedding", "llm", "tts")

#: How long a turn's memory/emotion stage may hold the reply (ms) before the
#: prosody emotion is demoted to a background tag_update.
_EMOTION_WAIT_S = 0.6
#: v0.4.13 (field report: "text understood but never reaches the LLM"):
#: hard ceiling on the emotion analyzer INITIALISATION inside a turn. The
#: construction imports funasr (-> torch; 30-180 s cold on Windows, worse
#: with antivirus, and a concurrent warm-up import serialises behind the
#: same module lock) - before this bound the first turn could freeze here
#: forever with no error, no answer and no diag line. On timeout the turn
#: CONTINUES with emotion disabled (semantic-only) and still reaches the
#: LLM; the background thread keeps loading, later turns pick it up.
_EMOTION_INIT_TIMEOUT_S = 5.0
#: LLM history window (entries) passed to the teacher persona.
_HISTORY_WINDOW = 8

#: v0.4.5 context guard (field report #6): llama-server runs with a fixed
#: 8192-token context and answers up to ``llm_max_tokens`` (512); a
#: long-running conversation (large memory block + 8 history entries) once
#: produced a 10459-token prompt the server rejected outright ("request ...
#: exceeds the available context size") and the turn lost its reply. Tokens
#: are estimated from characters (~2-3 chars/token for HU/EN text on the
#: Qwen3.6 tokenizer - deliberately conservative) and the prompt is trimmed
#: OLDEST-first: memory block capped, then oldest history entries dropped.
_MEMORY_CONTEXT_MAX_CHARS = 6000
_PROMPT_CHAR_BUDGET = 14000


def _cap_memory_context(memory_context: str) -> str:
    """v0.4.5: hard character cap on the memory block fed to the system prompt.

    Cuts at the last line boundary inside the cap (when one exists in the
    second half of the block) and appends a visible truncation note, so the
    model knows the block was shortened on purpose.
    """
    if not memory_context or not memory_context.strip():
        return memory_context
    text = memory_context.strip()
    if len(text) <= _MEMORY_CONTEXT_MAX_CHARS:
        return text
    cut = text[:_MEMORY_CONTEXT_MAX_CHARS]
    newline = cut.rfind("\n")
    if newline > _MEMORY_CONTEXT_MAX_CHARS // 2:
        cut = cut[:newline]
    return cut.rstrip() + "\n[... older memories truncated to fit the context]"


def _fit_history_budget(
    system_prompt: str, history: list[dict], user_text: str
) -> list[dict]:
    """v0.4.5: drop the OLDEST history entries until the prompt fits the budget.

    The current user text and the (already capped) system prompt always
    stay; only conversation history is shed, newest-first retention. Never
    raises and never returns more entries than given.
    """
    def _total(entries: list[dict]) -> int:
        return (
            len(system_prompt)
            + len(user_text)
            + sum(len(str(e.get("content", ""))) for e in entries)
        )

    if _total(history) <= _PROMPT_CHAR_BUDGET:
        return history
    trimmed = list(history)
    while trimmed and _total(trimmed) > _PROMPT_CHAR_BUDGET:
        trimmed.pop(0)
    return trimmed

#: v0.4.2 speech-chain safety: force a turn out of an utterance that stays in
#: speech longer than this (a VAD stuck above threshold on music/continuous
#: noise used to mean "I speak but nothing ever happens" - no speech_end, no
#: turn, no feedback). 12 s of continuous Hungarian/English speech is already
#: a very long sentence; beyond that we flush what we have.
_MAX_UTTERANCE_MS = 12_000

_ROOT = Path(__file__).resolve().parent.parent
_WEB_DIR = _ROOT / "web"

# ═══════════════════════════════════════════════════════════════════════════
# 1. Component status tracking (pipeline debug view)
# ═══════════════════════════════════════════════════════════════════════════

#: Component states rendered by the UI.
_STATE_INIT = "init"
_STATE_READY = "ready"
_STATE_PROCESSING = "processing"
_STATE_ERROR = "error"
_STATE_MISSING = "missing"
_STATE_MOCKED = "mocked"


class ComponentStatus:
    """One pipeline component's live state (thread-safe via the event loop)."""

    def __init__(self, key: str, detail: str = "") -> None:
        self.key = key
        self.detail = detail
        self.state = _STATE_INIT
        self.last_ms: Optional[float] = None
        self.error = ""
        self._t0 = 0.0
        self._prev_state = ""

    # -- lifecycle (called from the pipeline) -------------------------------- #

    def set_ready(self, detail: str = "") -> None:
        self.state = _STATE_READY
        if detail:
            self.detail = detail
        self.error = ""

    def set_missing(self, detail: str = "") -> None:
        self.state = _STATE_MISSING
        if detail:
            self.detail = detail

    def set_mocked(self, detail: str = "") -> None:
        self.state = _STATE_MOCKED
        if detail:
            self.detail = detail

    def set_error(self, message: str) -> None:
        self.state = _STATE_ERROR
        self.error = str(message)[:300]

    def begin(self) -> None:
        if self.state != _STATE_PROCESSING:
            # Remember where we came from: a transient processing span must
            # return to the PREVIOUS state (missing stays missing, error
            # stays error) - not magically become "ready".
            self._prev_state = self.state
        self.state = _STATE_PROCESSING
        self._t0 = time.perf_counter()

    def end(self) -> Optional[float]:
        """Finish a processing span; records last_ms, back to the prior state."""
        if self._t0:
            self.last_ms = round((time.perf_counter() - self._t0) * 1000.0, 1)
            self._t0 = 0.0
        if self.state == _STATE_PROCESSING:
            self.state = self._prev_state or _STATE_READY
        self._prev_state = ""
        return self.last_ms

    def fail(self, message: str) -> None:
        if self._t0:
            self.last_ms = round((time.perf_counter() - self._t0) * 1000.0, 1)
            self._t0 = 0.0
        self.set_error(message)

    # -- serialisation -------------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "state": self.state,
            "detail": self.detail,
            "last_ms": self.last_ms,
            "error": self.error,
        }


class PipelineStatus:
    """All six components + llama-server health + live session diagnostics."""

    def __init__(self) -> None:
        self.components = {k: ComponentStatus(k) for k in COMPONENT_KEYS}
        self.llama_healthy: Optional[bool] = None
        # v0.4.4: set when the LLM startup probe FAILED (empty reply / timeout
        # / error). "llama-server /health == 200" must not repaint that error
        # green - that is exactly the v0.4.3 silent-failure regression.
        self.llm_probe_failed = False
        self.llama_url = ""
        self.mode = "local"
        self.space = ""
        self.turn: dict = {}
        # v0.4.2 voice selection (UI Voice section): mode + selected voices +
        # the voice that actually spoke the last reply (runtime status:
        # "Voice: Imre / Language: Hungarian"). Maintained by WebComponents.
        self.voice: dict = {
            "mode": "auto",
            "hu": "",
            "en": "",
            "active": "",
            "language": "",
        }
        # Live session diagnostics (the "where does the chain stop" view).
        # One status object is shared by all WS sessions — the most recently
        # active session owns these fields (last writer wins; browsers almost
        # always run a single session). Populated by WebSession only.
        self.session: dict = {
            "connected": False,
            "mic_frames": 0,
            "last_frame_age_s": None,
            "vad_level": 0.0,
            # v0.4.8: rolling VAD peak + the configured threshold — the two
            # numbers that decide "the chain never even reaches ASR".
            # v0.4.11: the seed matches the new 0.25 default; WebComponents
            # .build() overwrites it with the CONFIGURED value so the
            # pre-session /api/pipeline response never shows a stale number.
            "vad_peak": 0.0,
            "vad_threshold": 0.25,
            "vad_in_speech": False,
            "speech_ms": 0,
            "utterances": 0,
            "turns": 0,
            "last_event": "",
            "events": [],
        }

    def snapshot(self) -> dict:
        session = dict(self.session)
        session["events"] = list(self.session.get("events", []))
        return {
            "mode": self.mode,
            "space": self.space,
            "llama": {
                "healthy": self.llama_healthy,
                "url": self.llama_url,
            },
            "components": {k: c.to_dict() for k, c in self.components.items()},
            "turn": dict(self.turn),
            "voice": dict(self.voice),
            "session": session,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Audio helpers (pure numpy)
# ═══════════════════════════════════════════════════════════════════════════


def resample_linear(x: Any, from_sr: int, to_sr: int) -> Any:
    """Linear-interpolation resampler for 1-D float32 arrays (never raises).

    v0.6.0 note (TASK-A forensics): the wire stays 24 kHz PCM16 and this
    converts 24k -> 16k linearly. Measured impact on the ASR boundary is
    small (chain correlation r=0.998 end to end; the ASR failure was the
    VAD context bug + the Qwen engine, NOT this resampler — see
    scripts/asr_forensics.py). Kept as-is deliberately: the audio path is
    NOT the broken boundary, and changing the wire format mid-task would
    churn the browser contract for no proven gain.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0 or from_sr == to_sr:
        return x
    n_out = int(x.size * to_sr / from_sr)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    idx = np.arange(n_out, dtype=np.float64) * (float(from_sr) / float(to_sr))
    i0 = np.floor(idx).astype(np.int64)
    i0 = np.clip(i0, 0, x.size - 1)
    i1 = np.minimum(i0 + 1, x.size - 1)
    frac = (idx - i0).astype(np.float32)
    return (x[i0] * (1.0 - frac) + x[i1] * frac).astype(np.float32)


def pcm16_to_float32(raw: bytes) -> Any:
    """PCM16 little-endian bytes -> float32 [-1, 1] (odd tails dropped)."""
    import numpy as np

    if len(raw) < 2:
        return np.zeros(0, dtype=np.float32)
    even = len(raw) - (len(raw) % 2)
    i16 = np.frombuffer(raw[:even], dtype=np.int16)
    return i16.astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(x: Any) -> bytes:
    """float32 [-1, 1] -> PCM16 little-endian bytes (clipped, never raises)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return b""
    clipped = np.clip(x, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Memory layer (REAL VoiceMem per space / DEMO in-memory store)
# ═══════════════════════════════════════════════════════════════════════════

_SPACE_SAFE_RE = re.compile(r"[^0-9A-Za-z_-]")


def sanitize_space_name(name: str) -> str:
    """Keep [0-9A-Za-z_-], cap 32 chars; '' when nothing usable is left."""
    safe = _SPACE_SAFE_RE.sub("", (name or "").strip())[:32]
    return safe


#: Right-brain prefix/suffix cleanup, shared with the original UI payload.
_RB_PREFIX_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?(?:[⚠✓✱*]\s*)?(?:[^：:\s]{2,8}[：:]\s*)?")
_RB_SUFFIX_RE = re.compile(r"[（(]\s*(?:下次|next time|内心OS|inner note)\s*[：:].*$", re.S | re.I)


def clean_rb_content(content: str) -> str:
    """Strip the date/slot prefix and advisory suffix from right-brain notes."""
    t = _RB_PREFIX_RE.sub("", str(content or ""))
    t = _RB_SUFFIX_RE.sub("", t)
    return t.strip()


#: Right-brain trait slots → English display labels. The vendored VoiceMem
#: package keeps the five slots as Chinese enum values (TraitStore.add
#: validates slot against them — they are machine-facing identifiers), but
#: the UI must never show them raw: the v0.4.6 field report was memory cards
#: reading "Profile · 应对方式 ×1". Unknown values pass through unchanged.
_SLOT_EN = {
    "情绪": "emotion",
    "应对方式": "coping style",
    "表达风格": "expression style",
    "思维模式": "thinking style",
    "喜好与厌恶": "likes and dislikes",
}


def localise_slot(slot: str) -> str:
    """Translate a vendor trait-slot enum value for display (unknown → as-is)."""
    return _SLOT_EN.get(str(slot or "").strip(), slot)


class RealMemoryLayer:
    """VoiceMem facades per Memory Space (text_mode, local E5, llama-server LLM).

    A "space" is one VoiceMem facade with ``user_id = webspace_<name>`` inside
    the agent's ``memory_root`` — the same isolation the M3 bridge uses. Facades
    are cached; switching spaces rebinds the active one without restarting.
    """

    def __init__(self, config: AgentConfig, status: PipelineStatus) -> None:
        self._config = config
        self._status = status
        self._spaces: dict[str, Any] = {}
        self._active = "demo"
        self._failed = False
        self._registry_path = Path(config.memory_root_path) / "web_spaces.json"
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        # v0.4.4: pin voicemem's E5 embedder to the LOCAL offline copy BEFORE
        # the first facade construction (the embedder resolves its model name
        # at import time; with the runtime offline env a hub fallback fails).
        # v0.5.1 (TASK 1): pin the vendor's OpenAI-compatible LLM target env
        # the same way — the vendor reads OPENAI_BASE_URL/MODEL/API_KEY at
        # call time and silently degrades (or aims at the SDK's built-in
        # CLOUD default) when a raw shell forgot to source
        # config/env.local.ps1.
        from app.voicemem_bridge import pin_e5_local_model, pin_vendor_llm_env

        pin_e5_local_model(config)
        pin_vendor_llm_env(config)
        # Make sure the default space exists on first boot.
        for name in self._registered():
            if name == "demo":
                self._active = "demo"

    # -- registry ------------------------------------------------------------ #

    def _registered(self) -> list[str]:
        names = ["demo"]
        try:
            if self._registry_path.is_file():
                data = json.loads(self._registry_path.read_text("utf-8"))
                extra = [str(n) for n in data.get("spaces", []) if n]
                for n in extra:
                    if n not in names:
                        names.append(n)
        except Exception as exc:  # noqa: BLE001 - registry is advisory
            logger.debug("web spaces registry unreadable: %s", exc)
        return names

    def _register(self, name: str) -> None:
        names = self._registered()
        if name not in names:
            names.append(name)
        try:
            self._registry_path.write_text(
                json.dumps({"spaces": names}, indent=2), "utf-8"
            )
        except OSError as exc:
            logger.warning("could not persist web spaces registry: %s", exc)

    # -- construction --------------------------------------------------------- #

    def _build_facade(self, safe: str) -> Any:
        """Construct ONE VoiceMem facade (called in a worker thread)."""
        from voicemem import VoiceMem

        cfg = self._config

        def embedding_factory():  # local multilingual E5, 0 network
            from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder

            return LocalE5Embedder()

        return VoiceMem(
            mode="text_mode",
            user_id=f"webspace_{safe}",
            base_url=cfg.llama_server_url,
            top_k=cfg.top_k,
            memory_root=str(cfg.memory_root_path),
            embedding=embedding_factory,
        )

    def _facade(self, safe: str) -> Optional[Any]:
        if safe in self._spaces:
            return self._spaces[safe]
        if self._failed:
            return None
        try:
            vm = self._build_facade(safe)
            self._spaces[safe] = vm
            return vm
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the server
            self._failed = True
            self._status.components["memory"].set_error(str(exc))
            logger.warning("VoiceMem facade construction failed: %s", exc)
            return None

    # -- MemoryLayer API (shared with DemoMemoryLayer) ------------------------ #

    @property
    def active(self) -> str:
        return self._active

    def list_spaces(self) -> list[dict]:
        out = []
        for name in self._registered():
            out.append(
                {
                    "id": name,
                    "name": name,
                    "count": self._count(name),
                    "active": name == self._active,
                    "open": name in self._spaces,
                }
            )
        return out

    def _count(self, safe: str) -> int:
        vm = self._spaces.get(safe) or self._facade(safe)
        if vm is None:
            return 0
        try:
            uid = f"webspace_{safe}"
            entries = vm._o._get_repo()._vector_store.list_entries(user_id=uid)
            return len([e for e in entries if e.get("role") != "assistant"])
        except Exception:  # noqa: BLE001
            return 0

    def create_space(self, name: str) -> dict:
        safe = sanitize_space_name(name)
        if not safe:
            raise ValueError("empty name")
        if safe in self._registered() and safe != "demo":
            raise FileExistsError(safe)
        self._register(safe)
        self._facade(safe)  # construct eagerly so the first turn is warm
        return {"id": safe, "name": safe, "count": 0}

    def use_space(self, name: str) -> str:
        safe = sanitize_space_name(name)
        if not safe:
            raise ValueError("empty name")
        registered = self._registered()
        if safe not in registered and safe != "demo":
            registered.append(safe)
            self._register(safe)
        vm = self._facade(safe)
        if vm is None and self._failed:
            raise RuntimeError("VoiceMem unavailable")
        self._active = safe
        return safe

    def search(self, query: str, emotion: str = "") -> Any:
        vm = self._facade(self._active)
        if vm is None:
            return None
        return vm.search(query, emotion=emotion or None)

    def ingest(self, text: str, agent_reply: str = "") -> None:
        vm = self._facade(self._active)
        if vm is None:
            return
        vm.ingest(text, agent_reply=agent_reply or None, async_facts=True)

    def classify(self, query: str) -> Any:
        vm = self._facade(self._active)
        if vm is None:
            return None
        return vm.classify(query)

    # -- snapshot for the memory graph ---------------------------------------- #

    def snapshot(self, limit: int = 48) -> dict:
        vm = self._facade(self._active)
        if vm is None:
            return {"left": [], "right": []}
        left: list[dict] = []
        right: list[dict] = []
        uid = f"webspace_{self._active}"
        try:
            repo = vm._o._get_repo()
            entries = repo._vector_store.list_entries(user_id=uid)
            slot_of = self._slot_map(repo, uid)
            entries = [e for e in entries if e.get("role") != "assistant"]
            for e in entries[:limit]:
                d = str(e.get("date", ""))
                left.append(
                    {
                        "text": e.get("text", ""),
                        "date": d if d[:4].isdigit() else "",
                        "slot": slot_of.get(e.get("id", ""), "daily_life"),
                        "entities": [],
                        "hit": False,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - graph is best-effort
            logger.debug("left-brain snapshot failed: %s", exc)
        try:
            right = self._right_tree(vm, uid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("right-brain snapshot failed: %s", exc)
        return {"left": left, "right": right}

    @staticmethod
    def _slot_map(repo: Any, uid: str) -> dict:
        out: dict = {}
        try:
            from voicemem.leftbrain.cognitive_graph.types import SlotV2

            cog = repo._cognitive_store
            for slot in SlotV2:
                for mid in cog.memory_ids_for_slots(uid, [slot]):
                    out.setdefault(mid, slot.value)
        except Exception as exc:  # noqa: BLE001
            logger.debug("slot map failed: %s", exc)
        return out

    @staticmethod
    def _right_tree(vm: Any, uid: str) -> list[dict]:
        out: list[dict] = []
        store = vm._o._right._traits()
        for t in store.all(uid, per_slot=6):
            notes = [
                {
                    "text": e.quote,
                    "emotion": getattr(e, "emotion", "") or "",
                    "cause": getattr(e, "cause", "") or "",
                }
                for e in (getattr(t, "evidence", None) or [])
            ]
            if not notes:
                continue
            out.append(
                {
                    "cluster": getattr(t, "cluster", "personality") or "personality",
                    "slot": localise_slot(getattr(t, "slot", "")),
                    "text": getattr(t, "claim", "") or "",
                    "desc": "",
                    "notes": notes,
                }
            )
        return out


# -- DEMO memory store (mock mode) --------------------------------------------- #


class _DemoHit:
    def __init__(self, text: str, score: float, memory_id: str) -> None:
        self.text = text
        self.score = score
        self.memory_id = memory_id
        self.attributed_to = ""
        self.source = ""


class _DemoRbHit:
    def __init__(self, content: str, slot: str, source: str, priority: float) -> None:
        self.content = content
        self.slot = slot
        self.source = source
        self.priority = priority
        self.metadata = {"slot_name": slot}


class _DemoClassification:
    def __init__(self, slots: list[str], entities: list[str]) -> None:
        self.slots = slots
        self.entities = entities


class _DemoSearchResult:
    def __init__(self, hits: list, rb_hits: list, classification: Any) -> None:
        self.hits = hits
        self.rb_hits = rb_hits
        self.classification = classification
        self.related_summaries: dict = {}
        self.current_scene = ""


_DEMO_SLOT_HINTS = (
    (("work", "job", "meeting", "project", "munka", "megbeszélés", "projekt"), "work"),
    (("run", "gym", "sport", "sleep", "doctor", "health", "edzés", "alvás"), "health"),
    (("friend", "family", "mom", "dad", "partner", "barát", "család"), "relationships"),
    (("money", "rent", "salary", "pénz", "bérlés"), "finance"),
    (("plan", "goal", "future", "cél", "terv"), "goals"),
    (("coffee", "lunch", "commute", "reggel", "kávé"), "daily_life"),
    (("learn", "read", "study", "english", "tanul", "olvas"), "knowledge"),
)


class DemoMemoryLayer:
    """Offline in-memory store with the MemoryLayer API (persists to JSON).

    Powers the DEMO mode: keyword-scored recall, naive entity extraction and
    a slot guess per turn — enough for the UI's recall panel, memory graph and
    Memory Space switching to be exercised without models.
    """

    def __init__(self, config: AgentConfig, status: PipelineStatus) -> None:
        self._config = config
        self._status = status
        self._active = "demo"
        self._path = Path(config.data_dir) / "web_demo_memory.json"
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.is_file():
                data = json.loads(self._path.read_text("utf-8"))
                self._spaces = data.get("spaces", {})
            else:
                self._spaces = {}
        except Exception:  # noqa: BLE001
            self._spaces = {}
        self._spaces.setdefault("demo", {"left": [], "right": [], "next_id": 1})

    def _persist(self) -> None:
        try:
            self._path.write_text(
                json.dumps({"spaces": self._spaces}, ensure_ascii=False, indent=1),
                "utf-8",
            )
        except OSError:
            pass

    @property
    def active(self) -> str:
        return self._active

    def list_spaces(self) -> list[dict]:
        return [
            {
                "id": name,
                "name": name,
                "count": len(s.get("left", [])),
                "active": name == self._active,
                "open": True,
            }
            for name, s in sorted(self._spaces.items())
        ]

    def create_space(self, name: str) -> dict:
        safe = sanitize_space_name(name)
        if not safe:
            raise ValueError("empty name")
        if safe in self._spaces:
            raise FileExistsError(safe)
        self._spaces[safe] = {"left": [], "right": [], "next_id": 1}
        self._persist()
        return {"id": safe, "name": safe, "count": 0}

    def use_space(self, name: str) -> str:
        safe = sanitize_space_name(name)
        if not safe:
            raise ValueError("empty name")
        if safe not in self._spaces:
            self._spaces[safe] = {"left": [], "right": [], "next_id": 1}
            self._persist()
        self._active = safe
        return safe

    @staticmethod
    def _slot_of(text: str) -> str:
        t = text.lower()
        for words, slot in _DEMO_SLOT_HINTS:
            if any(w in t for w in words):
                return slot
        return "daily_life"

    @staticmethod
    def _entities_of(text: str) -> list[str]:
        words = re.findall(r"\b[A-Z][a-zA-ZÀ-ÿ]{2,}\b", text or "")
        out: list[str] = []
        for w in words:
            if w.lower() not in ("i", "the", "my", "and", "then", "hello", "hi"):
                if w not in out:
                    out.append(w)
        return out[:6]

    def search(self, query: str, emotion: str = "") -> Any:
        store = self._spaces[self._active]
        words = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query or "") if len(w) > 2]
        hits = []
        for e in store["left"]:
            text_l = e["text"].lower()
            score = sum(1.0 for w in words if w in text_l) / max(1, len(words))
            if score > 0:
                hits.append(_DemoHit(e["text"], round(min(0.99, 0.4 + score), 2), e["id"]))
        hits.sort(key=lambda h: -h.score)
        rb = [
            _DemoRbHit(r["text"], r["slot"], "profile", 0.7)
            for r in store["right"]
            if any(w in r["text"].lower() for w in words)
        ]
        return _DemoSearchResult(hits[:5], rb[:3], _DemoClassification([self._slot_of(query)], self._entities_of(query)))

    def ingest(self, text: str, agent_reply: str = "") -> None:
        if not text.strip():
            return
        store = self._spaces[self._active]
        entry = {
            "id": f"m{store['next_id']}",
            "text": text.strip(),
            "slot": self._slot_of(text),
            "entities": self._entities_of(text),
            "date": time.strftime("%Y-%m-%d"),
        }
        store["next_id"] += 1
        store["left"].append(entry)
        # A simple right-brain profile note every few turns.
        if store["next_id"] % 2 == 0:
            store["right"].append(
                {
                    "text": f"The user mentioned: {text.strip()[:60]}",
                    "slot": "preference",
                    "cluster": "preference",
                    "notes": [{"text": text.strip()[:80], "emotion": "", "cause": ""}],
                }
            )
        self._persist()

    def classify(self, query: str) -> Any:
        return _DemoClassification([self._slot_of(query)], self._entities_of(query))

    def snapshot(self, limit: int = 48) -> dict:
        store = self._spaces[self._active]
        left = [
            {
                "text": e["text"],
                "date": e.get("date", ""),
                "slot": e.get("slot", "daily_life"),
                "entities": e.get("entities", []),
                "hit": False,
            }
            for e in store["left"][:limit]
        ]
        right = [
            {
                "cluster": r.get("cluster", "personality"),
                "slot": r.get("slot", ""),
                "text": r["text"],
                "desc": "",
                "notes": r.get("notes", []),
            }
            for r in store["right"][:limit]
        ]
        return {"left": left, "right": right}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Emotion helpers (semantic fallback is pure stdlib)
# ═══════════════════════════════════════════════════════════════════════════


def semantic_emotion_label(transcript: str) -> tuple[str, float, float]:
    """Text-only emotion estimate -> (teacher label, valence, arousal).

    Uses the pure-stdlib semantic anchors from app.emotion; returns
    ("", 0, 0) for empty input. The label is fused upstream with prosody
    (emotion2vec+) whenever audio is available.
    """
    if not transcript or not transcript.strip():
        return "", 0.0, 0.0
    try:
        from app.emotion import semantic_valence_arousal, teacher_label_from_va

        v, a = semantic_valence_arousal(transcript)
        return teacher_label_from_va(v, a), v, a
    except Exception:  # noqa: BLE001
        return "", 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Components factory (LOCAL vs DEMO)
# ═══════════════════════════════════════════════════════════════════════════


class EnergyVad:
    """Demo-mode VAD: speech probability from frame RMS (pure numpy)."""

    def __init__(self, threshold: float = 0.02) -> None:
        self.threshold = float(threshold)

    def prob(self, frame: Any) -> float:
        import numpy as np

        x = np.asarray(frame, dtype=np.float32)
        if x.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(x))))
        return min(1.0, rms / max(1e-6, self.threshold) * 0.5)

    def reset(self) -> None:
        return None


class DemoLlmClient:
    """Demo-mode LLM: scripted teacher reply, word-by-word (no network)."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def health_check(self) -> bool:
        return True

    async def chat_stream(self, messages: list[dict], **_: Any):
        self.calls.append([dict(m) for m in messages])
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = str(m.get("content", ""))
                break
        lang = detect_language(user)
        memory_used = any(
            "Long-term memory" in str(m.get("content", "")) for m in messages
        )
        if lang == LANG_HU:
            memory_note = "Emlékszem a korábbi beszélgetéseinkre, ezért tudom, hogy " if memory_used else ""
            reply = (
                "Rendben, értem amit mondtál. "
                + memory_note
                + f"ezt hallottam: {user.strip()[:80]}. "
                "Ez a bemódban egy előre megírt válasz, a valódi modell a célgépen fut. "
                "Folytassuk, mondj még valamit."
            )
        else:
            reply = (
                "Got it, I heard you. "
                + ("I remember our earlier conversations, so I know that " if memory_used else "")
                + f"you said: {user.strip()[:80]}. "
                "This is a scripted demo reply; the real model runs on the target machine. "
                "Keep going, say something else."
            )
        for word in reply.split(" "):
            await asyncio.sleep(0.012)
            yield word + " "

    async def chat_json(self, messages: list[dict], **_: Any) -> dict:
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = str(m.get("content", ""))
                break
        words = [w for w in re.findall(r"[A-Za-zÀ-ÿ0-9]+", user) if len(w) > 3][:4]
        return {"title": " ".join(words) if words else "Chat"}

    async def aclose(self) -> None:
        return None


class DemoTtsEngine:
    """Demo-mode TTS: soft modulated tone, duration from text length.

    v0.4.2 voice selection: the four selectable Piper voices map to distinct
    base frequencies, so changing the selection audibly changes the output and
    the whole flow (UI selection -> backend -> synthesis) is demonstrable on
    machines without Piper (the sandbox preview runs this engine).
    """

    SAMPLE_RATE = 22050

    #: Distinct demo tone per selectable voice (Anna/Berta/Imre/Lessac).
    VOICE_FREQ = {
        "hu_HU-anna-medium": 210.0,
        "hu_HU-berta-medium": 262.0,
        "hu_HU-imre-medium": 175.0,
        "en_US-lessac-medium": 294.0,
    }

    def __init__(self) -> None:
        self.synthesized: list[tuple[str, str]] = []
        self.last_voice = ""
        self.last_language = ""

    def is_available(self) -> bool:
        return True

    def list_voices(self) -> list[str]:
        return sorted(self.VOICE_FREQ)

    def synthesize(
        self,
        text: str,
        language: str,
        length_scale: Optional[float] = None,
        voice: Optional[str] = None,
    ) -> Any:
        import numpy as np

        if not text or not text.strip():
            return None
        self.synthesized.append((text[:40], language))
        self.last_voice = str(voice or "") or ("demo-tone-hu" if language == LANG_HU else "demo-tone-en")
        self.last_language = language
        chars = min(len(text), 160)
        dur = min(3.0, 0.35 + chars * 0.028) * float(length_scale or 1.0)
        n = int(self.SAMPLE_RATE * dur)
        t = np.arange(n, dtype=np.float32) / self.SAMPLE_RATE
        freq = self.VOICE_FREQ.get(str(voice or "")) or (196.0 if language == LANG_HU else 233.0)
        tone = np.sin(2 * np.pi * freq * t) * 0.5 + np.sin(2 * np.pi * freq * 1.5 * t) * 0.2
        env = np.minimum(1.0, t * 8.0) * np.minimum(1.0, (dur - t) * 8.0)
        out = np.clip(tone * env * 0.22, -1.0, 1.0)
        return (out * 32767.0).astype(np.int16)

    def synthesize_wav(
        self, text: str, language: str, voice: Optional[str] = None
    ) -> Optional[bytes]:
        """Preview helper: the demo tone wrapped in a real WAV container."""
        pcm = self.synthesize(text, language, voice=voice)
        if pcm is None:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())
        return buf.getvalue()

    def stop(self) -> None:
        return None


class WebComponents:
    """Everything a session needs, in LOCAL (real models) or DEMO (mock) mode."""

    def __init__(self, config: AgentConfig, mock: bool = False) -> None:
        self.config = config
        self.mock = bool(mock)
        self.status = PipelineStatus()
        self.status.mode = "demo" if self.mock else "local"
        self.status.llama_url = config.llama_server_url
        self.status.space = "demo"
        self.llm: Any = None
        self.tts: Any = None
        self.memory: Any = None
        # v0.4.2: persisted Piper voice selection (UI Voice section).
        self.voice: Optional[VoiceSettings] = None
        # v0.4.16: persisted LLM GGUF selection (UI "LLM model" section):
        # the operator picks any .gguf on ANY drive; only the absolute PATH
        # is stored (config/llm_model.json) — the file is never copied.
        self.llm_model: Optional[LlmModelSettings] = None
        self._emotion_analyzer: Any = None
        self._emotion_checked = False
        # Set True once the memory layer has completed ONE search (warm-up or
        # first turn) — the first search pays the E5 model load, so the turn
        # loop grants it a much longer wait than the steady-state 1.6 s.
        self.memory_warm = False
        self._warmup_started = False

    # -- lazy construction ----------------------------------------------------- #

    def build(self) -> "WebComponents":
        """Construct the memory layer, LLM, TTS and set component statuses."""
        st = self.status
        # M3 speaker stays optional and DISABLED in the web UI regardless of
        # the YAML config (task contract: disabled by default).
        self.config.enable_speaker = False
        # v0.4.11: publish the CONFIGURED VAD threshold from server start —
        # before, the pre-session /api/pipeline response carried the
        # hardcoded seed (0.5), which contradicted the config after the
        # 0.5 -> 0.25 change until the first WS session overwrote it.
        st.session["vad_threshold"] = round(float(self.config.vad_threshold), 3)

        # v0.4.2: persisted voice selection (config/voice_settings.json) —
        # loaded BEFORE the TTS status so the chip can name the active voices.
        self.voice = VoiceSettings.load(_ROOT, self.config)
        self._sync_voice_status()

        # v0.4.16: persisted LLM model selection (config/llm_model.json) —
        # loaded BEFORE the LLM status so the chip can name the ACTIVE GGUF
        # (any drive, read in place). The selection wins over the env profile
        # in AgentConfig.llm_model_file automatically (the property reads the
        # same JSON), so check_runtime_assets() and every /api/pipeline
        # snapshot report the user-selected file without further wiring.
        self.llm_model = LlmModelSettings.load(_ROOT)
        self._sync_llm_model_status()

        if self.mock:
            self.llm = DemoLlmClient()
            self.tts = DemoTtsEngine()
            self.memory = DemoMemoryLayer(self.config, st)
            st.components["llm"].set_mocked(self._llm_chip_detail(mock=True))
            st.components["tts"].set_mocked(
                f"tone generator (demo) · voice mode: {self.voice.mode}"
            )
            st.components["memory"].set_mocked("in-memory store (demo)")
            st.components["vad"].set_mocked("energy VAD (demo)")
            st.components["asr"].set_mocked("scripted demo engine (text input drives turns)")
            st.components["embedding"].set_mocked("not used in demo")
            st.components["mic"].set_mocked("uplink idle (demo)")
            st.llama_healthy = True
            return self

        from app.llm import LlmClient
        from app.tts import TtsEngine

        self.llm = LlmClient(self.config)
        self.tts = TtsEngine(self.config)
        self.memory = RealMemoryLayer(self.config, st)

        # TTS (Piper) readiness — the detail names the SELECTED voices so the
        # runtime status answers "which voice is active?" at a glance.
        if self.tts.is_available():
            st.components["tts"].set_ready(self._tts_detail())
        else:
            st.components["tts"].set_missing("piper binary/voices not found")

        # VAD (Silero) — v0.4.14: the detail names the energy fallback so the
        # pipeline panel answers "what exactly gates my speech?" at a glance.
        try:
            from app.vad import SileroVad

            probe = SileroVad(self.config)
            if probe.is_available():
                detail = "Silero v6 (onnx, CPU)"
                if getattr(self.config, "vad_energy_fallback", True):
                    detail += " + energy fallback"
                st.components["vad"].set_ready(detail)
            else:
                st.components["vad"].set_missing("silero_vad.onnx not found")
        except Exception as exc:  # noqa: BLE001
            st.components["vad"].set_error(str(exc))

        # ASR (v0.6.0 modular engine layer): the status names the SELECTED
        # engine + registry facts — construction is status-only; the model
        # loads lazily at first use, and a load failure surfaces as an
        # explicit stage error, never an engine switch.
        try:
            from app.asr_core import ENGINE_IDS, asr_backend_status, select_engine

            backend = asr_backend_status(self.config)
            self.asr_backend = backend
            chosen = backend["selected_engine"]
            spec = backend["registry"].get(chosen) or {}
            local_present = bool(spec.get("local_present"))
            try:
                engine = select_engine(self.config)
                detail = (
                    f"{engine.model_id} ({self.config.asr_device}"
                    f"{', local' if local_present else ', download needed'})"
                )
                st.components["asr"].set_ready(detail)
            except Exception as exc:  # noqa: BLE001 - selection error is a status
                st.components["asr"].set_error(str(exc)[:200])
        except Exception as exc:  # noqa: BLE001
            st.components["asr"].set_error(str(exc)[:200])
        try:
            st.components["mic"].set_missing("waiting for the first uplink frame")
        except Exception:  # noqa: BLE001
            pass

        # Embedding (multilingual E5 small, CPU)
        try:
            import importlib.util

            if importlib.util.find_spec("sentence_transformers") is not None:
                model_dir = self.config.embedding_model_dir
                st.components["embedding"].set_ready(
                    f"multilingual-e5-small ({model_dir.name})"
                    if model_dir.is_dir()
                    else "multilingual-e5-small"
                )
            else:
                st.components["embedding"].set_missing("sentence-transformers not installed")
        except Exception as exc:  # noqa: BLE001
            st.components["embedding"].set_error(str(exc))

        # Memory (VoiceMem facade) — construction is cheap; E5 loads lazily.
        # v0.5.0: the status reports the CONTROLLED-fork identity (source
        # path + pinned upstream commit) — the running application must make
        # its VoiceMem revision visible, not just importable.
        try:
            import importlib.util

            if importlib.util.find_spec("voicemem") is not None:
                detail = "VoiceMem text_mode (per space)"
                try:
                    import voicemem as _vm
                    from pathlib import Path as _P

                    src = str(_P(_vm.__file__).resolve())
                    fork = bool(getattr(_vm, "CONTROLLED_FORK", False))
                    commit = str(getattr(_vm, "CONTROLLED_UPSTREAM_COMMIT", ""))
                    if fork and commit:
                        where = "vendor" if "vendor" in src else src
                        detail = (f"VoiceMem controlled fork @ {commit[:7]} "
                                  f"({where}, text_mode, per space)")
                    else:
                        detail = (f"VoiceMem @ {src} — NOT the controlled fork "
                                  "(pin verification failed)")
                except Exception as _exc:  # noqa: BLE001
                    logger.debug("voicemem identity probe failed: %s", _exc)
                st.components["memory"].set_ready(detail)
            else:
                st.components["memory"].set_missing("voicemem package not installed")
        except Exception as exc:  # noqa: BLE001
            st.components["memory"].set_error(str(exc))

        # LLM: mark ready/error from the first health probe (async, in app start)
        st.components["llm"].set_ready(self._llm_chip_detail())
        return self

    # -- LLM model selection (v0.4.16) -------------------------------------------- #

    def _llm_model_status(self) -> dict:
        """Live status of the ACTIVE LLM GGUF (path, source, validation).

        Never raises; the payload feeds the /api/llm-model routes and the
        LLM chip detail. Validation (exists / readable / GGUF magic /
        metadata identity) runs live on every call so a file deleted or
        replaced AFTER selection is reported honestly. v0.4.17: the
        "no model configured" state is explicit (source "none"), split
        shards are flagged as NOT directly loadable, and the llama-server
        block carries the runtime load PROOF the starter records (served
        model id + the server's own log line + a completion round-trip).
        """
        root = _ROOT
        path, source = resolve_llm_model_path(
            root, os.environ.get("LLAMA_MODEL_PATH")
        )
        info: GgufInfo = read_gguf_metadata(path)
        check = expected_model_check(info)
        verdict = llama_server_load_verdict(info)
        marker = read_llama_server_loaded_model(root)
        loaded_path = str(marker.get("model_path", "") or "")
        # Cross-check: the llama-server starter writes the exact --model path
        # into logs/llama-server.resolved-model.json on every launch. A
        # mismatch means the RUNNING server still serves the previous file
        # (restart required after a selection change).
        if loaded_path:
            try:
                same = Path(loaded_path).resolve() == Path(path).resolve()
            except OSError:
                same = loaded_path == path
            server_match = "same" if same else "different"
        else:
            server_match = "unknown"
        selected = self.llm_model.llm_model_path if self.llm_model else ""
        no_model = source == SOURCE_NONE
        file_payload = {
            "exists": info.exists,
            "readable": info.readable,
            "is_gguf": info.is_gguf,
            "error": info.error,
            "name": info.name,
            "architecture": info.architecture,
            "file_type": info.file_type,
            "quant": info.quant,
            "size_bytes": info.file_size,
            # v0.4.17: split-shard facts (an Ollama blob may be ONE shard
            # of a gguf-split set — llama-server cannot load it directly).
            "split_shard": info.is_split_shard,
            "split_no": max(info.split_no, 0) if info.is_split_shard else -1,
            "split_count": (
                max(info.split_count, 0) if info.is_split_shard else -1
            ),
        }
        if no_model:
            # Honest empty state — never display a phantom path.
            file_payload["error"] = (
                "no model file configured - select the GGUF with the Browse "
                "button (any drive, Ollama blobs included), set "
                "LLAMA_MODEL_PATH, or place a GGUF in "
                "models/llm/qwen3.6-35b-a3b/"
            )
        payload = {
            "selected_llm": self.config.llm_model_name,
            "expected_model": EXPECTED_MODEL_LABEL,
            "expected_repo": EXPECTED_MODEL_REPO,
            "model_path": path,
            "source": source,
            "source_label": {
                SOURCE_USER: "user selection (config/llm_model.json)",
                SOURCE_ENV: "env profile (LLAMA_MODEL_PATH)",
                SOURCE_DEFAULT: "project default (models/llm/...)",
                SOURCE_NONE: "no model configured",
            }.get(source, source),
            "user_selected_path": selected,
            "selected_at": self.llm_model.selected_at if self.llm_model else "",
            "file": file_payload,
            "identity": check,
            "load": verdict,
            # v0.4.17: operator dir contents when the default is ambiguous
            # or empty (guidance for the "none" state).
            "default_candidates": (
                operator_default_candidates(root) if no_model else []
            ),
            "llama_server": {
                "url": self.config.llama_server_url,
                "loaded_path": loaded_path,
                "matches_selection": server_match,
                "resolved_at": str(marker.get("resolved_at", "") or ""),
                # Runtime load PROOF recorded by the starter (v0.4.17):
                "served_model_id": str(marker.get("served_model_id", "") or ""),
                "log_path_found": bool(marker.get("log_path_found", False)),
                "log_line": str(marker.get("log_line", "") or ""),
                "completion_ok": bool(marker.get("completion_ok", False)),
                "completion_reply": str(marker.get("completion_reply", "") or ""),
                "verified_at": str(marker.get("verified_at", "") or ""),
            },
            "picker_supported": picker_supported(),
        }
        return payload

    def _sync_llm_model_status(self) -> None:
        """Refresh the LLM chip detail + session fields from the live status."""
        try:
            payload = self._llm_model_status()
        except Exception as exc:  # noqa: BLE001 - status must never raise
            logger.warning("llm model status failed: %s", exc)
            return
        st = self.status
        st.session["llm_model_path"] = payload["model_path"]
        st.session["llm_model_source"] = payload["source"]
        comp = st.components["llm"]
        # v0.4.16: the chip names the ACTIVE MODEL (profile name + the exact
        # GGUF file), never a stale hardcoded model label.
        if comp.state == _STATE_PROCESSING:
            return  # a mid-turn LLM call owns the chip right now
        comp.detail = self._llm_chip_detail()

    def _llm_chip_detail(self, mock: bool = False) -> str:
        """LLM chip text: active model name + endpoint (never a hardcoded
        model label — v0.4.16)."""
        if mock or self.mock:
            return "scripted replies (demo; real mode: local llama-server)"
        return (
            f"{self.config.llm_model_name} (llama-server "
            f"{self.config.llama_server_url})"
        )

    def apply_llm_model_selection(self, path: str) -> dict:
        """Validate + apply + persist a model selection from the UI.

        Raises ValueError on unusable input (REST maps it to HTTP 400).
        The path is stored AS GIVEN (absolute, any drive) — the GGUF file
        itself is never copied into the project. v0.4.17: a SPLIT SHARD
        (an Ollama blob that is one part of a gguf-split set) is REJECTED
        — llama-server cannot load a single shard directly, so accepting
        it would silently produce a dead LLM.
        """
        cleaned = str(path or "").strip().strip('"')
        if not cleaned:
            raise ValueError("path must not be empty")
        info = read_gguf_metadata(cleaned)
        if not info.exists:
            raise ValueError(f"file does not exist: {cleaned}")
        if not info.readable:
            raise ValueError(f"file is not readable: {cleaned}")
        if not info.is_gguf:
            raise ValueError(f"not a GGUF file: {info.error or cleaned}")
        if info.is_split_shard:
            raise ValueError(
                llama_server_load_verdict(info)["reason"]
            )
        if self.llm_model is None:  # pragma: no cover - build() always sets it
            self.llm_model = LlmModelSettings.load(_ROOT)
        self.llm_model.apply_selection(
            cleaned,
            model_name=info.name,
            quant=info.quant,
            file_size=info.file_size,
        )
        # check_runtime_assets()/llm_model_file pick the JSON up on the next
        # read; refresh the chip + session status immediately instead.
        self._sync_llm_model_status()
        return self._llm_model_status()

    def reset_llm_model_selection(self) -> dict:
        """Clear the user selection -> the env profile / default applies."""
        if self.llm_model is None:  # pragma: no cover
            self.llm_model = LlmModelSettings.load(_ROOT)
        self.llm_model.clear()
        self._sync_llm_model_status()
        return self._llm_model_status()

    # -- voice selection (v0.4.2) ------------------------------------------------ #

    def _sync_voice_status(self) -> None:
        """Mirror the current selection into the status snapshot ("voice")."""
        voice = self.voice
        if voice is None:
            return
        prev = self.status.voice
        self.status.voice = {
            "mode": voice.mode,
            "hu": voice.hu_voice,
            "en": voice.en_voice,
            "hu_label": voice_label(voice.hu_voice),
            "en_label": voice_label(voice.en_voice),
            "active": prev.get("active", ""),
            "language": prev.get("language", ""),
            "active_label": prev.get("active_label", ""),
        }

    def _tts_detail(self) -> str:
        """TTS chip detail: Piper + the SELECTED voices + mode ("Voice: …")."""
        voice = self.voice
        if voice is None:
            return "Piper"
        if self.mock:
            return f"tone generator (demo) · voice mode: {voice.mode}"
        return (
            f"Piper · {voice_label(voice.hu_voice)} (HU), "
            f"{voice_label(voice.en_voice)} (EN) · mode: {voice.mode}"
        )

    def set_active_voice(self, voice_id: str, language: str) -> None:
        """Record the voice that actually spoke (runtime status display)."""
        st = self.status
        st.voice["active"] = voice_id
        st.voice["language"] = language
        st.voice["active_label"] = voice_label(voice_id)
        comp = st.components["tts"]
        # Update the chip detail WITHOUT touching the state (ready/processing
        # is owned by the synthesis lifecycle; the detail names the voice).
        comp.detail = (
            f"Piper · Voice: {voice_label(voice_id)} · "
            f"{language_name(language)}"
            if not self.mock
            else f"demo tone · Voice: {voice_label(voice_id)} · "
            f"{language_name(language)}"
        )

    def apply_voice_settings(
        self, mode: Optional[str], hu_voice: Optional[str], en_voice: Optional[str]
    ) -> dict:
        """Validate + apply + persist a voice selection from the UI.

        Returns the /api/voice payload. Raises ValueError on invalid input
        (the REST layer maps it to HTTP 400).
        """
        if self.voice is None:  # pragma: no cover - build() always sets it
            self.voice = VoiceSettings.load(_ROOT, self.config)
        voice = self.voice
        if mode is not None:
            if mode not in VOICE_MODES:
                raise ValueError(
                    f"mode must be one of {', '.join(VOICE_MODES)} (got {mode!r})"
                )
            voice.mode = mode
        if hu_voice is not None:
            voice.hu_voice = voice._validated(hu_voice, LANG_HU)
        if en_voice is not None:
            voice.en_voice = voice._validated(en_voice, LANG_EN)
        voice.save()
        self._sync_voice_status()
        # Keep the chip detail honest without resetting a mid-synthesis state.
        comp = self.status.components["tts"]
        if comp.state != _STATE_PROCESSING:
            comp.detail = self._tts_detail()
        payload = voice.to_dict()
        payload["active"] = self.status.voice.get("active", "")
        payload["language"] = self.status.voice.get("language", "")
        return payload

    # -- background warm-up ------------------------------------------------------- #

    def start_warmup(self) -> Any:
        """Spawn the daemon warm-up thread (idempotent); returns the Thread.

        Field report v0.4.1: the first spoken turn stalled for 1-3 minutes
        with EVERYTHING green — the ASR model, the E5 embedder and the
        emotion model all load lazily INSIDE the first turn, and the turn
        waits behind each load. Warming them at server start (right after the
        browser opens, while the user is still reading the page) makes the
        first spoken turn answer in the steady-state latency.
        """
        import threading

        if self._warmup_started or self.mock:
            return None
        self._warmup_started = True
        t = threading.Thread(
            target=self._warm_up_worker, name="voicemem-warmup", daemon=True
        )
        t.start()
        return t

    def _warm_up_worker(self) -> None:
        """Load the heavy models in the background; update the status chips.

        Never raises; every failure lands in the component's error field
        (visible in the Pipeline panel) and in logs/web-server.log.
        """
        st = self.status
        cfg = self.config
        session = st.session
        session["last_event"] = "warm-up: loading models in the background"
        self._event("warm-up started")

        # 1) ASR — the big one (the SELECTED engine onto its device).
        # v0.6.0: the warm-up drives the modular engine layer — the legacy
        # app/asr.py Qwen engine is NOT loaded here anymore (it was the
        # last hidden Qwen production route). A failed selection/load is an
        # EXPLICIT component error; no alternative engine is warmed.
        try:
            st.components["asr"].begin()
            from app.asr_core import select_engine

            engine = select_engine(cfg)
            if engine.warm_up():
                st.components["asr"].end()
                detail = f"{engine.model_id} ({cfg.asr_device}) · warm"
                if cfg.asr_language:
                    detail += f" · language: {cfg.asr_language}"
                st.components["asr"].set_ready(detail)
                self._event("ASR warm")
            else:
                status = engine.status()
                reason = (
                    status.get("last_error")
                    or "see logs/web-server.log"
                )
                st.components["asr"].set_error(f"ASR warm-up failed: {reason[:200]}")
                self._event("ASR warm-up FAILED")
        except Exception as exc:  # noqa: BLE001
            st.components["asr"].set_error(f"ASR warm-up: {exc}")
            self._event(f"ASR warm-up error: {exc}")

        # 2) Emotion (emotion2vec+ M2, CPU) — optional, degrades silently.
        try:
            analyzer = self.emotion_analyzer()
            if analyzer is not None and analyzer.warm_up():
                self._event("emotion warm")
        except Exception as exc:  # noqa: BLE001
            logger.debug("emotion warm-up failed: %s", exc)

        # 3) Memory / E5 embedder — construct the active facade and run ONE
        #    throwaway search so the embedder + llama-server round trip are
        #    both hot. A hang here only blocks this daemon thread.
        try:
            st.components["memory"].begin()
            st.components["embedding"].begin()
            if self.memory is not None:
                self.memory.search("warm up", "")
                st.components["memory"].end()
                st.components["embedding"].end()
                comp = st.components["memory"]
                if comp.state == _STATE_READY:
                    comp.set_ready((comp.detail or "VoiceMem") + " · warm")
                self.memory_warm = True
                self._event("memory warm")
        except Exception as exc:  # noqa: BLE001
            st.components["memory"].end()
            st.components["embedding"].end()
            self._event(f"memory warm-up skipped: {exc}")
            logger.warning("memory warm-up failed: %s", exc)

        session["last_event"] = "warm-up finished"
        self._event("warm-up finished")

    def _event(self, message: str) -> None:
        """Append a diagnostics event to the shared session ring buffer."""
        session = self.status.session
        stamp = time.strftime("%H:%M:%S")
        session["last_event"] = message
        events: list = session.setdefault("events", [])
        events.append(f"{stamp} {message}")
        del events[:-12]

    # -- per-session factories --------------------------------------------------- #

    def make_vad(self) -> Any:
        """Per-session VAD instance (state is per-connection).

        v0.6.0: real mode returns the RAW :class:`app.vad.SileroVad` — the
        energy/gain fallback gate (FusedVad, v0.4.14) is REMOVED. That gate
        existed to compensate for the ROOT-CAUSE Silero feed bug (512-sample
        windows without the official 64-sample rolling context: real speech
        scored ~0.003, the live chain never started a turn — the v0.4.13
        field report). With the context fix measured in
        scripts/asr_forensics.py (prob_max 1.000 on the same capture that
        scored 0.003), Silero alone is the production speech decision path;
        the fallback's side effect (fragmenting continuous speech into
        ~320-384 ms utterances through a noise-floor gate) is gone with it.
        """
        if self.mock:
            return EnergyVad()
        from app.vad import SileroVad

        return SileroVad(self.config)

    def make_vad_state_machine(self) -> Any:
        from app.vad import VadStateMachine

        return VadStateMachine(
            self.config.vad_threshold,
            self.config.vad_hangover_ms,
            self.config.vad_frame_ms,
        )

    def make_asr(self) -> Any:
        """Per-session ASR engine facade over the ONE selected engine.

        v0.6.0: the engine comes from :func:`app.asr_core.select_engine`
        (ASR_ENGINE / yaml asr.engine — parakeet by default, nemotron
        available). There is NO engine fallback: if the selected engine
        cannot load, every transcription returns an explicit
        ASR_MODEL_LOAD_ERROR AsrResult (code/stage/engine/reason) that the
        UI renders as a stage failure. The heavy model is shared per
        (model_dir, device) inside the engine module; the returned object
        only carries per-session state.

        Demo mode returns the scripted MockAsrEngine adapted to the same
        contract (engine_id "mock", non-streaming, preset transcripts).
        """
        if self.mock:
            from app.mock_components import MockAsrEngine

            return MockAsrEngine()
        from app.asr_core import AsrError, select_engine

        try:
            return select_engine(self.config)
        except AsrError as exc:
            from app.asr_core import _UnavailableEngine

            return _UnavailableEngine(exc)

    def emotion_analyzer(self) -> Any:
        """Shared EmotionAnalyzer (emotion2vec+ M2), None when unavailable."""
        if self.mock or self._emotion_checked:
            return self._emotion_analyzer
        self._emotion_checked = True
        if not self.config.enable_emotion:
            self._emotion_analyzer = None
            return None
        try:
            from app.emotion import EmotionAnalyzer

            analyzer = EmotionAnalyzer(
                self.config.emotion_model_dir,
                window_s=self.config.emotion_window_s,
                sample_rate=self.config.sample_rate,
            )
            if not analyzer.is_available():
                self.status.components["memory"].detail = (
                    self.status.components["memory"].detail or ""
                )
                logger.info(
                    "M2 emotion analyzer unavailable (model: %s) - semantic-only emotion",
                    self.config.emotion_model_dir,
                )
                self._emotion_analyzer = None
                return None
            self._emotion_analyzer = analyzer
            return analyzer
        except Exception as exc:  # noqa: BLE001
            logger.warning("emotion analyzer init failed: %s", exc)
            self._emotion_analyzer = None
            return None


# ═══════════════════════════════════════════════════════════════════════════
# 6. WS session (the original web/run.py protocol, local pipeline inside)
# ═══════════════════════════════════════════════════════════════════════════


class WebSession:
    """One browser WebSocket connection = one mic stream + turn loop.

    Client -> server:
      * binary frames: 24 kHz PCM16 mono microphone audio
      * ``{"type": "user_text", "text": ...}``: typed turn (ASR bypassed)

    Server -> client (same JSON shapes the original UI speaks):
      session_ready, partial_transcript, user_transcript (source: asr|text),
      memory_hits, tag_update, answer_start, answer_delta, answer_done,
      answer_interrupt, pipeline_status, error; binary: 24 kHz PCM16.
    """

    def __init__(self, sock: Any, components: WebComponents, vad: Any = None) -> None:
        self._sock = sock
        self._c = components
        self._config = components.config
        # v0.4.4: a pre-built VAD can be handed in (the WS route builds it in
        # a worker thread - real-mode SileroVad creates an onnxruntime
        # session, which must not block the event loop).
        self._vad = vad if vad is not None else components.make_vad()
        self._vsm = components.make_vad_state_machine()
        self._asr = components.make_asr()
        self._frame_buf: Any = None  # 16 kHz float32 waiting to be VAD-framed
        self._utterance: Any = None  # 16 kHz float32 collected while speaking
        self._in_speech = False
        self._turn_task: Optional[asyncio.Task] = None
        self._turn_started_at = 0.0
        self._history: list[dict] = []
        self._last_reply = ""
        self._speaking = False  # assistant audio currently streaming to browser
        # Barge-in bookkeeping (speech during answer playback)
        self._barge_frames = 0
        self._barge_ms = 0.0
        self._pending_bytes: asyncio.Queue = asyncio.Queue()
        self._audio_task: Optional[asyncio.Task] = None
        # v0.4.7: VAD observability — rolling peak speech probability while
        # NOT in speech, reported every ~5 s of frames (see _on_vad_frame).
        self._vad_peak = 0.0
        self._vad_frames_since_report = 0
        # v0.4.14: one-shot "ASR feed" stage event per utterance (the trail
        # must prove frames reach the engine even before any partial text).
        self._asr_fed = False
        # v0.6.0: True while a genuine cache-aware streaming session is open
        # on the selected engine (nemotron); parakeet (non-streaming) keeps
        # this False and transcribes the completed utterance at speech_end.
        self._asr_streaming = False
        # v0.4.4: keep a reference to the background ingest task so it cannot
        # be garbage-collected mid-flight (fire-and-forget tasks with no
        # reference can be dropped by the GC between checkpoints).
        self._bg_store_task: Optional[asyncio.Task] = None
        # v0.4.9 (report issue #9): registry of EVERY task this session owns
        # (turn, audio loop, background store, late-emotion flush). On WS
        # disconnect all of them are cancelled — an orphaned LLM streaming
        # turn kept pulling tokens from llama-server that nobody consumed
        # (LLM thrashing; the next user's turn waited behind the backlog).
        self._session_tasks: set = set()
        self._closed = False

    # -- send helpers ------------------------------------------------------------ #

    async def _send_json(self, payload: dict) -> None:
        if self._closed:
            return
        try:
            await self._sock.send_json(payload)
        except Exception:  # noqa: BLE001 - client went away
            self._closed = True

    async def _send_bytes(self, raw: bytes) -> None:
        if self._closed:
            return
        try:
            await self._sock.send_bytes(raw)
        except Exception:  # noqa: BLE001
            self._closed = True

    async def _send_status(self) -> None:
        await self._send_json(
            {"type": "pipeline_status", "pipeline": self._c.status.snapshot()}
        )

    def _diag(self, message: str) -> None:
        """Record one chain event in the live session diagnostics.

        v0.4.13: the ring keeps the last 24 events (was 12) — one full turn
        now emits a complete stage trail (turn received → memory → emotion
        → llm → tts → answer done), so a blocked turn shows WHERE it
        stopped instead of the trail being half-evicted by the next turn.
        v0.4.14: 24 → 48 — the LIVE MIC trail is longer (mic frame received
        → speech start → ASR feed → speech end → ASR flush start → flush
        done → final transcript → ASR turn dispatch → …turn trail…), so the
        ring now holds two complete microphone turns end to end.
        """
        session = self._c.status.session
        stamp = time.strftime("%H:%M:%S")
        session["last_event"] = message
        events: list = session.setdefault("events", [])
        events.append(f"{stamp} {message}")
        del events[:-48]
        logger.info("[chain] %s", message)

    # -- v0.6.0 generic stage events (TASK-B event contract) ------------------- #

    _STAGE_TO_COMPONENT = {
        "mic": "mic",
        "vad": "vad",
        "asr": "asr",
        "voicemem": "memory",
        "e5": "embedding",
        "llm": "llm",
        "tts": "tts",
    }

    async def _stage_event(
        self,
        stage: str,
        status: str,
        message: str = "",
        error: Optional[dict] = None,
    ) -> None:
        """Emit ONE generic stage event (started | completed | failed).

        This is the UI-facing runtime contract: the CHAIN panel consumes
        exactly this shape (stage / status / ts / message / error), with NO
        engine-specific internals — engine identity appears only inside the
        ``error`` payload (AsrError.to_dict) and diagnostics, never as a
        stage name. Component chips are driven from the same call so the
        strip and the events cannot disagree.
        """
        payload = {
            "type": "stage_event",
            "stage": stage,
            "status": status,  # started | completed | failed
            "ts": round(time.time(), 3),
            "message": message,
            "error": error,
        }
        comp_key = self._STAGE_TO_COMPONENT.get(stage)
        st = self._c.status
        if comp_key is not None:
            comp = st.components.get(comp_key)
            if comp is not None:
                if status == "started":
                    comp.begin()
                elif status == "completed":
                    comp.end()
                    if comp.state == _STATE_ERROR:  # recover from earlier error
                        comp.set_ready(comp.detail)
                elif status == "failed":
                    comp.fail(error.get("detail", message) if error else message)
        await self._send_json(payload)
        if error is not None:
            self._diag(
                f"stage {stage} FAILED: {error.get('code', '')} "
                f"({error.get('reason', '')}) {message}"
            )
        elif message:
            self._diag(f"stage {stage} {status}: {message}")

    def _dump_utterance(self, audio: Any, speech_ms: int, reason: str) -> None:
        """Write an empty-transcript utterance to logs/last_empty_utterance.wav.

        v0.4.2: when ASR returns no text the user gets a toast, but the audio
        itself was lost - the one artifact that would show WHY (level too low,
        wrong language, model hiccup) is the sound. Never raises.
        """
        try:
            import numpy as np

            if audio is None or not _size(audio):
                return
            log_dir = self._config.logs_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            out = log_dir / "last_empty_utterance.wav"
            x = np.asarray(audio, dtype=np.float32)
            pcm = np.clip(x, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16)
            with wave.open(str(out), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(int(self._config.sample_rate))
                wav_file.writeframes(pcm16.tobytes())
            logger.info(
                "empty utterance dumped: %s (%d ms, reason=%s)",
                out,
                speech_ms,
                reason[:80] or "empty transcript",
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
            logger.warning("utterance dump failed: %s", exc)

    # -- main loop --------------------------------------------------------------- #

    async def run(self) -> None:
        session = self._c.status.session
        session["connected"] = True
        session["mic_frames"] = 0
        session["utterances"] = 0
        session["turns"] = 0
        session["speech_ms"] = 0
        session["vad_peak"] = 0.0
        session["vad_threshold"] = round(float(self._config.vad_threshold), 3)
        await self._send_json(
            {
                "type": "session_ready",
                "mode": self._c.status.mode,
                "space": self._c.memory.active if self._c.memory else "demo",
                "speaker": "off",
            }
        )
        self._diag(f"session_ready (mode={self._c.status.mode})")
        await self._send_status()
        self._audio_task = self._spawn(self._audio_loop())
        try:
            while not self._closed:
                msg = await self._sock.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                text = msg.get("text")
                if text:
                    await self._on_text(text)
                    continue
                raw = msg.get("bytes")
                if raw:
                    if self._audio_task is None or self._audio_task.done():
                        # v0.4.4: the audio loop has died (its error was already
                        # reported to the browser). Stop queueing mic frames into
                        # a queue nothing drains - that grew memory without bound
                        # for the rest of the session. Close the session instead.
                        break
                    frames = session.get("mic_frames", 0) + 1
                    session["mic_frames"] = frames
                    session["last_frame_ts"] = time.monotonic()
                    # v0.4.7: mic-uplink diagnostics. The v0.4.6 field report
                    # ("speech produces no transcript, only typed text works")
                    # showed ZERO log lines for the whole speech path — the
                    # uplink itself was never observable. Log the first frame
                    # and a periodic milestone so the next report separates
                    # "mic frames arrive" from "VAD never fires".
                    if frames == 1:
                        logger.info(
                            "mic uplink started: first frame %d bytes", len(raw)
                        )
                        # v0.4.14 stage event: the mic path's trail starts HERE —
                        # "mic frame received" separates "frames arrive" from
                        # "VAD never fires" at a glance in the events panel.
                        self._diag(
                            f"mic frame received (first frame, {len(raw)} bytes PCM16)"
                        )
                        # v0.6.0: the MIC chain stage is live the moment real
                        # bytes arrive from the browser uplink.
                        self._c.status.components["mic"].set_ready("uplink streaming")
                        self._spawn(
                            self._stage_event("mic", "started", "uplink streaming")
                        )
                    elif frames % 250 == 0:
                        logger.info(
                            "mic uplink alive: %d frames, VAD level %.2f%s",
                            frames,
                            float(session.get("vad_level", 0.0) or 0.0),
                            " (in speech)" if session.get("vad_in_speech") else "",
                        )
                    self._pending_bytes.put_nowait(raw)
        except Exception as exc:  # noqa: BLE001 - disconnect and friends
            logger.debug("ws session ended: %s", exc)
        finally:
            # v0.6.0: the uplink closed — the MIC stage returns to idle.
            try:
                if int(session.get("mic_frames", 0) or 0) > 0:
                    self._diag("mic uplink closed (session end)")
                    self._c.status.components["mic"].set_ready("uplink closed")
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass
            self._closed = True
            session["connected"] = False
            self._diag("session closed")
            # v0.4.9 (report issue #9): tear down EVERY session-owned task.
            # The old finally called _cancel_turn() and then
            # _audio_task.cancel() unguarded — one failing cleanup step could
            # skip the rest, and the background store task was never touched
            # at all; a mid-stream disconnect left the LLM turn orphaned.
            await self._cancel_session_tasks()

    async def _on_text(self, text: str) -> None:
        try:
            data = json.loads(text)
        except ValueError:
            return
        if not isinstance(data, dict):
            # v0.4.4: valid non-object JSON ("null", [1], 42) used to raise
            # AttributeError here and silently tear down the whole WS session.
            return
        msg_type = data.get("type")
        # v0.4.19: browser-side capture diagnostics. The browser measures its
        # OWN signal (source RMS before PCM conversion, PCM RMS after it) and
        # reports it over the same socket the audio flows through — so the
        # field log can answer "browser source RMS vs backend PCM RMS" for
        # the EXACT same frames, live, without a second channel.
        if msg_type == "mic_diag":
            self._log_mic_diag(data)
            return
        if msg_type == "mic_probe":
            await self._handle_mic_probe(data)
            return
        if msg_type != "user_text":
            return
        user_text = str(data.get("text", "")).strip()
        if not user_text:
            return
        # v0.4.13: the stage trail starts in _run_turn ("turn received") so
        # every stage is named uniformly; this used to log "text turn: ..."
        # which duplicated the turn-received line.
        await self._start_turn(user_text, audio=None, source="text")

    def _log_mic_diag(self, data: dict) -> None:
        """Log the browser's own mic measurements (never raises).

        This is the browser-vs-backend comparison the v0.4.19 mic
        investigation needs: the browser reports the RMS it measured
        BEFORE PCM conversion (source) and AFTER it (pcm); the backend
        logs the line right where its own VAD peak lines appear, so one
        grep over web-server.log shows both sides of the wire.
        """
        try:
            logger.info(
                "browser mic diag: frames %s, src rms %s (peak %s), "
                "pcm rms %s (peak %s), nz samples %s/%s, ctx %s Hz, "
                "silent %s, device %r",
                data.get("frames"),
                data.get("src_rms"),
                data.get("src_peak"),
                data.get("pcm_rms"),
                data.get("pcm_peak"),
                data.get("nz"),
                data.get("samples"),
                data.get("ctx_rate"),
                bool(data.get("silent")),
                str(data.get("device", ""))[:80],
            )
            self._diag(
                "browser mic rms %s / pcm %s (backend peak %.2f)"
                % (
                    data.get("src_rms"),
                    data.get("pcm_rms"),
                    float(self._c.status.session.get("vad_peak", 0.0) or 0.0),
                )
            )
        except Exception:  # noqa: BLE001 - diagnostics must never raise
            pass

    async def _handle_mic_probe(self, data: dict) -> None:
        """Minimal diagnostic run over the WS: browser sends its captured PCM16
        clip + its own measurements; backend measures the SAME bytes and both
        sides land in one log line (see _mic_probe_core)."""
        try:
            audio_b64 = str(data.get("audio_b64", "") or "").strip()
            result = _mic_probe_core(
                self._c,
                audio_b64,
                src_rms=data.get("src_rms"),
                pcm_rms=data.get("pcm_rms"),
                note=str(data.get("note", "") or "")[:120],
                label=str(data.get("label", "") or "")[:40],
            )
            result["type"] = "mic_probe_result"
            await self._send_json(result)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
            logger.warning("mic-probe (ws) failed: %s", exc)
            try:
                await self._send_json(
                    {"type": "mic_probe_result", "error": str(exc)[:200]}
                )
            except Exception:  # noqa: BLE001
                pass

    # -- mic audio path ------------------------------------------------------- #

    async def _audio_loop(self) -> None:
        """Drain the mic queue: resample 24k->16k, VAD frames, ASR feed/flush.

        v0.4.1: a crash here used to kill the audio path SILENTLY (the task
        exception was never retrieved; the mic kept "working" in the browser
        but the chain stopped). The whole loop is now guarded — the user sees
        an error toast and the VAD chip turns red instead of nothing.
        """
        session = self._c.status.session
        try:
            while not self._closed:
                try:
                    raw = await asyncio.wait_for(self._pending_bytes.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                f32_24 = pcm16_to_float32(raw)
                f32_16 = resample_linear(f32_24, WEB_SAMPLE_RATE, PIPE_SAMPLE_RATE)
                self._frame_buf = (
                    f32_16
                    if self._frame_buf is None
                    else _concat(self._frame_buf, f32_16)
                )
                while _size(self._frame_buf) >= VAD_FRAME_SAMPLES:
                    frame = self._frame_buf[:VAD_FRAME_SAMPLES]
                    self._frame_buf = self._frame_buf[VAD_FRAME_SAMPLES:]
                    await self._on_vad_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the loop must never die silently
            logger.exception("audio processing loop crashed")
            session["last_event"] = f"audio loop crashed: {exc}"
            self._c.status.components["vad"].set_error(f"audio loop crashed: {exc}")
            await self._send_json(
                {"type": "error", "message": f"Audio processing stopped: {exc}"}
            )

    async def _on_vad_frame(self, frame: Any) -> None:
        st = self._c.status
        session = st.session
        t_frame = self._config.vad_frame_ms / 1000.0
        try:
            prob = await asyncio.to_thread(self._vad.prob, frame)
        except Exception as exc:  # noqa: BLE001
            st.components["vad"].set_error(str(exc))
            return
        # Live VAD level: the single most useful number when debugging
        # "I speak but nothing happens" (below threshold = VAD never fires).
        session["vad_level"] = round(float(prob), 3)
        if st.components["vad"].state == _STATE_ERROR:
            st.components["vad"].set_ready(st.components["vad"].detail)

        answering = self._turn_task is not None and not self._turn_task.done()

        # v0.4.7: while frames flow but the VAD never crosses the threshold,
        # log the rolling peak every ~5 s (150 frames x 32 ms). A peak near 0
        # means the mic uplink carries silence; a peak like 0.3-0.9 without a
        # speech_start means the threshold is too high for this mic/room —
        # lower vad_threshold in config/voicemem_config.yaml. Without this
        # line the failure mode is INVISIBLE (v0.4.6 field report: zero log
        # lines between session_ready and the typed turns).
        # v0.4.8 field-report fix: the line was gated on peak >= 0.05, so the
        # MOST important case — the v0.4.7 log, 93 s at peak 0.00 — still
        # printed NOTHING. Now every ~5 s prints a line, and the near-zero
        # case gets its own message that names the actual suspects (Windows
        # mic privacy / muted device / wrong input selected in the UI).
        if not self._in_speech and not answering:
            self._vad_peak = max(self._vad_peak, float(prob))
            session["vad_peak"] = round(self._vad_peak, 3)
            session["vad_threshold"] = round(float(self._config.vad_threshold), 3)
            self._vad_frames_since_report += 1
            if self._vad_frames_since_report >= 150:
                peak = self._vad_peak
                self._vad_peak = 0.0
                self._vad_frames_since_report = 0
                if peak < 0.05:
                    logger.warning(
                        "mic uplink carries silence: VAD peak %.2f over the last "
                        "~5 s (mic frames %d) — the browser is streaming frames "
                        "but the audio is (near-)zero. Check the mic test meter "
                        "in the UI, the selected input device, Windows mic "
                        "privacy/mute settings, or run the ASR test button.",
                        peak,
                        int(session.get("mic_frames", 0) or 0),
                    )
                else:
                    logger.info(
                        "VAD never reached the speech threshold: peak %.2f < %.2f "
                        "over the last ~5 s (mic frames %d)",
                        peak,
                        float(self._config.vad_threshold),
                        int(session.get("mic_frames", 0) or 0),
                    )

        # -- barge-in detection while the assistant is answering ----------------
        if answering:
            session["vad_in_speech"] = False
            # v0.4.14: barge-in reads the PRIMARY's own probability, never the
            # fused one — the energy fallback opens the speech GATE, but it
            # must not be able to interrupt the answer on line-level noise or
            # the agent's own TTS leaking into a Raw line-in capture.
            barge_prob = float(getattr(self._vad, "last_primary", prob))
            if barge_prob >= self._config.barge_in_threshold:
                self._barge_ms += t_frame * 1000.0
            else:
                self._barge_ms = max(0.0, self._barge_ms - t_frame * 200.0)
            since_start = (time.perf_counter() - self._turn_started_at) * 1000.0
            if (
                self._barge_ms >= self._config.barge_in_min_speech_ms
                and since_start > 500.0
            ):
                logger.info(
                    "[barge-in] sustained speech %.0f ms during answer - interrupting",
                    self._barge_ms,
                )
                self._diag(f"barge-in after {self._barge_ms:.0f} ms of speech")
                self._barge_ms = 0.0
                await self._interrupt_turn()
            return

        event = self._vsm.update(prob)

        if event.value == "speech_start":
            self._in_speech = True
            self._barge_ms = 0.0
            self._utterance = None
            self._asr_fed = False
            session["vad_in_speech"] = True
            session["speech_ms"] = 0
            session["utterances"] = session.get("utterances", 0) + 1
            st.components["vad"].begin()
            await self._send_status()
            self._diag(f"speech start (level {prob:.2f})")
            # v0.6.0: VAD decision made -> chain stage event. For engines
            # with genuine streaming, the ASR stream opens HERE (cache-aware
            # state, partial decode); non-streaming engines transcribe the
            # completed utterance at speech_end.
            await self._stage_event(
                "vad", "completed", f"speech start (level {prob:.2f})"
            )
            if getattr(self._asr, "capabilities", None) is not None and bool(
                getattr(self._asr.capabilities, "streaming", False)
            ):
                await self._stage_event("asr", "started", "stream open")
                try:
                    await asyncio.to_thread(self._asr.start)
                    self._asr_streaming = True
                except Exception as exc:  # noqa: BLE001
                    self._asr_streaming = False
                    await self._stage_event(
                        "asr",
                        "failed",
                        "stream open failed",
                        error=_asr_error_payload(exc, getattr(self._asr, "engine_id", "")),
                    )
            else:
                self._asr_streaming = False
            return

        if self._in_speech:
            # Collect + (streaming engines only) partial ASR while the user
            # is speaking. v0.6.0: non-streaming engines (parakeet) get the
            # COMPLETED utterance at speech_end — no chunked-batch fake
            # partials, the engine capability decides.
            self._utterance = (
                frame if self._utterance is None else _concat(self._utterance, frame)
            )
            session["speech_ms"] = session.get("speech_ms", 0) + int(t_frame * 1000.0)
            if self._asr_streaming:
                try:
                    if not self._asr_fed:
                        self._asr_fed = True
                        self._diag("ASR stream feed (frames flowing to the engine)")
                    partial = await asyncio.to_thread(
                        self._asr.feed,
                        _frame_buffer(frame),
                    )
                    if partial is not None and partial.text:
                        self._diag(f"partial: {partial.text[:60]!r}")
                        await self._send_json(
                            {
                                "type": "partial_transcript",
                                "text": partial.text,
                                "replace": True,
                                "asr": partial.to_dict(),
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    self._asr_streaming = False
                    await self._stage_event(
                        "asr",
                        "failed",
                        "stream feed failed",
                        error=_asr_error_payload(exc, getattr(self._asr, "engine_id", "")),
                    )

        # v0.4.2 safety: a VAD stuck IN SPEECH (continuous music/noise above
        # the threshold, or a hangover that never completes) previously meant
        # "I speak but nothing ever happens" - no speech_end, no turn, no
        # feedback. An utterance longer than _MAX_UTTERANCE_MS is force-ended
        # so the collected audio still becomes a turn.
        forced_end = (
            self._in_speech
            and event.value != "speech_end"
            and session.get("speech_ms", 0) >= _MAX_UTTERANCE_MS
        )

        if event.value == "speech_end" or forced_end:
            if forced_end:
                self._vsm.reset()
                self._diag(
                    f"forced utterance end after {session.get('speech_ms', 0)} ms "
                    "(VAD stayed in speech)"
                )
            st.components["vad"].end()
            self._in_speech = False
            session["vad_in_speech"] = False
            # v0.4.4: zero the Silero recurrent state between utterances (the
            # VAD documents this - stale hidden state can misjudge the first
            # frames of the next utterance). Best effort, never blocks.
            _vad_reset = getattr(self._vad, "reset", None)
            if callable(_vad_reset):
                try:
                    _vad_reset()
                except Exception:  # noqa: BLE001
                    pass
            utterance = self._utterance
            self._utterance = None
            speech_ms = session.get("speech_ms", 0)
            session["speech_ms"] = 0
            self._diag(f"speech end ({speech_ms} ms of speech)")
            await self._stage_event("vad", "completed", f"speech end ({speech_ms} ms)")
            await self._send_status()
            # v0.6.0: the ASR stage runs through the ENGINE CONTRACT —
            # exactly one authoritative path, structured errors, no
            # partial-join fallback, no engine switching:
            #   * streaming engine: finish() the cache-aware stream
            #   * non-streaming engine: transcribe(AudioBuffer(utterance))
            # The result is an AsrResult consumed identically by the UI
            # (user_transcript carries the generic asr dict) and the turn.
            if not self._asr_streaming:
                await self._stage_event("asr", "started", "transcribing utterance")
            t_flush0 = time.perf_counter()
            asr_result = None
            final_error = ""
            try:
                if self._asr_streaming:
                    self._asr_streaming = False
                    asr_result = await asyncio.to_thread(self._asr.finish)
                else:
                    buffer = _utterance_buffer(utterance)
                    asr_result = await asyncio.to_thread(self._asr.transcribe, buffer)
            except Exception as exc:  # noqa: BLE001 — engine raised outside its contract
                asr_result = None
                final_error = str(exc)
                await self._stage_event(
                    "asr",
                    "failed",
                    "transcription failed",
                    error=_asr_error_payload(exc, getattr(self._asr, "engine_id", "")),
                )
                logger.exception("ASR transcription raised outside the engine contract")
            result_dict = asr_result.to_dict() if asr_result is not None else None
            if asr_result is not None and asr_result.error is not None:
                final_error = asr_result.error.detail or asr_result.error.reason
                await self._stage_event(
                    "asr",
                    "failed",
                    "transcription failed",
                    error=asr_result.error.to_dict(),
                )
            final = (asr_result.text if asr_result is not None else "") or ""
            final = final.strip()
            if asr_result is not None and asr_result.status.value == "ok" and not final_error:
                self._diag(
                    f"ASR flush done ({(time.perf_counter() - t_flush0) * 1000.0:.0f} ms, "
                    f"engine {asr_result.engine_id})"
                )
                await self._stage_event(
                    "asr",
                    "completed",
                    f"transcript ready ({asr_result.inference_ms:.0f} ms engine time)",
                )
            if final:
                # Sound-only turns (noise, music) carry no text to answer in
                # text_mode - they are dropped instead of sending " " to the LLM.
                self._diag(f"ASR final transcript: {final[:60]!r}")
                self._diag("ASR turn dispatch (source=asr)")
                await self._start_turn(
                    final, audio=utterance, source="asr", asr_result=result_dict
                )
            elif speech_ms >= 500:
                # v0.4.1: this was the SILENT killer - speech detected, but the
                # transcript came back empty, so the turn was dropped with no
                # feedback at all. Now it is a first-class event.
                # v0.4.2: the event carries the REASON (exception text) and
                # the utterance is dumped to logs/last_empty_utterance.wav so
                # the failure is diagnosable after the fact.
                # v0.6.0: with the engine contract an EMPTY result is either
                # an explicit failure (already emitted as stage asr/failed)
                # or the engine's honest "no speech content" verdict — both
                # are visible; the turn is NOT dispatched on empties.
                if not final_error:
                    st.components["asr"].set_error(
                        f"last utterance ({speech_ms / 1000.0:.1f} s) produced no "
                        "transcript - see logs/web-server.log"
                    )
                self._diag(f"ASR EMPTY after {speech_ms} ms of speech: {final_error[:120]}")
                self._dump_utterance(utterance, speech_ms, final_error)
                await self._send_json(
                    {
                        "type": "asr_empty",
                        "audio_ms": speech_ms,
                        "reason": final_error,
                        "asr": result_dict,
                    }
                )
                await self._send_status()

    # -- turn processing --------------------------------------------------------- #

    async def _cancel_turn(self) -> None:
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._turn_task = None

    def _spawn(self, coro: Any) -> asyncio.Task:
        """v0.4.9 (issue #9): create a session-owned, tracked task.

        Every task spawned through here lands in ``_session_tasks`` (dropped
        automatically on completion), so ``_cancel_session_tasks`` can reach
        it on disconnect — no task can outlive its WebSocket session.
        """
        task = asyncio.get_running_loop().create_task(coro)
        self._session_tasks.add(task)
        task.add_done_callback(self._session_tasks.discard)
        return task

    async def _cancel_session_tasks(self) -> None:
        """v0.4.9 (issue #9): cancel and await every task this session owns.

        Cancelling the turn task mid-LLM-stream closes the httpx response and
        stops llama-server generating tokens nobody consumes. Each step is
        individually guarded so a failure in one cancellation never skips the
        remaining ones. The background memory store is cancelled too — its
        worker thread may finish the ingest server-side, which is fine.
        """
        tasks: list[asyncio.Task] = []
        for task in (self._turn_task, self._audio_task, self._bg_store_task):
            if task is not None and not task.done():
                tasks.append(task)
        tasks.extend(t for t in list(self._session_tasks) if not t.done())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._turn_task = None
        self._audio_task = None
        self._bg_store_task = None
        self._session_tasks.clear()

    async def _interrupt_turn(self) -> None:
        """User barged in: stop TTS, cancel the turn, notify the browser."""
        task = self._turn_task
        try:
            self._c.tts.stop()
        except Exception:  # noqa: BLE001
            pass
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._turn_task = None
        await self._send_json({"type": "answer_interrupt"})

    async def _start_turn(
        self,
        text: str,
        audio: Any = None,
        source: str = "asr",
        asr_result: Optional[dict] = None,
    ) -> None:
        """New turn: supersede any in-flight one, then run the local pipeline."""
        await self._cancel_turn()
        session = self._c.status.session
        if source == "text":
            # v0.4.4: a typed turn can start while the VSM is frozen mid-speech
            # (mic frames are dropped while the previous answer is streaming).
            # Reset the speech state and drop the partial utterance, otherwise
            # the frames after the reply - including the browser's own TTS
            # tail - splice onto the stale audio and contaminate the next
            # transcript (echo leak).
            self._vsm.reset()
            self._in_speech = False
            self._utterance = None
            self._barge_ms = 0.0
            session["vad_in_speech"] = False
            session["speech_ms"] = 0
        self._turn_started_at = time.perf_counter()
        session["turns"] = session.get("turns", 0) + 1
        self._turn_task = self._spawn(
            self._guarded_turn(text, audio=audio, source=source, asr_result=asr_result)
        )

    async def _guarded_turn(
        self,
        text: str,
        audio: Any = None,
        source: str = "asr",
        asr_result: Optional[dict] = None,
    ) -> None:
        """v0.4.1: a crash ANYWHERE inside a turn used to kill its task with the
        exception never retrieved - the user saw their message and then nothing.
        The guard reports the failure to the chat (error + answer_done) and
        marks whichever component was mid-processing as failed."""
        try:
            await self._run_turn(text, audio=audio, source=source, asr_result=asr_result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("turn crashed")
            st = self._c.status
            for key in ("memory", "embedding", "llm", "tts"):
                comp = st.components[key]
                if comp.state == _STATE_PROCESSING:
                    comp.fail(f"turn crashed: {exc}")
            self._diag(f"turn CRASHED: {str(exc)[:140]}")
            try:
                await self._send_json({"type": "error", "message": f"Turn failed: {exc}"})
                await self._send_json(
                    {"type": "answer_done", "timings": {}, "error": str(exc)}
                )
                await self._send_status()
            except Exception:  # noqa: BLE001 - client may be gone
                pass

    async def _run_turn(
        self,
        text: str,
        audio: Any = None,
        source: str = "asr",
        asr_result: Optional[dict] = None,
    ) -> None:
        c = self._c
        st = c.status
        timings: dict[str, Optional[float]] = {}
        t_turn0 = time.perf_counter()
        user_text = (text or "").strip()

        # v0.4.13 STAGE EVENTS: every pipeline stage writes one "[chain]"
        # diagnostic line (visible in the UI pipeline panel "events" list,
        # refreshed every 3 s, and in logs/web-server.log). A turn that dies
        # mid-pipeline now shows exactly WHERE it stopped — e.g. the v0.4.12
        # field report "the transcript appears then nothing happens" would
        # have printed "turn received → memory start → emotion start" and
        # gone silent, naming the emotion stage as the blocker immediately.
        self._diag(
            f"turn received (source={source}, {len(user_text)} chars: {user_text[:40]!r})"
        )
        await self._send_json(
            {
                "type": "user_transcript",
                "text": user_text,
                "source": source,
                "asr": asr_result,
            }
        )

        # --- memory recall (+ emotion in parallel) ---------------------------
        st.components["memory"].begin()
        st.components["embedding"].begin()
        await self._stage_event("voicemem", "started", "memory search")
        await self._stage_event("e5", "started", "embedding query")
        self._diag("memory start")
        emotion_hint, _v, _a = semantic_emotion_label(user_text)
        mem_task = asyncio.create_task(
            asyncio.to_thread(c.memory.search, user_text, emotion_hint)
        )
        emo_task = None
        fused: Any = None
        # v0.4.13 (field report: "the system understands the text but
        # something still blocks it from reaching the LLM"): the emotion
        # analyzer construction imports funasr (which imports torch:
        # 30-180 s cold on Windows, worse with antivirus, and a concurrent
        # warm-up import serialises behind the same module lock). This await
        # used to be UNBOUNDED, so the first turn could freeze here forever —
        # after the transcript, with no error, no answer and no diag line.
        # It is now bounded by _EMOTION_INIT_TIMEOUT_S: on timeout or failure
        # the turn CONTINUES with emotion disabled (semantic-only) and still
        # reaches the LLM. The background thread keeps loading; later turns
        # pick the analyzer up through WebComponents.emotion_analyzer()'s
        # one-shot cache (subsequent calls return instantly).
        self._diag("emotion start (init)")
        analyzer: Any = None
        try:
            analyzer = await asyncio.wait_for(
                asyncio.to_thread(c.emotion_analyzer),
                timeout=_EMOTION_INIT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self._diag(
                "emotion done (init timed out after "
                f"{_EMOTION_INIT_TIMEOUT_S:.1f} s — continuing without emotion)"
            )
            logger.warning(
                "emotion analyzer init timed out after %.1f s (funasr/torch "
                "import?) — turn continues without prosody emotion",
                _EMOTION_INIT_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - emotion must never kill a turn
            self._diag(
                "emotion done (init failed — continuing without emotion: "
                f"{str(exc)[:100]})"
            )
            logger.warning("emotion analyzer init failed: %s — turn continues", exc)
        else:
            if analyzer is not None and audio is not None and _size(audio) > 0:
                from app.emotion import tail_window

                window = tail_window(audio, c.config.sample_rate, c.config.emotion_window_s)
                emo_task = asyncio.create_task(asyncio.to_thread(analyzer.analyze, window))
                self._diag("emotion done (analyzer ready · prosody task started)")
            elif analyzer is None:
                self._diag("emotion done (analyzer unavailable — semantic-only)")
            else:
                self._diag("emotion done (no audio — text turn, semantic-only)")
        # First memory search pays the E5 model load (5-20 s on CPU): wait for
        # it generously once, then use the steady-state 1.6 s budget.
        wait_s = 25.0 if not c.memory_warm else _EMOTION_WAIT_S + 1.0
        done, pending = await asyncio.wait(
            [t for t in (mem_task, emo_task) if t is not None],
            timeout=wait_s,
        )
        # v0.4.4: only mark the memory path warm when the first search actually
        # finished. A timed-out E5 load used to set memory_warm=True anyway, so
        # every later turn was dropped to the 1.6 s budget while the model was
        # still loading - memory failed on all turns for no visible reason.
        if (
            mem_task in done
            and not mem_task.cancelled()
            and mem_task.exception() is None
        ):
            c.memory_warm = True
        elif mem_task not in done:

            def _drain_late(task: asyncio.Task) -> None:
                # v0.4.4: retrieve the outcome of the abandoned search so a
                # late failure does not surface as "Task exception was never
                # retrieved" at shutdown.
                try:
                    if not task.cancelled() and task.exception() is not None:
                        logger.warning(
                            "late memory search failed: %s", task.exception()
                        )
                except Exception:  # noqa: BLE001
                    pass

            mem_task.add_done_callback(_drain_late)
        result = None
        try:
            if mem_task in done:
                result = mem_task.result()
                st.components["memory"].end()
                n_hits = len(getattr(result, "hits", None) or []) + len(
                    getattr(result, "rb_hits", None) or []
                )
                self._diag(f"memory done ({n_hits} hits)")
                await self._stage_event(
                    "voicemem", "completed", f"memory search done ({n_hits} hits)"
                )
                await self._stage_event("e5", "completed", "embedding query done")
            else:
                st.components["memory"].fail(
                    f"search timeout after {wait_s:.0f} s (E5 first load?)"
                )
                self._diag(
                    f"memory done (timeout after {wait_s:.0f} s — E5 first load?)"
                )
                await self._stage_event(
                    "voicemem",
                    "failed",
                    f"search timeout after {wait_s:.0f} s (E5 first load?)",
                    error={
                        "code": "VOICEMEM_ERROR",
                        "stage": "voicemem",
                        "reason": "search_timeout",
                        "detail": f"memory search timed out after {wait_s:.0f} s",
                    },
                )
        except Exception as exc:  # noqa: BLE001
            st.components["memory"].fail(str(exc))
            self._diag(f"memory done (failed: {str(exc)[:120]})")
            await self._stage_event(
                "voicemem",
                "failed",
                "memory search failed",
                error={
                    "code": "VOICEMEM_ERROR",
                    "stage": "voicemem",
                    "reason": "search_failed",
                    "detail": str(exc)[:500],
                },
            )
        st.components["embedding"].end()

        # Prosody emotion: waited result, or late tag_update.
        if emo_task is not None:
            if emo_task in done:
                try:
                    prosody = emo_task.result()
                    if prosody is not None:
                        from app.emotion import fuse_emotion

                        fused = fuse_emotion(prosody, user_text, c.config.emotion_fusion_prosody_weight)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("prosody analysis failed: %s", exc)
            else:

                async def _late_emotion(task: asyncio.Task) -> None:
                    try:
                        prosody = await task
                        if prosody is None:
                            return
                        from app.emotion import fuse_emotion

                        fused_late = fuse_emotion(
                            prosody, user_text, c.config.emotion_fusion_prosody_weight
                        )
                        if fused_late is not None and fused_late.label:
                            await self._send_json(
                                {
                                    "type": "tag_update",
                                    "emotion": fused_late.label,
                                    "emotion_from": "prosody+semantic (fused)",
                                }
                            )
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass

                emo_task.add_done_callback(
                    lambda t: self._spawn(_late_emotion(t))
                )

        emotion_label = ""
        if fused is not None and fused.label:
            emotion_label = fused.label
        elif emotion_hint:
            emotion_label = emotion_hint

        memory_context = self._memory_context(result)
        await self._send_json(
            {
                "type": "memory_hits",
                "emotion": emotion_label,
                "emotion_from": "fused" if fused is not None and fused.label else "semantic",
                **self._hits_payload(result),
            }
        )

        # --- LLM streaming reply ----------------------------------------------
        from app.teacher_persona import build_messages, build_system_prompt

        emo_args: dict = {}
        if fused is not None and fused.label:
            emo_args = {
                "emotion_label": fused.label,
                "emotion_valence": fused.valence,
                "emotion_arousal": fused.arousal,
            }
        elif emotion_hint:
            _label, _hv, _ha = semantic_emotion_label(user_text)
            emo_args = {
                "emotion_label": _label,
                "emotion_valence": _hv,
                "emotion_arousal": _ha,
            }
        memory_context = _cap_memory_context(memory_context)
        system_prompt = build_system_prompt(memory_context, **emo_args)
        history = _fit_history_budget(
            system_prompt, self._history[-_HISTORY_WINDOW:], user_text
        )
        messages = build_messages(user_text, system_prompt, history=history)
        await self._send_json({"type": "answer_start"})
        await self._stage_event(
            "llm", "started", f"{self._config.llm_model_name} streaming"
        )
        await self._send_status()

        stream = SentenceStream(
            first_chunk_chars=self._config.tts_first_chunk_chars,
            chunk_chars=self._config.tts_chunk_chars,
        )
        speak_queue: asyncio.Queue = asyncio.Queue()

        async def speak_worker() -> None:
            nonlocal t_first_audio, tts_chunks
            while True:
                chunk = await speak_queue.get()
                if chunk is None:
                    return
                if not tts_chunks:
                    # v0.4.13 stage event: the FIRST TTS synthesis marks the
                    # tts stage start (per-chunk lines would spam the trail).
                    self._diag(f"tts start (chunk 1, {len(chunk)} chars)")
                    self._spawn(self._stage_event("tts", "started", "speaking the reply"))
                tts_chunks += 1
                pcm = await self._synthesize_chunk(chunk, fused)
                if pcm is not None:
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                        self._diag(
                            "first TTS audio sent to the browser "
                            f"({(t_first_audio - t_turn0) * 1000.0:.0f} ms)"
                        )
                    await self._send_bytes(pcm)

        speaker = asyncio.create_task(speak_worker())
        reply = ""
        t_first_token: Optional[float] = None
        t_first_audio: Optional[float] = None
        tts_chunks = 0
        llm_error = ""
        st.components["llm"].begin()
        # v0.4.13 stage event: names the model + endpoint so an "llm start"
        # line with no follow-up points straight at the llama-server.
        self._diag(
            f"llm start ({self._config.llm_model_name} @ {self._config.llama_server_url})"
        )
        t_llm0 = time.perf_counter()
        try:
            async for delta in c.llm.chat_stream(messages):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                    timings["llm_first_token_ms"] = round(
                        (t_first_token - t_turn0) * 1000.0, 1
                    )
                    self._diag(
                        f"llm first token ({(t_first_token - t_turn0) * 1000.0:.0f} ms)"
                    )
                reply += delta
                await self._send_json({"type": "answer_delta", "text": delta})
                for chunk in stream.add_delta(delta):
                    speak_queue.put_nowait(chunk)
            for chunk in stream.flush():
                speak_queue.put_nowait(chunk)
            st.components["llm"].end()
            self._diag(
                f"llm done ({len(reply)} chars, {(time.perf_counter() - t_llm0) * 1000.0:.0f} ms)"
            )
            await self._stage_event(
                "llm", "completed", f"{len(reply)} chars generated"
            )
        except LlmUnavailableError as exc:
            st.components["llm"].fail(str(exc))
            llm_error = str(exc)
            self._diag(f"llm failed ({str(exc)[:140]})")
            await self._stage_event(
                "llm",
                "failed",
                "LLM unavailable",
                error={
                    "code": "LLM_ERROR",
                    "stage": "llm",
                    "reason": "unavailable",
                    "detail": str(exc)[:500],
                },
            )
        except asyncio.CancelledError:
            speak_queue.put_nowait(None)
            speaker.cancel()
            raise
        finally:
            if llm_error:
                await self._send_json(
                    {"type": "error", "message": f"LLM unavailable: {llm_error}"}
                )

        speak_queue.put_nowait(None)
        try:
            await speaker
        except asyncio.CancelledError:
            # v0.4.4: this turn was cancelled (barge-in or a superseding turn)
            # while the TTS was still draining. Swallowing the cancellation let
            # the "cancelled" turn run to completion - answer_done arrived
            # AFTER answer_interrupt, history was appended and ingest fired.
            # Clean up and re-raise so the cancellation actually takes effect.
            self._speaking = False
            raise
        self._speaking = False
        # v0.4.13 stage event: the TTS stage ends when the speak worker has
        # drained every chunk (all audio has left for the browser).
        if tts_chunks:
            self._diag(
                f"tts done ({tts_chunks} chunks, "
                f"{(time.perf_counter() - t_turn0) * 1000.0:.0f} ms)"
            )
            await self._stage_event("tts", "completed", f"{tts_chunks} chunks played")
        else:
            self._diag("tts done (no chunks synthesised — silent reply?)")
            await self._stage_event(
                "tts",
                "failed",
                "no audio synthesised",
                error={
                    "code": "TTS_ERROR",
                    "stage": "tts",
                    "reason": "no_audio",
                    "detail": "no TTS chunks were synthesised for the reply",
                },
            )

        if llm_error:
            await self._send_json({"type": "answer_done", "timings": timings, "error": llm_error})
            return

        timings["total_ms"] = round((time.perf_counter() - t_turn0) * 1000.0, 1)
        st.turn = dict(timings)
        self._diag(
            f"answer done: {len(reply)} chars, total {timings['total_ms']:.0f} ms"
        )
        await self._send_json({"type": "answer_done", "timings": timings})
        await self._send_status()

        # --- remember the turn (background; the UI polls /api/memories) -------
        if user_text and reply:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": reply})

            async def _store() -> None:
                st.components["memory"].begin()
                try:
                    await asyncio.to_thread(c.memory.ingest, user_text, reply)
                    st.components["memory"].end()
                except Exception as exc:  # noqa: BLE001
                    st.components["memory"].fail(str(exc))
                await self._send_status()

            # v0.4.4: get_event_loop() inside a running loop is deprecated
            # and a task with no reference can be garbage-collected
            # mid-flight; create it on the running loop and keep a reference.
            # v0.4.9 (issue #9): spawn through the session task registry so
            # disconnect cancels it too.
            self._bg_store_task = self._spawn(_store())

    async def _synthesize_chunk(self, chunk: str, fused: Any) -> Optional[bytes]:
        """TTS one chunk via Piper, convert to 24 kHz PCM16 for the browser.

        v0.4.2 voice flow (task contract): LLM response -> language detection
        -> voice selection (mode-aware) -> Piper -> audio. The resolved voice
        is recorded in the runtime status ("Voice: Imre · Hungarian") and is
        what the Piper --model argument actually uses.
        """
        c = self._c
        st = c.status
        language = detect_language(chunk) if chunk.strip() else LANG_HU
        length_scale = None
        if fused is not None and getattr(fused, "label", "") in ("frustrated", "sad"):
            length_scale = c.config.emotion_slow_length_scale
        voice_id = ""
        if c.voice is not None:
            voice_id, language = c.voice.resolve(language)
            st.voice["active"] = voice_id
            st.voice["language"] = language
            st.voice["active_label"] = voice_label(voice_id)
            self._diag(
                f"TTS voice: {voice_label(voice_id)} "
                f"({language_name(language)}, mode {c.voice.mode})"
            )
        st.components["tts"].begin()
        try:
            pcm16k = await asyncio.to_thread(
                c.tts.synthesize, chunk, language, length_scale, voice_id or None
            )
        except Exception as exc:  # noqa: BLE001
            st.components["tts"].fail(str(exc))
            return None
        st.components["tts"].end()
        if voice_id:
            c.set_active_voice(voice_id, language)
        if pcm16k is None or not len(pcm16k):
            return None
        import numpy as np

        x = np.asarray(pcm16k).astype(np.float32) / 32768.0
        out = resample_linear(x, c.config.output_sample_rate, WEB_SAMPLE_RATE)
        return float32_to_pcm16_bytes(out)

    # -- payload helpers ---------------------------------------------------------- #

    @staticmethod
    def _memory_context(result: Any) -> str:
        """SearchResult -> the text block for the system prompt (capped)."""
        if result is None:
            return ""
        parts: list[str] = []
        hits = getattr(result, "hits", None) or []
        for h in hits[:5]:
            t = (getattr(h, "text", "") or "").strip()
            if t:
                parts.append(f"- {t}")
        rb = getattr(result, "rb_hits", None) or []
        for h in rb[:3]:
            t = clean_rb_content(getattr(h, "content", "") or "").strip()
            if t:
                parts.append(f"- {t}")
        ctx = "\n".join(parts)
        return ctx[:1200]

    @staticmethod
    def _hits_payload(result: Any) -> dict:
        """SearchResult -> the original UI's memory_hits shape."""
        hits = getattr(result, "hits", None) or []
        rb = getattr(result, "rb_hits", None) or []
        cls = getattr(result, "classification", None)
        return {
            "slots": list(getattr(cls, "slots", []) or []),
            "entities": list(getattr(cls, "entities", []) or []),
            "left_brain": [
                {
                    "text": getattr(h, "text", ""),
                    "score": getattr(h, "score", 0.0),
                    "attributed_to": getattr(h, "attributed_to", "") or "",
                    "memory_id": getattr(h, "memory_id", "") or "",
                    "has_audio": False,
                }
                for h in hits
            ],
            "right_brain_hits": [
                {
                    "content": clean_rb_content(getattr(h, "content", "") or ""),
                    "raw": getattr(h, "content", "") or "",
                    "internal": getattr(h, "source", "") == "response_experience",
                    "slot": localise_slot(
                        (getattr(h, "metadata", None) or {}).get("slot_name", "")
                        or getattr(h, "slot", "")
                    ),
                    "source": getattr(h, "source", "") or "",
                    "priority": getattr(h, "priority", 0.0),
                    "cluster": "",
                }
                for h in rb
                if getattr(h, "source", "") != "response_experience"
            ],
            "current_scene": None,
            "related_summaries": getattr(result, "related_summaries", {}) or {},
        }


def _concat(a: Any, b: Any) -> Any:
    import numpy as np

    return np.concatenate((np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))


def _size(x: Any) -> int:
    return int(getattr(x, "size", 0) or 0)


def _frame_buffer(frame: Any) -> Any:
    """Wrap one 16 kHz VAD frame into the canonical :class:`AudioBuffer`."""
    from app.asr_core import AudioBuffer

    return AudioBuffer.from_float(frame, PIPE_SAMPLE_RATE)


def _utterance_buffer(utterance: Any) -> Any:
    """Wrap a collected utterance into the canonical :class:`AudioBuffer`."""
    from app.asr_core import AudioBuffer

    if utterance is None or not _size(utterance):
        return AudioBuffer(samples=np_zeros(0), sample_rate=PIPE_SAMPLE_RATE)
    return AudioBuffer.from_float(utterance, PIPE_SAMPLE_RATE)


def np_zeros(n: int) -> Any:
    """numpy zeros helper (lazy import, float32 1-D)."""
    import numpy as np

    return np.zeros(int(n), dtype=np.float32)


def _asr_error_payload(exc: Exception, engine_id: str) -> dict:
    """Structured error dict for an exception raised outside the contract."""
    from app.asr_core import AsrError, AsrErrorCode

    if isinstance(exc, AsrError):
        return exc.to_dict()
    return AsrError(
        code=AsrErrorCode.ASR_INFERENCE_ERROR,
        stage="asr",
        engine=engine_id,
        reason="inference_failed",
        detail=str(exc)[:500],
    ).to_dict()


# ═══════════════════════════════════════════════════════════════════════════
# 6b. Mic → ASR one-shot test (v0.4.8)
# ═══════════════════════════════════════════════════════════════════════════

#: One ASR test at a time — the endpoint holds the GPU for a full inference.
_ASR_TEST_LOCK = threading.Lock()

#: Below this RMS the uplink audio is (near-)digital silence: the browser
#: sent zeros — a device/mute/privacy problem, NOT a model problem. The
#: v0.4.7 field log ran for 93 s at exactly this level.
_ASR_TEST_SILENT_RMS = 0.005
#: Shorter clips carry no speech worth transcribing.
_ASR_TEST_MIN_MS = 300

#: DEMO-mode transcript: honest about what the sandbox can and cannot do.
_ASR_TEST_DEMO_TEXT = (
    "DEMO — no ASR model is loaded in this browser preview. On your machine "
    "this box would show exactly what Qwen3-ASR heard in your recording."
)

# v0.4.19 — mic capture probe ("the minimal diagnostic path").
#: The probe's job is to answer ONE question with numbers: does real,
#: non-zero microphone audio reach the backend? Three RMS values are
#: compared for the SAME captured clip:
#:   1. browser source RMS  — the Float32 signal BEFORE any PCM conversion
#:      (measured in the ScriptProcessor tap, at the AudioContext rate)
#:   2. browser PCM RMS     — AFTER downsample + Int16 conversion (the exact
#:      bytes that go on the wire)
#:   3. backend PCM RMS     — measured here, on the bytes that arrived
#: If 1-2 are non-zero and 3 is zero, the bug is browser conversion or
#: WebSocket framing. If 1 is already zero, the bug is BEFORE the app:
#: getUserMedia constraints, the selected input device, the track state,
#: or the OS/permission layer. NO VAD, NO models, NO thresholds involved.
_MIC_PROBE_MIN_BYTES = 2  # one Int16 sample is enough for a zero/non-zero answer
#: The browser reports its values as numbers already; keep them as-is.


def _mic_probe_core(
    c: Any,
    audio_b64: str,
    src_rms: Any = None,
    pcm_rms: Any = None,
    note: str = "",
    label: str = "",
) -> dict:
    """Measure the backend side of a capture probe; log the RMS triple.

    Called from BOTH transports: the WS JSON ``{"type": "mic_probe"}`` frame
    and the REST ``POST /api/mic-probe`` (used when no WS session is live).
    Pure arithmetic on the received bytes — works identically in LOCAL and
    DEMO mode (no models are touched), and never raises on bad input.

    v0.4.20: ``label`` tags the probe phase ("A·constrained" /
    "B·minimal") so the A/B comparison lands in one log line per phase —
    the field answer shows both capture variants side by side.
    """
    import base64 as _b64

    result: dict = {
        "backend_rms": 0.0,
        "backend_peak": 0.0,
        "backend_samples": 0,
        "backend_nz": 0,
        "browser_src_rms": src_rms,
        "browser_pcm_rms": pcm_rms,
        "label": label[:40],
        "sample_rate": WEB_SAMPLE_RATE,
        "duration_ms": 0,
        "verdict": "no_audio",
        "recorded_at": time.strftime("%H:%M:%S"),
        "note": note,
    }
    try:
        raw = _b64.b64decode(audio_b64, validate=False) if audio_b64 else b""
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        raw = b""
    even = len(raw) - (len(raw) % 2)
    n = even // 2
    result["backend_samples"] = n
    result["duration_ms"] = round(1000.0 * n / float(WEB_SAMPLE_RATE))
    if n == 0 or even < _MIC_PROBE_MIN_BYTES:
        _log_mic_probe(c, result)
        return result

    # PCM16 little-endian (the exact wire format the browser sends):
    # decode straight from the bytes — endianness and interpretation are
    # part of what this probe verifies.
    import array

    samples = array.array("h")
    samples.frombytes(raw[:even])
    nz = 0
    peak = 0
    sq_sum = 0.0
    for s in samples:
        if s:
            nz += 1
            a = abs(s)
            if a > peak:
                peak = a
        sq_sum += float(s) * float(s)
    rms = (sq_sum / n) ** 0.5 / 32768.0 if n else 0.0
    result["backend_rms"] = round(rms, 5)
    result["backend_peak"] = round(peak / 32768.0, 5)
    result["backend_nz"] = nz

    # The verdict names the broken stage — the same decision tree the
    # v0.4.19 investigation prescribes, decided by numbers, not guesses:
    try:
        src_v = float(src_rms) if src_rms is not None else None
    except (TypeError, ValueError):
        src_v = None
    try:
        pcm_v = float(pcm_rms) if pcm_rms is not None else None
    except (TypeError, ValueError):
        pcm_v = None
    if src_v is not None and src_v > 0.001 and (
        pcm_v is None or (pcm_v is not None and pcm_v <= 0.001)
    ):
        # source non-zero, browser PCM (near-)zero: conversion bug in the
        # browser (scaling/serialization of the Int16 bytes).
        result["verdict"] = "browser_conversion_silent"
    elif rms < 0.001:
        if src_v is not None and src_v > 0.001:
            # browser measured non-zero source AND sent non-zero pcm, but
            # the backend decodes zeros: framing/transport bug.
            result["verdict"] = "transport_silent"
        else:
            # the browser itself captured silence: getUserMedia/track/
            # device/OS level — BEFORE any app conversion.
            result["verdict"] = "browser_source_silent"
    else:
        result["verdict"] = "ok"
    _log_mic_probe(c, result)
    return result


def _log_mic_probe(c: Any, result: dict) -> None:
    """One log line + one pipeline-panel event row per probe (never raises)."""
    try:
        label = str(result.get("label", "") or "")[:40]
        tag = f" [{label}]" if label else ""
        logger.info(
            "mic-probe%s: browser src RMS %s / browser PCM RMS %s / "
            "backend PCM RMS %s (peak %s, nz %s/%s samples, %s ms, "
            "verdict %s)%s",
            tag,
            result.get("browser_src_rms"),
            result.get("browser_pcm_rms"),
            result.get("backend_rms"),
            result.get("backend_peak"),
            result.get("backend_nz"),
            result.get("backend_samples"),
            result.get("duration_ms"),
            result.get("verdict"),
            (" — " + result["note"]) if result.get("note") else "",
        )
        c._event(
            "mic probe: browser {src} / backend {backend} → {verdict}".format(
                src=result.get("browser_pcm_rms"),
                backend=result.get("backend_rms"),
                verdict=result.get("verdict"),
            )
        )
    except Exception:  # noqa: BLE001
        pass


def _write_asr_test_wav(config: AgentConfig, f32_16: Any) -> Optional[Path]:
    """Save the last test clip to logs/asr_test_last.wav (best effort)."""
    try:
        import numpy as np

        if not _size(f32_16):
            return None
        log_dir = config.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / "asr_test_last.wav"
        x = np.asarray(f32_16, dtype=np.float32)
        pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(out), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(PIPE_SAMPLE_RATE)
            wav_file.writeframes(pcm16.tobytes())
        return out
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        logger.warning("asr-test wav dump failed: %s", exc)
        return None


def _run_asr_test(c: Any, raw: bytes) -> dict:
    """Synchronous core of POST /api/asr-test (runs in a worker thread).

    Stage-by-stage verdict — this is the machine-readable answer to
    "does the chain even get there, and what does ASR understand?":

    * ``too_short``     — clip under _ASR_TEST_MIN_MS, nothing to transcribe
    * ``silent_mic``    — RMS < _ASR_TEST_SILENT_RMS: the browser sent
                           (near-)zeros. Windows mic privacy/mute, wrong
                           device, or a muted track. NOT a model problem.
    * ``vad_would_fire``— False when the VAD peak stayed under the config
                           threshold: the LIVE chain (which gates ASR behind
                           VAD) would never start a turn at this level.
    * ``asr_error``     — the inference raised (error carries the reason)
    * ``asr_empty``     — inference ran, no text came back
    * ``ok``            — transcript non-empty

    The transcript itself is returned AND logged — "write the text out
    somewhere" — together with the audio stats, so the field log answers
    the question without a debugger.
    """
    import numpy as np

    st = c.status
    cfg = c.config
    threshold = float(cfg.vad_threshold)
    f32_24 = pcm16_to_float32(raw)
    f32_16 = resample_linear(f32_24, WEB_SAMPLE_RATE, PIPE_SAMPLE_RATE)
    n_samples = _size(f32_16)
    duration_ms = 1000.0 * n_samples / float(PIPE_SAMPLE_RATE)

    result: dict = {
        "mode": st.mode,
        "verdict": "too_short",
        "transcript": "",
        "demo": bool(c.mock),
        "duration_ms": round(duration_ms),
        "asr_wall_ms": 0,
        "audio": {"rms": 0.0, "peak": 0.0},
        "vad": {
            "peak": 0.0,
            "avg": 0.0,
            "threshold": round(threshold, 3),
            "frames_above": 0,
            "frames": 0,
        },
        "vad_would_fire": False,
        "wav_url": "/api/asr-test/audio",
        "error": "",
        "recorded_at": time.strftime("%H:%M:%S"),
    }

    # Always keep the recording: listening back to it is the single fastest
    # way to tell "the browser sent silence" from "ASR misheard real speech".
    _write_asr_test_wav(cfg, f32_16)

    if duration_ms < _ASR_TEST_MIN_MS:
        logger.info("ASR test: %.0f ms clip — too short, nothing transcribed", duration_ms)
        return result

    # --- stage 1: what did the mic actually send? ----------------------------- #
    x = np.asarray(f32_16, dtype=np.float32)
    rms = float(np.sqrt(float(np.mean(np.square(x))))) if n_samples else 0.0
    peak = float(np.max(np.abs(x))) if n_samples else 0.0
    result["audio"] = {"rms": round(rms, 4), "peak": round(peak, 4)}

    # --- stage 2: would the live VAD gate ever fire on this audio? ------------- #
    # A FRESH VAD instance: the live session's Silero state is per-connection
    # and must not be polluted by this clip (and vice versa).
    # v0.4.14: make_vad() now returns the FUSED VAD in real mode, so this
    # check answers what the LIVE gate (Silero + energy fallback) would do —
    # exactly the behaviour a live utterance experiences. The Silero-only
    # peak is reported separately (silero_peak) so a "deaf Silero, open
    # energy gate" verdict is visible at a glance.
    probs: list[float] = []
    try:
        vad = c.make_vad()
    except Exception as exc:  # noqa: BLE001
        vad = None
        logger.warning("asr-test: VAD unavailable (%s)", exc)
    if vad is not None:
        step = int(VAD_FRAME_SAMPLES)
        for i in range(0, n_samples - step + 1, step):
            try:
                probs.append(float(vad.prob(x[i : i + step])))
            except Exception:  # noqa: BLE001
                break
    if probs:
        above = sum(1 for p in probs if p >= threshold)
        fused_vad = {
            "peak": round(max(probs), 3),
            "avg": round(sum(probs) / len(probs), 3),
            "threshold": round(threshold, 3),
            "frames_above": above,
            "frames": len(probs),
        }
        silero_peak = getattr(vad, "primary_peak", None)
        if silero_peak is not None:
            fused_vad["silero_peak"] = round(float(silero_peak), 3)
            fused_vad["gate"] = (
                "energy fallback" if getattr(vad, "fallback_active", False) else "silero"
            )
        result["vad"] = fused_vad
        result["vad_would_fire"] = above > 0

    if rms < _ASR_TEST_SILENT_RMS:
        result["verdict"] = "silent_mic"
        _log_and_record(c, result, "mic carries digital silence")
        return result

    # --- stage 3: what does ASR understand? (ALSO runs when VAD would NOT
    #     fire — that is the whole point of the test: the answer names the
    #     broken stage even when the live chain never reaches ASR) ------------ #
    t0 = time.perf_counter()
    if c.mock:
        result["asr_wall_ms"] = round((time.perf_counter() - t0) * 1000.0)
        result["transcript"] = _ASR_TEST_DEMO_TEXT
        result["verdict"] = "ok"
        _log_and_record(c, result, "demo transcript")
        return result
    # v0.6.0: the test drives the PRODUCTION engine contract — select the
    # configured engine, build the canonical AudioBuffer, transcribe; the
    # structured AsrError becomes the machine-readable verdict.
    try:
        from app.asr_core import AudioBuffer, select_engine

        engine = select_engine(cfg)
        buffer = AudioBuffer.from_float(x, PIPE_SAMPLE_RATE)
        asr_result = engine.transcribe(buffer)
        result["asr_wall_ms"] = round((time.perf_counter() - t0) * 1000.0)
        result["engine"] = asr_result.engine_id
        result["model_id"] = asr_result.model_id
        result["language"] = asr_result.language
        result["asr"] = asr_result.to_dict()
        if asr_result.error is not None:
            result["verdict"] = "asr_error"
            err = asr_result.error.to_dict()
            result["error"] = (
                f"{err.get('code')} stage={err.get('stage')} "
                f"reason={err.get('reason')}: {err.get('detail', '')[:200]}"
            )
            st.components["asr"].set_error(
                f"ASR test failed: {err.get('reason')} — {err.get('detail', '')[:160]}"
            )
        else:
            result["transcript"] = (asr_result.text or "").strip()
            result["verdict"] = "ok" if result["transcript"] else "asr_empty"
            if result["verdict"] == "asr_empty":
                st.components["asr"].set_error(
                    "ASR test: transcript came back empty — listen to the recording "
                    "(logs/asr_test_last.wav) and check the language/model"
                )
    except Exception as exc:  # noqa: BLE001
        result["asr_wall_ms"] = round((time.perf_counter() - t0) * 1000.0)
        result["verdict"] = "asr_error"
        result["error"] = str(exc)[:300]
        st.components["asr"].set_error(f"ASR test failed: {str(exc)[:200]}")
    _log_and_record(c, result, "asr done")
    return result


def _log_and_record(c: Any, result: dict, note: str) -> None:
    """One log line + one pipeline-panel event row per ASR test (never raises)."""
    try:
        st = c.status
        vad = result.get("vad", {})
        audio = result.get("audio", {})
        gate_note = ""
        if "silero_peak" in vad:
            gate_note = f", silero peak {vad.get('silero_peak')} (gate: {vad.get('gate')})"
        summary = (
            f"ASR test: {result.get('duration_ms', 0)} ms, rms {audio.get('rms')}, "
            f"VAD peak {vad.get('peak')}/{vad.get('threshold')} "
            f"(fire: {result.get('vad_would_fire')}){gate_note}, verdict "
            f"{result.get('verdict')}, "
            f"transcript {result.get('transcript', '')[:80]!r} "
            f"({result.get('asr_wall_ms', 0)} ms) — {note}"
        )
        logger.info(summary)
        st.session["asr_test_last"] = (
            f"{result.get('recorded_at', '')} {result.get('verdict', '')} · "
            f"{result.get('transcript', '')[:60]!r}"
        )
        c._event(f"asr test: {result.get('verdict', '')} ({note})")
    except Exception:  # noqa: BLE001
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 7. FastAPI app
# ═══════════════════════════════════════════════════════════════════════════


def build_web_app(components: WebComponents) -> Any:
    """Wire the FastAPI app: UI file, REST, WebSocket. No OpenAI anywhere."""
    import contextlib

    @contextlib.asynccontextmanager
    async def _lifespan(app: Any):
        task = asyncio.create_task(_llama_health_loop(components))
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(
        title="VoiceMem local web backend",
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
    )
    c = components
    st = c.status

    # -- pages ------------------------------------------------------------------ #

    @app.get("/")
    def index():
        page = _WEB_DIR / "voicemem.html"
        if not page.is_file():
            return JSONResponse({"error": "voicemem.html missing"}, status_code=500)
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    images_dir = _WEB_DIR / "images"
    images_dir.mkdir(exist_ok=True)
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

    # -- health / pipeline debug ------------------------------------------------ #

    @app.get("/api/health")
    async def api_health() -> dict:
        return {
            "status": "ok",
            "mode": st.mode,
            "space": c.memory.active if c.memory else "demo",
            "llama": {
                "healthy": await c.llm.health_check(),
                "url": c.config.llama_server_url,
            },
            "version": _version(),
        }

    @app.get("/api/pipeline")
    async def api_pipeline() -> dict:
        st.llama_healthy = await c.llm.health_check()
        comp = st.components["llm"]
        if st.llama_healthy and comp.state in (_STATE_ERROR, _STATE_INIT):
            if st.llm_probe_failed:
                # v0.4.4: the startup probe itself reported a failure (empty
                # reply / timeout / exception). "/health == 200" does NOT prove
                # the model answers - keep the error visible instead of
                # painting the chip green (the v0.4.3 silent failure).
                pass
            else:
                comp.set_ready(f"llama-server {c.config.llama_server_url}")
        elif not st.llama_healthy and st.mode == "local":
            comp.set_error(f"llama-server unreachable at {c.config.llama_server_url}")
        st.space = c.memory.active if c.memory else ""
        # Live mic age: proves whether the browser is actually streaming.
        ts = st.session.get("last_frame_ts")
        if ts is not None:
            st.session["last_frame_age_s"] = round(max(0.0, time.monotonic() - ts), 1)
        return st.snapshot()

    # -- mic → ASR one-shot test (v0.4.8) ---------------------------------------- #
    #
    # The v0.4.7 field log: 2000 mic frames, VAD level 0.00 for 93 seconds,
    # zero VAD-peak lines, no ASR call, no transcript — "I speak but nothing
    # happens" with no way to tell WHERE the chain died. This endpoint
    # bypasses VAD/WS/turn state completely: the browser records a clip,
    # POSTs it here, and the answer says exactly what the mic + ASR heard
    # (the transcript is written into the UI AND the log), plus the
    # level/VAD/ASR stats that name the broken stage.

    @app.post("/api/asr-test")
    async def api_asr_test(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = None
        body = body if isinstance(body, dict) else {}
        audio_b64 = str(body.get("audio_b64", "") or "").strip()
        if not audio_b64:
            raise HTTPException(400, 'body must be {"audio_b64": "<PCM16 mono 24 kHz>"}')
        try:
            raw = base64.b64decode(audio_b64, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"audio_b64 is not valid base64: {exc}")
        if len(raw) < 2 * int(WEB_SAMPLE_RATE * 0.05):  # < 50 ms of audio
            raise HTTPException(400, "audio_b64 is shorter than 50 ms of PCM16")
        if not _ASR_TEST_LOCK.acquire(blocking=False):
            raise HTTPException(409, "another ASR test is still running")
        try:
            return await asyncio.to_thread(_run_asr_test, c, raw)
        finally:
            _ASR_TEST_LOCK.release()

    @app.get("/api/asr-test/audio")
    def api_asr_test_audio() -> Response:
        """Serve the last test recording (WAV) so the user can LISTEN to what
        the mic actually sent — 'it is silent' vs 'it is speech' in one click."""
        path = c.config.logs_dir / "asr_test_last.wav"
        if not path.is_file():
            raise HTTPException(404, "no ASR test recording yet")
        return FileResponse(
            str(path), media_type="audio/wav", headers={"Cache-Control": "no-store"}
        )

    # -- mic capture probe / browser diagnostics (v0.4.19) ---------------------- #

    @app.post("/api/mic-probe")
    async def api_mic_probe(req: Request) -> dict:
        """Minimal diagnostic path (HTTP twin of the WS ``mic_probe`` frame).

        Body: ``{"audio_b64": "<PCM16 mono 24 kHz>", "src_rms": <number>,
        "pcm_rms": <number>, "note": "<str>"}`` — the browser records a few
        seconds, measures its OWN signal before and after PCM conversion and
        posts the exact wire bytes; the backend measures the same bytes and
        both sides land in one ``mic-probe:`` log line. Pure arithmetic —
        no VAD, no models, no thresholds — so it works in DEMO mode too.
        """
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = None
        body = body if isinstance(body, dict) else {}
        audio_b64 = str(body.get("audio_b64", "") or "").strip()
        if not audio_b64:
            raise HTTPException(
                400, 'body must be {"audio_b64": "<PCM16 mono 24 kHz>"}'
            )
        try:
            base64.b64decode(audio_b64, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"audio_b64 is not valid base64: {exc}")
        return _mic_probe_core(
            c,
            audio_b64,
            src_rms=body.get("src_rms"),
            pcm_rms=body.get("pcm_rms"),
            note=str(body.get("note", "") or "")[:120],
            label=str(body.get("label", "") or "")[:40],
        )

    @app.post("/api/mic-diag")
    async def api_mic_diag(req: Request) -> dict:
        """Log a browser-side capture summary (device/track/constraint state).

        The browser POSTs its getUserMedia result — device label, track
        readyState/enabled/muted, track settings, AudioContext rate, the
        constraint actually used — so the field log carries the FULL browser
        capture state next to the backend's own measurements. Never raises;
        always ``{"ok": true}``.
        """
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = None
        body = body if isinstance(body, dict) else {}
        try:
            settings = body.get("track_settings") or {}
            note = str(body.get("note", "") or "")[:120]
            suffix = f" — {note}" if note else ""
            logger.info(
                "browser mic capture: device %r, ready %s, enabled %s, "
                "muted %s, track {rate %s Hz, ch %s, size %s, aec %s, "
                "agc %s, ns %s}, ctx %s Hz, constraint %s, source %r%s",
                str(body.get("device", ""))[:80],
                body.get("ready_state"),
                body.get("enabled"),
                body.get("muted"),
                settings.get("sampleRate"),
                settings.get("channelCount"),
                settings.get("sampleSize"),
                settings.get("echoCancellation"),
                settings.get("autoGainControl"),
                settings.get("noiseSuppression"),
                body.get("ctx_rate"),
                str(body.get("constraint", ""))[:200],
                str(body.get("source", ""))[:60],
                suffix,
            )
        except Exception:  # noqa: BLE001 - diagnostics must never raise
            pass
        return {"ok": True}

    # -- voice selection (v0.4.2) ------------------------------------------------- #

    @app.get("/api/voice")
    def api_voice() -> dict:
        """Current voice selection + catalog (populates the UI Voice section)."""
        if c.voice is None:  # pragma: no cover - build() always loads it
            c.voice = VoiceSettings.load(_ROOT, c.config)
            c._sync_voice_status()
        payload = c.voice.to_dict()
        payload["active"] = st.voice.get("active", "")
        payload["active_label"] = st.voice.get("active_label", "")
        payload["language"] = st.voice.get("language", "")
        return payload

    @app.post("/api/voice")
    async def api_voice_set(req: Request) -> dict:
        """Apply + PERSIST a voice selection (config/voice_settings.json)."""
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = None
        # v0.4.4: valid non-object JSON ([1], null, 42) used to raise
        # AttributeError -> HTTP 500.
        body = body if isinstance(body, dict) else {}
        try:
            payload = c.apply_voice_settings(
                mode=body.get("mode"),
                hu_voice=body.get("hu_voice"),
                en_voice=body.get("en_voice"),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except OSError as exc:
            raise HTTPException(500, f"could not persist voice settings: {exc}")
        self_event = f"voice settings: mode={payload['mode']} "
        f"hu={payload['hu_voice']} en={payload['en_voice']}"
        c._event(self_event)
        return payload

    @app.post("/api/voice/preview")
    async def api_voice_preview(req: Request) -> Response:
        """Synthesize a short local sentence with the SELECTED voice (WAV).

        Local Piper only - no network call in any mode. Demo mode answers
        with a voice-specific tone so the flow is testable without models.
        """
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = None
        # v0.4.4: valid non-object JSON ([1], null, 42) used to raise
        # AttributeError -> HTTP 500.
        body = body if isinstance(body, dict) else {}
        voice = str(body.get("voice", "") or "").strip()
        if not voice:
            raise HTTPException(400, "body must be {\"voice\": <voice id>}")
        if c.voice is not None and voice not in c.voice.available:
            raise HTTPException(400, f"unknown voice {voice!r}")
        language = language_of_voice(voice)
        text = PREVIEW_SENTENCES[language]
        if c.mock:
            wav_bytes = c.tts.synthesize_wav(text, language, voice=voice)
            if not wav_bytes:
                raise HTTPException(503, "demo preview synthesis failed")
        else:
            import tempfile

            def _synth() -> Optional[bytes]:
                with tempfile.TemporaryDirectory(prefix="voicemem_voice_") as tmp:
                    out = Path(tmp) / "preview.wav"
                    if not c.tts.synthesize_to_file(
                        text, language, out, voice=voice
                    ):
                        return None
                    return out.read_bytes()

            wav_bytes = await asyncio.to_thread(_synth)
            if not wav_bytes:
                raise HTTPException(
                    503,
                    "Piper could not synthesize the preview (is the voice "
                    "installed? see the TTS row in the Pipeline panel)",
                )
        c._event(f"voice preview: {voice_label(voice)} ({language})")
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    # -- memory spaces ------------------------------------------------------------ #

    # -- LLM model selection (v0.4.16) ---------------------------------------------- #
    #
    # The "LLM model" section of the web UI: the operator browses the LOCAL
    # Windows filesystem with a NATIVE file dialog (this backend opens it —
    # a browser <input type=file> would UPLOAD content, which we never want
    # for a 20 GB GGUF), the picked path is validated as a real GGUF (magic
    # + metadata + expected-model identity) and only the ABSOLUTE PATH is
    # persisted to config/llm_model.json. The file itself stays wherever
    # the operator keeps it (any drive); llama-server reads it in place.
    # The running llama-server keeps serving its own loaded model until the
    # operator restarts it (START.bat / start_llama_server.ps1) — the
    # /api/llm-model payload says exactly whether the loaded path matches.

    @app.get("/api/llm-model")
    def api_llm_model() -> dict:
        """Current model status: path, source, validation, identity,
        llama-server cross-check."""
        try:
            return c._llm_model_status()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"llm model status failed: {exc}")

    @app.post("/api/llm-model/browse")
    async def api_llm_model_browse() -> dict:
        """Open the NATIVE Windows file picker (GGUF filter, other files
        still selectable) and return the selected absolute path.

        The native dialog blocks, so it runs in a worker thread; the call
        only RETURNS a path — nothing is read, saved or copied here.
        """
        if not picker_supported():
            return {
                "supported": False,
                "path": None,
                "status": "unsupported",
                "reason": (
                    "native file dialogs are only available on Windows; "
                    "type the path manually below"
                ),
            }
        path, status = await asyncio.to_thread(
            pick_file,
            "Select the LLM model file (.gguf)",
        )
        return {"supported": True, "path": path, "status": status}

    @app.post("/api/llm-model/inspect")
    async def api_llm_model_inspect(req: Request) -> dict:
        """Validate a candidate path (exists / readable / GGUF magic /
        metadata identity) WITHOUT saving it."""
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = None
        path = str((body if isinstance(body, dict) else {}).get("path", "") or "").strip()
        if not path:
            raise HTTPException(400, 'body must be {"path": "<absolute file path>"}')
        info = read_gguf_metadata(path)
        check = expected_model_check(info)
        verdict = llama_server_load_verdict(info)
        return {
            "path": path,
            "exists": info.exists,
            "readable": info.readable,
            "is_gguf": info.is_gguf,
            "error": info.error,
            "name": info.name,
            "architecture": info.architecture,
            "file_type": info.file_type,
            "quant": info.quant,
            "size_bytes": info.file_size,
            # v0.4.17: split-shard detection (an Ollama blob may be ONE
            # shard of a gguf-split set — NOT directly loadable).
            "split_shard": info.is_split_shard,
            "split_no": max(info.split_no, 0) if info.is_split_shard else -1,
            "split_count": (
                max(info.split_count, 0) if info.is_split_shard else -1
            ),
            "identity": check,
            "load": verdict,
        }

    @app.post("/api/llm-model/select")
    async def api_llm_model_select(req: Request) -> dict:
        """Validate + PERSIST the selected model path (config/llm_model.json).

        The selection survives restart AND extract-over upgrades. The GGUF
        is never copied anywhere. A non-Qwen file is ACCEPTED (llama-server
        serves any GGUF; the UI shows the detected identity + a warning)
        — but a non-GGUF or missing file is rejected with HTTP 400.
        """
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = None
        path = str((body if isinstance(body, dict) else {}).get("path", "") or "").strip()
        try:
            payload = await asyncio.to_thread(c.apply_llm_model_selection, path)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except OSError as exc:
            raise HTTPException(500, f"could not persist the model selection: {exc}")
        check = payload.get("identity", {})
        load = payload.get("load", {})
        if load and not load.get("loadable", True):
            # v0.4.17: defensive — apply_llm_model_selection already rejects
            # shards; if a non-loadable file ever slips through, say it.
            c._event(
                "LLM model selected: NOT directly loadable — "
                f"{load.get('reason', '')}"
            )
        elif not check.get("is_expected"):
            c._event(
                "LLM model selected: "
                f"{check.get('detected', '')} (NOT the expected "
                f"{payload.get('expected_model', '')}) — llama-server restart "
                "needed to load it"
            )
        else:
            c._event(
                f"LLM model selected: {check.get('detected', '')} — "
                "llama-server restart needed to load it"
            )
        logger.info("LLM model selection: %s -> %r", payload.get("source"), path)
        return payload

    @app.post("/api/llm-model/reset")
    async def api_llm_model_reset() -> dict:
        """Clear the user selection: the env profile / project default
        applies again (LLAMA_MODEL_PATH in config/env.local.ps1)."""
        try:
            payload = await asyncio.to_thread(c.reset_llm_model_selection)
        except OSError as exc:
            raise HTTPException(500, f"could not persist the reset: {exc}")
        c._event("LLM model selection reset to the profile default")
        return payload

    @app.get("/api/spaces")
    def api_spaces() -> dict:
        return {"spaces": c.memory.list_spaces(), "active": c.memory.active}

    @app.post("/api/spaces")
    async def api_space_new(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = None
        # v0.4.4: valid non-object JSON ([1], null, 42) used to raise
        # AttributeError -> HTTP 500.
        name = (body if isinstance(body, dict) else {}).get("name", "")
        try:
            return c.memory.create_space(name)
        except FileExistsError as exc:
            raise HTTPException(409, f"Space '{exc}' already exists")
        except ValueError:
            raise HTTPException(400, "Invalid space name")

    @app.post("/api/spaces/{name}/use")
    def api_space_use(name: str) -> dict:
        try:
            active = c.memory.use_space(name)
            st.space = active
            return {"active": active}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not switch: {exc}")

    # -- memories / graph ----------------------------------------------------------- #

    @app.get("/api/memories")
    def api_memories() -> dict:
        try:
            return c.memory.snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory snapshot failed: %s", exc)
            return {"left": [], "right": []}

    @app.post("/api/classify")
    def api_classify(body: dict) -> dict:
        try:
            cls = c.memory.classify(str(body.get("query", "")))
            return {
                "slots": list(getattr(cls, "slots", []) or []),
                "entities": list(getattr(cls, "entities", []) or []),
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"classify failed: {exc}")

    # -- session title / language (compat with the original UI) --------------------- #

    @app.post("/api/title")
    async def api_title(body: dict) -> dict:
        text = str(body.get("text", ""))[:600]
        if not text.strip():
            return {"title": ""}
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Summarise this conversation in at most 12 words, as a title. "
                        "Output only the title itself, no quotes, no punctuation, "
                        "no 'About' prefix. Ignore greetings and mic tests."
                    ),
                },
                {"role": "user", "content": text},
            ]
            data = await c.llm.chat_json(messages)
            title = str(data.get("title", "")).strip()
            return {"title": title[:80]}
        except Exception:  # noqa: BLE001 - title is cosmetic
            return {"title": ""}

    @app.post("/api/lang")
    async def api_lang(req: Request) -> dict:
        # v0.4.4: malformed JSON used to raise -> HTTP 500; treat it as "en".
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            body = None
        data = body if isinstance(body, dict) else {}
        lang = data.get("lang", "en")
        app.state.lang = "hu" if str(lang).startswith("hu") else "en"
        # v0.4.12: the page reports its own build version — log it, so field
        # logs always name the page the browser actually ran (an open tab
        # across an upgrade silently keeps the OLD JS: the v0.4.11 report
        # reproduced the v0.4.9 capture numbers exactly for this reason).
        page = data.get("page")
        if page:
            logger.info(
                "web UI page version %s (backend %s)%s",
                page,
                _version(),
                " — STALE PAGE, reload needed" if str(page) != str(_version()) else "",
            )
        return {"lang": app.state.lang}

    @app.get("/api/audio/{memory_id}")
    def api_audio(memory_id: str):
        raise HTTPException(404, "No archived audio in local text mode")

    # -- WebSocket -------------------------------------------------------------------- #

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            # v0.4.4: real-mode SileroVad construction creates an onnxruntime
            # InferenceSession (blocking, 0.5-2 s) - build it in a worker
            # thread instead of on the event loop, and route ANY construction
            # failure through the error path rather than a raw ASGI traceback.
            vad = await asyncio.to_thread(c.make_vad)
            session = WebSession(sock, c, vad=vad)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ws session init failed: %s", exc)
            try:
                await sock.send_json({"type": "error", "message": str(exc)[:200]})
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            await session.run()
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - never crash the server
            logger.exception("ws session error: %s", exc)
            try:
                await sock.send_json({"type": "error", "message": str(exc)[:200]})
            except Exception:  # noqa: BLE001
                pass

    return app


async def _llama_health_loop(c: WebComponents) -> None:
    """Poll llama-server /health every 5 s; update the LLM component status.

    v0.4.4: on the FIRST successful probe in real mode an actual one-turn
    chat is sent (tiny, thinking-disabled) and the reply text is checked —
    "/health 200" alone does NOT prove the model answers. This is exactly
    the v0.4.3 field failure: llama-server was healthy, but every chat
    completion returned EMPTY content (the model's thinking channel ate
    the whole max_tokens budget) and the UI went silent on every turn.
    The verification lands in the LLM chip detail + the session event log
    so the Pipeline panel shows "reply verified" or the precise failure.
    """
    st = c.status
    verified = False
    attempts = 0
    while True:
        try:
            healthy = await c.llm.health_check()
        except Exception:  # noqa: BLE001
            healthy = False
        st.llama_healthy = healthy
        comp = st.components["llm"]
        if c.mock:
            if comp.state != _STATE_PROCESSING:
                # v0.4.4: same guard as the real branch - do not clobber a
                # mid-turn processing state every 5 s (the real branch got this
                # guard in v0.4.2; the demo branch was forgotten, so the LLM
                # chip flickered back to mocked mid-answer in DEMO mode).
                comp.set_mocked(c._llm_chip_detail(mock=True))
        elif healthy:
            if not verified and attempts < 5:
                attempts += 1
                verified = await _verify_llm_reply(c)
            # v0.4.16: names the ACTIVE model (profile name), never the
            # stale v0.4.3-era hardcoded label.
            detail = c._llm_chip_detail()
            if verified:
                if comp.state != _STATE_PROCESSING:
                    comp.set_ready(detail + " · reply verified")
            elif comp.state not in (_STATE_ERROR, _STATE_PROCESSING):
                # Probe not finished yet: keep the optimistic ready state,
                # but NEVER overwrite an error the probe itself reported.
                comp.set_ready(detail)
        else:
            verified = False
            attempts = 0
            # v0.4.4: server came back - clear the probe failure so the next
            # successful probe can turn the chip green again.
            st.llm_probe_failed = False
            comp.set_error(f"llama-server unreachable at {c.config.llama_server_url}")
        await asyncio.sleep(5.0)


async def _verify_llm_reply(c: WebComponents) -> bool:
    """Send ONE small chat turn; True when non-empty content comes back.

    Never raises; every failure is reported through the LLM component
    status (visible in the Pipeline panel) and the diagnostics event ring.
    """
    st = c.status
    comp = st.components["llm"]
    try:
        text = ""
        async def _collect() -> None:
            nonlocal text
            async for delta in c.llm.chat_stream(
                [
                    {
                        "role": "user",
                        "content": (
                            "Reply with the single word: ready"
                        ),
                    }
                ],
                max_tokens=24,
            ):
                text += delta

        await asyncio.wait_for(_collect(), timeout=60.0)
        if text.strip():
            st.session["llm_verified"] = True
            st.llm_probe_failed = False
            c._event("LLM reply verified (startup probe)")
            logger.info("LLM startup probe OK (%d chars)", len(text))
            return True
        # chat_stream raises on empty replies, so this is defensive:
        st.llm_probe_failed = True
        comp.set_error(
            "llama-server answered the startup probe with an empty reply — "
            "see logs/web-server.log"
        )
        c._event("LLM startup probe: EMPTY reply")
        return False
    except asyncio.TimeoutError:
        st.llm_probe_failed = True
        comp.set_error("LLM startup probe timed out (60 s) — model still loading?")
        c._event("LLM startup probe timeout")
        return False
    except Exception as exc:  # noqa: BLE001 - LlmUnavailableError and friends
        st.llm_probe_failed = True
        comp.set_error(f"LLM startup probe failed: {exc}")
        c._event(f"LLM startup probe failed: {str(exc)[:100]}")
        logger.warning("LLM startup probe failed: %s", exc)
        return False


def _version() -> str:
    try:
        return (_ROOT / "VERSION").read_text("utf-8").strip()
    except OSError:
        return "dev"


# ═══════════════════════════════════════════════════════════════════════════
# 8. CLI entry
# ═══════════════════════════════════════════════════════════════════════════


def _parse_args(argv: Optional[list[str]] = None):
    import argparse

    p = argparse.ArgumentParser(description="VoiceMem local web backend")
    p.add_argument("--config", default=os.environ.get("VOICEMEM_CONFIG", ""))
    p.add_argument(
        "--mock",
        action="store_true",
        default=os.environ.get("VOICEMEM_WEB_MOCK", "0") == "1",
        help="DEMO mode: offline stand-ins for every model (no GPU needed)",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("VOICEMEM_WEB_HOST", DEFAULT_WEB_HOST),
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VOICEMEM_WEB_PORT", DEFAULT_WEB_PORT)),
    )
    p.add_argument("--check", action="store_true", help="print the component status and exit")
    return p.parse_args(argv)


def load_config(path: str = "") -> AgentConfig:
    """Build AgentConfig: YAML (if any) + env overrides + web defaults."""
    cfg = AgentConfig(root=_ROOT)
    if path and Path(path).is_file():
        cfg = AgentConfig.from_yaml(Path(path))
        cfg.apply_env()
    else:
        cfg.apply_env()
    # M3 speaker stays optional and DISABLED in the web UI (the web layer
    # routes memory by Memory Space, not by speaker; the agent CLI keeps the
    # full M3 pipeline). WebComponents.build() enforces this again.
    cfg.enable_speaker = False
    return cfg


def _setup_logging(level: str, config: AgentConfig) -> Optional[Path]:
    """Console + rotating file logging (logs/web-server.log).

    v0.4.1 field report: the backend previously logged to stderr only, so a
    remote field failure left NO artifact to attach. The file captures the
    [chain] event stream, component warm-up failures and ASR tracebacks.
    Returns the log path (None when the file could not be opened).
    """
    from logging.handlers import RotatingFileHandler

    log_path: Optional[Path] = None
    handlers: list = [logging.StreamHandler()]
    try:
        log_dir = config.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "web-server.log"
        file_handler = RotatingFileHandler(
            str(log_path), maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handlers.append(file_handler)
    except OSError as exc:  # read-only disk and friends - console still works
        print(f"[web] WARNING: could not open the log file: {exc}", flush=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,  # v0.4.2: a pre-configured root (pytest, uvicorn) must
        # not silently swallow our file handler - force wins and keeps the
        # rotating file authoritative in every embedding.
    )
    return log_path


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else None)
    level = os.environ.get("VOICEMEM_WEB_LOG", "INFO")

    cfg = load_config(args.config)
    log_path = _setup_logging(level, cfg)
    components = WebComponents(cfg, mock=args.mock).build()

    if args.check:
        snap = components.status.snapshot()
        snap["llama"] = {
            "healthy": asyncio.run(components.llm.health_check()),
            "url": cfg.llama_server_url,
        }
        print(json.dumps(snap, indent=2, ensure_ascii=False))
        return 0

    import uvicorn

    # Warm the heavy models in the background BEFORE uvicorn takes over -
    # the first spoken turn then answers in steady-state latency.
    components.start_warmup()

    mode_label = "DEMO (mock components)" if args.mock else "LOCAL (real models)"
    url = f"http://{args.host}:{args.port}/"
    print(f"[web] VoiceMem local web backend - {mode_label}", flush=True)
    print(f"[web] LLM endpoint: {cfg.llama_server_url} (llama-server, local)", flush=True)
    print(f"[web] UI: {url}", flush=True)
    if log_path is not None:
        print(f"[web] Diagnostics log: {log_path}", flush=True)
    if not args.mock:  # v0.4.4: warm-up is a no-op in DEMO mode - do not claim it
        print("[web] Warming up ASR/memory/emotion models in the background.", flush=True)
    print("[web] Press Ctrl+C to stop.", flush=True)

    app = build_web_app(components)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
