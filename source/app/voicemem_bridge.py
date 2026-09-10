"""Bridge to the external VoiceMem long-term memory package.

The ``voicemem`` package (the CONTROLLED fork vendored at vendor/voicemem,
pinned to upstream e8384e0 - see VOICEMEM_PIN.json) is an OPTIONAL runtime
dependency: when it
is not importable, the bridge degrades to "no long-term memory" mode, logs a
single warning and returns empty memory contexts. All direct voicemem calls
run in worker threads via ``asyncio.to_thread`` so the event loop never blocks
on model I/O. The ``reply=`` hook keeps our teacher persona and llama-server
streaming under VoiceMem.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .config import AgentConfig
from .teacher_persona import (
    build_messages,
    build_system_prompt,
    fit_prompt_budget,
)

if TYPE_CHECKING:  # app.llm imports httpx; keep it out of import time here
    from .llm import LlmClient

try:  # optional dependency, NOT installed in the development sandbox
    import voicemem  # type: ignore

    _HAS_VOICEMEM = True
except ImportError:  # pragma: no cover - depends on the environment
    voicemem = None  # type: ignore[assignment]
    _HAS_VOICEMEM = False

logger = logging.getLogger(__name__)

#: Hard cap for the memory context string injected into the system prompt.
_MEMORY_CONTEXT_MAX_CHARS = 1200

#: v0.4.9 (report issue #3): cap for the memory context handed to the reply
#: hook by VoiceMem (uncapped before — a huge retrieval block could overflow
#: the fixed 8192-token llama-server context and lose the whole reply).
_REPLY_MEMORY_CONTEXT_MAX_CHARS = 6000

#: v0.4.9 (report issue #7): per-user facade cache bounds. 10 live facades
#: max (LRU eviction beyond that) and 1 h idle eviction — the old
#: ``{user_id: vm}`` dict kept every speaker's facade (SQLite handles +
#: Qdrant collection refs) FOREVER, a slow leak in multi-user deployments.
_VM_CACHE_MAX = 10
_VM_IDLE_S = 3600.0

#: v0.4.9 (report issue #5): VoiceMem construction failure is NOT cached
#: forever any more. Up to 3 consecutive failures are allowed; after that
#: retries pause for 30 s (a network blip on the first turn used to disable
#: long-term memory for the WHOLE session, silently).
_VM_MAX_FAILS = 3
_VM_RETRY_WINDOW_S = 30.0

#: Commit statuses returned by :meth:`VoiceMemBridge.commit_reply` (v0.4.9,
#: report issue #1 — the pipeline must be able to SEE that a reply was lost
#: from long-term memory instead of the old silent-None contract).
COMMIT_COMMITTED = "committed"
COMMIT_UNAVAILABLE = "unavailable"
COMMIT_FAILED = "failed"

_DEGRADED_WARNING = (
    "voicemem package not available - running in degraded mode without "
    "long-term memory (memory_context will be empty)"
)

#: v0.5.1: right-brain note cleaning (mirrors app/web_server.py — the proven
#: web rendering). The prefix strips "[date] ⚠label: "-style headers, the
#: suffix strips the advisory "(next time: ...)" tail VoiceMem appends.
_RB_PREFIX_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?(?:[⚠✓✱*]\s*)?(?:[^：:\s]{2,8}[：:]\s*)?")
_RB_SUFFIX_RE = re.compile(r"[（(]\s*(?:下次|next time|内心OS|inner note)\s*[：:].*$", re.S | re.I)


def clean_rb_content(content: str) -> str:
    """Strip the date/slot prefix and advisory suffix from right-brain notes."""
    t = _RB_PREFIX_RE.sub("", str(content or ""))
    t = _RB_SUFFIX_RE.sub("", t)
    return t.strip()


#: Adapter table for the VoiceMem streaming feed. The installed version may
#: expose any of these entry points (checked in order, hasattr-guarded).
#: v0.5.1 (TASK 1 forensic fix): the PINNED vendor (e8384e0) exposes none of
#: the legacy streaming-feed names — its public surface is
#: ingest/search/classify/reply/stream/flush (vendor core.py). The ingest
#: entry below is what store_fact() and any storage feed actually use on the
#: pinned package; ``process_turn`` retrieval now goes through
#: ``_retrieve_user_turn`` (Search-first) instead of this table.
_FEED_ADAPTERS: tuple[tuple[str, Callable[[Any, str], Any]], ...] = (
    ("feed_partial", lambda vm, text: vm.feed_partial(text, ended=True)),
    ("feed", lambda vm, text: vm.feed(text)),
    ("process_text", lambda vm, text: vm.process_text(text)),
    ("handle_text", lambda vm, text: vm.handle_text(text)),
    ("ingest", lambda vm, text: vm.ingest(text, async_facts=True)),
)

#: Adapter table for feeding the assistant reply back for consolidation.
_COMMIT_ADAPTERS: tuple[tuple[str, Callable[[Any, str], Any]], ...] = (
    (
        "add_message",
        lambda vm, text: vm.add_message({"role": "assistant", "content": text}),
    ),
    ("remember", lambda vm, text: vm.remember(text)),
    ("feed_partial", lambda vm, text: vm.feed_partial(text, ended=True)),
)

#: Attributes that may hold the memory context on a Turn/SearchResult object.
_CONTEXT_ATTRIBUTES: tuple[str, ...] = (
    "context",
    "memory_context",
    "search_result",
    "prompt_context",
)

#: Attributes that may hold the list of memories on a SearchResult-like object.
_MEMORY_LIST_ATTRIBUTES: tuple[str, ...] = ("memories", "results", "items", "top_memories")


def pin_e5_local_model(config: AgentConfig) -> Optional[str]:
    """Point VoiceMem's E5 embedder at the LOCAL offline model copy.

    v0.4.4 field report: the runtime environment is fully offline
    (``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1``, by design), but
    voicemem's ``hf_model("embedding", ...)`` resolution only accepts a
    local directory when the model files sit DIRECTLY in
    ``<models>/embedding/`` (flat layout, ``config.json`` / ``*.onnx``
    at the top). Our M0 layout is a PER-MODEL SUBDIRECTORY
    (``models/embedding/multilingual-e5-small/``), so the resolver fell
    back to the HF repo id -> ``SentenceTransformer`` tried the hub ->
    offline connection error -> memory/embedding ERROR on every turn.

    Fix: export ``VOICEMEM_E5_MODEL`` (the resolver's designed env
    override, highest priority) pointing at our local directory. Must
    run BEFORE voicemem's ``local_e5_embedder`` module is first imported
    (it resolves the model name at import time); the voicemem package
    uses PEP 562 lazy attribute loading, so importing ``voicemem`` has
    NOT imported it yet. A value the operator set explicitly is never
    overridden. ``config/env.local.ps1`` exports the same variable for
    the CLI path. Returns the pinned path (None when not applicable).
    """
    import os

    if os.environ.get("VOICEMEM_E5_MODEL"):
        return None
    model_dir = config.embedding_model_dir
    if model_dir.is_dir() and (model_dir / "config.json").is_file():
        os.environ["VOICEMEM_E5_MODEL"] = str(model_dir)
        logger.info("VoiceMem E5 embedder pinned to local model: %s", model_dir)
        return str(model_dir)
    return None


def pin_vendor_llm_env(config: AgentConfig) -> dict[str, str]:
    """Pin the vendor package's OpenAI-compatible env vars at our llama-server.

    v0.5.1 (TASK 1 forensic finding): the vendored VoiceMem package reads its
    LLM target from the environment at CALL time, with asymmetric fallbacks
    that silently break the memory chain when the process was started from a
    raw shell (config/env.local.ps1 not dot-sourced — only start_agent.ps1
    does it automatically):

    * ``orchestrator._llm_json/_llm_text``: ``model=os.environ.get(
      "OPENAI_MODEL", "gpt-4o-mini")`` and ``api_key=os.environ.get(
      "OPENAI_API_KEY")`` — with no env, the OpenAI() CONSTRUCTOR raises
      (api_key must be set), the call dies inside a ``try/except`` that only
      prints, and the memory extraction silently returns "".
    * ``memory_repository_v2``/``query_slot_classifier``:
      ``base_url=os.environ.get("OPENAI_BASE_URL") or None`` — with no env
      the OpenAI SDK DEFAULTS TO https://api.openai.com (an offline-policy
      violation: query text would leave the machine before failing).

    This pin (the LLM counterpart of :func:`pin_e5_local_model`, same
    "operator value is never overridden" rule) exports the app's own
    ``llama_server_url``/``llm_model_name``/dummy key so every vendor leg
    targets the local llama-server. Returns the pinned variables.
    """
    import os

    pinned: dict[str, str] = {}
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = config.llama_server_url
        pinned["OPENAI_BASE_URL"] = config.llama_server_url
    if not os.environ.get("OPENAI_API_KEY"):
        # dummy, exactly like config/env.local.ps1: the openai SDK demands a
        # key even for a local llama-server that ignores it
        os.environ["OPENAI_API_KEY"] = "not-needed-but-required-by-openai-lib"
        pinned["OPENAI_API_KEY"] = "(dummy)"
    if not os.environ.get("OPENAI_MODEL"):
        os.environ["OPENAI_MODEL"] = config.llm_model_name
        pinned["OPENAI_MODEL"] = config.llm_model_name
    if not os.environ.get("OPENAI_CHAT_MODEL"):
        os.environ["OPENAI_CHAT_MODEL"] = config.llm_model_name
        pinned["OPENAI_CHAT_MODEL"] = config.llm_model_name
    if pinned:
        logger.info(
            "VoiceMem vendor LLM env pinned to %s (model %s)",
            config.llama_server_url,
            config.llm_model_name,
        )
    return pinned


def verify_e5_local_model(config: AgentConfig) -> bool:
    """v0.4.9 (report issue #5): END-TO-END check that the pinned E5 model works.

    The v0.4.4 pin only exported the env var and HOPED voicemem would use it
    (``pin_e5_local_model`` returns None on success, nothing validated the
    embedder actually loaded from the local path, and a silent fallback to
    the HuggingFace hub could hang memory retrieval offline). This check:

    1. verifies the model directory + ``config.json`` exist (else False —
       the pin cannot happen, so say so LOUDLY instead of silently);
    2. when ``sentence_transformers`` is importable, loads the model from
       the LOCAL directory and encodes one dummy string — the embedder is
       proven to produce embeddings offline, before any turn depends on it.

    Returns True when the model is usable (or the dir is absent AND the
    operator pinned nothing — voicemem's own resolution then applies).
    Never raises; failures are logged as warnings the operator can act on.
    """
    import os

    model_dir = config.embedding_model_dir
    pinned = os.environ.get("VOICEMEM_E5_MODEL", "")
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        if pinned and Path(pinned) == model_dir:
            logger.warning(
                "E5 local model missing although VOICEMEM_E5_MODEL points at "
                "it: %s - offline memory retrieval may hang or fail; "
                "re-run scripts/download_models.ps1", 
                model_dir,
            )
            return False
        return True  # no local copy configured; not our failure to report
    # Local copy present: prove it embeds (when the library is installed).
    try:
        from sentence_transformers import SentenceTransformer  # heavy, lazy
    except ImportError:
        logger.debug(
            "sentence_transformers not installed; skipping the E5 warm-up "
            "encode (voicemem will load it itself)"
        )
        return True
    try:
        embedder = SentenceTransformer(str(model_dir))
        result = embedder.encode(["verify"], show_progress_bar=False)
        if result is None or len(result) == 0:
            raise RuntimeError("E5 model loaded but produced no embeddings")
        logger.info(
            "E5 local model verified end-to-end (dim=%d): %s",
            int(np.asarray(result).shape[-1]) if hasattr(result, "shape") else -1,
            model_dir,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - verification must never raise
        logger.warning(
            "E5 local model FAILED the warm-up encode (%s): %s - memory "
            "retrieval may fall back to the (offline-unreachable) hub",
            exc,
            model_dir,
        )
        return False


@dataclass
class TurnContext:
    """Result of one VoiceMem turn (or its degraded-mode fallback)."""

    transcript: str
    memory_context: str = ""
    speaker_id: str = "voice_user"
    turn: Optional[Any] = None


class VoiceMemBridge:
    """Wraps the controlled VoiceMem fork in ``text_mode`` with our LLM reply hook.

    Construction parameters mirror the VoiceMem facade: ``mode="text_mode"``,
    ``user_id=<per-turn speaker id>``, ``base_url=config.llama_server_url``,
    ``top_k=config.top_k``, ``memory_root=str(config.memory_root_path)`` and
    ``reply=`` our async generator (teacher persona + ``llm.chat_stream``).
    M3 (spec 9.3.4): the memory space follows the SPEAKER. The bridge keeps
    one lazily-constructed VoiceMem facade per ``user_id`` (``{user_id: vm}``
    dict) - each facade isolates its own SQLite + Qdrant collection, so
    memories never mix across speakers. When the package is missing ->
    degraded mode: ``memory_context=""``, warning logged exactly once.
    """

    def __init__(self, config: AgentConfig, llm: Optional["LlmClient"] = None) -> None:
        self._config = config
        self._llm = llm
        self._vm_by_user: dict[str, Any] = {}
        self._vm_last_access: dict[str, float] = {}
        # v0.4.9 (issue #5): failure bookkeeping with a retry window instead
        # of the old one-shot ``_vm_failed`` flag that disabled memory for the
        # whole session after a single transient failure.
        self._vm_fail_count = 0
        self._vm_last_fail_ts = 0.0
        self._e5_check_done = False
        self._degraded_warned = False
        self._cancelled = False
        self._feed_adapter: Optional[str] = None
        self._commit_adapter: Optional[str] = None
        # v0.5.1 (TASK 1): the last user transcript per speaker — the pinned
        # vendor consolidates a turn as ingest(text, agent_reply=reply), so
        # commit_reply needs the user text that produced the reply.
        self._last_user_text: dict[str, str] = {}
        pin_e5_local_model(config)
        # v0.5.1 (TASK 1): same pin discipline for the vendor LLM target env
        # (operator-set values are never overridden).
        pin_vendor_llm_env(config)

    def _ensure_e5_local_model(self) -> None:
        """v0.4.4: pin the E5 embedder to the local model (see module-level
        :func:`pin_e5_local_model`); kept as an instance alias for tests."""
        pin_e5_local_model(self._config)

    def is_available(self) -> bool:
        """True when the voicemem package is importable."""
        return _HAS_VOICEMEM

    # -- construction ----------------------------------------------------------- #

    async def _ensure_vm(self, user_id: str = "") -> Optional[Any]:
        """Lazily construct (and cache) the VoiceMem facade for ONE user.

        M3: one facade per speaker id - the per-user memory-space separation
        (spec 9.3.4) comes from the facade's own ``user_id`` parameter.

        v0.4.9 (report issue #5): a failed construction is NOT remembered
        forever any more — up to 3 consecutive attempts are made, then
        retries pause for ``_VM_RETRY_WINDOW_S`` seconds (30 s) before the
        next try, so a network blip on the first turn no longer silently
        disables long-term memory for the entire session. v0.4.9 (issue #7):
        the per-user cache is pruned (LRU cap 10 + 1 h idle) on every access.
        """
        if not _HAS_VOICEMEM:
            return None
        key = user_id or self._config.user_id
        if key in self._vm_by_user:
            self._vm_last_access[key] = time.monotonic()
            return self._vm_by_user[key]
        if not self._may_retry_vm():
            return None
        try:
            vm = await asyncio.to_thread(self._construct_vm, key)
            self._vm_by_user[key] = vm
            self._vm_last_access[key] = time.monotonic()
            self._vm_fail_count = 0
            self._prune_vm_cache()
            logger.info(
                "VoiceMem initialized (mode=text_mode, user_id=%s, memory_root=%s)",
                key,
                self._config.memory_root_path,
            )
        except Exception as exc:  # noqa: BLE001 - degrade on any construction error
            self._vm_fail_count += 1
            self._vm_last_fail_ts = time.monotonic()
            logger.warning(
                "VoiceMem construction failed (attempt %d): %s",
                self._vm_fail_count,
                exc,
                exc_info=True,
            )
        return self._vm_by_user.get(key)

    def _may_retry_vm(self) -> bool:
        """True when another VoiceMem construction attempt is allowed now."""
        if self._vm_fail_count < _VM_MAX_FAILS:
            return True
        paused_for = time.monotonic() - self._vm_last_fail_ts
        if paused_for >= _VM_RETRY_WINDOW_S:
            logger.info(
                "Retrying VoiceMem construction after %.0f s pause "
                "(previous failures: %d)",
                paused_for,
                self._vm_fail_count,
            )
            return True
        return False

    def _prune_vm_cache(self) -> None:
        """v0.4.9 (issue #7): bound the per-user facade cache.

        Evicts facades idle for over ``_VM_IDLE_S`` (1 h), then the
        least-recently-used beyond ``_VM_CACHE_MAX`` (10). A facade in active
        use by the current turn is safe: eviction only drops OUR reference,
        so the caller's local one keeps it alive until the turn ends. Close
        is best-effort (close/aclose/shutdown when the facade exposes it).
        """
        now = time.monotonic()
        # 1) idle eviction
        for key in [
            k
            for k, ts in self._vm_last_access.items()
            if now - ts > _VM_IDLE_S and k in self._vm_by_user
        ]:
            self._evict_vm(key, f"idle for {now - self._vm_last_access[k]:.0f} s")
        # 2) LRU cap
        while len(self._vm_by_user) > _VM_CACHE_MAX and self._vm_last_access:
            key = min(self._vm_last_access, key=lambda k: self._vm_last_access[k])
            self._evict_vm(key, "LRU cap exceeded")

    def _evict_vm(self, key: str, reason: str) -> None:
        """Drop one cached facade (best-effort close, never raises)."""
        vm = self._vm_by_user.pop(key, None)
        self._vm_last_access.pop(key, None)
        if vm is None:
            return
        logger.info("VoiceMem facade evicted for user %r (%s)", key, reason)
        for name in ("close", "aclose", "shutdown"):
            close_fn = getattr(vm, name, None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("VoiceMem close via %s() failed: %s", name, exc)
                return

    def _construct_vm(self, user_id: str) -> Any:
        """Build the VoiceMem facade for one user (runs in a worker thread)."""
        # v0.4.9 (issue #5): prove the pinned E5 model actually embeds BEFORE
        # the first facade depends on it (once per bridge; the check loads
        # sentence-transformers and runs one encode when the library exists).
        if not self._e5_check_done:
            self._e5_check_done = True
            verify_e5_local_model(self._config)
        factory = getattr(voicemem, "VoiceMem")

        def embedding_factory():  # local multilingual E5, 0 network
            # v0.5.0 (controlled fork): the memory embedder is EXPLICITLY
            # injected — the left-brain repo, the right-brain trait table and
            # the graph layers must all use the local E5 (see VOICEMEM_PIN.json
            # embedding_policy; the web server path app/web_server.py
            # _build_facade has injected this since v0.4.x, the CLI bridge
            # had not — its left-brain repo fell back to the package default,
            # which the controlled fork pins to the same local E5 anyway).
            from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder

            return LocalE5Embedder()

        return factory(
            mode="text_mode",
            user_id=user_id,
            base_url=self._config.llama_server_url,
            top_k=self._config.top_k,
            memory_root=str(self._config.memory_root_path),
            embedding=embedding_factory,
            reply=self.build_reply_fn(),
        )

    # -- turn processing --------------------------------------------------------- #

    async def process_turn(
        self, transcript: str, speaker_id: str = "voice_user"
    ) -> TurnContext:
        """Feed the user utterance to VoiceMem and return the retrieval context.

        Real path: ``asyncio.to_thread`` over the VoiceMem streaming feed
        (``feed_partial(text, ended=True)`` preferred; adapter table fallback)
        and extraction of the memory context string from the Turn/SearchResult.
        Degraded path: empty memory context. The ``_cancelled`` flag is checked
        between steps (speculative-retrieval cancel on barge-in).
        """
        self._cancelled = False
        vm = await self._ensure_vm(speaker_id)
        if vm is None:
            self._warn_degraded()
            return TurnContext(transcript=transcript, memory_context="", speaker_id=speaker_id)

        if self._cancelled:
            return TurnContext(transcript=transcript, memory_context="", speaker_id=speaker_id)

        # v0.5.1: remember the turn text for commit_reply's pair-consolidation
        # (the pinned vendor wants ingest(text, agent_reply=reply) AFTER the
        # reply exists — exactly the proven web_server MemoryLayer pattern).
        self._last_user_text[speaker_id] = transcript

        result = await self._retrieve_user_turn(vm, transcript)
        if self._cancelled:
            return TurnContext(transcript=transcript, memory_context="", speaker_id=speaker_id)

        memory_context = _extract_memory_context(result) if result is not None else ""
        return TurnContext(
            transcript=transcript,
            memory_context=memory_context,
            speaker_id=speaker_id,
            turn=result,
        )

    async def _retrieve_user_turn(self, vm: Any, transcript: str) -> Optional[Any]:
        """v0.5.1 (TASK 1 forensic fix): retrieval-first against the pinned vendor.

        The pinned vendor (e8384e0) has NO streaming-feed API — the bridge's
        legacy adapter table targets method names that never existed on this
        package (proven: "No compatible VoiceMem feed API found" on every CLI
        turn, so the voice chain ran with an empty memory_context forever).
        Its actual public retrieval API is ``vm.search(query)`` → SearchResult
        (hits + rb_hits), which is exactly what the web path has called all
        along (app/web_server.py MemoryLayer.search — the proven pattern).

        Turn storage deliberately does NOT happen here: the vendor wants the
        assistant reply in the same ``ingest(text, agent_reply=...)`` call
        (left-brain disambiguation + right-brain attribution need the pair),
        which commit_reply performs after the reply streamed. Falls back to
        the legacy adapter table when ``search`` is absent (forward-compat).
        """
        search_fn = getattr(vm, "search", None)
        if callable(search_fn):
            try:
                result = await asyncio.to_thread(search_fn, transcript)
                if self._feed_adapter != "search":
                    logger.info("VoiceMem user-turn retrieval using search()")
                    self._feed_adapter = "search"
                return result
            except Exception as exc:  # noqa: BLE001 - degrade on runtime errors
                logger.warning("VoiceMem search() failed: %s", exc)
                return None
        return await self._feed_user_turn(vm, transcript)

    async def _feed_user_turn(self, vm: Any, transcript: str) -> Optional[Any]:
        """Run the first compatible feed adapter in a worker thread."""
        for name, adapter in _FEED_ADAPTERS:
            method = getattr(vm, name, None)
            if not callable(method):
                continue
            try:
                result = await asyncio.to_thread(adapter, vm, transcript)
                if self._feed_adapter != name:
                    logger.info("VoiceMem user-turn feed using adapter %s()", name)
                    self._feed_adapter = name
                return result
            except TypeError as exc:
                logger.debug("VoiceMem adapter %s() signature mismatch: %s", name, exc)
                continue
            except Exception as exc:  # noqa: BLE001 - degrade on runtime errors
                logger.warning("VoiceMem feed via %s() failed: %s", name, exc)
                return None
        logger.warning("No compatible VoiceMem feed API found on the installed package")
        return None

    async def commit_reply(
        self, reply_text: str, speaker_id: str = "voice_user"
    ) -> str:
        """Feed the assistant reply back for long-term memory consolidation.

        Real path: first compatible adapter from the commit table
        (``add_message`` with an assistant role, ``remember``, ``feed_partial``)
        executed in a worker thread. Degraded path: no-op.

        v0.4.9 (report issue #1): RETURNS a status instead of a silent None —

        * ``COMMIT_COMMITTED`` ("committed") - the reply is in long-term memory;
        * ``COMMIT_UNAVAILABLE`` ("unavailable") - no voicemem package (degraded
          mode, nothing to store — not a data-loss event);
        * ``COMMIT_FAILED`` ("failed") - the reply was NOT stored (adapter/
          backend failure). Callers (the pipeline) now retry, queue and
          surface this instead of logging it away as a warning.

        Never raises.
        """
        try:
            vm = await self._ensure_vm(speaker_id)
            if vm is None:
                if _HAS_VOICEMEM:
                    # package present but the facade would not come up: the
                    # reply IS lost from long-term memory this turn.
                    return COMMIT_FAILED
                self._warn_degraded()
                return COMMIT_UNAVAILABLE
            if not reply_text or not reply_text.strip():
                return COMMIT_COMMITTED  # nothing to store is not a failure
            # v0.5.1 (TASK 1 forensic fix): the pinned vendor consolidates a
            # turn as ONE call — ingest(user_text, agent_reply=reply) — so the
            # left brain can disambiguate the user utterance with the reply
            # and the right brain can attribute emotion using the pair. The
            # legacy table below targets API names that do not exist on the
            # pinned package (add_message/remember/feed_partial — proven by
            # the TASK 1 forensic run), so it only serves as forward-compat.
            ingest_fn = getattr(vm, "ingest", None)
            user_text = self._last_user_text.get(speaker_id, "")
            if callable(ingest_fn) and user_text.strip():
                try:
                    await asyncio.to_thread(
                        ingest_fn, user_text, agent_reply=reply_text, async_facts=True
                    )
                    if self._commit_adapter != "ingest":
                        logger.info(
                            "VoiceMem reply committed using ingest(text, agent_reply=...)"
                        )
                        self._commit_adapter = "ingest"
                    return COMMIT_COMMITTED
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "VoiceMem commit via ingest(agent_reply=...) failed: %s",
                        exc,
                        exc_info=True,
                    )
                    return COMMIT_FAILED
            for name, adapter in _COMMIT_ADAPTERS:
                method = getattr(vm, name, None)
                if not callable(method):
                    continue
                try:
                    await asyncio.to_thread(adapter, vm, reply_text)
                    if self._commit_adapter != name:
                        logger.info("VoiceMem reply committed using adapter %s()", name)
                        self._commit_adapter = name
                    return COMMIT_COMMITTED
                except TypeError as exc:
                    logger.debug(
                        "VoiceMem commit adapter %s() signature mismatch: %s", name, exc
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "VoiceMem commit via %s() failed: %s",
                        name,
                        exc,
                        exc_info=True,
                    )
                    return COMMIT_FAILED
            logger.warning("No compatible VoiceMem commit API found on the installed package")
            return COMMIT_FAILED
        except Exception as exc:  # noqa: BLE001 - must never raise
            logger.warning("commit_reply failed: %s", exc, exc_info=True)
            return COMMIT_FAILED

    async def store_fact(self, text: str, speaker_id: str = "voice_user") -> None:
        """Feed one standalone observation into long-term memory (M2).

        Used by the pipeline's emotion layer (spec 8.3.4): a strong turn
        emotion becomes a fact ("Emotional state: the user sounded
        frustrated ... while saying ..."). Implementation: the same
        best-effort adapter machinery as the user-turn feed, executed in
        a worker thread. Degraded path: no-op. Never raises.
        """
        try:
            if not text or not text.strip():
                return
            vm = await self._ensure_vm(speaker_id)
            if vm is None:
                self._warn_degraded()
                return
            await self._feed_user_turn(vm, text)
        except Exception as exc:  # noqa: BLE001 - must never raise
            logger.warning("store_fact failed: %s", exc)

    # -- reply hook ----------------------------------------------------------------- #

    def build_reply_fn(self) -> Callable[..., Any]:
        """Return the async generator hook for ``VoiceMem(reply=...)``.

        Signature: ``reply(text, memory_context="") -> AsyncIterator[str]``.
        Internally builds the teacher system prompt from *memory_context* and
        streams our llama-server via ``self._llm.chat_stream``. Raises an
        :class:`LlmUnavailableError` (RuntimeError) when no LlmClient was
        injected. Stops yielding once ``cancel_pending()`` was called.
        """

        async def reply(text: str, memory_context: str = "") -> AsyncIterator[str]:
            if self._llm is None:
                from .llm import LlmUnavailableError  # lazy: keep httpx out of import time

                raise LlmUnavailableError(
                    "VoiceMemBridge has no LlmClient; cannot serve the reply hook"
                )
            # v0.4.9 (report issue #3): VoiceMem hands us ITS retrieval block as
            # memory_context, unvalidated. Cap it and run the whole message
            # list through the shared prompt budget guard (same 14000-char
            # rule web_server applies since v0.4.5) so the reply hook can no
            # longer overflow the fixed 8192-token llama-server context.
            capped_context = _cap(
                memory_context or "", limit=_REPLY_MEMORY_CONTEXT_MAX_CHARS
            )
            system_prompt = build_system_prompt(capped_context)
            messages = fit_prompt_budget(build_messages(text, system_prompt))
            async for delta in self._llm.chat_stream(messages):
                if self._cancelled:
                    return
                yield delta

        return reply

    # -- cancellation ----------------------------------------------------------------- #

    def cancel_pending(self) -> None:
        """Cancel in-flight speculative retrieval (barge-in). Never raises.

        Sets an internal flag checked between process_turn steps and inside the
        reply stream; also calls the vm-level cancel API when the installed
        VoiceMem exposes one (hasattr-guarded).
        """
        self._cancelled = True
        for vm in self._vm_by_user.values():
            if vm is None:
                continue
            for name in ("cancel", "cancel_pending", "abort"):
                cancel_fn = getattr(vm, name, None)
                if callable(cancel_fn):
                    try:
                        cancel_fn()
                        logger.debug("VoiceMem cancellation via %s()", name)
                        break
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("VoiceMem cancel via %s() failed: %s", name, exc)

    # -- internals ------------------------------------------------------------------- #

    def _warn_degraded(self) -> None:
        """Log the degraded-mode warning exactly once per bridge instance."""
        if not self._degraded_warned:
            self._degraded_warned = True
            logger.warning(_DEGRADED_WARNING)


# --------------------------------------------------------------------------- #
# Memory context extraction (best effort over unknown result shapes)
# --------------------------------------------------------------------------- #


def _extract_memory_context(result: Any) -> str:
    """Extract a memory-context string from a Turn/SearchResult-like object.

    v0.5.1 (TASK 1): the pinned vendor's ``Search()`` returns a SearchResult
    exposing ``hits`` (left-brain facts) and ``rb_hits`` (right-brain
    personality/emotion notes) — rendered here EXACTLY like the proven web
    path (app/web_server.py ``_memory_context``): top-5 facts + top-3
    right-brain notes as ``"- text"`` lines, right-brain content cleaned of
    the vendor's date/slot prefix and advisory suffix, capped at ~1200
    characters. Objects without hits/rb_attrs fall through to the generic
    attribute walk (context/memory_list) below.
    """
    hits = getattr(result, "hits", None)
    rb_hits = getattr(result, "rb_hits", None)
    if hits is not None or rb_hits is not None:
        parts: list[str] = []
        for h in (hits or [])[:5]:
            t = (getattr(h, "text", "") or "").strip()
            if t:
                parts.append(f"- {t}")
        for h in (rb_hits or [])[:3]:
            t = clean_rb_content(getattr(h, "content", "") or "").strip()
            if t:
                parts.append(f"- {t}")
        return "\n".join(parts)[:1200]
    for attr in _CONTEXT_ATTRIBUTES:
        value = getattr(result, attr, None)
        if value is None:
            continue
        rendered = _render_memory_value(value)
        if rendered:
            return _cap(rendered)
    for attr in _MEMORY_LIST_ATTRIBUTES:
        value = getattr(result, attr, None)
        if isinstance(value, (list, tuple)):
            rendered = _render_memory_value(value)
            if rendered:
                return _cap(rendered)
    return ""


def _render_memory_value(value: Any) -> str:
    """Render a context value (string / list / object) as '- {memory}' lines."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        lines = [f"- {text}" for text in (_memory_item_text(item) for item in value) if text]
        return "\n".join(lines)
    for attr in _MEMORY_LIST_ATTRIBUTES:
        items = getattr(value, attr, None)
        if isinstance(items, (list, tuple)):
            lines = [f"- {text}" for text in (_memory_item_text(item) for item in items) if text]
            if lines:
                return "\n".join(lines)
    text = str(value).strip()
    if not text or (text.startswith("<") and " object at " in text):
        return ""
    return text


def _memory_item_text(item: Any) -> str:
    """Text of one memory entry (str / dict / object with common attributes)."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("memory", "text", "content", "fact"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    for attr in ("memory", "text", "content", "fact"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    text = str(item).strip()
    if not text or (text.startswith("<") and " object at " in text):
        return ""
    return text


def _cap(text: str, limit: int = _MEMORY_CONTEXT_MAX_CHARS) -> str:
    """Cap *text* at *limit* characters (append '...' when truncated)."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."
