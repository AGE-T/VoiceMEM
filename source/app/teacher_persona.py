"""Conversational voice-assistant persona prompt builder (pure standard library).

Builds the system prompt for the local LLM and the OpenAI-style message list.
v0.4.14: the persona is the natural voice companion from the successful Qwen
Ollama experiments (short spoken turns, everyday Hungarian, latest-message
language matching) — the M1 language-teacher scaffold is retired. The M2
emotion parameters stay in the signature; when ``label`` is None the emotion
block is omitted entirely.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: v0.4.14: the conversational persona aligned with the successful Ollama
#: experiments (Qwen3.6 35B A3B: natural spoken Hungarian, short turns,
#: HU/EN switching). Replaces the v0.4.3 language-teacher scaffold — the
#: production use case is a natural VOICE companion, not grammar coaching;
#: the old prompt produced teacher-like corrections ("Szia Imre! Én egy
#: szöveges modell vagyok…" was the model answering the TEACHER frame). Memory
#: and emotion blocks are still appended conditionally below, so the fixed
#: scaffold stays well under 250 words.
_BASE_PROMPT = """\
You are a natural, friendly voice assistant.

Speak naturally, as if talking to another person.
Match the user's conversational style and response length.

Keep responses short in normal conversation.
For simple questions, usually answer in one or two sentences.
Do not over explain things during casual conversation.

Do not use lists, headings, numbered sections or long explanations unless \
the user explicitly asks for them.
Do not repeat the user's statement.
Do not provide multiple alternatives when one natural answer is enough.

Use natural everyday spoken Hungarian.
Avoid literal English translations and artificial Hungarian word formations.

Conversation is more important than information density.
Do not behave like an essay writer or customer support chatbot.

Always answer in the same language as the user's latest message unless the \
user explicitly requests another language."""

_MEMORY_HEADER = "Long-term memory about this user:"

#: v0.4.9 (report issue #3): character budget for the ASSEMBLED prompt. The
#: v0.4.5 fix guarded only web_server.py — the code-analysis report found the
#: same overflow UNGUARDED in the CLI pipeline and the VoiceMem reply hook:
#: llama-server runs a fixed ``llm_context_size`` (8192 tokens) and once
#: rejected a 10459-token prompt outright ("request exceeds the available
#: context size"), so a long memory block + history silently lost the reply.
#: HU/EN text is ~2-3 chars/token on the Qwen3.6 tokenizer — 14000 chars is the
#: same deliberately conservative budget web_server has used since v0.4.5,
#: now enforced everywhere the LLM is called.
_PROMPT_CHAR_BUDGET = 14000


def build_system_prompt(
    memory_context: str,
    emotion_label: Optional[str] = None,
    emotion_valence: Optional[float] = None,
    emotion_arousal: Optional[float] = None,
) -> str:
    """Build the conversational voice-assistant system prompt.

    Always contains: the natural voice-companion persona (short spoken
    turns, everyday Hungarian, no lists/headings, match the user's latest
    language — the prompt that produced the successful Qwen Ollama
    experiments), the reply-language rule and the voice-output constraints.

    The "Long-term memory about this user:" block is appended only when
    *memory_context* is non-empty. The emotion block is appended only when
    *emotion_label* is not None (M2: the fused prosody+semantic state) and
    includes valence/arousal, the verify-and-adjust instruction (spec 8.3.2:
    the LLM cross-checks the prosody assessment against the actual words)
    and the frustration/confidence pacing instructions.
    """
    parts: list[str] = [_BASE_PROMPT]

    if memory_context and memory_context.strip():
        parts.append(_MEMORY_HEADER + "\n" + memory_context.strip())

    if emotion_label is not None:
        valence = _format_number(emotion_valence)
        arousal = _format_number(emotion_arousal)
        parts.append(
            f"Emotional state of the user (prosody analysis): {emotion_label} "
            f"(valence={valence}, arousal={arousal}).\n"
            "Verify this assessment against what the user actually said and "
            "adjust it if the words suggest a different state.\n"
            "If the user seems frustrated, simplify the explanation, slow down, "
            "and give more concrete examples. If the user seems confident, "
            "continue at a normal pace."
        )

    return "\n\n".join(parts) + "\n"


def _format_number(value: Optional[float]) -> str:
    """Render a valence/arousal number compactly ('n/a' when missing)."""
    if value is None:
        return "n/a"
    return f"{value:g}"


def build_messages(
    transcript: str,
    system_prompt: str,
    history: Optional[list[dict]] = None,
) -> list[dict]:
    """Build the OpenAI-style message list.

    Layout: ``[{'role': 'system', ...}, *valid_history_entries,
    {'role': 'user', 'content': transcript}]``. History entries that are not
    dicts with string ``role`` and ``content`` values are skipped (logged at
    debug level) instead of corrupting the request.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for entry in history or []:
        if _is_valid_history_entry(entry):
            messages.append({"role": entry["role"], "content": entry["content"]})
        else:
            logger.debug("Skipping invalid history entry: %r", entry)
    messages.append({"role": "user", "content": transcript})
    return messages


def _is_valid_history_entry(entry: Any) -> bool:
    """True when *entry* is a dict with string 'role' and 'content' values."""
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("role"), str)
        and isinstance(entry.get("content"), str)
        and bool(entry.get("role"))
        and bool(entry.get("content"))
    )


def estimate_prompt_chars(messages: list[dict]) -> int:
    """Estimated size of the assembled prompt (total content characters)."""
    return sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))


def fit_prompt_budget(
    messages: list[dict], max_chars: int = _PROMPT_CHAR_BUDGET
) -> list[dict]:
    """v0.4.9: fit the message list into the LLM context budget (never raises).

    Algorithm (report issue #3, mirroring web_server's v0.4.5 guard):
    1. Drop the OLDEST non-system history entries first (the system message
       and the FINAL user message always stay).
    2. If the system + final user message alone still exceed *max_chars*,
       hard-truncate the SYSTEM message content to fit (the memory block is
       context, not instruction — a visible ``[...]`` marker is appended).
       The user text is never truncated (a half question is worse than a
       shortened memory block).

    Returns a NEW list when trimming happened, the original list otherwise
    (so callers can compare identity/length cheaply). Never raises and never
    returns an empty list.
    """
    if estimate_prompt_chars(messages) <= max_chars:
        return messages
    trimmed = list(messages)
    # Step 1: shed history oldest-first, keeping system (index 0) and the
    # final message (the current user turn).
    while len(trimmed) > 2 and estimate_prompt_chars(trimmed) > max_chars:
        dropped = trimmed.pop(1)  # oldest non-system entry
        logger.warning(
            "Prompt over %d-char budget: dropped history entry (%d chars)",
            max_chars,
            len(str(dropped.get("content", ""))),
        )
    if estimate_prompt_chars(trimmed) <= max_chars:
        return trimmed
    # Step 2: hard-cap the system message (memory block) as the last resort.
    capped = list(trimmed)
    system = capped[0] if capped else {"role": "system", "content": ""}
    if str(system.get("role", "")) == "system":
        content = str(system.get("content", ""))
        room = max_chars - sum(
            len(str(m.get("content", ""))) for m in capped[1:]
        )
        if room < 200:
            room = 200
        capped[0] = {
            "role": "system",
            "content": content[: room - 20].rstrip() + "\n[... truncated]",
        }
        logger.warning(
            "Prompt still over %d-char budget after shedding history: "
            "system prompt hard-truncated from %d to %d chars",
            max_chars,
            len(content),
            len(capped[0]["content"]),
        )
    return capped
