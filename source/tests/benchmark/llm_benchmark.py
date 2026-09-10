"""Main-LLM benchmark (Qwen3.6 35B A3B IQ4_XS via llama-server, TARGET machine).

Machine: TARGET with llama-server running on 127.0.0.1:8080 (spec Section 18.3).
Importing is safe anywhere; app.llm / app.teacher_persona are imported lazily.

Method (milestones doc Section 7.6.3, spec Section 19.3):
  - 10 Hungarian + 10 English conversations, 5 turns each.
  - Realistic teacher scenarios: grammar correction, grammar explanation in
    Hungarian, vocabulary training, memory use within the conversation,
    instruction following.
  - Each turn is scored by behavioural keyword checks on the reply:
    * any_of: at least one keyword (case-insensitive substring) must appear;
    * hu: the reply must contain at least one Hungarian explanation marker
      (word-level check) — the teacher explains errors in Hungarian.
  - Conversation history is passed through app.teacher_persona.build_messages.

v0.4.21 NO-THINKING VERIFICATION (the Ollama ``/set nothink`` comparison):
  the run additionally verifies on EVERY turn that the production
  no-thinking configuration is in effect — the exact profile of the
  successful Ollama ``/set nothink`` test:
    * the request carries chat_template_kwargs.enable_thinking=false and
      reasoning_effort "none" (captured from the wire on the first turn);
    * no reply contains a literal <think>/</think> block;
    * no turn dies with the "reasoning ate the budget" empty-reply error;
    * the first content token arrives immediately (no thought phase) —
      min/mean latency is printed in the summary.

Exit criteria: >= 80% of the turns PASS -> exit 0; 1 below that; 2 on
missing prerequisites (llama-server unreachable, modules not importable).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tests/<kind>/x.py -> 3 levels up)

# Word-level Hungarian markers: at least one must appear when a turn expects a
# Hungarian explanation. Words (not substrings) so English text cannot match.
HU_MARKERS = frozenset(
    {
        "az", "és", "egy", "ez", "nem", "igen", "meg", "mert", "van", "helyes",
        "helyesen", "helyett", "helytelen", "hibás", "hibát", "hibád", "javítás",
        "javítva", "példa", "példát", "példamondat", "magyarul", "magyar",
        "jelent", "kérdez", "fontos", "megjegyez", "emlékszel", "rendszerint",
    }
)

# (id, turns) where each turn = (user_text, any_of_keywords, require_hu).
# Empty any_of tuple: no keyword constraint (only the hu flag is checked).
HU_CONVERSATIONS: list[tuple[str, list[tuple[str, tuple[str, ...], bool]]]] = [
    (
        "hu1_present_perfect_javitas",
        [
            ("Szervusz! Angolul azt akarom mondani: I have went to the cinema. Ez így helyes?",
             ("i have gone", "i went"), True),
            ("Köszönöm! És miért nem jó ott a went?",
             ("present perfect", "gone", "went"), True),
            ("Rendben. Adnál két példát a have gone használatára?",
             ("have gone", "példa"), True),
            ("Most én próbálom: She has gone home. Helyes ez?",
             ("igen", "helyes", "pont", "jó"), True),
            ("Emlékszel, milyen hibát javítottál az elején?",
             ("have went", "went", "hibám", "hibát"), True),
        ],
    ),
    (
        "hu2_flat_white_szokincs",
        [
            ("Mit jelent angolul a flat white?",
             ("flat white", "kávé"), True),
            ("És mi a különbség a latte és a flat white között?",
             ("latte", "flat white"), True),
            ("Hogyan mondanám angolul: szeretném kevesebb tejjel?",
             ("less milk", "milk"), True),
            ("Írj egy angol mondatot a flat white-ról!",
             ("flat white",), True),
            ("Tanultunk ma valami új szót?",
             ("flat white", "igen", "ma"), True),
        ],
    ),
    (
        "hu3_borrow_lend",
        [
            ("Mi a különbség az angol borrow és lend között?",
             ("borrow", "lend"), True),
            ("Ha a barátom ad nekem egy könyvet, az lend vagy borrow?",
             ("lend", "borrow"), True),
            ("Adj egy példamondatot a borrow-val!",
             ("borrow",), True),
            ("Most én: Can I borrow your pen? Ez jó mondat?",
             ("igen", "helyes", "jó", "pont"), True),
            ("Emlékszel, melyik két szót vettük át ma?",
             ("borrow", "lend"), True),
        ],
    ),
    (
        "hu4_evszamok",
        [
            ("Hogyan mondják angolul az 1890-es éveket?",
             ("eighteen ninety", "1890"), True),
            ("És hogyan mondanád angolul, hogy 1900?",
             ("nineteen hundred", "1900"), True),
            ("Írj egy angol mondatot az 1890-es évekről!",
             ("1890", "eighteen"), True),
            ("Fordítsd le angolra: 1890-ben épült a ház.",
             ("1890", "eighteen", "built"), True),
            ("Milyen évszámot gyakoroltunk ma?",
             ("1890", "eighteen"), True),
        ],
    ),
    (
        "hu5_grammar_magyarazat",
        [
            ("Miért helyes az 'I have been' az 'I was' helyett?",
             ("present perfect", "have been"), True),
            ("Adj hozzá egy példát!",
             ("have been", "példa"), True),
            ("És mikor használjak egyszerű múltat?",
             ("past", "múlt", "simple"), True),
            ("Melyik helyes: I have been to London in 2019 vagy I went to London in 2019?",
             ("went", "i went"), True),
            ("Emlékszel, miről kérdeztelek a beszélgetés elején?",
             ("have been", "was", "present perfect"), True),
        ],
    ),
    (
        "hu6_utasitasok",
        [
            ("Magyarázd el két rövid mondattal, mi az espresso.",
             ("espresso",), True),
            ("Most ugyanezt angolul mondd el!",
             ("espresso",), False),
            ("Mondj három angol szót a kávé témában!",
             ("coffee", "espresso", "latte"), False),
            ("Fordítsd le magyarra: The coffee is hot.",
             ("forró", "meleg", "kávé"), True),
            ("Három szót kértem vagy kettőt?",
             ("három", "three", "kettő"), True),
        ],
    ),
    (
        "hu7_dont_doesnt",
        [
            ("Angolul: She don't like coffee. Ez így rendben van?",
             ("doesn't", "does not"), True),
            ("Miért kell ott a doesn't?",
             ("third person", "he", "she", "it", "doesn't"), True),
            ("Adj még egy példát a doesn't-vel!",
             ("doesn't", "does not"), True),
            ("He don't work here. Helyes ez így?",
             ("doesn't", "does not"), True),
            ("Milyen hibát javítottunk ma többször?",
             ("don't", "doesn't"), True),
        ],
    ),
    (
        "hu8_szokincs_gyakorlas",
        [
            ("Tanítsd meg nekem az 'inevitably' angol szót!",
             ("inevitably",), True),
            ("Használd egy angol mondatban!",
             ("inevitably",), False),
            ("Magyarul mit jelent?",
             (), True),
            ("Adj egy szinonimát vagy magyar körülírást!",
             ("mindig", "kikerül", "feltétlen", "biztosan", "inevitably"), True),
            ("Mi volt a mai új szó, amit tanultunk?",
             ("inevitably",), True),
        ],
    ),
    (
        "hu9_smalltalk",
        [
            ("Jó reggelt! Hogy vagy ma?",
             ("jó reggelt", "reggelt", "jól", "köszönöm"), True),
            ("Ma kissé fáradt vagyok, de tanulni szeretnék.",
             ("fáradt", "tanul", "rendben", "persze"), True),
            ("Milyen nyelvi célt tűznénk ki mára?",
             ("cél", "nyelv", "angol", "lecke"), True),
            ("Emlékszel, hogy mondtam, fáradt vagyok?",
             ("fáradt",), True),
            ("Foglald össze két mondatban a mai eddigi beszélgetésünket!",
             ("fáradt", "angol", "cél", "tanul"), True),
        ],
    ),
    (
        "hu10_hiba_emlekezet",
        [
            ("Why is I have went wrong?",
             ("i have gone", "i went"), True),
            ("Értem. Jegyezd meg, hogy ezt a hibát szoktam elkövetni!",
             ("megjegyez", "fog", "hib"), True),
            ("Gyakoroljunk mást: hogyan mondanád angolul, hogy szeretem a kávét?",
             ("i love coffee", "coffee", "love"), True),
            ("Közben eszembe jutott: emlékszel a hibámra?",
             ("have went", "went"), True),
            ("És hogyan mondanád helyesen?",
             ("i have gone", "i went"), True),
        ],
    ),
]

EN_CONVERSATIONS: list[tuple[str, list[tuple[str, tuple[str, ...], bool]]]] = [
    (
        "en1_present_perfect",
        [
            ("Why is I have went wrong?",
             ("i have gone", "i went"), True),
            ("Can you explain the rule in Hungarian?",
             ("present perfect", "gone", "went"), True),
            ("Give me two more examples with 'gone'.",
             ("gone",), False),
            ("Now I try: I have gone to school every day this week. Is this natural?",
             ("i have been going", "i went", "every day"), True),
            ("Do you remember my original mistake?",
             ("have went", "went"), True),
        ],
    ),
    (
        "en2_flat_white",
        [
            ("What is a flat white?",
             ("flat white", "espresso"), False),
            ("How is it different from a latte?",
             ("latte", "milk", "flat white"), False),
            ("Order one for me in English, politely.",
             ("would like", "can i", "may i", "could i"), False),
            ("How do you say the year 1890 in English?",
             ("eighteen ninety", "eighteen-ninety"), False),
            ("What did we talk about today?",
             ("flat white", "latte"), False),
        ],
    ),
    (
        "en3_vocabulary",
        [
            ("Teach me the word 'inevitably'.",
             ("inevitably",), False),
            ("Use it in a sentence about coffee.",
             ("inevitably",), False),
            ("What does it mean in Hungarian?",
             (), True),
            ("Now I try: Coffee is inevitably better in the morning. Correct?",
             ("correct", "good", "yes", "well"), True),
            ("What was our word today?",
             ("inevitably",), False),
        ],
    ),
    (
        "en4_grammar_explanation",
        [
            ("Why do we say 'I have been' instead of 'I was'?",
             ("present perfect", "have been"), False),
            ("Explain it in Hungarian, please.",
             (), True),
            ("Give me an example sentence with 'I have been'.",
             ("have been",), False),
            ("And now one with 'I was'.",
             ("was",), False),
            ("Do you remember what I asked about?",
             ("have been", "was", "present perfect"), False),
        ],
    ),
    (
        "en5_instruction_following",
        [
            ("Explain the difference between 'borrow' and 'lend' in three short sentences.",
             ("borrow", "lend"), False),
            ("Now give me one example sentence with 'borrow'.",
             ("borrow",), False),
            ("And one with 'lend'.",
             ("lend",), False),
            ("Translate into Hungarian: Can I borrow your pen?",
             ("kölcsön", "tudok", "kérhetek"), True),
            ("Summarize what we practiced.",
             ("borrow", "lend"), False),
        ],
    ),
    (
        "en6_smalltalk",
        [
            ("Good morning! How are you?",
             ("good morning", "morning", "fine", "well", "good"), False),
            ("I want to practice English today.",
             ("practice", "english", "great", "good", "start"), False),
            ("What should we start with?",
             ("grammar", "vocabulary", "warm", "start", "conversation"), False),
            ("Are you my teacher?",
             ("teacher", "yes", "tanár"), False),
            ("What did I say I wanted to do today?",
             ("practice", "english"), False),
        ],
    ),
    (
        "en7_dont_doesnt",
        [
            ("Is this correct: He don't have a car?",
             ("doesn't", "does not"), True),
            ("Why is 'doesn't' correct here?",
             ("third person", "he", "she", "it", "doesn't"), True),
            ("Give me two examples with 'doesn't'.",
             ("doesn't",), False),
            ("Now fix this: She don't like tea.",
             ("doesn't", "does not"), True),
            ("Which mistake did I make twice today?",
             ("don't", "doesn't"), True),
        ],
    ),
    (
        "en8_numbers",
        [
            ("How do you say 1890 in English?",
             ("eighteen ninety", "eighteen-ninety"), False),
            ("What about the year 2010?",
             ("twenty ten", "two thousand"), False),
            ("Say a sentence using the year 1890.",
             ("1890", "eighteen"), False),
            ("Now say the year 1890 in Hungarian.",
             ("ezerkilenc",), True),
            ("Which year did we practice?",
             ("1890", "eighteen"), False),
        ],
    ),
    (
        "en9_memory_recall",
        [
            ("I always make this mistake: I have went. Please remember it.",
             ("i have gone", "i went"), True),
            ("Let us talk about something else. What is a latte?",
             ("latte",), False),
            ("Do you remember my mistake?",
             ("have went", "went"), True),
            ("Correct it for me once more.",
             ("i have gone", "i went"), True),
            ("Thank you! Are you a good teacher?",
             ("teacher", "yes", "tanár"), False),
        ],
    ),
    (
        "en10_complex_instruction",
        [
            ("In two sentences, explain what espresso is, then ask me a question about it.",
             ("espresso", "?"), False),
            ("Now explain it in Hungarian.",
             (), True),
            ("Give me three English words related to coffee.",
             ("coffee", "espresso", "latte", "milk"), False),
            ("Ask me to make a sentence with one of them.",
             ("sentence", "mondat"), False),
            ("Do you remember the three words?",
             ("coffee", "espresso", "latte", "milk"), False),
        ],
    ),
]


# Literal Qwen thinking-channel tags (Qwen3 AND Qwen3.6 templates both use
# them; the production model's template renders '<think>\n\n</think>\n\n' when
# enable_thinking is false). A reply that still CONTAINS these literals means
# a thinking block leaked into the visible content — impossible when the
# no-thinking configuration is in effect.
THINKING_TAG_MARKERS = ("<think>", "</think>")


def reply_has_thinking_block(reply: str) -> bool:
    """True when the visible reply carries a literal thinking-channel tag."""
    return any(marker in (reply or "") for marker in THINKING_TAG_MARKERS)


def reply_has_hungarian(reply: str) -> bool:
    """Word-level Hungarian marker check on the reply (case-insensitive)."""
    words = set(reply.casefold().replace(",", " ").replace(".", " ").replace("!", " ")
                .replace("?", " ").replace(";", " ").replace(":", " ").split())
    return bool(words & HU_MARKERS)


def check_turn(reply: str, any_of: tuple[str, ...], require_hu: bool) -> list[str]:
    """Return the list of failed check descriptions for one turn."""
    failed: list[str] = []
    lowered = reply.casefold()
    if any_of and not any(keyword in lowered for keyword in any_of):
        failed.append(f"missing any of: {', '.join(any_of)}")
    if require_hu and not reply_has_hungarian(reply):
        failed.append("no Hungarian explanation marker in reply")
    # v0.4.21: a visible thinking block is ALWAYS a failure — the production
    # configuration (server --reasoning off + request-level suppression)
    # must never let one through (the Ollama /set nothink comparison).
    if reply_has_thinking_block(reply):
        failed.append("reply contains a literal <think> block (thinking not disabled)")
    return failed


async def run(args: argparse.Namespace) -> tuple[int, int, list[dict[str, Any]], dict[str, Any]]:
    """Run every conversation; return (passed, total, failures, thinking_stats)."""
    import json
    import time

    import httpx

    from app.config import AgentConfig
    from app.llm import LlmClient
    from app.teacher_persona import build_messages, build_system_prompt

    cfg = AgentConfig()
    cfg.apply_env()
    if args.config and Path(args.config).exists():
        try:
            cfg = AgentConfig.from_yaml(Path(args.config))
        except (TypeError, ValueError, RuntimeError) as exc:
            print(f"WARN: YAML config load failed ({exc}); using env + defaults")

    llm = LlmClient(cfg)

    # v0.4.21: capture the EXACT wire payload of the first request so the
    # benchmark itself documents the thinking switches the application sent
    # (observation only — the LlmClient request construction is untouched).
    wire_requests: list[dict] = []

    async def _capture_request(request: httpx.Request) -> None:
        try:
            wire_requests.append(json.loads(request.content.decode("utf-8")))
        except Exception:  # noqa: BLE001 - observation must never break the run
            pass

    if getattr(cfg, "llm_disable_thinking", True):
        llm._client = httpx.AsyncClient(
            base_url=cfg.llama_server_url,
            timeout=httpx.Timeout(300.0, connect=5.0),
            event_hooks={"request": [_capture_request]},
        )

    thinking_stats: dict[str, Any] = {
        "request_switches": None,
        "reasoning_eaten_errors": 0,
        "thinking_block_turns": 0,
        "empty_replies": 0,
        "first_token_s": [],
    }
    try:
        if not await llm.health_check():
            print(f"llama-server unreachable at {cfg.llama_server_health_url}")
            print("Start it first: scripts/start_llama_server.ps1")
            return (0, -1, [], thinking_stats)
        print(f"llama-server reachable: {cfg.llama_server_health_url} "
              f"(llm_disable_thinking={cfg.llm_disable_thinking})")
        conversations = HU_CONVERSATIONS + EN_CONVERSATIONS
        passed = 0
        total = 0
        failures: list[dict[str, Any]] = []
        for conv_id, turns in conversations:
            history: list[dict[str, str]] = []
            print(f"[{conv_id}]")
            for turn_idx, (user, any_of, require_hu) in enumerate(turns, start=1):
                system_prompt = build_system_prompt("")
                messages = build_messages(user, system_prompt, history or None)
                chunks: list[str] = []
                t_turn = time.perf_counter()
                first_tok_s: float | None = None
                try:
                    async for delta in llm.chat_stream(messages):
                        if first_tok_s is None:
                            first_tok_s = time.perf_counter() - t_turn
                            thinking_stats["first_token_s"].append(first_tok_s)
                        chunks.append(delta)
                except Exception as exc:  # noqa: BLE001 - transport error fails the turn
                    reply = ""
                    exc_text = str(exc)
                    if "reasoning" in exc_text.lower() or "thinking" in exc_text.lower():
                        thinking_stats["reasoning_eaten_errors"] += 1
                    print(f"  turn {turn_idx}: LLM ERROR ({exc})")
                    failures.append({"conversation": conv_id, "turn": turn_idx,
                                     "error": exc_text, "reply": ""})
                    total += 1
                    continue
                if wire_requests and thinking_stats["request_switches"] is None:
                    first = wire_requests[0]
                    thinking_stats["request_switches"] = {
                        "chat_template_kwargs": first.get("chat_template_kwargs"),
                        "reasoning_effort": first.get("reasoning_effort"),
                    }
                reply = "".join(chunks).strip()
                if not reply:
                    thinking_stats["empty_replies"] += 1
                if reply_has_thinking_block(reply):
                    thinking_stats["thinking_block_turns"] += 1
                total += 1
                failed_checks = check_turn(reply, any_of, require_hu)
                if failed_checks:
                    print(f"  turn {turn_idx}: FAIL ({'; '.join(failed_checks)})")
                    failures.append({"conversation": conv_id, "turn": turn_idx,
                                     "checks": failed_checks, "reply": reply})
                else:
                    passed += 1
                    print(f"  turn {turn_idx}: PASS")
                history.append({"role": "user", "content": user})
                history.append({"role": "assistant", "content": reply})
        return (passed, total, failures, thinking_stats)
    finally:
        await llm.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 main-LLM teacher benchmark (target machine)")
    parser.add_argument("--config", help="optional YAML config (config/voicemem_config.yaml)")
    args = parser.parse_args()

    try:
        import app.llm  # noqa: F401  (import guard with informative message)
        import app.teacher_persona  # noqa: F401
    except ImportError:
        print("app.llm / app.teacher_persona not importable — run from the repo root.")
        return 2

    passed, total, failures, thinking_stats = asyncio.run(run(args))
    if total < 0:  # llama-server unreachable
        return 2
    rate = passed / total if total else 0.0
    print("=" * 72)
    print("M1 LLM benchmark summary")
    print("=" * 72)
    print(f"conversations : {len(HU_CONVERSATIONS)} HU + {len(EN_CONVERSATIONS)} EN "
          f"({(len(HU_CONVERSATIONS) + len(EN_CONVERSATIONS)) * 5} turns total)")
    print(f"passed turns  : {passed}/{total} ({rate * 100:.1f}%)")
    print("-" * 72)
    # v0.4.21: the no-thinking profile — direct comparison with the successful
    # Ollama /set nothink run (immediate answers, no thought channel, no
    # budget-eaten empty replies).
    switches = thinking_stats.get("request_switches") or {}
    first_tokens: list[float] = thinking_stats.get("first_token_s") or []
    print("NO-THINKING profile (Ollama /set nothink comparison):")
    print(f"  request switches      : chat_template_kwargs={switches.get('chat_template_kwargs')} "
          f"reasoning_effort={switches.get('reasoning_effort')!r}")
    print(f"  thinking blocks       : {thinking_stats.get('thinking_block_turns', 0)} turns "
          f"(must be 0)")
    print(f"  reasoning-eaten errors: {thinking_stats.get('reasoning_eaten_errors', 0)} turns "
          f"(must be 0)")
    print(f"  empty replies         : {thinking_stats.get('empty_replies', 0)} turns (must be 0)")
    if first_tokens:
        print(f"  first content token   : min {min(first_tokens):.2f} s / mean "
              f"{sum(first_tokens) / len(first_tokens):.2f} s / max {max(first_tokens):.2f} s "
              f"(no thought phase — Ollama nothink behaviour)")
    nothink_ok = (
        thinking_stats.get("thinking_block_turns", 0) == 0
        and thinking_stats.get("reasoning_eaten_errors", 0) == 0
        and thinking_stats.get("empty_replies", 0) == 0
        and switches.get("chat_template_kwargs") == {"enable_thinking": False}
        and switches.get("reasoning_effort") == "none"
    )
    print(f"  NO-THINKING VERDICT   : {'MATCHES the Ollama /set nothink behaviour' if nothink_ok else 'DEVIATES - check the reasoning configuration!'}")
    print("-" * 72)
    if failures:
        print("Failed turns (details):")
        for failure in failures:
            print(f"  {failure['conversation']} turn {failure['turn']}: "
                  f"{failure.get('checks') or failure.get('error')}")
            reply = failure.get("reply", "")
            if reply:
                print(f"    reply: {reply[:160]}{'...' if len(reply) > 160 else ''}")
    print("=" * 72)
    verdict = "PASS" if rate >= 0.8 else "FAIL"
    print(f"RESULT: {verdict} (exit criterion: >= 80% of turns pass — milestones 7.6.3)")
    if not nothink_ok:
        print("RESULT: FAIL (no-thinking profile deviates from the required "
              "Ollama /set nothink behaviour)")
        return 1
    return 0 if rate >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
