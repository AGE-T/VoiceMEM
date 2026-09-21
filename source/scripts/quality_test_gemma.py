#!/usr/bin/env python3
"""scripts/quality_test_gemma.py - deterministic Gemma 4 12B quality suite.

v0.4.3 milestone contract, Section 6 (GEMMA QUALITY TEST). Runs on the
TARGET MACHINE against the local llama-server (127.0.0.1:8080 - LOCAL
ONLY, zero cloud calls).

Hungarian tests:
  1. normal conversation
  2. instruction following (word-count constraint)
  3. grammar correction
  4. explanation
  5. teaching
  6. memory related conversation
  7. multi-turn continuity (name recall across turns)
  8. natural conversational response

English tests:
  9. normal conversation
 10. instruction following (word-count constraint)
 11. grammar correction
 12. explanation
 13. teaching
 14. multi-turn continuity (name recall across turns)

Language switching (the reply language must follow the user's language,
detected with the EXISTING app.text_utils.detect_language logic - the
same detector the TTS voice routing uses; the architecture is unchanged):
 15. Hungarian conversation        -> detect_language(reply) == 'hu'
 16. switch to English             -> detect_language(reply) == 'en'
 17. switch back to Hungarian      -> detect_language(reply) == 'hu'

Determinism: temperature 0 (greedy). Every check verifies non-empty
reply + expected language + a concrete per-test expectation; the report
names the exact failing test - nothing is masked, nothing is greened by
weakening.

Usage (target machine, repo root; a healthy Gemma server must be
running - scripts/smoke_test_gemma.py --keep-server leaves one up):
    .venv\\Scripts\\python.exe scripts\\quality_test_gemma.py --reuse

Exit codes: 0 = all quality checks PASS; 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import AgentConfig  # noqa: E402
from app.text_utils import detect_language  # noqa: E402
from scripts.smoke_test_gemma import (  # noqa: E402
    GEMMA_DISPLAY,
    http_json,
    safe_print,
)

RESULT_JSON = "logs/gemma_quality_result.json"
CHAT_TIMEOUT_S = 180.0


def word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip(",.!?;:")])


class GemmaQualityTest:
    def __init__(self, cfg: AgentConfig, reuse_only: bool) -> None:
        self.cfg = cfg
        self.reuse_only = reuse_only
        self.results: list[dict[str, Any]] = []
        self.server_base = f"http://{cfg.llama_server_host}:{cfg.llama_server_port}"
        self.model_id = ""

    def record(self, num: int, name: str, ok: bool, detail: str,
               fix: str = "") -> bool:
        icon = "[PASS]" if ok else "[FAIL]"
        safe_print(f"{icon} {num:2d}. {name} - {detail}"
                   + (f"\n       FIX: {fix}" if (fix and not ok) else ""))
        self.results.append({"num": num, "name": name,
                             "status": "PASS" if ok else "FAIL",
                             "detail": detail, "fix": fix})
        return ok

    def chat(self, messages: list[dict], max_tokens: int = 128) -> str:
        payload = {
            "model": self.cfg.llm_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,   # deterministic (greedy) quality suite
            "stream": False,
        }
        body = http_json(f"{self.server_base}/v1/chat/completions",
                         payload, timeout=CHAT_TIMEOUT_S)
        try:
            content = body["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else ""
        except (KeyError, IndexError, TypeError):
            return ""

    def server_ok(self) -> bool:
        body = http_json(f"{self.server_base}/health", timeout=5.0)
        if body is None:
            return False
        models = http_json(f"{self.server_base}/v1/models", timeout=10.0)
        try:
            self.model_id = str(models["data"][0]["id"])
        except (KeyError, IndexError, TypeError):
            self.model_id = ""
        return "gemma" in self.model_id.lower()

    # -- individual checks --------------------------------------------------

    def check_reply(self, num: int, name: str, reply: str,
                    expect_lang: Optional[str] = None,
                    min_words: int = 3,
                    extra_ok: Optional[bool] = None,
                    extra_detail: str = "") -> bool:
        ok = bool(reply.strip()) and word_count(reply) >= min_words
        detail = f"reply: {reply.strip()[:70]!r}"
        if expect_lang is not None:
            lang = detect_language(reply)
            ok = ok and (lang == expect_lang)
            detail += f" | language: {lang} (expected {expect_lang})"
        if extra_ok is not None:
            ok = ok and extra_ok
            if extra_detail:
                detail += f" | {extra_detail}"
        return self.record(num, name, ok, detail,
                           "the reply was empty, wrong language or failed the "
                           "check's concrete expectation")

    def run(self) -> int:
        safe_print("=" * 70)
        safe_print(f"VoiceMem Agent - Gemma quality suite ({GEMMA_DISPLAY})")
        safe_print(f"endpoint : {self.server_base} (LOCAL ONLY)")
        safe_print("mode     : deterministic (temperature 0)")
        safe_print("=" * 70)

        if not self.server_ok():
            safe_print("[FAIL] no healthy GEMMA llama-server on "
                       f"{self.server_base} (model id: {self.model_id!r})")
            if self.reuse_only:
                safe_print("       Run scripts/smoke_test_gemma.py --keep-server "
                           "first, or start scripts/start_llama_server.ps1.")
            return 1
        safe_print(f"[INFO] using server model id: {self.model_id!r}")

        hu_history: list[dict] = []
        en_history: list[dict] = []

        # ---- Hungarian suite ------------------------------------------------
        # 1. normal conversation
        r = self.chat([{"role": "user",
                        "content": "Szia! Hogy vagy ma?"}])
        self.check_reply(1, "HU normal conversation", r, expect_lang="hu")

        # 2. instruction following (word count constraint)
        r = self.chat([{"role": "user",
                        "content": "Pontosan egyetlen egy szóval válaszolj: "
                                   "mekkora a magyar főváros?"}], max_tokens=16)
        self.check_reply(2, "HU instruction following", r, expect_lang=None,
                         min_words=1, extra_ok=(1 <= word_count(r) <= 3),
                         extra_detail=f"word count: {word_count(r)} (must be 1-3)")

        # 3. grammar correction
        r = self.chat([{"role": "user",
                        "content": "Javítsd ki nyelvtanilag ezt a mondatot, és "
                                   "írd le helyesen: „tegnap elmentünk boltba és "
                                   "vettünk kenyér”"}])
        self.check_reply(3, "HU grammar correction", r, expect_lang="hu",
                         extra_ok=any(w in r.lower() for w in
                                      ("kenyeret", "a boltba", "boltba")),
                         extra_detail="must contain a corrected form "
                                     "(kenyeret / a boltba)")

        # 4. explanation
        r = self.chat([{"role": "user",
                        "content": "Magyarázd el röviden, magyarul: mi az "
                                   "udvarias kezdőmondat egy e-mailben?"}])
        self.check_reply(4, "HU explanation", r, expect_lang="hu", min_words=4)

        # 5. teaching
        r = self.chat([{"role": "user",
                        "content": "Tanítsd meg nekem, magyarul, két mondatban: "
                                   "hogyan kell udvariasan kérni valamit?"}])
        self.check_reply(5, "HU teaching", r, expect_lang="hu", min_words=5)

        # 6. memory related conversation
        r = self.chat([{"role": "user",
                        "content": "A nevem Thomas, és imádom a kávét. Jegyezd "
                                   "meg, kérlek! Mondd el, mit jegyeztél meg "
                                   "rólam."}])
        self.check_reply(6, "HU memory related conversation", r, expect_lang="hu",
                         extra_ok=("thomas" in r.lower() and "káv" in r.lower()),
                         extra_detail="must recall Thomas and the coffee")

        # 7. multi-turn continuity (name recall across turns)
        hu_history = [
            {"role": "user", "content": "Szia! A nevem Thomas, örülök!"},
        ]
        r1 = self.chat(hu_history)
        hu_history.append({"role": "assistant", "content": r1})
        hu_history.append({"role": "user", "content": "Mi a nevem?"})
        r2 = self.chat(hu_history)
        self.check_reply(7, "HU multi-turn continuity", r2, expect_lang="hu",
                         extra_ok=("thomas" in r2.lower()),
                         extra_detail="must recall the name Thomas from turn 1")

        # 8. natural conversational response
        r = self.chat([{"role": "user",
                        "content": "Köszönöm szépen a segítséget!"}])
        self.check_reply(8, "HU natural conversational response", r,
                         expect_lang="hu",
                         extra_ok=any(w in r.lower() for w in
                                      ("szívesen", "nincs", "semmi", "örülök",
                                       "boldog", "mindent")),
                         extra_detail="must be a natural polite reply")

        # ---- English suite ----------------------------------------------------
        # 9. normal conversation
        r = self.chat([{"role": "user", "content": "Hi! How are you today?"}])
        self.check_reply(9, "EN normal conversation", r, expect_lang="en")

        # 10. instruction following (word count constraint)
        r = self.chat([{"role": "user",
                        "content": "Answer with exactly one word only: what is "
                                   "the capital of France?"}], max_tokens=16)
        self.check_reply(10, "EN instruction following", r, expect_lang=None,
                         min_words=1, extra_ok=(1 <= word_count(r) <= 3),
                         extra_detail=f"word count: {word_count(r)} (must be 1-3)")

        # 11. grammar correction
        r = self.chat([{"role": "user",
                        "content": "Correct this sentence grammatically and "
                                   "write it correctly: “Yesterday we go to "
                                   "the store and buy bread”"}])
        self.check_reply(11, "EN grammar correction", r, expect_lang="en",
                         extra_ok=("went" in r.lower()),
                         extra_detail="must use the past tense 'went'")

        # 12. explanation
        r = self.chat([{"role": "user",
                        "content": "Explain briefly: what is a polite opening "
                                   "line for an email?"}])
        self.check_reply(12, "EN explanation", r, expect_lang="en", min_words=4)

        # 13. teaching
        r = self.chat([{"role": "user",
                        "content": "Teach me in two sentences: how do I ask "
                                   "for something politely?"}])
        self.check_reply(13, "EN teaching", r, expect_lang="en", min_words=5)

        # 14. multi-turn continuity (name recall across turns)
        en_history = [{"role": "user", "content": "Hi! My name is Thomas, nice "
                                                  "to meet you!"}]
        r1 = self.chat(en_history)
        en_history.append({"role": "assistant", "content": r1})
        en_history.append({"role": "user", "content": "What is my name?"})
        r2 = self.chat(en_history)
        self.check_reply(14, "EN multi-turn continuity", r2, expect_lang="en",
                         extra_ok=("thomas" in r2.lower()),
                         extra_detail="must recall the name Thomas from turn 1")

        # ---- language switching ------------------------------------------------
        switch_history: list[dict] = [
            {"role": "user",
             "content": "Beszélgessünk magyarul egy kicsit. Hogy telt a napod?"},
        ]
        r_hu = self.chat(switch_history)
        self.check_reply(15, "language switch: Hungarian conversation", r_hu,
                         expect_lang="hu")
        switch_history.append({"role": "assistant", "content": r_hu})

        switch_history.append({"role": "user",
                               "content": "Let us switch to English now. "
                                          "What did we just talk about?"})
        r_en = self.chat(switch_history)
        self.check_reply(16, "switch to English", r_en, expect_lang="en")
        switch_history.append({"role": "assistant", "content": r_en})

        switch_history.append({"role": "user",
                               "content": "Most visszaváltunk magyarra. "
                                          "Kérlek, folytassuk magyarul!"})
        r_back = self.chat(switch_history)
        self.check_reply(17, "switch back to Hungarian", r_back,
                         expect_lang="hu")

        # ---- summary ---------------------------------------------------------
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        passed = len(self.results) - failed
        safe_print("")
        safe_print("=" * 70)
        safe_print(f"RESULT: {passed}/{len(self.results)} quality checks PASS"
                   + ("" if failed == 0 else f" ({failed} FAIL)")
                   + f" - LLM: {GEMMA_DISPLAY}")
        result = {
            "schema_version": 1,
            "llm": GEMMA_DISPLAY,
            "server_model_id": self.model_id,
            "checks": self.results,
            "passed": passed,
            "failed": failed,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            out_path = REPO_ROOT / RESULT_JSON
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
            safe_print(f"results written: {RESULT_JSON}")
        except OSError as exc:
            safe_print(f"[WARN] could not write {RESULT_JSON}: {exc}")
        return 1 if failed else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic Gemma 4 12B quality suite (HU/EN/switching).")
    parser.add_argument("--reuse", action="store_true",
                        help="require an already-running healthy Gemma server")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "voicemem_config.yaml"),
                        help="AgentConfig YAML path")
    args = parser.parse_args(argv)

    cfg = AgentConfig.from_yaml(Path(args.config))
    test = GemmaQualityTest(cfg, reuse_only=args.reuse)
    return test.run()


if __name__ == "__main__":
    sys.exit(main())
