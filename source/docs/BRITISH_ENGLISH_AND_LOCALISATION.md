# British English and Localisation Contract

Status: Active engineering requirement — audit executed 2026-09-18, deterministic regression scanner implemented
Scope: VoiceMemAgent product-owned source, UI labels, logs, prompts, documentation and test fixtures

## 1. Goal

All product-facing static terminology shall be consistently written in British English.

Asian-script labels must not appear in product UI, logs, prompts, runtime labels or product-owned documentation.

This is a localisation requirement, not an instruction to alter arbitrary user utterances or model/input data.

## 2. Canonical Memory Labels

| Existing (compat) label | Canonical English presentation label |
|---|---|
| `喜好与厌恶` | `likes and dislikes` |
| `表达风格` | `expression style` |
| `思维模式` | `thinking style` |
| `应对方式` | `coping style` |
| `情绪` | `emotion` |

The canonical labels must be used consistently in UI, logging and product-owned runtime formatting. The compat column values are internal identifiers (see §4) and must not be deleted.

## 3. Audit Result (2026-09-18)

Deterministic scan: 570 files scanned, 128 files with hits, 11,576 Asian-script occurrence runs (77,682 matched characters).

| Category | Files | Hits | Verdict |
|---|---|---|---|
| A — product-owned runtime/UI text | 0 | 0 | CLEAN. No Asian-script string reaches the user, TTS, or the app-side LLM prompt today. |
| B — product-owned non-runtime | 25 | 102 | 100 are vendor-format fixtures in tests/scripts (legitimate); 2 are Chinese comments in product code, queued for translation (app/web_server.py:537, app/emotion.py:229). |
| C — vendor/upstream/model data | 92 | 11,332 | preserved (vendored library prose, prompts, model cards). |
| D — historical forensic evidence | 8 | 110 | preserved (audit captures, verification evidence, CHANGELOG record). |
| E — runtime user/model-generated data | 1 | 5 | preserved (memory/memory.json vendor-written descriptors). |
| F — internal identifiers / compat values | 4 | 27 | preserved by design (§4). |

Five-label reachability verdict: localised at both UI payload sites (`_SLOT_EN` + `localise_slot()` — the mind-map tree and the memory-hits payload); not spoken (only LLM reply text enters TTS); not in the app-side LLM prompt (`clean_rb_content()` strips vendor Chinese prefixes/suffixes, `trait_prompt_suffix()` renders English provenance only). The browser UI contains none of the five labels.

## 4. Category F — the Compatibility Identifier Set (Must Stay Verbatim)

1. The five vendor trait-slot enum keys — 情绪 / 应对方式 / 表达风格 / 思维模式 / 喜好与厌恶 — persistence-stable keys of `TraitStore` (validated on write; `_SLOT_EN` maps them to the canonical English labels).
2. The eight bilingual emotion2vec tokens: 生气/angry, 厌恶/disgusted, 恐惧/fearful, 开心/happy, 中立/neutral, 其他/other, 难过/sad, 吃惊/surprised (model tokenizer compatibility).
3. Advisory-suffix strip markers: 下次, 内心OS, and the fullwidth （ ） ： literals in the three strip-regexes (app/web_server.py, app/voicemem_bridge.py, web/voicemem.html).
4. Vendor-prompt detection markers in scripts/tests (记忆清洁助手, 用户说了这句话, annotation forms, the ASR loop marker) — fixtures replicating vendor formats.

Changing any of these would break persistence, model tokenisation, or vendor-format matching. They are keys, not display text.

## 5. Residual Leak Channels (Advisory — Not Category A Today, Hardening Queued)

1. The `memory_hits` payload `raw` field ships the prefixed vendor original (Chinese prefix included) to the browser — debug-only, never rendered (used only by `bareNote()` matching). Hardening: strip server-side or mark debug-only.
2. `localise_slot()` passes unknown slots through raw — a vendor enum extension would leak untranslated. Hardening: warn and fall back to a stable English rendering for unmapped non-ASCII values.
3. Vendor stdout logging and vendor-internal LLM prompts remain Chinese (category C) — vendor-owned; product-side wrappers must stay pure English.

## 6. British English Rules

Use British spelling and terminology throughout product-owned text: behaviour, organisation, analyse, optimise, modelling, centre, licence (noun), favourite. Do not rewrite technical identifiers, package names, API names, model names or upstream licence titles merely to force British spelling.

## 7. Implemented Validation (2026-09-18)

- `tools/asian_script_scan.py` — the deterministic scanner + CLI (exit 1 on violation; `--json` report; `--update-baseline` refreshes the audited counts).
- `tools/asian_script_baseline.json` — the committed audit baseline (baseline-zone file → occurrence count).
- `tests/unit/test_asian_script_guard.py` — the gate test: hard-fail zone clean (literal allowlist + per-file caps), the canonical five English labels pinned present, the baseline not exceeded, scanner self-test.
- Zone policy: HARD-FAIL `app/`, `web/`, `config/` and the root product files; BASELINE `tests/`, `scripts/`, `docs/`, `tools/` (new occurrences fail against the committed baseline); REPORT-ONLY `memory/`, `data/` (runtime data, any language is legitimate); EXEMPT `vendor/`, `models/`, `logs/`, `audit/`, `releases/`, binaries, >2 MB tokenizer artifacts, `CHANGELOG.md` (append-only historical record).

## 8. Regression Requirements

The localisation rules must not alter: memory slot semantics; retrieval semantics; trait identifiers used as stable internal keys (unless a compatibility migration is explicitly implemented); VoiceMem persistence formats; LLM/ASR/TTS behaviour; user-generated memory content; evidence needed for forensic reproduction.

Where stable internal keys use the compat labels, preserve compatibility internally and expose only the English presentation label. Do not silently break historical data.

Queued for the implementation phase (small, isolated, full-gate rule): the two category-B comment translations (app/web_server.py:537, app/emotion.py:229) and the two §5 hardening items.
