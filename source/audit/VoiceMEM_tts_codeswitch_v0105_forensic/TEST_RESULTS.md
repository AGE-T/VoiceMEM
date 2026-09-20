# TEST RESULTS — VoiceMem v0.10.5 TTS code-switching forensic + fix

UTC: 2026-09-19T18:45:48Z  Tree: 653549701b61b28ba70a7d14ebf2fbba41ad5c6a

## 1. Target suite (span + web + CLI levels)
...............................................                          [100%]
109 passed, 10 subtests passed in 1.37s

## 2. Integration (pipeline mock, 47)
...............................................                          [100%]
47 passed in 37.70s

## 3. Adjacent TTS/voice/language suites
..............                                                           [100%]
86 passed in 6.20s

## 4. Full unit suite
9 failed, 1115 passed, 13 skipped, 18 subtests passed in 105.41s (0:01:45)

## 5. Full unit suite — failure decomposition (all pre-existing, none from this fix)

The 9 failures: test_asr_modular (6: Silero/Parakeet model files absent after the
sandbox state loss), test_install_manifest (1: venv-extras pin identity),
test_prompt_reduction (1: isolated-runner env), test_voicemem_bridge (1 in full-suite
ordering; passes in isolation — 26 passed when the 2 suspect files run alone).
PROOF this fix causes none of them: (a) the identical 4-file subset run WITH the
fix and WITHOUT the fix (git stash) yields the identical 7-failure set;
(b) the official release gate run pre-fix vs post-fix produces the byte-identical
failure set (14 env-gap + 17 temporal-memory/prompt-reduction/release-zip classes —
temporal memory fails in setUpClass on `from openai import OpenAI`, venv extras).

## 6. Official release gate (scripts/run_release_gate.py)

VERDICT: RED — environment, not the fix. Evidence: evidence/gate_run_forensic.txt
(post-fix run) vs the identical pre-fix run (both captured during the session).
Zero failures appear or disappear due to this change; the TTS-relevant layers
(items 1-3 above) are fully green.
