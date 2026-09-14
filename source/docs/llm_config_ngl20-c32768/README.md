# PRODUKCIÓS LLM-KONFIGURÁCIÓ — ngl20 + 32K (mért döntés)

Ez a mappa a production LLM-profil konfigurációs döntésének teljes
bizonyítékát tartalmazza, a v0.6.3 kiadástól kezdve **a release ZIP-ben
utazik** (korábban külön `VoiceMemAgent_v0.6.0_LLMConfig_ngl20-c32768.zip`
csomagban volt — az all-in-one v0.6.3 kiadással ez a külön csomag
kivezetésre került).

## Proveniencia

- A konfiguráció eredetileg a v0.6.0 kiadásra készült külön patch-csomagként
  (`identity.json`: baseline `34c5df9`, change `dd9fccd`), a mért
  döntéssel: **ngl 26 → 20**, **context 8192 → 32768**.
- A sandbox-rollback után a konfigurációt újraalkalmazták a visszaállított
  v0.6.2 forrásbázison (a merge során a v0.6.1/v0.6.2 javítások MEGMARADTAK,
  a konfigurációs értékek felülíródtak).
- **v0.6.3 ALL-IN-ONE**: a konfiguráció build-időben be van sütve a ZIP-be —
  a korábbi kétlépéses field-sorrend (ZIP telepítése + külön LLMConfig-csomag
  alkalmazása) elavult, egyetlen kicsomagolás + `START.bat` a production
  profilt adja.

## Tartalom

- `identity.json` — a változás összefoglalója, mérési alap, sandbox-verifikáció
- `RECOVERY.md` — mért táblázat, új production parancssor, field-ellenőrző lista
- `evidence/` — sandbox-bizonyítékok: starter python-bridge feloldás
  (`bridge_proof.txt`), élő llama.cpp b10717 szerverbizonyíték a production
  parancssorral (`server-proof.log`, `server-log-startup-config.txt`,
  `server-log-wire-dump.txt`), a valódi `app/llm.py` LlmClient élő kör
  (`llm_client_proof.txt/.py`), célzott teszteredmények
  (`targeted_tests.txt`), és a pontosan-egy-llama-server process-check
  (`process_check.txt`).

## Mért döntés

| profil | TTFT | teljes |
|---|---|---|
| ngl20 + 8192  | ~4,90 s | ~7,53 s |
| ngl20 + 32768 | ~4,89 s | ~7,82 s |

A 32K-ablak mért késleltetési költsége ~ nulla, miközben 8K feletti
production kérés már előfordult (kontextus-túlfutás). Production cél
ezért **32768, nem 8192**.
