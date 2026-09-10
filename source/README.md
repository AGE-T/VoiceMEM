# VoiceMem M1/M2/M3 — teljesen lokális magyar/angol hangügynök hosszú távú memóriával, érzelmi intelligenciával és beszélő-azonosítással

A VoiceMem M1 a V1-terv első mérföldköve: egy **teljesen lokális, offline, nulla
fizetős komponensű** kétnyelvű (magyar + angol) hangügynök, amely a mikrofontól a
hangszóróig egyetlen gépen fut:

```
mikrofon → Silero VAD → Qwen3-ASR → [ECAPA beszélő-azonosítás || emotion2vec+ érzelmi elemzés]
        → VoiceMem memória (beszélőnkénti tér) → llama-server LLM → Piper TTS → hangszóró
```

Az ügynök **tanár-personaként** viselkedik: ha a felhasználó angol nyelvtani hibát
ejt, kijavítja és magyarul elmagyarázza; a beszélgetés nyelvét az aktuális
megszólalás domináns nyelvéhez igazítja. A **hosszú távú memória**
(VoiceMem + multilingual-e5-small + lokális Qdrant) a korábbi hibákat és tényeket
visszaidézi ("ugyanazt a hibát csinálom még mindig?"). Támogatott a **barge-in**:
a felhasználó 500 ms folyamatos beszéddel megszakíthatja a folyó választ.

**M0-elv — önálló, egygyökerű alkalmazás:** a repó klónozás → telepítés →
modellletöltés → smoke test → futtatás egyetlen egység, minden a repo gyökerén
belül. A célgépen bárhol klónozható (példanév: `C:\VoiceMemAgent`); a modellek a
repo `models/` mappájában, a hosszú távú memória a `memory/`-ben, a HuggingFace
cache az izolált `models/hf/`-ben él, és minden Python-hívás a repo
`.venv\Scripts\python.exe`-jén át megy — a futtatás nem függ a gép globális
Python-csomagjaitól vagy felhasználói cache-ektől.

Célhardver: **NVIDIA RTX 5070 12 GB (Blackwell, sm_120), 32 GB RAM, Windows 11**.
Steady VRAM-igény (v0.4.14+, Qwen3.6 35B A3B IQ4_XS ~19 GB MoE, LLAMA_N_GPU_LAYERS=26 reszleges offload: figyelem+KV+compute GPU-n, a tobbi rendszer-RAM-bol; a TENYLEGES merest a scripts\measure_vram.ps1 vegzi), latenciacél: **p50 < 2,0 s,
p95 < 2,5 s** (beszéd végétől az első hangig). A fejlesztés GPU nélküli Linux
sandboxban történt; minden nehéz függőség guardolt import — a mock mód és a
teljes tesztkészlet GPU és hang nélkül is fut.

**v0.4.4 mező-javítás (field report: "a szöveges módra sem válaszol; ASR nem zöld") — három gyökérok, mindhárom fix:**

1. **LLM: 0 karakteres válaszok (thinking-csatorna).** A llama.cpp b10717
   chat-kezelője az `enable_thinking` defaultját TRUE-n hagyja; a hibrid-reasoning modell
   chat-template ekkor NYITVA hagyja a `<|channel>thought` gondolkodási
   csatornát a generálási promptban, a model a csatornán BELÜL válaszol, a
   szerver azt `reasoning_content`-be irányítja — a látható `content` üres
   marad, a 512 tokenos keretet a gondolkodás fogyasztja el (a
   valóságban élőben reprodukálva a b10717 forrásból épített szerverrel +
   valódi modellel). Fix három rétegben: (a) minden LlmClient-kérés
   request-szintű thinking-kikapcsolót visz (`chat_template_kwargs` +
   `reasoning_effort: none`); (b) a `start_llama_server.ps1` a szervert
   `--reasoning off` zászlóval indítja — ez a voicemem csomag SAJÁT
   OpenAI-lib hívásait is védi (azok nem tudnak request-szintű kwargs-ot
   küldeni); (c) a web backend induláskor EGYSZER VALÓBAN végigpróbál egy
   beszélgetési fordulót (`_verify_llm_reply`) — a "/health 200, de üres
   content" állapot mostantól LÁTHATÓ LLM ERROR, soha többé nem zöld csip.
2. **ASR: transformers pin-ütközés.** A vendored voicemem csomag
   `transformers==4.52.3`-at pinel — az a verzio NEM ismeri a `qwen3_asr`
   architekturát (v0.4.7: a nativ qwen3_asr modulhoz >= 5.0 kell), igy az
   ASR soha nem töltodött be ("ASR nem
   zöld"). Fix: a requirements/pyproject also padlo 5.0, es a telepito uj
   16. lepeseben egy OR (a funasr/speechbrain utan) visszairanyitja es
   verzio-asserttel lezarja.
3. **Memória/E5: offline feloldás.** A voicemem E5-feloldoje CSAK flat
   layout-ban eszi meg a lokalis `models\embedding\`-ot; a mi
   almappás (`multilingual-e5-small`) elrendezesunkre HF repo id-re esett,
   offline runtime pedig "could not connect to huggingface.co" hibával
   bukott minden memoria-kereses. Fix: a `VOICEMEM_E5_MODEL` env a pontos
   lokalis utvonalra mutat (env.local.ps1 + app-oldali pin a mindkét
   konstrukciós úton).

**v0.4.5 mező-javítás (field report: "ASR-rel még mindig baj van — csak szöveges beszélgetést enged"):**

A friss naplók (llama-server.starter / web-server / llama-server.err /
bootstrap) szerint a v0.4.4 javítások ÉLTEK a gépen: a llama-server a
a regi modelt töltötte `--reasoning off`-fal (/health 200), az induló
válasz-ellenőrzés PASS, a szöveges fordulók 3-4 s alatt válaszoltak TTS
hanggal, az E5 memória-keresések offline már rendben — DE az ASR továbbra
sem töltődött be, mert a venv-ben MEGMARADT a transformers 4.52.3 (a PyPI
voicemem csomag pinje), ami elutasítja a `qwen3_asr` architektúrát.

1. **Gyökérok: a bootstrap "dependencies importable" probe-ja csak a
   IMPORTÁLHATÓSÁGOT nézte, nem a VERZIÓT.** Egy helyben frissített venvben
   a 4.52.3 importálható volt → a bootstrap "environment already ready"
   üzenettel KIHAGYTA az idempotens telepítőt → a v0.4.4-es 16. lépés (a
   transformers>=5.0 OR) SOHA nem futott le → az ASR örökre piros maradt,
   miközben a szöveges mód működött. **Fix:** a probe most VERZIÓ-ÉRZÉKENY
   (egy idézőjel-mentes Python probe 5-ös kilépési kóddal jelzi, ha a
   transformers importálható, de régebbi, mint 5.0) — a verzóhiba egyenesen
   a telepítőhöz irányít, amelynek 16. lépése frissíti a transformersist, és
   a telepítés UTÁNI újra-probe is verziót ellenőriz: **EGYETLEN START.bat
   dupla kattintás megjavítja az ASR-t** (nem kell repair mód, nem kell
   kézi pip). A bootstrap kiírja a talált verziót és a Qwen3-ASR
   következményt; az ASR betöltési hiba szövege (a Pipeline panelen) is
   pontosan ezt az egy-kattintásos javítási utat írja.
2. **Kontextus-túlfutás (llama-server.err.log):** egy hosszú beszélgetés
   egyszer 10459 tokenes promptot küldött, amit a fix 8192 tokenes
   kontextus elutasított ("request exceeds the available context size") —
   az a forduló válasz nélkül halt el. **Fix:** prompt-költségvetés a web
   láncban — a memória-blokk maximum 6000 karakter (sorthatáron vágva,
   látható csonkolási jegyzettel), majd a LEGRÉGEBBI history-bejegyzések
   esnek ki, amíg az egész prompt be nem fér egy konzervatív 14000
   karakteres keretbe; az aktuális üzenet és a legfrissebb history mindig
   megmarad.
3. **Kozmetika:** a telepito banner most helyesen "22 lepes" (a "18 lepes"
   szöveg v0.4.4 óta elavult volt).

## A helyi webes felület (VoiceMem Web UI, v0.4.0)

A normál felhasználói út mostantól a **meglévő VoiceMem webes felület** a
böngészőben (`START.bat` -> LOCAL web backend, alapértelmezett port
**http://127.0.0.1:8787**). A böngésző SOHA nem hív külső (OpenAI) végpontot —
minden következtetés a lokális llama-serveren (127.0.0.1:8080) fut:

```
böngésző → helyi web backend (app/web_server.py :8787) → agent pipeline → llama-server :8080
```

A `START.bat` folyamata: (1) telepítés ellenőrzése, (2) llama-server indítása
ha nem fut, (3) a lokális web backend indítása, (4) a böngésző automatikus
megnyitása a VoiceMem UI-ra. A konzol kiírja a pontos URL-t. Nincs kézi
szerverindítás.

Pipeline (LOCAL mód):

```
mikrofon (24 kHz PCM a WS-en) → Silero VAD → Qwen3-ASR 0.6B → VoiceMem
  (multilingual E5 small CPU embedding) → Qwen3.6 35B A3B IQ4_XS (llama-server SSE)
  → Piper HU/EN → 24 kHz PCM vissza a böngészőnek
```

Az UI adottságai:

- **Mikrofonválasztó**: eszközlista frissítéssel, kiválasztott eszköz nevének
  kijelzésével, bemeneti szintjelzővel és külön "Start/Stop mic test"
  gombbal (a teszt nem megy fel a WS-re). Nem kényszerít egy adott mikrofont.
- **ASR teszt gomb** (v0.4.8): az "ASR test" felvesz pár másodperc hangot
  (ugyanazon eszköz/downsample/PCM16 úton, mint az élő uplink), majd el.POSTolja
  a `/api/asr-test` végpontnak — a szerver a VAD/WS/lánc állapotától FÜGGETLENÜL
  futtatja: szintstatisztika, friss Silero VAD-menet ("firingelt volna a
  kapu?"), majd a valódi Qwen3-ASR leiratás. **Az átirat kiíródik** az
  eredménypanelre ÉS a web-server.log-ba, a felvétel visszahallgatható
  (`logs/asr_test_last.wav`), és a verdikt-vezér megnevezi a törött láncszemet:
  `silent_mic` (a böngésző nullákat küldött — Windows mikrofon-adatvédelem
  /némítás/rossz eszköz; NEM modellhiba), VAD nem firingelne (a pontos
  `vad_threshold` értékkel, amit érdemes lejjebb venni), `asr_empty` /
  `asr_error`, vagy `ok`. A "beszélek, de semmi sem történik" hibakép így
  egyetlen gombnyomásból lokalizálható. **v0.4.12**: a verdikt okosabb lett —
  ha VAN szint (rms > 0,02) de a Silero süket (peak < 0,05), az NEM küszöb-
  probléma: a panel a régi (cache-elt) UI-oldalt és a Chrome
  hangfeldolgozását nevezi meg elsődleges gyanúsként ("Raw" kapcsoló +
  Ctrl+F5 újratöltés), nem küszöbcsökkentést javasol. A panel új
  **"Send to the agent" gombja** a hallott átiratot valódi `user_text` turnként
  küldi el a WS-úton — így a beszéd→LLM lánc akkor is végigmegy, amíg a
  VAD/capture-debug folyik.
- **Raw capture kapcsoló** (v0.4.12): a mikrofonlista melletti "Raw"
  jelölőnégyzet a Chrome MINDEN hangfeldolgozását kikapcsolja (AEC + AGC +
  NS — `echoCancellation:false, noiseSuppression:false,
  autoGainControl:false`), localStorage-ban perzisztál, és mindhárom capture-
  útra hat (mic test, ASR teszt, élő uplink). Tiszta line-in / pro láncokhoz
  (ahol a fizikai út maga tiszta hangot rögzít, és minden böngésző-oldali
  feldolgozás csak ront); laptopos mikrofonokhoz marad a default (AEC+AGC
  be, NS ki). **UI-verzió-ellenőrzés** (v0.4.12): az oldal a saját
  `PAGE_VERSION`-jét az induláskor a `/api/health` válaszával veti össze —
  eltérés esetén (pl. a frissítés után nyitva maradt fül a RÉGI JS-t
  futtatja) narancs figyelmeztető sor + toast jelenik meg a mikrofonsáv
  alatt ("reload with Ctrl+F5"), és az oldal `/api/lang`-on jelenti a saját
  verzióját, amit a backend a web-server.log-ba ír (`web UI page version X
  (backend Y) — STALE PAGE, reload needed`) — a terepi logok így mindig
  megnevezik, melyik oldal-build futott ténylegesen a böngészőben.
- **Automatikus mikrofon → agent átmenet + energia-tartalék VAD-kapu**
  (v0.4.14): a v0.4.13-as terepi jelentés ("beszélek, de csak a "Send to the
  agent" gombra jön válasz") kiváltó oka az volt, hogy az élő mikrofon-útvonal
  az ASR-t a Silero VAD MÖGÖTT kapuzza — és a Silero a jelentéskészítő
  line-in csatornáján súlyosan süket volt (valós, ASR által tökéletesen
  értett beszéd 0,003 valószínűséggel, 0/190 frame a 0,25 küszöb felett):
  `speech_start` sosem jött, így nem volt ASR-feed, flush, átirat és
  `_start_turn` sem — a gomb azért működött, mert a `user_text` a VAD-ot
  teljesen megkerüli. A javítás a meglévő architektúrán belül készült
  (`app/vad.py` **FusedVad**): a valódi módban a Silero ~10 s-os csúcs-
  valószínűségét figyelő "süketség-ablak" mögé energia-tartalék kapu kerül —
  amíg a Silero süket, egy két-időállandós zajpadkövető (gyors le, lassú fel
  — az állandó zajt önmagában limitálja) + abszolút pad beszédszintű jelet
  enged a state machine-be; amint a Silero életre kel, a tartalék passzív
  marad (egészséges csatornán semmi nem változik). A barge-in a Silero SAJÁT
  valószínűségét olvassa (`last_primary`), így az energia-kapu sosem
  szakíthatja meg magát a választ. Kapcsoló: `vad_energy_fallback`
  (default be; env: `VAD_ENERGY_FALLBACK=0` a kikapcsoláshoz). **Teljes mic
  stage-trail** a Pipeline events panelben (a gyűrű 48, a panel 24 sort mutat):
  mic frame received → speech start (kapu neve: "energy fallback") → ASR feed
  → speech end → ASR flush start → ASR flush done → ASR final transcript →
  **ASR turn dispatch** → turn received → …llm/tts… → answer done — ha az
  "ASR final transcript" megjelenik de az "ASR turn dispatch" nem, a törés
  pontja azonnal megnevezhető. Az ASR-teszt verdiktje is frissült: a
  gate-ellenőrzés a FÚZIONÁLT VAD-et futtatja (amit az élő lánc is használ),
  a `silero` csúcsot és a `gate`-et külön sorban nevezi meg, és a süket
  esetet úgy írja le, hogy "az energia-tartalék kapu most kezelni fogja — a
  beszéd automatikusan turn-t indít".
- **Qwen3.6 35B A3B IQ4_XS aktív LLM-profil** (v0.4.14): a
  `config/env.local.ps1` LLM-MODELPROFIL blokkja a Qwen3.6 35B A3B IQ4_XS-t
  az EGYETLEN aktív modell (v0.4.16: NINCS visszaesési profil). A GGUF
  operátori fájl: `models\llm\qwen3.6-35b-a3b\` (v0.4.17: tetszőleges
  fájlnevű, érvényes GGUF — a mappa szkennelése magic-alapú) VAGY bármelyik
  meghajtó (web UI "LLM model" picker → `config/llm_model.json`; README a
  mappában). **v0.4.17 OLLAMA-BLOB**: a tesztelés alatt használt modellt az
  Ollama töltötte le — a store-ban `sha256-...` nevű, kiterjesztés nélküli
  content-addressed blobként áll. A llama.cpp a GGUF **magic**-et nézi, nem a
  fájlnevet, így a blob HELYBEN, másolás és átnevezés nélkül betölthető. Az
  azonosító: `scripts\identify_ollama_blob.ps1` (manifest → tag →
  model-layer digest → blob; GGUF-header-azonosság + független sha256
  digest-ellenőrzés; IQ4_XS = `general.file_type 30`; split-blob esetén
  MEGÁLL és elmagyarázza, miért nem tölthető be közvetlenül — NEM készít
  20 GB duplikátumot; `-Select` = ugyanaz a kijelölés, mint a web UI picker).
  **v0.4.15: a fájl helyét NE találgasd** — a
  `scripts\find_qwen_gguf.ps1` pontos-modell kereső megkeresi a tényleges
  fájlt (repo, Ollama-store, HF-cache, LM Studio, letöltések), validálja a
  GGUF fejlécet (modell-azonosság + kvantizáció: az IQ4_XS a
  `general.file_type 30`), és
  kiírja a pontos abszolút Windows-útvonalat; `-SetEnv` átírja csak a
  `LLAMA_MODEL_PATH` sort (`.bak` biztonsági mentéssel), `-CopyToRepo` a
  kanonikus helyre másol. Ha a fájl hiányzik: MEGÁLL — más modell vagy más
  kvantizáció NEM helyettesíti be (a hiányó aktív GGUF hangos LLM-hiba marad,
  nincs visszaesési profil — a hiányzó GGUF hangos LLM ERROR marad). Hardver-igény: 12 GB VRAM + 32 GB RAM →
  `LLAMA_N_GPU_LAYERS=26` részleges offload (a figyelem + KV cache GPU-n, a
  többi réteg RAM-ból streamel), kontextus 8192 marad, thinking csatorna KI
  (`LLM_DISABLE_THINKING=1` + llama-server `--reasoning off`). A
  rendszerprompt a sikeres Ollama-kísérletek természetes
  hang-asszisztens personája (rövid beszélt fordulók, természetes magyar,
  legutolsó üzenet nyelvének követése) — a v0.4.3-as nyelvtanári scaffold
  visszavonult.
- **Robosztusság a kódelemző riport után** (v0.4.9): mind a tíz jelzett hiba
  javítva — a memória-bekötési (commit) hibák immár NEM csendesek (újrapróbálás
  + háttér-várólista + látható státusz), az ASR-modell hívása szálbiztos lett,
  az LLM prompt-költségvetés (8192 tokenes kontextus) MOST MINDEN hívási úton
  védett (web + CLI + VoiceMem reply-hook), a beállított mikrofon kiesésekor
  a rendszer-default eszközre esik vissza, az E5 offline modell betöltése
  végponttól-végpontig ellenőrzött, a VoiceMem-facade cache limitált (LRU 10 +
  1 óra idle), a VAD hosszabb frame-eket puffereli (nem csonkítja), a WS
  disconnect minden session-taskot leállít (nem árválkodik LLM-stream), és a
  numerikus env-változók elírása név szerinti, érthető hibát ad (nem
  néma default-ot).
- **Szöveges tesztút**: a szövegmező ugyanazt a backend-pipeline-t hajtja
  (memória → LLM → Piper), az ASR-t kihagyva; a beszélgetésben narancs
  "Text input · ASR bypassed" jelvény jelzi.
- **Pipeline debug nézet**: VAD / ASR / Memory / Embedding / LLM / TTS
  soronként ready/processing/error állapot + mért időzítők (WS
  `pipeline_status` + `/api/pipeline`).
- **Chat nézet**: felhasználói átírás, asszisztens válasz, emotion címke,
  Top-K memória-visszahívás, Piper hanglejátszás.
- **Memory Space**: létrehozás, váltás, szankenkénti memória, memória-grafikon
  (agytérkép), export — az eredeti UI funkcionalitása változatlanul.
- M2 emotion (emotion2vec+ + szemantikus fúzió) bekapcsolva marad; M3
  beszélő-azonosítás opcionális, alapból KI; scenefelismerés véglegesen zárva.
- Demó mód (`--mock` / `VOICEMEM_WEB_MOCK=1`): minden komponens offline
  helyettesítővel fut (GPU nélkül is tesztelhető) — az UI DEMO jelvényt mutat.

Manuális smoke test (célgép): `scripts\smoke_test_web.py` — elindítja a
backendet, kinyitja a böngészőt, és végigvezet a magyar beszédes ellenőrzőlistán
(átírás, LLM-válasz, Piper lejátszás, memóriabejegyzés, Memory Space, emotion).

A konzol-agent (korábbi alapértelmezés) elérhető marad: `START.bat cli`.

## Gyorsindítás — egyetlen dupla kattintás (START.bat)

**A felhasználó számára egyetlen belépési pont létezik: a `START.bat`.**
Kicsomagolás (vagy klónozás) után a projekt gyökerében:

```
dupla kattintás:  START.bat
```

Ennyi. A `START.bat` fully automatikusan (a `scripts\bootstrap.ps1` belső
orchestrátoron keresztül):

1. ellenőrzi a Python 3.11-et (64-bit; tiszta hibaüzenet, ha nincs),
2. létrehozza a `.venv`-et (csak ha még nincs),
3. telepíti a függőségeket (`requirements.lock`) és a Hugging Face
   `huggingface_hub` **kódtárat** (Python API a modellletöltéshez — a
   könyvtár hiánya SOHA nem user-facing hiba, automatikusan települ; a
   Hugging Face CLI NEM runtime függőség: v0.1.6-ban a CLI deprecation-
   warningjának emojija CP1252 konzolon UnicodeEncodeError-t okozott, ezért
   a letöltés a Python API-ra váltott — `hf_hub_download` /
   `snapshot_download` a `scripts/download_models_hf.py`-ból),
4. letölti a hiányzó M1 modelleket és binárisokat (`MODELS.lock.json`
   szerint; a meglévőket soha nem tölti újra),
5. elvégzi az install verificationt és az M1 smoke testet,
6. elindítja az agentet — és elindul a beszélgetés.

**Idempotens**: a második/harmadik `START.bat`-indítás felismeri a kész
részeket (`.install_state.json`), kihagyja a letöltést és a telepítést, és
egyből az agentet indítja. Megszakadt telepítés is bármikor folytatható:
`START.bat` újra. **Automatikus javítás**: ha egy csomag vagy modell
hiányzik/sérült, a bootstrap maga javítja (nem kapunk soha "telepítsd
kézzel" útmutatást addig, amíg a rendszer maga meg tudja oldani — pl. hiányzó
Python vagy NVIDIA driver esetén viszont érthető, pontos hibaüzenetet kapunk).

Opcionális módok (konzolban a fájl neve után írva):

| Mód | Hatás |
|-----|-------|
| `START.bat` | éles üzem: környezet-ellenőrzés → llama-server → LOCAL web backend (8787) → böngésző automatikusan nyílik |
| `START.bat web` | ugyanaz, mint a fenti alapértelmezés (explicit alias) |
| `START.bat cli` | konzol-agent (a korábbi alapértelmezés) |
| `START.bat mock` | szkriptelt 3 fordulós demo (mikrofon és GPU-modellek nélkül) |
| `START.bat check` | teljes ellenőrzés: assetek + smoke + teljes tesztkészlet (PASS/FAIL) |
| `START.bat benchmark` | latencia-benchmark készlet (p50/p95 limittel) |
| `START.bat repair` | automatikus környezetjavítás + utána mock-verify |
| `START.bat build` | verziózott release ZIP (a tesztkapu ELŐTTE fut — l. alább) |

**v0.3.6 robusztusság (field report):** ha a `START.bat`-et váratlan első
argumentummal indítja el valami (fájl ráhúzása drag-and-droppal, egyéni
`.bat` fájltársítás, szóközt tartalmazó útvonal), a szkript NEM akad el a
„Known modes" menün: rövid `[NOTE]` figyelmeztetést ír és az alapértelmezett
**webes felület** módban folytatja — a dupla kattintás tehát mindig a web
UI-hoz vezet. A `START.bat` emellett Windows CRLF sorvégjelekkel kerül a
release ZIP-be (a builder minden staged `.bat`-ot normalizál — az LF-only
`.bat` a cmd.exe parserének ismert kockázata).

A `scripts\` mappában lévő PowerShell-szkriptek **belső implementációs
részletek** — a felhasználónak soha nem kell őket külön elindítania. A
hibaüzenetek rövidek és elsődlegesen intézkedhetők (`[ERROR]` + Detected +
Required + következő lépés); a technikai részletek a
`logs\bootstrap.log`-ba kerülnek, soha nem a fő kimenetre.

A projekt bárhová másolható (bármely meghajtó, bármely mappa — pl.
`C:\VoiceMemAgent` vagy `F:\Voicemem\BUILD\voicemem-agent`): minden útvonal
a `START.bat` saját helyéből számított projektgyökérből indul, semmi sincs
beégetve.

## Architektúra

```
 EGY FORDULÓ ADATFOLYAMA (mic -> hangszóró):

   mikrofon (16 kHz mono PCM)
        |
        |  32 ms-es frame-ek
        v
   +--------------------+
   | Silero VAD v6.2.1  |  CPU / ONNX — beszédvég: 300 ms csend (hangover)
   +--------------------+
        |  beszéd-frame-ek
        v
   +--------------------+
   | Qwen3-ASR-0.6B     |  GPU — 600 ms-os kvázi-streaming -> végleges transcript
   +--------------------+
        |
        v
   +--------------------+   +----------------------+  PÁRHUZAMOSAN (M2)
   | VoiceMem memória   |   | emotion2vec+ base    |
   +--------------------+   | (proszódia-elemzés,  |
        |                   | CPU / funasr)        |
        |  memória-kontextus +----------------------+
        v                    |  fused érzelem (0.6/0.4)
   +--------------------+    |
   | teacher persona    |<---+  HU/EN tanár-prompt + memória-blokk + ÉRZELEM-BLOKK
   +--------------------+   (frusztrált user: egyszerűbb válasz + lassabb TTS)
        |  OpenAI-stílusú messages
        v
   +--------------------+        +-------------------------------------------+
   | LlmClient (httpx,  | -----> | llama-server 127.0.0.1:8080 (külön ablak) |
   | SSE streaming)     |  SSE   | Qwen3.6 35B A3B IQ4_XS, GPU, |
   +--------------------+        | 8K ctx, q8_0 KV; JSON-mód: REQUEST-szintű |
        |  token-delták          | (response_format a /v1/chat/completions   |
        v                        |  kérésben → llama.cpp constrained         |
   +--------------------+        |  generation — v0.3.1, nincs server-flag) |
   | SentenceStream     |  mondatvágás + nyelvdetektálás (HU/EN) chunkonként
   +--------------------+        +-------------------------------------------+
        |  beszélhető chunkok (frusztrált usernél: --length_scale 1.1)
        v
   +--------------------+
   | Piper TTS          |  CPU / subprocess — HU: anna (vagy berta/imre), EN: lessac
   +--------------------+
        |
        v
   hangszóró (22.05 kHz mono)

 BARGE-IN VEZÉRLÉS (a válasz alatt): a VAD tovább fut; ha 500 ms-on át
 a speech-probabilitás >= 0.30, a BargeInDetector leállítja a folyó
 TTS-t, LLM-streamet és memória-keresést. A llama-server JSON-módja NEM
server-flag: a `response_format` mező (`json_object` / `json_schema`)
KÉRÉS-SZINTŰ hatás (v0.3.1) — pontosan ezt használja a `chat_json()` és
a VoiceMem `_llm_json()`. Az EchoGate eközben loopback
 gatinget végez: a TTS alatt a mikrofon nem megy az ASR-be (V1.5: SpeexDSP
 igazi AEC, spec szerint).
```

| Komponens | Modell / verzió | Eszköz | Feladat |
|-----------|------------------|--------|---------|
| VAD | Silero VAD v6.2.1 (ONNX) | CPU | beszéd kezdet/vég, hangover 300 ms, barge-in |
| ASR | Qwen/Qwen3-ASR-0.6B | GPU (PyTorch cu128) | beszéd → szöveg, 600 ms-os kötegek |
| Memória-embedding | intfloat/multilingual-e5-small (384 d) | CPU | VoiceMem lokális embedder |
| Memória | VoiceMem VEZERLT FORRAS (vendor\voicemem, fork @ e8384e0 = upstream v0.0.1; identitas: VOICEMEM_PIN.json; NINCS klónozás) + mem0 + Qdrant embedded | CPU | hosszú távú felhasználói memória |
| **M2 érzelem** | **emotion2vec/emotion2vec_plus_base (~90M, FP32)** | **CPU (funasr AutoModel)** | **proszódia-elemzés: 9 kategória → fused valence/arousal** |
| LLM | Qwen3.6 35B A3B IQ4_XS GGUF (v0.4.16: az EGYETLEN LLM, NINCS fallback profil; operator-altal elhelyezett/barhol kivalaszthato GGUF; thinking CSAK kikapcsolva) | GPU (llama.cpp, 8K ctx, q8_0 KV, --reasoning off, ngl 26) | válaszgenerálás, JSON mód |
| TTS | Piper — hu_HU-anna/berta/imre, en_US-lessac (medium) | CPU (subprocess) | szöveg → beszéd (M2: --length_scale) |
| Hang I/O | sounddevice (PortAudio) | CPU | mikrofon + hangszóró |

## Követelmények

A telepítés ELŐTT a célgépen legyen meg:

- **Operációs rendszer**: Windows 11 (a PS1 indító- és telepítőszkriptek ehhez
  készültek; a Python-kód és a tesztek platformfüggetlenek).
- **Hardver**: NVIDIA RTX 5070 12 GB (Blackwell, sm_120), 32 GB RAM, NVMe SSD.
  A telepítő GPU smoke testet futtat (valódi mátrixművelet) — **compute
  capability < 12.0 esetén a telepítés sikertelen** (CPU-támogatás nincs).
- **Python 3.11 telepítve** (3.11.9 ajánlott, python.org) — **a telepítő NEM
  telepít Pythont**, csak használja: pontosan 3.11.x-et keres (`py -3.11`,
  majd `python3.11`, `python` fallback).
- **git** — a VoiceMem vendor-klónhoz (a telepítő 12. lépése); nélküle a
  telepítés folytatódik, de degraded módban (üres memória-kontextus).
- **NVIDIA driver**: 580+ ajánlott (CUDA 13 branch; minimum 570).
- **~10 GB szabad lemezterület** (modellek + PyTorch + binárisok + venv).
- **Internet A TELEPÍTÉS ALATT** (modell- és binárisletöltés) — a **futás
  teljesen offline**.

## Telepítés — egyetlen dupla kattintás

A felhasználói telepítés **EGY műveletből** áll (l. fent a Gyorsindítást): a
projekt gyökerében dupla kattintás a `START.bat`-ra. A `START.bat` a
`scripts\bootstrap.ps1` orchestrátort hívja, ami elvégzi a teljes telepítést,
az ellenőrzést és az agent indítását — egyetlen közbenső lépést sem
(PowerShell-indítás, venv-létrehozás, pip, modellletöltés,
kézi konfiguráció, környezeti változók) kell a felhasználónak kiadnia. Az első
futás ~15-60 perc (függőségek + ~4 GB modellletöltés); utána minden
indítás másodperceket vesz igénybe (idempotens gyorsútvonal).

Fejlesztői belső útvonal (ugyanaz a munka közvetlenül, fejlesztőknek):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_m1.ps1
```

A bootstrap/telepítő **18 lépést** futtat le (a felhasználói M0-specifikáció sorrendjében):

1. repo-gyökér ellenőrzése (app/ + scripts/ létezése)
2. Windows-verzió ellenőrzése (10+)
3. Python 3.11 ellenőrzése (pontosan 3.11.x; a telepítő SOHA nem telepít másik
   Python-verziót — hibát ír és leáll)
4. NVIDIA GPU ellenőrzése (nvidia-smi elérhető + GPU-név)
5. RTX 5070 ellenőrzése (informális; a döntést a 10. lépés cc-ellenőrzése hozza)
6. projektmappák létrehozása: models/, memory/, data/, logs/, bin/, vendor/
7. .venv létrehozása (Python 3.11)
8. pip frissítése
9. core függőségek (requirements.lock, ha van — a repóban van) +
   huggingface_hub **kódtár** a modellletöltéshez (Python API, CLI nélkül)
10. **PyTorch 2.7.0+cu128 telepítése + GPU smoke test** (a hivatalosan
    párosított hármast telepíti: torch 2.7.0 + torchvision 0.22.0 +
    torchaudio 2.7.0, mind a cu128 indexről, import-utáni verzió-ellenőrzéssel
    és a VoiceMem-telepítés utáni felülírás-guarddal; cc < 12.0 → a telepítés
    SIKERTELEN)
11. llama.cpp + Piper binárisok letöltése pin-elt GitHub release-ekből → `bin\`
12. VoiceMem: `vendor\voicemem` klón a pin-elt refre + `pip install -e` a
    .venv-be
13. modellletöltés: `scripts\download_models.ps1` (idempotens)
14. konfiguráció: `config\voicemem_config.yaml` + `.env` a
    `config\.env.example`-ből
15. smoke testek: `scripts\verify_m1.ps1 -WithServer` (FAIL → a telepítés
    sikertelen)
16. licenc-ellenőrzés (LICENSES.md + a Piper hangok `.onnx.json` fájljai)
17. offline környezet beállítása (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
    HF_HOME=`<gyökér>\models\hf` — a HF cache izolálva)
18. végső install report + INSTALL_MANIFEST.json

**Idempotens**: bármikor újrafuttatható — a kész részeket skip-if-exists
logikával átugorja (meglévő .venv, binárisok, modellek nem töltődnek újra), a
megszakadt letöltések pedig folytathatók. **Hamis sikert soha nem ír ki**: a
végső report minden PASS-ja a 15-17. lépések valódi ellenőrzésére épül.

### A telepítő kapcsolói (fejlesztői belső szkript — a felhasználó a `START.bat`-ot használja)

| Kapcsoló | Hatás |
|----------|-------|
| `-Root <útvonal>` | a repo-gyökér explicit megadása (alap: a szkript mappájának szülője) |
| `-SkipModels` | a modellletöltés kihagyása (offline újratelepítéshez) |
| `-SkipBinaries` | a llama.cpp/Piper binárisletöltés kihagyása (kézzel már telepítve) |
| `-SkipVoiceMem` | a VoiceMem kihagyása (a verify_m1 jelezni fogja) |
| `-Force` | a .venv újraépítése + a binárisok újraletöltése (a modelleket NEM törli) |
| `-ConnectivityAudit` | a 17. lépésben a `offline_check.ps1` audit is lefut (alap: kihagyva) |

### Pin-elt verziók (reprodukálhatóság)

| Komponens | Pin | Forrás | Cél |
|-----------|-----|--------|-----|
| llama.cpp | tag `b10717` — `llama-b10717-bin-win-cuda-13.3-x64.zip` + `cudart-llama-bin-win-cuda-13.3-x64.zip` (CUDA 13.3 runtime DLL-ek) | github.com/ggml-org/llama.cpp/releases | `bin\` (llama-server.exe + DLL-ek) |
| Piper | tag `2023.11.14-2` — `piper_windows_amd64.zip` | github.com/rhasspy/piper/releases | `bin\piper.exe` |
| VoiceMem | **VEZERLT FORK** — a repo-ban szállitott `vendor\voicemem` (alap: upstream commit `e8384e0` = tag v0.0.1) + `pip install -e` + pin-ellenőrzés | a kiadás ZIP tartalmazza (nincs letöltés a telepítéskor) | `vendor\voicemem` — identitás: `VOICEMEM_PIN.json` + `voicemem.CONTROLLED_UPSTREAM_COMMIT` (a manifest a `pin_verified` jelzést rögzíti) |
| PyTorch | `2.7.0+cu128` (RTX 5070 Blackwell sm_120) | download.pytorch.org/whl/cu128 | `.venv` |
| funasr (M2) | `1.4.11` — emotion2vec+ futtató, CPU (a requires_dist NEM tartalmaz torch-ot) | pypi.org/project/funasr | `.venv` (a telepítő 13. lépése + trio-guard) |

### Letöltött modellek (M1+M2+M3 szigor készlet — a 15. lépés)

| Modell | Cél |
|--------|-----|
| Qwen3-ASR-0.6B | `models\asr\qwen3-asr-0.6b` |
| Qwen3.6 35B A3B IQ4_XS (~19 GB; operator-altal elhelyezett, NEM auto-letoltve; v0.4.17: Ollama sha256-... blob is - magic-alapu azonositassal) | `models\llm\qwen3.6-35b-a3b` (barmelyik fajlnev) VAGY barmelyik meghajto / Ollama-blob (web UI "LLM model" picker / identify_ollama_blob.ps1 -Select / find_qwen_gguf.ps1 -> config/llm_model.json) |
| multilingual-e5-small | `models\embedding\multilingual-e5-small` |
| silero-vad v6.2.1 | `models\vad\silero-vad` |
| Piper hangok: hu_HU-anna/berta/imre + en_US-lessac (`.onnx` + `.onnx.json`) | `models\tts\piper` |
| **emotion2vec+ base (M2; model.pt ~1,1 GB + 3 config-fájl)** | **`models\emotion\emotion2vec-plus-base`** |
| **SpeechBrain ECAPA (M3; embedding_model.ckpt ~79,5 MB + 4 fájl)** | **`models\speaker\ecapa-voxceleb`** |
| HF cache gyökér (`HF_HOME` / `HF_HUB_CACHE` / `TRANSFORMERS_CACHE`) | `models\hf` |

(Ha a `download_models.ps1`-t kézzel futtatod: CSAK online, és az
`env.local.ps1` dot-source-olása ELŐTT — a `HF_HUB_OFFLINE=1` blokkolná a
letöltést. A telepítő ezt a sorrendet automatikusan tartja.)

### A telepítés vége

Sikeres futásnál a kimenet vége (a telepítő saját riportja):

```
INSTALLATION SUCCESS

Project: C:\VoiceMemAgent
Python: 3.11.9
GPU: NVIDIA GeForce RTX 5070
VRAM: 12.0 GB
PyTorch: 2.7.0+cu128
CUDA: 12.8
VoiceMem: 0.0.1
ASR: Qwen3 ASR 0.6B
LLM: Qwen3.6 35B A3B IQ4_XS (az EGYETLEN LLM)
TTS: Piper HU + EN
VAD: Silero
Embedding: multilingual E5 small
Offline: READY
License: PASS
Smoke tests: PASS

INSTALL_MANIFEST.json keszult a repo gyokerben.
```

A telepítő ezzel **INSTALL_MANIFEST.json**-t ír a repo gyökerébe
(`scripts\write_install_manifest.py`) — a reprodukálhatóság jegyzőkönyve:
OS-, Python-, CUDA- és torch-adatok, a repo git-commitja, a VoiceMem tényleges
commitja, a letöltött modellek fájllistája sha256-összefoglalóval és a
függőségverziók. A séma példája a repóban: `INSTALL_MANIFEST.example.json`.
(A manifest a telepítés után keletkezik — friss klónban még nincs meg.)

Sikertelen telepítésnél a kimenet `INSTALLATION FAILED` + a hibás lépés száma
és neve + konkrét javítási javaslat, majd a szkript azonnal leáll — hamis
SUCCESS soha nem jelenik meg. A javítás után a telepítő újrafuttatható
(idempotens).

## Ellenőrzés

Felhasználói útvonal (teljes ellenőrzés — assetek + smoke + teljes
tesztkészlet):

```
START.bat check
```

Fejlesztői belső útvonal:

```powershell
.\scripts\verify_m1.ps1                # gyors ellenőrzés (LLM-szerver: csak health)
.\scripts\verify_m1.ps1 -WithServer    # elindítja ÉS teszteli a llama-szervert is
                                        #   (health + „Hello" completion + JSON
                                        #   completion: json_object és json_schema;
                                        #   v0.3.3: elsődleges sikerfeltétel a
                                        #   GET /health -> HTTP 200, a starter
                                        #   exit kódja NEM bukta)
```

A szkript soronként `[PASS]` / `[FAIL]` / `[WARN]` jelzéssel futtatja le az M0
ellenőrzéseket (a WARN nem buktat meg); a végső összegző sor `VERIFY: PASS` vagy
`VERIFY: FAIL (N hibás ellenőrzés)`, kilépési kód 0 = PASS, 1 = FAIL. Mit
ellenőriz: a .venv Python-verziója (3.11.x) és a pip, CUDA/torch GPU smoke test
(valódi mátrixművelet) + compute capability >= 12.0, a modellfájlok (LLM GGUF
> 1 GB, ASR- és e5-config, silero_vad.onnx, 4 Piper hang `.onnx` +
`.onnx.json`), a VoiceMem-import, az ASR offline betöltése (AutoTokenizer,
local_files_only), a llama-server blokk (v0.3.3: `-WithServer` esetén a starter
indítása után a `GET /health`-et legfeljebb 30 s-ig poll-olja — ha HTTP 200,
az PASS, akkor is, ha a PowerShell starter wrapper időközben nem-0 exit
kóddal lépett ki; ha a wrapper kilépett és nincs életben llama-server
folyamat, a verify közvetlenül indítja a `bin\llama-server.exe`-t a
shellből közvetlenül levezetett konfigurációval; FAIL csak akkor, ha a
folyamat ténylegesen leállt ÉS a /health nem vált elérhetővé a timeouton
belül; a végeredmény hat külön jelzőt ír ki: server startup, health, chat
completion, JSON completion, model loaded, process alive — hibánál a
log-kivonatok először a start-szkript SAJÁT kimenetéből
(`logs\llama-server.starter.log`, Start-Transcript), majd a szerver
`logs\llama-server.err.log`-jából —, minimális „Hello” completion-teszt,
JSON completion-tesztek — `response_format: json_object` és `json_schema`
—, modell-betöltés (`GET /v1/models`), a python-bridge konfig-hiba pedig
külön WARNING, nem fatal, ha a közvetlen konfigurációval a szerver működik;
a verify a saját maga által indított szervert leállítja), Piper HU/EN
szintézis smoke (`data\audio\smoke_hu.wav` /
`smoke_en.wav`, > 1 KB), a `memory\` almappák írhatósága (write probe) és a
konfiguráció (`AgentConfig.from_yaml` + `validate()`).

## Indítás

Felhasználói útvonal — ugyanaz a dupla kattintás:

```
START.bat
```

(A gyorsindítási fejezetben leírt automatikus ellenőrzések után a bootstrap
megjeleníti a telepítési összefoglalót, majd elindítja az agentet.)

Fejlesztői belső útvonal (egyenértékű):

```powershell
.\scripts\start_agent.ps1
```

A szkript ellenőrzi a .venv-et, rövid asset-ellenőrzést futtat
(`--check`), ellenőrzi a konfigurációt, majd — ha a llama-server nem fut
(127.0.0.1:8080/health) — **új ablakban elindítja** a
`scripts\start_llama_server.ps1`-t és megvárja a felállását (max 60 s, 2
másodpercenkénti poll; ha a szerver hibával áll le, a start-szkript
transcriptjét — `logs\llama-server.starter.log` — és a
`logs\llama-server.err.log` utolsó sorait azonnal kiírja). Ezután beállítja az offline env-változókat
(`config\env.local.ps1`, csak ha még nincsenek beállítva), és a repo gyökeréből
elindítja: `.venv\Scripts\python.exe -m app.main`. (Kézi indítás is lehetséges:
`powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_llama_server.ps1`
külön ablakban — a Bypass forma policy- és MOTW-független, mindig működik;
a `-NoServer` / `--no-server` kapcsolóval az automatikus indítás
kihagyható.) Leállítás: Ctrl+C.

**Javasolt smoke-sorrend** (valós komponensek előtt):

```powershell
.\scripts\start_agent.ps1 --mock           # 3 fordulós szkriptelt demo:
#    1. forduló: EN nyelvtani hiba javítása ("Why is I have went wrong?")
#    2. forduló: HU kérdés magyar válasszal
#    3. forduló: memória-visszaidézés ("Do I still make the same mistake?")
.\scripts\start_agent.ps1 --check          # asset-lista + konfiguráció ellenőrzés
.\scripts\start_agent.ps1 --mock --barge-in-demo   # a TTS megszakításának demója
.\scripts\start_agent.ps1                 # éles üzem
```

A `start_agent.ps1` NINCS `param()` blokkja — minden argumentuma áthalad a
`.venv\Scripts\python.exe -m app.main`-nak (`--check`, `--mock`,
`--list-config`, `--text "..."`, `--barge-in-demo`). Kivétel a
`-Root <útvonal>` / `--root <útvonal>` és a `-NoServer` / `--no-server`. A
PowerShell-ben az egyszer-vesszős alak (pl. `-Mock`) paraméternek tűnne, ezért a
szkript az ilyen flageket automatikusan dupla vesszősre normalizálja
(`-Mock` → `--mock`); a biztonságos forma a dupla vessző.

## Használat

- **Beszélj** — a VAD a beszéd végét **300 ms csend** (hangover) után detektálja,
  erre indul az ASR véglegesítése, a memória-kontextus, az LLM és a TTS.
- **Barge-in**: a válasz alatt **500 ms folyamatos beszéd** (speech-probabilitás
  ≥ 0,30) megszakítja a folyó TTS-t/LLM-et/memória-keresést, és a te mondatod
  kerül sorra.
- Az ügynök a választ mondatonként szintetizálja (első chunk ~24 karakter) —
  az első hang gyorsan jön, a mondat többi része közben generálódik.
- Két nyelv ugyanabban a beszélgetésben: a válasz nyelvét az aktuális
  megszólalás domináns nyelve határozza meg (magyar kérdés → magyar válasz).
- **M2 érzelem**: minden megszólalásnál az utterance utolsó ~5 másodpercét az
  emotion2vec+ base **CPU-n** elemezí (a memória-kereséssel párhuzamosan), és
  a fúzionált érzelmi állapot (proszódia 0,6 + szöveges jelzők 0,4) bekerül a
  tanár-promptba. **Frusztrált** hangnál a válasz lassabb (`--length_scale 1.1`,
  kb. 10%-os lassítás), egyszerűbb, példaorientáltabb. Erős érzelmek a
  `data\emotion_log.jsonl`-be kerülnek, és VoiceMem-fact-ként tárolódnak.

## M2 — érzelmi intelligencia (emotion2vec+)

A **2. mérföldkő** (milestones 8. fejezet) implementációja:

| Elem | Megvalósítás |
|------|--------------|
| Prosody analízis | `app/emotion.py` `EmotionAnalyzer` — emotion2vec+ base (~90M), FP32, **CPU**, funasr `AutoModel`, az utterance utolsó 5 mp-je (`emotion_window_s`) |
| 9 emotion2vec+ kategória → 4 tanári kategória | angry/disgusted/fearful → **frustrated**; happy/surprised → **happy**; sad → **sad**; neutral/other/unknown → **neutral** |
| Fusion (spec 8.3.3) | `fuse_emotion`: 0,6·proszódia + 0,4·szemantika (determinisztikus HU/EN szöveges jelzők; a teljes tartalmi ellenőrzést az LLM végzi a promptban) |
| Prompt-injekció (spec 8.3.2) | a tanár-prompt ÉRZELEM-blokkja: címke + valence/arousal + „Verify this assessment…” utasítás |
| Adaptív válasz (spec 8.4) | frusztrált user → Piper `--length_scale 1.1` (10% lassítás) + „simplify, slow down, concrete examples” instrukció |
| Emotion memory (spec 8.3.4) | minden elemzett forduló → `data\emotion_log.jsonl`; erős érzelem (|valence| vagy arousal ≥ 0,5) → VoiceMem-fact (RightBrain) |
| Modularitás (spec 8.6) | `enable_emotion: false` (vagy `VOICEMEM_ENABLE_EMOTION=0`) → a pipeline **változatlanul** az M1 utat futtatja; a hiányzó modell/funasr NEM blokkol: graceful degradation |
| VRAM | **0 GB GPU-változás** — az emotion2vec+ kizárólag CPU-n fut (hardverpolitika 4.6) |

M2 demó (mock, sandboxban is futtatható):

```powershell
.\.venv\Scripts\python.exe -m app.main --mock --emotion-demo
# 3 forduló: frusztrált -> semleges -> vidám; látható: emotion-block a
# promptban, length_scale 1.1 az első fordulóban, emotion log sorok.
```

A modell a telepítő 14. lépése tölti (`MODELS.lock.json` `emotion` bejegyzés:
`emotion2vec/emotion2vec_plus_base` — 4 futtatási fájl, model.pt ~1,1 GB);
a futtató `funasr==1.4.11` a 13. lépés telepíti (CPU, a torch-trióst nem
érinti — a telepítő ezután is lefuttatja a trio-guardot).

**Licenc**: az emotion2vec+ súlyok a FunASR Model Open Source License
Agreement v1.1 alatt állnak (ingyenes használat **attribution mellett** —
a modell neve megtartandó; l. `LICENSES.md`), a funasr kód MIT.

## M3 — beszélő-azonosítás (SpeechBrain ECAPA)

A **3. mérföldkő** (milestones 9. fejezet) implementációja — a memória a
BESZÉLŐ-hez kötődik, a hangok nem keverednek:

```
audio → ECAPA embedding (CPU, ~30-50 ms) → cosine-matching a regisztrált
       beszélőkkel (küszöb 0,50) → VoiceMem(user_id=speaker_id)
       → beszélőnkénti SQLite + Qdrant memóriatér (pl. teacher_thomas)
```

| Elem | Megvalósítás |
|------|--------------|
| Beszélő-beágyazás | `app/speaker.py` `SpeakerEmbedder` — SpeechBrain **spkrec-ecapa-voxceleb** (ECAPA-TDNN, ~23M, 192-dim, **CPU**), lokális modellkönyvtárból (nincs hálózati hívás futásidőben), az utterance utolsó 5 mp-je (`speaker_window_s`) |
| Azonosítás (spec 9.3.2) | `identify_speaker`: legjobb cosine a regisztrált referenciák között; `speaker_match_threshold: 0,50` alatt → **ismeretlen** → `user_id="voice_user"` fallback (a memóriák SOHA nem keverednek) |
| Regisztráció (spec 9.3.3) | `--register-speaker <név>`: ~10 s tiszta beszéd (VAD-vezérelt gyűjtés), átlagolt embedding → `data/speaker_registry.json` (idempotens felülírás) |
| Memória-útválasztás (spec 9.3.4) | a feloldott `speaker_id` a VoiceMem `user_id`-je — a bridge **beszélőnkénti facade-t** tart (külön SQLite + Qdrant collection), a retrieval ÉS a commit is a helyes térbe kerül |
| Párhuzamosság | az ECAPA-beágyazás az M2 proszódia-elemzéssel PÁRHUZAMOSAN indul; a memória-keresés csak a gyors speaker-lépésre vár → additív késleltetés a 100 ms-os M3-költségvetésen belül |
| Modularitás (spec 9.5) | `enable_speaker: false` (vagy `VOICEMEM_ENABLE_SPEAKER=0`) → a pipeline **változatlanul** az M1/M2 utat futtatja; a hiányzó modell/speechbrain/regisztráció NEM blokkol: graceful degradation (minden hang `voice_user`) |
| VRAM | **0 GB GPU-változás** — az ECAPA kizárólag CPU-n fut (hardverpolitika 4.6) |

M3 demó (mock, sandboxban is futtatható):

```powershell
.\.venv\Scripts\python.exe -m app.main --mock --speaker-demo
# 4 forduló: thomas -> másik beszélő -> thomas -> ISMERETLEN;
# látható: [M3 beszelő: ...] sorok fordulónként, memória-útválasztás
# HELYES ellenőrzéssel és cross-speaker kontamináció: 0 összegzéssel.
```

Regisztráció a célgépen (valós mikrofon):

```powershell
.\.venv\Scripts\python.exe -m app.main --register-speaker thomas
# ~10 másodperc tiszta beszéd; utána a valós módban minden thomas-forduló
# a teacher_thomas memóriatérbe kerül.
```

A modell a telepítő 15. lépése tölti (`MODELS.lock.json` `speaker` bejegyzés:
`speechbrain/spkrec-ecapa-voxceleb` — 5 futtatási fájl, embedding_model.ckpt
~79,5 MB); a futtató `speechbrain==1.1.1` a **14. lépés** telepíti (CPU; a
requires_dist `torch>=2.1.0`/`torchaudio>=2.1.0` követelményt a pin-elt cu128
trió kielégíti — a telepítő ezután is lefuttatja a trio-guardot).

**Küszöb-megjegyzés**: a spec 9.3.2 a 0,50-es ECAPA-küszöbet pineli —
konzervatív választás (mért kereszthang-cosine ~0,13–0,18; azonos beszélő
0,37–0,75). Ismeretlen hang → közös `voice_user` tér (biztonságos fallback);
hangolni a `speaker_match_threshold`-dal lehet.

**Licenc**: a modell ÉS a speechbrain csomag is **Apache-2.0** (nem gated —
l. `LICENSES.md`).

## Tesztek és benchmarkok

```powershell
.\scripts\run_tests.ps1                            # bajtkód-ellenőrzés + teljes unittest + validációs riport
python -m unittest discover -s tests -t . -v       # kézzel (a -t . kötelező elem)
python tests\integration\offline_test.py           # Python-szintű offline audit
```

A tesztfa négy almappára bontott: **`tests/unit`** (modultesztek),
**`tests/integration`** (pipeline-mock + offline audit),
**`tests/benchmark`** (teljesítménymérések) és — M0.1-től —
**`tests/validation`**: **funkciónkénti validációs tesztek** (config,
runtime/GPU, modellek, VAD, ASR, LLM, TTS, embedding, memória, barge-in,
audio, teacher-persona, pipeline, offline, manifest, szkriptek, release,
emotion).
Jelenleg ~330 teszt. A tesztek a sandboxon és a célgépen egyaránt futnak —
nehéz függőség nem kell hozzájuk.

A `tests/validation` tesztek két szintet különböztetnek meg:

- **`test_logic_*`** — a funkció szerződése nehéz függőség nélkül
  (sandboxban és célgépen is fut; a ZIP-kapu része);
- **`test_deep_*`** — a **valdi** komponens ellenőrzése, ha az adott gépen
  elérhető (modellek, GPU, llama-server, Piper, VoiceMem csomag, hang).
  Hiányzó komponens esetén a teszt SKIP-pel jelz a
  `logs\validation_report.json` riportba (sosem “sikeres”, és sosem
  akad meg); a `build_release.ps1 -StrictValidation` ezt a SKIP-et FAIL-
  nek tekinti.

A `run_tests.ps1` minden futtatáskor írja a
`logs\validation_report.json`-t (a deep ellenőrzések funkció-mátrixa),
amit a `build_release.ps1` a `BUILD_INFO.json`-be és a
`RELEASE_INDEX.json`-be épít be.

Benchmarkok (csak célgépen, éles komponensekkel):

| Fájl | Mit mér |
|------|---------|
| `tests/benchmark/asr_benchmark_hu.py` | magyar ASR-helyesség (WER/CER) a spec tesztszövegén |
| `tests/benchmark/tts_benchmark.py` | Piper TTFB és valós idejű faktor 5 magyar mondaten |
| `tests/benchmark/llm_benchmark.py` | tanári minőség-benchmark (10 HU + 10 EN beszélgetés, viselkedéses kulcsszó-ellenőrzés, ≥ 80%) |
| `tests/benchmark/llm_server_latency.py` | llama-server saját latenciája: TTFT, első mondat, teljes válasz, token/s (p50/p95) + JSON-mód-mérés — a M1-es p50 < 2 s / p95 < 2,5 s end-to-end cél szerver-oldali komponense |
| `scripts/measure_vram.ps1` | VRAM-mérés fázisonként (idle → modell betöltve → első kérés → normál- és JSON-generáció → csúcs), folyamatonként szétbontva (llama-server vs. egyéb); riport: `logs\vram_report.json` |
| `tests/benchmark/latency_benchmark.py` | teljes kör (beszédvég → első hang), cél p50 < 2,0 s, p95 < 2,5 s |
| `tests/integration/offline_test.py` | Python-szintű socket/DNS audit (nincs kimenő hívás) |

## Frissítések és verziózott ZIP-kiadások

**Minden frissítés után egy dupla kattintás vagy egy parancs** — a teljes
validáció és a verziózott ZIP egy lépésben (a ZIP **csak** zöld tesztek után
készül el):

```
START.bat build
```

illetve (fejlesztői, paraméterezhető formában):

```powershell
.\scripts\build_release.ps1 -Notes "mi változott és miért"
```

A `build_release.ps1` futása (8 lépés):

1. cél-verzió kiszámítása (`VERSION` + `-Bump`, alap **patch** — minden
   frissítés új verzioszámot kap; `-Bump none|minor|major` vagy
   `-Version x.y.z` finomhangolás);
2. **KÖTELEZŐ TESZTKAPU**: a teljes tesztkészlet (unit + integration +
   benchmark + **validation**) lefut a ZIP előtt — ha egyetlen teszt is
   elbukik, **nem készül ZIP**, a kísérlet a
   `releases\BUILD_HISTORY.json`-be kerül (hamis siker soha);
3. verzió-írás a `VERSION` + `pyproject.toml` + `config/voicemem_config.yaml`
   `project.version` mezőkbe (szinkron-szabály);
4. staging megengedési-listával: a ZIP a **forrást** tartalmazza (`app/`,
   `config/`, `scripts/`, `tests/`, gyökéri dokumentumok + `START.bat` +
   `MODELS.lock.json` + generált `BUILD_INFO.json`), a futási állapot
   (`.venv/`, modellek, memória, binárisok, vendor-klón,
   `.install_state.json`) **nem** kerül bele — másik gépen a kicsomagolt
   mappából a `START.bat` dupla kattintása telepít (one-click);
5. `BUILD_INFO.json` a ZIP gyökerébe (verzió, UTC-idő, git-commit,
   teszt- és validációs-összefoglaló, notes);
6. `releases\VoiceMemAgent_vX.Y.Z.zip`;
7. `releases\VoiceMemAgent_vX.Y.Z.zip.sha256` (SHA256-ellenőrzőösszeg);
8. nyilvántartás: `releases\RELEASE_INDEX.json` (kiadások),
   `CHANGELOG.md` (változatenapló), `releases\BUILD_HISTORY.json`
   (minden kísérlet — sikeres és sikertelen).

Kapcsolók: `-Bump`, `-Version`, `-Notes`, `-Force` (létező verzió
újraépítése), `-StrictValidation`, `-Root`.

**`-StrictValidation`** — kiadási minőségű buildhez: a validációs deep
ellenőrzések SKIP-e is FAIL-t jelent. A telepített célgépen futtatva
(`.\scripts\build_release.ps1 -Notes "..." -StrictValidation`) a deep
ellenőrzések (modellek, GPU, llama-server, Piper, VoiceMem, hang) valóban
lefutnak — egy strict-zöld build így a teljes funkció-mátrix érvényességét
tanúsítja. Telepítetlen gépen a strict mód jogosan bukik el (a deep
ellenőrzések nem futtathatók); ilyenkor használd a sima buildet. A deep
ellenőrzések eredményét a `logs\validation_report.json` rögzíti (a
`run_tests.ps1` írja minden futtatáskor).

A ZIP-ből másik gépen való telepítés:

```
1. Csomagold ki a ZIP-et tetszőleges mappába (pl. C:\VoiceMemAgent).
2. Dupla kattintás: START.bat
```

Ennyi — a `START.bat` ugyanazt az egy-kattintásos telepítést végzi el, mint a
forrás-klón esetén (venv, függőségek, modellek, ellenőrzés, agent indítás).
(A ZIP csak a forrást tartalmazza — a modelleket, binárisokat és a `.venv`-et
a bootstrap tölti/hozza létre; a `MODELS.lock.json` a ZIP-ben van, így a
modell-zer pontosan követhető.)

## Konfiguráció

Fő szabály: **a környezeti változók felülírják a YAML-t, a YAML felülírja a
kóddefaultokat.** A környezeti változók teljes referenciája:
**`config/.env.example`** (a telepítő készíti el belőle a `.env`-et, ami
gitignore-olt). Windowson az aktív loader a `config/env.local.ps1` — a telepítő
és a `start_agent.ps1` állítja be a változóit.

| Kulcs | Hol | M1-érték / jelentés |
|-------|-----|---------------------|
| `app:` szekció | `config/voicemem_config.yaml` | minden AgentConfig-mező M1-értéke kommentelve |
| `vad_threshold` / `vad_hangover_ms` | YAML | 0,25 / 300 ms — beszédvég-detektálás (v0.4.11: 0,5 → 0,25) |
| `barge_in_threshold` / `barge_in_min_speech_ms` | YAML | 0,30 / 500 ms — megszakítás |
| `asr_model_name` | YAML | `Qwen/Qwen3-ASR-0.6B` (M1-szigor, NEM 1.7B) |
| `tts_hu_voice` / `tts_en_voice` | YAML | `hu_HU-anna-medium` / `en_US-lessac-medium` |
| `enable_emotion` / `enable_speaker` | YAML | **true (M2/M3 default)**; `VOICEMEM_ENABLE_EMOTION=0` / `VOICEMEM_ENABLE_SPEAKER=0` env-rel kikapcsolható (M1-azonos futás) |
| `speaker_match_threshold` / `speaker_window_s` | YAML | 0,50 (spec 9.3.2) / 5,0 s (9.3.1) — ECAPA azonosítási küszöb és ablak |
| `SPEAKER_MODEL_PATH`, `VOICEMEM_SPEAKER_THRESHOLD` | env | opcionális M3-modellútvonal / küszöb-felülírás |
| `voicemem_mode` | YAML | `text_mode` (multi_modal minden mérföldkőben kizárt) |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | env.local.ps1 | offline garancia |
| `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` | env.local.ps1 / .env.example | `models/hf/...` — HF-cache-izoláció a repo gyökerébe |
| `VOICEMEM_HOME` / `VOICEMEM_MEMORY_ROOT` | env.local.ps1 | repo-gyökér (alap) / `<gyökér>/memory` |
| `LLAMA_SERVER_HOST/PORT`, `LLAMA_MODEL_PATH` | env.local.ps1 | `127.0.0.1` / `8080` / (v0.4.17: nincs default — a sor kikommentezve; feloldas: config/llm_model.json > LLAMA_MODEL_PATH > operator-dir) |
| `LLAMA_CONTEXT_SIZE`, `LLAMA_N_GPU_LAYERS`, `LLAMA_CACHE_TYPE_K/V` | env.local.ps1 | 8192 / -1 / q8_0 / q8_0 |
| `OPENAI_BASE_URL`, `OPENAI_MODEL` | env.local.ps1 | `http://localhost:8080/v1` / `qwen3.6-35b-a3b` (dummy API-key, l. .env.example) |
| `QWEN3_ASR_MODEL_PATH`, `PIPER_VOICES_PATH`, `SILERO_VAD_PATH`, `EMBEDDING_MODEL_PATH` | env.local.ps1 | opcionális modellútvonal-felülírások (a defaultok már az M0-elrendezésre mutatnak) |

Dev (Linux/macOS) párnak: `source config/env.local.sh` — ugyanez a
változóhalmaz; a gyökér itt is a repo-gyökér (a szkript a saját helyéből
számolja ki, gépfüggetlen).

## Offline garancia

- **Env-változók**: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (a
  `config/env.local.ps1` állítja) — a transformers/huggingface_hub futás közben
  nem érhet hálózatot.
- **HF-cache-izoláció (M0)**: a `HF_HOME`, `HF_HUB_CACHE` és `TRANSFORMERS_CACHE`
  a repo `models\hf` almappájára mutat — a HuggingFace cache nem a felhasználói
  profilba (`~/.cache/huggingface`) kerül, hanem a projektgyökérbe. A telepítő
  és a `verify_m1.ps1` is ezt állítja be.
- **Modell-pin-ek**: a `MODELS.lock.json` rögzíti a letöltött modellek
  revízióit (a `scripts\download_models.ps1` írja a letöltés végén, ha még nem
  létezik; kézzel szerkeszthető modellzár). A repó friss klónjában még nincs
  meg — a modellletöltés után keletkezik.
- **Python-szintű audit**: `python tests\integration\offline_test.py` — a socket-
  és DNS-hívásokat műszerfalazza; eredmény: nincs nem-loopback kapcsolat.
- **OS-szintű audit** (a spec 21.2 heti ellenőrzése):
  ```powershell
  .\scripts\offline_check.ps1                       # pillanatkép
  .\scripts\offline_check.ps1 -Capture -Seconds 30  # 30 s mintavétel, amíg beszélsz az ügynökkel
  ```
  Elvárt kép: az ügynök pontosan **egy loopback kapcsolatot** tart
  (127.0.0.1:8080 → llama-server); minden nem-loopback távoli cím VIOLÁCIÓ.

## Mappastruktúra

A végleges M0-struktúra (a repo-gyökér bárhol lehet — minden útvonal relatív;
a példában `C:\VoiceMemAgent`):

```
C:\VoiceMemAgent\
├── START.bat                 (AZ EGYETLEN felhasználói belépési pont — dupla kattintás)
├── .venv\                    (a telepítő hozza létre — Python 3.11)
├── app\                      (main.py, pipeline.py, audio_io.py, ... — az M1 pipeline)
├── config\                   (voicemem_config.yaml, env.local.ps1, env.local.sh, .env.example)
├── models\
│   ├── asr\qwen3-asr-0.6b\
│   ├── llm\qwen3.6-35b-a3b\ (Qwen3.6-35B-A3B-IQ4_XS.gguf — az EGYETLEN LLM; a web UI picker barmelyik meghajtorol kivalaszthatja)
│   ├── tts\piper\            (4 hang: .onnx + .onnx.json)
│   ├── vad\silero-vad\       (silero_vad.onnx)
│   ├── embedding\multilingual-e5-small\
│   ├── emotion\              (ÜRES — M2 placeholder, M0/M1-ben tilos modellt tenni ide)
│   └── hf\                   (HuggingFace cache — HF_HOME ide mutat)
├── memory\                   (sqlite\, qdrant\, backups\ — VoiceMem hosszú távú memória)
├── data\                     (audio\ — smoke hangok; benchmarks\ — eredmények)
├── logs\                     (bootstrap.log, llama-server.starter.log = a start-szkript transcriptje, llama-server.out.log, llama-server.err.log, validation_report.json, vram_report.json, ...)
├── tests\                    (unit\, integration\, benchmark\, validation\)
├── scripts\                  (bootstrap.ps1 — one-click orchestrátor (a START.bat hívja);
│                              install_m1.ps1, download_models.ps1, download_models_hf.py
│                              (HF Python API letöltő), start_llama_server.ps1,
│                              start_agent.ps1, verify_m1.ps1, offline_check.ps1,
│                              run_tests.ps1, build_release.ps1, verify_setup.py,
│                              write_install_manifest.py)
├── bin\                      (llama-server.exe + piper.exe — a telepítő tölti)
├── vendor\voicemem\          (VoiceMem pinned klón — a telepítő klónoz)
├── releases\                 (verziózott ZIP-kiadások + index + build-history — a
│                              build_release.ps1 írja; a repóban csak a README)
├── requirements.txt / requirements.lock / pyproject.toml
├── CHANGELOG.md               (változatonapló — a build_release.ps1 bővíti)
├── INSTALL_MANIFEST.json     (a telepítő generálja; example a repóban)
├── .install_state.json       (a bootstrap állapotfájlja — futás közben keletkezik)
├── MODELS.lock.json          (repo-fájl: modell- és tool-pin-ek — a bootstrap
│                              jelenlét-ellenőrzése ezt használja)
├── VERSION                   (0.1.0)
└── LICENSES.md / README.md / CONTRACT.md
```

Megjegyzések a fához:

- A `bin\` és `vendor\voicemem\` sorok **kiegészítések a felhasználói
  M0-specifikáció fája mellett** (a binárisok és a VoiceMem-klón helye).
- A `INSTALL_MANIFEST.json` és `MODELS.lock.json` a telepítés, illetve a
  modellletöltés után keletkeznek — friss klónban még nem léteznek (az
  `INSTALL_MANIFEST.example.json` sémápélda viszont repo-fájl).
- A `models/` almappákban rövid magyar README.md-ok írják le, melyik fájl vár
  oda és melyik HF-repóból jön (repo-fájlok, a letöltés céljai nélkülük is
  létrejönnek).

## M0 exit criteria

Az M0 (környezet-bootstrap) elfogadásához a felhasználói spec 14 pontos
ellenőrzőlistája — **mind a 14 PASS szükséges** (a telepítő lépései futtatják le
és a smoke testek igazolják; utólag bármikor újratesztelhető a
`scripts\verify_m1.ps1`-lel):

| # | Szempont | Ellenőrzés |
|---|----------|------------|
| 1 | Python 3.11 | pontosan 3.11.x elérhető (py -3.11) |
| 2 | venv | a `.venv` a repo gyökerében, benne Python 3.11 |
| 3 | PyTorch | 2.7.0+cu128 telepítve és importálható a .venv-ből |
| 4 | CUDA | GPU smoke test: valódi mátrixművelet fut a GPU-n |
| 5 | RTX 5070 | GPU-név-ellenőrzés (informális; a döntő a cc >= 12.0, sm_120) |
| 6 | ASR | Qwen3-ASR-0.6B fájljai + offline tokenizer-betöltés |
| 7 | LLM | Q4_K_M GGUF (> 1 GB) + llama-server /health + 1-tokenes completion |
| 8 | Piper HU | magyar hangszintézis smoke (`data\audio\smoke_hu.wav` > 1 KB) |
| 9 | Piper EN | angol hangszintézis smoke (`data\audio\smoke_en.wav`) |
| 10 | VoiceMem | `import voicemem` működik a .venv-ben |
| 11 | Memory | `memory\sqlite` + `qdrant` + `backups` írhatósága (write probe) |
| 12 | Config | `AgentConfig.from_yaml` + `validate()` hibátlan |
| 13 | License | LICENSES.md megvan + minden Piper `.onnx.json` olvasható |
| 14 | Offline | HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE + `models\hf`-izoláció beállítva |

**Az M1 fejlesztés csak M0 PASS után kezdhető el.**

### M0.2 kiegészítés — one-click elfogadási tesztek (START.bat)

Az M0 (M0.2 óta) csak akkor PASS, ha a `START.bat` a fenti 14 pontot
**közbenső felhasználói lépés nélkül** teljesíti. Három kötelező elfogadási
teszt (mindhárom a `START.bat` egyetlen dupla kattintásából indul):

| # | Teszt | Kiindulás | Elvárt eredmény |
|---|-------|-----------|-----------------|
| 15 | **Clean machine test** | tiszta Windows: nincs `.venv`, nincs függőség, nincs `huggingface_hub`, nincs letöltött modell | egyetlen `START.bat`-indítás végigmegy a teljes folyamán (Python-ellenőrzés → venv → függőségek → HF könyvtár → M1 modellek → config → manifest → smoke testek → verification → agent indul), a felhasználó közben SEMMILYEN manuális Python/pip/PowerShell/HuggingFace-parancsot nem ad ki |
| 16 | **Existing environment test** | már telepített, konfigurált környezet | `START.bat` NEM telepíti újra a rendszert, NEM tölti újra a modelleket, ellenőrzi a verziókat, és elindítja az agentet (idempotens gyorsútvonal, `.install_state.json` + `MODELS.lock.json` jelenlét-ellenőrzés alapján) |
| 17 | **Repair test** | szándékosan törölt csomag vagy modell | `START.bat` felismeri a hiányt (import-probe / lock-jelenlét-ellenőrzés) és automatikusan javítja (újrapip / újradownloadol), utána az agent elindul — a felhasználó nem kap "javítsd kézzel" útmutatást |

## Ismert M1-korlátok

- **Kvázi-streaming ASR**: a transformers-backend nem ad valódi partial
  streameket — 600 ms-os kötegekben transzkribál, a végleges szöveg a hangvég
  után ~250 ms-mal kész (igazi streaming = későbbi mérföldkő/vLLM).
- **Loopback gating, nem igazi AEC**: amíg a TTS szól, a mikrofonjelet nem
  etetjük az ASR-be (csak a VAD fut tovább a barge-inhoz). Valódi echo
  cancellation (SpeexDSP) a V1.5-tervben szerepel.
- **Piper prozódia**: a neurális TTS természetes, de a humán dallamosságnál
  laposabb; a HU hangok mondat-szintű nyelvváltást kezelnek (a mondat közepi
  kódváltás nem az erőssége).
- **Single-user**: a memória egy felhasználóra (`voice_user`) épül; majd a
  beszélő-felismerés (M3) választja szét a több embert.
- **emotion/speaker/scene kizárva**: az M1 szándékosan nem tartalmaz
  érzelmi vagy beszélő-felismerést — ezek M2, illetve M3 anyagai.

## Troubleshooting

- **„not digitally signed" / „cannot be loaded" PowerShell-hiba kézi szkriptindításkor**
  (v0.3.2) — a böngészőből letöltött release ZIP Explorer-kicsomagolásánál a
  `.ps1` fájlok örökölhetik az Internet-zóna jelzést (Mark of the Web /
  Zone.Identifier), és az alapértelmezett RemoteSigned policy „not digitally
  signed" hibával blokkolja őket egy sima PowerShell-sessionben. A
  `START.bat` lánc mindent `-ExecutionPolicy Bypass`-szal futtat, ezért
  SOSEM érintett; kézi futtatásnál a garantáltan működő forma:

  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_m1.ps1 -WithServer
  ```

  vagy a jelzés egyszeri eltávolítása minden szkriptről (a telepítő és a
  verify ezt minden futásnál automatikusan megteszi):

  ```powershell
  Get-ChildItem .\scripts\*.ps1 | Unblock-File
  ```
- **llama-server 1–3 mp alatt néma módon kilép, a `logs\llama-server.err.log`
  üres** (v0.3.2) — klasszikus ok: a `bin\` CUDA DLL-készlet hiányos
  (`ggml-cuda*.dll` / `cudart*.dll` / `cublas*.dll`); ilyenkor a
  `llama-server.exe` `STATUS_DLL_NOT_FOUND`-dal (0xC0000135) hal meg,
  STDERR-KIMENET NÉLKÜL — ezért üres a log. A v0.3.2 starter ezt (a)
  név szerinti DLL-előellenőrzéssel fogja el, (b) a gyerekfolyamat exit
  kódját dekódolva kiírja, (c) a saját kimenetét
  `logs\llama-server.starter.log` transcriptbe naplózza. Javítás: a
  telepítő v0.3.2+ DLL-tudatosan újraletölti a binárisokat
  (`START.bat repair`, vagy `install_m1.ps1 -SkipModels`).
- **A `verify_m1.ps1` / `start_llama_server.ps1` indítás-detektálása és
  exit-kód-kezelése** (v0.3.3) — ha kézi tesztek bizonyítják, hogy maga a
  llama-server egészséges (`llama-server.exe --version` PASS,
  modell-betöltés PASS, `/health` HTTP 200, `127.0.0.1:8080` listening
  PASS), de a verify mégis FAIL-t ír a szerver-indításra: a v0.3.3 verify
  az elsődleges sikerfeltételt a `GET /health -> HTTP 200`-ra tette — a
  PowerShell starter wrapper exit kódja NEM bukta. Ha a wrapper kilép
  (tipikus ok: a python-bridge config-hiba — „a konfiguráció … nem tölthető
  be python-bridge-en keresztül”), a verify közvetlenül indítja a
  `bin\llama-server.exe`-t a shellből levezetett konfigurációval (ugyanazokkal
  a flagekkel: `-ngl -1`, `-c 8192`, `--parallel 1`, q8_0 KV-cache,
  `--temp 0.7`); a bridge-hiba külön `[WARN]`, nem fatal. A starter maga is
  így viselkedik: bridge-hiba esetén FIGYELEM + közvetlen konfigurációval
  indul, a llama-server meghívása változatlan. A végeredmény hat külön
  jelzőben jelenik meg (server startup / health / chat completion /
  JSON completion / model loaded / process alive).
- **A telepítő nem találja a Python 3.11-et** — a telepítő NEM telepít Pythont:
  töltsd fel a 3.11.x-et a python.org-ról (telepítőben jelöld be az "Add
  python.exe to PATH" opciót), majd futtasd újra a telepítőt.
- **GPU smoke test elbukik / "compute capability < 12.0"** — frissítsd az
  NVIDIA-drivert (580+ ajánlott), és RTX 5070 (Blackwell sm_120) szükséges;
  CPU-ra nincs támogatás, a telepítés nem lesz sikeres.
- **llama-server nem indul (`bin\llama-server.exe` hiányzik)** —
  `START.bat repair`: a bootstrap automatikusan újratölti a hiányzó binárist
  (a modelleket NEM tölti újra).
- **Piper-hang hiányzik** — `START.bat repair`: a letöltő idempotensen pótolja
  a hiányzó `.onnx` + `.onnx.json` fájlokat (nyelvi-almappás és flat útvonalat
  is automatikusan próbál, majd a hangokat a `models\tts\piper` gyökérbe
  lapítja).
- **HF-letöltés lassú / megszakad** — a letöltő újrapróbál és automatikusan
  folytatja a megszakadt átvitelt (resume). Az LLM GGUF NEM resze az auto-letoltesnek
  (v0.4.16: a Qwen3.6 35B A3B IQ4_XS operator-altal elhelyezett/barhol kivalaszthato,
  nincs tükör-lánc és nincs alternatív LLM — LLM FALLBACK: NONE); a VAD-nál
  tphakala re-host → eredeti silero repo. Csak indítsd újra a `START.bat`-ot —
  a letöltés ott folytatódik, ahol megszakadt.
- **VoiceMem import-hiba** — `START.bat repair`: a telepítő újraklónozza és
  újratelepíti a `vendor\voicemem` klónt (`pip install -e`).

## Mérföldkövek

- **M0 — kész**: környezet-bootstrap; ez a README Telepítés fejezete (START.bat
  egykattintásos telepítés, M0 exit criteria l. fent).
- **M0.1 — kész**: verziózott ZIP-kiadás minden frissítés után +
  funkcionkénti validációs tesztek + build előtti **kötelező** tesztkapu
  (`scripts\build_release.ps1`, `scripts\run_tests.ps1`,
  `tests\validation\` — l. a „Frissítések és verziózott ZIP-kiadások”
  fejezetet).
- **M1 — kész**: alap lokális hangügynök (VAD + ASR + memória + LLM + TTS +
  barge-in, tanár-persona). Exit-szempontok: latencia p50 < 2,0 s / p95 < 2,5 s,
  VRAM < 8 GB steady, offline audit PASS, licenc-ellenőrzés PASS.
- **M2 = kész (ez a repo)** — érzelmi intelligencia: emotion2vec+ base
  proszódia-elemzés CPU-n (a memóriával párhuzamosan), 0,6/0,4 fúzió,
  prompt-injekció, frusztrációnál lassabb TTS (`--length_scale 1.1`),
  emotion-log + VoiceMem-fact; `enable_emotion` zászlóval kikapcsolható
  (M1-azonos viselkedés). L. a „M2 — érzelmi intelligencia” fejezetet.
- **M3 = kész (ez a repo)** — beszélő-azonosítás: SpeechBrain ECAPA
  (192-dim, CPU, párhuzamos az emotion-ággyal), 0,50-es cosine-küszöb,
  `--register-speaker` regisztráció, beszélőnkénti VoiceMem-memóriatér
  (retrieval + commit + emotion-fact ugyanabba a térbe), ismeretlen hang →
  `voice_user` fallback; `enable_speaker` zászlóval kikapcsolható
  (M1/M2-azonos viselkedés). L. az „M3 — beszélő-azonosítás” fejezetet.
- **Scene/zene-felismerés: véglegesen kizárva** minden mérföldkőből.

A két spec-dokumentum a **szülő könyvtárban** él (nem a repóban):
`../voicemem-v1-spec.md` (komponens-szintű specifikáció) és
`../voicemem-v1-milestones.md` (mérföldkő- és licencterv).

## Licenc

Az M1 komponenseinek licencmanifesztje: **[LICENSES.md](LICENSES.md)** —
minden komponens PASS (lokális, offline, nem-NC, nem research-only). A **Piper
engine GPL-3.0-or-later** licencű, de a copyleft hatását az elválasztás
kerüli el: az app a `piper.exe`-t külön **subprocessként** hívja, így a GPL
kód nem linkelődik az (MIT) alkalmazáskódhoz. A repo saját kódja MIT.
