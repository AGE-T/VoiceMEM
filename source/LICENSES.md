# LICENSES.md — VoiceMem M1/M2/M3 licenc-manifeszt

Ez a fájl az **M1, M2 és M3 mérföldkő minden komponensének** licenc-nyilvántartása.
A V1-terv követelménye (voicemem-v1-milestones.md 5. fejezet, "Licencpolitika"):
minden komponens **lokális**, **offline futó**, **PASS licencű** — nem lehet
fizetős, nem kereskedelmi (non-commercial), research-only, vagy kötelező
cloud-szolgáltatást igénylő. Az alábbi M1-táblázat a milestones 5.3-as M1
licenc-tábláját, az M2 kiegészítés az 5.4-es M2 táblát tükrözi (Python-
függőségekkel és a repo saját kódjával kiegészítve).

A kapott binárisokat és modellfájlokat a `scripts/download_models.ps1` (illetve a
kézi llama.cpp/Piper letöltés) helyezi el — a licencfájlokat a telepítés után
érdemes még egyszer ellenőrizni a forrásoldalakon.

## M1 komponens-licenc táblázat

| # | Komponens | Szerep az M1-ben | Kód licenc | Modell licenc | Hang licenc | Fizetős | Runtime cloud | Eredmény |
|---|-----------|------------------|------------|---------------|-------------|---------|---------------|----------|
| 1 | Silero VAD v6.2.1 (ONNX) | beszédfelismerés mikrofonon (VAD) | MIT | MIT | n/a | NEM | NEM | **PASS** |
| 2 | Qwen3-ASR-0.6B | beszéd → szöveg (GPU) | Apache-2.0 | Apache-2.0 | n/a | NEM | NEM | **PASS** |
| 3 | Gemma 4 12B QAT Q4_0 (GGUF) | válaszgenerálás (LLM, v0.4.3: az EGYETLEN LLM) | n/a (Google kiadás) | Gemma Terms of Use (google/gemma-4-12B-it-qat-q4_0-gguf) | n/a | NEM | NEM | **PASS** |
| 4 | llama.cpp (llama-server.exe bináris, CUDA 13.3 prebuilt) | LLM backend szerver | MIT | n/a | n/a | NEM | NEM | **PASS** |
| 5 | Piper engine (piper.exe bináris, subprocess) | szöveg → beszéd (TTS) | GPL-3.0-or-later | n/a | n/a | NEM | NEM | **PASS** |
| 6 | Piper hangok: hu_HU anna, hu_HU berta, hu_HU imre, en_US lessac (medium) | TTS hangok | n/a | n/a | MIT | NEM | NEM | **PASS** |
| 7 | multilingual-e5-small | memória-embedding (CPU) | Apache-2.0 | Apache-2.0 | n/a | NEM | NEM | **PASS** |
| 8 | VoiceMem (VEZERLT FORK: vendor/voicemem, alap upstream v0.0.1 = e8384e0; l. VOICEMEM_PIN.json/UPSTREAM_POLICY.md) | hosszú távú memória-keretrendszer | Apache-2.0 | Apache-2.0 (LICENSE megtartva a fork-ban) | n/a | NEM | NEM | **PASS** |
| 9 | mem0ai (a VoiceMem telepíti) | memória-réteg | Apache-2.0 | n/a | n/a | NEM | NEM | **PASS** |
| 10 | Qdrant (embedded, a mem0 használja) | vektor tároló | Apache-2.0 | n/a | n/a | NEM | NEM | **PASS** |
| 11 | onnxruntime (CPU build) | VAD futtatás (ONNX) | MIT | n/a | n/a | NEM | NEM | **PASS** |
| 12 | PyTorch 2.7.0 (cu128 wheel) | ASR-inferencia (GPU) | BSD-3-Clause | n/a | n/a | NEM | NEM | **PASS** |
| 13 | sounddevice (PortAudio wrapper) | mikrofon/hangszóró I/O | MIT | n/a | n/a | NEM | NEM | **PASS** |
| 14 | sentence-transformers | e5-small embedder betöltés | Apache-2.0 | n/a | n/a | NEM | NEM | **PASS** |
| 15 | transformers | Qwen3-ASR betöltés | Apache-2.0 | n/a | n/a | NEM | NEM | **PASS** |
| 16 | voicemem-agent repo saját kódja (app/, tests/, scripts/, config/) | az ügynök maga | MIT (repo saját kódja) | n/a | n/a | NEM | NEM | **PASS** |

## M2 kiegészítő komponens-licenc táblázat (hozzáadott, milestones 5.4)

| # | Komponens | Szerep az M2-ben | Kód licenc | Modell licenc | Hang licenc | Fizetős | Runtime cloud | Eredmény |
|---|-----------|------------------|------------|---------------|-------------|---------|---------------|----------|
| 17 | emotion2vec_plus_base (emotion2vec/emotion2vec_plus_base, ~90M) | proszódia-alapú érzelemfelismerés (CPU) | n/a (modell) | **FunASR Model Open Source License Agreement v1.1** — ingyenes használat/copy/megosztás **attribution mellett** (forrás + szerző megnevezése, modellnév megtartása) | n/a | NEM | NEM | **PASS** (attribution teljesítve: ez a táblázat + a README M2-fejezete) |
| 18 | funasr (pip-csomag, 1.4.11) | emotion2vec+ futtató (AutoModel, CPU) | MIT (a FunASR repó LICENSE fájlja) | n/a | n/a | NEM | NEM | **PASS** |
| 19 | funasr közvetett függőségek (modelscope, librosa, oss2, kaldiio, jieba stb.) | telepítési-idejű csomagok (runtime-ban nem importáljuk őket) | vegyes (MIT/Apache-2.0/BSD) | n/a | n/a | NEM | NEM | **PASS** (kizárólag lokálisan települnek; hálózati hívást az app nem kezdeményez) |
| 20 | Semantikus fúzió + adaptív TTS (app/emotion.py, pipeline length_scale) | saját kód | MIT (repo saját kódja) | n/a | n/a | NEM | NEM | **PASS** |

**M2 licenc-eredmény: 100% PASS** (a milestones 5.4-es táblájával egyezően —
1 új modell + új pip-csomag, mind PASS; az emotion2vec+ attribution-kötelezettsége
ebben a fájlban és a README-ben teljesül).

## M3 kiegészítő komponens-licenc táblázat (hozzáadott, milestones 5.5)

| # | Komponens | Szerep az M3-ban | Kód licenc | Modell licenc | Hang licenc | Fizetős | Runtime cloud | Eredmény |
|---|-----------|------------------|------------|---------------|-------------|---------|---------------|----------|
| 21 | SpeechBrain spkrec-ecapa-voxceleb (ECAPA-TDNN, ~23M, 192-dim) | beszélő-beágyazás (CPU) | n/a (modell) | **Apache-2.0** (a HF modellkártya licence, hivatalos speechbrain-szervezeti re-host, NEM gated) | n/a | NEM | NEM | **PASS** |
| 22 | speechbrain (pip-csomag, 1.1.1) | ECAPA futtató (SpeakerRecognition interfész, CPU) | **Apache-2.0** | n/a | n/a | NEM | NEM | **PASS** |
| 23 | speechbrain közvetett függőségek (hyperpyyaml, joblib, scipy, sentencepiece, soundfile, tqdm) | telepítési-idejű csomagok (a futtató interfész használja őket) | vegyes (MIT/Apache-2.0/BSD) | n/a | n/a | NEM | NEM | **PASS** (kizárólag lokálisan települnek; hálózati hívást az app nem kezdeményez — a modell lokális könyvtárból töltődik) |
| 24 | Beszélő-regisztráció + memória-útválasztás (app/speaker.py, bridge per-user facade) | saját kód | MIT (repo saját kódja) | n/a | n/a | NEM | NEM | **PASS** |

**M3 licenc-eredmény: 100% PASS** (a milestones 5.5-ös táblájával egyezően —
1 új modell + 1 új pip-csomag, mind Apache-2.0, NEM gated, offline futó).

**M1 licenc-eredmény: 100% PASS, 0 CONDITIONAL, 0 FAIL** (a milestones 5.6
összegzésével egyezően: 11 M1-komponens, mind PASS — itt a Python-függőségekkel
és a saját kóddal kibontva). Az M1+M2 összesített eredmény szintén **100% PASS**
(l. a fenti M2 kiegészítő táblázatot).

## Megjegyzések

- **Piper engine (5. sor)**: az egyetlen copyleft (GPL-3.0-or-later) licenc a
  készletben. A copyleft **hatását elválasztással kerüljük el**: az app a
  `piper.exe`-t külön subprocessként hívja (`app/tts.py`), a GPL kód nem
  linkelődik, nem is importoltatik az MIT app-kóddal — így a mi kódunkra a GPL
  nem terjed át. Emiatt NEM veszünk fel `piper-tts` Python-csomagot függőségként
  (lásd requirements.txt).
- **multilingual-e5-small (7. sor)**: a HF-modellkártya Apache-2.0-t jelöl; a
  milestones 5.3-as táblázata a referencia-implementáció kódjára hivatkozva
  MIT-ként szerepelteti — üzleti szempontból mindkettő megfelel (permissive).
- **Qwen modellek (2-3. sor)**: Apache-2.0, NEM a Qwen Research License — ez volt
  az összetevőválasztás egyik fő szempontja (NC/research-only kizárás).
- **VoiceMem (8. sor)**: commit-SHA-pinning — a licencállapot a pinált SHA-val
  rögzített; a friss `main` ág licenc-változását a heti review nézi (spec 21.2).
- **mem0 telemetria**: kikapcsolva (`MEM0_ANONYMIZED_TELEMETRY=false`,
  `MEM0_TELEMETRY=false` a `config/env.local.ps1`-ben).

## Előretekintés (V2+)

Az alábbi komponensek **NEM részei az M1-nek/M2-nek/M3-nak** — saját
licenc-ellenőrzést kapnak a saját mérföldkövükben:

- **V2 — MOSS-TTS Realtime, Gemma audio, Qwen Omni**: mindegyik saját ellenőrzést
  kap (a milestones 5.5-ös M3-sora a speechbrain-ECAPA-t már leszállította —
  l. fent az M3 táblázatot); a pyannote-audio továbbra is kizárt (HF gated access
  sérti az offline követelményt).

---

*Ellenőrzés dátuma: 2026-08-30 (M1, spec audit) + M2-emotion2vec+ bővítés a
milestones 5.4 alapján + M3-speechbrain bővítés a milestones 5.5 alapján. Emlékeztető: minden ÚJ komponens (M3) telepítésekor ezt a
táblázatot ki kell egészíteni, és a release-rituálé részeként újra PASS-ra kell
minősíteni. Az emotion2vec+ attribution: „This repo uses emotion2vec+ base
(emotion2vec/emotion2vec_plus_base) under the FunASR Model Open Source License
Agreement v1.1 (Alibaba Group).”*
