<#
===============================================================================
config/env.local.ps1 - M0 kornyezeti valtozok (V1 spec 18.1 tukrozes, M0
repo-gyoker-elvvel).

M0 elv (spec 26): a REPO-GYOKER az egyetlen mukodesi egyseg - a Root-ot
ezzel a fajllal egyutt a sajat helyebol szamoljuk ki (config\ -> szulo =
repo-gyoker), tehat a szkript GYOKER-FUGGETLEN: barmelyik meghajtora
klonozod a repot, minden utvonal automatikusan jo lesz.

Hasznalat (dot-source, hogy a valtozok az aktualis shellben landjanak):
    . .\config\env.local.ps1

FIGYELEM:
- A modellletoltes ELOTT ne allitsd be (a HF_HUB_OFFLINE=1 blokkolja a
  letoltest):
    1. scripts/download_models.ps1        # letoltes (online)
    2. . .\config\env.local.ps1           # utana offline mod
- Az OPENAI_* valtozok DUMMY ertekek: a hivatalos `openai` python lib
  technikai kotelezettsege (a voicemem csomag hasznalja), VALOS API kulcs
  NINCS - a vegepont a helyi llama-server (127.0.0.1:8080).
#>

# --- Root: a repo-gyoker (a config\ mappa szuloje - gepfuggetlen) -----------
$Root = Split-Path -Parent $PSScriptRoot

# ---------------------------------------------------------------------------
# HF cache izolacio (M0): a HuggingFace cache NEM a user-profilba
# (~/.cache/huggingface) kerul, hanem a projekt models/hf ala.
# ---------------------------------------------------------------------------
$env:HF_HOME = "$Root\models\hf"                    # HF cache gyoker
$env:HF_HUB_CACHE = "$Root\models\hf\hub"           # hub snapshotok/blobok
$env:TRANSFORMERS_CACHE = "$Root\models\hf\transformers"  # transformers cache

# ---------------------------------------------------------------------------
# Offline mod (spec 11.3 / 18.1) - a letoltesek UTAN allitsd be.
# Megakadalyozza, hogy a transformers/huggingface_hub futaskozben halozatot
# erjen (offline kovetelmeny: zero runtime network dependency).
# ---------------------------------------------------------------------------
$env:HF_HUB_OFFLINE = "1"             # huggingface_hub: nincs halozati hozzaferes
$env:TRANSFORMERS_OFFLINE = "1"       # transformers: nincs modell-letoltes futaskozben
$env:TOKENIZERS_PARALLELISM = "false" # tokenizer warning-elnyomas

# ---------------------------------------------------------------------------
# VoiceMem konfiguracio (a voicemem csomag ES az app AgentConfig is ezeket olvassa)
# ---------------------------------------------------------------------------
# VOICEMEM_HOME = a projektgyoker feluliras (OPCIONALIS - alapertelmezes
# mar a repo-gyoker; csak akkor allitsd, ha tenyleg mashova akarod tenni):
# $env:VOICEMEM_HOME = $Root
$env:VOICEMEM_MEMORY_ROOT = "$Root\memory"        # memoria-gyoker (sqlite/qdrant/backups)
# v0.4.4 FIELD FIX: a voicemem E5-feloldoja (hf_model()) CSAK flat layout-ban
# eszi meg a lokalis models\embedding\-ot (config.json / *.onnx a MAPPA 
# TETEJEN); a mi M0 elrendezesunk almappa (multilingual-e5-small) - nelkule
# HF repo id-re esett es az offline runtime miatt "could not connect to
# huggingface.co" hibaval minden memoria-kereses elbukott. Az env-feluliras
# a feloldo LEGMAGASABB prioritasu aga - a pontos lokalis utvonalat adjuk.
# (Az app oldalon a app/voicemem_bridge.pin_e5_local_model() ugyanezt
# allitja be, ha ez a fajl nem volt dot-source-olva.)
$env:VOICEMEM_E5_MODEL = "$Root\models\embedding\multilingual-e5-small"  # lokalis E5 (offline)
$env:VOICEMEM_EMBED_DIM = "384"                   # local E5 (multilingual-e5-small, 384 dim)
$env:OPENAI_BASE_URL = "http://127.0.0.1:8080/v1" # llama.cpp llama-server (OpenAI-kompatibilis, loopback)
$env:OPENAI_API_KEY = "not-needed-but-required-by-openai-lib"  # dummy: az openai lib kotelezoen keri, valos kulcs NINCS
# v0.4.16: az EGYETLEN LLM a Qwen3.6 35B A3B IQ4_XS (l. az LLM-MODELPROFIL
# blokkot lentebb; NINCS visszaallitasi profil).
$env:OPENAI_MODEL = "qwen3.6-35b-a3b"     # modellnev a /v1/chat/completions hivasokhoz
$env:TTS_BACKEND = "local"                        # Piper az OpenAI TTS helyett (lokalis)

# ---------------------------------------------------------------------------
# Modellutvonal-felulirasok (OPCIONALIS - kommentben: az AgentConfig
# property-defaultjai mar az M0 #7 elrendezesre mutatnak):
# models/asr/qwen3-asr-0.6b, models/tts/piper, models/vad/silero-vad,
# models/llm/qwen3.6-35b-a3b, models/embedding/multilingual-e5-small
# ---------------------------------------------------------------------------
# $env:QWEN3_ASR_MODEL_PATH = "$Root\models\asr\qwen3-asr-0.6b"
# $env:PIPER_VOICES_PATH    = "$Root\models\tts\piper"
# $env:SILERO_VAD_PATH      = "$Root\models\vad\silero-vad\silero_vad.onnx"
# $env:EMBEDDING_MODEL_PATH = "$Root\models\embedding\multilingual-e5-small"
# M2 (emotion2vec) - opcionalis feluliras (az M0/M1-ben kizart funkcio,
# M2 ota legalis; a default mar a jo helyre mutat):
# $env:EMOTION2VEC_PATH = "$Root\models\emotion\emotion2vec-plus-base"
# M3 (SpeechBrain ECAPA speaker) - opcionalis feluliras (ugyanez a minta):
# $env:SPEAKER_MODEL_PATH = "$Root\models\speaker\ecapa-voxceleb"

# ---------------------------------------------------------------------------
# llama.cpp szerver (scripts/start_llama_server.ps1 ezeket hasznalja)
# v0.4.14: az aktiv profil a Qwen3.6 35B A3B IQ4_XS - a modell ~19 GB
# (IQ4_XS MoE), ami NEM fer bele a 12 GB VRAM-ba, ezert a retegek egy resze
# a rendszer-RAM-bol (32 GB) streamel: LLAMA_N_GPU_LAYERS=26 reszleges
# offload (figyelem + KV cache + compute buffer GPU-n, a tobbi CPU-n).
# Ha a telepitett llama.cpp build tamogatja, a MoE-szintu split a jobb:
# --n-cpu-moe / --override-tensor "exps=CPU" (expert-tenzorok CPU-ra) -
# addig is a reteg-szintu offload a biztos, univerzalisan tamogatott ut.
# Kontextus 8K marad (a v0.4.5 ota a prompt-budget 14000 karakterben
# korlatozva; nem noveljuk feleslegesen).
# ---------------------------------------------------------------------------
$env:LLAMA_SERVER_HOST = "127.0.0.1"   # loopback: nincs kuls halozati expozicio
$env:LLAMA_SERVER_PORT = "8080"
# v0.4.16: PORTABLE MODELLUT - a GGUF barhol lehet a gepen (masik SSD/meghajto,
# Ollama-blob, HF-cache). A tenyleges fajl kivalasztasa:
#   (a) a web UI "LLM model" szekcioja (Browse... native Windows file picker)
#       -> config/llm_model.json abszolut utvonal (EZ a JSON feluliras
#       elsobbsegu ezzel a sorral szemben); vagy
#   (b) scripts\find_qwen_gguf.ps1 -SetEnv atirja ezt a sort a felderitett
#       utvonalra (.bak biztonsagi mentessel); vagy kezzel.
# A GGUF NEM masolodik a projektbe - a llama-server a sajat helyen olvassa.
# Az Ollama NEM resze a futasi kornyezetnek, csak egy lehetseges tarolasihely.
# v0.4.17: a fenti projekt-relative default UT KIKOMMENTEZVE - a celgepen
# nem letezik (a v0.4.15 "nem talalom a GGUF modellt" hiba oka pont ez
# a megteveszto sor volt). Forras lehet: config/llm_model.json (web UI /
# identify_ollama_blob.ps1 -Select), vagy EZ a sor egy tenyleges abszolut
# utvonalra atirva (a find_qwen_gguf.ps1 -SetEnv es az
# identify_ollama_blob.ps1 -SetEnv automatikusan atirja/behelyezi):
# $env:LLAMA_MODEL_PATH = "D:\AI\Models\Qwen3.6-35B-A3B-IQ4_XS.gguf"
$env:LLAMA_CONTEXT_SIZE = "8192"       # 8K kontextus (parbeszed + memory befer)
$env:LLAMA_N_GPU_LAYERS = "26"         # RESZLEGES offload: 35B IQ4_XS ~19 GB > 12 GB VRAM
$env:LLAMA_CACHE_TYPE_K = "q8_0"       # 8-bit KV cache (VRAM-sparelas)
$env:LLAMA_CACHE_TYPE_V = "q8_0"
$env:LLM_DISABLE_THINKING = "1"        # Qwen3.6 hibrid-reasoning: thinking csatorna KI
                                        # (sub-second elso token; a llama-server
                                        # --reasoning off + a request-szintu
                                        # chat_template_kwargs egyuttes biztositja)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# v0.4.16 LLM-MODELPROFIL (EGYETLEN profil: a Qwen3.6 35B A3B IQ4_XS)
# Az LLM cserje NEM kodszintu: par valtozo atallitasa csereli a modellt a
# teljes pipeline-ban (scripts/start_llama_server.ps1 inditas + minden
# /v1/chat/completions hivas). NINCS masodik profil, NINCS visszaallitasi
# opcio es NINCS felho-fallback: a Qwen3.6 35B A3B IQ4_XS az EGYETLEN
# produkcios LLM (LLM MODEL COUNT: 1). A websocket/turn architektura, a
# memory, az emotion es a TTS modellfuggetlen - azok nem valtoznak.
#
# --- AZ EGYETLEN profil: Qwen3.6 35B A3B IQ4_XS (v0.4.14-) ------------------
#     Ollama-ban elozetes tesztelve: termeszetes magyar parbeszed, rovid
#     konverzacios valaszok, HU<->EN nyelvvaltas. Tesztelt forras:
#     hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS
#     A GGUF helye: BARMELYIK meghajto - a web UI "LLM model" szekcioval
#     kivalasztva (config/llm_model.json), vagy ide a projektbe helyezve:
#     v0.4.17 OLLAMA-BLOB: a teszteles alatt hasznalt modell az Ollama
#     store-ban van (a blobs\ konyvtarban sha256-... nevvel, content-addressed
#     blob, NEM .gguf kiterjesztesu). A llama.cpp a GGUF MAGIC-et nezi, nem
#     a fajlnevet - a blob HELYBEN, masolas nelkul betoltheto. Azonosito:
#     scripts\identify_ollama_blob.ps1  (manifest+GGUF-header+digest;
#     -Select = ugyanaz, mint a web UI kivalasztas). FIGYELEM: ha semmi
#     nincs konfiguralva (llm_model.json/env/projekt-dir), NINCS default - a
#     starter tiszta hibaval megall es a pickert javasolja.
#   $env:OPENAI_MODEL       = "qwen3.6-35b-a3b"
#   $env:LLAMA_MODEL_PATH   = "<abszolut GGUF/blob utvonal - l. a kikommentezett peldat fent>"
#   $env:LLAMA_CONTEXT_SIZE = "8192"    # MoE A3B: 8K kontextus fut rajta
#   $env:LLAMA_N_GPU_LAYERS = "26"      # reszleges offload (12 GB VRAM < ~19 GB modell)
#   $env:LLM_DISABLE_THINKING = "1"     # hibrid-reasoning: thinking csatorna KI
# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
$env:CUDA_VISIBLE_DEVICES = "0"        # csak a 0-s GPU (a VAD/ASR nem conflictol)

# ---------------------------------------------------------------------------
# Qdrant embedded (a memoria hasznalja - lokalis fajl, nem szerver)
# ---------------------------------------------------------------------------
$env:QDRANT_PATH = "$Root\memory\qdrant"

# ---------------------------------------------------------------------------
# Telemetria KI (HF / mem0 / altalanos) - adat nem hagyhatja el a gepet
# ---------------------------------------------------------------------------
$env:DO_NOT_TRACK = "1"                # altalanos opt-out (HF+mas eszkozok)
$env:HF_HUB_DISABLE_TELEMETRY = "1"    # huggingface_hub telemetria
$env:MEM0_ANONYMIZED_TELEMETRY = "false"
$env:MEM0_TELEMETRY = "false"

# M2 elokeszulet (FunASR cache) - M0/M1-ben NEM hasznalatban, kommentben:
# $env:FUNASR_CACHE_DIR = "$Root\.funasr_cache"

Write-Host "env.local.ps1 betoltve. Root = $Root"
Write-Host "HF cache izolalva: HF_HOME = $Root\models\hf"
Write-Host "Offline mod BE (HF_HUB_OFFLINE=1) - modellletolteshez indits uj shell-t es NE dot-source-old ezt."
