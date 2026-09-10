<#
===============================================================================
scripts/start_llama_server.ps1 - llama.cpp llama-server indito (KONFIGVEZERELT)

MIT csinal: elinditja a bin\llama-server.exe-t a konfiguralt LLM
GGUF-fel (v0.4.16-tol az EGYETLEN LLM a Qwen3.6 35B A3B IQ4_XS - nincs
visszaesi profil, l. config/env.local.ps1 LLM-MODELPROFIL; a modell NEM
hardcode-olt: a feloldasi sorrend config/llm_model.json (web UI model
picker kivalasztasa) > LLAMA_MODEL_PATH env > a models\llm\qwen3.6-35b-a3b\
konyvtarban tenylegesen levo GGUF), MEGVARJA A /health 200-AT, majd az
eloterben felugyeli a folyamatot (Ctrl+C allitja le). A szerver
OpenAI-kompatibilis /v1/chat/completions endpointot ad (SSE streaming) a
127.0.0.1:8080 cimen - ez fut az agent (scripts\start_agent.ps1) mellett,
MASIK ablakban (vagy az agent auto-inditja).

v0.4.17 OLLAMA-BLOB TAMOGATAS + BETOLTES-BIZONYITEK:
  - a modell lehet OLLAMA CONTENT-ADDRESSED BLOB is
    (pl. C:\AI_HOME\models\blobs\sha256-afc7238...): a llama.cpp a GGUF
    MAGIC-et nezi, NEM a fajlnevet, tehat a blob HELYBEN, atnevezes es
    masolas nelkul toltheto; az azonosito szkript:
    scripts\identify_ollama_blob.ps1 (-Select = ugyanaz, mint a web UI
    kivalasztas);
  - a /health 200 meg NEM bizonyitja a betoltott fajlt: a starter harom
    fuggetlen BIZONYITEKET futtat es a
    logs\llama-server.resolved-model.json markerbe ir (a web UI kiirja):
      [1] GET /v1/models served id,
      [2] a szerver SAJAT log-soraban szereplo pontos --model utvonal,
      [3] egy tenyleges /v1/chat/completions round-trip;
  - "nincs konfiguralva modell" allapot: a v0.4.16-os megteveszto
    projekt-default utvonal helyett most KOZVETLEN, tisztan lathato hiba
    + a harom javitasi ut (picker / identify / finder).

v0.3.1 FIX (llama.cpp b10717 field report): a regi szkript a --grammar-json
CLI argumentumot adta at, ami a telepitett b10717 buildben NEM letezik
("error: invalid argument: --grammar-json"), es a szerver azonnal kilepett.
A JSON-valaszt most NEM server-flag kezeli, hanem REQUEST SZINTEN, mert:
  - a VoiceMem _llm_json() es az app/llm.py chat_json() a
    POST /v1/chat/completions hivas response_format mezojet hasznalja
    ({"type":"json_object"} ill. {"type":"json_schema","json_schema":{...}}),
    es a llama-server ezt KERES-KORREKTEN kotezi GBNF constrained
    generation-re - tehat a JSON-mod mar request-szintu, globalis flag NINCS
    szuksege ra;
  - a globalis flag amugy is rossz architektura lenne: a sima (nem JSON)
    streaming valaszokat is JSON-ra kotezne.
  (A b10717 CLI-n letezo server-szintu alternativak: --grammar,
  --grammar-file, -j/--json-schema, -jf/--json-schema-file - ezeket szandekosan
  NEM hasznaljuk.)

v0.3.2 FIX (celgepi field report #2) - ket kulon hiba kep:
  (a) KEZI inditasnal "not digitally signed": a bongeszobol letoltott
      release ZIP-bol kicsomagolt .ps1 fajlok Internet-zona jelzest
      (Zone.Identifier / Mark of the Web) orokolhetnek, es a normal
      RemoteSigned policy blokkolja oket. A START.bat lanca mindent
      -ExecutionPolicy Bypass-szal futtat; kezi inditashoz a lenti
      Bypass-parancs, vagy:
      Get-ChildItem .\scripts\*.ps1 | Unblock-File
      (a bootstrap.ps1 / install_m1.ps1 / verify_m1.ps1 ezt minden futtas
      automatikusan elvegzi.)
  (b) A verify -WithServer altal inditott starter 2 s alatt NEMA modon
      kilepett (ures log): ha a bin\ CUDA DLL-keszlet (ggml-cuda/cudart/
      cublas) hianyos, a llama-server.exe STATUS_DLL_NOT_FOUND-tal
      (0xC0000135) hal meg STDERR KIMENET NELKUL - a hibauzenet eddig a
      minimizalt gyerekablakban veszett el. v0.3.2:
      (1) nev szerinti DLL-elozetes-ellenorzes,
      (2) a szkript SAJAT kimenete Start-Transcript-tel a
          logs\llama-server.starter.log fajlban (rejtett inditasnal is
          utolag olvashato),
      (3) a gyerekfolyamat exit kodjanak dekodolasa a FAIL uzenetben
          (0xC0000135 = hianyzo DLL),
      (4) az installer DLL-tudatosan ujraletolti a binarisokat.

v0.3.3 FIX (celgepi field report #3): a KEZI tesztek bizonyitottak, hogy a
  llama-server maga teljesen rendben van (llama-server.exe --version PASS,
  modellbetoltes PASS, /health -> HTTP 200, 127.0.0.1:8080 listening PASS)
  - a hiba a verify/starter INDITAS-DETEKTALASABAN es EXIT-KOD-KEZELESEBEN
  volt. Ezerv0.3.3:
  (a) a python-bridge (config\voicemem_config.yaml -> JSON) hibaja mar NEM
      fatal: FIGYELEM + KOZVETLEN (shell) konfiguracioval folytatunk. Az
      ertekek a config alapertekeivel AZONOSAK (+ az AgentConfig.apply_env()
      altal olvasott env-valtozok ugyanazok):
        LLAMA_SERVER_HOST / LLAMA_SERVER_PORT / LLAMA_MODEL_PATH /
        LLAMA_N_GPU_LAYERS / LLAMA_CONTEXT_SIZE / LLAMA_CACHE_TYPE_K /
        LLAMA_CACHE_TYPE_V
      A llama-server MEGHIVASA VALTOZATLAN: ugyanaz az exe, ugyanaz a
      parancssor (-ngl -1, -c 8192, --parallel 1, q8_0 KV cache, --temp 0.7,
      --metrics, --no-webui), ugyanaz a viselkedes.
  (b) a .venv hianya es a DLL-elozetes-ellenorzes talalata is csak FIGYELEM
      (nem fatal): a nev-alapu DLL-ellenorzes becsles - a vegso igazsag a
      llama-server sajat mukodese; ha a DLL-hiany miatt hal meg, az exit-kod
      dekodolasa (0xC0000135 = STATUS_DLL_NOT_FOUND) kiirja a tenyleges okot.
  (c) a verify_m1.ps1 (v0.3.3) mar NEM a starter exit kodjabol itelkezik: az
      elsodleges sikerfeltetel a GET /health -> HTTP 200; FAIL csak akkor,
      ha a llama-server folyamat tenylegesen leallt ES a /health nem valt
      elerhetove a timeouton belul.

KONFIGURACIO-VEZERELT: a szerver MINDEN parametere a config\voicemem_config.yaml
-bol valtozik ki. PowerShellben nincs nativ YAML, ezert PYTHON-BRIDGE: a
.venv pythonja az app.config.AgentConfig.from_yaml segitsegevel JSON-re forditja
a konfigot, a szkript pedig ConvertFrom-Json-nel veszi fel. A bridge a
repo-gyokerbol fut (Push-Location $Root), a YAML-relativ utvonal miatt.

LLAMA_* env-valtozok: ha a PS-sessionben mar be vannak allitva (pl. a
config\env.local.ps1 dot-source-olasa utan), azok az AgentConfig.apply_env()
-ben AUTOMATIKUSAN feluliraskent alkalmazodnak - a gyerekfolyamat orokli a
PS env-jet, igy ehhez a szkripthez SEMMI teendo nincs (a bridge mar
figyelembe veszi oket).

Miert ezek a flagek (mind a b10717 --help szerint tamogatott):
  --cache-type-k/-v q8_0 : 8-bit KV cache - a 8K kontextus VRAM-igenyet
                     kb. a felere csokkenti, minosegromlas nelkul.
  -ngl -1          : minden reteg GPU-ra (RTX 5070, sm_120 native CUDA build).
  -c 8192          : 8K kontextus-ablak (a 12B instruct modellhez is eleg).
  --parallel 1     : parhuzamos szerver-slotok szama (config: llm_parallel).
  --temp 0.7       : mintaveteli homerseklet (config: llm_temperature).
  --metrics        : /metrics endpoint a benchmarkokhoz (llm_benchmark.py).
  --no-webui       : nem kell a beepitett webes felulet.
  --reasoning off  : (v0.4.4) a chat-handler enable_thinking defaultja
                    TRUE lenne; a hibrid-reasoning modell gondolkodasi
                    csatornaja ilyenkor elnyeli a teljes valaszt (ures content).
                    Az off minden klienstol vegig tartja a gyors,
                    nem-gondolkodo valaszmodot.
  --host 127.0.0.1 : loopback - kulso halozati expozicio nincs (offline garancia).

ROBUSZTUSSAG (v0.3.1 - "ne csak javitsd, hanem tedd robusztussra"):
  1. projektgyoker meghatarozasa (-Root kapcsolove allithato)
  2. config betoltese a python-bridge-en keresztul (v0.3.3: hiba eseten
     FIGYELEM + kozvetlen shell-konfiguracio - NEM fatal)
  3. modellfajl ellenorzese
  4. llama-server.exe ellenorzese (fallback: $env:LLAMA_BIN)
  5. runtime DLL-ellenorzes a bin\ mappaban (v0.3.3: csak FIGYELEM - a
     vegso igazsag a llama-server sajat mukodese / exit-kodja)
  6. server inditasa GYEREKFOLYAMATKENT, stdout/stdlog atiranyitassal a
     logs\llama-server.out.log es logs\llama-server.err.log fajlokba
  7. /health ellenorzes (alap: max 90 s, -WaitSec kapcsolovel allithato)
  8. ha a folyamat INDULASKOR kilep: AZONNAL kiirjuk a stdout/stderr
     utolso sorait es FAIL-le lepunk ki - nincs 60 s-os vak varakozas
  8b. a szkript SAJAT kimenetet Start-Transcript naplozza a
      logs\llama-server.starter.log fajlba - rejtett/minimizalt ablakbol
      inditva (verify -WithServer, measure_vram) is utolag olvashato
  9. siker: egyertelmuen PASS + a szerver felugyelese (Ctrl+C = tiszta
     leallitas, a gyerekfolyamat is leall)
  10. sikertelenseeg: egyertelmuen FAIL + a tenyleges hibauzenet

VRAM megjegyzes (v0.4.14, Qwen3.6 35B A3B IQ4_XS): a modell ~19 GB
(IQ4_XS MoE), ami NEM fer a 12 GB VRAM-ba - LLAMA_N_GPU_LAYERS=26
reszleges offload (figyelem + KV cache q8_0 + compute buffer GPU-n, a
tobbseg rendszer-RAM-bol streamel). A TENYLEGES merest a
scripts\measure_vram.ps1 vegzi. Ha a tamogatott MoE-szintu split
(--n-cpu-moe / --override-tensor "exps=CPU") elerheto a buildben, az a
finomabb beallitas.

v0.4.16 LLM POLZIKA: a Qwen3.6 35B A3B IQ4_XS az EGYETLEN LLM (LLM MODEL
COUNT: 1, FALLBACK: NINCS - nincs masodik profil, nincs felho, nincs
Ollama runtime). Ha a Qwen GGUF nem talalhato, az LLM ERROR allapot
kovetkezik - NEM valtunk masik modellre, nincs rejtett fallback; a
kivalasztas a web UI "LLM model" pickerrel vagy a
scripts\find_qwen_gguf.ps1 keresovel tortenik.

Hasznalat (Bypass forma - policy-fuggetlen, MOTW-ellen is mindig mukodik):
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_llama_server.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_llama_server.ps1 -WaitSec 120
Rovid forma (csak ha a session policy engedi es a fajl nincs MOTW-jelezve):
    .\scripts\start_llama_server.ps1
    .\scripts\start_llama_server.ps1 -WaitSec 120     # hosszabb varakozas
    .\scripts\start_llama_server.ps1 -Root "C:\masik_root"   # VOICEMEM_HOME feluliras

Megjegyzes: a fajl szandekosan ASCII - a PowerShell 5.1 BOM nelkul
Windows-1252-kent olvasna az ekezetes karaktereket.
#>
param(
    [string]$Root = "",
    [int]$WaitSec = 150
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

# ---------------------------------------------------------------------------
# .venv feloldasa (a python-bridge es a konfig-feloldas erdekeben).
# v0.3.3: a .venv hianya mar NEM fatal - a llama-server nem szukseges a
# pythonra: a bridge kihagyasaval, KOZVETLEN (shell) konfiguracioval indul.
# ---------------------------------------------------------------------------
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "FIGYELEM: nem talalhato a .venv python: $VenvPython" -ForegroundColor Yellow
    Write-Host "         A python-bridge kihagyva - KOZVETLEN (shell) konfiguracioval folytatunk" -ForegroundColor Yellow
    Write-Host "         (a llama-server maga nem szukseges a pythonra)." -ForegroundColor Yellow
    Write-Host "         JAVITAS kesobbre: powershell -ExecutionPolicy Bypass -File scripts\install_m1.ps1" -ForegroundColor Yellow
}

# Explicit -Root eseten a VOICEMEM_HOME-on keresztul felulirjuk a konfig
# gyokeret (az AgentConfig a VOICEMEM_HOME-ot olvassa az apply_env-ben).
$env:VOICEMEM_HOME = $Root

# ---------------------------------------------------------------------------
# v0.3.2: a szkript SAJAT kimenetenek naplozasa (Start-Transcript).
# Hatterben / minimizalt ablakbol inditva (verify_m1 -WithServer,
# measure_vram.ps1) a konzolkimenet NEM lathato - eddig a tenyleges
# hibauzenet (pl. hianyzo CUDA DLL) a gyerekablakban veszett el. A
# transcript a logs\llama-server.starter.log fajlba irodik, es a verify /
# start_agent / measure_vram hibaelagaban is megjelenik. A tovabbi
# teljes futas try/finally-ba van zarva: minden exit elott a
# Stop-Transcript lefut, a log mindig zarva, olvashato marad.
# ---------------------------------------------------------------------------
$LogsDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
$StarterLog = Join-Path $LogsDir "llama-server.starter.log"
# v0.4.17: a szerver stdout/stderr utvonalait mar ITT definialjuk (a v0.4.16
# verzio csak a "mar fut" ag utan definialta oket) - a betoltes-BIZONYITEK
# (a szerver sajat log-soranak keresese) mar az idempotens agban is fut.
$OutLog = Join-Path $LogsDir "llama-server.out.log"
$ErrLog = Join-Path $LogsDir "llama-server.err.log"
try { Start-Transcript -Path $StarterLog -ErrorAction SilentlyContinue | Out-Null } catch { }
try {

# ---------------------------------------------------------------------------
# PYTHON-BRIDGE: config\voicemem_config.yaml -> JSON
# (az llm_parallel uj config-mezo - getattr-csel turelmes, ha meg nincs)
# v0.3.3 (celgepi field report #3): ha a bridge BARMILYEN okbol hibazik
# (pl. az app.config verzioja, hianyzo PyYAML, .venv-hiany), az mar NEM
# fatal: FIGYELEM + KOZVETLEN (shell) konfiguracioval folytatunk - a
# config alapertekeivel AZONOS ertekekkel (l. a fejlec v0.3.3 blokkja).
# ---------------------------------------------------------------------------
$Cfg = $null
if (-not (Test-Path $VenvPython)) {
    Write-Host "FIGYELEM: nem talalhato a .venv python: $VenvPython" -ForegroundColor Yellow
    Write-Host "         A python-bridge kihagyva - KOZVETLEN (shell) konfiguracioval folytatunk." -ForegroundColor Yellow
} else {
    $Bridge = @'
import json
import os
import sys

root = os.environ.get("VOICEMEM_HOME", os.getcwd())
if root and root not in sys.path:
    sys.path.insert(0, root)
from app.config import AgentConfig

c = AgentConfig.from_yaml(os.path.join(root, "config", "voicemem_config.yaml"))
print(json.dumps({
    "exe": str(c.piper_exe_path.parent),
    "model": (str(c.llm_model_file) if c.llm_model_file else ""),
    "host": c.llama_server_host,
    "port": c.llama_server_port,
    "ctx": c.llm_context_size,
    "ngl": c.llm_n_gpu_layers,
    "parallel": getattr(c, "llm_parallel", 1),
    "temp": c.llm_temperature,
    "ck_k": c.llm_cache_type_k,
    "ck_v": c.llm_cache_type_v,
}))
'@
    # v0.3.4 (field report #4): at `python -c $Bridge` the PowerShell 5.1
    # native argument passing STRIPS the inner double quotes of the Python
    # code (r"config\..." arrived as rconfig\... -> SyntaxError), so the
    # bridge now runs from an ATOMIC temp .py file instead - a file's
    # content has NO command-line quoting, this works on every PS version
    # (5.1 and 7+). sys.path is secured through BOTH PYTHONPATH and the
    # VOICEMEM_HOME env (both = $Root), and the YAML path is absolute.
    $BridgePy = Join-Path ([System.IO.Path]::GetTempPath()) ("voicemem_cfg_bridge_{0}.py" -f $PID)
    Set-Content -Path $BridgePy -Value $Bridge -Encoding ASCII
    $PrevPyPath = [string]$env:PYTHONPATH
    try {
        if ([string]::IsNullOrWhiteSpace($PrevPyPath)) { $env:PYTHONPATH = $Root } else { $env:PYTHONPATH = "$Root;$PrevPyPath" }
        Push-Location $Root
        $CfgJson = & $VenvPython $BridgePy 2>&1
        $BridgeExit = $LASTEXITCODE
    } finally {
        try { Pop-Location } catch { }
        $env:PYTHONPATH = $PrevPyPath
        Remove-Item -Path $BridgePy -Force -ErrorAction SilentlyContinue
    }
    if ($BridgeExit -ne 0) {
        Write-Host "FIGYELEM: a konfiguracio (config\voicemem_config.yaml) nem toltheto be a python-bridge-en keresztul." -ForegroundColor Yellow
        foreach ($Line in @($CfgJson)) { Write-Host ("    {0}" -f [string]$Line) }
        Write-Host "         A hiba NEM fatal (v0.3.3): a llama-server a KOZVETLEN (shell) konfiguracioval indul." -ForegroundColor Yellow
        Write-Host "         (JAVITAS kesobbre, repo-gyokerbol: .venv\Scripts\python.exe -m app.main --list-config)" -ForegroundColor Yellow
    } else {
        $JsonLine = (@($CfgJson) | Where-Object { ([string]$_).Trim().StartsWith('{') }) | Select-Object -First 1
        if ([string]$JsonLine -ne "") {
            $Cfg = $JsonLine | ConvertFrom-Json
        } else {
            Write-Host "FIGYELEM: a python-bridge kimeneteben nincs ertelmezheto JSON - kozvetlen konfiguracio." -ForegroundColor Yellow
        }
    }
}
if ($null -eq $Cfg) {
    # --- KOZVETLEN (shell) konfiguracio: a config alapertekeivel AZONOS
    # ertekek (+ az apply_env altal olvasott LLAMA_* env felulirasok).
    # A llama-server meghivasa (flagek, modell, -ngl -1) VALTOZATLAN.
    $FbHost = "127.0.0.1"
    if ($env:LLAMA_SERVER_HOST) { $FbHost = [string]$env:LLAMA_SERVER_HOST }
    $FbPort = "8080"
    if ($env:LLAMA_SERVER_PORT) { $FbPort = [string]$env:LLAMA_SERVER_PORT }
    $FbModel = Join-Path $Root "models\llm\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-IQ4_XS.gguf"
    if ($env:LLAMA_MODEL_PATH) { $FbModel = $env:LLAMA_MODEL_PATH }
    $FbNgl = "26"   # v0.4.14: aktiv profil = 35B IQ4_XS (~19 GB) > 12 GB VRAM -> partial offload
    if ($env:LLAMA_N_GPU_LAYERS) { $FbNgl = [string]$env:LLAMA_N_GPU_LAYERS }
    $FbCtx = "8192"
    if ($env:LLAMA_CONTEXT_SIZE) { $FbCtx = [string]$env:LLAMA_CONTEXT_SIZE }
    $FbCkK = "q8_0"
    if ($env:LLAMA_CACHE_TYPE_K) { $FbCkK = [string]$env:LLAMA_CACHE_TYPE_K }
    $FbCkV = "q8_0"
    if ($env:LLAMA_CACHE_TYPE_V) { $FbCkV = [string]$env:LLAMA_CACHE_TYPE_V }
    $Cfg = New-Object PSObject -Property @{
        exe = (Join-Path $Root "bin")
        model = $FbModel
        host = $FbHost
        port = $FbPort
        ctx = $FbCtx
        ngl = $FbNgl
        parallel = "1"
        temp = "0.7"
        ck_k = $FbCkK
        ck_v = $FbCkV
    }
    Write-Host ""
    Write-Host "Konfiguracio: KOZVETLEN (shell) - a config\voicemem_config.yaml alapertekei:"
} else {
    Write-Host ""
    Write-Host "Konfiguracio a config\voicemem_config.yaml-bol (python-bridge, env felulirasokkal):"
}
# ---------------------------------------------------------------------------
# v0.4.16: PORTABLE MODELLUT - a config/llm_model.json-ben tartositott
# kivalasztas (a web UI \"LLM model\" szekcio) FELULIRJA a bridge/env altal
# feloldott utvonalat. A python-bridge mar ugyanezt a sorrendet alkalmazza
# (AgentConfig.llm_model_file: JSON > LLAMA_MODEL_PATH env > default); itt a
# KOZVETLEN (shell) konfiguracios ut es az egyformasag miatt fut meg egyszer.
# A GGUF NEM masolodik - a llama-server a sajat helyen, barmelyik meghajton
# olvassa.
# ---------------------------------------------------------------------------
$LlmModelJson = Join-Path $Root "config\llm_model.json"
$ModelSource = "env/profile (python-bridge)"
if (Test-Path $LlmModelJson) {
    try {
        $LlmSel = Get-Content -LiteralPath $LlmModelJson -Raw -Encoding UTF8 | ConvertFrom-Json
        $LlmSelPath = [string]$LlmSel.llm_model_path
        if (-not [string]::IsNullOrWhiteSpace($LlmSelPath)) {
            if ([string]$Cfg.model -ne $LlmSelPath) {
                Write-Host ("MODELLUT FELULIRAS: a web UI kivalasztasa ert ervenyesben ({0})" -f $LlmSelPath) -ForegroundColor Cyan
                Write-Host ("        (config/llm_model.json; korabbi feloldas: {0})" -f $Cfg.model)
            }
            $Cfg.model = $LlmSelPath
            $ModelSource = "user selection (config/llm_model.json)"
        }
    } catch {
        Write-Host "FIGYELEM: a config/llm_model.json nem olvashato - a profil utvonal marad." -ForegroundColor Yellow
    }
}
Write-Host ("    modell : {0}" -f $Cfg.model)
Write-Host ("    cim    : http://{0}:{1} (loopback)" -f $Cfg.host, $Cfg.port)
Write-Host ("    ctx    : {0} | ngl: {1} | parallel: {2} | temp: {3}" -f $Cfg.ctx, $Cfg.ngl, $Cfg.parallel, $Cfg.temp)
Write-Host ("    KV     : k={0} v={1}" -f $Cfg.ck_k, $Cfg.ck_v)
Write-Host ("    modellut forrasa: {0}" -f $ModelSource)

# ---------------------------------------------------------------------------
# A bin-konyvtar (a configban feloldott exe-k helye) a PATH elejere kerul -
# igy a llama-server megtalalja a melle hallo DLL-eket a bin\ mappaban.
# ---------------------------------------------------------------------------
$BinDir = [string]$Cfg.exe
if ($BinDir -ne "" -and (Test-Path $BinDir)) {
    if (-not ($env:PATH.StartsWith($BinDir))) {
        $env:PATH = "$BinDir;$env:PATH"
    }
}

# ---------------------------------------------------------------------------
# llama-server.exe feloldasa: $Root\bin\llama-server.exe, fallback $env:LLAMA_BIN
# ---------------------------------------------------------------------------
$LlamaServer = Join-Path $Root "bin\llama-server.exe"
if (-not (Test-Path $LlamaServer)) {
    if ($env:LLAMA_BIN) {
        $Candidate = Join-Path $env:LLAMA_BIN "llama-server.exe"
        if (Test-Path $Candidate) {
            $LlamaServer = $Candidate
        }
    }
}

# ---------------------------------------------------------------------------
# Repulo-ellenorzesek (pre-flight) - vilagos hibauzenet + javitasi javaslat
# ---------------------------------------------------------------------------
if (-not (Test-Path $LlamaServer)) {
    Write-Host "HIBA: nem talalom a llama-server.exe-t." -ForegroundColor Red
    Write-Host "Keresett utvonal : $LlamaServer"
    Write-Host "JAVITAS: a binarisokat a telepito tolti le (pin-elt llama.cpp release):" -ForegroundColor Yellow
    Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\install_m1.ps1"
    Write-Host "    (vagy kezzel: https://github.com/ggml-org/llama.cpp/releases ->"
    Write-Host "     llama-bXXXX-bin-win-cuda-13.3-x64.zip + cudart-llama-bin-win-cuda-13.3-x64.zip"
    Write-Host "     kicsomagolasa a bin\ mappaba)"
    exit 1
}

$ModelPath = [string]$Cfg.model
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    Write-Host "HIBA: nincs konfiguralva LLM modell." -ForegroundColor Red
    Write-Host "A feloldasi sorrend: config\llm_model.json (web UI kivalasztas) >"
    Write-Host "LLAMA_MODEL_PATH env > a models\llm\qwen3.6-35b-a3b\ konyvtarban"
    Write-Host "tenylegesen levo GGUF fajl. JAVITAS - harom ut (a modell NEM" -ForegroundColor Yellow
    Write-Host "masolodik automatikusan):"
    Write-Host "  (a) a web UI \"LLM model\" szekcioja (Browse... gomb): barmelyik"
    Write-Host "      meghajtorol kivalasztod a GGUF fajlot (Ollama sha256-... blob is)"
    Write-Host "      - az abszolut ut a config/llm_model.json-be kerul, majd"
    Write-Host "      ujrainditod a START.bat-ot;"
    Write-Host "  (b) az Ollama-blob azonosito (v0.4.17): manifest + GGUF-header +"
    Write-Host "      digeriten azonositja a PONTOS sha256-... blobot, es -Select-tel"
    Write-Host "      kivalasztja (ugyanaz, mint a web UI kivalasztas):"
    Write-Host "      powershell -NoProfile -ExecutionPolicy Bypass -File scripts\identify_ollama_blob.ps1 -Select"
    Write-Host "  (c) az altalanos pontos-modell kereso (Ollama-store / HF-cache /"
    Write-Host "      letoltesek):"
    Write-Host "      powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1"
    Write-Host "  Tesztelt forras: hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF (IQ4_XS tag)."
    Write-Host "  NINCS visszaesi profil - a Qwen3.6 az egyetlen produkcios LLM."
    exit 1
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    Write-Host "HIBA: nem talalom a GGUF modellt." -ForegroundColor Red
    Write-Host "Keresett utvonal : $ModelPath"
    Write-Host "v0.4.17 JAVITAS - harom ut (a modell NEM masolodik automatikusan):" -ForegroundColor Yellow
    Write-Host "  (a) a web UI \"LLM model\" szekcioja (Browse... gomb): barmelyik"
    Write-Host "      meghajtorol kivalasztod a GGUF fajlot (Ollama sha256-... blob is)"
    Write-Host "      - az abszolut ut a config/llm_model.json-be kerul, majd"
    Write-Host "      ujrainditod a START.bat-ot;"
    Write-Host "  (b) az Ollama-blob azonosito (v0.4.17): manifest + GGUF-header +"
    Write-Host "      digeriten azonositja a PONTOS sha256-... blobot, es -Select-tel"
    Write-Host "      kivalasztja (ugyanaz, mint a web UI kivalasztas):"
    Write-Host "      powershell -NoProfile -ExecutionPolicy Bypass -File scripts\identify_ollama_blob.ps1 -Select"
    Write-Host "  (c) az altalanos pontos-modell kereso (Ollama-store / HF-cache /"
    Write-Host "      letoltesek):"
    Write-Host "      powershell -NoProfile -ExecutionPolicy Bypass -File scripts\find_qwen_gguf.ps1"
    Write-Host "  (d) kezzel helyezz EGY GGUF fajlot a models\llm\qwen3.6-35b-a3b\"
    Write-Host "      konyvtarba (tetszoleges fajlnev - a GGUF magic dont, nem a nev)."
    Write-Host "  Tesztelt forras: hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF (IQ4_XS tag)."
    Write-Host "  NINCS visszaesi profil - a Qwen3.6 az egyetlen produkcios LLM."
    exit 1
}

# --- DLL-ellenorzes (v0.3.2: NEV SZERINT, nem csak "van-e valami .dll") ---
#     A CUDA build FAJLGESZ DLL-keszlete (a ket pin-elt zip egyuttesen):
#       ggml-cuda*.dll : a CUDA backend      (llama-b...-bin-win-cuda zip)
#       cudart*.dll    : a CUDA runtime      (cudart-llama-bin-win-cuda zip)
#       cublas*.dll    : a cuBLAS            (cudart-llama-bin-win-cuda zip)
#     Ha ezekbol barmelyik hianyzik, a llama-server.exe INDULASKOR
#     AZONNAL, NEMA modon hal meg (STATUS_DLL_NOT_FOUND, 0xC0000135) - a
#     stderr-log URES marad! Ezt elore kiszurni a celgepi talalgatas helyett.
$LlamaBinDir = Split-Path -Parent $LlamaServer
$BinDlls = @(Get-ChildItem -Path $LlamaBinDir -Filter "*.dll" -ErrorAction SilentlyContinue)
$DllNames = @($BinDlls | ForEach-Object { $_.Name })
$HasGgmlCuda = (@($DllNames | Where-Object { $_ -like "ggml-cuda*" })).Count -gt 0
$HasCudart   = (@($DllNames | Where-Object { $_ -like "cudart*" })).Count -gt 0
$HasCublas   = (@($DllNames | Where-Object { $_ -like "cublas*" })).Count -gt 0
if ($BinDlls.Count -eq 0) {
    Write-Host "FIGYELEM: nem talalhato EGYETLEN .dll sem a bin mappaban - a CUDA" -ForegroundColor Yellow
    Write-Host "         build (llama-bXXXX-bin-win-cuda + cudart-llama-bin-win-cuda) a runtime"
    Write-Host "         DLL-ek (cudart/cublas/ggml-cuda) nelkul nem tud elindulni."
    Write-Host ("Keresett mappa : {0}" -f $LlamaBinDir)
    Write-Host "MEGJEGYZES (v0.3.3): ez mar NEM fatal - megprobaljuk inditani; ha a" -ForegroundColor Yellow
    Write-Host "         DLL-hiany miatt hal meg, az exit-kod dekodolasa mutatja (0xC0000135)."
    Write-Host "JAVITAS (ha megall): a telepito ujraletolti a binarisokat:" -ForegroundColor Yellow
    Write-Host "    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_m1.ps1 -SkipModels"
} elseif (-not ($HasGgmlCuda -and $HasCudart -and $HasCublas)) {
    $DllMissing = @()
    if (-not $HasGgmlCuda) { $DllMissing += "ggml-cuda*.dll (CUDA backend - llama-bin-win-cuda zip)" }
    if (-not $HasCudart)   { $DllMissing += "cudart*.dll (CUDA runtime - cudart-llama zip)" }
    if (-not $HasCublas)   { $DllMissing += "cublas*.dll (cuBLAS - cudart-llama zip)" }
    Write-Host "FIGYELEM: a bin mappa CUDA DLL-keszlete feltehetoleg HIANYZOS:" -ForegroundColor Yellow
    foreach ($D in $DllMissing) { Write-Host ("       hianyzik : {0}" -f $D) }
    Write-Host ("Mappa : {0} ({1} db .dll talalhato)" -f $LlamaBinDir, $BinDlls.Count)
    Write-Host "MEGJEGYZES (v0.3.3): ez mar NEM fatal (a nev-alapu ellenorzes becsles -" -ForegroundColor Yellow
    Write-Host "         pl. mas DLL-keszlettel epulo binary is mukodhet) - megprobaljuk"
    Write-Host "         inditani; ha a DLL-hiany miatt hal meg, az exit-kod dekodolasa"
    Write-Host "         mutatja (0xC0000135 = STATUS_DLL_NOT_FOUND, ures log)."
    Write-Host "JAVITAS (ha megall): a telepito v0.3.2+ DLL-tudatosan ujraletolti a binarisokat:" -ForegroundColor Yellow
    Write-Host "    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_m1.ps1 -SkipModels"
}
Write-Host ("DLL-ellenorzes : {0} db .dll (ggml-cuda={1} cudart={2} cublas={3})" -f $BinDlls.Count, $HasGgmlCuda, $HasCudart, $HasCublas)

# ---------------------------------------------------------------------------
# /health segedfv. + "mar fut" ellenorzes (idempotens inditas)
# v0.4.16: az idempotens ugras most a BETOLTOTT MODELLT is ellenorzi - a
# /v1/models valaszbol kiolvassuk a szerver altal szolgalt modell-azonositojat
# es osszehasonlitjuk a feloldott GGUF-fajlnevvel + az OPENAI_MODEL-lel. Ha a
# futtato szerver MAS modellt toltoott be (a v0.4.16 elotti leggyakoribb eset:
# egy regi peldany marad futva a :8080-on a korabbi verzio modelljevel), a
# starter LEALLITJA es a
# HELYES modellel ujrainditja - kulonben a web UI egesz nap a regen modellt
# hasznalna anelkul, hogy barmi hibara utalna (field report 2026-09-05: egy
# v0.4.16-elotti inditas maradvanya futott meg a :8080-on).
# ---------------------------------------------------------------------------
function Test-LlamaHealth {
    param([string]$H, [int]$P, [int]$TimeoutSec = 3)
    try {
        $R = Invoke-WebRequest -Uri ("http://{0}:{1}/health" -f $H, $P) -Method Get -TimeoutSec $TimeoutSec -UseBasicParsing
        if (($R.StatusCode -ge 200) -and ($R.StatusCode -lt 300)) { return $true }
    } catch { }
    return $false
}

function Get-LlamaServedModel {
    param([string]$H, [int]$P)
    try {
        $M = Invoke-RestMethod -Uri ("http://{0}:{1}/v1/models" -f $H, $P) -Method Get -TimeoutSec 5
        return [string]$M.data[0].id
    } catch { return "" }
}

function Test-LlamaServesModel {
    param([string]$ServedId, [string]$ExpectedModelPath, [string]$ExpectedName)
    if ([string]::IsNullOrWhiteSpace($ServedId)) { return "unknown" }
    $Norm = {
        param([string]$S)
        return ([string]$S).ToLower() -replace '[^0-9a-z]', ''
    }
    $ServedN = & $Norm $ServedId
    # v0.4.17: egy Ollama blob (sha256-afc723...) neve NINCS kiterjesztes -
    # a GetFileNameWithoutExtension a TELJES nevet adja (sha256-afc723...),
    # teart az osszehasonlitas az ilyen blobokra is mukodik.
    $StemN = & $Norm ([System.IO.Path]::GetFileNameWithoutExtension($ExpectedModelPath))
    $NameN = & $Norm $ExpectedName
    if ($ServedN.Length -eq 0) { return "unknown" }
    if ($StemN.Length -gt 0 -and (($ServedN.Contains($StemN)) -or ($StemN.Contains($ServedN)))) { return "same" }
    if ($NameN.Length -gt 0 -and (($ServedN.Contains($NameN)) -or ($NameN.Contains($ServedN)))) { return "same" }
    return "different"
}

# ---------------------------------------------------------------------------
# v0.4.17: BETOLTES-BIZONYITEK. A felhasznalo kovetelmenye: "prove that
# llama-server actually loads the exact identified file - verify the actual
# # running llama-server process and its loaded model". Harom fuggetlen
# bizonyitek, mind a markerbe irva (a web UI \"LLM model\" szekcioja kiirja):
#   [1] GET /v1/models      - a szerver ALTAL szolgaltatott modell-azonosito
#   [2] a szerver SAJAT logja - llama-server.out.log tartalmazza a pontos
#       --model utvonalat (a betolteskor maga a szerver irja ki)
#   [3] /v1/chat/completions round-trip - tenyleges generacio (nem csak
#       \"health 200\")
# ---------------------------------------------------------------------------
function Get-LlamaLoadLogLine {
    param([string]$ModelPath, [string]$OutLog, [string]$ErrLog)
    foreach ($log in @($OutLog, $ErrLog)) {
        if (-not (Test-Path -LiteralPath $log)) { continue }
        try {
            $hit = Select-String -LiteralPath $log -SimpleMatch $ModelPath -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -ne $hit) {
                $line = [string]$hit.Line
                if ($line.Length -gt 200) { $line = $line.Substring(0, 200) }
                return @{ found = $true; line = $line.Trim(); log = $log }
            }
        } catch { }
    }
    return @{ found = $false; line = ""; log = "" }
}

function Invoke-LlamaCompletionProbe {
    param([string]$H, [int]$P, [int]$TimeoutSec = 90)
    $out = @{ ok = $false; reply = "" }
    try {
        $Body = @{
            messages = @(@{ role = "user"; content = "Reply with the single word OK." })
            max_tokens = 16
            temperature = 0
            stream = $false
        } | ConvertTo-Json -Depth 4
        $R = Invoke-RestMethod -Uri ("http://{0}:{1}/v1/chat/completions" -f $H, $P) -Method Post -Body $Body -ContentType "application/json" -TimeoutSec $TimeoutSec
        $txt = ""
        try { $txt = [string]$R.choices[0].message.content } catch { }
        if (-not [string]::IsNullOrWhiteSpace($txt)) {
            $out.ok = $true
            if ($txt.Length -gt 60) { $txt = $txt.Substring(0, 60) }
            $out.reply = $txt.Trim()
        } else {
            $out.reply = "empty completion content"
        }
    } catch {
        $out.reply = ("probe failed: {0}" -f $_.Exception.Message)
    }
    return $out
}

function Invoke-LlamaLoadProof {
    param([string]$ModelPath)
    Write-Host ""
    Write-Host "BETOLTES-BIZONYITEK (a FUTTO szerver tenyleg ezt a fajlt tolti-e):"
    $ServedId = Get-LlamaServedModel -H $Cfg.host -P $Cfg.port
    if (-not [string]::IsNullOrWhiteSpace($ServedId)) {
        Write-Host ("  [1] GET /v1/models   -> served id : {0}" -f $ServedId)
    } else {
        Write-Host "  [1] GET /v1/models   -> a modell-azonosito nem olvashato ki"
    }
    $LogHit = Get-LlamaLoadLogLine -ModelPath $ModelPath -OutLog $OutLog -ErrLog $ErrLog
    if ($LogHit.found) {
        Write-Host "  [2] a szerver SAJAT logja tartalmazza a pontos utvonalat:" -ForegroundColor Green
        Write-Host ("      {0}" -f $LogHit.line)
    } else {
        Write-Host "  [2] a szerver logja nem tartalmazza az utvonalat (a log inditaskor" -ForegroundColor Yellow
        Write-Host "      felulirodhatott; az [1] es [3] bizonyitek attol meg ervenyesek)"
    }
    $Comp = Invoke-LlamaCompletionProbe -H $Cfg.host -P $Cfg.port
    if ($Comp.ok) {
        Write-Host ("  [3] /v1/chat/completions round-trip OK - valasz: {0}" -f $Comp.reply) -ForegroundColor Green
    } else {
        Write-Host ("  [3] /v1/chat/completions round-trip NEM sikerult: {0}" -f $Comp.reply) -ForegroundColor Yellow
    }
    return @{
        served = [string]$ServedId
        log_found = [bool]$LogHit.found
        log_line = [string]$LogHit.line
        completion_ok = [bool]$Comp.ok
        completion_reply = [string]$Comp.reply
    }
}

function Write-LlamaResolvedMarker {
    param([string]$ModelPath, [string]$ServedId, [string]$ModelSource,
          [bool]$LogFound, [string]$LogLine, [bool]$CompletionOk, [string]$CompletionReply)
    try {
        $Marker = @{
            model_path = [string]$ModelPath
            served_model_id = [string]$ServedId
            model_name = [string]$env:OPENAI_MODEL
            source = [string]$ModelSource
            resolved_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
            log_path_found = [bool]$LogFound
            log_line = [string]$LogLine
            completion_ok = [bool]$CompletionOk
            completion_reply = [string]$CompletionReply
            verified_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
        } | ConvertTo-Json -Compress
        Set-Content -LiteralPath (Join-Path $LogsDir "llama-server.resolved-model.json") -Value $Marker -Encoding UTF8
    } catch { }
}

function Stop-LlamaServerOnPort {
    param([int]$P)
    $Stopped = $false
    try {
        $Conns = @(Get-NetTCPConnection -LocalPort $P -State Listen -ErrorAction SilentlyContinue)
        foreach ($Conn in $Conns) {
            $OwnerPid = [int]$Conn.OwningProcess
            if ($OwnerPid -gt 0) {
                try {
                    $Proc = Get-Process -Id $OwnerPid -ErrorAction Stop
                    if ($Proc.ProcessName -like "llama*") {
                        Write-Host ("    leallitas: llama-server PID {0} ({1})" -f $OwnerPid, $Proc.ProcessName) -ForegroundColor Yellow
                        Stop-Process -Id $OwnerPid -Force -ErrorAction SilentlyContinue
                        $Stopped = $true
                    } else {
                        Write-Host ("    FIGYELEM: a {0} porton a '{1}' folyamat figyel (PID {2}) - NEM allitom le automatikusan." -f $P, $Proc.ProcessName, $OwnerPid) -ForegroundColor Yellow
                        Write-Host "    Allitsd le kezzel, majd futtasd ujra a START.bat-ot." -ForegroundColor Yellow
                    }
                } catch { }
            }
        }
    } catch {
        Write-Host ("    FIGYELEM: nem tudtam lekerdezni a {0} port tulajdonosat - kezi leallitas szukseges." -f $P) -ForegroundColor Yellow
    }
    if ($Stopped) {
        # varjuk meg, hogy a port felszabaduljon (max 15 s)
        $WaitEnd = (Get-Date).AddSeconds(15)
        while ((Get-Date) -lt $WaitEnd) {
            if (-not (Test-LlamaHealth -H $Cfg.host -P $P -TimeoutSec 1)) { break }
            Start-Sleep -Milliseconds 500
        }
    }
    return $Stopped
}

if (Test-LlamaHealth -H $Cfg.host -P $Cfg.port -TimeoutSec 3) {
    $ServedId = Get-LlamaServedModel -H $Cfg.host -P $Cfg.port
    $ServesOk = Test-LlamaServesModel -ServedId $ServedId -ExpectedModelPath ([string]$Cfg.model) -ExpectedName ([string]$env:OPENAI_MODEL)
    if ($ServesOk -eq "same") {
        Write-Host ""
        Write-Host ("PASS: a llama-server mar fut a http://{0}:{1} cimen (/health 200)." -f $Cfg.host, $Cfg.port) -ForegroundColor Green
        Write-Host ("      Betoltott modell: {0} - megegyezik a feloldott GGUF-fel, nem inditok masodik peldanyt." -f $ServedId)
        # v0.4.17: betoltes-bizonyitek (served id + a szerver sajat log-sora +
        # completion round-trip) es a KITERJESZTETT marker irasa - a web UI
        # "LLM model" szekcioja ebbol mutatja meg, hogy a FUTTO szerver a
        # kivalasztott fajlt tolti-e.
        $Proof = Invoke-LlamaLoadProof -ModelPath ([string]$Cfg.model)
        Write-LlamaResolvedMarker -ModelPath ([string]$Cfg.model) -ServedId $Proof.served `
            -ModelSource $ModelSource -LogFound $Proof.log_found -LogLine $Proof.log_line `
            -CompletionOk $Proof.completion_ok -CompletionReply $Proof.completion_reply
        exit 0
    } elseif ($ServesOk -eq "different") {
        Write-Host ""
        Write-Host "FIGYELEM: a llama-server mar fut, de MAS MODELLT toltott be!" -ForegroundColor Yellow
        Write-Host ("      Futtato szerver modellje : {0}" -f $ServedId)
        Write-Host ("      Feloldott (kivant) modell: {0}" -f $Cfg.model)
        Write-Host "      A v0.4.16 autojavitas: a regi peldany LEALLITASA es ujrainditas a helyes modellel..."
        $null = Stop-LlamaServerOnPort -P ([int]$Cfg.port)
        if (Test-LlamaHealth -H $Cfg.host -P $Cfg.port -TimeoutSec 3) {
            Write-Host "HIBA: a regi llama-server allitoan meg mindig fut - allitsd le kezzel es ujrainditast." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host ""
        Write-Host ("PASS: a llama-server mar fut a http://{0}:{1} cimen (/health 200; a modell-azonosito nem olvashato ki)." -f $Cfg.host, $Cfg.port) -ForegroundColor Green
        Write-Host "      Nem inditok masodik peldanyt (idempotens inditas)."
        # v0.4.17: bizonyitek itt is (best effort - az id nem olvashato ki,
        # de a log-sor es a completion round-trip meg mutathatja a betoltott fajlt).
        $Proof = Invoke-LlamaLoadProof -ModelPath ([string]$Cfg.model)
        Write-LlamaResolvedMarker -ModelPath ([string]$Cfg.model) -ServedId $Proof.served `
            -ModelSource $ModelSource -LogFound $Proof.log_found -LogLine $Proof.log_line `
            -CompletionOk $Proof.completion_ok -CompletionReply $Proof.completion_reply
        exit 0
    }
}

# ---------------------------------------------------------------------------
# Logfajlok: a szerver stdout/stderr ide irodik (indulasi hibanal ez az
# egyertelmen olvashato bizonyitek - nem csak egy eltuno ablak).
# (v0.4.17: a $LogsDir / $OutLog / $ErrLog definicio a transcript-blokkba
# kerult, hogy a fenti idempotens ag is hasznalni tudja oket.)
# ---------------------------------------------------------------------------

function Show-LogTail {
    param([string]$Path, [string]$Label, [int]$N = 25)
    if (-not (Test-Path $Path)) { return }
    $Tail = @(Get-Content -Path $Path -Tail $N -ErrorAction SilentlyContinue)
    if ($Tail.Count -gt 0) {
        Write-Host ""
        Write-Host $Label -ForegroundColor Yellow
        foreach ($Line in $Tail) { Write-Host ("    {0}" -f $Line) }
    }
}

# ---------------------------------------------------------------------------
# Parancssor osszeallitasa (a b10717 --help szerint tamogatott flagek CSAK),
# majd inditas gyerekfolyamatkent naplo-atiranyitassal.
# FONTOS: --grammar-json NINCS - a JSON-mod REQUEST szintu (response_format),
# l. a fajl fejlecat. Ha egy jovo build mas flageket kernene, a
# bin\llama-server.exe --help mutatja a tamogatott lista minden elemet.
# ---------------------------------------------------------------------------
$ModelPathQ = '"' + $ModelPath + '"'
$LlamaArgs = @(
    "--model", $ModelPathQ,
    "--host", [string]$Cfg.host,
    "--port", [string]$Cfg.port,
    "-ngl", [string]$Cfg.ngl,
    "-c", [string]$Cfg.ctx,
    "--parallel", [string]$Cfg.parallel,
    "--cache-type-k", [string]$Cfg.ck_k,
    "--cache-type-v", [string]$Cfg.ck_v,
    "--temp", [string]$Cfg.temp,
    "--metrics",
    "--no-webui",
    # v0.4.4: a b10717 llama-server chat-handler enable_thinking DEFAULTJA
    # TRUE - a hibrid-reasoning sablonok ilyenkor NYITVA hagyjak a thought
    # csatornat a generacios promptban, a modell a gondolkodasi csatornaban
    # valaszol, a szerver azt reasoning_content-be kuldi es a content URES
    # marad (a v0.4.3-as "minden valasz 0 karakter" mezo-mode kieses).
    # --reasoning off = szerveroldali default MINDEN kliensre (a voicemem
    # csomag sajat OpenAI-lib hivasait is vedi, amik nem tudnak
    # request-szintu kwargs-ot kuldeni). Az app raadasul request-szintu
    # chat_template_kwargs / reasoning_effort "none"-t is kuld
    # (app/llm.py _thinking_control_kwargs) - a ket mechanizmus ugyanazt
    # az enable_thinking=false allapotot allitja be, nem utkozik.
    "--reasoning", "off"
)

Write-Host ""
Write-Host "llama-server inditasa:"
Write-Host "  exe    : $LlamaServer"
Write-Host "  modell : $ModelPath"
Write-Host "  log    : $OutLog"
Write-Host "          $ErrLog"
Write-Host "Teljes parancssor:"
Write-Host ("  `"{0}`" {1}" -f $LlamaServer, ($LlamaArgs -join " "))
Write-Host "Leallitas: Ctrl+C"
Write-Host ""

# ---------------------------------------------------------------------------
# v0.4.16: RESOLVED-MODEL MARKER - a pontos feloldott --model utvonalat
# irjuk a logs/llama-server.resolved-model.json-be. A web UI "LLM model"
# szekcioja ezt olvassa vissza es mutatja, hogy a FUTTO szerver tenyleg
# ezt a fajlt tolti-e be ("llama-server loads the same file").
# ---------------------------------------------------------------------------
try {
    $Marker = @{
        model_path = [string]$ModelPath
        model_name = [string]$env:OPENAI_MODEL
        source = $ModelSource
        resolved_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    } | ConvertTo-Json -Compress
    Set-Content -LiteralPath (Join-Path $LogsDir "llama-server.resolved-model.json") -Value $Marker -Encoding UTF8
} catch { }

$Proc = Start-Process -FilePath $LlamaServer `
    -ArgumentList $LlamaArgs `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru
$StartedAt = Get-Date

# ---------------------------------------------------------------------------
# /health varakozas GYORS HIBAJELENTEssel: ha a folyamat kilep (pl. ervenytelen
# CLI argumentum, hianyzo DLL, foglalt port), NEM varunk tovabb - azonnal
# kiirjuk a tenyleges stdout/stderr hibauzenetet es FAIL-le lepunk ki.
# ---------------------------------------------------------------------------
$Healthy = $false
while ($true) {
    if ($Proc.HasExited) { break }
    if (Test-LlamaHealth -H $Cfg.host -P $Cfg.port -TimeoutSec 2) { $Healthy = $true; break }
    if (((Get-Date) - $StartedAt).TotalSeconds -ge $WaitSec) { break }
    Start-Sleep -Milliseconds 500
}

if ($Healthy) {
    $UpSec = [math]::Round(((Get-Date) - $StartedAt).TotalSeconds, 1)
    Write-Host ""
    Write-Host ("PASS: a llama-server felfutott {0} s alatt - /health 200 a http://{1}:{2} cimen." -f $UpSec, $Cfg.host, $Cfg.port) -ForegroundColor Green
    Write-Host "      OpenAI-kompatibilis endpointok:"
    Write-Host ("        POST http://{0}:{1}/v1/chat/completions  (SSE streaming + response_format" -f $Cfg.host, $Cfg.port)
    Write-Host "             json_object / json_schema -> llama.cpp constrained generation)"
    Write-Host ("        GET  http://{0}:{1}/health   GET /metrics (a --metrics miatt)" -f $Cfg.host, $Cfg.port)
    Write-Host "      JSON-mod: REQUEST szintu (response_format mezo) - server-flag nincs ra."
    # v0.4.17: BETOLTES-BIZONYITEK - a /health 200 meg NEM bizonyitja, hogy a
    # szerver a PONTOS kivant fajlt toltotte be. Harom fuggetlen bizonyitek:
    # a /v1/models served id, a szerver SAJAT log-sora (tartalmazza a pontos
    # --model utvonalat), es egy tenyleges /v1/chat/completions round-trip.
    # Az eredmeny a KITERJESZTETT markerbe iratik (felulirja az inditas elott
    # irt alap-markert) - a web UI es a verify_m1 ebbol olvassa.
    $Proof = Invoke-LlamaLoadProof -ModelPath $ModelPath
    Write-LlamaResolvedMarker -ModelPath $ModelPath -ServedId $Proof.served `
        -ModelSource $ModelSource -LogFound $Proof.log_found -LogLine $Proof.log_line `
        -CompletionOk $Proof.completion_ok -CompletionReply $Proof.completion_reply
    Write-Host ""
    Write-Host "A szerver az eloterben fut, ez a szkript felugyeli. Ctrl+C -> tiszta leallitas."
    Write-Host ""
    try {
        Wait-Process -Id $Proc.Id -ErrorAction SilentlyContinue
    } finally {
        if (-not $Proc.HasExited) {
            Write-Host "llama-server leallitasa (Ctrl+C)..."
            Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    # A felugyelet vegen: ha a szerver magatol kilepett, mutatjuk miert.
    try { $null = $Proc.WaitForExit(2000) } catch { }
    if ($Proc.HasExited) {
        $ExitCode = 0
        try { $ExitCode = [int]$Proc.ExitCode } catch { $ExitCode = 1 }
        if ($ExitCode -ne 0) {
            Write-Host ("FIGYELEM: a llama-server kilepett (exit kod: {0})." -f $ExitCode) -ForegroundColor Yellow
            Show-LogTail -Path $ErrLog -Label "A szerver utolso hibauzenetei (stderr):"
            exit $ExitCode
        }
    }
    exit 0
}

if ($Proc.HasExited) {
    # GYORS HIBAJELZES: a szerver mar kilepett - kiirjuk a tenyleges hibat.
    # v0.3.2: az exit kod DEKODOLASA - a 0xC0000135 (hianyzo DLL) es a
    # 0xC0000139 (DLL verzio-osszeferes) NEM irnak semmit a stderr-re,
    # csak a kod mondja meg a tenyleges okot.
    $ExitCode = 1
    try { $ExitCode = [int]$Proc.ExitCode } catch { $ExitCode = 1 }
    $ExitReason = ""
    if ($ExitCode -eq -1073741515) {
        $ExitReason = " <- STATUS_DLL_NOT_FOUND (0xC0000135): HIANYZO DLL (cudart/cublas/ggml-cuda)"
    } elseif ($ExitCode -eq -1073741519) {
        $ExitReason = " <- STATUS_ENTRYPOINT_NOT_FOUND (0xC0000139): DLL verzio-osszeferes"
    } elseif ($ExitCode -eq -1073741819) {
        $ExitReason = " <- STATUS_ACCESS_VIOLATION (0xC0000005)"
    }
    Write-Host ""
    Write-Host ("FAIL: a llama-server.exe az inditas utan azonnal kilepett (exit kod: {0}){1}." -f $ExitCode, $ExitReason) -ForegroundColor Red
    Show-LogTail -Path $ErrLog -Label "A szerver tenyleges hibauzenete (stderr, utolso sorok):"
    Show-LogTail -Path $OutLog -Label "A szerver stdout-ja (utolso sorok):"
    Write-Host ""
    Write-Host "TIPPEK a leggyakoribb okokra:" -ForegroundColor Yellow
    Write-Host "  - a szkript teljes sajat kimenete: logs\llama-server.starter.log (transcript)"
    Write-Host "  - ervenytelen CLI argumentum: futtasd kezzel a"
    Write-Host ("      `"{0}`" --help" -f $LlamaServer)
    Write-Host "    parancsot, es hasonlitsd ossze a tamogatott flagekkel."
    Write-Host "  - hianyzo CUDA DLL: telepitsd ujra a binarisokat (install_m1.ps1 -SkipModels)."
    Write-Host ("  - foglalt port: valami mas figyel a {0}:{1} cimen (netstat -ano | findstr {1})." -f $Cfg.host, $Cfg.port)
    exit 1
}

# Idotullepes: a folyamat meg fut, de nem valaszolt a /health-re.
try { Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue } catch { }
Write-Host ""
Write-Host ("FAIL: a llama-server {0} s alatt nem valaszolt a /health-re (a folyamat fut, de nem ready)." -f $WaitSec) -ForegroundColor Red
Show-LogTail -Path $ErrLog -Label "A szerver stderr-e (utolso sorok):"
Show-LogTail -Path $OutLog -Label "A szerver stdout-ja (utolso sorok):"
Write-Host ""
Write-Host "TIPP: a modellbetoltes lassu lehet - probald ujra -WaitSec 240 kapcsolovel." -ForegroundColor Yellow
exit 1

# ---------------------------------------------------------------------------
# (v0.3.2) A transcript lezarasa: minden fenti exit elott a finally blokk
# lefut - a logs\llama-server.starter.log igy mindig teljes, zarott,
# utolag olvashato hibanyom (a verify_m1 / start_agent / measure_vram
# pontosan ezt mutatja a hibaelagaban).
# ---------------------------------------------------------------------------
} finally {
    try { Stop-Transcript -ErrorAction SilentlyContinue | Out-Null } catch { }
}
