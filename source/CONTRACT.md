# VoiceMem M1 — Implementation Contract (Task 7)

This file is the **single source of truth** for module interfaces. All agents must code
against these exact signatures. `app/config.py` is already implemented — read it first.

## Project context

- Goal: **Milestone 1** of the VoiceMem V1 plan — a fully local HU/EN voice agent:
  `mic → VAD → ASR → VoiceMem memory → local LLM → local TTS → speaker`,
  with barge-in (loopback gating), long-term memory, fully offline, zero paid components.
- Full component spec: `/home/z/my-project/voicemem-v1-spec.md` (read Sections 2, 14, 17, 18, 19 as needed).
- Milestone plan: `/home/z/my-project/voicemem-v1-milestones.md` (read Sections 2, 3.3, 4.4, 6, 10).
- Target runtime: Windows 11 + RTX 5070. This sandbox is Linux without GPU/audio — all heavy
  dependencies (torch, onnxruntime, sounddevice, transformers, the `voicemem` package) are
  **NOT installable here**. Code must be importable and unit-testable WITHOUT them.
- Development sandbox CAN rely on: Python 3.12, numpy, PyYAML, httpx.

## Hard conventions (all agents)

1. **Python 3.11+ compatible syntax**, full type hints, `from __future__ import annotations` at top.
2. **Lazy heavy imports**: `torch`, `transformers`, `onnxruntime`, `sounddevice`, `soundfile`,
   `voicemem`, `sentence_transformers` — import ONLY inside functions/methods or guarded by
   `try/except ImportError` at module level with a `_HAS_X` flag. A module that imports a heavy
   dep at top level will fail the sandbox test suite.
   `numpy`, `yaml`, `httpx` are allowed at module top level (available in sandbox + target env).
3. **Pure-stdlib modules**: `app/text_utils.py`, `app/barge_in.py`, `app/teacher_persona.py`,
   `app/config.py` must import NOTHING beyond the Python standard library.
4. **Tests**: stdlib `unittest` only (no pytest). Every test file ends with:
   `if __name__ == "__main__": unittest.main()`.
   All tests MUST pass in this sandbox with zero heavy deps installed, run from the
   `voicemem-agent/` repo root: `python3 -m unittest discover -s tests -t . -v`
   (test packages: `tests/unit`, `tests/integration`, `tests/benchmark` — see the
   M0 addendum at the bottom of this file).
5. **Logging**: `logging.getLogger(__name__)`, never `print` in library modules (print is OK in
   `main.py` CLI output and benchmark scripts).
6. **No emojis** in code or comments. English identifiers; concise English docstrings
   (the docs/README are Hungarian — code stays English).
7. **M1 exclusion policy**: NO emotion recognition, NO speaker recognition, NO scene/music
   recognition anywhere in the M1 code. The only allowed traces of M2/M3 are:
   - `config.enable_emotion: bool = False` and `config.enable_speaker: bool = False` flags (already in config.py)
   - `teacher_persona.build_system_prompt(...)` accepts optional `emotion_label/valence/arousal`
     params which are ALWAYS `None` in M1 (prompt omits the block when `None`)
   - `speaker_id: str = "voice_user"` parameter on memory/pipeline APIs (M3 hook, fixed value in M1)
   - `config.voicemem_mode` must stay `"text_mode"` (never `"multi_modal"`) — `validate()` enforces this.
8. **Repo-root paths (M0)**: use `pathlib.Path`, forward slashes OK. The default
   root is THE REPO ROOT itself (`app/config.py: default_root()` — parent of
   `app/`), overridable by the `VOICEMEM_HOME` env var. No machine-global
   locations: models/memory/data/logs all live under the repo (M0 §7/§26).
9. **Every public class/function gets a docstring.** Keep modules under ~400 lines.

## Shared dataclasses & enums

```python
# app/pipeline.py (owned by 7-a)
@dataclass
class TurnTimings:
    speech_end_s: Optional[float] = None      # VAD hangover end (t0 of latency measurement)
    asr_final_s: Optional[float] = None
    memory_s: Optional[float] = None          # VoiceMem retrieval duration
    first_token_s: Optional[float] = None
    first_audio_s: Optional[float] = None
    full_response_s: Optional[float] = None
    def to_dict(self) -> dict[str, Optional[float]]: ...

@dataclass
class TurnResult:
    transcript: str
    reply: str
    language: str                              # "hu" | "en" (reply language)
    timings: TurnTimings
    cancelled: bool = False                    # barge-in aborted this turn's TTS
    barge_in: bool = False
```

```python
# app/vad.py (owned by 7-a) — pure part
class VadEvent(Enum):     # NONE, SPEECH_START, SPEECH_END
```

## Module interfaces (VERBATIM — do not change signatures)

### app/text_utils.py — 7-b — PURE STDLIB

```python
LANG_HU = "hu"
LANG_EN = "en"

def detect_language(text: str) -> str:
    """Heuristic HU/EN detection. Hungarian diacritic-bearing tokens weigh 3,
    HU stopwords weigh 2, EN stopwords 1. Ties -> 'hu' iff any HU diacritic else 'en'."""

def split_sentences(text: str) -> list[str]:
    """Sentence split keeping abbreviations (pl. "Dr.", "Mr.", "stb.", "pl.", "vs.")
    and decimal numbers intact. Strips whitespace-only results."""

class SentenceStream:
    """Buffers streaming LLM deltas and emits speakable text chunks."""
    def __init__(self, first_chunk_chars: int = 24, chunk_chars: int = 80) -> None: ...
    def add_delta(self, delta: str) -> list[str]:  # 0+ complete speakable chunks
    def flush(self) -> list[str]:                 # remaining partial chunk(s) if non-trivial
```

### app/teacher_persona.py — 7-b — PURE STDLIB

```python
def build_system_prompt(memory_context: str,
                        emotion_label: Optional[str] = None,
                        emotion_valence: Optional[float] = None,
                        emotion_arousal: Optional[float] = None) -> str:
    """Bilingual HU/EN teacher persona. ALWAYS includes grammar-correction duty:
    if user's English has an error -> correct it, explain in Hungarian, give 2 examples.
    Replies in the user's language (match dominant language of the utterance).
    Voice-output constraints: no markdown, short sentences, numbers spelled naturally.
    Memory context block only if non-empty. Emotion block ONLY if emotion_label is not None
    (M2 hook — always None in M1)."""

def build_messages(transcript: str, system_prompt: str,
                   history: Optional[list[dict]] = None) -> list[dict]:
    """OpenAI-style messages: [{'role':'system',...}, *history, {'role':'user','content':transcript}]"""
```

### app/llm.py — 7-b — httpx allowed

```python
class LlmClient:
    """llama.cpp llama-server OpenAI-compatible client. Base URL from config.llama_server_url."""
    def __init__(self, config: AgentConfig) -> None: ...          # lazy httpx.AsyncClient
    async def health_check(self) -> bool: ...                     # GET {base}/health, 2s timeout
    async def chat_stream(self, messages: list[dict],
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None) -> AsyncIterator[str]:
        """POST /v1/chat/completions stream=True; yields content deltas. Parses SSE lines;
        tolerates comments/keep-alives. Raises LlmUnavailableError on connection errors."""
    async def chat_json(self, messages: list[dict],
                        temperature: Optional[float] = None) -> dict:
        """Non-streaming with response_format={"type":"json_object"} (VoiceMem _llm_json compat).
        Parses and json.loads the content; raises LlmUnavailableError on transport errors,
        ValueError on invalid JSON."""
    async def aclose(self) -> None: ...

class LlmUnavailableError(RuntimeError): ...

def parse_sse_content_delta(line: str) -> str:
    """PURE: one SSE data line -> content delta ('' if not a content chunk/keep-alive).
    Unit-tested without any server."""
```

### app/tts.py — 7-b — numpy allowed; piper via subprocess

```python
class TtsEngine:
    """Piper subprocess wrapper. Voice per language: config.tts_hu_voice / tts_en_voice,
    resolved under config.voices_dir (config property)."""
    def __init__(self, config: AgentConfig) -> None: ...
    def is_available(self) -> bool: ...   # piper executable + both voice onnx files exist
    def synthesize_to_file(self, text: str, language: str, out_path: Path) -> bool:
        """Run: <piper> --model <voice.onnx> --output_file <wav>; text via stdin (utf-8).
        Returns success. Timeout 10s, logs stderr on failure."""
    def synthesize(self, text: str, language: str) -> Optional["np.ndarray"]:
        """synthesize_to_file to a tempfile, read WAV via stdlib `wave`,
        return int16 mono numpy array at config.output_sample_rate, else None."""
    def stop(self) -> None: ...           # kill in-flight subprocess (barge-in)
    def list_voices(self) -> list[str]:   # available .onnx voice names in voices_dir
```

### app/voicemem_bridge.py — 7-b — `voicemem` pkg lazy

```python
@dataclass
class TurnContext:
    transcript: str
    memory_context: str = ""
    speaker_id: str = "voice_user"
    turn: Optional[Any] = None           # VoiceMem Turn object when real integration active

class VoiceMemBridge:
    """Wraps the CONTROLLED VoiceMem fork (vendor/voicemem @ e8384e0, v0.5.0 ownership): mode='text_mode', embedding local, reply= our LLM streaming fn.
    If the `voicemem` package is not installed -> degraded mode: memory_context="", log once."""
    def __init__(self, config: AgentConfig, llm: Optional[LlmClient] = None) -> None: ...
    def is_available(self) -> bool: ...
    async def process_turn(self, transcript: str,
                           speaker_id: str = "voice_user") -> TurnContext: ...
        """Real: asyncio.to_thread over VoiceMem feed_partial(text, ended=True) -> Turn;
        extract memory context string from the Turn/SearchResult. Degraded: empty context."""
    async def commit_reply(self, reply_text: str,
                           speaker_id: str = "voice_user") -> None: ...
        """Feed assistant reply back for long-term memory consolidation."""
    def build_reply_fn(self) -> Callable[..., Any]: ...
        """Async generator fn for VoiceMem(reply=...): signature (text, memory_context="") ->
        AsyncIterator[str]; internally uses llm.chat_stream with the teacher system prompt."""
    def cancel_pending(self) -> None: ...  # barge-in: cancel speculative retrieval
```

### app/vad.py — 7-a — pure state machine + lazy onnxruntime

```python
class VadEvent(Enum): NONE, SPEECH_START, SPEECH_END

class VadStateMachine:
    """PURE frame-level state: speech start/end + 300ms hangover. threshold from config."""
    def __init__(self, threshold: float, hangover_ms: int, frame_ms: int) -> None: ...
    def update(self, prob: float) -> VadEvent: ...
    def reset(self) -> None: ...
    @property
    def in_speech(self) -> bool: ...

class SileroVad:
    """Silero VAD v6.2.1 ONNX via onnxruntime (CPU). frame = 512 samples @16k float32 (32ms).
    Lazy import; raises RuntimeError with install hint if onnxruntime/model missing."""
    def __init__(self, config: AgentConfig) -> None: ...
    def prob(self, frame: "np.ndarray") -> float: ...
    def reset(self) -> None: ...
```

### app/barge_in.py — 7-a — PURE STDLIB

```python
class EchoGate:
    """Loopback gating: while TTS plays, mic input is captured but NOT fed to ASR.
    VAD keeps running (barge-in detection needs it)."""
    def __init__(self) -> None: ...
    def on_tts_start(self) -> None: ...
    def on_tts_end(self) -> None: ...
    @property
    def tts_playing(self) -> bool: ...
    def should_process_mic(self) -> bool: ...  # False while TTS playing

class BargeInDetector:
    """Fires True once when speech prob >= threshold for >= min_speech_ms DURING TTS playback.
    Does nothing outside TTS playback. reset() after each turn."""
    def __init__(self, threshold: float, min_speech_ms: int, frame_ms: int) -> None: ...
    def update(self, prob: float, tts_playing: bool) -> bool: ...
    def reset(self) -> None: ...
```

### app/asr.py — 7-a — torch/transformers lazy

```python
class AsrEngine:
    """Qwen3-ASR-0.6B (config.asr_model_name) via transformers on config.asr_device.
    Quasi-streaming: feed() accumulates 600ms batches -> partial string; flush() -> final.
    Lazy import; is_available() False when torch/transformers missing.
    The exact generate() call must follow the model card; keep it in ONE clearly-marked
    method `_transcribe(samples) -> str` so it is easy to fix on the target machine."""
    def __init__(self, config: AgentConfig) -> None: ...
    def is_available(self) -> bool: ...
    def feed(self, samples: "np.ndarray") -> str: ...
    def flush(self) -> str: ...
    def reset(self) -> None: ...
```

### app/audio_io.py — 7-a — sounddevice lazy

```python
class MicStream:
    """sounddevice InputStream @ config.sample_rate, blocksize = config.vad_frame_samples (512).
    frame_callback(frame: np.ndarray) called per block. Context manager start/stop.
    Lazy import with RuntimeError install-hint. NO-OP (callback loop over silence) is NOT
    needed here — mock mode uses its own fake clock driver, not MicStream."""
    def __init__(self, config: AgentConfig, frame_callback) -> None: ...
    def __enter__(self) -> "MicStream": ...
    def __exit__(self, *exc) -> None: ...
    def stop(self) -> None: ...

class SpeakerOutput:
    """sounddevice OutputStream playback of int16 mono PCM. play() blocking, play_async via to_thread."""
    def __init__(self, config: AgentConfig) -> None: ...
    def play(self, pcm: "np.ndarray", sample_rate: Optional[int] = None) -> None: ...
    async def play_async(self, pcm: "np.ndarray", sample_rate: Optional[int] = None) -> None: ...
    def stop(self) -> None: ...
```

### app/pipeline.py — 7-a

```python
class VoicePipeline:
    """Turn orchestrator. Components injected (real or mock)."""
    def __init__(self, config: AgentConfig, asr: AsrEngine, llm: LlmClient,
                 tts: TtsEngine, voicemem: VoiceMemBridge, vad: SileroVad,
                 audio_out: SpeakerOutput, logger: Optional[logging.Logger] = None) -> None: ...
    async def handle_utterance(self, audio: "np.ndarray",
                               speaker_id: str = "voice_user",
                               vad_prob_fn: Optional[Callable[[], float]] = None) -> TurnResult:
        """1) asr.feed/flush -> transcript; 2) voicemem.process_turn; 3) system prompt
        (teacher_persona, emotion always None in M1); 4) llm.chat_stream;
        5) SentenceStream chunks -> detect_language per chunk -> tts.synthesize ->
        audio_out.play; between chunks poll vad_prob_fn + EchoGate/BargeInDetector ->
        cancel TTS/LLM/voicemem on barge-in; 6) timings recorded; 7) commit_reply."""
    def cancel(self) -> None: ...
```

### app/mock_components.py — 7-a (drives sandbox-verifiable end-to-end demo)

```python
class MockAsrEngine:      # preset transcript queue: .queue = ["...", ...], feed/flush/is_available/reset
class MockLlmClient:      # scripted replies; chat_stream yields word-by-word with tiny sleeps;
                          # chat_json returns {"ok": True}; health_check True; SSE helpers
class MockTtsEngine:      # synthesize returns zeros(22050) after tiny sleep; stop() sets flag; logs
class MockVoiceMemBridge: # scripted memory contexts per turn; records commit_reply calls
class MockVad:            # prob(frame) from a scripted probability timeline
```
All mocks satisfy the real interfaces above (duck-typed; the pipeline accepts them).

### app/main.py — 7-a

```
CLI: python -m app.main [--config PATH] [--mock] [--text "..."] [--barge-in-demo]
                        [--check] [--list-config]
  --mock            : run scripted 3-turn demo conversation with mock components:
                      turn 1: EN grammar error ("Why is I have went wrong?")
                      turn 2: follow-up HU question
                      turn 3: memory recall check ("Do I still make the same mistake?")
                      Prints transcript/reply/timings JSON per turn. Exit 0 on success.
  --mock --text "…" : single custom turn instead of the scripted demo.
  --mock --barge-in-demo : scripted turn where user interrupts TTS -> cancelled=True.
  --check           : print asset checklist (config.check_runtime_assets()) and exit.
  --list-config     : print resolved config as JSON and exit.
main() is async (asyncio.run). No heavy deps imported unless real mode is requested.
```

## File ownership

| Task | Files |
|------|-------|
| 7-a (audio-side) | `app/vad.py`, `app/barge_in.py`, `app/asr.py`, `app/audio_io.py`, `tests/unit/test_vad_state_machine.py`, `tests/unit/test_barge_in.py`, `tests/unit/test_config.py` (M0-bővítés: 10-a) |
| 7-b (text & brain) | `app/text_utils.py`, `app/teacher_persona.py`, `app/llm.py`, `app/tts.py`, `app/voicemem_bridge.py`, `tests/unit/test_text_utils.py`, `tests/unit/test_teacher_persona.py`, `tests/unit/test_llm_sse.py`, `tests/unit/test_voicemem_bridge.py` |
| 7-c (infra & benchmarks) | `config/env.local.ps1`, `config/env.local.sh` (M0-átírás: 10-a), `config/voicemem_config.yaml` (M0-bővítés: 10-a), `scripts/download_models.ps1`, `scripts/start_llama_server.ps1`, `scripts/start_agent.ps1`, `scripts/offline_check.ps1`, `scripts/verify_setup.py` (M0-átírás: 10-b), `scripts/run_tests.ps1` (M0-frissítés: 10-a), `requirements.txt` + `requirements.lock` (10-a), `pyproject.toml` (10-a), `README.md` (10-b), `LICENSES.md` (10-a), `.gitignore` (10-a), `tests/benchmark/metrics.py` (pure-python WER/CER), `tests/benchmark/asr_benchmark_hu.py`, `tests/benchmark/tts_benchmark.py`, `tests/benchmark/llm_benchmark.py`, `tests/benchmark/latency_benchmark.py`, `tests/integration/offline_test.py` |
| 8 (main orchestrator) | `app/pipeline.py`, `app/mock_components.py`, `app/main.py`, `tests/integration/test_pipeline_mock.py` — written LAST, during integration, once 7-a/7-b modules exist |
| 10-a (M0 layout/config/manifest) | lásd a fájltulajdon-jegyzéket az „M0 kiegészítés” szekcióban |
| 10-b (M0 installer scripts) | lásd a fájltulajdon-jegyzéket az „M0 kiegészítés” szekcióban |

The pipeline (`Task 8`) imports 7-a and 7-b modules — it is written by the integrator after
both land. 7-a and 7-b do NOT import each other's modules, so they can run fully in parallel.

## Verification each agent must run (from `/home/z/my-project/voicemem-agent/`)

```
python3 -m py_compile app/<your modules>.py
python3 -m unittest discover -s tests -t . -v     # only your tests may exist yet — that's fine
```

All tests must pass. Do not weaken a test to make it pass — fix the code.

---

## M0 kiegészítés (Task 10-a)

A 10-a ügynök beépítette az M0 (bootstrap) követelményeket a repo szerkezetébe.
Ez a szekció a 10-b ügynökkel (telepítő `.ps1` szkriptek) közös szerződés.

### M0 path-szerződés — modell-elrendezés (spec §7)

| Komponens | Könyvtár (repo-gyökérhez képest) | Config property | Env felülírás |
|---|---|---|---|
| ASR (Qwen3-ASR-0.6B) | `models/asr/qwen3-asr-0.6b/` | `asr_model_dir` | `QWEN3_ASR_MODEL_PATH` |
| LLM (Gemma 4 12B QAT Q4_0 — v0.4.3: az EGYETLEN LLM, fallback NINCS) | `models/llm/gemma-4-12b/` | `llm_model_file` | `LLAMA_MODEL_PATH` |
| TTS (Piper hangok) | `models/tts/piper/` | `voices_dir` | `PIPER_VOICES_PATH` |
| VAD (Silero) | `models/vad/silero-vad/silero_vad.onnx` | `silero_vad_path` | `SILERO_VAD_PATH` |
| Embedding (e5-small) | `models/embedding/multilingual-e5-small/` | `embedding_model_dir` (ÚJ) | `EMBEDDING_MODEL_PATH` |
| Emotion (M2 placeholder) | `models/emotion/` — M0/M1-ben ÜRES, TILOS modellt tenni ide | — | — |
| HF cache gyökér | `models/hf/` (`HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` — a tartalom gitignore-olt) | — | — |
| Memória | `memory/` (`sqlite/`, `qdrant/`, `backups/`) | `memory_root_path` | `VOICEMEM_MEMORY_ROOT` |
| Adat | `data/audio/`, `data/benchmarks/` | `data_dir` (ÚJ) | — |
| Logok | `logs/` | `logs_dir` (ÚJ) | — |
| Binárisok | `bin/` (llama-server.exe, piper.exe — a telepítő tölti) | `bin_dir` | — |
| VoiceMem pinned klón | `vendor/voicemem/` (a telepítő klónoz; `pip install -e`) | — | — |

A `default_root()` a REPO-GYÖKÉR (a régi Windows meghajtó-alapú default és a
`<repo>/runtime` fallback TÖRLÉSRE került); a `VOICEMEM_HOME` env felülírás
megmaradt. A memória default `root/memory` (a "memoryspace" elnevezés törlődött).

### app/config.py új mezők és property-k

Új dataclass mezők: `model_root: str = "models"`, `data_root: str = "data"`,
`logs_root: str = "logs"`, `llm_parallel: int = 1` (llama-server `--parallel`;
`validate()` >= 1-t követel), `audio_input_device: str = ""` és
`audio_output_device: str = ""` (üres = rendszer-default sounddevice),
`embedding_model_path: str = ""` (env: `EMBEDDING_MODEL_PATH`).
Új property-k: `models_dir` (= root/model_root), `data_dir`, `logs_dir`,
`embedding_model_dir`. A `from_yaml` toleráns maradt: az új `project:`/`paths:`
top-level szekciókat figyelmen kívül hagyja (csak AgentConfig mezőneveket
vesz fel — az `app:` szekció a forrás).

### scripts/write_install_manifest.py interfész (10-a tulajdona)

```python
build_manifest(root: Path, fake_gpu: bool = False) -> dict   # tiszta logika, unit-tesztelhető
write_manifest(manifest: dict, out_path: Path) -> Path       # vékony JSON-író réteg
main(argv=None) -> int    # CLI: --root (default: repo-gyökér), --out, --fake-gpu
```

Kimenet: `<root>/INSTALL_MANIFEST.json` — determinisztikus
(`json.dumps(indent=2, sort_keys=True)`), a modellek lista útvonal szerint
rendezve; a `models/hf` cache-karból és a `.gitkeep`/`README.md` repo-dokokból
kimarad; komponens-név + `hf_repo` mapping (asr/llm/tts/vad/embedding).
Példa a kimenetre: `INSTALL_MANIFEST.example.json` (fikív adatokkal, commitolva).
A GPU/torch blokk csak valódi torch esetén töltődik (sandboxban `--fake-gpu`
kapcsolóval tesztelhető); a voicemem blokk `not_installed` jelzőt ad,
amíg a `vendor/voicemem` klón nem áll készen.

### tests/ átszervezés (10-a)

| Régi hely (tests/) | Új hely | Megjegyzés |
|---|---|---|
| `test_config.py` | `tests/unit/` | M0-tesztekkel bővítve |
| `test_text_utils.py`, `test_teacher_persona.py`, `test_llm_sse.py`, `test_voicemem_bridge.py`, `test_vad_state_machine.py`, `test_barge_in.py`, `test_metrics.py` | `tests/unit/` | sys.path-bootstrap `parents[2]` (repo-gyökér) |
| `test_pipeline_mock.py`, `offline_test.py` | `tests/integration/` | a CLI-alprocessz-teszt `cwd=REPO_ROOT`-ja `parents[2]`-re módosult |
| `latency_benchmark.py`, `asr_benchmark_hu.py`, `llm_benchmark.py`, `tts_benchmark.py`, `metrics.py` | `tests/benchmark/` | a `from tests.metrics import ...` `from tests.benchmark.metrics import ...`-ra módosult |
| — | `tests/unit/test_install_manifest.py` | ÚJ: 13 teszt a manifest-szkriptre |

Minden tesztcsomag `__init__.py`-t kapott (`tests/`, `tests/unit/`,
`tests/integration/`, `tests/benchmark/`). Futtatás a repo-gyökérből:
`python -m unittest discover -s tests -t . -v` (a `-t .` kötelező elem).

### Fájltulajdon-jegyzék (10-a vs 10-b)

| Tulajdonos | Fájlok |
|---|---|
| 10-a (struktúra/config/manifest) | `models/**` (.gitkeep + README.md), `memory/**`, `data/**`, `logs/`, `bin/`, `vendor/` (.gitkeep), `tests/unit|integration|benchmark/` átszervezés, `app/config.py`, `config/voicemem_config.yaml`, `config/env.local.ps1`, `config/env.local.sh`, `config/.env.example`, `requirements.txt`, `requirements.lock`, `VERSION`, `pyproject.toml`, `INSTALL_MANIFEST.example.json`, `.gitignore`, `scripts/run_tests.ps1`, `scripts/write_install_manifest.py`, `tests/unit/test_install_manifest.py`, `tests/unit/test_config.py`, `CONTRACT.md` (ez a szekció), `LICENSES.md` |
| 10-b (telepítő szkriptek + README) | `scripts/download_models.ps1`, `scripts/start_llama_server.ps1`, `scripts/start_agent.ps1`, `scripts/offline_check.ps1`, `scripts/verify_setup.py`, ÚJ M0 telepítő szkriptek (pl. `install.ps1`, smoke-test), `README.md` |

M0 kizárások (mindkét ügynökre): M2/M3 modellek (emotion2vec, SpeechBrain
ECAPA, scene recognition, MOSS TTS, Gemma audio, Qwen Omni) NEM települnek —
a `models/emotion/` üres placeholder marad; a HF cache a `models/hf/` alá
izolált; a `.venv`-et a telepítő hozza létre (a sandboxban nem jön létre).

---

## M0.1 kiegészítés (Task 12 — release-munkafolyamat és validáció)

A felhasználói követelmény: **verziózott ZIP minden frissítés után**, és
**funkciónkénti validációs tesztek, amelyek minden ZIP-build ELŐTT lefutnak**.
Ez a szekció a szerződéseket rögzíti — minden későbbi ügynök ezekhez kódol.

### tests/validation/ csomag (12-a)

* Csak **stdlib unittest** (nincs pytest, nincs conftest). Minden fájl:
  `from __future__ import annotations`, `sys.path.insert(0, ...parents[2])`,
  `if __name__ == "__main__": unittest.main()`.
* Alaposztály: `tests.validation._report.FeatureValidationTest` —
  osztályattribútum `FEATURE` (a riport funkcióneve), `deep_skip(reason)`
  (SKIP + ok-riport), `deep_pass(detail)`.
* Metódus-elnevezés: `test_logic_*` = mindig fut (sandbox + célgép, a
  ZIP-kapu része); `test_deep_*` = valódi komponens, ha elérhető — hiányzó
  komponens esetén `deep_skip` okkal (sosem csendes PASS, sosem akadás:
  hálózati hívás időtúllépés-köteles). Komponens jelenléte + hibás
  működés = FAIL (nem SKIP).
* Nehéz modulok (torch/transformers/onnxruntime/sounddevice/
  sentence_transformers/voicemem) CSAK a deep metódusokon belül, guard után.
* Riport: `_report` atexit-kor JSON-t ír a `VMA_VALIDATION_REPORT` env
  által megadott útvonalra; sémája: `{schema_version, generated_at, python,
  platform, features: {<feature>: {status: pass|skipped, detail|reason}}}`.
* Env-semlegesítés (AgentConfig-et érintő teszteknél): a
  `tests/integration/test_pipeline_mock.py` `_ENV_KEYS` + `patch.dict`
  mintája.

### scripts/run_tests.ps1 (12, frissítve)

* Python-választás (M0-elv): `.venv\Scripts\python.exe` →
  `.venv/bin/python` → PATH `python` → `python3`.
* A futtatás alatt beállítja `VMA_VALIDATION_REPORT` =
  `logs\validation_report.json`-t (a végén visszaállítja/törli).
* Lépések: `compileall -q app` → `unittest discover -s tests -t . -v`;
  kilépési kód = a unittest kódja. Zéró paraméteres (szándékos).

### scripts/build_release.ps1 (12 — ÚJ)

* param-blokk: `-Bump patch|minor|major|none` (alap: **patch**),
  `-Version x.y.z`, `-Notes "..."`, `-Force`, `-StrictValidation`, `-Root`.
* 8 lépés: cél-verzió → **KÖTELEZO tesztkapu** (`run_tests.ps1` child
  powershell.exe/pwsh-ben; FAIL → nincs ZIP, history-bejegyzés, exit 1) →
  verzió-írás → staging → BUILD_INFO.json → ZIP → SHA256 → index/changelog.
* Staging **allowlist** (a ZIP tartalma): `app/`, `config/` (`.env`
  kizárva!), `scripts/`, `tests/`, + `README.md`, `CHANGELOG.md`,
  `LICENSES.md`, `CONTRACT.md`, `VERSION`, `INSTALL_MANIFEST.example.json`,
  `requirements.txt`, `requirements.lock`, `pyproject.toml`, `.gitignore`,
  `START.bat`, `MODELS.lock.json`, és generált `BUILD_INFO.json`. Minden más
  (`.venv`, `models`, `memory`, `data`, `logs`, `bin`, `vendor`, `releases`,
  `INSTALL_MANIFEST.json`, `.install_state.json`, `__pycache__`) NEM kerül a
  ZIP-be. (M0.2: a `START.bat` + `MODELS.lock.json` mostantól ZIP-tartalom —
  a kicsomagolt ZIP-et is egy dupla kattintással kell telepíteni.)
* Kimenetek: `releases/VoiceMemAgent_vX.Y.Z[.zip|.zip.sha256]`,
  `releases/RELEASE_INDEX.json` (séma 1, bejegyzésenként: version,
  built_at_utc, zip, sha256, size_bytes, notes, git_commit, test_gate*,
  validation_*), `releases/BUILD_HISTORY.json` (minden kísérlet),
  `CHANGELOG.md` (bejegyzés `## [X.Y.Z] - YYYY-MM-DD` formában), valamint a
  verzió-sync: `VERSION` + `pyproject.toml [project] version` +
  `config/voicemem_config.yaml project.version` (a
  `tests/validation/test_feature_release.py` kikényszeríti).
* PS 5.1-biztos: pure ASCII, nincs `??`/`?.`/`&&`/`||`/ternary; JSON-írás
  UTF-8 BOM nélkül (`[IO.File]::WriteAllText` + `UTF8Encoding($false)`);
  egyelemű tömb-gyűrűzés elkerülve (bejegyzésenkénti ConvertTo-Json).
* `-StrictValidation`: a deep SKIP-ek FAIL-t okoznak (kiadási build a
  célgépen; telepítetlen gépen jogosan elbukik).

### Fájltulajdon (Task 12)

| Tulajdonos | Fájlok |
|---|---|
| 12 (main orchestrator) | `scripts/build_release.ps1`, `scripts/run_tests.ps1` (frissítés), `tests/validation/__init__.py`, `tests/validation/_report.py`, `tests/validation/test_feature_release.py`, `CHANGELOG.md`, `releases/` (README, .gitkeep, RELEASE_INDEX seed), `config/.env.example` (pótlás), `.gitignore` (releases-blokk), `README.md` + `CONTRACT.md` M0.1-fejezetek |
| 12-a (alagent) | `tests/validation/test_feature_{config,runtime,models,vad,asr,llm,tts,embedding,memory,barge_in,audio,teacher,pipeline,offline,manifest,scripts}.py` (16 fájl) |

---

## M0.2 kiegészítés (Task 13 — One Click UX)

A felhasználói követelmény: **a `START.bat` az egyetlen felhasználói belépési
pont** — dupla kattintás, és a rendszer maga végzi el a Python-ellenőrzést, a
venv-létrehozást, a függőségtelepítést, a HF toolingot, a modellletöltést, a
konfigurációt, a manifestet, a smoke testeket, a verificationt és az agent
indítását. Egyetlen user-facing üzenet sem terelheti vissza a felhasználóra
a megoldható munkát ("telepítsd kézzel", "futtasd előbb az installert",
"pip install ..."). Ez a szekció rögzíti a szerződéseket.

### START.bat (repo-gyökér, 13 tulajdona)

* Pure ASCII batch; gyökér: `%~dp0` + `cd /d` (a projekt bárhová másolható).
* Módok: `` (run), `mock`, `check`, `benchmark`, `repair`, `build` —
  érvényesítve a BAT-ban; ismeretlen mód → tiszta hiba + `pause`.
* Mindig `powershell -NoProfile -ExecutionPolicy Bypass -File
  scripts\bootstrap.ps1 -Mode "<mód>"` hívás (a PowerShell belső részlet).
* Hibánál ÉS normál leállásnál is `pause` (dupla kattintásnál olvasható marad
  a konzol); a kilépési kód propagál (`exit /b %ERRORLEVEL%`).

### scripts/bootstrap.ps1 (13 tulajdona — belső orchestrátor)

* `param(` az első végrehajtható utasítás:
  `-Mode [ValidateSet('run','mock','check','benchmark','repair','build')]`,
  `-Root`, valamint build-továbbítás: `-Notes`, `-Bump`, `-StrictValidation`.
* Gyökér: `$PSScriptRoot` szülője (nincs beégetett útvonal).
* **Állapotgép**: `.install_state.json` (séma 1; a felhasználói spec 15
  kötelező kulcsai: `python_ready`, `venv_ready`, `dependencies_ready`,
  `models_ready`, `config_ready`, `smoke_tests_passed` + kiegészítők:
  `hf_tooling_ready`, `binaries_ready`, `manifest_ready`, `voicemem_ready`,
  `python_version`, `python_exe`, `gpu_name`, `cuda_ready`, `version`,
  `runs_count`, `last_run_at_utc`, `last_mode`, `last_result`). Minden lépés
  után ment; megszakadt telepítés újraindítással folytatható.
* **Gyorsútvonal (idempotencia, spec 16/29)**: Ha a probe-ok (venv fut,
  `import yaml, numpy, httpx, soundfile, torch`, `huggingface_hub`
  kódtár a .venv-ben (Python API — CLI NEM kell),
  `bin\llama-server.exe` + `bin\piper.exe`, MODELS.lock-jelenlét, config +
  .env) mind rendben ÉS `smoke_tests_passed` → a telepítő kimarad
  ("environment already ready"), csak verify + dispatch fut.
* **Nehéz útvonal**: bármi hiányzik / első futás / `repair` mód → az
  idempotens `scripts\install_m1.ps1` **child folyamatként** fut
  (powershell.exe, fallback pwsh; az `exit` nem ölheti meg a bootstrapet),
  utána pip/setuptools/wheel frissítés + újraprobe + állapotfrissítés.
  Olcsó javítások (spec 17) a telepítő előtt/után külön: `.env` generálás
  `.env.example`-ből, `INSTALL_MANIFEST.json` regenerálás.
* **Mód-diszpécs**: run → `start_agent.ps1`; mock → `start_agent.ps1
  -NoServer --mock`; check → `verify_m1.ps1 -WithServer` + `run_tests.ps1`
  (PASS/FAIL összegző; FAIL → exit 2); benchmark → `latency_benchmark.py`
  (+ tts/asr/llm best-effort); repair → installer + mock post-verify; build →
  `build_release.ps1`.
* **Hiba-UX (spec 21)**: elsődleges hiba = rövid, tiszta üzenet (Detected /
  Required / következő lépés sablonnal); a traceback és minden technikai
  részlet `logs\bootstrap.log`-ba megy; globális try/catch.
* Log (spec 22): minden futás időbélyeg, mód, OS, Python, GPU (nvidia-smi),
  probe-eredmények, installer-futtatások, tesztkimenetek, hibák/figyelmeztetések,
  végstátusz.
* 64-bit Python-ellenőrzés: quote-mentes szondák (`sys.maxsize > 2**32`,
  másodelagosan `platform.architecture()`); Python-keresési sorrend:
  `py -3.11` → `python3.11` → `python` (sosem telepít Pythont).
* Python-invokációs szerződés (v0.1.4): a szkriptek SOHA nem adnak
  több elemű tömböt a call operatornak (`& $Var`) — a PS a tömböt egyetlen
  (nem létező) parancsnévvé fűzi össze. Minden python-hívás
  `& $PythonExe @PythonArgs ...` minta (exe-útvonal + argumentumtömb);
  regressziós teszt: `tests/validation/test_feature_pyexec.py`
  (FEATURE `pyexec`, statikus tömb-szkenner + élő deep-invokáció).
* NVIDIA GPU-ellenőrzés (v0.1.5): a régi PATH-alapú egyetlen próba
  (`Get-Command nvidia-smi`) a START.bat-lánc gyermekfolyamatában fals
  "nincs GPU/driver" hibát adott élő gépen — helyette közös könyvtár
  `scripts/gpu_check.ps1` (bootstrap + install_m1 dot-source): lánc
  Get-Command → standard abszolút útvonalak (System32/NVSMI/SysWOW64) →
  futtatás → name/driver/VRAM (nvidia-smi `memory.total`, SOHA WMI
  AdapterRAM — uint32, 4 GiB fölött csomagol) → WMI
  `Win32_VideoController` fallback. 8 külön hibakód
  (GPU_NOT_FOUND / DRIVER_NOT_FOUND / NVIDIA_SMI_NOT_FOUND /
  NVIDIA_SMI_EXEC_FAILED / NVIDIA_SMI_QUERY_FAILED / VRAM_QUERY_FAILED /
  GPU_VALIDATION_FAILED / OK), hibánál diagnosztikai blokk
  (PS-verzió, bitness, PATH, Get-Command, where.exe, standard-path
  próbák, smi exit/stdout, WMI) a `logs\bootstrap.log`-ba UGYANABBÓL a
  gyermekfolyamatból. Teszt-hookok: `-SearchDirs`,
  `VM_GPU_TEST_NO_STD_PATHS` (csak validációs suite). Regressziós teszt:
  `tests/validation/test_feature_gpu_check.py` (FEATURE `gpu-check`,
  A–F forgatókönyvek +élő deep a START.bat gyermekmintán).

### MODELS.lock.json (repo-fájl, 13 tulajdona — séma 2)

* Mostantól **a repóval együtt szállítjuk** (a `.gitignore`-ból kivettük; a
  release ZIP tartalma). A `download_models.ps1` csak akkor írja, ha nem
  létezik (a szállított változat megmarad).
* `models[]`: `component` (llm|asr|embedding|vad|tts), `repo`, `target_dir`
  (M0 §7 elrendezés), `files`, `snapshot` (true = teljes repo-snapshot;
  jelenlét: `snapshot_probe` fájl + `min_total_mb` összméret-padló),
  `pinned_revision`, `min_bytes` (fájlonkénti minimális méret; GGUF >= 1 GiB).
* `tools[]`: `component` (llama-server|piper), `provider` github-release,
  `repo`, `tag`, `assets`, `target_dir` = `bin`, `files`. A pin-eknek
  szinkronban kell maradniuk az `install_m1.ps1` fejében lévő változókkal.
* A bootstrap `Test-ModelsPresent` jelenlét-ellenőrzése CSAK ezt a fájlt
  olvassa és sosem tölt le semmit.

### Self-heal + Python-API szabály (download_models.ps1 / download_models_hf.py)

* **v0.1.7 architektúra**: a modellletöltés a `huggingface_hub` PYTHON
  API-jával fut (`hf_hub_download` / `snapshot_download` a
  `scripts/download_models_hf.py`-ban), mindig a projekt `.venv`
  interpreterével (`.venv\Scripts\python.exe -u`). A Hugging Face CLI
  NEM runtime függőség (v0.1.6 blocker: a CLI deprecation-warningjének
  emojija CP1252 konzolon UnicodeEncodeError-t okozott még a letöltés
  előtt) — CLI subprocess, CLI output parsing és PATH-függés TILOS.
  A gyermek Python kap `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`
  biztonsági övet, a kimenet tiszta ASCII.
* A `huggingface_hub` kódtár hiánya esetén AUTOMATIKUS `pip install
  "huggingface_hub"` a `.venv` pythonjával (a `[cli]` extra NINCS
  telepítve); csak a sikertelen automatikus telepítés lehet hiba — és
  akkor is a javítás `START.bat` újr indítása, NEM kézi parancs.
* Letöltési szerződés: lock-vezérelt (MODELS.lock.json), retry
  (alap 3 próbálkozás 5s/15s backoff-fal), resume (megszakadt átvitel
  folytatása), `pinned_revision` támogatás, helyi cél a `models/` fába,
  tükör-láncok, és a lépés CSAK akkor sikeres, ha minden kötelező fájl
  ténylegesen létezik a `min_bytes` / `min_total_mb` küszöbbökkel
  ellenőrizve (`ALL_MODELS_OK` / `MODELS_FAILED` egyértelmű státusz).
* LLM GGUF: az automatikus tükör-lánc beépített (nem TIP üzenet):
  Qwen → unsloth (azonos fájlnev, drop-in) → bartowski (alulvonalas fájlnev +
  automatikus átnevezés a kanonikus névre) — a hivatalos Qwen GGUF-repo
  2026-08-31-én élőben ellenőrizve nem elérhető. VAD: tphakala re-host
  (flat fájl) → eredeti silero repo + VAD-lapítás. Piper hangok:
  nyelvi-almappás (elsődleges) + flat (fallback) útvonal-próba és lapítás
  (flatten) a `models\tts\piper` gyökérbe (a config `voices_dir` ezt várja).

### Tiltott user-facing üzenetek (spec 3 — a teszt kikényszeríti)

A `START.bat`, `scripts\bootstrap.ps1`, `scripts/download_models.ps1`,
`scripts/start_agent.ps1` kódjában (kommentek nélkül) NEM szerepelhet:
`Telepitsd a .venv-be`, `Telepitsd a huggingface`, `Futtasd elobb`,
`install manually`, `vagy kezzel:`. Megengedettek az OS-szintű
előfeltételek üzenetei (Python 3.11 python.org-ról, NVIDIA driver, hálózat).

### M0.2 exit criteria (bővíti a 14 pontos M0-listát)

15. **Clean machine test**: tiszta gépen egyetlen `START.bat`-dupla kattintás
    a teljes telepítéstől az agent indulásáig vezet, manuális
    Python/pip/PowerShell/HF-parancs nélkül.
16. **Existing environment test**: kész környezetben a `START.bat` nem
    telepít/letölt újra, csak ellenőriz és indít.
17. **Repair test**: törölt csomag/modell esetén a `START.bat` felismeri és
    automatikusan javítja.

### Fájltulajdon (Task 13)

| Tulajdonos | Fájlok |
|---|---|
| 13 (main orchestrator) | `START.bat`, `scripts/bootstrap.ps1`, `MODELS.lock.json`, `tests/validation/test_feature_oneclick.py`, `scripts/download_models.ps1` (self-heal + fallback + flatten), `scripts/install_m1.ps1` (auto-venv-rebuild + START.bat-hintek), `scripts/start_agent.ps1` (START.bat-hintek), `scripts/build_release.ps1` (staging + START.bat + MODELS.lock.json), `.gitignore` (lock ki, state be), `tests/validation/test_feature_release.py` (STAGED_FILES + gate-markerek), `tests/validation/test_feature_scripts.py` (bootstrap.ps1 a kötelező/param-listákban), `README.md` + `CONTRACT.md` M0.2-fejezetek, `voicemem-v1-milestones.md` M0.2-alfejezet |

---

## M2 addendum — érzelmi intelligencia (Task 20, v0.2.0)

Az M2 (milestones 8. fejezet) bővíti az M1 interface-eket. Minden bővítés
BACKWARD-COMPATIBILIS: opcionális paraméterek, a régi hívási módok változatlanul
működnek (a mock demo / benchmark / régi tesztek M1-azonos úton futnak).

### Új modul: `app/emotion.py` (pure stdlib a modul-szinten)

```python
@dataclass
class EmotionResult:
    raw_label: str          # emotion2vec+ category (e.g. "angry"; the model's tokens.txt uses bilingual zh/en forms, both accepted)
    label: str              # tanári kategória: frustrated|sad|neutral|happy
    valence: float          # -1..1
    arousal: float          # -1..1
    confidence: float = 0.0
    fused: bool = False     # True: a 0.6/0.4 blend már lefutott
    scores: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    def to_dict(self) -> dict: ...

def map_raw_label(raw_label: str) -> str: ...          # 9 -> 4 kategória
def prosody_valence_arousal(labels, scores) -> tuple: ...
def semantic_valence_arousal(transcript: str) -> tuple: ...
def teacher_label_from_va(valence, arousal) -> str: ...
def fuse_emotion(prosody, transcript, prosody_weight=0.6) -> Optional[EmotionResult]
def tail_window(audio, sample_rate, seconds) -> Any

class EmotionAnalyzer:      # funasr LAZY importtal (CONTRACT szabály 2)
    REQUIRED_FILES = ("model.pt", "config.yaml", "tokens.txt")
    def __init__(self, model_dir, window_s=5.0, sample_rate=16000, model_factory=None)
    def is_available(self) -> bool
    def analyze(self, audio) -> Optional[EmotionResult]   # None = degradáció

class EmotionMemory:        # data/emotion_log.jsonl (spec 8.3.4)
    def __init__(self, log_path: Path)
    def record_turn(self, result, transcript, stored_as_fact=False) -> bool
    @staticmethod
    def should_store(result, threshold=0.5) -> bool
    @staticmethod
    def fact_text(result, transcript) -> str
```

### Megváltozott M1-interface-ek (opcionális bővítések)

```python
# app/pipeline.py
@dataclass
class TurnTimings:
    ...  # ÚJ mező: emotion_s: Optional[float] = None  (DURÁCIÓ, mint memory_s)
class TurnResult:
    ...  # ÚJ mező: emotion: Optional[dict] = None (EmotionResult.to_dict() vagy None)
class VoicePipeline:
    def __init__(self, ..., emotion: Optional[EmotionAnalyzer] = None,
                 emotion_memory: Optional[EmotionMemory] = None)
    # handle_utterance: ha emotion van ÉS config.enable_emotion -> a
    # _retrieve_memory-mel PÁRHUZAMOSAN (asyncio.gather) fut a proszódia-elemzés;
    # a fúzionált érzelem a system promptba kerül (teacher_persona), frusztrációnál
    # length_scale=config.emotion_slow_length_scale megy a TTS-be.

# app/tts.py (MockTtsEngine tükrözi)
def synthesize(self, text, language, length_scale: Optional[float] = None)
def synthesize_to_file(self, text, language, out_path, length_scale=None)
    # length_scale=None -> piper default (M1-azonos); --length_scale csak ha nem None

# app/voicemem_bridge.py
async def store_fact(self, text: str, speaker_id: str = "voice_user") -> None
    # M2 emotion-fact horog (best-effort, degraded no-op, sosem raise-el)

# app/teacher_persona.py — az emotion-blokk tartalmazza a spec 8.3.2
# "Verify this assessment ..." utasítást is.
```

### Config (M2)

- `enable_emotion: bool = True` (M2 default; `VOICEMEM_ENABLE_EMOTION=0|false`
  kikapcsolja — ekkor a pipeline szigorúan M1-azonos utat fut).
- Új mezők: `emotion_model_name`, `emotion_model_path` (env
  `EMOTION2VEC_MODEL_PATH`), `emotion_window_s` (5.0),
  `emotion_fusion_prosody_weight` (0.6), `emotion_slow_length_scale` (1.1),
  `emotion_store_threshold` (0.5); property: `emotion_model_dir`,
  `emotion_log_path`, `check_emotion_assets()`.
- `validate()`: az `enable_emotion` értéke immáron LEGÁLIS (az M1-es tiltás
  törölve); `enable_speaker` továbbra is tiltott (M3); az új mezők tartomány-
  validálása hozzáadva.
- `check_runtime_assets()` NEM tartalmazza az emotion-modellt (a hiánya sosem
  blokkol indítást — graceful degradation).

### M2-kizárások (a 7. pont M1-kizárási listájának feloldásai)

- Az emotion recognition M2-ben IMPLEMENTÁLT (emotion2vec+ base, CPU).
- A speaker recognition TOVÁBBRA IS kizárt (M3): `enable_speaker` marad false.
- A scene/zene-felismerés TOVÁBBRA IS véglegesen kizárva.
- VRAM-szabály: az emotion2vec+ CSAK CPU-n fut (device="cpu") — 0 GB GPU-bővülés.

### Telepítés (M2)

- `install_m1.ps1`: 19 lépés (új 13. lépés: `funasr==1.4.11` telepítése +
  import-probe + torch-triós guard a telepítése után).
- `MODELS.lock.json`: új `emotion` bejegyzés (emotion2vec/emotion2vec_plus_base,
  models/emotion/emotion2vec-plus-base, 4 fájl, model.pt padló 1 GiB).
- `requirements.txt`/`.lock`: funasr==1.4.11 target-only sor + dokumentáció.

### Fájltulajdon (Task 20)

| Tulajdonos | Fájlok |
|---|---|
| 20 (main orchestrator) | `app/emotion.py`, `app/pipeline.py` (M2-stage), `app/config.py` (M2-mezők), `app/tts.py` (length_scale), `app/teacher_persona.py` (verify-sor), `app/mock_components.py` (MockEmotionAnalyzer + 3-args synthesize), `app/main.py` (--emotion-demo, valós mód), `app/voicemem_bridge.py` (store_fact), `MODELS.lock.json` + `download_models_hf.py` (emotion bejegyzés), `install_m1.ps1` (19 lépés), `requirements.txt/.lock`, `config/voicemem_config.yaml`, `models/emotion/README.md`, `LICENSES.md` (M2-táblázat), `README.md` (M2-fejezet), `tests/unit/test_emotion.py`, `tests/integration/test_pipeline_mock.py` (PipelineEmotionTests), `tests/validation/test_feature_emotion.py`, ez a CONTRACT-addendum |


---

## M3 addendum (Task 21 — Speaker Recognition, v0.3.0)

A 3. mérföldkő (milestones 9. fejezet) szerződéses kiegészítése. Az M1/M2
szerződések változatlanul érvényesek; az itt feloldott kizárások és az új
interfészek:

### M3-kizárások feloldása / új kizárások

- A speaker recognition M3-ban IMPLEMENTÁLVA (SpeechBrain ECAPA, CPU).
- `enable_speaker: bool = True` az M3-alapértelmezés (9.5 zászló);
  `VOICEMEM_ENABLE_SPEAKER=0|false|no|off` env-rel kikapcsolható — ekkor a
  pipeline M1/M2-azonos (a recognizer sosem hívódik meg).
- `validate()`-ban az `enable_speaker`-tiltás TÖRÖLVE; új mezők tartomány-
  validálása: `speaker_match_threshold (0,1]`, `speaker_window_s [0.5,30]`,
  `speaker_registration_min_s [2,60]`.
- A scene/zene-felismerés TOVÁBBRA IS véglegesen kizárva.
- VRAM-szabály: az ECAPA CSAK CPU-n fut — 0 GB GPU-bővülés (9.7 exit 4).

### Új modul: app/speaker.py (interface-szerződések)

- `SpeakerEmbedder(model_dir: Path, window_s: float = 5.0,
  sample_rate: int = 16000, model_factory=None)`
  - `is_available() -> bool` — speechbrain importálható ÉS a 5 fájl a lemezen.
  - `warm_up() -> bool` — egyszeri betöltés; a hiba megjegyzése (nincs
    fordulónkénti újrapróbálkozás).
  - `embed(audio) -> Optional[List[float]]` — 192-dim síkos lista; bármilyen
    hiba `None` (graceful degradation). LAZY speechbrain/torch import
    (CONTRACT 2. pont); torch nélkül numpy-fallback a fake-model tesztekhez.
  - `REQUIRED_FILES = (embedding_model.ckpt, hyperparams.yaml,
    mean_var_norm_emb.ckpt, classifier.ckpt, label_encoder.txt)`.
- `SpeakerRegistry(path: Path)` — `speaker_ids() -> List[str]`,
  `references() -> Dict[str, List[float]]`, `register(id, embedding, samples,
  audio_seconds) -> bool` (felülírás idempotens), `remove(id) -> bool`;
  JSON séma 1; thread-safe; sosem dob kivételt.
- `SpeakerRecognizer(embedder, registry, threshold=0.5,
  registration_min_s=10.0, sample_rate=16000)`
  - `identify(audio) -> Optional[SpeakerIdentification]` — `None` = halott
    analizátor; `SpeakerIdentification(id=None, ...)` = ismeretlen beszélő
    (fallback `user_id="voice_user"`); sosem dob kivételt.
  - `register(speaker_id, audio_chunks) -> Tuple[bool, float]` — átlagolt
    referencia (spec 9.3.3), a második érték a felhasznált beszéd-másodpercek.
- Tiszta függvények: `cosine_similarity(a, b) -> float`,
  `identify_speaker(embedding, references, threshold) ->
  Optional[SpeakerIdentification]`, `average_embeddings(list) -> List[float]`,
  `tail_window(audio, sample_rate, seconds)`.

### Pipeline (M3-stage)

- `VoicePipeline(..., speaker: Optional[SpeakerRecognizer] = None)` — új
  opcionális injektálható komponens (kacsatípus: `identify(audio)`).
- `handle_utterance(audio, speaker_id="voice_user", ...)`:
  - az azonosítás az ASR UTÁN, a memória-keresés ELŐTT fut; az emotion-task
    indítása PÁRHUZAMOS vele (create_task + await csak a speaker-lépésre).
  - a feloldott `speaker_id` (identified id vagy a paraméter fallback)
    irányítja a `_retrieve_memory`-t, a `commit_reply`-t és a `store_fact`-et.
- `TurnTimings.speaker_s: Optional[float]` (duration); `TurnResult.speaker:
  Optional[dict]` + `TurnResult.speaker_id: str`; `to_dict()` bővítve.

### VoiceMem bridge (M3 routing)

- `_ensure_vm(user_id)` — **beszélőnkénti** facade (`_vm_by_user` dict);
  a `process_turn`/`commit_reply`/`store_fact` mind a speaker_id-hoz tartozó
  facadet használja (9.3.4 memóriatér-szeparáció); `cancel_pending()` minden
  élő facadeten végigiterál. Az M1-es „single-user” figyelmeztetés törölve.

### Mock-komponens (kacsatípus)

- `MockSpeakerRecognizer(scripted: List[Optional[SpeakerIdentification]],
  fail=False, dead=False, delay_s=0.001)` — `identify(audio)` időzített
  szkript (ciklikus), `analyzed_sizes` rögzítés; `dead=True` -> None (halott),
  `fail=True` -> raise (degradációs út).
- `MockVoiceMemBridge`: új M3-rögzítők: `turn_speakers`, `commit_speakers`,
  `fact_speakers` (a routing/kontamináció-assertekhez; a korábbi
  `committed`/`facts` listák változatlanok).

### Telepítés (M3)

- `install_m1.ps1`: 20 lépés (új 14. lépés: `speechbrain==1.1.1` telepítése +
  import-probe + torch-triós guard utána; a modellletöltés 15. lépés lett).
- `MODELS.lock.json` + a letöltő REPO_DEFAULT_LOCK-ja: új `speaker` bejegyzés
  (speechbrain/spkrec-ecapa-voxceleb, models/speaker/ecapa-voxceleb, 5 fájl,
  embedding_model.ckpt padló 80 MB — a generic letöltő útvonal kezeli,
  tükör-lánc nem szükséges).
- `requirements.txt`/`.lock`: speechbrain==1.1.1 target-only blokk
  (a `torch>=2.1.0` követelményt a cu128 trió kielégíti — trio-guard utána).

### Fájltulajdon (Task 21)

| Tulajdonos | Fájlok |
|---|---|
| 21 (main orchestrator) | `app/speaker.py` (új), `app/pipeline.py` (M3-stage + timings/result), `app/config.py` (M3-mezők + speaker_model_dir/registry/check_speaker_assets), `app/voicemem_bridge.py` (per-user facades), `app/mock_components.py` (MockSpeakerRecognizer + routing-rögzítők), `app/main.py` (--speaker-demo, --register-speaker, valós mód, --check), `MODELS.lock.json` + `download_models_hf.py` (speaker bejegyzés), `install_m1.ps1` (20 lépés), `requirements.txt/.lock`, `config/voicemem_config.yaml` (M3-mezők), `models/speaker/README.md`, `LICENSES.md` (M3-táblázat), `README.md` (M3-fejezet), `tests/unit/test_speaker.py`, `tests/integration/test_pipeline_mock.py` (PipelineSpeakerTests + CLI-tesztek), `tests/validation/test_feature_speaker.py`, ez a CONTRACT-addendum |
