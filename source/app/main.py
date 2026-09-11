"""M1/M2/M3 CLI entry point (CONTRACT.md).

Usage::

    python -m app.main [--config PATH] [--mock] [--text "..."]
                       [--barge-in-demo] [--emotion-demo] [--speaker-demo]
                       [--check] [--list-config] [--register-speaker NAME]

  --mock             : scripted 3-turn demo with mock components
                       (EN grammar error -> HU follow-up -> memory recall).
  --mock --text "..." : one custom scripted turn.
  --mock --barge-in-demo : scripted turn where the user interrupts the TTS
                       playback -> cancelled=True.
  --mock --emotion-demo : M2 scripted demo - three turns with prosody
                       emotions (frustrated -> neutral -> happy) showing
                       the emotion prompt block and the slowed-down TTS.
  --mock --speaker-demo : M3 scripted demo - four turns alternating TWO
                       speakers (thomas -> other -> thomas -> unknown),
                       showing per-speaker memory routing and the unknown
                       fallback.
  --register-speaker NAME : M3 registration - records ~10 s of mic speech
                       (real mode only; requires --mock absent) and stores
                       the reference embedding in data/speaker_registry.json.
  --check            : runtime asset checklist, exit 0/1 (emotion and
                       speaker assets are printed as info - they degrade,
                       never block).
  --list-config      : resolved configuration as JSON, exit 0.

Import safety: this module imports ONLY the standard library plus the light
app modules (config/pipeline) at import time — ``tests/offline_test.py``
imports it. Real-mode heavy imports (torch, sounddevice, onnxruntime,
voicemem) happen inside :func:`run_real`.

Real mode runs two loops: ``_vad_loop`` drains mic frames, runs Silero VAD +
the state machine and emits complete utterances (loopback gating: frames
while TTS plays never reach ASR, the VAD state machine is reset on gate
close — see app/pipeline.py docstring); ``_turn_loop`` feeds utterances into
:class:`VoicePipeline.handle_utterance` and prints the exchange.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import AgentConfig
from app.pipeline import TurnResult, VoicePipeline

logger = logging.getLogger(__name__)

# --- scripted demo material (Hungarian teacher persona, M1 mock demo) -------

DEMO_TRANSCRIPTS = [
    "Why is I have went wrong?",
    "És magyarul hogy mondanád ezt?",
    "Do I still make the same mistake?",
]

DEMO_REPLIES = [
    "Kis javítás következik. Nem azt mondjuk angolul, hogy why is I have went. "
    "A helyes kérdés: why have I gone wrong? "
    "Első példa: Why have I gone wrong so many times? "
    "Második példa: Where have you gone today? "
    "Most mondd utánam lassan: why have I gone wrong.",
    "Magyarul így mondanád: miért tettem rosszul? "
    "Angolban a have és a gone mindig együtt jár ebben az igeidőben. "
    "A magyar mondat egyszerűbb, mert nincs segédige benne. "
    "Próbáld meg még egyszer angolul, és javítom ha kell.",
    "Igen, emlékszem a korábbi hibádra. "
    "Te hajlamos vagy összekeveri a have és a gone párosát. "
    "A helyes alak: why have I gone wrong. "
    "Figyeld a have gone párost a kérdésben. "
    "Csináljunk még két példát a gyakorláshoz?",
]

DEMO_MEMORY_CONTEXTS = [
    "",
    "",
    "- The user repeatedly says 'why is I have went' instead of "
    "'why have I gone wrong'.\n"
    "- The user is practising the English present perfect tense.",
]

BARGE_IN_TRANSCRIPT = (
    "Meséld el még egyszer, mi a különbség a flat white és a tejeskávé között."
)

BARGE_IN_REPLY = (
    "Röviden összefoglalom a különbséget. "
    "A flat white kisebb, kevesebb tejjel készül. "
    "Vékony mikrohabréteget használ, ami sima felületet ad. "
    "A tejeskávé nagyobb arányban tartalmaz forró tejet. "
    "Vastagabb habrétege krémesebb italt eredményez. "
    "Ezért tompítja az espresso keserűségét is. "
    "A latté tehát ennél még lágyabb ízű ital lesz."
)

# --- M2 emotion demo material (scripted prosody states) ----------------------
# Turn 1: the learner is stuck and angry -> fused "frustrated" -> the reply
# slows down (length_scale 1.1). Turn 2: neutral. Turn 3: happy/confident.

DEMO_EMOTION_TRANSCRIPTS = [
    "Why is I have went wrong? I do not understand this at all!",
    "És magyarul hogy mondanád ezt?",
    "Got it, köszönöm, now I understand the rule!",
]

DEMO_EMOTION_RAW_LABELS = ["angry", "neutral", "happy"]

DEMO_EMOTION_REPLIES = [
    "Semmi gond, nézzük meg lassan, lépésről lépésre. "
    "A kérdés helyesen: why have I gone wrong? "
    "Példa: Why have I gone wrong so many times? "
    "Ezt most lassabban mondom, figyeld a have gone párost.",
    "Magyarul így mondanád: miért tettem rosszul? "
    "Angolban a have és a gone mindig együtt jár ebben az igeidőben. "
    "Próbáld meg még egyszer angolul, és javítom ha kell.",
    "Nagyon szuper, hogy sikerült megértened! "
    "A have gone párost most már jól használod. "
    "Csináljunk még két gyors példát a gyakorláshoz?",
]


# --- M3 speaker demo material (scripted identifications) ----------------------
# Four turns alternating two REGISTERED speakers plus one unknown voice:
# turn 1/3 = thomas (memory space A), turn 2 = other_speaker (memory space
# B), turn 4 = unknown speaker -> fallback user "voice_user". The per-turn
# [M3 speaker: ...] lines and the routing summary prove the separation.

DEMO_SPEAKER_TRANSCRIPTS = [
    "Why is I have went wrong?",
    "Why is I have went wrong?",
    "Do I still make the same mistake?",
    "Meseld el meg egyszer, mi a kulonbseg.",
]

DEMO_SPEAKER_REPLIES = [
    "Kis javitas kovetkezik. Nem azt mondjuk angolul, hogy why is I have went. "
    "A helyes kerdes: why have I gone wrong? Most mondd utanam lassan.",
    "Roviden: a have es a gone mindig egyutt jar ebben az igeidoben. "
    "Pelda: Where have you gone today? Probald meg meg egyszer angolul.",
    "Igen, emlekszem a korabbi hibadra. Te hajlamos vagy osszekeverni a "
    "have es a gone parosat. Figyeld a have gone parost a kerdesben.",
    "Rendben, lassan elmondom meg egyszer. A flat white kisebb, kevesebb "
    "tejjel keszul, vastag mikrohabreteggel.",
]

DEMO_SPEAKER_MEMORY_CONTEXTS = [
    "- The user repeatedly says 'why is I have went' instead of "
    "'why have I gone wrong'.",
    "- The user is practising the present perfect tense.",  # other space
    "- The user repeatedly says 'why is I have went' instead of "
    "'why have I gone wrong'.\\n- The user is practising the English "
    "present perfect tense.",
    "",  # unknown speaker -> default space (no memories)
]


# --- mock demo ----------------------------------------------------------------


def _barge_in_prob_fn(start_after: int = 2) -> Callable[[], float]:
    """Scripted VAD probe: silence first, then sustained speech (barge-in)."""
    state = {"calls": 0}

    def prob() -> float:
        state["calls"] += 1
        return 0.95 if state["calls"] > start_after else 0.02

    return prob


async def run_mock_demo(
    config: AgentConfig,
    custom_text: Optional[str] = None,
    barge_in: bool = False,
    emotion_demo: bool = False,
    speaker_demo: bool = False,
) -> int:
    """Scripted demo through the full pipeline with mock components."""
    import numpy as np

    from app.emotion import EmotionMemory, EmotionResult
    from app.mock_components import (
        MockAsrEngine,
        MockEmotionAnalyzer,
        MockLlmClient,
        MockSpeaker,
        MockSpeakerRecognizer,
        MockTtsEngine,
        MockVad,
        MockVoiceMemBridge,
    )
    from app.speaker import SpeakerIdentification

    if custom_text is not None:
        transcripts: list[str] = [custom_text]
        replies: list[str] = [DEMO_REPLIES[0]]
        contexts: list[str] = [DEMO_MEMORY_CONTEXTS[-1]]
    elif barge_in:
        transcripts = [BARGE_IN_TRANSCRIPT]
        replies = [BARGE_IN_REPLY]
        contexts = [""]
    elif emotion_demo:
        transcripts = list(DEMO_EMOTION_TRANSCRIPTS)
        replies = list(DEMO_EMOTION_REPLIES)
        contexts = [""] * len(transcripts)
    elif speaker_demo:
        transcripts = list(DEMO_SPEAKER_TRANSCRIPTS)
        replies = list(DEMO_SPEAKER_REPLIES)
        contexts = list(DEMO_SPEAKER_MEMORY_CONTEXTS)
    else:
        transcripts = DEMO_TRANSCRIPTS
        replies = DEMO_REPLIES
        contexts = DEMO_MEMORY_CONTEXTS

    asr = MockAsrEngine(transcripts)
    llm = MockLlmClient(replies)
    tts = MockTtsEngine()
    voicemem = MockVoiceMemBridge(contexts)
    vad = MockVad()
    speaker = MockSpeaker()

    emotion_analyzer: Optional[MockEmotionAnalyzer] = None
    emotion_memory: Optional[EmotionMemory] = None
    if emotion_demo and not config.enable_emotion:
        print(
            "FIGYELEM: enable_emotion=false - az emotion demó M1-modban fut "
            "(nincs proszodia-elemzés)."
        )

    speaker_recognizer: Optional[MockSpeakerRecognizer] = None
    if speaker_demo:
        if config.enable_speaker:
            scripted_ids: list[Optional[str]] = [
                "thomas",
                "other_speaker",
                "thomas",
                None,  # unknown speaker -> fallback user
            ]
            scripted = [
                SpeakerIdentification(
                    id=spk,
                    similarity=0.82 if spk else 0.31,
                    best_id=spk or "thomas",
                    registered=2,
                    latency_ms=38.0,
                )
                for spk in scripted_ids
            ]
            speaker_recognizer = MockSpeakerRecognizer(scripted)
        else:
            print(
                "FIGYELEM: enable_speaker=false - a speaker demó M1/M2-modban "
                "fut (nincs beszelo-azonositas, minden fordulo voice_user)."
            )

    if emotion_demo:
        if config.enable_emotion:
            scripted: list[Optional[EmotionResult]] = [
                EmotionResult(
                    raw_label=raw,
                    label={
                        "angry": "frustrated",
                        "neutral": "neutral",
                        "happy": "happy",
                    }[raw],
                    valence={"angry": -0.8, "neutral": 0.0, "happy": 0.8}[raw],
                    arousal={"angry": 0.8, "neutral": 0.0, "happy": 0.5}[raw],
                    confidence=0.9,
                )
                for raw in DEMO_EMOTION_RAW_LABELS
            ]
            emotion_analyzer = MockEmotionAnalyzer(scripted)
            emotion_memory = EmotionMemory(config.data_dir / "emotion_demo_log.jsonl")

    pipeline = VoicePipeline(
        config,
        asr=asr,
        llm=llm,
        tts=tts,
        voicemem=voicemem,
        vad=vad,
        audio_out=speaker,
        emotion=emotion_analyzer,
        emotion_memory=emotion_memory,
        speaker=speaker_recognizer,
    )

    if emotion_demo:
        mode = "emotion demo (M2 proszodia: frusztralt -> semleges -> vidam)"
    elif speaker_demo:
        mode = "speaker demo (M3 beszelok: thomas -> masik -> thomas -> ismeretlen)"
    elif barge_in:
        mode = "barge-in demo (a felhasználó megszakítja a választ)"
    else:
        mode = "mock demo (3 forgatókönyv szerinti forduló)"
    print("=" * 72)
    print(f"VoiceMem M1/M2/M3 — {mode}")
    print("=" * 72)

    exit_code = 0
    for index in range(len(transcripts)):
        audio = np.zeros(int(config.sample_rate * 1.5), dtype=np.float32)
        prob_fn: Callable[[], float] = _barge_in_prob_fn() if barge_in else (lambda: 0.0)
        result = await pipeline.handle_utterance(audio, vad_prob_fn=prob_fn)
        payload: dict[str, Any] = result.to_dict()
        payload["turn"] = index + 1
        payload["memory_context"] = contexts[index] or None
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if speaker_demo:
            spk = result.speaker
            if spk is not None:
                print(
                    f"  [M3 beszelő: {spk.get('id') or 'ISMERETLEN'} "
                    f"(hasonlóság={spk.get('similarity')}, "
                    f"{spk.get('latency_ms')} ms) -> memóriatér: {result.speaker_id}]"
                )
            else:
                print(f"  [M3 kikapcsolva -> memóriatér: {result.speaker_id}]")
        if emotion_demo and tts.synthesized:
            scales = sorted(
                {ls for _t, _l, ls, _v in tts.synthesized},
                key=lambda value: (value is None, value if value is not None else 0.0),
            )
            print(f"  TTS length_scale ebben a fordulóban: {scales}")
        print("-" * 72)
        if not result.reply and not result.cancelled:
            print(f"HIBA: {index + 1}. forduló üres választ adott.", file=sys.stderr)
            exit_code = 1

    if barge_in:
        ok = result.cancelled and result.barge_in
        print(f"barge-in eredmény: cancelled={result.cancelled} barge_in={result.barge_in}")
        if not ok:
            print("HIBA: a barge-in demó nem szakította meg a választ.", file=sys.stderr)
            exit_code = 1
    elif speaker_demo:
        if config.enable_speaker:
            routed = voicemem.turn_speakers
            expected = ["thomas", "other_speaker", "thomas", "voice_user"]
            routed_ok = routed == expected
            committed = voicemem.commit_speakers
            committed_ok = all(
                spk in ("thomas", "other_speaker", "voice_user") for spk in committed
            ) and len(committed) == len(voicemem.committed)
            contamination = [
                (i, spk)
                for i, spk in enumerate(routed)
                if spk not in expected
            ]
            print(
                f"Összegzés: {len(transcripts)} forduló, memória-útválasztás: "
                f"{'HELYES' if routed_ok else 'HIBÁS'} {routed}, "
                f"commit-útválasztás: {'HELYES' if committed_ok else 'HIBÁS'} {committed}, "
                f"cross-speaker kontamináció: {len(contamination)}"
            )
            if not routed_ok or not committed_ok or contamination:
                print(
                    "HIBA: a speaker demó útválasztása hibás (elvárt: "
                    f"{expected}).",
                    file=sys.stderr,
                )
                exit_code = 1
        else:
            print(
                "Összegzés: enable_speaker=false - a beszélő-azonosítás ki volt "
                "kapcsolva (M1/M2-mod, minden forduló voice_user)."
            )
    elif emotion_demo:
        if config.enable_emotion:
            slowed = any(ls is not None and ls > 1.0 for _t, _l, ls, _v in tts.synthesized)
            with_emotion = any(
                "frustrated" in (call[0]["content"] if call else "") for call in llm.calls
            )
            print(
                f"Összegzés: {len(transcripts)} forduló, emotion-block a promptban: "
                f"{'igen' if with_emotion else 'NEM'}, "
                f"lassított TTS (length_scale > 1.0): {'igen' if slowed else 'NEM'}, "
                f"emotion log: {emotion_memory.path if emotion_memory else '-'}"
            )
            if not with_emotion or not slowed:
                print(
                    "HIBA: az emotion demó nem aktiválta az adaptációt "
                    "(prompt-block es/vagy length_scale).",
                    file=sys.stderr,
                )
                exit_code = 1
        else:
            print(
                "Összegzés: enable_emotion=false - az emotion adaptáció ki volt "
                "kapcsolva (M1-mod)."
            )
    else:
        committed = len(voicemem.committed)
        print(
            f"Összegzés: {len(transcripts)} forduló, {committed} válasz került a "
            "hosszú távú memóriába (commit_reply)."
        )
    return exit_code


# --- real mode -----------------------------------------------------------------


def _print_check(config: AgentConfig) -> int:
    """--check: runtime asset checklist; 0 when everything is present."""
    assets = config.check_runtime_assets()
    print("VoiceMem M1/M2/M3 — futásidejű eszközök ellenőrzése")
    print(f"root: {config.root}")
    print("-" * 60)
    width = max(len(name) for name in assets)
    for name, ok in assets.items():
        print(f"  {name:<{width}}  {'OK' if ok else 'HIÁNYZIK'}")
    print("-" * 60)
    emotion_assets = config.check_emotion_assets()
    emotion_ok = all(emotion_assets.values())
    for name, ok in emotion_assets.items():
        print(f"  {name:<{width}}  {'OK' if ok else 'HIÁNYZIK'} (M2, opcionális)")
    print("-" * 60)
    speaker_assets = config.check_speaker_assets()
    speaker_ok = all(speaker_assets.values())
    for name, ok in speaker_assets.items():
        print(f"  {name:<{width}}  {'OK' if ok else 'HIÁNYZIK'} (M3, opcionális)")
    print("-" * 60)
    if all(assets.values()):
        if config.enable_speaker and not speaker_ok:
            print(
                "Eredmény: az M1 eszközök rendben. Az M3 beszélő-elemző hiányzik -"
            )
            print(
                "          a futás M1/M2-modban folytatódik (graceful degradation)."
            )
            print("          Letöltés: scripts/download_models.ps1 (speaker komponens),")
            print("          regisztráció: python -m app.main --register-speaker <név>.")
        elif config.enable_emotion and not emotion_ok:
            print(
                "Eredmény: az M1 eszközök rendben. Az M2 érzelemmodell hiányzik -"
            )
            print(
                "          a futás M1-modban folytatódik (graceful degradation)."
            )
            print("          Letöltés: scripts/download_models.ps1 (emotion komponens).")
        else:
            print("Eredmény: minden eszköz jelen van. Indítás: scripts/start_agent.ps1")
        return 0
    print("Eredmény: hiányzó eszközök — futtasd a scripts/download_models.ps1-t")
    print("(a két kézi letöltés: llama.cpp CUDA zip és piper.exe — lásd README).")
    return 1


async def run_register_speaker(config: AgentConfig, speaker_name: str) -> int:
    """M3 spec 9.3.3: register a speaker from ~10 s of clean mic speech.

    Records while you speak (VAD-gated: only speech frames accumulate),
    embeds the collected audio, averages the embeddings and stores the
    reference in ``data/speaker_registry.json``. Re-registering the same
    name OVERWRITES the reference (idempotent).
    """
    import numpy as np

    from app.speaker import SpeakerEmbedder, SpeakerRecognizer, SpeakerRegistry
    from app.vad import SileroVad, VadStateMachine, VadEvent

    name = (speaker_name or "").strip()
    if not name:
        print("HIBA: adj meg egy beszélő-nevet: --register-speaker <név>", file=sys.stderr)
        return 2

    embedder = SpeakerEmbedder(
        config.speaker_model_dir,
        window_s=config.speaker_window_s,
        sample_rate=config.sample_rate,
    )
    if not embedder.is_available():
        print("HIBA: az M3 beszélőmodell nem elérhető (models/speaker/ecapa-voxceleb).")
        print("      Telepítés: scripts/download_models.ps1 (speaker komponens).")
        return 1
    if not embedder.warm_up():
        print("HIBA: az M3 beszélőmodell betöltése nem sikerült (graceful: nem regisztráltunk).")
        return 1

    from app.audio_io import MicStream

    target_s = config.speaker_registration_min_s
    frame_s = config.vad_frame_ms / 1000.0
    print("=" * 72)
    print(f"M3 regisztráció: {name} — beszélj ~{target_s:.0f} másodpercig tiszta beszéddel")
    print("(a VAD csak a tényleges beszédet számolja; Ctrl+C = megszakítás)")
    print("=" * 72)

    collected: list[Any] = []
    speech_seconds = 0.0
    done = {"flag": False}

    try:
        vad = SileroVad(config)
        state = VadStateMachine(
            config.vad_threshold, config.vad_hangover_ms, config.vad_frame_ms
        )

        def _on_frame(frame: Any) -> None:
            nonlocal speech_seconds
            if done["flag"]:
                return
            prob = float(vad.prob(frame))
            event = state.update(prob)
            if state.in_speech:
                collected.append(frame)
                speech_seconds += frame_s
            if event == VadEvent.SPEECH_END and collected:
                # keep every completed speech burst; stop at the target
                if speech_seconds >= target_s:
                    done["flag"] = True
                return
            if speech_seconds >= target_s:
                done["flag"] = True

        with MicStream(config, _on_frame):
            import time as _time

            deadline = _time.time() + 60.0
            while not done["flag"] and _time.time() < deadline:
                await asyncio.sleep(0.05)
    except KeyboardInterrupt:
        print("\nMegszakítás — a már gyűjtött beszéddel próbálkozunk.")

    if not collected:
        print("HIBA: nem gyűlt össze beszéd — nem regisztráltunk.", file=sys.stderr)
        return 1

    audio = np.concatenate(collected)
    registry = SpeakerRegistry(config.speaker_registry_file)
    recognizer = SpeakerRecognizer(
        embedder,
        registry,
        threshold=config.speaker_match_threshold,
        registration_min_s=config.speaker_registration_min_s,
        sample_rate=config.sample_rate,
    )
    ok, seconds = recognizer.register(name, [audio])
    if not ok:
        print("HIBA: a regisztráció nem sikerült (embedding üres?).", file=sys.stderr)
        return 1
    print(f"Regisztrálva: {name} ({seconds:.1f} s beszéd, {registry.count()} beszélő a registry-ben)")
    print(f"Registry: {config.speaker_registry_file}")
    if seconds < target_s * 0.6:
        print(f"FIGYELEM: csak {seconds:.1f} s beszéd — a spec ~{target_s:.0f} s-t ajánl.")
    return 0


async def run_real(config: AgentConfig) -> int:
    """Real conversation loop: mic -> VAD -> pipeline -> speaker."""
    missing = [name for name, ok in config.check_runtime_assets().items() if not ok]
    if missing:
        print("Hiányzó futásidejű eszközök: " + ", ".join(missing))
        print("Telepítés: scripts/download_models.ps1, ellenőrzés: python scripts/verify_setup.py")
        return 1

    import numpy as np  # local import: heavy stack comes with real mode

    from app.asr_core import select_engine
    from app.audio_io import MicStream, SpeakerOutput
    from app.emotion import EmotionAnalyzer, EmotionMemory
    from app.llm import LlmClient
    from app.speaker import SpeakerEmbedder, SpeakerRecognizer, SpeakerRegistry
    from app.tts import TtsEngine
    from app.vad import SileroVad, VadEvent, VadStateMachine
    from app.voicemem_bridge import VoiceMemBridge

    # v0.6.0: the ONE selected engine (ASR_ENGINE / yaml asr.engine); a
    # failed load raises an explicit AsrError with the exact reason — there
    # is NO fallback to another engine.
    try:
        asr = select_engine(config)
    except Exception as exc:
        print(f"HIBA: az ASR motor ({config.asr_engine}) nem elérhető: {exc}")
        print("      Állítsd be ASR_ENGINE-t / asr.engine-t, vagy töltsd le a modellt")
        print("      (scripts/download_models.ps1). NINCS automatikus motor-váltás.")
        return 1
    llm = LlmClient(config)
    tts = TtsEngine(config)
    voicemem = VoiceMemBridge(config, llm)
    vad = SileroVad(config)
    speaker = SpeakerOutput(config)

    emotion = None
    emotion_memory = None
    if config.enable_emotion:
        emotion = EmotionAnalyzer(
            config.emotion_model_dir,
            window_s=config.emotion_window_s,
            sample_rate=config.sample_rate,
        )
        if not emotion.is_available():
            print(
                "FIGYELEM: az M2 érzelem-elemző nem elérhető (modell vagy funasr "
                "hiányzik) - M1-modban fut tovább a pipeline."
            )
            print("          Modell: models/emotion/emotion2vec-plus-base (download_models).")
            emotion = None
        elif not emotion.warm_up():
            print(
                "FIGYELEM: az M2 érzelemmodell betöltése nem sikerült - "
                "M1-modban fut tovább a pipeline (graceful degradation)."
            )
            emotion = None
        else:
            emotion_memory = EmotionMemory(config.emotion_log_path)
            print(
                f"M2 emotion: aktiv (emotion2vec+ base, CPU, ablak={config.emotion_window_s:.0f}s, betoltve)"
            )

    speaker_recognizer = None
    if config.enable_speaker:
        embedder = SpeakerEmbedder(
            config.speaker_model_dir,
            window_s=config.speaker_window_s,
            sample_rate=config.sample_rate,
        )
        registry = SpeakerRegistry(config.speaker_registry_file)
        speaker_recognizer = SpeakerRecognizer(
            embedder,
            registry,
            threshold=config.speaker_match_threshold,
            registration_min_s=config.speaker_registration_min_s,
            sample_rate=config.sample_rate,
        )
        if not speaker_recognizer.is_available():
            print(
                "FIGYELEM: az M3 beszélő-felismerő nem elérhető (modell vagy "
                "speechbrain hiányzik) - M1/M2-modban fut tovább a pipeline."
            )
            print("          Modell: models/speaker/ecapa-voxceleb (download_models).")
            speaker_recognizer = None
        elif not speaker_recognizer.warm_up():
            print(
                "FIGYELEM: az M3 beszélőmodell betöltése nem sikerült - "
                "M1/M2-modban fut tovább a pipeline (graceful degradation)."
            )
            speaker_recognizer = None
        else:
            registered = registry.count()
            print(
                f"M3 speaker: aktiv (ECAPA-voxceleb, CPU, ablak={config.speaker_window_s:.0f}s, "
                f"küszöb={config.speaker_match_threshold:.2f}, regisztrált beszélők: {registered})"
            )
            if not registered:
                print(
                "          Még nincs regisztrált beszélő - minden hang 'voice_user' "
                "lesz, amíg nem regisztrálsz:"
                )
                print("          python -m app.main --register-speaker <név> (~10 s beszéd)")

    pipeline = VoicePipeline(
        config,
        asr=asr,
        llm=llm,
        tts=tts,
        voicemem=voicemem,
        vad=vad,
        audio_out=speaker,
        emotion=emotion,
        emotion_memory=emotion_memory,
        speaker=speaker_recognizer,
    )

    if not asr.is_available():
        print("HIBA: torch/transformers nincs telepítve — az ASR nem tud működni.")
        print("      Telepítés: lásd README (torch cu128 + requirements.txt).")
        return 1
    if not await llm.health_check():
        print(f"FIGYELEM: a llama-server nem érhető el: {config.llama_server_health_url}")
        print("          Indítsd el: scripts/start_llama_server.ps1 (külön ablakban).")
    if not tts.is_available():
        print("FIGYELEM: Piper hangok hiányoznak — a válaszok csak szövegben jelennek meg.")
    if not voicemem.is_available():
        print("FIGYELEM: a voicemem csomag nincs telepítve — hosszú távú memória NÉLKÜL fut.")

    state = VadStateMachine(
        config.vad_threshold, config.vad_hangover_ms, config.vad_frame_ms
    )
    frame_q: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
    utterance_q: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
    latest: dict[str, float] = {"prob": 0.0}
    running: dict[str, bool] = {"flag": True}
    utterance: list[Any] = []
    speech_ms: dict[str, float] = {"value": 0.0}
    # v0.4.4: the web server has had a 12 s forced utterance end since v0.4.2
    # (web_server.py _MAX_UTTERANCE_MS); the CLI loop never got it. A VAD stuck
    # in speech (continuous music/noise above the threshold) collected the
    # utterance forever: no turn ever fired, then one giant ASR flush.
    max_utterance_ms = 12_000.0

    def _on_frame(frame: Any) -> None:
        frame_q.put(frame)

    async def _vad_loop() -> None:
        """Drain mic frames, run VAD, emit complete utterances (loopback gate)."""
        nonlocal utterance
        while running["flag"]:
            drained = False
            while True:
                try:
                    frame = frame_q.get_nowait()
                except queue.Empty:
                    break
                drained = True
                try:
                    prob = float(vad.prob(frame))
                except Exception:  # noqa: BLE001 - one bad frame never stops the loop
                    logger.exception("VAD frame failed (frame dropped)")
                    continue
                latest["prob"] = prob
                if pipeline.echo_gate.tts_playing:
                    # Loopback gating: frames during TTS playback never reach
                    # ASR; a mid-speech gate close drops the partial utterance
                    # (the user re-speaks; real AEC arrives in V1.5).
                    if state.in_speech:
                        state.reset()
                        utterance.clear()
                        speech_ms["value"] = 0.0
                    continue
                event = state.update(prob)
                if event == VadEvent.SPEECH_START:
                    utterance.clear()
                    speech_ms["value"] = 0.0
                    print("\n  [beszéd indul ...]", end="", flush=True)
                if state.in_speech:
                    utterance.append(frame)
                    speech_ms["value"] += config.vad_frame_ms
                forced_end = (
                    state.in_speech
                    and event != VadEvent.SPEECH_END
                    and speech_ms["value"] >= max_utterance_ms
                )
                if event == VadEvent.SPEECH_END or forced_end:
                    if forced_end:
                        state.reset()
                        speech_ms["value"] = 0.0
                        print(
                            f"\n  [kényszerlezárás {max_utterance_ms / 1000.0:.0f} s után "
                            "(a VAD beszédben ragadt)]",
                            end="",
                            flush=True,
                        )
                    audio = None
                    if utterance:
                        audio = np.concatenate(utterance)
                    utterance.clear()
                    if audio is not None and audio.size:
                        utterance_q.put(audio)
            if not drained:
                await asyncio.sleep(0.005)

    async def _turn_loop() -> None:
        """Consume completed utterances and drive the pipeline."""
        while running["flag"]:
            try:
                audio = utterance_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            result: TurnResult = await pipeline.handle_utterance(
                audio, vad_prob_fn=lambda: latest["prob"]
            )
            print()
            print(f"Te    : {result.transcript}")
            print(f"Agent : {result.reply or '(üres válasz)'}")
            if result.emotion:
                emo = result.emotion
                print(
                    f"      [M2 érzelem: {emo.get('label')} (valence={emo.get('valence')}, "
                    f"arousal={emo.get('arousal')}, {emo.get('latency_ms')} ms)]"
                )
            if result.speaker:
                spk = result.speaker
                print(
                    f"      [M3 beszélő: {spk.get('id') or 'ISMERETLEN'} "
                    f"(hasonlóság={spk.get('similarity')}, {spk.get('latency_ms')} ms) "
                    f"-> memóriatér: {result.speaker_id}]"
                )
            t = result.timings
            if t.speech_end_s is not None and t.first_audio_s is not None:
                print(
                    f"      [{result.language}] első hang: "
                    f"{t.first_audio_s - t.speech_end_s:.2f} s"
                )
            if result.barge_in:
                print("      [megszakítottad a választ — figellek újra]")

    print("=" * 72)
    print("VoiceMem M1/M2/M3 — valós mód (beszélj; Ctrl+C a kilépés)")
    print(f"root: {config.root} | llama-server: {config.llama_server_url}")
    print("=" * 72)
    tasks: list[asyncio.Task[Any]] = []
    try:
        with MicStream(config, _on_frame):
            tasks = [
                asyncio.create_task(_vad_loop(), name="vad_loop"),
                asyncio.create_task(_turn_loop(), name="turn_loop"),
            ]
            await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass  # Ctrl+C: asyncio.run cancels this coroutine -> finally cleans up
    finally:
        running["flag"] = False
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await llm.aclose()
        print("\nViszlát!")
    return 0


# --- CLI ------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.main", description="VoiceMem M1/M2/M3 voice agent (mock demó és valós mód)"
    )
    parser.add_argument("--config", help="opcionális YAML konfig (config/voicemem_config.yaml)")
    parser.add_argument("--mock", action="store_true", help="mock demó komponensekkel")
    parser.add_argument("--text", help="egyéni forduló szövege (csak --mock mellett)")
    parser.add_argument(
        "--barge-in-demo", action="store_true", help="megszakítási demó (csak --mock mellett)"
    )
    parser.add_argument(
        "--emotion-demo",
        action="store_true",
        help="M2 érzelem demó: frusztrált -> semleges -> vidam (csak --mock mellett)",
    )
    parser.add_argument(
        "--speaker-demo",
        action="store_true",
        help="M3 beszélő demó: thomas -> másik -> thomas -> ismeretlen (csak --mock mellett)",
    )
    parser.add_argument(
        "--register-speaker",
        metavar="NÉV",
        help="M3 regisztráció: ~10 s mikrofon-beszéd a data/speaker_registry.json-be (valós mód)",
    )
    parser.add_argument("--check", action="store_true", help="eszköz-ellenőrzés és kilépés")
    parser.add_argument("--list-config", action="store_true", help="konfiguráció JSON-ként")
    return parser


def load_config(path: Optional[str]) -> AgentConfig:
    """YAML (ha van) + env változók + alapértémek sorrend feloldása."""
    if path:
        return AgentConfig.from_yaml(Path(path))
    config = AgentConfig()
    config.apply_env()
    return config


def _make_stdio_resilient() -> None:
    """Console-encoding crash guard (target field report #4).

    On Windows the launcher scripts capture this process's output through
    a pipe, so Python encodes stdout/stderr with the ANSI code page
    (cp1252) instead of using the UTF-8 console API - and the Hungarian
    checklist text then died with UnicodeEncodeError ("'charmap' codec
    can't encode '\\u0171'") before printing a single line. Reconfiguring
    the streams with errors="replace" guarantees no entry point (--check,
    --list-config, mock, real) can crash on encoding; unencodable
    characters degrade to '?' instead. (The PowerShell launchers also set
    PYTHONIOENCODING=utf-8 so the text usually renders correctly.)
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - exotic stdout setups (pythonw)
            pass


async def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _make_stdio_resilient()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.list_config:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False, default=str))
        return 0
    if args.check:
        return _print_check(config)
    if args.text is not None and not args.mock:
        print("HIBA: a --text csak --mock mellett használható.", file=sys.stderr)
        return 2
    if args.barge_in_demo and not args.mock:
        print("HIBA: a --barge-in-demo csak --mock mellett használható.", file=sys.stderr)
        return 2
    if args.emotion_demo and not args.mock:
        print("HIBA: a --emotion-demo csak --mock mellett használható.", file=sys.stderr)
        return 2
    if args.speaker_demo and not args.mock:
        print("HIBA: a --speaker-demo csak --mock mellett használható.", file=sys.stderr)
        return 2
    if args.register_speaker is not None and args.mock:
        print("HIBA: a --register-speaker NEM --mock módban fut (mikrofon kell).", file=sys.stderr)
        return 2
    if args.register_speaker is not None:
        return await run_register_speaker(config, args.register_speaker)
    if args.mock:
        return await run_mock_demo(
            config,
            custom_text=args.text,
            barge_in=args.barge_in_demo,
            emotion_demo=args.emotion_demo,
            speaker_demo=args.speaker_demo,
        )
    return await run_real(config)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nMegszakítás — kilépés.")
        sys.exit(130)
