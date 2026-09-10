"""VoiceMem M1 — fully local Hungarian-English voice agent (Milestone 1).

Pipeline: mic -> VAD -> ASR -> VoiceMem memory -> local LLM -> local TTS -> speaker.
All components run offline on the target machine (Windows 11 + RTX 5070).
"""

__version__ = "0.1.0-m1"
