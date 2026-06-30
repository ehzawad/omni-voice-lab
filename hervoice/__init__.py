"""HerVoice: a Bengali-first, local, turn-based voice-to-voice assistant.

Modular pipeline (NOT a single end-to-end network): spoken Bengali question ->
ASR (faster-whisper) -> brain (Qwen2.5-3B, optional gated RAG) -> Bengali speech
out (orpheus-bangla-tts + LoRA + SNAC). One process, sequential/lazy GPU
residency (one heavy model resident at a time), turn-based (not full-duplex).
"""
from .core import HerVoice

__all__ = ["HerVoice"]
