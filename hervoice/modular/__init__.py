"""Modular English voice assistant pipeline.

Stack:  spoken English -> ASR -> Qwen3.5-4B GGUF brain (llama-server) -> Qwen3-TTS -> spoken English.

Submodules:
  asr.py       unified transcribe(model_key, wav) over 5 ASR backends (2 venvs)
  brain.py     llama-server OpenAI client with a concise-spoken-answer system prompt
  tts.py       Qwen3-TTS voice-clone wrapper (text -> wav)
  pipeline.py  end-to-end wav -> ASR -> brain -> TTS -> wav (+ manifest)
  asr_bench.py the 5-model English ASR bake-off (WER/CER/latency/RTF/VRAM/features)

VENVs (see docs/MODULAR_PIPELINE.md):
  .venv-qwen-asr   : Qwen3-ASR (0.6B + 1.7B; transformers 5.x)
  .venv-funasr     : SenseVoiceSmall + paraformer-zh + Fun-ASR-Nano  (funasr)
  .venv-qwen-audio : Qwen3-TTS (qwen-tts pins transformers 4.57.x)

The brain is a separate llama-server process (no python deps); talk to it over HTTP.
GPU: everything is pinned to GPU0 by the launch scripts. GPU1 is never touched.
"""
