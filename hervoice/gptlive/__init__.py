"""gptlive — a local, open-weight mirror of OpenAI's (API-only) GPT-Live.

A native full-duplex voice FRONT (Moshi) that DELEGATES hard questions to a
local reasoning brain (Qwen3.5-4B GGUF via llama-server), all on one RTX A5000.

Modules:
  front.py     -- Moshi driver: full-duplex listen + teacher-forced speak.
  delegate.py  -- the controller: classify a user turn, call the brain.
  pipeline.py  -- wires it end to end and writes headless proof artifacts.
"""
