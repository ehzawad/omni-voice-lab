#!/bin/bash
# HerVoice (English path): SINGLE-NETWORK voice-to-voice.
#
# Unlike the Bengali path (modular: python -m hervoice), English runs end-to-end in ONE neural
# network: MiniCPM-o 4.5 does ASR (its audio encoder), reasoning (the shared LLM "Thinker"), and
# speech-out (the "Talker" + vocoder) in a single model. This works for English because the omni
# Talker is trained on English speech; it does NOT work for Bengali (the Talker is EN/ZH-only),
# which is exactly why the Bengali path is modular.
#
# Spoken English question -> MiniCPM-o ASR -> RAG over fifa_kb.md -> grounded English spoken answer.
#
#   ./hervoice_en.sh [audio_in.wav] [answer_out.wav]
#
# Measured (RTX A5000): grounded answer correct, ~14.2 GB VRAM, ~19 s generation. A plain
# (non-RAG) single-network voice turn is minicpm_voice.py; streaming with first-audio latency
# (~2.2 s, RTF ~1.12 at int4) is minicpm_stream.py.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-0}"
AUDIO="${1:-examples/in_fifa_question.wav}"
OUT="${2:-runs/hervoice_en/answer.wav}"
mkdir -p "$(dirname "$OUT")"
exec .venv/bin/python fifa_voice.py --audio "$AUDIO" --ref examples/ref_female.wav --out "$OUT"
