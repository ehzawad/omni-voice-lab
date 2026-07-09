#!/usr/bin/env python3
"""Independent re-ASR readability check for arch-2 MiniCPM-o S2S output.

Runs Qwen3-ASR (a SEPARATE network, in .venv-qwen-asr / transformers 5.x) on an
output wav to re-transcribe what the MiniCPM-o Talker actually spoke. This is a
READABILITY / intelligibility proxy, NOT a WER measurement (there is no
ground-truth transcript for a free-form spoken reply).

Kept deliberately separate from the generation path: the MiniCPM-o network alone
does speech-understanding AND speech-generation for arch 2. This ASR touches only
the already-written output wav, never the generation.

Usage (must run in .venv-qwen-asr):
    python -m hervoice.arch2.reasr_minicpm <wav> [model_key]
Prints one JSON line: {"text": ..., "language": ..., "model": ...}
"""
import json
import sys

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

HF_IDS = {
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
}


def main():
    wav = sys.argv[1]
    key = sys.argv[2] if len(sys.argv) > 2 else "qwen3-asr-0.6b"
    hf_id = HF_IDS[key]
    processor = AutoProcessor.from_pretrained(hf_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16, device_map="cuda").eval()
    inputs = processor.apply_transcription_request(audio=wav, language="English")
    inputs = inputs.to(model.device, model.dtype)
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256)
    gen = out[:, n_in:]
    parsed = processor.decode(gen, return_format="parsed")[0]
    print(json.dumps({
        "text": parsed.get("transcription", "").strip(),
        "language": parsed.get("language", "?"),
        "model": key,
    }))


if __name__ == "__main__":
    main()
