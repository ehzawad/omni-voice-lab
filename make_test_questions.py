#!/usr/bin/env python3
"""Synthesize clean English question clips (16k mono wav) to probe the Omni model."""
import sys, numpy as np, soundfile as sf, torch
from transformers import VitsModel, AutoTokenizer

QUESTIONS = [
    ("q1_capital", "What is the capital of France?"),
    ("q2_math",    "What is twelve multiplied by eight?"),
    ("q3_advice",  "I feel nervous before a job interview. Can you give me one quick tip?"),
]

tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
sr = model.config.sampling_rate
for name, text in QUESTIONS:
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        wav = model(**inp).waveform[0].cpu().numpy()
    # normalize to ~ -3 dBFS
    wav = wav / (np.abs(wav).max() + 1e-8) * 0.7
    out = f"/mnt/sdb/arafat/hervoice/{name}.wav"
    sf.write(out, wav, sr)
    print(f"{out}  ({len(wav)/sr:.1f}s)  text={text!r}")
print("DONE")
