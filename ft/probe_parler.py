#!/usr/bin/env python3
"""Probe AI4Bharat Indic Parler-TTS: confirm load, generate a baseline Bengali clip,
and dump candidate LoRA target module names. Run in .venv-tts-lora on cuda:0."""
import os, torch, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

REPO = "ai4bharat/indic-parler-tts"
DEV = "cuda:0"
os.makedirs("examples/bench", exist_ok=True)

print("[load] model")
model = ParlerTTSForConditionalGeneration.from_pretrained(REPO).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(REPO)
desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
sr = model.config.sampling_rate
print(f"[ok] loaded. sampling_rate={sr}  text_encoder={model.config.text_encoder._name_or_path}")

# baseline Bengali generation
prompt = "আমি আজ খুব খুশি, কারণ আমরা একসাথে নতুন কিছু তৈরি করছি।"
desc = "A female speaker speaks clearly and naturally in a clean, close-sounding recording."
d = desc_tok(desc, return_tensors="pt").to(DEV)
p = tok(prompt, return_tensors="pt").to(DEV)
with torch.no_grad():
    gen = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                         prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
audio = gen.cpu().numpy().squeeze()
out = "examples/bench/probe_bn_base.wav"
sf.write(out, audio, sr)
import numpy as np
print(f"[gen] wrote {out}  dur={len(audio)/sr:.2f}s  rms={np.sqrt((audio**2).mean()):.4f}  peak={np.abs(audio).max():.3f}")

# candidate LoRA targets: linear layers in the decoder
lin = {}
for name, mod in model.named_modules():
    if mod.__class__.__name__ == "Linear" and "decoder" in name:
        leaf = name.split(".")[-1]
        lin[leaf] = lin.get(leaf, 0) + 1
print("[decoder Linear leaf names -> count]")
for k, v in sorted(lin.items(), key=lambda x: -x[1]):
    print(f"   {k}: {v}")
print("[params] total =", sum(p.numel() for p in model.parameters())/1e6, "M")
