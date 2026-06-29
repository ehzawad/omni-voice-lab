#!/usr/bin/env python3
"""Baseline: generate Bengali speech from held-out FLEURS texts with the un-finetuned
orpheus-bangla-tts, decode via SNAC, save WAVs. Establishes the 'before'.
Run in .venv-tts-lora on cuda:0."""
import os, sys, time, numpy as np, torch, soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import bn_tts as B

REPO = "asif00/orpheus-bangla-tts"
DEV = "cuda:0"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TAG = sys.argv[2] if len(sys.argv) > 2 else "base"
ADAPTER = sys.argv[3] if len(sys.argv) > 3 else None
OUT = "examples/bench"; os.makedirs(OUT, exist_ok=True)

print("[load] orpheus-bangla" + (f" + adapter {ADAPTER}" if ADAPTER else ""))
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(DEV).eval()
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER).to(DEV).eval()
print("[load] SNAC 24khz")
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEV).eval()

rows = B.load_fleurs_split("test", n=N)
print(f"[data] {len(rows)} held-out FLEURS bn texts")

manifest = []
for i, r in enumerate(rows):
    text = r["text"].strip()
    inp = B.build_gen_input(tok, text).to(DEV)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=1200, do_sample=True, temperature=0.6,
                             top_p=0.95, repetition_penalty=1.1, eos_token_id=B.END_SPEECH,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][inp.shape[1]:]
    codes = B.tokens_to_codes(gen)
    audio = B.codes_to_audio(snac, codes, DEV)
    dt = time.time() - t0
    path = os.path.join(OUT, f"bn_{TAG}_{i:02d}.wav")
    sf.write(path, audio, B.SNAC_SR)
    rms = float(np.sqrt((audio**2).mean())) if audio.size else 0.0
    dur = len(audio) / B.SNAC_SR
    print(f"[{i}] codes={len(codes)} dur={dur:.2f}s rms={rms:.4f} gen={dt:.1f}s | {text[:45]}")
    manifest.append(dict(i=i, text=text, wav=path, dur=round(dur, 2), rms=round(rms, 4),
                         n_codes=len(codes), gen_s=round(dt, 1)))

import json
mpath = os.path.join(OUT, f"bn_{TAG}_manifest.json")
json.dump(manifest, open(mpath, "w"), ensure_ascii=False, indent=2)
print(f"[wrote] {mpath}  | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
