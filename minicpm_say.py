#!/usr/bin/env python3
"""Make MiniCPM-o 4.5 SPEAK a fixed sentence (controlled voice probe).

Default voice, or clone a reference voice with --ref ref.wav.
TTS-on-text requires an audio_assistant SYSTEM message (a plain user prompt won't emit audio).
"""
import argparse, time, librosa, numpy as np, torch
from transformers import AutoModel, AutoTokenizer
MODEL_ID = "openbmb/MiniCPM-o-4_5"

VC_SUFFIX = ("Please assist users while maintaining this voice style. Please answer the user's "
             "questions seriously and in a high quality. Please chat with the user in a highly "
             "human-like and oral style. You are a helpful assistant developed by ModelBest: MiniCPM-Omni.")

ap = argparse.ArgumentParser()
ap.add_argument("--text", required=True)
ap.add_argument("--out", default="minicpm_say.wav")
ap.add_argument("--ref", default=None, help="optional reference wav to clone the voice from")
args = ap.parse_args()

model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model.init_tts()

if args.ref:
    ref_audio, _ = librosa.load(args.ref, sr=16000, mono=True)
    sys_msg = {"role": "system", "content": ["Clone the voice in the provided audio prompt.", ref_audio, VC_SUFFIX]}
    print(f"[voice] cloning from {args.ref}")
else:
    sys_msg = {"role": "system", "content": ["Use the <reserved_53> voice.", VC_SUFFIX]}
    print("[voice] default")

instruction = ("Repeat the following sentence exactly, word for word, with a warm friendly tone. "
               "Do not add anything else. Sentence: " + args.text)
msgs = [sys_msg, {"role": "user", "content": [instruction]}]

t0 = time.time()
res = model.chat(msgs=msgs, tokenizer=tok, use_tts_template=True, generate_audio=True,
                 output_audio_path=args.out, max_new_tokens=128)
print("[said-text]", res if isinstance(res, str) else getattr(res, "text", str(res)))
print(f"[wrote] {args.out}  [gen] {time.time()-t0:.1f}s")
peak = torch.cuda.max_memory_allocated() / 1e9
reserved = torch.cuda.max_memory_reserved() / 1e9
print(f"[VRAM] peak allocated {peak:.1f} GB | peak reserved {reserved:.1f} GB")
