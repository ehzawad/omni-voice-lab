#!/usr/bin/env python3
"""Prepare Orpheus training sequences from FLEURS bn: SNAC-encode each clip and build
[SOH]+text+[EOT,EOH]+[SOS]+audio_tokens+[EOS] sequences with a prompt mask length.
Saves to data/tts_bn/orpheus_fleurs_<n>.pt. Run in .venv-tts-lora on cuda:0.
Usage: prep_orpheus_data.py <split> <n>"""
import os, sys, unicodedata, re, torch
from transformers import AutoTokenizer
from snac import SNAC
import bn_tts as B

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
DEV = "cuda:0"
os.makedirs("data/tts_bn", exist_ok=True)

_lat = re.compile(r"[a-zA-Z]")
def ok_text(t):
    t = t.strip()
    if len(t) < 3: return False
    lat = len(_lat.findall(t))
    return lat / max(len(t), 1) <= 0.15   # drop heavy code-switch

tok = AutoTokenizer.from_pretrained("asif00/orpheus-bangla-tts")
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEV).eval()

rows = B.load_fleurs_split(SPLIT, n=None)
seqs, kept, durs = [], 0, 0.0
for r in rows:
    if kept >= N: break
    text = unicodedata.normalize("NFC", r["text"])
    if not ok_text(text) or r["audio"] is None: continue
    dur = len(r["audio"]) / (r["sr"] or 16000)
    if dur < 2.0 or dur > 12.0: continue
    try:
        atoks = B.audio_to_tokens(snac, r["audio"], r["sr"], DEV)
    except Exception:
        continue
    if len(atoks) < 7: continue
    text_ids = tok(text, add_special_tokens=False).input_ids
    prompt = [B.START_HUMAN] + text_ids + [B.EOT, B.END_HUMAN]
    full = prompt + [B.START_SPEECH] + atoks + [B.END_SPEECH]
    seqs.append({"input_ids": full, "prompt_len": len(prompt)})
    kept += 1; durs += dur
    if kept % 50 == 0: print(f"  prepared {kept}/{N} ({durs/60:.1f} min audio)")

out = f"data/tts_bn/orpheus_fleurs_{SPLIT}_{kept}.pt"
torch.save(seqs, out)
print(f"[wrote] {out}  clips={kept}  audio={durs/60:.1f}min  "
      f"avg_seq_len={sum(len(s['input_ids']) for s in seqs)//max(kept,1)}")
