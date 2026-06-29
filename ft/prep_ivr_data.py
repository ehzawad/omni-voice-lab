#!/usr/bin/env python3
"""Prepare Orpheus training sequences from IndicVoices-R Bengali (gated, clean TTS corpus).
Downloads train shards, filters on quality metadata, SNAC-encodes into Orpheus sequences.
Also writes a held-out in-domain eval text list from the test split.
Run in .venv-tts-lora on cuda:0. Usage: prep_ivr_data.py <target_clips> <max_train_shards>"""
import os, sys, io, gc, unicodedata, re, torch, soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi
from transformers import AutoTokenizer
from snac import SNAC
import bn_tts as B

# Memory-frugal: the box is RAM-contended, so stream parquet in small batches via pyarrow
# and never load a whole shard (with audio bytes) into memory at once.
COLS = ["normalized", "text", "audio", "duration", "cer", "snr", "speaker_id"]

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MAX_SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
DEV = "cuda:0"
REPO = "ai4bharat/indicvoices_r"
os.makedirs("data/tts_bn", exist_ok=True)

_lat = re.compile(r"[a-zA-Z]")
def ok_text(t):
    t = (t or "").strip()
    if len(t) < 5: return False
    return len(_lat.findall(t)) / max(len(t), 1) <= 0.15

def cerval(x):
    # IndicVoices-R stores cer as a string like 'tensor(0.0194)'
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
    return float(m.group()) if m else 1.0

api = HfApi()
files = api.list_repo_files(REPO, repo_type="dataset")
train_shards = sorted(f for f in files if f.startswith("Bengali/train-") and f.endswith(".parquet"))
test_shards = sorted(f for f in files if f.startswith("Bengali/test-") and f.endswith(".parquet"))
print(f"[ivr] {len(train_shards)} train shards, {len(test_shards)} test shards")

tok = AutoTokenizer.from_pretrained("asif00/orpheus-bangla-tts")
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEV).eval()

# ---- held-out in-domain eval texts from test split (metadata cols only, no audio) ----
import json
tp = hf_hub_download(REPO, test_shards[0], repo_type="dataset")
ev = []
for batch in pq.ParquetFile(tp).iter_batches(batch_size=64, columns=["normalized", "text", "duration"]):
    for r in batch.to_pylist():
        t = unicodedata.normalize("NFC", str(r.get("normalized") or r.get("text") or ""))
        if ok_text(t) and 2.0 <= float(r["duration"]) <= 12.0:
            ev.append(t)
    if len(ev) >= 20: break
ev = ev[:20]
json.dump(ev, open("data/tts_bn/ivr_test_texts.json", "w"), ensure_ascii=False, indent=2)
print(f"[ivr] wrote {len(ev)} held-out eval texts", flush=True)

# ---- training subset: stream each shard in small batches, filter + speaker-cap + SNAC ----
seqs, durs, per_spk = [], 0.0, {}
for shard in train_shards[:MAX_SHARDS]:
    if len(seqs) >= TARGET: break
    path = hf_hub_download(REPO, shard, repo_type="dataset")
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=16, columns=COLS):
        if len(seqs) >= TARGET: break
        for r in batch.to_pylist():
            if len(seqs) >= TARGET: break
            dur = float(r["duration"])
            if not (2.0 <= dur <= 12.0): continue
            if cerval(r.get("cer", 1.0)) > 0.05: continue
            if float(r.get("snr", 0)) < 20: continue
            spk = str(r.get("speaker_id", "?"))
            if per_spk.get(spk, 0) >= 12: continue
            text = unicodedata.normalize("NFC", str(r.get("normalized") or r.get("text") or ""))
            if not ok_text(text): continue
            a = r["audio"]
            if not (isinstance(a, dict) and a.get("bytes")): continue
            try:
                audio, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                atoks = B.audio_to_tokens(snac, audio, sr, DEV)
            except Exception: continue
            if len(atoks) < 7: continue
            text_ids = tok(text, add_special_tokens=False).input_ids
            prompt = [B.START_HUMAN] + text_ids + [B.EOT, B.END_HUMAN]
            seqs.append({"input_ids": prompt + [B.START_SPEECH] + atoks + [B.END_SPEECH],
                         "prompt_len": len(prompt)})
            per_spk[spk] = per_spk.get(spk, 0) + 1
            durs += dur
            if len(seqs) % 200 == 0:
                print(f"  kept {len(seqs)}/{TARGET} ({durs/60:.0f}min, {len(per_spk)} spk)", flush=True)
        del batch
    gc.collect()

out = f"data/tts_bn/orpheus_ivr_train_{len(seqs)}.pt"
torch.save(seqs, out)
print(f"[wrote] {out}  clips={len(seqs)}  audio={durs/60:.0f}min  speakers={len(per_spk)}  "
      f"avg_seq_len={sum(len(s['input_ids']) for s in seqs)//max(len(seqs),1)}")
