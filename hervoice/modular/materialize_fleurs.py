#!/usr/bin/env python3
"""Materialize ~100 English clips (16kHz mono wav + refs) from FLEURS en_us test split.

Writes to data/asr_eval/fleurs_en/ with manifest.jsonl (id, wav, ref, audio_s).
Audio decoding is done via soundfile on the raw bytes (datasets audio-decode needs
torchcodec which is not installed in these venvs). FLEURS audio is already 16kHz mono.
"""
import io, json, os, sys
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "asr_eval", "fleurs_en")
os.makedirs(OUT, exist_ok=True)

print(f"[load] google/fleurs en_us test (streaming), target {N} clips")
ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

manifest = []
for ex in ds:
    if len(manifest) >= N:
        break
    ref = (ex.get("transcription") or "").strip()
    if not ref:
        continue
    b = ex["audio"].get("bytes")
    if not b:
        continue
    data, sr = sf.read(io.BytesIO(b))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    cid = str(ex.get("id", len(manifest)))
    fname = f"{len(manifest):04d}_{cid}.wav"
    wav_path = os.path.join(OUT, fname)
    sf.write(wav_path, data.astype(np.float32), sr, subtype="PCM_16")
    manifest.append({"id": cid, "wav": fname, "ref": ref,
                     "audio_s": round(len(data) / sr, 3)})

mpath = os.path.join(OUT, "manifest.jsonl")
with open(mpath, "w") as f:
    for m in manifest:
        f.write(json.dumps(m) + "\n")

tot = sum(m["audio_s"] for m in manifest)
print(f"[done] {len(manifest)} clips, {tot:.1f}s total audio -> {OUT}")
print(f"[manifest] {mpath}")
