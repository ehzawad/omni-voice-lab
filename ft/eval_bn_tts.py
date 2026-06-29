#!/usr/bin/env python3
"""Evaluate generated Bengali TTS by re-ASR: transcribe each generated WAV with
faster-whisper large-v3 (Bengali), compute CER/WER vs the prompt text.
Run in .venv-tts-lora. Usage: eval_bn_tts.py <tag>  (reads examples/bench/bn_<tag>_manifest.json)"""
import os, sys, json, unicodedata, re, numpy as np
import jiwer
from faster_whisper import WhisperModel

TAG = sys.argv[1] if len(sys.argv) > 1 else "base"
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"   # cpu keeps the GPU free for training
MAN = f"examples/bench/bn_{TAG}_manifest.json"

_punct = re.compile(r"[।,.?!৷;:\"'`()\[\]{}—\-–…]+")
def norm_bn(s):
    s = unicodedata.normalize("NFC", s or "")
    s = _punct.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def cer(ref, hyp):
    ref, hyp = norm_bn(ref), norm_bn(hyp)
    if not ref: return None
    return round(jiwer.cer(ref, hyp), 4)
def wer(ref, hyp):
    ref, hyp = norm_bn(ref), norm_bn(hyp)
    if not ref: return None
    return round(jiwer.wer(ref, hyp), 4)

print(f"[load] faster-whisper large-v3 ({DEVICE})")
ct = "int8" if DEVICE == "cpu" else "float16"
asr = WhisperModel("large-v3", device=DEVICE, compute_type=ct)

man = json.load(open(MAN))
rows = []
for m in man:
    wavp = m["wav"]
    if not os.path.exists(wavp) or m.get("dur", 0) < 0.2:
        rows.append({**m, "asr": "", "cer": None, "wer": None, "valid": False}); continue
    segs, _ = asr.transcribe(wavp, language="bn", beam_size=5)
    hyp = " ".join(s.text for s in segs).strip()
    rows.append({**m, "asr": hyp, "cer": cer(m["text"], hyp), "wer": wer(m["text"], hyp),
                 "valid": bool(hyp.strip())})
    print(f"[{m['i']}] CER={rows[-1]['cer']} WER={rows[-1]['wer']} | hyp={hyp[:45]}")

cers = [r["cer"] for r in rows if r["cer"] is not None]
wers = [r["wer"] for r in rows if r["wer"] is not None]
summary = dict(tag=TAG, n=len(rows), valid_rate=round(np.mean([r["valid"] for r in rows]), 3),
               cer_mean=round(float(np.mean(cers)), 4) if cers else None,
               wer_mean=round(float(np.mean(wers)), 4) if wers else None, rows=rows)
out = f"results_bn_tts_{TAG}.json"
json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
print(f"\n[{TAG}] valid={summary['valid_rate']} CER_mean={summary['cer_mean']} "
      f"WER_mean={summary['wer_mean']}  -> {out}")
