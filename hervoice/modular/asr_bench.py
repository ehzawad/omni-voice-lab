#!/usr/bin/env python3
"""English ASR bake-off harness.

Runs one or more of the 5 ASR backends over the 4 English eval prompts in bench_common.py,
computing WER + CER (jiwer, normalized: lowercase + strip punctuation), transcription latency,
RTF, load VRAM, and a features column. Because the two families live in two venvs, run this
once per venv and merge:

  .venv-qwen-asr/bin/python -m hervoice.modular.asr_bench --keys qwen3-asr-1.7b,qwen3-asr-0.6b
  .venv-funasr/bin/python   -m hervoice.modular.asr_bench --keys sensevoice,paraformer-zh,funasr-nano

Each run appends its rows into results_asr_bench.json (merge-by-model_key). Models are loaded
one at a time and freed before the next, so VRAM stays within the 24GB card. GPU0 only.
"""
import argparse
import gc
import json
import os
import re
import string
import subprocess
import time

import jiwer
import librosa
import soundfile as sf

import bench_common as bc
from hervoice.modular import asr as asr_mod

RESULTS = os.path.join(bc.ROOT, "results_asr_bench.json")
RAW_DIR = os.path.join(bc.ROOT, "runs", "modular")

_PUNCT = str.maketrans("", "", string.punctuation)


def _normalize(s):
    """lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()


def wer_cer(ref, hyp):
    ref_n, hyp_n = _normalize(ref), _normalize(hyp)
    if not hyp_n:
        return 1.0, 1.0
    return jiwer.wer(ref_n, hyp_n), jiwer.cer(ref_n, hyp_n)


def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def audio_seconds(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return len(y) / sr


def bench_one(key):
    print(f"\n=== [{key}] {asr_mod.ASR_MODELS[key]['note']} ===")
    base_mib = gpu_used_mib()
    asr_mod.load(key)  # load once, measure VRAM
    load_mib = gpu_used_mib()
    load_vram_gb = round((load_mib - base_mib) / 1024, 2) if (load_mib and base_mib) else None
    rows = []
    for pid, wav, ref in bc.PROMPTS:
        r = asr_mod.transcribe(key, wav, language="en")
        w, c = wer_cer(ref, r["text"])
        dur = audio_seconds(wav)
        rtf = round(r["latency_s"] / max(dur, 1e-6), 3)
        row = dict(id=pid, ref=ref, hyp=r["text"], raw=r["raw"], features=r["features"],
                   wer=round(w, 4), cer=round(c, 4), latency_s=r["latency_s"],
                   audio_s=round(dur, 3), rtf=rtf)
        print(f"  [{pid}] wer={row['wer']:.3f} cer={row['cer']:.3f} lat={row['latency_s']}s "
              f"rtf={rtf} | '{r['text'][:60]}'" + (f" | {r['features']}" if r['features'] else ""))
        rows.append(row)
    n = len(rows)
    summary = dict(
        model_key=key, hf_id=asr_mod.ASR_MODELS[key]["hf_id"], note=asr_mod.ASR_MODELS[key]["note"],
        avg_wer=round(sum(x["wer"] for x in rows) / n, 4),
        avg_cer=round(sum(x["cer"] for x in rows) / n, 4),
        avg_latency_s=round(sum(x["latency_s"] for x in rows) / n, 3),
        avg_rtf=round(sum(x["rtf"] for x in rows) / n, 3),
        load_vram_gb=load_vram_gb, rows=rows)
    print(f"  -> avg_wer={summary['avg_wer']} avg_cer={summary['avg_cer']} "
          f"avg_lat={summary['avg_latency_s']}s load_vram={load_vram_gb}GB")
    # free
    asr_mod._CACHE.pop(key, None)
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return summary


def merge_save(new_summaries):
    data = {"benchmark": "English ASR bake-off (4 prompts)", "models": {}}
    if os.path.exists(RESULTS):
        try:
            data = json.load(open(RESULTS))
            data.setdefault("models", {})
        except Exception:
            pass
    for s in new_summaries:
        data["models"][s["model_key"]] = s
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(data, open(RESULTS, "w"), indent=2)
    print(f"\n[wrote] {RESULTS} (models: {list(data['models'])})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True, help="comma-separated ASR keys")
    args = ap.parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    summaries = []
    for k in keys:
        try:
            summaries.append(bench_one(k))
        except Exception as e:
            import traceback
            traceback.print_exc()
            summaries.append(dict(model_key=k, hf_id=asr_mod.ASR_MODELS.get(k, {}).get("hf_id"),
                                  error=str(e), rows=[]))
    merge_save(summaries)
