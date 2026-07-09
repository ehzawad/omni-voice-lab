#!/usr/bin/env python3
"""Larger, statistically defensible English ASR benchmark over FLEURS en_us.

Runs ONE model (by key) over all clips in data/asr_eval/fleurs_en/manifest.jsonl,
scores WER/CER with a single shared normalization (lowercase, strip punctuation,
collapse whitespace, spell digits as words so "12"=="twelve"), and reports:
  - macro-WER (mean over clips) with bootstrap 95% CI
  - micro/overall-WER (total edits / total ref words)
  - macro-CER
  - latency: steady-state median + p90 (warmup call discarded), RTF
  - VRAM (nvidia-smi delta across load)
  - n_failed clips (errors are counted, never silently dropped)

Because the two backend families live in separate venvs, run this once per venv,
each writing/merging into results_asr_bench_large.json.

Usage:
  <venv>/bin/python hervoice/modular/asr_bench_large.py <model_key> [more_keys...]
"""
import json, os, re, sys, time, subprocess, random
import statistics as st
import jiwer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "hervoice", "modular"))
import asr as ASR

EVAL_DIR = os.path.join(ROOT, "data", "asr_eval", "fleurs_en")
MANIFEST = os.path.join(EVAL_DIR, "manifest.jsonl")
RESULTS = os.path.join(ROOT, "results_asr_bench_large.json")

# ---- shared normalization (matches rescore_asr.numnorm) ----
_UNITS = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
          "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
          "eighteen","nineteen"]
_TENS = {20:"twenty",30:"thirty",40:"forty",50:"fifty",60:"sixty",70:"seventy",80:"eighty",90:"ninety"}
def _int_to_words(n):
    n = int(n)
    if n < 20: return _UNITS[n]
    if n < 100:
        t = (n//10)*10; r = n%10
        return _TENS[t] + ("" if r==0 else " "+_UNITS[r])
    if n < 1000:
        h = n//100; r = n%100
        return _UNITS[h]+" hundred" + ("" if r==0 else " "+_int_to_words(r))
    return str(n)
_base = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                       jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
def numnorm(s):
    s = _base(s)
    return re.sub(r"\b\d+\b", lambda m: _int_to_words(m.group()), s)

def gpu_mem_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID"))
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return float("nan")

def bootstrap_ci(per_clip_edits, per_clip_refwords, n_boot=1000, seed=13):
    """Bootstrap 95% CI for micro-WER (resample utterances)."""
    rng = random.Random(seed)
    n = len(per_clip_edits)
    idx = list(range(n))
    wers = []
    for _ in range(n_boot):
        samp = [rng.choice(idx) for _ in range(n)]
        e = sum(per_clip_edits[i] for i in samp)
        w = sum(per_clip_refwords[i] for i in samp)
        if w > 0:
            wers.append(e / w)
    wers.sort()
    lo = wers[int(0.025 * len(wers))]
    hi = wers[int(0.975 * len(wers))]
    return round(lo, 4), round(hi, 4)

def load_manifest():
    rows = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def run_model(key, clips):
    print(f"\n===== {key} =====")
    meta = ASR.ASR_MODELS[key]
    mem_before = gpu_mem_used_mb()
    ASR.load(key)
    time.sleep(0.5)
    mem_after = gpu_mem_used_mb()
    vram_gb = round(max(0.0, mem_after - mem_before) / 1024.0, 2)

    # warmup (discarded)
    warm_wav = os.path.join(EVAL_DIR, clips[0]["wav"])
    try:
        ASR.transcribe(key, warm_wav, language="en")
    except Exception as e:
        print(f"  [warmup error, ignored] {e}")

    per_clip_edits, per_clip_refwords = [], []
    macro_wers, macro_cers, lats, rtfs = [], [], [], []
    tot_edits = tot_refwords = 0
    n_failed = 0
    rows_out = []
    feats_seen = set()

    for i, c in enumerate(clips):
        wav = os.path.join(EVAL_DIR, c["wav"])
        ref = c["ref"]
        try:
            r = ASR.transcribe(key, wav, language="en")
            hyp = r["text"]
            lat = r["latency_s"]
            if r.get("features"):
                feats_seen.add(r["features"].split(",")[0] if "tags=" not in r["features"] else "tags")
            rn, hn = numnorm(ref), numnorm(hyp)
            # edit ops for micro-WER
            m = jiwer.process_words(rn, hn)
            edits = m.substitutions + m.deletions + m.insertions
            refwords = m.hits + m.substitutions + m.deletions
            wer = edits / refwords if refwords else 0.0
            cer = jiwer.cer(rn, hn) if rn.strip() else 0.0
            per_clip_edits.append(edits); per_clip_refwords.append(refwords)
            tot_edits += edits; tot_refwords += refwords
            macro_wers.append(wer); macro_cers.append(cer)
            lats.append(lat)
            if c.get("audio_s"): rtfs.append(lat / c["audio_s"])
            rows_out.append({"id": c["id"], "ref": ref, "hyp": hyp,
                             "wer": round(wer,4), "cer": round(cer,4),
                             "latency_s": lat, "audio_s": c.get("audio_s")})
        except Exception as e:
            n_failed += 1
            print(f"  [FAIL {c['id']}] {type(e).__name__}: {e}")
            rows_out.append({"id": c["id"], "ref": ref, "hyp": None,
                             "error": f"{type(e).__name__}: {e}"})
        if (i+1) % 25 == 0:
            print(f"  ...{i+1}/{len(clips)} done")

    macro_wer = round(st.mean(macro_wers), 4) if macro_wers else None
    macro_cer = round(st.mean(macro_cers), 4) if macro_cers else None
    micro_wer = round(tot_edits / tot_refwords, 4) if tot_refwords else None
    ci = bootstrap_ci(per_clip_edits, per_clip_refwords) if per_clip_edits else (None, None)
    # steady-state latency: warmup already discarded above; lats are all post-warmup
    median_lat = round(st.median(lats), 3) if lats else None
    p90_lat = round(sorted(lats)[int(0.9*len(lats))-1], 3) if len(lats) >= 2 else (lats[0] if lats else None)
    rtf = round(st.median(rtfs), 3) if rtfs else None

    entry = {
        "model_key": key, "hf_id": meta["hf_id"], "note": meta["note"],
        "n_clips": len(clips), "n_scored": len(macro_wers), "n_failed": n_failed,
        "macro_wer": macro_wer, "micro_wer": micro_wer,
        "wer_ci95": [ci[0], ci[1]], "cer": macro_cer,
        "median_lat": median_lat, "p90_lat": p90_lat, "rtf": rtf,
        "vram_gb": vram_gb, "features": sorted(feats_seen),
        "rows": rows_out,
    }
    print(f"  n={len(clips)} scored={len(macro_wers)} failed={n_failed}")
    print(f"  macro_WER={macro_wer}  micro_WER={micro_wer}  CI95={ci}  CER={macro_cer}")
    print(f"  median_lat={median_lat}s  p90={p90_lat}s  RTF={rtf}  VRAM={vram_gb}GB")
    return entry

def main():
    keys = sys.argv[1:]
    if not keys:
        print("usage: asr_bench_large.py <model_key> [more...]"); sys.exit(1)
    clips = load_manifest()
    print(f"[eval] {len(clips)} clips from {MANIFEST}")
    if os.path.exists(RESULTS):
        db = json.load(open(RESULTS))
    else:
        db = {"benchmark": "English ASR (FLEURS en_us test, materialized)",
              "eval_dir": EVAL_DIR, "n_clips": len(clips),
              "normalization": "lowercase+strip_punct+collapse_ws+digits_to_words",
              "models": {}}
    for k in keys:
        entry = run_model(k, clips)
        db["models"][k] = entry
        db["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(db, open(RESULTS, "w"), indent=2)
        print(f"[wrote] {RESULTS}")

if __name__ == "__main__":
    main()
