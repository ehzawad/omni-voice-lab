#!/usr/bin/env python3
"""Fair re-score of the ASR bake-off from stored hypotheses (no models reloaded).

Adds number normalization so a model that writes "12" is not penalized against a
reference that says "twelve" (a formatting/ITN difference, not a recognition error).
Reports raw WER/CER (as run) alongside number-normalized WER/CER. Reads
results_asr_bench.json, writes results_asr_bench_normalized.json."""
import json, re, statistics as st
import jiwer

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
    return str(n)  # leave large numbers as-is

_base = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                       jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
def norm(s):        # raw normalization (matches the original bench)
    return _base(s)
def numnorm(s):     # additionally spell digit tokens as words, so 12 == twelve
    s = _base(s)
    return re.sub(r"\b\d+\b", lambda m: _int_to_words(m.group()), s)

def score(ref, hyp, fn):
    r, h = fn(ref), fn(hyp)
    if not r.strip(): return None, None
    return round(jiwer.wer(r, h), 4), round(jiwer.cer(r, h), 4)

d = json.load(open("results_asr_bench.json"))
out = {"benchmark": d.get("benchmark"), "note":
       "Re-scored from stored hyps. num_norm spells digits as words so ITN/formatting "
       "differences (12 vs twelve) are not counted as recognition errors. 4 prompts only.",
       "models": {}}
rows_out = []
for k, m in d["models"].items():
    raw_w, raw_c, nn_w, nn_c = [], [], [], []
    for r in m["rows"]:
        w0, c0 = score(r["ref"], r["hyp"], norm)
        w1, c1 = score(r["ref"], r["hyp"], numnorm)
        if w0 is not None: raw_w.append(w0); raw_c.append(c0)
        if w1 is not None: nn_w.append(w1); nn_c.append(c1)
    entry = {"raw_wer": round(st.mean(raw_w),4), "raw_cer": round(st.mean(raw_c),4),
             "numnorm_wer": round(st.mean(nn_w),4), "numnorm_cer": round(st.mean(nn_c),4),
             "features": m.get("note","")}
    out["models"][k] = entry
    rows_out.append((k, entry))

json.dump(out, open("results_asr_bench_normalized.json","w"), indent=2)
rows_out.sort(key=lambda x: x[1]["numnorm_wer"])
print(f"{'model':16s} {'raw_WER':>8s} {'numnorm_WER':>12s} {'numnorm_CER':>12s}")
for k, e in rows_out:
    print(f"{k:16s} {e['raw_wer']:>8.3f} {e['numnorm_wer']:>12.3f} {e['numnorm_cer']:>12.3f}")
print("\n[wrote] results_asr_bench_normalized.json")
