#!/usr/bin/env python3
"""NFE sweep: what fewer flow-matching steps cost in QUALITY, not just what they save in time.

TTS is ~1.6 s of a ~3.1 s response, so the solver step count is the single biggest latency
lever in the system. 32 -> 16 was taken on latency evidence alone (3074 -> 1553 ms) and the
quality cost was never measured. This measures it, and tests whether 8 or 12 is also safe.

Quality is judged three ways, because none alone is sufficient:
  * intelligibility  -- re-ASR CER through the SAME Bengali FastConformer the bot uses. A
                        synthesiser that drops or slurs words shows up here.
  * voice drift      -- ECAPA speaker similarity against the NFE 32 rendering of the SAME
                        sentence with the SAME seed. Catches timbre changing as steps drop,
                        which CER cannot see.
  * duration drift   -- audio seconds vs the NFE 32 rendering. Catches the solver truncating
                        or stretching, which neither of the above reliably catches.

NFE 32 is the reference, not ground truth: it is the released model's default, so "no drift
from 32" means "indistinguishable from the setting the model was tuned for".

    CUDA_VISIBLE_DEVICES=1 .venv-bnweb/bin/python -m hervoice.eval.bench_nfe
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.svc import config as C                    # noqa: E402
from hervoice.eval.run_scenarios import _cer            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Sentences the ASSISTANT would actually say: replies from the scripted runs plus a few
# government-domain sentences. Short and long, with conjuncts and numerals in words.
SENTENCES = [
    "বাংলাদেশের রাজধানীর নাম ঢাকা।",
    "এটি দেশের প্রধান প্রশাসনিক ও অর্থনৈতিক কেন্দ্র।",
    "বাংলাদেশের জাতীয় ফুল হলো শাপলা।",
    "নতুন জাতীয় পরিচয়পত্র করতে জন্ম নিবন্ধন সনদ ও ছবি লাগবে।",
    "জমির নামজারি করতে হলে স্থানীয় ভূমি অফিসে যোগাযোগ করুন।",
    "ই-পাসপোর্টের মেয়াদ সাধারণত পাঁচ বছর বা দশ বছর হয়ে থাকে।",
    "চট্টগ্রামের আবহাওয়া সাধারণত উষ্ণ এবং আর্দ্র হয়।",
    "আপনি অনলাইনে আবেদন করতে পারেন অথবা নিকটস্থ অফিসে যোগাযোগ করতে পারেন।",
]


def asr_text(wav16k_path_or_array, sr=16000):
    import urllib.request
    a = wav16k_path_or_array
    pcm = np.ascontiguousarray(a, dtype="<f4").tobytes()
    r = urllib.request.urlopen(urllib.request.Request(
        f"{C.ASR_URL}/transcribe", data=pcm,
        headers={"Content-Type": "application/octet-stream"}), timeout=120)
    return json.load(r)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfe", default="8,12,16,24,32")
    ap.add_argument("--repeats", type=int, default=2, help="timed repeats per sentence")
    ap.add_argument("--out", default="runs/svc/nfe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nfes = [int(x) for x in a.nfe.split(",")]
    assert 32 in nfes, "32 is the reference"

    import torch
    import torchaudio
    from hervoice.bn.models import BnTts
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    print(f"[gpu] {gpu}")

    tts = BnTts(repo=C.TTS_REPO, device=dev, nfe=16)
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         run_opts={"device": dev})

    def embed(w24):
        w16 = torchaudio.functional.resample(torch.from_numpy(w24)[None], 24000, 16000)
        with torch.inference_mode():
            e = enc.encode_batch(w16.to(dev)).squeeze()
        return torch.nn.functional.normalize(e, dim=-1).cpu()

    # warm-up: first call carries CUDA/compile cost that would pollute NFE 8
    tts.synth_chunk(SENTENCES[0], seed=1, nfe=16)
    torch.cuda.synchronize() if dev == "cuda" else None

    audio, timing = {}, {}
    for nfe in sorted(nfes):
        waves, ms = [], []
        for si, s in enumerate(SENTENCES):
            for r in range(a.repeats):
                t = time.time()
                w = tts.synth_chunk(s, seed=4000 + si, nfe=nfe)   # SAME seed across NFE
                if dev == "cuda":
                    torch.cuda.synchronize()
                dt = (time.time() - t) * 1000
                if r == a.repeats - 1:
                    waves.append(w)
                ms.append(dt)
        audio[nfe] = waves
        timing[nfe] = ms
        print(f"  nfe {nfe:2d}: median {np.median(ms):7.0f} ms/sentence", flush=True)

    ref = audio[32]
    ref_emb = [embed(w) for w in ref]
    rows = []
    print(f"\n  {'nfe':>3s} {'ms/sent':>8s} {'RTF':>5s} {'re-ASR CER':>11s} {'SECS vs nfe32':>14s} {'dur drift':>10s}")
    for nfe in sorted(nfes):
        ws = audio[nfe]
        cers, secs, durs = [], [], []
        for i, (w, s) in enumerate(zip(ws, SENTENCES)):
            w16 = torchaudio.functional.resample(torch.from_numpy(w)[None], 24000, 16000)[0].numpy()
            cers.append(_cer(s, asr_text(w16)))
            secs.append(float(torch.dot(embed(w), ref_emb[i])))
            durs.append(len(w) / max(len(ref[i]), 1))
        aud_s = sum(len(w) for w in ws) / 24000
        rtf = (sum(timing[nfe]) / a.repeats / 1000) / aud_s
        rows.append(dict(nfe=nfe, ms_median=round(float(np.median(timing[nfe])), 1),
                         rtf=round(rtf, 3), cer_mean=round(float(np.mean(cers)), 4),
                         cer_median=round(float(np.median(cers)), 4),
                         secs_vs32=round(float(np.mean(secs)), 4),
                         dur_ratio=round(float(np.mean(durs)), 3),
                         gpu=gpu, per_sentence_cer=[round(c, 3) for c in cers]))
        r = rows[-1]
        print(f"  {nfe:3d} {r['ms_median']:8.0f} {r['rtf']:5.2f} {r['cer_mean']:11.4f} "
              f"{r['secs_vs32']:14.4f} {r['dur_ratio']:10.3f}")
        for i, w in enumerate(ws):
            sf.write(os.path.join(a.out, f"nfe{nfe:02d}_s{i}.wav"), w, 24000)

    json.dump(dict(gpu=gpu, sentences=SENTENCES, rows=rows),
              open(os.path.join(a.out, "nfe.json"), "w"), ensure_ascii=False, indent=1)
    base = [r for r in rows if r["nfe"] == 32][0]
    print(f"\n  reference nfe 32: CER {base['cer_mean']:.4f}")
    print("  SECS vs nfe32 is voice drift: 1.000 = indistinguishable timbre from the default setting.")
    print("  A CER at or below the nfe-32 reference means no intelligibility was lost.")
    print(f"  wrote {a.out}/nfe.json and {len(SENTENCES)} wavs per setting for listening")


if __name__ == "__main__":
    main()
