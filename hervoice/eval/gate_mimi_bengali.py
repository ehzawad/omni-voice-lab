#!/usr/bin/env python3
"""GATE 0 for any Moshi-style Bengali model: can Mimi reproduce Bengali speech at all?

Moshi generates audio as Mimi codec tokens. Whatever the model learns, its output can never
be better than what Mimi can reconstruct — exactly as an Orpheus-style model could never beat
its SNAC round-trip. That earlier gate decided a whole project before a single training step:
SNAC round-tripped the target speaker at 0.660 speaker similarity against 0.984 for the
mel→Vocos path, and the model that used SNAC duly topped out at 0.436 while the mel-based one
reached 0.611. Ten minutes of measurement predicted weeks of training.

So: round-trip REAL Bengali speech through Mimi and measure what survives.

  * intelligibility — re-ASR CER through the same Bengali FastConformer the bot uses, against
    the corpus transcript. The number that matters is the DELTA from the original clip's CER,
    because the ASR is not perfect on the originals either.
  * speaker identity — ECAPA cosine between the original and the reconstruction.
  * codebook depth — Mimi is residual: 1 codebook is the semantic one, 8 is full quality.
    Moshi uses 8. Measuring 1/2/4/8 shows how much of Bengali survives at each rate, which
    determines what a smaller model could get away with.

LOUDNESS IS A CONFOUND AND MUST BE CONTROLLED. Mimi reconstructs quiet audio much worse:
FLEURS en_us at its native level scored 0.304 speaker similarity at 8 codebooks and 0.671
after RMS normalisation to 0.04 — a 0.37 swing from gain alone. Before that was found, the
same comparison appeared to show Mimi favouring Bengali over English by +0.41, which is not
true: level-matched, FLEURS Bengali 0.713 vs English 0.671, a difference small enough to be
corpus variation. Mimi has no Bengali penalty; it has a loudness sensitivity. Any Mimi-based
pipeline should normalise input loudness, and any codec comparison must do so before drawing
a conclusion about language.

Reference points already measured on this box, same metrics, same ASR:
    mel → Vocos (what IndicF5 uses)   speaker similarity 0.984
    SNAC 24 kHz round-trip             speaker similarity 0.660  → capped that project

    CUDA_VISIBLE_DEVICES=1 .venv-bnweb/bin/python -m hervoice.eval.gate_mimi_bengali --n 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.svc import config as C                 # noqa: E402
from hervoice.eval.run_scenarios import _cer         # noqa: E402
from hervoice.eval.run_real_speech import load_clips  # noqa: E402

MIMI_SR = 24000


def asr(a16k):
    import urllib.request
    r = urllib.request.urlopen(urllib.request.Request(
        f"{C.ASR_URL}/transcribe", data=np.ascontiguousarray(a16k, dtype="<f4").tobytes(),
        headers={"Content-Type": "application/octet-stream"}), timeout=120)
    return json.load(r)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--codebooks", default="1,2,4,8,16,32")
    ap.add_argument("--target-rms", type=float, default=0.04,
                    help="normalise input loudness before encoding; 0 disables. Mimi is "
                         "strongly level-sensitive -- see the module docstring.")
    ap.add_argument("--out", default="runs/svc/gate_mimi")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    import torchaudio
    from transformers import MimiModel, AutoFeatureExtractor
    from speechbrain.inference.speaker import EncoderClassifier

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
    fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    n_q_max = mimi.config.num_quantizers
    frame_hz = mimi.config.frame_rate
    print(f"[mimi] on {gpu}: {n_q_max} codebooks max, {frame_hz} Hz frames, "
          f"{mimi.config.sampling_rate} Hz audio")
    cbs = [c for c in (int(x) for x in a.codebooks.split(",")) if c <= n_q_max]

    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         run_opts={"device": dev})

    def embed(w16):
        with torch.inference_mode():
            e = enc.encode_batch(torch.from_numpy(w16)[None].to(dev)).squeeze()
        return torch.nn.functional.normalize(e, dim=-1).cpu()

    clips, _ = load_clips(a.n)
    print(f"[data] {a.n} real Bengali clips, {sum(c['dur'] for c in clips)/60:.1f} min, "
          f"SNR {min(c['snr'] for c in clips):.0f}-{max(c['snr'] for c in clips):.0f} dB")

    rows, orig_cers = [], []
    recon = {c: [] for c in cbs}
    for k, c in enumerate(clips):
        a16 = c["audio"]                                   # 16 kHz, what the ASR wants
        if a.target_rms > 0:
            r = float(np.sqrt(np.mean(a16 ** 2)))
            if r > 1e-6:
                a16 = np.clip(a16 * (a.target_rms / r), -1.0, 1.0)
        a24 = torchaudio.functional.resample(torch.from_numpy(a16)[None], C.SR_IN, MIMI_SR)
        orig_cer = _cer(c["text"], asr(a16))
        orig_cers.append(orig_cer)
        e0 = embed(a16)
        inp = fe(raw_audio=a24[0].numpy(), sampling_rate=MIMI_SR, return_tensors="pt")
        for nq in cbs:
            with torch.inference_mode():
                codes = mimi.encode(inp["input_values"].to(dev), num_quantizers=nq).audio_codes
                out = mimi.decode(codes).audio_values[0, 0].float().cpu()
            w16 = torchaudio.functional.resample(out[None], MIMI_SR, C.SR_IN)[0].numpy()
            n = min(len(w16), len(a16))
            cer = _cer(c["text"], asr(w16))
            secs = float(torch.dot(embed(w16[:n]), e0))
            recon[nq].append(dict(cer=cer, secs=secs, d_cer=cer - orig_cer))
            if k == 0:
                sf.write(os.path.join(a.out, f"recon_nq{nq}.wav"), w16, C.SR_IN)
        if k == 0:
            sf.write(os.path.join(a.out, "original.wav"), a16, C.SR_IN)
        print(f"  [{k:2d}] orig_cer={orig_cer:.3f} " +
              " ".join(f"nq{nq}:{recon[nq][-1]['cer']:.3f}/{recon[nq][-1]['secs']:.3f}" for nq in cbs),
              flush=True)

    kbps = {nq: nq * frame_hz * 11 / 1000 for nq in cbs}    # 2**11 entries per codebook
    print(f"\n[GATE] Mimi round-trip of REAL Bengali speech, n={len(clips)}")
    print(f"  original clips, ASR CER through our Bengali model: mean {np.mean(orig_cers):.4f}")
    print(f"\n  {'codebooks':>9s} {'~kbps':>6s} {'CER':>8s} {'ΔCER vs orig':>13s} {'speaker sim':>12s}")
    for nq in cbs:
        r = recon[nq]
        row = dict(num_quantizers=nq, kbps=round(kbps[nq], 2),
                   cer_mean=round(float(np.mean([x["cer"] for x in r])), 4),
                   d_cer_mean=round(float(np.mean([x["d_cer"] for x in r])), 4),
                   secs_mean=round(float(np.mean([x["secs"] for x in r])), 4),
                   secs_min=round(float(np.min([x["secs"] for x in r])), 4))
        rows.append(row)
        print(f"  {nq:9d} {row['kbps']:6.1f} {row['cer_mean']:8.4f} {row['d_cer_mean']:+13.4f} "
              f"{row['secs_mean']:12.4f}")
    json.dump(dict(gpu=gpu, n=len(clips), orig_cer=float(np.mean(orig_cers)),
                   frame_hz=frame_hz, rows=rows),
              open(os.path.join(a.out, "gate_mimi.json"), "w"), indent=1)
    print("\n  Reference points measured on this box with the same metrics:")
    print("    mel -> Vocos (IndicF5's path)  speaker similarity 0.984")
    print("    SNAC 24 kHz round-trip          speaker similarity 0.660  -> capped that project at 0.436")
    print(f"  A Moshi-style Bengali model cannot exceed the row it is built on.")
    print(f"  wrote {a.out}/gate_mimi.json and recon_nq*.wav for listening")


if __name__ == "__main__":
    main()
