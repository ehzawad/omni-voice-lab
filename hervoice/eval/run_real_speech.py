#!/usr/bin/env python3
"""Robustness on REAL human speech: IndicVoices-R Bengali, extempore (spontaneous) clips.

The scripted scenarios use clean synthetic user audio, which tests memory but flatters ASR.
This run feeds real recordings from real speakers -- rural and urban, different ages, a range
of SNRs, spontaneous rather than read -- one utterance each through the live gateway, and
asks the questions a scripted test cannot:

  * does ASR hold up on spontaneous speech across SNR?            (CER vs the dataset's text)
  * does the bot still produce a valid Bengali reply to whatever it heard? (non-empty,
    Bengali script, no code-switch characters)
  * what is first-audio latency on real-length real utterances?

These clips are STATEMENTS, not questions, so the reply is judged for validity, not
correctness -- a bot hearing "they taught me knitting carefully" should respond sensibly in
Bengali, not be graded on facts. The speakers are West Bengal Bengali (Nadia district), which
is a dialect shift from the Bangladeshi target and is stated as such.

    HV_GW_TOKEN=... .venv-bnweb/bin/python -m hervoice.eval.run_real_speech --n 30
"""
import argparse
import asyncio
import glob
import io
import json
import os
import random
import re
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.svc import config as C
from hervoice.svc import protocol as P                       # noqa: E402
from hervoice.eval.run_scenarios import _cer, script_ok    # noqa: E402

# IndicVoices-R Bengali shards, read from the local Hub cache. Override with HV_INDICVOICES_GLOB,
# or let huggingface_hub fetch them (about 4.6 GB) if the cache is empty.
_HF = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
PARQ = os.environ.get(
    "HV_INDICVOICES_GLOB",
    os.path.join(_HF, "hub", "datasets--ai4bharat--indicvoices_r", "snapshots", "*", "Bengali", "*.parquet"))


def _ensure_shards():
    """Return the shard list, downloading from the Hub if the cache has none."""
    fs = sorted(glob.glob(PARQ))
    if fs:
        return fs
    from huggingface_hub import snapshot_download
    d = snapshot_download("ai4bharat/indicvoices_r", repo_type="dataset",
                          allow_patterns=["Bengali/*.parquet"])
    return sorted(glob.glob(os.path.join(d, "Bengali", "*.parquet")))


def load_clips(n, seed=17, min_s=1.5, max_s=12.0):
    import pyarrow.parquet as pq
    import torch
    import torchaudio
    rows = []
    for f in _ensure_shards():
        t = pq.read_table(f, columns=["normalized", "verbatim", "audio", "speaker_id", "scenario",
                                      "gender", "age_group", "area", "district", "snr", "duration"])
        for i in range(t.num_rows):
            d = float(t.column("duration")[i].as_py() or 0)
            if not (min_s <= d <= max_s):
                continue
            rows.append({k: t.column(k)[i].as_py() for k in t.column_names})
    random.Random(seed).shuffle(rows)
    # stratify by SNR tercile so the sample is not all clean studio audio
    snrs = sorted(float(r["snr"]) for r in rows)
    lo, hi = snrs[len(snrs) // 3], snrs[2 * len(snrs) // 3]
    buckets = {"low": [], "mid": [], "high": []}
    for r in rows:
        s = float(r["snr"])
        buckets["low" if s < lo else "mid" if s < hi else "high"].append(r)
    per = max(1, n // 3)
    picked = buckets["low"][:per] + buckets["mid"][:per] + buckets["high"][:n - 2 * per]
    out = []
    for r in picked:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if sr != C.SR_IN:
            a = torchaudio.functional.resample(torch.from_numpy(a)[None], sr, C.SR_IN)[0].numpy()
        out.append(dict(audio=a, text=r["normalized"] or r["verbatim"], speaker=r["speaker_id"],
                        gender=r["gender"], age=r["age_group"], area=r["area"], district=r["district"],
                        snr=round(float(r["snr"]), 1), dur=round(len(a) / C.SR_IN, 2),
                        snr_bucket="low" if float(r["snr"]) < lo else "mid" if float(r["snr"]) < hi else "high"))
    return out, (lo, hi)


async def run(clips, token, out_dir):
    import websockets
    n = int(C.SR_IN * 0.02)
    results = []
    async with websockets.connect(f"ws://127.0.0.1:{C.GW_PORT}/ws", max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "token": token, "sample_rate": C.SR_IN}))
        ready = asyncio.Event()
        st = {"ended": 0, "err": [], "turns": {}}   # turns keyed by server turn number

        async def reader():
            async for m in ws:
                if isinstance(m, (bytes, bytearray)):
                    _, ep, sq, _ = P.unpack_header(m)
                    await ws.send(json.dumps({"type": "played", "epoch": ep, "seq": sq}))   # instant playback
                    continue
                if isinstance(m, str):
                    d = json.loads(m); t = d.get("type"); tn = d.get("turn")
                    if t == "ready": ready.set()
                    if tn is not None:
                        rec = st["turns"].setdefault(tn, {"asr": "", "reply": "", "first": None, "first_se": None, "ended": False})
                    if t == "asr": rec["asr"] = d.get("text", "")
                    elif t == "text": rec["reply"] += d.get("delta", "")
                    elif t == "metrics":
                        if d.get("first_audio_ms"): rec["first"] = d["first_audio_ms"]
                        if d.get("first_audio_from_speech_end_ms"): rec["first_se"] = d["first_audio_from_speech_end_ms"]
                    elif t == "turn_end": rec["ended"] = True; st["ended"] += 1
                    elif t == "error": st["err"].append(d.get("message"))
        rt = asyncio.create_task(reader())
        await asyncio.wait_for(ready.wait(), timeout=30)   # never stream before the server is ready

        async def silence(sec):
            z = np.zeros(n, dtype="<f4").tobytes()
            for _ in range(int(sec * 50)):
                await ws.send(z); await asyncio.sleep(0.02)

        for k, c in enumerate(clips):
            st["err"] = []; before = set(k for k, v in st["turns"].items() if v["ended"])
            want = st["ended"] + 1
            await ws.send(json.dumps({"type": "reset"}))      # each clip is its own conversation
            await silence(0.4)
            for i in range(0, len(c["audio"]), n):
                await ws.send(np.ascontiguousarray(c["audio"][i:i + n], dtype="<f4").tobytes())
                await asyncio.sleep(0.02)
            deadline = time.time() + 60
            while st["ended"] < want and time.time() < deadline:
                await silence(0.2)
            timed_out = st["ended"] < want
            # every server turn this ONE clip produced; >1 means a premature endpoint split it
            new = sorted(k for k, v in st["turns"].items() if v["ended"] and k not in before)
            parts = [st["turns"][k] for k in new]
            asr_all = " | ".join(p_["asr"] for p_ in parts)                 # everything heard, in order
            asr_joined = " ".join(p_["asr"] for p_ in parts)                # for CER against the reference
            reply = (parts[-1]["reply"] if parts else "").strip()           # the reply to the LAST part
            cer = _cer(c["text"], asr_joined)
            ok, frac = script_ok(reply) if reply else (False, 0.0)
            rec = dict(idx=k, snr=c["snr"], snr_bucket=c["snr_bucket"], gender=c["gender"], age=c["age"],
                       area=c["area"], dur=c["dur"], ref=c["text"], asr=asr_all, cer=round(cer, 3),
                       split_into_turns=len(parts), reply=reply, reply_ok=bool(reply) and ok, bn_frac=frac,
                       first_audio_ms=parts[0]["first"] if parts else None,
                       first_audio_from_speech_end_ms=parts[0]["first_se"] if parts else None,
                       timed_out=timed_out, errors=st["err"])
            results.append(rec)
            print(f"  [{k:2d}] snr={c['snr']:5.1f} {c['snr_bucket']:4s} {c['gender'][:1]} {c['area'][:1]} "
                  f"{c['dur']:4.1f}s cer={cer:.2f} reply={'ok ' if rec['reply_ok'] else 'BAD'} "
                  f"first={rec['first_audio_ms'] or 0:5.0f}ms{' SPLIT x'+str(len(parts)) if len(parts)>1 else ''} | {asr_all[:40]}", flush=True)
            await silence(0.6)
        await ws.send(json.dumps({"type": "stop"})); await asyncio.sleep(0.3); rt.cancel()
    return results


def summarize(res, thresholds):
    print(f"\n[real speech] {len(res)} IndicVoices-R Bengali extempore clips, SNR terciles split at {thresholds[0]:.1f}/{thresholds[1]:.1f} dB")
    for b in ("low", "mid", "high"):
        rs = [r for r in res if r["snr_bucket"] == b]
        if not rs: continue
        cers = [r["cer"] for r in rs]
        print(f"  SNR {b:4s} (n={len(rs):2d}): ASR CER mean {np.mean(cers):.3f} median {np.median(cers):.3f} "
              f"| valid Bengali reply {sum(r['reply_ok'] for r in rs)}/{len(rs)}")
    cers = [r["cer"] for r in res]
    fa = [r["first_audio_ms"] for r in res if r["first_audio_ms"]]
    fse = [r["first_audio_from_speech_end_ms"] for r in res if r.get("first_audio_from_speech_end_ms")]
    print(f"  utterances split into >1 turn by a premature endpoint: {sum(1 for r in res if r.get('split_into_turns',1)>1)}/{len(res)}")
    if fse: print(f"  first audio from SPEECH END (includes the silence wait): median {np.median(fse):.0f} ms, p90 {np.percentile(fse,90):.0f} ms  <- the honest number")
    print(f"  ALL: CER mean {np.mean(cers):.3f} median {np.median(cers):.3f} p90 {np.percentile(cers,90):.3f} "
          f"| valid reply {sum(r['reply_ok'] for r in res)}/{len(res)} | timeouts {sum(r['timed_out'] for r in res)}")
    if fa:
        print(f"  first audio from ENDPOINT on real utterances: median {np.median(fa):.0f} ms, p90 {np.percentile(fa,90):.0f} ms")
    bad = [r for r in res if not r["reply_ok"]]
    for r in bad[:5]:
        print(f"    BAD reply example: heard={r['asr'][:40]!r} reply={r['reply'][:60]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--token", default=os.environ.get("HV_GW_TOKEN", ""))
    ap.add_argument("--out", default="runs/svc/real_speech")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clips, th = load_clips(a.n)
    print(f"[load] {len(clips)} clips; {sum(1 for c in clips if c['gender']=='Female')} female, "
          f"{sum(1 for c in clips if c['area']=='Rural')} rural; durations {min(c['dur'] for c in clips):.1f}-{max(c['dur'] for c in clips):.1f}s")
    res = asyncio.run(run(clips, a.token, a.out))
    summarize(res, th)
    json.dump(res, open(os.path.join(a.out, "real_speech.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[out] {a.out}/real_speech.json")


if __name__ == "__main__":
    main()
