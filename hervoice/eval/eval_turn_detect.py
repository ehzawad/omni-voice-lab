#!/usr/bin/env python3
"""Turn-end detection on REAL spontaneous Bengali: acoustic silence vs a semantic model.

The live run showed the failure: Silero with min_silence 220 ms fires SPEECH_END inside
utterances at natural pauses -- 7.8 s of speech heard as one word -- and every high-CER clip
had 1-4 such false ends while every clean clip had zero. Raising min_silence to 800 ms
removes most of them but adds ~600 ms to EVERY turn, including the ones that were fine.

Smart Turn v3 (pipecat-ai, BSD-2) is a Whisper-tiny encoder + linear head that looks at the
last 8 s of audio at a candidate pause and predicts whether the speaker is DONE. Its published
Bengali accuracy is 83.8 % (v3.2-cpu, 1000 samples, FPR 10.9 %, FNR 5.3 %). This script uses
Pipecat's exact preprocessing (vendored in hervoice/svc/whisper_features.py): keep the LAST
8 s, left-pad with zeros, Whisper log-mel with normalisation; the ONNX returns a sigmoid
probability; complete if > 0.5; a `stop_secs` fallback forces the end after long silence.

Streaming simulation per clip (20 ms frames, 1.2 s of trailing silence appended):
  * SILERO@X   : SPEECH_END with min_silence_ms = X ends the turn. Baseline.
  * GATED@220  : Silero@220 proposes; Smart Turn decides. A rejected proposal keeps the turn
                 open; the turn ends on accept, or on stop_secs of continuous silence.

Ground truth is the clip boundary. These are dataset SEGMENTS of spontaneous speech, so a
segment end is not always a semantic turn end -- a "miss" at the boundary is reported but
read with that caveat. The hard, unambiguous number is PREMATURE ENDS: any accepted end
before the speech has actually finished is a cut-off user.

    CUDA_VISIBLE_DEVICES= .venv-bnweb/bin/python -m hervoice.eval.eval_turn_detect --n 30
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.live.turn_detector import TurnDetector, VadEvent          # noqa: E402
from hervoice.svc.whisper_features import compute_whisper_log_mel_features  # noqa: E402
from hervoice.eval.run_real_speech import load_clips                    # noqa: E402

SR = 16000
FRAME = 320                      # 20 ms
TRAIL_S = 1.2                    # silence appended so the true end can be detected
GT_TOL_S = 0.30                  # an end within this of the true end counts as correct


class SmartTurn:
    def __init__(self, repo="pipecat-ai/smart-turn-v3", fname="smart-turn-v3.2-cpu.onnx", threshold=0.5):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, fname)
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.fname = fname
        self.ms = []

    def complete_prob(self, audio16k):
        n = 8 * SR
        a = np.asarray(audio16k, dtype=np.float32)
        a = a[-n:] if len(a) >= n else np.pad(a, (n - len(a), 0))      # keep END, left-pad
        t = time.time()
        feats = compute_whisper_log_mel_features(a, do_normalize=True)
        out = self.s.run(None, {"input_features": feats[None].astype(np.float32)})[0]
        self.ms.append((time.time() - t) * 1000)
        return float(np.asarray(out).reshape(-1)[0])


def simulate(audio, true_end_s, min_silence_ms, gate=None, stop_secs=3.0):
    """Return dict(ends=[accepted end times], premature=int, final_latency_ms|None, decisions)."""
    a = np.concatenate([audio, np.zeros(int(TRAIL_S * SR), dtype=np.float32)])
    det = TurnDetector(min_silence_ms=min_silence_ms)
    ends, decisions = [], []
    in_speech = False
    silence_since = None
    for i in range(0, len(a), FRAME):
        fr = a[i:i + FRAME]
        now = (i + len(fr)) / SR
        for ev in det.process(fr):
            if ev.kind == VadEvent.SPEECH_START:
                in_speech = True; silence_since = None
            elif ev.kind == VadEvent.SPEECH_END:
                if gate is None:
                    ends.append(now); in_speech = False
                else:
                    p = gate.complete_prob(a[:i + len(fr)])
                    decisions.append((round(now, 2), round(p, 3)))
                    if p > gate.threshold:
                        ends.append(now); in_speech = False; silence_since = None
                    else:
                        silence_since = now        # keep the turn open, start the fallback clock
                        det.triggered = True       # Silero must not re-fire START for the same speech
        if gate is not None and silence_since is not None and (now - silence_since) >= stop_secs:
            ends.append(now); silence_since = None; in_speech = False
    premature = sum(1 for e in ends if e < true_end_s - GT_TOL_S)
    final = [e for e in ends if e >= true_end_s - GT_TOL_S]
    return dict(ends=[round(e, 2) for e in ends], premature=premature,
                final_latency_ms=round((final[0] - true_end_s) * 1000, 0) if final else None,
                decisions=decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--model", default="smart-turn-v3.2-cpu.onnx")
    ap.add_argument("--out", default="runs/svc/turn_detect")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clips, _ = load_clips(a.n)
    st = SmartTurn(fname=a.model)
    print(f"[load] {len(clips)} IndicVoices-R extempore clips; Smart Turn {a.model}")

    methods = {"silero@220": dict(ms=220), "silero@500": dict(ms=500), "silero@800": dict(ms=800),
               "gated@220": dict(ms=220, gate=True)}
    rows = []
    for k, c in enumerate(clips):
        true_end = len(c["audio"]) / SR
        r = dict(idx=k, dur=round(true_end, 2), snr=c["snr"])
        for name, cfg in methods.items():
            r[name] = simulate(c["audio"], true_end, cfg["ms"], st if cfg.get("gate") else None)
        rows.append(r)
        g = r["gated@220"]; s2 = r["silero@220"]
        print(f"  [{k:2d}] {true_end:5.1f}s  silero@220 prem={s2['premature']} | gated prem={g['premature']} "
              f"final={g['final_latency_ms']} ms  probs={[p for _, p in g['decisions']][:5]}", flush=True)

    print("\n[turn-end detection on real spontaneous Bengali]")
    print(f"  {'method':12s} {'clips cut off':>13s} {'total premature ends':>21s} {'final detected':>15s} {'median final latency':>21s}")
    for name in methods:
        cut = sum(1 for r in rows if r[name]["premature"] > 0)
        prem = sum(r[name]["premature"] for r in rows)
        fin = [r[name]["final_latency_ms"] for r in rows if r[name]["final_latency_ms"] is not None]
        med = f"{np.median(fin):.0f} ms" if fin else "-"
        print(f"  {name:12s} {cut:>7d}/{len(rows):<5d} {prem:>21d} {len(fin):>9d}/{len(rows):<5d} {med:>21s}")
    print(f"\n  Smart Turn inference: median {np.median(st.ms):.1f} ms, p95 {np.percentile(st.ms,95):.1f} ms per decision (CPU, 1 thread)")
    print("  'clips cut off' = at least one accepted end BEFORE the speech finished. That is the user being interrupted.")
    print("  'final detected' counts a clip boundary as a turn end; segments are not always semantic ends, so read the miss count with that caveat.")
    json.dump(rows, open(os.path.join(a.out, "turn_detect.json"), "w"), indent=1)
    print(f"[out] {a.out}/turn_detect.json")


if __name__ == "__main__":
    main()
