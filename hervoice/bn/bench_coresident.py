#!/usr/bin/env python3
"""GATE: prove co-residency and measure a real turn, BEFORE any web code is written.

docs/HERVOICE_DEMO.md records ~71 s warm per turn for the Bengali path and attributes it to
per-stage model load/free under sequential GPU residency, not to inference. This script tests
the claim that making all three models co-resident removes that cost, and it measures the one
number a voice bot lives or dies by: time from end-of-speech to FIRST AUDIO.

It reports per-stage latency honestly, including the part that cannot be hidden: IndicF5 is
flow matching, so the first chunk must complete in full before a single sample exists.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      .venv-bnweb/bin/python -m hervoice.bn.bench_coresident --audio examples/in_bn_question.wav
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.bn.models import BnBrain, load_all, peak_vram_gb, vram_gb   # noqa: E402

SYSTEM_BN = (
    "তুমি একজন সহায়ক বাংলা কণ্ঠ-সহকারী। সবসময় বাংলায় উত্তর দাও। "
    "উত্তর সংক্ষিপ্ত রাখো — দুই থেকে তিনটি ছোট বাক্য।"
)


def first_sentence(buf, tts):
    """Return (sentence, rest) once a complete danda-delimited sentence is available."""
    for mark in ("।", "?", "!"):
        i = buf.find(mark)
        if i >= 0:
            return buf[:i + 1].strip(), buf[i + 1:]
    return None, buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="examples/in_bn_question.wav")
    ap.add_argument("--brain", default=None, help="override the brain model id (e.g. Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--tts-repo", default=None)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--turns", type=int, default=3, help="measured turns AFTER the warm-up turn")
    ap.add_argument("--out", default="runs/bnweb_bench")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] device={dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}", flush=True)
    kw = {}
    if a.tts_repo:
        kw["tts_repo"] = a.tts_repo
    asr, brain, tts, rep = load_all(device=dev, **kw)
    if a.brain:
        del brain
        torch.cuda.empty_cache()
        t = time.time()
        brain = BnBrain(model_id=a.brain, device=dev)
        rep["brain"] = dict(load_s=round(time.time() - t, 1), vram_gb=vram_gb(), model=a.brain)
        print(f"  brain swapped to {a.brain}: {rep['brain']['load_s']}s  resident {rep['brain']['vram_gb']} GiB", flush=True)
        rep["resident_total_gb"] = vram_gb()
    tts.nfe = a.nfe

    print(f"[load] ALL THREE RESIDENT: {rep['resident_total_gb']} GiB allocated, "
          f"{peak_vram_gb()} GiB peak\n", flush=True)

    audio, sr = sf.read(a.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    print(f"[turn] input {a.audio}  {len(audio)/sr:.2f}s", flush=True)

    def one_turn(label, verbose=True):
        """One full turn. Returns the timing record; `label` distinguishes warm-up."""
        t_turn = time.time()
        t = time.time()
        asr_text = asr.transcribe(audio)
        t_asr = (time.time() - t) * 1000
        if not asr_text:
            return None
        buf, waves, sents = "", [], []
        t_first_tok = t_first_audio = None
        tts_ms = []
        t_gen = time.time()
        full = ""
        for piece in brain.stream(SYSTEM_BN, asr_text, max_new_tokens=a.max_new_tokens):
            if t_first_tok is None:
                t_first_tok = (time.time() - t_gen) * 1000
            buf += piece
            full += piece
            sent, buf = first_sentence(buf, tts)
            if sent:
                for ci, ch in enumerate(tts.chunks(sent)):
                    ts = time.time()
                    w = tts.synth_chunk(ch, seed=1234 + 100 * len(sents) + ci)
                    tts_ms.append((time.time() - ts) * 1000)
                    waves.append(w)
                    if t_first_audio is None:
                        t_first_audio = (time.time() - t_turn) * 1000
                sents.append(sent)
        tail = buf.strip()
        if tail:
            for ci, ch in enumerate(tts.chunks(tail)):
                ts = time.time()
                w = tts.synth_chunk(ch, seed=9000 + ci)
                tts_ms.append((time.time() - ts) * 1000)
                waves.append(w)
                if t_first_audio is None:
                    t_first_audio = (time.time() - t_turn) * 1000
            sents.append(tail)
        wav = np.concatenate(waves) if waves else np.zeros(0, dtype=np.float32)
        rec = dict(label=label, asr_ms=round(t_asr, 1),
                   brain_first_token_ms=round(t_first_tok or 0, 1),
                   tts_first_chunk_ms=round(tts_ms[0] if tts_ms else 0, 1),
                   first_audio_ms=round(t_first_audio or 0, 1),
                   turn_total_ms=round((time.time() - t_turn) * 1000, 1),
                   audio_out_s=round(len(wav) / 24000, 2),
                   tts_rtf=round((sum(tts_ms) / 1000) / max(len(wav) / 24000, 1e-6), 2),
                   asr_text=asr_text, reply=full.strip(), sentences=len(sents))
        if verbose:
            print(f"  [{label}] ASR {rec['asr_ms']:6.0f} | brain-1st-tok {rec['brain_first_token_ms']:7.0f}"
                  f" | TTS-1st {rec['tts_first_chunk_ms']:7.0f} | FIRST AUDIO {rec['first_audio_ms']:7.0f} ms"
                  f" | {rec['audio_out_s']}s out, RTF {rec['tts_rtf']}", flush=True)
        return rec, wav

    print("[warm] one warm-up turn (CUDA kernels, NeMo dataloader, allocator) -- discarded", flush=True)
    w0 = one_turn("warmup")
    if w0 is None:
        print("  ASR returned empty; aborting"); return

    torch.cuda.reset_peak_memory_stats()
    recs = []
    for i in range(a.turns):
        r = one_turn(f"turn{i+1}")
        if r:
            recs.append(r[0])
            last_wav = r[1]
    if not recs:
        print("  no measured turns"); return

    def med(k):
        v = sorted(x[k] for x in recs)
        return round(v[len(v) // 2], 1)

    sf.write(os.path.join(a.out, "answer.wav"), last_wav, 24000)
    out = dict(
        audio=a.audio, input_s=round(len(audio) / sr, 2),
        brain=a.brain or "Qwen/Qwen2.5-3B-Instruct", nfe=a.nfe, turns=len(recs),
        cold=w0[0], warm_median=dict(
            asr_ms=med("asr_ms"), brain_first_token_ms=med("brain_first_token_ms"),
            tts_first_chunk_ms=med("tts_first_chunk_ms"), first_audio_ms=med("first_audio_ms"),
            turn_total_ms=med("turn_total_ms"), tts_rtf=med("tts_rtf")),
        turns_detail=recs, resident_gb=rep["resident_total_gb"], peak_gb=peak_vram_gb(), load=rep,
    )
    json.dump(out, open(os.path.join(a.out, "bench.json"), "w"), ensure_ascii=False, indent=1)
    m = out["warm_median"]
    print(f"\n[WARM MEDIAN over {len(recs)} turns] ASR {m['asr_ms']:.0f} ms | brain first token "
          f"{m['brain_first_token_ms']:.0f} ms | TTS first chunk {m['tts_first_chunk_ms']:.0f} ms"
          f"\n[WARM MEDIAN] >>> FIRST AUDIO {m['first_audio_ms']:.0f} ms <<< | full turn "
          f"{m['turn_total_ms']:.0f} ms | TTS RTF {m['tts_rtf']}")
    print(f"[vram] resident {out['resident_gb']} GiB, peak {out['peak_gb']} GiB")
    print(f"[reply] {recs[-1]['reply']!r}")
    print(f"[out] {a.out}/answer.wav + bench.json")


if __name__ == "__main__":
    main()
