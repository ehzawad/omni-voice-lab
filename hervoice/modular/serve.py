#!/usr/bin/env python3
"""Warm turn orchestrator: run one voice turn against the three RESIDENT workers.

This is the persistent-worker counterpart of pipeline.py. Instead of spawning a fresh subprocess
that reloads each model (the ~58 s cold-start tax), it POSTs to long-lived workers whose models are
already loaded and warmed:

    wav ─▶ ASR worker (8091) ─▶ brain llama-server (8090) ─▶ TTS worker (8092) ─▶ wav

A warm turn is inference-only. The honest failure states and manifest schema match pipeline.py:
  empty transcript      -> asr_failed   (no brain call, no wav)
  empty brain answer    -> brain_failed (no TTS,        no wav)
  degenerate TTS audio  -> tts_failed   (tts_reason,    no wav)   [guard lives in tts.synth]

Run from any python with `requests` (e.g. .venv-funasr), workers already up:

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv-funasr/bin/python -m hervoice.modular.serve \
      --wav examples/in_fifa_question.wav --out runs/modular/warm_demo.wav --ref-text "$REF"
"""
import argparse
import json
import os
import sys
import time
import urllib.request

from hervoice.modular import brain as brain_mod
from hervoice.modular import asr as asr_mod

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASR_URL = "http://127.0.0.1:8091"
TTS_URL = "http://127.0.0.1:8092"


def _post(url, obj, timeout=300):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _health(url, timeout=5):
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:
            return json.loads(r.read()).get("loaded") is True
    except Exception:
        return False


def run_turn(wav, out, asr_model, brain_url, ref_text=None):
    """One warm turn. Returns the manifest dict (also written to <out dir>/manifest_warm.json)."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    man_path = os.path.join(os.path.dirname(os.path.abspath(out)), "manifest_warm.json")
    t_start = time.time()
    transcript, answer, usage = "", "", {}
    t_asr = t_brain = t_tts = 0.0
    a, tts_r = {}, {}

    def _finish(status):
        total = time.time() - t_start
        manifest = {
            "mode": "warm",
            "status": status,
            "input_wav": wav, "asr_model": asr_model,
            "asr_hf_id": asr_mod.ASR_MODELS[asr_model]["hf_id"],
            "transcript": transcript, "asr_features": a.get("features"),
            "brain": "Qwen3.5-4B-Q4_K_M.gguf (llama-server)", "answer": answer, "brain_usage": usage,
            "tts_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "output_wav": tts_r.get("out_path"), "tts_status": tts_r.get("status"),
            "tts_reason": tts_r.get("reason"),
            "output_audio_s": tts_r.get("duration_s"), "output_sr": tts_r.get("sr"),
            "output_rms": tts_r.get("rms"),
            "latency_s": {"asr": round(t_asr, 3), "brain": round(t_brain, 3),
                          "tts": round(t_tts, 3), "total": round(total, 3)},
        }
        json.dump(manifest, open(man_path, "w"), indent=2)
        return manifest

    # --- ASR (warm) ---
    t0 = time.time()
    a = _post(f"{ASR_URL}/transcribe", {"wav": wav, "language": "en"})
    t_asr = time.time() - t0
    transcript = a.get("text", "")
    if not transcript.strip():
        return _finish("asr_failed")

    # --- brain (already persistent) ---
    answer, t_brain, usage = brain_mod.ask(transcript, base_url=brain_url)
    if not answer.strip():
        return _finish("brain_failed")

    # --- TTS (warm) ---
    t0 = time.time()
    tts_r = _post(f"{TTS_URL}/synth", {"text": answer, "out_path": out, "ref_text": ref_text})
    t_tts = time.time() - t0
    if tts_r.get("status") != "ok":
        return _finish("tts_failed")
    return _finish("ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(ROOT, "examples", "in_fifa_question.wav"))
    ap.add_argument("--asr", default="qwen3-asr-0.6b", choices=list(asr_mod.ASR_MODELS))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "modular", "warm_demo.wav"))
    ap.add_argument("--brain-url", default=brain_mod.DEFAULT_URL)
    ap.add_argument("--ref-text", default=None)
    args = ap.parse_args()

    print("[serve] health: brain=", brain_mod.health(args.brain_url),
          " asr=", _health(ASR_URL), " tts=", _health(TTS_URL), sep="", flush=True)
    if not (brain_mod.health(args.brain_url) and _health(ASR_URL) and _health(TTS_URL)):
        print("[serve] one or more workers are DOWN -- run start_workers.sh first.", file=sys.stderr)
        sys.exit(1)

    m = run_turn(args.wav, args.out, args.asr, args.brain_url, ref_text=args.ref_text)
    print(json.dumps(m, indent=2))
    ls = m["latency_s"]
    print(f"[serve] {m['status']}  ASR {ls['asr']}s | brain {ls['brain']}s | "
          f"TTS {ls['tts']}s | total {ls['total']}s", flush=True)
    if m["status"] != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
