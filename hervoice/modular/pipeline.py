#!/usr/bin/env python3
"""End-to-end modular voice assistant: wav -> ASR -> Qwen3.5-4B brain -> Qwen3-TTS -> wav.

The three stages live in three venvs (deps conflict), so this orchestrator runs ASR and TTS as
subprocesses in their own venv, and talks to the brain over HTTP. Running each model stage as a
short-lived subprocess also frees its VRAM before the next stage, so peak GPU stays low
(brain server + one model at a time). GPU0 only; GPU1 is never touched.

Run from any python that has `requests` (e.g. the funasr venv):

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv-funasr/bin/python -m hervoice.modular.pipeline \
      --wav examples/in_fifa_question.wav --asr qwen3-asr-1.7b
"""
import argparse
import json
import os
import subprocess
import sys
import time

from hervoice.modular import brain as brain_mod
from hervoice.modular import asr as asr_mod

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = {
    "qwen": os.path.join(ROOT, ".venv-qwen-asr", "bin", "python"),
    "funasr": os.path.join(ROOT, ".venv-funasr", "bin", "python"),
    "tts": os.path.join(ROOT, ".venv-qwen-audio", "bin", "python"),
}
GPU_ENV = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"}


def _run_json(py, code):
    """Run `code` in interpreter `py`; the code must print one JSON line last."""
    p = subprocess.run([py, "-c", code], cwd=ROOT, env=GPU_ENV,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"subprocess failed:\n{p.stderr[-2000:]}")
    line = [l for l in p.stdout.strip().splitlines() if l.strip().startswith("{")][-1]
    return json.loads(line)


def run_asr(key, wav):
    family = asr_mod.ASR_MODELS[key]["family"]
    py = PY["qwen"] if family == "qwen" else PY["funasr"]
    code = (
        "import json;from hervoice.modular import asr;"
        f"r=asr.transcribe({key!r},{wav!r},language='en');"
        "print(json.dumps(r))"
    )
    return _run_json(py, code)


def run_tts(text, out_wav, ref_text=None):
    code = (
        "import json;from hervoice.modular import tts;"
        f"r=tts.synth({text!r},{out_wav!r},ref_text={ref_text!r});"
        "print(json.dumps(r))"
    )
    return _run_json(PY["tts"], code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(ROOT, "examples", "in_fifa_question.wav"))
    ap.add_argument("--asr", default="qwen3-asr-0.6b", choices=list(asr_mod.ASR_MODELS))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "modular", "pipeline_demo.wav"))
    ap.add_argument("--brain-url", default=brain_mod.DEFAULT_URL)
    ap.add_argument("--ref-text", default=None, help="transcript of the TTS reference voice (optional)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t_start = time.time()

    print(f"[STATE] input wav      : {args.wav}")
    print(f"[STATE] brain health   : ", end="", flush=True)
    if not brain_mod.health(args.brain_url):
        print("DOWN -- start llama-server first (see brain.py docstring)")
        sys.exit(1)
    print("ok")

    print(f"[STATE] ASR ({args.asr}) ...", flush=True)
    t0 = time.time()
    a = run_asr(args.asr, args.wav)
    t_asr = time.time() - t0
    transcript = a["text"]
    print(f"[ASR]   transcript     : {transcript!r}  (features: {a.get('features') or '-'}, {t_asr:.2f}s)")

    # honest failure states -- never fabricate a downstream turn on empty input.
    status, answer, usage, t_brain, tts_r, t_tts = "ok", "", {}, 0.0, {}, 0.0
    man_path = os.path.join(os.path.dirname(args.out), "manifest.json")

    def _finish(status, code=0):
        total = time.time() - t_start
        manifest = {
            "status": status,
            "input_wav": args.wav, "asr_model": args.asr,
            "asr_hf_id": asr_mod.ASR_MODELS[args.asr]["hf_id"],
            "transcript": transcript, "asr_features": a.get("features"),
            "brain": "Qwen3.5-4B-Q4_K_M.gguf (llama-server)", "answer": answer, "brain_usage": usage,
            "tts_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "output_wav": tts_r.get("out_path"), "tts_status": tts_r.get("status"),
            "tts_reason": tts_r.get("reason"),
            "output_audio_s": tts_r.get("duration_s"), "output_sr": tts_r.get("sr"),
            "output_rms": tts_r.get("rms"),
            "latency_s": {"asr": round(t_asr, 2), "brain": round(t_brain, 2),
                          "tts": round(t_tts, 2), "total": round(total, 2)},
        }
        json.dump(manifest, open(man_path, "w"), indent=2)
        print(f"[STATE] {status} in {total:.2f}s -> manifest {man_path}")
        if code:
            sys.exit(code)
        return manifest

    if not transcript.strip():
        print("[ASR]   FAILED -- empty transcript; aborting (no fabricated answer).")
        return _finish("asr_failed", code=2)

    print(f"[STATE] brain thinking ...", flush=True)
    answer, t_brain, usage = brain_mod.ask(transcript, base_url=args.brain_url)
    print(f"[BRAIN] answer         : {answer!r}  ({t_brain:.2f}s, {usage.get('completion_tokens','?')} tok)")
    if not answer.strip():
        print("[BRAIN] FAILED -- empty answer; aborting (no TTS on empty text).")
        return _finish("brain_failed", code=2)

    print(f"[STATE] TTS synthesizing ...", flush=True)
    t0 = time.time()
    tts_r = run_tts(answer, args.out, ref_text=args.ref_text)
    t_tts = time.time() - t0
    if tts_r.get("status") != "ok":
        print(f"[TTS]   FAILED -- {tts_r.get('reason')}; no wav written.")
        return _finish("tts_failed", code=2)
    print(f"[TTS]   wav            : {tts_r['out_path']}  ({tts_r['duration_s']}s @ {tts_r['sr']}Hz, "
          f"rms={tts_r['rms']}, {t_tts:.2f}s)")
    return _finish("ok")


if __name__ == "__main__":
    main()
