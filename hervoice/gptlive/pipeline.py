#!/usr/bin/env python3
"""pipeline.py -- wire the gptlive front + controller + brain end to end.

Headless flow (no mic on this box), one spoken English question in:

  spoken question wav
    -> [ASR] accurate user transcript (faster-whisper, CPU, .venv-hervoice)
    -> [Moshi FRONT] full-duplex listen: Moshi's own inner monologue + audio
    -> [CONTROLLER] delegate yes/no on the user transcript
    -> [BRAIN] Qwen3.5-4B GGUF (llama-server) -> concise reasoned answer
    -> [Moshi FRONT] TEACHER-FORCE Moshi's text stream to SPEAK the answer
    -> [ASR] transcribe the forced output wav to verify Moshi vocalised it
    -> write runs/gptlive/{manifest.json, *.wav, log} + results_gptlive.json

GPU0 only. Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv-duplex/bin/python -m hervoice.gptlive.pipeline
(The brain -- llama-server on :8090 -- must already be running on GPU0.)
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from .front import MoshiFront, nvidia_smi_used_mb
from . import delegate

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "runs" / "gptlive"
ASR_PY = REPO_ROOT / ".venv-hervoice" / "bin" / "python"
ASR_WORKER = Path(__file__).resolve().parent / "_asr_worker.py"


def asr(wav_path: str, log) -> str:
    """Transcribe a wav via faster-whisper in .venv-hervoice (CPU subprocess)."""
    try:
        out = subprocess.check_output(
            [str(ASR_PY), str(ASR_WORKER), str(wav_path)],
            text=True, stderr=subprocess.DEVNULL, timeout=180)
        return out.strip()
    except Exception as e:
        log(f"[asr] FAILED on {wav_path}: {e}")
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",
                    default=str(REPO_ROOT / "examples" / "in_en_question.wav"))
    ap.add_argument("--gap", type=int, default=2,
                    help="PAD frames inserted between forced word-pieces (pacing)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(OUT_DIR / "log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    timings = {}
    vram = {}

    log("=== gptlive: local open-weight GPT-Live mirror ===")
    log(f"infile = {args.infile}")

    # -- 0. brain must be up --------------------------------------------------
    if not delegate.brain_healthy():
        log("[brain] /health NOT ok -- start llama-server on :8090 first. Aborting.")
        logf.close(); raise SystemExit(1)
    log("[brain] /health ok (Qwen3.5-4B on :8090)")
    vram["after_brain_only_mb"] = nvidia_smi_used_mb(0)
    log(f"[vram] GPU0 used with brain only = {vram['after_brain_only_mb']:.0f} MB")

    # -- 1. accurate user transcript (for the controller) ---------------------
    t = time.time()
    user_transcript = asr(args.infile, log)
    timings["asr_user_s"] = round(time.time() - t, 2)
    log(f"[asr] USER transcript = {user_transcript!r}  ({timings['asr_user_s']}s)")

    # -- 2. load Moshi front (co-resident with the brain) ---------------------
    t = time.time()
    front = MoshiFront(log=log)
    timings["load_moshi_s"] = round(time.time() - t, 2)
    vram["after_both_resident_mb"] = nvidia_smi_used_mb(0)
    log(f"[vram] GPU0 used with Moshi + brain BOTH resident = "
        f"{vram['after_both_resident_mb']:.0f} MB")

    user_chunks = front.load_user_wav(args.infile)
    log(f"[front] user audio -> {len(user_chunks)} frames "
        f"({len(user_chunks)/front.frame_rate:.2f}s @ {front.frame_rate}Hz)")

    # -- 3. LISTEN: Moshi full-duplex, capture its own inner monologue --------
    t = time.time()
    listen = front.run_turn(user_chunks, speak_schedule=None)
    timings["moshi_listen_s"] = round(time.time() - t, 2)
    moshi_inner = listen["listen_inner_text"]
    log(f"[front] Moshi inner monologue (its OWN reply, not a user ASR) = "
        f"{moshi_inner!r}")
    front.write_wav(OUT_DIR / "moshi_listen_reply.wav", listen["listen_audio"])

    # -- 4. CONTROLLER: delegate? ---------------------------------------------
    delegate_yes, reason = delegate.classify(user_transcript)
    log(f"[controller] delegate = {delegate_yes}  ({reason})")

    brain_answer = None
    if delegate_yes:
        t = time.time()
        brain_answer, brain_dt, _ = delegate.ask_brain(user_transcript)
        timings["brain_answer_s"] = round(brain_dt, 2)
        log(f"[brain] answer = {brain_answer!r}  ({timings['brain_answer_s']}s)")
    else:
        log("[controller] no delegation -> front would answer conversationally")

    # -- 5. SPEAK: teacher-force Moshi to vocalise the brain's answer ---------
    forced_asr = None
    speak_wav_path = None
    if brain_answer:
        ids, sched = front.build_speak_schedule(brain_answer, gap=args.gap)
        log(f"[front] teacher-forcing {len(ids)} answer word-pieces over "
            f"{len(sched)} speak frames (gap={args.gap})")
        t = time.time()
        spoken = front.run_turn(user_chunks, speak_schedule=sched)
        timings["moshi_speak_s"] = round(time.time() - t, 2)
        speak_wav_path = OUT_DIR / "gptlive_spoken_answer.wav"
        front.write_wav(speak_wav_path, spoken["speak_audio"])
        dur = len(spoken["speak_audio"]) / front.sample_rate
        log(f"[front] forced-speech wav = {speak_wav_path.name} "
            f"({dur:.2f}s, {spoken['n_forced_frames']} forced frames)")
        log(f"[front] text Moshi emitted while forced = "
            f"{spoken['speak_forced_text']!r}")
        # objective loop-closure: what does ASR hear in the forced output?
        t = time.time()
        forced_asr = asr(speak_wav_path, log)
        timings["asr_forced_out_s"] = round(time.time() - t, 2)
        log(f"[verify] ASR of Moshi's teacher-forced output = {forced_asr!r}")

    vram["peak_both_resident_mb"] = nvidia_smi_used_mb(0)
    log(f"[vram] GPU0 peak with both resident = {vram['peak_both_resident_mb']:.0f} MB")

    # -- 6. artifacts ---------------------------------------------------------
    manifest = {
        "design": "GPT-Live mirror: Moshi full-duplex FRONT + delegated Qwen3.5-4B brain",
        "infile": args.infile,
        "user_transcript_asr": user_transcript,
        "moshi_inner_monologue_own_reply": moshi_inner,
        "delegation": {"delegate": delegate_yes, "reason": reason},
        "brain_answer": brain_answer,
        "spoken_output_wav": (str(speak_wav_path.relative_to(REPO_ROOT))
                              if speak_wav_path else None),
        "spoken_output_asr_verification": forced_asr,
        "moshi_listen_reply_wav": "runs/gptlive/moshi_listen_reply.wav",
        "timings_s": timings,
        "vram_mb": vram,
        "peak_vram_gb": round(vram["peak_both_resident_mb"] / 1024.0, 2),
        "gpu": "GPU0 (RTX A5000) only; GPU1 untouched",
        "teacher_forcing": {
            "achieved": brain_answer is not None,
            "mechanism": "LMGen(on_text_hook=...) overwrites the sampled text "
                         "token in place before the depformer vocalises it",
            "pacing": f"heuristic: 1 word-piece + {args.gap} pad frames, "
                      "no learned alignment model",
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (REPO_ROOT / "results_gptlive.json").write_text(json.dumps(manifest, indent=2))
    log(f"[done] wrote {OUT_DIR/'manifest.json'} and results_gptlive.json")
    logf.close()


if __name__ == "__main__":
    main()
