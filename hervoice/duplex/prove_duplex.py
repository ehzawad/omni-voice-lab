#!/usr/bin/env python3
"""Headless native full-duplex proof for Moshi (kyutai/moshiko-pytorch-bf16).

WHAT THIS DEMONSTRATES
----------------------
Moshi is ONE autoregressive model that jointly models two audio streams (the
user's and its own) plus a time-aligned text "inner monologue". Turn-taking is
decided INSIDE the network, frame by frame, at 12.5 Hz (Mimi codec, 80 ms
frames). There is NO external voice-activity detector, NO turn-taking state
machine, NO energy gate anywhere in this file. We simply:

  1. feed the user's audio stream frame-by-frame (Mimi-encoded),
  2. after the user's audio ends, keep stepping the model on SILENCE frames,
  3. read back, per frame, the model's OWN outputs:
       - inner-monologue text token  (out[:, 0])
       - 8 audio codebook tokens      (out[:, 1:])  -> decoded to waveform via Mimi

The model's inner-monologue text stream emits a "pad" token (id 0) when it has
nothing to say (it is listening) and real word-pieces when it decides to speak.
That transition -- chosen by the network, not by us -- is the turn-take.

NOTE ON MEASUREMENT: we compute the RMS of the model's OWN decoded output audio
purely as descriptive evidence for the log. It is never fed back into any
decision and never gates anything. The only signal we read to say "the model
chose to speak" is the model's own inner-monologue text stream.

GPU: GPU0 only. Run with:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv-duplex/bin/python -m hervoice.duplex.prove_duplex
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import sphn
import torch

from moshi.models import LMGen, loaders

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "runs" / "duplex"
PAD_TOKEN = 0          # Moshi text "pad" = model has no word this frame (listening)
SPECIAL_TOKENS = {0, 3}  # 0 = pad, 3 = other special; skipped when detokenizing


def nvidia_smi_used_mb(gpu_index: int = 0) -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(gpu_index)],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(REPO_ROOT / "examples" / "in_fifa_question.wav"))
    ap.add_argument("--silence-seconds", type=float, default=10.0,
                    help="trailing silence fed AFTER user audio ends, so the model "
                         "has room to take its turn (chosen by the model, not us)")
    ap.add_argument("--repo", default=loaders.DEFAULT_REPO)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "log"
    logf = open(log_path, "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available"
    log(f"device0 = {torch.cuda.get_device_name(0)}")
    log(f"repo = {args.repo}")
    torch.cuda.reset_peak_memory_stats()

    # ---- load model (Mimi codec + Moshi LM + text tokenizer) --------------
    log("retrieving checkpoint ...")
    ckpt = loaders.CheckpointInfo.from_hf_repo(args.repo)
    log(f"model_type = {ckpt.model_type}")
    log("loading mimi ...")
    mimi = ckpt.get_mimi(device=device)
    text_tokenizer = ckpt.get_text_tokenizer()
    log("loading moshi lm ...")
    lm = ckpt.get_moshi(device=device, dtype=torch.bfloat16)
    log(f"dep_q = {lm.dep_q}, delays(max) = {max(lm.delays)}")

    lm_gen = LMGen(lm, **ckpt.lm_gen_config)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)   # 1920 samples @ 24kHz
    log(f"sample_rate = {mimi.sample_rate}, frame_rate = {mimi.frame_rate} Hz, "
        f"frame_size = {frame_size} samples ({1000/mimi.frame_rate:.0f} ms)")

    peak_after_load_mb = nvidia_smi_used_mb(0)
    log(f"nvidia-smi used after load (GPU0) = {peak_after_load_mb:.0f} MB")

    # ---- load user audio, resample to 24kHz mono --------------------------
    in_pcms, _ = sphn.read(args.infile, sample_rate=mimi.sample_rate)  # [C, T]
    in_pcms = torch.from_numpy(in_pcms).to(device=device)[None, 0:1]   # [1,1,T]
    user_samples = in_pcms.shape[-1]
    user_frames = user_samples // frame_size
    log(f"user audio: {args.infile}")
    log(f"user audio: {user_samples} samples = {user_samples/mimi.sample_rate:.2f} s "
        f"= {user_frames} full frames @ 12.5 Hz")

    user_chunks = [c for c in in_pcms.split(frame_size, dim=2)
                   if c.shape[-1] == frame_size]
    n_silence = int(round(args.silence_seconds * mimi.frame_rate))
    log(f"trailing silence frames = {n_silence} ({args.silence_seconds:.1f} s)")

    silence_chunk = torch.zeros((1, mimi.channels, frame_size), device=device)

    # ---- the streaming step loop (the whole point) ------------------------
    # per-frame records: (global_frame_idx, phase, text_token, piece, out_audio_rms)
    records = []
    out_audio_chunks = []
    inner_tokens = []

    log("=== stepping the model frame-by-frame (12.5 Hz) ===")
    t0 = time.time()
    first_frame = True
    step_idx = -1  # index into emitted (post-delay) outputs
    with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
        total_input_frames = len(user_chunks) + n_silence
        for i in range(total_input_frames):
            if i < len(user_chunks):
                phase = "USER"
                codes = mimi.encode(user_chunks[i])
            else:
                phase = "SILENCE"
                codes = mimi.encode(silence_chunk)

            if first_frame:
                # warm the transformer so the very first real codes are seen
                _ = lm_gen.step(codes)
                first_frame = False

            tokens = lm_gen.step(codes)
            if tokens is None:
                # still inside the model's output delay window
                continue
            step_idx += 1
            assert tokens.shape[1] == lm.dep_q + 1
            text_tok = int(tokens[0, 0].item())
            audio_tok = tokens[:, 1:]
            wav = mimi.decode(audio_tok)[0, 0].float().cpu().numpy()  # [frame_size]
            rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-12)
            out_audio_chunks.append(wav)

            piece = ""
            if text_tok not in SPECIAL_TOKENS:
                piece = text_tokenizer.id_to_piece(text_tok).replace("▁", " ")
                inner_tokens.append(text_tok)

            records.append({
                "frame": step_idx,
                "input_phase": phase,
                "text_token": text_tok,
                "piece": piece,
                "out_audio_rms": rms,
            })

    dt = time.time() - t0
    peak_alloc_gb = torch.cuda.max_memory_allocated(0) / 1e9
    peak_smi_mb = nvidia_smi_used_mb(0)
    frames_stepped = len(records)
    log(f"stepped {frames_stepped} output frames in {dt:.1f}s "
        f"({1000*dt/max(frames_stepped,1):.0f} ms/step)")
    log(f"peak torch.cuda.max_memory_allocated (GPU0) = {peak_alloc_gb:.2f} GB")
    log(f"nvidia-smi used at end (GPU0) = {peak_smi_mb:.0f} MB")

    # ---- analyse emergent turn-taking (reading ONLY the model's own text) -
    # A frame counts as "model speaking" when its inner-monologue emits a real
    # word-piece (text token not in {pad, special}). This is the network's own
    # decision -- we do not detect anything on the user's audio.
    speaking_frames = [r["frame"] for r in records if r["piece"] != ""]
    model_spoke_frames = len(speaking_frames)
    first_model_speech_frame = speaking_frames[0] if speaking_frames else None

    # How many "model speaking" frames fell during USER audio vs during SILENCE?
    spoke_during_user = sum(
        1 for r in records if r["piece"] != "" and r["input_phase"] == "USER")
    spoke_during_silence = sum(
        1 for r in records if r["piece"] != "" and r["input_phase"] == "SILENCE")

    inner_text = "".join(text_tokenizer.id_to_piece(t).replace("▁", " ")
                         for t in inner_tokens).strip()

    log("--- emergent turn-taking evidence ---")
    log(f"model-speaking frames (inner monologue emitted a word): {model_spoke_frames}")
    log(f"first frame the model chose to speak: {first_model_speech_frame}")
    log(f"  ... during USER audio: {spoke_during_user} frames")
    log(f"  ... during SILENCE   : {spoke_during_silence} frames")
    if first_model_speech_frame is not None:
        log(f"user audio occupied output frames ~0..{len(user_chunks)-1}; "
            f"the model began speaking at frame {first_model_speech_frame}")
    log(f"INNER MONOLOGUE TEXT: {inner_text!r}")

    # ---- save the model's output audio (decoded via Mimi) -----------------
    reply = np.concatenate(out_audio_chunks) if out_audio_chunks else np.zeros(1)
    reply_path = OUT_DIR / "moshi_reply.wav"
    sphn.write_wav(str(reply_path), reply[None, :], sample_rate=mimi.sample_rate)
    log(f"wrote model reply audio: {reply_path} "
        f"({len(reply)/mimi.sample_rate:.2f} s)")

    (OUT_DIR / "inner_monologue.txt").write_text(inner_text + "\n")

    # per-frame trace (compact) for auditing
    with open(OUT_DIR / "frame_trace.jsonl", "w") as ftr:
        for r in records:
            ftr.write(json.dumps(r) + "\n")

    # ---- machine-readable summary -----------------------------------------
    summary = {
        "peak_vram_gb": round(peak_smi_mb / 1024.0, 2),
        "peak_vram_torch_alloc_gb": round(peak_alloc_gb, 2),
        "frames_stepped": frames_stepped,
        "model_spoke_frames": model_spoke_frames,
        "model_spoke_frames_during_user_audio": spoke_during_user,
        "model_spoke_frames_during_trailing_silence": spoke_during_silence,
        "user_stream_frames": len(user_chunks),
        "trailing_silence_frames": n_silence,
        "first_model_audio_frame": first_model_speech_frame,
        "frame_rate_hz": mimi.frame_rate,
        "frame_ms": 1000.0 / mimi.frame_rate,
        "inner_monologue_text": inner_text,
        "uses_external_vad": False,
        "model": args.repo,
        "notes": (
            "Native full-duplex: a single autoregressive Moshi model models the "
            "user audio stream and its own audio stream in parallel and emits a "
            "time-aligned inner-monologue text stream. Turn-taking is decided "
            "inside the network at 12.5 Hz. No Silero/webrtcvad/energy gate and no "
            "turn-taking controller exist in hervoice/duplex/. 'first_model_audio_"
            "frame' = first frame the model's inner monologue emitted a real word, "
            "i.e. the frame the MODEL chose to start speaking. out_audio_rms in "
            "frame_trace.jsonl is descriptive logging only and gates nothing."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log("wrote summary.json")
    log(json.dumps(summary, indent=2))
    logf.close()


if __name__ == "__main__":
    main()
