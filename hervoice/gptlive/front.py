#!/usr/bin/env python3
"""front.py -- the full-duplex Moshi FRONT for gptlive.

Reuses the exact working pattern from hervoice/duplex/prove_duplex.py: load the
Moshi 7B LM + the Mimi codec via the `moshi` package, and drive `LMGen.step`
frame-by-frame at 12.5 Hz (Mimi = 80 ms frames, 24 kHz).

Two capabilities on top of the bare duplex loop:

  1. LISTEN  -- feed the user's audio frame-by-frame and read back Moshi's own
     time-aligned inner-monologue TEXT stream and its audio stream. This is the
     always-on conversational layer (native turn-taking, no VAD).

  2. SPEAK (teacher-forced) -- the HARD part. Moshi normally speaks from text it
     SAMPLES itself. We instead make it vocalize an arbitrary string (the
     brain's delegated answer) by TEACHER-FORCING its text stream.

     Mechanism (public `moshi` API, no monkey-patching of internals): LMGen
     exposes `on_text_hook(text_token)`, called every frame with the freshly
     sampled text token (shape [B]) *before* the depformer turns it into audio
     tokens and *before* it is written back into the KV cache
     (moshi/models/lm.py `_step`, lines ~736-763). Because the token is a live
     tensor, the hook can overwrite it in place -- `text_token[:] = forced_id` --
     and the depformer then generates the acoustic tokens for OUR token, and the
     forced token is what enters the autoregressive context. That is exactly
     teacher-forcing Moshi's inner monologue. This is the same principle the
     delayed-streams TTS models use.

     Timing: Moshi's text runs at 12.5 Hz with a "pad" token (id 0) held between
     word-pieces while a word is acoustically realised. We approximate that by
     emitting each answer word-piece then `gap` pad frames. There is no learned
     alignment model here, so pacing is heuristic -- see README for honesty.

GPU: GPU0 only. Callers must set CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0.
"""
import subprocess
import time
from pathlib import Path

import numpy as np
import sphn
import torch

from moshi.models import LMGen, loaders

REPO_ROOT = Path(__file__).resolve().parents[2]
PAD_TOKEN = 0
SPECIAL_TOKENS = {0, 3}


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


class MoshiFront:
    """Loads Moshi + Mimi once, then runs full-duplex turns."""

    def __init__(self, repo: str = loaders.DEFAULT_REPO, log=print):
        self.log = log
        device = "cuda"
        assert torch.cuda.is_available(), "CUDA not available"
        torch.cuda.reset_peak_memory_stats()
        log(f"[front] device0 = {torch.cuda.get_device_name(0)}")
        log(f"[front] repo = {repo}")
        ckpt = loaders.CheckpointInfo.from_hf_repo(repo)
        self.mimi = ckpt.get_mimi(device=device)
        self.text_tokenizer = ckpt.get_text_tokenizer()
        self.lm = ckpt.get_moshi(device=device, dtype=torch.bfloat16)
        self.lm_gen_config = ckpt.lm_gen_config
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.sample_rate = self.mimi.sample_rate
        self.frame_rate = self.mimi.frame_rate
        self.vram_after_load_mb = nvidia_smi_used_mb(0)
        log(f"[front] sample_rate={self.sample_rate} frame_rate={self.frame_rate}Hz "
            f"frame_size={self.frame_size} dep_q={self.lm.dep_q}")
        log(f"[front] nvidia-smi used after Moshi load (GPU0) = "
            f"{self.vram_after_load_mb:.0f} MB")

    def load_user_wav(self, path: str):
        pcms, _ = sphn.read(path, sample_rate=self.sample_rate)  # resample to 24k
        pcms = torch.from_numpy(pcms).to(device=self.device)[None, 0:1]
        chunks = [c for c in pcms.split(self.frame_size, dim=2)
                  if c.shape[-1] == self.frame_size]
        return chunks

    def id_to_piece_text(self, ids):
        return "".join(
            self.text_tokenizer.id_to_piece(t).replace("▁", " ")
            for t in ids
        ).strip()

    def build_speak_schedule(self, answer_text: str, gap: int = 2,
                             lead_pad: int = 4, tail_pad: int = 8):
        """Return a per-frame list of forced text-token ids for the SPEAK phase.

        Each answer word-piece is followed by `gap` PAD frames so Moshi has time
        to acoustically realise the word. `lead_pad` PAD frames prime the speak
        phase; `tail_pad` PAD frames flush the acoustic delay at the end.
        """
        ids = self.text_tokenizer.encode(answer_text)
        sched = [PAD_TOKEN] * lead_pad
        for tid in ids:
            sched.append(tid)
            sched.extend([PAD_TOKEN] * gap)
        sched.extend([PAD_TOKEN] * tail_pad)
        return ids, sched

    @torch.no_grad()
    def run_turn(self, user_chunks, speak_schedule=None, listen_tail_frames: int = 12):
        """Run one full-duplex turn in a single streaming session.

        Phases:
          LISTEN  : feed user_chunks (Moshi hears the question), then
                    `listen_tail_frames` silence frames so its own inner
                    monologue can react. Moshi runs FREE here (no forcing).
          SPEAK   : if `speak_schedule` given, feed silence frames while
                    TEACHER-FORCING the text stream to `speak_schedule`.

        Returns dict with inner-monologue text (listen phase, Moshi's own),
        forced pieces actually vocalised, output audio for each phase, records.
        """
        lm_gen_kwargs = dict(self.lm_gen_config)

        # forcing control shared with the hook; set per-frame by the main loop.
        ctl = {"force_id": None, "forced_log": []}

        def on_text_hook(text_token):
            fid = ctl["force_id"]
            if fid is not None:
                text_token[:] = fid  # teacher-force Moshi's inner monologue
                ctl["forced_log"].append(fid)

        lm_gen = LMGen(self.lm, on_text_hook=on_text_hook, **lm_gen_kwargs)

        silence = torch.zeros((1, self.mimi.channels, self.frame_size),
                              device=self.device)

        listen_frames = len(user_chunks) + listen_tail_frames
        speak = speak_schedule or []

        # Build the full input plan: (phase, input_audio_chunk, force_id)
        plan = []
        for i in range(len(user_chunks)):
            plan.append(("USER", user_chunks[i], None))
        for _ in range(listen_tail_frames):
            plan.append(("LISTEN_SIL", silence, None))
        for fid in speak:
            plan.append(("SPEAK", silence, fid))

        listen_audio, speak_audio = [], []
        listen_inner_ids, listen_records = [], []
        speak_pieces = []

        t0 = time.time()
        first_frame = True
        step_idx = -1
        with self.mimi.streaming(1), lm_gen.streaming(1):
            for phase, audio_chunk, force_id in plan:
                codes = self.mimi.encode(audio_chunk)
                ctl["force_id"] = force_id  # hook reads this for THIS step
                if first_frame:
                    _ = lm_gen.step(codes)  # warm-up (hook may fire; harmless)
                    first_frame = False
                tokens = lm_gen.step(codes)
                if tokens is None:
                    continue  # inside output-delay window
                step_idx += 1
                text_tok = int(tokens[0, 0].item())
                wav = self.mimi.decode(tokens[:, 1:])[0, 0].float().cpu().numpy()
                rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-12)

                if phase == "SPEAK":
                    speak_audio.append(wav)
                    if text_tok not in SPECIAL_TOKENS:
                        speak_pieces.append(text_tok)
                else:
                    listen_audio.append(wav)
                    if text_tok not in SPECIAL_TOKENS:
                        listen_inner_ids.append(text_tok)
                    listen_records.append(
                        {"frame": step_idx, "phase": phase,
                         "text_token": text_tok, "rms": rms})
        dt = time.time() - t0

        return {
            "listen_inner_text": self.id_to_piece_text(listen_inner_ids),
            "speak_forced_text": self.id_to_piece_text(speak_pieces),
            "listen_audio": (np.concatenate(listen_audio)
                             if listen_audio else np.zeros(1, dtype=np.float32)),
            "speak_audio": (np.concatenate(speak_audio)
                            if speak_audio else np.zeros(1, dtype=np.float32)),
            "n_forced_frames": len(ctl["forced_log"]),
            "listen_records": listen_records,
            "wall_seconds": dt,
            "frames_stepped": step_idx + 1,
        }

    def write_wav(self, path, audio):
        sphn.write_wav(str(path), audio[None, :], sample_rate=self.sample_rate)
