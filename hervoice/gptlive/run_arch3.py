#!/usr/bin/env python3
"""run_arch3.py -- drive and MEASURE the four ARCH 3 behaviours end to end.

ARCH 3 = continuous full-duplex interaction with delegated reasoning: the open
local analog of GPT-Live's continuous mode. Moshi is the always-listening,
always-speaking front; gpt-oss-20B (open reasoning model, llama-server on :8093)
is the delegated brain. The front never blocks on the brain.

This runner produces, in runs/arch3/, real logged evidence for:
  1. FULL-DUPLEX OVERLAP  -- frames with user input energy AND assistant output
                             energy at the same time.
  2. ASYNC DELEGATION     -- gpt-oss runs on a background thread; Moshi
                             backchannels while it reasons, then vocalises the
                             streamed answer token by token.
  3. BARGE-IN             -- user energy during assistant speech aborts the
                             forced answer mid-utterance.
  4. GAP SWEEP            -- teacher-force pacing swept over gap in {1,2,3,4},
                             re-ASR intelligibility + duration measured.

GPU1 (A6000) ONLY. Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    .venv-duplex/bin/python -m hervoice.gptlive.run_arch3
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .front import MoshiFront, nvidia_smi_used_mb
from .duplex_session import DuplexSession
from . import delegate_oss

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "runs" / "arch3"
ASR_PY = REPO_ROOT / ".venv-hervoice" / "bin" / "python"
ASR_WORKER = Path(__file__).resolve().parent / "_asr_worker.py"

GPU_PHYS = 1  # physical A6000 index for nvidia-smi (CUDA_VISIBLE_DEVICES=1)

# overlap thresholds (documented; silence chunks sit near 1e-6)
U_THR = 0.02   # user input energy
A_THR = 0.010  # assistant output energy


def asr(wav_path, log) -> str:
    try:
        out = subprocess.check_output(
            [str(ASR_PY), str(ASR_WORKER), str(wav_path)],
            text=True, stderr=subprocess.DEVNULL, timeout=180)
        return out.strip()
    except Exception as e:  # noqa: BLE001
        log(f"[asr] FAILED on {wav_path}: {e}")
        return ""


def _norm_words(s):
    return [w for w in "".join(c.lower() if (c.isalnum() or c == " ") else " "
                               for c in s).split() if w]


def word_recall(asr_text, ref_text):
    ref = _norm_words(ref_text)
    got = set(_norm_words(asr_text))
    if not ref:
        return 0.0
    return round(sum(1 for w in ref if w in got) / len(ref), 3)


def cer(asr_text, ref_text):
    a = "".join(_norm_words(asr_text))
    b = "".join(_norm_words(ref_text))
    if not b:
        return 1.0
    # Levenshtein
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return round(prev[len(b)] / len(b), 3)


def overlay(timeline, chunks, start_frame, gain=1.0):
    """Add scaled user chunks onto a copy of the silence/question timeline."""
    tl = list(timeline)
    need = start_frame + len(chunks)
    # pad with silence chunks if needed
    while len(tl) < need:
        tl.append(torch.zeros_like(chunks[0]))
    for k, c in enumerate(chunks):
        tl[start_frame + k] = tl[start_frame + k] + gain * c
    return tl


def overlap_evidence(records):
    ov = [r for r in records
          if r["user_rms_in"] > U_THR and r["assistant_rms_out"] > A_THR]
    by_phase = {}
    for r in ov:
        by_phase[r["phase"]] = by_phase.get(r["phase"], 0) + 1
    return {
        "u_thr": U_THR, "a_thr": A_THR,
        "n_overlap_frames": len(ov),
        "overlap_frames_by_phase": by_phase,
        "sample_overlap_frames": [
            {k: r[k] for k in ("in_frame", "out_frame", "phase",
                               "user_rms_in", "assistant_rms_out", "piece")}
            for r in ov[:12]],
    }


def frames_to_ms(n):
    return round(n * 80.0, 1)   # 12.5 Hz -> 80 ms/frame of audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question",
                    default="What is the capital of France?")
    ap.add_argument("--question_wav",
                    default=str(REPO_ROOT / "examples" / "in_en_question.wav"))
    ap.add_argument("--bargein_question",
                    default="What is the capital of France?")
    ap.add_argument("--interrupt_wav",
                    default=str(REPO_ROOT / "examples" / "in_fifa_question.wav"))
    ap.add_argument("--gap", type=int, default=2)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(OUT_DIR / "run.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    results = {"arch": "ARCH 3 -- continuous full-duplex + delegated reasoning",
               "front": "Moshi (kyutai/moshiko-pytorch-bf16)",
               "delegate_brain": "gpt-oss-20B Q4_K_M (llama-server :8093)",
               "gpu": "RTX A6000 48GB (physical GPU1) only",
               "note": "open local analog of GPT-Live; delegate is gpt-oss-20B, NOT GPT-5.5"}
    vram = {}

    log("=== ARCH 3: continuous full-duplex with delegated reasoning ===")
    if not delegate_oss.brain_healthy():
        log("[brain] gpt-oss :8093 /health NOT ok. Start llama-server first. Abort.")
        logf.close(); raise SystemExit(1)
    vram["brain_only_mb"] = nvidia_smi_used_mb(GPU_PHYS)
    log(f"[brain] gpt-oss :8093 healthy. GPU1 used (brain only) = {vram['brain_only_mb']:.0f} MB")

    # -- load Moshi (co-resident with gpt-oss on the A6000) ----------------
    t = time.time()
    front = MoshiFront(log=log)
    log(f"[front] Moshi loaded in {time.time()-t:.1f}s")
    torch.cuda.reset_peak_memory_stats()
    vram["both_resident_mb"] = nvidia_smi_used_mb(GPU_PHYS)
    log(f"[vram] GPU1 used, Moshi + gpt-oss BOTH resident = {vram['both_resident_mb']:.0f} MB")

    sess = DuplexSession(front, log=log)
    q_chunks = front.load_user_wav(args.question_wav)
    n_prime = len(q_chunks) + 4
    log(f"[in] question wav -> {len(q_chunks)} frames; delegation fires at frame {n_prime}")

    # ==================================================================
    # SCENARIO A -- continuous async delegation + overlap
    # ==================================================================
    log("--- SCENARIO A: async streaming delegation (behaviours 1 & 2) ---")
    # overlay a real spoken utterance during the answer to force clear
    # listen-while-speaking overlap frames (barge-in OFF here).
    tlA = overlay(q_chunks, q_chunks, start_frame=n_prime + 22, gain=0.6)
    A = sess.run({
        "input_timeline": tlA,
        "n_prime_frames": n_prime,
        "brain_mode": "stream",
        "question": args.question,
        "gap": args.gap,
        "backchannel": "let me think.",
        "bargein": None,
        "tail_pad": 10,
        "max_frames": 1500,
    })
    front.write_wav(OUT_DIR / "scenarioA_session.wav", A["audio"])
    front.write_wav(OUT_DIR / "demo_arch3.wav", A["audio"])
    (OUT_DIR / "scenarioA_frames.json").write_text(json.dumps(A["records"], indent=1))
    vram["peak_after_A_mb"] = nvidia_smi_used_mb(GPU_PHYS)
    log(f"[A] brain answer = {A['brain']['full_answer']!r}")
    log(f"[A] Moshi forced-vocalised = {A['forced_answer_text']!r}")
    log(f"[A] markers = {A['markers']}")
    log(f"[A] reasoning deltas before first spoken word = {A['brain']['reasoning_deltas']}; "
        f"brain first-content latency = {A['brain']['first_content_latency_s']}s")
    log(f"[A] frames spoken while waiting = {A['frames_spoken_while_waiting']} "
        f"(~{frames_to_ms(A['frames_spoken_while_waiting'])} ms of audio)")
    A_overlap = overlap_evidence(A["records"])
    log(f"[A] OVERLAP frames (user_in>{U_THR} AND asst_out>{A_THR}) = "
        f"{A_overlap['n_overlap_frames']} by phase {A_overlap['overlap_frames_by_phase']}")
    A_asr = asr(OUT_DIR / "scenarioA_session.wav", log)
    log(f"[A] re-ASR of full session wav = {A_asr!r}")

    # async delegation timeline (in output frames; delegation start is the datum)
    m = A["markers"]
    dstart = m["delegation_start_frame"]
    # map input-frame markers to output frames via records
    def out_of(inframe):
        if inframe is None:
            return None
        for r in A["records"]:
            if r["in_frame"] >= inframe:
                return r["out_frame"]
        return None
    async_timeline = {
        "delegation_start_frame": dstart,
        "first_brain_token_frame": m["first_brain_token_frame"],
        "answer_complete_frame": m["answer_complete_frame"],
        "frames_spoken_while_waiting": A["frames_spoken_while_waiting"],
        "audio_ms_kept_alive_while_brain_reasoned":
            frames_to_ms(A["frames_spoken_while_waiting"]),
        "brain_reasoning_deltas_before_answer": A["brain"]["reasoning_deltas"],
        "brain_first_content_latency_s": A["brain"]["first_content_latency_s"],
    }

    # ==================================================================
    # SCENARIO B -- barge-in / interrupt
    # ==================================================================
    log("--- SCENARIO B: barge-in during assistant speech (behaviour 3) ---")
    int_chunks = front.load_user_wav(args.interrupt_wav)
    # A WIDE interruption overlay starting a little before the typical answer
    # onset; the detector is armed only during phase SPEAK (see engine), so it
    # fires on the first sustained user energy WHILE the answer is being spoken,
    # robust to the brain's run-to-run latency/length.
    interrupt_start = n_prime + 38
    tlB = overlay(q_chunks, int_chunks, start_frame=interrupt_start, gain=1.0)
    B = sess.run({
        "input_timeline": tlB,
        "n_prime_frames": n_prime,
        "brain_mode": "stream",
        "question": args.bargein_question,
        "gap": args.gap,
        "backchannel": "let me think.",
        "bargein": {"rms_thresh": 0.02, "sustain": 3, "arm_from_frame": 0},
        "tail_pad": 10,
        "max_frames": 1500,
    })
    # let the brain finish in the background so we can log the FULL answer that
    # gpt-oss would have spoken had Moshi not been cut off.
    if B.get("_brain_obj") is not None:
        B["_brain_obj"].join(timeout=5.0)
        B["brain"]["full_answer"] = B["_brain_obj"].full_answer
    front.write_wav(OUT_DIR / "scenarioB_bargein.wav", B["audio"])
    (OUT_DIR / "scenarioB_frames.json").write_text(json.dumps(B["records"], indent=1))
    (OUT_DIR / "bargein.log").write_text(json.dumps(B["markers"], indent=2))
    bm = B["markers"]
    log(f"[B] scripted interruption starts at input frame {interrupt_start}")
    log(f"[B] brain answer = {B['brain']['full_answer']!r}")
    log(f"[B] Moshi vocalised (before cutoff) = {B['forced_answer_text']!r}")
    log(f"[B] markers = {bm}")
    bargein_shown = bm["cutoff_frame"] is not None
    if bargein_shown:
        # confirm assistant was actually speaking just before the cutoff
        pre = [r for r in B["records"]
               if r["out_frame"] is not None and bm["cutoff_frame"] - 6 <= r["in_frame"] < bm["cutoff_frame"]]
        spoke_before = sum(1 for r in pre if r["assistant_rms_out"] > A_THR)
        post = [r for r in B["records"] if r["in_frame"] >= bm["cutoff_frame"]]
        log(f"[B] BARGE-IN detected+cutoff at frame {bm['cutoff_frame']} "
            f"(interrupt first seen frame {bm['interrupt_frame']}); "
            f"assistant speaking in {spoke_before}/{len(pre)} frames just before cutoff; "
            f"{len(post)} forced-yield frames after")
    else:
        log("[B] BARGE-IN NOT triggered (answer may have completed before the "
            "scripted interruption) -- reported honestly.")

    # ==================================================================
    # SCENARIO C -- gap sweep (behaviour 4)
    # ==================================================================
    log("--- SCENARIO C: gap sweep (behaviour 4) ---")
    sweep_answer = A["brain"]["full_answer"] or "The capital of France is Paris."
    N_TRIALS = 5   # Moshi's audio depformer samples at temp>0 -> average trials
    log(f"[C] sweeping gap with fixed answer text = {sweep_answer!r} "
        f"({N_TRIALS} trials/gap, depformer is stochastic)")
    sweep = []
    for g in (1, 2, 3, 4):
        trials = []
        audios = []
        for k in range(N_TRIALS):
            C = sess.run({
                "input_timeline": [],
                "n_prime_frames": 4,
                "brain_mode": "prefill",
                "prefill_text": sweep_answer,
                "gap": g,
                "tail_pad": 10,
                "max_frames": 1500,
            })
            wavp = OUT_DIR / f"sweep_gap{g}_t{k}.wav"
            front.write_wav(wavp, C["audio"])
            audios.append(C["audio"])
            rasr = asr(wavp, log)
            trials.append({
                "trial": k,
                "duration_s": C["duration_s"],
                "reasr_text": rasr,
                "word_recall": word_recall(rasr, sweep_answer),
                "cer": cer(rasr, sweep_answer),
            })
        mean_recall = round(float(np.mean([t["word_recall"] for t in trials])), 3)
        mean_cer = round(float(np.mean([t["cer"] for t in trials])), 3)
        mean_dur = round(float(np.mean([t["duration_s"] for t in trials])), 2)
        # keep a representative wav (best-recall trial) at the canonical name
        best_i = sorted(range(len(trials)),
                        key=lambda i: (-trials[i]["word_recall"], trials[i]["cer"]))[0]
        best_t = trials[best_i]
        front.write_wav(OUT_DIR / f"sweep_gap{g}.wav", audios[best_i])
        # balanced intelligibility score: reward answer words present, penalise
        # spurious/wrong characters (Moshi's greeting/echo inflates CER). A word
        # recall of 1 with CER 0 -> 1.0; big insertions push CER>1 -> negative.
        score = round(mean_recall - mean_cer, 3)
        row = {
            "gap": g,
            "mean_duration_s": mean_dur,
            "mean_word_recall_vs_answer": mean_recall,
            "mean_cer_vs_answer": mean_cer,
            "intelligibility_score": score,
            "trials": trials,
            "best_trial_reasr": best_t["reasr_text"],
        }
        sweep.append(row)
        log(f"[C] gap={g}: mean_recall={mean_recall} mean_cer={mean_cer} "
            f"score={score} mean_dur={mean_dur}s | best_asr={best_t['reasr_text']!r}")
    # recommend: best balanced intelligibility score, tie-break shorter duration
    best = sorted(sweep, key=lambda r: (-r["intelligibility_score"],
                                        r["mean_duration_s"]))[0]
    log(f"[C] RECOMMENDED gap = {best['gap']} "
        f"(score {best['intelligibility_score']}, recall "
        f"{best['mean_word_recall_vs_answer']}, cer {best['mean_cer_vs_answer']})")

    vram["peak_all_mb"] = nvidia_smi_used_mb(GPU_PHYS)
    try:
        vram["torch_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 0)
    except Exception:
        pass
    log(f"[vram] GPU1 peak (both resident, whole run) = {vram['peak_all_mb']:.0f} MB")

    # ==================================================================
    # RESULTS
    # ==================================================================
    results["behaviour_1_full_duplex_overlap"] = {
        "shown": A_overlap["n_overlap_frames"] > 0,
        "evidence": A_overlap,
        "explanation": ("frames where the input stream carries speech energy "
                        "AND Moshi simultaneously emits audio; native 12.5 Hz "
                        "duplex, no VAD gate."),
    }
    results["behaviour_2_async_streaming_delegation"] = {
        "shown": (A["markers"]["first_brain_token_frame"] is not None
                  and A["frames_spoken_while_waiting"] > 0),
        "timeline_output_frames": async_timeline,
        "brain_answer": A["brain"]["full_answer"],
        "moshi_forced_vocalised": A["forced_answer_text"],
        "reasr_full_session": A_asr,
        "explanation": ("gpt-oss ran on a background thread; Moshi backchanneled "
                        "for the frames_spoken_while_waiting frames while the brain "
                        "reasoned, then teacher-forced the streamed answer tokens "
                        "and vocalised them incrementally."),
    }
    results["behaviour_3_barge_in"] = {
        "shown": bargein_shown,
        "scripted_interrupt_input_frame": interrupt_start,
        "markers": bm,
        "rms_thresh": 0.02, "sustain_frames": 3,
        "moshi_vocalised_before_cutoff": B["forced_answer_text"],
        "brain_answer_full": B["brain"]["full_answer"],
        "explanation": ("energy-threshold detector armed during assistant "
                        "speech; on sustained user energy the forced answer is "
                        "aborted (force -> None) and Moshi yields. Heuristic, "
                        "not a trained interruption model."),
    }
    results["behaviour_4_gap_sweep"] = {
        "shown": True,
        "fixed_answer_text": sweep_answer,
        "table": sweep,
        "recommended_gap": best["gap"],
        "n_trials_per_gap": N_TRIALS,
        "metric": ("intelligibility_score = mean word_recall - mean CER over "
                   "trials (recall: answer words present, higher better; CER "
                   "proxy penalises Moshi's spurious/echo chars, lower better)"),
    }
    results["vram_mb"] = vram
    results["vram_peak_gb"] = round(vram["peak_all_mb"] / 1024.0, 2)
    results["reasr_verifications"] = {
        "scenarioA_session": A_asr,
        "gap_sweep": {r["gap"]: r["best_trial_reasr"] for r in sweep},
    }
    results["teacher_forcing_mechanism"] = (
        "Moshi is a full-duplex front; the delegated answer text is rendered "
        "back through Moshi by teacher-forcing its inner-monologue text stream "
        "(on_text_hook overwrites the sampled text token in place before the "
        "depformer vocalises it), so the acoustics are Moshi's but the WORDS are "
        "gpt-oss's. This is NOT Moshi autonomously deciding to say the answer.")
    results["honesty_caveats"] = [
        "Pacing is heuristic (gap PAD frames per word-piece); no learned "
        "alignment. Moshi's audio depformer samples at temp>0 so run-to-run "
        "variance / occasional echo is possible.",
        "moshiko's own inner monologue is its OWN unreliable reply, so the user "
        "turn is not read from it; behaviours use scripted input wavs.",
        "barge-in is an energy-threshold heuristic on the scripted input stream, "
        "not a trained interruption model.",
        "Naturalness needs human MOS; not claimed here.",
        "Open local analog of GPT-Live continuous mode; delegate is gpt-oss-20B "
        "(open weights), single A6000, English, file-driven (no live mic).",
    ]

    (REPO_ROOT / "hervoice" / "gptlive" / "results_arch3.json").write_text(
        json.dumps(results, indent=2))
    manifest = {
        "arch": results["arch"],
        "inputs": {
            "scenarioA_question": args.question,
            "scenarioA_question_wav": args.question_wav,
            "bargein_question": args.bargein_question,
            "interrupt_wav": args.interrupt_wav,
            "gap_default": args.gap,
        },
        "outputs": {
            "demo": "runs/arch3/demo_arch3.wav",
            "scenarioA_session": "runs/arch3/scenarioA_session.wav",
            "scenarioA_frames": "runs/arch3/scenarioA_frames.json",
            "scenarioB_bargein": "runs/arch3/scenarioB_bargein.wav",
            "scenarioB_frames": "runs/arch3/scenarioB_frames.json",
            "bargein_log": "runs/arch3/bargein.log",
            "gap_sweep_wavs": [f"runs/arch3/sweep_gap{g}.wav" for g in (1, 2, 3, 4)],
            "oss_server_log": "runs/arch3/oss_server.log",
            "run_log": "runs/arch3/run.log",
            "results": "hervoice/gptlive/results_arch3.json",
        },
        "vram_mb": vram,
        "vram_peak_gb": results["vram_peak_gb"],
        "behaviours_shown": {
            "1_overlap": results["behaviour_1_full_duplex_overlap"]["shown"],
            "2_async_delegation": results["behaviour_2_async_streaming_delegation"]["shown"],
            "3_barge_in": results["behaviour_3_barge_in"]["shown"],
            "4_gap_sweep": results["behaviour_4_gap_sweep"]["shown"],
        },
        "recommended_gap": best["gap"],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"[done] wrote results_arch3.json and {OUT_DIR/'manifest.json'}")
    logf.close()


if __name__ == "__main__":
    main()
