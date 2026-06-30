#!/usr/bin/env python3
"""ADVERSARIAL stress harness for the HerVoice live English voice loop.

Loads ONE resident MiniCPM-o 4.5 engine (GPU0) and hammers the LiveLoop /
LiveVoiceEngine with endurance, barge-in edge cases, malformed input, and race
scenarios. The goal is to FIND BUGS, not to prove green. Everything is logged to
runs/live_stress/ and machine-readable findings go to runs/live_stress/bugs.json.

Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -m hervoice.live.stress_test --turns 14
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import librosa
import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "runs", "live_stress")
os.makedirs(OUT, exist_ok=True)

LOG_PATH = os.path.join(OUT, "stress.log")
_logf = open(LOG_PATH, "a", buffering=1)


def log(*a):
    msg = " ".join(str(x) for x in a)
    line = f"[{time.time():.3f}] {msg}"
    print(line, flush=True)
    _logf.write(line + "\n")


def nvidia_used_mb(gpu=0):
    """GPU0 *process-external-inclusive* memory (ASR/CT2/token2wav live outside torch)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(gpu)], text=True, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception as e:
        return -1


def thread_snapshot():
    ts = threading.enumerate()
    return {"count": len(ts), "names": sorted(t.name for t in ts)}


# --------------------------------------------------------------------------- #
#  Loop driver: runs a LiveLoop in a thread, scripts frames, collects events.
# --------------------------------------------------------------------------- #
class LoopRunner:
    def __init__(self, engine, detector_factory, mode="plain", max_new_tokens=96,
                 barge_guard_chunks=1):
        from .loop import LiveLoop
        self.engine = engine
        self.detector = detector_factory()
        self.events = []
        self.ev_lock = threading.Lock()
        self.turn_audio = {}

        def on_event(evt):
            with self.ev_lock:
                self.events.append(evt)

        def on_audio(turn_idx, audio):
            self.turn_audio.setdefault(turn_idx, []).append(np.asarray(audio, dtype=np.float32))

        self.loop = LiveLoop(engine, self.detector, on_event=on_event, on_audio=on_audio,
                             mode=mode, response_max_new_tokens=max_new_tokens,
                             barge_guard_chunks=barge_guard_chunks)
        self.frame = int(16000 * 0.05)  # 50ms frames
        self._thr = None

    def start(self):
        self._thr = threading.Thread(target=self.loop.run, name="loop-run", daemon=True)
        self._thr.start()

    def feed(self, audio, realtime=False, frame_ms=50):
        frame = int(16000 * frame_ms / 1000)
        for i in range(0, len(audio), frame):
            self.loop.submit_frame(audio[i:i + frame])
            if realtime:
                time.sleep(frame_ms / 1000.0)

    def feed_silence(self, n, frame_ms=50, realtime=False):
        f = np.zeros(int(16000 * frame_ms / 1000), dtype=np.float32)
        for _ in range(n):
            self.loop.submit_frame(f.copy())
            if realtime:
                time.sleep(frame_ms / 1000.0)

    def wait(self, pred, timeout, poll=0.01):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if pred():
                return True
            time.sleep(poll)
        return False

    def stop_and_join(self, timeout=60):
        self.loop.stop()
        self._thr.join(timeout=timeout)
        return not self._thr.is_alive()

    def text_for(self, turn):
        with self.ev_lock:
            return "".join(e.get("text", "") for e in self.events
                           if e.get("type") == "text_delta" and e.get("turn") == turn)

    def find(self, etype, **m):
        with self.ev_lock:
            for e in self.events:
                if e["type"] == etype and all(e.get(k) == v for k, v in m.items()):
                    return e
        return None

    def all(self, etype, **m):
        with self.ev_lock:
            return [e for e in self.events if e["type"] == etype
                    and all(e.get(k) == v for k, v in m.items())]


# --------------------------------------------------------------------------- #
BUGS = []


def bug(severity, scenario, observed, expected, repro, evidence, fix):
    b = {"severity": severity, "scenario": scenario, "observed": observed,
         "expected": expected, "reproduction": repro, "evidence": evidence,
         "proposed_fix": fix}
    BUGS.append(b)
    log(f"  *** BUG [{severity}] {scenario}: {observed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=14)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--ref", default=os.path.join(REPO, "examples/ref_female.wav"))
    ap.add_argument("--scenarios", default="all",
                    help="comma list: endurance,bargein,malformed,race,all")
    args = ap.parse_args()

    sel = set(args.scenarios.split(",")) if args.scenarios != "all" else {
        "endurance", "bargein", "malformed", "race", "shortchunk"}

    import torch
    from .engine import LiveVoiceEngine
    from .turn_detector import TurnDetector

    log("=" * 70)
    log(f"STRESS START  scenarios={sel}  turns={args.turns}  mnt={args.max_new_tokens}")
    log(f"GPU0 used before model load: {nvidia_used_mb()} MB")

    boot_threads = thread_snapshot()
    log(f"boot threads: {boot_threads}")

    t0 = time.time()
    engine = LiveVoiceEngine(ref_audio_path=args.ref, quant="int4",
                             response_max_new_tokens=args.max_new_tokens)
    log(f"engine loaded in {time.time()-t0:.1f}s  GPU0 used now: {nvidia_used_mb()} MB")

    # one shared VAD model -> cheap detector factory
    from silero_vad import load_silero_vad
    vad_model = load_silero_vad()

    def det():
        return TurnDetector(threshold=0.5, min_speech_ms=120, min_silence_ms=220,
                            model=vad_model)

    # inputs
    q_en, _ = librosa.load(os.path.join(REPO, "examples/in_en_question.wav"), sr=16000, mono=True)
    q_fifa, _ = librosa.load(os.path.join(REPO, "examples/in_fifa_question.wav"), sr=16000, mono=True)

    # synth edge-case wavs
    rng = np.random.default_rng(0)
    edge = {
        "empty": np.zeros(0, dtype=np.float32),
        "silence_2s": np.zeros(16000 * 2, dtype=np.float32),
        "clip_0p2s": q_en[:int(16000 * 0.2)].copy(),
        "white_noise": (rng.standard_normal(16000 * 2).astype(np.float32) * 0.1),
        "tone_440": (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000 * 2) / 16000)).astype(np.float32),
        "long_60s": np.tile(q_en, int(np.ceil(60 * 16000 / len(q_en))))[:60 * 16000].astype(np.float32),
    }
    edir = os.path.join(OUT, "edge_wavs")
    os.makedirs(edir, exist_ok=True)
    for k, v in edge.items():
        if v.size:
            sf.write(os.path.join(edir, f"{k}.wav"), v, 16000)

    report = {"vram_mb_over_turns": [], "scenarios": {}}

    # ===================================================================== #
    # SCENARIO 1: ENDURANCE  (one process, many turns, same/changing prompt)
    # ===================================================================== #
    if "endurance" in sel:
        log("\n" + "#" * 60 + "\n# SCENARIO 1: ENDURANCE\n" + "#" * 60)
        runner = LoopRunner(engine, det, mode="plain", max_new_tokens=args.max_new_tokens)
        runner.start()
        per_turn = []
        # alternate: 3x SAME prompt (contamination), then alternate changing prompts
        plan = []
        for i in range(args.turns):
            if i < 3:
                plan.append(("same", q_en))
            else:
                plan.append(("alt", q_en if i % 2 == 0 else q_fifa))
        prev_used = nvidia_used_mb()
        t_first = None
        for i, (kind, audio) in enumerate(plan, 1):
            tstart = time.time()
            torch.cuda.reset_peak_memory_stats()
            target_turn = len(runner.loop.turns) + 1
            runner.feed(audio)
            runner.feed_silence(14)  # drive end-of-turn
            ok_gen = runner.wait(
                lambda tt=target_turn: any(t["turn"] == tt for t in runner.loop.turns)
                and runner.loop.state == "GENERATING", timeout=60)
            # drive to natural completion: keep feeding silence so _on_frame can finish
            done = runner.wait(lambda: runner.loop._gen_done.is_set(), timeout=90)
            runner.feed_silence(3)  # let _finish_turn_natural run
            runner.wait(lambda: runner.loop.state == "IDLE", timeout=10)
            dt = time.time() - tstart
            used = nvidia_used_mb()
            torchmax = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            txt = runner.text_for(target_turn)
            chunks = len(runner.turn_audio.get(target_turn, []))
            thr = thread_snapshot()
            rec = {"turn": i, "kind": kind, "secs": round(dt, 1), "nvsmi_used_mb": used,
                   "delta_mb": used - prev_used, "torch_peak_gb": torchmax,
                   "audio_chunks": chunks, "threads": thr["count"],
                   "text": txt[:160], "started_gen": ok_gen, "natural_done": done}
            per_turn.append(rec)
            report["vram_mb_over_turns"].append(used)
            log(f"  turn {i:2d} [{kind}] {dt:5.1f}s used={used}MB d={used-prev_used:+d} "
                f"torch={torchmax}GB chunks={chunks} thr={thr['count']} txt={txt[:70]!r}")
            prev_used = used
        runner.stop_and_join()

        # ---- analyze endurance ----
        report["scenarios"]["endurance"] = per_turn
        used_seq = [r["nvsmi_used_mb"] for r in per_turn if r["nvsmi_used_mb"] > 0]
        # leak heuristic: compare last-3 avg vs turns 2-4 avg (skip warmup turn1)
        if len(used_seq) >= 6:
            base = np.mean(used_seq[1:4])
            tail = np.mean(used_seq[-3:])
            growth = tail - base
            log(f"  VRAM base(t2-4)={base:.0f}MB tail={tail:.0f}MB growth={growth:+.0f}MB")
            report["scenarios"]["endurance_vram"] = {
                "base_mb": round(base, 0), "tail_mb": round(tail, 0), "growth_mb": round(growth, 0)}
            if growth > 800:
                bug("HIGH", "endurance-leak",
                    f"GPU0 VRAM grew {growth:.0f}MB across {len(used_seq)} turns (base {base:.0f}->tail {tail:.0f})",
                    "flat VRAM across turns (each turn resets the session)",
                    "python -m hervoice.live.stress_test --scenarios endurance --turns 14",
                    {"used_mb_seq": used_seq}, "investigate session/cache not freed per turn")
        # thread leak
        thr_seq = [r["threads"] for r in per_turn]
        if thr_seq and max(thr_seq) - thr_seq[0] > 2:
            bug("HIGH", "endurance-threads",
                f"thread count grew {thr_seq[0]}->{max(thr_seq)} across turns",
                "thread count stable per turn (worker joins each turn)",
                "python -m hervoice.live.stress_test --scenarios endurance",
                {"thread_seq": thr_seq}, "join gen worker; ensure no zombie accumulation")
        # contamination: first 3 same-prompt turns should be similar; alt should differ
        same_txts = [r["text"] for r in per_turn[:3]]
        log(f"  same-prompt turn texts: {same_txts}")
        report["scenarios"]["contamination_same"] = same_txts
        # slowdown
        secs = [r["secs"] for r in per_turn]
        if len(secs) >= 6 and np.mean(secs[-3:]) > 1.5 * np.mean(secs[1:4]):
            bug("MEDIUM", "endurance-slowdown",
                f"per-turn time grew {np.mean(secs[1:4]):.1f}s -> {np.mean(secs[-3:]):.1f}s",
                "stable per-turn latency", "stress_test --scenarios endurance",
                {"secs": secs}, "profile growth")

    # ===================================================================== #
    # SCENARIO 2: BARGE-IN EDGE CASES
    # ===================================================================== #
    if "bargein" in sel:
        log("\n" + "#" * 60 + "\n# SCENARIO 2: BARGE-IN EDGE CASES\n" + "#" * 60)
        bres = {}

        # --- 2a: barge-in at t=0 (before any audio emitted; guard window) -----
        log("  [2a] barge-in during guard window (before first audio chunk)")
        r = LoopRunner(engine, det, mode="plain", max_new_tokens=args.max_new_tokens,
                       barge_guard_chunks=1)
        r.start()
        r.feed(q_en)
        r.feed_silence(14)
        r.wait(lambda: runner_state(r) == "GENERATING", timeout=60)
        # inject barge-in IMMEDIATELY (likely before first audio chunk) and keep talking
        thr_before = thread_snapshot()
        r.feed(q_fifa)            # user starts talking right at gen start
        r.feed_silence(14)
        r.wait(lambda: r.loop._gen_done.is_set(), timeout=90)
        r.feed_silence(3)
        r.wait(lambda: runner_state(r) in ("IDLE", "GENERATING"), timeout=20)
        # drain a possible second turn
        r.wait(lambda: r.loop._gen_done.is_set(), timeout=60)
        r.feed_silence(3)
        time.sleep(1.0)
        thr_after = thread_snapshot()
        barge = r.find("barge_in")
        nturns = len(r.loop.turns)
        gc_at = r.all("first_audio")
        bres["2a_guard_window"] = {"barge_fired": barge is not None, "n_turns": nturns,
                                   "thr_before": thr_before["count"], "thr_after": thr_after["count"]}
        log(f"    barge_fired={barge is not None} n_turns={nturns} "
            f"thr {thr_before['count']}->{thr_after['count']}")
        r.stop_and_join()
        # Known suspicion: if user speaks during guard window, VAD latches triggered=True
        # and the barge-in for that whole utterance is dropped.
        if barge is None and nturns >= 1:
            bug("MEDIUM", "bargein-guard-latch",
                "User speech that STARTS during the barge-guard window (before the 1st "
                "audio chunk) never triggers barge-in for the entire utterance: VAD latches "
                "triggered=True on the dropped speech_start, so no further speech_start fires.",
                "barge-in should fire as soon as guard clears, OR the latched onset should "
                "be re-evaluated when _gen_chunks reaches the guard.",
                "stress_test --scenarios bargein",
                {"barge_fired": False, "n_turns": nturns},
                "On a dropped (guarded) speech_start, do NOT leave the detector latched; or "
                "re-check pending onset each frame while GENERATING once guard clears.")

        # --- 2b: two barge-ins back-to-back ----------------------------------
        log("  [2b] two barge-ins back-to-back")
        r = LoopRunner(engine, det, mode="plain", max_new_tokens=args.max_new_tokens)
        r.start()
        r.feed(q_en); r.feed_silence(14)
        r.wait(lambda: r.loop._gen_chunks >= 1, timeout=60)
        r.feed(q_fifa)            # barge 1
        r.wait(lambda: len(r.loop.turns) >= 2 and r.loop.state == "USER_SPEAKING", timeout=30)
        r.feed_silence(14)
        r.wait(lambda: r.loop._gen_chunks >= 1 and r.loop.state == "GENERATING", timeout=60)
        r.feed(q_en)             # barge 2
        r.wait(lambda: len(r.loop.turns) >= 3, timeout=30)
        r.feed_silence(14)
        r.wait(lambda: r.loop._gen_done.is_set(), timeout=90)
        r.feed_silence(3)
        time.sleep(0.5)
        barges = r.all("barge_in")
        sids = [t["session_id"] for t in r.loop.turns]
        bres["2b_double_barge"] = {"n_barges": len(barges), "n_turns": len(r.loop.turns),
                                   "unique_sids": len(set(sids)), "sids": sids}
        log(f"    barges={len(barges)} turns={len(r.loop.turns)} uniq_sids={len(set(sids))}")
        if len(set(sids)) != len(sids):
            bug("HIGH", "bargein-dup-sid", f"session_ids not unique across turns: {sids}",
                "every turn gets a fresh session_id",
                "stress_test --scenarios bargein", {"sids": sids}, "new uuid per turn")
        r.stop_and_join()

        # --- 2c: barge-in then immediate silence -----------------------------
        log("  [2c] barge-in then immediate silence (no real follow-up speech)")
        r = LoopRunner(engine, det, mode="plain", max_new_tokens=args.max_new_tokens)
        r.start()
        r.feed(q_en); r.feed_silence(14)
        r.wait(lambda: r.loop._gen_chunks >= 1, timeout=60)
        r.feed(q_fifa[:int(16000 * 0.4)])   # brief barge then silence
        r.feed_silence(20)
        ok = r.wait(lambda: r.loop._gen_done.is_set(), timeout=60)
        r.feed_silence(3)
        time.sleep(0.5)
        bres["2c_barge_then_silence"] = {"recovered": ok, "state": runner_state(r),
                                         "n_turns": len(r.loop.turns)}
        log(f"    recovered={ok} state={runner_state(r)} turns={len(r.loop.turns)}")
        r.stop_and_join()

        # --- 2d: many rapid barge-ins (zombie thread hunt) -------------------
        log("  [2d] many rapid barge-ins -- zombie thread hunt")
        r = LoopRunner(engine, det, mode="plain", max_new_tokens=args.max_new_tokens)
        r.start()
        thr0 = thread_snapshot()["count"]
        r.feed(q_en); r.feed_silence(14)
        peak_thr = thr0
        for k in range(5):
            if not r.wait(lambda: r.loop._gen_chunks >= 1 and r.loop.state == "GENERATING",
                          timeout=60):
                break
            r.feed(q_fifa if k % 2 == 0 else q_en)  # barge
            r.wait(lambda kk=k: len(r.loop.turns) >= kk + 2, timeout=30)
            peak_thr = max(peak_thr, thread_snapshot()["count"])
            r.feed_silence(14)
        r.wait(lambda: r.loop._gen_done.is_set(), timeout=90)
        r.feed_silence(3)
        time.sleep(1.5)
        thr_end = thread_snapshot()["count"]
        live_gen = [t.name for t in threading.enumerate() if "loop-run" not in t.name]
        bres["2d_rapid_barge"] = {"thr0": thr0, "peak": peak_thr, "end": thr_end,
                                  "n_turns": len(r.loop.turns)}
        log(f"    thr0={thr0} peak={peak_thr} end={thr_end} turns={len(r.loop.turns)}")
        if thr_end - thr0 > 1:
            bug("HIGH", "bargein-zombie-threads",
                f"thread count did not return to baseline after rapid barge-ins: {thr0}->{thr_end}",
                "all generation worker threads join after each barge-in",
                "stress_test --scenarios bargein",
                {"thr0": thr0, "peak": peak_thr, "end": thr_end},
                "ensure _join_generation actually joins; investigate 10s join timeout zombies")
        r.stop_and_join()

        report["scenarios"]["bargein"] = bres

    # ===================================================================== #
    # SCENARIO 3: MALFORMED INPUT
    # ===================================================================== #
    if "malformed" in sel:
        log("\n" + "#" * 60 + "\n# SCENARIO 3: MALFORMED INPUT\n" + "#" * 60)
        mres = {}
        for name in ["silence_2s", "clip_0p2s", "white_noise", "tone_440", "long_60s"]:
            audio = edge[name]
            log(f"  [3] feeding {name} ({len(audio)/16000:.2f}s)")
            r = LoopRunner(engine, det, mode="plain", max_new_tokens=args.max_new_tokens)
            r.start()
            crashed = None
            t0 = time.time()
            try:
                r.feed(audio)
                r.feed_silence(16)
                # bounded wait: a turn either starts quickly or never (non-speech)
                started = r.wait(lambda: len(r.loop.turns) > 0, timeout=8)
                if started:
                    r.wait(lambda: r.loop.state == "GENERATING", timeout=30)
                    r.wait(lambda: r.loop._gen_done.is_set(), timeout=90)
                    r.feed_silence(3)
                    r.wait(lambda: r.loop.state == "IDLE", timeout=10)
            except Exception as e:
                crashed = str(e)
            dt = round(time.time() - t0, 1)
            gerr = r.find("generation_error")
            n_turns = len(r.loop.turns)
            txt = r.text_for(n_turns) if n_turns else ""
            mres[name] = {"n_turns": n_turns, "secs": dt, "crashed": crashed,
                          "gen_error": gerr.get("error") if gerr else None,
                          "text": txt[:120]}
            log(f"    {name}: turns={n_turns} secs={dt} err={gerr.get('error') if gerr else None} "
                f"txt={txt[:60]!r}")
            joined = r.stop_and_join(timeout=30)
            if not joined:
                bug("HIGH", f"malformed-hang-{name}",
                    f"loop did not stop within timeout after feeding {name}",
                    "loop drains and stops", "stress_test --scenarios malformed",
                    {"name": name}, "add bounded wait / timeout on generation")
            if crashed:
                bug("HIGH", f"malformed-crash-{name}",
                    f"feeding {name} raised: {crashed}", "graceful handling, no crash",
                    "stress_test --scenarios malformed", {"err": crashed},
                    "validate/clamp input length")
            if gerr:
                bug("MEDIUM", f"malformed-generror-{name}",
                    f"generation_error on {name}: {gerr.get('error')}",
                    "no exception in worker for benign input",
                    "stress_test --scenarios malformed", {"err": gerr.get("error")},
                    "guard generate path")
        # pure silence / tone should NOT fabricate a spoken turn
        for nonspeech in ["silence_2s", "tone_440"]:
            if mres.get(nonspeech, {}).get("n_turns", 0) > 0:
                bug("MEDIUM", f"malformed-fabricated-{nonspeech}",
                    f"{nonspeech} (no human speech) fabricated {mres[nonspeech]['n_turns']} turn(s) "
                    f"-> text={mres[nonspeech]['text']!r}",
                    "VAD should not trigger a turn on non-speech; no fabricated answer",
                    "stress_test --scenarios malformed",
                    {nonspeech: mres[nonspeech]}, "raise VAD threshold / require speech energy")
        report["scenarios"]["malformed"] = mres

    # ===================================================================== #
    # SCENARIO 4: CANCELLATION / RACE
    # ===================================================================== #
    if "race" in sel:
        log("\n" + "#" * 60 + "\n# SCENARIO 4: CANCELLATION / RACE\n" + "#" * 60)
        rres = {}
        # mid-chunk cancel -> does NEXT turn's audio come out corrupted/silent/wrong-length?
        log("  [4a] mid-chunk cancel then measure next-turn audio health")
        r = LoopRunner(engine, det, mode="plain", max_new_tokens=args.max_new_tokens)
        r.start()
        # baseline clean turn for length reference
        r.feed(q_en); r.feed_silence(14)
        r.wait(lambda: r.loop.state == "GENERATING", timeout=60)
        r.wait(lambda: r.loop._gen_done.is_set(), timeout=90)
        r.feed_silence(3); r.wait(lambda: r.loop.state == "IDLE", timeout=10)
        clean_chunks = len(r.turn_audio.get(1, []))
        clean_samps = sum(len(a) for a in r.turn_audio.get(1, []))
        # now a turn that gets barged mid-chunk, then a clean follow-up turn
        r.feed(q_en); r.feed_silence(14)
        r.wait(lambda: r.loop._gen_chunks >= 1, timeout=60)
        r.feed(q_en)  # barge mid-chunk
        r.wait(lambda: len(r.loop.turns) >= 3 and r.loop.state == "USER_SPEAKING", timeout=30)
        r.feed_silence(14)
        r.wait(lambda: r.loop._gen_done.is_set(), timeout=90)
        r.feed_silence(3); r.wait(lambda: r.loop.state == "IDLE", timeout=10)
        post = len(r.loop.turns)
        post_chunks = len(r.turn_audio.get(post, []))
        post_samps = sum(len(a) for a in r.turn_audio.get(post, []))
        # health: non-empty, finite, not absurd length vs clean
        arr = np.concatenate(r.turn_audio[post]) if r.turn_audio.get(post) else np.zeros(0)
        health = {"clean_chunks": clean_chunks, "clean_samps": clean_samps,
                  "post_chunks": post_chunks, "post_samps": post_samps,
                  "post_finite": bool(np.all(np.isfinite(arr))) if arr.size else None,
                  "post_max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0}
        log(f"    {health}")
        rres["4a_post_cancel_health"] = health
        if arr.size and (not np.all(np.isfinite(arr))):
            bug("HIGH", "race-postcancel-nan", "post-barge-in turn audio contains NaN/Inf",
                "clean finite audio after a mid-chunk cancel",
                "stress_test --scenarios race", health, "reset token2wav state after cancel")
        if post_chunks == 0:
            bug("HIGH", "race-postcancel-silent",
                "post-barge-in turn produced ZERO audio (silent)",
                "follow-up turn after a barge-in still speaks",
                "stress_test --scenarios race", health,
                "investigate token2wav/session state after mid-chunk GeneratorExit")
        r.stop_and_join()
        report["scenarios"]["race"] = rres

    # ===================================================================== #
    # SCENARIO 5: SHORT FINAL CHUNK (regression for avg_pool1d encoder crash)
    # ===================================================================== #
    if "shortchunk" in sel:
        log("\n" + "#" * 60 + "\n# SCENARIO 5: SHORT FINAL CHUNK (encoder crash regression)\n" + "#" * 60)
        sres = {}
        # craft a user turn whose total length leaves a tiny (<1024 sample) leftover
        # after 1s-aligned chunking: 1*16000 + 256 samples of real speech.
        speech = q_en[:16000 + 256].copy() if len(q_en) >= 16000 + 256 else \
            np.concatenate([q_en, q_en])[:16000 + 256]
        r = LoopRunner(engine, det, mode="plain", max_new_tokens=args.max_new_tokens)
        r.start()
        r.feed(speech)
        r.feed_silence(16)
        # the loop thread must survive; a turn should still be produced
        started = r.wait(lambda: len(r.loop.turns) > 0, timeout=15)
        done = r.wait(lambda: r.loop._gen_done.is_set(), timeout=90)
        r.feed_silence(3)
        r.wait(lambda: r.loop.state == "IDLE", timeout=10)
        loop_alive = r._thr.is_alive()
        ferr = r.find("frame_error")
        n_turns = len(r.loop.turns)
        # feed a SECOND turn to prove the loop is still usable after the short chunk
        r.feed(q_en)
        r.feed_silence(16)
        second = r.wait(lambda: len(r.loop.turns) >= n_turns + 1, timeout=20)
        sres = {"loop_alive": loop_alive, "frame_error": ferr.get("error") if ferr else None,
                "first_turn_started": started, "first_turn_done": done,
                "second_turn_accepted": second, "n_turns": len(r.loop.turns)}
        log(f"    {sres}")
        report["scenarios"]["shortchunk"] = sres
        if not loop_alive:
            bug("HIGH", "shortchunk-loop-death",
                "consumer loop thread DIED after a short (<1024 sample) final user "
                "chunk: streaming audio encoder avg_pool1d raised and was uncaught, "
                "permanently killing the loop (frames silently swallowed thereafter).",
                "loop survives any chunk length; turn still produced",
                "stress_test --scenarios shortchunk",
                sres, "pad short prefill chunks to >=1024 samples in "
                      "engine.prefill_user_chunk AND guard loop.run's _on_frame.")
        elif ferr:
            bug("MEDIUM", "shortchunk-frame-error",
                f"short final chunk raised but loop recovered: {ferr.get('error')}",
                "no exception for a normal short leftover",
                "stress_test --scenarios shortchunk", sres,
                "pad short prefill chunks before streaming_prefill")
        r.stop_and_join()

    # ===================================================================== #
    log("\n" + "=" * 70)
    log(f"FINAL GPU0 used: {nvidia_used_mb()} MB   torch peak: "
        f"{round(torch.cuda.max_memory_allocated()/1e9,2)} GB")
    log(f"final threads: {thread_snapshot()}")
    report["final_vram_mb"] = nvidia_used_mb()
    report["bugs"] = BUGS
    report["n_bugs"] = len(BUGS)

    with open(os.path.join(OUT, "bugs.json"), "w") as f:
        json.dump(BUGS, f, indent=2)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    log(f"\n==== {len(BUGS)} BUG(S) FOUND ====")
    for b in sorted(BUGS, key=lambda x: ["HIGH", "MEDIUM", "LOW"].index(x["severity"])):
        log(f"  [{b['severity']}] {b['scenario']}: {b['observed']}")
    log(f"\nbugs.json  -> {os.path.join(OUT,'bugs.json')}")
    log(f"report.json-> {os.path.join(OUT,'report.json')}")

    sys.stdout.flush()
    os._exit(0)


def runner_state(r):
    return r.loop.state


if __name__ == "__main__":
    main()
