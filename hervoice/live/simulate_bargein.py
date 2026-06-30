#!/usr/bin/env python3
"""HEADLESS barge-in proof for the live loop (no mic, no speaker).

Drives the SAME LiveLoop the real mic client uses, but feeds audio from files in
real-time-sized chunks:

  1. stream examples/in_en_question.wav  -> turn 1 (user asks)
  2. feed silence so VAD fires end-of-turn -> turn 1 GENERATING (assistant speaks)
  3. once turn 1 has emitted >= --bargein-after-chunks audio packets, INJECT
     examples/in_fifa_question.wav as a simulated barge-in (user interrupts)
  4. feed silence so turn 2 ends -> turn 2 GENERATING (assistant speaks again)

Then ASSERT, from the recorded events, that:
  - turn 1 produced assistant audio
  - the barge-in SET the cancel event and STOPPED turn-1 generation (cancelled=True)
  - a NEW session_id was created for turn 2 (distinct from turn 1)
  - turn 2 produced assistant audio

Writes runs/<out>/{events.jsonl, summary.json, turn_1_assistant.wav,
turn_2_assistant.wav}. Exits non-zero if any assertion fails. No faked success.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -m hervoice.live.simulate_bargein \
      --out-dir runs/live_bargein_smoke --chunk-ms 50
"""
import argparse
import json
import os
import sys
import threading
import time

import librosa
import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/live_bargein_smoke")
    ap.add_argument("--chunk-ms", type=int, default=50, help="mic frame size fed to the loop")
    ap.add_argument("--turn1-audio", default=os.path.join(REPO, "examples/in_en_question.wav"))
    ap.add_argument("--bargein-audio", default=os.path.join(REPO, "examples/in_fifa_question.wav"))
    ap.add_argument("--ref", default=os.path.join(REPO, "examples/ref_female.wav"))
    ap.add_argument("--mode", choices=["plain", "rag"], default="plain")
    ap.add_argument("--bargein-after-chunks", type=int, default=1,
                    help="inject barge-in once turn 1 emitted this many audio packets")
    ap.add_argument("--quant", choices=["none", "int4"], default="int4")
    ap.add_argument("--realtime", action="store_true",
                    help="pace frame injection at wall-clock chunk-ms (off => as-fast-as-loop)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    from .engine import LiveVoiceEngine
    from .turn_detector import TurnDetector
    from .loop import LiveLoop

    retriever = None
    if args.mode == "rag":
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from fifa_rag import FifaRetriever
        retriever = FifaRetriever(device="cuda")

    print(f"[boot] loading MiniCPM-o ({args.quant}) + voice cache ...", flush=True)
    engine = LiveVoiceEngine(ref_audio_path=args.ref, quant=args.quant, retriever=retriever,
                             response_max_new_tokens=args.max_new_tokens)

    # ---- sinks -----------------------------------------------------------
    events = []
    events_lock = threading.Lock()

    def on_event(evt):
        with events_lock:
            events.append(evt)
        keys = {k: v for k, v in evt.items() if k not in ("t",)}
        print(f"[evt {evt['t']:.3f}] {keys}", flush=True)

    turn_audio = {}

    def on_audio(turn_idx, audio24k):
        turn_audio.setdefault(turn_idx, []).append(np.asarray(audio24k, dtype=np.float32))

    detector = TurnDetector(threshold=0.5, min_speech_ms=120, min_silence_ms=220)
    loop = LiveLoop(engine, detector, on_event=on_event, on_audio=on_audio,
                    mode=args.mode, response_max_new_tokens=args.max_new_tokens)

    # ---- load inputs (16k mono) -----------------------------------------
    q1, _ = librosa.load(args.turn1_audio, sr=16000, mono=True)
    q2, _ = librosa.load(args.bargein_audio, sr=16000, mono=True)
    frame = int(16000 * args.chunk_ms / 1000)
    sil = np.zeros(frame, dtype=np.float32)

    def feed(audio):
        for i in range(0, len(audio), frame):
            loop.submit_frame(audio[i:i + frame])
            if args.realtime:
                time.sleep(args.chunk_ms / 1000.0)

    def feed_silence(n):
        for _ in range(n):
            loop.submit_frame(sil.copy())
            if args.realtime:
                time.sleep(args.chunk_ms / 1000.0)

    def wait_state(pred, timeout, poll=0.01):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if pred():
                return True
            time.sleep(poll)
        return False

    # ---- producer: drives turn1 -> silence -> barge-in -> silence --------
    def producer():
        feed(q1)                                   # turn 1: the question
        feed_silence(12)                           # ~> VAD end-of-turn (>=220ms silence)
        # wait until turn 1 is GENERATING and has emitted some audio packets
        wait_state(lambda: loop.state == "GENERATING" and loop._gen_chunks >= 1, timeout=120)
        # keep VAD fed with silence while we wait for enough audio chunks
        waited = 0
        while not (loop._gen_chunks >= args.bargein_after_chunks) and waited < 600:
            loop.submit_frame(sil.copy())
            time.sleep(0.02)
            waited += 1
        # INJECT the barge-in mid-generation
        on_event({"t": time.time(), "type": "inject_bargein", "state": loop.state,
                  "gen_chunks_at_inject": loop._gen_chunks})
        feed(q2)
        feed_silence(12)                           # ~> VAD end-of-turn for turn 2
        # let turn 2 generate to completion (or a bounded wall)
        wait_state(lambda: any(t["turn"] == 2 for t in loop.turns), timeout=60)
        wait_state(lambda: loop.state in ("IDLE",) or
                   (len(loop.turns) >= 2 and loop._gen_done.is_set()), timeout=180)
        feed_silence(4)
        loop.stop()

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()
    loop.run()                                     # blocks until stop sentinel
    prod.join(timeout=5)

    peak_vram = engine.peak_vram_gb()

    # ---- write events.jsonl ---------------------------------------------
    ev_path = os.path.join(out_dir, "events.jsonl")
    with open(ev_path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    # ---- write per-turn audio -------------------------------------------
    for ti in (1, 2):
        chunks = turn_audio.get(ti, [])
        wav_path = os.path.join(out_dir, f"turn_{ti}_assistant.wav")
        if chunks:
            sf.write(wav_path, np.concatenate(chunks), samplerate=24000)

    # ---- derive metrics from events -------------------------------------
    def find(etype, **match):
        for e in events:
            if e["type"] == etype and all(e.get(k) == v for k, v in match.items()):
                return e
        return None

    turn1_audio_n = len(turn_audio.get(1, []))
    turn2_audio_n = len(turn_audio.get(2, []))
    barge = find("barge_in")
    gen_end_t1 = None
    for e in events:
        if e["type"] == "generation_end" and e.get("turn") == 1:
            gen_end_t1 = e
    new_sess = find("new_session")
    cancel_to_new_turn_ms = new_sess.get("cancel_to_new_turn_ms") if new_sess else None

    sids = [t["session_id"] for t in loop.turns]
    summary = {
        "mode": args.mode,
        "quant": args.quant,
        "chunk_ms": args.chunk_ms,
        "turns": [
            {"turn": t["turn"], "session_id": t["session_id"],
             "prefill_chunks": t["prefill_chunks"],
             "audio_chunks_out": t["audio_chunks_out"],
             "barge_in": t.get("barge_in", False)}
            for t in loop.turns
        ],
        "session_ids": sids,
        "unique_session_ids": len(set(sids)),
        "turn1_audio_chunks": turn1_audio_n,
        "turn2_audio_chunks": turn2_audio_n,
        "barge_in_fired": barge is not None,
        "turn1_generation_cancelled": bool(gen_end_t1 and gen_end_t1.get("cancelled")),
        "cancel_to_new_turn_ms": cancel_to_new_turn_ms,
        "peak_vram_gb": peak_vram,
        "n_events": len(events),
    }

    # ---- ASSERTIONS (honest; non-zero exit on failure) ------------------
    failures = []
    if turn1_audio_n <= 0:
        failures.append("turn 1 produced NO assistant audio")
    if not summary["barge_in_fired"]:
        failures.append("barge-in did NOT fire")
    if not summary["turn1_generation_cancelled"]:
        failures.append("turn 1 generation was NOT cancelled by barge-in")
    if len(loop.turns) < 2:
        failures.append("no second turn was created")
    elif len(set(sids[:2])) < 2:
        failures.append(f"turn 2 did not get a NEW session_id (sids={sids[:2]})")
    if turn2_audio_n <= 0:
        failures.append("turn 2 produced NO assistant audio")

    summary["assertions_passed"] = (len(failures) == 0)
    summary["failures"] = failures

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print(f"\n[events]  {ev_path}")
    print(f"[summary] {os.path.join(out_dir, 'summary.json')}")
    if failures:
        print("\n[FAIL] barge-in assertions failed:")
        for fmsg in failures:
            print("   - " + fmsg)
        code = 1
    else:
        print("\n[PASS] live barge-in proven headlessly.")
        code = 0
    # a background MiniCPM/CUDA daemon thread can SIGABRT during interpreter
    # teardown; hard-exit so the honest assertion code is what the shell sees.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
