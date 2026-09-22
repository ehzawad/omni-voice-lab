#!/usr/bin/env python3
"""Drive the gateway over a real WebSocket from a wav file: the box has no microphone, so
this is how the browser path is proven, the same way hervoice/live/simulate_bargein.py does
it for the in-process loop.

Two scenarios:

  --scenario turn     one utterance, then silence; expect asr -> text -> audio -> turn_end.
  --scenario bargein  one utterance, wait until the assistant is actually SPEAKING, then feed
                      speech again; expect a `cancel` for the old epoch, a new turn_start with
                      barge_in=true, and NO further audio stamped with the old epoch.

The second scenario is the real test: it proves the stale-audio drop works, which is the
failure the in-process design could not exhibit because nothing was ever buffered downstream.

    .venv-bnweb/bin/python -m hervoice.svc.simulate_ws --scenario bargein
"""
import argparse
import asyncio
import json
import os
import struct
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.svc import config as C       # noqa: E402
from hervoice.svc import protocol as P     # noqa: E402

FRAME_MS = 20


def load16k(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    assert sr == C.SR_IN, f"expected {C.SR_IN} Hz, got {sr}"
    return a


async def feed(ws, audio, label=""):
    n = int(C.SR_IN * FRAME_MS / 1000)
    for i in range(0, len(audio), n):
        await ws.send(np.ascontiguousarray(audio[i:i + n], dtype="<f4").tobytes())
        await asyncio.sleep(FRAME_MS / 1000)


async def feed_silence(ws, seconds):
    n = int(C.SR_IN * FRAME_MS / 1000)
    frames = int(seconds * 1000 / FRAME_MS)
    z = np.zeros(n, dtype="<f4").tobytes()
    for _ in range(frames):
        await ws.send(z)
        await asyncio.sleep(FRAME_MS / 1000)


async def main():
    import websockets

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="examples/in_bn_question.wav")
    ap.add_argument("--scenario", choices=["turn", "bargein"], default="turn")
    ap.add_argument("--url", default=f"ws://127.0.0.1:{C.GW_PORT}/ws")
    ap.add_argument("--token", default=os.environ.get("HV_GW_TOKEN", ""))
    ap.add_argument("--out", default="runs/svc/sim")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    audio = load16k(a.audio)

    events, audio_by_epoch, stale = [], {}, []
    t0 = time.time()
    state = {"cur": 0, "speaking": False, "cancelled_epochs": set(), "done": False}

    async with websockets.connect(a.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "token": a.token, "sample_rate": C.SR_IN}))
        ready = asyncio.Event()

        async def reader():
            async for m in ws:
                if isinstance(m, str):
                    d = json.loads(m)
                    d["at_ms"] = round((time.time() - t0) * 1000, 1)
                    events.append(d)
                    t = d.get("type")
                    if t == "ready": ready.set()
                    if t == "turn_start":
                        state["cur"] = d["epoch"]; state["speaking"] = False
                    elif t == "state":
                        state["speaking"] = (d["state"] == "speaking")
                    elif t == "cancel":
                        state["cancelled_epochs"].add(d["epoch"])
                    elif t == "turn_end":
                        state["done"] = True
                    print(f"  {d['at_ms']:8.0f}ms  {json.dumps(d, ensure_ascii=False)[:150]}")
                else:
                    kind, epoch, seq, sr = P.unpack_header(m)
                    pcm = np.frombuffer(m, dtype="<f4", offset=P.HEADER_SIZE)
                    # instant-playback client: acknowledge each chunk so the memory ledger commits
                    await ws.send(json.dumps({"type": "played", "epoch": epoch, "seq": seq}))
                    audio_by_epoch.setdefault(epoch, []).append(pcm)
                    if epoch in state["cancelled_epochs"]:
                        stale.append((epoch, seq, round((time.time() - t0) * 1000, 1)))

        rtask = asyncio.create_task(reader())
        await asyncio.wait_for(ready.wait(), timeout=30)   # never stream before the server is ready

        print(f"[sim] scenario={a.scenario}  feeding {len(audio)/C.SR_IN:.2f}s of speech")
        await feed(ws, audio)
        await feed_silence(ws, 0.8)          # let the VAD endpoint the utterance

        if a.scenario == "turn":
            deadline = time.time() + 45
            while not state["done"] and time.time() < deadline:
                await feed_silence(ws, 0.2)
        else:
            # wait until audio is actually playing, then interrupt
            deadline = time.time() + 45
            while not state["speaking"] and time.time() < deadline:
                await feed_silence(ws, 0.1)
            if state["speaking"]:
                print("[sim] assistant is SPEAKING -> barging in now")
                await asyncio.sleep(0.5)
                bargein_at = time.time()
                await feed(ws, audio)
                await feed_silence(ws, 0.8)
                deadline = time.time() + 45
                while time.time() < deadline:
                    await feed_silence(ws, 0.2)
                    if any(e["type"] == "turn_start" and e.get("barge_in") for e in events):
                        break
                print(f"[sim] barge-in fed at +{(bargein_at-t0)*1000:.0f}ms")
            else:
                print("[sim] never reached SPEAKING; cannot test barge-in")

        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.5)
        rtask.cancel()

    # ---- verdict
    print("\n[verdict]")
    ts = [e for e in events if e["type"] == "turn_start"]
    cancels = [e for e in events if e["type"] == "cancel"]
    print(f"  turns started      : {len(ts)}  (barge_in: {sum(1 for e in ts if e.get('barge_in'))})")
    print(f"  cancels emitted    : {len(cancels)}  epochs={[e['epoch'] for e in cancels]}")
    for ep, chunks in sorted(audio_by_epoch.items()):
        n = sum(len(c) for c in chunks)
        print(f"  audio epoch {ep:>3}    : {len(chunks)} frames, {n/C.SR_OUT:.2f}s")
    print(f"  STALE frames (arrived after their epoch was cancelled): {len(stale)}")
    if stale:
        print(f"    -> the client drops these by epoch; server sent {len(stale)} after cancel")
    asr = [e for e in events if e["type"] == "asr"]
    if asr:
        print(f"  ASR                : {asr[0].get('text','')!r}  (turn audio {asr[0].get('turn_audio_s')}s, "
              f"dropped frames {asr[0].get('dropped_frames')})")
        if asr[0].get("dropped_frames"):
            print("    !! frames were dropped -- the client streamed before the server was ready, or the consumer stalled")
    fa = [e.get("first_audio_ms") for e in events if e["type"] == "metrics" and e.get("first_audio_ms")]
    cn = [e.get("cancel_to_new_turn_ms") for e in events if e["type"] == "metrics" and e.get("cancel_to_new_turn_ms")]
    if fa: print(f"  first audio        : {fa[0]:.0f} ms")
    if cn: print(f"  cancel -> new turn : {cn[0]:.0f} ms")

    for ep, chunks in audio_by_epoch.items():
        if chunks:
            sf.write(os.path.join(a.out, f"epoch{ep}.wav"), np.concatenate(chunks), C.SR_OUT)
    json.dump({"events": events, "stale": stale}, open(os.path.join(a.out, "sim.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"  wrote {a.out}/sim.json and per-epoch wavs")


if __name__ == "__main__":
    asyncio.run(main())
