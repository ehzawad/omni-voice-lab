#!/usr/bin/env python3
"""LOCAL real-mic / real-speaker client for the live barge-in loop.

Run this on YOUR OWN machine (laptop/desktop with a mic+speaker), NOT on the
headless A5000 box. It uses the SAME LiveVoiceEngine + TurnDetector + LiveLoop as
the headless simulator -- the ONLY difference is the audio I/O endpoints:

    mic  (sounddevice InputStream, 16k mono)  -> loop.submit_frame(...)
    loop -> on_audio(turn, audio24k)          -> speaker (sounddevice OutputStream)

Requirements that exist ONLY on the user's machine:
    pip install sounddevice            # needs PortAudio (apt install libportaudio2)
The model itself still needs a CUDA GPU. If you run the model remotely, you must
ship 16k frames to it and stream 24k audio back yourself; this file assumes the
model runs locally for simplicity.

Imports of sounddevice are wrapped so that merely importing this module on the
headless box does not crash; it only fails if you actually try to run() it there.

Barge-in playback note: when you interrupt, the engine STOPS emitting audio. The
CLIENT is responsible for stopping/cross-fading whatever is still in the speaker
buffer (here we just abort the OutputStream); that jitter/crossfade is a client
concern, not the engine's.
"""
import argparse
import os
import queue
import threading

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import sounddevice as sd  # noqa: F401  (optional; only needed to actually run)
    _SD_OK = True
    _SD_ERR = None
except Exception as e:  # ImportError or PortAudio load failure on a headless box
    sd = None
    _SD_OK = False
    _SD_ERR = e

FRAME_MS = 50
SR_IN = 16000
SR_OUT = 24000


class SpeakerPlayer:
    """Background 24k playback that can be flushed instantly on barge-in."""

    def __init__(self):
        self.q = queue.Queue()
        self.stream = sd.OutputStream(samplerate=SR_OUT, channels=1, dtype="float32")
        self.stream.start()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def play(self, audio24k):
        self.q.put(np.asarray(audio24k, dtype=np.float32).reshape(-1, 1))

    def flush(self):
        """Drop everything queued (barge-in): client-side stop of stale audio."""
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                block = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.stream.write(block)

    def close(self):
        self._stop.set()
        self.stream.stop()
        self.stream.close()


def run(args):
    if not _SD_OK:
        raise RuntimeError(
            "sounddevice/PortAudio unavailable -- this client must run on a machine "
            f"with a mic+speaker (not the headless box). Import error: {_SD_ERR}")

    from .engine import LiveVoiceEngine
    from .turn_detector import TurnDetector
    from .loop import LiveLoop

    retriever = None
    if args.mode == "rag":
        import sys
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from fifa_rag import FifaRetriever
        retriever = FifaRetriever(device="cuda")

    print("[boot] loading model + voice cache ...", flush=True)
    engine = LiveVoiceEngine(ref_audio_path=args.ref, quant=args.quant, retriever=retriever)
    detector = TurnDetector(threshold=0.5, min_speech_ms=120, min_silence_ms=220)

    player = SpeakerPlayer()

    def on_event(evt):
        if evt["type"] in ("turn_start", "barge_in", "new_session", "generation_end",
                           "turn_complete", "rag", "first_audio"):
            print(f"[evt] {evt['type']} {dict((k, evt[k]) for k in evt if k != 't')}",
                  flush=True)
        if evt["type"] == "barge_in":
            player.flush()   # stop stale speaker audio the instant the user cuts in

    def on_audio(turn_idx, audio24k):
        player.play(audio24k)

    loop = LiveLoop(engine, detector, on_event=on_event, on_audio=on_audio, mode=args.mode)

    frame = int(SR_IN * FRAME_MS / 1000)

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", flush=True)
        loop.submit_frame(indata[:, 0].copy().astype(np.float32))

    print("[live] speak now (Ctrl-C to quit). Interrupt any time -- it will barge in.",
          flush=True)
    loop_thread = threading.Thread(target=loop.run, daemon=True)
    loop_thread.start()
    try:
        with sd.InputStream(samplerate=SR_IN, channels=1, dtype="float32",
                            blocksize=frame, callback=mic_callback):
            loop_thread.join()
    except KeyboardInterrupt:
        print("\n[quit]")
    finally:
        loop.stop()
        player.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default=os.path.join(REPO, "examples/ref_female.wav"))
    ap.add_argument("--mode", choices=["plain", "rag"], default="plain")
    ap.add_argument("--quant", choices=["none", "int4"], default="int4")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
