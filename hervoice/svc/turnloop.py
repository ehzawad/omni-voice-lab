#!/usr/bin/env python3
"""The turn state machine for the service world.

This REUSES hervoice.live.turn_detector.TurnDetector unchanged -- the Silero VAD and its
hysteresis are stress-tested and correct -- but it does NOT reuse hervoice.live.loop.LiveLoop,
for reasons that are specific and not stylistic:

  1. LiveLoop drives an engine that owns ONE model session: reset_for_new_turn() -> sid, then
     prefill_system(sid), prefill_user_chunk(sid, ...), generate(sid). Across three
     independent HTTP services there is no such session, and passing a meaningless sid through
     every call would hide rather than model that.
  2. LiveLoop has a verified defect in the RAG path: _end_user_turn() binds `sid` before
     calling _apply_rag(), which creates a NEW session and prefills the retrieved facts into
     it, after which _start_generation(sid) runs on the STALE session and the grounding is
     silently discarded.
  3. _finish_turn_natural() returns to IDLE when the GENERATOR finishes, not when the listener
     has finished HEARING the answer. Over a network the client can still hold seconds of
     queued speech, so the next utterance starts a fresh turn with no cancel for the transport
     to flush on -- and the user hears the old answer continue over their new question.
  4. submit_frame() appends to an unbounded queue while the consumer does synchronous work.
  5. Onset is clipped: frames are VAD-inspected while idle but never retained, so a turn
     begins ~128 ms after the user actually started speaking and initial consonants are lost.

What is different here:

  * EPOCH. Every turn gets an epoch, and any cancel bumps it. Audio is stamped with it, so the
    client can discard what is already in its buffer. This is the network analogue of the
    in-process cancel_event.
  * SPEAKING is a real state, distinct from GENERATING. The turn is not over when the last
    sentence is synthesised; it is over when the client acknowledges playback or the audio it
    was sent has had time to play. Speech arriving in that window is a barge-in.
  * PREROLL. A rolling pre-speech buffer is prepended to every turn so the onset survives.
  * BOUNDED admission. The frame queue has a limit and drops oldest-first under overload
    rather than turning live speech into delayed speech.
"""
import collections
import logging
import queue
import threading
import time

import numpy as np

log = logging.getLogger("turnloop")

PREROLL_MS = 320          # > the detector's ~128 ms confirmation, with margin
IDLE, USER_SPEAKING, THINKING, SPEAKING = "listening", "user_speaking", "thinking", "speaking"


class TurnLoop:
    """Single-threaded consumer; generation runs in a worker thread.

    engine must provide:
        transcribe(audio16k) -> {"text": str, "ms": float}
        respond(text, cancel_event, on_delta, on_sentence, on_audio) -> None
    on_audio(pcm24k) is called once per synthesised chunk and must not block for long.
    """

    def __init__(self, engine, detector, on_event, on_audio, sr=16000,
                 max_queue=200, max_turn_s=30.0, barge_guard_ms=350):
        self.engine = engine
        self.detector = detector
        self.on_event = on_event
        self.on_audio = on_audio
        self.sr = sr
        self.max_turn_samples = int(max_turn_s * sr)
        self.barge_guard_ms = barge_guard_ms

        self.q = queue.Queue(maxsize=max_queue)
        self.cancel = threading.Event()
        self.state = IDLE
        self.epoch = 0
        self.turn = 0
        self._dropped = 0
        self._turn_degraded = False
        self._wedged = False

        self._buf = []
        self._buf_n = 0
        self._preroll = collections.deque()
        self._preroll_n = 0
        self._preroll_max = int(sr * PREROLL_MS / 1000)

        self._gen_thread = None
        self._gen_done = threading.Event()
        self._speak_started = None
        self._audio_sent_s = 0.0

    # ------------------------------------------------------------------ public
    def submit_frame(self, frame16k):
        """Bounded admission.

        Dropping frames while the user is mid-utterance silently corrupts the transcript --
        measured: 13 dropped frames turned "বাংলাদেশের রাজধানীর নাম কি" into "কানির নাম কী".
        So while capturing we wait briefly for room rather than dropping, and if the queue is
        still full the loss is COUNTED and surfaced on the turn, never silent. Outside an
        utterance, dropping the oldest frame is harmless and keeps latency honest.
        """
        item = ("frame", np.asarray(frame16k, dtype=np.float32).reshape(-1))
        capturing = self.state == USER_SPEAKING
        try:
            self.q.put(item, timeout=0.05 if capturing else 0)
            return
        except queue.Full:
            pass
        try:
            self.q.get_nowait()
            self._dropped += 1
            if capturing:
                self._turn_degraded = True
            self.q.put_nowait(item)
        except (queue.Empty, queue.Full):
            self._dropped += 1

    def stop(self):
        try:
            self.q.put_nowait(("stop", None))
        except queue.Full:
            with self.q.mutex:
                self.q.queue.clear()
            self.q.put_nowait(("stop", None))

    def run(self):
        self._emit("state", state=self.state)
        while True:
            kind, payload = self.q.get()
            if kind == "stop":
                if self._gen_thread is not None:
                    self._cancel_generation("client")
                self._emit("stopped")
                return
            try:
                self._on_frame(payload)
            except Exception as e:                      # noqa: BLE001
                log.exception("frame error")
                self._emit("error", message=repr(e))
                self._recover()

    # ------------------------------------------------------------------ frames
    def _on_frame(self, frame):
        if self.state in (IDLE,):
            self._push_preroll(frame)
        elif self.state == USER_SPEAKING:
            self._accumulate(frame)
        else:
            self._push_preroll(frame)   # keep context during THINKING/SPEAKING for barge-in

        for ev in self.detector.process(frame):
            self._on_vad(ev)

        if self.state == THINKING and self._gen_done.is_set():
            self._finish_generation()
        elif self.state == SPEAKING and self._playback_probably_done():
            self._end_turn(cancelled=False)

    def _on_vad(self, ev):
        from hervoice.live.turn_detector import VadEvent
        if self._wedged:
            return
        if ev.kind == VadEvent.SPEECH_START:
            if self.state == IDLE:
                self._begin_turn(barge_in=False)
            elif self.state in (THINKING, SPEAKING):
                # Barge-in covers BOTH: while the brain is writing, and while the client is
                # still playing audio we already sent. LiveLoop only handled the first.
                if self._past_barge_guard():
                    self._barge_in()
        elif ev.kind == VadEvent.SPEECH_END:
            if self.state == USER_SPEAKING:
                self._end_user_speech()

    # ------------------------------------------------------------------ turn
    def _begin_turn(self, barge_in):
        self.turn += 1
        self.epoch += 1
        self.state = USER_SPEAKING
        self._buf, self._buf_n = [], 0
        self._turn_degraded = False
        # onset survives: the pre-speech ring is the start of the utterance
        if self._preroll:
            pre = np.concatenate(list(self._preroll))
            self._buf.append(pre)
            self._buf_n += pre.size
        self._reset_preroll()
        self.detector.reset()
        self.detector.triggered = True
        self._emit("turn_start", turn=self.turn, epoch=self.epoch, barge_in=barge_in)
        self._emit("state", state=self.state)

    def _accumulate(self, frame):
        self._buf.append(frame)
        self._buf_n += frame.size
        if self._buf_n > self.max_turn_samples:      # hard cap on one utterance
            self._end_user_speech()

    def _end_user_speech(self):
        audio = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)
        self._buf, self._buf_n = [], 0
        self.state = THINKING
        self._emit("state", state=self.state)
        self.cancel.clear()
        self._gen_done.clear()
        self._audio_sent_s = 0.0
        self._speak_started = None
        self.detector.reset()
        epoch = self.epoch
        self._gen_thread = threading.Thread(
            target=self._gen_worker, args=(audio, epoch), daemon=True)
        self._gen_thread.start()

    def _gen_worker(self, audio, epoch):
        t0 = time.time()
        try:
            asr = self.engine.transcribe(audio)
            self._emit("asr", turn=self.turn, text=asr.get("text", ""), ms=asr.get("ms", 0),
                       turn_audio_s=round(len(audio) / self.sr, 3),
                       dropped_frames=self._dropped, degraded=self._turn_degraded)
            if self._turn_degraded:
                self._emit("error", message="audio frames were dropped during this "
                           "utterance; the transcript may be wrong")
            if not asr.get("text"):
                return
            first = {"t": None}

            def on_delta(d):
                self._emit("text", turn=self.turn, delta=d)

            def on_sentence(s):
                self._emit("sentence", turn=self.turn, text=s)

            def on_audio(pcm):
                if self.cancel.is_set() or epoch != self.epoch:
                    return
                if first["t"] is None:
                    first["t"] = time.time()
                    self._emit("metrics", turn=self.turn,
                               first_audio_ms=round((first["t"] - t0) * 1000, 1))
                    if self._speak_started is None:
                        self._speak_started = time.time()
                        self.state = SPEAKING
                        self._emit("state", state=self.state)
                self._audio_sent_s += len(pcm) / 24000.0
                self.on_audio(epoch, pcm)

            self.engine.respond(asr["text"], self.cancel, on_delta, on_sentence, on_audio)
        except Exception as e:                          # noqa: BLE001
            log.exception("generation failed")
            self._emit("error", message=f"generation failed: {e!r}")
        finally:
            self._gen_done.set()

    def _finish_generation(self):
        """The brain and TTS are done. The LISTENER is not: audio is still in flight."""
        self._join(timeout=5.0)
        if self._audio_sent_s <= 0:
            self._end_turn(cancelled=False)
            return
        if self.state != SPEAKING:
            self.state = SPEAKING
            self._speak_started = self._speak_started or time.time()
            self._emit("state", state=self.state)

    def _playback_probably_done(self):
        """Server-side estimate, used only as a backstop; the client's `played` ack is
        authoritative and arrives via note_played()."""
        if self._speak_started is None:
            return True
        return (time.time() - self._speak_started) > (self._audio_sent_s + 0.35)

    def note_played(self, epoch, seq):
        """Client acknowledged playback. Authoritative end-of-speaking for the current turn."""
        if epoch == self.epoch and self.state == SPEAKING and self._gen_done.is_set():
            self._end_turn(cancelled=False)

    def _end_turn(self, cancelled):
        self._emit("turn_end", turn=self.turn, epoch=self.epoch, cancelled=cancelled)
        self.state = IDLE
        self._speak_started = None
        self._audio_sent_s = 0.0
        self.detector.reset()
        self._emit("state", state=self.state)

    # ------------------------------------------------------------------ barge-in
    def _past_barge_guard(self):
        if self._speak_started is None:
            return True     # still THINKING: nothing is playing, interrupt freely
        return (time.time() - self._speak_started) * 1000 >= self.barge_guard_ms

    def _barge_in(self):
        t0 = time.time()
        self._cancel_generation("barge_in")
        self._begin_turn(barge_in=True)
        self._emit("metrics", turn=self.turn,
                   cancel_to_new_turn_ms=round((time.time() - t0) * 1000, 1))

    def _cancel_generation(self, reason):
        self.cancel.set()
        old = self.epoch
        self.epoch += 1                       # everything stamped `old` is now void
        self._emit("cancel", epoch=old, reason=reason)
        self._join(timeout=10.0)
        self._audio_sent_s = 0.0
        self._speak_started = None

    def _join(self, timeout):
        th = self._gen_thread
        if th is None:
            return
        th.join(timeout=timeout)
        if th.is_alive():                     # LiveLoop discarded the ref without checking
            # Do NOT clear the reference and do NOT allow another turn: the worker may still
            # be inside a flow-matching call on the GPU. Wedging loudly beats two concurrent
            # generations sharing one budget.
            self._wedged = True
            log.error("generation thread still alive after %.1fs; refusing further turns", timeout)
            self._emit("error", message="generation did not stop; session wedged, reconnect")
        else:
            self._gen_thread = None

    # ------------------------------------------------------------------ helpers
    def _push_preroll(self, frame):
        self._preroll.append(frame)
        self._preroll_n += frame.size
        while self._preroll_n > self._preroll_max and len(self._preroll) > 1:
            self._preroll_n -= self._preroll.popleft().size

    def _reset_preroll(self):
        self._preroll.clear()
        self._preroll_n = 0

    def _recover(self):
        try:
            if self._gen_thread is not None:
                self._cancel_generation("error")
        finally:
            self._buf, self._buf_n = [], 0
            self._reset_preroll()
            try:
                self.detector.reset()
            except Exception:
                pass
            self.state = IDLE
            self._emit("state", state=self.state)

    def _emit(self, etype, **f):
        ev = {"type": etype, "t": time.time()}
        ev.update(f)
        if self.on_event:
            self.on_event(ev)
