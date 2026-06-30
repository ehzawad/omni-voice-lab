#!/usr/bin/env python3
"""LiveLoop: the VAD-gated streaming turn state machine with barge-in.

ONE engine, ONE Silero detector, ONE cancel_event. Frames arrive via submit_frame
(from the simulator OR a real mic thread -- SAME code path, no forked behavior).
The consumer loop runs the VAD on every frame so it keeps detecting speech even
while the model is GENERATING; that is what makes barge-in possible.

States:
    IDLE/LISTENING -> USER_SPEAKING -> GENERATING -> (natural end) IDLE
                                         |
                                         +-- barge-in -> INTERRUPTING -> RESETTING -> USER_SPEAKING

Generation runs in a worker THREAD (GPU). The consumer loop stays on the main
thread doing only cheap CPU VAD + cancel-event signalling, so the two never make
concurrent CUDA calls on the big model (VAD is a separate Silero model).

Concurrency note for a real client: audio playback jitter / crossfade on cancel
is a CLIENT concern (the engine just stops emitting); not handled here.
"""
import queue
import threading
import time

import numpy as np

from .turn_detector import VadEvent

# loop tuning (independent of the mic frame size).
# MiniCPM-o's streaming audio encoder expects ~1 s aligned chunks (model.CHUNK_MS
# = 1000, FIRST_CHUNK_MS = 1035, the first chunk is auto-padded up to 1035 ms);
# feeding shorter chunks underflows the apm conv front-end. So aggregate to 1000 ms.
PREFILL_CHUNK_MS = 1000          # == model.CHUNK_MS (proven in minicpm_stream.py)
BARGEIN_CTX_MS = 1200            # keep this much recent audio during GENERATING to
                                 # seed the barge-in turn with the speech already spoken


class LiveLoop:
    def __init__(self, engine, detector, on_event=None, on_audio=None,
                 mode="plain", sr=16000, prefill_chunk_ms=PREFILL_CHUNK_MS,
                 response_max_new_tokens=256, barge_guard_chunks=1):
        """
        Args:
            engine: LiveVoiceEngine
            detector: TurnDetector
            on_event(dict): typed event sink (timestamps added here)
            on_audio(turn_idx, audio24k): assistant audio sink
            mode: "plain" (fast, no RAG) or "rag" (ASR->retrieve->grounded reply)
            barge_guard_chunks: ignore barge-in until turn has emitted this many
                                audio chunks (avoids self-trigger on first packet)
        """
        self.engine = engine
        self.detector = detector
        self.on_event = on_event
        self.on_audio = on_audio
        self.mode = mode
        self.sr = sr
        self.prefill_chunk = int(sr * prefill_chunk_ms / 1000)
        self.response_max_new_tokens = response_max_new_tokens
        self.barge_guard_chunks = barge_guard_chunks

        self.frame_q = queue.Queue()
        self.cancel_event = threading.Event()
        self.state = "IDLE"
        self.turns = []                  # per-turn metadata dicts
        self._turn_idx = 0

        # current-turn user audio buffer (16k mono frames)
        self._buf = []
        self._buf_samples = 0
        self._turn_audio_all = []        # full user audio for this turn (for RAG ASR)

        # rolling recent audio while GENERATING (to seed a barge-in turn)
        self._barge_ctx = []
        self._barge_ctx_samples = 0
        self._barge_ctx_max = int(sr * BARGEIN_CTX_MS / 1000)

        # generation worker plumbing
        self._gen_thread = None
        self._gen_done = threading.Event()
        self._gen_chunks = 0
        self._gen_sid = None
        # a barge-in onset seen during the guard window, fired once the guard clears
        self._pending_guarded_onset = False
        self._cancel_set_ts = None

    # --------------------------------------------------------------- public API
    def submit_frame(self, frame16k):
        self.frame_q.put(("frame", np.asarray(frame16k, dtype=np.float32).reshape(-1)))

    def stop(self):
        self.frame_q.put(("stop", None))

    def run(self):
        """Single-threaded consumer; returns when a stop sentinel is seen and any
        in-flight generation has been drained."""
        self._emit("loop_start", state=self.state, mode=self.mode)
        while True:
            kind, payload = self.frame_q.get()
            if kind == "stop":
                # let a natural turn finish if one is running; cancel otherwise
                if self.state == "GENERATING":
                    self._join_generation(cancel=False)
                self._emit("loop_stop", state=self.state)
                return
            try:
                self._on_frame(payload)
            except Exception as e:
                # An engine/model exception on the consumer thread (e.g. a malformed
                # prefill chunk) must NOT silently kill this loop -- otherwise the
                # loop becomes a black hole that swallows frames forever. Surface it
                # and recover to a clean LISTENING state, abandoning the broken turn.
                self._emit("frame_error", state=self.state, error=repr(e))
                self._recover_after_error()

    # --------------------------------------------------------------- frame path
    def _on_frame(self, frame):
        # keep a rolling context of recent audio while we are GENERATING so a
        # barge-in turn starts from the words the user already began speaking
        if self.state == "GENERATING":
            self._push_barge_ctx(frame)

        # in USER_SPEAKING we accumulate the user's audio for prefill
        if self.state == "USER_SPEAKING":
            self._accumulate(frame)

        for ev in self.detector.process(frame):
            self._on_vad_event(ev)

        # a barge-in onset that arrived during the guard window fires once the guard
        # clears (the latched VAD will not re-emit SPEECH_START for the same utterance)
        if (self.state == "GENERATING" and self._pending_guarded_onset
                and self._gen_chunks >= self.barge_guard_chunks):
            self._pending_guarded_onset = False
            self._barge_in()
            return

        # natural end of a turn: generation worker finished on its own
        if self.state == "GENERATING" and self._gen_done.is_set():
            self._finish_turn_natural()

    def _on_vad_event(self, ev):
        if ev.kind == VadEvent.SPEECH_START:
            if self.state in ("IDLE", "LISTENING"):
                self._begin_user_turn(prev_ctx=None)
            elif self.state == "GENERATING":
                # BARGE-IN (only after the turn has actually started speaking)
                if self._gen_chunks >= self.barge_guard_chunks:
                    self._barge_in()
                else:
                    # Onset during the guard window: Silero latches triggered=True and
                    # will not re-emit SPEECH_START for this same utterance. Remember it
                    # and fire once the guard clears (handled in _on_frame).
                    self._pending_guarded_onset = True
        elif ev.kind == VadEvent.SPEECH_END:
            if self.state == "USER_SPEAKING":
                self._end_user_turn()
            elif self.state == "GENERATING":
                # guarded speech ended before the guard cleared -> not a real barge-in
                self._pending_guarded_onset = False

    # --------------------------------------------------------------- user turn
    def _begin_user_turn(self, prev_ctx):
        is_barge = prev_ctx is not None and getattr(prev_ctx, "size", 0) > 0
        self._turn_idx += 1
        sid = self.engine.reset_for_new_turn()
        self.engine.prefill_system(sid)
        self.detector.reset()
        self.detector.triggered = True       # we are mid-speech right now
        self._buf, self._buf_samples = [], 0
        self._turn_audio_all = []
        turn = {"turn": self._turn_idx, "session_id": sid, "prefill_chunks": 0,
                "audio_chunks_out": 0, "mode": self.mode, "barge_in": is_barge}
        self.turns.append(turn)
        self.state = "USER_SPEAKING"
        self._emit("turn_start", turn=self._turn_idx, session_id=sid,
                   barge_in=is_barge)
        if is_barge:
            # seed with the audio the user already spoke during the barge-in
            self._accumulate(prev_ctx)

    def _accumulate(self, frame):
        self._buf.append(frame)
        self._buf_samples += frame.size
        self._turn_audio_all.append(frame)
        while self._buf_samples >= self.prefill_chunk:
            chunk = np.concatenate(self._buf)
            head, tail = chunk[:self.prefill_chunk], chunk[self.prefill_chunk:]
            self.engine.prefill_user_chunk(self._gen_sid_or_turn(), head, is_last=False)
            self.turns[-1]["prefill_chunks"] += 1
            self._emit("prefill_chunk", turn=self._turn_idx, is_last=False,
                       samples=int(head.size))
            self._buf = [tail] if tail.size else []
            self._buf_samples = tail.size

    def _gen_sid_or_turn(self):
        return self.turns[-1]["session_id"]

    def _end_user_turn(self):
        sid = self.turns[-1]["session_id"]
        leftover = np.concatenate(self._buf) if self._buf else np.zeros(512, dtype=np.float32)
        self.engine.prefill_user_chunk(sid, leftover, is_last=True)
        self.turns[-1]["prefill_chunks"] += 1
        self._emit("prefill_chunk", turn=self._turn_idx, is_last=True,
                   samples=int(leftover.size))
        self._buf, self._buf_samples = [], 0

        # optional RAG: transcribe + retrieve, then re-seed the system prompt.
        # This loses the fast first-audio path -- stated honestly in events.
        if self.mode == "rag":
            self._apply_rag(sid)

        self._start_generation(sid)

    def _apply_rag(self, sid):
        full = (np.concatenate(self._turn_audio_all) if self._turn_audio_all
                else np.zeros(512, dtype=np.float32))
        t0 = time.time()
        asr = self.engine.asr_text(full)
        facts = self.engine.retrieve_facts(asr)
        self._emit("rag", turn=self._turn_idx, asr=asr,
                   retrieved=bool(facts), rag_ms=round((time.time() - t0) * 1000, 1))
        # rebuild the turn's system prompt with facts, then re-prefill the user
        # audio in the SAME 1 s-aligned chunks the plain path uses (the streaming
        # encoder breaks on a single oversized chunk).
        new_sid = self.engine.reset_for_new_turn()
        self.turns[-1]["session_id"] = new_sid
        self.engine.prefill_system(new_sid, extra_facts=facts)
        self._prefill_audio_chunked(new_sid, full)

    def _prefill_audio_chunked(self, sid, audio):
        step = self.prefill_chunk
        n = max(1, (len(audio) + step - 1) // step)
        for i in range(n):
            ch = audio[i * step:(i + 1) * step]
            self.engine.prefill_user_chunk(sid, ch, is_last=(i == n - 1))
            self.turns[-1]["prefill_chunks"] += 1

    # --------------------------------------------------------------- generation
    def _start_generation(self, sid):
        self.state = "GENERATING"
        self.cancel_event.clear()
        self._gen_done.clear()
        self._gen_chunks = 0
        self._gen_sid = sid
        self._pending_guarded_onset = False
        self._reset_barge_ctx()
        self.detector.reset()            # fresh VAD state to watch for barge-in
        self._emit("generation_start", turn=self._turn_idx, session_id=sid)
        self._gen_thread = threading.Thread(target=self._gen_worker, args=(sid,), daemon=True)
        self._gen_thread.start()

    def _gen_worker(self, sid):
        first_ts = None
        t0 = time.time()
        try:
            for audio, text in self.engine.generate(
                    sid, self.cancel_event, self.response_max_new_tokens):
                if self.cancel_event.is_set():
                    break
                if audio is not None and audio.size:
                    if first_ts is None:
                        first_ts = time.time()
                        self._emit("first_audio", turn=self._turn_idx,
                                   first_audio_ms=round((first_ts - t0) * 1000, 1))
                    self._gen_chunks += 1
                    self.turns[-1]["audio_chunks_out"] = self._gen_chunks
                    if self.on_audio:
                        self.on_audio(self._turn_idx, audio)
                if text:
                    self._emit("text_delta", turn=self._turn_idx, text=text)
        except Exception as e:  # honest failure surfacing, never fake success
            self._emit("generation_error", turn=self._turn_idx, error=str(e))
        finally:
            self._emit("generation_end", turn=self._turn_idx,
                       audio_chunks_out=self._gen_chunks,
                       cancelled=self.cancel_event.is_set())
            self._gen_done.set()

    def _barge_in(self):
        self._cancel_set_ts = time.time()
        self.cancel_event.set()
        self._emit("barge_in", turn=self._turn_idx, session_id=self._gen_sid,
                   audio_chunks_before_cancel=self._gen_chunks)
        # INTERRUPTING -> join the worker (it bails on cancel within ~one chunk)
        self.state = "INTERRUPTING"
        self._join_generation(cancel=True)
        # RESETTING: drop the abandoned generator, keep voice cache, new session
        prev_ctx = (np.concatenate(self._barge_ctx) if self._barge_ctx
                    else np.zeros(0, dtype=np.float32))
        self._reset_barge_ctx()
        self.state = "RESETTING"
        self._emit("resetting", turn=self._turn_idx)
        self._begin_user_turn(prev_ctx=prev_ctx)
        new_sid = self.turns[-1]["session_id"]
        self._emit("new_session", turn=self._turn_idx, session_id=new_sid,
                   cancel_to_new_turn_ms=round((time.time() - self._cancel_set_ts) * 1000, 1))

    def _join_generation(self, cancel):
        if cancel:
            self.cancel_event.set()
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=10.0)
        self._gen_thread = None

    def _recover_after_error(self):
        """Abandon the current (broken) turn and return to a clean listening state
        without tearing down the loop. Best-effort; never raises."""
        try:
            if self._gen_thread is not None:
                self._join_generation(cancel=True)
        except Exception:
            self._gen_thread = None
        self._buf, self._buf_samples = [], 0
        self._turn_audio_all = []
        self._reset_barge_ctx()
        try:
            self.detector.reset()
        except Exception:
            pass
        try:
            self.engine.reset_for_new_turn()
        except Exception:
            pass
        self.state = "IDLE"
        self._emit("recovered", state=self.state)

    def _finish_turn_natural(self):
        self._join_generation(cancel=False)
        self.state = "IDLE"
        self.detector.reset()
        self._emit("turn_complete", turn=self._turn_idx,
                   audio_chunks_out=self.turns[-1]["audio_chunks_out"])

    # --------------------------------------------------------------- barge ctx
    def _push_barge_ctx(self, frame):
        self._barge_ctx.append(frame)
        self._barge_ctx_samples += frame.size
        while self._barge_ctx_samples > self._barge_ctx_max and len(self._barge_ctx) > 1:
            drop = self._barge_ctx.pop(0)
            self._barge_ctx_samples -= drop.size

    def _reset_barge_ctx(self):
        self._barge_ctx, self._barge_ctx_samples = [], 0

    # --------------------------------------------------------------- events
    def _emit(self, etype, **fields):
        evt = {"t": time.time(), "type": etype, "state": self.state}
        evt.update(fields)
        if self.on_event:
            self.on_event(evt)
