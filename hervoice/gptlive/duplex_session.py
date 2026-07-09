#!/usr/bin/env python3
"""duplex_session.py -- the CONTINUOUS full-duplex session engine for ARCH 3.

ARCH 3 is the open local analog of GPT-Live's continuous mode: a full-duplex
front (Moshi) that listens and speaks at the same time, decides when to speak,
handles barge-in, and DELEGATES hard reasoning to a stronger model (gpt-oss-20B)
without freezing the audio loop.

Everything below happens inside ONE streaming context
(`mimi.streaming(1)` + `lm_gen.streaming(1)`) driven frame by frame at 12.5 Hz
(Mimi = 80 ms frames, 24 kHz). Per frame we log: input frame index, output
frame index, wall-clock ms, phase, user_audio_rms_in, assistant_audio_rms_out,
and the forced text token / piece. No claim in ARCH 3 is asserted -- it is read
back out of these frame logs.

Teacher-forcing (stated narrowly, reused verbatim from front.py):
    Moshi is a full-duplex front; the delegated answer text is rendered back
    through Moshi by teacher-forcing its inner-monologue text stream
    (`on_text_hook` overwrites the sampled text token in place before the
    depformer vocalises it), so the ACOUSTICS are Moshi's but the WORDS are the
    brain's. This is NOT Moshi autonomously deciding to say the answer.

The four behaviours the engine can exercise (selected by params):
  1. OVERLAP    : input audio can carry energy in the SAME frames the assistant
                  emits answer audio (native duplex; no VAD gate).
  2. ASYNC      : when the controller delegates, the gpt-oss call runs on a
     DELEGATION background thread; the Moshi loop never blocks. Moshi backchannels
                  while the brain reasons, then splices the streamed answer
                  tokens into the teacher-force schedule and vocalises them
                  incrementally.
  3. BARGE-IN   : an energy threshold on the input stream, armed while the
                  assistant is speaking, aborts the forced answer mid-utterance.
  4. GAP SWEEP  : the `gap` PAD-frames-per-word-piece pacing is a plain
                  parameter, so it can be swept and measured (prefill mode).
"""
import collections
import threading
import time

import numpy as np
import torch

from moshi.models import LMGen

from . import delegate_oss

PAD_TOKEN = 0
SPECIAL_TOKENS = {0, 3}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


class BrainStream:
    """Runs delegate_oss.ask_stream on a background thread and commits whole
    answer word-pieces into a thread-safe deque as they arrive."""

    def __init__(self, question: str, tokenizer, max_tokens: int = 512):
        self.question = question
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.committed = collections.deque()   # answer token ids, FIFO
        self.reasoning_deltas = 0
        self.reasoning_chars = 0
        self.full_answer = ""
        self.done = False
        self.error = None
        self.first_content_wall = None
        self.start_wall = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.start_wall = time.time()
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout)

    def _run(self):
        acc = ""
        pushed = 0
        try:
            for ch, delta in delegate_oss.ask_stream(self.question,
                                                     max_tokens=self.max_tokens):
                if ch == "reasoning":
                    self.reasoning_deltas += 1
                    self.reasoning_chars += len(delta)
                    continue
                # content channel = the spoken answer
                if self.first_content_wall is None:
                    self.first_content_wall = time.time()
                acc += delta
                idx = acc.rfind(" ")
                if idx > 0:
                    commit = acc[:idx]
                    ids = self.tokenizer.encode(commit)
                    for tid in ids[pushed:]:
                        self.committed.append(tid)
                    pushed = len(ids)
            # flush the final partial word
            final = acc.strip()
            ids = self.tokenizer.encode(final)
            for tid in ids[pushed:]:
                self.committed.append(tid)
            self.full_answer = final
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        finally:
            self.done = True


class DuplexSession:
    """Continuous full-duplex driver over an already-loaded MoshiFront."""

    def __init__(self, front, log=print):
        self.front = front
        self.log = log
        self.fr = front.frame_rate
        self.fs = front.frame_size
        self.sr = front.sample_rate

    def id_to_piece(self, tid: int) -> str:
        return self.front.text_tokenizer.id_to_piece(tid).replace("▁", " ")

    @torch.no_grad()
    def run(self, params):
        """Run one continuous session.

        params keys:
          input_timeline : list of input audio chunks (torch [1,C,fs]); frames
                           past the end are silence.
          n_prime_frames : frames before delegation fires (LISTEN / priming).
          brain_mode     : 'stream'  -> async gpt-oss delegation (behaviours 1-3)
                           'prefill' -> answer known up front (gap sweep).
          question       : user question (brain_mode='stream').
          prefill_text   : answer text to force (brain_mode='prefill').
          gap            : PAD frames per forced word-piece.
          backchannel    : filler text forced while waiting for the brain.
          bargein        : None or dict(rms_thresh, sustain, arm_from_frame).
          tail_pad       : PAD frames after the last answer word.
          max_frames     : hard safety cap.
        Returns a dict with the full session audio, per-frame records, and the
        measured timeline markers.
        """
        front = self.front
        gap = params.get("gap", 2)
        tail_pad = params.get("tail_pad", 10)
        max_frames = params.get("max_frames", 4000)
        timeline = params["input_timeline"]
        n_prime = params["n_prime_frames"]
        brain_mode = params.get("brain_mode", "stream")
        bargein = params.get("bargein")
        backchannel_text = params.get("backchannel", "let me think.")

        silence = torch.zeros((1, front.mimi.channels, self.fs),
                              device=front.device)

        # ---- brain source ------------------------------------------------
        brain = None
        prefill_committed = collections.deque()
        if brain_mode == "stream":
            brain = BrainStream(params["question"], front.text_tokenizer,
                                max_tokens=params.get("max_tokens", 512))
        else:
            for tid in front.text_tokenizer.encode(params["prefill_text"]):
                prefill_committed.append(tid)

        def committed():
            return brain.committed if brain is not None else prefill_committed

        def brain_done():
            return True if brain is None else brain.done

        bc_ids = front.text_tokenizer.encode(backchannel_text)

        # ---- teacher-force control shared with the hook ------------------
        ctl = {"force_id": None}

        def on_text_hook(text_token):
            fid = ctl["force_id"]
            if fid is not None:
                text_token[:] = fid   # teacher-force Moshi's inner monologue

        lm_gen = LMGen(front.lm, on_text_hook=on_text_hook,
                       **dict(front.lm_gen_config))

        # ---- session state ----------------------------------------------
        phase = "LISTEN"
        pad_countdown = 0
        bc_ptr = 0
        bc_pad = 0
        answer_started = False
        last_assistant_rms = 0.0
        sustain = 0

        markers = {
            "delegation_start_frame": None,
            "first_brain_token_frame": None,
            "answer_complete_frame": None,
            "interrupt_frame": None,
            "interrupt_detect_frame": None,
            "cutoff_frame": None,
        }
        frames_spoken_while_waiting = 0
        answer_word_frames = 0
        tail_left = None
        interrupt_tail = None

        session_audio = []
        records = []
        out_frame = -1
        t0 = time.time()

        with front.mimi.streaming(1), lm_gen.streaming(1):
            for i in range(max_frames):
                chunk = timeline[i] if i < len(timeline) else silence
                in_np = chunk[0, 0].float().cpu().numpy()
                user_rms_in = _rms(in_np)

                # -- phase transition: fire delegation once priming is done --
                if phase == "LISTEN" and i >= n_prime:
                    phase = "WAIT"
                    markers["delegation_start_frame"] = i
                    if brain is not None:
                        brain.start()

                # -- barge-in: energy threshold armed ONLY while the assistant is
                #    actually vocalising the answer (phase SPEAK), so the cutoff
                #    lands on the answer, robust to brain latency/length --
                if (bargein and phase == "SPEAK"
                        and i >= bargein.get("arm_from_frame", 0)):
                    if markers["interrupt_frame"] is None and user_rms_in > bargein["rms_thresh"]:
                        markers["interrupt_frame"] = i
                    if user_rms_in > bargein["rms_thresh"]:
                        sustain += 1
                    else:
                        sustain = 0
                    if (sustain >= bargein["sustain"]
                            and markers["cutoff_frame"] is None):
                        markers["interrupt_detect_frame"] = i
                        markers["cutoff_frame"] = i
                        phase = "INTERRUPTED"
                        interrupt_tail = 8   # a few frames to fall silent

                # -- decide the forced text token for THIS frame -------------
                force = None
                piece_kind = "none"
                if phase == "LISTEN":
                    force = None
                elif phase == "INTERRUPTED":
                    force = None            # abort forced speech; yield to user
                    piece_kind = "yield"
                    if interrupt_tail is not None:
                        interrupt_tail -= 1
                elif phase in ("WAIT", "SPEAK"):
                    cq = committed()
                    if pad_countdown > 0:
                        pad_countdown -= 1
                        force = PAD_TOKEN
                        piece_kind = "pad"
                    elif len(cq) > 0:
                        force = cq.popleft()
                        pad_countdown = gap
                        piece_kind = "answer"
                        answer_word_frames += 1
                        if not answer_started:
                            answer_started = True
                            markers["first_brain_token_frame"] = i
                            phase = "SPEAK"
                    elif not answer_started and not brain_done():
                        # still waiting on the brain -> keep the stream alive
                        if bc_pad > 0:
                            bc_pad -= 1
                            force = PAD_TOKEN
                            piece_kind = "bc_pad"
                        elif bc_ptr < len(bc_ids):
                            force = bc_ids[bc_ptr]
                            bc_ptr += 1
                            bc_pad = gap
                            piece_kind = "backchannel"
                        else:
                            force = PAD_TOKEN
                            piece_kind = "wait_pad"
                        frames_spoken_while_waiting += 1
                    else:
                        # answer stream drained
                        if brain_done() and len(cq) == 0:
                            if answer_started and markers["answer_complete_frame"] is None:
                                markers["answer_complete_frame"] = i
                                tail_left = tail_pad
                            if tail_left is not None:
                                tail_left -= 1
                        force = PAD_TOKEN
                        piece_kind = "tail"

                ctl["force_id"] = force

                # -- step Moshi (hook fires inside) --------------------------
                codes = front.mimi.encode(chunk)
                if i == 0:
                    _ = lm_gen.step(codes)   # warm-up (matches proven pattern)
                tokens = lm_gen.step(codes)
                if tokens is None:
                    continue                 # inside the output-delay window
                out_frame += 1
                text_tok = int(tokens[0, 0].item())
                wav = front.mimi.decode(tokens[:, 1:])[0, 0].float().cpu().numpy()
                a_rms = _rms(wav)
                last_assistant_rms = a_rms
                session_audio.append(wav)

                piece = "" if force is None else self.id_to_piece(force)
                records.append({
                    "in_frame": i,
                    "out_frame": out_frame,
                    "wall_ms": round((time.time() - t0) * 1000.0, 1),
                    "phase": phase,
                    "user_rms_in": round(user_rms_in, 5),
                    "assistant_rms_out": round(a_rms, 5),
                    "forced_token": (None if force is None else int(force)),
                    "piece_kind": piece_kind,
                    "piece": piece,
                    "moshi_text_tok": text_tok,
                })

                # -- termination -------------------------------------------
                if phase == "INTERRUPTED" and interrupt_tail is not None and interrupt_tail <= 0:
                    break
                if (phase in ("WAIT", "SPEAK") and answer_started
                        and brain_done() and len(committed()) == 0
                        and tail_left is not None and tail_left <= 0):
                    break

        dt = time.time() - t0
        audio = (np.concatenate(session_audio) if session_audio
                 else np.zeros(1, dtype=np.float32))

        # forced answer pieces actually vocalised (for a quick text read-back)
        answer_pieces = [r["piece"] for r in records
                         if r["piece_kind"] == "answer"]
        forced_answer_text = "".join(answer_pieces).strip()

        result = {
            "audio": audio,
            "records": records,
            "markers": markers,
            "frames_spoken_while_waiting": frames_spoken_while_waiting,
            "answer_word_frames": answer_word_frames,
            "forced_answer_text": forced_answer_text,
            "wall_seconds": round(dt, 2),
            "out_frames": out_frame + 1,
            "duration_s": round(len(audio) / self.sr, 2),
        }
        if brain is not None:
            result["_brain_obj"] = brain
            result["brain"] = {
                "full_answer": brain.full_answer,
                "reasoning_deltas": brain.reasoning_deltas,
                "reasoning_chars": brain.reasoning_chars,
                "error": brain.error,
                "first_content_latency_s": (
                    round(brain.first_content_wall - brain.start_wall, 3)
                    if brain.first_content_wall else None),
            }
        return result
