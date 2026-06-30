#!/usr/bin/env python3
"""Silero-VAD turn detector for the live loop.

Consumes 16k mono float32 frames of ANY size (mic frame size is decoupled from
the VAD window). Internally re-chunks into the exact 512-sample windows Silero
needs at 16 kHz (~32 ms) and runs a small hysteresis state machine that emits:

    speech_start  -- onset confirmed after >= min_speech_ms of voiced frames
    speech_end    -- offset confirmed after >= min_silence_ms of silence while
                     in speech

The loop interprets a `speech_start` during GENERATING as a BARGE-IN. The
detector itself stays state-machine-pure and event-only; it never talks to the
big model. Thresholds are configurable.
"""
import numpy as np
import torch

VAD_SR = 16000
VAD_WINDOW = 512                 # Silero requires exactly 512 samples @ 16k (~32 ms)
VAD_FRAME_MS = VAD_WINDOW / VAD_SR * 1000.0


class VadEvent:
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"

    def __init__(self, kind, prob):
        self.kind = kind
        self.prob = float(prob)

    def __repr__(self):
        return f"VadEvent({self.kind}, p={self.prob:.2f})"


class TurnDetector:
    """Streaming Silero-VAD onset/offset detector with hysteresis.

    Args:
        threshold:       per-window speech probability cutoff (0..1).
        min_speech_ms:   voiced run required to confirm a speech onset.
        min_silence_ms:  silence run required to confirm end-of-turn.
        sampling_rate:   must be 16000 for this loop.
    """

    def __init__(self, threshold=0.5, min_speech_ms=120, min_silence_ms=220,
                 sampling_rate=VAD_SR, model=None):
        assert sampling_rate == VAD_SR, "live loop is 16k mono only"
        if model is None:
            from silero_vad import load_silero_vad
            model = load_silero_vad()
        self.model = model
        self.model.eval()
        self.threshold = float(threshold)
        self.min_speech_ms = float(min_speech_ms)
        self.min_silence_ms = float(min_silence_ms)
        self.sr = sampling_rate
        self._resid = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self):
        """Clear VAD RNN state + hysteresis counters (call between sessions)."""
        try:
            self.model.reset_states()
        except Exception:
            pass
        self._resid = np.zeros(0, dtype=np.float32)
        self.triggered = False        # currently inside a speech segment
        self.speech_ms = 0.0          # voiced run while not yet triggered
        self.silence_ms = 0.0         # silence run while triggered
        self.last_prob = 0.0

    @torch.no_grad()
    def _prob(self, window512):
        t = torch.from_numpy(np.ascontiguousarray(window512, dtype=np.float32))
        return float(self.model(t, self.sr).item())

    def process(self, frame):
        """Feed one (arbitrary-length) 16k mono frame; return a list of VadEvent.

        A single frame can yield 0, 1 or 2 events (e.g. a 50 ms frame spans ~1.5
        VAD windows), so the loop must iterate the returned list.
        """
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        buf = np.concatenate([self._resid, frame]) if self._resid.size else frame
        events = []
        i = 0
        n = buf.size
        while i + VAD_WINDOW <= n:
            win = buf[i:i + VAD_WINDOW]
            i += VAD_WINDOW
            p = self._prob(win)
            self.last_prob = p
            voiced = p >= self.threshold
            if not self.triggered:
                if voiced:
                    self.speech_ms += VAD_FRAME_MS
                    if self.speech_ms >= self.min_speech_ms:
                        self.triggered = True
                        self.silence_ms = 0.0
                        events.append(VadEvent(VadEvent.SPEECH_START, p))
                else:
                    self.speech_ms = 0.0
            else:
                if voiced:
                    self.silence_ms = 0.0
                else:
                    self.silence_ms += VAD_FRAME_MS
                    if self.silence_ms >= self.min_silence_ms:
                        self.triggered = False
                        self.speech_ms = 0.0
                        events.append(VadEvent(VadEvent.SPEECH_END, p))
        self._resid = buf[i:].copy()
        return events
