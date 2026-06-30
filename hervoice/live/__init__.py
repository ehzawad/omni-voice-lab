"""HerVoice LIVE: a VAD-gated streaming turn loop with barge-in (interruption).

This is NOT true simultaneous Omni-Flow full duplex. It is a fast turn-based
streaming loop on ONE resident MiniCPM-o 4.5 process (GPU0): Silero VAD gates
end-of-turn and detects barge-in; on barge-in the in-flight generator is
cooperatively cancelled and the session is reset for a fresh turn.

Public surface:
    LiveVoiceEngine  (engine.py)   -- one resident model; prefill / generate / reset / asr
    TurnDetector     (turn_detector.py) -- Silero VAD; speech_start / speech_end events
    LiveLoop         (loop.py)     -- state machine wiring detector + engine + cancel_event

Proof (headless): hervoice.live.simulate_bargein
Real mic (user box only): hervoice.live.local_mic_client
"""

from .engine import LiveVoiceEngine
from .turn_detector import TurnDetector, VadEvent
from .loop import LiveLoop

__all__ = ["LiveVoiceEngine", "TurnDetector", "VadEvent", "LiveLoop"]
