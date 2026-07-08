#!/usr/bin/env python3
"""Qwen3-TTS wrapper: text -> wav. Runs in .venv-qwen-audio (qwen-tts + transformers 4.57.3).

Qwen3-TTS-12Hz-1.7B-Base is a *voice-clone* base model: give it a reference clip (ref_audio)
and, ideally, that clip's transcript (ref_text). Without a transcript, pass x_vector_only_mode=
True to clone from the speaker embedding alone (lower fidelity but no transcript needed).

We default the reference to examples/ref_female.wav (the repo's stock female voice).
"""
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REF = os.path.join(ROOT, "examples", "ref_female.wav")
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

_MODEL = None


def load(model_id: str = MODEL_ID):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    from qwen_tts import Qwen3TTSModel
    # flash_attention_2 is recommended but may be absent; sdpa is always available.
    try:
        m = Qwen3TTSModel.from_pretrained(model_id, device_map="cuda:0",
                                          dtype=torch.bfloat16,
                                          attn_implementation="flash_attention_2")
    except Exception:
        m = Qwen3TTSModel.from_pretrained(model_id, device_map="cuda:0",
                                          dtype=torch.bfloat16,
                                          attn_implementation="sdpa")
    _MODEL = m
    return m


def synth(text, out_path, ref_audio: str = DEFAULT_REF, ref_text: str | None = None,
          language: str = "English"):
    """Synthesize `text` in the reference voice, write a wav, return dict(out_path, sr, duration_s, latency_s)."""
    import soundfile as sf
    model = load()
    t0 = time.time()
    if ref_text:
        wavs, sr = model.generate_voice_clone(text=text, language=language,
                                              ref_audio=ref_audio, ref_text=ref_text)
    else:
        # No transcript for the reference -> speaker-embedding-only clone.
        wavs, sr = model.generate_voice_clone(text=text, language=language,
                                              ref_audio=ref_audio, x_vector_only_mode=True)
    dt = time.time() - t0
    import numpy as np
    wav = np.asarray(wavs[0], dtype="float32").reshape(-1)
    dur = len(wav) / sr if sr else 0.0
    rms = float(np.sqrt((wav.astype("float64") ** 2).mean())) if wav.size else 0.0
    # Guard against fake success: never report a valid wav for empty / too-short /
    # silent (degenerate) audio. Return an explicit tts_failed status and write no file.
    if wav.size == 0 or dur < 0.2 or rms < 0.005:
        return {"out_path": None, "sr": sr, "duration_s": round(dur, 3),
                "rms": round(rms, 4), "latency_s": round(dt, 3),
                "status": "tts_failed",
                "reason": f"degenerate audio (dur={dur:.2f}s rms={rms:.4f})"}
    sf.write(out_path, wav, sr)
    return {"out_path": out_path, "sr": sr, "duration_s": round(dur, 3),
            "rms": round(rms, 4), "latency_s": round(dt, 3), "status": "ok"}


if __name__ == "__main__":
    import sys
    txt = sys.argv[1] if len(sys.argv) > 1 else "Brazil has won the men's World Cup five times."
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "runs", "modular", "tts_smoke.wav")
    print(synth(txt, out))
