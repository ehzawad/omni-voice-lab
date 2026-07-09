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
    # Reject empty/whitespace input up front (defense-in-depth): the model will vocalize
    # something for "" that can pass the output-audio floor, so never synthesize on no text.
    if not (text or "").strip():
        return {"out_path": None, "status": "tts_failed", "reason": "empty input text",
                "duration_s": 0.0, "rms": 0.0, "latency_s": 0.0}
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


def synth_stream(text, out_prefix, ref_audio: str = DEFAULT_REF, ref_text: str | None = None,
                 language: str = "English"):
    """Sentence-chunked streaming synth: split `text`, synthesize each sentence with the resident
    model, write each chunk wav (`out_prefix_00.wav`, `_01.wav`, ...) AS SOON AS it is ready.

    Reuses synth() verbatim per chunk -- so every chunk keeps the same empty-text / degenerate-audio
    guards. A chunk that fails its guard is recorded as tts_failed and skipped (no fabrication) and
    does NOT abort the rest. Returns a dict with per-chunk records, the TTFA (cumulative latency when
    the FIRST valid chunk was written), the total, and a concatenated `out_prefix_full.wav` of all
    the ok chunks in order.

    This lowers *perceived* latency (first audio sooner), not total generation time -- generation is
    still autoregressive.
    """
    import numpy as np
    import soundfile as sf
    from hervoice.modular.chunk import split_sentences

    out_prefix = os.path.abspath(out_prefix)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    sentences = split_sentences(text)

    t0 = time.time()
    chunks = []
    ttfa = None
    concat = []
    sr_ref = None

    for idx, sent in enumerate(sentences):
        wav_path = f"{out_prefix}_{idx:02d}.wav"
        r = synth(sent, wav_path, ref_audio=ref_audio, ref_text=ref_text, language=language)
        cum = round(time.time() - t0, 3)
        entry = {"index": idx, "text": sent, "status": r.get("status"),
                 "duration_s": r.get("duration_s", 0.0), "cumulative_latency_s": cum}
        if r.get("status") == "ok":
            entry["wav"] = r["out_path"]
            if ttfa is None:
                ttfa = cum  # first valid chunk -> time-to-first-audio
            w, sr = sf.read(wav_path, dtype="float32")
            if sr_ref is None:
                sr_ref = sr
            if sr == sr_ref:
                concat.append(w)
        else:
            entry["wav"] = None
            entry["reason"] = r.get("reason")
        chunks.append(entry)

    full_path, full_dur = None, None
    if concat:
        full = np.concatenate(concat)
        full_path = f"{out_prefix}_full.wav"
        sf.write(full_path, full, sr_ref)
        full_dur = round(len(full) / sr_ref, 3) if sr_ref else 0.0

    total = round(time.time() - t0, 3)
    n_ok = sum(1 for c in chunks if c["status"] == "ok")
    return {
        "status": "ok" if n_ok > 0 else "tts_failed",
        "n_sentences": len(sentences), "n_ok": n_ok,
        "ttfa_s": ttfa, "total_s": total,
        "full_wav": full_path, "full_duration_s": full_dur, "sr": sr_ref,
        "chunks": chunks,
    }


if __name__ == "__main__":
    import sys
    txt = sys.argv[1] if len(sys.argv) > 1 else "Brazil has won the men's World Cup five times."
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "runs", "modular", "tts_smoke.wav")
    print(synth(txt, out))
