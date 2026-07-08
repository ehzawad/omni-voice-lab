#!/usr/bin/env python3
"""Unified ASR dispatcher over 5 English ASR backends.

    transcribe(model_key, wav_path, language="en") -> dict(text, raw, features, ...)

Two backend families live in two venvs (their deps conflict), so a given key only works in the
matching venv:

  .venv-qwen-asr  (transformers 5.x):  qwen3-asr-1.7b, qwen3-asr-0.6b
  .venv-funasr    (funasr 1.3.x):      sensevoice, paraformer-zh, funasr-nano

transcribe() lazily imports the backend and raises a clear error if you call a key in the wrong
venv. Models are cached module-side so a bench can reuse a load across prompts.
"""
import re
import time

# key -> metadata. family decides which venv / backend.
ASR_MODELS = {
    "qwen3-asr-1.7b": {"family": "qwen", "hf_id": "Qwen/Qwen3-ASR-1.7B-hf",
                       "note": "Qwen3-ASR 1.7B, transformers-native, 52 langs + language id"},
    "qwen3-asr-0.6b": {"family": "qwen", "hf_id": "Qwen/Qwen3-ASR-0.6B-hf",
                       "note": "Qwen3-ASR 0.6B, transformers-native, fast"},
    "sensevoice":     {"family": "funasr", "hf_id": "FunAudioLLM/SenseVoiceSmall",
                       "note": "SenseVoiceSmall, emits language + emotion + audio-event tags"},
    "paraformer-zh":  {"family": "funasr", "hf_id": "funasr/paraformer-zh",
                       "note": "Paraformer (Chinese-first); expected WEAK on English"},
    "funasr-nano":    {"family": "funasr", "hf_id": "FunAudioLLM/Fun-ASR-Nano-2512",
                       "note": "Fun-ASR-Nano-2512, newer LLM-ASR decoder path"},
}

_CACHE = {}
_LANG_NAME = {"en": "English", "zh": "Chinese"}

# ---------- Qwen3-ASR (transformers) ----------

def _load_qwen(hf_id):
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    processor = AutoProcessor.from_pretrained(hf_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16, device_map="cuda").eval()
    return {"processor": processor, "model": model}

def _run_qwen(obj, wav_path, language):
    import torch
    processor, model = obj["processor"], obj["model"]
    lang_hint = _LANG_NAME.get(language, language)
    inputs = processor.apply_transcription_request(audio=wav_path, language=lang_hint)
    inputs = inputs.to(model.device, model.dtype)
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256)
    gen = out[:, n_in:]
    raw = processor.decode(gen)[0]
    parsed = processor.decode(gen, return_format="parsed")[0]
    text = parsed.get("transcription", "").strip()
    feats = f"lang={parsed.get('language','?')}"
    return text, raw, feats

# ---------- funasr family ----------

def _load_funasr(hf_id):
    from funasr import AutoModel
    return AutoModel(model=hf_id, hub="hf", disable_update=True, device="cuda:0")

_TAG_RE = re.compile(r"<\|[^|]*\|>")

def _run_funasr(model, wav_path, language, is_sensevoice):
    kwargs = dict(input=wav_path, cache={}, batch_size_s=60)
    if is_sensevoice:
        kwargs.update(language=language, use_itn=True)
    else:
        # paraformer/nano accept a language hint on some builds; pass defensively
        try:
            res = model.generate(**dict(kwargs, language=language))
        except TypeError:
            res = model.generate(**kwargs)
        raw = (res[0].get("text", "") if res else "").strip()
        clean = _TAG_RE.sub("", raw).strip()
        return clean, raw, ""
    res = model.generate(**kwargs)
    raw = (res[0].get("text", "") if res else "").strip()
    # SenseVoice raw looks like: <|en|><|NEUTRAL|><|Speech|><|withitn|>the text
    tags = _TAG_RE.findall(raw)
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    clean = rich_transcription_postprocess(raw)
    feats = "tags=" + ",".join(t.strip("<|>") for t in tags) if tags else ""
    return clean, raw, feats

# ---------- unified entry ----------

def load(model_key):
    if model_key in _CACHE:
        return _CACHE[model_key]
    if model_key not in ASR_MODELS:
        raise KeyError(f"unknown ASR key '{model_key}'. Known: {list(ASR_MODELS)}")
    meta = ASR_MODELS[model_key]
    if meta["family"] == "qwen":
        obj = _load_qwen(meta["hf_id"])
    else:
        obj = _load_funasr(meta["hf_id"])
    _CACHE[model_key] = obj
    return obj

def transcribe(model_key, wav_path, language="en"):
    """Return dict(text, raw, features, latency_s, model_key)."""
    meta = ASR_MODELS[model_key]
    obj = load(model_key)
    t0 = time.time()
    if meta["family"] == "qwen":
        text, raw, feats = _run_qwen(obj, wav_path, language)
    else:
        text, raw, feats = _run_funasr(obj, wav_path, language,
                                       is_sensevoice=(model_key == "sensevoice"))
    dt = time.time() - t0
    return {"model_key": model_key, "text": text, "raw": raw,
            "features": feats, "latency_s": round(dt, 3)}


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "qwen3-asr-1.7b"
    wav = sys.argv[2] if len(sys.argv) > 2 else "examples/in_fifa_question.wav"
    r = transcribe(key, wav)
    print(r)
