#!/usr/bin/env python3
"""MiniCPM-o 4.5 STREAMING demo: prefill audio in chunks, stream audio+text out.

Measures time-to-first-audio-packet (the realtime metric) and writes the streamed reply.
This exercises the same engine the full-duplex server (model.as_duplex) sits on top of.
"""
import argparse, time, librosa, numpy as np, soundfile as sf, torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
MODEL_ID = "openbmb/MiniCPM-o-4_5"
# keep audio encoder / TTS / vision in full precision; only 4-bit the LLM
KEEP_FP = ["vpm", "apm", "resampler", "tts", "audio", "vision", "Token2wav", "embed", "tokenizer"]
VC_SUFFIX = ("Please assist users while maintaining this voice style. Answer seriously and in high quality. "
             "Chat in a highly human-like, oral style. You are a helpful assistant.")

ap = argparse.ArgumentParser()
ap.add_argument("--audio", required=True, help="user question audio")
ap.add_argument("--ref", required=True, help="reference voice clip to clone")
ap.add_argument("--out", default="minicpm_stream_reply.wav")
ap.add_argument("--chunk-ms", type=int, default=1000)
ap.add_argument("--quant", choices=["none", "int4"], default="none")
args = ap.parse_args()

load_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
if args.quant == "int4":
    load_kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=KEEP_FP)
    load_kw["device_map"] = "cuda"
    model = AutoModel.from_pretrained(MODEL_ID, **load_kw).eval()
else:
    model = AutoModel.from_pretrained(MODEL_ID, **load_kw).eval().cuda()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model.init_tts()

sid = "hervoice-stream-1"
ref_audio, _ = librosa.load(args.ref, sr=16000, mono=True)
user_audio, _ = librosa.load(args.audio, sr=16000, mono=True)

# streaming TTS needs the voice cache built from the reference clip up front
model.init_token2wav_cache(ref_audio)

# 1) prefill system message (voice clone)
model.streaming_prefill(session_id=sid,
    msgs=[{"role": "system", "content": ["Clone the voice in the provided audio prompt.", ref_audio, VC_SUFFIX]}],
    tokenizer=tok)

# 2) prefill the user audio in fixed chunks (simulating mic stream)
step = int(16000 * args.chunk_ms / 1000)
n = max(1, (len(user_audio) + step - 1) // step)
for i in range(n):
    ch = user_audio[i*step:(i+1)*step]
    model.streaming_prefill(session_id=sid,
        msgs=[{"role": "user", "content": [ch]}], is_last_chunk=(i == n-1), tokenizer=tok)

# 3) stream the reply out; measure first-packet latency
t0 = time.time()
first_pkt = None
waves, text_acc = [], ""
for wav_chunk, new_text in model.streaming_generate(session_id=sid, generate_audio=True, tokenizer=tok):
    if wav_chunk is not None and len(wav_chunk) > 0:
        if first_pkt is None:
            first_pkt = time.time() - t0
        waves.append(wav_chunk.reshape(-1).float().cpu().numpy() if torch.is_tensor(wav_chunk) else np.asarray(wav_chunk).reshape(-1))
    if new_text:
        text_acc += new_text

total = time.time() - t0
if waves:
    sf.write(args.out, np.concatenate(waves), samplerate=24000)
print(f"\n[TEXT] {text_acc.strip()}")
print(f"[first-audio-packet] {first_pkt*1000:.0f} ms" if first_pkt else "[no audio]")
print(f"[total stream] {total:.2f}s   chunks={len(waves)}   -> {args.out}")
print(f"[VRAM] peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
