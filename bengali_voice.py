#!/usr/bin/env python3
"""Bengali voice-to-voice (MODULAR): MiniCPM-o understands Bengali + answers in Bengali TEXT,
then a dedicated Bengali TTS (facebook/mms-tts-ben) speaks it. This sidesteps MiniCPM-o's
broken Bengali speech-out (EN/ZH talker) while keeping its smart multilingual brain.
"""
import argparse, time, librosa, numpy as np, soundfile as sf, torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, VitsModel

MODEL_ID = "openbmb/MiniCPM-o-4_5"
KEEP_FP = ["vpm", "apm", "resampler", "tts", "audio", "vision", "Token2wav", "embed", "tokenizer"]

ap = argparse.ArgumentParser()
ap.add_argument("--audio", required=True)
ap.add_argument("--out", default="bengali_answer.wav")
args = ap.parse_args()

# 1) MiniCPM-o int4 = Bengali brain (ASR + reason + Bengali text reply)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=KEEP_FP)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", quantization_config=bnb, device_map="cuda").eval()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

q_audio, _ = librosa.load(args.audio, sr=16000, mono=True)
sys_msg = {"role": "system", "content": ["You are a helpful assistant. Always reply in Bengali (Bangla) "
           "using Bengali script, in one concise sentence."]}
t0 = time.time()
ans = model.chat(msgs=[sys_msg, {"role": "user", "content": [q_audio]}], tokenizer=tok,
                 generate_audio=False, max_new_tokens=96)
ans = (ans if isinstance(ans, str) else str(ans)).strip()
print(f"[Bengali text answer] {ans}   ({time.time()-t0:.1f}s)")

del model; torch.cuda.empty_cache()

# 2) dedicated Bengali TTS speaks the answer
tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
tts = VitsModel.from_pretrained("facebook/mms-tts-ben").eval().cuda()
with torch.no_grad():
    wav = tts(**tts_tok(ans, return_tensors="pt").to("cuda")).waveform[0].cpu().numpy()
wav = wav / (np.abs(wav).max() + 1e-8) * 0.8
sf.write(args.out, wav, tts.config.sampling_rate)
print(f"[wrote] {args.out}  {len(wav)/tts.config.sampling_rate:.1f}s @ {tts.config.sampling_rate}Hz")
print(f"[VRAM] peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
