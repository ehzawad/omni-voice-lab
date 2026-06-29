#!/usr/bin/env python3
"""Pipeline A (SINGLE-NETWORK) benchmark runner: MiniCPM-o 4.5 int4.

Run in .venv. Writes results_a.json. ASR transcript via a transcribe turn; the answer uses the
streaming path so first-audio latency and RTF are measured the same way as Pipeline B
(from full user audio available to first response audio).
"""
import json, time, os, librosa, numpy as np, soundfile as sf, torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import bench_common as bc

MODEL_ID = "openbmb/MiniCPM-o-4_5"
KEEP_FP = ["vpm","apm","resampler","tts","audio","vision","Token2wav","embed","tokenizer"]
VC_SUFFIX = "Please assist users while maintaining this voice style. Answer in one or two concise sentences."

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=KEEP_FP)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", quantization_config=bnb, device_map="cuda").eval()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model.init_tts()

ref_audio, _ = librosa.load(bc.REF_VOICE, sr=16000, mono=True)
model.init_token2wav_cache(ref_audio)

def asr(audio_np):
    t0 = time.time()
    r = model.chat(msgs=[{"role":"user","content":[audio_np, "Transcribe the user's question exactly, text only."]}],
                   tokenizer=tok, generate_audio=False, max_new_tokens=64)
    return (r if isinstance(r,str) else str(r)).strip(), time.time()-t0

def answer_stream(audio_np, sid, out_path):
    model.streaming_prefill(session_id=sid, tokenizer=tok,
        msgs=[{"role":"system","content":["Clone the voice in the provided audio prompt.", ref_audio, VC_SUFFIX]}])
    step = 16000
    n = max(1, (len(audio_np)+step-1)//step)
    for i in range(n):
        model.streaming_prefill(session_id=sid, tokenizer=tok,
            msgs=[{"role":"user","content":[audio_np[i*step:(i+1)*step]]}], is_last_chunk=(i==n-1))
    t0 = time.time(); first=None; waves=[]; text=""
    for wav_chunk, new_text in model.streaming_generate(session_id=sid, tokenizer=tok, generate_audio=True):
        if wav_chunk is not None and len(wav_chunk) > 0:
            if first is None: first = time.time()-t0
            waves.append(wav_chunk.reshape(-1).float().cpu().numpy() if torch.is_tensor(wav_chunk) else np.asarray(wav_chunk).reshape(-1))
        if new_text: text += new_text
    total = time.time()-t0
    if waves:
        sf.write(out_path, np.concatenate(waves), samplerate=24000)
        dur = sum(len(w) for w in waves)/24000
    else:
        dur = 0.0
    return text.strip(), first, total, dur

results = []
for pid, audio_path, ref in bc.PROMPTS:
    torch.cuda.reset_peak_memory_stats()
    a = librosa.load(audio_path, sr=16000, mono=True)[0]
    hyp, t_asr = asr(a)
    out_wav = os.path.join(bc.OUTDIR, f"out_a_{pid}.wav")
    text, first, total, dur = answer_stream(a, f"bench-{pid}", out_wav)
    vram = torch.cuda.max_memory_allocated()/1e9
    row = dict(id=pid, asr_hyp=hyp, answer=text, t_asr=round(t_asr,3),
               first_audio_s=round(first,3) if first else None, total_s=round(total,3),
               out_audio_s=round(dur,3), rtf=round(total/max(dur,1e-6),3), vram_gb=round(vram,2))
    print(f"[{pid}] asr='{hyp[:50]}' | ans='{text[:50]}' | first_audio={row['first_audio_s']}s rtf={row['rtf']} vram={row['vram_gb']}GB")
    results.append(row)

with open(os.path.join(bc.ROOT, "results_a.json"), "w") as f:
    json.dump({"pipeline": "A-single-network (MiniCPM-o 4.5 int4, streaming)", "rows": results}, f, indent=2)
print("\n[wrote] results_a.json")
