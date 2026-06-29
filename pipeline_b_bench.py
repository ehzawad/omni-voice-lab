#!/usr/bin/env python3
"""Pipeline B (MODULAR) benchmark runner: faster-whisper ASR -> Qwen3-8B brain -> Chatterbox TTS.

Run in .venv-modular. Writes results_b.json. Latency is measured from "full user audio available"
to "first response audio". Chatterbox here is batch (non-streaming), so first-audio == total pipeline
time; this is reported honestly as a property of this implementation.
"""
import json, time, os, torch, torchaudio, soundfile as sf
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from chatterbox.tts import ChatterboxTTS
import bench_common as bc

# NOTE: ideally Qwen3-8B to match MiniCPM-o's 8B backbone, but on this shared box only ~14.8GB
# was free and the 8B int4 load-time spike OOMs. Using Qwen2.5-3B-Instruct so the run completes;
# the brain-size asymmetry is flagged in docs/COMPARISON.md. Swap back to Qwen3-8B on a free GPU.
BRAIN_ID = "Qwen/Qwen2.5-3B-Instruct"
DEV = "cuda"

print("[load] faster-whisper (CPU, to leave GPU room for the 8B brain on a contended box)")
asr = WhisperModel("base.en", device="cpu", compute_type="int8")

print(f"[load] brain {BRAIN_ID} (int4)")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
brain = AutoModelForCausalLM.from_pretrained(BRAIN_ID, quantization_config=bnb, device_map=DEV).eval()
btok = AutoTokenizer.from_pretrained(BRAIN_ID)

print("[load] Chatterbox TTS")
tts = ChatterboxTTS.from_pretrained(device=DEV)

def transcribe(path):
    t0 = time.time()
    segs, _ = asr.transcribe(path, language="en")
    text = " ".join(s.text for s in segs).strip()
    return text, time.time() - t0

def think(question):
    msgs = [{"role": "system", "content": bc.SYSTEM_PROMPT}, {"role": "user", "content": question}]
    text = btok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = btok(text, return_tensors="pt").to(DEV)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.inference_mode():
        out = brain.generate(**inputs, max_new_tokens=bc.MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=btok.eos_token_id)
    ans = btok.decode(out[0, n_in:], skip_special_tokens=True).strip()
    return ans, time.time() - t0

def speak(text, out_path):
    t0 = time.time()
    wav = tts.generate(text, audio_prompt_path=bc.REF_VOICE)
    dt = time.time() - t0
    torchaudio.save(out_path, wav, tts.sr)
    dur = wav.shape[-1] / tts.sr
    return dur, dt

results = []
for pid, audio, ref in bc.PROMPTS:
    torch.cuda.reset_peak_memory_stats()
    hyp, t_asr = transcribe(audio)
    ans, t_llm = think(hyp)
    out_wav = os.path.join(bc.OUTDIR, f"out_b_{pid}.wav")
    dur, t_tts = speak(ans, out_wav)
    total = t_asr + t_llm + t_tts
    vram = torch.cuda.max_memory_allocated() / 1e9
    row = dict(id=pid, asr_hyp=hyp, answer=ans, t_asr=round(t_asr,3), t_llm=round(t_llm,3),
               t_tts=round(t_tts,3), first_audio_s=round(total,3), total_s=round(total,3),
               out_audio_s=round(dur,3), rtf=round(total/max(dur,1e-6),3), vram_gb=round(vram,2))
    print(f"[{pid}] asr='{hyp[:50]}' | ans='{ans[:50]}' | first_audio={total:.2f}s rtf={row['rtf']} vram={row['vram_gb']}GB")
    results.append(row)

with open(os.path.join(bc.ROOT, "results_b.json"), "w") as f:
    json.dump({"pipeline": f"B-modular (faster-whisper base.en [CPU] + {BRAIN_ID} int4 + Chatterbox)",
               "brain": BRAIN_ID, "rows": results}, f, indent=2)
print("\n[wrote] results_b.json")
