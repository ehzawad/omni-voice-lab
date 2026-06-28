#!/usr/bin/env python3
"""FIFA-expert voice pipeline: spoken question -> ASR -> RAG retrieve -> grounded voice answer.

MiniCPM-o 4.5 (int4) for ASR + answer; Qwen3-Embedding for retrieval. Runs on the free A5000.
"""
import argparse, time, librosa, numpy as np, torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from fifa_rag import FifaRetriever

MODEL_ID = "openbmb/MiniCPM-o-4_5"
KEEP_FP = ["vpm", "apm", "resampler", "tts", "audio", "vision", "Token2wav", "embed", "tokenizer"]
VC_SUFFIX = "Speak in this voice style, in a warm, concise, expert tone."

ap = argparse.ArgumentParser()
ap.add_argument("--audio", required=True)
ap.add_argument("--ref", default="examples/ref_female.wav")
ap.add_argument("--out", default="fifa_answer.wav")
ap.add_argument("--k", type=int, default=3)
args = ap.parse_args()

# 1) retriever (small, load first)
retr = FifaRetriever(device="cuda")

# 2) MiniCPM-o int4
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=KEEP_FP)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", quantization_config=bnb, device_map="cuda").eval()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model.init_tts()

q_audio, _ = librosa.load(args.audio, sr=16000, mono=True)
ref_audio, _ = librosa.load(args.ref, sr=16000, mono=True)

# 3) ASR: transcribe the spoken question (text needed to retrieve)
asr = model.chat(msgs=[{"role": "user", "content": [q_audio, "Transcribe the user's question exactly, text only."]}],
                 tokenizer=tok, generate_audio=False, max_new_tokens=64)
asr = asr if isinstance(asr, str) else str(asr)
print(f"\n[ASR] {asr.strip()}")

# 4) retrieve grounding facts
hits = retr.retrieve(asr.strip(), k=args.k)
facts = "\n".join(f"- {c}" for _, c in hits)
print("[RETRIEVED]")
for s, c in hits:
    print(f"  [{s:.3f}] {c[:80]}...")

# 5) grounded voice answer (facts injected into system prompt, voice cloned)
sys_content = [("You are a knowledgeable FIFA and football expert. Answer the user's spoken question "
                "accurately and concisely using ONLY these verified facts:\n" + facts +
                "\nIf the facts don't cover it, say so briefly."),
               "Clone the voice in the provided audio prompt.", ref_audio, VC_SUFFIX]
msgs = [{"role": "system", "content": sys_content}, {"role": "user", "content": [q_audio]}]
t0 = time.time()
ans = model.chat(msgs=msgs, tokenizer=tok, use_tts_template=True, generate_audio=True,
                 output_audio_path=args.out, max_new_tokens=200)
print(f"\n[ANSWER] {ans if isinstance(ans, str) else ans}")
print(f"[wrote] {args.out}  [gen] {time.time()-t0:.1f}s  [VRAM] {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
