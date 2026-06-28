#!/usr/bin/env python3
"""Diagnose WHY MiniCPM-o 4.5 speaks Bengali badly.

Hypothesis: the TTS (talker) text tokenizer is EN/ZH-only, so Bengali script maps to UNK/empty
-> the talker gets no phonetic signal -> babble/repetition. We find the real TTS text tokenizer
by introspection, then test Bengali vs English coverage.
"""
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "openbmb/MiniCPM-o-4_5"
KEEP_FP = ["vpm","apm","resampler","tts","audio","vision","Token2wav","embed","tokenizer"]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=KEEP_FP)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", quantization_config=bnb, device_map="cuda").eval()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model.init_tts()
try:
    model.prepare_processor(tokenizer=tok)
except Exception as e:
    print("[prepare_processor]", e)

def has_encode(o):
    return hasattr(o, "encode") and callable(getattr(o, "encode"))

# Introspect candidate holders for the TTS text tokenizer
print("=== introspect tts-related attributes ===")
candidates = {}
for holder_name in ["tts_processor", "tts", "processor"]:
    holder = getattr(model, holder_name, None)
    if holder is None:
        print(f"model.{holder_name} = None"); continue
    attrs = [a for a in dir(holder) if not a.startswith("__")]
    toklike = [a for a in attrs if "token" in a.lower()]
    print(f"model.{holder_name} ({type(holder).__name__}) token-like attrs: {toklike}")
    for a in toklike:
        try:
            obj = getattr(holder, a)
            if has_encode(obj):
                candidates[f"{holder_name}.{a}"] = obj
        except Exception:
            pass

print(f"\n=== tokenizers with .encode found: {list(candidates.keys())} ===")

EN = "Dhaka is the capital of Bangladesh."
BN = "ঢাকা বাংলাদেশের রাজধানী।"
def probe(tag, t):
    print(f"\n##### {tag}  ({type(t).__name__}, vocab_size={getattr(t,'vocab_size','?')}) #####")
    for name, text in [("EN", EN), ("BN", BN)]:
        try:
            ids = t.encode(text, add_special_tokens=False) if "add_special_tokens" in t.encode.__code__.co_varnames else t.encode(text)
        except Exception:
            ids = t.encode(text)
        back = t.decode(ids) if hasattr(t, "decode") else "<no decode>"
        # which chars are LOST (not recoverable through the tokenizer)
        lost = []
        for c in dict.fromkeys(text):
            if not c.strip():
                continue
            try:
                cc = t.encode(c, add_special_tokens=False) if "add_special_tokens" in t.encode.__code__.co_varnames else t.encode(c)
                if c not in (t.decode(cc) if hasattr(t,"decode") else ""):
                    lost.append(c)
            except Exception:
                lost.append(c)
        print(f"  [{name}] chars={len(text)} tokens={len(ids)} decode={back!r}")
        print(f"       LOST chars (cannot be represented): {lost}")

for tag, t in candidates.items():
    probe(tag, t)

print("\n=== contrast: MAIN brain tokenizer (known to handle Bengali text) ===")
for name, text in [("EN", EN), ("BN", BN)]:
    ids = tok.encode(text, add_special_tokens=False)
    print(f"  [{name}] brain tokens={len(ids)} decode={tok.decode(ids)!r}")
