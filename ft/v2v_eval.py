#!/usr/bin/env python
"""STEP 3 - eval base vs base+LoRA on 20 held-out SPOKEN questions (audio-in).

Runs Qwen2.5-Omni thinker on the held-out spoken questions (REAL audio input) and
produces a TEXT answer (speech-out via the talker is left out of scope per plan; the
adapted path is listen+respond). Scores answer correctness against KB ground truth by
slot/keyword match (years, numbers, key entities) and reports a before/after delta.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
     .venv-qwenomni/bin/python ft/v2v_eval.py
"""
import json, os, re, time
import torch, librosa
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel

ROOT = "/mnt/sdb/arafat/hervoice"
MODEL = "Qwen/Qwen2.5-Omni-7B"
MANIFEST = os.path.join(ROOT, "data/v2v/manifest.jsonl")
ADAPTER_DIR = os.path.join(ROOT, "adapters/v2v_qwen25omni_lora")
SR = 16000
SYS = ("You are a concise FIFA expert. Answer the spoken question in one short, "
       "factual sentence.")

NUMWORDS = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6",
            "seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12",
            "thirteen":"13","fourteen":"14","fifteen":"15","sixteen":"16","seventeen":"17"}

STOP = {"the","a","an","in","on","of","and","or","to","is","was","by","with","not",
        "who","what","it","its","at","for","beat","won","win","wins","title","titles",
        "times","time","each","after","draw","final","held","penalties","penalty",
        "metres","metre","yards","yard","minutes","minute","goals","goal","one","short",
        "factual","sentence","co-hosted","still","then","decides","decided","first",
        "record","scored","record."}

def norm(s):
    s = s.lower()
    for w, d in NUMWORDS.items():
        s = re.sub(rf"\b{w}\b", d, s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def keywords(answer):
    n = norm(answer)
    toks = n.split()
    keys = set()
    for t in toks:
        if t.isdigit():
            keys.add(t)
        elif len(t) >= 4 and t not in STOP:
            keys.add(t)
    return keys

def score(pred, answer):
    keys = keywords(answer)
    if not keys:
        return 1.0, 0, 0
    np_ = norm(pred)
    pset = set(np_.split())
    hit = sum(1 for k in keys if k in pset or k in np_)
    return hit / len(keys), hit, len(keys)

def load_test():
    rows = []
    with open(MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test":
                rows.append(r)
    return rows

def run(gen_model, processor, rows, dev):
    out = []
    for r in rows:
        audio, _ = librosa.load(r["question_wav"], sr=SR)
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "audio", "audio": r["question_wav"]}]},
        ]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = processor(text=text, audio=[audio], return_tensors="pt", padding=True, sampling_rate=SR)
        inp = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
        with torch.no_grad():
            gen = gen_model.generate(**inp, max_new_tokens=48, do_sample=False, use_audio_in_video=False)
        new = gen[0, inp["input_ids"].shape[1]:]
        pred = processor.batch_decode([new], skip_special_tokens=True)[0].strip()
        frac, hit, tot = score(pred, r["answer_text"])
        out.append({"id": r["id"], "q": r["question_text"], "gt": r["answer_text"],
                    "pred": pred, "frac": round(frac, 3), "hit": hit, "tot": tot})
    return out

def summarize(res):
    fracs = [x["frac"] for x in res]
    correct = sum(1 for x in res if x["frac"] >= 0.6)
    return {"mean_slot_frac": round(sum(fracs)/len(fracs), 3),
            "correct_at_0.6": correct, "n": len(res)}

def main():
    dev = "cuda"
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    rows = load_test()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL)
    print(f"[1] loading base ({len(rows)} held-out) ...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        enable_audio_output=False).to(dev).eval()

    print("[2] BASE eval (audio-in) ...", flush=True)
    base = run(model.thinker, processor, rows, dev)
    base_sum = summarize(base)
    print("    base:", base_sum, flush=True)

    print("[3] attaching adapter + LoRA eval ...", flush=True)
    model.thinker = PeftModel.from_pretrained(model.thinker, ADAPTER_DIR).eval()
    lora = run(model.thinker, processor, rows, dev)
    lora_sum = summarize(lora)
    print("    lora:", lora_sum, flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9

    tr = {}
    tp = os.path.join(ROOT, "ft/logs/v2v_train_result.json")
    if os.path.exists(tp):
        tr = json.load(open(tp))

    result = {
        "model": MODEL,
        "path": "Qwen2.5-Omni-7B thinker LoRA, audio-in -> text-answer",
        "trained_through_audio": tr.get("trained_through_audio", True),
        "steps": tr.get("steps"),
        "trainable_params": tr.get("trainable_params"),
        "peak_vram_gb": round(peak, 2),
        "train_peak_vram_gb": tr.get("peak_vram_gb"),
        "final_train_loss": tr.get("final_loss"),
        "base_correct": base_sum["correct_at_0.6"],
        "lora_correct": lora_sum["correct_at_0.6"],
        "base_mean_slot_frac": base_sum["mean_slot_frac"],
        "lora_mean_slot_frac": lora_sum["mean_slot_frac"],
        "n_heldout": base_sum["n"],
        "speech_out": "not generated for Qwen (text answer scored; talker out of scope)",
        "notes": ("Genuine audio-input path: spoken-question waveform -> Omni audio encoder "
                  "(frozen) -> thinker LLM (LoRA). Slot match = years/numbers/entities "
                  "from KB ground-truth answer present in model answer."),
        "error": None,
        "eval_seconds": round(time.time() - t0, 1),
        "base_rows": base,
        "lora_rows": lora,
    }
    with open(os.path.join(ROOT, "results_voice2voice.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[4] wrote results_voice2voice.json", flush=True)
    print(json.dumps({k: result[k] for k in
          ["base_correct","lora_correct","base_mean_slot_frac","lora_mean_slot_frac",
           "trainable_params","steps","peak_vram_gb"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
