#!/usr/bin/env python
"""STEP 2 - GENUINE audio-in -> text-answer LoRA on Qwen2.5-Omni-7B.

The training signal flows through the REAL AUDIO INPUT path: each example feeds the
spoken-question waveform through the Omni audio encoder into the thinker LLM, and the
loss is computed only on the assistant answer tokens. LoRA is on the thinker's LLM
projections; the audio encoder is frozen.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
     .venv-qwenomni/bin/python ft/v2v_train.py
"""
import json, os, time, math, random
import torch, librosa
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import LoraConfig, get_peft_model

ROOT = "/mnt/sdb/arafat/hervoice"
MODEL = "Qwen/Qwen2.5-Omni-7B"
MANIFEST = os.path.join(ROOT, "data/v2v/manifest.jsonl")
ADAPTER_DIR = os.path.join(ROOT, "adapters/v2v_qwen25omni_lora")
EPOCHS = 3
LR = 1e-4
SR = 16000
SYS = ("You are a concise FIFA expert. Answer the spoken question in one short, "
       "factual sentence.")

def load_rows(split):
    rows = []
    with open(MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split:
                rows.append(r)
    return rows

def build_example(processor, wav_path, answer_text, device, dtype):
    """Return thinker inputs with labels masking the prompt (audio path)."""
    audio, _ = librosa.load(wav_path, sr=SR)
    prompt_conv = [
        {"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user", "content": [{"type": "audio", "audio": wav_path}]},
    ]
    full_conv = prompt_conv + [
        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
    ]
    prompt_text = processor.apply_chat_template(prompt_conv, add_generation_prompt=True, tokenize=False)
    full_text = processor.apply_chat_template(full_conv, add_generation_prompt=False, tokenize=False)

    full = processor(text=full_text, audio=[audio], return_tensors="pt", padding=True,
                     sampling_rate=SR)
    prompt = processor(text=prompt_text, audio=[audio], return_tensors="pt", padding=True,
                       sampling_rate=SR)
    plen = prompt["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100

    out = {}
    for k, v in full.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
        else:
            out[k] = v
    out["labels"] = labels.to(device)
    return out

def main():
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = {"model": MODEL, "path": "qwen2.5-omni thinker LoRA (audio-in)",
              "trained_through_audio": False, "error": None}
    print("[1] loading processor + base (enable_audio_output=False) ...", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        enable_audio_output=False)
    model.to(dev)
    dtype = torch.bfloat16

    # Freeze everything; LoRA only on the thinker LLM projections.
    for p in model.parameters():
        p.requires_grad = False
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model.thinker = get_peft_model(model.thinker, lcfg)
    model.thinker.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result["trainable_params"] = int(trainable)
    print(f"[2] LoRA attached to thinker; trainable={trainable}", flush=True)

    rows = load_rows("train")
    print(f"[3] {len(rows)} train pairs; sanity-checking one audio example ...", flush=True)
    ex0 = build_example(processor, rows[0]["question_wav"], rows[0]["answer_text"], dev, dtype)
    print("    input keys:", [k for k in ex0.keys()],
          "| has input_features:", "input_features" in ex0,
          "| seq_len:", ex0["input_ids"].shape, flush=True)
    assert "input_features" in ex0, "no audio features -> not the audio path!"

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.thinker.train()
    steps = 0
    losses = []
    print("[4] training (audio-in -> answer-text) ...", flush=True)
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(rows)
        for r in rows:
            inp = build_example(processor, r["question_wav"], r["answer_text"], dev, dtype)
            out = model.thinker(**inp, use_audio_in_video=False)
            loss = out.loss
            loss.backward()
            opt.step(); opt.zero_grad()
            steps += 1
            losses.append(float(loss))
            if steps % 20 == 0 or steps == 1:
                print(f"    ep{ep} step{steps} loss={loss.item():.4f}", flush=True)
    result["trained_through_audio"] = True
    result["steps"] = steps
    result["final_loss"] = round(sum(losses[-10:]) / min(10, len(losses)), 4)
    peak = torch.cuda.max_memory_allocated() / 1e9
    result["peak_vram_gb"] = round(peak, 2)
    print(f"[5] done {steps} steps, final_loss={result['final_loss']}, peak={peak:.1f}GB", flush=True)

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.thinker.save_pretrained(ADAPTER_DIR)
    print(f"[6] adapter saved -> {ADAPTER_DIR}", flush=True)
    result["adapter_dir"] = ADAPTER_DIR
    result["train_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(ROOT, "ft/logs/v2v_train_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[7] result:", json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
