#!/usr/bin/env python3
"""LoRA fine-tune orpheus-bangla on prepared Orpheus sequences (Bengali speech-out).
Manual loop with grad accumulation; loss only on the speech span (prompt masked).
Run in .venv-tts-lora on cuda:0.
Usage: train_orpheus_lora.py <data.pt> <out_adapter_dir> [max_steps] [grad_accum]"""
import os, sys, time, math, random, torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import bn_tts as B

DATA = sys.argv[1]
OUT = sys.argv[2]
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
GRAD_ACCUM = int(sys.argv[4]) if len(sys.argv) > 4 else 4
LR = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-4
MICRO_BS = 1
DEV = "cuda:0"
REPO = "asif00/orpheus-bangla-tts"
os.makedirs(OUT, exist_ok=True)
random.seed(0); torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(REPO)
PAD = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(DEV)
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False

lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()

data = torch.load(DATA)
print(f"[data] {len(data)} sequences from {DATA}")

def batches():
    order = list(range(len(data)))
    while True:
        random.shuffle(order)
        for i in order:
            yield data[i]

def collate(items):
    ids = [torch.tensor(it["input_ids"], dtype=torch.long) for it in items]
    inp = pad_sequence(ids, batch_first=True, padding_value=PAD)
    att = (inp != PAD).long()
    lab = inp.clone()
    for j, it in enumerate(items):
        lab[j, :it["prompt_len"]] = -100          # mask the text prompt
        lab[j, len(it["input_ids"]):] = -100      # mask padding
    return inp.to(DEV), att.to(DEV), lab.to(DEV)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=MAX_STEPS,
                                            pct_start=0.05, anneal_strategy="cos")
gen = batches()
model.train()
t0 = time.time(); running = 0.0; step = 0
torch.cuda.reset_peak_memory_stats()
while step < MAX_STEPS:
    opt.zero_grad()
    acc = 0.0
    for _ in range(GRAD_ACCUM):
        inp, att, lab = collate([next(gen) for _ in range(MICRO_BS)])
        out = model(input_ids=inp, attention_mask=att, labels=lab)
        (out.loss / GRAD_ACCUM).backward()
        acc += out.loss.item() / GRAD_ACCUM
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step(); sched.step(); step += 1; running += acc
    if step % 20 == 0 or step == 1:
        print(f"  step {step}/{MAX_STEPS} loss {acc:.4f} (avg {running/step:.4f}) "
              f"lr {sched.get_last_lr()[0]:.2e} vram {torch.cuda.max_memory_allocated()/1e9:.1f}GB "
              f"{(time.time()-t0)/step:.2f}s/step")

model.save_pretrained(OUT)
print(f"[saved] adapter -> {OUT}  final_avg_loss {running/max(step,1):.4f}  "
      f"peak_vram {torch.cuda.max_memory_allocated()/1e9:.1f}GB  time {(time.time()-t0)/60:.1f}min")
