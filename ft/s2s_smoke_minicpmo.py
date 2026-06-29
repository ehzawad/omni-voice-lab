#!/usr/bin/env python
"""LoRA SMOKE TEST (plumbing proof) for MiniCPM-o-4_5 (single-network omni model).

Path: direct HF + peft fallback. LLaMA-Factory could not drive MiniCPM-o-4_5
because the model's bespoke `forward(self, data, **kwargs)` re-passes `input_ids`
into the inner LLM and is incompatible with the standard SFT collator/peft path
(see ft/logs/minicpmo_train.log). Instead we attach LoRA to the model's LLM
backbone (model.llm, a Qwen3ForCausalLM) and take a few text optimizer steps.
This proves: load omni base -> attach LoRA -> step -> save adapter -> reload
base+adapter -> run one inference, without crashing.

GPU: run with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (A6000 only).
"""
import json, os, time, gc
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel

MODEL = "openbmb/MiniCPM-o-4_5"
ADAPTER_DIR = "/mnt/sdb/arafat/hervoice/adapters/minicpmo_lora_smoke"
STEPS = 12

def load_base():
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    # keep audio encoder (genuine omni backbone); skip heavy TTS for the smoke
    cfg.init_tts = False
    cfg.init_audio = True
    model = AutoModel.from_pretrained(
        MODEL, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    return model

def main():
    t0 = time.time()
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print("[1] loading base omni model ...", flush=True)
    model = load_base()
    model.to(dev)

    # ---- attach LoRA to the LLM backbone ----
    lcfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lcfg)
    model.llm.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    print(f"[2] LoRA attached; trainable params = {trainable}", flush=True)

    # ---- tiny text dataset (transcripts from the audio demo) ----
    samples = [
        "It is the sound of glass shattering.",
        "A woman is coughing.",
        "Mister Quiller is the apostle of the middle classes.",
        "The bird is singing in the morning light.",
        "Rain is falling steadily on the roof.",
        "A car engine starts and revs loudly.",
    ]
    model.llm.train()
    opt = torch.optim.AdamW(
        [p for p in model.llm.parameters() if p.requires_grad], lr=1e-4
    )

    print("[3] training ...", flush=True)
    losses = []
    for step in range(STEPS):
        text = samples[step % len(samples)]
        enc = tok(text, return_tensors="pt").to(dev)
        labels = enc["input_ids"].clone()
        out = model.llm(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"], labels=labels)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(float(loss))
        print(f"    step {step+1}/{STEPS} loss={loss.item():.4f}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[4] peak VRAM = {peak:.1f} GB", flush=True)

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.llm.save_pretrained(ADAPTER_DIR)
    print(f"[5] adapter saved to {ADAPTER_DIR}", flush=True)
    saved_ok = os.path.exists(os.path.join(ADAPTER_DIR, "adapter_model.safetensors"))

    # ---- free and reload base + adapter ----
    del model, opt
    gc.collect(); torch.cuda.empty_cache()

    print("[6] reloading base + adapter ...", flush=True)
    reloaded_ok = False
    inference_ok = False
    err = None
    try:
        model2 = load_base().to(dev)
        model2.llm = PeftModel.from_pretrained(model2.llm, ADAPTER_DIR)
        model2.llm.eval()
        reloaded_ok = True
        print("[7] adapter reloaded; running one inference ...", flush=True)
        enc = tok("The sound I hear is", return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model2.llm.generate(**enc, max_new_tokens=16, do_sample=False)
        txt = tok.decode(gen[0], skip_special_tokens=True)
        print("    generated:", repr(txt), flush=True)
        inference_ok = True
    except Exception as e:
        err = repr(e)
        print("    reload/inference error:", err, flush=True)

    result = {
        "model": MODEL,
        "path_used": "direct-peft",
        "status": "pass" if (saved_ok and reloaded_ok and inference_ok) else "fail",
        "steps_run": STEPS,
        "trainable_params": trainable,
        "peak_vram_gb": round(peak, 2),
        "adapter_reloaded": reloaded_ok,
        "inference_ok": inference_ok,
        "first_loss": round(losses[0], 4) if losses else None,
        "last_loss": round(losses[-1], 4) if losses else None,
        "notes": ("LLaMA-Factory failed on MiniCPM-o-4_5: bespoke forward(data,**kwargs) "
                  "re-passes input_ids into inner LLM -> 'multiple values for input_ids'; "
                  "also bnb-4bit breaks audio-tower native MHA and enable_input_require_grads "
                  "breaks in-place scatter_. Fallback: LoRA on model.llm (Qwen3) text smoke."),
        "wall_clock_s": round(time.time() - t0),
        "error": err,
    }
    out_path = "/mnt/sdb/arafat/hervoice/results_s2s_lora_smoke.json"
    existing = []
    if os.path.exists(out_path):
        try:
            existing = json.load(open(out_path))
        except Exception:
            existing = []
    existing = [e for e in existing if e.get("model") != MODEL]
    existing.append(result)
    json.dump(existing, open(out_path, "w"), indent=2)
    print("[8] wrote", out_path, flush=True)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
