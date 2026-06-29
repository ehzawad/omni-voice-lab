#!/usr/bin/env python
"""LoRA SMOKE TEST (pipeline proof) for Qwen2.5-Omni-7B (single-network omni).

Path: direct HF + peft (NOT LLaMA-Factory). Proves: load omni base in bf16 ->
disable talker (speech head) to save memory -> attach LoRA to the THINKER's text
decoder (q_proj/v_proj) -> take a few optimizer steps -> save adapter -> reload
base+adapter -> run one inference, without crashing.

This is plumbing, not quality. Env: .venv-qwenomni (transformers>=4.57).
GPU: run with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (A6000 only).
"""
import json, os, time, gc
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel

MODEL = "Qwen/Qwen2.5-Omni-7B"
ADAPTER_DIR = "/mnt/sdb/arafat/hervoice/adapters/qwen25omni_lora_smoke"
STEPS = 16
DEV = "cuda"


def load_base():
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    # speech generation head is irrelevant to a text LoRA smoke; drop it to save VRAM
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    return model


def main():
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer

    print("[1] loading base omni model (bf16) ...", flush=True)
    model = load_base()
    model.to(DEV)
    print("    thinker type:", type(model.thinker).__name__, flush=True)

    # ---- attach LoRA to the thinker's text decoder only ----
    lcfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model.thinker = get_peft_model(model.thinker, lcfg)
    model.thinker.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.thinker.parameters() if p.requires_grad)
    print(f"[2] LoRA attached to thinker; trainable params = {trainable}", flush=True)

    # gradient checkpointing on the thinker text backbone to keep VRAM modest
    try:
        model.thinker.gradient_checkpointing_enable()
        model.thinker.enable_input_require_grads()
    except Exception as e:
        print("    (gradient checkpointing skipped:", repr(e), ")", flush=True)

    # ---- tiny text dataset (instruction-ish strings; plumbing only) ----
    samples = [
        "It is the sound of glass shattering.",
        "A woman is coughing in the next room.",
        "Mister Quiller is the apostle of the middle classes.",
        "The bird is singing in the morning light.",
        "Rain is falling steadily on the metal roof.",
        "A car engine starts and revs loudly.",
        "The kettle whistles as the water boils.",
        "Footsteps echo down the empty hallway.",
    ]

    model.thinker.train()
    opt = torch.optim.AdamW(
        [p for p in model.thinker.parameters() if p.requires_grad], lr=1e-4
    )

    print("[3] training ...", flush=True)
    losses = []
    for step in range(STEPS):
        text = samples[step % len(samples)]
        enc = tok(text, return_tensors="pt").to(DEV)
        labels = enc["input_ids"].clone()
        out = model.thinker(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"], labels=labels)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(float(loss))
        print(f"    step {step+1}/{STEPS} loss={loss.item():.4f}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[4] peak VRAM = {peak:.2f} GB", flush=True)

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.thinker.save_pretrained(ADAPTER_DIR)
    print(f"[5] adapter saved to {ADAPTER_DIR}", flush=True)
    saved_ok = os.path.exists(os.path.join(ADAPTER_DIR, "adapter_model.safetensors"))

    # ---- free and reload base + adapter ----
    del model, opt
    gc.collect(); torch.cuda.empty_cache()

    print("[6] reloading base + adapter ...", flush=True)
    reloaded_ok = False
    inference_ok = False
    gen_txt = None
    err = None
    try:
        model2 = load_base().to(DEV)
        model2.thinker = PeftModel.from_pretrained(model2.thinker, ADAPTER_DIR)
        model2.thinker.eval()
        reloaded_ok = True
        print("[7] adapter reloaded; running one inference ...", flush=True)
        enc = tok("The sound I hear is", return_tensors="pt").to(DEV)
        with torch.no_grad():
            gen = model2.thinker.generate(**enc, max_new_tokens=16, do_sample=False)
        gen_txt = tok.decode(gen[0], skip_special_tokens=True)
        print("    generated:", repr(gen_txt), flush=True)
        inference_ok = True
    except Exception as e:
        err = repr(e)
        print("    reload/inference error:", err, flush=True)

    import transformers
    result = {
        "model": MODEL,
        "path_used": "direct-peft",
        "env": ".venv-qwenomni",
        "transformers_version": transformers.__version__,
        "status": "pass" if (saved_ok and reloaded_ok and inference_ok) else "fail",
        "steps_run": STEPS,
        "trainable_params": trainable,
        "peak_vram_gb": round(peak, 2),
        "adapter_reloaded": reloaded_ok,
        "inference_ok": inference_ok,
        "first_loss": round(losses[0], 4) if losses else None,
        "last_loss": round(losses[-1], 4) if losses else None,
        "generated": gen_txt,
        "notes": ("Direct HF+peft smoke. Loaded full Qwen2_5OmniForConditionalGeneration "
                  "in bf16, disable_talker() to save VRAM, attached LoRA (r=8/a=16, "
                  "q_proj/v_proj) to the thinker text decoder, trained on tiny text samples "
                  "with gradient checkpointing, saved+reloaded adapter, ran one text "
                  "inference through the thinker."),
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
