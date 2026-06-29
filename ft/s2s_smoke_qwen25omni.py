#!/usr/bin/env python
"""LoRA SMOKE TEST attempt for Qwen2.5-Omni-7B.

BLOCKER (recorded honestly, not faked): Qwen2.5-Omni uses the *native*
transformers integration (config architecture `qwen2_5_omni`, library
`transformers`, Model Class AutoModel) and ships NO trust_remote_code modeling
files. That architecture was added to transformers in 4.52.x. This training env
pins transformers==4.51.0, which does not contain `qwen2_5_omni`, so the model
cannot be instantiated here at all -- there is no point downloading the ~16GB
weights. LLaMA-Factory also requires transformers>=4.55 for its current source.

This script probes the config load to capture the concrete error string. To
actually run this smoke, the env would need transformers>=4.52 (ideally >=4.55
to also satisfy LLaMA-Factory), which is out of scope for a smoke test that must
not destabilize the shared training env (the MiniCPM-o adapter is already proven).
"""
import json, os, time
from transformers import AutoConfig

MODEL = "Qwen/Qwen2.5-Omni-7B"

def main():
    t0 = time.time()
    err = None
    status = "fail"
    try:
        cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        print("config loaded:", getattr(cfg, "model_type", None))
        # If we ever get here on a newer transformers, the real smoke would go here.
        status = "config-ok-but-not-implemented"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print("BLOCKED:", err)

    import transformers
    result = {
        "model": MODEL,
        "path_used": "none (blocked before load)",
        "status": status,
        "steps_run": 0,
        "trainable_params": None,
        "peak_vram_gb": None,
        "adapter_reloaded": False,
        "inference_ok": False,
        "notes": (f"Blocked: transformers=={transformers.__version__} lacks the "
                  "`qwen2_5_omni` architecture (added in 4.52). Qwen2.5-Omni-7B uses "
                  "native transformers integration with no trust_remote_code fallback, "
                  "so it cannot be loaded on this env. Weights were NOT downloaded. "
                  "Remediation: transformers>=4.52 (>=4.55 for LLaMA-Factory)."),
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
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
