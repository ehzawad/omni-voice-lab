"""Shared F5/IndicF5 model construction and weight loading (no whole-checkpoint GPU mapping; strict key report)."""
import json, torch
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer

MEL = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, target_sample_rate=24000, mel_spec_type="vocos")
BASE_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False, conv_layers=4, pe_attn_head=1)

def read_cfg(config_json=None):
    """IndicF5 ships a config.json; fall back to F5TTS_Base. Only known DiT keys are forwarded."""
    cfg = dict(BASE_CFG)
    if config_json:
        c = json.load(open(config_json)); c = c.get("model", c).get("arch", c) if isinstance(c, dict) else c
        for k in ("dim", "depth", "heads", "ff_mult", "text_dim", "text_mask_padding", "conv_layers", "pe_attn_head", "qk_norm", "long_skip_connection"):
            if k in c: cfg[k] = c[k]
    return cfg

def build_model(vocab_file, cfg, checkpoint_activations=False, ode_method="euler"):
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    transformer = DiT(**cfg, text_num_embeds=vocab_size, mel_dim=MEL["n_mel_channels"], checkpoint_activations=checkpoint_activations)
    model = CFM(transformer=transformer, mel_spec_kwargs=MEL, odeint_kwargs=dict(method=ode_method), vocab_char_map=vocab_char_map)
    return model, vocab_char_map, vocab_size

def load_weights(model, path, prefer="ema"):
    """Load a released checkpoint (.safetensors or .pt) onto CPU into our CFM.
    Handles: torch.compile '_orig_mod.' prefixes, 'ema_model.' prefixes, a bundled vocoder in the same file,
    and mel_spec buffers. Returns (missing, unexpected) — callers must assert these are acceptable."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file; sd = load_file(path, device="cpu")
    else:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # our own training checkpoints carry RNG state (numpy arrays), which weights_only=True rejects
            ck = torch.load(path, map_location="cpu", weights_only=False)
        if prefer == "ema" and ck.get("ema_model_state_dict") is not None: sd = ck["ema_model_state_dict"]
        elif "model_state_dict" in ck: sd = ck["model_state_dict"]
        else: sd = ck
    has_ema = any(k.startswith("ema_model.") for k in sd)
    out = {}
    for k, v in sd.items():
        if k.startswith("vocoder."): continue                      # IndicF5 ships the vocoder in the same file
        if has_ema and prefer == "ema" and not k.startswith("ema_model."): continue
        k2 = k[len("ema_model."):] if k.startswith("ema_model.") else k
        k2 = k2.replace("_orig_mod.", "")                          # torch.compile wrapper
        if k2 in ("initted", "step") or ".mel_spec." in k2 or k2.startswith("mel_spec"): continue
        out[k2] = v
    res = model.load_state_dict(out, strict=False)
    return list(res.missing_keys), list(res.unexpected_keys)
