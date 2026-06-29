#!/usr/bin/env python3
"""Shared helpers for the Bengali Orpheus (Llama+SNAC) TTS fine-tune.

Orpheus encodes audio as SNAC tokens packed into the Llama vocab. Token scheme
(canopylabs Orpheus convention):
  START_HUMAN=128259, EOT=128009, END_HUMAN=128260,
  START_SPEECH=128257, END_SPEECH=128258, AUDIO_OFFSET=128266
SNAC frame = 7 codes redistributed across 3 hierarchical codebooks.
"""
import glob, os, numpy as np, torch

START_HUMAN, EOT, END_HUMAN = 128259, 128009, 128260
START_SPEECH, END_SPEECH, AUDIO_OFFSET = 128257, 128258, 128266
SNAC_SR = 24000

def fleurs_bn_dir():
    base = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--google--fleurs/snapshots/*/parquet-data/bn_in"))
    if not base:
        raise FileNotFoundError("FLEURS bn_in parquet not cached")
    return base[0]

def load_fleurs_split(split, n=None):
    """Return list of dicts {text, raw, audio(np float32 @sr), sr} from cached parquet."""
    import pandas as pd, soundfile as sf, io
    path = os.path.join(fleurs_bn_dir(), f"{split}-00000-of-00001.parquet")
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        a = r["audio"]
        audio = sr = None
        if isinstance(a, dict) and a.get("bytes"):
            audio, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        elif isinstance(a, dict) and a.get("array") is not None:
            audio, sr = np.asarray(a["array"], dtype=np.float32), a.get("sampling_rate", 16000)
        rows.append(dict(text=r.get("transcription", ""), raw=r.get("raw_transcription", ""),
                         audio=audio, sr=sr))
        if n and len(rows) >= n:
            break
    return rows

# ---------- prompt construction ----------
def build_gen_input(tokenizer, text, voice=None):
    prompt = f"{voice}: {text}" if voice else text
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    start = torch.tensor([[START_HUMAN]], dtype=torch.long)
    end = torch.tensor([[EOT, END_HUMAN]], dtype=torch.long)
    return torch.cat([start, ids, end], dim=1)

# ---------- decode generated tokens -> SNAC codes -> waveform ----------
def tokens_to_codes(generated_ids_row):
    """generated_ids_row: 1D tensor of generated token ids (full sequence)."""
    row = generated_ids_row
    idx = (row == START_SPEECH).nonzero(as_tuple=True)[0]
    if len(idx) > 0:
        row = row[idx[-1].item() + 1:]
    row = row[row != END_SPEECH]
    n = (row.numel() // 7) * 7
    row = row[:n]
    return [int(t) - AUDIO_OFFSET for t in row.tolist()]

def redistribute(code_list):
    l1, l2, l3 = [], [], []
    for i in range(len(code_list) // 7):
        l1.append(code_list[7*i])
        l2.append(code_list[7*i+1] - 4096)
        l3.append(code_list[7*i+2] - 2*4096)
        l3.append(code_list[7*i+3] - 3*4096)
        l2.append(code_list[7*i+4] - 4*4096)
        l3.append(code_list[7*i+5] - 5*4096)
        l3.append(code_list[7*i+6] - 6*4096)
    return l1, l2, l3

def codes_to_audio(snac_model, code_list, device):
    l1, l2, l3 = redistribute(code_list)
    if not l1:
        return np.zeros(1, dtype=np.float32)
    codes = [torch.tensor(l1, device=device).unsqueeze(0),
             torch.tensor(l2, device=device).unsqueeze(0),
             torch.tensor(l3, device=device).unsqueeze(0)]
    # guard: SNAC codebook range is [0,4095]
    for c in codes:
        c.clamp_(0, 4095)
    with torch.no_grad():
        wav = snac_model.decode(codes)
    return wav.squeeze().detach().cpu().numpy().astype(np.float32)

# ---------- SNAC-encode an audio waveform -> 7-per-frame token ids (for training labels) ----------
def audio_to_tokens(snac_model, audio_np, in_sr, device):
    import librosa
    if in_sr != SNAC_SR:
        audio_np = librosa.resample(audio_np, orig_sr=in_sr, target_sr=SNAC_SR)
    wav = torch.tensor(audio_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        codes = snac_model.encode(wav)  # list of 3 tensors: [1,T], [1,2T], [1,4T]
    l1 = codes[0][0].tolist(); l2 = codes[1][0].tolist(); l3 = codes[2][0].tolist()
    toks = []
    for i in range(len(l1)):
        toks.append(l1[i] + AUDIO_OFFSET)
        toks.append(l2[2*i]   + 4096 + AUDIO_OFFSET)
        toks.append(l3[4*i]   + 2*4096 + AUDIO_OFFSET)
        toks.append(l3[4*i+1] + 3*4096 + AUDIO_OFFSET)
        toks.append(l2[2*i+1] + 4*4096 + AUDIO_OFFSET)
        toks.append(l3[4*i+2] + 5*4096 + AUDIO_OFFSET)
        toks.append(l3[4*i+3] + 6*4096 + AUDIO_OFFSET)
    return toks
