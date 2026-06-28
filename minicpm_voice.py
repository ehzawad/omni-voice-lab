#!/usr/bin/env python3
"""MiniCPM-o 4.5 voice-to-voice smoke test: audio question in -> text + 24kHz speech out."""
import argparse, time, librosa, torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "openbmb/MiniCPM-o-4_5"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="minicpm_response.wav")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.init_tts()
    print(f"[load] {time.time()-t0:.1f}s")

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)

    # Try the audio-assistant system prompt if the model exposes one.
    msgs = []
    try:
        sys_msg = model.get_sys_prompt(mode="audio_assistant", language="en")
        msgs.append(sys_msg)
    except Exception as e:
        print(f"[info] no get_sys_prompt ({e}); proceeding without it")
    msgs.append({"role": "user", "content": [audio]})

    common = dict(msgs=msgs, tokenizer=tok, generate_audio=True, output_audio_path=args.out,
                  max_new_tokens=args.max_new_tokens)
    t1 = time.time()
    try:
        res = model.chat(**common)
    except TypeError:
        common.pop("tokenizer", None)
        res = model.chat(**common)
    dt = time.time() - t1

    text = res if isinstance(res, str) else getattr(res, "text", str(res))
    print("\n[TEXT RESPONSE]\n" + text)
    print(f"\n[wrote audio] {args.out}")
    print(f"[gen time] {dt:.2f}s")

if __name__ == "__main__":
    main()
