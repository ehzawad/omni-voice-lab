#!/usr/bin/env python3
"""Render the USER side of the scripted scenarios to audio, in a voice that is NOT the bot's.

There is no microphone on this box and nobody available to record, so the user turns are
synthesised. Two deliberate choices keep this honest:

  * a DIFFERENT voice from the assistant: the released ai4bharat/IndicF5 conditioned on a REAL
    BENGALI MALE speaker (IndicVoices-R, SNR 74 dB), against the assistant's fine-tune on a
    Punjabi female prompt. If the same voice spoke both sides, the ASR would be tested on the
    one voice the TTS was tuned to produce, which is the easiest possible input.

    The Bengali reference is not cosmetic. The first version of this script used ai4bharat's
    MARATHI prompt, and the resulting Bengali was measurably harder for a Bengali ASR than
    real human speech: direct-to-ASR CER 0.282 against 0.112 for real IndicVoices-R speakers.
    Cross-lingual prompting gives IndicF5 a non-Bengali accent, and the evaluation was then
    measuring that accent rather than the pipeline. Condition on the target language.
  * CPU only, at low priority, so the GPU budget on the shared card is untouched. Slow (tens
    of seconds per sentence) but this is offline preparation, not the serving path.

Output: hervoice/eval/audio/<scenario_id>/t<N>.wav, 16 kHz mono float32, plus a manifest.
Synthetic input is CLEAN; realism comes from the separate IndicVoices-R real-speech run.

    OMP_NUM_THREADS=8 nice -n 10 .venv-bnweb/bin/python -m hervoice.eval.render_user_voice
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HERE = os.path.dirname(os.path.abspath(__file__))

# A real Bengali male speaker, extracted from IndicVoices-R by pick_user_ref (see DECISIONS.md).
# Regenerate with a different speaker by replacing these two files.
USER_REF_WAV = os.path.join(HERE, "user_ref.wav")
USER_REF_TXT = os.path.join(HERE, "user_ref.txt")


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    from huggingface_hub import hf_hub_download
    import torchaudio
    from hervoice.bn.models import BnTts

    scen = json.load(open(os.path.join(HERE, "scenarios_bn.json"), encoding="utf-8"))
    out_root = os.path.join(HERE, "audio")
    os.makedirs(out_root, exist_ok=True)

    ref_text = open(USER_REF_TXT, encoding="utf-8").read().strip()
    t = time.time()
    tts = BnTts(repo="ai4bharat/IndicF5", ref_wav=USER_REF_WAV, ref_text=ref_text,
                device="cpu", nfe=16)
    print(f"[load] released IndicF5 on CPU in {time.time()-t:.1f}s", flush=True)

    manifest = []
    for sc in scen["scenarios"]:
        d = os.path.join(out_root, sc["id"])
        os.makedirs(d, exist_ok=True)
        for i, turn in enumerate(sc["turns"]):
            path = os.path.join(d, f"t{i}.wav")
            if os.path.exists(path):
                manifest.append({"scenario": sc["id"], "turn": i, "path": path, "text": turn["user"], "cached": True})
                continue
            t = time.time()
            waves = [tts.synth_chunk(ch, seed=7000 + 10 * i + ci) for ci, ch in enumerate(tts.chunks(turn["user"]))]
            w24 = np.concatenate(waves) if waves else np.zeros(2400, dtype=np.float32)
            w16 = torchaudio.functional.resample(torch.from_numpy(w24)[None], 24000, 16000)[0].numpy()
            sf.write(path, w16, 16000)
            dt = time.time() - t
            print(f"  {sc['id']}/t{i}: {len(w16)/16000:.2f}s audio in {dt:.0f}s  {turn['user'][:50]}", flush=True)
            manifest.append({"scenario": sc["id"], "turn": i, "path": path, "text": turn["user"],
                             "seconds": round(len(w16) / 16000, 2), "render_s": round(dt, 1)})
    json.dump(manifest, open(os.path.join(out_root, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[done] {len(manifest)} user turns -> {out_root}")


if __name__ == "__main__":
    main()
