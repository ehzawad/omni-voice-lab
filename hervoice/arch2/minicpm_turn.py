#!/usr/bin/env python3
"""ARCHITECTURE 2 -- turn-based SINGLE speech-to-speech with MiniCPM-o 4.5.

ONE omni network does audio understanding AND audio generation in STRICT DISCRETE
TURNS: a full user utterance goes in, a full spoken assistant reply comes out, then
the turn ends. This mirrors GPT-Live "Advanced Voice Mode": smoother than a cascaded
ASR->LLM->TTS pipeline (arch 1) because a single net carries the whole path
(audio encoder -> shared Thinker LLM -> Talker + Token2wav vocoder), but the
back-and-forth is still rigid.

Guardrail: this is NOT full-duplex. No barge-in, no streaming-input overlap, no
`hervoice/live/` loop (that is arch 3). No external ASR and no external TTS sit in
the generation path -- the same MiniCPM-o net does both ends. (A separate Qwen3-ASR
re-transcribes the OUTPUT wav afterwards purely as a readability check; it never
touches generation.)

We reuse the proven streaming call pattern from minicpm_stream.py / engine.py, but
in a discrete-turn framing so we can timestamp the FIRST assistant audio frame
(time-to-first-audio, TTFA) while still concatenating all chunks into one reply wav.
"""
import argparse
import json
import subprocess
import time
import uuid

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "openbmb/MiniCPM-o-4_5"
# 4-bit the LLM only; keep audio encoder / TTS / vision in full precision
KEEP_FP = ["vpm", "apm", "resampler", "tts", "audio", "vision", "Token2wav", "embed", "tokenizer"]
VC_SUFFIX = ("Please assist users while maintaining this voice style. Answer seriously and in high "
             "quality. Chat in a highly human-like, oral style. You are a helpful assistant.")
SYSTEM_TEXT = "You are a warm, concise English voice assistant. Keep replies short and spoken."

OUT_SR = 24000          # Talker/Token2wav output sample rate
SILENCE_RMS = 0.005     # below this the whole reply is treated as silent
MIN_DUR_S = 0.2         # shorter than this is degenerate
MIN_PREFILL_SAMPLES = 1600  # 0.1 s @ 16k; avoids MiniCPM avg_pool1d short-chunk crash


def _to_np_audio(wav_chunk):
    if wav_chunk is None:
        return None
    if torch.is_tensor(wav_chunk):
        return wav_chunk.reshape(-1).float().cpu().numpy()
    return np.asarray(wav_chunk, dtype=np.float32).reshape(-1)


class MiniCPMTurnS2S:
    """Load MiniCPM-o 4.5 once, build the voice cache once, run discrete turns."""

    def __init__(self, ref_audio_path, quant="int4", chunk_ms=1000, max_new_tokens=200):
        self.quant = quant
        self.chunk_ms = chunk_ms
        self.max_new_tokens = max_new_tokens

        load_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if quant == "int4":
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=KEEP_FP)
            load_kw["device_map"] = "cuda"
            self.model = AutoModel.from_pretrained(MODEL_ID, **load_kw).eval()
        else:
            self.model = AutoModel.from_pretrained(MODEL_ID, **load_kw).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model.init_tts()

        # voice cache built ONCE at boot from the reference clip (never per turn)
        self.ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        self.model.init_token2wav_cache(self.ref_audio)

    def run_turn(self, wav_in, out_wav):
        """One discrete turn: user audio in -> spoken reply out. Returns a metrics dict."""
        sid = f"arch2-minicpm-{uuid.uuid4().hex[:12]}"
        # fresh session, keep the voice cache
        self.model.reset_session(reset_token2wav_cache=False)

        user_audio, _ = librosa.load(wav_in, sr=16000, mono=True)

        # 1) prefill system (voice clone)
        self.model.streaming_prefill(
            session_id=sid,
            msgs=[{"role": "system",
                   "content": [SYSTEM_TEXT, "Clone the voice in the provided audio prompt.",
                               self.ref_audio, VC_SUFFIX]}],
            tokenizer=self.tok)

        # 2) prefill the full user utterance in fixed chunks (no streaming overlap:
        #    we push the entire utterance, THEN generate -- strict discrete turn)
        step = int(16000 * self.chunk_ms / 1000)
        n = max(1, (len(user_audio) + step - 1) // step)
        for i in range(n):
            ch = user_audio[i * step:(i + 1) * step]
            if ch.size < MIN_PREFILL_SAMPLES:
                ch = np.concatenate([ch, np.zeros(MIN_PREFILL_SAMPLES - ch.size, dtype=np.float32)])
            self.model.streaming_prefill(
                session_id=sid,
                msgs=[{"role": "user", "content": [ch]}],
                is_last_chunk=(i == n - 1), tokenizer=self.tok)

        # 3) generate the spoken reply; timestamp the first non-silent audio frame
        t0 = time.time()
        first_audio_s = None
        waves, text_acc = [], ""
        for wav_chunk, new_text in self.model.streaming_generate(
                session_id=sid, generate_audio=True, tokenizer=self.tok,
                max_new_tokens=self.max_new_tokens):
            audio = _to_np_audio(wav_chunk)
            if audio is not None and audio.size > 0:
                if first_audio_s is None and float(np.sqrt(np.mean(audio ** 2))) > 1e-4:
                    first_audio_s = time.time() - t0
                waves.append(audio)
            if new_text:
                text_acc += new_text
        total_s = time.time() - t0

        reply_text = text_acc.strip()
        m = {
            "input_wav": wav_in,
            "measured_first_audio_s": round(first_audio_s, 3) if first_audio_s is not None else None,
            "total_s": round(total_s, 3),
            "reply_text": reply_text,
        }

        # honest guards -- no fake wav on failure
        if not waves:
            m.update(state="s2s_failed", reason="no audio frames produced",
                     reply_audio_duration_s=0.0, rtf=None, out_wav=None)
            return m
        reply = np.concatenate(waves)
        dur = len(reply) / OUT_SR
        rms = float(np.sqrt(np.mean(reply ** 2)))
        if not reply_text:
            m.update(state="s2s_failed", reason="empty reply text",
                     reply_audio_duration_s=round(dur, 3), rtf=None, out_wav=None)
            return m
        if dur < MIN_DUR_S:
            m.update(state="s2s_failed", reason=f"degenerate audio dur={dur:.3f}s",
                     reply_audio_duration_s=round(dur, 3), rtf=None, out_wav=None)
            return m
        if rms < SILENCE_RMS:
            m.update(state="s2s_failed", reason=f"silent audio rms={rms:.5f}",
                     reply_audio_duration_s=round(dur, 3), rtf=None, out_wav=None)
            return m

        sf.write(out_wav, reply, samplerate=OUT_SR)
        m.update(state="ok",
                 reply_audio_duration_s=round(dur, 3),
                 reply_rms=round(rms, 4),
                 rtf=round(total_s / dur, 3),
                 out_wav=out_wav)
        return m

    @staticmethod
    def peak_vram_gb():
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
        return 0.0


def reasr(out_wav, venv_python, model_key="qwen3-asr-0.6b"):
    """Independent readability check: re-transcribe the OUTPUT wav with Qwen3-ASR.

    Runs in a SEPARATE venv/process; not part of the S2S generation path. Returns a
    dict or {'error': ...}. This is a readability proxy, NOT WER (no ground truth).
    """
    try:
        proc = subprocess.run(
            [venv_python, "-m", "hervoice.arch2.reasr_minicpm", out_wav, model_key],
            capture_output=True, text=True, timeout=600,
            cwd="/mnt/sdb/arafat/hervoice",
            env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
                 "PATH": "/usr/bin:/bin", "HOME": "/mnt/sdb/arafat"})
        if proc.returncode != 0:
            return {"error": (proc.stderr or proc.stdout).strip()[-500:]}
        line = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")][-1]
        return json.loads(line)
    except Exception as e:
        return {"error": str(e)}


def median(xs):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    n = len(xs)
    return round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--ref", default="examples/ref_female.wav")
    ap.add_argument("--outdir", default="runs/arch2/minicpm")
    ap.add_argument("--quant", choices=["none", "int4"], default="int4")
    ap.add_argument("--reasr-venv", default=".venv-qwen-asr/bin/python")
    ap.add_argument("--results", default="hervoice/arch2/results_minicpm.json")
    ap.add_argument("--manifest", default="runs/arch2/minicpm/manifest.json")
    args = ap.parse_args()

    t_load = time.time()
    engine = MiniCPMTurnS2S(args.ref, quant=args.quant)
    load_s = round(time.time() - t_load, 2)

    turns = []
    for i, wav_in in enumerate(args.inputs):
        out_wav = f"{args.outdir}/turn{i}_reply.wav"
        print(f"\n=== turn {i}: {wav_in} ===", flush=True)
        m = engine.run_turn(wav_in, out_wav)
        m["turn"] = i
        print(f"  state={m['state']} TTFA={m['measured_first_audio_s']}s "
              f"total={m['total_s']}s dur={m.get('reply_audio_duration_s')}s "
              f"RTF={m.get('rtf')}", flush=True)
        print(f"  reply_text: {m['reply_text'][:160]}", flush=True)
        if m["state"] == "ok":
            r = reasr(out_wav, args.reasr_venv)
            m["reasr_readability"] = r
            print(f"  re-ASR(readability): {r.get('text', r)}", flush=True)
        turns.append(m)

    peak_vram = engine.peak_vram_gb()
    ok = [t for t in turns if t["state"] == "ok"]
    summary = {
        "n_turns": len(turns),
        "n_ok": len(ok),
        "median_ttfa_s": median([t["measured_first_audio_s"] for t in ok]),
        "median_rtf": median([t["rtf"] for t in ok]),
        "median_turn_total_s": median([t["total_s"] for t in ok]),
        "load_s": load_s,
        "peak_vram_gb": peak_vram,
    }
    print(f"\n=== SUMMARY === {json.dumps(summary, indent=2)}", flush=True)

    results = {"model_id": MODEL_ID, "quant": args.quant, "architecture": "arch2-turn-based-single-s2s",
               "turns": turns, "summary": summary}
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)

    # VRAM used from nvidia-smi (device-level) as a cross-check
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True,
            env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
                 "PATH": "/usr/bin:/bin"})
        smi_mb = int(smi.stdout.strip().splitlines()[0])
    except Exception:
        smi_mb = None
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd="/mnt/sdb/arafat/hervoice").stdout.strip()

    manifest = {
        "architecture": "arch2-turn-based-single-speech-to-speech",
        "model_id": MODEL_ID,
        "quant": args.quant,
        "single_network": True,
        "external_asr_in_generation_path": False,
        "external_tts_in_generation_path": False,
        "reasr_is_readability_check_only": True,
        "gpu": "RTX A5000 (GPU0, PCI_BUS_ID)",
        "git_commit": git_commit,
        "ref_voice": args.ref,
        "inputs": args.inputs,
        "vram_peak_torch_gb": peak_vram,
        "vram_device_used_mb_nvidia_smi": smi_mb,
        "turns": turns,
        "summary": summary,
        "caveats": [
            "Naturalness/quality of speech needs human MOS; not measured here.",
            "re-ASR is an independent readability check (Qwen3-ASR on the output wav), NOT WER -- no ground-truth transcript for free-form replies.",
            "int4 quantization of the LLM (audio/TTS/vision kept fp).",
            "Single RTX A5000, English only, file-driven single discrete turns, no live mic.",
            "Strict discrete-turn framing: NOT full-duplex / no barge-in (that is architecture 3).",
            "Open local analog of GPT-Live Advanced Voice Mode, NOT GPT-Live itself.",
        ],
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    # copy best (longest ok reply) to demo_minicpm.wav
    if ok:
        best = max(ok, key=lambda t: t["reply_audio_duration_s"])
        import shutil
        shutil.copy(best["out_wav"], f"{args.outdir}/demo_minicpm.wav")
        print(f"[demo] {best['out_wav']} -> {args.outdir}/demo_minicpm.wav", flush=True)

    print(f"[wrote] {args.results}, {args.manifest}", flush=True)
    print(f"[VRAM] torch peak {peak_vram} GB; nvidia-smi device {smi_mb} MB", flush=True)


if __name__ == "__main__":
    main()
