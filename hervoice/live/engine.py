#!/usr/bin/env python3
"""LiveVoiceEngine: ONE resident MiniCPM-o 4.5 process for the live loop.

Loads the model exactly once (no per-turn reload), builds the token2wav voice
cache from the reference clip at boot, and exposes the proven streaming API in a
shape the state machine can drive:

    prefill_system(sid)                  -- seed voice clone + (optional) RAG facts
    prefill_user_chunk(sid, a16k, last)  -- push a 16k mono user chunk
    generate(sid, cancel_event)          -- yield (audio24k, text_delta), checking
                                            cancel_event every chunk (cooperative
                                            barge-in cancel: stop + abandon generator)
    reset_for_new_turn()                 -- reset_session(reset_token2wav_cache=False),
                                            return a fresh session_id (voice cache kept)
    asr_text(a16k)                       -- text-only transcription for RAG retrieval

Signatures mirror minicpm_stream.py / fifa_voice.py exactly. GPU0 only is the
caller's responsibility (CUDA_VISIBLE_DEVICES=0).
"""
import time
import uuid

import librosa
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "openbmb/MiniCPM-o-4_5"
# keep audio encoder / TTS / vision in full precision; only 4-bit the LLM
KEEP_FP = ["vpm", "apm", "resampler", "tts", "audio", "vision", "Token2wav", "embed", "tokenizer"]
VC_SUFFIX = ("Please assist users while maintaining this voice style. Answer seriously and in high "
             "quality. Chat in a highly human-like, oral style. You are a helpful assistant.")
DEFAULT_SYSTEM = "You are a warm, concise English voice assistant. Keep replies short and spoken."


def _to_np_audio(wav_chunk):
    if wav_chunk is None:
        return None
    if torch.is_tensor(wav_chunk):
        return wav_chunk.reshape(-1).float().cpu().numpy()
    return np.asarray(wav_chunk, dtype=np.float32).reshape(-1)


class LiveVoiceEngine:
    def __init__(self, ref_audio_path, model_id=MODEL_ID, quant="int4",
                 retriever=None, system_text=DEFAULT_SYSTEM,
                 response_max_new_tokens=256):
        self.model_id = model_id
        self.quant = quant
        self.retriever = retriever
        self.system_text = system_text
        self.response_max_new_tokens = response_max_new_tokens

        load_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if quant == "int4":
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=KEEP_FP)
            load_kw["device_map"] = "cuda"
            self.model = AutoModel.from_pretrained(model_id, **load_kw).eval()
        else:
            self.model = AutoModel.from_pretrained(model_id, **load_kw).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model.init_tts()

        # voice cache is built ONCE at boot from the reference clip; never per turn
        self.ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        self.model.init_token2wav_cache(self.ref_audio)

        self._sid = None

    # ------------------------------------------------------------------ session
    def new_session_id(self):
        self._sid = f"hervoice-live-{uuid.uuid4().hex[:12]}"
        return self._sid

    def reset_for_new_turn(self):
        """Abandon any in-flight turn and start clean, KEEPING the voice cache."""
        self.model.reset_session(reset_token2wav_cache=False)
        return self.new_session_id()

    # ------------------------------------------------------------------ prefill
    def prefill_system(self, sid, extra_facts=None):
        """Prefill the system message: voice clone + optional grounding facts."""
        sys_text = self.system_text
        if extra_facts:
            sys_text = (sys_text + " Answer using ONLY these verified facts:\n" + extra_facts +
                        "\nIf the facts don't cover it, say so briefly.")
        self.model.streaming_prefill(
            session_id=sid,
            msgs=[{"role": "system",
                   "content": [sys_text, "Clone the voice in the provided audio prompt.",
                               self.ref_audio, VC_SUFFIX]}],
            tokenizer=self.tok)

    # MiniCPM-o's streaming audio encoder pools with audio_pool_step=5: a chunk
    # that produces < 5 conv frames makes avg_pool1d output length 0 and raises
    # RuntimeError ("Output size is too small"), which (uncaught) kills the loop's
    # consumer thread. Empirically a trailing chunk < 1024 samples (after at least
    # one full chunk in the same turn) triggers it. Pad short chunks to a safe
    # floor so a tiny leftover / empty end-of-turn buffer can never crash.
    MIN_PREFILL_SAMPLES = 1600       # 0.1 s @ 16k; comfortably above the 1024 cliff

    def prefill_user_chunk(self, sid, audio16k, is_last):
        """Push one 16k mono user audio chunk (is_last=True only on final chunk)."""
        a = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        if a.size < self.MIN_PREFILL_SAMPLES:
            a = np.concatenate([a, np.zeros(self.MIN_PREFILL_SAMPLES - a.size,
                                            dtype=np.float32)])
        self.model.streaming_prefill(
            session_id=sid,
            msgs=[{"role": "user", "content": [a]}],
            is_last_chunk=bool(is_last),
            tokenizer=self.tok)

    # ------------------------------------------------------------------ generate
    def generate(self, sid, cancel_event=None, max_new_tokens=None):
        """Stream the reply; yield (audio24k_np_or_None, text_delta).

        Cooperative barge-in cancel: cancel_event is checked BEFORE each yield and
        after each produced chunk. On cancel we stop iterating and let the
        generator be abandoned (the caller then reset_session()s).
        """
        mnt = max_new_tokens or self.response_max_new_tokens
        gen = self.model.streaming_generate(
            session_id=sid, generate_audio=True, tokenizer=self.tok, max_new_tokens=mnt)
        for wav_chunk, new_text in gen:
            if cancel_event is not None and cancel_event.is_set():
                # abandon the generator immediately; do NOT continue the old turn
                try:
                    gen.close()
                except Exception:
                    pass
                return
            audio = _to_np_audio(wav_chunk)
            if (audio is not None and audio.size == 0):
                audio = None
            yield audio, (new_text or "")

    # ------------------------------------------------------------------ ASR (RAG)
    def asr_text(self, audio16k):
        """Text-only transcription of the user turn (used to drive RAG retrieval)."""
        a = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        out = self.model.chat(
            msgs=[{"role": "user",
                   "content": [a, "Transcribe the user's question exactly, text only."]}],
            tokenizer=self.tok, generate_audio=False, max_new_tokens=64)
        return out.strip() if isinstance(out, str) else str(out).strip()

    def retrieve_facts(self, query, k=3):
        """RAG: return a newline bullet list of facts (or None) for the system prompt."""
        if self.retriever is None or not query:
            return None
        try:
            hits = self.retriever.retrieve(query, k=k)
        except Exception:
            return None
        if not hits:
            return None
        return "\n".join(f"- {c}" for _, c in hits)

    @staticmethod
    def peak_vram_gb():
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
        return 0.0
