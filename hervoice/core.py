#!/usr/bin/env python3
"""HerVoice engine.

A modular, turn-based Bengali voice-to-voice pipeline. NOT a single end-to-end
network and NOT full-duplex. Stages run in one process with sequential/lazy GPU
residency: each heavy model is loaded, used, then freed (del + gc.collect() +
torch.cuda.empty_cache()) before the next heavy model loads. The brain (~6GB)
and orpheus (~7GB) are never deliberately co-resident.

Stages:
  transcribe()  faster-whisper large-v3 (language="bn")        -> Bengali text
  retrieve()    Qwen3-Embedding-0.6B, cross-lingual, GATED      -> context|None
  think()       Qwen2.5-3B-Instruct (bf16), Bengali-only prompt -> answer text
  say()         orpheus-bangla-tts + LoRA + SNAC, validated     -> answer wav

The Bengali TTS LoRA improves intelligibility (re-ASR CER 0.640 -> 0.498 on
FLEURS-20, an ASR proxy, NOT human naturalness). RAG is optional and gated; it
never blocks or crashes the voice loop.
"""
import os
import sys
import gc
import numpy as np
import torch

# --- locate repo root + ft/ so we can reuse the proven helpers ---
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
_FT = os.path.join(_REPO, "ft")
for _p in (_REPO, _FT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bn_tts as B  # noqa: E402  (ft/bn_tts.py: build_gen_input, tokens_to_codes, codes_to_audio, ...)

# ---- model ids / paths (all cached locally; no downloads expected) ----
ASR_MODEL = "large-v3"
BRAIN_ID = "Qwen/Qwen2.5-3B-Instruct"
ORPHEUS_ID = "asif00/orpheus-bangla-tts"
ORPHEUS_ADAPTER = os.path.join(_REPO, "adapters", "orpheus_bn_ivr_lora")
SNAC_ID = "hubertsiuzdak/snac_24khz"
EMB_ID = "Qwen/Qwen3-Embedding-0.6B"

# ---- knobs ----
BRAIN_MAX_NEW_TOKENS = 96
RAG_TOP_K = 3
RAG_SIM_THRESHOLD = 0.45          # below this best-sim -> rag_disabled_retrieval_weak
# (measured: Bengali FIFA queries score 0.55-0.68 against the English KB; an
#  off-topic Bengali query tops out ~0.41, so 0.45 cleanly gates off-topic out.)
TTS_MIN_DUR, TTS_MAX_DUR = 0.3, 30.0
TTS_MIN_RMS = 0.005
TTS_MIN_UNIQUE_CODES = 8          # degenerate token-loop guard

# fixed Bengali sentence for the intelligibility self-check
SELFCHECK_TEXT = "আমি বাংলায় কথা বলতে পারি।"


def _free(*objs):
    """Drop python refs, then force CUDA cache release (sequential residency)."""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _norm_bn(s):
    """NFC + strip punctuation + collapse whitespace (mirrors ft/eval_bn_tts.norm_bn)."""
    import unicodedata
    import re
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[।,.?!৷;:\"'`()\[\]{}—\-–…]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---- brain prompts: force fluent Bengali, short, spoken, no markdown/Latin ----
_BRAIN_RULES = (
    "তুমি একজন বাংলা ভয়েস সহকারী। সবসময় শুধুমাত্র সাবলীল বাংলায় উত্তর দাও। "
    "এক বা দুটি ছোট কথ্য বাক্যে উত্তর দাও। কোনো মার্কডাউন, তালিকা, ইংরেজি বা "
    "ল্যাটিন অক্ষর ব্যবহার করবে না। সংখ্যা অঙ্কে নয়, বাংলা শব্দে লেখো।"
)
_BRAIN_GROUNDED = (
    _BRAIN_RULES
    + " নিচের তথ্যের উপর ভিত্তি করে উত্তর দাও। তথ্যে উত্তর না থাকলে বিনয়ের সাথে "
    "বাংলায় বলো যে তুমি জানো না — অনুমান করবে না।"
)


class HerVoice:
    """Turn-based Bengali voice-to-voice engine with sequential GPU residency.

    Public API:
        __init__(device="cuda:0", kb_path=None)
        transcribe(audio_path) -> str                # Bengali transcript; raises on failure
        retrieve(query)        -> dict               # {status, retrieved_chunk, context, score}
        think(question, context=None) -> str         # Bengali answer text
        say(text, out_path)    -> dict               # {status, dur_s, rms, out_path}
        self_check(out_path)   -> dict               # {text, asr, cer, ...} intelligibility proxy

    Attributes (filled as stages run): asr_transcript, rag, answer_text, tts.
    """

    def __init__(self, device="cuda:0", kb_path=None):
        self.device = device
        self.kb_path = kb_path
        # per-run state (also consumed by __main__ to build the manifest)
        self.asr_transcript = None
        self.rag = {"status": "rag_off", "retrieved_chunk": None, "context": None, "score": None}
        self.answer_text = None
        self.tts = {"status": None, "dur_s": None, "rms": None, "out_path": None}

    # ---- model id surface for the manifest ----
    def model_info(self):
        return {
            "asr_model": ASR_MODEL,
            "brain_id": BRAIN_ID,
            "tts_id": ORPHEUS_ID,
            "tts_adapter": ORPHEUS_ADAPTER,
            "snac_id": SNAC_ID,
            "embedding_id": EMB_ID,
            "device": self.device,
            "single_network": False,
            "duplex": False,
            "residency": "sequential",
        }

    # ------------------------------------------------------------------ ASR
    def transcribe(self, audio_path):
        """Load whisper, transcribe Bengali, free. Raises RuntimeError on failure."""
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError(f"asr_failed: audio not found: {audio_path}")
        from faster_whisper import WhisperModel

        ct = "float16" if str(self.device).startswith("cuda") else "int8"
        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        model = None
        try:
            model = WhisperModel(ASR_MODEL, device=dev, compute_type=ct)
            segs, _ = model.transcribe(audio_path, language="bn", beam_size=5)
            text = " ".join(s.text for s in segs).strip()
        except Exception as e:  # noqa: BLE001
            _free(model)
            raise RuntimeError(f"asr_failed: {e}")
        finally:
            _free(model)
        if not text:
            raise RuntimeError("asr_failed: empty transcript")
        self.asr_transcript = text
        return text

    # ------------------------------------------------------------------ RAG
    def retrieve(self, query):
        """Cross-lingual gated retrieval. Never raises into the voice loop.

        Returns dict {status, retrieved_chunk, context, score}. status is one of
        rag_off / rag_ok / rag_disabled_retrieval_weak / rag_error.
        """
        if not self.kb_path:
            self.rag = {"status": "rag_off", "retrieved_chunk": None, "context": None, "score": None}
            return self.rag
        retr = None
        try:
            import fifa_rag

            # Pass the KB path explicitly so a custom --kb is honoured (mutating the
            # module global has no effect: load_chunks binds its default at def-time).
            retr = fifa_rag.FifaRetriever(device=self.device, kb_path=self.kb_path)
            hits = retr.retrieve(query, k=RAG_TOP_K)  # [(score, chunk), ...] sorted desc
            best = hits[0][0] if hits else 0.0
            if not hits or best < RAG_SIM_THRESHOLD:
                self.rag = {
                    "status": "rag_disabled_retrieval_weak",
                    "retrieved_chunk": None,
                    "context": None,
                    "score": round(float(best), 4),
                }
            else:
                context = "\n".join(f"- {c}" for _, c in hits)
                self.rag = {
                    "status": "rag_ok",
                    "retrieved_chunk": hits[0][1],
                    "context": context,
                    "score": round(float(best), 4),
                }
        except Exception as e:  # noqa: BLE001  RAG must never break the loop
            self.rag = {
                "status": "rag_error",
                "retrieved_chunk": None,
                "context": None,
                "score": None,
                "error": str(e),
            }
        finally:
            _free(retr)
        return self.rag

    # ---------------------------------------------------------------- BRAIN
    def think(self, question, context=None):
        """Qwen2.5-3B -> short fluent Bengali answer. Raises on failure."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = model = None
        try:
            tok = AutoTokenizer.from_pretrained(BRAIN_ID)
            model = AutoModelForCausalLM.from_pretrained(
                BRAIN_ID, torch_dtype=torch.bfloat16
            ).to(self.device).eval()

            if context:
                system = _BRAIN_GROUNDED
                user = f"তথ্য:\n{context}\n\nপ্রশ্ন: {question}"
            else:
                system = _BRAIN_RULES
                user = f"প্রশ্ন: {question}"
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=BRAIN_MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=tok.eos_token_id,
                )
            ans = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:  # noqa: BLE001
            _free(model, tok)
            raise RuntimeError(f"brain_failed: {e}")
        finally:
            _free(model, tok)
        if not ans:
            raise RuntimeError("brain_failed: empty answer")
        self.answer_text = ans
        return ans

    # ------------------------------------------------------------------ TTS
    def _validate_audio(self, audio, codes):
        """Return (ok, dur_s, rms, reason)."""
        if audio is None or audio.size == 0:
            return False, 0.0, 0.0, "empty"
        dur = float(len(audio)) / B.SNAC_SR
        rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))
        if not (TTS_MIN_DUR <= dur <= TTS_MAX_DUR):
            return False, dur, rms, f"dur_out_of_range({dur:.2f}s)"
        if rms <= TTS_MIN_RMS:
            return False, dur, rms, f"rms_too_low({rms:.4f})"
        if len(codes) < 7 or len(set(codes)) < TTS_MIN_UNIQUE_CODES:
            return False, dur, rms, "degenerate_token_loop"
        return True, dur, rms, "ok"

    def _load_orpheus(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from snac import SNAC

        tok = AutoTokenizer.from_pretrained(ORPHEUS_ID)
        model = AutoModelForCausalLM.from_pretrained(
            ORPHEUS_ID, torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        if ORPHEUS_ADAPTER and os.path.isdir(ORPHEUS_ADAPTER):
            model = PeftModel.from_pretrained(model, ORPHEUS_ADAPTER).to(self.device).eval()
        snac = SNAC.from_pretrained(SNAC_ID).to(self.device).eval()
        return tok, model, snac

    def _generate_once(self, tok, model, snac, text):
        inp = B.build_gen_input(tok, text).to(self.device)
        with torch.no_grad():
            out = model.generate(
                inp,
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=B.END_SPEECH,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inp.shape[1]:]
        codes = B.tokens_to_codes(gen)
        audio = B.codes_to_audio(snac, codes, self.device)
        return audio, codes

    def say(self, text, out_path):
        """Synthesize Bengali speech with validation + ONE retry.

        On success writes the wav and returns {status:'ok', dur_s, rms, out_path}.
        On failure writes NO wav and returns {status:'tts_failed', reason, ...}.
        """
        import soundfile as sf

        tok = model = snac = None
        try:
            tok, model, snac = self._load_orpheus()
            ok = False
            dur = rms = 0.0
            reason = "not_run"
            audio = None
            for attempt in range(2):  # generate once, retry once
                audio, codes = self._generate_once(tok, model, snac, text)
                ok, dur, rms, reason = self._validate_audio(audio, codes)
                if ok:
                    break
            if ok:
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                sf.write(out_path, audio, B.SNAC_SR)
                self.tts = {
                    "status": "ok",
                    "dur_s": round(dur, 2),
                    "rms": round(rms, 4),
                    "out_path": out_path,
                }
            else:
                # no fake-success wav written
                self.tts = {
                    "status": "tts_failed",
                    "dur_s": round(dur, 2),
                    "rms": round(rms, 4),
                    "out_path": None,
                    "reason": reason,
                }
        except Exception as e:  # noqa: BLE001
            self.tts = {
                "status": "tts_failed",
                "dur_s": None,
                "rms": None,
                "out_path": None,
                "reason": str(e),
            }
        finally:
            _free(model, snac, tok)
        return self.tts

    # ----------------------------------------------------------- SELF-CHECK
    def self_check(self, out_path):
        """Synthesize a FIXED Bengali sentence, re-ASR it, report CER/WER.

        Intelligibility proxy ONLY (re-ASR), not human naturalness. Reuses the
        same norm_bn + jiwer.cer/wer as ft/eval_bn_tts.py.
        """
        result = {
            "selfcheck_text": SELFCHECK_TEXT,
            "selfcheck_asr": None,
            "selfcheck_cer": None,
            "selfcheck_wer": None,
            "tts_status": None,
            "out_path": None,
        }
        tts = self.say(SELFCHECK_TEXT, out_path)
        result["tts_status"] = tts["status"]
        result["out_path"] = tts.get("out_path")
        if tts["status"] != "ok":
            return result
        try:
            asr = self.transcribe(out_path)  # loads/frees whisper internally
        except Exception as e:  # noqa: BLE001
            result["selfcheck_asr"] = f"asr_failed: {e}"
            return result
        import jiwer

        ref, hyp = _norm_bn(SELFCHECK_TEXT), _norm_bn(asr)
        result["selfcheck_asr"] = asr
        if ref:
            result["selfcheck_cer"] = round(jiwer.cer(ref, hyp), 4)
            result["selfcheck_wer"] = round(jiwer.wer(ref, hyp), 4)
        return result
