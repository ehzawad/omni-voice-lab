#!/usr/bin/env python3
"""The three Bengali models, CO-RESIDENT in one process.

HerVoice's original Bengali path uses sequential GPU residency (load -> use -> free per
stage) because faster-whisper large-v3 + Qwen2.5-3B + Orpheus-3B + SNAC do not fit in 24 GB
together; docs/HERVOICE_DEMO.md measures ~71 s warm per turn and attributes it to that
load/free cycle rather than to inference.

Swapping two stages for much smaller models removes the constraint:

    ASR   ehzawad/stt_bn_fastconformer_ctc   115.6 M  (was faster-whisper large-v3, ~1.5 B)
    brain Qwen2.5-3B-Instruct                  3.1 B  (unchanged)
    TTS   ehzawad/indicf5-bangla-tts           337 M  (was Orpheus-3B + SNAC)

so all three stay resident and a turn costs inference only.

The IndicF5 sampling path here is the one from the training recipe
(github.com/ehzawad/indicf5-bangla-tts, scripts/f5_infer.py) and must stay identical to it:
nfe 32, cfg 2.0, sway -1.0, Euler, byte-ratio duration, RMS match to 0.1 then restore,
0.15 s cross-fade between chunks. Two IndicF5 defaults are load-bearing and are pinned in
f5_common.read_cfg / passed explicitly here -- see the note on use_epss below.
"""
import os
from collections.abc import Mapping
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f5_common import MEL, build_model, load_weights, read_cfg   # noqa: E402
from f5_text import chunk_bn, est_duration_frames, normalize      # noqa: E402

ASR_ID = "ehzawad/stt_bn_fastconformer_ctc"
BRAIN_ID = "Qwen/Qwen2.5-3B-Instruct"
BRAIN_ID_7B = "Qwen/Qwen2.5-7B-Instruct"   # fits the 20 GB budget alongside ASR+TTS
TTS_REPO = "ehzawad/indicf5-bangla-tts"
TTS_BASE_REPO = "ai4bharat/IndicF5"

# ai4bharat's own reference prompt and its transcript, straight from the IndicF5 model card.
# The BnTTS speaker's own reference clip is corpus audio and is not redistributable, so the
# released prompt is the default voice; pass your own ref_wav/ref_text to change it.
DEFAULT_REF_REPO = TTS_BASE_REPO
DEFAULT_REF_FILE = "prompts/PAN_F_HAPPY_00001.wav"
DEFAULT_REF_TEXT = (
    "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, "
    "ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)

SR_IN = 16000
SR_OUT = 24000


def vram_gb():
    """PyTorch ALLOCATOR bytes only -- NOT whole-process VRAM.

    Allocations outside the caching allocator (CUDA context, cuBLAS/cuDNN workspaces, NCCL)
    are invisible here, so this always UNDER-reports. For capacity decisions use NVML per
    process (`nvidia-smi --query-compute-apps=pid,used_memory`), which is what the service
    supervisor reports.
    """
    return round(torch.cuda.memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0.0


def peak_vram_gb():
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0.0


# --------------------------------------------------------------------------------- ASR
class BnAsr:
    """FastConformer-CTC, greedy, no LM. 16 kHz mono float32 in, normalised Bengali out."""

    def __init__(self, model_id=ASR_ID, device="cuda"):
        import nemo.collections.asr as nemo_asr
        self.m = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location=device)
        self.m.eval()
        self.device = device

    def transcribe(self, audio16k):
        a = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        # 8x subsampling + the conv front-end: a very short clip decodes to nothing and
        # NeMo can raise on an empty batch. Treat sub-100 ms turns as silence.
        if a.size < SR_IN // 10:
            return ""
        with torch.inference_mode():
            out = self.m.transcribe([a], batch_size=1, verbose=False)
        if not out:
            return ""
        first = out[0]
        return (getattr(first, "text", first) or "").strip()


# ------------------------------------------------------------------------------- brain
class BnBrain:
    """Qwen2.5-3B-Instruct, streamed token by token so the first sentence can start
    synthesising while the rest is still being written."""

    def __init__(self, model_id=BRAIN_ID, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.device = device

    def stream(self, system, user, max_new_tokens=256, cancel_event=None, timeout=120.0):
        """Yield decoded text deltas. Stops early when cancel_event is set.

        Two failure modes are handled explicitly, because both hang forever otherwise:
          * apply_chat_template returns a BatchEncoding (not a bare tensor) on transformers
            5.x, so the token ids must be unwrapped before generate() sees them;
          * if generate() raises inside the worker thread, nothing ever puts the sentinel
            into the streamer queue and the consumer blocks for good. The worker's exception
            is captured and re-raised here, and the streamer has a timeout as a backstop.
        """
        import threading

        from transformers import TextIteratorStreamer

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )
        # BatchEncoding subclasses UserDict, NOT dict -- isinstance(enc, dict) is False for it,
        # so test for Mapping or a bare tensor gets wrapped as input_ids and generate() dies.
        if not isinstance(enc, Mapping):         # older transformers return a bare tensor
            enc = {"input_ids": enc}
        enc = {k: v.to(self.device) for k, v in enc.items() if hasattr(v, "to")}

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True,
                                        timeout=timeout)
        err = {}

        def _worker():
            try:
                self.m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                streamer=streamer)
            except BaseException as e:           # noqa: BLE001 -- must not die silently
                err["e"] = e
                streamer.end()                   # unblock the consumer

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        try:
            for piece in streamer:
                if cancel_event is not None and cancel_event.is_set():
                    break
                if piece:
                    yield piece
        finally:
            th.join(timeout=5.0)
            if err:
                raise RuntimeError(f"brain generation failed: {err['e']!r}") from err["e"]


# --------------------------------------------------------------------------------- TTS
class BnTts:
    """IndicF5 flow matching + frozen Vocos, one fixed reference voice.

    Flow matching runs a fixed number of function evaluations over the WHOLE utterance mel
    before any audio exists, so there is no way to stream *within* a sentence. The unit of
    both latency and cancellation is therefore one chunk of `chunk_bn`.
    """

    def __init__(self, repo=TTS_REPO, ref_wav=None, ref_text=None, device="cuda",
                 nfe=32, cfg=2.0, sway=-1.0, speed=1.0, max_bytes=400):
        import soundfile as sf
        import torchaudio
        from huggingface_hub import hf_hub_download
        from vocos import Vocos

        ckpt = hf_hub_download(repo, "model.safetensors")
        vocab = hf_hub_download(repo, "checkpoints/vocab.txt")
        model, _, _ = build_model(vocab, read_cfg(None))
        missing, unexpected = load_weights(model, ckpt, prefer="ema")
        bad = [k for k in missing if not k.startswith("mel_spec")]
        assert not bad and not unexpected, (bad[:8], unexpected[:8])
        self.model = model.to(device).float().eval()
        self.voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval().to(device)

        if ref_wav is None:
            ref_wav = hf_hub_download(DEFAULT_REF_REPO, DEFAULT_REF_FILE)
            ref_text = DEFAULT_REF_TEXT
        y, sr = sf.read(ref_wav, dtype="float32")
        ref = torch.from_numpy(y)[None]
        if ref.ndim == 3:
            ref = ref.mean(-1)
        if sr != SR_OUT:
            ref = torchaudio.functional.resample(ref, sr, SR_OUT)
        self.rms = float(torch.sqrt(torch.mean(ref ** 2)))
        self.target_rms = 0.1
        if self.rms < self.target_rms:
            ref = ref * self.target_rms / self.rms
        self.ref = ref.to(device)
        self.ref_frames = self.ref.shape[-1] // MEL["hop_length"]
        rt = normalize(ref_text)
        self.ref_text = rt if rt.endswith(("।", ".", "?", "!")) else rt + "।"

        self.device, self.nfe, self.cfg, self.sway = device, nfe, cfg, sway
        self.speed, self.max_bytes = speed, max_bytes

    def chunks(self, text):
        return chunk_bn(normalize(text), self.max_bytes)

    def synth_chunk(self, chunk, seed=1234, nfe=None):
        """One chunk -> float32 @ 24 kHz. Not interruptible; this call is the cancel unit.

        nfe is per call: mutating self.nfe from a request handler races other requests.
        """
        steps = int(nfe or self.nfe)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        dur = est_duration_frames(self.ref_frames, self.ref_text + " ", chunk, self.speed)
        with torch.inference_mode():
            gen, _ = self.model.sample(
                cond=self.ref, text=[self.ref_text + " " + chunk], duration=dur,
                steps=steps, cfg_strength=self.cfg, sway_sampling_coef=self.sway,
                seed=seed,
                use_epss=False,   # IndicF5's vendored sampler has no EPSS; upstream defaults it ON
            )
            mel = gen[:, self.ref_frames:, :].float().permute(0, 2, 1)
            w = self.voc.decode(mel).squeeze().cpu().numpy()
        if self.rms < self.target_rms:
            w = w * self.rms / self.target_rms
        return w.astype(np.float32)


def load_all(device="cuda", tts_repo=TTS_REPO, ref_wav=None, ref_text=None, verbose=True):
    """Load all three and report VRAM after each. Returns (asr, brain, tts, report)."""
    rep = {}
    t = time.time(); asr = BnAsr(device=device)
    rep["asr"] = dict(load_s=round(time.time() - t, 1), vram_gb=vram_gb())
    if verbose: print(f"  ASR   loaded {rep['asr']['load_s']:>5}s  resident {rep['asr']['vram_gb']} GiB", flush=True)

    t = time.time(); brain = BnBrain(device=device)
    rep["brain"] = dict(load_s=round(time.time() - t, 1), vram_gb=vram_gb())
    if verbose: print(f"  brain loaded {rep['brain']['load_s']:>5}s  resident {rep['brain']['vram_gb']} GiB", flush=True)

    t = time.time(); tts = BnTts(repo=tts_repo, ref_wav=ref_wav, ref_text=ref_text, device=device)
    rep["tts"] = dict(load_s=round(time.time() - t, 1), vram_gb=vram_gb())
    if verbose: print(f"  TTS   loaded {rep['tts']['load_s']:>5}s  resident {rep['tts']['vram_gb']} GiB", flush=True)

    rep["resident_total_gb"] = vram_gb()
    return asr, brain, tts, rep
