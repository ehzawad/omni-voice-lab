#!/usr/bin/env python3
"""TTS microservice: IndicF5 flow matching + frozen Vocos, one fixed assistant voice.

Contract:

    GET  /health                 -> {"ok": true}
    GET  /ready                  -> {"ready": bool, ...}
    POST /chunk    {"text": str} -> {"chunks": [str, ...]}     danda-aware split, no GPU
    POST /synthesize             -> raw little-endian float32 mono PCM at 24 kHz
         {"text": str, "nfe": int?, "seed": int?}

ONE CHUNK PER REQUEST, on purpose. IndicF5 is flow matching: the solver runs a fixed number
of steps over the WHOLE chunk mel before a single sample exists, so a chunk is physically
non-interruptible. Making the chunk the request makes the unit of work, the unit of retry
and the unit of cancellation all the same thing -- to cancel, the caller simply does not send
the next request. Worst-case cancel latency is therefore one chunk: measured ~1553 ms at
NFE 16 and ~3074 ms at NFE 32 for ~2.5 s of Bengali audio.

/chunk is separated from /synthesize so the gateway can split text without holding the model:
the splitter is pure text processing and needs no GPU.

    HV_TTS_PORT=8002 .venv-bnweb/bin/python -m hervoice.svc.tts_service
"""
import logging
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.svc import config as C   # noqa: E402

log = logging.getLogger("tts-svc")

_tts = None
_ready = False
_err = None
# FastAPI runs non-async endpoints in a threadpool, so two requests CAN reach the GPU model
# at once. One flow-matching call already saturates the card, and overlapping them only makes
# both slower while doubling peak VRAM. Serialise explicitly.
_gpu = threading.Lock()

MAX_TEXT_BYTES = 2000
MIN_NFE, MAX_NFE = 4, 64


def _load():
    global _tts, _ready, _err
    try:
        from hervoice.bn.models import BnTts
        t = time.time()
        ref_wav = C.TTS_REF_WAV or None
        ref_text = C.TTS_REF_TEXT or None
        _tts = BnTts(repo=C.TTS_REPO, ref_wav=ref_wav, ref_text=ref_text, device=C.DEVICE,
                     nfe=C.TTS_NFE, cfg=C.TTS_CFG, sway=C.TTS_SWAY, speed=C.TTS_SPEED,
                     max_bytes=C.TTS_MAX_BYTES)
        _ready = True
        log.info("loaded %s in %.1fs (nfe=%d)", C.TTS_REPO, time.time() - t, C.TTS_NFE)
    except BaseException as e:   # noqa: BLE001
        _err = repr(e)
        log.exception("TTS load failed")


def build_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel

    class ChunkReq(BaseModel):
        text: str

    class SynthReq(BaseModel):
        text: str
        nfe: int | None = None
        seed: int = 1234

    app = FastAPI(title="hervoice-tts")

    @app.on_event("startup")
    def _startup():
        _load()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/ready")
    def ready():
        return JSONResponse({"ready": _ready, "repo": C.TTS_REPO, "nfe": C.TTS_NFE,
                             "sr": C.SR_OUT, "error": _err},
                            status_code=200 if _ready else 503)

    @app.post("/chunk")
    def chunk(req: ChunkReq):
        # Pure text; usable before the model finishes loading.
        from hervoice.bn.f5_text import chunk_bn, normalize
        text = normalize(req.text)
        if not text.strip():
            return {"chunks": []}
        return {"chunks": chunk_bn(text, C.TTS_MAX_BYTES)}

    @app.post("/synthesize")
    def synthesize(req: SynthReq):
        if not _ready:
            raise HTTPException(status_code=503, detail=f"model not ready: {_err}")
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="empty text")
        if len(req.text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise HTTPException(status_code=413, detail=f"text over {MAX_TEXT_BYTES} bytes; "
                                                       "split it with /chunk first")
        nfe = int(req.nfe or C.TTS_NFE)
        if not (MIN_NFE <= nfe <= MAX_NFE):
            raise HTTPException(status_code=400, detail=f"nfe must be {MIN_NFE}..{MAX_NFE}")
        t = time.time()
        # nfe is passed per call; mutating the shared model's attribute raced between threads.
        with _gpu:
            w = _tts.synth_chunk(req.text, seed=req.seed, nfe=nfe)
        ms = (time.time() - t) * 1000
        pcm = np.ascontiguousarray(w, dtype="<f4").tobytes()
        return Response(
            content=pcm, media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(C.SR_OUT), "X-Samples": str(len(w)),
                     "X-Synth-Ms": f"{ms:.1f}", "X-NFE": str(nfe),
                     "X-Audio-Seconds": f"{len(w)/C.SR_OUT:.3f}"},
        )

    return app


app = build_app()

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(app, host=C.TTS_HOST, port=C.TTS_PORT, log_level="warning")
