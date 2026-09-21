#!/usr/bin/env python3
"""ASR microservice: NeMo FastConformer-CTC, greedy, no LM.

Contract (deliberately tiny and stateless, so a retry is always safe):

    GET  /health                    -> {"ok": true}                 process is up
    GET  /ready                     -> {"ready": bool, ...}         model is loaded
    POST /transcribe                -> {"text": str, "ms": float}
         body: raw little-endian float32 mono PCM at 16 kHz, Content-Type
               application/octet-stream

Why the whole endpointed utterance rather than streaming chunks: this checkpoint is NOT a
cache-aware streaming model. Its config carries att_context_size [-1, -1] and
causal_downsampling false, i.e. unrestricted attention context, so independently decoding
chunks changes the context each frame sees and damages boundary recognition. The measured
warm cost of decoding a whole short utterance is ~52 ms, consistent with the 78 ms median in
the owner's own published benchmark for this model, so there is nothing worth streaming for.

    HV_ASR_PORT=8001 .venv-bnweb/bin/python -m hervoice.svc.asr_service
"""
import logging
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.svc import config as C   # noqa: E402

log = logging.getLogger("asr-svc")

_model = None
_ready = False
_err = None
_gpu = threading.Lock()   # non-async endpoints share a threadpool; serialise GPU access


def _load():
    global _model, _ready, _err
    try:
        import torch  # noqa: F401
        import nemo.collections.asr as nemo_asr
        t = time.time()
        m = nemo_asr.models.ASRModel.from_pretrained(C.ASR_MODEL, map_location=C.DEVICE)
        m.eval()
        _model = m
        _ready = True
        log.info("loaded %s in %.1fs", C.ASR_MODEL, time.time() - t)
    except BaseException as e:   # noqa: BLE001 -- readiness must report, never hang
        _err = repr(e)
        log.exception("ASR load failed")


def build_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="hervoice-asr")

    @app.on_event("startup")
    def _startup():
        _load()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/ready")
    def ready():
        return JSONResponse({"ready": _ready, "model": C.ASR_MODEL, "error": _err},
                            status_code=200 if _ready else 503)

    @app.post("/transcribe")
    async def transcribe(request: Request):
        if not _ready:
            raise HTTPException(status_code=503, detail=f"model not ready: {_err}")
        cl = request.headers.get("content-length")
        max_bytes = int(C.MAX_TURN_SECONDS * C.SR_IN * 4) + 1024
        if cl is not None and int(cl) > max_bytes:
            raise HTTPException(status_code=413, detail=f"body over {max_bytes} bytes")
        raw = await request.body()
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail=f"body over {max_bytes} bytes")
        if len(raw) % 4:
            raise HTTPException(status_code=400, detail="body must be float32 PCM (length % 4 == 0)")
        audio = np.frombuffer(raw, dtype="<f4")
        # 8x subsampling plus the conv front-end: a sub-100 ms clip decodes to nothing and
        # can raise on an empty batch downstream. Report it as silence, not as an error.
        if audio.size < C.SR_IN // 10:
            return {"text": "", "ms": 0.0, "samples": int(audio.size), "short": True}
        max_s = C.MAX_TURN_SECONDS
        if audio.size > int(max_s * C.SR_IN):
            audio = audio[-int(max_s * C.SR_IN):]     # keep the most recent speech
        t = time.time()
        import torch
        with _gpu, torch.inference_mode():
            out = _model.transcribe([np.ascontiguousarray(audio, dtype=np.float32)],
                                    batch_size=1, verbose=False)
        text = ""
        if out:
            first = out[0]
            text = (getattr(first, "text", first) or "").strip()
        return {"text": text, "ms": round((time.time() - t) * 1000, 1), "samples": int(audio.size)}

    return app


app = build_app()

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(app, host=C.ASR_HOST, port=C.ASR_PORT, log_level="warning")
