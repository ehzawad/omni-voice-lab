#!/usr/bin/env python3
"""Persistent ASR worker: load Qwen3-ASR ONCE, keep it resident, serve warm inference over HTTP.

Runs in .venv-qwen-asr (transformers 5.x). The whole point is to pay the model LOAD + CUDA warmup
cost exactly once at startup, then every turn is warm inference only (no per-turn reload tax that
the one-shot subprocess path in pipeline.py pays).

Start (GPU0 only; never GPU1):

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv-qwen-asr/bin/python -m hervoice.modular.asr_worker --model qwen3-asr-0.6b --port 8091

API (localhost only):
  GET  /health              -> {"status":"ok","model":<key>,"loaded":true}
  POST /transcribe {"wav": <path>, "language"?: "en"}
        -> {"text","raw","features","latency_s","model_key"}   (from asr.transcribe)

The warmup transcription uses examples/in_fifa_question.wav so the first real turn is already warm.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hervoice.modular import asr as asr_mod

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WARMUP_WAV = os.path.join(ROOT, "examples", "in_fifa_question.wav")

STATE = {"model_key": None, "loaded": False}


def make_handler(model_key):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # keep stderr quiet
            pass

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok" if STATE["loaded"] else "loading",
                                 "model": STATE["model_key"], "loaded": STATE["loaded"]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/transcribe":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                wav = req["wav"]
                language = req.get("language", "en")
                if not os.path.isfile(wav):
                    self._send(400, {"error": f"wav not found: {wav}"})
                    return
                r = asr_mod.transcribe(model_key, wav, language=language)
                self._send(200, r)
            except Exception as e:  # never crash the worker on a bad turn
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-asr-0.6b", choices=list(asr_mod.ASR_MODELS))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--warmup-wav", default=WARMUP_WAV)
    args = ap.parse_args()

    STATE["model_key"] = args.model
    print(f"[asr_worker] loading {args.model} ...", flush=True)
    t0 = time.time()
    asr_mod.load(args.model)
    print(f"[asr_worker] loaded in {time.time()-t0:.2f}s; warming up ...", flush=True)
    if os.path.isfile(args.warmup_wav):
        t1 = time.time()
        w = asr_mod.transcribe(args.model, args.warmup_wav, language="en")
        print(f"[asr_worker] warmup transcribe {time.time()-t1:.2f}s -> {w['text']!r}", flush=True)
    STATE["loaded"] = True

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(args.model))
    print(f"[asr_worker] READY on http://{args.host}:{args.port} "
          f"(pid {os.getpid()}, model {args.model})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[asr_worker] shutting down", flush=True)
        srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
