#!/usr/bin/env python3
"""Persistent TTS worker: load Qwen3-TTS ONCE, keep it resident, serve warm synthesis over HTTP.

Runs in .venv-qwen-audio (qwen-tts pins transformers 4.57.x). Pays the ~model-LOAD cost once at
startup; every turn is warm generation only. NOTE: warm generation is still autoregressive -- it
takes seconds for a multi-second answer. This worker removes the model-LOAD tax, NOT the
generation time; the next lever is sentence-chunked streaming / a faster TTS backend.

Start (GPU0 only; never GPU1):

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv-qwen-audio/bin/python -m hervoice.modular.tts_worker --port 8092

API (localhost only):
  GET  /health           -> {"status":"ok","loaded":true}
  POST /synth {"text": <str>, "out_path": <path>, "ref_text"?: <str>, "ref_audio"?: <path>}
        -> tts.synth(...) dict: {status,out_path,sr,duration_s,rms,latency_s,[reason]}
  POST /synth_stream {"text": <str>, "out_prefix": <path>, "ref_text"?: <str>, "ref_audio"?: <path>}
        -> tts.synth_stream(...) dict: sentence-chunked synth. Splits text into sentences, writes
           each chunk wav (out_prefix_00.wav, _01.wav, ...) as soon as it is ready, plus a
           concatenated out_prefix_full.wav. Returns {status,n_sentences,n_ok,ttfa_s,total_s,
           full_wav,full_duration_s,sr,chunks:[{index,text,wav,duration_s,status,
           cumulative_latency_s,[reason]}]}. TTFA = cumulative latency when the FIRST valid chunk
           was written -> lowers *perceived* latency (first audio sooner), not total generation.

The tts.synth() failure guards (empty text, degenerate/silent audio -> tts_failed, no wav) are
preserved verbatim -- this worker only changes WHERE the model lives, not the honesty guards.
synth_stream applies the SAME guards per chunk; a failing chunk is recorded tts_failed and skipped
(no fabrication) without aborting the rest.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hervoice.modular import tts as tts_mod

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE = {"loaded": False}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok" if STATE["loaded"] else "loading",
                             "loaded": STATE["loaded"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/synth", "/synth_stream"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            text = req.get("text", "")
            ref_text = req.get("ref_text")
            kwargs = {}
            if req.get("ref_audio"):
                kwargs["ref_audio"] = req["ref_audio"]
            if self.path == "/synth_stream":
                out_prefix = req["out_prefix"]
                os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
                # synth_stream applies tts.synth's guards per chunk (empty/degenerate -> skipped).
                r = tts_mod.synth_stream(text, out_prefix, ref_text=ref_text, **kwargs)
            else:
                out_path = req["out_path"]
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                # tts.synth keeps its own failure guards (empty text, degenerate audio).
                r = tts_mod.synth(text, out_path, ref_text=ref_text, **kwargs)
            self._send(200, r)
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    args = ap.parse_args()

    print("[tts_worker] loading Qwen3-TTS ...", flush=True)
    t0 = time.time()
    tts_mod.load()
    print(f"[tts_worker] loaded in {time.time()-t0:.2f}s; warming up ...", flush=True)
    t1 = time.time()
    warm_out = os.path.join(ROOT, "runs", "modular", "tts_warmup.wav")
    os.makedirs(os.path.dirname(warm_out), exist_ok=True)
    w = tts_mod.synth("Warming up the speech engine.", warm_out)
    print(f"[tts_worker] warmup synth {time.time()-t1:.2f}s -> status={w.get('status')} "
          f"dur={w.get('duration_s')}s", flush=True)
    STATE["loaded"] = True

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[tts_worker] READY on http://{args.host}:{args.port} (pid {os.getpid()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[tts_worker] shutting down", flush=True)
        srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
