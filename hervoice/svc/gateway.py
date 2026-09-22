#!/usr/bin/env python3
"""Gateway: the browser's WebSocket, the Silero VAD, and the turn state machine.

This is the only process the browser talks to. It holds no model of its own except the VAD
(a few MB on CPU), so the GPU budget belongs entirely to the three model services.

    HV_GW_TOKEN=... .venv-bnweb/bin/python -m hervoice.svc.gateway
    # then from a laptop:  ssh -N -L 8100:127.0.0.1:8100 <box>
    #                      open http://localhost:8100/

Binding to 127.0.0.1 and reaching it through an SSH tunnel is deliberate and does double duty:
it keeps a microphone endpoint off a shared box's network, and http://localhost is a SECURE
CONTEXT, so getUserMedia works without a TLS certificate. A token is still required, because
anyone else with an account on this box can reach 127.0.0.1.
"""
import asyncio
import json
import logging
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.svc import config as C           # noqa: E402
from hervoice.svc import protocol as P         # noqa: E402
from hervoice.svc.conversation import Conversation  # noqa: E402
from hervoice.svc.engine import ServiceEngine  # noqa: E402
from hervoice.svc.turnloop import TurnLoop     # noqa: E402

log = logging.getLogger("gateway")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_sessions = 0
_sessions_lock = threading.Lock()


def _make_detector():
    """Build the VAD and WARM IT before any real audio arrives.

    Silero's first load + first inference in a fresh process takes seconds. Measured: with
    the detector built INSIDE the connection handler, a client that started streaming right
    after `hello` piled up ~5 s of frames, the bounded queue dropped 47 of them (~940 ms),
    and the first transcript lost its onset ('বাংলাদেশের রাজধানীর' -> 'সে রাজধানীর').
    So the cold path runs ONCE at process start (see warm_at_startup), per-connection
    construction is then fast, and the client is told to wait for `ready` anyway.
    """
    from hervoice.live.turn_detector import TurnDetector
    d = TurnDetector(min_silence_ms=C.MIN_SILENCE_MS, min_speech_ms=C.MIN_SPEECH_MS)
    silence = np.zeros(512, dtype=np.float32)
    for _ in range(4):
        d.process(silence)
    d.reset()
    return d


def warm_at_startup():
    """Pay Silero's cold load once, before the first connection exists."""
    import time
    t = time.time()
    _make_detector()
    log.info("VAD warm in %.2fs", time.time() - t)


def build_app():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="hervoice-gateway")

    @app.on_event("startup")
    def _startup():
        warm_at_startup()

    if os.path.isdir(STATIC):
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/ready")
    def ready():
        import urllib.request
        out = {}
        for name, url in (("asr", f"{C.ASR_URL}/ready"), ("tts", f"{C.TTS_URL}/ready"),
                          ("llm", f"{C.LLM_URL}/health")):
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    out[name] = r.status == 200
            except Exception:
                out[name] = False
        ok = all(out.values())
        return JSONResponse({"ready": ok, "services": out, "config": C.summary()},
                            status_code=200 if ok else 503)

    @app.get("/")
    def index():
        p = os.path.join(STATIC, "index.html")
        if not os.path.isfile(p):
            return JSONResponse({"error": "client not built"}, status_code=404)
        return FileResponse(p)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        global _sessions
        await sock.accept()
        loop = asyncio.get_running_loop()

        # --- admission -------------------------------------------------------
        try:
            hello = await asyncio.wait_for(sock.receive_text(), timeout=10.0)
            msg = json.loads(hello)
        except Exception:
            await sock.close(code=4400); return
        if msg.get("type") != "hello":
            await sock.close(code=4400); return
        # Fail CLOSED. An unset token used to mean "allow everyone", which on a shared box
        # means any other account here can open a microphone session.
        if not C.GW_TOKEN or msg.get("token") != C.GW_TOKEN:
            await sock.send_text(json.dumps({"type": "error", "message": "bad token"}))
            await sock.close(code=4401); return
        with _sessions_lock:
            if _sessions >= C.MAX_SESSIONS:
                await sock.send_text(json.dumps(
                    {"type": "error", "message": "busy: one conversation at a time"}))
                await sock.close(code=4409); return
            _sessions += 1

        outq: asyncio.Queue = asyncio.Queue(maxsize=256)

        def emit(ev):            # called from the loop thread
            loop.call_soon_threadsafe(_put, ("json", ev))

        def send_audio(epoch, seq, pcm):
            loop.call_soon_threadsafe(
                _put, ("bin", (epoch, seq, np.ascontiguousarray(pcm, dtype="<f4").tobytes())))

        def _put(item):
            try:
                outq.put_nowait(item)
            except asyncio.QueueFull:
                log.warning("client too slow; dropping outbound frame")

        engine = ServiceEngine()
        conv = Conversation(system=C.SYSTEM_PROMPT, max_turns=C.MEM_MAX_TURNS,
                            max_chars=C.MEM_MAX_CHARS)
        tl = TurnLoop(engine, _make_detector(), on_event=emit, on_audio=send_audio,
                      conversation=conv, sr=C.SR_IN, max_queue=C.MAX_INBOUND_FRAMES,
                      max_turn_s=C.MAX_TURN_SECONDS, min_silence_ms=C.MIN_SILENCE_MS)

        th = threading.Thread(target=tl.run, daemon=True)
        th.start()
        # Contract: the client MUST NOT stream audio until it has received `ready`. Frames
        # sent before this point sit in the socket buffer and arrive in a burst that the
        # bounded admission queue will (correctly) drop from.
        await sock.send_text(json.dumps({"type": "ready", "config": C.summary()}))

        async def pump():
            while True:
                kind, payload = await outq.get()
                if kind == "json":
                    await sock.send_text(json.dumps(payload, ensure_ascii=False))
                else:
                    epoch, seq, pcm = payload      # seq assigned by the turn loop's ledger
                    await sock.send_bytes(P.pack_audio(epoch, seq, C.SR_OUT, pcm))

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                m = await sock.receive()
                if m.get("type") == "websocket.disconnect":
                    break
                if (b := m.get("bytes")) is not None:
                    if len(b) % 4 == 0 and b:
                        tl.submit_frame(np.frombuffer(b, dtype="<f4"))
                elif (t := m.get("text")) is not None:
                    try:
                        d = json.loads(t)
                    except json.JSONDecodeError:
                        continue
                    if d.get("type") == "played":
                        tl.note_played(int(d.get("epoch", 0)), int(d.get("seq", 0)))
                    elif d.get("type") == "flushed":
                        tl.note_flushed(int(d.get("epoch", 0)))
                    elif d.get("type") == "reset":
                        conv.reset()
                        emit({"type": "memory", "turns_kept": 0, "reset": True})
                    elif d.get("type") == "stop":
                        break
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("ws error")
        finally:
            pump_task.cancel()
            tl.stop()                       # cancels any in-flight generation
            # join() is blocking; awaiting it in a thread keeps the event loop responsive.
            stopped = await asyncio.get_running_loop().run_in_executor(
                None, lambda: (th.join(timeout=15.0), not th.is_alive())[1])
            if stopped:
                with _sessions_lock:
                    _sessions -= 1
            else:
                # The slot is NOT released: work may still be touching the GPU, and handing
                # the budget to a new session while that is true is how two generations end
                # up running at once.
                log.error("turn loop did not stop within 15s; holding the session slot")
            try:
                await sock.close()
            except Exception:
                pass

    return app


app = build_app()

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not C.GW_TOKEN:
        raise SystemExit(
            "HV_GW_TOKEN is not set. This endpoint accepts live microphone audio and binds to "
            "127.0.0.1, which every other account on this shared box can reach. Refusing to "
            "start without a token.")
    uvicorn.run(app, host=C.GW_HOST, port=C.GW_PORT, log_level="warning")
