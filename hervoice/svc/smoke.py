#!/usr/bin/env python3
"""End-to-end smoke test across the three services, with the same timing points as the
co-resident benchmark so the two are directly comparable.

Answers one question: what do the HTTP hops cost against the measured 2818 ms single-process
baseline (Qwen2.5-3B, NFE 16)? Everything here goes over the wire; nothing is imported from
the model layer except the text splitter, which is pure text.

    .venv-bnweb/bin/python -m hervoice.svc.smoke --audio examples/in_bn_question.wav
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.svc import config as C   # noqa: E402


def _post(url, data, headers, timeout):
    req = urllib.request.Request(url, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def transcribe(audio16k):
    pcm = np.ascontiguousarray(audio16k, dtype="<f4").tobytes()
    r = _post(f"{C.ASR_URL}/transcribe", pcm,
              {"Content-Type": "application/octet-stream"}, C.HTTP_TIMEOUT_S)
    return json.load(r)


def brain_stream(user_text, cancel=None, history=None):
    """Yield text deltas from vLLM's OpenAI-compatible SSE stream."""
    body = json.dumps({
        "model": C.LLM_MODEL,
        "messages": [{"role": "system", "content": C.SYSTEM_PROMPT}] + (history or [])
                    + [{"role": "user", "content": user_text}],
        "max_tokens": C.LLM_MAX_TOKENS, "temperature": C.LLM_TEMPERATURE, "stream": True,
    }).encode()
    r = _post(f"{C.LLM_URL}/v1/chat/completions", body,
              {"Content-Type": "application/json"}, C.HTTP_TIMEOUT_S)
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
        if delta:
            yield delta
        if cancel is not None and cancel.is_set():
            r.close()
            break


def split_chunks(text):
    r = _post(f"{C.TTS_URL}/chunk", json.dumps({"text": text}).encode(),
              {"Content-Type": "application/json"}, C.HTTP_TIMEOUT_S)
    return json.load(r)["chunks"]


def synth(chunk, seed=1234, nfe=None):
    body = {"text": chunk, "seed": seed}
    if nfe:
        body["nfe"] = nfe
    r = _post(f"{C.TTS_URL}/synthesize", json.dumps(body).encode(),
              {"Content-Type": "application/json"}, C.HTTP_TIMEOUT_S)
    raw = r.read()
    return np.frombuffer(raw, dtype="<f4"), float(r.headers.get("X-Synth-Ms", 0))


def first_sentence(buf):
    for mark in ("।", "?", "!"):
        i = buf.find(mark)
        if i >= 0:
            return buf[:i + 1].strip(), buf[i + 1:]
    return None, buf


def one_turn(audio, label, nfe=None):
    t0 = time.time()
    t = time.time()
    asr = transcribe(audio)
    asr_ms = (time.time() - t) * 1000
    if not asr["text"]:
        print(f"  [{label}] ASR empty"); return None
    buf, waves, sents, tts_ms = "", [], [], []
    t_first_tok = t_first_audio = None
    tgen = time.time()
    full = ""
    for piece in brain_stream(asr["text"]):
        if t_first_tok is None:
            t_first_tok = (time.time() - tgen) * 1000
        buf += piece; full += piece
        s, buf = first_sentence(buf)
        if s:
            for ci, ch in enumerate(split_chunks(s)):
                w, ms = synth(ch, seed=1234 + 100 * len(sents) + ci, nfe=nfe)
                tts_ms.append(ms); waves.append(w)
                if t_first_audio is None:
                    t_first_audio = (time.time() - t0) * 1000
            sents.append(s)
    tail = buf.strip()
    if tail:
        for ci, ch in enumerate(split_chunks(tail)):
            w, ms = synth(ch, seed=9000 + ci, nfe=nfe)
            tts_ms.append(ms); waves.append(w)
            if t_first_audio is None:
                t_first_audio = (time.time() - t0) * 1000
        sents.append(tail)
    wav = np.concatenate(waves) if waves else np.zeros(0, dtype=np.float32)
    rec = dict(label=label, asr_ms=round(asr_ms, 1), asr_svc_ms=asr["ms"],
               brain_first_token_ms=round(t_first_tok or 0, 1),
               tts_first_chunk_ms=round(tts_ms[0] if tts_ms else 0, 1),
               first_audio_ms=round(t_first_audio or 0, 1),
               turn_total_ms=round((time.time() - t0) * 1000, 1),
               audio_out_s=round(len(wav) / C.SR_OUT, 2),
               asr_text=asr["text"], reply=full.strip(), sentences=len(sents))
    print(f"  [{label}] ASR {rec['asr_ms']:6.0f} (svc {rec['asr_svc_ms']:.0f}) | brain-1st-tok "
          f"{rec['brain_first_token_ms']:7.0f} | TTS-1st {rec['tts_first_chunk_ms']:7.0f}"
          f" | FIRST AUDIO {rec['first_audio_ms']:7.0f} ms | {rec['audio_out_s']}s out", flush=True)
    return rec, wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="examples/in_bn_question.wav")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--out", default="runs/svc/smoke")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    audio, sr = sf.read(a.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    assert sr == C.SR_IN, f"expected {C.SR_IN} Hz, got {sr}"
    print(f"[cfg] {json.dumps(C.summary(), ensure_ascii=False)}")
    print(f"[in ] {a.audio} {len(audio)/sr:.2f}s")
    print("[warm] discarded warm-up turn", flush=True)
    one_turn(audio, "warmup", a.nfe)
    recs = []
    for i in range(a.turns):
        r = one_turn(audio, f"turn{i+1}", a.nfe)
        if r:
            recs.append(r[0]); last = r[1]
    if not recs:
        print("no turns"); return
    sf.write(os.path.join(a.out, "answer.wav"), last, C.SR_OUT)

    def med(k):
        v = sorted(x[k] for x in recs); return round(v[len(v) // 2], 1)

    out = dict(turns=len(recs), nfe=a.nfe or C.TTS_NFE, detail=recs,
               warm_median=dict(asr_ms=med("asr_ms"), brain_first_token_ms=med("brain_first_token_ms"),
                                tts_first_chunk_ms=med("tts_first_chunk_ms"),
                                first_audio_ms=med("first_audio_ms"), turn_total_ms=med("turn_total_ms")))
    json.dump(out, open(os.path.join(a.out, "smoke.json"), "w"), ensure_ascii=False, indent=1)
    m = out["warm_median"]
    print(f"\n[WARM MEDIAN over {len(recs)}] ASR {m['asr_ms']:.0f} ms | brain 1st token "
          f"{m['brain_first_token_ms']:.0f} ms | TTS 1st chunk {m['tts_first_chunk_ms']:.0f} ms")
    print(f"[WARM MEDIAN] >>> FIRST AUDIO {m['first_audio_ms']:.0f} ms <<< | full turn {m['turn_total_ms']:.0f} ms")
    print(f"[reply] {recs[-1]['reply']!r}")
    print(f"[out] {a.out}/answer.wav + smoke.json")


if __name__ == "__main__":
    main()
