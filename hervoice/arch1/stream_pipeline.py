#!/usr/bin/env python3
"""ARCH 1 streaming cascade: STT -> (streamed) LLM -> sentence-pipelined TTS.

    wav -> ASR worker (8091) -> llama-server SSE (8090) -> per-sentence TTS (8092) -> wav chunks

The honest latency win of this cascaded architecture is STAGE PIPELINING, not sub-sentence audio
packet streaming. The `qwen_tts` package cannot emit audio packets incrementally: its
generate_voice_clone returns a FULL waveform (the non_streaming_mode flag only *simulates*
streaming text input). So we do the next best, genuinely measurable thing:

  * Stream the LLM output token-by-token (brain_stream.ask_stream, SSE).
  * Detect COMPLETED sentences in the growing buffer (chunk.split_sentences: all-but-last element
    is complete; the last is still being generated).
  * Dispatch each completed sentence to the TTS worker /synth ON A BACKGROUND THREAD as soon as it
    completes, so TTS of sentence N overlaps LLM generation of sentence N+1. The FIRST sentence's
    audio is ready long before the whole answer finishes.

We also run the OLD sequential BASELINE (wait for the full LLM answer, THEN synthesize it once) on
the same turn, so the improvement is a MEASURED delta, not a claim.

Honest failure states (same taxonomy as modular/serve.py):
  empty transcript      -> asr_failed   (no LLM, no wav)
  empty LLM answer      -> brain_failed (no TTS, no wav)
  no chunk ever passes  -> tts_failed   (per-chunk guards live in tts.synth; a failing chunk is
                                          recorded and skipped, it does not abort the turn)

Run in .venv-funasr (has requests + soundfile + numpy); the 3 workers must be up:

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv-funasr/bin/python -m hervoice.arch1.stream_pipeline \
      --wav examples/in_fifa_question.wav --out-prefix runs/arch1/turn_fifa --ref-text "$REF"

NOTE on t0: on this headless box the input is a single-shot file, so t0 (turn start) is when we
begin processing the file, not a real microphone voice-onset. Recorded honestly; no live mic.
"""
import argparse
import json
import os
import queue
import threading
import time
import urllib.request

import numpy as np
import soundfile as sf

from hervoice.modular.brain import DEFAULT_URL as BRAIN_URL, ask as brain_ask
from hervoice.modular.chunk import split_sentences
from hervoice.arch1.brain_stream import ask_stream

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASR_URL = "http://127.0.0.1:8091"
TTS_URL = "http://127.0.0.1:8092"


def _post(url, obj, timeout=300):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _health(url, timeout=5):
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:
            return json.loads(r.read()).get("loaded") is True
    except Exception:
        return False


def _brain_health(timeout=5):
    try:
        with urllib.request.urlopen(f"{BRAIN_URL}/health", timeout=timeout) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def _concat_ok_chunks(chunks, full_path):
    """Concatenate the ok chunk wavs (same sample rate) in order -> full_path. Returns (dur, sr)."""
    concat, sr_ref = [], None
    for c in chunks:
        if c["status"] != "ok" or not c.get("wav"):
            continue
        w, sr = sf.read(c["wav"], dtype="float32")
        if sr_ref is None:
            sr_ref = sr
        if sr == sr_ref:
            concat.append(w)
    if not concat:
        return None, None, None
    full = np.concatenate(concat)
    sf.write(full_path, full, sr_ref)
    return full_path, round(len(full) / sr_ref, 3) if sr_ref else 0.0, sr_ref


def run_turn(wav, out_prefix, ref_text=None, run_baseline=True):
    """One ARCH 1 streaming cascade turn + the sequential baseline. Returns a metrics dict."""
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    result = {
        "input_wav": os.path.abspath(wav),
        "asr_model": "qwen3-asr-0.6b",
        "asr_hf_id": "Qwen/Qwen3-ASR-0.6B-hf",
        "brain": "Qwen3.5-4B-Q4_K_M.gguf (llama-server, enable_thinking=False)",
        "tts_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "ref_text": ref_text,
    }
    t0 = time.time()

    # --- ASR ---
    a = _post(f"{ASR_URL}/transcribe", {"wav": os.path.abspath(wav), "language": "en"})
    t_asr = time.time() - t0
    transcript = (a.get("text") or "").strip()
    result["asr_s"] = round(t_asr, 3)
    result["transcript"] = transcript
    result["asr_features"] = a.get("features")
    if not transcript:
        result["status"] = "asr_failed"
        return result

    # --- streaming LLM + sentence-pipelined TTS -----------------------------------------------
    # Background TTS worker thread: pulls (index, sentence) off a queue, synthesizes each with a
    # blocking /synth POST, records per-chunk timing. It runs WHILE the LLM keeps streaming, which
    # is the whole point -- TTS of sentence N overlaps generation of sentence N+1.
    q = queue.Queue()
    chunks = []
    tts_state = {"first_audio_s": None, "first_chunk_bytes": None}
    tts_lock = threading.Lock()

    def tts_worker():
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            idx, sent = item
            wav_path = f"{out_prefix}_{idx:02d}.wav"
            try:
                r = _post(f"{TTS_URL}/synth",
                          {"text": sent, "out_path": os.path.abspath(wav_path),
                           "ref_text": ref_text})
            except Exception as e:
                r = {"status": "tts_failed", "reason": f"{type(e).__name__}: {e}"}
            cum = round(time.time() - t0, 3)
            entry = {"index": idx, "text": sent, "status": r.get("status"),
                     "duration_s": r.get("duration_s", 0.0), "cumulative_latency_s": cum}
            if r.get("status") == "ok":
                entry["wav"] = r.get("out_path")
                with tts_lock:
                    if tts_state["first_audio_s"] is None:
                        tts_state["first_audio_s"] = cum  # turn-relative TTFA
                        try:
                            tts_state["first_chunk_bytes"] = os.path.getsize(r["out_path"])
                        except OSError:
                            tts_state["first_chunk_bytes"] = None
            else:
                entry["wav"] = None
                entry["reason"] = r.get("reason")
            with tts_lock:
                chunks.append(entry)
            q.task_done()

    worker = threading.Thread(target=tts_worker, daemon=True)
    worker.start()

    buf = ""
    emitted = 0
    llm_first_token_s = None
    t_llm0 = time.time()
    for delta in ask_stream(transcript):
        if llm_first_token_s is None:
            llm_first_token_s = round(time.time() - t0, 3)
        buf += delta
        sents = split_sentences(buf)
        # All sentences except the last are complete; the last is still being generated.
        while emitted < len(sents) - 1:
            q.put((emitted, sents[emitted]))
            emitted += 1
    llm_total_s = round(time.time() - t_llm0, 3)

    answer = buf.strip()
    result["llm_first_token_s"] = llm_first_token_s
    result["llm_total_s"] = llm_total_s
    result["answer_text"] = answer

    if not answer:
        q.put(None)
        worker.join()
        result["n_sentences"] = 0
        result["status"] = "brain_failed"
        return result

    # Flush the trailing partial/final sentence.
    final_sents = split_sentences(buf)
    while emitted < len(final_sents):
        q.put((emitted, final_sents[emitted]))
        emitted += 1
    result["n_sentences"] = len(final_sents)

    q.put(None)
    worker.join()
    chunks.sort(key=lambda c: c["index"])

    n_ok = sum(1 for c in chunks if c["status"] == "ok")
    full_path, full_dur, full_sr = _concat_ok_chunks(chunks, f"{out_prefix}_full.wav")
    total_s = round(time.time() - t0, 3)

    result.update({
        "chunks": chunks,
        "tts_chunk_count": len(chunks),
        "tts_n_ok": n_ok,
        "measured_first_audio_s": tts_state["first_audio_s"],   # streaming TTFA (turn-relative)
        "tts_first_chunk_bytes": tts_state["first_chunk_bytes"],
        "full_wav": full_path,
        "full_duration_s": full_dur,
        "full_sr": full_sr,
        "total_s": total_s,
    })
    if n_ok == 0:
        result["status"] = "tts_failed"
    else:
        result["status"] = "ok"

    # --- BASELINE: sequential path (full LLM answer, THEN synthesize the whole thing once) ------
    # First audio in the baseline cannot arrive until the ENTIRE answer is generated AND the whole
    # answer is synthesized once -> baseline_first_audio == baseline_total. To isolate the
    # PIPELINING effect from LLM sampling noise (temperature 0.3 re-samples a different answer), we
    # synthesize the EXACT SAME streaming answer as one shot and reuse the already-measured ASR and
    # streamed-LLM times. brain.ask (independent non-stream reproduction) is timed too, for
    # reference, but the reported speedup uses the same-answer baseline so it is confound-free.
    if run_baseline and result["status"] == "ok":
        base_wav = f"{out_prefix}_baseline_full.wav"
        tb0 = time.time()
        rb = _post(f"{TTS_URL}/synth",
                   {"text": answer, "out_path": os.path.abspath(base_wav), "ref_text": ref_text})
        t_tts_b = time.time() - tb0
        base_first = round(t_asr + llm_total_s + t_tts_b, 3)
        ans_b, t_brain_b, _ = brain_ask(transcript)  # reference: non-stream full-answer latency
        result["baseline"] = {
            "answer_text": answer,   # same answer synthesized whole (confound-free)
            "llm_s": llm_total_s,
            "tts_full_s": round(t_tts_b, 3),
            "baseline_first_audio_s": base_first,   # == baseline_total_s (single-shot)
            "baseline_total_s": base_first,
            "baseline_wav": rb.get("out_path"),
            "baseline_tts_status": rb.get("status"),
            "reference_nonstream_brain_ask_s": round(t_brain_b, 3),
            "reference_nonstream_answer": ans_b.strip(),
        }
        stream_ttfa = result["measured_first_audio_s"]
        if stream_ttfa is not None:
            speedup = round(base_first - stream_ttfa, 3)
            ratio = round(base_first / stream_ttfa, 3) if stream_ttfa > 0 else None
            result["improvement"] = {
                "first_audio_speedup_s": speedup,
                "first_audio_speedup_ratio": ratio,
                "note": ("For a single-sentence answer there is only one split point, so streaming "
                         "TTFA ~= baseline TTFA (the win requires >1 sentence). See a multi-sentence "
                         "turn for the real pipelining gain."
                         if len(final_sents) <= 1 else
                         "Streaming emits the first sentence's audio while the LLM is still "
                         "generating later sentences and before the whole answer is synthesized.")
            }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=os.path.join(ROOT, "examples", "in_fifa_question.wav"))
    ap.add_argument("--out-prefix", default=os.path.join(ROOT, "runs", "arch1", "turn"))
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    if not (_brain_health() and _health(ASR_URL) and _health(TTS_URL)):
        print(f"[arch1] workers DOWN: brain={_brain_health()} asr={_health(ASR_URL)} "
              f"tts={_health(TTS_URL)} -- start start_workers.sh first.", flush=True)
        raise SystemExit(1)

    m = run_turn(args.wav, args.out_prefix, ref_text=args.ref_text,
                 run_baseline=not args.no_baseline)
    print(json.dumps(m, indent=2))
    if m.get("status") == "ok":
        imp = m.get("improvement", {})
        print(f"[arch1] ok  ASR {m['asr_s']}s | LLM ttft {m['llm_first_token_s']}s "
              f"total {m['llm_total_s']}s | stream TTFA {m['measured_first_audio_s']}s | "
              f"baseline TTFA {m.get('baseline', {}).get('baseline_first_audio_s')}s | "
              f"speedup {imp.get('first_audio_speedup_s')}s "
              f"({m['n_sentences']} sentence(s), {m['tts_n_ok']} ok chunk(s))", flush=True)
    else:
        print(f"[arch1] FAILURE state: {m.get('status')}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
