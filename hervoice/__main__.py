#!/usr/bin/env python3
"""HerVoice CLI: one-command Bengali voice-to-voice turn.

  spoken Bengali question -> ASR -> brain (optional gated RAG) -> Bengali answer wav

Run (GPU0 only; never touch GPU1):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv-hervoice/bin/python -m hervoice [flags]

Modular pipeline, sequential GPU residency, turn-based (not full-duplex, not a
single end-to-end network). Prints state lines and writes a manifest.json with
explicit success/failure states (no fake success).
"""
import os
import sys
import json
import argparse

from hervoice.core import HerVoice, SELFCHECK_TEXT  # noqa: F401

DEFAULT_AUDIO = "examples/in_bn_question.wav"
DEFAULT_OUT = "runs/hervoice_demo/answer.wav"
MANIFEST_DIR = "runs/hervoice_demo"


def _write_manifest(payload):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    path = os.path.join(MANIFEST_DIR, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hervoice", description="Bengali voice-to-voice (modular, turn-based)")
    ap.add_argument("--audio", default=DEFAULT_AUDIO, help="spoken Bengali question wav")
    ap.add_argument("--text", default=None, help="skip ASR, use this Bengali text")
    ap.add_argument("--kb", default=None, help="enable gated RAG over this KB (e.g. fifa_kb.md)")
    ap.add_argument("--no-rag", action="store_true", help="force plain mode (ignore --kb)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="answer wav path")
    ap.add_argument("--self-check", action="store_true",
                    help="synthesize a fixed Bengali sentence, re-ASR it, report CER/WER, exit")
    ap.add_argument("--device", default="cuda:0", help="default cuda:0")
    args = ap.parse_args(argv)

    kb_path = None if args.no_rag else args.kb
    hv = HerVoice(device=args.device, kb_path=kb_path)

    # ---- self-check path: no ASR of a user question, fixed sentence only ----
    if args.self_check:
        sc = hv.self_check(args.out)
        print(f"[selfcheck] text={sc['selfcheck_text']}")
        print(f"[tts] status={sc['tts_status']} out={sc.get('out_path')}")
        print(f"[selfcheck] asr={sc['selfcheck_asr']}")
        print(f"[selfcheck] CER={sc['selfcheck_cer']} WER={sc['selfcheck_wer']} "
              f"(re-ASR intelligibility proxy, NOT naturalness)")
        manifest = {
            **hv.model_info(),
            "mode": "self_check",
            "selfcheck_text": sc["selfcheck_text"],
            "selfcheck_asr": sc["selfcheck_asr"],
            "selfcheck_cer": sc["selfcheck_cer"],
            "selfcheck_wer": sc["selfcheck_wer"],
            "tts_status": sc["tts_status"],
            "out_path": sc.get("out_path"),
        }
        mpath = _write_manifest(manifest)
        print(f"[manifest] {mpath}")
        return 0 if sc["tts_status"] == "ok" else 1

    manifest = {**hv.model_info(), "mode": "voice_turn", "audio_in": None}

    # ---- 1. transcript (ASR or provided --text) ----
    if args.text:
        transcript = args.text.strip()
        hv.asr_transcript = transcript
        print(f"[asr] transcript={transcript} (provided via --text, ASR skipped)")
    else:
        try:
            transcript = hv.transcribe(args.audio)
            manifest["audio_in"] = args.audio
            print(f"[asr] transcript={transcript}")
        except Exception as e:  # noqa: BLE001
            print(f"[asr] status=asr_failed err={e}")
            manifest.update({"asr_transcript": None, "asr_status": "asr_failed", "error": str(e)})
            print(f"[manifest] {_write_manifest(manifest)}")
            return 1
    manifest["asr_transcript"] = transcript

    # ---- 2. RAG (optional, gated, non-blocking) ----
    rag = hv.retrieve(transcript)
    print(f"[rag] status={rag['status']} score={rag.get('score')}")
    manifest["rag_status"] = rag["status"]
    manifest["retrieved_chunk"] = rag.get("retrieved_chunk")
    context = rag.get("context")  # only set when status == rag_ok

    # ---- 3. brain ----
    try:
        answer = hv.think(transcript, context=context)
        print(f"[brain] answer={answer}")
    except Exception as e:  # noqa: BLE001
        print(f"[brain] status=brain_failed err={e}")
        manifest.update({"answer_text": None, "brain_status": "brain_failed", "error": str(e)})
        print(f"[manifest] {_write_manifest(manifest)}")
        return 1
    manifest["answer_text"] = answer

    # ---- 4. say ----
    tts = hv.say(answer, args.out)
    print(f"[tts] status={tts['status']} dur={tts.get('dur_s')} rms={tts.get('rms')}"
          + (f" reason={tts.get('reason')}" if tts.get("reason") else ""))
    manifest["tts_status"] = tts["status"]
    manifest["tts_dur_s"] = tts.get("dur_s")
    manifest["tts_rms"] = tts.get("rms")
    manifest["out_path"] = tts.get("out_path")

    if tts["status"] == "ok":
        print(f"[out] path={tts['out_path']}")
    else:
        print("[out] path=None (tts_failed; no wav written)")

    print(f"[manifest] {_write_manifest(manifest)}")
    return 0 if tts["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
