#!/usr/bin/env python3
"""Tiny ASR worker, run in .venv-hervoice (has faster-whisper 1.2.1).

Used by pipeline.py as a subprocess for two honest, objective purposes:
  1. transcribe the USER turn accurately for the delegation controller
     (moshiko's own inner monologue is its REPLY, not a verbatim user ASR);
  2. transcribe Moshi's TEACHER-FORCED output wav to objectively check that
     Moshi actually vocalised the brain's answer (close the loop with evidence).

Runs on CPU (int8) to avoid contending for GPU0 with Moshi + the brain.
Usage: python _asr_worker.py <wav_path>  ->  prints the transcript to stdout.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU; do not touch any GPU

from faster_whisper import WhisperModel  # noqa: E402


def main():
    wav = sys.argv[1]
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(wav, language="en", beam_size=1)
    text = " ".join(s.text.strip() for s in segments).strip()
    print(text)


if __name__ == "__main__":
    main()
