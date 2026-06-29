#!/usr/bin/env python3
"""Shared benchmark definition for the Pipeline A vs Pipeline B comparison.

Kept dependency-free (no torch) so both venvs can import it. Each runner writes a
results_{a,b}.json; make_comparison.py merges them and computes WER.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_VOICE = os.path.join(ROOT, "examples", "ref_female.wav")

# (id, audio path, reference transcript) -- fixed English prompt set
PROMPTS = [
    ("p1_capital", os.path.join(ROOT, "examples", "in_en_question.wav"),
     "What is the capital of France?"),
    ("p2_fifa", os.path.join(ROOT, "examples", "in_fifa_question.wav"),
     "How many times has Brazil won the men's World Cup, and which years?"),
    ("p3_math", os.path.join(ROOT, "examples", "bench", "p3_math.wav"),
     "What is twelve multiplied by eight?"),
    ("p4_advice", os.path.join(ROOT, "examples", "bench", "p4_advice.wav"),
     "I feel nervous before a job interview. Can you give me one quick tip?"),
]

SYSTEM_PROMPT = "You are a helpful voice assistant. Answer in one or two concise spoken sentences."
MAX_NEW_TOKENS = 120
OUTDIR = os.path.join(ROOT, "examples", "bench")
