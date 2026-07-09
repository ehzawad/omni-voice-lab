#!/usr/bin/env python3
"""delegate.py -- the DELEGATION CONTROLLER.

GPT-Live's design: the fast full-duplex front handles conversation, but when the
user asks something that needs real reasoning or facts, it hands the turn to a
stronger reasoning model. Here the front is Moshi and the "stronger model" is a
local Qwen3.5-4B GGUF served by llama-server (OpenAI-compatible) on 127.0.0.1:8090.

This module: (1) classifies whether a user turn needs delegation, and (2) calls
the brain and returns a concise answer.
"""
import re
import time

import requests

BRAIN_URL = "http://127.0.0.1:8090/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8090/health"

# Heuristic triggers: is this a question / does it need facts or reasoning?
_TRIGGER_WORDS = [
    "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
    "how many", "how much", "calculate", "compute", "explain", "define",
    "list", "name", "count", "difference", "compare",
]


def brain_healthy(timeout: float = 2.0) -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=timeout)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


def classify(user_text: str):
    """Return (delegate: bool, reason: str). Cheap, transparent heuristic."""
    t = (user_text or "").strip().lower()
    if not t:
        return False, "empty transcript"
    if "?" in t:
        return True, "contains a question mark"
    words = re.findall(r"[a-z']+", t)
    hit = next((w for w in _TRIGGER_WORDS if w in words
                or (" " in w and w in t)), None)
    if hit:
        return True, f"question/reasoning trigger word: {hit!r}"
    return False, "no question/reasoning trigger -> handle on the front"


def ask_brain(user_text: str, max_tokens: int = 120, timeout: float = 60.0):
    """Call the local reasoning brain. Returns (answer_text, latency_s, raw)."""
    payload = {
        "model": "qwen3.5-4b",
        "messages": [
            {"role": "system",
             "content": ("You are a concise voice assistant. Answer in ONE short "
                         "spoken sentence, plain words, no markdown, no lists.")},
            {"role": "user", "content": user_text},
        ],
        # Qwen3.5 is a reasoning model: without disabling thinking it returns
        # empty content in `content` (the reasoning goes elsewhere).
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    r = requests.post(BRAIN_URL, json=payload, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    data = r.json()
    answer = data["choices"][0]["message"]["content"].strip()
    # collapse whitespace so it teacher-forces cleanly into Moshi's text stream
    answer = re.sub(r"\s+", " ", answer)
    return answer, dt, data
