#!/usr/bin/env python3
"""delegate_oss.py -- streaming delegation client for ARCH 3.

The delegated "brain" is gpt-oss-20B (OpenAI's open reasoning model) served by
llama-server (OpenAI-compatible, harmony chat template) on 127.0.0.1:8093.

gpt-oss is a harmony-format reasoning model: it first streams its chain of
thought on the `reasoning_content` channel, then the spoken answer on the
`content` channel. For ARCH 3 that split is a feature: while gpt-oss is still
*reasoning* (content still empty) the Moshi front keeps the stream alive with a
short backchannel; the moment the first `content` delta arrives we start
teacher-forcing Moshi to VOCALIZE the answer, word by word, as it streams.

Two entry points:
  ask(...)        -- blocking; returns (answer_text, latency_s, raw_json).
  ask_stream(...) -- SSE generator; yields ('reasoning', delta) and
                     ('content', delta) tuples in real time for async splicing.

This is the open local analog of GPT-Live delegating hard reasoning to a
stronger model. The delegate here is gpt-oss-20B (open weights), NOT GPT-5.5.
"""
import json
import re
import time

import requests

BASE = "http://127.0.0.1:8093"
CHAT_URL = f"{BASE}/v1/chat/completions"
HEALTH_URL = f"{BASE}/health"

SYSTEM_PROMPT = ("Answer in one or two concise spoken sentences, no markdown, "
                 "no lists, no emoji.")


def brain_healthy(timeout: float = 2.0) -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=timeout)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


def _payload(user_text: str, max_tokens: int, stream: bool):
    return {
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        # gpt-oss (harmony) spends tokens on reasoning first; keep it light so
        # the spoken answer arrives quickly, and give enough budget to REACH the
        # final channel (with too few tokens content comes back empty).
        "reasoning_effort": "low",
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def ask(user_text: str, max_tokens: int = 512, timeout: float = 120.0):
    """Blocking call. Returns (answer_text, latency_s, raw_json)."""
    t0 = time.time()
    r = requests.post(CHAT_URL, json=_payload(user_text, max_tokens, False),
                      timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    data = r.json()
    answer = (data["choices"][0]["message"].get("content") or "").strip()
    answer = re.sub(r"\s+", " ", answer)
    return answer, dt, data


def ask_stream(user_text: str, max_tokens: int = 512, timeout: float = 120.0):
    """Streaming call. Yields (channel, delta_text) as tokens arrive.

    channel is 'reasoning' (chain of thought) or 'content' (the spoken answer).
    """
    with requests.post(CHAT_URL, json=_payload(user_text, max_tokens, True),
                       timeout=timeout, stream=True) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except Exception:
                continue
            delta = d["choices"][0].get("delta", {})
            rc = delta.get("reasoning_content")
            if rc:
                yield "reasoning", rc
            c = delta.get("content")
            if c:
                yield "content", c
