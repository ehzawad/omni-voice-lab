#!/usr/bin/env python3
"""Streaming brain client for ARCH 1: Qwen3.5-4B GGUF via llama-server, SSE token streaming.

This is the STREAMING counterpart of hervoice.modular.brain.ask (which returns the whole answer
in one shot). Same payload (system prompt, enable_thinking=False, temperature 0.3, max_tokens 200)
but with "stream": true, so llama-server emits OpenAI-style Server-Sent Events. We parse the
`data:` lines and yield content deltas as they arrive, letting the orchestrator start synthesizing
each completed sentence while the model keeps generating the next one.

Do NOT edit modular/brain.py -- this lives in arch1/ and only reads its constants.

Run in any venv with `requests` (e.g. .venv-funasr). llama-server must be up on 8090.
"""
import json
import time

import requests

from hervoice.modular.brain import DEFAULT_URL, SYSTEM_PROMPT


def ask_stream(question, base_url=DEFAULT_URL, system=SYSTEM_PROMPT,
               max_tokens=200, temperature=0.3, timeout=120.0):
    """Open the SSE stream and yield content-delta strings as they arrive.

    Yields str deltas. The caller times the first yield (time-to-first-token) and accumulates
    the deltas into the full answer. Matches brain.ask's payload exactly except stream=True.
    """
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Qwen3.5 reasoning switch -- must be off or content comes back empty.
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }
    with requests.post(f"{base_url}/v1/chat/completions", json=payload,
                       stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].lstrip()
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "What is the capital of France, and what is the capital of Japan?"
    t0 = time.time()
    first = None
    buf = []
    for d in ask_stream(q):
        if first is None:
            first = time.time() - t0
        buf.append(d)
    total = time.time() - t0
    print("Q:", q)
    print("A:", "".join(buf).strip())
    print(f"first_token_s={first:.3f} total_s={total:.3f}")
