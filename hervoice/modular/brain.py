#!/usr/bin/env python3
"""Brain client: Qwen3.5-4B GGUF served by llama-server, OpenAI chat API over HTTP.

The GGUF is a *reasoning* model. Left in thinking mode it burns the whole token budget on
`reasoning_content` and returns an empty `content`, so we disable thinking via
`chat_template_kwargs={"enable_thinking": false}` -- verified to yield clean final answers.

Start the server (GPU0, all layers offloaded); default port 8090 because 8080 was already taken
on this box:

  BIN=/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/llama.cpp/build/bin/llama-server
  GGUF=/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/q4b/Qwen3.5-4B-Q4_K_M.gguf
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $BIN -m $GGUF -ngl 99 \
      --host 127.0.0.1 --port 8090 -c 4096 --no-warmup > runs/modular/llama_server.log 2>&1 &
"""
import time
import requests

DEFAULT_URL = "http://127.0.0.1:8090"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Answer in one or two concise spoken sentences. "
    "Do not use lists, markdown, or emoji -- your reply will be read aloud."
)


def health(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


def ask(question: str, base_url: str = DEFAULT_URL, system: str = SYSTEM_PROMPT,
        max_tokens: int = 200, temperature: float = 0.3, timeout: float = 120.0):
    """Send one question, return (answer_text, latency_seconds, usage_dict)."""
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Qwen3.5 reasoning switch -- must be off or content comes back empty.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    dt = time.time() - t0
    data = r.json()
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    return text, dt, data.get("usage", {})


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "How many times has Brazil won the men's World Cup, and which years?"
    print("health:", health())
    ans, dt, usage = ask(q)
    print(f"Q: {q}")
    print(f"A: {ans}")
    print(f"latency={dt:.2f}s usage={usage}")
