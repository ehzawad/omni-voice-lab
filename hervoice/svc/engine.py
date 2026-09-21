#!/usr/bin/env python3
"""The gateway's view of the three services: one object, three HTTP endpoints.

Cancellation is checked at every point where work can still be avoided:

  * before and during the brain stream -- the SSE response is CLOSED, which is what actually
    frees the sequence in vLLM. Merely stopping to read would leave it generating into the
    KV cache.
  * before each TTS chunk -- a chunk is a single non-interruptible flow-matching call
    (~1611 ms measured at NFE 16), so the worst case is one chunk. Not starting a chunk we
    already know is stale is the only real lever, and it is taken here.
"""
import json
import logging
import re
import time
import urllib.error
import urllib.request

import numpy as np

from hervoice.svc import config as C

log = logging.getLogger("engine")

SENT_MARKS = ("।", "?", "!")


def _post(url, data, headers, timeout):
    return urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers), timeout=timeout)


_MD = re.compile(r"(\*\*|__|\*|`|#{1,6}\s|^\s*[-*]\s+)", re.M)


def _despeak_markdown(text):
    """LLMs emit markdown; a TTS would read the asterisks aloud. Strip formatting only --
    never touch letters, dandas or digits."""
    return _MD.sub("", text).replace("  ", " ")


def _split_sentence(buf):
    best = -1
    for m in SENT_MARKS:
        i = buf.find(m)
        if i >= 0 and (best < 0 or i < best):
            best = i
    if best < 0:
        return None, buf
    return buf[:best + 1].strip(), buf[best + 1:]


class ServiceEngine:
    def __init__(self, asr_url=None, tts_url=None, llm_url=None, model=None,
                 system=None, nfe=None, timeout=None):
        self.asr_url = asr_url or C.ASR_URL
        self.tts_url = tts_url or C.TTS_URL
        self.llm_url = llm_url or C.LLM_URL
        self.model = model or C.LLM_MODEL
        self.system = system or C.SYSTEM_PROMPT
        self.nfe = nfe or C.TTS_NFE
        self.timeout = timeout or C.HTTP_TIMEOUT_S

    # ---------------------------------------------------------------------- ASR
    def transcribe(self, audio16k):
        pcm = np.ascontiguousarray(audio16k, dtype="<f4").tobytes()
        t = time.time()
        try:
            r = _post(f"{self.asr_url}/transcribe", pcm,
                      {"Content-Type": "application/octet-stream"}, self.timeout)
            d = json.load(r)
        except (urllib.error.URLError, OSError) as e:
            log.error("ASR call failed: %r", e)
            return {"text": "", "ms": round((time.time() - t) * 1000, 1), "error": repr(e)}
        d.setdefault("ms", 0.0)
        return d

    # -------------------------------------------------------------------- brain
    def _brain_stream(self, messages, cancel):
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": C.LLM_MAX_TOKENS, "temperature": C.LLM_TEMPERATURE, "stream": True,
        }).encode()
        r = _post(f"{self.llm_url}/v1/chat/completions", body,
                  {"Content-Type": "application/json"}, self.timeout)
        try:
            for raw in r:
                if cancel.is_set():
                    break
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
        finally:
            # Closing is what frees the sequence server-side; do it on every exit path.
            try:
                r.close()
            except Exception:
                pass

    # ---------------------------------------------------------------------- TTS
    def _chunks(self, text):
        try:
            r = _post(f"{self.tts_url}/chunk", json.dumps({"text": text}).encode(),
                      {"Content-Type": "application/json"}, self.timeout)
            return json.load(r).get("chunks", [])
        except (urllib.error.URLError, OSError) as e:
            log.error("TTS /chunk failed: %r", e)
            return []

    def _synth(self, chunk, seed):
        body = json.dumps({"text": chunk, "seed": seed, "nfe": self.nfe}).encode()
        r = _post(f"{self.tts_url}/synthesize", body,
                  {"Content-Type": "application/json"}, self.timeout)
        return np.frombuffer(r.read(), dtype="<f4")

    # -------------------------------------------------------------------- turn
    def respond(self, messages, cancel, on_delta, on_sentence, on_audio):
        """Stream a reply for `messages` (system + history + current user turn).

        Callbacks: on_delta(text), on_sentence(idx, text) when a sentence is handed to TTS,
        on_audio(pcm, idx) per synthesised chunk, on_sentence_done(idx) is NOT a callback --
        completeness is signalled by returning; the caller tracks per-sentence chunk sets.

        Returns (generated_text, emitted_text). `emitted_text` is what was HANDED TO THE
        SOCKET, not what the listener heard: the browser may still discard it on a cancel.
        Memory must therefore be committed from playback acknowledgements (turnloop ledger),
        never from this return value.
        """
        buf, n_sent, generated, spoken = "", 0, [], []
        for delta in self._brain_stream(messages, cancel):
            if cancel.is_set():
                break
            on_delta(delta)
            buf += delta
            generated.append(delta)
            sent, buf = _split_sentence(buf)
            while sent:
                if cancel.is_set():
                    return "".join(generated), " ".join(spoken)
                if self._speak(sent, n_sent, cancel, on_sentence, on_audio):
                    spoken.append(sent)
                n_sent += 1
                sent, buf = _split_sentence(buf)
        tail = buf.strip()
        if tail and not cancel.is_set():
            if self._speak(tail, n_sent, cancel, on_sentence, on_audio):
                spoken.append(tail)
        return "".join(generated), " ".join(spoken)

    def _speak(self, sentence, idx, cancel, on_sentence, on_audio):
        """Synthesise one sentence. Returns True only if ALL its audio was emitted."""
        sentence = _despeak_markdown(sentence).strip()
        if not sentence:
            return False
        on_sentence(idx, sentence)
        chunks = self._chunks(sentence)
        for ci, ch in enumerate(chunks):
            if cancel.is_set():
                return False    # the only real cancel lever: do not START a stale chunk
            try:
                pcm = self._synth(ch, seed=1234 + 100 * idx + ci)
            except (urllib.error.URLError, OSError) as e:
                log.error("TTS /synthesize failed: %r", e)
                return False
            if cancel.is_set():
                return False    # finished, but already stale -- drop rather than play
            on_audio(pcm, idx)
        return bool(chunks)
