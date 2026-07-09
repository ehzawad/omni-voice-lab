"""ARCH 1: the CASCADED voice architecture (STT -> LLM -> TTS, three separate nets).

An open local analog of GPT-Live's "Standard Voice Mode": three independent models chained,
with the honest latency win coming from STAGE PIPELINING -- streaming the LLM output and
synthesizing each sentence as soon as it completes, overlapping TTS of sentence N with LLM
generation of sentence N+1. First audio arrives before the LLM finishes the whole answer.

Reuses hervoice.modular workers read-only (ASR 8091, llama-server 8090, TTS 8092).
"""
