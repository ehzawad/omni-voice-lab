# HerVoice

A Bengali-first, local, turn-based voice-to-voice assistant. You ask a question in spoken
Bengali, it transcribes, thinks, and speaks a Bengali answer back, on your own RTX A5000, in one
command.

This is an honest product doc. HerVoice is a modular pipeline, not a single end-to-end network,
and not full-duplex. Numbers below are measured from a real integration run, not aspirational.

## What it is

HerVoice is a single process that runs four stages in sequence:

1. ASR — faster-whisper large-v3, `language="bn"`, Bengali speech to text.
2. Brain — Qwen2.5-3B-Instruct (bf16), produces a short fluent Bengali answer.
3. TTS — `asif00/orpheus-bangla-tts` (Llama-3.2-3B) + the `orpheus_bn_ivr_lora` adapter + SNAC
   24 kHz decoder, the Bengali speech-out centerpiece.
4. Optional RAG — Qwen3-Embedding-0.6B over a knowledge base, gated and off by default.

It is explicitly **modular** (`single_network=false`), **turn-based** (`duplex=false`), and uses
**sequential GPU residency** (`residency=sequential`). The brain (~6 GB) and orpheus (~7 GB) are
never co-resident; each heavy model is loaded, used, then freed (`del` + `gc.collect()` +
`torch.cuda.empty_cache()`) before the next loads. Latency is the accepted cost of staying well
under 24 GB.

## One-command demo

The core demo is the plain Bengali voice loop. No RAG needed.

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv-hervoice/bin/python -m hervoice
```

This transcribes `examples/in_bn_question.wav`, thinks, and writes a Bengali answer wav to
`runs/hervoice_demo/answer.wav` plus a `manifest.json`.

## CLI

`python -m hervoice` with flags:

| Flag | Meaning |
| --- | --- |
| `--audio PATH` | spoken Bengali question (default `examples/in_bn_question.wav`) |
| `--text "..."` | skip ASR, use this Bengali text directly (isolation / fallback) |
| `--kb PATH` | enable gated RAG over this KB (e.g. `fifa_kb.md`); omit for plain mode |
| `--no-rag` | force plain mode |
| `--out PATH` | answer wav (default `runs/hervoice_demo/answer.wav`) |
| `--self-check` | synthesize a fixed Bengali sentence, re-ASR it, report CER/WER, exit |
| `--device` | default `cuda:0` |

It prints state lines while running:

```
[asr]   transcript=...
[rag]   status=...
[brain] answer=...
[tts]   status=... dur=... rms=...
[out]   path=...
```

and writes `runs/hervoice_demo/manifest.json` with model ids, adapter path, device, input audio,
ASR transcript, RAG status + retrieved chunk (if any), answer text, TTS status/duration/RMS,
output path, and (for `--self-check`) the self-check text, re-ASR, and CER.

Failure states are explicit: `asr_failed`, `brain_failed`, `tts_failed`,
`rag_disabled_retrieval_weak`. There is no fake success: `say()` validates the decoded audio
(non-empty, duration in [0.3, 30] s, RMS > 0.005, not a degenerate token loop), retries once, and
writes no wav on failure.

## Architecture

Sequential residency, one heavy model at a time:

```
audio --> [whisper large-v3 bn] --> transcript
                                      |
              (optional, gated) [Qwen3-Embedding-0.6B vs KB] --> context
                                      |
transcript (+context) --> [Qwen2.5-3B-Instruct bf16] --> Bengali answer (<=96 tokens)
                                      |
answer --> [orpheus-bangla-tts + orpheus_bn_ivr_lora + SNAC] --> answer.wav
```

The brain is prompted to answer in fluent Bengali, 1-2 short spoken sentences, no markdown, no
Latin/English code-switch, numbers spelled in Bengali words. When RAG context is present, the
brain is grounded-only and prefers a grounded refusal over hallucination.

### Optional gated RAG

RAG is off by default and is not the headline. With `--kb fifa_kb.md` HerVoice embeds the Bengali
transcript with Qwen3-Embedding-0.6B and retrieves top-3 chunks from the English KB
(cross-lingual). It is gated by a similarity threshold (`RAG_SIM_THRESHOLD = 0.45`):

- Bengali FIFA question -> `rag_ok`, similarity 0.682, grounds to the correct clean answer
  "আর্জেন্টিনা".
- Off-topic Bengali question -> `rag_disabled_retrieval_weak`, similarity 0.297, answers plain
  instead of pulling irrelevant context.

RAG never blocked or crashed the voice loop. The plain demo manifest records `rag=off`.

> Note: the threshold was raised from 0.40 to 0.45 during integration so that off-topic Bengali
> queries (measured ~0.41) gate off instead of pulling irrelevant FIFA context, while real FIFA
> queries (0.55-0.68) still pass.

## Measured results (integration run, A5000 / GPU0, `.venv-hervoice`)

All four stages ran end-to-end with no runtime crashes and produced a valid Bengali answer wav +
manifest.

**Self-check (`--self-check`).** Synthesized the fixed sentence "আমি বাংলায় কথা বলতে পারি।",
re-ASR'd it to "আমি বাংগায় কথো বলতে পারি।":

- CER = 0.08, WER = 0.40, `tts_status=ok`
- output: `runs/hervoice_demo/selfcheck.wav`

This is a re-ASR **intelligibility** proxy only. It is not a measure of naturalness.

**Plain voice demo.** `examples/in_bn_question.wav --no-rag` produced a valid `answer.wav`:
duration 2.39 s, RMS 0.1056, `tts_status=ok`.

**Resources.**

- Peak VRAM on GPU0 = 10930 MiB (~10.7 GB) via `nvidia-smi`, whole-process including
  CTranslate2 memory outside torch. Sequential residency keeps it well under 24 GB; brain and
  orpheus are never co-resident.
- Latency: warm wall-clock ~71 s; cold first run ~163 s. Dominated by per-stage model
  load/free, not inference, the accepted tradeoff for low VRAM.

**The Bengali TTS adapter.** The `orpheus_bn_ivr_lora` adapter beats base orpheus, CER 0.640 ->
0.498 on FLEURS-20. This is a re-ASR intelligibility proxy, not human naturalness.

## Limitations

Read these. They are the honest boundary of the product.

- **The TTS CER of ~0.498 is rough.** It means the synthesized Bengali is often re-transcribable,
  not that it is clean or natural. The adapter is a measured improvement over base, not a solved
  problem.
- **Naturalness is unmeasured.** Every CER/WER number here is a re-ASR intelligibility proxy.
  There is no human MOS or naturalness evaluation. Do not read "the model speaks well" into these
  numbers.
- **Re-ASR is a proxy.** Both the self-check (CER 0.08) and the adapter win (0.640 -> 0.498) are
  measured by transcribing the synthesized audio back with the same ASR family. This conflates ASR
  quality and TTS quality and says nothing about how the speech sounds.
- **The small brain is factually weak.** On the default audio the ASR transcript was partly
  garbled ("বামলাদেশে রাজানি নাম কি?") and the plain ungrounded answer was weak/gibberish
  ("মৌলভী আলীজাবাদীরা"). Even with a clean `--text` question, Qwen2.5-3B showed factual errors and
  a Latin code-switch leak ("আগ্রha"). This is a small-model limitation, recorded honestly. The
  RAG-grounded FIFA path, by contrast, returned the correct clean answer "আর্জেন্টিনা".
- **No full-duplex, no GPT-4o parity.** HerVoice is turn-based. There is no streaming, no
  barge-in, no live mic loop, and no claim of GPT-4o-class behavior.
- **English is a text-only fallback.** HerVoice is Bengali-first. There is no English speech-out
  path here.
- **It is not a single network.** Four separate models in sequence, unified at the product level
  only.

## Known latent issue (not fixed)

`fifa_rag.load_chunks` binds its default `path=KB_PATH` at definition time, so `core.retrieve`'s
module-level `KB_PATH` override only works because the bundled default happens to equal
`--kb fifa_kb.md`. A different `--kb` path would silently load the bundled KB. Not fixed here to
avoid changing shared reuse code; flagged for the next pass.

## Files

- `hervoice/core.py` — the engine (`HerVoice` class: `transcribe`, `retrieve`, `think`, `say`,
  `self_check`, sequential residency + guards)
- `hervoice/__main__.py` — CLI
- `runs/hervoice_demo/manifest.json` — run manifest
- `runs/hervoice_demo/answer.wav`, `selfcheck.wav`, `answer_rag_fifa.wav`,
  `answer_rag_offtopic.wav` — outputs

No Hugging Face token is stored in this repo.

## English: the single-network path

The Bengali path above is modular because the omni Talker cannot speak Bengali. English does not
have that wall, so for English HerVoice runs as a true **single end-to-end network**: MiniCPM-o 4.5
does ASR (its audio encoder), reasoning (the shared LLM "Thinker"), and speech-out (the "Talker" +
vocoder) in ONE model — audio in, audio out, no text-to-separate-TTS handoff. RAG grounding is the
only external add-on.

```
./hervoice_en.sh examples/in_fifa_question.wav runs/hervoice_en/answer.wav
```

Measured on the RTX A5000 (one real run):

```
[ASR]    How many times has Brazil won the men's World Cup and which years?
[ANSWER] Brazil has won the FIFA Men's World Cup a record five times: 1958, 1962, 1970, 1994, and 2002.
[wrote]  answer.wav   generation ~19 s   peak VRAM ~14.2 GB
```

One model produced the transcript and the spoken answer; the answer is grounded and correct.
This is the single-network voice-to-voice that the project originally aimed for — it works for
English precisely because the Talker is trained on English speech. Sample output:
`examples/hervoice_en_fifa_answer.wav`. Plain (non-RAG) single-network turns: `minicpm_voice.py`;
streaming with first-audio latency (~2.2 s, RTF ~1.12 at int4): `minicpm_stream.py`.

Honest scope: this is turn-based, uses a cloned reference voice (MiniCPM-o has no usable default
voice). The point proven here is architectural: ASR + LLM + TTS as one network, end to end,
for English.

## English: live, interruptible voice (barge-in)

The remaining gap for English was the live serving layer. It is now built as a **VAD-gated
streaming turn loop with barge-in** on the single resident MiniCPM-o 4.5 process. It is **not**
true simultaneous full-duplex (Omni-Flow) and **not** sub-second; it is turn-based streaming that
you can interrupt. State machine: `IDLE -> USER_SPEAKING -> GENERATING -> INTERRUPTING ->
RESETTING -> USER_SPEAKING`. Silero VAD (32 ms frames) detects turn boundaries; while the
assistant speaks, VAD keeps watching, and new user speech sets a `cancel_event` that preempts
generation between streamed chunks, then `reset_session(reset_token2wav_cache=False)` starts a
fresh turn (new session id, voice cache kept).

`hervoice/live/`: `engine.py` (LiveVoiceEngine), `turn_detector.py` (Silero VAD), `loop.py`
(state machine), `simulate_bargein.py` (headless proof), `local_mic_client.py` (your machine).

Because this box has no microphone or speaker, the loop is proven by a **file-driven barge-in
simulation** that runs the real engine: feed a wav in real-time chunks, inject a second wav as a
simulated interruption during generation, and assert that turn-1 generation is cancelled and a new
session begins.

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m hervoice.live.simulate_bargein --out-dir runs/live_bargein_smoke
```

Measured (RTX A5000, int4), from `results_live_bargein.json`:

- assertions passed; barge-in fired; turn-1 generation cancelled; 2 distinct session ids.
- cancel -> new-turn ~1.5 s; first audio ~2.4 s; peak VRAM ~14.5 GB (plain), ~15.7 GB (live-RAG).
- live-RAG mode also passes (ASR -> retrieve -> grounded spoken answer), adding ~2-3 s before first
  audio for the retrieval step.

Honest scope: cancellation is **cooperative** — `streaming_generate` yields ~1 s TTS chunks and the
cancel is checked at the chunk boundary, so the in-flight chunk finishes before bailing (not a hard
mid-chunk kill). MiniCPM-o's streaming audio encoder needs ~1 s-aligned prefill chunks, so the loop
aggregates mic frames to 1000 ms. Live mic capture and playback are confirmable only on your machine:
run `hervoice/live/local_mic_client.py` (needs `sounddevice`/PortAudio, which the headless box lacks);
it uses the identical engine and loop the simulation exercises.

### Stress test and known issues

An adversarial stress harness (`hervoice/live/stress_test.py`) runs the resident engine through
endurance, barge-in edge cases, malformed input, and cancellation races. It exists to FIND bugs, not
to prove green; full results are in `results_live_stress.json`. Sustained testing found and fixed two
HIGH-severity bugs the happy-path simulation missed:

- Short final prefill chunk crashed the streaming audio encoder. The leftover after 1 s-aligned
  chunking is `total_samples % 16000` (uniform 0-15999), so roughly 6 percent of normal turns ended
  with a chunk short enough to underflow the audio pooler (`RuntimeError`). Fixed by padding short
  chunks before prefill.
- An engine exception on the consumer thread killed the loop silently (no error, an undrained queue,
  effective deadlock). Fixed: `run()` now catches it, surfaces a `frame_error`, and recovers to a
  clean LISTENING state.

Clean under stress: 14-turn endurance held VRAM flat (about 16.1 GB, +31 MB over 14 turns, no leak),
threads flat at 4 (no zombies after rapid barge-ins), no cross-turn contamination (same prompt gives
the same answer; alternating prompts give correct distinct answers), malformed inputs (silence, 0.2 s,
noise, tone) produced zero fabricated turns, and a mid-chunk cancel left the next turn's audio finite
and non-silent.

Known limitation (honest): a barge-in that begins within the first ~1 s guard window of a SHORT (1-2 s)
response can be missed at the default `barge_guard_chunks=1`. The guard exists to stop the VAD
self-triggering on the assistant's own audio when there is no echo cancellation. A deferred-fire now
catches guard-window barge-ins on longer responses, but the sub-guard case on a short response is a
tradeoff: with client-side echo cancellation (or headphones), set `barge_guard_chunks=0` for immediate
barge-in. Real-mic barge-in feel and VAD robustness to room noise are confirmable only on your machine.
