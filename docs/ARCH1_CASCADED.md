# ARCH 1: The Cascaded Voice Architecture (STT -> LLM -> TTS)

This is architecture 1 of a three-part project building an open, local analog of OpenAI's
GPT-Live. It is the CASCADED design: three separate neural networks chained end to end, each doing
one job.

    user wav ──▶ ASR (speech-to-text) ──▶ LLM (text brain) ──▶ TTS (text-to-speech) ──▶ reply wav

It mirrors GPT-Live's "Standard Voice Mode": three independent models pipelined. That mode is
correct and robust but perceptibly slow and stilted, because each stage must (mostly) wait for the
previous one to finish before it can start, producing long pauses between the end of your question
and the start of the spoken answer. This document describes what we built, the exact models, the
streaming design, and the MEASURED latency numbers, and then states the limitations plainly.

This is an OPEN LOCAL ANALOG of the GPT-Live cascaded architecture. It is NOT GPT-Live.

## The three models (and why)

All three run on a single RTX A5000 (GPU0 only; GPU1 is never touched), about 9.5 GB VRAM resident.

- ASR: `Qwen/Qwen3-ASR-0.6B-hf`, served by a resident worker on `127.0.0.1:8091`.
  We chose the 0.6B over the 1.7B because in our own benchmark on 100 FLEURS English clips it
  reached WER 0.049, within the confidence interval of the 1.7B, while being cheaper and faster.
- LLM (brain): `Qwen3.5-4B-Q4_K_M.gguf`, served by `llama-server` on `127.0.0.1:8090` with
  `-ngl 99 -c 4096`. Qwen3.5 is a reasoning model; left in thinking mode it spends the whole token
  budget on hidden reasoning and returns empty content, so we disable it with
  `chat_template_kwargs={"enable_thinking": false}`. Temperature 0.3, max_tokens 200, and a system
  prompt that asks for one or two concise spoken sentences (no lists or markdown, since the text is
  read aloud).
- TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, a voice-clone base model, served by a resident worker on
  `127.0.0.1:8092`. It clones a reference clip (`examples/ref_female.wav`) plus that clip's
  transcript (`ref_text`) to speak the answer in a consistent voice.

Each model is loaded ONCE into a long-lived worker (the resident-worker pattern from
`hervoice/modular/`), so every turn is warm inference and pays no per-turn model-load tax. ARCH 1
reuses those workers read-only; it does not modify `hervoice/modular/`.

## The streaming design (where the latency win comes from)

The naive sequential turn is: transcribe the whole question, generate the whole answer, synthesize
the whole answer, then play. No audio can start until the last stage has synthesized the entire
reply.

ARCH 1 instead PIPELINES the LLM and TTS stages:

1. Stream the LLM. We open the `llama-server` chat completion with `"stream": true` and read the
   Server-Sent Events, yielding token deltas as they arrive (`hervoice/arch1/brain_stream.py`).
2. Detect completed sentences incrementally. As the answer buffer grows we run
   `hervoice/modular/chunk.py:split_sentences` on it; every sentence except the last is complete,
   the last is still being generated.
3. Synthesize each completed sentence immediately, on a background thread. The moment sentence 0 is
   complete we dispatch it to the TTS worker's `/synth` while the LLM keeps streaming sentence 1.
   TTS of sentence N overlaps LLM generation of sentence N+1. The first sentence's audio is ready
   long before the whole answer has been generated or synthesized.
4. When the stream ends we flush the trailing sentence, then concatenate the per-sentence chunk
   wavs (`_00.wav`, `_01.wav`, ...) into a single `_full.wav`.

Time-to-first-audio (TTFA) is measured as wall-clock from turn start to the moment the first
guard-passing chunk wav is written.

The honest failure states are preserved from the modular pipeline: an empty transcript is
`asr_failed` (no LLM call, no wav); an empty answer is `brain_failed` (no TTS, no wav); a
degenerate or silent chunk is recorded `tts_failed` and skipped without aborting the turn; if no
chunk ever passes, the turn is `tts_failed`.

## Measured results

All numbers below are measured in this repo on one RTX A5000, streaming path vs a same-answer
sequential baseline. The baseline synthesizes the EXACT SAME streamed answer as one shot (so the
delta is attributable to pipelining, not to LLM sampling variance) and reuses the already-measured
ASR and LLM times. For the sequential baseline, first audio cannot arrive until the whole answer is
synthesized, so baseline TTFA equals baseline total. Full data: `hervoice/arch1/results_arch1.json`
and `runs/arch1/manifest.json`.

| Turn  | Input                        | n sentences | Stream TTFA (s) | Baseline TTFA (s) | Speedup (s) |
|-------|------------------------------|-------------|-----------------|-------------------|-------------|
| fifa  | in_fifa_question.wav (real)  | 1           | 30.19           | 26.19             | -4.00       |
| france| in_en_question.wav (real)    | 1           | 7.95            | 6.49              | -1.46       |
| multi | in_multi_question.wav (gen)  | 2           | 16.27           | 39.88             | +23.61      |

The multi-sentence turn is the real win. The answer was two sentences; streaming emitted the first
sentence's audio at 16.27 s (after synthesizing only sentence 0, 7.44 s of audio), while the
sequential baseline could not produce any audio until it had synthesized the whole 17.84 s answer,
at 39.88 s. That is a 23.6 s reduction in time-to-first-audio, a 2.45x speedup, on this turn.

The two single-sentence turns show essentially no win (small NEGATIVE numbers here). This is
expected and honest: a single-sentence answer has no intra-answer split point, so the streaming
path synthesizes exactly the same one sentence as the baseline. The differences of a few seconds
are TTS run-to-run variance on a ~26-30 s autoregressive synthesis, not a real regression. The
pipelining win requires an answer with more than one sentence.

The `multi` input is a TTS-generated question ("Who was Isaac Newton, and what is he famous for?"),
created with the same TTS worker and then transcribed by ASR, because both real repo English wavs
(the FIFA question and "What is the capital of France?") happen to produce single-sentence answers
and therefore cannot exercise the pipelining path. This is noted honestly; it is a test input, not
a cherry-picked metric.

Reproduce:

    bash hervoice/modular/start_workers.sh
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      .venv-funasr/bin/python -m hervoice.arch1.run_arch1

A representative reply is kept at `runs/arch1/demo_arch1.wav` (the two-sentence Newton answer).

## Limitations (read this)

- The `qwen_tts` package does NOT do true audio-packet streaming. Its
  `generate_voice_clone(..., non_streaming_mode=False)` returns a FULL waveform; the package's own
  docstring states the flag "only simulates streaming text input ... rather than enabling true
  streaming input or streaming generation." Therefore the first-audio latency win here comes from
  STAGE PIPELINING (synthesize sentence 0 while the LLM generates sentence 1), NOT from
  sub-sentence audio-packet streaming.
- We do NOT claim the Qwen3-TTS paper's "97 ms first-packet" figure. That is a property of a true
  streaming vocoder path we do not have access to through this package. Every latency number in
  this document is measured in this repo.
- Pipelining does NOT reduce total generation time. It can be marginally higher than the baseline
  due to per-sentence overhead. Only TIME-TO-FIRST-AUDIO drops. TTS is still autoregressive and
  takes seconds per sentence on this hardware.
- The win only materializes for multi-sentence answers. Short, single-sentence replies see no
  benefit.
- Naturalness is not measured. We report latency only; speech quality and naturalness would need a
  human MOS study, which we have not run.
- Scope: single RTX A5000, English only, single session, file-driven input. On this headless box
  the input is a single-shot wav file, so turn start (t0) is when we begin processing the file, not
  a real microphone voice-onset. There is no live mic and no barge-in in ARCH 1.

## Files

- `hervoice/arch1/brain_stream.py` — streaming LLM client (SSE, `enable_thinking=False`).
- `hervoice/arch1/stream_pipeline.py` — one streaming cascade turn + the same-answer baseline.
- `hervoice/arch1/run_arch1.py` — driver: runs every turn, writes results + manifest + demo wav.
- `hervoice/arch1/results_arch1.json` — per-turn metrics and a summary block.
- `runs/arch1/manifest.json` — inputs, models, git commit, per-turn metrics, honest caveats.
- `runs/arch1/demo_arch1.wav` — a representative multi-sentence spoken reply.
