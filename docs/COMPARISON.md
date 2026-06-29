# Pipeline A vs Pipeline B: measured comparison

This is a real run, not a projection. Both pipelines were executed on the same fixed prompt set
(4 spoken English questions in `examples/`), same reference voice
(`examples/ref_female.wav`), on the same GPU. Raw numbers are in `results_a.json` / `results_b.json`.

- Pipeline A: A-single-network (MiniCPM-o 4.5 int4, streaming)
- Pipeline B: B-modular (faster-whisper base.en [CPU] + Qwen/Qwen2.5-3B-Instruct int4 + Chatterbox)

## Headline (means over prompts 2-4; prompt 1 excluded as cold-start warmup)

| metric | A (single-network) | B (modular) |
| --- | --- | --- |
| First-audio latency (s) | 2.31 | 11.68 |
| Real-time factor (RTF) | 1.45 | 2.43 |
| ASR WER (all prompts) | 0.0 | 0.122 |
| Peak VRAM (GB) | 14.78 | 5.87 |

## Pipeline A (single-network, MiniCPM-o 4.5, streaming)

| prompt | ASR WER | first-audio (s) | RTF | VRAM (GB) | answer |
| --- | --- | --- | --- | --- | --- |
| p1_capital | 0.0 | 3.497 | 2.045 | 14.44 | The capital of France is Paris. |
| p2_fifa | 0.0 | 2.405 | 1.495 | 14.77 | Brazil has won the Men's World Cup a total of five times. Th |
| p3_math | 0.0 | 2.328 | 1.455 | 14.76 | Twelve multiplied by eight is 96. |
| p4_advice | 0.0 | 2.208 | 1.406 | 14.78 | Absolutely! Take a few slow, deep breaths before walking int |

## Pipeline B (modular, ASR + LLM + TTS, batch)

| prompt | ASR WER | first-audio (s) | RTF | VRAM (GB) | answer |
| --- | --- | --- | --- | --- | --- |
| p1_capital | 0.0 | 32.459 | 13.991 | 5.66 | The capital of France is Paris. |
| p2_fifa | 0.154 | 11.858 | 1.611 | 5.87 | Brazil has won the men's World Cup five times. Rich ears is  |
| p3_math | 0.333 | 9.466 | 3.114 | 5.87 | 12 multiplied by 8 is 96. |
| p4_advice | 0.0 | 13.705 | 2.557 | 5.87 | Take deep breaths and focus on your strengths—this can help  |

## Reading the results

- Latency: Pipeline A reaches first audio in about 2.3 s because it streams output as it
  generates. Pipeline B is much slower to first audio because this implementation is batch:
  it finishes ASR, then the full LLM answer, then synthesizes the entire utterance before any
  audio plays. A streaming modular pipeline (incremental TTS) would narrow this; it was not built.
- RTF above 1.0 on both means generation is slower than playback on this hardware, so long
  answers drift behind. Neither is yet a smooth real-time experience at this configuration.
- ASR: both transcribe these clean prompts essentially correctly.
- VRAM: Pipeline A is one ~14.8 GB model; Pipeline B here is smaller (~5.9 GB) only because its
  brain was downsized (see caveats).

## Caveats (important for fairness)

- Brain asymmetry: Pipeline A uses MiniCPM-o's Qwen3-8B brain; Pipeline B had to use
  Qwen2.5-3B-Instruct because the shared machine left only ~14.8 GB free and the 8B int4
  load-time spike OOMed. This favors A on answer quality and B on VRAM. Re-run B with Qwen3-8B
  on a free GPU for a clean brain-parity comparison (one-line change in pipeline_b_bench.py).
- B is batch, A is streaming. The first-audio gap is partly architecture and partly this
  implementation choice. A fair latency comparison needs a streaming TTS in B.
- Shared GPU: both ran on the A6000 with another user occupying most of both GPUs, so absolute
  timings include contention and are not best-case.
- TTS naturalness / MOS is NOT measured here; it requires human listening tests. Demo outputs
  are in examples/bench/ (out_a_*.wav, out_b_*.wav) for manual judgment.
- Prompt 1 is a cold-start outlier (first CUDA/compile pass) and is excluded from the means.

## Reproduce

```
# Pipeline B (modular)
.venv-modular/bin/python pipeline_b_bench.py
# Pipeline A (single-network)
.venv/bin/python pipeline_a_bench.py
# Merge
.venv-modular/bin/python make_comparison.py
```
