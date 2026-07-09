# Architecture 2 — Turn-Based Single Speech-to-Speech (MiniCPM-o 4.5)

This is the primary MiniCPM-o candidate for Architecture 2 of the three-part
"open local analog of OpenAI GPT-Live" project. It is the turn-based single
speech-to-speech architecture: ONE omni neural network takes audio in and produces
audio out, in strict discrete turns.

## What turn-based single-S2S is

A full user utterance goes in, a full spoken assistant reply comes out, and then the
turn ends. There is no overlap between listening and speaking. This mirrors GPT-Live
"Advanced Voice Mode": it feels smoother than a cascaded pipeline because a single
network carries the whole path, but the back-and-forth is still rigid — you speak,
it speaks, you speak.

It is deliberately NOT full-duplex. There is no barge-in, no streaming-input overlap,
no interrupt handling. That live, interruptible behavior is Architecture 3 and lives
in a separate loop; none of it is used here.

## Why one network (contrast with Architecture 1)

Architecture 1 is a cascade of three separate networks: an external ASR transcribes
speech to text, an LLM produces a text reply, and an external TTS renders that text
to audio. Three models, three failure surfaces, and text as a lossy bottleneck
between them.

Architecture 2 collapses all of that into a single MiniCPM-o 4.5 network:

- an **audio encoder** consumes the user's speech,
- a **shared Thinker LLM** understands it and forms the reply,
- a **Talker + Token2wav vocoder** speaks the reply directly.

The same weights do speech understanding AND speech generation. No external ASR and
no external TTS sit anywhere in the generation path. That single-net property is the
entire point of Architecture 2 versus Architecture 1's cascade.

The reply voice is cloned from a short reference clip (`examples/ref_female.wav`), and
the Token2wav voice cache is built once at boot, not per turn.

## The discrete-turn guardrail

This runner enforces strict discrete turns: it prefills the ENTIRE user utterance,
then generates the ENTIRE reply, then stops. It does not import or use
`hervoice/live/` (the full-duplex loop). It does not stream user input concurrently
with generation, and it never interrupts. Explicitly: this is NOT full-duplex —
full-duplex is Architecture 3.

## Measured numbers

Model `openbmb/MiniCPM-o-4_5`, int4 LLM (audio encoder / TTS / vision kept in full
precision), single RTX A5000 (GPU0). Three real English input wavs from the repo,
file-driven single turns. Numbers from `hervoice/arch2/results_minicpm.json`.

Per turn:

| Turn | Input | TTFA (s) | Total (s) | Reply audio (s) | RTF |
|------|-------|----------|-----------|-----------------|-----|
| 0 | in_en_question.wav | 3.638 | 5.171 | 2.48 | 2.085 |
| 1 | in_fifa_question.wav | 1.770 | 14.704 | 12.56 | 1.171 |
| 2 | hervoice_en_live_turn2.wav | 2.166 | 17.603 | 13.60 | 1.294 |

Medians across the three turns:

- median TTFA (time-to-first-audio): **2.166 s**
- median RTF (total / reply audio duration): **1.294**
- median turn total: **14.704 s**

TTFA is measured honestly as the wall-clock time from feeding the user audio to the
first NON-silent assistant audio frame emitted by the streaming generate. All audio
chunks are concatenated into the reply wav. Turn 0's higher TTFA and RTF are a warm-up
effect on the first turn; later turns settle near real time (RTF ~1.2–1.3).

VRAM: torch peak **14.71 GB**; nvidia-smi device peak **16098 MB** — comfortably within
the A5000's 24 GB.

Reply text (the model's own text stream, produced jointly with the audio):

- Turn 0: "The capital of France is Paris."
- Turn 1: "Brazil has won the men's World Cup a total of five times. They won in 1958, 1962, 1970, 1994, and 2002."
- Turn 2: "Brazil has won the men's World Cup a total of five times. They won in the following years—1958, 1962, 1970, 1994, and 2002."

### Intelligibility / readability check

The OUTPUT wav of each turn is re-transcribed by an INDEPENDENT network, Qwen3-ASR
(0.6B), running in a separate venv and process. This is only to confirm the spoken
reply is intelligible; it never touches the generation path. The re-ASR transcripts
matched the model's own reply text closely (Qwen3-ASR normalizes some digits to
words, e.g. "nineteen fifty eight" for "1958" — still the same content).

This is a READABILITY proxy, NOT a WER measurement. There is no ground-truth
transcript for a free-form spoken reply, so word error rate cannot be computed and is
not claimed.

## How to reproduce

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m hervoice.arch2.minicpm_turn \
  --inputs examples/in_en_question.wav examples/in_fifa_question.wav examples/hervoice_en_live_turn2.wav
```

Outputs: per-turn reply wavs and `demo_minicpm.wav` in `runs/arch2/minicpm/`,
metrics in `hervoice/arch2/results_minicpm.json`, and a full run manifest (inputs,
model id, quant, git commit, VRAM, per-turn metrics, caveats) in
`runs/arch2/minicpm/manifest.json`. The independent re-ASR runs automatically in
`.venv-qwen-asr` as a subprocess per successful turn.

## Limitations

- Naturalness and voice quality need a human MOS study; they are not measured here.
- The re-ASR is an independent readability check, not WER — there is no ground truth
  for free-form replies.
- The LLM runs at int4 quantization (audio/TTS/vision are full precision).
- Single RTX A5000, English only, file-driven single discrete turns, no live mic.
- Honest failure guards: if the model emits empty text or degenerate/silent audio
  (duration < 0.2 s or rms < 0.005) the turn is recorded as `s2s_failed` with a reason
  and NO wav is written. All three turns here passed.
- This is an open, local analog of GPT-Live Advanced Voice Mode. It is NOT GPT-Live.
