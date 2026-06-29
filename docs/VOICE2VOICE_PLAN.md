# End-to-end voice-to-voice fine-tune: plan and first run

The earlier S2S smokes proved the LoRA loop on the models' text core only. This is the plan for a
genuine audio-in -> audio-out fine-tune: the model listens to a spoken question and speaks an
answer, and the training signal flows through the audio input path (not text tokens).

## Honest framing of "end to end"

- Input: speech (a spoken question waveform).
- Output: speech (a spoken answer waveform), produced by the model's own talker.
- Trained path: audio encoder -> LLM comprehension/response (LoRA). The English talker already
  works, so it is used as-is for speech output; we do not retrain the talker in this first run.
  This is end-to-end audio->audio at inference, with the adapted part being listen+respond.
- A later run can also LoRA the talker, which is the only way to move a low-resource language
  (e.g. Bengali) speech-output and would need target speech in that language.

## Task and data

- Domain: a concise spoken FIFA expert (reuses `fifa_kb.md`), so correctness is checkable.
- Build paired data: take question texts grounded in the KB, synthesize the spoken question with
  an available TTS (Chatterbox in `.venv-modular`, or the model's own TTS), pair each with a short
  ground-truth answer text. ~80-150 pairs for a first run; held-out 20 for eval.
- Each training example is a chat: user content = [question_audio], assistant content = answer
  text. Trained with the model's native multimodal message format (the bespoke forward that broke
  LLaMA-Factory must be driven through the model's own training/prepare utilities, not a generic
  collator).

## Model and method

- Primary: MiniCPM-o 4.5 (full-duplex, strategic target), bf16, LoRA on the inner `model.llm`
  attention/MLP projections; audio encoder frozen. If its native training path is intractable in
  the time budget, fall back to Qwen2.5-Omni-7B (cleaner transformers integration) for the same
  audio-in -> text LoRA.
- VRAM: bf16 9B fits the A6000 (48 GB) comfortably; micro-batch 1, grad-accum, audio <= 8 s.
- GPU: A6000 (`cuda:1`), `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

## Evaluation (measurable, no human listening required)

- Feed 20 held-out spoken questions; model speaks answers (`generate_audio=True`).
- Re-transcribe the spoken answers with faster-whisper; score answer correctness against the KB
  ground truth (keyword / slot match) and report a before/after delta.
- Also record a behavioral probe: does the adapter make answers more concise / on-style than base?

## Success criteria

- Smoke: LoRA attaches over the real audio-input training path, steps run, adapter saves/reloads,
  and a held-out spoken question produces a spoken answer without crashing.
- Win: measurable lift in answer correctness or style-compliance over base on the 20 held-out
  spoken questions. If no lift, report it straight as a pipeline proof (as with the Bengali TTS).

## Status

First run started the night of 2026-06-29; results and the before/after delta will be appended
here and as `results_voice2voice_*.json`.
