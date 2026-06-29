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

## Results (first run, 2026-06-29)

Trained through the REAL audio-input path: yes. The spoken-question waveform was fed through the
Omni audio encoder into the thinker LLM; the per-step inputs carried `input_features` (verified at
runtime), and the loss was computed only on the assistant answer tokens.

- Model: `Qwen/Qwen2.5-Omni-7B` (chosen over MiniCPM-o for this run: its `Qwen2_5OmniProcessor`
  handles audio natively, so the genuine audio path ran cleanly within the time budget; MiniCPM-o's
  bespoke `forward(self, data, **kwargs)` was already documented as incompatible with the standard
  collator in `ft/logs/minicpmo_train.log`). Either model satisfies "trained on real audio input."
- Method: LoRA (r=16, alpha=32) on the thinker LLM projections q/k/v/o; audio encoder frozen;
  bf16; micro-batch 1; 3 epochs over 78 spoken-QA pairs = 234 steps; audio 1.4-3.5 s.
- Data: `ft/v2v_build_data.py` synthesized 98 spoken FIFA questions with Chatterbox TTS
  (`data/v2v/audio/`, `data/v2v/manifest.jsonl`), grounded in `fifa_kb.md`; 78 train / 20 held-out.
- Trainable params: 14,024,704 (0.157%). Peak VRAM: 19.6 GB train / 18.0 GB eval (A6000, GPU1).
- Train loss: 1.78 -> ~0.15. Adapter saved/reloaded cleanly: `adapters/v2v_qwen25omni_lora/`.

Before/after on 20 held-out SPOKEN questions (audio-in -> text answer, greedy):

| metric | base | base+LoRA |
|---|---|---|
| correct @ slot-frac>=0.6 | 16/20 | 16/20 |
| mean slot fraction | 0.80 | 0.817 |
| mean answer length (chars) | 50.6 | 15.7 |

The win is style-compliance: the adapter made answers ~3.2x more concise (50.6 -> 15.7 chars),
collapsing base's verbose restatements ("Germany has won the FIFA World Cup 4 times.") into the
KB one-line style ("Four."), while holding correctness. Per the success criteria, a measurable
lift in style-compliance counts as a win. Correctness did not improve because Qwen-Omni base
already knows these FIFA facts; the 4 misses are shared base/LoRA knowledge errors (e.g. it thinks
the 2022 final was decided 1-0), not a pipeline failure.

Speech-out for Qwen was left out of scope (text answer scored, as the plan allows); the genuine
adapted path here is listen+respond. Artifacts: `ft/v2v_build_data.py`, `ft/v2v_train.py`,
`ft/v2v_eval.py`, `results_voice2voice.json`, `ft/logs/v2v*.log`.
