# Single-network S2S LoRA: plumbing smoke tests

Two proof-of-pipeline LoRA smoke tests on single-network (omni) speech models, covering both
interaction classes the project targets: full-duplex and turn-based. The goal here is narrow and
honest: prove that a LoRA training loop attaches, takes optimizer steps, saves, reloads, and runs
inference on these models within the VRAM budget. These are **plumbing proofs on the text core**,
not voice-to-voice training runs.

## Scope and honesty

- What ran: LoRA on the models' inner text LLM, trained on a handful of text samples (10-16 steps).
- What did NOT run: training the audio-input or speech-output (talker) paths. The smoke proves the
  trainer works; it does not adapt the voice path. A real audio-in -> audio-out fine-tune is a
  separate, larger job (see docs/VOICE2VOICE_PLAN.md).

## Results

| model | class | path | transformers | trainable | peak VRAM | adapter reload | inference | status |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `openbmb/MiniCPM-o-4_5` | full-duplex | direct PEFT | 4.51.0 | 3.83M | 18.3 GB | yes | yes | PASS |
| `Qwen/Qwen2.5-Omni-7B` | turn-based | direct PEFT | 5.12.1 | 3.83M | 18.0 GB | yes | yes | PASS |

Raw numbers and notes: `results_s2s_lora_smoke.json`.

## What we learned about the trainers

- **MiniCPM-o 4.5 does not fit LLaMA-Factory's SFT path.** Its bespoke `forward(self, data, **kwargs)`
  re-passes `input_ids` into the inner LLM, colliding with the standard collator/PEFT call
  (`multiple values for keyword argument 'input_ids'`). Additional blockers: 4-bit quantization
  breaks the audio tower's native `nn.MultiheadAttention`, and `enable_input_require_grads`
  (from gradient checkpointing) breaks an in-place `scatter_`. The working approach attaches LoRA
  directly to `model.llm` (a standard Qwen3 causal LM), bypassing the multimodal forward.
- **Qwen2.5-Omni needs transformers >= 4.52** for the `qwen2_5_omni` architecture; 4.51 cannot
  instantiate it. It was run in a separate venv (transformers 5.12.1). `disable_talker()` drops the
  speech head to save VRAM during text-core training; LoRA is attached to the thinker text decoder.

Both used direct HF + PEFT, not LLaMA-Factory.

## Reproduce

Both env recipes are in `requirements-s2s-lora.txt`. After building them:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-lora-lf/bin/python ft/s2s_smoke_minicpmo.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-qwenomni/bin/python ft/s2s_smoke_qwen25omni_v2.py
```
