# Architecture 2 — Qwen3-Omni-30B-A3B-Instruct candidate (smoke-test verdict)

This documents an evaluation of `Qwen/Qwen3-Omni-30B-A3B-Instruct` as an alternative
candidate for Architecture 2 of the "open local analog of OpenAI GPT-Live" project.
Architecture 2 is the turn-based single speech-to-speech design: ONE omni neural
network takes audio in and produces audio out, in strict discrete turns (no
full-duplex, no barge-in, no streaming-input overlap).

The proven primary for Architecture 2 is MiniCPM-o 4.5 (see `docs/ARCH2_MINICPM.md`).
The question here was narrow: is Qwen3-Omni-30B a viable, and ideally better,
turn-based S2S than MiniCPM-o on a single RTX A6000 (48 GB)?

Short answer: no, not on this hardware. Qwen3-Omni-30B is architecturally a strong
speech-to-speech model, but it cannot be loaded and run with its native audio-out
path within 48 GB in the current transformers / compressed-tensors / accelerate
stack. MiniCPM-o remains the Architecture 2 primary. This is a real, useful negative
result, and no audio or metrics were fabricated to make it look otherwise.

## Smoke-test gates

The smoke test asked three questions. Evidence is in
`runs/arch2/qwenomni/smoke_result.json`, `runs/arch2/qwenomni/smoke_run.log`, and
`runs/arch2/qwenomni/smoke_offload.log`.

### 1. Does it load on the A6000 within 48 GB? NO.

Qwen3-Omni-30B-A3B-Instruct is a 35.26B-parameter MoE. Measured on a meta device:
thinker 31.72B, talker 3.32B, code2wav vocoder 0.22B. In bf16 the raw weights are
about 70 GB, and the model card's own minimum for the Instruct thinker+talker is
78.85 GB (bf16, FlashAttention2, 15 s of context). That does not fit 48 GB — it is an
~80 GB (multi-GPU) model in bf16.

Every attempt to shrink it to 48 GB failed:

- **bitsandbytes int4 / int8** — OOM at ~75% of weight loading, filling all 47.4 GiB.
  Root cause: the MoE experts are a fused custom module
  (`Qwen3OmniMoeThinkerTextExperts`, one packed tensor per layer), not `nn.Linear`.
  bitsandbytes only quantizes `nn.Linear`, so the ~31B of experts stay bf16 (~64 GB)
  and OOM. Skipping the audio-generation modules to protect speech quality made it
  worse; skipping only the small vocoder still OOM'd.

- **Community int4 checkpoint** (`cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`,
  compressed-tensors "pack-quantized" int4, 27.5 GB on disk) — OOM at the last layer.
  transformers' compressed-tensors path **decompresses** the packed int4 fused experts
  back to bf16 at load time (`DecompressExperts`), so the runtime footprint returns to
  ~64 GB and OOM's. It is int4 on disk, not int4 at runtime for the fused experts.
  (transformers 5.12.1 did not even recognize the packed expert tensors and
  re-initialized empty full-size experts; upgrading to 5.13.0 made it recognize and
  decompress them, which then OOM'd — the fit problem is unchanged either way.)

- **Community int4 + CPU offload** (`device_map="auto"`, cap GPU at 40 GiB, spill to
  100 GiB CPU RAM) — this DID load the weights across GPU and CPU. But inference then
  crashed with `AttributeError: 'Linear' object has no attribute 'weight'` inside
  `compressed_tensors ... quantized_forward` under an accelerate offload hook:
  accelerate's parameter-offload hooks and compressed-tensors' quantized forward do
  not compose. So even the offload path cannot produce speech without patching library
  internals — and a CPU-offloaded 30B MoE would be far too slow for turn-based S2S
  anyway.

VRAM peak observed across all paths: 47.4 GiB (the 48 GB card was the binding
constraint in every case).

### 2. Can it do native audio-out (speech-to-speech), not just ASR+text? YES (capability), NO (on this hardware).

The Instruct checkpoint is genuinely speech-to-speech capable — it is NOT a text-out-
only model. Evidence from its config and structure: `enable_audio_output=true`; a
Thinker–Talker design with a `talker` (3.32B), a `code2wav` vocoder (0.22B), and a
`code_predictor`; documented output voices Ethan / Chelsie / Aiden; audio output at
24 kHz. The community int4 checkpoint preserves the talker and code2wav tensors.

So the audio-out capability is real at the architecture level. It simply could not be
exercised on a single 48 GB A6000, because the model would not load and run there
(gate 1).

### 3. Feed one real English wav and produce a valid spoken reply? NOT ACHIEVED.

No reply wav was produced on the A6000. Every load/run path either OOM'd or hit the
accelerate-vs-compressed-tensors incompatibility during inference. No audio was
fabricated, and there is deliberately no `demo_qwenomni.wav` — producing one would
have meant inventing output the hardware never generated.

## Recommendation

MiniCPM-o 4.5 remains the Architecture 2 primary.

- MiniCPM-o is proven end-to-end on this exact A6000: it loads (int4 LLM, audio
  encoder / TTS / vision kept in full precision), runs discrete-turn S2S, and produces
  measured turn latencies and real reply audio (`docs/ARCH2_MINICPM.md`).
- Qwen3-Omni-30B, despite strong published audio benchmarks, does not fit a single
  48 GB A6000 with its audio-out path. In bf16 it is an ~80 GB (multi-GPU) model, and
  its fused MoE experts have no working single-GPU true-int4 runtime in the current
  transformers / compressed-tensors / accelerate stack.

Qwen3-Omni-30B would become a candidate again on hardware with roughly 80 GB or more
(H100 / A100-80G, or two A6000s) running bf16, or if/when a true-int4 fused-MoE
runtime with audio-out lands (for example a vLLM audio-out path, or GPTQ/AWQ kernels
that execute the fused experts in int4 rather than decompressing them). None of that
applies to the single-A6000 target here.

Note on variants: the 30B comes in Instruct / Thinking / Captioner cuts, but only the
Instruct cut carries the talker (audio out) — so there is no smaller same-family cut
that keeps speech output. A genuinely A6000-sized omni S2S with a talker, if one
appears in this family, would be the thing to re-test; the 30B is simply too large
here.

## Limitations of this evaluation

- This is a load-and-run feasibility gate on ONE machine (single A6000, 48 GB),
  English only, file-driven, single-turn. It is not a full benchmark.
- Because the smoke gate failed, there are NO latency numbers (TTFA, RTF,
  turn-total), NO quality comparison, and NO reply audio for Qwen3-Omni. Those would
  only exist if the model had run.
- Naturalness of synthesized speech, had any been produced, would still need human MOS
  to judge; the project's re-ASR check (`runs/arch2/qwenomni/reasr_qwenomni.py`, via
  Qwen3-ASR in `.venv-qwen-asr`) measures readability, not word error rate.
- Quantization changes model behavior; none of the quant paths here even reached a
  working forward pass, so no quantized-quality claim is made.
- This whole effort targets an open, local, turn-based analog of GPT-Live "Advanced
  Voice Mode" — it is not GPT-Live itself and makes no equivalence claim.

## Reproduce

```bash
# Download the base model (~67 GB) and the community int4 checkpoint (~28 GB)
.venv-qwenomni/bin/hf download Qwen/Qwen3-Omni-30B-A3B-Instruct
HF_HUB_DISABLE_XET=1 .venv-qwenomni/bin/hf download cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit

# Smoke test on GPU1 (A6000) — expect OOM / library-incompatibility, per this doc
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  QWENOMNI_MODEL=cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit QWENOMNI_QUANT=awq \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-qwenomni/bin/python runs/arch2/qwenomni/smoke_qwenomni.py

# CPU-offload variant (loads, then inference crashes on the offload incompatibility)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  QWENOMNI_MODEL=cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit QWENOMNI_QUANT=awq \
  QWENOMNI_OFFLOAD=1 QWENOMNI_GPU_GIB=40 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-qwenomni/bin/python runs/arch2/qwenomni/smoke_qwenomni.py
```

Env note: `.venv-qwenomni` was moved from transformers 5.12.1 to 5.13.0 during this
task (to test packed-MoE compressed-tensors loading), and `compressed_tensors` 0.17.1
was installed. torch was re-pinned to 2.8.0+cu128 / torchvision 0.23.0+cu128 after an
install accidentally pulled torch 2.13.0.
