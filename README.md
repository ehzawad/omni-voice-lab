# omni-voice-lab

An open, locally fine-tunable, low-latency voice-assistant lab built on a single-network
omni model (MiniCPM-o 4.5), with a modular fallback for low-resource languages. The goal is a
voice-to-voice assistant that listens, reasons, and speaks back, runnable and trainable on a
single 24 GB GPU.

This repository is a working set of validated components and experiments, not a finished
product. Status is stated honestly per component below.

## What this is

Two architectures are explored and compared:

- Pipeline A (single network): MiniCPM-o 4.5 does ASR, reasoning, and speech generation in one
  model with a built-in streaming/duplex path and voice cloning.
- Pipeline B (modular): a strong text brain plus a dedicated streaming TTS, used where the single
  network falls short (notably Bengali speech output).

The single-network path is the strategic target; the modular path is the reliability and
low-resource-language fallback.

## Status

| Component | State | Notes |
| --- | --- | --- |
| English voice-to-voice (MiniCPM-o) | Working | Understands speech, answers correctly, speaks back |
| Streaming engine (`minicpm_stream.py`) | Working | int4, ~14.7 GB; first audio ~2.2 s; RTF ~1.12 |
| Voice cloning from a reference clip | Working | A reference clip is required; there is no usable default voice |
| FIFA domain expert via RAG | Working | Spoken question -> ASR -> retrieval -> grounded spoken answer |
| Bengali (modular) | Working | MiniCPM-o brain (Bengali text) + dedicated Bengali TTS |
| Live full-duplex server (mic, barge-in) | Not built | The hard real-time serving layer is future work |
| Single-net vs modular head-to-head benchmark | In progress | See `docs/` for the model/voice/dataset surveys |

Honest caveat on real time: at int4 on a 24 GB card the real-time factor is about 1.12, meaning
generation is slightly slower than playback and drifts behind on long answers. A larger GPU,
flash-attention, or smaller chunking are the levers; none are validated yet.

## Hardware and environment

Developed on a single NVIDIA RTX A5000 (24 GB). Target budget is at most 40 GB VRAM. int4
inference fits comfortably (about 11 to 15 GB).

The environment uses uv with Python 3.12 (no conda):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

GPU selection note on multi-GPU machines: set `CUDA_DEVICE_ORDER=PCI_BUS_ID` so device indices
match `nvidia-smi`, then pick a GPU with `CUDA_VISIBLE_DEVICES`.

The MiniCPM-o weights are downloaded on first use from the Hugging Face Hub
(`openbmb/MiniCPM-o-4_5`) and require `trust_remote_code=True`, which runs the model's own code.
Pin a specific revision for reproducibility.

## Quickstart

Generate test inputs (optional; example inputs are in `examples/`):

```bash
.venv/bin/python make_test_questions.py
```

English voice-to-voice (audio in, text and speech out):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python minicpm_voice.py --audio examples/in_en_question.wav --out reply.wav
```

Speak a fixed sentence in a cloned voice:

```bash
.venv/bin/python minicpm_say.py --text "Hello, I am your assistant." --ref examples/ref_female.wav --out say.wav
```

Streaming engine with latency and RTF (int4):

```bash
.venv/bin/python minicpm_stream.py --quant int4 --audio examples/in_en_question.wav --ref examples/ref_female.wav --out stream.wav
```

FIFA expert (RAG, grounded spoken answer):

```bash
.venv/bin/python fifa_voice.py --audio examples/in_fifa_question.wav --ref examples/ref_female.wav --out fifa.wav
```

Bengali (modular: MiniCPM-o brain plus Bengali TTS):

```bash
.venv/bin/python bengali_voice.py --audio examples/in_bn_question.wav --out bn.wav
```

## Why Bengali speech output fails on the single network

Empirically diagnosed in `diag_bengali_tts.py`. The text path is not the problem: the model's
tokenizer represents Bengali script perfectly (no lost characters). The failure is in the talker,
the learned text-to-speech-token model, which was trained on English and Chinese speech. Bengali
is out of distribution for that mapping, so the talker emits degenerate, repeating audio tokens
(observed as long babble instead of a short answer). The fix used here is modular: keep the
MiniCPM-o brain for Bengali understanding and text, and route the text to a dedicated Bengali TTS.

## Repository layout

```
minicpm_voice.py     English voice-to-voice (audio in, text and speech out)
minicpm_say.py       Speak a fixed sentence in a cloned voice
minicpm_stream.py    Streaming engine; reports first-audio latency and RTF; int4 option
fifa_rag.py          Retrieval over a FIFA knowledge base using Qwen3-Embedding
fifa_voice.py        FIFA expert: spoken question -> ASR -> retrieval -> grounded spoken answer
fifa_kb.md           FIFA knowledge base (editable; swap for a live feed for volatile facts)
bengali_voice.py     Modular Bengali pipeline (MiniCPM-o brain + Bengali TTS)
diag_bengali_tts.py  Diagnostic for the Bengali speech-output limitation
make_test_questions.py  Generate clean English test inputs
docs/                Model, voice, and dataset surveys
examples/            Example input clips, a reference voice, and demo outputs
```

## Models and licenses

This repository's own code is under the MIT License (see LICENSE). The models it downloads have
their own licenses, which apply to your use of them:

| Model | Use here | License |
| --- | --- | --- |
| `openbmb/MiniCPM-o-4_5` | Single-network brain, ASR, English TTS | Apache-2.0 (per upstream) |
| `Qwen/Qwen3-Embedding-0.6B` | RAG retrieval | Apache-2.0 (per upstream) |
| `facebook/mms-tts-ben` | Bengali TTS (modular) | CC-BY-NC-4.0 (non-commercial) |
| `facebook/mms-tts-eng` | Test-input generation | CC-BY-NC-4.0 (non-commercial) |

Important: the Bengali and test-input TTS paths use MMS, which is non-commercial. Do not treat
the end-to-end system as commercially usable while those components are in the loop. Some
alternative voices (for example Orpheus) are access-gated and require approval before download.

## Notes

This is research and demo code on a shared machine. It is not a hardened service. The live
full-duplex serving layer, a fair single-net vs modular benchmark, and any fine-tuning are
future work tracked in `docs/`.
