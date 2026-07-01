# omni-voice-lab

An open, locally fine-tunable, low-latency voice-assistant lab built on a single-network
omni model (MiniCPM-o 4.5), with a modular fallback for low-resource languages. The goal is a
voice-to-voice assistant that listens, reasons, and speaks back, runnable and trainable on a
single 24 GB GPU.

This repository is a working set of validated components and experiments, not a finished
product. Status is stated honestly per component below.

## HerVoice: the assembled assistant

The components below come together in **HerVoice**, a Bengali-first, local, turn-based
voice-to-voice assistant: ask a question in spoken Bengali and it transcribes, thinks, and speaks
a Bengali answer back, on a single RTX A5000, in one command.

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv-hervoice/bin/python -m hervoice
```

It is a **modular** pipeline (not a single end-to-end network, not full-duplex): faster-whisper
large-v3 (Bengali ASR) -> Qwen2.5-3B-Instruct (brain) -> orpheus-bangla + the `orpheus_bn_ivr_lora`
adapter + SNAC (Bengali speech-out, the measured win) -> optional gated cross-lingual RAG over a
knowledge base. It uses **sequential GPU residency** so it stays well under 24 GB.

For **English**, HerVoice runs as a true **single end-to-end network**: MiniCPM-o 4.5 does ASR,
reasoning, and speech-out in one model (audio in, audio out), grounded by RAG. This is the
single-network voice-to-voice the project originally aimed for; it works for English because the
omni Talker speaks English (the only reason Bengali must be modular). Run `./hervoice_en.sh`;
measured on the A5000 a spoken FIFA question grounds to the correct spoken answer (~14.2 GB, ~19 s).

English also has a **live, interruptible** path (`hervoice/live/`): a VAD-gated streaming turn loop
with barge-in on the resident MiniCPM-o process. While the assistant speaks, Silero VAD keeps
listening; new user speech preempts generation and starts a fresh turn. It is turn-based streaming
with interruption, not sub-second true full-duplex. Because this box has no mic, it is proven by a
file-driven barge-in simulation (`python -m hervoice.live.simulate_bargein`, passes: turn-1
generation cancelled, new session, ~1.5 s cancel-to-new-turn, ~14.5 GB; numbers in
`results_live_bargein.json`); talk to it with a real mic via `hervoice/live/local_mic_client.py`.
An adversarial stress harness (`hervoice/live/stress_test.py`) found and fixed two HIGH-severity bugs
(a ~6%-of-turns encoder crash on short chunks; a silent consumer-thread deadlock) that the happy path
missed; endurance is clean (flat VRAM, no leaks/zombies/contamination). One known limitation remains
(sub-1 s barge-in on short answers; tunable via `barge_guard_chunks`). See `docs/HERVOICE_DEMO.md`.

Measured from a real run: peak VRAM about 10.7 GB; self-check re-ASR CER about 0.08; the Bengali
TTS adapter cuts CER from 0.640 to 0.498 on FLEURS-20 (a re-ASR intelligibility proxy, not human
naturalness). Honest limits: the 3B brain is weak on open-domain Bengali facts and can code-switch,
so the strongest path is RAG-grounded (a Bengali FIFA question grounds to the correct answer); plain
mode runs but answer quality is brain-limited. Full setup, CLI, manifest, and limitations are in
`docs/HERVOICE_DEMO.md`.

```bash
uv venv --python 3.12 .venv-hervoice
uv pip install --python .venv-hervoice/bin/python torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-hervoice/bin/python transformers==4.51.0 peft accelerate snac \
    faster-whisper==1.2.1 sentencepiece protobuf soundfile librosa jiwer
```

New to the concepts? `system-knowledge.md` is a learning guide that maps the papers behind this
system (single-network omni, neural audio codecs, LoRA, RAG, VAD/barge-in) to the exact files here
and the measured results, organized as a reading path.

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
| Single-net vs modular benchmark | Done (measured) | Numbers in `docs/COMPARISON.md`; raw `results_a.json`/`results_b.json` |
| LoRA training pipelines | Proven | Full-duplex (MiniCPM-o) and turn-based (Qwen2.5-Omni) smokes pass; `docs/FINETUNE_S2S_LORA.md` |
| Bengali TTS fine-tune | Measured win | LoRA on clean IndicVoices-R corpus cuts CER 22% / WER 16% vs base on FLEURS-20 (FLEURS-data LoRA did not); `docs/FINETUNE_BENGALI_TTS.md` |
| HerVoice assistant (`python -m hervoice`) | Working | Bengali-first voice-to-voice from the proven parts; one command, ~10.7 GB; RAG-grounded answers correct; `docs/HERVOICE_DEMO.md` |

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

## Benchmark: single-network vs modular

A measured head-to-head on a fixed set of spoken English prompts, same reference voice, same GPU.
Full results, per-prompt tables, and caveats are in `docs/COMPARISON.md`.

Headline (means over the warm prompts):

| metric | Pipeline A (single-network) | Pipeline B (modular) |
| --- | --- | --- |
| First-audio latency | about 2.3 s | about 11.7 s |
| Real-time factor | about 1.45 | about 2.43 |
| ASR word error rate | 0.0 | 0.12 |
| Peak VRAM | about 14.8 GB | about 5.9 GB |

Pipeline A streams output, so it reaches first audio quickly. Pipeline B here is batch (full ASR,
then full answer, then full synthesis), which is the main reason its first-audio latency is higher;
a streaming TTS would narrow that. Important fairness caveat: Pipeline B's brain was downsized to
Qwen2.5-3B (from the intended Qwen3-8B) because the shared machine left only about 14.8 GB free, so
the VRAM and answer-quality columns are not brain-parity. See `docs/COMPARISON.md` for the details.

Reproduce:

```bash
.venv-modular/bin/python pipeline_b_bench.py   # modular env
.venv/bin/python pipeline_a_bench.py           # single-network env
.venv-modular/bin/python make_comparison.py    # writes docs/COMPARISON.md
```

Pipeline B uses a second environment (Chatterbox pins transformers 5.2.0, which conflicts with
MiniCPM-o's pinned 4.51.0):

```bash
uv venv --python 3.12 .venv-modular
uv pip install --python .venv-modular/bin/python -r requirements-modular.txt
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
