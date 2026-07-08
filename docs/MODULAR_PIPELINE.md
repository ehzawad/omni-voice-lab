# Modular English voice assistant

Spoken English in, spoken English out, via four decoupled stages:

```
  in.wav ──▶ ASR ──▶ text ──▶ Qwen3.5-4B brain (llama-server) ──▶ text ──▶ Qwen3-TTS ──▶ out.wav
             │                        │                                       │
   Qwen3-ASR-0.6B            Qwen3.5-4B-Q4_K_M.gguf                Qwen3-TTS-12Hz-1.7B-Base
   (.venv-qwen-asr)          (HTTP, port 8090)                    (.venv-qwen-audio)
```

Code: `hervoice/modular/` — `asr.py`, `brain.py`, `tts.py`, `pipeline.py`, `asr_bench.py`.

## Why three venvs (honest constraint)

The three model stages have **mutually incompatible dependencies**, so they cannot share one
environment:

| venv | stage | key pins |
|---|---|---|
| `.venv-qwen-asr` | Qwen3-ASR (0.6B + 1.7B) | transformers **5.13** (needs `AutoModelForMultimodalLM` / `qwen3_asr`), torch 2.11 cu128 |
| `.venv-funasr` | SenseVoice / paraformer-zh / Fun-ASR-Nano | funasr 1.3.14, torch 2.11 cu128 |
| `.venv-qwen-audio` | Qwen3-TTS | `qwen-tts` pkg, which **pins transformers 4.57.3**, torch 2.11 cu128 + torchaudio cu128 |

The original plan of one `.venv-qwen-audio` for *both* Qwen3-ASR and Qwen3-TTS is **not
feasible**: `qwen-tts` forces transformers 4.57.3, which predates `qwen3_asr`. So `pipeline.py`
runs ASR and TTS as **subprocesses in their own venv** and talks to the brain over HTTP. A happy
side effect: each model stage is a short-lived process, so its VRAM is released before the next
stage loads — peak GPU stays low.

> torchaudio gotcha: `qwen-tts` pulls a cu13 torchaudio that mismatches the cu128 torch. Fixed
> by `uv pip install --index-url .../cu128 torchaudio==2.11.0` into `.venv-qwen-audio`.

## GPU policy

Everything is pinned to **GPU0** (RTX A5000, 24 GB) with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`. GPU1 (the other user's A6000) is never
touched.

## 1. Start the brain (llama-server)

Port 8090 (8080 was already occupied on this box by another service).

```bash
BIN=/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/llama.cpp/build/bin/llama-server
GGUF=/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/q4b/Qwen3.5-4B-Q4_K_M.gguf
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$BIN" -m "$GGUF" -ngl 99 \
    --host 127.0.0.1 --port 8090 -c 4096 --no-warmup > runs/modular/llama_server.log 2>&1 &
curl -s http://127.0.0.1:8090/health         # {"status":"ok"}
```

**Reasoning-model note:** Qwen3.5 is a thinking model. Left in thinking mode it spends the whole
token budget on `reasoning_content` and returns empty `content`. `brain.py` sends
`chat_template_kwargs={"enable_thinking": false}` — verified to give clean final answers.

## 2. Run the full pipeline (exact command)

```bash
REF="Printing, in the only sense with which we are at present concerned, differs from most, if \
not from all, the arts and crafts represented in the exhibition."   # transcript of examples/ref_female.wav

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv-qwen-asr/bin/python -m hervoice.modular.pipeline \
    --wav examples/in_fifa_question.wav --asr qwen3-asr-0.6b \
    --out runs/modular/pipeline_demo.wav --ref-text "$REF"
```

`--asr` accepts any of the 5 keys (`qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `funasr-nano`,
`sensevoice`, `paraformer-zh`). Default `qwen3-asr-0.6b` (provisional smoke-test pick — see
`docs/ASR_COMPARISON.md`). `--ref-text` is optional; omit it to clone the reference voice from
its speaker embedding alone (`x_vector_only_mode`).

## 3. Proven end-to-end run (real numbers)

Input `examples/in_fifa_question.wav` → `runs/modular/pipeline_demo.wav` (manifest:
`runs/modular/manifest.json`):

| Stage | Output | Latency |
|---|---|---|
| ASR (qwen3-asr-0.6b) | `"How many times has Brazil won the men's World Cup and which years?"` (lang=English) | 15.15 s |
| Brain (Qwen3.5-4B) | `"Brazil has won the men's World Cup five times, in the years 1958, 1962, 1970, 1994, and 2002."` | 0.49 s |
| TTS (Qwen3-TTS-1.7B-Base) | 11.04 s wav @ 24 kHz | 42.79 s |
| **Total** | | **58.42 s** |

**Round-trip check:** feeding `pipeline_demo.wav` back through ASR returns *"Brazil has won the
men's World Cup five times in the years 1958,1962,1970,1994 and 2002."* This proves the generated
answer is ASR-readable on this one clip; it is evidence of intelligibility, not a human MOS or
general TTS-quality score.

### Latency is cold-start, and honest about it

Because each stage is a fresh subprocess, these times **include one-time model load / CUDA
warmup**, not steady-state:

- ASR 15.15 s ≈ ~14 s CUDA-graph warmup on the first `generate` + ~0.7 s real work (the bake-off
  measured 0.71 s once warm).
- TTS 42.79 s includes loading the 4.3 GB model from disk + generating 11 s of audio.
- Brain 0.49 s is already warm (server stays resident).

A persistent-worker deployment (keep each model loaded) should remove much of the model-load /
warmup overhead, but this run did **not** measure interactive persistent-worker latency. This build
optimizes for **VRAM safety and venv isolation**, not latency.

## VRAM

Stages run sequentially, so peak GPU0 = brain (resident) + the single active model:

| Component | GPU0 VRAM |
|---|---|
| Brain server (Qwen3.5-4B Q4_K_M, all layers offloaded) | ~3.3 GB resident |
| + Qwen3-ASR-0.6B (bench-measured load) | +1.47 GB |
| + Qwen3-TTS-1.7B-Base | loaded only during the TTS subprocess; exact delta not recorded in the manifest |

The architecture is designed to stay well under the 24 GB card by loading only one speech model
at a time alongside the resident brain. The ASR bake-off ran one model at a time (see
`results_asr_bench.json` for per-model load VRAM: 0.84-4.0 GB). The manifest does not record an
exact TTS VRAM delta or exact end-to-end peak.

## Honest limits

- **Latency** is cold-start dominated (above). Not tuned for real-time; this is a batch/offline
  quality demo, not the live-duplex loop (`hervoice/live/`).
- **Three venvs** are a hard consequence of upstream dependency pins, not a design choice.
- **TTS reference:** `Qwen3-TTS-12Hz-1.7B-Base` is a voice-*clone* model. We clone
  `examples/ref_female.wav`; quality depends on that clip and its transcript. Without `--ref-text`
  it falls back to embedding-only cloning (lower fidelity).
- **ASR eval set is tiny** (4 clean prompts). WER=0 for the 0.6B model means "no errors on these
  four", not "perfect". See the caveats in `docs/ASR_COMPARISON.md`.
- **Brain** is served separately; the pipeline assumes it is already up on port 8090 and exits
  with a clear message if `/health` is down.

## Failure states (no fake success)

The pipeline never fabricates a downstream turn. `pipeline.py` and `tts.py` now guard each stage and
write an explicit `status` to the manifest:

- empty ASR transcript -> `status: asr_failed` (no brain call), exit 2;
- empty brain answer -> `status: brain_failed` (no TTS), exit 2;
- degenerate TTS audio (empty / < 0.2 s / RMS < 0.005) -> `status: tts_failed`, **no wav written**,
  with a `tts_reason`, exit 2.

A successful run records `status: ok`, `tts_status: ok`, and the output `output_rms`/`output_audio_s`.
The verification that "re-transcribing the output recovers the answer" is an **ASR-intelligibility
proxy on one clip**, not human audio-quality proof.

## Reproducibility and what is not built yet

Honest scope so this is not mistaken for a finished product:

- **Machine-specific coupling.** The brain reuses an *external* llama.cpp build and GGUF at fixed
  paths under `/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/` (see `brain.py`). On another machine
  those paths, the port (8090), and the three venv locations must be adjusted; there is no config
  file yet. Treat the paths as this-host defaults.
- **Per-turn model reload.** ASR and TTS run as one-shot subprocesses that load their model every
  call (the 15 s / 41 s costs above). This is VRAM-safe but not interactive. A persistent
  worker/service per venv (localhost HTTP or JSON-RPC) is the real-latency path and is **not built**.
- **Not built:** streaming (partial ASR/LLM/TTS), a single entrypoint that starts/health-checks all
  services, config-driven model/path selection, and RAG grounding. `fifa_rag.py` already exists and
  the clean slot is between ASR and brain (an optional `--rag-kb` + `brain.ask_grounded`); not wired
  in here. These are the smallest high-value next steps.
