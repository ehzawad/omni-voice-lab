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

A persistent-worker deployment (keep each model loaded) removes the model-load / warmup overhead.
This is now **built and measured** — see [Persistent-worker mode (warm)](#persistent-worker-mode-warm)
below. The one-shot subprocess path (`pipeline.py`) remains available and optimizes for **VRAM
safety and venv isolation**; the warm path (`serve.py`) optimizes for **latency**.

## Persistent-worker mode (warm)

Instead of reloading each model per turn, keep ASR and TTS **resident** in long-lived worker
processes that expose a tiny localhost HTTP API (Python stdlib `http.server`, no web framework).
The brain (llama-server) was already persistent. All three models co-reside on GPU0.

```
  wav ─▶ ASR worker :8091 ─▶ brain llama-server :8090 ─▶ TTS worker :8092 ─▶ wav
        (.venv-qwen-asr,      (Qwen3.5-4B, resident)     (.venv-qwen-audio,
         model resident)                                  model resident)
```

Code: `asr_worker.py`, `tts_worker.py`, `serve.py` (warm turn), `start_workers.sh` (launch + health).

### Start all three workers (GPU0 only)

```bash
bash hervoice/modular/start_workers.sh      # loads brain + ASR + TTS, waits for /health, prints PIDs
# stop: kill $(cat runs/modular/worker_pids.txt)
```

Model load + one warmup pass happens once at startup (measured: ASR load 12.6 s + warmup 14.1 s;
TTS load 30.8 s + warmup 6.2 s; brain server ~seconds). After that every turn is warm inference.

### Run one warm turn

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv-funasr/bin/python -m hervoice.modular.serve \
    --wav examples/in_fifa_question.wav --out runs/modular/warm_demo.wav --ref-text "$REF"
```

### Warm vs cold — real measured numbers

Input `examples/in_fifa_question.wav`, `qwen3-asr-0.6b`, 5 turns, **turn 1 discarded** (4 counted).
Full data in `results_modular_warm.json`. Every turn produced the correct answer
(*"Brazil has won the men's World Cup five times, in the years 1958, 1962, 1970, 1994, and 2002."*)
and a valid ~11.3 s wav (`tts_status: ok`, rms ≈ 0.077).

| Stage | Cold (one-shot subprocess) | Warm median | Warm p90 | What warm removes |
|---|---|---|---|---|
| ASR (qwen3-asr-0.6b) | 15.15 s | **0.74 s** | 0.84 s | ~14 s CUDA-graph warmup + model load |
| Brain (Qwen3.5-4B) | 0.49 s | **0.48 s** | 0.48 s | already resident both ways |
| TTS (Qwen3-TTS-1.7B) | 42.79 s | **22.05 s** | 22.72 s | ~20 s model load (see honest note) |
| **Total** | **58.42 s** | **23.36 s** | 24.04 s | **~2.5× faster per turn** |

### Co-resident VRAM (all three models loaded, GPU0)

Measured peak GPU0 with brain + ASR + TTS all resident and serving: **10 803 MiB ≈ 10.55 GB**,
well under the 24 GB card. Per-process at load: brain ~3.23 GB, ASR-0.6B ~1.83 GB, TTS ~4.49 GB.
GPU1 (the other user's A6000) stayed at 6 MiB idle throughout — never touched.

### Honest note: this removes the LOAD tax, not the generation time

Warm TTS is still **22 s for a ~11 s answer** because Qwen3-TTS generation is autoregressive — that
is generation, not model load. Warm mode kills the per-turn *model-reload* tax (the ~14 s ASR warmup
and ~20 s TTS load), which is why ASR drops 20× and total drops ~2.5×. The **next** lever is TTS
*generation* time — sentence-chunked streaming (start speaking sentence 1 while sentence 2
synthesizes) and/or a faster backend (e.g. faster-qwen3-tts, reported ~6×). That is a separate
improvement and is **not** what this change does.

The warm path keeps the exact same honest failure states as `pipeline.py`, verified through the
workers: empty text → `tts_failed: empty input text` (no wav); missing/empty transcript →
`asr_failed`; empty answer → `brain_failed`; degenerate audio → `tts_failed`. Manifest schema is
unchanged except for an added `"mode": "warm"` field (`runs/modular/manifest_warm.json`).

## Streaming TTS (sentence-chunked, lower time-to-first-audio)

Warm mode removed the model-*load* tax, but warm TTS still spends ~22 s because it synthesizes the
**entire** answer autoregressively before returning any audio. The cheapest perceived-latency win is
to synthesize the answer **sentence by sentence** and emit the first sentence's audio as soon as it
is ready, while the rest keeps generating. This reuses the **existing** Qwen3-TTS model (no new
dependency) and the exact same per-chunk failure guards.

Code: `chunk.py` (`split_sentences`), `tts.py` (`synth_stream`), `tts_worker.py`
(`POST /synth_stream`), `serve.py` (`--stream`).

```bash
# workers up (start_workers.sh); one streaming turn:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv-funasr/bin/python -m hervoice.modular.serve \
    --wav examples/in_fifa_question.wav --out runs/modular/stream_demo.wav --ref-text "$REF" --stream
```

`/synth_stream` splits the text on `.?!`/newline boundaries (keeping list-y content like
"1958, 1962, 1970, 1994, and 2002." in one chunk, and not splitting decimals/abbreviations/initials),
writes each chunk `out_prefix_00.wav`, `_01.wav`, ... **as soon as it is ready**, plus a concatenated
`out_prefix_full.wav`. It returns per-chunk `{index,text,wav,duration_s,status,cumulative_latency_s}`,
the **TTFA** (cumulative latency when the first valid chunk was written) and the total. A chunk that
fails the degenerate/empty guard is recorded `tts_failed` and skipped — no fabrication, no abort.

### Measured — real numbers (warm workers, GPU0)

**Required input `examples/in_fifa_question.wav`, `qwen3-asr-0.6b`, 5 turns, turn 1 discarded (4
counted).** Full data in `results_modular_stream.json`.

| Metric | Streaming (this change) | Whole-answer warm (baseline) |
|---|---|---|
| TTFA (turn start → first-sentence wav) | **22.30 s** median / 22.59 s p90 | — |
| Full-answer total | 22.36 s median / 22.65 s p90 | 23.36 s median / 24.04 s p90 |
| Whole-answer warm TTS | — | ~22.05 s median |

**Honest result for the FIFA answer:** the brain's FIFA answer is a **single sentence**
(*"Brazil has won the men's World Cup five times, in the years 1958, 1962, 1970, 1994, and 2002."*),
so it splits into **exactly one chunk** — there are no intra-answer split points, and TTFA ≈ total
(**22.30 s ≈ 22.36 s**). Streaming gives **no** win for a one-sentence answer, and we report that
straight rather than dress it up. (The concise "one or two spoken sentences" system prompt often
yields single sentences; "What is the capital of France?" is likewise one sentence.)

**Where the win actually lands (multi-sentence answer).** Feeding a genuinely multi-sentence answer
through the **same resident** Qwen3-TTS `/synth_stream` (isolating the TTS-level behavior):

> "Brazil has won the men's World Cup five times. The years were 1958, 1962, 1970, 1994, and 2002.
> It is the most successful team in the tournament's history."

| Chunk | Text | Cumulative latency when its wav is ready |
|---|---|---|
| 0 | "Brazil has won the men's World Cup five times." | **7.22 s ← TTFA** |
| 1 | "The years were 1958, 1962, 1970, 1994, and 2002." | 23.01 s |
| 2 | "It is the most successful team in the tournament's history." | 31.45 s |

First audio plays at **7.22 s** instead of **31.5 s** — the listener starts hearing the answer
**~4.4× sooner**. Both the FIFA full wav and this multi-sentence full wav **round-trip** correctly
(re-ASR recovers *"Brazil has won the men's World Cup five times ... 1958, 1962, 1970, 1994, and
2002 ..."*), and every emitted chunk passed the duration/RMS guard.

### Honest note: this cuts *perceived* latency, not total generation time

Streaming does **not** make generation faster — it is still autoregressive. In the multi-sentence
case the **total** is actually *higher* (31.5 s for ~15 s of audio, across three separate synth
calls with per-call overhead) than a single whole-answer call would be; what drops dramatically is
**time-to-first-audio**. The win is that the user hears the first sentence at ~7 s instead of
waiting ~31 s for the whole thing. The **next** lever — a genuinely faster TTS *backend* (e.g.
`faster-qwen3-tts`, reported ~6×) that lowers total generation time — is a **separate** change and is
deliberately **not** done here.

VRAM/GPU unchanged by this feature: co-resident peak GPU0 **10 851 MiB ≈ 10.6 GB** (vs 10 803 MiB
baseline), well under the 24 GB card; **GPU1 stayed at 6 MiB — never touched**. The failure guards
are preserved per chunk: empty/whitespace text → `tts_failed`, no wav, zero sentences (verified).

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

**Verified (2026-07):** these branches were exercised with degenerate inputs, not just coded.
Full pipeline on 1.5 s of silence -> ASR returned `''` -> `status: asr_failed`, no brain call, no wav
written. `tts.synth` on `"   "` -> `tts_failed` (degenerate audio, rms 0.0001); this test also caught
that empty text `""` slipped through the output-audio floor (the model vocalizes noise), so an upfront
empty-text guard was added -- `""`/`"   "`/`"\n\t "` now all return `tts_failed: empty input text`
with no wav. `brain_failed` is coded (symmetric to `asr_failed`) but not independently triggered, since
forcing an empty answer from the brain is impractical.

## Reproducibility and what is not built yet

Honest scope so this is not mistaken for a finished product:

- **Machine-specific coupling.** The brain reuses an *external* llama.cpp build and GGUF at fixed
  paths under `/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/` (see `brain.py`). On another machine
  those paths, the port (8090), and the three venv locations must be adjusted; there is no config
  file yet. Treat the paths as this-host defaults.
- **Per-turn model reload — FIXED (warm path).** `pipeline.py` still runs ASR/TTS as one-shot
  subprocesses (VRAM-safe, cold-start), but the **persistent-worker path is now built and measured**:
  `asr_worker.py` + `tts_worker.py` (stdlib `http.server`, localhost) keep each model resident, and
  `serve.py` runs a warm inference-only turn. See
  [Persistent-worker mode (warm)](#persistent-worker-mode-warm) — warm total 23.4 s vs 58.4 s cold,
  ASR 0.74 s vs 15.15 s. `start_workers.sh` is the single entrypoint that starts + health-checks all
  three services.
- **Not built:** streaming (partial ASR/LLM/TTS — the next lever, since warm TTS is still
  autoregressive), config-driven model/path selection, and RAG grounding. `fifa_rag.py` already
  exists and the clean slot is between ASR and brain (an optional `--rag-kb` + `brain.ask_grounded`);
  not wired in here. These are the smallest high-value next steps.
