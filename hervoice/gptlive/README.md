# hervoice/gptlive — a local, open-weight mirror of OpenAI's GPT-Live

OpenAI's **GPT-Live** is an API-only, closed voice system: a native full-duplex
voice front that hands hard questions to a stronger reasoning model. This
directory replicates that **design** with **open weights**, entirely on **one
RTX A5000 (GPU0)**:

- **FULL-DUPLEX FRONT = Moshi** (`kyutai/moshiko-pytorch-bf16`). One ~7B
  autoregressive model that jointly models the user's audio and its own audio at
  12.5 Hz (Mimi codec, 80 ms frames) and emits a time-aligned inner-monologue
  text stream. Turn-taking is decided inside the network — **no VAD**. This is
  the always-on conversational layer. (Same working pattern as
  `hervoice/duplex/prove_duplex.py`.)
- **DELEGATED REASONER = Qwen3.5-4B GGUF** served by `llama-server`
  (OpenAI-compatible, `127.0.0.1:8090`) — the open-weight "GPT-5.5" analog.
  Qwen3.5 is a reasoning model, so we send `chat_template_kwargs={"enable_thinking": false}`
  (otherwise `content` comes back empty).
- **DELEGATION CONTROLLER** (`delegate.py`): decide whether a user turn needs
  real reasoning/facts (question mark / who·what·when·why·how·which·calculate…
  triggers); if yes, send the transcript to the brain and get a concise answer.

## The hard part — making Moshi SPEAK the brain's answer (achieved)

Moshi speaks from text it **samples itself**. To make it vocalise an arbitrary
string (the brain's delegated answer) we **teacher-force its inner monologue**,
using only the public `moshi` API — no monkey-patching of internals:

`LMGen(on_text_hook=…)` calls our hook every frame with the freshly sampled text
token (shape `[B]`) **before** the depformer turns it into audio tokens and
**before** it is written into the KV cache (`moshi/models/lm.py::_step`,
lines ~736–763). The token is a live tensor, so the hook overwrites it in place:

```python
def on_text_hook(text_token):
    if force_id is not None:
        text_token[:] = force_id   # teacher-force Moshi's inner monologue
```

The depformer then generates the acoustic tokens for **our** token, and the
forced token enters the autoregressive context. This is the same principle the
delayed-streams TTS models use. **We drive it and Moshi speaks the answer.**

**Objective proof (loop closure):** we re-transcribe Moshi's teacher-forced
output wav with faster-whisper. Example (`in_en_question.wav`):

| stage | value |
|---|---|
| user transcript (ASR) | `What is the capital of France?` |
| Moshi's own inner monologue | `Hi there, how can I help you?` |
| delegate? | **yes** (question mark) |
| brain answer | `The capital of France is Paris.` |
| ASR of Moshi's **forced** output | `the capital of France is Paris` |

On the FIFA example the 15-word brain answer came back through Moshi almost
verbatim, showing forcing holds over longer utterances too.

## Honest limitations (read this)

- **No learned alignment model for pacing.** We emit `1 word-piece + gap pad
  frames`; `gap=2` is the intelligibility sweet spot (a pacing sweep is in the
  git history / `runs/gptlive/sweep_gap2.wav`). Moshi's audio depformer samples
  at temp 0.8, so there is **run-to-run variance** — occasional word echoes or a
  trailing artifact. This is an architecture proof of teacher-forcing, not a
  polished TTS.
- **moshiko's inner monologue is Moshi's OWN reply, not a verbatim user ASR**,
  and it is factually unreliable (FIFA: it volunteered *"She won it twice in
  1980"*). That unreliability is exactly **why** you delegate to the brain. So
  the controller transcribes the **user turn** with a separate faster-whisper
  pass (`_asr_worker.py`, CPU, `.venv-hervoice`) rather than trusting Moshi's
  monologue as the question.
- **Front-end ASR quality cascades.** `base.en` garbled the noisier FIFA input
  (`…has brazen one the men's world cup…`), and the brain answered accordingly.

## Files

- `front.py` — `MoshiFront`: loads Moshi + Mimi once; `run_turn()` does a single
  streaming session (LISTEN over the user wav → optional TEACHER-FORCED SPEAK).
- `delegate.py` — `classify()` + `ask_brain()` (the controller and brain client).
- `pipeline.py` — wires it end to end and writes the proof artifacts.
- `_asr_worker.py` — faster-whisper subprocess (user transcript + output verify).

## Run (headless, GPU0 only)

```bash
# 1) start the brain on GPU0
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/llama.cpp/build/bin/llama-server \
  -m /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/q4b/Qwen3.5-4B-Q4_K_M.gguf \
  -ngl 99 --host 127.0.0.1 --port 8090 -c 4096 &

# 2) run the pipeline (Moshi in .venv-duplex)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv-duplex/bin/python -m hervoice.gptlive.pipeline \
  --infile examples/in_en_question.wav
```

Artifacts land in `runs/gptlive/` (`manifest.json`, `log`,
`gptlive_spoken_answer.wav`, `moshi_listen_reply.wav`) and `results_gptlive.json`.

## Evidence (this run)

- Moshi + brain **co-resident**, peak **~20.5 GB / 24 GB** — fits.
- Timings: brain answer ~0.2–0.3 s; user ASR ~4 s (CPU); Moshi teacher-forced
  speak ~4 s; Moshi load ~73 s (one-time).
- **GPU0 only; GPU1 (A6000) untouched.**
