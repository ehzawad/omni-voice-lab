# ARCH 3 — Continuous full-duplex interaction with delegated reasoning

This is the third and hardest architecture in a three-part project that builds an
open, local analog of OpenAI's GPT-Live (the API-only continuous voice system).
ARCH 3 is the continuous full-duplex mode: a fast front model that listens and
speaks at the same time, decides when to speak, handles being interrupted, and
delegates hard reasoning to a stronger model without freezing the audio loop.

It mirrors GPT-Live-1's design — active listening, speaking while listening, and
handing difficult questions to a stronger reasoning model — but with open weights
on a single GPU. The front is Moshi; the delegated brain is gpt-oss-20B, an open
reasoning model. This is NOT GPT-Live and the delegate is NOT GPT-5.5; it is the
open local analog. English only, file-driven (no live microphone on this host).

## What "continuous full-duplex with delegated reasoning" means

- Full-duplex: the front jointly models the incoming user audio and its own
  outgoing audio at 12.5 Hz (Mimi codec, 80 ms frames, 24 kHz). It can receive
  and emit audio in the same frame. Turn-taking is decided inside the network,
  not by a voice-activity-detection gate.
- Delegated reasoning: when a turn needs facts or reasoning the front does not
  answer from itself; it calls a stronger model. Crucially the audio loop keeps
  running while that model thinks — the front backchannels, then speaks the
  delegated answer as it streams back.

## Design

Front (full-duplex): Moshi, `kyutai/moshiko-pytorch-bf16`, ~7B, driven frame by
frame through the public `moshi` package (`LMGen.step` over Mimi codes).

Delegated brain: gpt-oss-20B (`gpt-oss-20b-Q4_K_M.gguf`) served by llama-server
(OpenAI-compatible, harmony chat template) on `127.0.0.1:8093`. gpt-oss is a
harmony reasoning model: it streams its chain of thought on a `reasoning_content`
channel first, then the spoken answer on the `content` channel. That split is
used directly — the front backchannels while gpt-oss reasons and starts speaking
the moment the first `content` token arrives.

Async delegation: when the controller decides to delegate, the gpt-oss call runs
on a background thread (`BrainStream` in `duplex_session.py`). The Moshi frame
loop never blocks on it. Answer tokens are committed, whole word by whole word,
into a thread-safe queue as they stream; the loop drains that queue and splices
the words into the teacher-force schedule, so Moshi begins vocalizing the first
words while gpt-oss is still generating the rest.

Teacher-forced vocalization (stated narrowly): Moshi is a full-duplex front; the
delegated answer text is rendered back through Moshi by teacher-forcing its
inner-monologue text stream. `LMGen(on_text_hook=...)` is called every frame with
the freshly sampled text token before the depformer turns it into audio and
before it enters the KV cache; the hook overwrites that token in place
(`text_token[:] = forced_id`). The depformer then generates the acoustics for the
forced token. So the acoustics are Moshi's but the WORDS are gpt-oss's. This is
NOT Moshi autonomously deciding to say the answer. Loop closure is checked by
re-transcribing Moshi's output wav with faster-whisper.

Everything below happens inside ONE streaming session
(`mimi.streaming(1)` + `lm_gen.streaming(1)`), driven frame by frame. Every frame
is logged: input frame index, output frame index, wall-clock ms, phase,
`user_audio_rms_in`, `assistant_audio_rms_out`, and the forced text token / piece.
No claim here is asserted; each is read back out of those frame logs
(`runs/arch3/scenarioA_frames.json`, `scenarioB_frames.json`).

## The four demonstrated behaviors (measured, from `results_arch3.json`)

Inputs used (real English wavs in the repo):
- `examples/in_en_question.wav` — the spoken question ("what is the capital of
  France"), 2.48 s, drives delegation.
- `examples/in_fifa_question.wav` — reused as the scripted barge-in interruption
  (4.46 s of spoken energy).

### 1. Full-duplex overlap (listens while speaking)

Overlap = frames where the input stream carries speech energy AND Moshi is
simultaneously emitting audio, at thresholds `user_rms_in > 0.02` and
`assistant_rms_out > 0.01`. In the continuous session: 19 overlapping frames,
distributed across phases as LISTEN 6, WAIT 4, SPEAK 9. The SPEAK overlaps are
the strongest evidence — Moshi is vocalizing the delegated answer while the input
stream simultaneously carries user energy, with no VAD gate. Native 12.5 Hz
duplex; timing is Moshi's.

### 2. Async / streaming delegation (keep talking while the brain thinks)

Timeline (output frames; 80 ms each):

- delegation_start_frame: 35
- first_brain_token_frame: 67
- answer_complete_frame: 88
- frames_spoken_while_waiting: 32 (≈ 2560 ms of assistant audio kept alive while
  gpt-oss reasoned)
- gpt-oss reasoning deltas before the first spoken word: 52
- brain first-content latency: 1.59 s

So after delegating at frame 35, the Moshi loop stayed live for 32 frames
(~2.56 s of audio), backchanneling ("let me think") while gpt-oss streamed 52
reasoning deltas, then teacher-forced the streamed answer and vocalized it. The
brain answered "The capital of France is Paris."; Moshi's forced text stream
carried exactly that. Re-ASR of the full session wav
(`runs/arch3/scenarioA_session.wav`) recovers the answer words ("the capital of
France ... Paris") around Moshi's own greeting and a trailing acoustic artifact
(see limitations).

### 3. Barge-in / interrupt

The interruption wav is scheduled to overlap the SPEAK window; the detector is an
energy threshold (`rms > 0.02`, sustained 3 frames) armed only while Moshi is
actually vocalizing the answer (phase SPEAK). Frame log around the cutoff
(`runs/arch3/scenarioB_frames.json`):

- Frames 70–79: assistant speaking the answer "Paris is the capital"
  (`assistant_rms_out` 0.04–0.07), user quiet.
- Frame 79: user interruption energy arrives (`user_rms_in` 0.181) while the
  assistant is still emitting ("capital") — an overlap frame.
- Frame 81: sustained user energy crosses threshold → detected; phase flips to
  INTERRUPTED, teacher-forcing stops (force → None), the answer is aborted before
  "of France". Markers: interrupt_frame 79, interrupt_detect_frame 81,
  cutoff_frame 81.
- Frames 81–88: assistant yields (forced silence), user energy continues.

gpt-oss's full intended answer (captured by joining the brain thread after the
cutoff) was "Paris is the capital of France."; Moshi vocalized only "Paris is the
capital" before being cut off. This is a genuine mid-answer interruption. The
detector is a heuristic, not a trained interruption model.

### 4. Gap sweep (pacing is measured, not hand-waved)

The teacher-force pacing inserts `gap` PAD frames per forced word-piece. Sweeping
gap in {1,2,3,4}, 5 trials each (Moshi's audio depformer samples at temperature,
so trials are averaged), re-ASR each output and score intelligibility as
`word_recall − CER` (word_recall = fraction of answer words present, higher
better; CER proxy penalizes Moshi's spurious/echo characters, lower better),
against the fixed answer "The capital of France is Paris.":

| gap | mean word_recall | mean CER | score | mean duration (s) |
|----:|-----------------:|---------:|------:|------------------:|
| 1   | 0.367            | 0.704    | -0.337| 2.24              |
| 2   | 0.666            | 0.528    |  0.138| 2.80              |
| 3   | 0.600            | 0.672    | -0.072| 3.36              |
| 4   | 0.733            | 1.112    | -0.379| 3.92              |

Recommended gap = 2. At gap 1 word-pieces collide (recall drops); at gap 4
duration inflates and the CER blows past 1.0 as Moshi adds spurious/echo words;
gap 2 is the balanced sweet spot, with gap 3 the next usable setting. Per-gap
representative wavs are `runs/arch3/sweep_gap{1..4}.wav` (all trials kept as
`sweep_gap{g}_t{k}.wav`).

## Co-residence and cost

Moshi and gpt-oss-20B are co-resident on one RTX A6000 (48 GB, physical GPU1):
gpt-oss alone ≈ 11.2 GB; both resident ≈ 26.8 GB; peak across the whole run
28.9 GB (28.2 GB), well within 48 GB. gpt-oss first-content latency ≈ 1.6 s;
brain answer streams word by word after that.

## How to reproduce

```
# 1) start the delegated brain on GPU1 (A6000), free port 8093
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/llama.cpp/build/bin/llama-server \
  -m /mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/gptoss20b/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 99 --host 127.0.0.1 --port 8093 -c 4096 --no-warmup --jinja \
  > runs/arch3/oss_server.log 2>&1 &
# poll http://127.0.0.1:8093/health until {"status":"ok"}

# 2) run the continuous session + all four behaviors (Moshi in .venv-duplex)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv-duplex/bin/python -m hervoice.gptlive.run_arch3
```

Artifacts land in `runs/arch3/` (session and barge-in wavs, per-frame JSON logs,
gap-sweep wavs, `oss_server.log`, `run.log`, `manifest.json`) and
`hervoice/gptlive/results_arch3.json`. `runs/arch3/demo_arch3.wav` is the
async-delegated spoken answer.

Code:
- `hervoice/gptlive/delegate_oss.py` — gpt-oss client: blocking `ask()` and
  streaming `ask_stream()` (SSE, yields reasoning/content deltas).
- `hervoice/gptlive/duplex_session.py` — the continuous single-session engine
  (`BrainStream` async delegate + `DuplexSession`); reuses the teacher-forcing
  `on_text_hook` mechanism from `front.py`.
- `hervoice/gptlive/run_arch3.py` — drives and measures the four behaviors.
- `hervoice/gptlive/front.py` — the loaded Moshi/Mimi front (reused).

## Limitations (read this)

- Teacher-forcing, narrowly: Moshi is a full-duplex front; the delegated answer
  is rendered back through Moshi by teacher-forcing its inner-monologue text
  stream (the `on_text_hook` overwrites the sampled token in place before the
  depformer vocalizes it). The acoustics are Moshi's; the words are gpt-oss's.
  This is NOT Moshi autonomously deciding to say the answer.
- Pacing is heuristic. There is no learned alignment model; we emit one
  word-piece then `gap` PAD frames. Moshi's audio depformer samples at
  temperature, so there is run-to-run variance — occasional word echoes or a
  trailing acoustic artifact (visible in the scenario-A re-ASR tail). This is an
  architecture proof of continuous duplex + async delegation, not a polished TTS;
  naturalness would need a human MOS study, which is not claimed here.
- moshiko's own inner monologue is its OWN unreliable reply, not a verbatim user
  transcript, so the user turn is not read from it; the behaviors are driven by
  scripted input wavs, and the user transcript (where needed) comes from a
  separate faster-whisper pass.
- Barge-in here is an energy-threshold heuristic on the scripted input stream,
  not a trained interruption model.
- All latency, overlap, and barge-in numbers come from the logged frames, not
  assertion.
- This is an open local analog of GPT-Live's continuous mode. The delegate is
  gpt-oss-20B (open weights), NOT GPT-5.5; a single RTX A6000; English; file-
  driven (no live microphone).
