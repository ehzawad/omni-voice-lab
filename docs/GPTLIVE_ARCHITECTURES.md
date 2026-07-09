# Three voice architectures, built as open local analogs of GPT-Live

OpenAI's GPT-Live post lays out three voice architectures. This repo implements all three
from open weights, English, each on its own git branch, on a single-workstation setup
(RTX A5000 24 GB = GPU0, RTX A6000 48 GB = GPU1). Every latency and behavior number below
is MEASURED in this repo on file-driven turns (no live mic on this box), not quoted from
papers. These are open local analogs of the GPT-Live architectures, not GPT-Live; where a
delegate model stands in for GPT-5.5 it is an open model (gpt-oss-20B), not GPT-5.5.

## The three architectures

| # | GPT-Live name | This build | Branch |
|---|---|---|---|
| 1 | Cascaded voice system (Standard Voice Mode) | Qwen3-ASR-0.6B -> Qwen3.5-4B GGUF -> Qwen3-TTS-12Hz-1.7B, three separate nets, LLM->TTS stage pipelining | `arch-1-cascaded` |
| 2 | Turn-based single speech-to-speech (Advanced Voice Mode) | MiniCPM-o 4.5, ONE omni net audio-in -> audio-out, strict discrete turns | `arch-2-turnbased-s2s` |
| 3 | Continuous interaction / full-duplex (GPT-Live-1) | Moshi full-duplex front + gpt-oss-20B streaming async delegate, teacher-forced vocalization | `arch-3-continuous-duplex` |

## What distinguishes each (the architectural point)

- Arch 1 is THREE separate models wired in series. Speech is understood by one net, reasoned
  by a second, spoken by a third. This is what makes it "slow and stilted with long pauses":
  each stage must finish before the next starts. Our one honest speedup is to stream the LLM
  and start speaking sentence N while the LLM writes sentence N+1.
- Arch 2 is ONE omni network: its audio encoder, shared reasoning LLM, and speech Talker/vocoder
  all live in a single model, so there is no cross-model handoff. It is smoother, but still
  strictly turn-based (full utterance in, full reply out, no overlap). We deliberately keep it
  in discrete turns; it is NOT full-duplex (that is arch 3).
- Arch 3 is genuinely full-duplex: the front model listens and speaks in the same streaming
  session, overlaps, backchannels, handles barge-in, and delegates hard reasoning to a stronger
  model while it keeps talking. This is the GPT-Live-1 pattern.

## Measured comparison

Latency is not perfectly apples-to-apples across rows and the table says why; read each number
against its own definition, not as a single leaderboard.

| Metric | Arch 1 cascaded | Arch 2 turn-based S2S | Arch 3 continuous duplex |
|---|---|---|---|
| Time-to-first-audio | 7.95 s (single-sentence turn); 16.27 s streaming vs 39.88 s sequential-baseline on a 2-sentence turn (2.45x) | 2.17 s median (first audio chunk of the reply) | first answer word spoken ~2.56 s into the wait while the brain streams; overlaps continue |
| First-audio definition | user-audio-start -> first fully-synthesized sentence wav (whole-sentence TTS) | user-audio-start -> first streamed audio chunk from the one net | delegate fired -> Moshi keeps stream alive backchanneling, then vocalizes answer as it streams |
| Real-time factor | not the design axis (batch TTS per sentence) | 1.29 median | n/a (continuous session, not turn-scored) |
| Nets in the path | 3 (ASR + LLM + TTS) | 1 (omni) | 2 (Moshi front + delegate brain) |
| Full-duplex / overlap | none (serial stages) | none (strict turns, by design) | yes: 19 frames with simultaneous user-in and assistant-out energy, 9 during SPEAK |
| Barge-in | n/a | n/a | yes: user energy during SPEAK detected (frame 81), forcing aborted, assistant yields |
| GPU | A5000 (GPU0) | A5000 (GPU0) | A6000 (GPU1) |
| Peak VRAM | ~11 GB (co-resident workers) | 14.7 GB (int4) | 28.9 GB (Moshi + gpt-oss-20B co-resident) |

The ordering the GPT-Live figure predicts holds in our measurements: the cascade pays the most
for its first word (each stage serial; whole-sentence TTS), the single omni net is markedly faster
to first audio (2.17 s) and smoother, and the continuous build is the only one that overlaps
listening and speaking and handles interruption.

## Model-selection findings

- Arch 1 ASR: Qwen3-ASR-0.6B chosen over 1.7B (WER 0.049 at 100 FLEURS clips, within CI of the
  larger model, cheaper). Brain: Qwen3.5-4B GGUF with thinking disabled. TTS: Qwen3-TTS-12Hz-1.7B.
  The qwen_tts package does NOT stream audio packets (its non_streaming_mode flag only simulates
  streaming text input), so we do not claim the paper's 97 ms first-packet; the first-audio win is
  stage pipelining, measured locally.
- Arch 2: MiniCPM-o 4.5 is the primary. Qwen3-Omni-30B-A3B was evaluated on the A6000 and rejected
  with evidence: it is a genuine speech-to-speech model (enable_audio_output, Thinker-Talker +
  code2wav) but does not fit a single 48 GB card (~80 GB bf16; its fused MoE experts have no working
  single-GPU true-int4 runtime -- int4 checkpoints decompress to bf16 at load; every path OOM'd at
  47.4 GiB). It would be a candidate again only on ~80 GB+ hardware.
- Arch 3: Moshi front + gpt-oss-20B delegate (the open mirror of "delegate to GPT-5.5"). gpt-oss
  needs `--jinja` and low reasoning effort or its content channel comes back empty. gap=2 is the
  best teacher-forcing pacing (5-trial sweep, scored word_recall - CER).

## Honesty and scope (applies to all three)

- All numbers are measured on this hardware on file-driven turns; there is no live microphone on
  this box. Single A5000 / single A6000, single session, English only.
- Naturalness and voice quality are NOT measured here; they need a human MOS study.
- Arch 3's teacher-forcing is stated narrowly: the delegated answer's WORDS are rendered through
  Moshi's acoustics by overwriting its inner-monologue text token per frame; Moshi is not
  autonomously deciding to say the answer. Loop-closure is verified by re-ASR, which also exposes
  the heuristic-pacing artifact (forced tokens read "...France is Paris" while the acoustics
  re-ASR as "...Alice is the Paris") -- disclosed, not hidden.
- Barge-in is an energy-threshold heuristic on a scripted input stream, not a trained interruption
  model. moshiko's own inner monologue is its own unreliable reply, so the user turn is transcribed
  separately.
- Re-ASR of S2S output is a readability/loop-closure check, not a WER measurement (no ground truth
  for free-form spoken replies).

## Reproduce

Each branch carries its own runner, results JSON, manifest, demo wav, and doc:

- `arch-1-cascaded`: `docs/ARCH1_CASCADED.md`, `hervoice/arch1/`
- `arch-2-turnbased-s2s`: `docs/ARCH2_MINICPM.md`, `docs/ARCH2_QWENOMNI.md`, `hervoice/arch2/`
- `arch-3-continuous-duplex`: `docs/ARCH3_CONTINUOUS.md`, `hervoice/gptlive/`
