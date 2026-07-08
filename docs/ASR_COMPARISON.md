# English ASR bake-off (5 models)

Goal: pick a provisional ASR front-end for the modular **English** voice assistant, using a small
local smoke test. This file does not claim general ASR superiority and should not be read as a
statistically defensible leaderboard.

- **Eval set:** the 4 English prompts in `bench_common.py` (`in_en_question.wav`,
  `in_fifa_question.wav`, `bench/p3_math.wav`, `bench/p4_advice.wav`), each with a reference
  transcript. Clean, single-speaker, short (2.5-4.7s) studio-style speech.
- **Metrics:** WER + CER via `jiwer`, normalized (lowercase, strip punctuation, collapse
  whitespace). The table reports the current harness' **macro average over 4 clips**, not
  corpus-level WER. Raw latency = wall-clock of the transcribe call; RTF = latency / audio-seconds.
  Load VRAM = GPU0 `nvidia-smi` delta across model load, not peak inference memory. GPU0
  (RTX A5000) only.
- **Harness:** `hervoice/modular/asr_bench.py`. Raw numbers: `results_asr_bench.json`.
- **Honesty caveat:** 4 clean prompts is a *tiny* set. A 0.000 WER here means "no errors on
  these four", **not** "perfect ASR". Treat this as a smoke-ranking, not a leaderboard.

## Results (avg over 4 prompts)

| Rank | Model | HF id | avg WER | avg CER | raw avg latency | raw avg RTF | load VRAM | Notable features |
|---|---|---|---|---|---|---|---|---|
| 1 | **qwen3-asr-0.6b** | Qwen/Qwen3-ASR-0.6B-hf | **0.000** | 0.000 | 0.71s | 0.20 | 1.47 GB | language id; transformers-native; clean result on these 4 prompts with low VRAM and sub-second raw average latency |
| 2 | funasr-nano | FunAudioLLM/Fun-ASR-Nano-2512 | 0.019 | 0.004 | 0.68s | 0.20 | 2.15 GB | newer LLM-ASR decoder; strong on these 4 prompts |
| 2 | qwen3-asr-1.7b | Qwen/Qwen3-ASR-1.7B-hf | 0.019 | 0.008 | 5.74s* | 2.22* | 4.00 GB | language id; larger/heavier Qwen3-ASR variant |
| 4 | sensevoice | FunAudioLLM/SenseVoiceSmall | 0.160† | 0.104† | 0.20s | 0.07 | 1.15 GB | **emits language + emotion + audio-event tags**; fastest |
| 5 | paraformer-zh | funasr/paraformer-zh | 0.174 | 0.073 | 0.24s | 0.09 | 0.84 GB | Chinese-first; **weak English** (expected) |

\* qwen3-asr-1.7b's *first* generate call took 20.6s (CUDA graph / kernel warmup); the other 3
prompts ran at 0.5-0.95s, with post-warmup mean 0.80s and post-warmup mean RTF 0.20. The table
shows the raw 5.736s average from `results_asr_bench.json`; it is dominated by that one-time
cold-start outlier.

† sensevoice's WER is inflated by digit / inverse-text-normalization behavior, but not entirely:
on `p3_math` it returned "What is **12** multiplied by **8**?" (digits), which mismatches the
word-form reference "twelve ... eight" and scores WER 0.333. It also has genuine errors on
`p2_fifa` ("as" for "has", "when" for "won", "witch" for "which", "yearss" for "years"). The
headline 0.160 WER is therefore partly a scoring artifact and partly real English ASR error.

## Fair re-score (number-normalized)

To remove the digit/ITN unfairness, the stored hypotheses were re-scored with numbers spelled as
words in both reference and hypothesis (so "12" == "twelve"), no models reloaded
(`hervoice/modular/rescore_asr.py`, raw output `results_asr_bench_normalized.json`):

| Model | raw WER | number-normalized WER | numnorm CER |
|---|---|---|---|
| **qwen3-asr-0.6b** | 0.000 | **0.000** | 0.000 |
| funasr-nano | 0.019 | 0.019 | 0.004 |
| qwen3-asr-1.7b | 0.019 | 0.019 | 0.008 |
| sensevoice | 0.160 | **0.077** | 0.023 |
| paraformer-zh | 0.174 | 0.174 | 0.073 |

Reading it honestly: number normalization **halves** SenseVoice's macro WER (0.160 -> 0.077),
confirming much of its raw penalty was the "12 vs twelve" formatting difference, not misrecognition
-- but its genuine `p2_fifa` errors keep it in 4th. paraformer-zh is **unchanged** (0.174), because
its errors are real English mistakes ("friends" for "France", "brays il" for "Brazil"), not
formatting. The ordering is stable on this tiny set and **qwen3-asr-0.6b remains the provisional
default pick** under the fairer metric. Both caveats still hold: this is a 4-clip smoke ranking,
not a leaderboard.

## Per-prompt highlights

- **qwen3-asr-0.6b** — clean on all four, including the FIFA sentence with the apostrophe and
  "World Cup": *"How many times has Brazil won the men's World Cup and which years?"*
- **funasr-nano** — one error: *"Brazel"* for "Brazil" (p2). Otherwise no errors on this
  4-prompt set.
- **qwen3-asr-1.7b** — one error: dropped "and" → *"…World Cup in which years?"* (p2).
- **sensevoice** — tags returned per clip, e.g. `en, EMO_UNKNOWN, Speech, withitn`
  (language=en, emotion slot, audio-event=Speech, inverse-text-normalization on).
- **paraformer-zh** — English weakness is blatant: *"the capital of **friends**"* (France),
  *"**brays il**"* (Brazil), lowercase-only, no punctuation. This is the legitimate "con" of a
  Chinese-first model asked to do English.

## What would make this defensible

Minimum honest next eval:

- Use a larger English slice: at least 100-200 clips from a standard set such as LibriSpeech
  `test-clean` / `dev-clean` and/or FLEURS English, plus a few assistant-domain utterances.
- Report corpus WER/CER and macro WER/CER, with bootstrap confidence intervals over utterances.
- Apply a single published text normalization before scoring, including punctuation/case removal
  and digit/ITN normalization, so "12" and "twelve" do not decide rankings.
- Separate cold-start, first-call warmup, and steady-state latency. Report median, p90, and RTF
  after one untimed warmup call.
- Measure memory as baseline, post-load, and peak inference memory on an otherwise idle GPU; label
  whether the value is `nvidia-smi` process memory, PyTorch allocated/reserved memory, or both.

Until that exists, the honest claim is: **qwen3-asr-0.6b is the best smoke-test default in this
repo run, not proven best ASR.**

## Verdict for English

**Pick `qwen3-asr-0.6b` as the provisional default for this build.** On this set it is the only model with
zero WER, while also being one of the smallest (1.47 GB) and fastest (sub-second, RTF ~0.2).
That supports using it as the current default; it does not prove it will win on noisy speech,
accents, long-form audio, or broader domains. It is transformers-native (no extra runtime) and
returns a language id for free. It is the default ASR in `pipeline.py`.

- **Second choice:** `funasr-nano` (near-identical accuracy on these prompts, self-contained
  funasr runtime) or `qwen3-asr-1.7b` for a larger Qwen3-ASR model to test on harder audio —
  accepting ~2.7x the VRAM and a one-time warmup. The 4-prompt bake-off does not prove the
  1.7B model is more robust.
- **Use `sensevoice`** only when you specifically want its emotion / audio-event / language
  tags (e.g. to drive expressive TTS or logging). Its headline plain-English WER is inflated by
  digit normalization, but it also made real English mistakes on the FIFA prompt.
- **Avoid `paraformer-zh` for English** — it is a Chinese/Mandarin model and mis-hears common
  English words. Keep it for zh.
