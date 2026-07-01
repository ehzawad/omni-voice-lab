# System knowledge: how this voice assistant actually works

A learning guide to the ideas behind this repository, written to be read alongside the code. Each
section connects a concept to the paper it comes from, the exact file in this repo where it lives,
and what we actually measured. It is organized around the questions that drove the project, so it
doubles as the "why" behind the design decisions.

Everything here is honest about limits. Where a number is quoted it was measured on the project's
hardware (1x RTX A5000 24 GB, sometimes a shared RTX A6000); where something is unproven it says so.

The mechanism details below were read from the primary papers, not paraphrased from abstracts: the
SNAC (2410.14411), LoRA (2106.09685), MiniCPM-o 4.5 (2604.27393), and IndicVoices-R (2409.05356) PDFs
were read directly (the specific sections are cited inline), and each is tied to the exact file it
governs. CosyVoice 2, Moshi, Whisper, RAG, and Qwen3-Embedding are summarized from their papers and
model cards. The point of the exercise: a paper fact next to the line of code it explains.

---

## 0. The one-paragraph mental model

A voice assistant has to do three things: **hear** (speech to meaning), **think** (produce an
answer), and **speak** (meaning to speech). The central design question of this whole project was:
*must those be three separate models, or can they be one neural network?* The honest answer this
repo arrived at: **for English they can be one network; for Bengali they cannot yet, so it is
modular.** Understanding why is most of the education here, and it comes down to one component --
the part that turns text/thought into audio, called the **talker** -- and what language it was
trained on.

---

## 1. Two architectures, and why this repo has both

There are two ways to build voice-to-voice:

- **Single network (omni / end-to-end).** One transformer backbone with an audio encoder on the
  input and a speech decoder on the output. Audio goes in, audio comes out, with no intermediate
  text hand-off. Examples: MiniCPM-o 4.5, Qwen-Omni, Moshi.
- **Modular (cascade).** Three specialized models chained: ASR -> LLM -> TTS. More moving parts,
  but each part can be swapped or fine-tuned independently.

This repo builds **both** and compares them in `docs/COMPARISON.md` (`pipeline_a_bench.py` is the
single network, `pipeline_b_bench.py` is modular). The measured tradeoff: the single network reaches
first audio much faster (it streams, no hand-off), while the modular path is easier to fix for a
weak spot -- which is exactly what Bengali needed.

The product, **HerVoice**, ended up split along this line on purpose: English runs single-network
(`hervoice_en.sh`, and the live loop `hervoice/live/`), Bengali runs modular (`python -m hervoice`).
That split is not a compromise; it is what the evidence dictated (section 5).

---

## 2. Can ASR + LLM + TTS really be one neural network?

Yes. This was the question you kept asking, and the answer is that modern "omni" speech models are
**one shared LLM backbone with extra encoders and decoders attached**:

```
speech in ─▶ [audio encoder] ─┐
                              ▼
text in ───▶ [embeddings] ─▶ [ shared transformer LLM = "Thinker" ] ─▶ text out
                              │
                              └▶ [speech decoder = "Talker"] ─▶ [vocoder/codec] ─▶ speech out
```

- The **audio encoder** (a Whisper-style encoder) converts speech into embeddings the LLM consumes
  directly -- no "transcribe to text first" bottleneck.
- The **Thinker** is a normal LLM (in MiniCPM-o 4.5 it is Qwen3-8B) reasoning over interleaved
  audio and text tokens.
- The **Talker** generates audio tokens from the LLM's hidden states; a codec/vocoder turns those
  into a waveform.

So "multiple encoders/decoders but one network" is exactly right. This is the **Thinker-Talker**
design used by Qwen-Omni and MiniCPM-o, and the closely related single-transformer design of Moshi.

Papers to read for this:
- MiniCPM-o 4.5, *Towards Real-Time Full-Duplex Omni-Modal Interaction* (arXiv 2604.27393; read
  Section 2 for the architecture, Section 3 for Omni-Flow). Concretely, its ~9B all-differentiable
  stack is: (1) **encoders** -- a SigLIP ViT for vision and a **Whisper-Medium (0.3B) encoder** for
  audio in chunk-based streaming (50 feature tokens/s, MLP-compressed 5x to **10 audio tokens/s**
  into the LLM, to save token budget); (2) the **Qwen3-8B backbone** ("Thinker") that generates text
  and hidden states -- notably it only decodes in the *text* domain at ~3-4 steps/s (human speech
  rate), delegating audio to keep language quality intact; (3) the **Talker**: a lightweight ~0.3B
  Llama *speech-token decoder* that takes the backbone's summed hidden states plus each text token and
  emits discrete "S3" speech tokens, then a **streaming flow-matching decoder** turns those into a
  waveform **conditioned on a reference-audio clip**. That last fact is why the code must call
  `init_token2wav_cache(ref_audio)` and why "there is no usable default voice" -- the vocoder is
  reference-conditioned by design (`hervoice/live/engine.py`, `minicpm_stream.py`). And it is why the
  Talker is language-bound: it was trained on English/Chinese S3 speech tokens (section 4).
- Qwen3-Omni (arXiv 2509.17765) -- the Thinker-Talker split stated plainly.
- Moshi (arXiv 2410.00037) -- a single transformer over audio codec tokens with an "inner
  monologue" (it predicts text time-aligned with the audio it generates), reaching ~200 ms latency.

In this repo, the single network is exercised by `minicpm_voice.py` (one turn), `fifa_voice.py`
(one turn + retrieval), and `minicpm_stream.py` (streaming). All three are the same MiniCPM-o model
doing hear+think+speak.

---

## 3. Component by component (concept -> paper -> file -> measurement)

### 3a. Hearing: the audio encoder / ASR

Speech is turned into something the model understands by an audio encoder. Two flavors appear here:
- Inside the omni model, an encoder feeds the LLM directly (MiniCPM-o uses a Whisper-medium encoder).
- In the modular path, a standalone ASR model produces text: **faster-whisper** (a fast reimpl of
  OpenAI Whisper). Whisper: *Robust Speech Recognition via Large-Scale Weak Supervision*
  (arXiv 2212.04356) -- trained on 680k hours of weakly-labeled web audio, which is why it is
  robust and multilingual (including Bengali) out of the box.

Where: `pipeline_b_bench.py` and `hervoice/live/engine.py` (ASR), and it is reused as the **evaluator**
in `ft/eval_bn_tts.py` (re-transcribe generated speech to score it -- section 7).

### 3b. Thinking: the LLM brain

A normal decoder-only LLM. In the single network it is MiniCPM-o's built-in Qwen3-8B; in the modular
Bengali path it is Qwen2.5-3B-Instruct (chosen small so the whole pipeline fits one GPU -- section 6).
The honest cost of "small": the 3B brain is weak on open-domain Bengali facts and can code-switch,
which is why grounding (section 8) matters.

Where: `hervoice/core.py` (`think()`), `pipeline_b_bench.py`.

### 3c. Speaking: the Talker, and why it is the whole story

This is the component that decides whether a language works. The talker turns text/thought into
**audio tokens**, which a codec decodes to a waveform. Two different talker designs appear here:

- **CosyVoice 2** (arXiv 2412.10117) is MiniCPM-o 4.5's speech component. It uses a finite-scalar-
  quantized (FSQ) speech tokenizer, a unified text-speech LM, and a **chunk-aware causal flow-matching**
  decoder so it can stream. Flow matching is a diffusion-like way to generate the waveform smoothly
  in chunks.
- **Orpheus** (used for Bengali) is different and simpler to fine-tune: it is a Llama-3.2-3B that
  emits **SNAC** audio tokens instead of text. See `asif00/orpheus-bangla-tts`.

The key fact: **a talker only speaks the languages its speech training data covered.** MiniCPM-o's
talker was trained on English and Chinese speech. That is the entire reason Bengali needed a
different path (section 5).

Where: `hervoice/live/engine.py` (MiniCPM-o talker, streaming), `ft/orpheus_baseline.py` +
`ft/bn_tts.py` (Orpheus talker).

### 3d. Audio codecs: how sound becomes tokens

An LLM can only emit discrete tokens, so audio must be quantized into a token stream and back. This
is a **neural audio codec**. Understanding it demystifies `ft/bn_tts.py` completely.

- **SNAC** -- *Multi-Scale Neural Audio Codec* (arXiv 2410.14411; read Section 3 + 4.1). It extends
  Residual Vector Quantization (RVQ, the cascade of codebooks in SoundStream/EnCodec/DAC) by letting
  each codebook operate at a **different temporal resolution**: at quantizer i the residual is
  average-pooled down by a factor W_i, looked up, then upsampled back. Coarse codebooks therefore
  cover more time per token (structure/prosody), fine ones less (detail). The **24 kHz speech codec**
  Orpheus uses has RVQ strides **[4, 2, 1]**, giving three token streams at **~12 / 23 / 47 Hz** --
  a **1 : 2 : 4** ratio. That ratio *is* the "7 tokens per audio frame" (1 coarse + 2 mid + 4 fine),
  and each codebook holds **4096 entries (12-bit)** at an effective **~0.98 kbps** (a purely
  convolutional ~19.8 M-param model). Every one of those exact numbers is in `ft/bn_tts.py`:
  `redistribute()` un-interleaves the 7 flat tokens back into the 1/2/4 layers, `audio_to_tokens()`
  does the inverse with the per-position `+ i*4096` offsets, and codes sit at `AUDIO_OFFSET = 128266`
  so they occupy a slice of the Llama vocabulary that does not collide with text. Read that file next
  to the paper's Section 4.1 and the strides/offsets line up one-to-one. (SNAC beats EnCodec/DAC on
  speech at a *lower* bitrate -- ViSQOL 4.14, MUSHRA 88.4 at 0.98 kbps -- which is why an LLM-TTS can
  afford to model its tokens autoregressively.)
- **Mimi** is Moshi's codec (same purpose, different design), and **FSQ** is CosyVoice 2's. Same
  idea: continuous audio <-> discrete tokens an LLM can model.

A practical bug this created (section 9): SNAC is 24 kHz, but a dataset may be 48 kHz stereo, and a
naive resample can hang -- exactly what happened and was fixed in `ft/bn_tts.py`.

### 3e. Full-duplex and streaming: listening while speaking

- **Turn-based**: you talk, it answers, repeat. Simplest.
- **Streaming**: it starts speaking before it has finished thinking (lower latency to first sound).
- **Full-duplex**: it listens and speaks at the same time and can be interrupted (barge-in). This
  is what MiniCPM-o's Omni-Flow and Moshi's dual-stream aim for.

How **Omni-Flow** actually does it (MiniCPM-o paper Section 3): time-division multiplexing. The
interaction is cut into fixed time windows; each window's env-visual, env-audio, and out-stream tokens
are concatenated into one sequence `g_k = [v_k; a_k; o_k]` fed to a normal causal LLM, so within each
window the model first ingests what it just perceived, then decides *whether* to speak (a binary
Listen/Speak control token -- their ablation shows deciding *whether* separately from *what* is more
stable) and *what* to say, keeping text and speech time-aligned (their "TAIL" interleaving stops the
text running ahead of playback). Two facts from that paper directly shaped this repo's code:
- Their ablation finds a **1.0 s window is the sweet spot** (0.2 s / 0.1 s degrade). That is exactly
  why the live loop aggregates mic audio to `CHUNK_MS = 1000` before prefill -- sub-second chunks also
  underflow the audio encoder (the crash section 9 fixed). The "~1 Hz speak/no-speak decision" is this
  1 s window plus the Listen/Speak token.
- Because the model itself decides whether to output each window, native Omni-Flow **reduces reliance
  on an external VAD**. This repo does *not* run native Omni-Flow; it drives MiniCPM-o as a turn loop,
  so it *adds back* Silero VAD to detect turn boundaries -- an honest, simpler substitute for the
  harder full-duplex controller.

This repo's live English loop (`hervoice/live/`) is honestly a **VAD-gated streaming turn loop with
barge-in**, not proven simultaneous full-duplex. The serving pieces:
- **Silero VAD** (voice activity detection, github.com/snakers4/silero-vad) decides when the user
  starts and stops speaking (<1 ms per chunk). `hervoice/live/turn_detector.py`.
- **Barge-in**: while the assistant speaks, VAD keeps listening; new user speech sets a cancel flag
  that stops generation between chunks, then the session resets for a fresh turn.
  `hervoice/live/loop.py` (the `IDLE -> USER_SPEAKING -> GENERATING -> INTERRUPTING -> RESETTING`
  state machine).

Measured: first audio ~2.2-2.4 s at int4; cancel-to-new-turn ~1.5 s. Not Moshi's sub-200 ms -- that
is a different model class, stated plainly in `docs/LIVE_ENGLISH_ASSISTANT_SCOPE.md`.

---

## 4. Why Bengali is the hard part (and it is not what you would guess)

Diagnosed empirically in `diag_bengali_tts.py`. The intuitive guess -- "the model can't read Bengali
text" -- is **wrong**: the tokenizer represents Bengali perfectly and the LLM understands it. The
wall is the **talker**: it was trained on English/Chinese speech, so Bengali is out-of-distribution
for the text->audio-token mapping, and it produces degenerate/babbling audio.

This is the single most important lesson in the repo: **understanding a language and speaking it are
separate capabilities, and the speaking one is gated by speech training data, not by the tokenizer or
the brain.** Every open omni model shares this limitation (Qwen3-Omni lists 10 speech-output
languages; Bengali is not among them).

The consequence: for Bengali you keep the omni brain for understanding/text but route speech-out to a
**dedicated Bengali TTS** -- the modular path. That is `python -m hervoice` (`hervoice/core.py`).

---

## 5. Fixing a language is a data problem, not an architecture problem

The project tried to improve Bengali speech-out two ways, same model and recipe, only the training
data changed -- and the result is the cleanest lesson in the repo:

- LoRA on **FLEURS** Bengali (read speech, multi-speaker): **no improvement** (CER 0.640 -> 0.645).
- LoRA on **IndicVoices-R** Bengali (a purpose-built Indic TTS corpus): **CER 0.640 -> 0.498,
  about 22% better** on the same held-out set; WER about 16% better.

IndicVoices-R: *Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS*
(arXiv 2409.05356; read Section 1 + 3). The details explain *why* it worked where FLEURS did not:
- It is **derived from IndicVoices, an ASR corpus**, then **restored to TTS quality** by denoising
  and speech-enhancement models -- trained on English but applied cross-lingually (the same
  cross-lingual-generalization trick that shows up again in the RAG embedder, section 7).
- **1,704 hours, 10,496 speakers, 22 languages, 93.25% extempore** (spontaneous, not read) speech.
  Both properties matter: huge speaker diversity and natural prosody are what a TTS needs, whereas
  FLEURS is read news speech from few speakers.
- Because it came from restored ASR audio, each clip carries **quality metadata (SNR, C50, and a
  per-clip CER)** -- which is exactly what `ft/prep_ivr_data.py` filters on (`cer <= 0.05`,
  `snr >= 20`, 2-12 s, speaker-capped) to keep only clean, well-aligned clips.
- The paper's own recipe is: **fine-tune an English-pretrained TTS (VoiceCraft) on IV-R** to unlock
  Indian languages. This repo did the same move with a different base (Orpheus), which is why the
  result transfers. The corpus was the lever, not the architecture -- exactly what an earlier code
  review predicted. Prep: `ft/prep_ivr_data.py`; write-up: `docs/FINETUNE_BENGALI_TTS.md`.

Honest ceiling: CER ~0.498 still means roughly half the characters are off, and naturalness was never
measured (no human listening test). "Better," not "solved."

---

## 6. Fine-tuning without a data center: LoRA

You do not retrain a 3B model to teach it something; you add a small adapter. **LoRA** -- *Low-Rank
Adaptation of Large Language Models* (arXiv 2106.09685). Core idea: a weight update during fine-tuning
has low "intrinsic rank," so instead of changing a big matrix W you learn two small matrices B and A
and use `W + (alpha/r) * B*A`. You train ~0.1-1% of the parameters.

From the paper (Section 4.1, read it -- it is only two pages): for a frozen weight `W0`, the update is
`W0 + BA` with `B` (d x r) and `A` (r x k) and rank `r << min(d,k)`; the forward pass is just
`h = W0 x + (alpha/r) B A x`. `A` is random-Gaussian and `B` is zero-initialized, so `BA = 0` at the
start (training begins as the exact base model). `alpha/r` is a fixed scaling knob the authors "set to
the first r we try and do not tune." The memory win is precise: because `W0` is frozen you keep **no
optimizer states** for it, cutting VRAM by up to ~2/3, and the saved adapter is tiny (they report a
~10,000x smaller checkpoint than the full model). The paper deliberately adapted **only attention
weights** (`Wq, Wk, Wv, Wo`) and left MLP/LayerNorm "to future work."

In this repo: `ft/train_orpheus_lora.py` uses `r=16`, `alpha=32` (so `alpha/r = 2` scaling), and
targets `q,k,v,o` **plus** the MLP `gate,up,down` -- the common modern choice that goes beyond the
paper's attention-only default. Result: **24.3 M trainable params (0.73% of the 3.3B model)**, peak
VRAM ~9.8 GB (the no-optimizer-states win in action), ~16 min, and a **97 MB adapter** versus the
6.6 GB base (the paper's tiny-checkpoint claim, concretely). QLoRA (arXiv 2305.14314) stacks this on a
4-bit base so the 8-9B omni models fine-tune inside 24 GB (`docs/FINETUNE_S2S_LORA.md`).

Related: **QLoRA** (arXiv 2305.14314) = LoRA on top of a 4-bit-quantized base model, which is how the
larger omni models are fine-tuned inside 24 GB (the S2S smokes in `docs/FINETUNE_S2S_LORA.md`).

A concrete lesson learned: learning rate matters more than you would think on a small dataset. LoRA at
2e-4 caused **catastrophic forgetting** (the model got worse, CER 0.54 -> 0.77); dropping to 5e-5
fixed it. That is in the fine-tune doc, reported straight.

---

## 7. Grounding: making it answer correctly (RAG)

A small brain hallucinates facts. **Retrieval-Augmented Generation** -- Lewis et al.,
*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (arXiv 2005.11401): retrieve
relevant text from a knowledge base and put it in the prompt so the model answers from facts, not
memory.

The retriever here embeds text with **Qwen3-Embedding-0.6B** and compares by cosine similarity.
Two details worth knowing:
- It uses **last-token pooling** with a causal mask -- because it is a decoder-only model, the final
  token's hidden state has "seen" the whole input, so it serves as the sentence embedding (unlike
  BERT's `[CLS]`).
- It is **multilingual/cross-lingual**, which is why a **Bengali** question can retrieve from an
  **English** knowledge base directly (`hervoice/core.py` `retrieve()`), no translation step. This
  was verified: a Bengali FIFA question grounds to the correct answer ("আর্জেন্টিনা জিতেছে").

Where: `fifa_rag.py` (the retriever, `fifa_kb.md` the KB), gated by a similarity threshold so weak
retrieval degrades to an honest "no reliable answer" instead of a confident wrong one.

---

## 8. How you measure speech quality here: CER vs WER

You cannot eyeball audio, so quality is scored by **re-ASR**: generate speech, transcribe it back with
Whisper, and compare to the intended text.

- **CER** (Character Error Rate) and **WER** (Word Error Rate) are both edit distance /
  reference-length: `(substitutions + deletions + insertions) / N`. CER counts characters, WER counts
  words. One wrong character makes a whole word wrong, so WER is always harsher.
- For Bengali specifically CER is the more trustworthy signal: words are long and morphologically
  dense, and inconsistent spacing inflates WER even when the sounds are right.
- Big caveat: re-ASR CER is an **intelligibility proxy**, not naturalness. It says "a recognizer can
  recover the words," not "it sounds human." Naturalness needs human MOS, which was not run.

Where: `ft/eval_bn_tts.py` (uses `jiwer` for CER/WER, with Bengali text normalization).

---

## 9. Engineering realities that shaped the design

The papers describe models; running them on one shared GPU taught the rest:

- **VRAM budget forces choices.** All five HerVoice models resident at once was a No-Go on 24 GB, so
  the live loop uses **sequential residency** (load one heavy model, use it, free it). This is why
  latency is the accepted cost, not OOM.
- **Quantization.** int4 (via bitsandbytes) roughly quarters weight memory and is how a 9B omni model
  runs in ~14.7 GB. See `minicpm_stream.py --quant int4`.
- **Environments matter.** A single wrong dependency version breaks everything: `Qwen3-Embedding` is
  the `qwen3` architecture and needs transformers >= 4.51, while Parler-TTS pins 4.46 -- this repo
  runs several isolated `uv` venvs on purpose (`requirements-*.txt`).
- **Sustained testing finds real bugs the happy path hides.** `hervoice/live/stress_test.py` found a
  crash on ~6% of normal turns (a short audio chunk underflowing the encoder) and a silent deadlock
  (an exception killing a worker thread with no error surfaced). Both fixed; see
  `results_live_stress.json`. Lesson: a passing demo is not a tested system.

---

## 10. Concept-to-file map

| Concept | Paper | Where in this repo |
| --- | --- | --- |
| Single-network omni (Thinker-Talker, Omni-Flow) | MiniCPM-o 4.5 (2604.27393), Qwen3-Omni (2509.17765) | `minicpm_voice.py`, `fifa_voice.py`, `hervoice_en.sh` |
| Full-duplex single transformer | Moshi (2410.00037) | (studied; not run) |
| Single vs modular tradeoff | -- | `pipeline_a_bench.py`, `pipeline_b_bench.py`, `docs/COMPARISON.md` |
| ASR / audio encoder | Whisper (2212.04356) | `hervoice/live/engine.py`, `pipeline_b_bench.py` |
| Streaming TTS / talker (flow matching) | CosyVoice 2 (2412.10117) | inside MiniCPM-o; `hervoice/live/engine.py` |
| Neural audio codec (tokens <-> audio) | SNAC (2410.14411) | `ft/bn_tts.py` |
| Bengali speech-out diagnosis | -- | `diag_bengali_tts.py`, `docs/FINETUNE_BENGALI_TTS.md` |
| Bengali TTS corpus | IndicVoices-R (2409.05356) | `ft/prep_ivr_data.py` |
| Parameter-efficient fine-tuning | LoRA (2106.09685), QLoRA (2305.14314) | `ft/train_orpheus_lora.py`, `docs/FINETUNE_S2S_LORA.md` |
| Retrieval grounding | RAG (2005.11401) | `fifa_rag.py`, `hervoice/core.py` |
| Turn detection / barge-in | Silero VAD | `hervoice/live/turn_detector.py`, `loop.py` |
| Speech-quality metrics | -- | `ft/eval_bn_tts.py` |

---

## 11. Reading list (start here, in order)

1. LoRA -- arXiv 2106.09685 (read directly, Section 4). The fine-tuning idea; short and foundational.
2. Whisper -- arXiv 2212.04356. How robust ASR is trained.
3. SNAC -- arXiv 2410.14411 (read directly, Section 3-4). Read next to `ft/bn_tts.py`; the strides
   [4,2,1] -> 12/23/47 Hz -> 7 tokens/frame and the 4096 codebooks map straight onto the code.
4. CosyVoice 2 -- arXiv 2412.10117. Streaming TTS with flow matching (MiniCPM-o's talker lineage).
5. MiniCPM-o 4.5 -- arXiv 2604.27393 (read directly, Section 2-3). Single-network omni + Omni-Flow;
   the 1.0 s window explains the live loop's `CHUNK_MS`.
6. Moshi -- arXiv 2410.00037. The purest full-duplex design; the latency frontier.
7. Qwen3-Omni -- arXiv 2509.17765. Thinker-Talker stated cleanly.
8. IndicVoices-R -- arXiv 2409.05356 (read directly, Section 1+3). Why the Bengali win was a data
   story; its SNR/C50/CER metadata is what `ft/prep_ivr_data.py` filters on.
9. RAG -- arXiv 2005.11401. Grounding answers in facts.
10. QLoRA -- arXiv 2305.14314. Fine-tuning big models in small VRAM.

The repo's own docs are the applied companion to these: `docs/COMPARISON.md` (single vs modular
numbers), `docs/FINETUNE_BENGALI_TTS.md` (the data lesson), `docs/HERVOICE_DEMO.md` (the assembled
product), `docs/LIVE_ENGLISH_ASSISTANT_SCOPE.md` (honest scope of "live"). Read a paper, then read
the file it maps to in section 10 -- that pairing is the fastest way to actually learn this.
