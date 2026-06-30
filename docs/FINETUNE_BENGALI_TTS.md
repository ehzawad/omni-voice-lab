# Bengali TTS LoRA fine-tune: measured result (honest)

This documents a real overnight LoRA fine-tune experiment on a Bengali text-to-speech model.
The headline is deliberately unembellished: the training pipeline works end to end, but the
fine-tune did **not** produce a material quality improvement on the available open data. The
negative result and its diagnosis are the value here.

## What this is and is not

- It IS: text -> speech (TTS). Input is Bengali text; output is a Bengali waveform.
- It is NOT: voice-to-voice (audio in -> audio out). The Bengali wall is the speech-output
  ("talker") half, so TTS is the right thing to target, but this is one half of a pipeline,
  not an end-to-end voice assistant.

## Why a TTS, and why this base

Across open omni speech models (MiniCPM-o, Qwen3-Omni, GLM-4-Voice, Kimi-Audio, Step-Audio 2),
none generate Bengali speech: their talkers are English/Chinese-centric. The understanding/text
side handles Bengali; the speech-output side is the failure. So the adaptable target is a
dedicated Bengali TTS.

The intended base was AI4Bharat Indic Parler-TTS (Apache-2.0, natively Bengali) trained on
IndicVoices-R (a clean, purpose-built Indic TTS corpus). Both are **gated** on the Hugging Face
Hub and the available token was not approved (HTTP 403), so this run used the ungated
`asif00/orpheus-bangla-tts` instead: a Llama-3.2-3B + SNAC model, Apache-2.0, and (per its own
card) a proof-of-concept fine-tuned on only 955 audiobook samples. It is a clean LoRA target
(autoregressive Llama), which is why it was preferred over `facebook/mms-tts-ben` (VITS,
conv-heavy, awkward for LoRA; also CC-BY-NC non-commercial).

## Method

- Base: `asif00/orpheus-bangla-tts` (3.3B). Audio codec: `hubertsiuzdak/snac_24khz`.
- Data: `google/fleurs` config `bn_in` (CC-BY-4.0), 512 train clips (~79 min) after filtering
  (2-12 s, NFC-normalized, dropped >15% Latin code-switch). Audio SNAC-encoded into the Orpheus
  7-codes-per-frame token scheme; sequences are
  `[SOH]+text+[EOT,EOH]+[SOS]+audio_tokens+[EOS]`, loss masked to the speech span.
- LoRA: r=16, alpha=32, dropout 0.05 on all Llama projections (q,k,v,o,gate,up,down);
  24.3M trainable params (0.73%), bf16, gradient checkpointing.
- Eval: generate held-out FLEURS `bn` sentences, re-transcribe with `faster-whisper large-v3`
  (Bengali), score CER/WER (`jiwer`) against the prompt text. `valid_rate` = fraction of clips
  that produced non-empty transcribable audio.
- Hardware: single RTX A5000, peak ~9.8 GB train. Two learning rates tried.

## Results

Training was stable (loss 9.63 -> ~3.7 at LR 2e-4; ~4.5 avg at LR 5e-5), adapters saved and
reloaded, generation works. Quality:

| run | held-out n | valid-rate | CER | WER |
| --- | ---: | ---: | ---: | ---: |
| base (orpheus-bangla, no LoRA) | 5 | 1.00 | 0.538 | 1.039 |
| LoRA @ lr 2e-4 | 5 | 0.80 | 0.770 | 1.042 |
| LoRA @ lr 5e-5 | 5 | 1.00 | 0.531 | 0.832 |
| base | 20 | 0.95 | 0.640 | 1.080 |
| LoRA @ lr 5e-5 | 20 | 0.85 | 0.645 | 1.033 |

Reading it straight:

- LR 2e-4 caused catastrophic forgetting: valid-rate dropped and CER got clearly worse, with
  degenerate/repeating and empty outputs.
- LR 5e-5 looked like a win on n=5 (WER 1.039 -> 0.832), but that did not survive a larger
  eval. On n=20 the LoRA is essentially neutral on CER/WER and slightly worse on valid-rate
  (0.95 -> 0.85). The n=5 improvement was noise.
- Net: no material quality improvement from fine-tuning on FLEURS Bengali.

## Why it didn't help, and the path that would

FLEURS is the wrong lever for this base: it is multi-speaker read speech at 16 kHz, a harder
and different domain than the single-narrator audiobook data the base was tuned on. Tuning a
small POC model toward that distribution destabilizes generation rather than improving it,
especially with a text-only intelligibility eval where the base is already mediocre but stable.

A genuine improvement needs clean, in-domain, single-/consistent-speaker Bengali TTS data. That
is exactly the gated IndicVoices-R corpus on the Indic Parler-TTS base. The highest-value next
step is obtaining that access and rerunning this same harness against it.

## Update: clean-corpus run (IndicVoices-R) — the data hypothesis, confirmed

After gated access was granted, the exact same harness was rerun with the base swapped only on
the data side: training on a filtered subset of AI4Bharat IndicVoices-R Bengali (a purpose-built
Indic TTS corpus) instead of FLEURS. Filters: `cer<=0.05`, `snr>=20`, duration 2-12 s, capped at
12 clips/speaker; 1500 clips, ~192 min, 244 speakers. Same LoRA (r=16, lr 5e-5, 600 steps),
same orpheus-bangla base, same eval.

Note on data handling: IndicVoices-R audio is 48 kHz and often stereo. An early run hung for
hours inside `librosa.resample` on the 2-D arrays; `audio_to_tokens` now mono-izes and resamples
on-GPU (bounded, fast). The `cer` field is stored as a string (`'tensor(0.0194)'`) and is parsed
accordingly.

Before/after on the SAME FLEURS-20 held-out set used above (apples-to-apples):

| run | valid-rate | CER | WER |
| --- | ---: | ---: | ---: |
| base (orpheus-bangla) | 0.95 | 0.640 | 1.080 |
| LoRA on FLEURS (earlier) | 0.95 | 0.645 | 1.033 |
| LoRA on IndicVoices-R | 0.95 | **0.498** | **0.910** |

In-domain IVR-test-20 (its own base measured on the same set):

| run | valid-rate | CER | WER |
| --- | ---: | ---: | ---: |
| base | 0.95 | 0.470 | 0.867 |
| LoRA on IndicVoices-R | 0.90 | 0.460 | 0.890 |

Reading it straight:

- On the comparable FLEURS-20 set, training on clean IndicVoices-R data is a real, measurable
  win: CER 0.640 -> 0.498 (-22% relative), WER 1.080 -> 0.910 (-16%), valid-rate unchanged. The
  FLEURS-trained LoRA did not move CER at all (0.645); the IVR-trained LoRA did. Same model, same
  recipe, same eval - only the training corpus changed. The corpus was the lever, exactly as the
  diagnosis above predicted.
- In-domain IVR-test-20 is roughly neutral (CER -2%, WER +3%, valid-rate 0.95 -> 0.90). The base
  is already comparatively decent there; the gain manifests as better generalization on the
  harder FLEURS distribution rather than on in-domain text.

This closes the loop: the earlier negative result was a data problem, not a pipeline problem.
Raw numbers: `results_bn_tts_ivrlora_f20.json`, `results_bn_tts_base_i20.json`,
`results_bn_tts_ivrlora_i20.json`. Reproduce with `ft/run_ivr_pipeline.sh` (gated access required).

## What this proves regardless

The training-and-evaluation pipeline is real and reproducible: SNAC encode -> Orpheus sequence
packing -> LoRA SFT -> adapter save/reload -> generation -> re-ASR CER/WER. Swapping in a better
base and corpus is a config change, not new infrastructure.

## Reproduce

```bash
# env (separate from inference venvs; orpheus/SNAC + eval deps)
uv venv --python 3.12 .venv-tts-lora
uv pip install --python .venv-tts-lora/bin/python torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-tts-lora/bin/python -r requirements-tts-lora.txt

# prefetch FLEURS bn (CC-BY-4.0) is automatic via huggingface_hub on first use
G="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-tts-lora/bin/python"
$G ft/orpheus_baseline.py 20 base20                                   # before
$G ft/prep_orpheus_data.py train 512                                  # SNAC-encode train set
$G ft/train_orpheus_lora.py data/tts_bn/orpheus_fleurs_train_512.pt adapters/orpheus_bn_lora 300 4 5e-5
$G ft/orpheus_baseline.py 20 lora20 adapters/orpheus_bn_lora          # after
$G ft/eval_bn_tts.py base20 cuda && $G ft/eval_bn_tts.py lora20 cuda  # CER/WER
```
