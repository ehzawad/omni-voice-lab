# Clean English Speech Datasets — survey for hervoice (Qwen2.5-Omni voice-to-voice)

Architectural note: Qwen2.5-Omni is **audio-in → text + speech-out** (Thinker–Talker split).
Two different data shapes are needed:
- **Thinker (understanding / ASR):** any clean speech+transcript corpus; also the *spoken question* side of dialogue sets.
- **Talker (voice out):** clean `text → 24 kHz speech` pairs in one consistent voice.
  WARNING: "answer audio" in voice-assistant corpora is stored as foreign neural codecs
  (SNAC for Mini-Omni, CosyVoice2 tokens for VocalNet) — NOT Qwen-native. Decode to waveform
  and re-tokenize with Qwen's own audio tokenizer; you cannot feed those codec tokens directly.

## Ranked "start small & clean" pilot picks
1. **LibriTTS-R** (`mythicinfinity/libritts_r`) — best clean VOICE/TTS pilot. 24 kHz studio-clean
   (matches Omni's speech-out SR), has `text_normalized`/`text_original` + `speaker_id`. CC-BY-4.0.
   Pilot: config `clean`, split `train.clean.100`, filter ONE speaker → ~500 utterances for a voice LoRA.
2. **LibriSpeech** (`openslr/librispeech_asr`) — best clean ASR/UNDERSTANDING pilot. ~1000h, 16 kHz,
   CC-BY-4.0, ungated. Pilot: ~500–1000 utterances from `train.clean.100` for audio→text LoRA.
3. **SDF English** (`minghanw/sdf_dataset_en`) — best small clean true SPEECH-DIALOGUE pilot.
   1K–10K dialogues, Apache-2.0, conversational, 24 kHz-class. Pilot: ~300–1000 turns end-to-end.
   (Synthetic + voice-cloned; eyeball quality before scaling.)

## Fuller option table
| Dataset | HF id | Size | License | Quality | SR | Transcripts | Best for |
|---|---|---|---|---|---|---|---|
| LibriSpeech | `openslr/librispeech_asr` | ~1000h | CC-BY-4.0 | clean read | 16k | yes | ASR/Thinker |
| LibriTTS | `mythicinfinity/libritts` | ~585h, 2456 spk | CC-BY-4.0 | clean read | 24k | yes | multi-spk TTS |
| LibriTTS-R | `mythicinfinity/libritts_r` | ~585h restored | CC-BY-4.0 | studio-clean | 24k | yes | Talker/voice persona |
| People's Speech | `MLCommons/peoples_speech` | 30000h+ | CC-BY / CC-BY-SA (split!) | mixed/noisy | 16k | yes | large ASR (not TTS) |
| Common Voice 17 | `mozilla-foundation/common_voice_17_0` | thousands h | CC0-1.0 | variable | 48k | yes | accent-robust ASR |
| GigaSpeech | `speechcolab/gigaspeech` (gated) | 10000h | apache-2.0 + agreement | spontaneous | 16k | yes | spontaneous ASR |
| VoiceAssistant-400K | `gpt-omni/VoiceAssistant-400K` | 470k ex | apache-2.0 | answer=SNAC tokens | TTS | yes | spoken-instruction (audio-in usable) |
| VoiceAssistant-430K | `VocalNet/VoiceAssistant-430K-vocalnet` | ~430k | apache-2.0 | answer=CosyVoice2 tokens | — | yes | speech-dialogue scaling |
| UltraChat-vocalnet | `VocalNet/UltraChat-vocalnet` | ~300k | apache-2.0 | CosyVoice2 tokens | — | yes | multi-turn speech-dialogue |
| SDF English | `minghanw/sdf_dataset_en` | 1K–10K | apache-2.0 | synthetic, clean | 24k | yes | small clean dialogue pilot |
| Spoken-SQuAD | `alinet/spoken_squad` | ~37k/5k | unknown | TTS, noisy ASR | 16k | yes | spoken-QA EVAL only |

## Licensing gotchas
- Common Voice now via Mozilla Data Collective (Oct 2025); data is CC0 but expect an access step.
- GigaSpeech gated + usage agreement; underlying YouTube/podcast rights — research-leaning.
- People's Speech: `*_sa` configs are CC-BY-SA (copyleft) — prefer plain `clean`/`dirty` CC-BY.
- LibriTTS-R: arXiv says CC-BY-NC-ND but HF/OpenSLR release is CC-BY-4.0 — use CC-BY-4.0, attribute.
- Spoken-SQuAD license "unknown" (derived from SQuAD CC-BY-SA) → eval only, not product training.
- Synthetic/voice-cloned sets (SDF, VocalNet): Apache-2.0 covers data, but add consent/likeness checks
  before shipping a cloned persona voice.

## Concrete first moves
1. Thinker LoRA: `load_dataset("openslr/librispeech_asr","clean",split="train.clean.100")`, ~1000 → audio→text.
2. Talker/voice LoRA: `load_dataset("mythicinfinity/libritts_r","clean",split="train.clean.100")`, one speaker, ~500.
3. End-to-end dialogue smoke test: `load_dataset("minghanw/sdf_dataset_en")`, ~300–1000 turns.
4. Scale: add People's Speech `clean` (CC-BY only); re-tokenize VocalNet answer waveforms for big dialogue speech-out.
