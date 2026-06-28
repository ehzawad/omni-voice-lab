# hervoice — Open-Weight Speech-to-Speech / Full-Duplex LLM Survey (June 2026)

Target HW: 1× RTX A5000 (24GB) + 1× RTX A6000 (48GB) = 72GB. Goal: GPT-4o-Voice / Gemini-Live feel
(full-duplex or low-latency streaming) + smart fine-tunable brain + open weights + local LoRA.

## Decision: MiniCPM-o 4.5 (primary base for hervoice)
The only model that is simultaneously: truly full-duplex, smart-brained, Apache-2.0, easy LoRA, fits HW.

## Top-3
1. **MiniCPM-o 4.5** `openbmb/MiniCPM-o-4_5` — Feb 2026, 9.37B, Apache-2.0. TRUE full-duplex (interleaved
   text/speech decoder + 1Hz proactive speak/listen loop). Brain = Qwen3-8B, OpenCompass 77.6. ~18-20GB
   bf16 inference (fits A5000 alone); LoRA fits A6000 easily. Voice clone from reference. Fine-tune via
   LLaMA-Factory. Paper arXiv:2604.27393. Repo github.com/OpenBMB/MiniCPM-o.
   Watch-outs: no single published ms latency; duplex pipeline more complex to stand up; EN/ZH focus.
2. **Fun-Audio-Chat-8B** `FunAudioLLM/Fun-Audio-Chat-8B` — Dec 2025, 9.45B, Apache-2.0. Strongest ~8B
   Apache audio brain; speech function-calling; has a `Fun-Audio-Chat-Duplex` full-duplex variant.
   ~24GB inference (A6000 fine). LoRA fits 72GB; FULL SFT does NOT (authors use 4×80GB). arXiv:2512.20156.
3. **Qwen3-Omni-30B-A3B-Instruct** `Qwen/Qwen3-Omni-30B-A3B-Instruct` — Sep 2025, 35.3B MoE (~3B active),
   Apache-2.0. Smartest brain; 211ms first-packet; but STREAMING TURN-BASED, not full-duplex. bf16 does
   NOT fit 72GB → must run 4-bit AWQ (vLLM) + QLoRA. arXiv:2509.17765.

## Truly full-duplex vs marketing "real-time"
TRUE full-duplex (listen-while-speak + native barge-in): Moshi, **MiniCPM-o 4.5**, FLM-Audio (research),
BayLing-Duplex (Jun 2026), SALMONN-omni, DuplexSLA, Fun-Audio-Chat-Duplex variant.
NOT full-duplex (streaming/turn-based, "real-time"=low TTFT): Qwen2.5/3-Omni, Kimi-Audio, Step-Audio 2,
GLM-4-Voice, VITA-1.5/Audio, Baichuan, LLaMA-Omni 1/2, SpeechGPT-2 (near), Freeze-Omni (partial),
Mini-Omni2 (keyword interrupt only).

## Notable others
- Kimi-Audio `moonshotai/Kimi-Audio-7B-Instruct` — MIT, half-duplex, strong audio brain, but full-SFT only
  (NO LoRA) → can't fully fine-tune on 72GB. Good understanding, wrong shape for full-duplex hervoice.
- Moshi `kyutai/moshika-pytorch-bf16` — CC-BY-4.0, TRUE full-duplex, best ~200ms latency, but weak Helium-7B
  brain. Official LoRA (kyutai-labs/moshi-finetune). Data: stereo 2-channel + timestamped transcripts.
- Higgs Audio v2 — TTS + best-in-class voice cloning ("other" license). Candidate for hervoice OUTPUT voice
  layer, not the conversation loop.
- License landmines (avoid/commercial-restricted): GLM-4-Voice, Baichuan-Omni-1.5, LLaMA-Omni 1/2,
  FLM-Audio, Higgs, Qwen2.5-Omni-**3B** (research-only; the 7B is Apache).

## HW hard limits
- Full SFT of any 8-9B model exceeds 72GB → use LoRA/QLoRA (fits everything here).
- Qwen3-Omni bf16 won't fit → 4-bit AWQ + QLoRA.
- MiniCPM-o 4.5: LoRA on A6000, keep A5000 for streaming encoder/decoder + ASR fallback.

## To verify before/while committing
- Exact published latency for MiniCPM-o 4.5.
- That the OPEN `Fun-Audio-Chat-Duplex` checkpoint (not just dense 8B) is actually downloadable.
