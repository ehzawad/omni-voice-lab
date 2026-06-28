# hervoice — Expressive TTS (voice-layer) survey, June 2026

Context: these are CSM-like LLM-backbone SPEECH GENERATORS (no brain) = the VOICE layer for a MODULAR stack.
They are NOT single-network assistants. Use to (a) benchmark MiniCPM-o 4.5's voice ceiling, (b) as the
escape hatch if voice quality ever outranks the single-network requirement.

Key finding: **Sesame never open-sourced a CSM successor** (Tiny-1B/Small-3B/Medium-8B stayed internal;
only `sesame/csm-1b` is public, Apache-2.0, English, no text/reasoning). "Stronger backbone than CSM"
means a different family.

## Top-3 voice layers
1. **Higgs Audio v2** `bosonai/higgs-audio-v2-generation-3B-base` — Llama-3.2-3B + 2.2B DualFFN (~5.8B), 24k.
   BEST measured English naturalness (EmergentTTS/Seed-TTS SOTA), zero-shot clone 3-10s, realtime. ~12-13GB
   infer (fits A5000). License = **"other"** (verify before commercial). Paper arXiv:2505.23009.
2. **Orpheus-3B** `canopylabs/orpheus-3b-0.1-ft` — Llama-3.2-3B + SNAC 24k, **Apache-2.0**, streaming
   **~100-200ms TTFA**, emotion tags, zero-shot clone, **official Unsloth LoRA notebook**. ~8GB infer.
   Cleanest permissive production pick. github.com/canopyai/Orpheus-TTS.
3. **CosyVoice 3** `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` — Qwen2.5-0.5B+flow, **best English WER ~2.24%**,
   **~150ms** bi-streaming, Apache-2.0, ~3s clone. Backbone only 0.5B (not "bigger than CSM") but punches up.

Bigger-backbone alt: **Llasa-8B** (Llama-3.1-8B) but **CC-BY-NC**, 16kHz, no official streaming.

## Streaming (needed for realtime): Orpheus ~100-200ms, CosyVoice2/3 ~150ms, XTTS-v2 <200ms,
Kyutai TTS ~220ms, CSM ~300ms, Higgs ~600ms. Batch-only (NOT live): F5-TTS, MaskGCT, MegaTTS3, Zonos,
Spark-TTS, Kokoro (also NO cloning).

## June-2026 leads (UNVERIFIED — confirm repo/license before betting): Higgs Audio v3 (Qwen3-4B, streaming,
non-commercial), MisoTTS-8B (Llama-8B+Mimi, ~110ms, "closest open CSM-successor by feel"), VibeVoice-Realtime-0.5B,
NeuTTS Air (CPU realtime), Kani-TTS-2.

## Verdict for hervoice
If staying SINGLE-NETWORK (user's stated requirement) → MiniCPM-o 4.5 stays; improve its voice via cloning +
TTS LoRA. These TTS models = the measuring stick + modular escape hatch. Best modular voice pair would be
LLM(Qwen3/MiniCPM) + Orpheus-3B (permissive) or Higgs v2 (max naturalness).
