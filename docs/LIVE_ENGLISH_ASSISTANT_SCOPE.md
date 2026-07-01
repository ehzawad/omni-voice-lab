# Live English Assistant Scope

This is the build scope for closing the English "live voice" gap on one shared RTX A5000.
It is intentionally narrower than MiniCPM-o 4.5's upstream full-duplex claims because this
repository has not measured a true simultaneous-duplex serving loop.

## Honest Claim

The target is a live, interruptible English voice assistant implemented as a streaming
turn loop:

- User audio is accepted in small chunks.
- Silero VAD detects speech start and end.
- MiniCPM-o receives the completed spoken turn through `streaming_prefill`.
- MiniCPM-o streams text and speech through `streaming_generate`.
- If VAD detects new user speech while the assistant is speaking, playback/generation is
  cancelled and a new user turn begins.

This means **turn-based streaming with VAD barge-in**, not proven simultaneous full-duplex.
On the current A5000 baseline, realistic responsiveness is about **1-3 seconds** from
end-of-user-speech to first assistant audio, with prior measured first-audio around **2.2s**
and RTF around **1.12** at int4. Long answers may drift because generation is slightly
slower than real-time.

Do not claim sub-second response, Moshi-style sub-200ms interaction, or true simultaneous
full-duplex unless those numbers are measured in this repository on the target GPU.

## Minimal Deliverable

The smallest deliverable that genuinely proves the live, interruptible English assistant is:

1. A reusable live engine loop around MiniCPM-o 4.5 streaming APIs.
2. Silero VAD turn detection for 16 kHz mono input.
3. Barge-in cancellation: user speech while assistant audio is streaming stops the current
   assistant output and starts a new user turn.
4. A runnable headless simulation that feeds WAV files in real time, injects a second WAV as
   an interruption, and logs timings.
5. A local mic client the user can run on their own machine.
6. This honest documentation, kept current with only measured numbers.

If any of these are missing, the live layer is not materially proven.

## Engine Design

Use the existing MiniCPM-o streaming path from `minicpm_stream.py`, not a new model stack.

The engine should load once:

- `openbmb/MiniCPM-o-4_5`
- int4 LLM quantization with the existing `KEEP_FP` skip list
- `AutoTokenizer`
- `model.init_tts()`
- `model.init_token2wav_cache(ref_audio)`
- one system prefill that includes the cloned reference voice and assistant policy

Session handling should be conservative. Start with one active user session and a fresh
MiniCPM session id per conversation or per interrupted turn if cancellation leaves the
remote-code generation state uncertain. Prefer correctness over preserving long context
tonight. If context is preserved, it must be proven by the simulation after at least one
barge-in.

The first version should use the proven turn-based streaming mechanics:

1. Accumulate user audio chunks while VAD reports speech.
2. After end-of-speech, call `streaming_prefill(..., is_last_chunk=True)` for the final
   user chunk.
3. Start `streaming_generate(..., generate_audio=True)`.
4. Stream audio chunks to the output sink as they arrive.
5. Between generated chunks, check a cancellation event set by VAD.
6. On cancellation, stop yielding assistant audio immediately, reset or replace the
   MiniCPM session safely, and begin collecting the new user turn.

Do not implement MiniCPM-o Omni-Flow/proactive 1 Hz speaking tonight unless it is already
available through a stable API and can be measured. The deliverable does not depend on it.

## VAD And Barge-In State Machine

Use Silero VAD as the only turn-taking dependency tonight. Suggested initial parameters:

- Input: 16 kHz mono PCM float32 or int16 converted consistently.
- Chunk size: 20-100 ms for VAD.
- Speech-start confirmation: 100-200 ms.
- End-of-speech threshold: 250-500 ms of silence.
- Pre-roll: keep 200-300 ms before speech start so words are not clipped.

States:

- `IDLE`: no active user speech or assistant output.
- `USER_SPEAKING`: VAD is collecting a user utterance.
- `MODEL_PREFILL`: the completed user utterance is being submitted to MiniCPM-o.
- `ASSISTANT_SPEAKING`: generated audio/text is streaming out.
- `INTERRUPTED`: VAD detected user speech during `ASSISTANT_SPEAKING`; stop output and
  move to `USER_SPEAKING` for the new turn.

Barge-in rule:

- During `ASSISTANT_SPEAKING`, speech-start confirmation sets `cancel_generation`.
- The output sink must stop accepting assistant chunks immediately.
- The generator loop must check `cancel_generation` between every yielded chunk and exit.
- The new user utterance must include pre-roll audio captured before the interrupt was
  confirmed.

The implementation should log the state transition and the elapsed time from interrupt
speech start to assistant audio stop. That is the barge-in proof metric.

## Headless Simulation

The shared box has no microphone or speaker, so the authoritative proof must be file-driven.

The simulation should:

- Load a first user WAV, a second user WAV, and the reference voice WAV.
- Feed the first WAV in real time to the engine at 16 kHz.
- Let the assistant begin streaming.
- Inject the second WAV while assistant output is active.
- Assert that the assistant stream is cancelled and a new user turn starts.
- Write any assistant audio chunks to files for inspection.
- Emit JSONL timing logs.

Required timings in the log:

- `vad_speech_start_ms`
- `vad_speech_end_ms`
- `prefill_start_ms`
- `prefill_done_ms`
- `generate_start_ms`
- `first_audio_ms`
- `interrupt_speech_start_ms`
- `assistant_audio_stop_ms`
- `barge_in_stop_latency_ms`
- `turn_total_ms`
- `rtf`
- `peak_vram_gb`

Passing simulation criteria:

- First turn produces non-empty text or audio.
- First assistant audio timing is recorded from the real run.
- Barge-in is detected while assistant output is active.
- Assistant audio stops after the interrupt event.
- A second user turn is accepted after cancellation.
- The run logs exact hardware, quantization, chunk sizes, VAD thresholds, and model id.

If the simulation cannot run or does not show these events, say "the live/interruptible
layer is not materially proven yet."

## Local Mic Client

The local client is for the user's machine, not the shared headless GPU box.

Minimal client:

- Captures mic audio with `sounddevice`.
- Resamples or records as 16 kHz mono.
- Sends chunks to the server over a simple localhost/WebSocket protocol.
- Plays streamed assistant audio chunks as they arrive.
- Sends an interrupt signal or raw mic chunks continuously enough for server-side VAD to
  detect barge-in.

For tonight, a plain WebSocket client is enough. WebRTC is a production concern and should
not block the proof.

## Cut Tonight

Cut these explicitly to protect the core:

- WebRTC, NAT traversal, TURN/STUN, browser production audio.
- Multi-user serving, queues, auth, tenancy, autoscaling.
- MiniCPM-o proactive 1 Hz speaking / Omni-Flow simultaneous-duplex mode.
- Video input or video grounding.
- Polished UI.
- Mobile support.
- Wake word.
- Persistent memory.
- Latency claims from vendor demos or papers.

## Required Documentation Language

The README or demo doc must state plainly:

- This live layer is **turn-based streaming with VAD barge-in**, not proven simultaneous
  full-duplex, unless a later measured run proves otherwise.
- The initial implementation is **single-user**.
- The assistant speaks in a **cloned reference voice** and requires a reference clip; there
  is no usable default voice.
- Latency numbers are reported only from real local runs, with hardware, quantization,
  chunk size, VAD settings, and prompt length recorded.
- Current measured baseline is about 2.2s first-audio and RTF about 1.12 at int4 on an
  RTX A5000, not sub-second and not sub-200ms.

Do not soften these caveats. They are part of the deliverable.
