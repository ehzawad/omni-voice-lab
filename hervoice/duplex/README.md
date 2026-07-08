# hervoice/duplex — Moshi, a NATIVE full-duplex speech dialogue model

This directory stands up **Moshi** (`kyutai/moshiko-pytorch-bf16`) locally on the
A5000 (GPU0) and proves, headlessly, that its turn-taking is **native** — decided
inside the neural network — not stitched together by our code.

## What Moshi is (and why it is different from `hervoice/live/`)

`hervoice/live/` is a **VAD-gated streaming turn loop**: Silero VAD + a Python
state machine decide when the user stopped, when to answer, and when to
barge-in. The turn-taking lives in *our* code.

**Moshi is the opposite.** It is a single ~7B autoregressive Transformer that:

- models **two audio streams in parallel** — the user's incoming audio *and* its
  own outgoing audio — using the **Mimi** neural codec (**12.5 Hz**, 80 ms
  frames, 24 kHz audio compressed to **~1.1 kbps** with 8 residual codebooks);
- emits a time-aligned **text "inner monologue"** stream (one text token per
  frame) that is what it is about to say;
- decides **when to speak, when to stay silent, and when to stop because it was
  interrupted entirely inside the network**. There is no external VAD and no
  turn-taking controller. Reported end-to-end latency in the Moshi paper is
  **~200 ms**.

The whole point of this build is to demonstrate that difference on real audio,
with logged, machine-readable evidence.

## Files

- `prove_duplex.py` — the headless native-duplex proof (drives the streaming
  step loop frame-by-frame; **no VAD anywhere**).
- `runs/duplex/summary.json` — machine-readable result.
- `runs/duplex/moshi_reply.wav` — the model's own generated speech, decoded from
  its audio tokens via Mimi.
- `runs/duplex/inner_monologue.txt` — the model's time-aligned inner-monologue text.
- `runs/duplex/frame_trace.jsonl` — per-frame audit trail (input phase, text
  token, decoded piece, output-audio RMS).
- `runs/duplex/log` — human-readable run log.

## How to run the headless proof

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv-duplex/bin/python -m hervoice.duplex.prove_duplex
```

It feeds a user question WAV (resampled to 24 kHz mono) into the model
frame-by-frame, then feeds **trailing silence** so the model has room to take its
turn. Per frame it captures the model's own audio tokens (decoded to
`moshi_reply.wav`) and its inner-monologue text token.

### Measured result (this hardware — RTX A5000, GPU0)

From `runs/duplex/summary.json` (input: `examples/in_fifa_question.wav`, a spoken
"how many World Cups has Brazil won" question):

- **Peak VRAM (GPU0):** 16.85 GB nvidia-smi / 17.43 GB `torch.max_memory_allocated`
  — comfortably under the 24 GB A5000.
- **Frames stepped:** 181 at 12.5 Hz (56 user frames = 4.48 s, + 125 trailing
  silence frames = 10 s). ~178 ms/step (first step compiles CUDA graphs).
- **Model-speaking frames (inner monologue emitted a word):** 17 — of which
  **9 fell during the user's audio** and **8 during the trailing silence**.
- **Inner-monologue text (verbatim):**
  `"there! How is it going? Brazil has won the World Cup 5 times."`

**Emergent turn-taking, frame by frame** (see `runs/duplex/frame_trace.jsonl`):

| frames | input phase | what the model's inner monologue does |
|--------|-------------|----------------------------------------|
| 0–4    | USER        | pad tokens (id 0/3), output near-silent — **listening** |
| 5–21   | USER        | a spoken backchannel greeting *"there! How is it going?"* |
| 22–50  | USER        | pad again — **listening** to the World-Cup question |
| 51–56  | USER (ends 56) | begins the answer: *"Brazil has …"* |
| 57–70  | **SILENCE** | **continues its own turn after the user stopped**: *"won the World Cup 5 times."* |

The substantive answer **starts while the user is still finishing and continues
into the silence** — the model, not our code, decided when to take the turn. (The
answer is also factually correct: Brazil has 5 World Cups.)

### How this shows there is NO external VAD

- `prove_duplex.py` contains no `silero`, no `webrtcvad`, no energy gate, and no
  turn-taking state machine. Confirm:

  ```bash
  grep -rInE 'silero|webrtcvad|vad|energy|threshold|turn_detector|state.?machine' \
    hervoice/duplex/*.py
  ```

  The only matches are inside comments/docstrings and the summary field name
  `uses_external_vad` (set to `false`) — never as control logic. No `import` of
  any VAD library exists (the package does not even depend on one).
- The single signal we read to say "the model started speaking" is the model's
  **own inner-monologue text stream** (a real word-piece token instead of the
  pad token). That decision is made by the network, frame by frame.
- The output-audio RMS logged in `frame_trace.jsonl` is **descriptive only** — it
  is never fed back into any decision and gates nothing.

## How the user talks to it live (server + web UI)

The `moshi` package ships a server with a browser mic UI. On the (headless) box:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  .venv-duplex/bin/python -m moshi.server --host 0.0.0.0 --port 8998
```

It loads Moshiko and binds `:8998`. From a laptop with a mic, open an SSH tunnel
and use the browser (mic + speakers) — the browser talks to the model over a
WebSocket; **all** turn-taking happens in the model:

```bash
ssh -N -L 8998:localhost:8998 user@this-box
# then open http://localhost:8998 in a browser and click to start talking
```

(We verified the server process starts and binds the port here, then stopped it.
We do **not** run a live mic on this headless box.)

## Honest limits

- **English-only, single voice** (Moshiko is one fixed male voice; Moshika is a
  female voice). No language switching.
- **Single session** at a time in this proof (`batch_size = 1`).
- The proof feeds audio **from a file**, not a live mic (headless box). It drives
  the exact same streaming step API the live server uses, so it is a faithful
  demonstration of the model's native duplex behaviour, but it is not a live
  conversation.
- Moshiko is a base full-duplex model, not instruction-tuned for tasks; its
  replies are conversational, not a grounded QA answer.
- **PersonaPlex (`nvidia/personaplex-7b-v1`) is the gated upgrade** (better
  voices / multi-speaker) and is **not** used here — it returns 403 until the
  NVIDIA license is accepted on the Hub.
</content>
