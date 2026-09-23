# Attribution for evaluation audio

`user_ref.wav` / `user_ref.txt` — a 5.5 s utterance by speaker `S4258185100319593` from
**IndicVoices-R** (Bengali, extempore), AI4Bharat, IIT Madras, licensed **CC-BY-4.0**
(https://huggingface.co/datasets/ai4bharat/indicvoices_r). Used unmodified except resampling
to 24 kHz, as the reference voice for synthesising the *user* side of the scripted scenarios.

`audio/*/t*.wav` — synthetic user turns generated from that reference with
`ai4bharat/IndicF5` (MIT) by `render_user_voice.py`. Not human speech.
