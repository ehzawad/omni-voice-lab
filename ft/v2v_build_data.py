#!/usr/bin/env python
"""STEP 1 - build paired spoken-QA data for the voice-to-voice LoRA.

Question/answer texts are grounded in fifa_kb.md. The spoken QUESTION audio is
synthesized with Chatterbox TTS (.venv-modular). Outputs:
  data/v2v/audio/q_XXXX.wav        - spoken question waveforms (24 kHz mono)
  data/v2v/manifest.jsonl          - {id, split, question_text, answer_text, question_wav}

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-modular/bin/python ft/v2v_build_data.py
"""
import json, os, time
import torch, torchaudio
from chatterbox.tts import ChatterboxTTS

ROOT = "/mnt/sdb/arafat/hervoice"
OUTAUD = os.path.join(ROOT, "data/v2v/audio")
MANIFEST = os.path.join(ROOT, "data/v2v/manifest.jsonl")
os.makedirs(OUTAUD, exist_ok=True)

# (question, concise ground-truth answer) grounded in fifa_kb.md
QA = [
    ("How many times has Brazil won the World Cup?", "Five times, in 1958, 1962, 1970, 1994, and 2002."),
    ("Which country has won the most Men's World Cups?", "Brazil, with five titles."),
    ("In what years did Brazil win the World Cup?", "1958, 1962, 1970, 1994, and 2002."),
    ("How many World Cups have Germany and Italy each won?", "Four each."),
    ("How many World Cups has Argentina won?", "Three, in 1978, 1986, and 2022."),
    ("In which years did Argentina win the World Cup?", "1978, 1986, and 2022."),
    ("Who won the 2022 World Cup?", "Argentina, beating France on penalties after a 3-3 draw."),
    ("Where was the 2022 World Cup held?", "In Qatar."),
    ("Who won the Golden Ball at the 2022 World Cup?", "Lionel Messi."),
    ("Who won the 2018 World Cup?", "France, who beat Croatia 4-2 in the final."),
    ("Where was the 2018 World Cup held?", "In Russia."),
    ("What was the score in the 2018 World Cup final?", "France beat Croatia 4-2."),
    ("Who is the all-time top scorer in World Cup history?", "Miroslav Klose of Germany, with 16 goals."),
    ("How many World Cup goals did Miroslav Klose score?", "Sixteen goals."),
    ("When and where was the first World Cup held?", "In 1930 in Uruguay, which Uruguay won."),
    ("Who won the first World Cup?", "Uruguay, in 1930."),
    ("How often is the World Cup held?", "Every four years."),
    ("When was the first Women's World Cup held?", "In 1991."),
    ("Which team has won the most Women's World Cups?", "The United States, with four titles."),
    ("How many Women's World Cups has the United States won?", "Four."),
    ("Who won the 2023 Women's World Cup?", "Spain, who beat England 1-0 in the final."),
    ("Where was the 2023 Women's World Cup held?", "Co-hosted by Australia and New Zealand."),
    ("What was the score in the 2023 Women's World Cup final?", "Spain beat England 1-0."),
    ("How long is a standard football match?", "90 minutes, in two halves of 45 minutes."),
    ("How long is each half of a football match?", "45 minutes."),
    ("How long can the half-time interval last?", "No more than 15 minutes."),
    ("How many players are on a football team?", "Eleven, including one goalkeeper."),
    ("What is the minimum number of players to continue a match?", "Seven players."),
    ("How far is the penalty spot from the goal line?", "11 metres, or 12 yards."),
    ("What does a yellow card mean?", "A caution."),
    ("What does a red card mean?", "A sending-off."),
    ("What happens after two yellow cards in a match?", "They equal a red card and the player is sent off."),
    ("What is a hat-trick?", "When a single player scores three goals in one match."),
    ("How many goals make a hat-trick?", "Three goals in one match."),
    ("Who maintains the Laws of the Game?", "The International Football Association Board, IFAB."),
    ("How many Laws of the Game are there?", "Seventeen."),
    ("How many situations does VAR review?", "Four match-changing situations."),
    ("What situations does VAR review?", "Goals, penalty decisions, direct red-card incidents, and mistaken identity."),
    ("When was FIFA founded?", "In 1904."),
    ("Where is FIFA headquartered?", "In Zurich, Switzerland."),
    ("How many substitutions are teams now allowed?", "Up to five substitutions during a match."),
    ("How wide is a football goal?", "7.32 metres."),
    ("How tall is a football goal?", "2.44 metres."),
    ("How far is the penalty area from the goal line?", "16.5 metres."),
    ("What happens if a knockout match is level after 90 minutes?", "30 minutes of extra time, then a penalty shootout."),
    ("What decides a knockout match still level after extra time?", "A penalty shootout."),
    ("Which confederation governs football in Europe?", "UEFA."),
    ("Which confederation governs football in South America?", "CONMEBOL."),
    ("Which confederation governs football in Africa?", "CAF."),
    ("Which confederation governs football in Asia?", "AFC."),
    ("Who organises the Ballon d'Or?", "France Football, not FIFA."),
    ("What is FIFA's own award for the best player called?", "The Best FIFA Men's Player."),
    ("What is awarded for a direct-free-kick offence inside the penalty area?", "A penalty kick."),
    ("Can a goal be scored directly from a direct free kick?", "Yes."),
    ("What must happen before a goal counts from an indirect free kick?", "The ball must touch another player first."),
    ("Who won the most Ballon d'Or awards?", "Lionel Messi."),
    ("What is the Golden Boot awarded for?", "Scoring the most goals in a World Cup tournament."),
    ("Which country hosted and won the first World Cup?", "Uruguay, in 1930."),
    ("How many halves are in a football match?", "Two."),
    ("Who beat France in the 2022 World Cup final?", "Argentina, on penalties."),
    ("What was the score before penalties in the 2022 final?", "3-3."),
    ("Which team did Spain beat to win the 2023 Women's World Cup?", "England."),
    ("Which team did France beat in the 2018 final?", "Croatia."),
    ("Is being in an offside position an offence by itself?", "No, it is not an offence in itself."),
    ("How many continental confederations are under FIFA?", "Six."),
    ("Name the six FIFA confederations.", "UEFA, CONMEBOL, CONCACAF, CAF, AFC, and OFC."),
    ("Which body governs international football worldwide?", "FIFA."),
    ("What position is one of the eleven players required to be?", "Goalkeeper."),
    ("When did the five-substitution rule start?", "In 2020."),
    ("What is the penalty spot distance in yards?", "12 yards."),
    ("How many goals did the all-time World Cup top scorer record?", "Sixteen."),
    ("Which nation won the World Cup in 1970?", "Brazil."),
    ("Which nation won the World Cup in 1994?", "Brazil."),
    ("Which nation won the World Cup in 1986?", "Argentina."),
    ("Who won the World Cup in 2002?", "Brazil."),
    ("Who hosted the 2022 World Cup?", "Qatar."),
    ("Which player won the Golden Ball in 2022?", "Lionel Messi."),
    ("How many World Cup titles does Italy have?", "Four."),
    ("How many World Cup titles does Germany have?", "Four."),
    ("What is the width of the goal in metres?", "7.32 metres."),
    ("What shape result decided the 2022 World Cup?", "A penalty shootout."),
    ("Which award goes to the top scorer of a World Cup?", "The Golden Boot."),
    ("What is the maximum half-time interval?", "15 minutes."),
    ("How many minutes of extra time in a knockout?", "30 minutes."),
    ("Who won the Women's World Cup in 2023?", "Spain."),
    ("What confederation is CONCACAF?", "North, Central America and the Caribbean."),
    ("What does OFC stand for in football regions?", "Oceania."),
    ("How many players minimum before a match is abandoned?", "Fewer than seven."),
    ("When was the Women's World Cup first held?", "In 1991."),
    ("Which country won four Women's World Cups?", "The United States."),
    ("What year was FIFA founded?", "1904."),
    ("In which city is FIFA based?", "Zurich."),
    ("How many Laws of the Game exist?", "Seventeen."),
    ("Who keeps the Laws of the Game?", "IFAB."),
    ("What card equals two yellows?", "A red card."),
    ("How many goals is a hat-trick?", "Three."),
    ("What is the height of a football goal?", "2.44 metres."),
    ("How far from goal is the penalty area edge?", "16.5 metres."),
]

N_HELD = 20

def main():
    dev = "cuda"
    print(f"[load] Chatterbox TTS on {dev}", flush=True)
    tts = ChatterboxTTS.from_pretrained(device=dev)
    sr = tts.sr
    rows = []
    t0 = time.time()
    n = len(QA)
    for i, (q, a) in enumerate(QA):
        split = "test" if i >= n - N_HELD else "train"
        wav = tts.generate(q)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav_path = os.path.join(OUTAUD, f"q_{i:04d}.wav")
        torchaudio.save(wav_path, wav.cpu(), sr)
        dur = wav.shape[-1] / sr
        rows.append({"id": i, "split": split, "question_text": q,
                     "answer_text": a, "question_wav": wav_path, "dur_s": round(dur, 2)})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] dur={dur:.1f}s elapsed={time.time()-t0:.0f}s", flush=True)
    with open(MANIFEST, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    durs = [r["dur_s"] for r in rows]
    print(f"[done] {len(rows)} pairs, {sum(s['split']=='train' for s in rows)} train / "
          f"{sum(s['split']=='test' for s in rows)} test, sr={sr}, "
          f"dur min/mean/max={min(durs):.1f}/{sum(durs)/len(durs):.1f}/{max(durs):.1f}s", flush=True)
    print(f"[wrote] {MANIFEST}", flush=True)

if __name__ == "__main__":
    main()
