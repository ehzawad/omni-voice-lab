#!/usr/bin/env python3
"""Multi-turn conversation test: does the bot actually USE memory, and does it survive audio?

Two modes, run both, in this order:

  --mode text   drives ServiceEngine + Conversation directly (no ASR, no WebSocket). Isolates
                the memory LOGIC: if a follow-up fails here, it is the brain or the history
                budget, not the microphone path. Fast.
  --mode audio  drives the real gateway over a WebSocket with the rendered user-voice wavs,
                realistic inter-turn silence, and waits for each turn_end. This is the
                end-to-end claim; ASR errors show up here as the difference from text mode.

Judging is two-signal, and both are reported separately because neither alone is trusted:
  * keyword hit   -- at least one of `expect.any` appears in the reply (cheap, brittle);
  * LLM judge     -- the same Gemma, at temperature 0, shown the dialogue so far and asked
                     whether the reply correctly resolves the reference. A model judging its
                     own output is a known weak judge; it is reported as such, not as truth.

The verdict that matters is the MEMORY column: turns marked memory=true are unanswerable
without earlier turns, so a keyword hit there is direct evidence the history was used.

    .venv-bnweb/bin/python -m hervoice.eval.run_scenarios --mode text
    HV_GW_TOKEN=... .venv-bnweb/bin/python -m hervoice.eval.run_scenarios --mode audio
"""
import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
import urllib.request

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hervoice.svc import config as C                       # noqa: E402
from hervoice.svc import protocol as P                     # noqa: E402
from hervoice.svc.conversation import Conversation         # noqa: E402
from hervoice.svc.engine import ServiceEngine              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BN = re.compile(r"[ঀ-৿]")
# U+0964/U+0965 (danda, double danda) sit in the Devanagari block but ARE Bengali punctuation.
NON_BN_LETTER = re.compile(r"[一-鿿぀-ヿ가-힯ᄀ-ᇿЀ-ӿ؀-ۿ฀-๿ऀ-ॣ०-ॿ]")  # +Hangul, +Thai


def _cer(ref, hyp):
    ref, hyp = re.sub(r"[^\w\s]", "", ref), re.sub(r"[^\w\s]", "", hyp)
    ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(hyp, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
            prev = cur
    return d[len(hyp)] / len(ref)


def keyword_hit(reply, expect):
    return any(k in reply for k in expect.get("any", []))


REFERENT_ALIASES = {"ঢাকা": ["ঢাকা"], "জাতীয় পরিচয়পত্র": ["জাতীয় পরিচয়পত্র", "এনআইডি", "NID", "পরিচয়পত্র"],
                    "ই-পাসপোর্ট": ["পাসপোর্ট"], "নামজারি": ["নামজারি", "ভূমি"], "চট্টগ্রাম": ["চট্টগ্রাম"],
                    "রফিক": ["রফিক"], "শাপলা": ["শাপলা"]}


def resolved(reply, refers_to):
    """Did the reply name the thing the pronoun/ellipsis pointed at? This is the MEMORY
    signal proper: a wrong fact about the right referent still counts as resolved."""
    if not refers_to:
        return None
    return any(a in reply for a in REFERENT_ALIASES.get(refers_to, [refers_to]))


def script_ok(reply):
    """Bengali-script reply with no stray CJK/Cyrillic/Arabic/Devanagari letters -- the
    code-switch failure class Qwen produced ('কুব达রা')."""
    # numerator and denominator must count the SAME category: alphabetic letters only,
    # so the fraction is bounded by 1 (it previously exceeded 1 by counting Bengali marks).
    letters = [c for c in reply if c.isalpha()]
    if not letters:
        return False, 0.0
    bn = sum(1 for c in letters if "\u0980" <= c <= "\u09ff")
    frac = bn / len(letters)
    return (frac > 0.6 and not NON_BN_LETTER.search(reply)), round(frac, 2)


def llm_judge(dialogue, question, reply, refers_to):
    """Ask the brain to grade. Returns (verdict, reason). Reported as a weak signal."""
    hist = "\n".join(f"{'ব্যবহারকারী' if m['role']=='user' else 'সহকারী'}: {m['text']}" for m in dialogue)
    prompt = (
        "You are grading a Bengali voice assistant. Below is the conversation so far, then the "
        "user's latest question and the assistant's reply.\n\n"
        f"CONVERSATION SO FAR:\n{hist or '(none)'}\n\n"
        f"LATEST QUESTION: {question}\n"
        + (f"(This question refers back to: {refers_to})\n" if refers_to else "")
        + f"ASSISTANT REPLY: {reply}\n\n"
        "Does the reply correctly answer the latest question, resolving any pronoun or reference "
        "to the earlier conversation? Answer with exactly one word first, YES or NO, then one "
        "short sentence of reason in English."
    )
    body = json.dumps({"model": C.LLM_MODEL, "temperature": 0, "max_tokens": 60,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            f"{C.LLM_URL}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"}), timeout=60)
        txt = json.load(r)["choices"][0]["message"]["content"].strip()
    except Exception as e:                               # noqa: BLE001
        return "ERR", repr(e)
    v = "YES" if txt.upper().startswith("YES") else ("NO" if txt.upper().startswith("NO") else "?")
    return v, txt[:160]


# ------------------------------------------------------------------------------- text mode
def run_text(scen, out):
    eng = ServiceEngine()
    results = []
    for sc in scen["scenarios"]:
        conv = Conversation(system=C.SYSTEM_PROMPT, max_turns=C.MEM_MAX_TURNS, max_chars=C.MEM_MAX_CHARS)
        print(f"\n=== {sc['id']} ({sc['domain']}) ===")
        for i, turn in enumerate(sc["turns"]):
            dialogue_before = conv.snapshot()
            msgs = conv.messages(pending_user=turn["user"])
            t0 = time.time()
            gen, spoken = eng.respond(msgs, threading.Event(), lambda d: None, lambda i, s: None, lambda p, i: None)
            dt = (time.time() - t0) * 1000
            conv.add_user(turn["user"]); conv.add_assistant(spoken or gen)
            reply = (spoken or gen).strip()
            kw = keyword_hit(reply, turn["expect"]); sc_ok, frac = script_ok(reply)
            jv, jr = llm_judge(dialogue_before, turn["user"], reply, turn.get("refers_to"))
            rec = dict(scenario=sc["id"], turn=i, memory=turn["memory"], user=turn["user"], reply=reply,
                       keyword=kw, resolved=resolved(reply, turn.get("refers_to")), script_ok=sc_ok,
                       bn_frac=frac, judge=jv, judge_reason=jr,
                       turn_ms=round(dt, 1), history_turns=len(dialogue_before))
            results.append(rec)
            flag = "MEM" if turn["memory"] else "   "
            rs = rec["resolved"]; rtxt = "-" if rs is None else ("Y" if rs else "n")
            print(f"  t{i} {flag} resolved={rtxt} kw={'Y' if kw else 'n'} judge={jv:3s} "
                  f"script={'ok' if sc_ok else 'BAD'} {dt:5.0f}ms | U: {turn['user'][:38]}")
            print(f"        A: {reply[:110]}")
    return results


# ------------------------------------------------------------------------------ audio mode
async def run_audio(scen, out, token, pause_s=1.2):
    import websockets
    results = []
    _root = os.path.dirname(os.path.dirname(HERE))
    manifest = {}
    for m in json.load(open(os.path.join(HERE, "audio", "manifest.json"), encoding="utf-8")):
        if not os.path.isabs(m["path"]):
            m["path"] = os.path.join(_root, m["path"])
        manifest[(m["scenario"], m["turn"])] = m
    n = int(C.SR_IN * 0.02)

    for sc in scen["scenarios"]:
        print(f"\n=== {sc['id']} ({sc['domain']}) [audio] ===")
        ev_log = []
        # keyed by the server's turn number, so a premature split (which creates an extra
        # turn) cannot shift attribution onto the wrong scripted turn
        turns = {}   # turn -> {"asr":..., "reply":"", "first":..., "ended":bool, "epoch":..}
        state = {"turn_end": 0}
        async with websockets.connect(f"ws://127.0.0.1:{C.GW_PORT}/ws", max_size=None) as ws:
            await ws.send(json.dumps({"type": "hello", "token": token, "sample_rate": C.SR_IN}))
            ready = asyncio.Event()

            async def reader():
                async for m in ws:
                    if isinstance(m, (bytes, bytearray)):
                        _, ep, sq, _ = P.unpack_header(m)
                        await ws.send(json.dumps({"type": "played", "epoch": ep, "seq": sq}))
                        continue
                    if isinstance(m, str):
                        d = json.loads(m); ev_log.append(d); t = d.get("type")
                        if t == "ready": ready.set()
                        tn = d.get("turn")
                        if tn is not None:
                            rec = turns.setdefault(tn, {"asr": "", "reply": "", "first": None,
                                                        "first_from_speech_end": None, "ended": False,
                                                        "epoch": d.get("epoch")})
                        if t == "asr": rec["asr"] = d.get("text", "")
                        elif t == "text": rec["reply"] += d.get("delta", "")
                        elif t == "metrics":
                            if d.get("first_audio_ms"): rec["first"] = d["first_audio_ms"]
                            if d.get("first_audio_from_speech_end_ms"): rec["first_from_speech_end"] = d["first_audio_from_speech_end_ms"]
                        elif t == "turn_end":
                            rec["ended"] = True; state["turn_end"] += 1
            rt = asyncio.create_task(reader())
            await asyncio.wait_for(ready.wait(), timeout=30)   # never stream before the server is ready

            async def silence(sec):
                z = np.zeros(n, dtype="<f4").tobytes()
                for _ in range(int(sec * 50)):
                    await ws.send(z); await asyncio.sleep(0.02)

            dialogue = []
            seen_turns = set()
            for i, turn in enumerate(sc["turns"]):
                m = manifest.get((sc["id"], i))
                if not m:
                    print(f"  t{i}: no rendered audio, skipping"); break
                a, sr = sf.read(m["path"], dtype="float32")
                await silence(0.4)
                for k in range(0, len(a), n):
                    await ws.send(np.ascontiguousarray(a[k:k + n], dtype="<f4").tobytes())
                    await asyncio.sleep(0.02)
                want = i + 1
                deadline = time.time() + 60
                while state["turn_end"] < want and time.time() < deadline:
                    await silence(0.2)
                if state["turn_end"] < want:
                    print(f"  t{i}: TIMEOUT waiting for turn_end"); break
                # the scripted turn i is the LAST server turn that ended during this window;
                # if the server produced more than one turn for one scripted utterance, the
                # utterance was split -- record that explicitly instead of misattributing
                ended_turns = sorted(k for k, v in turns.items() if v["ended"])
                new_turns = [k for k in ended_turns if k > max(seen_turns, default=0)]
                seen_turns.update(new_turns)
                split = len(new_turns) > 1
                last = turns[new_turns[-1]] if new_turns else {"asr": "", "reply": "", "first": None, "first_from_speech_end": None}
                asr_txt = " | ".join(turns[k]["asr"] for k in new_turns) if split else last["asr"]
                reply = last["reply"].strip()
                first = last["first"]; first_se = last.get("first_from_speech_end")
                kw = keyword_hit(reply, turn["expect"]); sc_ok, frac = script_ok(reply)
                jv, jr = llm_judge(dialogue, turn["user"], reply, turn.get("refers_to"))
                cer = _cer(turn["user"], asr_txt)
                rec = dict(scenario=sc["id"], turn=i, memory=turn["memory"], user=turn["user"],
                           asr=asr_txt, asr_cer=round(cer, 3), reply=reply, keyword=kw,
                           split_into_turns=len(new_turns), first_audio_from_speech_end_ms=first_se,
                           resolved=resolved(reply, turn.get("refers_to")),
                           script_ok=sc_ok, bn_frac=frac, judge=jv, judge_reason=jr,
                           first_audio_ms=first, history_turns=len(dialogue))
                results.append(rec)
                dialogue += [{"role": "user", "text": asr_txt}, {"role": "assistant", "text": reply}]
                flag = "MEM" if turn["memory"] else "   "
                print(f"  t{i} {flag} asr_cer={cer:.2f} kw={'Y' if kw else 'n'} judge={jv:3s} "
                      f"first_audio={first or 0:5.0f}ms (from speech end {first_se or 0:5.0f}) "
                      f"{'SPLIT x'+str(len(new_turns)) if split else ''} | heard: {asr_txt[:34]}")
                print(f"        A: {reply[:110]}")
                await silence(pause_s)
            await ws.send(json.dumps({"type": "stop"})); await asyncio.sleep(0.3); rt.cancel()
    return results


def summarize(results, mode):
    mem = [r for r in results if r["memory"]]
    non = [r for r in results if not r["memory"]]
    def rate(rs, k): return f"{sum(1 for r in rs if r[k])}/{len(rs)}" if rs else "-"
    def jrate(rs): return f"{sum(1 for r in rs if r['judge']=='YES')}/{len(rs)}" if rs else "-"
    print(f"\n[{mode}] SUMMARY over {len(results)} turns in {len({r['scenario'] for r in results})} scenarios")
    res_turns = [r for r in mem if r.get("resolved") is not None]
    print(f"  memory-dependent turns : RESOLVED {rate(res_turns,'resolved'):>6}   <- the memory claim (referent named)")
    print(f"                           keyword  {rate(mem,'keyword'):>6}  judge {jrate(mem):>6}   (answer quality)")
    print(f"  independent turns      : keyword {rate(non,'keyword'):>6}  judge {jrate(non):>6}")
    print(f"  script clean (no code-switch chars): {rate(results,'script_ok')}")
    if mode == "audio":
        cers = [r["asr_cer"] for r in results]
        fa = [r["first_audio_ms"] for r in results if r.get("first_audio_ms")]
        fse = [r["first_audio_from_speech_end_ms"] for r in results if r.get("first_audio_from_speech_end_ms")]
        splits = sum(1 for r in results if r.get("split_into_turns", 1) > 1)
        if cers: print(f"  ASR CER on synthetic user voice: mean {np.mean(cers):.3f}, max {max(cers):.3f}")
        if fa: print(f"  first audio from ENDPOINT: median {np.median(fa):.0f} ms, p90 {np.percentile(fa,90):.0f} ms")
        if fse: print(f"  first audio from SPEECH END (includes silence wait): median {np.median(fse):.0f} ms, p90 {np.percentile(fse,90):.0f} ms  <- the honest number")
        print(f"  utterances split into >1 turn (premature endpoint): {splits}/{len(results)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["text", "audio"], default="text")
    ap.add_argument("--token", default=os.environ.get("HV_GW_TOKEN", ""))
    ap.add_argument("--out", default="runs/svc/scenarios")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scen = json.load(open(os.path.join(HERE, "scenarios_bn.json"), encoding="utf-8"))
    if a.mode == "text":
        res = run_text(scen, a.out)
    else:
        res = asyncio.run(run_audio(scen, a.out, a.token))
    summarize(res, a.mode)
    p = os.path.join(a.out, f"scenarios_{a.mode}.json")
    json.dump(res, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"[out] {p}")


if __name__ == "__main__":
    main()
