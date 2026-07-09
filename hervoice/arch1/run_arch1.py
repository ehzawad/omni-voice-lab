#!/usr/bin/env python3
"""Driver for ARCH 1: run the streaming cascade + baseline on each input wav, then write
results_arch1.json (per-turn dicts + a summary block) and runs/arch1/manifest.json, and copy the
best multi-sentence full wav to runs/arch1/demo_arch1.wav.

Run in .venv-funasr with the 3 workers up:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    .venv-funasr/bin/python -m hervoice.arch1.run_arch1
"""
import json
import os
import shutil
import statistics
import subprocess
import time

from hervoice.arch1 import stream_pipeline as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_TEXT = ("Printing, in the only sense with which we are at present concerned, differs from "
            "most, if not from all, the arts and crafts represented in the exhibition.")

TURNS = [
    {"name": "fifa", "wav": os.path.join(ROOT, "examples", "in_fifa_question.wav"),
     "note": "real repo wav; single-sentence answer"},
    {"name": "france", "wav": os.path.join(ROOT, "examples", "in_en_question.wav"),
     "note": "real repo wav; single-sentence answer"},
    {"name": "multi", "wav": os.path.join(ROOT, "examples", "in_multi_question.wav"),
     "note": "TTS-generated question ('Who was Isaac Newton, and what is he famous for?') to "
             "exercise a genuine >1-sentence answer and show the pipelining win"},
]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return None


def main():
    if not (sp._brain_health() and sp._health(sp.ASR_URL) and sp._health(sp.TTS_URL)):
        raise SystemExit("workers DOWN -- start start_workers.sh first")

    results = []
    for t in TURNS:
        out_prefix = os.path.join(ROOT, "runs", "arch1", f"turn_{t['name']}")
        r = sp.run_turn(t["wav"], out_prefix, ref_text=REF_TEXT, run_baseline=True)
        r["turn_name"] = t["name"]
        r["input_note"] = t["note"]
        results.append(r)
        imp = r.get("improvement", {})
        print(f"[{t['name']}] {r.get('status')}  n_sent={r.get('n_sentences')}  "
              f"stream_TTFA={r.get('measured_first_audio_s')}  "
              f"baseline_TTFA={r.get('baseline', {}).get('baseline_first_audio_s')}  "
              f"speedup={imp.get('first_audio_speedup_s')}s", flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    multi = [r for r in ok if r.get("n_sentences", 0) >= 2]

    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 3) if vals else None

    summary = {
        "n_turns": len(results),
        "n_ok": len(ok),
        "failure_states": [{"turn": r["turn_name"], "status": r["status"]}
                           for r in results if r.get("status") != "ok"],
        "median_stream_ttfa_s_all_ok": med([r.get("measured_first_audio_s") for r in ok]),
        "median_baseline_ttfa_s_all_ok": med([r["baseline"]["baseline_first_audio_s"]
                                              for r in ok if r.get("baseline")]),
        "median_speedup_s_all_ok": med([r.get("improvement", {}).get("first_audio_speedup_s")
                                        for r in ok]),
        "multi_sentence_turns": {
            "n": len(multi),
            "median_stream_ttfa_s": med([r.get("measured_first_audio_s") for r in multi]),
            "median_baseline_ttfa_s": med([r["baseline"]["baseline_first_audio_s"] for r in multi]),
            "median_speedup_s": med([r.get("improvement", {}).get("first_audio_speedup_s")
                                     for r in multi]),
        },
        "honest_caveats": [
            "The qwen_tts package does NOT stream audio packets; generate_voice_clone returns a "
            "FULL waveform. The measured first-audio win is STAGE PIPELINING (synthesize sentence 0 "
            "while the LLM generates sentence 1), not sub-sentence packet streaming.",
            "We do NOT claim the Qwen3-TTS paper's 97ms first-packet number; every latency here is "
            "measured in this repo on one RTX A5000.",
            "The pipelining win only appears when the answer has >1 sentence (a single-sentence "
            "answer has no intra-answer split point, so streaming TTFA ~= baseline TTFA).",
            "Total generation time is NOT reduced by pipelining (it can be marginally higher due to "
            "per-sentence overhead); only TIME-TO-FIRST-AUDIO drops.",
            "Baseline synthesizes the SAME streaming answer whole, so the speedup is attributable to "
            "pipelining and not to LLM sampling variance.",
            "Single-shot file input on a headless box: t0 is file-processing start, not a live mic "
            "voice-onset. English, single session.",
        ],
    }

    out = {"architecture": "ARCH 1 -- cascaded voice (STT -> LLM -> TTS), open local analog of "
                           "GPT-Live Standard Voice Mode",
           "git_commit": git_commit(),
           "gpu": "RTX A5000 (GPU0 only; CUDA_VISIBLE_DEVICES=0)",
           "models": {"asr": "Qwen/Qwen3-ASR-0.6B-hf",
                      "llm": "Qwen3.5-4B-Q4_K_M.gguf (llama-server, enable_thinking=False)",
                      "tts": "Qwen/Qwen3-TTS-12Hz-1.7B-Base (voice-clone, ref_female.wav)"},
           "ref_text": REF_TEXT,
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "summary": summary,
           "turns": results}

    res_path = os.path.join(ROOT, "hervoice", "arch1", "results_arch1.json")
    json.dump(out, open(res_path, "w"), indent=2)
    print("wrote", res_path)

    # Manifest (inputs, models, commit, per-turn metrics, caveats).
    manifest = {
        "architecture": out["architecture"],
        "git_commit": out["git_commit"],
        "gpu": out["gpu"],
        "models": out["models"],
        "ref_audio": os.path.join(ROOT, "examples", "ref_female.wav"),
        "ref_text": REF_TEXT,
        "inputs": [{"turn": t["name"], "wav": t["wav"], "note": t["note"]} for t in TURNS],
        "workers": {"asr": "127.0.0.1:8091", "llama_server": "127.0.0.1:8090",
                    "tts": "127.0.0.1:8092"},
        "per_turn": [{"turn": r["turn_name"], "status": r["status"],
                      "transcript": r.get("transcript"),
                      "answer": r.get("answer_text"),
                      "n_sentences": r.get("n_sentences"),
                      "asr_s": r.get("asr_s"), "llm_first_token_s": r.get("llm_first_token_s"),
                      "llm_total_s": r.get("llm_total_s"),
                      "stream_first_audio_s": r.get("measured_first_audio_s"),
                      "baseline_first_audio_s": (r.get("baseline") or {}).get("baseline_first_audio_s"),
                      "first_audio_speedup_s": (r.get("improvement") or {}).get("first_audio_speedup_s"),
                      "full_wav": r.get("full_wav")} for r in results],
        "summary": summary,
        "reproduce": ("bash hervoice/modular/start_workers.sh; then "
                      "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "
                      ".venv-funasr/bin/python -m hervoice.arch1.run_arch1"),
    }
    man_path = os.path.join(ROOT, "runs", "arch1", "manifest.json")
    json.dump(manifest, open(man_path, "w"), indent=2)
    print("wrote", man_path)

    # Demo wav: prefer a multi-sentence ok turn's full wav.
    demo_src = None
    for r in (multi or ok):
        fw = r.get("full_wav")
        if fw and os.path.isfile(fw):
            demo_src = fw
            break
    if demo_src:
        demo_dst = os.path.join(ROOT, "runs", "arch1", "demo_arch1.wav")
        shutil.copyfile(demo_src, demo_dst)
        print("demo wav:", demo_dst, "(from", demo_src + ")")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
