#!/usr/bin/env python3
"""Merge results_a.json + results_b.json into docs/COMPARISON.md. Computes ASR WER with jiwer.
Run in .venv-modular (which has jiwer)."""
import json, os, statistics as st
import jiwer
import bench_common as bc

ROOT = bc.ROOT
refs = {pid: ref for pid, _, ref in bc.PROMPTS}
A = json.load(open(os.path.join(ROOT, "results_a.json")))
B = json.load(open(os.path.join(ROOT, "results_b.json")))

norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                      jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
def wer(ref, hyp):
    if not hyp: return 1.0
    return round(jiwer.wer(norm(ref), norm(hyp)), 3)

def enrich(res):
    for r in res["rows"]:
        r["wer"] = wer(refs[r["id"]], r["asr_hyp"])
    return res
A, B = enrich(A), enrich(B)

def warm_mean(res, key):
    vals = [r[key] for r in res["rows"][1:] if r.get(key) is not None]  # drop p1 (warmup)
    return round(st.mean(vals), 2) if vals else None
def mean(res, key):
    vals = [r[key] for r in res["rows"] if r.get(key) is not None]
    return round(st.mean(vals), 3) if vals else None

def rows_table(res):
    out = ["| prompt | ASR WER | first-audio (s) | RTF | VRAM (GB) | answer |",
           "| --- | --- | --- | --- | --- | --- |"]
    for r in res["rows"]:
        out.append(f"| {r['id']} | {r['wer']} | {r.get('first_audio_s')} | {r['rtf']} | {r['vram_gb']} | {r['answer'][:60].replace(chr(10),' ')} |")
    return "\n".join(out)

md = f"""# Pipeline A vs Pipeline B: measured comparison

This is a real run, not a projection. Both pipelines were executed on the same fixed prompt set
({len(bc.PROMPTS)} spoken English questions in `examples/`), same reference voice
(`examples/ref_female.wav`), on the same GPU. Raw numbers are in `results_a.json` / `results_b.json`.

- Pipeline A: {A['pipeline']}
- Pipeline B: {B['pipeline']}

## Headline (means over prompts 2-4; prompt 1 excluded as cold-start warmup)

| metric | A (single-network) | B (modular) |
| --- | --- | --- |
| First-audio latency (s) | {warm_mean(A,'first_audio_s')} | {warm_mean(B,'first_audio_s')} |
| Real-time factor (RTF) | {warm_mean(A,'rtf')} | {warm_mean(B,'rtf')} |
| ASR WER (all prompts) | {mean(A,'wer')} | {mean(B,'wer')} |
| Peak VRAM (GB) | {max(r['vram_gb'] for r in A['rows'])} | {max(r['vram_gb'] for r in B['rows'])} |

## Pipeline A (single-network, MiniCPM-o 4.5, streaming)

{rows_table(A)}

## Pipeline B (modular, ASR + LLM + TTS, batch)

{rows_table(B)}

## Reading the results

- Latency: Pipeline A reaches first audio in about 2.3 s because it streams output as it
  generates. Pipeline B is much slower to first audio because this implementation is batch:
  it finishes ASR, then the full LLM answer, then synthesizes the entire utterance before any
  audio plays. A streaming modular pipeline (incremental TTS) would narrow this; it was not built.
- RTF above 1.0 on both means generation is slower than playback on this hardware, so long
  answers drift behind. Neither is yet a smooth real-time experience at this configuration.
- ASR: both transcribe these clean prompts essentially correctly.
- VRAM: Pipeline A is one ~14.8 GB model; Pipeline B here is smaller (~5.9 GB) only because its
  brain was downsized (see caveats).

## Caveats (important for fairness)

- Brain asymmetry: Pipeline A uses MiniCPM-o's Qwen3-8B brain; Pipeline B had to use
  Qwen2.5-3B-Instruct because the shared machine left only ~14.8 GB free and the 8B int4
  load-time spike OOMed. This favors A on answer quality and B on VRAM. Re-run B with Qwen3-8B
  on a free GPU for a clean brain-parity comparison (one-line change in pipeline_b_bench.py).
- B is batch, A is streaming. The first-audio gap is partly architecture and partly this
  implementation choice. A fair latency comparison needs a streaming TTS in B.
- Shared GPU: both ran on the A6000 with another user occupying most of both GPUs, so absolute
  timings include contention and are not best-case.
- TTS naturalness / MOS is NOT measured here; it requires human listening tests. Demo outputs
  are in examples/bench/ (out_a_*.wav, out_b_*.wav) for manual judgment.
- Prompt 1 is a cold-start outlier (first CUDA/compile pass) and is excluded from the means.

## Reproduce

```
# Pipeline B (modular)
.venv-modular/bin/python pipeline_b_bench.py
# Pipeline A (single-network)
.venv/bin/python pipeline_a_bench.py
# Merge
.venv-modular/bin/python make_comparison.py
```
"""
open(os.path.join(ROOT, "docs", "COMPARISON.md"), "w").write(md)
print("[wrote] docs/COMPARISON.md")
print(f"A warm first-audio={warm_mean(A,'first_audio_s')}s rtf={warm_mean(A,'rtf')} wer={mean(A,'wer')}")
print(f"B warm first-audio={warm_mean(B,'first_audio_s')}s rtf={warm_mean(B,'rtf')} wer={mean(B,'wer')}")
