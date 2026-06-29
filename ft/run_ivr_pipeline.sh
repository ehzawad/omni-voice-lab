#!/bin/bash
# Full IndicVoices-R Bengali run: prep -> train -> gen -> eval. Runs on GPU1 only.
# Launched by the resource watcher once RAM+GPU are sufficient.
set -euo pipefail
cd /mnt/sdb/arafat/hervoice
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
PY=".venv-tts-lora/bin/python -u"

echo "[driver] $(date) prep IndicVoices-R subset"
$PY ft/prep_ivr_data.py 1500 30
PT=$(ls -S data/tts_bn/orpheus_ivr_train_*.pt | head -1)
echo "[driver] data=$PT"

echo "[driver] train LoRA (IVR, lr 5e-5)"
$PY ft/train_orpheus_lora.py "$PT" adapters/orpheus_bn_ivr_lora 600 4 5e-5

echo "[driver] generate held-out clips"
$PY ft/orpheus_baseline.py 20 ivrlora_f20 adapters/orpheus_bn_ivr_lora            # FLEURS-20, IVR-LoRA
TEXTS_JSON=data/tts_bn/ivr_test_texts.json $PY ft/orpheus_baseline.py 20 base_i20                          # IVR-20, base
TEXTS_JSON=data/tts_bn/ivr_test_texts.json $PY ft/orpheus_baseline.py 20 ivrlora_i20 adapters/orpheus_bn_ivr_lora  # IVR-20, IVR-LoRA

echo "[driver] eval (re-ASR CER/WER)"
for t in ivrlora_f20 base_i20 ivrlora_i20; do $PY ft/eval_bn_tts.py "$t" cuda; done

echo "[driver] $(date) DONE"
