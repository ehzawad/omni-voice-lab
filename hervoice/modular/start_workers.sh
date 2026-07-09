#!/usr/bin/env bash
# Start the three resident models for the warm modular pipeline, all on GPU0 (never GPU1):
#   brain   llama-server         port 8090
#   ASR     asr_worker.py        port 8091   (.venv-qwen-asr)
#   TTS     tts_worker.py        port 8092   (.venv-qwen-audio)
# Waits for every /health, then prints PIDs. See the STOP note at the bottom.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p runs/modular

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

ASR_MODEL="${ASR_MODEL:-qwen3-asr-0.6b}"
BIN="${LLAMA_BIN:-/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/llama.cpp/build/bin/llama-server}"
GGUF="${LLAMA_GGUF:-/mnt/sdb/arafat/llm-stuff/qwen35-gguf-bench/models/q4b/Qwen3.5-4B-Q4_K_M.gguf}"

wait_health () {  # url name maxtries
  local url="$1" name="$2" tries="${3:-120}"
  for _ in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      # for workers, require loaded=true (not just listening)
      if curl -s "$url" 2>/dev/null | grep -q '"loaded": *true' \
         || curl -s "$url" 2>/dev/null | grep -q '"status": *"ok"'; then
        echo "[start_workers] $name healthy: $url"; return 0
      fi
    fi
    sleep 2
  done
  echo "[start_workers] ERROR: $name never became healthy at $url" >&2
  return 1
}

echo "[start_workers] brain (llama-server) :8090 ..."
"$BIN" -m "$GGUF" -ngl 99 --host 127.0.0.1 --port 8090 -c 4096 --no-warmup \
  > runs/modular/llama_server.log 2>&1 &
BRAIN_PID=$!

echo "[start_workers] ASR worker ($ASR_MODEL) :8091 ..."
.venv-qwen-asr/bin/python -m hervoice.modular.asr_worker --model "$ASR_MODEL" --port 8091 \
  > runs/modular/asr_worker.log 2>&1 &
ASR_PID=$!

echo "[start_workers] TTS worker :8092 ..."
.venv-qwen-audio/bin/python -m hervoice.modular.tts_worker --port 8092 \
  > runs/modular/tts_worker.log 2>&1 &
TTS_PID=$!

wait_health "http://127.0.0.1:8090/health" brain 120
wait_health "http://127.0.0.1:8091/health" asr   180
wait_health "http://127.0.0.1:8092/health" tts   180

echo
echo "[start_workers] ALL READY (GPU0 only)."
echo "  brain (llama-server) PID=$BRAIN_PID  :8090"
echo "  asr_worker           PID=$ASR_PID  :8091"
echo "  tts_worker           PID=$TTS_PID  :8092"
echo "$BRAIN_PID $ASR_PID $TTS_PID" > runs/modular/worker_pids.txt
echo
echo "[start_workers] STOP them with:  kill $BRAIN_PID $ASR_PID $TTS_PID"
echo "                or:              kill \$(cat runs/modular/worker_pids.txt)"
