#!/usr/bin/env bash
# Start / stop / status for the HerVoice-BN services on a SHARED box.
#
#   hervoice/svc/run.sh start [llm|asr|tts|gw|all]
#   hervoice/svc/run.sh status
#   hervoice/svc/run.sh stop [llm|asr|tts|gw|all]
#
# SAFETY: this script NEVER uses pkill/pgrep pattern matching. On this box a `-f` pattern has
# repeatedly matched the issuing shell's own command line and killed it. Instead each service
# writes a PID file, and before sending any signal we verify that the pid still exists, is
# owned by us, has OUR repo as its cwd, and has the expected module in its cmdline. A pid that
# fails any check is treated as stale and left alone.
set -uo pipefail
R="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$R/runs/svc"; mkdir -p "$RUN"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VLLM_PY="$R/.venv-vllm/bin/python"
BN_PY="$R/.venv-bnweb/bin/python"

LLM_MODEL_ID="${HV_LLM_MODEL_ID:-google/gemma-4-E4B-it-qat-w4a16-ct}"
LLM_PORT="${HV_LLM_PORT:-8090}"
LLM_UTIL="${HV_LLM_GPU_UTIL:-0.65}"
# Prefer an ABSOLUTE KV-cache size. --gpu-memory-utilization is a fraction of TOTAL card memory
# and vLLM sizes the cache by profiling FREE memory at startup, so its footprint depends on
# what else is resident at that instant -- which is why start had to be sequential, and why it
# would break on Kubernetes (no ordered startup). vLLM itself printed this value for the
# footprint. NOTE vLLM printed 2641641472 (2.46 GiB) as the value that FITS the 0.65 request; the
# measured run actually USED 2.89 GiB, which is what preserves the 26x concurrency figure.
# Set HV_LLM_KV_CACHE_BYTES=0 to omit the absolute setting.
LLM_KV_BYTES="${HV_LLM_KV_CACHE_BYTES:-3103113871}"   # 2.89 GiB = what the measured run actually used
LLM_MAXLEN="${HV_LLM_MAX_MODEL_LEN:-2048}"

# identity check: pid alive AND ours AND cwd==repo AND cmdline mentions the marker
_owned() {  # _owned <pid> <marker>
  local pid="$1" marker="$2"
  [ -n "$pid" ] || return 1
  [ -d "/proc/$pid" ] || return 1
  [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$(id -un)" ] || return 1
  [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" = "$(readlink -f "$R")" ] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "$marker" || return 1
  return 0
}

_marker() { case "$1" in llm) echo "vllm.entrypoints";; asr) echo "hervoice.svc.asr_service";;
                         tts) echo "hervoice.svc.tts_service";; gw) echo "hervoice.svc.gateway";; esac; }
_port()   { case "$1" in llm) echo "$LLM_PORT";; asr) echo "${HV_ASR_PORT:-8001}";;
                         tts) echo "${HV_TTS_PORT:-8002}";; gw) echo "${HV_GW_PORT:-8100}";; esac; }

_pidfile() { echo "$RUN/$1.pid"; }

_start_one() {
  local svc="$1" pf; pf="$(_pidfile "$svc")"
  if [ -f "$pf" ] && _owned "$(cat "$pf" 2>/dev/null)" "$(_marker "$svc")"; then
    echo "  $svc already running (pid $(cat "$pf"))"; return 0
  fi
  cd "$R" || return 1
  case "$svc" in
    llm) nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
            --model "$LLM_MODEL_ID" --served-model-name "${HV_LLM_MODEL:-gemma4-e4b}" \
            --host 127.0.0.1 --port "$LLM_PORT" \
            --gpu-memory-utilization "$LLM_UTIL" \
            $( [ "$LLM_KV_BYTES" != "0" ] && echo "--kv-cache-memory $LLM_KV_BYTES" ) \
            --max-model-len "$LLM_MAXLEN" \
            --no-enable-log-requests >> "$RUN/llm.log" 2>&1 & ;;
    asr) nohup "$BN_PY" -m hervoice.svc.asr_service >> "$RUN/asr.log" 2>&1 & ;;
    tts) nohup "$BN_PY" -m hervoice.svc.tts_service >> "$RUN/tts.log" 2>&1 & ;;
    gw)  nohup "$BN_PY" -m hervoice.svc.gateway     >> "$RUN/gw.log"  2>&1 & ;;
    *)   echo "  unknown service: $svc"; return 1 ;;
  esac
  echo $! > "$pf"
  echo "  $svc started (pid $(cat "$pf"), port $(_port "$svc"))"
}

_wait_ready() {  # _wait_ready <svc> <timeout_s>
  local svc="$1" timeout="${2:-600}" port url t0 path
  port="$(_port "$svc")"
  path="/ready"; [ "$svc" = "llm" ] && path="/health"
  url="http://127.0.0.1:$port$path"
  t0=$(date +%s)
  while :; do
    if curl -s -m 3 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null | grep -q '^200$'; then
      # A 200 alone is not proof it is OURS: a failed bind plus somebody else's listener on
      # that port looks identical. Require our verified pid to still be alive too.
      if _owned "$(cat "$(_pidfile "$svc")" 2>/dev/null)" "$(_marker "$svc")"; then
        echo "  $svc READY after $(( $(date +%s) - t0 ))s"; return 0
      fi
      echo "  $svc: port $port answers 200 but our process is gone -- someone else holds it"
      return 1
    fi
    if ! _owned "$(cat "$(_pidfile "$svc")" 2>/dev/null)" "$(_marker "$svc")"; then
      echo "  $svc DIED during startup -- see $RUN/$svc.log"; return 1
    fi
    [ $(( $(date +%s) - t0 )) -ge "$timeout" ] && { echo "  $svc not ready after ${timeout}s"; return 1; }
    sleep 3
  done
}

_stop_one() {
  local svc="$1" pf pid; pf="$(_pidfile "$svc")"
  pid="$(cat "$pf" 2>/dev/null)"
  if ! _owned "$pid" "$(_marker "$svc")"; then
    echo "  $svc not running (no verified pid)"; rm -f "$pf"; return 0
  fi
  echo "  stopping $svc (pid $pid)"
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 30); do [ -d "/proc/$pid" ] || break; sleep 1; done
  if [ -d "/proc/$pid" ]; then
    echo "    still alive after 30s; SIGKILL"; kill -KILL "$pid" 2>/dev/null; sleep 2
  fi
  rm -f "$pf"
}

_status() {
  printf "  %-4s %-7s %-6s %s\n" SVC PID PORT STATE
  for svc in llm asr tts gw; do
    local pid state port; pid="$(cat "$(_pidfile "$svc")" 2>/dev/null)"; port="$(_port "$svc")"
    if _owned "$pid" "$(_marker "$svc")"; then
      local path="/ready"; [ "$svc" = "llm" ] && path="/health"
      if curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port$path" 2>/dev/null | grep -q '^200$'
        then state="ready"; else state="starting/unready"; fi
    else pid="-"; state="stopped"; fi
    printf "  %-4s %-7s %-6s %s\n" "$svc" "${pid:--}" "$port" "$state"
  done
  echo "  --- GPU (ours only; other users' processes are never touched) ---"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while read -r l; do
    p="${l%%,*}"
    if [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -un)" ] \
       && [ "$(readlink -f "/proc/$p/cwd" 2>/dev/null)" = "$(readlink -f "$R")" ]; then
      echo "    ours: pid=$p mem=${l#*,}"
    else
      echo "    other user: pid=$p mem=${l#*,}  (leave alone)"
    fi
  done
}

cmd="${1:-status}"; target="${2:-all}"
case "$cmd" in
  start)
    # STRICTLY SEQUENTIAL, start-then-verify, fail fast.
    # Two reasons this is not just tidiness:
    #  1. vLLM profiles free GPU memory at startup to size its KV cache, and that profiling
    #     assumes other processes are NOT changing their allocation while it runs. Loading
    #     ASR and TTS concurrently corrupts that measurement.
    #  2. Starting everything before checking anything means a failure surfaces only after
    #     three more processes have already taken GPU memory.
    [ "$target" = "all" ] && set -- llm asr tts gw || set -- "$target"
    for s in "$@"; do
      _start_one "$s" || { echo "  ABORT: $s failed to start"; exit 1; }
      _wait_ready "$s" "${HV_READY_TIMEOUT_S:-600}" || {
        echo "  ABORT: $s never became ready; stopping what we started"
        for t in "$@"; do _stop_one "$t"; done
        exit 1; }
    done
    _status ;;
  stop)
    [ "$target" = "all" ] && set -- gw tts asr llm || set -- "$target"
    for s in "$@"; do _stop_one "$s"; done ;;
  status) _status ;;
  *) echo "usage: $0 {start|stop|status} [llm|asr|tts|gw|all]"; exit 2 ;;
esac
