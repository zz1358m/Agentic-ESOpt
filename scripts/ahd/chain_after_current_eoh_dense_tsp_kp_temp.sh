#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
AFTER_PID="${AFTER_PID:-3080635}"
CHAIN_STAMP="${CHAIN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_chain_after_current_eoh_dense_tsp_kp_temp_${CHAIN_STAMP}}"
LOGDIR="$RUN_ROOT/logs"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PORTS="${PORTS:-11313 11314 11315 11316 11317 11318 11319 11320}"
SHARED_URLS="${SHARED_URLS:-http://127.0.0.1:11313/completions,http://127.0.0.1:11314/completions,http://127.0.0.1:11315/completions,http://127.0.0.1:11316/completions,http://127.0.0.1:11317/completions,http://127.0.0.1:11318/completions,http://127.0.0.1:11319/completions,http://127.0.0.1:11320/completions}"

mkdir -p "$LOGDIR"
cd "$ROOT"

echo "chain_root=$RUN_ROOT"
echo "chain_stamp=$CHAIN_STAMP"
echo "after_pid=$AFTER_PID"
echo "gpus=$GPUS"
echo "ports=$PORTS"
echo "tsp_temps=0.8 1.2"
echo "kp_temps=0.6 0.8 1.2 1.5"
echo "reps=1 2 3"

if ps -p "$AFTER_PID" >/dev/null 2>&1; then
  echo "[wait] waiting for current chain pid=${AFTER_PID}"
  while ps -p "$AFTER_PID" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[wait-done] pid=${AFTER_PID} exited"
else
  echo "[wait-skip] pid=${AFTER_PID} is not running"
fi

wait_ports_free() {
  local waited=0
  while true; do
    local busy=0
    for port in $PORTS; do
      if ss -ltn "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q .; then
        busy=1
      fi
    done
    if [ "$busy" -eq 0 ]; then
      return 0
    fi
    if [ "$waited" -ge 600 ]; then
      echo "timeout waiting for ports to become free: $PORTS" >&2
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

run_temp_reps() {
  local task="$1"
  local temp="$2"
  local temp_label="${temp/./p}"
  local rep
  for rep in 1 2 3; do
    echo "[eoh-temp] task=${task} temp=${temp} rep=${rep}"
    wait_ports_free
    env \
      ROOT="$ROOT" \
      PY="$PY" \
      MODEL="$MODEL" \
      STAMP="${task}_temp${temp_label}_rep${rep}_${CHAIN_STAMP}" \
      RUN_ROOT="$ROOT/runs/ahd_eoh_${task}_temp${temp_label}_rep${rep}_${CHAIN_STAMP}" \
      TASKS="$task" \
      REPS="$rep" \
      GPUS="$GPUS" \
      PORTS="$PORTS" \
      SHARED_URLS="$SHARED_URLS" \
      LLM_TEMPERATURE="$temp" \
      LLM_TOP_P="0.98" \
      LLM_LOCAL_TIMEOUT="600" \
      EVA_TIMEOUT="600" \
        bash "$ROOT/scripts/ahd/run_all_eoh_llama31_8b_train_triplicate.sh" \
        >"$LOGDIR/${task}_temp${temp_label}_rep${rep}.log" 2>&1
  done
}

echo "[stage-1] EoH construct_tsp dense temperature add-ons"
for temp in 0.8 1.2; do
  run_temp_reps "construct_tsp" "$temp"
done
echo "[stage-1-done]"

echo "[stage-2] EoH construct_kp temperature sweep"
for temp in 0.6 0.8 1.2 1.5; do
  run_temp_reps "construct_kp" "$temp"
done
echo "[stage-2-done]"

echo "[chain-done] run_root=$RUN_ROOT"
