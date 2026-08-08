#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
AFTER_PID="${AFTER_PID:-2945646}"
CHAIN_STAMP="${CHAIN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_chain_after_noiseonly_${CHAIN_STAMP}}"
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

if ps -p "$AFTER_PID" >/dev/null 2>&1; then
  echo "[wait] waiting for current experiment pid=${AFTER_PID}"
  while ps -p "$AFTER_PID" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[wait-done] pid=${AFTER_PID} exited"
else
  echo "[wait-skip] pid=${AFTER_PID} is not running"
fi

echo "[stage-1] ES full-scope ACO tasks, 3 reps"
env \
  ROOT="$ROOT" \
  PY="$PY" \
  MODEL="$MODEL" \
  STAMP="es_aco3_full_${CHAIN_STAMP}" \
  RUN_ROOT="$ROOT/runs/ahd_es_full_aco3_3rep_reload8_${CHAIN_STAMP}" \
  TASKS="aco_tsp aco_cvrp aco_bpp" \
  REPS="1 2 3" \
  MODES="full" \
  GPUS="$GPUS" \
  PORTS="$PORTS" \
  ES_OPERATORS="m1,m2" \
  FULL_SIGMAS="1e-3" \
  FULL_ALPHA_ES="5e-4" \
  ES_DISABLE_UPDATE="0" \
  LLM_LOCAL_TIMEOUT="600" \
    bash "$ROOT/scripts/ahd/run_es_reload_lora_full_tsp_kp_8gpu.sh" \
    >"$LOGDIR/es_aco3_full.log" 2>&1
echo "[stage-1-done]"

echo "[stage-2] EoH construct_tsp temperature sweep"
for temp in 0.6 1.5; do
  temp_label="${temp/./p}"
  for rep in 1 2 3; do
    echo "[eoh-temp] temp=${temp} rep=${rep}"
    env \
      ROOT="$ROOT" \
      PY="$PY" \
      MODEL="$MODEL" \
      STAMP="eoh_tsp_temp${temp_label}_rep${rep}_${CHAIN_STAMP}" \
      RUN_ROOT="$ROOT/runs/ahd_eoh_construct_tsp_temp${temp_label}_rep${rep}_${CHAIN_STAMP}" \
      TASKS="construct_tsp" \
      REPS="$rep" \
      GPUS="$GPUS" \
      PORTS="$PORTS" \
      SHARED_URLS="$SHARED_URLS" \
      LLM_TEMPERATURE="$temp" \
      LLM_TOP_P="0.98" \
      LLM_LOCAL_TIMEOUT="600" \
      EVA_TIMEOUT="600" \
        bash "$ROOT/scripts/ahd/run_all_eoh_llama31_8b_train_triplicate.sh" \
        >"$LOGDIR/eoh_construct_tsp_temp${temp_label}_rep${rep}.log" 2>&1
  done
done
echo "[stage-2-done]"

echo "[chain-done] run_root=$RUN_ROOT"
