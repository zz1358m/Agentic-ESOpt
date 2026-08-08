#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
MODE="${1:-}"
TASKS=(${TASKS:-construct_tsp construct_kp construct_asp aco_tsp aco_cvrp aco_bpp})
REPS=(${REPS:-1 2 3})
BUDGET="${BUDGET:-2000}"
BATCH_SIZE="${BATCH_SIZE:-20}"
POP_SIZE="${AHD_POP_SIZE:-${EC_POP_SIZE:-10}}"
GENERATIONS="${AHD_GENERATIONS:-${EC_GENERATIONS:-25}}"
EOH_K="${EOH_K:-}"
LLM_LOCAL_URL="${LLM_LOCAL_URL:-http://127.0.0.1:11013/completions}"
ES_ENGINE_URLS="${ES_ENGINE_URLS:-http://127.0.0.1:11013/completions,http://127.0.0.1:11014/completions,http://127.0.0.1:11015/completions,http://127.0.0.1:11016/completions}"
ES_SIGMA_START="${ES_SIGMA_START:-${ES_SIGMA:-0.001}}"
ES_SIGMA_END="${ES_SIGMA_END:-0}"
ES_ALPHA="${ES_ALPHA:-0.0005}"
ES_SIGMA_SCHEDULE="${ES_SIGMA_SCHEDULE:-cosine}"
ES_SIGMA_WARMUP_STEPS="${ES_SIGMA_WARMUP_STEPS:-0}"
ES_SEED="${ES_SEED:-2024}"
ES_INVALID_REWARD_STRATEGY="${ES_INVALID_REWARD_STRATEGY:-current}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUNNER="$ROOT/ahd-test-time/scripts/run_eoh_ahd.py"

usage() {
  echo "usage: $0 {eoh|sample|dynamic-eoh|dynamic-sample}" >&2
  exit 2
}

case "$MODE" in
  eoh|sample|dynamic-eoh|dynamic-sample) ;;
  *) usage ;;
esac

if [ "$MODE" = "eoh" ] || [ "$MODE" = "dynamic-eoh" ]; then
  if [ -z "$EOH_K" ]; then
    base_budget=$((2 * POP_SIZE * GENERATIONS))
    if [ "$BUDGET" -le "$base_budget" ] || [ $((BUDGET % base_budget)) -ne 0 ]; then
      echo "cannot infer EOH_K: BUDGET must be an exact multiple of 2*POP_SIZE*GENERATIONS and produce k>0" >&2
      echo "set EOH_K explicitly for a non-standard EoH budget" >&2
      exit 2
    fi
    EOH_K=$((BUDGET / base_budget - 1))
  fi
fi

if [ ! -f "$RUNNER" ]; then
  echo "runner not found: $RUNNER" >&2
  exit 1
fi
if [ "$MODE" = "dynamic-sample" ] && [ $((BUDGET % BATCH_SIZE)) -ne 0 ]; then
  echo "dynamic-sample requires BUDGET to be divisible by BATCH_SIZE" >&2
  exit 2
fi

for task in "${TASKS[@]}"; do
  for rep in "${REPS[@]}"; do
    run_id="${task}_${MODE//-/_}_b${BUDGET}_rep${rep}_${STAMP}"
    echo "[start] mode=$MODE task=$task rep=$rep run_id=$run_id"
    case "$MODE" in
      eoh)
        LLM_LOCAL_URL="$LLM_LOCAL_URL" \
          "$PY" -u "$RUNNER" --task "$task" --split train --method eoh \
          --ec-pop-size "$POP_SIZE" --ec-generations "$GENERATIONS" \
          --ec-m1m2-multiplier "$EOH_K" --run-id "$run_id"
        ;;
      sample)
        LLM_LOCAL_URL="$LLM_LOCAL_URL" \
          "$PY" -u "$RUNNER" --task "$task" --split train --method sample \
          --sample-total "$BUDGET" --sample-batch-size "$BATCH_SIZE" --run-id "$run_id"
        ;;
      dynamic-eoh)
        ES_ENGINE_URLS="$ES_ENGINE_URLS" ES_SIGMA_START="$ES_SIGMA_START" \
        ES_SIGMA_END="$ES_SIGMA_END" ES_ALPHA="$ES_ALPHA" ES_SEED="$ES_SEED" \
        ES_SIGMA_SCHEDULE="$ES_SIGMA_SCHEDULE" ES_SIGMA_WARMUP_STEPS="$ES_SIGMA_WARMUP_STEPS" \
        ES_INVALID_REWARD_STRATEGY="$ES_INVALID_REWARD_STRATEGY" \
          "$PY" -u "$RUNNER" --task "$task" --split train --method es \
          --ec-pop-size "$POP_SIZE" --ec-generations "$GENERATIONS" \
          --ec-m1m2-multiplier "$EOH_K" --run-id "$run_id"
        ;;
      dynamic-sample)
        ES_ENGINE_URLS="$ES_ENGINE_URLS" ES_SIGMA_START="$ES_SIGMA_START" \
        ES_SIGMA_END="$ES_SIGMA_END" ES_ALPHA="$ES_ALPHA" ES_SEED="$ES_SEED" \
        ES_SIGMA_SCHEDULE="$ES_SIGMA_SCHEDULE" ES_SIGMA_WARMUP_STEPS="$ES_SIGMA_WARMUP_STEPS" \
        ES_INVALID_REWARD_STRATEGY="$ES_INVALID_REWARD_STRATEGY" \
          "$PY" -u "$RUNNER" --task "$task" --split train --method sample_es \
          --sample-batch-size "$BATCH_SIZE" --sample-generations "$((BUDGET / BATCH_SIZE))" \
          --run-id "$run_id"
        ;;
    esac
    echo "[done] mode=$MODE task=$task rep=$rep"
  done
done
