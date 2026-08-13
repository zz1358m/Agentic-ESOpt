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
ES_OPERATORS="${ES_OPERATORS:-m1,m2}"
ES_SIGMA_START="${ES_SIGMA_START:-${ES_SIGMA:-0.001}}"
ES_SIGMA_END="${ES_SIGMA_END:-0}"
ES_ALPHA="${ES_ALPHA:-0.0005}"
ES_SIGMA_SCHEDULE="${ES_SIGMA_SCHEDULE:-cosine}"
ES_SIGMA_WARMUP_STEPS="${ES_SIGMA_WARMUP_STEPS:-0}"
ES_SEED="${ES_SEED:-2024}"
ES_INVALID_REWARD_STRATEGY="${ES_INVALID_REWARD_STRATEGY:-current}"
AHD_EVALUATION_SEED="${AHD_EVALUATION_SEED:-1234}"
export AHD_EVALUATION_SEED
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUNNER="$ROOT/ahd-test-time/scripts/run_eoh_ahd.py"
ACO_TSP_VALIDATION_GATE="$ROOT/scripts/ahd/aco_tsp_validation_gate.py"

usage() {
  echo "usage: $0 {eoh|sample|agentic-esopt-eoh|agentic-esopt-sample}" >&2
  exit 2
}

case "$MODE" in
  eoh|sample|agentic-esopt-eoh|agentic-esopt-sample) ;;
  *) usage ;;
esac

if [ "$MODE" = "eoh" ] || [ "$MODE" = "agentic-esopt-eoh" ]; then
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
if [ "$MODE" = "agentic-esopt-sample" ] && [ $((BUDGET % BATCH_SIZE)) -ne 0 ]; then
  echo "agentic-esopt-sample requires BUDGET to be divisible by BATCH_SIZE" >&2
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
      agentic-esopt-eoh)
        ES_ENGINE_URLS="$ES_ENGINE_URLS" ES_SIGMA_START="$ES_SIGMA_START" \
        ES_SIGMA_END="$ES_SIGMA_END" ES_ALPHA="$ES_ALPHA" ES_SEED="$ES_SEED" \
        ES_OPERATORS="$ES_OPERATORS" \
        ES_SIGMA_SCHEDULE="$ES_SIGMA_SCHEDULE" ES_SIGMA_WARMUP_STEPS="$ES_SIGMA_WARMUP_STEPS" \
        ES_INVALID_REWARD_STRATEGY="$ES_INVALID_REWARD_STRATEGY" \
          "$PY" -u "$RUNNER" --task "$task" --split train --method es \
          --ec-pop-size "$POP_SIZE" --ec-generations "$GENERATIONS" \
          --ec-m1m2-multiplier "$EOH_K" --run-id "$run_id"
        ;;
      agentic-esopt-sample)
        ES_ENGINE_URLS="$ES_ENGINE_URLS" ES_SIGMA_START="$ES_SIGMA_START" \
        ES_SIGMA_END="$ES_SIGMA_END" ES_ALPHA="$ES_ALPHA" ES_SEED="$ES_SEED" \
        ES_OPERATORS="$ES_OPERATORS" \
        ES_SIGMA_SCHEDULE="$ES_SIGMA_SCHEDULE" ES_SIGMA_WARMUP_STEPS="$ES_SIGMA_WARMUP_STEPS" \
        ES_INVALID_REWARD_STRATEGY="$ES_INVALID_REWARD_STRATEGY" \
          "$PY" -u "$RUNNER" --task "$task" --split train --method sample_es \
          --sample-batch-size "$BATCH_SIZE" --sample-generations "$((BUDGET / BATCH_SIZE))" \
          --run-id "$run_id"
        ;;
    esac
    if [ "$MODE" = "agentic-esopt-sample" ] && [ "$task" = "aco_tsp" ]; then
      generation=$((BUDGET / BATCH_SIZE))
      mapfile -d '' run_roots < <(
        find "$ROOT/cache/active_runs" -mindepth 1 -maxdepth 1 -type d \
          -name "*_${run_id}" -print0
      )
      if [ "${#run_roots[@]}" -ne 1 ]; then
        echo "expected exactly one completed ACO-TSP run root for run_id=$run_id; found ${#run_roots[@]}" >&2
        exit 1
      fi
      population_json="${run_roots[0]}/results/pops_best/population_generation_${generation}.json"
      if [ ! -f "$population_json" ]; then
        echo "final ACO-TSP population artifact not found: $population_json" >&2
        exit 1
      fi
      validation_root="${run_roots[0]}/frozen_validation"
      "$PY" -u "$ACO_TSP_VALIDATION_GATE" \
        --population-json "$population_json" \
        --output "$validation_root/validation.json"
    fi
    echo "[done] mode=$MODE task=$task rep=$rep"
  done
done
