#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BASE_STAMP="${BASE_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-$ROOT/runs/ahd_dynamic_eoh_construct_tsp_kp_3rep_split4_${BASE_STAMP}}"
STATUS="$QUEUE_ROOT/status.log"
DYNAMIC_STAMP="dynamic_k1_construct_tsp_kp_3rep_gpu0_3_${BASE_STAMP}"
EOH_STAMP="eoh_construct_tsp_kp_3rep_gpu4_7_${BASE_STAMP}"
DYNAMIC_ROOT="$ROOT/runs/ahd_es_${DYNAMIC_STAMP}"
EOH_ROOT="$ROOT/runs/ahd_eoh_llama31_8b_train_triplicate_${EOH_STAMP}"

mkdir -p "$QUEUE_ROOT"
echo "$$" > "$QUEUE_ROOT/queue.pid"

mark() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"
}

cleanup() {
  local pid
  for pid in ${DYNAMIC_PID:-} ${EOH_PID:-}; do
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

mark "queue_started total_runs=12 split=dynamic:gpu0-3,eoh:gpu4-7"
mark "dynamic_config method=Dynamic-EoH-k1 tasks=construct_tsp,construct_kp reps=1,2,3 population=10 generations=25"
mark "eoh_config method=EoH-k1 tasks=construct_tsp,construct_kp reps=1,2,3 population=10 generations=25"

env \
  ROOT="$ROOT" \
  STAMP="$DYNAMIC_STAMP" \
  RUN_ROOT="$DYNAMIC_ROOT" \
  TASKS="construct_tsp construct_kp" \
  REPS="1 2 3" \
  MODES="full" \
  GPUS="0 1 2 3" \
  PORTS="12113 12114 12115 12116" \
  ES_OPERATORS="m1,m2" \
  ES_DIRECTIONS="10" \
  EC_M1M2_MULTIPLIER="1" \
  ES_SIGMA_SCHEDULE="cosine" \
  ES_SIGMA_SCHEDULE_PLATEAU_FRACTION="0" \
  ES_INVALID_REWARD_STRATEGY="current" \
  FULL_SIGMAS="1e-3" \
  FULL_ALPHA_ES="5e-4" \
  ES_DISABLE_UPDATE="0" \
  LLM_LOCAL_TIMEOUT="600" \
    bash "$ROOT/scripts/ahd/run_es_reload_lora_full_tsp_kp_8gpu.sh" \
    >"$QUEUE_ROOT/dynamic.queue.log" 2>&1 &
DYNAMIC_PID=$!
echo "$DYNAMIC_PID" > "$QUEUE_ROOT/dynamic.pid"
mark "stage_started stage=dynamic pid=$DYNAMIC_PID run_root=$DYNAMIC_ROOT"

env \
  ROOT="$ROOT" \
  STAMP="$EOH_STAMP" \
  RUN_ROOT="$EOH_ROOT" \
  TASKS="construct_tsp construct_kp" \
  REPS="1 2 3" \
  GPUS="4 5 6 7" \
  PORTS="12117 12118 12119 12120" \
  EC_M1M2_MULTIPLIER="1" \
  LLM_TEMPERATURE="1.0" \
  LLM_TOP_P="1.0" \
  LLM_LOCAL_TIMEOUT="600" \
  EVA_TIMEOUT="600" \
    bash "$ROOT/scripts/ahd/run_all_eoh_llama31_8b_train_triplicate.sh" \
    >"$QUEUE_ROOT/eoh.queue.log" 2>&1 &
EOH_PID=$!
echo "$EOH_PID" > "$QUEUE_ROOT/eoh.pid"
mark "stage_started stage=eoh pid=$EOH_PID run_root=$EOH_ROOT"

wait "$DYNAMIC_PID"
DYNAMIC_RC=$?
mark "stage_finished stage=dynamic pid=$DYNAMIC_PID exit_code=$DYNAMIC_RC run_root=$DYNAMIC_ROOT"

wait "$EOH_PID"
EOH_RC=$?
mark "stage_finished stage=eoh pid=$EOH_PID exit_code=$EOH_RC run_root=$EOH_ROOT"

if [ "$DYNAMIC_RC" -eq 0 ] && [ "$EOH_RC" -eq 0 ]; then
  mark "queue_completed total_runs=12"
  exit 0
fi

mark "queue_failed dynamic_exit_code=$DYNAMIC_RC eoh_exit_code=$EOH_RC"
exit 1
