#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BASE_STAMP="${BASE_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
ROUNDS="${ROUNDS:-3}"
QUEUE_ROOT="${QUEUE_ROOT:-$ROOT/runs/ahd_dynamic_eoh_construct_triplicate_rounds_by_task_split4_${BASE_STAMP}}"
STATUS="$QUEUE_ROOT/status.log"

if ! [[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROUNDS must be a positive integer: $ROUNDS" >&2
  exit 2
fi

mkdir -p "$QUEUE_ROOT"
echo "$$" > "$QUEUE_ROOT/queue.pid"

mark() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"
}

cleanup_lane() {
  if [ -n "${ACTIVE_CHILD:-}" ]; then
    kill -TERM "$ACTIVE_CHILD" >/dev/null 2>&1 || true
    wait "$ACTIVE_CHILD" >/dev/null 2>&1 || true
  fi
  exit 143
}

run_dynamic() {
  local task="$1" round="$2" gpus="$3" ports="$4"
  local stamp="dynamic_k1_${task}_3rep_round${round}_${BASE_STAMP}"
  local run_root="$ROOT/runs/ahd_es_${stamp}"
  local log="$QUEUE_ROOT/${task}_round${round}_dynamic.queue.log"
  local rc

  mark "stage_started lane=$task round=$round method=dynamic_k1 run_root=$run_root"
  env \
    ROOT="$ROOT" \
    STAMP="$stamp" \
    RUN_ROOT="$run_root" \
    TASKS="$task" \
    REPS="1 2 3" \
    MODES="full" \
    GPUS="$gpus" \
    PORTS="$ports" \
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
      >"$log" 2>&1 &
  ACTIVE_CHILD=$!
  wait "$ACTIVE_CHILD"
  rc=$?
  ACTIVE_CHILD=""
  mark "stage_finished lane=$task round=$round method=dynamic_k1 exit_code=$rc run_root=$run_root"
  return "$rc"
}

run_eoh() {
  local task="$1" round="$2" gpus="$3" ports="$4"
  local stamp="eoh_${task}_3rep_round${round}_${BASE_STAMP}"
  local run_root="$ROOT/runs/ahd_eoh_llama31_8b_train_triplicate_${stamp}"
  local log="$QUEUE_ROOT/${task}_round${round}_eoh.queue.log"
  local rc

  mark "stage_started lane=$task round=$round method=eoh_k1 run_root=$run_root"
  env \
    ROOT="$ROOT" \
    STAMP="$stamp" \
    RUN_ROOT="$run_root" \
    TASKS="$task" \
    REPS="1 2 3" \
    GPUS="$gpus" \
    PORTS="$ports" \
    EC_M1M2_MULTIPLIER="1" \
    LLM_TEMPERATURE="1.0" \
    LLM_TOP_P="1.0" \
    LLM_LOCAL_TIMEOUT="600" \
    EVA_TIMEOUT="600" \
      bash "$ROOT/scripts/ahd/run_all_eoh_llama31_8b_train_triplicate.sh" \
      >"$log" 2>&1 &
  ACTIVE_CHILD=$!
  wait "$ACTIVE_CHILD"
  rc=$?
  ACTIVE_CHILD=""
  mark "stage_finished lane=$task round=$round method=eoh_k1 exit_code=$rc run_root=$run_root"
  return "$rc"
}

run_lane() {
  local task="$1" gpus="$2" ports="$3"
  local round rc
  ACTIVE_CHILD=""
  trap cleanup_lane INT TERM

  for round in $(seq 1 "$ROUNDS"); do
    mark "round_started lane=$task round=$round"
    if [ $((round % 2)) -eq 1 ]; then
      run_dynamic "$task" "$round" "$gpus" "$ports"
      rc=$?
      [ "$rc" -eq 0 ] || return "$rc"
      run_eoh "$task" "$round" "$gpus" "$ports"
      rc=$?
      [ "$rc" -eq 0 ] || return "$rc"
    else
      run_eoh "$task" "$round" "$gpus" "$ports"
      rc=$?
      [ "$rc" -eq 0 ] || return "$rc"
      run_dynamic "$task" "$round" "$gpus" "$ports"
      rc=$?
      [ "$rc" -eq 0 ] || return "$rc"
    fi
    mark "round_finished lane=$task round=$round"
  done
}

cleanup() {
  local pid
  for pid in ${TSP_PID:-} ${KP_PID:-}; do
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

TOTAL_RUNS=$((ROUNDS * 12))
mark "queue_started rounds=$ROUNDS total_runs=$TOTAL_RUNS split=tsp:gpu0-3,kp:gpu4-7"
mark "dynamic_config method=Dynamic-EoH-k1 population=10 generations=25 operators=m1,m2 directions=10 sigma=1e-3 sigma_schedule=cosine alpha=5e-4 invalid_reward=current"
mark "eoh_config method=EoH-k1 population=10 generations=25 temperature=1.0 top_p=1.0"
mark "round_order odd=dynamic_then_eoh even=eoh_then_dynamic"

run_lane "construct_tsp" "0 1 2 3" "12213 12214 12215 12216" &
TSP_PID=$!
echo "$TSP_PID" > "$QUEUE_ROOT/tsp_lane.pid"
mark "lane_started lane=construct_tsp pid=$TSP_PID gpus=0,1,2,3"

run_lane "construct_kp" "4 5 6 7" "12217 12218 12219 12220" &
KP_PID=$!
echo "$KP_PID" > "$QUEUE_ROOT/kp_lane.pid"
mark "lane_started lane=construct_kp pid=$KP_PID gpus=4,5,6,7"

wait "$TSP_PID"
TSP_RC=$?
mark "lane_finished lane=construct_tsp pid=$TSP_PID exit_code=$TSP_RC"

wait "$KP_PID"
KP_RC=$?
mark "lane_finished lane=construct_kp pid=$KP_PID exit_code=$KP_RC"

if [ "$TSP_RC" -eq 0 ] && [ "$KP_RC" -eq 0 ]; then
  mark "queue_completed rounds=$ROUNDS total_runs=$TOTAL_RUNS"
  exit 0
fi

mark "queue_failed tsp_exit_code=$TSP_RC kp_exit_code=$KP_RC"
exit 1
