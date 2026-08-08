#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BASE_STAMP="${BASE_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
START_PAIR="${START_PAIR:-10}"
QUEUE_ROOT="${QUEUE_ROOT:-$ROOT/runs/ahd_dynamic_eoh_construct_continuous_pairs_by_task_split4_${BASE_STAMP}}"
STATUS="$QUEUE_ROOT/status.log"
STOP_FILE="$QUEUE_ROOT/STOP"

if ! [[ "$START_PAIR" =~ ^[1-9][0-9]*$ ]]; then
  echo "START_PAIR must be a positive integer: $START_PAIR" >&2
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

run_dynamic_attempt() {
  local task="$1" pair="$2" attempt="$3" gpus="$4" ports="$5"
  local stamp="continuous_dynamic_k1_${task}_pair${pair}_attempt${attempt}_${BASE_STAMP}"
  local run_root="$ROOT/runs/ahd_es_${stamp}"
  local log="$QUEUE_ROOT/${task}_pair${pair}_dynamic_attempt${attempt}.queue.log"
  local rc

  mark "stage_started lane=$task pair=$pair method=dynamic_k1 attempt=$attempt run_root=$run_root"
  env \
    ROOT="$ROOT" \
    STAMP="$stamp" \
    RUN_ROOT="$run_root" \
    TASKS="$task" \
    REPS="$pair" \
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
  mark "stage_finished lane=$task pair=$pair method=dynamic_k1 attempt=$attempt exit_code=$rc run_root=$run_root"
  return "$rc"
}

run_eoh_attempt() {
  local task="$1" pair="$2" attempt="$3" gpus="$4" ports="$5"
  local stamp="continuous_eoh_${task}_pair${pair}_attempt${attempt}_${BASE_STAMP}"
  local run_root="$ROOT/runs/ahd_eoh_llama31_8b_train_triplicate_${stamp}"
  local log="$QUEUE_ROOT/${task}_pair${pair}_eoh_attempt${attempt}.queue.log"
  local rc

  mark "stage_started lane=$task pair=$pair method=eoh_k1 attempt=$attempt run_root=$run_root"
  env \
    ROOT="$ROOT" \
    STAMP="$stamp" \
    RUN_ROOT="$run_root" \
    TASKS="$task" \
    REPS="$pair" \
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
  mark "stage_finished lane=$task pair=$pair method=eoh_k1 attempt=$attempt exit_code=$rc run_root=$run_root"
  return "$rc"
}

run_until_success() {
  local method="$1" task="$2" pair="$3" gpus="$4" ports="$5"
  local attempt=1 rc
  while [ ! -e "$STOP_FILE" ]; do
    if [ "$method" = "dynamic" ]; then
      run_dynamic_attempt "$task" "$pair" "$attempt" "$gpus" "$ports"
    else
      run_eoh_attempt "$task" "$pair" "$attempt" "$gpus" "$ports"
    fi
    rc=$?
    if [ "$rc" -eq 0 ]; then
      return 0
    fi
    mark "stage_retry lane=$task pair=$pair method=$method failed_attempt=$attempt retry_delay_seconds=60"
    attempt=$((attempt + 1))
    sleep 60
  done
  return 143
}

run_lane() {
  local task="$1" gpus="$2" ports="$3"
  local pair="$START_PAIR" rc
  ACTIVE_CHILD=""
  trap cleanup_lane INT TERM

  while [ ! -e "$STOP_FILE" ]; do
    mark "pair_started lane=$task pair=$pair"
    if [ $((pair % 2)) -eq 0 ]; then
      run_until_success "eoh" "$task" "$pair" "$gpus" "$ports"
      rc=$?
      [ "$rc" -eq 0 ] || break
      run_until_success "dynamic" "$task" "$pair" "$gpus" "$ports"
      rc=$?
      [ "$rc" -eq 0 ] || break
    else
      run_until_success "dynamic" "$task" "$pair" "$gpus" "$ports"
      rc=$?
      [ "$rc" -eq 0 ] || break
      run_until_success "eoh" "$task" "$pair" "$gpus" "$ports"
      rc=$?
      [ "$rc" -eq 0 ] || break
    fi
    mark "pair_finished lane=$task pair=$pair completed_runs=2"
    pair=$((pair + 1))
  done
  mark "lane_stopped lane=$task next_pair=$pair stop_file=$STOP_FILE"
}

cleanup() {
  local pid
  for pid in ${TSP_PID:-} ${KP_PID:-}; do
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

mark "queue_started mode=continuous start_pair=$START_PAIR split=tsp:gpu0-3,kp:gpu4-7 stop_file=$STOP_FILE"
mark "dynamic_config method=Dynamic-EoH-k1 population=10 generations=25 operators=m1,m2 directions=10 sigma=1e-3 sigma_schedule=cosine alpha=5e-4 invalid_reward=current"
mark "eoh_config method=EoH-k1 population=10 generations=25 temperature=1.0 top_p=1.0"
mark "pair_order even=eoh_then_dynamic odd=dynamic_then_eoh retry=until_success"

run_lane "construct_tsp" "0 1 2 3" "12313 12314 12315 12316" &
TSP_PID=$!
echo "$TSP_PID" > "$QUEUE_ROOT/tsp_lane.pid"
mark "lane_started lane=construct_tsp pid=$TSP_PID gpus=0,1,2,3"

run_lane "construct_kp" "4 5 6 7" "12317 12318 12319 12320" &
KP_PID=$!
echo "$KP_PID" > "$QUEUE_ROOT/kp_lane.pid"
mark "lane_started lane=construct_kp pid=$KP_PID gpus=4,5,6,7"

wait "$TSP_PID"
TSP_RC=$?
mark "lane_finished lane=construct_tsp pid=$TSP_PID exit_code=$TSP_RC"

wait "$KP_PID"
KP_RC=$?
mark "lane_finished lane=construct_kp pid=$KP_PID exit_code=$KP_RC"
mark "queue_stopped tsp_exit_code=$TSP_RC kp_exit_code=$KP_RC"
