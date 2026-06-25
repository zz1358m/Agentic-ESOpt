#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
MODEL="${MODEL:-}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_eoh_llama31_8b_train_triplicate_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
SUMMARY_JSON="$RUN_ROOT/summary.json"
SKIP_SERVER_START="${SKIP_SERVER_START:-0}"

TASKS=(${TASKS:-construct_kp construct_asp aco_tsp aco_cvrp})
PORTS=(${PORTS:-11013 11014 11015 11016})
GPUS=(${GPUS:-0 1 2 3})
REPS=(${REPS:-1 2 3})
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"
EVA_TIMEOUT="${EVA_TIMEOUT:-600}"
SHARED_URLS="${SHARED_URLS:-http://127.0.0.1:${PORTS[0]}/completions,http://127.0.0.1:${PORTS[1]}/completions,http://127.0.0.1:${PORTS[2]}/completions,http://127.0.0.1:${PORTS[3]}/completions}"

mkdir -p "$LOGDIR"
cd "$ROOT"

if [ ! -x "$PY" ]; then
  echo "python not executable: $PY" >&2
  exit 1
fi

if [ -z "$MODEL" ] && [ "$SKIP_SERVER_START" != "1" ]; then
  echo "Set MODEL to a local HF model path or model id." >&2
  exit 2
fi

cleanup() {
  if [ -n "${RUNNER_PIDS:-}" ]; then
    for pid in $RUNNER_PIDS; do
      kill "$pid" >/dev/null 2>&1 || true
    done
  fi
  if [ -n "${SERVER_PIDS:-}" ]; then
    for pid in $SERVER_PIDS; do
      kill "$pid" >/dev/null 2>&1 || true
    done
  fi
}
trap cleanup EXIT INT TERM

is_port_busy() {
  local port="$1"
  ss -ltn "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q .
}

wait_server_ready() {
  local port="$1"
  local log="$2"
  local waited=0
  while true; do
    if grep -q "Running on http://127.0.0.1:${port}" "$log" 2>/dev/null; then
      return 0
    fi
    if grep -Eq "Traceback|Error|RuntimeError|ModuleNotFoundError" "$log" 2>/dev/null; then
      echo "server on port ${port} failed, see $log" >&2
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
    if [ "$waited" -ge 900 ]; then
      echo "timeout waiting for server on port ${port}" >&2
      return 1
    fi
  done
}

start_servers() {
  local i gpu port log pid
  SERVER_PIDS=""
  for i in "${!PORTS[@]}"; do
    gpu="${GPUS[$i]}"
    port="${PORTS[$i]}"
    log="$LOGDIR/server_gpu${gpu}_port${port}.log"
    if is_port_busy "$port"; then
      echo "port ${port} already in use; refusing to start over it" >&2
      exit 1
    fi
    setsid "$PY" "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
      --path "$MODEL" --d "$gpu" --port "$port" --host 127.0.0.1 \
      >"$log" 2>&1 < /dev/null &
    pid=$!
    SERVER_PIDS="${SERVER_PIDS} ${pid}"
    echo "$pid" > "$LOGDIR/server_gpu${gpu}_port${port}.pid"
  done

  for i in "${!PORTS[@]}"; do
    wait_server_ready "${PORTS[$i]}" "$LOGDIR/server_gpu${GPUS[$i]}_port${PORTS[$i]}.log"
  done
}

result_path_for() {
  local task="$1"
  local run_id="$2"
  echo "$ROOT/cache/active_runs/${task}_train_eoh_${run_id}/results/pops_best/population_generation_25.json"
}

queue_job() {
  local task="$1"
  local rep="$2"
  local run_id="${task}_train_eoh_rep${rep}_${STAMP}"
  local log="$LOGDIR/${task}_rep${rep}.log"
  local result_path pid
  result_path="$(result_path_for "$task" "$run_id")"

  echo "[launch] task=${task} rep=${rep} run_id=${run_id}"
  (
    cd "$ROOT"
    RUN_ID="$run_id" \
    LLM_LOCAL_URL="$SHARED_URLS" \
    LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
    EVA_TIMEOUT="$EVA_TIMEOUT" \
      "$PY" -u "$ROOT/ahd-test-time/scripts/run_eoh_ahd.py" \
      --task "$task" \
      --split train \
      --method eoh \
      --run-id "$run_id"
  ) >"$log" 2>&1 &
  pid=$!
  RUNNER_PIDS="${RUNNER_PIDS:-} ${pid}"
  ACTIVE_PIDS+=("$pid")
  ACTIVE_TASKS+=("$task")
  ACTIVE_REPS+=("$rep")
  ACTIVE_RUN_IDS+=("$run_id")
  ACTIVE_RESULTS+=("$result_path")
}

if [ "$SKIP_SERVER_START" != "1" ]; then
  start_servers
fi

declare -a ACTIVE_PIDS ACTIVE_TASKS ACTIVE_REPS ACTIVE_RUN_IDS ACTIVE_RESULTS

ACTIVE_PIDS=()
ACTIVE_TASKS=()
ACTIVE_REPS=()
ACTIVE_RUN_IDS=()
ACTIVE_RESULTS=()

for task in "${TASKS[@]}"; do
  for rep in "${REPS[@]}"; do
    run_id="${task}_train_eoh_rep${rep}_${STAMP}"
    log="$LOGDIR/${task}_rep${rep}.log"
    echo "[launch] task=${task} rep=${rep} run_id=${run_id}"
    (
      cd "$ROOT"
      RUN_ID="$run_id" \
      LLM_LOCAL_URL="$SHARED_URLS" \
      LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
      EVA_TIMEOUT="$EVA_TIMEOUT" \
        "$PY" -u "$ROOT/ahd-test-time/scripts/run_eoh_ahd.py" \
        --task "$task" \
        --split train \
        --method eoh \
        --run-id "$run_id"
    ) >"$log" 2>&1
    echo "[done] task=${task} rep=${rep} run_id=${run_id}"
  done
  echo "[task-finished] task=${task}"
done

"$PY" "$ROOT/scripts/ahd/summarize_eoh_triplicate.py" \
  --root "$ROOT" \
  --stamp "$STAMP" \
  --tasks "$(IFS=,; echo "${TASKS[*]}")" \
  --reps "$(IFS=,; echo "${REPS[*]}")" \
  --out "$SUMMARY_JSON"

echo "summary: $SUMMARY_JSON"
