#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_sample_all6_rep2_rep3_8gpu_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"
WAIT_PID="${WAIT_PID:-0}"

TASKS=(${TASKS:-construct_tsp construct_kp construct_asp aco_tsp aco_cvrp aco_bpp})
REPS=(${REPS:-2 3})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
PORTS=(${PORTS:-11513 11514 11515 11516 11517 11518 11519 11520})
SAMPLE_TOTAL="${SAMPLE_TOTAL:-1000}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-20}"
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"

mkdir -p "$LOGDIR"
cd "$ROOT"
echo "$$" > "$RUN_ROOT/queue.pid"

if [ "${#GPUS[@]}" -ne "${#PORTS[@]}" ]; then
  echo "GPUS and PORTS must have the same length." >&2
  exit 1
fi

join_by_comma() {
  local IFS=","
  echo "$*"
}

is_port_busy() {
  local port="$1"
  ss -ltn "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q .
}

wait_server_ready() {
  local port="$1" log="$2" waited=0
  while true; do
    if grep -q "Running on http://127.0.0.1:${port}" "$log" 2>/dev/null; then
      return 0
    fi
    if grep -Eq "Traceback|RuntimeError|ModuleNotFoundError|Address already in use" "$log" 2>/dev/null; then
      echo "server on port ${port} failed, see $log" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge 900 ]; then
      echo "timeout waiting for server on port ${port}" >&2
      return 1
    fi
  done
}

cleanup() {
  local pid
  for pid in ${RUN_PID:-}; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  for pid in ${SERVER_PIDS:-}; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  wait ${RUN_PID:-} >/dev/null 2>&1 || true
  wait ${SERVER_PIDS:-} >/dev/null 2>&1 || true
}

on_signal() {
  trap - EXIT
  cleanup
  exit 143
}
trap cleanup EXIT
trap on_signal INT TERM

write_progress() {
  local task="$1" rep="$2" run_id="$3" status="$4" rc="$5" result_path="$6"
  "$PY" - "$PROGRESS_JSONL" "$task" "$rep" "$run_id" "$status" "$rc" "$result_path" \
    "$SAMPLE_TOTAL" "$SAMPLE_BATCH_SIZE" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, task, rep, run_id, status, rc, result_path, total, batch = sys.argv[1:]
row = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "task": task,
    "method": "sample",
    "rep": int(rep),
    "run_id": run_id,
    "status": status,
    "exit_code": int(rc),
    "result_path": result_path,
    "sample_total": int(total),
    "sample_batch_size": int(batch),
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
PY
}

if [ "$WAIT_PID" -gt 0 ] 2>/dev/null; then
  echo "[waiting] pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" >/dev/null 2>&1; do
    sleep 30
  done
  echo "[wait-finished] pid=$WAIT_PID"
fi

SERVER_PIDS=""
for i in "${!PORTS[@]}"; do
  port="${PORTS[$i]}"
  gpu="${GPUS[$i]}"
  log="$LOGDIR/server_gpu${gpu}_port${port}.log"
  if is_port_busy "$port"; then
    echo "port ${port} already in use" >&2
    exit 1
  fi
  setsid "$PY" \
    "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
    --path "$MODEL" --d "$gpu" --port "$port" --host 127.0.0.1 \
    >"$log" 2>&1 < /dev/null &
  pid=$!
  SERVER_PIDS="$SERVER_PIDS $pid"
  echo "$pid" > "$LOGDIR/server_gpu${gpu}_port${port}.pid"
done
for i in "${!PORTS[@]}"; do
  wait_server_ready "${PORTS[$i]}" "$LOGDIR/server_gpu${GPUS[$i]}_port${PORTS[$i]}.log" || exit 1
done

URLS=()
for port in "${PORTS[@]}"; do
  URLS+=("http://127.0.0.1:${port}/completions")
done
URLS_CSV="$(join_by_comma "${URLS[@]}")"

echo "[servers-ready]"
echo "tasks=${TASKS[*]} reps=${REPS[*]} total=$SAMPLE_TOTAL batch=$SAMPLE_BATCH_SIZE"

FAILURES=0
TOTAL_RUNS=$(( ${#TASKS[@]} * ${#REPS[@]} ))
INDEX=0
GENERATION=$(( (SAMPLE_TOTAL + SAMPLE_BATCH_SIZE - 1) / SAMPLE_BATCH_SIZE ))
for rep in "${REPS[@]}"; do
  for task in "${TASKS[@]}"; do
    INDEX=$((INDEX + 1))
    run_id="${task}_sample_t${SAMPLE_TOTAL}_rep${rep}_${STAMP}"
    result="$ROOT/cache/active_runs/${task}_train_sample_t${SAMPLE_TOTAL}_${run_id}/results/pops_best/population_generation_${GENERATION}.json"
    log="$LOGDIR/${task}_sample_rep${rep}.log"
    echo "[${INDEX}/${TOTAL_RUNS} start] task=$task rep=$rep"
    write_progress "$task" "$rep" "$run_id" started 0 "$result"
    LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
    SAMPLE_TOTAL="$SAMPLE_TOTAL" \
    SAMPLE_BATCH_SIZE="$SAMPLE_BATCH_SIZE" \
      "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
        --task "$task" --split train --method sample --run-id "$run_id" \
        --llm-local-url "$URLS_CSV" >"$log" 2>&1 &
    RUN_PID=$!
    wait "$RUN_PID"
    rc=$?
    RUN_PID=""
    if [ "$rc" -eq 0 ]; then
      write_progress "$task" "$rep" "$run_id" completed "$rc" "$result"
    else
      write_progress "$task" "$rep" "$run_id" failed "$rc" "$result"
      FAILURES=$((FAILURES + 1))
    fi
    echo "[${INDEX}/${TOTAL_RUNS} done] task=$task rep=$rep rc=$rc"
  done
done

echo "[all-done] failures=$FAILURES run_root=$RUN_ROOT"
exit "$FAILURES"
