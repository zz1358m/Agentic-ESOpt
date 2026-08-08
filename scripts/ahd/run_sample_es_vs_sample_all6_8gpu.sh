#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_sample_es_vs_sample_all6_8gpu_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"

ES_TASKS=(${ES_TASKS:-construct_tsp construct_kp construct_asp aco_tsp aco_cvrp aco_bpp})
SAMPLE_TASKS=(${SAMPLE_TASKS:-construct_tsp construct_kp construct_asp aco_tsp aco_cvrp aco_bpp})
ES_GPUS=(${ES_GPUS:-0 1 2 3})
SAMPLE_GPUS=(${SAMPLE_GPUS:-4 5 6 7})
ES_PORTS=(${ES_PORTS:-11413 11414 11415 11416})
SAMPLE_PORTS=(${SAMPLE_PORTS:-11417 11418 11419 11420})
SAMPLE_TOTAL="${SAMPLE_TOTAL:-1000}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-20}"
SAMPLE_GENERATIONS="${SAMPLE_GENERATIONS:-50}"
ES_SIGMA="${ES_SIGMA:-1e-3}"
ES_ALPHA="${ES_ALPHA:-5e-4}"
ES_PARAMETER_SCOPE="${ES_PARAMETER_SCOPE:-full}"
ES_SIGMA_SCHEDULE="${ES_SIGMA_SCHEDULE:-constant}"
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"

mkdir -p "$LOGDIR"
cd "$ROOT"
echo "$$" > "$RUN_ROOT/queue.pid"

if [ "${#ES_GPUS[@]}" -ne "${#ES_PORTS[@]}" ] || \
   [ "${#SAMPLE_GPUS[@]}" -ne "${#SAMPLE_PORTS[@]}" ]; then
  echo "GPU and port counts must match for both groups." >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "python not executable: $PY" >&2
  exit 1
fi
if [ ! -d "$MODEL" ]; then
  echo "model path not found: $MODEL" >&2
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
  local port="$1"
  local log="$2"
  local waited=0
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
  for pid in ${QUEUE_PIDS:-}; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  for pid in ${SERVER_PIDS:-}; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  wait ${QUEUE_PIDS:-} >/dev/null 2>&1 || true
  wait ${SERVER_PIDS:-} >/dev/null 2>&1 || true
}
on_signal() {
  trap - EXIT
  cleanup
  exit 143
}
trap cleanup EXIT
trap on_signal INT TERM

start_server_group() {
  local group="$1"
  shift
  local -a gpus=("$@")
  local -a ports
  if [ "$group" = "es" ]; then
    ports=("${ES_PORTS[@]}")
  else
    ports=("${SAMPLE_PORTS[@]}")
  fi

  local i gpu port log pid
  for i in "${!gpus[@]}"; do
    gpu="${gpus[$i]}"
    port="${ports[$i]}"
    log="$LOGDIR/server_${group}_gpu${gpu}_port${port}.log"
    if is_port_busy "$port"; then
      echo "port ${port} already in use" >&2
      exit 1
    fi
    setsid "$PY" \
      "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
      --path "$MODEL" --d "$gpu" --port "$port" --host 127.0.0.1 \
      >"$log" 2>&1 < /dev/null &
    pid=$!
    SERVER_PIDS="${SERVER_PIDS:-} ${pid}"
    echo "$pid" > "$LOGDIR/server_${group}_gpu${gpu}_port${port}.pid"
  done

  for i in "${!gpus[@]}"; do
    wait_server_ready "${ports[$i]}" "$LOGDIR/server_${group}_gpu${gpus[$i]}_port${ports[$i]}.log" || exit 1
  done
}

write_progress() {
  local task="$1"
  local method="$2"
  local run_id="$3"
  local status="$4"
  local rc="$5"
  local result_path="$6"
  "$PY" - "$PROGRESS_JSONL" "$task" "$method" "$run_id" "$status" "$rc" "$result_path" \
    "$SAMPLE_TOTAL" "$SAMPLE_BATCH_SIZE" "$SAMPLE_GENERATIONS" "$ES_SIGMA" "$ES_ALPHA" <<'PY'
import json
import sys
from datetime import datetime, timezone

(path, task, method, run_id, status, rc, result_path, total, batch, generations, sigma, alpha) = sys.argv[1:]
row = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "task": task,
    "method": method,
    "run_id": run_id,
    "status": status,
    "exit_code": int(rc),
    "result_path": result_path,
    "sample_total": int(total),
    "sample_batch_size": int(batch),
    "sample_generations": int(generations),
}
if method == "sample_es":
    row.update({"sigma": sigma, "alpha": alpha, "reward_normalization": "zscore"})
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
PY
}

reset_es_engines() {
  local port
  for port in "${ES_PORTS[@]}"; do
    curl --connect-timeout 5 --max-time 600 -fsS \
      -X POST -H 'Content-Type: application/json' -d '{}' \
      "http://127.0.0.1:${port}/es/reset" >/dev/null 2>&1 || true
  done
}

run_es() {
  local task="$1"
  local run_id="$2"
  local result_path="$3"
  local log="$LOGDIR/${task}_sample_es.log"
  write_progress "$task" sample_es "$run_id" started 0 "$result_path"
  ES_ENGINE_URLS="$ES_URLS_CSV" \
  ES_SIGMA="$ES_SIGMA" \
  ES_ALPHA="$ES_ALPHA" \
  ES_PARAMETER_SCOPE="$ES_PARAMETER_SCOPE" \
  ES_SIGMA_SCHEDULE="$ES_SIGMA_SCHEDULE" \
  SAMPLE_BATCH_SIZE="$SAMPLE_BATCH_SIZE" \
  SAMPLE_GENERATIONS="$SAMPLE_GENERATIONS" \
  LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
    "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
      --task "$task" --split train --method sample_es --run-id "$run_id" \
      >"$log" 2>&1
}

run_sample() {
  local task="$1"
  local run_id="$2"
  local result_path="$3"
  local log="$LOGDIR/${task}_sample.log"
  write_progress "$task" sample "$run_id" started 0 "$result_path"
  LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
  SAMPLE_TOTAL="$SAMPLE_TOTAL" \
  SAMPLE_BATCH_SIZE="$SAMPLE_BATCH_SIZE" \
    "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
      --task "$task" --split train --method sample --run-id "$run_id" \
      --llm-local-url "$SAMPLE_URLS_CSV" \
      >"$log" 2>&1
}

ES_URLS=()
for port in "${ES_PORTS[@]}"; do
  ES_URLS+=("http://127.0.0.1:${port}/completions")
done
ES_URLS_CSV="$(join_by_comma "${ES_URLS[@]}")"

SAMPLE_URLS=()
for port in "${SAMPLE_PORTS[@]}"; do
  SAMPLE_URLS+=("http://127.0.0.1:${port}/completions")
done
SAMPLE_URLS_CSV="$(join_by_comma "${SAMPLE_URLS[@]}")"

echo "run_root=$RUN_ROOT"
echo "es_tasks=${ES_TASKS[*]}"
echo "sample_tasks=${SAMPLE_TASKS[*]}"
echo "es_gpus=${ES_GPUS[*]} es_ports=${ES_PORTS[*]}"
echo "sample_gpus=${SAMPLE_GPUS[*]} sample_ports=${SAMPLE_PORTS[*]}"
echo "sample_total=$SAMPLE_TOTAL batch_size=$SAMPLE_BATCH_SIZE generations=$SAMPLE_GENERATIONS"
echo "es_sigma=$ES_SIGMA es_alpha=$ES_ALPHA es_scope=$ES_PARAMETER_SCOPE normalization=zscore"

SERVER_PIDS=""
start_server_group es "${ES_GPUS[@]}"
start_server_group sample "${SAMPLE_GPUS[@]}"
echo "[servers-ready]"

run_es_queue() {
  local failures=0 index=0 task run_id result rc
  for task in "${ES_TASKS[@]}"; do
    index=$((index + 1))
    run_id="${task}_sample_es_dynamic_invalid_pop${SAMPLE_BATCH_SIZE}_gen${SAMPLE_GENERATIONS}_rep1_${STAMP}"
    result="$ROOT/cache/active_runs/${task}_train_sample_es_pop${SAMPLE_BATCH_SIZE}_gen${SAMPLE_GENERATIONS}_sigma0.001_alpha0.0005_${run_id}/results/pops_best/population_generation_${SAMPLE_GENERATIONS}.json"
    echo "[es ${index}/${#ES_TASKS[@]} start] task=$task"
    run_es "$task" "$run_id" "$result"
    rc=$?
    reset_es_engines
    if [ "$rc" -eq 0 ]; then
      write_progress "$task" sample_es "$run_id" completed "$rc" "$result"
    else
      write_progress "$task" sample_es "$run_id" failed "$rc" "$result"
      failures=$((failures + 1))
    fi
    echo "[es ${index}/${#ES_TASKS[@]} done] task=$task rc=$rc"
  done
  return "$failures"
}

run_sample_queue() {
  local failures=0 index=0 task run_id result rc
  local generation=$(( (SAMPLE_TOTAL + SAMPLE_BATCH_SIZE - 1) / SAMPLE_BATCH_SIZE ))
  for task in "${SAMPLE_TASKS[@]}"; do
    index=$((index + 1))
    run_id="${task}_sample_t${SAMPLE_TOTAL}_rep1_${STAMP}"
    result="$ROOT/cache/active_runs/${task}_train_sample_t${SAMPLE_TOTAL}_${run_id}/results/pops_best/population_generation_${generation}.json"
    echo "[sample ${index}/${#SAMPLE_TASKS[@]} start] task=$task"
    run_sample "$task" "$run_id" "$result"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      write_progress "$task" sample "$run_id" completed "$rc" "$result"
    else
      write_progress "$task" sample "$run_id" failed "$rc" "$result"
      failures=$((failures + 1))
    fi
    echo "[sample ${index}/${#SAMPLE_TASKS[@]} done] task=$task rc=$rc"
  done
  return "$failures"
}

run_es_queue &
ES_QUEUE_PID=$!
run_sample_queue &
SAMPLE_QUEUE_PID=$!
QUEUE_PIDS="$ES_QUEUE_PID $SAMPLE_QUEUE_PID"

wait "$ES_QUEUE_PID"
ES_FAILURES=$?
wait "$SAMPLE_QUEUE_PID"
SAMPLE_FAILURES=$?
QUEUE_PIDS=""
TOTAL_FAILURES=$((ES_FAILURES + SAMPLE_FAILURES))
echo "[all-done] failures=$TOTAL_FAILURES run_root=$RUN_ROOT"
exit "$TOTAL_FAILURES"
