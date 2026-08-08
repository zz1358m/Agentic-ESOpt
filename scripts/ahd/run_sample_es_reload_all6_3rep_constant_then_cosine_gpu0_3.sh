#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_sample_es_reload_all6_3rep_constant_then_cosine_gpu0_3_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"

TASKS=(${TASKS:-construct_tsp construct_kp construct_asp aco_tsp aco_cvrp aco_bpp})
REPS=(${REPS:-1 2 3})
SCHEDULES=(${SCHEDULES:-constant cosine})
STRATEGIES=(${STRATEGIES:-current})
GPUS=(${GPUS:-0 1 2 3})
PORTS=(${PORTS:-11413 11414 11415 11416})
POPULATION="${POPULATION:-20}"
GENERATIONS="${GENERATIONS:-50}"
ES_SIGMA="${ES_SIGMA:-1e-3}"
ES_ALPHA="${ES_ALPHA:-5e-4}"
ES_PARAMETER_SCOPE="${ES_PARAMETER_SCOPE:-full}"
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"

mkdir -p "$LOGDIR"
cd "$ROOT"
echo "$$" > "$RUN_ROOT/queue.pid"

if [ "${#GPUS[@]}" -ne "${#PORTS[@]}" ]; then echo "GPU/port count mismatch" >&2; exit 1; fi
if [ ! -x "$PY" ] || [ ! -d "$MODEL" ]; then echo "missing python or model" >&2; exit 1; fi

join_by_comma() { local IFS=","; echo "$*"; }
is_port_busy() { ss -ltn "( sport = :${1} )" 2>/dev/null | tail -n +2 | grep -q .; }

wait_server_ready() {
  local port="$1" log="$2" waited=0
  while true; do
    grep -q "Running on http://127.0.0.1:${port}" "$log" 2>/dev/null && return 0
    if grep -Eq "Traceback|RuntimeError|ModuleNotFoundError|Address already in use" "$log" 2>/dev/null; then
      echo "server on port ${port} failed, see $log" >&2; return 1
    fi
    sleep 5; waited=$((waited + 5))
    [ "$waited" -lt 900 ] || { echo "timeout waiting for port $port" >&2; return 1; }
  done
}

stop_servers() {
  local pid
  for pid in ${SERVER_PIDS:-}; do kill "$pid" >/dev/null 2>&1 || true; done
  wait ${SERVER_PIDS:-} >/dev/null 2>&1 || true
  SERVER_PIDS=""
  sleep 3
}

cleanup() {
  for pid in ${RUN_PID:-}; do kill "$pid" >/dev/null 2>&1 || true; done
  wait ${RUN_PID:-} >/dev/null 2>&1 || true
  stop_servers
}
on_signal() { trap - EXIT; cleanup; exit 143; }
trap cleanup EXIT
trap on_signal INT TERM

start_servers() {
  local schedule="$1" strategy="$2" task="$3" rep="$4" i port gpu log pid
  SERVER_PIDS=""
  for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"; gpu="${GPUS[$i]}"
    if is_port_busy "$port"; then echo "port $port is busy" >&2; stop_servers; return 1; fi
    log="$LOGDIR/server_${schedule}_${strategy}_${task}_rep${rep}_gpu${gpu}_port${port}.log"
    setsid "$PY" "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
      --path "$MODEL" --d "$gpu" --port "$port" --host 127.0.0.1 >"$log" 2>&1 < /dev/null &
    pid=$!; SERVER_PIDS="$SERVER_PIDS $pid"
    echo "$pid" > "$LOGDIR/server_${schedule}_${strategy}_${task}_rep${rep}_gpu${gpu}_port${port}.pid"
  done
  for i in "${!PORTS[@]}"; do
    wait_server_ready "${PORTS[$i]}" "$LOGDIR/server_${schedule}_${strategy}_${task}_rep${rep}_gpu${GPUS[$i]}_port${PORTS[$i]}.log" || { stop_servers; return 1; }
  done
}

write_progress() {
  local schedule="$1" strategy="$2" task="$3" rep="$4" run_id="$5" status="$6" rc="$7" result="$8"
  "$PY" - "$PROGRESS_JSONL" "$schedule" "$strategy" "$task" "$rep" "$run_id" "$status" "$rc" "$result" \
    "$POPULATION" "$GENERATIONS" "$ES_SIGMA" "$ES_ALPHA" <<'PY'
import json, sys
from datetime import datetime, timezone
path, schedule, strategy, task, rep, run_id, status, rc, result, population, generations, sigma, alpha = sys.argv[1:]
row = {
    "ts": datetime.now(timezone.utc).isoformat(), "task": task,
    "method": "sample_es", "schedule": schedule, "rep": int(rep),
    "invalid_reward_strategy": strategy,
    "run_id": run_id, "status": status, "exit_code": int(rc),
    "result_path": result, "population": int(population),
    "generations": int(generations), "sigma": sigma, "alpha": alpha,
    "sigma_final": "0" if schedule == "cosine" else sigma,
    "reward_normalization": "zscore",
    "invalid_reward": (
        "batch_relative_below_worst_valid"
        if strategy == "current" else "excluded_from_zscore_then_zero"
    ),
    "parameter_scope": "full", "generation_concurrency": 4,
    "evaluation_concurrency": 4, "reload_model_before_run": True,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
PY
}

URLS=(); for port in "${PORTS[@]}"; do URLS+=("http://127.0.0.1:${port}/completions"); done
URLS_CSV="$(join_by_comma "${URLS[@]}")"
echo "tasks=${TASKS[*]} reps=${REPS[*]} schedules=${SCHEDULES[*]} strategies=${STRATEGIES[*]}"
echo "gpus=${GPUS[*]} ports=${PORTS[*]} population=$POPULATION generations=$GENERATIONS"
echo "sigma=$ES_SIGMA alpha=$ES_ALPHA scope=$ES_PARAMETER_SCOPE reload_before_every_run=1 concurrency=4"

FAILURES=0; INDEX=0; TOTAL_RUNS=$(( ${#SCHEDULES[@]} * ${#REPS[@]} * ${#TASKS[@]} * ${#STRATEGIES[@]} ))
for schedule in "${SCHEDULES[@]}"; do
  for rep in "${REPS[@]}"; do
    for task in "${TASKS[@]}"; do
      for strategy in "${STRATEGIES[@]}"; do
      INDEX=$((INDEX + 1))
      run_id="${task}_sample_es_reload_${schedule}_${strategy}_pop${POPULATION}_gen${GENERATIONS}_rep${rep}_${STAMP}"
      result="$ROOT/cache/active_runs/${task}_train_sample_es_pop${POPULATION}_gen${GENERATIONS}_sigma0.001_alpha0.0005_${run_id}/results/pops_best/population_generation_${GENERATIONS}.json"
      log="$LOGDIR/${schedule}_${strategy}_${task}_rep${rep}.log"
      echo "[${INDEX}/${TOTAL_RUNS} reload-start] schedule=$schedule strategy=$strategy task=$task rep=$rep"
      write_progress "$schedule" "$strategy" "$task" "$rep" "$run_id" loading 0 "$result"
      if ! start_servers "$schedule" "$strategy" "$task" "$rep"; then
        write_progress "$schedule" "$strategy" "$task" "$rep" "$run_id" failed 90 "$result"; FAILURES=$((FAILURES + 1)); continue
      fi
      write_progress "$schedule" "$strategy" "$task" "$rep" "$run_id" started 0 "$result"
      ES_ENGINE_URLS="$URLS_CSV" ES_SIGMA="$ES_SIGMA" ES_ALPHA="$ES_ALPHA" \
      ES_PARAMETER_SCOPE="$ES_PARAMETER_SCOPE" ES_SIGMA_SCHEDULE="$schedule" \
      ES_INVALID_REWARD_STRATEGY="$strategy" \
      SAMPLE_BATCH_SIZE="$POPULATION" SAMPLE_GENERATIONS="$GENERATIONS" \
      LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
        "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
          --task "$task" --split train --method sample_es --run-id "$run_id" >"$log" 2>&1 &
      RUN_PID=$!; wait "$RUN_PID"; rc=$?; RUN_PID=""
      stop_servers
      if [ "$rc" -eq 0 ]; then
        write_progress "$schedule" "$strategy" "$task" "$rep" "$run_id" completed "$rc" "$result"
      else
        write_progress "$schedule" "$strategy" "$task" "$rep" "$run_id" failed "$rc" "$result"; FAILURES=$((FAILURES + 1))
      fi
      echo "[${INDEX}/${TOTAL_RUNS} done] schedule=$schedule strategy=$strategy task=$task rep=$rep rc=$rc"
      done
    done
  done
done
echo "[all-done] failures=$FAILURES run_root=$RUN_ROOT"
exit "$FAILURES"
