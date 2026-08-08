#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_dynamic_es_invalid_reward_tsp_kp_3rep_gpu0_3_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"

TASKS=(${TASKS:-construct_tsp construct_kp})
REPS=(${REPS:-1 2 3})
STRATEGIES=(${STRATEGIES:-current zero})
GPUS=(${GPUS:-0 1 2 3})
PORTS=(${PORTS:-11413 11414 11415 11416})
ES_OPERATORS="${ES_OPERATORS:-m1,m2}"
ES_DIRECTIONS="${ES_DIRECTIONS:-10}"
ES_SIGMA="${ES_SIGMA:-1e-3}"
ES_ALPHA="${ES_ALPHA:-5e-4}"
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"

mkdir -p "$LOGDIR"
cd "$ROOT"
echo "$$" > "$RUN_ROOT/queue.pid"

join_by_comma() { local IFS=","; echo "$*"; }
is_port_busy() { ss -ltn "( sport = :${1} )" 2>/dev/null | tail -n +2 | grep -q .; }

wait_server_ready() {
  local port="$1" log="$2" waited=0
  while true; do
    grep -q "Running on http://127.0.0.1:${port}" "$log" 2>/dev/null && return 0
    if grep -Eq "Traceback|RuntimeError|ModuleNotFoundError|Address already in use" "$log" 2>/dev/null; then
      echo "server on port ${port} failed; see $log" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
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
  if [ -n "${RUN_PID:-}" ]; then
    kill "$RUN_PID" >/dev/null 2>&1 || true
    wait "$RUN_PID" >/dev/null 2>&1 || true
  fi
  stop_servers
}
on_signal() { trap - EXIT; cleanup; exit 143; }
trap cleanup EXIT
trap on_signal INT TERM

start_servers() {
  local strategy="$1" task="$2" rep="$3" i port gpu log pid
  SERVER_PIDS=""
  for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"; gpu="${GPUS[$i]}"
    if is_port_busy "$port"; then echo "port $port is busy" >&2; stop_servers; return 1; fi
    log="$LOGDIR/server_${strategy}_${task}_rep${rep}_gpu${gpu}_port${port}.log"
    setsid "$PY" "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
      --path "$MODEL" --d "$gpu" --port "$port" --host 127.0.0.1 >"$log" 2>&1 < /dev/null &
    pid=$!; SERVER_PIDS="$SERVER_PIDS $pid"
    echo "$pid" > "$LOGDIR/server_${strategy}_${task}_rep${rep}_gpu${gpu}_port${port}.pid"
  done
  for i in "${!PORTS[@]}"; do
    wait_server_ready "${PORTS[$i]}" "$LOGDIR/server_${strategy}_${task}_rep${rep}_gpu${GPUS[$i]}_port${PORTS[$i]}.log" \
      || { stop_servers; return 1; }
  done
}

write_progress() {
  local strategy="$1" task="$2" rep="$3" run_id="$4" status="$5" rc="$6" result="$7"
  "$PY" - "$PROGRESS_JSONL" "$strategy" "$task" "$rep" "$run_id" "$status" "$rc" "$result" \
    "$ES_OPERATORS" "$ES_DIRECTIONS" "$ES_SIGMA" "$ES_ALPHA" <<'PY'
import json, sys
from datetime import datetime, timezone

path, strategy, task, rep, run_id, status, rc, result, operators, directions, sigma, alpha = sys.argv[1:]
row = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "method": "dynamic_es",
    "task": task,
    "rep": int(rep),
    "invalid_reward_strategy": strategy,
    "invalid_reward": "existing_population_worst_fallback" if strategy == "current" else "zero_for_raw_invalid_only",
    "run_id": run_id,
    "status": status,
    "exit_code": int(rc),
    "result_path": result,
    "operators": operators,
    "directions": int(directions),
    "population": 10,
    "generations": 25,
    "sigma": sigma,
    "alpha": alpha,
    "reward": "parent_objective_minus_offspring_objective",
    "reward_normalization": "zscore",
    "parameter_scope": "full",
    "generation_concurrency": 4,
    "evaluation_concurrency": 4,
    "reload_model_before_run": True,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
PY
}

if [ "${#GPUS[@]}" -ne "${#PORTS[@]}" ]; then echo "GPU/port count mismatch" >&2; exit 1; fi
if [ ! -x "$PY" ] || [ ! -d "$MODEL" ]; then echo "missing python or model" >&2; exit 1; fi

URLS=(); for port in "${PORTS[@]}"; do URLS+=("http://127.0.0.1:${port}/completions"); done
URLS_CSV="$(join_by_comma "${URLS[@]}")"
TOTAL_RUNS=$(( ${#STRATEGIES[@]} * ${#TASKS[@]} * ${#REPS[@]} ))
echo "run_root=$RUN_ROOT"
echo "tasks=${TASKS[*]} reps=${REPS[*]} strategies=${STRATEGIES[*]} total_runs=$TOTAL_RUNS"
echo "gpus=${GPUS[*]} operators=$ES_OPERATORS directions=$ES_DIRECTIONS sigma=$ES_SIGMA alpha=$ES_ALPHA"
echo "reload_before_every_run=1 concurrency=4 sample_queues_untouched=1"

FAILURES=0; INDEX=0
# Interleave strategies within each task/rep so the comparison is not confounded
# by one treatment always running much later than the other.
for rep in "${REPS[@]}"; do
  for task in "${TASKS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
      INDEX=$((INDEX + 1))
      run_id="${task}_dynamic_es_invalid_${strategy}_m1m2_full_rep${rep}_${STAMP}"
      result="$ROOT/cache/active_runs/${task}_train_es_sigma0.001_alpha0.0005_${run_id}/results/pops_best/population_generation_25.json"
      log="$LOGDIR/${strategy}_${task}_rep${rep}.log"
      echo "[$INDEX/$TOTAL_RUNS reload-start] strategy=$strategy task=$task rep=$rep"
      write_progress "$strategy" "$task" "$rep" "$run_id" loading 0 "$result"
      if ! start_servers "$strategy" "$task" "$rep"; then
        write_progress "$strategy" "$task" "$rep" "$run_id" failed 90 "$result"
        FAILURES=$((FAILURES + 1)); continue
      fi
      write_progress "$strategy" "$task" "$rep" "$run_id" started 0 "$result"
      ES_ENGINE_URLS="$URLS_CSV" ES_OPERATORS="$ES_OPERATORS" ES_DIRECTIONS="$ES_DIRECTIONS" \
      ES_SIGMA="$ES_SIGMA" ES_ALPHA="$ES_ALPHA" ES_PARAMETER_SCOPE=full \
      ES_INVALID_REWARD_STRATEGY="$strategy" LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
        "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
          --task "$task" --split train --method es --run-id "$run_id" >"$log" 2>&1 &
      RUN_PID=$!; wait "$RUN_PID"; rc=$?; RUN_PID=""
      stop_servers
      if [ "$rc" -eq 0 ]; then
        write_progress "$strategy" "$task" "$rep" "$run_id" completed "$rc" "$result"
      else
        write_progress "$strategy" "$task" "$rep" "$run_id" failed "$rc" "$result"
        FAILURES=$((FAILURES + 1))
      fi
      echo "[$INDEX/$TOTAL_RUNS done] strategy=$strategy task=$task rep=$rep rc=$rc"
    done
  done
done
echo "[all-done] failures=$FAILURES run_root=$RUN_ROOT"
exit "$FAILURES"
