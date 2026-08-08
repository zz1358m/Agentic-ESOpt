#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
PLAN="${PLAN:?PLAN is required}"
QUEUE_NAME="${QUEUE_NAME:?QUEUE_NAME is required}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_sample_es_current_cosine_${QUEUE_NAME}_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"

GPUS=(${GPUS:?GPUS is required})
PORTS=(${PORTS:?PORTS is required})
POPULATION="${POPULATION:-20}"
ES_SIGMA="${ES_SIGMA:-1e-3}"
ES_ALPHA="${ES_ALPHA:-5e-4}"
ES_PARAMETER_SCOPE="${ES_PARAMETER_SCOPE:-full}"
EVALUATION_WORKERS="${EVALUATION_WORKERS:-4}"
ACO_EVA_TIMEOUT="${ACO_EVA_TIMEOUT:-30}"
CONSTRUCT_TSP_EVA_TIMEOUT="${CONSTRUCT_TSP_EVA_TIMEOUT:-20}"
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"

mkdir -p "$LOGDIR"
cd "$ROOT"
echo "$$" > "$RUN_ROOT/queue.pid"
cp "$PLAN" "$RUN_ROOT/job_plan.tsv"

if [ "${#GPUS[@]}" -ne 4 ] || [ "${#PORTS[@]}" -ne 4 ]; then
  echo "sample_es queue requires exactly four GPUs and ports" >&2
  exit 1
fi
if [ ! -x "$PY" ] || [ ! -d "$MODEL" ] || [ ! -f "$PLAN" ]; then
  echo "missing python, model, or plan" >&2
  exit 1
fi

join_by_comma() { local IFS=","; echo "$*"; }
is_port_busy() { ss -ltn "( sport = :${1} )" 2>/dev/null | tail -n +2 | grep -q .; }

wait_server_ready() {
  local port="$1" log="$2" waited=0
  while true; do
    grep -q "Running on http://127.0.0.1:${port}" "$log" 2>/dev/null && return 0
    if grep -Eq "Traceback|RuntimeError|ModuleNotFoundError|Address already in use" "$log" 2>/dev/null; then
      echo "server on port ${port} failed, see $log" >&2
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
  local total="$1" task="$2" rep="$3" i port gpu log pid
  SERVER_PIDS=""
  for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"; gpu="${GPUS[$i]}"
    if is_port_busy "$port"; then
      echo "port $port is busy" >&2
      stop_servers
      return 1
    fi
    log="$LOGDIR/server_t${total}_${task}_rep${rep}_gpu${gpu}_port${port}.log"
    setsid "$PY" "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
      --path "$MODEL" --d "$gpu" --port "$port" --host 127.0.0.1 >"$log" 2>&1 < /dev/null &
    pid=$!
    SERVER_PIDS="$SERVER_PIDS $pid"
    echo "$pid" > "$LOGDIR/server_t${total}_${task}_rep${rep}_gpu${gpu}_port${port}.pid"
  done
  for i in "${!PORTS[@]}"; do
    wait_server_ready "${PORTS[$i]}" \
      "$LOGDIR/server_t${total}_${task}_rep${rep}_gpu${GPUS[$i]}_port${PORTS[$i]}.log" \
      || { stop_servers; return 1; }
  done
}

write_progress() {
  local total="$1" task="$2" rep="$3" estimate="$4" run_id="$5" status="$6" rc="$7" result="$8" task_timeout="$9"
  "$PY" - "$PROGRESS_JSONL" "$QUEUE_NAME" "$total" "$task" "$rep" "$estimate" \
    "$run_id" "$status" "$rc" "$result" "$POPULATION" "$ES_SIGMA" "$ES_ALPHA" \
    "$EVALUATION_WORKERS" "$task_timeout" "$LLM_LOCAL_TIMEOUT" <<'PY'
import json, sys
from datetime import datetime, timezone
(
    path, queue, total, task, rep, estimate, run_id, status, rc, result,
    population, sigma, alpha, evaluation_workers, eva_timeout, llm_timeout,
) = sys.argv[1:]
total = int(total)
population = int(population)
row = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "queue": queue,
    "task": task,
    "rep": int(rep),
    "method": "sample_es",
    "invalid_reward_strategy": "current",
    "schedule": "cosine",
    "sample_total": total,
    "population": population,
    "generations": total // population,
    "sigma": sigma,
    "sigma_final": "0",
    "alpha": alpha,
    "reward_normalization": "zscore",
    "generation_concurrency": 4,
    "evaluation_concurrency": int(evaluation_workers),
    "evaluation_timeout_seconds": float(eva_timeout),
    "llm_timeout_seconds": float(llm_timeout),
    "reload_model_before_run": True,
    "estimated_minutes": int(estimate),
    "run_id": run_id,
    "status": status,
    "exit_code": int(rc),
    "result_path": result,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
PY
}

URLS=()
for port in "${PORTS[@]}"; do URLS+=("http://127.0.0.1:${port}/completions"); done
URLS_CSV="$(join_by_comma "${URLS[@]}")"
TOTAL_RUNS=$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$PLAN")
ESTIMATED_MINUTES=$(awk 'NF && $1 !~ /^#/ {sum+=$4} END {print sum+0}' "$PLAN")
echo "queue=$QUEUE_NAME plan=$PLAN runs=$TOTAL_RUNS estimated_minutes=$ESTIMATED_MINUTES"
echo "gpus=${GPUS[*]} ports=${PORTS[*]} population=$POPULATION schedule=cosine current=1"
echo "evaluation_workers=$EVALUATION_WORKERS aco_eva_timeout=$ACO_EVA_TIMEOUT construct_tsp_eva_timeout=$CONSTRUCT_TSP_EVA_TIMEOUT llm_timeout=$LLM_LOCAL_TIMEOUT"

FAILURES=0
INDEX=0
while read -r total task rep estimate; do
  [ -n "${total:-}" ] || continue
  [[ "$total" == \#* ]] && continue
  if [ "$total" -ne 1000 ] && [ "$total" -ne 2000 ]; then
    echo "invalid sample total in plan: $total" >&2
    FAILURES=$((FAILURES + 1))
    continue
  fi
  generations=$((total / POPULATION))
  case "$task" in
    aco_*) task_eva_timeout="$ACO_EVA_TIMEOUT" ;;
    construct_tsp) task_eva_timeout="$CONSTRUCT_TSP_EVA_TIMEOUT" ;;
    *) task_eva_timeout=30 ;;
  esac
  INDEX=$((INDEX + 1))
  run_id="${task}_sample_es_current_cosine_t${total}_rep${rep}_${QUEUE_NAME}_${STAMP}"
  result="$ROOT/cache/active_runs/${task}_train_sample_es_pop${POPULATION}_gen${generations}_sigma0.001_alpha0.0005_${run_id}/results/pops_best/population_generation_${generations}.json"
  log="$LOGDIR/t${total}_${task}_rep${rep}.log"
  echo "[$INDEX/$TOTAL_RUNS reload-start] total=$total task=$task rep=$rep generations=$generations"
  write_progress "$total" "$task" "$rep" "$estimate" "$run_id" loading 0 "$result" "$task_eva_timeout"
  if ! start_servers "$total" "$task" "$rep"; then
    write_progress "$total" "$task" "$rep" "$estimate" "$run_id" failed 90 "$result" "$task_eva_timeout"
    FAILURES=$((FAILURES + 1))
    continue
  fi
  write_progress "$total" "$task" "$rep" "$estimate" "$run_id" started 0 "$result" "$task_eva_timeout"
  ES_ENGINE_URLS="$URLS_CSV" ES_SIGMA="$ES_SIGMA" ES_ALPHA="$ES_ALPHA" \
  ES_PARAMETER_SCOPE="$ES_PARAMETER_SCOPE" ES_SIGMA_SCHEDULE=cosine \
  ES_INVALID_REWARD_STRATEGY=current SAMPLE_BATCH_SIZE="$POPULATION" \
  SAMPLE_GENERATIONS="$generations" EVALUATION_WORKERS="$EVALUATION_WORKERS" \
  EVA_TIMEOUT="$task_eva_timeout" LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
    "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
      --task "$task" --split train --method sample_es --run-id "$run_id" \
      --sample-batch-size "$POPULATION" --sample-generations "$generations" \
      --es-invalid-reward-strategy current --es-sigma-schedule cosine \
      --evaluation-workers "$EVALUATION_WORKERS" --eva-timeout "$task_eva_timeout" \
      >"$log" 2>&1 &
  RUN_PID=$!
  wait "$RUN_PID"
  rc=$?
  RUN_PID=""
  stop_servers
  if [ "$rc" -eq 0 ]; then
    write_progress "$total" "$task" "$rep" "$estimate" "$run_id" completed "$rc" "$result" "$task_eva_timeout"
  else
    write_progress "$total" "$task" "$rep" "$estimate" "$run_id" failed "$rc" "$result" "$task_eva_timeout"
    FAILURES=$((FAILURES + 1))
  fi
  echo "[$INDEX/$TOTAL_RUNS done] total=$total task=$task rep=$rep rc=$rc"
done < "$PLAN"

echo "[all-done] failures=$FAILURES run_root=$RUN_ROOT"
exit "$FAILURES"
