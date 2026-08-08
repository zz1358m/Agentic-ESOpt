#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_es_m1m2_llama31_8b_train_fixed_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"

TASKS=(${TASKS:-construct_tsp construct_kp construct_asp aco_tsp aco_cvrp aco_bpp})
REPS=(${REPS:-1 2 3})
GPUS=(${GPUS:-4 5 6 7})
PORTS=(${PORTS:-11313 11314 11315 11316})
SIGMA="${SIGMA:-1e-3}"
ALPHA="${ALPHA:-5e-4}"
ES_OPERATORS="${ES_OPERATORS:-m1,m2}"

mkdir -p "$LOGDIR"
cd "$ROOT"

if [ "${#GPUS[@]}" -ne "${#PORTS[@]}" ]; then
  echo "GPUS and PORTS must have the same length." >&2
  exit 1
fi

join_by_comma() {
  local IFS=","
  echo "$*"
}

format_g() {
  "$PY" - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}")
PY
}

http_post_json() {
  local url="$1"
  local payload="$2"
  curl --connect-timeout 5 --max-time 20 -sS -X POST -H 'Content-Type: application/json' -d "$payload" "$url"
}

kill_own_ports() {
  local port
  for port in "${PORTS[@]}"; do
    pkill -f "llama31_instruct_server.py.*--port ${port} " || true
  done
  sleep 5
}

start_servers() {
  local i gpu port log
  for i in "${!PORTS[@]}"; do
    gpu="${GPUS[$i]}"
    port="${PORTS[$i]}"
    log="$LOGDIR/server_gpu${gpu}_port${port}.log"
    "$PY" "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
      --path "$MODEL" --d "$gpu" --port "$port" --host 127.0.0.1 \
      >"$log" 2>&1 < /dev/null &
  done
}

wait_servers() {
  local i gpu port log waited
  for i in "${!PORTS[@]}"; do
    gpu="${GPUS[$i]}"
    port="${PORTS[$i]}"
    log="$LOGDIR/server_gpu${gpu}_port${port}.log"
    waited=0
    until grep -q "Running on http://127.0.0.1:${port}" "$log" 2>/dev/null; do
      if grep -Eq "Traceback|Error|RuntimeError|ModuleNotFoundError|Address already in use" "$log" 2>/dev/null; then
        echo "server failed: $log" >&2
        exit 1
      fi
      sleep 10
      waited=$((waited + 10))
      if [ "$waited" -ge 900 ]; then
        echo "timeout waiting for server $port" >&2
        exit 1
      fi
    done
  done
}

reset_engines() {
  local port
  for port in "${PORTS[@]}"; do
    http_post_json "http://127.0.0.1:${port}/es/reset" '{}' >/dev/null 2>&1 || true
  done
}

write_progress() {
  local task="$1"
  local rep="$2"
  local run_id="$3"
  local status="$4"
  local exit_code="$5"
  local result_path="$6"
  "$PY" - "$PROGRESS_JSONL" "$task" "$SIGMA" "$ALPHA" "$ES_OPERATORS" "$rep" "$run_id" "$status" "$exit_code" "$result_path" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, task, sigma, alpha, operators, rep, run_id, status, exit_code, result_path = sys.argv[1:]
record = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "task": task,
    "sigma": sigma,
    "alpha": alpha,
    "operators": operators,
    "rep": int(rep),
    "run_id": run_id,
    "status": status,
    "exit_code": int(exit_code),
    "result_path": result_path,
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=True) + "\n")
PY
}

ES_URLS=()
for port in "${PORTS[@]}"; do
  ES_URLS+=("http://127.0.0.1:${port}/completions")
done
ES_ENGINE_URLS_CSV="$(join_by_comma "${ES_URLS[@]}")"
SIGMA_G="$(format_g "$SIGMA")"
ALPHA_G="$(format_g "$ALPHA")"
TOTAL_RUNS=$(( ${#TASKS[@]} * ${#REPS[@]} ))
RUN_INDEX=0

echo "run_root=$RUN_ROOT"
echo "progress_jsonl=$PROGRESS_JSONL"
echo "tasks=${TASKS[*]}"
echo "reps=${REPS[*]}"
echo "sigma=$SIGMA alpha=$ALPHA operators=$ES_OPERATORS"
echo "gpus=${GPUS[*]} ports=${PORTS[*]}"

kill_own_ports
start_servers
wait_servers

for task in "${TASKS[@]}"; do
  for rep in "${REPS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    run_id="${task}_train_es_m1m2_sigma${SIGMA}_alpha${ALPHA}_rep${rep}_${STAMP}"
    log="$LOGDIR/${task}_rep${rep}.log"
    result_path="$ROOT/cache/active_runs/${task}_train_es_sigma${SIGMA_G}_alpha${ALPHA_G}_${run_id}/results/pops_best/population_generation_25.json"

    echo "[${RUN_INDEX}/${TOTAL_RUNS}] start task=${task} rep=${rep} run_id=${run_id}"
    write_progress "$task" "$rep" "$run_id" "started" 0 "$result_path"
    reset_engines

    set +e
    (
      cd "$ROOT"
      RUN_ID="$run_id" \
      ES_SIGMA="$SIGMA" \
      ES_ALPHA="$ALPHA" \
      ES_OPERATORS="$ES_OPERATORS" \
      ES_ENGINE_URLS="$ES_ENGINE_URLS_CSV" \
        "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
        --task "$task" \
        --split train \
        --method es \
        --run-id "$run_id"
    ) >"$log" 2>&1
    rc=$?
    set -e

    reset_engines
    if [ "$rc" -eq 0 ]; then
      status=completed
    else
      status=failed
    fi
    write_progress "$task" "$rep" "$run_id" "$status" "$rc" "$result_path"
    echo "[${RUN_INDEX}/${TOTAL_RUNS}] $status task=${task} rep=${rep} rc=${rc}"
  done
done

echo "done"
