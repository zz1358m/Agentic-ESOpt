#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_es_ablation_8gpu_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
PORTS=(${PORTS:-11313 11314 11315 11316 11317 11318 11319 11320})
SIGMA="${SIGMA:-1e-3}"
ALPHA="${ALPHA:-5e-4}"
ES_OPERATORS="${ES_OPERATORS:-m1,m2}"
REPS=(${REPS:-1 2 3})
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"

mkdir -p "$LOGDIR"
cd "$ROOT"

if [ "${#GPUS[@]}" -ne "${#PORTS[@]}" ]; then
  echo "GPUS and PORTS must have the same length." >&2
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

format_g() {
  "$PY" - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}")
PY
}

is_port_busy() {
  local port="$1"
  ss -ltn "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q .
}

http_post_json() {
  local url="$1"
  local payload="$2"
  curl --connect-timeout 5 --max-time 30 -sS -X POST -H 'Content-Type: application/json' -d "$payload" "$url"
}

reset_engines() {
  local port
  for port in "${PORTS[@]}"; do
    http_post_json "http://127.0.0.1:${port}/es/reset" '{}' >/dev/null 2>&1 || true
  done
}

cleanup() {
  if [ -n "${SERVER_PIDS:-}" ]; then
    for pid in $SERVER_PIDS; do
      kill "$pid" >/dev/null 2>&1 || true
    done
  fi
}
trap cleanup EXIT INT TERM

wait_server_ready() {
  local port="$1"
  local log="$2"
  local waited=0
  while true; do
    if grep -q "Running on http://127.0.0.1:${port}" "$log" 2>/dev/null; then
      return 0
    fi
    if grep -Eq "Traceback|Error|RuntimeError|ModuleNotFoundError|Address already in use" "$log" 2>/dev/null; then
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
    port="${PORTS[$i]}"
    gpu="${GPUS[$i]}"
    if is_port_busy "$port"; then
      echo "port ${port} already in use" >&2
      exit 1
    fi
    log="$LOGDIR/server_gpu${gpu}_port${port}.log"
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

write_progress() {
  local phase="$1"
  local task="$2"
  local rep="$3"
  local run_id="$4"
  local status="$5"
  local rc="$6"
  local result_path="$7"
  "$PY" - "$PROGRESS_JSONL" "$phase" "$task" "$SIGMA" "$ALPHA" "$ES_OPERATORS" "$rep" "$run_id" "$status" "$rc" "$result_path" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, phase, task, sigma, alpha, operators, rep, run_id, status, rc, result_path = sys.argv[1:]
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "task": task,
        "sigma": sigma,
        "alpha": alpha,
        "operators": operators,
        "rep": int(rep),
        "run_id": run_id,
        "status": status,
        "exit_code": int(rc),
        "result_path": result_path,
    }, ensure_ascii=True) + "\n")
PY
}

run_one() {
  local phase="$1"
  local task="$2"
  local rep="$3"
  local disable_update="$4"
  local run_id="$5"
  local log="$LOGDIR/${phase}_${task}_rep${rep}.log"
  local sigma_g alpha_g result_path rc
  sigma_g="$(format_g "$SIGMA")"
  alpha_g="$(format_g "$ALPHA")"
  result_path="$ROOT/cache/active_runs/${task}_train_es_sigma${sigma_g}_alpha${alpha_g}_${run_id}/results/pops_best/population_generation_25.json"

  echo "[start] phase=${phase} task=${task} rep=${rep} run_id=${run_id}"
  write_progress "$phase" "$task" "$rep" "$run_id" "started" 0 "$result_path"

  set +e
  (
    cd "$ROOT"
    RUN_ID="$run_id" \
    ES_SIGMA="$SIGMA" \
    ES_ALPHA="$ALPHA" \
    ES_OPERATORS="$ES_OPERATORS" \
    ES_ENGINE_URLS="$ES_ENGINE_URLS_CSV" \
    ES_DISABLE_UPDATE="$disable_update" \
    LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
      "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
      --task "$task" \
      --split train \
      --method es \
      --run-id "$run_id"
  ) >"$log" 2>&1
  rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    echo "[done] phase=${phase} task=${task} rep=${rep} rc=${rc}"
    write_progress "$phase" "$task" "$rep" "$run_id" "completed" "$rc" "$result_path"
  else
    echo "[failed] phase=${phase} task=${task} rep=${rep} rc=${rc}; see $log" >&2
    write_progress "$phase" "$task" "$rep" "$run_id" "failed" "$rc" "$result_path"
    exit "$rc"
  fi
}

ES_URLS=()
for port in "${PORTS[@]}"; do
  ES_URLS+=("http://127.0.0.1:${port}/completions")
done
ES_ENGINE_URLS_CSV="$(join_by_comma "${ES_URLS[@]}")"

echo "run_root=$RUN_ROOT"
echo "progress_jsonl=$PROGRESS_JSONL"
echo "model=$MODEL"
echo "python=$PY"
echo "gpus=${GPUS[*]}"
echo "ports=${PORTS[*]}"
echo "sigma=$SIGMA alpha=$ALPHA operators=$ES_OPERATORS"
echo "reps=${REPS[*]}"

start_servers

reset_engines
for rep in "${REPS[@]}"; do
  run_id="construct_tsp_train_es_noise_only_sigma${SIGMA}_alpha${ALPHA}_rep${rep}_${STAMP}"
  reset_engines
  run_one "noise_only" "construct_tsp" "$rep" "1" "$run_id"
done

reset_engines
for rep in "${REPS[@]}"; do
  for task in construct_tsp construct_kp construct_asp; do
    run_id="${task}_train_es_no_refresh_seq_sigma${SIGMA}_alpha${ALPHA}_rep${rep}_${STAMP}"
    run_one "no_refresh_sequence" "$task" "$rep" "0" "$run_id"
  done
done

echo "[all-done] run_root=$RUN_ROOT"
