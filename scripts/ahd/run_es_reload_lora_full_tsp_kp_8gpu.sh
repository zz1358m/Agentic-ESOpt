#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_es_reload_lora_full_tsp_kp_8gpu_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"

TASKS=(${TASKS:-construct_tsp construct_kp})
REPS=(${REPS:-1 2 3})
MODES=(${MODES:-lora full})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
PORTS=(${PORTS:-11313 11314 11315 11316 11317 11318 11319 11320})
ES_OPERATORS="${ES_OPERATORS:-m1,m2}"
ES_DIRECTIONS="${ES_DIRECTIONS:-10}"
EC_M1M2_MULTIPLIER="${EC_M1M2_MULTIPLIER:-1}"
ES_SIGMA_SCHEDULE="${ES_SIGMA_SCHEDULE:-constant}"
ES_SIGMA_SCHEDULE_PLATEAU_FRACTION="${ES_SIGMA_SCHEDULE_PLATEAU_FRACTION:-0}"
ES_INVALID_REWARD_STRATEGY="${ES_INVALID_REWARD_STRATEGY:-current}"
LORA_SIGMA="${LORA_SIGMA:-2e-2}"
LORA_SIGMAS=(${LORA_SIGMAS:-$LORA_SIGMA})
LORA_ALPHA_ES="${LORA_ALPHA_ES:-2e-3}"
FULL_SIGMA="${FULL_SIGMA:-1e-3}"
FULL_SIGMAS=(${FULL_SIGMAS:-$FULL_SIGMA})
FULL_ALPHA_ES="${FULL_ALPHA_ES:-5e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES=(${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj up_proj down_proj gate_proj})
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"
ES_DISABLE_UPDATE="${ES_DISABLE_UPDATE:-0}"
ES_CONTINUE_PATH="${ES_CONTINUE_PATH:-}"
ES_CONTINUE_PROGRESS_JSONL="${ES_CONTINUE_PROGRESS_JSONL:-}"
ES_CONTINUE_ID="${ES_CONTINUE_ID:-0}"

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

continue_path_for() {
  local task="$1"
  local rep="$2"
  if [ -n "$ES_CONTINUE_PATH" ]; then
    "$PY" - "$ES_CONTINUE_PATH" <<'PY'
import json
import os
import sys

path = sys.argv[1]
candidates = [path]
if "/results/pops_best/" in path:
    candidates.insert(0, path.replace("/results/pops_best/", "/results/pops/"))

for candidate in candidates:
    if not candidate or not os.path.exists(candidate):
        continue
    try:
        with open(candidate, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        continue
    if isinstance(data, list):
        print(candidate)
        break
else:
    print(path)
PY
    return 0
  fi
  if [ -n "$ES_CONTINUE_PROGRESS_JSONL" ]; then
    "$PY" - "$ES_CONTINUE_PROGRESS_JSONL" "$task" "$rep" <<'PY'
import json
import os
import sys

progress, task, rep = sys.argv[1], sys.argv[2], int(sys.argv[3])
best = ""
with open(progress, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "completed" and row.get("task") == task and int(row.get("rep")) == rep:
            best = row.get("result_path", "")
candidates = [best]
if "/results/pops_best/" in best:
    candidates.insert(0, best.replace("/results/pops_best/", "/results/pops/"))

for candidate in candidates:
    if not candidate or not os.path.exists(candidate):
        continue
    try:
        with open(candidate, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    if isinstance(data, list):
        print(candidate)
        break
else:
    print(best)
PY
    return 0
  fi
  echo ""
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

stop_servers() {
  if [ -n "${SERVER_PIDS:-}" ]; then
    for pid in $SERVER_PIDS; do
      kill "$pid" >/dev/null 2>&1 || true
    done
    wait $SERVER_PIDS >/dev/null 2>&1 || true
    SERVER_PIDS=""
    sleep 5
  fi
}

cleanup() {
  stop_servers
}
trap cleanup EXIT INT TERM

start_servers() {
  local mode="$1"
  local run_id="$2"
  local i gpu port log pid
  SERVER_PIDS=""
  for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    gpu="${GPUS[$i]}"
    if is_port_busy "$port"; then
      echo "port ${port} already in use" >&2
      exit 1
    fi
    log="$LOGDIR/server_${mode}_${run_id}_gpu${gpu}_port${port}.log"
    if [ "$mode" = "lora" ]; then
      setsid "$PY" "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
        --path "$MODEL" \
        --d "$gpu" \
        --port "$port" \
        --host 127.0.0.1 \
        --enable-lora \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --lora-dropout "$LORA_DROPOUT" \
        --lora-target-modules "${LORA_TARGET_MODULES[@]}" \
        >"$log" 2>&1 < /dev/null &
    else
      setsid "$PY" "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
        --path "$MODEL" \
        --d "$gpu" \
        --port "$port" \
        --host 127.0.0.1 \
        >"$log" 2>&1 < /dev/null &
    fi
    pid=$!
    SERVER_PIDS="${SERVER_PIDS} ${pid}"
    echo "$pid" > "$LOGDIR/server_${mode}_${run_id}_gpu${gpu}_port${port}.pid"
  done

  for i in "${!PORTS[@]}"; do
    wait_server_ready "${PORTS[$i]}" "$LOGDIR/server_${mode}_${run_id}_gpu${GPUS[$i]}_port${PORTS[$i]}.log"
  done
}

write_progress() {
  local mode="$1"
  local task="$2"
  local sigma="$3"
  local alpha="$4"
  local rep="$5"
  local run_id="$6"
  local status="$7"
  local rc="$8"
  local result_path="$9"
  local continue_path
  continue_path="$(continue_path_for "$task" "$rep")"
  "$PY" - "$PROGRESS_JSONL" "$mode" "$task" "$sigma" "$alpha" "$ES_OPERATORS" "$rep" "$run_id" "$status" "$rc" "$result_path" "$LORA_R" "$LORA_ALPHA" "$ES_DISABLE_UPDATE" "$ES_SIGMA_SCHEDULE" "$ES_SIGMA_SCHEDULE_PLATEAU_FRACTION" "$EC_M1M2_MULTIPLIER" "$ES_DIRECTIONS" "$continue_path" "$ES_CONTINUE_ID" "$ES_INVALID_REWARD_STRATEGY" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, mode, task, sigma, alpha, operators, rep, run_id, status, rc, result_path, lora_r, lora_alpha, es_disable_update, sigma_schedule, sigma_schedule_plateau_fraction, m1m2_multiplier, es_directions, continue_path, continue_id, invalid_reward_strategy = sys.argv[1:]
record = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "task": task,
    "sigma": sigma,
    "alpha": alpha,
    "operators": operators,
    "parameter_scope": "lora" if mode == "lora" else "full",
    "rep": int(rep),
    "run_id": run_id,
    "status": status,
    "exit_code": int(rc),
    "result_path": result_path,
    "es_disable_update": es_disable_update in {"1", "true", "True"},
    "sigma_schedule": sigma_schedule,
    "sigma_schedule_plateau_fraction": float(sigma_schedule_plateau_fraction),
    "m1m2_multiplier": float(m1m2_multiplier),
    "es_directions": int(es_directions),
    "continue_path": continue_path,
    "continue_id": int(continue_id),
    "invalid_reward_strategy": invalid_reward_strategy,
}
if mode == "lora":
    record["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    record["lora_r"] = int(lora_r)
    record["lora_alpha"] = int(lora_alpha)
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=True) + "\n")
PY
}

run_one() {
  local mode="$1"
  local task="$2"
  local rep="$3"
  local configured_sigma="$4"
  local sigma alpha parameter_scope target_modules run_id log sigma_g alpha_g result_path rc continue_path
  if [ "$mode" = "lora" ]; then
    sigma="$configured_sigma"
    alpha="$LORA_ALPHA_ES"
    parameter_scope="lora"
    target_modules="$(join_by_comma "${LORA_TARGET_MODULES[@]}")"
    if [ "$ES_DISABLE_UPDATE" = "1" ]; then
      run_id="${task}_train_es_lora_noise_only_reload_sigma${sigma}_alpha${alpha}_r${LORA_R}_la${LORA_ALPHA}_rep${rep}_${STAMP}"
    else
      run_id="${task}_train_es_lora_reload_sigma${sigma}_alpha${alpha}_r${LORA_R}_la${LORA_ALPHA}_rep${rep}_${STAMP}"
    fi
  else
    sigma="$configured_sigma"
    alpha="$FULL_ALPHA_ES"
    parameter_scope="full"
    target_modules=""
    if [ "$ES_DISABLE_UPDATE" = "1" ]; then
      run_id="${task}_train_es_full_noise_only_reload_sigma${sigma}_alpha${alpha}_rep${rep}_${STAMP}"
    else
      run_id="${task}_train_es_full_reload_sigma${sigma}_alpha${alpha}_rep${rep}_${STAMP}"
    fi
  fi

  sigma_g="$(format_g "$sigma")"
  alpha_g="$(format_g "$alpha")"
  log="$LOGDIR/${mode}_${task}_rep${rep}_sigma${sigma}_alpha${alpha}.log"
  result_path="$ROOT/cache/active_runs/${task}_train_es_sigma${sigma_g}_alpha${alpha_g}_${run_id}/results/pops_best/population_generation_25.json"
  continue_path="$(continue_path_for "$task" "$rep")"

  echo "[start] mode=${mode} task=${task} rep=${rep} sigma=${sigma} alpha=${alpha} run_id=${run_id}"
  if [ -n "$continue_path" ]; then
    echo "[continue] task=${task} rep=${rep} from=${continue_path} continue_id=${ES_CONTINUE_ID}"
  fi
  write_progress "$mode" "$task" "$sigma" "$alpha" "$rep" "$run_id" "started" 0 "$result_path"
  start_servers "$mode" "$run_id"

  set +e
  (
    cd "$ROOT"
    RUN_ID="$run_id" \
    ES_SIGMA="$sigma" \
    ES_ALPHA="$alpha" \
    ES_DIRECTIONS="$ES_DIRECTIONS" \
    EC_M1M2_MULTIPLIER="$EC_M1M2_MULTIPLIER" \
    ES_CONTINUE_PATH="$continue_path" \
    ES_CONTINUE_ID="$ES_CONTINUE_ID" \
    ES_SIGMA_SCHEDULE="$ES_SIGMA_SCHEDULE" \
    ES_SIGMA_SCHEDULE_PLATEAU_FRACTION="$ES_SIGMA_SCHEDULE_PLATEAU_FRACTION" \
    ES_OPERATORS="$ES_OPERATORS" \
    ES_ENGINE_URLS="$ES_ENGINE_URLS_CSV" \
    ES_PARAMETER_SCOPE="$parameter_scope" \
    ES_TARGET_MODULES="$target_modules" \
    ES_DISABLE_UPDATE="$ES_DISABLE_UPDATE" \
    ES_INVALID_REWARD_STRATEGY="$ES_INVALID_REWARD_STRATEGY" \
    LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" \
      "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
      --task "$task" \
      --split train \
      --method es \
      --run-id "$run_id"
  ) >"$log" 2>&1
  rc=$?
  set -e

  stop_servers
  if [ "$rc" -eq 0 ]; then
    echo "[done] mode=${mode} task=${task} rep=${rep} rc=${rc}"
    write_progress "$mode" "$task" "$sigma" "$alpha" "$rep" "$run_id" "completed" "$rc" "$result_path"
  else
    echo "[failed] mode=${mode} task=${task} rep=${rep} rc=${rc}; see $log" >&2
    write_progress "$mode" "$task" "$sigma" "$alpha" "$rep" "$run_id" "failed" "$rc" "$result_path"
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
echo "modes=${MODES[*]}"
echo "tasks=${TASKS[*]}"
echo "reps=${REPS[*]}"
echo "gpus=${GPUS[*]}"
echo "ports=${PORTS[*]}"
echo "operators=$ES_OPERATORS"
echo "es_directions=$ES_DIRECTIONS"
echo "m1m2_multiplier=$EC_M1M2_MULTIPLIER"
echo "continue_progress_jsonl=$ES_CONTINUE_PROGRESS_JSONL"
echo "continue_path=$ES_CONTINUE_PATH continue_id=$ES_CONTINUE_ID"
echo "sigma_schedule=$ES_SIGMA_SCHEDULE plateau_fraction=$ES_SIGMA_SCHEDULE_PLATEAU_FRACTION"
echo "invalid_reward_strategy=$ES_INVALID_REWARD_STRATEGY"
echo "lora_sigmas=${LORA_SIGMAS[*]} lora_alpha=$LORA_ALPHA_ES lora_r=$LORA_R lora_lora_alpha=$LORA_ALPHA"
echo "full_sigmas=${FULL_SIGMAS[*]} full_alpha=$FULL_ALPHA_ES"
echo "reload_servers_each_run=1"
echo "es_disable_update=$ES_DISABLE_UPDATE"

TOTAL_RUNS=0
for mode in "${MODES[@]}"; do
  if [ "$mode" = "lora" ]; then
    TOTAL_RUNS=$(( TOTAL_RUNS + ${#LORA_SIGMAS[@]} * ${#TASKS[@]} * ${#REPS[@]} ))
  elif [ "$mode" = "full" ]; then
    TOTAL_RUNS=$(( TOTAL_RUNS + ${#FULL_SIGMAS[@]} * ${#TASKS[@]} * ${#REPS[@]} ))
  else
    echo "unsupported mode: $mode" >&2
    exit 1
  fi
done
RUN_INDEX=0
for mode in "${MODES[@]}"; do
  if [ "$mode" = "lora" ]; then
    SIGMAS=("${LORA_SIGMAS[@]}")
  else
    SIGMAS=("${FULL_SIGMAS[@]}")
  fi
  for sigma in "${SIGMAS[@]}"; do
    for task in "${TASKS[@]}"; do
      for rep in "${REPS[@]}"; do
        RUN_INDEX=$((RUN_INDEX + 1))
        echo "[${RUN_INDEX}/${TOTAL_RUNS}]"
        run_one "$mode" "$task" "$rep" "$sigma"
      done
    done
  done
done

echo "[all-done] run_root=$RUN_ROOT"
