#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_es_llama31_8b_train_grid_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
SUMMARY_JSON="$RUN_ROOT/summary.json"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"

TASKS=(${TASKS:-construct_kp construct_asp aco_tsp aco_cvrp aco_bpp})
SIGMAS=(${SIGMAS:-1e-4 5e-4 1e-3 5e-3})
ALPHAS=(${ALPHAS:-5e-4 1e-3 5e-3})
REPS=(${REPS:-1 2 3})
ES_URLS=(${ES_URLS:-http://127.0.0.1:11013/completions http://127.0.0.1:11014/completions http://127.0.0.1:11015/completions http://127.0.0.1:11016/completions})

mkdir -p "$LOGDIR"
cd "$ROOT"

if [ ! -x "$PY" ]; then
  echo "python not executable: $PY" >&2
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

reset_engines() {
  local completion_url base_url
  for completion_url in "${ES_URLS[@]}"; do
    base_url="${completion_url%/completions}"
    http_post_json "$base_url/es/reset" '{}' >/dev/null 2>&1 || true
  done
}

status_engines() {
  local completion_url base_url
  for completion_url in "${ES_URLS[@]}"; do
    base_url="${completion_url%/completions}"
    echo "== $base_url/es/status =="
    http_post_json "$base_url/es/status" '{}' || true
    echo
  done
}

write_progress() {
  local task="$1"
  local sigma="$2"
  local alpha="$3"
  local rep="$4"
  local run_id="$5"
  local status="$6"
  local exit_code="$7"
  local result_path="$8"
  "$PY" - "$PROGRESS_JSONL" "$task" "$sigma" "$alpha" "$rep" "$run_id" "$status" "$exit_code" "$result_path" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, task, sigma, alpha, rep, run_id, status, exit_code, result_path = sys.argv[1:]
record = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "task": task,
    "sigma": sigma,
    "alpha": alpha,
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

TOTAL_RUNS=$(( ${#TASKS[@]} * ${#SIGMAS[@]} * ${#ALPHAS[@]} * ${#REPS[@]} ))
RUN_INDEX=0
ES_ENGINE_URLS_CSV="$(join_by_comma "${ES_URLS[@]}")"

echo "run_root=$RUN_ROOT"
echo "progress_jsonl=$PROGRESS_JSONL"
echo "total_runs=$TOTAL_RUNS"
echo "tasks=${TASKS[*]}"
echo "sigmas=${SIGMAS[*]}"
echo "alphas=${ALPHAS[*]}"
echo "reps=${REPS[*]}"

status_engines || true

for task in "${TASKS[@]}"; do
  for sigma in "${SIGMAS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      for rep in "${REPS[@]}"; do
        RUN_INDEX=$((RUN_INDEX + 1))
        run_id="${task}_train_es_grid_sigma${sigma}_alpha${alpha}_rep${rep}_${STAMP}"
        log="$LOGDIR/${task}_sigma${sigma}_alpha${alpha}_rep${rep}.log"
        sigma_g="$(format_g "$sigma")"
        alpha_g="$(format_g "$alpha")"
        result_path="$ROOT/cache/active_runs/${task}_train_es_sigma${sigma_g}_alpha${alpha_g}_${run_id}/results/pops_best/population_generation_25.json"

        echo "[${RUN_INDEX}/${TOTAL_RUNS}] start task=${task} sigma=${sigma} alpha=${alpha} rep=${rep} run_id=${run_id}"
        write_progress "$task" "$sigma" "$alpha" "$rep" "$run_id" "started" 0 "$result_path"
        reset_engines

        set +e
        (
          cd "$ROOT"
          RUN_ID="$run_id" \
          ES_SIGMA="$sigma" \
          ES_ALPHA="$alpha" \
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
          echo "[${RUN_INDEX}/${TOTAL_RUNS}] done task=${task} sigma=${sigma} alpha=${alpha} rep=${rep}"
          write_progress "$task" "$sigma" "$alpha" "$rep" "$run_id" "completed" "$rc" "$result_path"
        else
          echo "[${RUN_INDEX}/${TOTAL_RUNS}] failed task=${task} sigma=${sigma} alpha=${alpha} rep=${rep} rc=${rc}"
          write_progress "$task" "$sigma" "$alpha" "$rep" "$run_id" "failed" "$rc" "$result_path"
        fi
      done
    done
  done
done

"$PY" "$ROOT/scripts/ahd/summarize_es_grid.py" \
  --progress "$PROGRESS_JSONL" \
  --out "$SUMMARY_JSON"

echo "summary: $SUMMARY_JSON"
