#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_es_grid_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"

TASKS=(${TASKS:-construct_tsp construct_kp construct_asp aco_tsp aco_cvrp aco_bpp})
SIGMAS=(${SIGMAS:-1e-4 5e-4 1e-3})
ALPHAS=(${ALPHAS:-5e-4 1e-3})
REPS=(${REPS:-1 2 3})

mkdir -p "$LOGDIR"

write_progress() {
  "$PY" - "$PROGRESS_JSONL" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, task, sigma, alpha, rep, run_id, status, rc = sys.argv[1:]
record = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "task": task,
    "sigma": sigma,
    "alpha": alpha,
    "rep": int(rep),
    "run_id": run_id,
    "status": status,
    "exit_code": int(rc),
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=True) + "\n")
PY
}

total=$(( ${#TASKS[@]} * ${#SIGMAS[@]} * ${#ALPHAS[@]} * ${#REPS[@]} ))
idx=0
echo "run_root=$RUN_ROOT"
echo "total_runs=$total"

for task in "${TASKS[@]}"; do
  for sigma in "${SIGMAS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      for rep in "${REPS[@]}"; do
        idx=$((idx + 1))
        run_id="${task}_train_es_sigma${sigma}_alpha${alpha}_rep${rep}_${STAMP}"
        log="$LOGDIR/${task}_sigma${sigma}_alpha${alpha}_rep${rep}.log"
        echo "[$idx/$total] $run_id"
        write_progress "$task" "$sigma" "$alpha" "$rep" "$run_id" started 0
        set +e
        TASK="$task" SPLIT=train METHOD=es RUN_ID="$run_id" ES_SIGMA="$sigma" ES_ALPHA="$alpha" \
          bash "$ROOT/scripts/ahd/run.sh" >"$log" 2>&1
        rc=$?
        set -e
        if [ "$rc" -eq 0 ]; then status=completed; else status=failed; fi
        write_progress "$task" "$sigma" "$alpha" "$rep" "$run_id" "$status" "$rc"
      done
    done
  done
done

echo "progress=$PROGRESS_JSONL"
