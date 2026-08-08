#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_dynamic_es_current_cosine_aco3_3rep_gpu0_4_${STAMP}}"
DYNAMIC_ROOT="${DYNAMIC_ROOT:-$ROOT/runs/ahd_dynamic_es_invalid_reward_tsp_kp_3rep_gpu0_3_20260719_071107}"
DYNAMIC_PID="${DYNAMIC_PID:-$(cat "$DYNAMIC_ROOT/queue.pid")}"
DYNAMIC_PROGRESS="$DYNAMIC_ROOT/progress.jsonl"
SAMPLE_PID="${SAMPLE_PID:-$(cat "$ROOT/runs/ahd_sample_extend_all6_3rep_to2000_gpu4_7_20260718_145101/queue.pid")}"
CHILD_PID=""

mkdir -p "$RUN_ROOT"
echo "$$" > "$RUN_ROOT/queue.pid"

cleanup() {
  if [ -n "$CHILD_PID" ]; then
    kill -TERM "$CHILD_PID" >/dev/null 2>&1 || true
    wait "$CHILD_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "waiting for in-flight constant-current construct_kp rep2"
while ! "$PY" - "$DYNAMIC_PROGRESS" <<'PY'
import json, os, sys
found = False
if os.path.isfile(sys.argv[1]):
    with open(sys.argv[1], encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("status") == "completed"
                and row.get("task") == "construct_kp"
                and int(row.get("rep", -1)) == 2
                and row.get("invalid_reward_strategy") == "current"
            ):
                found = True
                break
raise SystemExit(0 if found else 1)
PY
do
  sleep 10
done

echo "retiring obsolete zero-reward comparison queue pid=$DYNAMIC_PID"
kill -TERM "$DYNAMIC_PID" >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  kill -0 "$DYNAMIC_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$DYNAMIC_PID" 2>/dev/null; then
  echo "dynamic comparison queue did not stop cleanly" >&2
  exit 2
fi

echo "waiting for sample queue pid=$SAMPLE_PID to release GPU4"
while kill -0 "$SAMPLE_PID" 2>/dev/null; do sleep 20; done

echo "starting cosine-current ACO TSP/CVRP/BPP x3 on GPUs 0-4"
env ROOT="$ROOT" PY="$PY" STAMP="$STAMP" RUN_ROOT="$RUN_ROOT" \
  MODES="full" TASKS="aco_tsp aco_cvrp aco_bpp" REPS="1 2 3" \
  GPUS="0 1 2 3 4" PORTS="11813 11814 11815 11816 11817" \
  ES_OPERATORS="m1,m2" ES_DIRECTIONS="10" FULL_SIGMAS="1e-3" FULL_ALPHA_ES="5e-4" \
  ES_SIGMA_SCHEDULE="cosine" ES_SIGMA_SCHEDULE_PLATEAU_FRACTION="0" \
  ES_INVALID_REWARD_STRATEGY="current" \
  bash "$ROOT/scripts/ahd/run_es_reload_lora_full_tsp_kp_8gpu.sh" &
CHILD_PID=$!; wait "$CHILD_PID"; RC=$?; CHILD_PID=""
[ "$RC" -eq 0 ] || exit "$RC"

echo "[all-done] cosine-current ACO three tasks x3"
