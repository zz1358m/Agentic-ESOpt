#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WAIT_PID="${WAIT_PID:?Set WAIT_PID to the current evaluator PID.}"
RUN_ID="${RUN_ID:-qwen35_4b_noskill_trajectory_dapo400x16_docvqa50x16_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/trace2skill_trajectories/${RUN_ID}}"
STATUS_LOG="${STATUS_LOG:-${ROOT}/logs/${RUN_ID}_queue.log}"
LAUNCHER_LOG="${LAUNCHER_LOG:-${ROOT}/logs/${RUN_ID}_launcher.log}"

mkdir -p "$ROOT/logs" "$OUT_DIR"
timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

echo "[$(timestamp)] queued; waiting for evaluator pid=${WAIT_PID}" >>"$STATUS_LOG"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 10
done
echo "[$(timestamp)] evaluator exited; waiting for vLLM ports to be released" >>"$STATUS_LOG"

for _ in $(seq 1 60); do
  if ! ss -ltn 2>/dev/null | grep -Eq ':(18080|18081|18082|18083)[[:space:]]'; then
    break
  fi
  sleep 1
done
if ss -ltn 2>/dev/null | grep -Eq ':(18080|18081|18082|18083)[[:space:]]'; then
  echo "[$(timestamp)] ports still busy; collection not started" >>"$STATUS_LOG"
  exit 1
fi

echo "[$(timestamp)] starting no-skill trajectory collection run_id=${RUN_ID}" >>"$STATUS_LOG"
set +e
RUN_ID="$RUN_ID" OUT_DIR="$OUT_DIR" \
  "$ROOT/scripts/trace2skill/collect_noskill_trajectories_4gpu_vllm.sh" \
  >"$LAUNCHER_LOG" 2>&1
exit_code=$?
set -e
echo "[$(timestamp)] collection exited code=${exit_code}" >>"$STATUS_LOG"
exit "$exit_code"
