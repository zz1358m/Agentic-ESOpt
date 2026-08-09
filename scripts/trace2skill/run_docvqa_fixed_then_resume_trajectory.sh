#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DOC_RUN_ID="${DOC_RUN_ID:-qwen35_4b_docvqa_s4_envfix_$(date -u +%Y%m%d_%H%M%S)}"
TRAJECTORY_RUN_ID="${TRAJECTORY_RUN_ID:?Set TRAJECTORY_RUN_ID to the trajectory run to resume.}"
TRAJECTORY_OUT_DIR="${TRAJECTORY_OUT_DIR:-${ROOT}/runs/trace2skill_trajectories/${TRAJECTORY_RUN_ID}}"
STATUS_LOG="${ROOT}/logs/${DOC_RUN_ID}_wrapper.log"
DOC_LOG="${ROOT}/logs/${DOC_RUN_ID}_launcher.log"
RESUME_LOG="${ROOT}/logs/${TRAJECTORY_RUN_ID}_resume_after_docvqa_eval.log"

mkdir -p "$ROOT/logs"
timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

echo "[$(timestamp)] starting fixed DocVQA 100x4 eval" >>"$STATUS_LOG"
RUN_ID="$DOC_RUN_ID" \
TRACE2SKILL_EVAL_DATASETS=docvqa \
TRACE2SKILL_EVAL_SAMPLES=4 \
TRACE2SKILL_EVAL_SEED=20260629 \
TRACE2SKILL_EVAL_DOCVQA_MAX_TURNS=50 \
TRACE2SKILL_EVAL_DOCVQA_MAX_TOKENS=512 \
TRACE2SKILL_EVAL_DOCVQA_MAX_TOTAL_TOKENS=32768 \
TRACE2SKILL_EVAL_DOCVQA_LIMIT=100 \
  "$ROOT/scripts/trace2skill/eval16_react_4gpu_vllm.sh" >"$DOC_LOG" 2>&1
doc_exit=$?
echo "[$(timestamp)] DocVQA eval exited code=${doc_exit}" >>"$STATUS_LOG"

echo "[$(timestamp)] resuming trajectory run=${TRAJECTORY_RUN_ID}" >>"$STATUS_LOG"
RUN_ID="$TRAJECTORY_RUN_ID" OUT_DIR="$TRAJECTORY_OUT_DIR" \
  "$ROOT/scripts/trace2skill/collect_noskill_trajectories_4gpu_vllm.sh" \
  >"$RESUME_LOG" 2>&1
trajectory_exit=$?
echo "[$(timestamp)] trajectory exited code=${trajectory_exit}" >>"$STATUS_LOG"

if [[ "$doc_exit" -ne 0 ]]; then
  exit "$doc_exit"
fi
exit "$trajectory_exit"
