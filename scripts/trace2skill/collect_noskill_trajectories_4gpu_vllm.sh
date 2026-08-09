#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-qwen35_4b_noskill_trajectory_dapo400x16_docvqa50x16_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/trace2skill_trajectories/${RUN_ID}}"

export RUN_ID OUT_DIR
export TRACE2SKILL_EVAL_DATASETS="dapo_evolve,docvqa_evolve"
export TRACE2SKILL_EVAL_SAMPLES="16"
export TRACE2SKILL_EVAL_MATH_MAX_TURNS="50"
export TRACE2SKILL_EVAL_DOCVQA_MAX_TURNS="50"
export TRACE2SKILL_EVAL_DOCVQA_MAX_TOKENS="512"
export TRACE2SKILL_EVAL_DOCVQA_LIMIT="0"
export TRACE2SKILL_EVAL_MATH_PYTHON_TIMEOUT="20.0"
export TRACE2SKILL_EVAL_DOCVQA_PYTHON_TIMEOUT="20.0"
export TRACE2SKILL_EVAL_CONCURRENCY="${TRACE2SKILL_EVAL_CONCURRENCY:-64}"

mkdir -p "$OUT_DIR/trace_logs"
exec "$ROOT/scripts/trace2skill/eval16_react_4gpu_vllm.sh" \
  --trace-log-dir "$OUT_DIR/trace_logs" \
  "$@"
