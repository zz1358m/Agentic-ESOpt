#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -f "$ROOT/scripts/settings.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

STAGE="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi

case "$STAGE" in
  train)
    exec "$ROOT/scripts/math/run_grpo.sh" "$@"
    ;;
  eval)
    PY="${PY:-python}"
    MODEL_PATH="${MATH_GRPO_EVAL_MODEL_PATH:-${MODEL_PATH:-}}"
    OUT_DIR="${MATH_GRPO_EVAL_OUT:-$ROOT/runs/multiturn_grpo/eval/math}"
    if [[ -z "$MODEL_PATH" ]]; then
      echo "Set MATH_GRPO_EVAL_MODEL_PATH (or MODEL_PATH) to a base model or VERL-exported Hugging Face checkpoint." >&2
      exit 2
    fi
    exec "$PY" "$ROOT/scripts/math/run_four_gpu_eval.py" \
      --model-path "$MODEL_PATH" \
      --out-dir "$OUT_DIR" \
      --physical-gpus "${MATH_PHYSICAL_GPU_IDS:-0,1,2,3}" \
      --samples "${MATH_GRPO_EVAL_SAMPLES:-4}" \
      --profile "${MATH_GRPO_EVAL_PROFILE:-repo-react-v1-50x4096}" \
      --resume \
      "$@"
    ;;
  *)
    echo "usage: $0 <train|eval> [arguments...]" >&2
    exit 2
    ;;
esac
