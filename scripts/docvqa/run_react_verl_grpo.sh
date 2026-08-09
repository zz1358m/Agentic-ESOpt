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
    exec "$ROOT/scripts/docvqa/run_grpo.sh" "$@"
    ;;
  eval)
    PY="${PY:-python}"
    MODEL_PATH="${DOCVQA_GRPO_EVAL_MODEL_PATH:-${MODEL_PATH:-}}"
    OUT_DIR="${DOCVQA_GRPO_EVAL_OUT:-$ROOT/runs/multiturn_grpo/eval/docvqa}"
    DOCVQA_ROOT="${DOCVQA_ROOT:-$ROOT}"
    if [[ -z "$MODEL_PATH" ]]; then
      echo "Set DOCVQA_GRPO_EVAL_MODEL_PATH (or MODEL_PATH) to a base model or VERL-exported Hugging Face checkpoint." >&2
      exit 2
    fi
    exec "$PY" "$ROOT/scripts/docvqa/run_four_gpu_eval.py" \
      --model-path "$MODEL_PATH" \
      --out-dir "$OUT_DIR" \
      --docvqa-root "$DOCVQA_ROOT" \
      --physical-gpus "${DOCVQA_PHYSICAL_GPU_IDS:-auto}" \
      --samples "${DOCVQA_GRPO_EVAL_SAMPLES:-4}" \
      --resume \
      "$@"
    ;;
  *)
    echo "usage: $0 <train|eval> [arguments...]" >&2
    exit 2
    ;;
esac
