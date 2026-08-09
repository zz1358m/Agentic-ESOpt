#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ACCELERATE="${ACCELERATE:-accelerate}"
MASK_COUNT="${1:-15}"

if [[ -z "${SUDOKU_GRPO_MODEL:-}" ]]; then
  echo "Set SUDOKU_GRPO_MODEL to a local Qwen3.5-4B path or Hugging Face model ID." >&2
  exit 2
fi

case "$MASK_COUNT" in
  5|10|15) ;;
  *)
    echo "Usage: $0 {5|10|15} [trainer args...]" >&2
    exit 2
    ;;
esac
shift $(( $# > 0 ? 1 : 0 ))

RUN_ID="${RUN_ID:-sudoku_grpo_qwen35_4b_mask${MASK_COUNT}_raw_policy_kl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/sudoku_grpo_multiturn/$RUN_ID}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/train_eval.log}"
mkdir -p "$OUTPUT_DIR"

exec > >(tee "$LOG_PATH") 2>&1
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  "$ACCELERATE" launch --num_processes "${NUM_PROCESSES:-4}" \
  "$ROOT/sudoku-train-time/scripts/run_sudoku_multiturn_grpo_train.py" \
  --model "$SUDOKU_GRPO_MODEL" \
  --mask-count "$MASK_COUNT" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 100 \
  --global-batch-size 32 \
  --num-generations 8 \
  --rollout-micro-batch-size 8 \
  --train-micro-batch-size 2 \
  --policy-batch-size 512 \
  --max-turns "$((MASK_COUNT * 3))" \
  --learning-rate 1e-6 \
  --beta 1e-3 \
  --clip-epsilon 0.2 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0 \
  --eval-before \
  --eval-interval 20 \
  --eval-repeats 3 \
  --no-eval-after \
  --log-interval 1 \
  "$@"
