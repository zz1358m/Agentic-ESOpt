#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
PROFILE="${1:-agentic-esopt-es32}"
MASK_COUNT="${2:-15}"

case "$MASK_COUNT" in
  5|10|15) ;;
  *)
    echo "Mask count must be one of 5, 10, or 15." >&2
    exit 2
    ;;
esac

POPULATION=32
case "$PROFILE" in
  vanilla-es32)
    if [[ "$MASK_COUNT" == "15" ]]; then
      SIGMA_START=5e-4
    else
      SIGMA_START=1e-3
    fi
    SIGMA_END="$SIGMA_START"
    SIGMA_SCHEDULE=constant
    ;;
  agentic-esopt-es32)
    if [[ "$MASK_COUNT" == "15" ]]; then
      SIGMA_START=7e-4
      SIGMA_END=5e-4
    else
      SIGMA_START=1e-3
      SIGMA_END=2.5e-4
    fi
    SIGMA_SCHEDULE=cosine
    ;;
  *)
    echo "Usage: $0 {vanilla-es32|agentic-esopt-es32} {5|10|15} [trainer args...]" >&2
    exit 2
    ;;
esac
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

ENDPOINTS="${SUDOKU_ES_ENDPOINTS:-http://127.0.0.1:12100,http://127.0.0.1:12101,http://127.0.0.1:12102,http://127.0.0.1:12103,http://127.0.0.1:12104,http://127.0.0.1:12105,http://127.0.0.1:12106,http://127.0.0.1:12107,http://127.0.0.1:12108,http://127.0.0.1:12109,http://127.0.0.1:12110,http://127.0.0.1:12111,http://127.0.0.1:12112,http://127.0.0.1:12113,http://127.0.0.1:12114,http://127.0.0.1:12115}"
RUN_ID="${RUN_ID:-sudoku_es_qwen35_4b_mask${MASK_COUNT}_${PROFILE}}"
LOG_PATH="${LOG_PATH:-$ROOT/runs/sudoku_es/$RUN_ID/train_eval.log}"
mkdir -p "$(dirname "$LOG_PATH")"

exec > >(tee "$LOG_PATH") 2>&1
exec "$PY" "$ROOT/sudoku-train-time/scripts/run_sudoku_es_train.py" \
  --endpoints "$ENDPOINTS" \
  --run-id "$RUN_ID" \
  --model "${SUDOKU_MODEL_NAME:-Qwen3.5-4B}" \
  --mask-count "$MASK_COUNT" \
  --generations 100 \
  --population "$POPULATION" \
  --case-batch-size 32 \
  --case-workers 4 \
  --sigma-start "$SIGMA_START" \
  --sigma-end "$SIGMA_END" \
  --sigma-schedule "$SIGMA_SCHEDULE" \
  --sigma-warmup-steps 0 \
  --alpha 5e-4 \
  --parameter-scope full \
  --reward-normalization zscore \
  --max-turns "$((MASK_COUNT * 3))" \
  --max-tokens 64 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0 \
  --batched-eval \
  --endpoint-batch-size 32 \
  --eval-interval 10 \
  --eval-repeats 3 \
  "$@"
