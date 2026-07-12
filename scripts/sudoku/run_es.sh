#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

SUDOKU_TARGET_MASK_COUNT="${SUDOKU_TARGET_MASK_COUNT:-50}"

if [ ! -f "${SUDOKU_TRAIN_DATA:-$ROOT/data/sudoku/train.jsonl}" ]; then
  "$PY" "$ROOT/sudoku-train-time/scripts/generate_sudoku_data.py" \
    --output-dir "$ROOT/data/sudoku" \
    --train-size "${SUDOKU_TRAIN_SIZE:-128}" \
    --eval-size "${SUDOKU_EVAL_SIZE:-128}" \
    --mask-counts "${SUDOKU_MASK_COUNTS:-5,10,15,20}"
fi

"$PY" "$ROOT/sudoku-train-time/scripts/run_sudoku_es_train.py" \
  --endpoints "${SUDOKU_ES_ENDPOINTS:-${WEBARENA_ES_ENDPOINTS:-http://127.0.0.1:11013}}" \
  --run-id "${RUN_ID:-sudoku_es_mask${SUDOKU_TARGET_MASK_COUNT}}" \
  --train-data "${SUDOKU_TRAIN_DATA:-$ROOT/data/sudoku/train.jsonl}" \
  --eval-data "${SUDOKU_EVAL_DATA:-$ROOT/data/sudoku/eval.jsonl}" \
  --mask-count "$SUDOKU_TARGET_MASK_COUNT" \
  --generations "${SUDOKU_ES_GENERATIONS:-1}" \
  --population "${SUDOKU_ES_POPULATION:-8}" \
  --case-batch-size "${SUDOKU_ES_CASE_BATCH:-8}" \
  --sigma "${SUDOKU_ES_SIGMA:-5e-4}" \
  --alpha "${SUDOKU_ES_ALPHA:-5e-4}" \
  --max-turns "${SUDOKU_MAX_TURNS:-90}" \
  ${SUDOKU_ES_EXTRA_ARGS:-}
