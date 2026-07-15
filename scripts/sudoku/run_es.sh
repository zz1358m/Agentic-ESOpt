#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

SUDOKU_TARGET_MASK_COUNT="${SUDOKU_TARGET_MASK_COUNT:-15}"

if [ ! -f "${SUDOKU_TRAIN_DATA:-$ROOT/data/sudoku/train.jsonl}" ]; then
  "$PY" "$ROOT/sudoku-train-time/scripts/generate_sudoku_data.py" \
    --output-dir "$ROOT/data/sudoku" \
    --train-size "${SUDOKU_TRAIN_SIZE:-128}" \
    --eval-size "${SUDOKU_EVAL_SIZE:-128}" \
    --mask-counts "${SUDOKU_MASK_COUNTS:-5,10,15,20}"
fi

EXTRA_ARGS=()
if [[ -n "${SUDOKU_ES_HISTORY_FILE:-}" ]]; then
  EXTRA_ARGS+=(--history-file "$SUDOKU_ES_HISTORY_FILE")
fi
if [[ -n "${SUDOKU_ES_RESUME_HISTORY:-}" ]]; then
  EXTRA_ARGS+=(--resume-history "$SUDOKU_ES_RESUME_HISTORY")
fi
if [[ -n "${SUDOKU_ES_EXTRA_ARGS:-}" ]]; then
  read -r -a LEGACY_EXTRA_ARGS <<< "$SUDOKU_ES_EXTRA_ARGS"
  EXTRA_ARGS+=("${LEGACY_EXTRA_ARGS[@]}")
fi

exec "$PY" "$ROOT/sudoku-train-time/scripts/run_sudoku_es_train.py" \
  --endpoints "${SUDOKU_ES_ENDPOINTS:-${WEBARENA_ES_ENDPOINTS:-http://127.0.0.1:11013}}" \
  --run-id "${RUN_ID:-sudoku_es_mask${SUDOKU_TARGET_MASK_COUNT}}" \
  --train-data "${SUDOKU_TRAIN_DATA:-$ROOT/data/sudoku/train.jsonl}" \
  --eval-data "${SUDOKU_EVAL_DATA:-$ROOT/data/sudoku/eval.jsonl}" \
  --mask-count "$SUDOKU_TARGET_MASK_COUNT" \
  --generations "${SUDOKU_ES_GENERATIONS:-1}" \
  --population "${SUDOKU_ES_POPULATION:-8}" \
  --case-batch-size "${SUDOKU_ES_CASE_BATCH:-8}" \
  --sigma-start "${SUDOKU_ES_SIGMA_START:-5e-4}" \
  --sigma-end "${SUDOKU_ES_SIGMA_END:-${SUDOKU_ES_SIGMA_START:-5e-4}}" \
  --sigma-schedule "${SUDOKU_ES_SIGMA_SCHEDULE:-constant}" \
  --sigma-warmup-steps "${SUDOKU_ES_SIGMA_WARMUP_STEPS:-0}" \
  --alpha "${SUDOKU_ES_ALPHA:-5e-4}" \
  --max-turns "${SUDOKU_MAX_TURNS:-90}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
