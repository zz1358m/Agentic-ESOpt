#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

TRAIN_DATA="${MATH_TRAIN_DATA:-$ROOT/data/trace2skill/math_reasoning/dapo_evolve.jsonl}"
EVAL_DATA="${MATH_EVAL_DATA:-$ROOT/data/trace2skill/math_reasoning/dapo_test.jsonl}"
AIME_DATA="${MATH_AIME_DATA:-$ROOT/data/trace2skill/math_reasoning/aime_2026.jsonl}"

if [[ ! -f "$TRAIN_DATA" || ! -f "$EVAL_DATA" || ! -f "$AIME_DATA" ]]; then
  cat >&2 <<EOF
Missing math reasoning data.

Expected:
  $TRAIN_DATA
  $EVAL_DATA
  $AIME_DATA

Prepare it with:
  $PY $ROOT/algorithms/trace2skill-settings/scripts/prepare_data.py --setting math_reasoning \\
    --train-source /path/to/dapo_evolve.jsonl \\
    --eval-source /path/to/dapo_test.jsonl \\
    --aime-source /path/to/aime_2026.jsonl
EOF
  exit 1
fi

HISTORY_ARGS=()
if [[ -n "${MATH_ES_HISTORY_FILE:-}" ]]; then
  HISTORY_ARGS+=(--history-file "$MATH_ES_HISTORY_FILE")
fi
if [[ -n "${MATH_ES_RESUME_HISTORY:-}" ]]; then
  HISTORY_ARGS+=(--resume-history "$MATH_ES_RESUME_HISTORY")
fi

exec "$PY" "$ROOT/math-train-time/scripts/run_math_es_train.py" \
  --endpoints "${MATH_ES_ENDPOINTS:-http://127.0.0.1:11013}" \
  --run-id "${RUN_ID:-math_reasoning_es_$(date -u +%Y%m%d_%H%M%S)}" \
  --train-data "$TRAIN_DATA" \
  --eval-data "$EVAL_DATA" \
  --aime-data "$AIME_DATA" \
  --skill-file "${MATH_SKILL_FILE:-}" \
  --generations "${MATH_ES_GENERATIONS:-1}" \
  --population "${MATH_ES_POPULATION:-8}" \
  --case-batch-size "${MATH_ES_CASE_BATCH:-8}" \
  --case-workers "${MATH_ES_CASE_WORKERS:-4}" \
  --inference-batch-size "${MATH_INFERENCE_BATCH_SIZE:-16}" \
  --train-samples "${MATH_TRAIN_SAMPLES:-1}" \
  --eval-samples "${MATH_EVAL_SAMPLES:-16}" \
  --max-turns "${MATH_MAX_TURNS:-${MATH_MAX_REACT_ROUNDS:-0}}" \
  --python-timeout "${MATH_PYTHON_TIMEOUT:-20.0}" \
  --tool-observation-limit "${MATH_TOOL_OBSERVATION_LIMIT:-6000}" \
  --request-retries "${MATH_REQUEST_RETRIES:-3}" \
  --eval-limit "${MATH_EVAL_LIMIT:-100}" \
  --aime-limit "${MATH_AIME_LIMIT:-30}" \
  --sigma-start "${MATH_ES_SIGMA_START:-5e-4}" \
  --sigma-end "${MATH_ES_SIGMA_END:-${MATH_ES_SIGMA_START:-5e-4}}" \
  --sigma-schedule "${MATH_ES_SIGMA_SCHEDULE:-constant}" \
  --sigma-warmup-steps "${MATH_ES_SIGMA_WARMUP_STEPS:-0}" \
  --alpha "${MATH_ES_ALPHA:-5e-4}" \
  --seed "${MATH_ES_SEED:-20260627}" \
  --parameter-scope "${MATH_ES_SCOPE:-full}" \
  --reward-normalization "${MATH_ES_REWARD_NORMALIZATION:-zscore}" \
  --model "${MATH_MODEL_NAME:-Qwen3.5-4B}" \
  --max-tokens "${MATH_MAX_TOKENS:-0}" \
  --temperature "${MATH_TEMPERATURE:-1.0}" \
  --top-p "${MATH_TOP_P:-1.0}" \
  --top-k "${MATH_TOP_K:-40}" \
  --min-p "${MATH_MIN_P:-0.0}" \
  --presence-penalty "${MATH_PRESENCE_PENALTY:-2.0}" \
  --repetition-penalty "${MATH_REPETITION_PENALTY:-1.0}" \
  --timeout "${MATH_TIMEOUT:-1800}" \
  --eval-interval "${MATH_ES_EVAL_INTERVAL:-1}" \
  "${HISTORY_ARGS[@]}" \
  "$@"
