#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

EXTRA_ARGS=()
if [[ -n "${DOCVQA_ES_HISTORY_FILE:-}" ]]; then
  EXTRA_ARGS+=(--history-file "$DOCVQA_ES_HISTORY_FILE")
fi
if [[ -n "${DOCVQA_ES_RESUME_HISTORY:-}" ]]; then
  EXTRA_ARGS+=(--resume-history "$DOCVQA_ES_RESUME_HISTORY")
fi
if [[ -n "${DOCVQA_EXTRA_ARGS:-}" ]]; then
  read -r -a LEGACY_EXTRA_ARGS <<< "$DOCVQA_EXTRA_ARGS"
  EXTRA_ARGS+=("${LEGACY_EXTRA_ARGS[@]}")
fi

exec "$PY" "$ROOT/docvqa-train-time/scripts/run_docvqa_es_train.py" \
  --endpoints "${DOCVQA_ES_ENDPOINTS:-http://127.0.0.1:11013}" \
  --run-id "${RUN_ID:-docvqa_es}" \
  --endpoint-mode "${DOCVQA_ENDPOINT_MODE:-openai_vision_chat}" \
  --generations "${DOCVQA_ES_GENERATIONS:-1}" \
  --population "${DOCVQA_ES_POPULATION:-8}" \
  --case-batch-size "${DOCVQA_ES_CASE_BATCH:-8}" \
  --sigma-start "${DOCVQA_ES_SIGMA_START:-5e-4}" \
  --sigma-end "${DOCVQA_ES_SIGMA_END:-${DOCVQA_ES_SIGMA_START:-5e-4}}" \
  --sigma-schedule "${DOCVQA_ES_SIGMA_SCHEDULE:-constant}" \
  --sigma-warmup-steps "${DOCVQA_ES_SIGMA_WARMUP_STEPS:-0}" \
  --alpha "${DOCVQA_ES_ALPHA:-5e-4}" \
  --skill-file "${DOCVQA_SKILL_FILE:-}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
