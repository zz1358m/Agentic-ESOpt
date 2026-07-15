#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

TASK_POSITIONAL=""
SPLIT_POSITIONAL=""
METHOD_POSITIONAL=""
if [[ $# -gt 0 && "$1" != -* ]]; then TASK_POSITIONAL="$1"; shift; fi
if [[ $# -gt 0 && "$1" != -* ]]; then SPLIT_POSITIONAL="$1"; shift; fi
if [[ $# -gt 0 && "$1" != -* ]]; then METHOD_POSITIONAL="$1"; shift; fi
TASK="${TASK:-${TASK_POSITIONAL:-construct_tsp}}"
SPLIT="${SPLIT:-${SPLIT_POSITIONAL:-train}}"
METHOD="${METHOD:-${METHOD_POSITIONAL:-eoh}}"
RUN_ID="${RUN_ID:-${TASK}_${SPLIT}_${METHOD}}"

EXTRA_ARGS=()
if [[ -n "${ES_HISTORY_FILE:-}" ]]; then
  EXTRA_ARGS+=(--es-history-file "$ES_HISTORY_FILE")
fi
if [[ -n "${ES_RESUME_HISTORY:-}" ]]; then
  EXTRA_ARGS+=(--resume-history "$ES_RESUME_HISTORY")
fi
if [[ -n "${AHD_CONTINUE_PATH:-}" ]]; then
  EXTRA_ARGS+=(--continue-path "$AHD_CONTINUE_PATH" --continue-id "${AHD_CONTINUE_ID:-0}")
fi
if [[ -n "${AHD_EXTRA_ARGS:-}" ]]; then
  read -r -a LEGACY_EXTRA_ARGS <<< "$AHD_EXTRA_ARGS"
  EXTRA_ARGS+=("${LEGACY_EXTRA_ARGS[@]}")
fi

exec "$PY" "$ROOT/ahd-test-time/scripts/run_eoh_ahd.py" \
  --task "$TASK" \
  --split "$SPLIT" \
  --method "$METHOD" \
  --run-id "$RUN_ID" \
  --llm-local-url "${LLM_LOCAL_URL:-http://127.0.0.1:11013/completions}" \
  --es-engine-urls "${ES_ENGINE_URLS:-}" \
  --es-operators "${ES_OPERATORS:-e1,e2,m1,m2}" \
  --eoh-k "${EOH_K:-1}" \
  --ec-pop-size "${AHD_POP_SIZE:-10}" \
  --ec-generations "${AHD_GENERATIONS:-25}" \
  --es-sigma-start "${ES_SIGMA_START:-1e-3}" \
  --es-sigma-end "${ES_SIGMA_END:-${ES_SIGMA_START:-1e-3}}" \
  --es-sigma-schedule "${ES_SIGMA_SCHEDULE:-constant}" \
  --es-sigma-warmup-steps "${ES_SIGMA_WARMUP_STEPS:-0}" \
  --es-alpha "${ES_ALPHA:-5e-4}" \
  --es-seed "${ES_SEED:-2024}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
