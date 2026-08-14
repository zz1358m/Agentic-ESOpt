#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
HISTORY="${WEBARENA_ES_HISTORY_FILE:-}"
PREFIX="${WEBARENA_FINAL_EVAL_PREFIX:-webarena_final_eval}"

if [[ -z "$HISTORY" || ! -f "$HISTORY" ]]; then
  echo "Set WEBARENA_ES_HISTORY_FILE to the completed NoSkill ES history.json." >&2
  exit 2
fi
if [[ ! -s "$ROOT/apikey" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "The 40 fuzzy tasks require OPENAI_API_KEY or a non-empty $ROOT/apikey." >&2
  exit 2
fi

export WEBARENA_EVAL_REPEATS="${WEBARENA_EVAL_REPEATS:-3}"

"$PY" "$ROOT/webarena-train-time/scripts/install_vab_extensions.py" \
  --vab-root "${VAB_ROOT:-$ROOT/data/webarena/vab-lite}" --check

run_setting() {
  local setting=$1
  local history=${2:-}
  echo "[suite] setting=$setting judge_model=gpt-4.1-mini repeats=$WEBARENA_EVAL_REPEATS"
  RUN_ID="${PREFIX}_${setting}" WEBARENA_ES_HISTORY_FILE="$history" \
    "$ROOT/scripts/webarena/run.sh" "$setting" test
}

run_setting noskill_no-finetune
run_setting noskill_agentic_esopt "$HISTORY"
run_setting trace2skill_no-finetune
run_setting trace2skill_agentic_esopt "$HISTORY"
