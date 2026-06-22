#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

METHOD="${METHOD:-${1:-skillopt}}"
STAGE="${STAGE:-${2:-train_test}}"

case "$METHOD:$STAGE" in
  skillopt:train)
    "$ROOT/webarena-train-time/scripts/train_skillopt_webagent_epoch_eval.sh" \
      "${EPOCHS:-6}" "${RUN_ID:-skillopt_train}" "${MODEL_PORT:-11013}" "${EVAL_PORT:-11015}" "0"
    ;;
  skillopt:test)
    "$ROOT/webarena-train-time/scripts/run_webrl_lite_vab_subset.sh" \
      skillopt "${EVAL_PORT:-11015}" "${EVAL_SITES:-shopping,shopping_admin,reddit,gitlab,map}" "${RUN_ID:-skillopt_test}" "${TEST_LIMIT:-0}"
    ;;
  skillopt:train_test)
    "$ROOT/webarena-train-time/scripts/train_skillopt_webagent_epoch_eval.sh" \
      "${EPOCHS:-6}" "${RUN_ID:-skillopt_train_test}" "${MODEL_PORT:-11013}" "${EVAL_PORT:-11015}" "${TEST_LIMIT:-0}"
    ;;
  trace2skill:train|trace2skill:test|trace2skill:train_test)
    TRACE2SKILL_ROOT="${TRACE2SKILL_ROOT:-$ROOT/webarena-train-time/methods/trace2skill/source}"
    export TRACE2SKILL_TRAIN_SPLIT="${TRACE2SKILL_TRAIN_SPLIT:-$ROOT/data/webarena/skillopt_splits/train/items.json}"
    export TRACE2SKILL_VAL_SPLIT="${TRACE2SKILL_VAL_SPLIT:-$ROOT/data/webarena/skillopt_splits/val/items.json}"
    export TRACE2SKILL_TEST_SPLIT="${TRACE2SKILL_TEST_SPLIT:-$ROOT/data/webarena/skillopt_splits/test/items.json}"
    if [ ! -f "$TRACE2SKILL_ROOT/run_traintest.sh" ]; then
      echo "Trace2Skill runner not found: $TRACE2SKILL_ROOT/run_traintest.sh" >&2
      exit 4
    fi
    STAGE="$STAGE" sh "$TRACE2SKILL_ROOT/run_traintest.sh"
    ;;
  *)
    echo "usage: METHOD=(skillopt|trace2skill) STAGE=(train|test|train_test) $0" >&2
    exit 2
    ;;
esac
