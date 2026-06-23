#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi
PY="${PY:-python}"

METHOD="${METHOD:-${1:-trace2skill_es}}"
STAGE="${STAGE:-${2:-train}}"

ES_ENDPOINTS="${WEBARENA_ES_ENDPOINTS:-http://127.0.0.1:11013,http://127.0.0.1:11014,http://127.0.0.1:11015,http://127.0.0.1:11016}"
WEBARENA_ES_COMMON=(
  --endpoints "$ES_ENDPOINTS"
  --run-id "${RUN_ID:-webarena_${METHOD}_${STAGE}}"
  --split "${WEBARENA_TRAIN_SPLIT:-$ROOT/data/webarena/vab_lite_split/items.json}"
  --eval-split "${WEBARENA_EVAL_SPLIT:-$ROOT/data/webarena/vab_lite_split/items.json}"
  --config-dir "${WEBARENA_CONFIG_DIR:-$ROOT/data/webarena/vab-lite/config_files/wa/test_webarena_lite}"
  --sites "${EVAL_SITES:-reddit,gitlab,map,shopping,shopping_admin}"
  --generations "${WEBARENA_ES_GENERATIONS:-1}"
  --population "${WEBARENA_ES_POPULATION:-8}"
  --case-batch-size "${WEBARENA_ES_CASE_BATCH:-8}"
  --sigma "${WEBARENA_ES_SIGMA:-5e-4}"
  --alpha "${WEBARENA_ES_ALPHA:-5e-4}"
  --parameter-scope "${WEBARENA_ES_SCOPE:-full}"
  --model-name "${WEBARENA_MODEL_NAME:-Qwen3.5-27B}"
  --instruction-path "${WEBARENA_INSTRUCTION_PATH:-agent/prompts/jsons/p_webrl_chat_qwen_action.json}"
  --mode "${WEBARENA_MODE:-chat}"
  --stop-token "${WEBARENA_STOP_TOKEN:-}"
  --eval-limit "${TEST_LIMIT:-0}"
)

case "$METHOD:$STAGE" in
  no_skill_es:train)
    "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${WEBARENA_ES_COMMON[@]}" \
      --skill-file ""
    ;;
  no_skill_es:test)
    "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${WEBARENA_ES_COMMON[@]}" \
      --skill-file "" \
      --eval-only
    ;;
  trace2skill_es:train)
    "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${WEBARENA_ES_COMMON[@]}" \
      --skill-file "${TRACE2SKILL_SKILL_FILE:-$ROOT/webarena-train-time/skills/dynamic_agent_trace2skill_generation.md}"
    ;;
  trace2skill_es:test)
    "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${WEBARENA_ES_COMMON[@]}" \
      --skill-file "${TRACE2SKILL_SKILL_FILE:-$ROOT/webarena-train-time/skills/dynamic_agent_trace2skill_generation.md}" \
      --eval-only
    ;;
  trace2skill:train|trace2skill:test|trace2skill:train_test)
    TRACE2SKILL_ROOT="${TRACE2SKILL_ROOT:-$ROOT/webarena-train-time/methods/trace2skill/source}"
    export TRACE2SKILL_TRAIN_SPLIT="${TRACE2SKILL_TRAIN_SPLIT:-$ROOT/data/webarena/skillopt_splits/train/items.json}"
    export TRACE2SKILL_VAL_SPLIT="${TRACE2SKILL_VAL_SPLIT:-$ROOT/data/webarena/skillopt_splits/val/items.json}"
    export TRACE2SKILL_TEST_SPLIT="${TRACE2SKILL_TEST_SPLIT:-$ROOT/data/webarena/skillopt_splits/test/items.json}"
    export TRACE2SKILL_MAX_SKILL_LINES="${TRACE2SKILL_MAX_SKILL_LINES:-20}"
    if [ ! -f "$TRACE2SKILL_ROOT/run_traintest.sh" ]; then
      echo "Trace2Skill runner not found: $TRACE2SKILL_ROOT/run_traintest.sh" >&2
      exit 4
    fi
    STAGE="$STAGE" sh "$TRACE2SKILL_ROOT/run_traintest.sh"
    ;;
  *)
    echo "usage: METHOD=(no_skill_es|trace2skill_es|trace2skill) STAGE=(train|test|train_test) $0" >&2
    exit 2
    ;;
esac
