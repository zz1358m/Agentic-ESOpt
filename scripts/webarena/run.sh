#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi
PY="${PY:-python}"

if [[ -z "${METHOD:-}" ]]; then
  METHOD="${1:-trace2skill_es}"
  if [[ $# -gt 0 ]]; then shift; fi
fi
if [[ -z "${STAGE:-}" ]]; then
  STAGE="${1:-train}"
  if [[ $# -gt 0 ]]; then shift; fi
fi

ES_ENDPOINTS="${WEBARENA_ES_ENDPOINTS:-http://127.0.0.1:11013,http://127.0.0.1:11014,http://127.0.0.1:11015,http://127.0.0.1:11016}"
WEBARENA_ES_COMMON=(
  --endpoints "$ES_ENDPOINTS"
  --run-id "${RUN_ID:-webarena_${METHOD}_${STAGE}}"
  --split "${WEBARENA_TRAIN_SPLIT:-$ROOT/data/webarena/vab_nonlite_split/train/items.json}"
  --eval-split "${WEBARENA_EVAL_SPLIT:-$ROOT/data/webarena/vab_lite_split/items.json}"
  --config-dir "${WEBARENA_CONFIG_DIR:-$ROOT/data/webarena/vab-lite/config_files/wa/test_webarena_lite}"
  --train-config-dir "${WEBARENA_TRAIN_CONFIG_DIR:-$ROOT/data/webarena/vab-lite/config_files/wa/test_webarena}"
  --sites "${EVAL_SITES:-reddit,gitlab,map,shopping,shopping_admin}"
  --generations "${WEBARENA_ES_GENERATIONS:-1}"
  --population "${WEBARENA_ES_POPULATION:-8}"
  --case-batch-size "${WEBARENA_ES_CASE_BATCH:-8}"
  --sigma-start "${WEBARENA_ES_SIGMA_START:-5e-4}"
  --sigma-end "${WEBARENA_ES_SIGMA_END:-${WEBARENA_ES_SIGMA_START:-5e-4}}"
  --sigma-schedule "${WEBARENA_ES_SIGMA_SCHEDULE:-constant}"
  --sigma-warmup-steps "${WEBARENA_ES_SIGMA_WARMUP_STEPS:-0}"
  --alpha "${WEBARENA_ES_ALPHA:-5e-4}"
  --parameter-scope "${WEBARENA_ES_SCOPE:-full}"
  --model-name "${WEBARENA_MODEL_NAME:-Qwen3.5-27B}"
  --instruction-path "${WEBARENA_INSTRUCTION_PATH:-agent/prompts/jsons/p_webrl_chat_qwen_action.json}"
  --mode "${WEBARENA_MODE:-chat}"
  --stop-token "${WEBARENA_STOP_TOKEN:-}"
  --eval-limit "${TEST_LIMIT:-0}"
)

if [[ -n "${WEBARENA_ES_HISTORY_FILE:-}" ]]; then
  WEBARENA_ES_COMMON+=(--history-file "$WEBARENA_ES_HISTORY_FILE")
fi
if [[ -n "${WEBARENA_ES_RESUME_HISTORY:-}" ]]; then
  WEBARENA_ES_COMMON+=(--resume-history "$WEBARENA_ES_RESUME_HISTORY")
fi
case "${WEBARENA_TRACE2SKILL_EVERY_GENERATION:-0}" in
  1|true|yes|on) WEBARENA_ES_COMMON+=(--trace2skill-every-generation --init-empty-skill) ;;
esac

case "$METHOD:$STAGE" in
  no_skill_es:train)
    "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${WEBARENA_ES_COMMON[@]}" \
      --skill-file "" \
      "$@"
    ;;
  no_skill_es:test)
    "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${WEBARENA_ES_COMMON[@]}" \
      --skill-file "" \
      --eval-only \
      "$@"
    ;;
  trace2skill_es:train)
    TRACE_RUN_ID="${TRACE2SKILL_RUN_ID:-webarena_trace2skill}"
    "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${WEBARENA_ES_COMMON[@]}" \
      --skill-file "${TRACE2SKILL_SKILL_FILE:-$ROOT/runs/trace2skill_webarena_sft/$TRACE_RUN_ID/skill/SKILL.md}" \
      "$@"
    ;;
  trace2skill_es:test)
    TRACE_RUN_ID="${TRACE2SKILL_RUN_ID:-webarena_trace2skill}"
    "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${WEBARENA_ES_COMMON[@]}" \
      --skill-file "${TRACE2SKILL_SKILL_FILE:-$ROOT/runs/trace2skill_webarena_sft/$TRACE_RUN_ID/skill/SKILL.md}" \
      --eval-only \
      "$@"
    ;;
  trace2skill:train|trace2skill:train_test)
    TRACE_RUN_ID="${RUN_ID:-webarena_trace2skill}"
    TRACE_ARGS=(
      --run-id "$TRACE_RUN_ID"
      --split-dir "${TRACE2SKILL_SPLIT_DIR:-$ROOT/data/webarena/vab_nonlite_split}"
      --steps "${TRACE2SKILL_STEPS:-1}"
      --train-instances-per-epoch "${TRACE2SKILL_TRAIN_INSTANCES:-8}"
      --samples-per-instance "${TRACE2SKILL_SAMPLES_PER_INSTANCE:-1}"
      --eval-interval "${TRACE2SKILL_EVAL_INTERVAL:-1}"
      --train-workers "${TRACE2SKILL_TRAIN_WORKERS:-8}"
      --test-workers "${TRACE2SKILL_TEST_WORKERS:-32}"
      --analysis-workers "${TRACE2SKILL_WORKERS:-8}"
      --optimizer-model "${TRACE2SKILL_OPTIMIZER_MODEL:-gpt-4.1-mini}"
      --target-model-name "${WEBARENA_MODEL_NAME:-Qwen3.5-27B}"
      --instruction-path "${WEBARENA_INSTRUCTION_PATH:-agent/prompts/jsons/p_webrl_chat_qwen_action.json}"
    )
    if [[ -n "${TRACE2SKILL_MODEL_ENDPOINTS:-}" ]]; then
      TRACE_ARGS+=(--model-endpoints "$TRACE2SKILL_MODEL_ENDPOINTS")
    fi
    case "${TRACE2SKILL_EMPTY_SKILL:-0}" in
      1|true|yes|on) TRACE_ARGS+=(--empty-skill) ;;
    esac
    "$PY" "$ROOT/webarena-train-time/scripts/run_trace2skill_webarena_sft.py" \
      "${TRACE_ARGS[@]}" \
      "$@"
    ;;
  trace2skill:test)
    TRACE_RUN_ID="${RUN_ID:-webarena_trace2skill}"
    TRACE_TEST_ARGS=(
      --out-dir "${TRACE2SKILL_TEST_OUT:-$ROOT/runs/trace2skill_webarena_sft/$TRACE_RUN_ID/test}"
      --skill-file "${TRACE2SKILL_SKILL_FILE:-$ROOT/runs/trace2skill_webarena_sft/$TRACE_RUN_ID/skill/SKILL.md}"
      --workers "${TRACE2SKILL_TEST_WORKERS:-32}"
      --temperature "${TRACE2SKILL_TEST_TEMPERATURE:-0.0}"
      --model-name "${WEBARENA_MODEL_NAME:-Qwen3.5-27B}"
      --instruction-path "${WEBARENA_INSTRUCTION_PATH:-agent/prompts/jsons/p_webrl_chat_qwen_action.json}"
    )
    if [[ -n "${TRACE2SKILL_MODEL_ENDPOINTS:-}" ]]; then
      TRACE_TEST_ARGS+=(--model-endpoints "$TRACE2SKILL_MODEL_ENDPOINTS")
    fi
    "$PY" "$ROOT/scripts/webarena/eval_skill_lite165.py" \
      "${TRACE_TEST_ARGS[@]}" \
      "$@"
    ;;
  *)
    echo "usage: METHOD=(no_skill_es|trace2skill_es|trace2skill) STAGE=(train|test|train_test) $0" >&2
    exit 2
    ;;
esac
