#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -f "$ROOT/scripts/settings.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

PY="${PY:-python}"
if [[ -z "${METHOD:-}" ]]; then
  METHOD="${1:-noskill_agentic_esopt}"
  if [[ $# -gt 0 ]]; then shift; fi
fi
if [[ -z "${STAGE:-}" ]]; then
  STAGE="${1:-train}"
  if [[ $# -gt 0 ]]; then shift; fi
fi

RUN_ID="${RUN_ID:-webarena_${METHOD}_${STAGE}}"
ES_ENDPOINTS="${WEBARENA_ES_ENDPOINTS:-http://127.0.0.1:11013,http://127.0.0.1:11014,http://127.0.0.1:11015,http://127.0.0.1:11016}"
TRAIN_SPLIT="${WEBARENA_TRAIN_SPLIT:-$ROOT/data/webarena/vab_nonlite_split/train/items.json}"
EVAL_SPLIT="${WEBARENA_EVAL_SPLIT:-$ROOT/data/webarena/vab_lite_split/items.json}"
TRAIN_CONFIG_DIR="${WEBARENA_TRAIN_CONFIG_DIR:-$ROOT/data/webarena/vab-lite/config_files/wa/test_webarena}"
EVAL_CONFIG_DIR="${WEBARENA_CONFIG_DIR:-$ROOT/data/webarena/vab-lite/config_files/wa/test_webarena_lite}"
EVAL_SITES="${EVAL_SITES:-reddit,gitlab,map,shopping,shopping_admin}"

# Final Agentic-ESOpt setting: cosine(1.5e-3 -> 1.5e-3), hence fixed noise.
ES_GENERATIONS="${WEBARENA_ES_GENERATIONS:-70}"
ES_POPULATION="${WEBARENA_ES_POPULATION:-8}"
ES_CASE_BATCH="${WEBARENA_ES_CASE_BATCH:-8}"
ES_SIGMA_START="${WEBARENA_ES_SIGMA_START:-${WEBARENA_ES_SIGMA:-1.5e-3}}"
ES_SIGMA_END="${WEBARENA_ES_SIGMA_END:-$ES_SIGMA_START}"
ES_SIGMA_SCHEDULE="${WEBARENA_ES_SIGMA_SCHEDULE:-cosine}"
ES_ALPHA="${WEBARENA_ES_ALPHA:-2.5e-4}"
ES_REWARD_NORMALIZATION="${WEBARENA_ES_REWARD_NORMALIZATION:-zscore}"
ES_PARAMETER_SCOPE="${WEBARENA_ES_SCOPE:-full}"

# Shared final-evaluation decoding setting.
MODEL_NAME="${WEBARENA_MODEL_NAME:-Qwen3.5-27B}"
INSTRUCTION_PATH="${WEBARENA_INSTRUCTION_PATH:-agent/prompts/jsons/p_webrl_chat_qwen_action.json}"
MODE="${WEBARENA_MODE:-chat}"
STOP_TOKEN="${WEBARENA_STOP_TOKEN:-}"
TEMPERATURE="${WEBARENA_TEMPERATURE:-0.7}"
TOP_P="${WEBARENA_TOP_P:-0.8}"
TOP_K="${WEBARENA_TOP_K:-20}"
MIN_P="${WEBARENA_MIN_P:-0.0}"
PRESENCE_PENALTY="${WEBARENA_PRESENCE_PENALTY:-1.5}"
REPETITION_PENALTY="${WEBARENA_REPETITION_PENALTY:-1.0}"

TRACE_RUN_ID="${TRACE2SKILL_RUN_ID:-webarena_trace2skill_noft}"
TRACE_SKILL_FILE="${TRACE2SKILL_SKILL_FILE:-$ROOT/runs/trace2skill_webarena_sft/$TRACE_RUN_ID/skill/SKILL.md}"

TRACE_MODEL_ENDPOINTS="${TRACE2SKILL_MODEL_ENDPOINTS:-}"
if [[ -z "$TRACE_MODEL_ENDPOINTS" ]]; then
  IFS=',' read -r -a endpoint_array <<< "$ES_ENDPOINTS"
  for endpoint in "${endpoint_array[@]}"; do
    endpoint="${endpoint%/}"
    TRACE_MODEL_ENDPOINTS+="${TRACE_MODEL_ENDPOINTS:+ }${endpoint}/completions"
  done
fi

ES_TRAIN_ARGS=(
  --endpoints "$ES_ENDPOINTS"
  --run-id "$RUN_ID"
  --split "$TRAIN_SPLIT"
  --eval-split "$EVAL_SPLIT"
  --config-dir "$EVAL_CONFIG_DIR"
  --train-config-dir "$TRAIN_CONFIG_DIR"
  --sites "$EVAL_SITES"
  --generations "$ES_GENERATIONS"
  --population "$ES_POPULATION"
  --case-batch-size "$ES_CASE_BATCH"
  --case-workers-per-sample "${WEBARENA_ES_CASE_WORKERS:-8}"
  --eval-workers-per-endpoint "${WEBARENA_EVAL_WORKERS_PER_ENDPOINT:-8}"
  --eval-interval "${WEBARENA_ES_EVAL_INTERVAL:-10}"
  --sigma-start "$ES_SIGMA_START"
  --sigma-end "$ES_SIGMA_END"
  --sigma-schedule "$ES_SIGMA_SCHEDULE"
  --sigma-warmup-steps "${WEBARENA_ES_SIGMA_WARMUP_STEPS:-0}"
  --alpha "$ES_ALPHA"
  --reward-normalization "$ES_REWARD_NORMALIZATION"
  --parameter-scope "$ES_PARAMETER_SCOPE"
  --seed "${WEBARENA_ES_SEED:-20260605}"
  --model-name "$MODEL_NAME"
  --instruction-path "$INSTRUCTION_PATH"
  --mode "$MODE"
  --stop-token "$STOP_TOKEN"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --min-p "$MIN_P"
  --presence-penalty "$PRESENCE_PENALTY"
  --repetition-penalty "$REPETITION_PENALTY"
  --eval-limit "${TEST_LIMIT:-0}"
)

case "${WEBARENA_ES_SKIP_INITIAL_EVAL:-1}" in
  1|true|yes|on) ES_TRAIN_ARGS+=(--skip-initial-eval) ;;
esac

ES_REPLAY_HISTORY="${WEBARENA_ES_REPLAY_HISTORY:-}"
if [[ -n "$ES_REPLAY_HISTORY" ]]; then
  ES_TRAIN_ARGS+=(
    --replay-history "$ES_REPLAY_HISTORY"
    --replay-generations "${WEBARENA_ES_REPLAY_GENERATIONS:--1}"
  )
fi

run_final_eval() {
  local skill_file=$1
  local source_history=$2
  local require_history=$3
  shift 3
  local replay_args=()

  if [[ "$require_history" == 1 && -z "$source_history" ]]; then
    echo "Set WEBARENA_ES_HISTORY_FILE or WEBARENA_ES_TRAIN_RUN_ID for an Agentic-ESOpt evaluation." >&2
    exit 2
  fi
  if [[ -n "$source_history" ]]; then
    replay_args+=(--source-history "$source_history")
    if [[ -n "${WEBARENA_ES_EVAL_GENERATIONS:-}" ]]; then
      replay_args+=(--generations "$WEBARENA_ES_EVAL_GENERATIONS")
    fi
  else
    replay_args+=(--generations 0)
  fi
  if [[ -n "$skill_file" ]]; then
    replay_args+=(--skill-file "$skill_file")
  fi

  exec "$PY" "$ROOT/scripts/webarena/replay_es_history_and_eval.py" \
    --run-id "$RUN_ID" \
    --endpoints "$ES_ENDPOINTS" \
    --alpha "$ES_ALPHA" \
    --reward-normalization "$ES_REWARD_NORMALIZATION" \
    --parameter-scope "$ES_PARAMETER_SCOPE" \
    --eval-split "$EVAL_SPLIT" \
    --config-dir "$EVAL_CONFIG_DIR" \
    --sites "$EVAL_SITES" \
    --eval-limit "${TEST_LIMIT:-0}" \
    --eval-workers-per-endpoint "${WEBARENA_EVAL_WORKERS_PER_ENDPOINT:-8}" \
    --eval-repeats "${WEBARENA_EVAL_REPEATS:-3}" \
    --instruction-path "$INSTRUCTION_PATH" \
    --model-name "$MODEL_NAME" \
    --mode "$MODE" \
    --stop-token "$STOP_TOKEN" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    --presence-penalty "$PRESENCE_PENALTY" \
    --repetition-penalty "$REPETITION_PENALTY" \
    "${replay_args[@]}" \
    "$@"
}

ES_HISTORY_FILE="${WEBARENA_ES_HISTORY_FILE:-}"
if [[ -z "$ES_HISTORY_FILE" && -n "${WEBARENA_ES_TRAIN_RUN_ID:-}" ]]; then
  ES_HISTORY_FILE="$ROOT/runs/webrl_lite_full_es/$WEBARENA_ES_TRAIN_RUN_ID/history.json"
fi

case "$METHOD:$STAGE" in
  noskill_noft:test)
    run_final_eval "" "" 0 "$@"
    ;;
  noskill_agentic_esopt:train)
    exec "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${ES_TRAIN_ARGS[@]}" \
      --skill-file "" \
      "$@"
    ;;
  noskill_agentic_esopt:test)
    run_final_eval "" "$ES_HISTORY_FILE" 1 "$@"
    ;;
  trace2skill_noft:train|trace2skill_noft:train_test)
    trace_args=(
      --run-id "$TRACE_RUN_ID"
      --split-dir "${TRACE2SKILL_SPLIT_DIR:-$ROOT/data/webarena/vab_nonlite_split}"
      --steps "${TRACE2SKILL_STEPS:-70}"
      --train-instances-per-epoch "${TRACE2SKILL_TRAIN_INSTANCES:-8}"
      --samples-per-instance "${TRACE2SKILL_SAMPLES_PER_INSTANCE:-8}"
      --eval-interval "${TRACE2SKILL_EVAL_INTERVAL:-10}"
      --train-workers "${TRACE2SKILL_TRAIN_WORKERS:-32}"
      --test-workers "${TRACE2SKILL_TEST_WORKERS:-32}"
      --analysis-workers "${TRACE2SKILL_WORKERS:-16}"
      --optimizer-model "${TRACE2SKILL_OPTIMIZER_MODEL:-gpt-5.4-mini}"
      --analysis-reasoning-effort "${TRACE2SKILL_ANALYSIS_REASONING_EFFORT:-medium}"
      --skill-reasoning-effort "${TRACE2SKILL_SKILL_REASONING_EFFORT:-medium}"
      --consolidation-reasoning-effort "${TRACE2SKILL_CONSOLIDATION_REASONING_EFFORT:-medium}"
      --target-model-name "$MODEL_NAME"
      --instruction-path "$INSTRUCTION_PATH"
      --model-endpoints "$TRACE_MODEL_ENDPOINTS"
      --mode "$MODE"
      --stop-token "$STOP_TOKEN"
      --train-temperature "${TRACE2SKILL_TRAIN_TEMPERATURE:-$TEMPERATURE}"
      --test-temperature "${TRACE2SKILL_TEST_TEMPERATURE:-$TEMPERATURE}"
      --top-p "$TOP_P"
      --top-k "$TOP_K"
      --min-p "$MIN_P"
      --presence-penalty "$PRESENCE_PENALTY"
      --repetition-penalty "$REPETITION_PENALTY"
      --max-steps "${WEBARENA_MAX_STEPS:-30}"
    )
    case "${TRACE2SKILL_EMPTY_SKILL:-1}" in
      1|true|yes|on) trace_args+=(--empty-skill) ;;
    esac
    exec "$PY" "$ROOT/webarena-train-time/scripts/run_trace2skill_webarena_sft.py" \
      "${trace_args[@]}" \
      "$@"
    ;;
  trace2skill_noft:distill)
    if [[ -z "${WEBARENA_TRAJECTORY_RUN:-}" ]]; then
      echo "Set WEBARENA_TRAJECTORY_RUN to the ES run directory containing gen_* trajectory folders." >&2
      exit 2
    fi
    distill_args=(
      --es-run-dir "$WEBARENA_TRAJECTORY_RUN"
      --run-id "$TRACE_RUN_ID"
      --generations "${TRACE2SKILL_SOURCE_GENERATIONS:-10}"
      --max-traces "${TRACE2SKILL_MAX_TRACES:-0}"
      --html-limit "${TRACE2SKILL_HTML_LIMIT:-12000}"
      --optimizer-model "${TRACE2SKILL_OPTIMIZER_MODEL:-gpt-5.4-mini}"
      --analysis-workers "${TRACE2SKILL_WORKERS:-16}"
      --analysis-reasoning-effort "${TRACE2SKILL_ANALYSIS_REASONING_EFFORT:-medium}"
      --skill-reasoning-effort "${TRACE2SKILL_SKILL_REASONING_EFFORT:-medium}"
      --consolidation-reasoning-effort "${TRACE2SKILL_CONSOLIDATION_REASONING_EFFORT:-medium}"
      --seed "${TRACE2SKILL_SEED:-20260721}"
    )
    if [[ -n "${TRACE2SKILL_INITIAL_SKILL:-}" ]]; then
      distill_args+=(--initial-skill "$TRACE2SKILL_INITIAL_SKILL")
    else
      distill_args+=(--empty-skill)
    fi
    exec env \
      TRACE2SKILL_MAX_SKILL_LINES="${TRACE2SKILL_MAX_SKILL_LINES:-0}" \
      TRACE2SKILL_MAX_SKILL_TOKENS="${TRACE2SKILL_MAX_SKILL_TOKENS:-0}" \
      TRACE2SKILL_MAX_REFERENCES="${TRACE2SKILL_MAX_REFERENCES:-0}" \
      "$PY" "$ROOT/webarena-train-time/scripts/run_trace2skill_from_es_traces.py" \
      "${distill_args[@]}" \
      "$@"
    ;;
  trace2skill_noft:test)
    run_final_eval "$TRACE_SKILL_FILE" "" 0 "$@"
    ;;
  trace2skill_agentic_esopt:train)
    online_args=()
    case "${WEBARENA_TRACE2SKILL_EVERY_GENERATION:-0}" in
      1|true|yes|on) online_args+=(--trace2skill-every-generation) ;;
    esac
    exec "$PY" "$ROOT/webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py" \
      "${ES_TRAIN_ARGS[@]}" \
      --skill-file "$TRACE_SKILL_FILE" \
      "${online_args[@]}" \
      "$@"
    ;;
  trace2skill_agentic_esopt:test)
    run_final_eval "$TRACE_SKILL_FILE" "$ES_HISTORY_FILE" 1 "$@"
    ;;
  *)
    echo "usage: $0 METHOD STAGE" >&2
    echo "  noskill_noft test" >&2
    echo "  noskill_agentic_esopt train|test" >&2
    echo "  trace2skill_noft train|distill|test|train_test" >&2
    echo "  trace2skill_agentic_esopt train|test" >&2
    exit 2
    ;;
esac
