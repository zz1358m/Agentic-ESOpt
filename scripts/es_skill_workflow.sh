#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/es_skill_workflow.sh <math|docvqa> <action> [runner arguments...]

Actions:
  es-train       Train model weights with ES and save no-skill trajectories.
  eval           Replay an ES history and run no-skill evaluation/trajectories.
  distill-skill  Select no-skill ES trajectories and distill one SKILL.md.
  skill-eval     Replay the same ES history and evaluate with the distilled skill.

Machine paths and experiment settings are supplied through environment variables
or scripts/settings.local.env. See scripts/es_skill_workflow.example.env.
EOF
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

TASK="$1"
ACTION="$2"
shift 2

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
if [[ -f "$ROOT/scripts/settings.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

bool_args() {
  local value="$1"
  case "${value,,}" in
    1|true|yes|on) printf '%s\n' "--enforce-eager" ;;
    *) printf '%s\n' "--no-enforce-eager" ;;
  esac
}

case "$TASK" in
  math)
    SETTING="math_reasoning"
    RUNNER="$ROOT/math-train-time/scripts/run_math_es_vllm_train.py"
    RESULT_SUBDIR="${RESULT_SUBDIR:-runs/math_es_vllm}"
    TRAIN_DATA="${TRAIN_DATA:-${MATH_TRAIN_DATA:-$ROOT/data/trace2skill/math_reasoning/dapo_evolve.jsonl}}"
    EVAL_DATA="${EVAL_DATA:-${MATH_EVAL_DATA:-$ROOT/data/trace2skill/math_reasoning/dapo_test.jsonl}}"
    AIME_DATA="${AIME_DATA:-${MATH_AIME_DATA:-$ROOT/data/trace2skill/math_reasoning/aime_2026.jsonl}}"
    MODEL_PATH="${MODEL_PATH:-${MATH_MODEL_PATH:-Qwen/Qwen3.5-4B}}"
    MODEL_NAME="${MODEL_NAME:-${MATH_MODEL_NAME:-Qwen3.5-4B}}"
    GENERATIONS="${ES_GENERATIONS:-${MATH_ES_GENERATIONS:-25}}"
    SEED="${ES_SEED:-${MATH_ES_SEED:-20260627}}"
    MAX_TOKENS="${MAX_TOKENS:-${MATH_MAX_TOKENS:-4096}}"
    MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-${MATH_MAX_TOTAL_TOKENS:-0}}"
    VLLM_DEFAULT_MAX_TOKENS="${VLLM_DEFAULT_MAX_TOKENS:-${MATH_VLLM_DEFAULT_MAX_TOKENS:-4096}}"
    EVAL_LIMIT="${EVAL_LIMIT:-${MATH_EVAL_LIMIT:-100}}"
    AIME_LIMIT="${AIME_LIMIT:-${MATH_AIME_LIMIT:-30}}"
    INITIAL_SKILL="${INITIAL_SKILL:-$ROOT/algorithms/trace2skill-settings/skills/math_reasoning/SKILL.md}"
    ;;
  docvqa)
    SETTING="docvqa"
    RUNNER="$ROOT/docvqa-train-time/scripts/run_docvqa_es_vllm_train.py"
    RESULT_SUBDIR="${RESULT_SUBDIR:-runs/docvqa_es_vllm}"
    TRAIN_DATA="${TRAIN_DATA:-${DOCVQA_TRAIN_DATA:-$ROOT/data/trace2skill/docvqa/evolve.jsonl}}"
    EVAL_DATA="${EVAL_DATA:-${DOCVQA_EVAL_DATA:-$ROOT/data/trace2skill/docvqa/test.jsonl}}"
    AIME_DATA="$EVAL_DATA"
    MODEL_PATH="${MODEL_PATH:-${DOCVQA_MODEL_PATH:-Qwen/Qwen3.5-4B}}"
    MODEL_NAME="${MODEL_NAME:-${DOCVQA_MODEL_NAME:-Qwen3.5-4B}}"
    GENERATIONS="${ES_GENERATIONS:-${DOCVQA_ES_GENERATIONS:-40}}"
    SEED="${ES_SEED:-${DOCVQA_ES_SEED:-20260627}}"
    MAX_TOKENS="${MAX_TOKENS:-${DOCVQA_MAX_TOKENS:-512}}"
    MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-${DOCVQA_MAX_TOTAL_TOKENS:-32768}}"
    VLLM_DEFAULT_MAX_TOKENS="${VLLM_DEFAULT_MAX_TOKENS:-${DOCVQA_VLLM_DEFAULT_MAX_TOKENS:-512}}"
    EVAL_LIMIT="${EVAL_LIMIT:-${DOCVQA_EVAL_LIMIT:-100}}"
    AIME_LIMIT="$EVAL_LIMIT"
    INITIAL_SKILL="${INITIAL_SKILL:-$ROOT/algorithms/trace2skill-settings/skills/docvqa/SKILL.md}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

TRAIN_RUN_ID="${TRAIN_RUN_ID:-${TASK}_es_train}"
EVAL_RUN_ID="${EVAL_RUN_ID:-${TRAIN_RUN_ID}_raw_eval}"
SKILL_EVAL_RUN_ID="${SKILL_EVAL_RUN_ID:-${TRAIN_RUN_ID}_skill_eval}"
DISTILL_RUN_ID="${DISTILL_RUN_ID:-${TRAIN_RUN_ID}_distilled_skill}"
RESULT_ROOT="$ROOT/$RESULT_SUBDIR"
ES_HISTORY="${ES_HISTORY:-$RESULT_ROOT/$TRAIN_RUN_ID/history.json}"
TRACE_ROOT="${TRACE_ROOT:-$RESULT_ROOT/$TRAIN_RUN_ID/trace_logs/train}"
DISTILL_ROOT="${DISTILL_ROOT:-$ROOT/runs/es_trajectory_distill/$DISTILL_RUN_ID}"
ANALYSIS_LOGS="${ANALYSIS_LOGS:-$DISTILL_ROOT/analysis_logs}"
TRAJECTORY_MANIFEST="${TRAJECTORY_MANIFEST:-${ANALYSIS_LOGS}_manifest.json}"
SKILL_FILE="${SKILL_FILE:-$ROOT/runs/trace2skill_extra/$DISTILL_RUN_ID/skill_step_001.md}"

NUM_ENGINES="${NUM_ENGINES:-4}"
GPU_FRACTION="${GPU_FRACTION:-1.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"
ROLLOUT_TOKEN_BUDGET="${ROLLOUT_TOKEN_BUDGET:-131072}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1}"
TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-1}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
MAX_TURNS="${MAX_TURNS:-50}"
PYTHON_TIMEOUT="${PYTHON_TIMEOUT:-20}"
TOOL_OBSERVATION_LIMIT="${TOOL_OBSERVATION_LIMIT:-6000}"
POPULATION="${ES_POPULATION:-16}"
CASE_BATCH_SIZE="${ES_CASE_BATCH_SIZE:-16}"
SIGMA_START="${ES_SIGMA_START:-0.001}"
SIGMA_END="${ES_SIGMA_END:-0.0005}"
SIGMA_SCHEDULE="${ES_SIGMA_SCHEDULE:-cosine}"
SIGMA_WARMUP_STEPS="${ES_SIGMA_WARMUP_STEPS:-0}"
ALPHA="${ES_ALPHA:-0.0005}"
EVAL_INTERVAL="${ES_EVAL_INTERVAL:-10}"
EVAL_SEED="${EVAL_SEED:-$SEED}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

export MATH_ES_RESULT_SUBDIR="$RESULT_SUBDIR"
export PYTHONPATH="$ROOT/math-train-time/scripts:$ROOT/math-train-time/envs:$ROOT/docvqa-train-time/scripts:$ROOT/docvqa-train-time/envs:$ROOT/algorithms/verl_trace2skill:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

runner_args() {
  local samples="$1"
  COMMON_ARGS=(
    --train-data "$TRAIN_DATA"
    --eval-data "$EVAL_DATA"
    --aime-data "$AIME_DATA"
    --model-path "$MODEL_PATH"
    --tokenizer-path "${TOKENIZER_PATH:-$MODEL_PATH}"
    --model "$MODEL_NAME"
    --num-engines "$NUM_ENGINES"
    --gpu-fraction "$GPU_FRACTION"
    --dtype "${DTYPE:-bfloat16}"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --gdn-prefill-backend "${GDN_PREFILL_BACKEND:-triton}"
    --vllm-default-max-tokens "$VLLM_DEFAULT_MAX_TOKENS"
    --trust-remote-code
    "$(bool_args "$ENFORCE_EAGER")"
    --inference-batch-size "$INFERENCE_BATCH_SIZE"
    --rollout-token-budget "$ROLLOUT_TOKEN_BUDGET"
    --max-total-tokens "$MAX_TOTAL_TOKENS"
    --train-samples "$TRAIN_SAMPLES"
    --eval-samples "$samples"
    --max-turns "$MAX_TURNS"
    --python-timeout "$PYTHON_TIMEOUT"
    --tool-observation-limit "$TOOL_OBSERVATION_LIMIT"
    --no-trim-context
    --write-trace-logs
    --eval-limit "$EVAL_LIMIT"
    --aime-limit "$AIME_LIMIT"
    --generations "$GENERATIONS"
    --population "$POPULATION"
    --case-batch-size "$CASE_BATCH_SIZE"
    --sigma-start "$SIGMA_START"
    --sigma-end "$SIGMA_END"
    --sigma-schedule "$SIGMA_SCHEDULE"
    --sigma-warmup-steps "$SIGMA_WARMUP_STEPS"
    --alpha "$ALPHA"
    --seed "$SEED"
    --eval-seed "$EVAL_SEED"
    --parameter-scope "${ES_PARAMETER_SCOPE:-full}"
    --reward-normalization "${ES_REWARD_NORMALIZATION:-zscore}"
    --reward-normalization-ddof "${ES_REWARD_NORMALIZATION_DDOF:-0}"
    --reward-normalization-eps "${ES_REWARD_NORMALIZATION_EPS:-1e-8}"
    --max-tokens "$MAX_TOKENS"
    --temperature "${TEMPERATURE:-1.0}"
    --top-p "${TOP_P:-1.0}"
    --top-k "${TOP_K:-40}"
    --min-p "${MIN_P:-0.0}"
    --presence-penalty "${PRESENCE_PENALTY:-2.0}"
    --repetition-penalty "${REPETITION_PENALTY:-1.0}"
  )
}

run_eval() {
  local run_id="$1"
  local skill_file="${2:-}"
  shift 2
  [[ -f "$ES_HISTORY" ]] || { echo "Missing ES_HISTORY: $ES_HISTORY" >&2; exit 2; }
  runner_args "$EVAL_SAMPLES"
  local skill_args=()
  if [[ -n "$skill_file" ]]; then
    [[ -f "$skill_file" ]] || { echo "Missing SKILL_FILE: $skill_file" >&2; exit 2; }
    skill_args=(--skill-file "$skill_file")
  fi
  exec "$PY" -u "$RUNNER" \
    --run-id "$run_id" \
    --eval-only \
    --resume-history "$ES_HISTORY" \
    --resume-generations "$GENERATIONS" \
    "${COMMON_ARGS[@]}" \
    "${skill_args[@]}" \
    "$@"
}

case "$ACTION" in
  es-train)
    runner_args "$TRAIN_EVAL_SAMPLES"
    exec "$PY" -u "$RUNNER" \
      --run-id "$TRAIN_RUN_ID" \
      --eval-interval "$EVAL_INTERVAL" \
      "${COMMON_ARGS[@]}" \
      "$@"
    ;;
  eval)
    run_eval "$EVAL_RUN_ID" "" "$@"
    ;;
  distill-skill)
    [[ -f "$ES_HISTORY" ]] || { echo "Missing ES_HISTORY: $ES_HISTORY" >&2; exit 2; }
    [[ -d "$TRACE_ROOT" ]] || { echo "Missing no-skill TRACE_ROOT: $TRACE_ROOT" >&2; exit 2; }
    mkdir -p "$DISTILL_ROOT"
    if [[ ! -d "$ANALYSIS_LOGS" ]]; then
      if [[ "$TASK" == "math" ]]; then
        "$PY" "$ROOT/algorithms/trace2skill-settings/scripts/prepare_es_trajectory_logs.py" \
          math_reasoning \
          --trace-roots "$TRACE_ROOT" \
          --history "$ES_HISTORY" \
          --checkpoint-step "${DISTILL_CHECKPOINT_STEP:-$GENERATIONS}" \
          --task-count "${DISTILL_TASK_COUNT:-$((GENERATIONS * CASE_BATCH_SIZE))}" \
          --population "$POPULATION" \
          --case-batch-size "$CASE_BATCH_SIZE" \
          --one-error-per-task \
          --output-dir "$ANALYSIS_LOGS"
      else
        "$PY" "$ROOT/algorithms/trace2skill-settings/scripts/prepare_es_trajectory_logs.py" \
          docvqa \
          --history "$ES_HISTORY" \
          --trace-root "$TRACE_ROOT" \
          --checkpoint-step "${DISTILL_CHECKPOINT_STEP:-$GENERATIONS}" \
          --task-count "${DISTILL_TASK_COUNT:-50}" \
          --population "$POPULATION" \
          --one-per-outcome-per-task \
          --output-dir "$ANALYSIS_LOGS"
      fi
    fi
    exec "$PY" "$ROOT/algorithms/trace2skill-settings/scripts/evolve_from_trace_logs.py" \
      --setting "$SETTING" \
      --trace-logs "$ANALYSIS_LOGS" \
      --trajectory-manifest "$TRAJECTORY_MANIFEST" \
      --run-id "$DISTILL_RUN_ID" \
      --initial-skill "$INITIAL_SKILL" \
      --analysis-model "${TRACE2SKILL_ANALYSIS_MODEL:-${TRACE2SKILL_OPTIMIZER_MODEL:-gpt-5.4-nano}}" \
      --evolution-model "${TRACE2SKILL_EVOLUTION_MODEL:-${TRACE2SKILL_OPTIMIZER_MODEL:-gpt-5.4-nano}}" \
      --analysis-generation-config "${TRACE2SKILL_ANALYSIS_GENERATION_CONFIG:-{\"reasoning_effort\":\"none\"}}" \
      --evolution-generation-config "${TRACE2SKILL_EVOLUTION_GENERATION_CONFIG:-{\"reasoning_effort\":\"medium\"}}" \
      --workers "${TRACE2SKILL_WORKERS:-32}" \
      --max-skill-lines "${TRACE2SKILL_MAX_SKILL_LINES:-80}" \
      --max-references "${TRACE2SKILL_MAX_REFERENCES:-0}" \
      --evolution-temperature "${TRACE2SKILL_EVOLUTION_TEMPERATURE:-1.0}" \
      "$@"
    ;;
  skill-eval)
    run_eval "$SKILL_EVAL_RUN_ID" "$SKILL_FILE" "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
