#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
RUN_ID="${RUN_ID:-math_reasoning_es_vllm4gpu_$(date -u +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TRAIN_DATA="${MATH_TRAIN_DATA:-$ROOT/data/trace2skill/math_reasoning/dapo_evolve.jsonl}"
EVAL_DATA="${MATH_EVAL_DATA:-$ROOT/data/trace2skill/math_reasoning/dapo_test.jsonl}"
AIME_DATA="${MATH_AIME_DATA:-$ROOT/data/trace2skill/math_reasoning/aime_2026.jsonl}"
SKILL_FILE="${MATH_SKILL_FILE:-}"
NUM_ENGINES="${MATH_VLLM_NUM_ENGINES:-4}"
GPU_FRACTION="${MATH_VLLM_GPU_FRACTION:-1.0}"
GPU_MEMORY_UTILIZATION="${MATH_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MATH_VLLM_MAX_MODEL_LEN:-32768}"
ENFORCE_EAGER="${MATH_VLLM_ENFORCE_EAGER:-0}"
INFERENCE_BATCH_SIZE="${MATH_INFERENCE_BATCH_SIZE:-16}"
TRAIN_SAMPLES="${MATH_TRAIN_SAMPLES:-1}"
EVAL_SAMPLES="${MATH_EVAL_SAMPLES:-16}"
MAX_TURNS="${MATH_MAX_TURNS:-${MATH_MAX_REACT_ROUNDS:-0}}"
PYTHON_TIMEOUT="${MATH_PYTHON_TIMEOUT:-20.0}"
TOOL_OBSERVATION_LIMIT="${MATH_TOOL_OBSERVATION_LIMIT:-6000}"
EVAL_LIMIT="${MATH_EVAL_LIMIT:-100}"
AIME_LIMIT="${MATH_AIME_LIMIT:-30}"
MAX_TOKENS="${MATH_MAX_TOKENS:-0}"
VLLM_DEFAULT_MAX_TOKENS="${MATH_VLLM_DEFAULT_MAX_TOKENS:-4096}"
GDN_PREFILL_BACKEND="${MATH_VLLM_GDN_PREFILL_BACKEND:-triton}"
TEMPERATURE="${MATH_TEMPERATURE:-1.0}"
TOP_P="${MATH_TOP_P:-1.0}"
TOP_K="${MATH_TOP_K:-40}"
MIN_P="${MATH_MIN_P:-0.0}"
PRESENCE_PENALTY="${MATH_PRESENCE_PENALTY:-2.0}"
REPETITION_PENALTY="${MATH_REPETITION_PENALTY:-1.0}"
GENERATIONS="${MATH_ES_GENERATIONS:-1}"
POPULATION="${MATH_ES_POPULATION:-8}"
CASE_BATCH_SIZE="${MATH_ES_CASE_BATCH:-8}"
SIGMA_START="${MATH_ES_SIGMA_START:-5e-4}"
SIGMA_END="${MATH_ES_SIGMA_END:-$SIGMA_START}"
SIGMA_SCHEDULE="${MATH_ES_SIGMA_SCHEDULE:-constant}"
SIGMA_WARMUP_STEPS="${MATH_ES_SIGMA_WARMUP_STEPS:-0}"
ALPHA="${MATH_ES_ALPHA:-5e-4}"
PARAMETER_SCOPE="${MATH_ES_SCOPE:-full}"
REWARD_NORMALIZATION="${MATH_ES_REWARD_NORMALIZATION:-zscore}"
EVAL_INTERVAL="${MATH_ES_EVAL_INTERVAL:-1}"
SKIP_FINAL_INTERVAL_EVAL="${MATH_ES_SKIP_FINAL_INTERVAL_EVAL:-0}"
FINAL_EVAL="${MATH_FINAL_EVAL:-0}"
FINAL_EVAL_SAMPLES="${MATH_FINAL_EVAL_SAMPLES:-4}"
FINAL_EVAL_MAX_TURNS="${MATH_FINAL_EVAL_MAX_TURNS:-0}"
FINAL_EVAL_MAX_TOKENS="${MATH_FINAL_EVAL_MAX_TOKENS:-0}"
FINAL_EVAL_VLLM_DEFAULT_MAX_TOKENS="${MATH_FINAL_EVAL_VLLM_DEFAULT_MAX_TOKENS:-4096}"
RESUME_HISTORY="${MATH_ES_RESUME_HISTORY:-}"
HISTORY_FILE="${MATH_ES_HISTORY_FILE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FINAL_EVAL_ARGS=(--no-final-eval)
case "${FINAL_EVAL,,}" in
  1|true|yes|on) FINAL_EVAL_ARGS=(--final-eval) ;;
esac
ENFORCE_EAGER_ARGS=(--no-enforce-eager)
case "${ENFORCE_EAGER,,}" in
  1|true|yes|on) ENFORCE_EAGER_ARGS=(--enforce-eager) ;;
esac
SKIP_FINAL_INTERVAL_EVAL_ARGS=(--no-skip-final-interval-eval)
case "${SKIP_FINAL_INTERVAL_EVAL,,}" in
  1|true|yes|on) SKIP_FINAL_INTERVAL_EVAL_ARGS=(--skip-final-interval-eval) ;;
esac
RESUME_ARGS=()
if [[ -n "$RESUME_HISTORY" ]]; then
  RESUME_ARGS=(--resume-history "$RESUME_HISTORY")
fi
if [[ -n "$HISTORY_FILE" ]]; then
  RESUME_ARGS+=(--history-file "$HISTORY_FILE")
fi
LEGACY_EXTRA_ARGS=()
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a LEGACY_EXTRA_ARGS <<< "$EXTRA_ARGS"
fi

export CUDA_VISIBLE_DEVICES

cd "$ROOT"
mkdir -p logs

if [[ ! -f "$EVAL_DATA" || ! -f "$AIME_DATA" ]]; then
  cat >&2 <<EOF
Missing math eval data.

Expected:
  $EVAL_DATA
  $AIME_DATA
EOF
  exit 1
fi

exec "$PY" "$ROOT/math-train-time/scripts/run_math_es_vllm_train.py" \
  --run-id "$RUN_ID" \
  --train-data "$TRAIN_DATA" \
  --eval-data "$EVAL_DATA" \
  --aime-data "$AIME_DATA" \
  --skill-file "$SKILL_FILE" \
  --model-path "$MODEL_PATH" \
  --model "$MODEL_NAME" \
  --num-engines "$NUM_ENGINES" \
  --gpu-fraction "$GPU_FRACTION" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  "${ENFORCE_EAGER_ARGS[@]}" \
  --inference-batch-size "$INFERENCE_BATCH_SIZE" \
  --train-samples "$TRAIN_SAMPLES" \
  --eval-samples "$EVAL_SAMPLES" \
  --max-turns "$MAX_TURNS" \
  --python-timeout "$PYTHON_TIMEOUT" \
  --tool-observation-limit "$TOOL_OBSERVATION_LIMIT" \
  --eval-limit "$EVAL_LIMIT" \
  --aime-limit "$AIME_LIMIT" \
  --max-tokens "$MAX_TOKENS" \
  --vllm-default-max-tokens "$VLLM_DEFAULT_MAX_TOKENS" \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --min-p "$MIN_P" \
  --presence-penalty "$PRESENCE_PENALTY" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --generations "$GENERATIONS" \
  --population "$POPULATION" \
  --case-batch-size "$CASE_BATCH_SIZE" \
  --sigma-start "$SIGMA_START" \
  --sigma-end "$SIGMA_END" \
  --sigma-schedule "$SIGMA_SCHEDULE" \
  --sigma-warmup-steps "$SIGMA_WARMUP_STEPS" \
  --alpha "$ALPHA" \
  --parameter-scope "$PARAMETER_SCOPE" \
  --reward-normalization "$REWARD_NORMALIZATION" \
  --eval-interval "$EVAL_INTERVAL" \
  "${SKIP_FINAL_INTERVAL_EVAL_ARGS[@]}" \
  "${FINAL_EVAL_ARGS[@]}" \
  --final-eval-samples "$FINAL_EVAL_SAMPLES" \
  --final-eval-max-turns "$FINAL_EVAL_MAX_TURNS" \
  --final-eval-max-tokens "$FINAL_EVAL_MAX_TOKENS" \
  --final-eval-vllm-default-max-tokens "$FINAL_EVAL_VLLM_DEFAULT_MAX_TOKENS" \
  "${RESUME_ARGS[@]}" \
  "${LEGACY_EXTRA_ARGS[@]}" \
  "$@"
