#!/usr/bin/env bash
set -euo pipefail

# Template launcher for the shared local Hugging Face model server.
# Example:
#   MODEL_PROFILE=qwen3_14b GPUS="0 1 2 3" PORT=11012 bash templates/local_models/start_model_server.sh

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
MODEL_PROFILE="${MODEL_PROFILE:-llama31_8b}"
PORT="${PORT:-11012}"
HOST="${HOST:-127.0.0.1}"
GPUS="${GPUS:-0}"
DTYPE="${DTYPE:-}"
MODEL_PATH="${MODEL_PATH:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-}"
CHAT_TEMPLATE_ENABLE_THINKING="${CHAT_TEMPLATE_ENABLE_THINKING:-}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-false}"
MAX_REPEAT_PROMPT="${MAX_REPEAT_PROMPT:-8}"

case "$MODEL_PROFILE" in
  llama31_8b)
    MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
    DTYPE="${DTYPE:-float16}"
    TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
    CHAT_TEMPLATE_ENABLE_THINKING="${CHAT_TEMPLATE_ENABLE_THINKING:-auto}"
    ;;
  qwen3_14b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
    DTYPE="${DTYPE:-bfloat16}"
    TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
    CHAT_TEMPLATE_ENABLE_THINKING="${CHAT_TEMPLATE_ENABLE_THINKING:-false}"
    ;;
  qwen3_32b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
    DTYPE="${DTYPE:-bfloat16}"
    TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
    CHAT_TEMPLATE_ENABLE_THINKING="${CHAT_TEMPLATE_ENABLE_THINKING:-false}"
    ;;
  *)
    echo "Unknown MODEL_PROFILE: $MODEL_PROFILE" >&2
    exit 2
    ;;
esac

args=(
  "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py"
  --path "$MODEL_PATH"
  --port "$PORT"
  --host "$HOST"
  --dtype "$DTYPE"
  --max-repeat-prompt "$MAX_REPEAT_PROMPT"
  --chat-template-enable-thinking "$CHAT_TEMPLATE_ENABLE_THINKING"
)

if [ -n "$GPUS" ] && [ "$GPUS" != "cpu" ]; then
  args+=(--d $GPUS)
else
  args+=(--d cpu)
fi

if [ "$TRUST_REMOTE_CODE" = "true" ]; then
  args+=(--trust-remote-code)
fi
if [ "$LOAD_IN_4BIT" = "true" ]; then
  args+=(--load-in-4bit)
fi
if [ "$LOAD_IN_8BIT" = "true" ]; then
  args+=(--quantization)
fi

cd "$ROOT"
exec "$PY" "${args[@]}"
