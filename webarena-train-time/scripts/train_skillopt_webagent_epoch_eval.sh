#!/usr/bin/env sh
set -eu

EPOCHS=${1:-6}
RUN_ID=${2:-skillopt_epoch_eval}
MODEL_PORT=${3:-11013}
EVAL_PORT=${4:-11015}
TEST_LIMIT=${5:-0}

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python}
SKILLOPT=${SKILLOPT_ROOT:-$ROOT/webarena-train-time/methods/skillopt/source}
VAB=${VAB_ROOT:-$ROOT/data/webarena/vab-lite}
CONFIG=$SKILLOPT/configs/webagent_lite/default.yaml
SPLIT_SOURCE=${SKILLOPT_SPLIT_SOURCE:-$ROOT/data/webarena/skillopt_splits}
SPLIT_DIR=${SKILLOPT_SPLIT_DIR:-$ROOT/data/webarena/skillopt_splits_available}
OUT_ROOT=$ROOT/runs/skillopt_webagent_lite/${RUN_ID}
EVAL_SITES=shopping,shopping_admin,reddit,gitlab,wikipedia
OPTIMIZER_BACKEND=${SKILLOPT_OPTIMIZER_BACKEND:-openai_chat}
TARGET_BACKEND=${SKILLOPT_TARGET_BACKEND:-qwen_chat}
OPTIMIZER_MODEL=${SKILLOPT_OPTIMIZER_MODEL:-gpt-4.1-mini}
TARGET_MODEL=${SKILLOPT_TARGET_MODEL:-Llama-3.1-8B-Instruct}
TARGET_BASE_URL=${SKILLOPT_TARGET_BASE_URL:-http://127.0.0.1:${MODEL_PORT}/v1}

mkdir -p "$OUT_ROOT"

if [ ! -d "$SKILLOPT" ]; then
  echo "SkillOpt source not found: $SKILLOPT" >&2
  echo "Set SKILLOPT_ROOT or place SkillOpt at webarena-train-time/methods/skillopt/source." >&2
  exit 2
fi
if [ ! -d "$VAB" ]; then
  echo "VAB/WebArena-Lite source not found: $VAB" >&2
  echo "Set VAB_ROOT or place it at data/webarena/vab-lite." >&2
  exit 2
fi

"$PY" "$ROOT/webarena-train-time/scripts/prepare_standard_webarena_data.py"

"$PY" "$ROOT/webarena-train-time/scripts/prepare_webarena_lite_available_split.py" \
  --source "$SPLIT_SOURCE" \
  --output "$SPLIT_DIR"

OPENAI_KEY_FILE=$ROOT/apikey
if [ -s "$OPENAI_KEY_FILE" ]; then
  OPENAI_KEY=$(tr -d '\r\n' < "$OPENAI_KEY_FILE")
else
  OPENAI_KEY=${OPENAI_API_KEY:-}
fi

if [ "$OPTIMIZER_BACKEND" = "openai_chat" ] && [ -z "${OPENAI_KEY:-}" ]; then
  echo "missing OpenAI API key: $OPENAI_KEY_FILE or OPENAI_API_KEY" >&2
  exit 1
fi

export AZURE_OPENAI_ENDPOINT=https://api.openai.com/v1
export AZURE_OPENAI_API_KEY="${OPENAI_KEY:-dummy}"
export AZURE_OPENAI_AUTH_MODE=openai_compatible
export OPENAI_API_KEY="${OPENAI_KEY:-dummy}"
export WEBRL_EVAL_MODEL=${WEBRL_EVAL_MODEL:-gpt-4.1-mini}

cd "$SKILLOPT"

epoch=1
while [ "$epoch" -le "$EPOCHS" ]; do
  echo "===== SkillOpt train through epoch $epoch/$EPOCHS ====="
  "$PY" scripts/train.py \
    --config "$CONFIG" \
    --num_epochs "$epoch" \
    --out_root "$OUT_ROOT" \
    --split_dir "$SPLIT_DIR" \
    --optimizer_backend "$OPTIMIZER_BACKEND" \
    --target_backend "$TARGET_BACKEND" \
    --optimizer_model "$OPTIMIZER_MODEL" \
    --target_model "$TARGET_MODEL" \
    --target_qwen_chat_base_url "$TARGET_BASE_URL" \
    --cfg-options \
      env.webarena_root=$VAB \
      env.python=$PY \
      env.model_endpoint=http://127.0.0.1:${MODEL_PORT}/completions \
      env.model_name="$TARGET_MODEL" \
      env.instruction_path=agent/prompts/jsons/p_webrl.json \
      env.max_steps=30 \
      env.max_tokens=2048 \
      env.temperature=0.0 \
      env.top_p=0.9 \
      env.workers=1 \
      train.batch_size=4 \
      train.train_size=79 \
      gradient.minibatch_size=4 \
      evaluation.sel_env_num=8 \
      evaluation.test_env_num=8 \
      evaluation.eval_test=false

  skill="$OUT_ROOT/best_skill.md"
  if [ ! -s "$skill" ]; then
    echo "missing best skill after epoch $epoch: $skill" >&2
    exit 1
  fi

  echo "===== VAB test eval after epoch $epoch ====="
  cd "$ROOT"
  WEBRL_SKILL_FILE="$skill" \
    env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY \
    "$ROOT/webarena-train-time/scripts/run_webrl_lite_vab_subset.sh" \
      skillopt "$EVAL_PORT" "$EVAL_SITES" "${RUN_ID}_epoch${epoch}_test" "$TEST_LIMIT"
  cd "$SKILLOPT"

  epoch=$((epoch + 1))
done
