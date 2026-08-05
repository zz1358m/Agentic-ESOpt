#!/usr/bin/env sh
set -eu

METHOD=${1:?method: base|base_es|skillopt|skillopt_es}
PORT=${2:?local completion port}
START=${3:-0}
END=${4:-165}
RUN_ID=${5:-run1}

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
VAB=${VAB_ROOT:-$ROOT/data/webarena/vab-lite}
LITE_CONFIG_ROOT=${LITE_CONFIG_ROOT:-$VAB/config_files/wa/test_webarena_lite}
PY=${PY:-python}
MODEL_LABEL=${MODEL_LABEL:-webrl-sft-llama-3.1-8b}
PROVIDER=${PROVIDER:-local_completion}
PLANNER_IP=${PLANNER_IP:-http://127.0.0.1:${PORT}}
INSTRUCTION_PATH=${INSTRUCTION_PATH:-agent/prompts/jsons/p_webrl.json}
MODE=${MODE:-completion}
STOP_TOKEN=${STOP_TOKEN:-<|eot_id|>}
RESULT_DIR=$ROOT/runs/webrl_lite_llama8b_full/${METHOD}_${RUN_ID}_${START}_${END}
OPENAI_KEY_FILE=$ROOT/apikey
OPENAI_KEY=dummy
if [ -s "$OPENAI_KEY_FILE" ]; then
  OPENAI_KEY=$(tr -d '\r\n' < "$OPENAI_KEY_FILE")
fi

mkdir -p "$RESULT_DIR"
if [ ! -d "$VAB" ]; then
  echo "VAB/WebArena-Lite source not found: $VAB" >&2
  echo "Set VAB_ROOT to the VisualAgentBench WebArena-Lite checkout or place it at data/webarena/vab-lite." >&2
  exit 2
fi
if [ ! -d "$LITE_CONFIG_ROOT" ]; then
  echo "WebArena-Lite config root not found: $LITE_CONFIG_ROOT" >&2
  echo "Run: python webarena-train-time/scripts/prepare_standard_webarena_data.py" >&2
  exit 2
fi

SKILL_ENV=""
case "$METHOD" in
  skillopt|skillopt_es)
    SKILL_ENV="WEBRL_SKILL_FILE=${WEBRL_SKILL_FILE:-$ROOT/webarena-train-time/skills/webrl_lite_skillopt_v1.md}"
    ;;
  base|base_es)
    ;;
  *)
    echo "unknown method $METHOD" >&2
    exit 1
    ;;
esac

cd "$VAB"
HOST=${WEBARENA_HOST:-127.0.0.1}
HOST=${HOST#http://}
HOST=${HOST#https://}
HOST=${HOST%/}
env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY \
  DATASET=webarena \
  SHOPPING=${SHOPPING:-http://${HOST}:7770} \
  SHOPPING_ADMIN=${SHOPPING_ADMIN:-http://${HOST}:7780/admin} \
  REDDIT=${REDDIT:-http://${HOST}:9999} \
  GITLAB=${GITLAB:-http://${HOST}:8023} \
  MAP=${MAP:-http://${HOST}:3000} \
  WIKIPEDIA=${WIKIPEDIA:-http://${HOST}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing} \
  HOMEPAGE=${HOMEPAGE:-http://${HOST}:4399} \
  OPENAI_API_KEY="$OPENAI_KEY" \
  WEBRL_EVAL_MODEL=${WEBRL_EVAL_MODEL:-gpt-4o-mini} \
  PYTHONPATH="$VAB" \
  $SKILL_ENV \
  "$PY" run.py \
    --instruction_path "$INSTRUCTION_PATH" \
    --test_start_idx "$START" \
    --test_end_idx "$END" \
    --result_dir "$RESULT_DIR" \
    --test_config_base_dir "$LITE_CONFIG_ROOT" \
    --provider "$PROVIDER" \
    --model "$MODEL_LABEL" \
    --mode "$MODE" \
    --planner_ip "$PLANNER_IP" \
    --stop_token "$STOP_TOKEN" \
    --temperature 0.0 \
    --max_obs_length 0 \
    --max_tokens 2048 \
    --viewport_width 1280 \
    --viewport_height 720 \
    --parsing_failure_th 5 \
    --repeating_action_failure_th 5 \
    --action_set_tag webrl_id \
    --observation_type webrl \
    --save_trace_enabled
