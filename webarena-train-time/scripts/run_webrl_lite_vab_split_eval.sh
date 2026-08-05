#!/usr/bin/env sh
set -eu

METHOD=${1:?method: base|skillopt}
PORT=${2:?local completion port}
SPLIT_JSON=${3:?split items.json}
RUN_ID=${4:?run id}
LIMIT=${5:-0}
OFFSET=${6:-0}

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
VAB=${VAB_ROOT:-$ROOT/data/webarena/vab-lite}
PY=${PY:-python}
MODEL_LABEL=${MODEL_LABEL:-Llama-3.1-8B-Instruct}
CONFIG_SRC=${LITE_CONFIG_ROOT:-$VAB/config_files/wa/test_webarena_lite}
PROVIDER=${PROVIDER:-openai}
PLANNER_IP=${PLANNER_IP:-http://127.0.0.1:${PORT}}
SUBSET_DIR=$ROOT/runs/webrl_lite_llama8b_full/config_subsets/${RUN_ID}_${METHOD}
RESULT_DIR=$ROOT/runs/webrl_lite_llama8b_full/${METHOD}_${RUN_ID}
OPENAI_KEY_FILE=$ROOT/apikey
OPENAI_KEY=dummy

if [ -s "$OPENAI_KEY_FILE" ]; then
  OPENAI_KEY=$(tr -d '\r\n' < "$OPENAI_KEY_FILE")
fi

if [ ! -d "$VAB" ]; then
  echo "VAB/WebArena-Lite source not found: $VAB" >&2
  echo "Set VAB_ROOT to the VisualAgentBench WebArena-Lite checkout or place it at data/webarena/vab-lite." >&2
  exit 2
fi
if [ ! -d "$CONFIG_SRC" ]; then
  echo "WebArena-Lite config root not found: $CONFIG_SRC" >&2
  echo "Run: python webarena-train-time/scripts/prepare_standard_webarena_data.py" >&2
  exit 2
fi
if [ -d "$SUBSET_DIR" ]; then
  case "$SUBSET_DIR" in
    "$ROOT"/runs/*) rm -rf "$SUBSET_DIR" ;;
    *) echo "Refusing to remove SUBSET_DIR outside runs: $SUBSET_DIR" >&2; exit 2 ;;
  esac
fi
mkdir -p "$SUBSET_DIR" "$RESULT_DIR"

SPLIT_JSON="$SPLIT_JSON" LIMIT="$LIMIT" OFFSET="$OFFSET" CONFIG_SRC="$CONFIG_SRC" SUBSET_DIR="$SUBSET_DIR" "$PY" - <<'PY'
import json
import os
import shutil
from pathlib import Path

items = json.loads(Path(os.environ["SPLIT_JSON"]).read_text())
limit = int(os.environ.get("LIMIT", "0"))
offset = int(os.environ.get("OFFSET", "0"))
config_src = Path(os.environ["CONFIG_SRC"])
subset_dir = Path(os.environ["SUBSET_DIR"])
tail = items[offset:]
selected = tail[:limit] if limit else tail

for idx, item in enumerate(selected):
    task_id = int(item["task_id"])
    shutil.copy2(config_src / f"{task_id}.json", subset_dir / f"{idx}.json")

manifest = {
    "split_json": os.environ["SPLIT_JSON"],
    "offset": offset,
    "count": len(selected),
    "source_task_ids": [int(item["task_id"]) for item in selected],
}
(subset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

COUNT=$(SUBSET_DIR="$SUBSET_DIR" "$PY" - <<'PY'
import json
import os
from pathlib import Path
print(json.loads((Path(os.environ["SUBSET_DIR"]) / "manifest.json").read_text())["count"])
PY
)

SKILL_ENV=""
case "$METHOD" in
  skillopt)
    SKILL_ENV="WEBRL_SKILL_FILE=${WEBRL_SKILL_FILE:-$ROOT/webarena-train-time/skills/webrl_lite_skillopt_v1.md}"
    ;;
  base)
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
    --instruction_path agent/prompts/jsons/p_webrl.json \
    --test_start_idx 0 \
    --test_end_idx "$COUNT" \
    --result_dir "$RESULT_DIR" \
    --test_config_base_dir "$SUBSET_DIR" \
    --provider "$PROVIDER" \
    --model "$MODEL_LABEL" \
    --mode completion \
    --planner_ip "$PLANNER_IP" \
    --stop_token "<|eot_id|>" \
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
