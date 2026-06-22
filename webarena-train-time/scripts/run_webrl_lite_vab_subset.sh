#!/usr/bin/env sh
set -eu

METHOD=${1:?method: base|base_es|skillopt|skillopt_es}
PORT=${2:?local completion port}
SITES=${3:?comma-separated allowed sites, e.g. shopping,shopping_admin,reddit}
RUN_ID=${4:-subset1}
LIMIT=${5:-0}

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
VAB=${VAB_ROOT:-$ROOT/data/webarena/vab-lite}
LITE_CONFIG_ROOT=${LITE_CONFIG_ROOT:-$VAB/config_files/wa/test_webarena_lite}
PY=${PY:-python}
MODEL_LABEL=${MODEL_LABEL:-Llama-3.1-8B-Instruct}
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
if [ ! -d "$LITE_CONFIG_ROOT" ]; then
  echo "WebArena-Lite config root not found: $LITE_CONFIG_ROOT" >&2
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

LITE_CONFIG_ROOT="$LITE_CONFIG_ROOT" METHOD="$METHOD" SITES="$SITES" LIMIT="$LIMIT" SUBSET_DIR="$SUBSET_DIR" "$PY" - <<'PY'
import json
import os
import shutil
from pathlib import Path

src = Path(os.environ["LITE_CONFIG_ROOT"])
dst = Path(os.environ["SUBSET_DIR"])
allowed = {s.strip() for s in os.environ["SITES"].split(",") if s.strip()}
limit = int(os.environ.get("LIMIT", "0"))

selected = []
for path in sorted(src.glob("*.json"), key=lambda p: int(p.stem)):
    data = json.load(open(path))
    sites = set(data.get("sites", []))
    if sites and sites.issubset(allowed):
        selected.append(path)
        if limit and len(selected) >= limit:
            break

for idx, path in enumerate(selected):
    shutil.copy2(path, dst / f"{idx}.json")

manifest = {
    "allowed_sites": sorted(allowed),
    "count": len(selected),
    "source_task_ids": [int(p.stem) for p in selected],
}
(dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

COUNT=$(SUBSET_DIR="$SUBSET_DIR" "$PY" - <<'PY'
import json
import os
from pathlib import Path
manifest = json.load(open(Path(os.environ["SUBSET_DIR"]) / "manifest.json"))
print(manifest["count"])
PY
)

if [ "$COUNT" -le 0 ]; then
  echo "no tasks selected for sites: $SITES" >&2
  exit 1
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
