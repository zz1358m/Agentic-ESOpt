#!/usr/bin/env sh
set -eu

EPOCHS=${1:-1}
RUN_ID=${2:-skillopt_webarena_sft_smoke}
MODEL_PORT=${3:-11013}

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}
SKILLOPT=$ROOT/webarena-train-time/methods/skillopt/source
OUT_ROOT=$ROOT/runs/skillopt_webarena_sft/$RUN_ID
SPLIT_DIR=$ROOT/data/webarena/skillopt_nonlite_sft

"$PY" "$ROOT/webarena-train-time/scripts/prepare_webarena_nonlite_skillopt_split.py" \
  --output-dir "$SPLIT_DIR"

TRAIN_SUBSET_SIZE=${WEBARENA_SKILLOPT_TRAIN_SUBSET_SIZE:-0}
if [ "$TRAIN_SUBSET_SIZE" -gt 0 ]; then
  BASE_SPLIT_DIR=$SPLIT_DIR
  SPLIT_DIR=$OUT_ROOT/split_train${TRAIN_SUBSET_SIZE}_valfull
  BASE_SPLIT_DIR="$BASE_SPLIT_DIR" SPLIT_DIR="$SPLIT_DIR" TRAIN_SUBSET_SIZE="$TRAIN_SUBSET_SIZE" "$PY" - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["BASE_SPLIT_DIR"])
out = Path(os.environ["SPLIT_DIR"])
n = int(os.environ["TRAIN_SUBSET_SIZE"])
for split in ("train", "val", "test"):
    items = json.loads((base / split / "items.json").read_text(encoding="utf-8"))
    if split == "train":
        items = items[:n]
    target = out / split / "items.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(items, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
manifest = {
    "source_split_dir": str(base.resolve()),
    "train_subset_size": n,
    "counts": {
        split: len(json.loads((out / split / "items.json").read_text(encoding="utf-8")))
        for split in ("train", "val", "test")
    },
    "note": "Train is subsetted for runtime. Val is full non-Lite validation; test is empty because Lite-165 is evaluated externally.",
}
(out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY
fi

OPENAI_KEY_FILE=$ROOT/apikey
if [ -s "$OPENAI_KEY_FILE" ]; then
  OPENAI_KEY=$(tr -d '\r\n' < "$OPENAI_KEY_FILE")
else
  OPENAI_KEY=${OPENAI_API_KEY:-}
fi

export AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-https://api.openai.com/v1}
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-${OPENAI_KEY:-dummy}}"
export AZURE_OPENAI_AUTH_MODE=${AZURE_OPENAI_AUTH_MODE:-openai_compatible}
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENAI_KEY:-dummy}}"

cd "$SKILLOPT"
"$PY" -m pip install -e . >/dev/null

"$PY" scripts/train.py \
  --config configs/webarena_sft/default.yaml \
  --num_epochs "$EPOCHS" \
  --out_root "$OUT_ROOT" \
  --split_dir "$SPLIT_DIR" \
  --optimizer_backend openai_chat \
  --optimizer_model "${OPTIMIZER_MODEL:-gpt-4.1-mini}" \
    --target_model "${TARGET_MODEL:-Qwen3-14B}" \
  --cfg-options \
    env.webarena_root="$ROOT/data/webarena/vab-lite" \
    env.python="$PY" \
    env.model_endpoint="http://127.0.0.1:${MODEL_PORT}/completions" \
    env.model_endpoints="${WEBARENA_SKILLOPT_MODEL_ENDPOINTS:-http://127.0.0.1:12013/completions http://127.0.0.1:12014/completions http://127.0.0.1:12015/completions http://127.0.0.1:12016/completions}" \
    env.model_name="${TARGET_MODEL:-Qwen3-14B}" \
    env.instruction_path="${WEBARENA_INSTRUCTION_PATH:-agent/prompts/jsons/p_webrl_chat.json}" \
    env.mode="${WEBARENA_MODE:-chat}" \
    env.stop_token="${WEBARENA_STOP_TOKEN:-}" \
    env.local_enable_thinking="${WEBARENA_LOCAL_ENABLE_THINKING:-false}" \
    env.max_steps="${WEBARENA_MAX_STEPS:-30}" \
    env.workers="${WEBARENA_SKILLOPT_WORKERS:-1}" \
    env.limit="${WEBARENA_SKILLOPT_LIMIT:-0}" \
    model.reasoning_effort="" \
    model.rewrite_reasoning_effort="" \
    train.batch_size="${WEBARENA_SKILLOPT_BATCH_SIZE:-4}" \
    train.train_size="${WEBARENA_SKILLOPT_TRAIN_SIZE:-0}" \
    gradient.minibatch_size="${WEBARENA_SKILLOPT_MINIBATCH_SIZE:-4}" \
    gradient.analyst_workers="${WEBARENA_SKILLOPT_ANALYST_WORKERS:-2}" \
    evaluation.sel_env_num="${WEBARENA_SKILLOPT_SEL_ENV_NUM:-8}" \
    ${WEBARENA_SKILLOPT_EXTRA_CFG:-}
