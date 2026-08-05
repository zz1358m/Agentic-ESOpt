#!/usr/bin/env sh
set -eu

EPOCHS=${1:-3}
RUN_ID=${2:-skillopt_webarena_sft_epoch_eval}
TRAIN_PORT=${3:-11013}

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}
LOG_DIR=$ROOT/runs/skillopt_webarena_sft/${RUN_ID}_logs
OUT_ROOT=$ROOT/runs/skillopt_webarena_sft/$RUN_ID
mkdir -p "$LOG_DIR"

PORTS=${WEBARENA_EVAL_PORTS:-"11013 11014 11015 11016"}
STARTS="0 42 84 126"
ENDS="42 84 126 165"

epoch=1
while [ "$epoch" -le "$EPOCHS" ]; do
  echo "===== SkillOpt train through epoch $epoch/$EPOCHS ====="
  env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY \
    WEBARENA_SKILLOPT_LIMIT="${WEBARENA_SKILLOPT_LIMIT:-0}" \
    WEBARENA_SKILLOPT_TRAIN_SUBSET_SIZE="${WEBARENA_SKILLOPT_TRAIN_SUBSET_SIZE:-0}" \
    WEBARENA_SKILLOPT_TRAIN_SIZE="${WEBARENA_SKILLOPT_TRAIN_SIZE:-0}" \
    WEBARENA_SKILLOPT_BATCH_SIZE="${WEBARENA_SKILLOPT_BATCH_SIZE:-4}" \
    WEBARENA_SKILLOPT_SEL_ENV_NUM="${WEBARENA_SKILLOPT_SEL_ENV_NUM:-65}" \
    WEBARENA_SKILLOPT_MINIBATCH_SIZE="${WEBARENA_SKILLOPT_MINIBATCH_SIZE:-4}" \
    WEBARENA_SKILLOPT_ANALYST_WORKERS="${WEBARENA_SKILLOPT_ANALYST_WORKERS:-2}" \
    WEBARENA_SKILLOPT_WORKERS="${WEBARENA_SKILLOPT_WORKERS:-1}" \
    sh "$ROOT/webarena-train-time/scripts/run_skillopt_webarena_sft.sh" \
      "$epoch" "$RUN_ID" "$TRAIN_PORT" \
      > "$LOG_DIR/train_epoch_${epoch}.log" 2>&1

  skill=$OUT_ROOT/best_skill.md
  if [ ! -s "$skill" ]; then
    echo "missing skill after epoch $epoch: $skill" >&2
    exit 1
  fi

  echo "===== SkillOpt Lite-165 eval after epoch $epoch ====="
  idx=0
  pids=""
  for port in $PORTS; do
    start=$(echo "$STARTS" | awk -v n=$((idx + 1)) '{print $n}')
    end=$(echo "$ENDS" | awk -v n=$((idx + 1)) '{print $n}')
    if [ -z "$start" ] || [ -z "$end" ]; then
      break
    fi
    WEBRL_SKILL_FILE="$skill" \
      env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY \
      sh "$ROOT/webarena-train-time/scripts/run_webrl_lite_vab_full.sh" \
        skillopt "$port" "$start" "$end" "${RUN_ID}_epoch${epoch}" \
        > "$LOG_DIR/eval_epoch_${epoch}_${start}_${end}.log" 2>&1 &
    pids="$pids $!"
    idx=$((idx + 1))
  done
  for pid in $pids; do
    wait "$pid"
  done

  RUN_ID="$RUN_ID" EPOCH="$epoch" "$PY" - <<'PY'
import json, os
from pathlib import Path
root = Path("/home/zhi/Dynamic-Agent")
run_id = os.environ["RUN_ID"]
epoch = os.environ["EPOCH"]
parts = [(0,42),(42,84),(84,126),(126,165)]
scores = []
for start, end in parts:
    action_dir = root / "runs" / "webrl_lite_llama8b_full" / f"skillopt_{run_id}_epoch{epoch}_{start}_{end}" / "actions"
    for path in sorted(action_dir.glob("*.json"), key=lambda p: int(p.stem)):
        scores.append(float(json.loads(path.read_text()).get("score", 0.0)))
summary = {
    "epoch": int(epoch),
    "done": len(scores),
    "pass": sum(1 for score in scores if score >= 1.0),
    "average_score": sum(scores) / max(len(scores), 1),
}
summary["success_rate"] = summary["pass"] / max(summary["done"], 1)
out = root / "runs" / "skillopt_webarena_sft" / run_id / "lite165_epoch_summaries.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("a", encoding="utf-8") as f:
    f.write(json.dumps(summary, ensure_ascii=True) + "\n")
print(json.dumps(summary, indent=2))
PY

  epoch=$((epoch + 1))
done
