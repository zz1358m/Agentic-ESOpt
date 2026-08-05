#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zhi/Dynamic-Agent}
PY=${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}
ENDPOINTS=${ENDPOINTS:-http://127.0.0.1:12013,http://127.0.0.1:12014,http://127.0.0.1:12015,http://127.0.0.1:12016}
CURRENT_PID=${CURRENT_PID:-3403197}
COSINE_RUN=${COSINE_RUN:-$ROOT/runs/webrl_lite_full_es/qwen35_27b_noskill_cosine1p75e3_to1p25e3_replay60_rerun_last10_g70_20260721_132316}
CONSTANT_RUN=${CONSTANT_RUN:-$ROOT/runs/webrl_lite_full_es/qwen35_27b_true_noskill_strict_vab_pop8_batch8_eval10_sigma1p5e3_alpha2p5e4_4gpu_20260630_045638}
EMPTY_SKILL=${EMPTY_SKILL:-$ROOT/runs/trace2skill_reanalysis/webarena_629_grouped_gpt54mini_unlimited_medium_20260721/skill_backup_20260721_083359/SKILL.md}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
SUITE_ROOT="$ROOT/runs/cosine_constant_trace2skill_suite/$STAMP"
REPLAY="$ROOT/scripts/webarena/replay_es_history_and_eval.py"
TRACE2SKILL="$ROOT/webarena-train-time/scripts/run_trace2skill_from_es_traces.py"

mkdir -p "$SUITE_ROOT" "$ROOT/logs"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

record() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$SUITE_ROOT/queue.log"
}

wait_for_current() {
  record "waiting for cosine training pid=$CURRENT_PID"
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep 30
  done
  "$PY" - "$COSINE_RUN/history.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
history = json.loads(path.read_text())
final = [row for row in history if row.get("generation") == 69]
if len(final) != 1 or not final[0].get("eval"):
    raise SystemExit(f"cosine run did not finish generation 69 with final eval: {path}")
PY
  record "cosine generation 70 and built-in eval complete"
}

eval_args() {
  printf '%s\n' \
    --endpoints "$ENDPOINTS" \
    --alpha 2.5e-4 \
    --reward-normalization zscore \
    --parameter-scope full \
    --eval-split data/webarena/vab_lite_split/items.json \
    --config-dir data/webarena/vab-lite/config_files/wa/test_webarena_lite \
    --eval-workers-per-endpoint 8 \
    --instruction-path agent/prompts/jsons/p_webrl_chat_qwen_action.json \
    --model-name Qwen3.5-27B \
    --mode chat \
    --stop-token "" \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --repetition-penalty 1.0
}

run_eval() {
  local run_id=$1
  local source_history=$2
  local generations=$3
  local repeats=$4
  local reset_mode=$5
  local skill_file=${6:-}
  local log="$ROOT/logs/${run_id}.log"
  local args=()
  mapfile -t args < <(eval_args)
  if [[ "$reset_mode" == "keep" ]]; then
    args+=(--skip-reset --skip-init)
  fi
  if [[ -n "$skill_file" ]]; then
    args+=(--skill-file "$skill_file")
  fi
  record "starting eval run_id=$run_id generations=$generations repeats=$repeats reset_mode=$reset_mode"
  env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY WEBRL_LOCAL_ENABLE_THINKING=false \
    "$PY" "$REPLAY" \
      --source-history "$source_history" \
      --run-id "$run_id" \
      --generations "$generations" \
      --eval-repeats "$repeats" \
      "${args[@]}" >"$log" 2>&1
  record "finished eval run_id=$run_id"
}

build_skill() {
  local label=$1
  local source_run=$2
  local run_id="${label}_last10_gpt54mini_unlimited_medium_${STAMP}"
  local out_root="$ROOT/runs/trace2skill_webarena_sft/$run_id"
  local log="$ROOT/logs/${run_id}.log"
  record "starting Trace2Skill label=$label source=$source_run"
  env \
    TRACE2SKILL_MAX_SKILL_LINES=0 \
    TRACE2SKILL_MAX_SKILL_TOKENS=0 \
    TRACE2SKILL_MAX_REFERENCES=0 \
    "$PY" "$TRACE2SKILL" \
      --es-run-dir "$source_run" \
      --run-id "$run_id" \
      --initial-skill "$EMPTY_SKILL" \
      --generations 10 \
      --max-traces 0 \
      --html-limit 12000 \
      --optimizer-model gpt-5.4-mini \
      --analysis-workers 16 \
      --analysis-reasoning-effort medium \
      --skill-reasoning-effort medium \
      --consolidation-reasoning-effort medium \
      --seed 20260721 >"$log" 2>&1
  test -s "$out_root/skill/SKILL.md"
  printf '%s\n' "$out_root/skill/SKILL.md" >"$SUITE_ROOT/${label}_skill_path.txt"
  record "finished Trace2Skill label=$label skill=$out_root/skill/SKILL.md"
}

cd "$ROOT"
record "suite start root=$SUITE_ROOT"
wait_for_current

# The cosine trainer already performs repeat 1 at generation 70; add repeats 2 and 3 in-place.
run_eval \
  "cosine_es_g70_noskill_additional2_${STAMP}" \
  "$COSINE_RUN/history.json" 0 2 keep

# Roll cosine updates back to base, replay the constant ES history, then evaluate it three times.
run_eval \
  "constant_es_g70_noskill_eval3_${STAMP}" \
  "$CONSTANT_RUN/history.json" 70 3 reset

# Restore the base policy before evaluating skills.
run_eval \
  "restore_base_after_constant_${STAMP}" \
  "$CONSTANT_RUN/history.json" 0 0 reset

build_skill cosine "$COSINE_RUN"
build_skill constant "$CONSTANT_RUN"

COSINE_SKILL=$(<"$SUITE_ROOT/cosine_skill_path.txt")
CONSTANT_SKILL=$(<"$SUITE_ROOT/constant_skill_path.txt")
run_eval \
  "cosine_last10_trace2skill_eval3_${STAMP}" \
  "$COSINE_RUN/history.json" 0 3 keep "$COSINE_SKILL"
run_eval \
  "constant_last10_trace2skill_eval3_${STAMP}" \
  "$CONSTANT_RUN/history.json" 0 3 keep "$CONSTANT_SKILL"

record "suite complete"
