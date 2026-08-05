#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhi/Dynamic-Agent"
PYTHON="/home/zhi/miniconda3/envs/es4llm/bin/python"
REPLAY_SCRIPT="$ROOT/scripts/webarena/replay_es_history_and_eval.py"
ENDPOINTS="http://127.0.0.1:12013,http://127.0.0.1:12014,http://127.0.0.1:12015,http://127.0.0.1:12016"
CONSTANT_HISTORY="$ROOT/runs/webrl_lite_full_es/qwen35_27b_true_noskill_strict_vab_pop8_batch8_eval10_sigma1p5e3_alpha2p5e4_4gpu_20260630_045638/history.json"
FINAL_SKILL="$ROOT/runs/webrl_lite_full_es/qwen35_27b_es_trace2skill_strict_vab_empty_pop8_batch8_eval10_g70_lines50_noref_webprompt_41mini_sigma1p5e3_constant_alpha2p5e4_4gpu_20260715_143725/skill/SKILL.md"

: "${WAIT_PID:?WAIT_PID must be set to the active decay eval process}"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

run_eval() {
  local label="$1"
  local generations="$2"
  local skill_file="$3"
  local timestamp run_id log_path
  local skill_args=()

  timestamp="$(date -u +%Y%m%d_%H%M%S)"
  run_id="qwen35_27b_${label}_eval3_${timestamp}"
  log_path="$ROOT/logs/${run_id}.log"
  if [[ -n "$skill_file" ]]; then
    skill_args=(--skill-file "$skill_file")
  fi

  printf '[queue] starting run_id=%s log=%s\n' "$run_id" "$log_path"
  env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY WEBRL_LOCAL_ENABLE_THINKING=false \
    "$PYTHON" "$REPLAY_SCRIPT" \
      --source-history "$CONSTANT_HISTORY" \
      --run-id "$run_id" \
      --endpoints "$ENDPOINTS" \
      --generations "$generations" \
      --alpha 2.5e-4 \
      --reward-normalization zscore \
      --parameter-scope full \
      --eval-split data/webarena/vab_lite_split/items.json \
      --config-dir data/webarena/vab-lite/config_files/wa/test_webarena_lite \
      --eval-workers-per-endpoint 8 \
      --eval-repeats 3 \
      --instruction-path agent/prompts/jsons/p_webrl_chat_qwen_action.json \
      --model-name Qwen3.5-27B \
      --mode chat \
      --stop-token "" \
      --temperature 0.7 \
      --top-p 0.8 \
      --top-k 20 \
      --min-p 0.0 \
      --presence-penalty 1.5 \
      --repetition-penalty 1.0 \
      "${skill_args[@]}" \
      >"$log_path" 2>&1
  printf '[queue] finished run_id=%s\n' "$run_id"
}

cd "$ROOT"
run_eval "constant_es_g70_noskill" 70 ""
run_eval "base_noskill" 0 ""
run_eval "base_trace2skill_final" 0 "$FINAL_SKILL"
printf '[queue] all follow-up evals finished\n'
