#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhi/Dynamic-Agent"
PYTHON="/home/zhi/miniconda3/envs/es4llm/bin/python"
EVAL_SCRIPT="$ROOT/scripts/webarena/eval_skill_lite165.py"

: "${WAIT_PID:?WAIT_PID must be the active no-skill collection runner PID}"
: "${FINAL_RUN:?FINAL_RUN must point to the active Trace2Skill run directory}"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

FINAL_SKILL="$FINAL_RUN/skill/SKILL.md"
FINAL_SUMMARY="$FINAL_RUN/step_070/summary.json"
if [[ ! -s "$FINAL_SKILL" || ! -s "$FINAL_SUMMARY" ]]; then
  printf '[queue] final run did not finish successfully: %s\n' "$FINAL_RUN" >&2
  exit 1
fi

timestamp="$(date -u +%Y%m%d_%H%M%S)"
eval_root="$ROOT/runs/trace2skill_eval_repeats/final70_${timestamp}"
common_args=(
  --workers 32
  --temperature 0.7
  --top-p 0.8
  --top-k 20
  --min-p 0.0
  --presence-penalty 1.5
  --repetition-penalty 1.0
  --timeout 1200
  --max-steps 30
  --model-name Qwen3.5-27B
  --instruction-path agent/prompts/jsons/p_webrl_chat_qwen_action.json
  --model-endpoints "http://127.0.0.1:12013/completions http://127.0.0.1:12014/completions http://127.0.0.1:12015/completions http://127.0.0.1:12016/completions"
)

cd "$ROOT"
printf '[queue] starting two additional final-skill evals\n'
"$PYTHON" "$EVAL_SCRIPT" \
  --out-dir "$eval_root/final_skill" \
  --skill-file "$FINAL_SKILL" \
  --repeats 2 \
  "${common_args[@]}"

printf '[queue] starting three no-skill evals\n'
"$PYTHON" "$EVAL_SCRIPT" \
  --out-dir "$eval_root/no_skill" \
  --repeats 3 \
  "${common_args[@]}"

printf '[queue] all follow-up evals finished: %s\n' "$eval_root"
