#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MATH_PID="${MATH_PID:-}"
DOCVQA_PID="${DOCVQA_PID:-}"
MATH_RUN_ID="${MATH_RUN_ID:-qwen35_4b_dapo_failed2219_t2s_gpt54nano_anone_smedium_20260806}"
DOCVQA_RUN_ID="${DOCVQA_RUN_ID:-qwen35_4b_docvqa_all800_taskprompts_t2s_gpt54nano_anone_smedium_v2_20260806}"
EVAL_RUN_ID="${EVAL_RUN_ID:-qwen35_4b_distilled_skills_math_docvqa_eval4_$(date -u +%Y%m%d_%H%M%S)}"

T2S_ROOT="${T2S_ROOT:-${ROOT}/runs/trace2skill_extra}"
MATH_SKILL_FILE="${T2S_ROOT}/${MATH_RUN_ID}/skill_step_001.md"
DOCVQA_SKILL_FILE="${T2S_ROOT}/${DOCVQA_RUN_ID}/skill_step_001.md"
SKILL_VALIDATOR="${SKILL_VALIDATOR:-}"
QUEUE_LOG="${ROOT}/logs/${EVAL_RUN_ID}_queue.log"

mkdir -p "${ROOT}/logs"
exec >>"$QUEUE_LOG" 2>&1

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

count_reports() {
  local directory="$1"
  local pattern="$2"
  if [[ ! -d "$directory" ]]; then
    printf '0'
    return
  fi
  find "$directory" -maxdepth 1 -name "$pattern" ! -name '*_prompt.md' | wc -l
}

is_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

echo "[$(timestamp)] waiting math_pid=${MATH_PID} docvqa_pid=${DOCVQA_PID}"
while is_running "$MATH_PID" || is_running "$DOCVQA_PID"; do
  math_reports=$(count_reports "${T2S_ROOT}/${MATH_RUN_ID}/step_001/update_001/error_analysis" 'error_analysis_*.md')
  doc_error_reports=$(count_reports "${T2S_ROOT}/${DOCVQA_RUN_ID}/step_001/update_001/error_analysis" 'error_analysis_*.md')
  doc_success_reports=$(count_reports "${T2S_ROOT}/${DOCVQA_RUN_ID}/step_001/update_001/success_analysis" 'success_analysis_*.md')
  echo "[$(timestamp)] progress math_error=${math_reports}/2219 doc_error=${doc_error_reports}/470 doc_success=${doc_success_reports}/330"
  sleep 30
done

for skill_file in "$MATH_SKILL_FILE" "$DOCVQA_SKILL_FILE"; do
  if [[ ! -s "$skill_file" ]]; then
    echo "[$(timestamp)] missing distilled skill: $skill_file"
    exit 3
  fi
  if [[ -n "$SKILL_VALIDATOR" ]]; then
    python3 "$SKILL_VALIDATOR" "$(dirname "$skill_file")/skill"
  fi
done

if rg -n -i '\b(document|image|ocr|tesseract|docvqa)\b' "$MATH_SKILL_FILE"; then
  echo "[$(timestamp)] math domain audit failed"
  exit 4
fi
if rg -n -i '\b(spreadsheet|workbook|browser|webarena|webpage|dom|click)\b' "$DOCVQA_SKILL_FILE"; then
  echo "[$(timestamp)] docvqa domain audit failed"
  exit 5
fi

echo "[$(timestamp)] distilled skills validated; starting eval run_id=${EVAL_RUN_ID}"
exec env \
  RUN_ID="$EVAL_RUN_ID" \
  MATH_SKILL_FILE="$MATH_SKILL_FILE" \
  DOCVQA_SKILL_FILE="$DOCVQA_SKILL_FILE" \
  TRACE2SKILL_EVAL_DATASETS="dapo100,aime2026,docvqa" \
  TRACE2SKILL_EVAL_SAMPLES=4 \
  TRACE2SKILL_EVAL_DOCVQA_LIMIT=100 \
  TRACE2SKILL_EVAL_MATH_MAX_TOKENS=4096 \
  TRACE2SKILL_EVAL_MATH_MAX_TURNS=50 \
  TRACE2SKILL_EVAL_DOCVQA_MAX_TOKENS=512 \
  TRACE2SKILL_EVAL_DOCVQA_MAX_TOTAL_TOKENS=32768 \
  TRACE2SKILL_EVAL_DOCVQA_MAX_TURNS=50 \
  bash "${ROOT}/scripts/trace2skill/eval16_react_4gpu_vllm.sh"
