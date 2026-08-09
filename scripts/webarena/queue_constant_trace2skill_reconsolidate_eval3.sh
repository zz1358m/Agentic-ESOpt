#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
ENDPOINTS=${ENDPOINTS:-http://127.0.0.1:12013,http://127.0.0.1:12014,http://127.0.0.1:12015,http://127.0.0.1:12016}
SOURCE_ANALYSIS=${SOURCE_ANALYSIS:-$ROOT/runs/trace2skill_webarena_sft/constant_last10_gpt54mini_unlimited_medium_20260721_152905/step_001/update_001}
EMPTY_SKILL=${EMPTY_SKILL:-$ROOT/runs/trace2skill_reanalysis/webarena_629_grouped_gpt54mini_unlimited_medium_20260721/skill_backup_20260721_083359/SKILL.md}
CONSTANT_HISTORY=${CONSTANT_HISTORY:-$ROOT/runs/webrl_lite_full_es/qwen35_27b_true_noskill_strict_vab_pop8_batch8_eval10_sigma1p5e3_alpha2p5e4_4gpu_20260630_045638/history.json}

RUN_ID="constant_last10_gpt54mini_reconsolidated_medium_${STAMP}"
OUT_ROOT="$ROOT/runs/trace2skill_reconsolidation/$RUN_ID"
SKILL_DIR="$OUT_ROOT/skill"
EVOLVE_LOG="$ROOT/logs/${RUN_ID}.log"
EVAL_RUN_ID="${RUN_ID}_eval3"
EVAL_LOG="$ROOT/logs/${EVAL_RUN_ID}.log"
STATUS_LOG="$OUT_ROOT/queue.log"

mkdir -p "$SKILL_DIR" "$OUT_ROOT/evolution_intermediates" "$ROOT/logs"
cp "$EMPTY_SKILL" "$SKILL_DIR/SKILL.md"

record() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS_LOG"
}

record "reconsolidation start source=$SOURCE_ANALYSIS"
cd "$ROOT/webarena-train-time/methods/trace2skill/source"
env \
  OPENAI_API_KEY="$(<"$ROOT/apikey")" \
  OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
  TRACE2SKILL_SKILL_DOMAIN="WebArena web-agent" \
  TRACE2SKILL_RUNTIME_POLICY="Use only the available WebRL actions and visible page evidence. Before exiting, satisfy the task's full completion contract, including requested cardinality, ties, multiple entities, and persisted state when applicable. When that complete answer is visible, exit immediately with it. When the requested state-changing task is fully and visibly confirmed complete, exit immediately. Prefer the shortest valid action sequence; learned guidance must not require extra exploration or verification after these completion conditions are satisfied." \
  "$PY" -m skill_evolver.run_parallel_combined_skill_evolution \
    --error-json "$SOURCE_ANALYSIS/error_analysis/parsed_error_records.json" \
    --success-json "$SOURCE_ANALYSIS/success_analysis/parsed_success_records.json" \
    --skill-dir "$SKILL_DIR" \
    --model gpt-5.4-mini \
    --base-url "${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
    --max-workers 16 \
    --batch-size 1 \
    --merge-batch-size 5 \
    --patch-pipeline json \
    --save-intermediates \
    --intermediates-dir "$OUT_ROOT/evolution_intermediates" \
    --changelog "$OUT_ROOT/change.log" \
    --seed 20260722 \
    --max-skill-lines 0 \
    --max-skill-tokens 0 \
    --max-references 0 \
    --max-verification-rounds 0 \
    --group-records-by-task \
    --generation-config '{"reasoning_effort":"medium"}' \
    --reasoning-effort medium \
    --consolidation-reasoning-effort medium >"$EVOLVE_LOG" 2>&1

test -s "$SKILL_DIR/SKILL.md"
record "reconsolidation complete skill=$SKILL_DIR/SKILL.md"
record "eval3 start run_id=$EVAL_RUN_ID"

cd "$ROOT"
env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY WEBRL_LOCAL_ENABLE_THINKING=false \
  "$PY" scripts/webarena/replay_es_history_and_eval.py \
    --source-history "$CONSTANT_HISTORY" \
    --run-id "$EVAL_RUN_ID" \
    --generations 0 \
    --eval-repeats 3 \
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
    --repetition-penalty 1.0 \
    --skill-file "$SKILL_DIR/SKILL.md" >"$EVAL_LOG" 2>&1

record "eval3 complete run_id=$EVAL_RUN_ID"
