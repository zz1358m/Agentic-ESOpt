#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}
RUN_ID=${RUN_ID:-qwen35_27b_noskill_full_epoch_es_pop16_batch4_sigma1e3_alpha1e3_$(date -u +%Y%m%d_%H%M%S)}
SIGMA_SCHEDULE=${SIGMA_SCHEDULE:-constant}
SIGMA_WARMUP_STEPS=${SIGMA_WARMUP_STEPS:--1}

cd "$ROOT"
mkdir -p logs

exec "$PY" webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py \
  --endpoints http://127.0.0.1:12013,http://127.0.0.1:12014,http://127.0.0.1:12015,http://127.0.0.1:12016 \
  --run-id "$RUN_ID" \
  --split data/webarena/jitrl_skillopt_splits/train/items.json \
  --train-config-dir data/webarena/vab-lite/config_files/wa/test_webarena \
  --eval-split data/webarena/vab_lite_split/items.json \
  --config-dir data/webarena/vab-lite/config_files/wa/test_webarena_lite \
  --episodes 0 \
  --generations 146 \
  --population 16 \
  --case-batch-size 4 \
  --case-workers-per-sample 4 \
  --eval-workers-per-endpoint 2 \
  --eval-interval 146 \
  --skip-initial-eval \
  --sigma 1e-3 \
  --sigma-schedule "$SIGMA_SCHEDULE" \
  --sigma-warmup-steps "$SIGMA_WARMUP_STEPS" \
  --alpha 1e-3 \
  --reward-normalization zscore \
  --parameter-scope full \
  --skill-file "" \
  --instruction-path agent/prompts/jsons/p_webrl_chat_qwen_action.json \
  --model-name Qwen3.5-27B \
  --mode chat \
  --stop-token ""
