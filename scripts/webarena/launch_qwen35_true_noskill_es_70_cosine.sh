#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zhi/Dynamic-Agent}
PY=${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}
RUN_ID=${RUN_ID:-qwen35_27b_true_noskill_strict_vab_pop8_batch8_eval10_g70_sigma1p5e3_to5e4_cosine_warmup0_alpha2p5e4_4gpu_$(date -u +%Y%m%d_%H%M%S)}

cd "$ROOT"
mkdir -p logs

exec env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY \
  WEBRL_LOCAL_ENABLE_THINKING=false \
  "$PY" webarena-train-time/scripts/run_webrl_lite_distributed_es_train.py \
  --endpoints http://127.0.0.1:12013,http://127.0.0.1:12014,http://127.0.0.1:12015,http://127.0.0.1:12016 \
  --run-id "$RUN_ID" \
  --split data/webarena/jitrl_skillopt_splits/train/items.json \
  --train-config-dir data/webarena/vab-lite/config_files/wa/test_webarena \
  --eval-split data/webarena/vab_lite_split/items.json \
  --config-dir data/webarena/vab-lite/config_files/wa/test_webarena_lite \
  --episodes 0 \
  --generations 70 \
  --population 8 \
  --case-batch-size 8 \
  --case-workers-per-sample 8 \
  --eval-workers-per-endpoint 8 \
  --eval-interval 10 \
  --skip-initial-eval \
  --sigma 1.5e-3 \
  --sigma-schedule cosine-after-warmup \
  --sigma-warmup-steps 0 \
  --sigma-min-ratio 0.3333333333333333 \
  --alpha 2.5e-4 \
  --reward-normalization zscore \
  --parameter-scope full \
  --skill-file "" \
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
