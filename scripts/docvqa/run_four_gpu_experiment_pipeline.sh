#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUNS_ROOT="${DOCVQA_RUNS_ROOT:-${ROOT}/runs/docvqa_grpo}"
PY="${PY:-python}"

if ! command -v "${PY}" >/dev/null 2>&1 && [[ ! -x "${PY}" ]]; then
  echo "Python environment is unavailable: ${PY}" >&2
  exit 2
fi

export DOCVQA_PHYSICAL_GPU_IDS="${DOCVQA_PHYSICAL_GPU_IDS:-auto}"
export DOCVQA_GPU_UUIDS="${DOCVQA_GPU_UUIDS:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT}/algorithms/verl_trace2skill:${ROOT}:${ROOT}/algorithms/verl${PYTHONPATH:+:${PYTHONPATH}}"

gpu_args=(--physical-devices "${DOCVQA_PHYSICAL_GPU_IDS}")
if [[ -n "${DOCVQA_GPU_UUIDS}" ]]; then
  gpu_args+=(--expected-uuids "${DOCVQA_GPU_UUIDS}")
fi
export DOCVQA_PHYSICAL_GPU_IDS="$(${PY} "${ROOT}/scripts/docvqa/gpu_visibility.py" \
  "${gpu_args[@]}" --format physical)"
gpu_args=(--physical-devices "${DOCVQA_PHYSICAL_GPU_IDS}")
if [[ -n "${DOCVQA_GPU_UUIDS}" ]]; then
  gpu_args+=(--expected-uuids "${DOCVQA_GPU_UUIDS}")
fi
export CUDA_VISIBLE_DEVICES="$(${PY} "${ROOT}/scripts/docvqa/gpu_visibility.py" \
  "${gpu_args[@]}" --format cuda)"

BASE_MODEL="${RUNS_ROOT}/assets/Qwen3.5-4B-text"
PRE_EVAL_DIR="${RUNS_ROOT}/pre_eval"
DATA_DIR="${RUNS_ROOT}/full_data"
CKPT_ROOT="${RUNS_ROOT}/checkpoints"
RUN_TAG="qwen35-4b-docvqa-grpo-four-gpu-seed42"
CKPT_DIR="${CKPT_ROOT}/${RUN_TAG}"
TRAIN_LOG_DIR="${RUNS_ROOT}/full_train_logs/${RUN_TAG}"
POST_EVAL_DIR="${RUNS_ROOT}/post_eval"
REPORT_DIR="${RUNS_ROOT}/report"
PIPELINE_DIR="${RUNS_ROOT}/pipeline"
STATUS_FILE="${PIPELINE_DIR}/pipeline_status.tsv"
FINAL_HF="${CKPT_DIR}/global_step_180/actor/huggingface"
CURRENT_STAGE="initialization"

mkdir -p "${PIPELINE_DIR}" "${TRAIN_LOG_DIR}" "${REPORT_DIR}"

status() {
  printf '%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${STATUS_FILE}"
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  if [[ "${exit_code}" -eq 0 ]]; then
    status "COMPLETE stage=${CURRENT_STAGE}"
  else
    status "FAILED stage=${CURRENT_STAGE} exit_code=${exit_code}"
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Converted base model is unavailable: ${BASE_MODEL}" >&2
  exit 2
fi
if [[ ! -f "${ROOT}/data/trace2skill/docvqa/test.jsonl" ]]; then
  echo "Aligned DocVQA test data is unavailable" >&2
  exit 2
fi

CURRENT_STAGE="four_gpu_resource_check"
status "START stage=${CURRENT_STAGE}"
"${PY}" "${ROOT}/scripts/docvqa/gpu_visibility.py" \
  "${gpu_args[@]}" \
  --format json \
  --out "${PIPELINE_DIR}/gpu_resources_start.json" >/dev/null
"${PY}" "${ROOT}/scripts/docvqa/experiment_config.py" \
  --visible-devices "${DOCVQA_PHYSICAL_GPU_IDS}" \
  --effective-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --train-records 50 \
  --train-batch-size 4 \
  --ppo-mini-batch-size 4 \
  --rollout-n 8 \
  --epochs 15 \
  --world-size 4 \
  --check-torch \
  --out "${PIPELINE_DIR}/experiment_config_start.json" >/dev/null
"${PY}" - <<'PY'
import ray

ray.init(include_dashboard=False, num_cpus=1, log_to_driver=False)
try:
    detected = int(ray.cluster_resources().get("GPU", 0))
    if detected != 4:
        raise RuntimeError(f"Ray sees {detected} GPUs; expected 4")
    print(f"Ray resource check PASS: GPUs={detected}")
finally:
    ray.shutdown()
PY
status "PASS stage=${CURRENT_STAGE} torch_gpus=4 ray_gpus=4 physical_gpus=${DOCVQA_PHYSICAL_GPU_IDS}"

CURRENT_STAGE="pre_eval"
status "START stage=${CURRENT_STAGE} resume=true expected_records=400 requested_concurrency=8"
"${PY}" "${ROOT}/scripts/docvqa/run_four_gpu_eval.py" \
  --model-path "${BASE_MODEL}" \
  --out-dir "${PRE_EVAL_DIR}" \
  --docvqa-root "${ROOT}" \
  --python "${PY}" \
  --port-base 18080 \
  --samples 4 \
  --limit 100 \
  --seed 42 \
  --concurrency 8 \
  --fallback-concurrency 4 \
  --context-length 131072 \
  --physical-gpus "${DOCVQA_PHYSICAL_GPU_IDS}" \
  --gpu-uuids "${DOCVQA_GPU_UUIDS}" \
  --resume
EVAL_CONCURRENCY="$(${PY} -c 'import json,sys; print(json.load(open(sys.argv[1]))["concurrency"])' \
  "${PRE_EVAL_DIR}/four_gpu_manifest.json")"
if [[ "${EVAL_CONCURRENCY}" != "8" && "${EVAL_CONCURRENCY}" != "4" ]]; then
  echo "Unexpected selected evaluation concurrency: ${EVAL_CONCURRENCY}" >&2
  exit 3
fi
status "PASS stage=${CURRENT_STAGE} records=400 selected_concurrency=${EVAL_CONCURRENCY}"

CURRENT_STAGE="training"
LATEST_FILE="${CKPT_DIR}/latest_checkpointed_iteration.txt"
LATEST_STEP=""
if [[ -f "${LATEST_FILE}" ]]; then
  LATEST_STEP="$(tr -d '[:space:]' < "${LATEST_FILE}")"
fi
if [[ "${LATEST_STEP}" != "180" ]]; then
  status "START stage=${CURRENT_STAGE} resume_step=${LATEST_STEP:-0} target_step=180"
  CONDA_ENV="" \
  PY="${PY}" \
  MODEL_PATH="${BASE_MODEL}" \
  DATA_DIR="${DATA_DIR}" \
  CKPT_ROOT="${CKPT_ROOT}" \
  CKPT_DIR="${CKPT_DIR}" \
  LOG_DIR="${TRAIN_LOG_DIR}" \
  RUN_TAG="${RUN_TAG}" \
  DOCVQA_TRAIN_LIMIT=50 \
  DOCVQA_VAL_LIMIT=100 \
  TRAIN_BATCH_SIZE=4 \
  PPO_MINI_BATCH_SIZE=4 \
  PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
  ROLLOUT_N=8 \
  TOTAL_EPOCHS=15 \
  DATA_SEED=42 \
  LR=1e-6 \
  KL_LOSS_COEF=0.001 \
  TEMPERATURE=1.0 \
  TOP_P=1.0 \
  TOP_K=40 \
  PRESENCE_PENALTY=2.0 \
  MAX_MODEL_LEN=131072 \
  MAX_RESPONSE_LENGTH=32768 \
  MAX_TURN_RESPONSE_LENGTH=512 \
  MAX_USER_TURNS=50 \
  MAX_ASSISTANT_TURNS=50 \
  N_GPUS_PER_NODE=4 \
  AGENT_LOOP_WORKERS=4 \
  GPU_MEMORY_UTILIZATION=0.30 \
  MAX_NUM_SEQS=16 \
  USE_FUSED_KERNELS=True \
  FUSED_KERNELS_BACKEND=torch \
  SAVE_FREQ=1 \
  MAX_ACTOR_CKPT_TO_KEEP=1 \
  VERL_PROTECTED_CHECKPOINT_STEPS=60,120,180 \
  TEST_FREQ=-1 \
  VAL_BEFORE_TRAIN=False \
  MULTI_TURN_FORMAT=paper_react_cli \
  bash "${ROOT}/scripts/docvqa/run_grpo.sh"
else
  status "SKIP stage=${CURRENT_STAGE} reason=global_step_180_already_complete"
fi

if [[ ! -f "${LATEST_FILE}" ]] || [[ "$(tr -d '[:space:]' < "${LATEST_FILE}")" != "180" ]]; then
  echo "Training did not reach global step 180" >&2
  exit 3
fi
REQUIRED_CHECKPOINTS=(global_step_60 global_step_120 global_step_180)
for checkpoint in "${REQUIRED_CHECKPOINTS[@]}"; do
  if [[ ! -d "${CKPT_DIR}/${checkpoint}/actor/huggingface" ]]; then
    echo "Required Hugging Face checkpoint is missing: ${checkpoint}" >&2
    exit 3
  fi
done
status "PASS stage=${CURRENT_STAGE} final_step=180 checkpoints=60,120,180"

CURRENT_STAGE="checkpoint_verification"
status "START stage=${CURRENT_STAGE} checkpoint=${FINAL_HF}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
  "${PY}" "${ROOT}/scripts/docvqa/verify_text_checkpoint.py" \
  "${FINAL_HF}" \
  --device cuda:0 \
  --out "${REPORT_DIR}/final_checkpoint_verification.json"
status "PASS stage=${CURRENT_STAGE}"

CURRENT_STAGE="post_eval"
status "START stage=${CURRENT_STAGE} resume=true expected_records=400 concurrency=${EVAL_CONCURRENCY}"
"${PY}" "${ROOT}/scripts/docvqa/run_four_gpu_eval.py" \
  --model-path "${FINAL_HF}" \
  --out-dir "${POST_EVAL_DIR}" \
  --docvqa-root "${ROOT}" \
  --python "${PY}" \
  --port-base 19080 \
  --samples 4 \
  --limit 100 \
  --seed 42 \
  --concurrency "${EVAL_CONCURRENCY}" \
  --fallback-concurrency "${EVAL_CONCURRENCY}" \
  --strict-concurrency \
  --context-length 131072 \
  --physical-gpus "${DOCVQA_PHYSICAL_GPU_IDS}" \
  --gpu-uuids "${DOCVQA_GPU_UUIDS}" \
  --resume
status "PASS stage=${CURRENT_STAGE} records=400"

CURRENT_STAGE="report"
status "START stage=${CURRENT_STAGE}"
"${PY}" "${ROOT}/scripts/docvqa/report_grpo_experiment.py" \
  --before "${PRE_EVAL_DIR}/outputs/docvqa.jsonl" \
  --after "${POST_EVAL_DIR}/outputs/docvqa.jsonl" \
  --json-out "${REPORT_DIR}/final_report.json" \
  --markdown-out "${REPORT_DIR}/final_report.md" \
  --bootstrap-samples 10000 \
  --seed 42
status "PASS stage=${CURRENT_STAGE} report=${REPORT_DIR}/final_report.md"
