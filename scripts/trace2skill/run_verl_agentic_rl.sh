#!/usr/bin/env bash
set -euo pipefail

if [[ "${TRACE2SKILL_XTRACE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export ROOT

if [[ -f "${ROOT}/scripts/settings.local.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/settings.local.env"
fi

VERL_ROOT="${VERL_ROOT:-${ROOT}/verl}"
CONDA_ENV="${CONDA_ENV-grpo}"
PY="${PY:-python}"
TASK="${TASK:-math}"

case "${TASK}" in
  math)
    export MATH_PHYSICAL_GPU_IDS="${MATH_PHYSICAL_GPU_IDS:-3,4,5,6}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-20}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-20}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    TEST_FREQ="${TEST_FREQ:-5}"
    SAVE_FREQ="${SAVE_FREQ:-20}"
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
    # Async agent workers each own a one-CPU RewardManagerWorker.  The Ray
    # driver, four FSDP workers, and four embedded SGLang servers also reserve
    # CPUs, so the legacy 12-CPU default deadlocks five of eight reward actors.
    RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
    # Qwen3Next's hybrid recurrent-state pool is sized per concurrent request;
    # 0.85/1024 cannot initialize even on an 80 GiB A100.  These values tune
    # service capacity without changing any trajectory turn/token limit.
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${ROOT}/verl_trace2skill/math_bash_tool_config.yaml}"
    export TRACE2SKILL_PATCH_DENSE_QWEN3NEXT="${TRACE2SKILL_PATCH_DENSE_QWEN3NEXT:-1}"
    export TRACE2SKILL_REGISTER_TOOL_PARSER="${TRACE2SKILL_REGISTER_TOOL_PARSER:-1}"
    # A 512-token turn normally completes in well under this bound.  Retry
    # embedded SGLang/Ray replies that disappear after the server goes idle;
    # this does not alter the trajectory-level 100-turn/8192-token limits.
    export TRACE2SKILL_GENERATE_TIMEOUT_SECONDS="${TRACE2SKILL_GENERATE_TIMEOUT_SECONDS:-600}"
    export TRACE2SKILL_GENERATE_MAX_ATTEMPTS="${TRACE2SKILL_GENERATE_MAX_ATTEMPTS:-3}"
    export TRACE2SKILL_REWARD_TIMEOUT_SECONDS="${TRACE2SKILL_REWARD_TIMEOUT_SECONDS:-120}"
    export TRACE2SKILL_REWARD_MAX_ATTEMPTS="${TRACE2SKILL_REWARD_MAX_ATTEMPTS:-3}"
    DEFAULT_MODEL_PATH="${ROOT}/runs/docvqa_grpo/assets/Qwen3.5-4B-text"
    ;;
  docvqa)
    export DOCVQA_PHYSICAL_GPU_IDS="${DOCVQA_PHYSICAL_GPU_IDS:-auto}"
    export DOCVQA_TRAIN_LIMIT="${DOCVQA_TRAIN_LIMIT:-50}"
    # Use the first 100 rows of the held-out split for the aligned experiment.
    export DOCVQA_VAL_LIMIT="${DOCVQA_VAL_LIMIT:-100}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    TEST_FREQ="${TEST_FREQ:--1}"
    SAVE_FREQ="${SAVE_FREQ:-1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
    RAY_NUM_CPUS="${RAY_NUM_CPUS:-${PBS_NCPUS:-12}}"
    MAX_USER_TURNS="${MAX_USER_TURNS:-50}"
    MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-50}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-40960}"
    LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-40960}"
    REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-40960}"
    MULTI_TURN_FORMAT="${MULTI_TURN_FORMAT:-paper_react_cli}"
    TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-${ROOT}/verl_trace2skill/local_bash_tool_config.yaml}"
    ;;
  *)
    echo "TASK must be math or docvqa, got ${TASK}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${VERL_ROOT}/verl/trainer/main_ppo.py" ]]; then
  echo "VERL source not found at ${VERL_ROOT}. Clone the repository with its bundled verl/ directory or set VERL_ROOT." >&2
  exit 2
fi
if [[ ! -f "${ROOT}/verl_trace2skill/reward.py" ]]; then
  echo "verl_trace2skill integration not found under ${ROOT}." >&2
  exit 2
fi

if [[ -n "${CONDA_ENV}" ]]; then
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "Set CONDA_SH or put conda on PATH; use CONDA_ENV='' to use the current environment." >&2
    exit 3
  fi
  conda activate "${CONDA_ENV}"
fi

if [[ "${TASK}" == "docvqa" ]]; then
  requested_gpu_plan="${DOCVQA_PHYSICAL_GPU_IDS}"
  GPU_VISIBILITY_ARGS=(--physical-devices "${requested_gpu_plan}")
  if [[ -n "${DOCVQA_GPU_UUIDS:-}" ]]; then
    GPU_VISIBILITY_ARGS+=(--expected-uuids "${DOCVQA_GPU_UUIDS}")
  fi
  export DOCVQA_PHYSICAL_GPU_IDS="$(${PY} "${ROOT}/scripts/docvqa/gpu_visibility.py" \
    "${GPU_VISIBILITY_ARGS[@]}" --format physical)"
  GPU_VISIBILITY_ARGS=(--physical-devices "${DOCVQA_PHYSICAL_GPU_IDS}")
  if [[ -n "${DOCVQA_GPU_UUIDS:-}" ]]; then
    GPU_VISIBILITY_ARGS+=(--expected-uuids "${DOCVQA_GPU_UUIDS}")
  fi
  export CUDA_VISIBLE_DEVICES="$(${PY} "${ROOT}/scripts/docvqa/gpu_visibility.py" \
    "${GPU_VISIBILITY_ARGS[@]}" --format cuda)"
else
  requested_gpu_plan="${MATH_PHYSICAL_GPU_IDS}"
  GPU_VISIBILITY_ARGS=(--physical-devices "${requested_gpu_plan}")
  if [[ -n "${MATH_GPU_UUIDS:-}" ]]; then
    GPU_VISIBILITY_ARGS+=(--expected-uuids "${MATH_GPU_UUIDS}")
  fi
  export MATH_PHYSICAL_GPU_IDS="$(${PY} "${ROOT}/scripts/docvqa/gpu_visibility.py" \
    "${GPU_VISIBILITY_ARGS[@]}" --format physical)"
  GPU_VISIBILITY_ARGS=(--physical-devices "${MATH_PHYSICAL_GPU_IDS}")
  if [[ -n "${MATH_GPU_UUIDS:-}" ]]; then
    GPU_VISIBILITY_ARGS+=(--expected-uuids "${MATH_GPU_UUIDS}")
  fi
  export CUDA_VISIBLE_DEVICES="$(${PY} "${ROOT}/scripts/docvqa/gpu_visibility.py" \
    "${GPU_VISIBILITY_ARGS[@]}" --format cuda)"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TMPDIR="${TRACE2SKILL_TMPDIR:-${TMPDIR:-/tmp}}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"
SITE_PACKAGES="$(${PY} - <<'PY'
import site

paths = site.getsitepackages()
print(paths[0] if paths else site.getusersitepackages())
PY
)"

LIB_PARTS=()
GCC_LIB64="${GCC_LIB64:-}"
GCC_LIB="${GCC_LIB:-}"
if [[ -n "${GCC_LIB64}" && -d "${GCC_LIB64}" ]]; then
  LIB_PARTS+=("${GCC_LIB64}")
  if [[ -f "${GCC_LIB64}/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${GCC_LIB64}/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
fi
if [[ -n "${GCC_LIB}" && -d "${GCC_LIB}" ]]; then
  LIB_PARTS+=("${GCC_LIB}")
fi
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  LIB_PARTS+=("${CONDA_PREFIX}/lib")
fi
if [[ -d "${SITE_PACKAGES}/nvidia" ]]; then
  while IFS= read -r lib_dir; do
    LIB_PARTS+=("${lib_dir}")
  done < <(find "${SITE_PACKAGES}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null)
fi
if (( ${#LIB_PARTS[@]} )); then
  JOINED_LIB_PATH="$(IFS=:; echo "${LIB_PARTS[*]}")"
  export LD_LIBRARY_PATH="${JOINED_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH:-Qwen/Qwen3.5-4B}}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${MODEL_PATH}}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/trace2skill/verl}"
RUN_TAG="${RUN_TAG:-trace2skill-verl-qwen35-4b-${TASK}-$(date -u +%Y%m%d_%H%M%S)}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/runs/multiturn_grpo/checkpoints}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/multiturn_grpo/logs/${RUN_TAG}}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-${ROOT}/runs/multiturn_grpo/trajectories/${RUN_TAG}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${TRAJECTORY_ROOT}/train_raw}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${TRAJECTORY_ROOT}/validation_raw}"
export TRACE2SKILL_MATH_TOOL_CWD="${TRACE2SKILL_MATH_TOOL_CWD:-${TRAJECTORY_ROOT}/tool_workspace}"
mkdir -p "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}" "${TRACE2SKILL_MATH_TOOL_CWD}"
export TRACE2SKILL_TRITON_CACHE_ROOT="${TRACE2SKILL_TRITON_CACHE_ROOT:-${LOG_DIR}/triton_cache}"
mkdir -p "${TRACE2SKILL_TRITON_CACHE_ROOT}"

if [[ "${TASK}" == "docvqa" ]]; then
  "${PY}" "${ROOT}/scripts/docvqa/gpu_visibility.py" \
    "${GPU_VISIBILITY_ARGS[@]}" \
    --format json \
    --out "${LOG_DIR}/gpu_resources.json" >/dev/null
  "${PY}" "${ROOT}/scripts/docvqa/experiment_config.py" \
    --visible-devices "${DOCVQA_PHYSICAL_GPU_IDS}" \
    --effective-visible-devices "${CUDA_VISIBLE_DEVICES}" \
    --train-records "${DOCVQA_TRAIN_LIMIT}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --ppo-mini-batch-size "${PPO_MINI_BATCH_SIZE}" \
    --rollout-n "${ROLLOUT_N:-8}" \
    --epochs "${TOTAL_EPOCHS}" \
    --check-torch \
    --out "${LOG_DIR}/experiment_config.json"
else
  "${PY}" "${ROOT}/scripts/docvqa/gpu_visibility.py" \
    "${GPU_VISIBILITY_ARGS[@]}" \
    --format json \
    --out "${LOG_DIR}/gpu_resources.json" >/dev/null
  "${PY}" "${ROOT}/scripts/math/experiment_config.py" \
    --physical-gpus "${MATH_PHYSICAL_GPU_IDS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --ppo-mini-batch-size "${PPO_MINI_BATCH_SIZE}" \
    --rollout-n "${ROLLOUT_N:-8}" \
    --epochs "${TOTAL_EPOCHS}" \
    --world-size "${N_GPUS_PER_NODE}" \
    --test-freq "${TEST_FREQ}" \
    --save-freq "${SAVE_FREQ}" \
    --ray-num-cpus "${RAY_NUM_CPUS}" \
    --max-user-turns "${MAX_USER_TURNS:-100}" \
    --max-assistant-turns "${MAX_ASSISTANT_TURNS:-100}" \
    --max-response-length "${MAX_RESPONSE_LENGTH}" \
    --max-turn-response-length "${MAX_TURN_RESPONSE_LENGTH:-512}" \
    --rollout-data-dir "${ROLLOUT_DATA_DIR}" \
    --validation-data-dir "${VALIDATION_DATA_DIR}" \
    --tool-config-path "${TOOL_CONFIG_PATH}" \
    --model-path "${MODEL_PATH}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --learning-rate "${LR:-1e-6}" \
    --use-kl-loss "${USE_KL_LOSS:-True}" \
    --kl-loss-coef "${KL_LOSS_COEF:-0.001}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --top-p "${TOP_P:-1.0}" \
    --top-k "${TOP_K:-40}" \
    --presence-penalty "${PRESENCE_PENALTY:-2.0}" \
    --repetition-penalty "${REPETITION_PENALTY:-1.0}" \
    --data-shuffle "${DATA_SHUFFLE:-True}" \
    --data-seed "${DATA_SEED:-1}" \
    --val-before-train "${VAL_BEFORE_TRAIN:-True}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --generate-timeout-seconds "${TRACE2SKILL_GENERATE_TIMEOUT_SECONDS}" \
    --generate-max-attempts "${TRACE2SKILL_GENERATE_MAX_ATTEMPTS}" \
    --reward-timeout-seconds "${TRACE2SKILL_REWARD_TIMEOUT_SECONDS}" \
    --reward-max-attempts "${TRACE2SKILL_REWARD_MAX_ATTEMPTS}" \
    --parser-enabled "${TRACE2SKILL_REGISTER_TOOL_PARSER:-0}" \
    --dense-qwen3next-patch-enabled "${TRACE2SKILL_PATCH_DENSE_QWEN3NEXT:-0}" \
    --check-torch \
    --out "${LOG_DIR}/experiment_config.json" >/dev/null
fi

"${PY}" "${ROOT}/scripts/trace2skill/prepare_verl_trace2skill_data.py" \
  --task "${TASK}" \
  --out-dir "${DATA_DIR}"

# Adding the package root imports verl_trace2skill; adding the package directory
# itself exposes sitecustomize.py so every Ray worker registers the tool parser.
export PYTHONPATH="${ROOT}/verl_trace2skill:${ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${TASK}" == "docvqa" ]]; then
  export TRACE2SKILL_PATCH_DENSE_QWEN3NEXT=1
fi

nvidia-smi || true

cd "${VERL_ROOT}"

"${PY}" -m verl.trainer.main_ppo \
  --config-path="${VERL_ROOT}/examples/sglang_multiturn/config" \
  --config-name="gsm8k_multiturn_grpo" \
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
  +ray_kwargs.ray_init.include_dashboard=False \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_DIR}/${TASK}/train.parquet" \
  data.val_files="${DATA_DIR}/${TASK}/val.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.shuffle="${DATA_SHUFFLE:-True}" \
  +data.seed="${DATA_SEED:-1}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  data.return_multi_modal_inputs=False \
  data.trust_remote_code=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  custom_reward_function.path="${ROOT}/verl_trace2skill/reward.py" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels="${USE_FUSED_KERNELS:-False}" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend="${FUSED_KERNELS_BACKEND:-torch}" \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}" \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS:-True}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}" \
  actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode="${ROLLOUT_MODE:-async}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.85}" \
  actor_rollout_ref.rollout.multi_stage_wake_up="${MULTI_STAGE_WAKE_UP:-True}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${TOP_P:-1.0}" \
  actor_rollout_ref.rollout.top_k="${TOP_K:-40}" \
  +actor_rollout_ref.rollout.presence_penalty="${PRESENCE_PENALTY:-2.0}" \
  +actor_rollout_ref.rollout.repetition_penalty="${REPETITION_PENALTY:-1.0}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-40960}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-32768}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-1024}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-32768}" \
  actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS:-8}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS:-100}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS:-100}" \
  actor_rollout_ref.rollout.multi_turn.max_turn_response_length="${MAX_TURN_RESPONSE_LENGTH:-512}" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH:-6000}" \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
  actor_rollout_ref.rollout.multi_turn.format="${MULTI_TURN_FORMAT:-trace2skill}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K:-40}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-32768}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD:-True}" \
  +actor_rollout_ref.ref.model.path="${REF_MODEL_PATH}" \
  trainer.logger="${TRAINER_LOGGER:-[\"console\"]}" \
  trainer.project_name="${PROJECT_NAME:-trace2skill-agentic-rl}" \
  trainer.experiment_name="${RUN_TAG}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.save_freq="${SAVE_FREQ:-5}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-null}" \
  trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP:-null}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-null}" \
  "$@" 2>&1 | tee -a "${LOG_DIR}/train.log"

echo "ckpt_dir=${CKPT_DIR}"
echo "log_dir=${LOG_DIR}"
echo "rollout_data_dir=${ROLLOUT_DATA_DIR}"
echo "validation_data_dir=${VALIDATION_DATA_DIR}"
