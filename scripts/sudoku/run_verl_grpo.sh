#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

SUDOKU_TARGET_MASK_COUNT="${SUDOKU_TARGET_MASK_COUNT:-50}"
VERL_TOOL_ROOT="${VERL_TOOL_ROOT:?Set VERL_TOOL_ROOT to your verl-tool checkout.}"
SUDOKU_VERL_DATA_DIR="${SUDOKU_VERL_DATA_DIR:-$ROOT/data/sudoku/verl/mask${SUDOKU_TARGET_MASK_COUNT}}"

"$PY" "$ROOT/sudoku-train-time/scripts/prepare_verl_data.py" \
  --output-dir "$SUDOKU_VERL_DATA_DIR" \
  --mask-count "$SUDOKU_TARGET_MASK_COUNT"

"$PY" "$ROOT/sudoku-train-time/scripts/install_verl_tool_adapter.py" \
  --verl-tool-root "$VERL_TOOL_ROOT" \
  --repo-root "$ROOT"

export DYNAMIC_AGENT_ROOT="$ROOT"
export VERL_RUN_ID="${RUN_ID:-sudoku_verl_grpo_mask${SUDOKU_TARGET_MASK_COUNT}}"

cd "$VERL_TOOL_ROOT"

host="${SUDOKU_TOOL_HOST:-127.0.0.1}"
port="${SUDOKU_TOOL_PORT:-5500}"
tool_server_url="http://${host}:${port}/get_observation"
mkdir -p logs
python -m verl_tool.servers.serve \
  --host "$host" \
  --port "$port" \
  --tool_type "sudoku" \
  --workers_per_tool "${SUDOKU_TOOL_WORKERS:-512}" \
  --max_concurrent_requests "${SUDOKU_TOOL_MAX_CONCURRENT:-8192}" \
  --router_workers "${SUDOKU_TOOL_ROUTER_WORKERS:-16}" \
  --use_ray=False \
  --log_level "${SUDOKU_TOOL_LOG_LEVEL:-info}" > logs/sudoku_tool_server.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

action_stop_tokens_file="$(mktemp)"
printf '\n' > "$action_stop_tokens_file"

python -m verl_tool.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$SUDOKU_VERL_DATA_DIR/train.parquet" \
  data.val_files="[$SUDOKU_VERL_DATA_DIR/eval.parquet]" \
  data.train_batch_size="${SUDOKU_VERL_BATCH_SIZE:-32}" \
  data.val_batch_size="${SUDOKU_VERL_VAL_BATCH_SIZE:-32}" \
  data.max_prompt_length="${SUDOKU_VERL_MAX_PROMPT_LENGTH:-1024}" \
  data.max_response_length="${SUDOKU_VERL_MAX_RESPONSE_LENGTH:-4096}" \
  data.truncation=right \
  reward_model.reward_manager=sudoku_binary \
  reward_model.launch_reward_fn_async=True \
  actor_rollout_ref.model.path="${SUDOKU_GRPO_MODEL:-meta-llama/Llama-3.1-8B-Instruct}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr="${SUDOKU_VERL_LR:-1e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${SUDOKU_VERL_PPO_MINI_BATCH:-32}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${SUDOKU_VERL_MICRO_BATCH:-1}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.strategy="${SUDOKU_VERL_STRATEGY:-fsdp}" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.agent.enable_agent=True \
  actor_rollout_ref.agent.tool_server_url="$tool_server_url" \
  actor_rollout_ref.agent.max_prompt_length="${SUDOKU_VERL_MAX_PROMPT_LENGTH:-1024}" \
  actor_rollout_ref.agent.max_response_length="${SUDOKU_VERL_MAX_RESPONSE_LENGTH:-4096}" \
  actor_rollout_ref.agent.max_start_length="${SUDOKU_VERL_MAX_PROMPT_LENGTH:-1024}" \
  actor_rollout_ref.agent.max_obs_length="${SUDOKU_VERL_MAX_OBS_LENGTH:-1024}" \
  actor_rollout_ref.agent.max_turns="${SUDOKU_VERL_MAX_TURNS:-90}" \
  actor_rollout_ref.agent.mask_observations=True \
  actor_rollout_ref.agent.action_stop_tokens="$action_stop_tokens_file" \
  actor_rollout_ref.agent.enable_mtrl=True \
  actor_rollout_ref.agent.max_action_length="${SUDOKU_VERL_MAX_ACTION_LENGTH:-32}" \
  +actor_rollout_ref.agent.retokenization=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${SUDOKU_VERL_NUM_GENERATIONS:-8}" \
  actor_rollout_ref.rollout.temperature="${SUDOKU_VERL_TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.top_p="${SUDOKU_VERL_TOP_P:-0.95}" \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${SUDOKU_VERL_TP:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${SUDOKU_VERL_GPU_MEMORY_UTILIZATION:-0.6}" \
  actor_rollout_ref.rollout.max_num_seqs="${SUDOKU_VERL_MAX_NUM_SEQS:-512}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  trainer.logger="${SUDOKU_VERL_LOGGER:-['console']}" \
  trainer.project_name="${SUDOKU_VERL_PROJECT:-dynamic-agent-sudoku}" \
  trainer.experiment_name="$VERL_RUN_ID" \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.rollout_data_dir="$ROOT/runs/sudoku_verl_step_records/$VERL_RUN_ID" \
  trainer.validation_data_dir=null \
  trainer.n_gpus_per_node="${SUDOKU_VERL_GPUS:-1}" \
  trainer.nnodes="${SUDOKU_VERL_NODES:-1}" \
  trainer.save_freq="${SUDOKU_VERL_SAVE_FREQ:-10}" \
  trainer.test_freq="${SUDOKU_VERL_TEST_FREQ:-10}" \
  trainer.total_epochs="${SUDOKU_VERL_TOTAL_EPOCHS:-10}" \
  ${SUDOKU_VERL_EXTRA_ARGS:-}
