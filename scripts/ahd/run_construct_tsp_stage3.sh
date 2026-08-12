#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/bayp/anaconda3/envs/esopt_recheck_py310/bin/python}"
MODEL="${MODEL:-/mnt/data7t/ES4LLM/data/model/Llama-3.1-8B-Instruct}"
RUN_ROOT="${RUN_ROOT:-}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
EXPECTED_REMOTE_URL="${EXPECTED_REMOTE_URL:-https://github.com/zz1358m/Agentic-ESOpt.git}"
EXPECTED_REMOTE_HEAD="${EXPECTED_REMOTE_HEAD:-1068ab4250d8d5875c9d6092a8e494de19d30e6c}"
EXPECTED_ENV_PREFIX="${EXPECTED_ENV_PREFIX:-/home/bayp/anaconda3/envs/esopt_recheck_py310}"
EXPECTED_ENV_ID="${EXPECTED_ENV_ID:-esopt_recheck_py310}"
UPSTREAM_BASE_COMMIT="${UPSTREAM_BASE_COMMIT:-1068ab4250d8d5875c9d6092a8e494de19d30e6c}"
INTEGRATION_BASE_COMMIT="${INTEGRATION_BASE_COMMIT:-780e7f2969ba7101eb16e780cf8f34f58fcb76a4}"
AHD_STAGE1_GATE="${AHD_STAGE1_GATE:-}"
AHD_STAGE2_GATE="${AHD_STAGE2_GATE:-}"
EXPECTED_MODEL_CONFIG_SHA256="${EXPECTED_MODEL_CONFIG_SHA256:-29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e}"
EXPECTED_MODEL_INDEX_SHA256="${EXPECTED_MODEL_INDEX_SHA256:-146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b}"
EXPECTED_TOKENIZER_CONFIG_SHA256="${EXPECTED_TOKENIZER_CONFIG_SHA256:-177c7b61e616fecb84c17ce0591acb92c6d60e9ac5ababfb940ff23bbcd424}"
EXPECTED_MODEL_SHARD_SHA256=(
  2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668
  09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15
  fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa
  92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b
)
PORTS=(11013 11014 11015 11016 11017 11018 11019 11020)
GPUS=(0 1 2 3 4 5 6 7)
ENDPOINTS="http://127.0.0.1:11013/completions,http://127.0.0.1:11014/completions,http://127.0.0.1:11015/completions,http://127.0.0.1:11016/completions,http://127.0.0.1:11017/completions,http://127.0.0.1:11018/completions,http://127.0.0.1:11019/completions,http://127.0.0.1:11020/completions"
TOOL="$ROOT/scripts/ahd/construct_tsp_stage3.py"
LAUNCHER="$ROOT/scripts/ahd/run_ahd_1000.sh"
SERVER_LAUNCHER="$ROOT/scripts/ahd/start_llama31_8b_servers.sh"
ACTIVE_SERVICE_ROOT=""

die() {
  echo "[stage3] ERROR: $*" >&2
  exit 1
}

usage() {
  echo "usage: RUN_ROOT=/absolute/audit/root EXPECTED_COMMIT=<sha> $0 {plan|preflight|smoke|formal|cleanup}" >&2
  exit 2
}

require_runtime_inputs() {
  [[ -n "$RUN_ROOT" && "$RUN_ROOT" = /* ]] || die "RUN_ROOT must be an absolute, dedicated audit directory"
  [[ -n "$EXPECTED_COMMIT" ]] || die "EXPECTED_COMMIT is required; pin the reviewed local commit"
  [[ -n "$AHD_STAGE1_GATE" && -n "$AHD_STAGE2_GATE" ]] || die "AHD_STAGE1_GATE and AHD_STAGE2_GATE are required"
}

port_is_busy() {
  local port="$1"
  ss -ltn "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q .
}

file_value() {
  tr -d '\r\n' < "$1"
}

verify_stage_gate() {
  local gate="$1" stage="$2" pip_sha="$3" conda_sha="$4"
  [[ -d "$gate" ]] || die "$stage gate directory is missing: $gate"
  for file in acceptance.md exit_status.txt repo_commit.txt upstream_base_commit.txt \
    integration_base_commit.txt task_environment_id.txt task_environment_path.txt \
    python_executable.txt pip_freeze.txt conda_explicit.txt validation.json; do
    [[ -s "$gate/$file" ]] || die "$stage gate artifact is missing: $gate/$file"
  done
  [[ "$(file_value "$gate/exit_status.txt")" = "0" ]] || die "$stage did not exit successfully"
  "$PY" - "$gate/validation.json" "$stage" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
status = str(payload.get("status", "")).upper()
if status not in {"PASS", "COMPLETE"}:
    raise SystemExit(f"{sys.argv[2]} validation status is not fail-closed PASS/COMPLETE: {status!r}")
PY
  [[ "$(file_value "$gate/repo_commit.txt")" = "$EXPECTED_COMMIT" ]] || die "$stage commit differs from Stage 3"
  [[ "$(file_value "$gate/upstream_base_commit.txt")" = "$UPSTREAM_BASE_COMMIT" ]] || die "$stage upstream base differs"
  [[ "$(file_value "$gate/integration_base_commit.txt")" = "$INTEGRATION_BASE_COMMIT" ]] || die "$stage integration base differs"
  [[ "$(file_value "$gate/task_environment_id.txt")" = "$EXPECTED_ENV_ID" ]] || die "$stage environment ID differs"
  [[ "$(file_value "$gate/task_environment_path.txt")" = "$EXPECTED_ENV_PREFIX" ]] || die "$stage environment path differs"
  [[ "$(file_value "$gate/python_executable.txt")" = "$PY" ]] || die "$stage Python executable differs"
  [[ "$(sha256sum "$gate/pip_freeze.txt" | awk '{print $1}')" = "$pip_sha" ]] || die "$stage pip freeze differs"
  [[ "$(sha256sum "$gate/conda_explicit.txt" | awk '{print $1}')" = "$conda_sha" ]] || die "$stage Conda explicit list differs"
}

preflight() {
  require_runtime_inputs
  [[ -x "$PY" ]] || die "frozen Python is not executable: $PY"
  [[ -f "$MODEL/config.json" ]] || die "frozen model is incomplete: $MODEL/config.json"
  [[ -x "$LAUNCHER" && -x "$SERVER_LAUNCHER" ]] || die "canonical AHD launchers are not executable"
  git -C "$ROOT" fetch --all --prune
  [[ "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT" ]] || die "local commit differs from EXPECTED_COMMIT"
  git -C "$ROOT" diff --quiet -- . || die "tracked worktree changes are present"
  git -C "$ROOT" diff --cached --quiet -- . || die "staged worktree changes are present"
  [[ "$(git -C "$ROOT" remote get-url origin)" = "$EXPECTED_REMOTE_URL" ]] || die "origin URL differs"
  [[ "$(git -C "$ROOT" ls-remote origin refs/heads/main | awk '{print $1}')" = "$EXPECTED_REMOTE_HEAD" ]] || die "origin/main differs or is unreachable"
  [[ "$EXPECTED_REMOTE_HEAD" = "$UPSTREAM_BASE_COMMIT" ]] || die "remote head differs from frozen upstream base"
  git -C "$ROOT" merge-base --is-ancestor "$UPSTREAM_BASE_COMMIT" "$EXPECTED_COMMIT" || die "task commit is not an upstream descendant"
  git -C "$ROOT" merge-base --is-ancestor "$INTEGRATION_BASE_COMMIT" "$EXPECTED_COMMIT" || die "task commit is not an integration descendant"
  [[ "$($PY -c 'import sys; print(sys.prefix)')" = "$EXPECTED_ENV_PREFIX" ]] || die "Python environment prefix differs"
  [[ "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = "$EXPECTED_MODEL_CONFIG_SHA256" ]] || die "model config SHA-256 differs"
  [[ "$(sha256sum "$MODEL/model.safetensors.index.json" | awk '{print $1}')" = "$EXPECTED_MODEL_INDEX_SHA256" ]] || die "model index SHA-256 differs"
  [[ "$(sha256sum "$MODEL/tokenizer_config.json" | awk '{print $1}')" = "$EXPECTED_TOKENIZER_CONFIG_SHA256" ]] || die "tokenizer config SHA-256 differs"
  for shard_index in 1 2 3 4; do
    shard="$(printf '%s/model-%05d-of-00004.safetensors' "$MODEL" "$shard_index")"
    [[ "$(sha256sum "$shard" | awk '{print $1}')" = "${EXPECTED_MODEL_SHARD_SHA256[$((shard_index - 1))]}" ]] || die "model shard $shard_index SHA-256 differs"
  done
  mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  [[ "${#gpu_rows[@]}" -eq 8 ]] || die "exactly 8 physical GPUs are required"
  [[ "$(printf '%s\n' "${gpu_rows[@]}" | awk -F, '{gsub(/ /,"",$2); print $2}' | sort -u | wc -l)" -eq 8 ]] || die "GPU UUIDs are not unique"
  for index in "${!gpu_rows[@]}"; do
    [[ "${gpu_rows[$index]%%,*}" = "$index" ]] || die "GPU index mapping is not exactly 0..7"
    used_memory="$(awk -F, '{gsub(/ /,"",$3); print $3}' <<< "${gpu_rows[$index]}")"
    [[ "$used_memory" -le 100 ]] || die "GPU $index is not idle: ${used_memory} MiB used"
  done
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]] || die "existing GPU compute processes are present"
  for port in "${PORTS[@]}"; do
    port_is_busy "$port" && die "frozen port $port is already occupied"
  done

  mkdir -p "$RUN_ROOT/preflight"
  "$PY" -m pip freeze > "$RUN_ROOT/preflight/pip_freeze.txt"
  conda list -p "$EXPECTED_ENV_PREFIX" --explicit > "$RUN_ROOT/preflight/conda_explicit.txt"
  pip_sha="$(sha256sum "$RUN_ROOT/preflight/pip_freeze.txt" | awk '{print $1}')"
  conda_sha="$(sha256sum "$RUN_ROOT/preflight/conda_explicit.txt" | awk '{print $1}')"
  verify_stage_gate "$AHD_STAGE1_GATE" "Stage 1" "$pip_sha" "$conda_sha"
  verify_stage_gate "$AHD_STAGE2_GATE" "Stage 2" "$pip_sha" "$conda_sha"
  {
    echo "checked_at=$(date --iso-8601=seconds)"
    echo "repo=$ROOT"
    echo "commit=$EXPECTED_COMMIT"
    echo "remote_url=$EXPECTED_REMOTE_URL"
    echo "remote_main=$EXPECTED_REMOTE_HEAD"
    echo "upstream_base_commit=$UPSTREAM_BASE_COMMIT"
    echo "integration_base_commit=$INTEGRATION_BASE_COMMIT"
    echo "python=$PY"
    echo "environment=$EXPECTED_ENV_PREFIX"
    echo "task_environment_id=$EXPECTED_ENV_ID"
    echo "python_executable=$PY"
    echo "pip_freeze_sha256=$pip_sha"
    echo "conda_explicit_sha256=$conda_sha"
    echo "model=$MODEL"
    echo "model_config_sha256=$EXPECTED_MODEL_CONFIG_SHA256"
    echo "model_index_sha256=$EXPECTED_MODEL_INDEX_SHA256"
    echo "tokenizer_config_sha256=$EXPECTED_TOKENIZER_CONFIG_SHA256"
    echo "model_shard_sha256=${EXPECTED_MODEL_SHARD_SHA256[*]}"
    echo "gpus=${GPUS[*]}"
    echo "ports=${PORTS[*]}"
    echo "endpoints=$ENDPOINTS"
    echo "stage1_gate=$AHD_STAGE1_GATE"
    echo "stage2_gate=$AHD_STAGE2_GATE"
    echo "evidence_boundary=8 physical GPUs with one endpoint each is the closest reconstruction supported by the original 8-client log; the original physical GPU mapping was not recorded and is not claimed as proven."
  } > "$RUN_ROOT/preflight/frozen_inputs.txt"
  printf '%s\n' "${gpu_rows[@]}" > "$RUN_ROOT/preflight/gpu_uuid_map.csv"
  git -C "$ROOT" status --short > "$RUN_ROOT/preflight/git_status.txt"
  git -C "$ROOT" log --oneline "$INTEGRATION_BASE_COMMIT..$EXPECTED_COMMIT" > "$RUN_ROOT/preflight/commit_list.txt"
  git -C "$ROOT" diff --stat "$INTEGRATION_BASE_COMMIT...$EXPECTED_COMMIT" > "$RUN_ROOT/preflight/integration_diff_stat.txt"
  git -C "$ROOT" diff --stat "$UPSTREAM_BASE_COMMIT...$EXPECTED_COMMIT" > "$RUN_ROOT/preflight/upstream_diff_stat.txt"
  git -C "$ROOT" diff "$INTEGRATION_BASE_COMMIT...$EXPECTED_COMMIT" > "$RUN_ROOT/preflight/integration.diff"
  git -C "$ROOT" diff "$UPSTREAM_BASE_COMMIT...$EXPECTED_COMMIT" > "$RUN_ROOT/preflight/upstream.diff"
  env | sort > "$RUN_ROOT/preflight/environment_variables.txt"
  nvidia-smi > "$RUN_ROOT/preflight/nvidia_smi.txt"
  "$PY" - <<'PY' > "$RUN_ROOT/preflight/cuda_environment.txt"
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_devices={torch.cuda.device_count()}")
PY
  sha256sum "$AHD_STAGE1_GATE/acceptance.md" "$AHD_STAGE1_GATE/validation.json" \
    "$AHD_STAGE2_GATE/acceptance.md" "$AHD_STAGE2_GATE/validation.json" > "$RUN_ROOT/preflight/stage_gate_sha256.txt"
  sha256sum "$ROOT/scripts/ahd/run_ahd_1000.sh" "$ROOT/scripts/ahd/run_four_method_ahd.sh" \
    "$ROOT/ahd-test-time/scripts/run_eoh_ahd.py" "$ROOT/scripts/ahd/run_construct_tsp_stage3.sh" \
    "$ROOT/scripts/ahd/construct_tsp_stage3.py" "$ROOT/algorithms/es/model_es_client.py" \
    "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/methods/eoh/eoh_interface_EC.py" \
    > "$RUN_ROOT/preflight/task_code_sha256.txt"
  "$PY" "$TOOL" plan > "$RUN_ROOT/preflight/protocol_plan.json"
  (cd "$MODEL" && find . -maxdepth 1 -type f -printf '%P,%s\n' | sort) > "$RUN_ROOT/preflight/model_files_and_sizes.csv"
  find "$RUN_ROOT/preflight" -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/preflight/SHA256SUMS"
  echo "[stage3] preflight PASS: $RUN_ROOT/preflight"
}

stop_owned_servers() {
  local service_root="$1"
  local pid starttime pgid gpu port command current_pgid current_starttime
  local ownership="$service_root/ownership.csv"
  [[ -s "$ownership" ]] || return 0
  while IFS=, read -r pid starttime pgid gpu port; do
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || continue
    command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    read -r current_pgid current_starttime < <(awk '{print $5, $22}' "/proc/$pid/stat")
    if [[ "$current_starttime" = "$starttime" && "$current_pgid" = "$pgid" && "$pgid" = "$pid" \
      && "$command" == *"llama31_instruct_server.py"* && "$command" == *"--port $port"* \
      && "$command" == *"--d $gpu"* && "$command" == *"--path $MODEL"* ]]; then
      kill -TERM -- "-$pgid" 2>/dev/null || true
    else
      echo "[stage3] refuse to signal reused or mismatched pid=$pid" >&2
    fi
  done < "$ownership"
  for _ in {1..30}; do
    local live=0
    while IFS=, read -r pid starttime pgid gpu port; do
      if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/stat" ]]; then
        current_starttime="$(awk '{print $22}' "/proc/$pid/stat")"
        [[ "$current_starttime" = "$starttime" ]] && live=1
      fi
    done < "$ownership"
    [[ "$live" -eq 0 ]] && return 0
    sleep 1
  done
  die "owned servers did not terminate within 30 seconds: $service_root"
}

capture_owned_servers() {
  local service_root="$1" gpu port pid pid_file command pgid starttime
  local ownership="$service_root/ownership.csv"
  : > "$ownership"
  for gpu in "${GPUS[@]}"; do
    port=$((11013 + gpu))
    pid_file="$service_root/logs/server_gpu${gpu}_port${port}.pid"
    [[ -s "$pid_file" ]] || continue
    pid="$(<"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || continue
    command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    read -r pgid starttime < <(awk '{print $5, $22}' "/proc/$pid/stat")
    if [[ "$pgid" = "$pid" && "$command" == *"llama31_instruct_server.py"* \
      && "$command" == *"--port $port"* && "$command" == *"--d $gpu"* \
      && "$command" == *"--path $MODEL"* ]]; then
      printf '%s,%s,%s,%s,%s\n' "$pid" "$starttime" "$pgid" "$gpu" "$port" >> "$ownership"
    fi
  done
}

cleanup_on_exit() {
  local status=$?
  if [[ -n "$ACTIVE_SERVICE_ROOT" ]]; then
    stop_owned_servers "$ACTIVE_SERVICE_ROOT" || true
  fi
  exit "$status"
}

start_servers() {
  local service_root="$1"
  local launcher_status pid_file_count
  ACTIVE_SERVICE_ROOT="$service_root"
  mkdir -p "$service_root"
  set +e
  MODEL="$MODEL" PY="$PY" RUN_ROOT="$service_root" \
    PORTS="${PORTS[*]}" GPUS="${GPUS[*]}" \
    "$SERVER_LAUNCHER" | tee "$service_root/startup.txt"
  launcher_status="${PIPESTATUS[0]}"
  set -e
  capture_owned_servers "$service_root"
  [[ "$launcher_status" -eq 0 ]] || die "server launcher failed with status $launcher_status"
  pid_file_count="$(find "$service_root/logs" -maxdepth 1 -name '*.pid' -type f | wc -l)"
  [[ "$pid_file_count" -eq 8 ]] || die "server launcher did not create exactly 8 PID files"
  [[ "$(wc -l < "$service_root/ownership.csv")" -eq 8 ]] || die "server ownership audit did not capture exactly 8 process groups"
  for port in "${PORTS[@]}"; do
    port_is_busy "$port" || die "endpoint port $port is not listening"
  done
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits > "$service_root/gpu_uuid_map.csv"
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits > "$service_root/compute_pid_uuid_map.csv"
  "$PY" "$TOOL" validate-topology \
    --gpu-inventory "$RUN_ROOT/preflight/gpu_uuid_map.csv" \
    --compute-inventory "$service_root/compute_pid_uuid_map.csv" \
    --pid-dir "$service_root/logs" --output "$service_root/topology.json"
}

raw_run_for() {
  local stamp="$1"
  echo "$ROOT/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_agentic_esopt_eoh_b1000_rep1_${stamp}"
}

run_smoke() {
  preflight
  local attempt="$RUN_ROOT/smoke"
  [[ ! -e "$attempt/PASS" ]] || die "smoke was already accepted: $attempt"
  if [[ -d "$attempt" ]]; then
    [[ "${RETRY_SMOKE:-0}" = "1" ]] || die "partial smoke exists; set RETRY_SMOKE=1 to archive and restart it"
    failed_smoke="$RUN_ROOT/failed/smoke_$(date -u +%Y%m%d_%H%M%S)"
    mkdir -p "$(dirname "$failed_smoke")"
    mv "$attempt" "$failed_smoke"
  fi
  mkdir -p "$attempt"
  local stamp="stage3_smoke_$(date -u +%Y%m%d_%H%M%S)"
  local raw
  raw="$(raw_run_for "$stamp")"
  [[ ! -e "$raw" ]] || die "raw smoke path already exists: $raw"
  start_servers "$attempt/services"
  {
    echo "task=construct_tsp"
    echo "repeat=smoke"
    echo "launcher=$LAUNCHER agentic-esopt-eoh"
    echo "generations=1 population=10 k=1 operators=m1"
    echo "directions=10 endpoints=$ENDPOINTS generation_workers=8 evaluation_workers=4"
    echo "seed=2024 sigma_start=0.001 sigma_end=0 schedule=cosine alpha=0.0005"
  } > "$attempt/run_parameters.txt"
  TASKS="construct_tsp" REPS="1" STAMP="$stamp" PY="$PY" \
    ES_ENGINE_URLS="$ENDPOINTS" ES_MAX_WORKERS="8" EVALUATION_WORKERS="4" \
    AHD_POP_SIZE="10" AHD_GENERATIONS="1" EOH_K="1" ES_OPERATORS="m1" \
    ES_DIRECTIONS="10" ES_SIGMA_START="0.001" ES_SIGMA_END="0" \
    ES_SIGMA_SCHEDULE="cosine" ES_ALPHA="0.0005" ES_SEED="2024" \
    "$LAUNCHER" agentic-esopt-eoh 2>&1 | tee "$attempt/runner.log"
  [[ -d "$raw" ]] || die "canonical launcher did not create expected smoke run: $raw"
  ln -s "$raw" "$attempt/raw"
  "$PY" "$TOOL" validate --smoke --repeat 0 --raw-run "$raw" \
    --runner-log "$attempt/runner.log" --output "$attempt/validation.json"
  sha256sum "$attempt/runner.log" "$attempt/validation.json" \
    "$raw/results/es/history.json" "$raw/results/es/evaluator_processes.json" \
    "$raw/results/history/operator_candidates.json" \
    "$attempt/services/ownership.csv" "$attempt/services/topology.json" \
    "$attempt/services/gpu_uuid_map.csv" "$attempt/services/compute_pid_uuid_map.csv" \
    "$attempt/services/startup.txt" "$attempt/run_parameters.txt" > "$attempt/SHA256SUMS"
  stop_owned_servers "$ACTIVE_SERVICE_ROOT"
  ACTIVE_SERVICE_ROOT=""
  touch "$attempt/PASS"
  echo "[stage3] directed smoke PASS: $attempt"
}

archive_partial_repeat() {
  local repeat_root="$1" raw="$2" repeat="$3"
  [[ "${RETRY_PARTIAL:-0}" = "1" ]] || die "repeat $repeat is partial; set RETRY_PARTIAL=1 to archive and restart it from the base model"
  local failed="$RUN_ROOT/failed/rep${repeat}_$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "$(dirname "$failed")" "$failed"
  [[ ! -d "$repeat_root" ]] || mv "$repeat_root" "$failed/audit"
  [[ ! -d "$raw" ]] || mv "$raw" "$failed/raw"
  echo "[stage3] archived partial repeat $repeat at $failed"
}

run_formal_repeat() {
  local repeat="$1"
  local repeat_root="$RUN_ROOT/rep${repeat}"
  local stamp="stage3_$(basename "$RUN_ROOT")_rep${repeat}"
  local raw="$ROOT/cache/active_runs/construct_tsp_train_es_sigma0.001_alpha0.0005_construct_tsp_agentic_esopt_eoh_b1000_rep${repeat}_${stamp}"
  if [[ -e "$repeat_root/VALIDATED" ]]; then
    sha256sum -c "$repeat_root/SHA256SUMS"
    echo "[stage3] resume: repeat $repeat already validated"
    return 0
  fi
  if [[ -d "$repeat_root" || -d "$raw" ]]; then
    archive_partial_repeat "$repeat_root" "$raw" "$repeat"
  fi
  mkdir -p "$repeat_root"
  start_servers "$repeat_root/services"
  {
    echo "task=construct_tsp"
    echo "repeat=$repeat"
    echo "launcher=$LAUNCHER agentic-esopt-eoh"
    echo "generations=25 population=10 k=1 operators=m1,m2"
    echo "directions=10 endpoints=$ENDPOINTS generation_workers=8 evaluation_workers=4"
    echo "seed=2024 sigma_start=0.001 sigma_end=0 schedule=cosine alpha=0.0005"
  } > "$repeat_root/run_parameters.txt"
  TASKS="construct_tsp" REPS="$repeat" STAMP="$stamp" PY="$PY" \
    ES_ENGINE_URLS="$ENDPOINTS" ES_MAX_WORKERS="8" EVALUATION_WORKERS="4" \
    AHD_POP_SIZE="10" AHD_GENERATIONS="25" EOH_K="1" ES_OPERATORS="m1,m2" \
    ES_DIRECTIONS="10" ES_SIGMA_START="0.001" ES_SIGMA_END="0" \
    ES_SIGMA_SCHEDULE="cosine" ES_ALPHA="0.0005" ES_SEED="2024" \
    "$LAUNCHER" agentic-esopt-eoh 2>&1 | tee "$repeat_root/runner.log"
  [[ -d "$raw" ]] || die "canonical launcher did not create expected formal run: $raw"
  ln -s "$raw" "$repeat_root/raw"
  "$PY" "$TOOL" validate --repeat "$repeat" --raw-run "$raw" \
    --runner-log "$repeat_root/runner.log" \
    --final-code "$repeat_root/final_best_code.py" --output "$repeat_root/trend.json"
  "$PY" "$TOOL" evaluate --repeat "$repeat" --code "$repeat_root/final_best_code.py" \
    --output "$repeat_root/cpu_final_eval.json" | tee "$repeat_root/cpu_final_eval.log"
  sha256sum "$repeat_root/runner.log" "$repeat_root/trend.json" \
    "$repeat_root/final_best_code.py" "$repeat_root/cpu_final_eval.json" \
    "$raw/results/es/history.json" \
    "$raw/results/es/evaluator_processes.json" \
    "$raw/results/history/operator_candidates.json" \
    "$raw/results/pops_best/population_generation_25.json" \
    "$repeat_root/services/ownership.csv" "$repeat_root/services/topology.json" \
    "$repeat_root/services/gpu_uuid_map.csv" "$repeat_root/services/compute_pid_uuid_map.csv" \
    "$repeat_root/services/startup.txt" "$repeat_root/run_parameters.txt" > "$repeat_root/SHA256SUMS"
  stop_owned_servers "$ACTIVE_SERVICE_ROOT"
  ACTIVE_SERVICE_ROOT=""
  touch "$repeat_root/VALIDATED"
  echo "[stage3] repeat $repeat structurally validated"
}

run_formal() {
  preflight
  [[ -e "$RUN_ROOT/smoke/PASS" ]] || die "directed 8-endpoint smoke must pass before formal training"
  (cd "$RUN_ROOT/smoke" && sha256sum -c SHA256SUMS)
  for repeat in 1 2 3; do
    run_formal_repeat "$repeat"
  done
  "$PY" "$TOOL" summarize --run-root "$RUN_ROOT" --output "$RUN_ROOT/summary.json"
  sha256sum "$RUN_ROOT"/rep*/SHA256SUMS "$RUN_ROOT/summary.json" > "$RUN_ROOT/FORMAL_SHA256SUMS"
  touch "$RUN_ROOT/READY_FOR_MANUAL_ACCEPTANCE"
  echo "[stage3] all three repeats complete; manual acceptance required: $RUN_ROOT"
}

cleanup() {
  require_runtime_inputs
  local service_root
  while IFS= read -r service_root; do
    stop_owned_servers "$service_root"
  done < <(find "$RUN_ROOT" -type d -name services -print 2>/dev/null)
  echo "[stage3] cleanup checked only PID files owned by $RUN_ROOT"
}

action="${1:-}"
case "$action" in
  plan) exec "$PY" "$TOOL" plan ;;
  preflight) preflight ;;
  smoke) trap cleanup_on_exit EXIT; run_smoke; trap - EXIT ;;
  formal) trap cleanup_on_exit EXIT; run_formal; trap - EXIT ;;
  cleanup) cleanup ;;
  *) usage ;;
esac
