#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_eoh_staged_train_${STAMP}}"
SERVER_RUN_ROOT="${SERVER_RUN_ROOT:-$RUN_ROOT/servers}"
PHASE1_RUN_ROOT="${PHASE1_RUN_ROOT:-$RUN_ROOT/phase1_construct}"
PHASE2_RUN_ROOT="${PHASE2_RUN_ROOT:-$RUN_ROOT/phase2_aco}"
PORTS="${PORTS:-11013 11014 11015 11016}"
GPUS="${GPUS:-0 1 2 3}"
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"
EVA_TIMEOUT="${EVA_TIMEOUT:-600}"
SHARED_URLS="${SHARED_URLS:-http://127.0.0.1:11013/completions,http://127.0.0.1:11014/completions,http://127.0.0.1:11015/completions,http://127.0.0.1:11016/completions}"

mkdir -p "$RUN_ROOT"
cd "$ROOT"

cleanup() {
  if [ -n "${SERVER_PIDS:-}" ]; then
    for pid in $SERVER_PIDS; do
      kill "$pid" >/dev/null 2>&1 || true
    done
  fi
}
trap cleanup EXIT INT TERM

env STAMP="${STAMP}_servers" RUN_ROOT="$SERVER_RUN_ROOT" PORTS="$PORTS" GPUS="$GPUS" \
  bash "$ROOT/scripts/ahd/start_llama31_8b_servers.sh" > "$RUN_ROOT/server_start.log" 2>&1

SERVER_PIDS="$(find "$SERVER_RUN_ROOT/logs" -maxdepth 1 -name 'server_gpu*_port*.pid' -print0 | xargs -0 cat)"

env STAMP="${STAMP}_phase1" RUN_ROOT="$PHASE1_RUN_ROOT" TASKS="construct_asp construct_kp" \
  SKIP_SERVER_START=1 SHARED_URLS="$SHARED_URLS" LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" EVA_TIMEOUT="$EVA_TIMEOUT" \
  bash "$ROOT/scripts/ahd/run_all_eoh_llama31_8b_train_triplicate.sh" > "$RUN_ROOT/phase1.log" 2>&1

env STAMP="${STAMP}_phase2" RUN_ROOT="$PHASE2_RUN_ROOT" TASKS="aco_tsp aco_cvrp" \
  SKIP_SERVER_START=1 SHARED_URLS="$SHARED_URLS" LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" EVA_TIMEOUT="$EVA_TIMEOUT" \
  bash "$ROOT/scripts/ahd/run_all_eoh_llama31_8b_train_triplicate.sh" > "$RUN_ROOT/phase2.log" 2>&1
