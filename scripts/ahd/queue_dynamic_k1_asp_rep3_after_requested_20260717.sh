#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WAIT_PID="${WAIT_PID:-138387}"
WAIT_SCRIPT="${WAIT_SCRIPT:-$ROOT/scripts/ahd/run_requested_reruns_20260716.sh}"
BASE_STAMP="${BASE_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
K1_STAMP="rerun_dynamic_k1_asp_rep3_8gpu_${BASE_STAMP}"
K1_RUN_ROOT="$ROOT/runs/ahd_es_${K1_STAMP}"
K3_STAMP="rerun_dynamic_k3_asp_3rep_8gpu_${BASE_STAMP}"
K3_RUN_ROOT="$ROOT/runs/ahd_es_${K3_STAMP}"
QUEUE_ROOT="$ROOT/runs/ahd_queue_dynamic_asp_k1rep3_k3_3rep_${BASE_STAMP}"
STATUS="$QUEUE_ROOT/status.log"
mkdir -p "$QUEUE_ROOT"

mark() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"
}

mark "queued stages=dynamic_k1_asp_rep3,dynamic_k3_asp_3rep wait_pid=$WAIT_PID total_runs=4"

while [[ -r "/proc/$WAIT_PID/cmdline" ]] && tr '\0' ' ' < "/proc/$WAIT_PID/cmdline" | grep -Fq "$WAIT_SCRIPT"; do
  sleep 30
done

mark "stage_started stage=dynamic_k1_asp_rep3 run_root=$K1_RUN_ROOT runs=1"
ROOT="$ROOT" \
STAMP="$K1_STAMP" \
RUN_ROOT="$K1_RUN_ROOT" \
TASKS="construct_asp" \
REPS="3" \
MODES="full" \
GPUS="0 1 2 3 4 5 6 7" \
PORTS="11313 11314 11315 11316 11317 11318 11319 11320" \
ES_OPERATORS="m1,m2" \
ES_DIRECTIONS="10" \
EC_M1M2_MULTIPLIER="1" \
ES_SIGMA_SCHEDULE="cosine" \
ES_SIGMA_SCHEDULE_PLATEAU_FRACTION="0" \
FULL_SIGMAS="1e-3" \
FULL_ALPHA_ES="5e-4" \
ES_DISABLE_UPDATE="0" \
  bash "$ROOT/scripts/ahd/run_es_reload_lora_full_tsp_kp_8gpu.sh" 2>&1 | tee -a "$QUEUE_ROOT/queue.log"

mark "stage_completed stage=dynamic_k1_asp_rep3 run_root=$K1_RUN_ROOT runs=1"

mark "stage_started stage=dynamic_k3_asp_3rep run_root=$K3_RUN_ROOT runs=3"
ROOT="$ROOT" \
STAMP="$K3_STAMP" \
RUN_ROOT="$K3_RUN_ROOT" \
TASKS="construct_asp" \
REPS="1 2 3" \
MODES="full" \
GPUS="0 1 2 3 4 5 6 7" \
PORTS="11313 11314 11315 11316 11317 11318 11319 11320" \
ES_OPERATORS="m1,m2" \
ES_DIRECTIONS="10" \
EC_M1M2_MULTIPLIER="3" \
ES_SIGMA_SCHEDULE="cosine" \
ES_SIGMA_SCHEDULE_PLATEAU_FRACTION="0" \
FULL_SIGMAS="1e-3" \
FULL_ALPHA_ES="5e-4" \
ES_DISABLE_UPDATE="0" \
  bash "$ROOT/scripts/ahd/run_es_reload_lora_full_tsp_kp_8gpu.sh" 2>&1 | tee -a "$QUEUE_ROOT/queue.log"

mark "stage_completed stage=dynamic_k3_asp_3rep run_root=$K3_RUN_ROOT runs=3"
mark "queue_completed total_runs=4"
