#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BASE_STAMP="${BASE_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-$ROOT/runs/ahd_requested_reruns_${BASE_STAMP}}"
STATUS="$QUEUE_ROOT/status.log"
mkdir -p "$QUEUE_ROOT"

mark() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"
}

mark "queue_started total_runs=15"

EOH_STAMP="rerun_eoh_k1_bpp_3rep_8gpu_${BASE_STAMP}"
EOH_ROOT="$ROOT/runs/ahd_${EOH_STAMP}"
mark "stage_started stage=eoh_k1_bpp run_root=$EOH_ROOT runs=3"
ROOT="$ROOT" \
STAMP="$EOH_STAMP" \
RUN_ROOT="$EOH_ROOT" \
TASKS="aco_bpp" \
REPS="1 2 3" \
EC_M1M2_MULTIPLIER="1" \
GPUS="0 1 2 3 4 5 6 7" \
PORTS="11213 11214 11215 11216 11217 11218 11219 11220" \
LLM_LOCAL_TIMEOUT="600" \
EVA_TIMEOUT="600" \
  bash "$ROOT/scripts/ahd/run_all_eoh_llama31_8b_train_triplicate.sh"
mark "stage_completed stage=eoh_k1_bpp run_root=$EOH_ROOT runs=3"

DYN_K1_STAMP="rerun_dynamic_k1_kp_asp_3rep_8gpu_${BASE_STAMP}"
DYN_K1_ROOT="$ROOT/runs/ahd_es_${DYN_K1_STAMP}"
mark "stage_started stage=dynamic_k1_kp_asp run_root=$DYN_K1_ROOT runs=6"
ROOT="$ROOT" \
STAMP="$DYN_K1_STAMP" \
RUN_ROOT="$DYN_K1_ROOT" \
TASKS="construct_kp construct_asp" \
REPS="1 2 3" \
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
  bash "$ROOT/scripts/ahd/run_es_reload_lora_full_tsp_kp_8gpu.sh"
mark "stage_completed stage=dynamic_k1_kp_asp run_root=$DYN_K1_ROOT runs=6"

DYN_K3_STAMP="rerun_dynamic_k3_kp_tspaco_3rep_8gpu_${BASE_STAMP}"
DYN_K3_ROOT="$ROOT/runs/ahd_es_${DYN_K3_STAMP}"
mark "stage_started stage=dynamic_k3_kp_tspaco run_root=$DYN_K3_ROOT runs=6"
ROOT="$ROOT" \
STAMP="$DYN_K3_STAMP" \
RUN_ROOT="$DYN_K3_ROOT" \
TASKS="construct_kp aco_tsp" \
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
  bash "$ROOT/scripts/ahd/run_es_reload_lora_full_tsp_kp_8gpu.sh"
mark "stage_completed stage=dynamic_k3_kp_tspaco run_root=$DYN_K3_ROOT runs=6"

mark "queue_completed total_runs=15"
