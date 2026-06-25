#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
"$PY" "$ROOT/ahd-test-time/scripts/run_eoh_ahd.py" --task aco_tsp --split train --method es --run-id "${RUN_ID:-aco_tsp_train_es}"
