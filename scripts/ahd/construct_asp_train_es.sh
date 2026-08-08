#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
"$PY" "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" --task construct_asp --split train --method es --run-id "${RUN_ID:-construct_asp_train_es}"
