#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
"$PY" "$ROOT/ahd-test-time/scripts/run_eoh_ahd.py" --task construct_kp --split test --method eoh --run-id "${RUN_ID:-construct_kp_test_eoh}"
