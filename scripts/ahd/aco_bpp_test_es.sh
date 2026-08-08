#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
"$PY" "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" --task aco_bpp --split test --method es --run-id "${RUN_ID:-aco_bpp_test_es}"
