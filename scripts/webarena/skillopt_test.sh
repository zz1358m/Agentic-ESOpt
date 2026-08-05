#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
"$ROOT/webarena-train-time/scripts/run_webrl_lite_vab_subset.sh" skillopt "${EVAL_PORT:-11015}" "${SITES:-shopping,shopping_admin,reddit,gitlab,wikipedia,map}" "${RUN_ID:-skillopt_test}" "${TEST_LIMIT:-0}"
