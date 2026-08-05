#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
"$ROOT/webarena-train-time/scripts/train_skillopt_webagent_epoch_eval.sh" "${EPOCHS:-6}" "${RUN_ID:-skillopt_train}" "${MODEL_PORT:-11013}" "${EVAL_PORT:-11015}" "0"
