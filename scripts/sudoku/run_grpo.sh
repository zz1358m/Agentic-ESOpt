#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

MASK_COUNT="${SUDOKU_TARGET_MASK_COUNT:-15}"

exec "$ROOT/sudoku-train-time/scripts/run_grpo_hyperparams.sh" "$MASK_COUNT" "$@"
