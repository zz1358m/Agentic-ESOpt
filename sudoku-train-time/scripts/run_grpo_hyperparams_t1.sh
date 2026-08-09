#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MASK_COUNT="${1:-15}"

case "$MASK_COUNT" in
  5|10|15) ;;
  *)
    echo "Usage: $0 {5|10|15} [trainer args...]" >&2
    exit 2
    ;;
esac
shift $(( $# > 0 ? 1 : 0 ))

RUN_ID="${RUN_ID:-sudoku_grpo_qwen35_4b_mask${MASK_COUNT}_t1_p1_kneg1_raw_policy_kl}"
export RUN_ID

exec "$ROOT/sudoku-train-time/scripts/run_grpo_hyperparams.sh" "$MASK_COUNT" \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k -1 \
  "$@"
