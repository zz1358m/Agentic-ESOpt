#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# 1000 calls = 25 generations * (e1 + e2 + one m1 + one m2) * population 10.
exec env BUDGET=1000 EOH_K=1 \
  bash "$ROOT/scripts/ahd/run_four_method_ahd.sh" "$@"
