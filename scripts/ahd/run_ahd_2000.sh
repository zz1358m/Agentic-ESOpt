#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# 2000 calls = 25 generations * (e1 + e2 + three m1 + three m2) * population 10.
exec env BUDGET=2000 EOH_K=3 \
  bash "$ROOT/scripts/ahd/run_four_method_ahd.sh" "$@"
