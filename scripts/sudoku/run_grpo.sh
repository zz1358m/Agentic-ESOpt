#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

exec "$ROOT/scripts/sudoku/run_verl_grpo.sh" "$@"
