#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ROOT TASK=docvqa
exec "$ROOT/scripts/trace2skill/run_verl_agentic_rl.sh" "$@"
