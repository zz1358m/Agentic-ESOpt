#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-math_trace2skill_es}"
TRACE_RUN_ID="${TRACE_RUN_ID:-${RUN_ID}_skill}"

RUN_ID="$TRACE_RUN_ID" "$ROOT/scripts/math/run_trace2skill.sh"
export MATH_SKILL_FILE="$ROOT/runs/trace2skill_extra/$TRACE_RUN_ID/skill_step_001.md"
export RUN_ID
exec "$ROOT/scripts/math/run.sh" "$@"
