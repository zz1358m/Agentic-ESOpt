#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-docvqa_trace2skill_es}"
TRACE_RUN_ID="${TRACE_RUN_ID:-${RUN_ID}_skill}"

RUN_ID="$TRACE_RUN_ID" "$ROOT/scripts/docvqa/run_trace2skill.sh"
export DOCVQA_SKILL_FILE="$ROOT/runs/trace2skill_extra/$TRACE_RUN_ID/skill_step_001.md"
export RUN_ID
exec "$ROOT/scripts/docvqa/run.sh" "$@"
