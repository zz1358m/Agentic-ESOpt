#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

TRACE_LOGS="${TRACE_LOGS:?Set TRACE_LOGS to a directory of *_FAILED.md and *_SUCCEED.md traces.}"
RUN_ID="${RUN_ID:-docvqa_trace2skill}"

exec "$PY" "$ROOT/trace2skill-settings/scripts/evolve_from_trace_logs.py" \
  --setting docvqa \
  --trace-logs "$TRACE_LOGS" \
  --run-id "$RUN_ID" \
  "$@"
