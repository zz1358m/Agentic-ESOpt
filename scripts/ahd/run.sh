#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

TASK="${TASK:-${1:-construct_tsp}}"
SPLIT="${SPLIT:-${2:-train}}"
METHOD="${METHOD:-${3:-eoh}}"
RUN_ID="${RUN_ID:-${TASK}_${SPLIT}_${METHOD}}"

"$PY" "$ROOT/ahd-test-time/scripts/run_eoh_ahd.py" \
  --task "$TASK" \
  --split "$SPLIT" \
  --method "$METHOD" \
  --run-id "$RUN_ID" \
  --llm-local-url "${LLM_LOCAL_URL:-http://127.0.0.1:11013/completions}" \
  --es-engine-urls "${ES_ENGINE_URLS:-}" \
  --es-operators "${ES_OPERATORS:-e1,e2,m1,m2}" \
  --es-sigma "${ES_SIGMA:-1e-3}" \
  --es-alpha "${ES_ALPHA:-5e-4}" \
  ${AHD_EXTRA_ARGS:-}
