#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

GAME="${GAME:-${1:-library}}"
AGENT_TYPE="${AGENT_TYPE:-${2:-memory}}"
RUN_ID="${RUN_ID:-${GAME}_${AGENT_TYPE}}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/jericho/$RUN_ID}"
POLICY_COMPLETIONS_URL="${POLICY_COMPLETIONS_URL:-http://127.0.0.1:11013/completions}"

PYTHONPATH="$ROOT/data/jericho/source${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" "$ROOT/data/jericho/jitrl/main.py" \
  --rom_path "$ROOT/data/jericho/jitrl/jericho-games" \
  --game_name "$GAME" \
  --agent_type "$AGENT_TYPE" \
  --eval_runs "${RUNS:-50}" \
  --env_step_limit "${STEPS:-110}" \
  --seed "${SEED:-0}" \
  --llm_model "${MODEL_NAME:-local-policy}" \
  --evolution_llm_model "${EVOLUTION_MODEL_NAME:-${MODEL_NAME:-local-policy}}" \
  --policy-completions-url "$POLICY_COMPLETIONS_URL" \
  --output_path "$OUT_ROOT" \
  ${JERICHO_EXTRA_ARGS:-}
