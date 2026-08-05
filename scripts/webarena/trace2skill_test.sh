#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TRACE2SKILL_ROOT="${TRACE2SKILL_ROOT:-$ROOT/webarena-train-time/methods/trace2skill/source}"
export TRACE2SKILL_TRAIN_SPLIT="${TRACE2SKILL_TRAIN_SPLIT:-$ROOT/data/webarena/skillopt_splits/train/items.json}"
export TRACE2SKILL_VAL_SPLIT="${TRACE2SKILL_VAL_SPLIT:-$ROOT/data/webarena/skillopt_splits/val/items.json}"
export TRACE2SKILL_TEST_SPLIT="${TRACE2SKILL_TEST_SPLIT:-$ROOT/data/webarena/skillopt_splits/test/items.json}"
if [ ! -f "$TRACE2SKILL_ROOT/run_traintest.sh" ]; then
  echo "Trace2Skill runner not found: $TRACE2SKILL_ROOT/run_traintest.sh" >&2
  exit 4
fi
STAGE=test sh "$TRACE2SKILL_ROOT/run_traintest.sh"
