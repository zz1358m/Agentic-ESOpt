#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"

if [ -f "$ROOT/scripts/settings.local.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

MODEL="${DOCVQA_MODEL_PATH:-${MODEL:-}}"
if [[ -z "$MODEL" ]]; then
  echo "Set DOCVQA_MODEL_PATH or MODEL to a vision-language checkpoint." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ -n "${DOCVQA_SERVER_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "$DOCVQA_SERVER_EXTRA_ARGS"
fi

exec "$PY" "$ROOT/docvqa-train-time/scripts/hf_vision_es_server.py" \
  --path "$MODEL" \
  --d "${DOCVQA_SERVER_GPU:-0}" \
  --host "${DOCVQA_SERVER_HOST:-127.0.0.1}" \
  --port "${DOCVQA_SERVER_PORT:-11013}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
