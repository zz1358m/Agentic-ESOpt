#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <math|docvqa|all> <hf_model_dir> [qsub args...]" >&2
  exit 2
fi

TASK="$1"
MODEL_PATH="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
JOB="${EVAL_JOB:-${JOB:-}}"

if [[ -z "${JOB}" || ! -f "${JOB}" ]]; then
  echo "Set EVAL_JOB (or JOB) to the site-specific PBS script used for checkpoint evaluation." >&2
  exit 2
fi

case "${TASK}" in
  math) DATASETS="dapo100,aime2026" ;;
  docvqa) DATASETS="docvqa" ;;
  all) DATASETS="dapo100,aime2026,docvqa" ;;
  *) echo "TASK must be math, docvqa, or all" >&2; exit 2 ;;
esac

RUN_TAG="${RUN_TAG:-trace2skill-verl-eval16-$(basename "$(dirname "$(dirname "${MODEL_PATH}")")")-${TASK}-$(date -u +%Y%m%d_%H%M%S)}"

qsub -v "ROOT=${ROOT},MODEL_PATH=${MODEL_PATH},SERVED_MODEL_NAME=$(basename "${MODEL_PATH}"),DATASETS=${DATASETS},RUN_TAG=${RUN_TAG}" "$@" "${JOB}"
