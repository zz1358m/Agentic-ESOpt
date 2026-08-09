#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-${ROOT}/algorithms/verl}"
JOB="${JOB:-}"

if [[ -z "${JOB}" || ! -f "${JOB}" ]]; then
  echo "Set JOB to the site-specific PBS script that invokes scripts/trace2skill/run_verl_agentic_rl.sh." >&2
  exit 2
fi

submit_one() {
  local task="$1"
  qsub -v "ROOT=${ROOT},VERL_ROOT=${VERL_ROOT},TASK=${task}" "$@" "${JOB}"
}

if [[ "${1:-}" == "--both" ]]; then
  shift
  submit_one math "$@"
  submit_one docvqa "$@"
elif [[ "${1:-}" == "math" || "${1:-}" == "docvqa" ]]; then
  task="$1"
  shift
  submit_one "${task}" "$@"
else
  echo "Usage: JOB=/path/to/job.pbs $0 --both [qsub args...] | math [qsub args...] | docvqa [qsub args...]" >&2
  exit 2
fi
