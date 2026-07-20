#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
JOB="${JOB:-}"

if [[ -z "${JOB}" || ! -f "${JOB}" ]]; then
  echo "Set JOB to the site-specific PBS script used to launch the vLLM evaluation." >&2
  exit 2
fi

qsub -v "ROOT=${ROOT}" "$@" "${JOB}"
