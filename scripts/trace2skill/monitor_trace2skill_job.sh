#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:?usage: monitor_trace2skill_job.sh JOB_ID [LOG_FILE]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNS_ROOT="${TRACE2SKILL_VLLM_RUN_ROOT:-${ROOT}/runs/trace2skill_vllm}"
LOG_FILE="${2:-${RUNS_ROOT}/monitor-${JOB_ID%%.*}.log}"
INTERVAL="${INTERVAL:-30}"
PBS_SERVER_NAME="${PBS_SERVER_NAME:-}"
RUN_TAG="${RUN_TAG:-}"

mkdir -p "$(dirname "${LOG_FILE}")"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_FILE}"
}

latest_run_dir() {
  if [[ -n "${RUN_TAG}" ]]; then
    ls -td "${RUNS_ROOT}/${RUN_TAG}" 2>/dev/null | head -1 || true
  else
    ls -td "${RUNS_ROOT}"/trace2skill-* 2>/dev/null | head -1 || true
  fi
}

find_job_log() {
  local run_dir="$1"
  local pattern="$2"
  find "${run_dir}/pbs_logs" -maxdepth 1 -type f -name "${pattern}" -print 2>/dev/null \
    | sort | head -1 || true
}

summarize_outputs() {
  local run_dir="$1"
  [[ -n "${run_dir}" && -d "${run_dir}" ]] || return 0
  for dataset in dapo100 aime2026 docvqa; do
    local path="${run_dir}/outputs/${dataset}.jsonl"
    if [[ -f "${path}" ]]; then
      local stats
      stats="$(python - "${path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = 0
errors = 0
valid = 0
last_error = None
with path.open(encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if not line.strip():
            continue
        records += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors += 1
            last_error = f"json: {exc}"
            continue
        if row.get("error"):
            errors += 1
            last_error = str(row.get("error"))
        elif float(row.get("score", -1.0)) >= 0.0:
            valid += 1
print(f"records={records} valid={valid} errors={errors} last_error={last_error!r}"[:600])
PY
)"
      log "output ${dataset}: ${stats}"
    fi
  done
  if [[ -f "${run_dir}/summary.json" ]]; then
    log "summary: ${run_dir}/summary.json"
  fi
}

log "monitor start job=${JOB_ID} log=${LOG_FILE}"

while true; do
  qstat_out="$(qstat -f "${JOB_ID}" 2>/dev/null || true)"
  if [[ -z "${qstat_out}" && -n "${PBS_SERVER_NAME}" ]]; then
    qstat_out="$(qstat -f "${JOB_ID}" "@${PBS_SERVER_NAME}" 2>/dev/null || true)"
  fi
  if [[ -z "${qstat_out}" && -n "${PBS_SERVER_NAME}" ]]; then
    qstat_line="$(qstat -u "${USER}" "@${PBS_SERVER_NAME}" 2>/dev/null | awk -v id="${JOB_ID%%.*}" '$1 ~ "^" id {print; exit}' || true)"
    if [[ -n "${qstat_line}" ]]; then
      state="$(awk '{print $(NF-1)}' <<<"${qstat_line}")"
      qstat_out="job_state = ${state}"
    fi
  fi
  if [[ -z "${qstat_out}" ]]; then
    log "job left qstat"
    stdout="${PBS_STDOUT:-}"
    if [[ -z "${stdout}" ]]; then
      stdout="$(find "${ROOT}" -maxdepth 1 -type f -name "*.o${JOB_ID%%.*}" -print 2>/dev/null | sort | head -1 || true)"
    fi
    if [[ -f "${stdout}" ]]; then
      log "pbs stdout: ${stdout}"
      tail -n 80 "${stdout}" 2>&1 | sed 's/^/[pbs] /' | tee -a "${LOG_FILE}"
    fi
    run_dir="$(latest_run_dir)"
    log "latest_run_dir=${run_dir}"
    if [[ -n "${run_dir}" ]]; then
      find "${run_dir}" -maxdepth 2 -type f | sort | sed 's/^/[file] /' | tee -a "${LOG_FILE}"
      server_log="$(find_job_log "${run_dir}" "vllm_server.${JOB_ID%%.*}*.log")"
      if [[ -f "${server_log}" ]]; then
        log "server log: ${server_log}"
        tail -n 120 "${server_log}" 2>&1 | sed 's/^/[server] /' | tee -a "${LOG_FILE}"
      fi
      summarize_outputs "${run_dir}"
    fi
    exit 0
  fi

  state="$(awk -F'= ' '/job_state =/{print $2; exit}' <<<"${qstat_out}")"
  comment="$(awk -F'= ' '/comment =/{print $2; exit}' <<<"${qstat_out}")"
  exec_host="$(awk -F'= ' '/exec_host =/{print $2; exit}' <<<"${qstat_out}")"
  walltime="$(awk -F'= ' '/resources_used.walltime =/{print $2; exit}' <<<"${qstat_out}")"
  log "state=${state:-unknown} exec_host=${exec_host:-none} walltime=${walltime:-none} comment=${comment:-none}"

  if [[ "${state:-}" == "R" ]]; then
    run_dir="$(latest_run_dir)"
    log "latest_run_dir=${run_dir}"
    if [[ -n "${run_dir}" ]]; then
      server_log="$(find_job_log "${run_dir}" "vllm_server.${JOB_ID%%.*}*.log")"
      if [[ -f "${server_log}" ]]; then
        tail -n 40 "${server_log}" 2>&1 | sed 's/^/[server] /' | tee -a "${LOG_FILE}"
      fi
      for stream in stdout stderr; do
        client_log="$(find_job_log "${run_dir}" "client.${JOB_ID%%.*}*.${stream}")"
        if [[ -f "${client_log}" ]]; then
          tail -n 25 "${client_log}" 2>&1 | sed "s/^/[client-${stream}] /" | tee -a "${LOG_FILE}"
        fi
      done
      summarize_outputs "${run_dir}"
    fi
  fi

  sleep "${INTERVAL}"
done
