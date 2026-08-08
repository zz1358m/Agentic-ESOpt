#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL="${MODEL:-/data/xiangru/my_cache/Llama-3.1-8B-Instruct}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_sample_extend_all6_3rep_to2000_gpu4_7_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"
PROGRESS_JSONL="$RUN_ROOT/progress.jsonl"
WAIT_PID="${WAIT_PID:?WAIT_PID is required}"
REP23_STAMP="${REP23_STAMP:?REP23_STAMP is required}"
REP1_CONSTRUCT_STAMP="${REP1_CONSTRUCT_STAMP:-20260718_060041}"
REP1_ACO_STAMP="${REP1_ACO_STAMP:-20260718_091608}"

TASKS=(${TASKS:-construct_tsp construct_kp construct_asp aco_tsp aco_cvrp aco_bpp})
REPS=(${REPS:-1 2 3})
GPUS=(${GPUS:-4 5 6 7})
PORTS=(${PORTS:-11617 11618 11619 11620})
SOURCE_TOTAL="${SOURCE_TOTAL:-1000}"
TARGET_TOTAL="${TARGET_TOTAL:-2000}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-20}"
LLM_LOCAL_TIMEOUT="${LLM_LOCAL_TIMEOUT:-600}"

mkdir -p "$LOGDIR"
cd "$ROOT"
echo "$$" > "$RUN_ROOT/queue.pid"

join_by_comma() { local IFS=","; echo "$*"; }
is_port_busy() { ss -ltn "( sport = :${1} )" 2>/dev/null | tail -n +2 | grep -q .; }

wait_server_ready() {
  local port="$1" log="$2" waited=0
  while true; do
    grep -q "Running on http://127.0.0.1:${port}" "$log" 2>/dev/null && return 0
    if grep -Eq "Traceback|RuntimeError|ModuleNotFoundError|Address already in use" "$log" 2>/dev/null; then
      echo "server on port ${port} failed, see $log" >&2; return 1
    fi
    sleep 5; waited=$((waited + 5))
    [ "$waited" -lt 900 ] || { echo "timeout waiting for port $port" >&2; return 1; }
  done
}

cleanup() {
  local pid
  for pid in ${RUN_PID:-}; do kill "$pid" >/dev/null 2>&1 || true; done
  for pid in ${SERVER_PIDS:-}; do kill "$pid" >/dev/null 2>&1 || true; done
  wait ${RUN_PID:-} >/dev/null 2>&1 || true
  wait ${SERVER_PIDS:-} >/dev/null 2>&1 || true
}
on_signal() { trap - EXIT; cleanup; exit 143; }
trap cleanup EXIT
trap on_signal INT TERM

source_path_for() {
  local task="$1" rep="$2" source_stamp
  if [ "$rep" -eq 1 ]; then
    case "$task" in
      construct_*) source_stamp="$REP1_CONSTRUCT_STAMP" ;;
      aco_*) source_stamp="$REP1_ACO_STAMP" ;;
      *) echo "unsupported task: $task" >&2; return 1 ;;
    esac
  else
    source_stamp="$REP23_STAMP"
  fi
  echo "$ROOT/cache/active_runs/${task}_train_sample_t${SOURCE_TOTAL}_${task}_sample_t${SOURCE_TOTAL}_rep${rep}_${source_stamp}"
}

validate_source() {
  local source="$1"
  "$PY" - "$source" "$SOURCE_TOTAL" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
samples = root / "results" / "samples.jsonl"
if not samples.is_file():
    raise SystemExit(f"missing source samples: {samples}")
count = sum(1 for line in samples.open() if line.strip())
if count != expected:
    raise SystemExit(f"source {root} has {count} samples, expected {expected}")
summary = json.loads((root / "results" / "sample_summary.json").read_text())
if int(summary.get("total_samples", -1)) != expected:
    raise SystemExit(f"source summary is not complete: {root}")
PY
}

write_progress() {
  local task="$1" rep="$2" status="$3" rc="$4" source="$5" destination="$6" result="$7"
  "$PY" - "$PROGRESS_JSONL" "$task" "$rep" "$status" "$rc" "$source" "$destination" "$result" <<'PY'
import json, sys
from datetime import datetime, timezone
path, task, rep, status, rc, source, destination, result = sys.argv[1:]
row = {
    "ts": datetime.now(timezone.utc).isoformat(), "task": task,
    "method": "sample", "rep": int(rep), "status": status,
    "exit_code": int(rc), "source_1000": source,
    "destination_2000": destination, "result_path": result,
    "source_total": 1000, "target_total": 2000,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
PY
}

echo "[waiting] pid=$WAIT_PID"
while kill -0 "$WAIT_PID" >/dev/null 2>&1; do sleep 30; done
echo "[wait-finished] pid=$WAIT_PID"

if [ "${#GPUS[@]}" -ne "${#PORTS[@]}" ]; then echo "GPU/port count mismatch" >&2; exit 1; fi
SERVER_PIDS=""
for i in "${!PORTS[@]}"; do
  port="${PORTS[$i]}"; gpu="${GPUS[$i]}"; log="$LOGDIR/server_gpu${gpu}_port${port}.log"
  is_port_busy "$port" && { echo "port $port is busy" >&2; exit 1; }
  setsid "$PY" "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
    --path "$MODEL" --d "$gpu" --port "$port" --host 127.0.0.1 >"$log" 2>&1 < /dev/null &
  pid=$!; SERVER_PIDS="$SERVER_PIDS $pid"; echo "$pid" > "$LOGDIR/server_gpu${gpu}_port${port}.pid"
done
for i in "${!PORTS[@]}"; do
  wait_server_ready "${PORTS[$i]}" "$LOGDIR/server_gpu${GPUS[$i]}_port${PORTS[$i]}.log" || exit 1
done
URLS=(); for port in "${PORTS[@]}"; do URLS+=("http://127.0.0.1:${port}/completions"); done
URLS_CSV="$(join_by_comma "${URLS[@]}")"
echo "[servers-ready] source_total=$SOURCE_TOTAL target_total=$TARGET_TOTAL"

FAILURES=0; INDEX=0; TOTAL_RUNS=$(( ${#TASKS[@]} * ${#REPS[@]} )); TARGET_GENERATION=$((TARGET_TOTAL / SAMPLE_BATCH_SIZE))
for rep in "${REPS[@]}"; do
  for task in "${TASKS[@]}"; do
    INDEX=$((INDEX + 1)); source="$(source_path_for "$task" "$rep")"
    destination="$ROOT/cache/active_runs/${task}_train_sample_t${TARGET_TOTAL}_${task}_sample_t${TARGET_TOTAL}_from_rep${rep}_${STAMP}"
    result="$destination/results/pops_best/population_generation_${TARGET_GENERATION}.json"
    log="$LOGDIR/${task}_sample_extend_rep${rep}.log"
    echo "[${INDEX}/${TOTAL_RUNS} start] task=$task rep=$rep source=$source"
    if ! validate_source "$source"; then
      write_progress "$task" "$rep" failed 2 "$source" "$destination" "$result"; FAILURES=$((FAILURES + 1)); continue
    fi
    if [ -e "$destination" ]; then
      echo "destination already exists: $destination" >&2
      write_progress "$task" "$rep" failed 3 "$source" "$destination" "$result"; FAILURES=$((FAILURES + 1)); continue
    fi
    cp -a "$source" "$destination"
    write_progress "$task" "$rep" started 0 "$source" "$destination" "$result"
    LLM_LOCAL_TIMEOUT="$LLM_LOCAL_TIMEOUT" SAMPLE_TOTAL="$TARGET_TOTAL" SAMPLE_BATCH_SIZE="$SAMPLE_BATCH_SIZE" \
      "$PY" -u "$ROOT/ahd-test-time/scripts/run_ahd_four_methods.py" \
        --task "$task" --split train --method sample \
        --sample-total "$TARGET_TOTAL" --sample-batch-size "$SAMPLE_BATCH_SIZE" \
        --sample-resume-path "$destination" --llm-local-url "$URLS_CSV" \
        --run-id "${task}_sample_t${TARGET_TOTAL}_from_rep${rep}_${STAMP}" >"$log" 2>&1 &
    RUN_PID=$!; wait "$RUN_PID"; rc=$?; RUN_PID=""
    if [ "$rc" -eq 0 ]; then
      write_progress "$task" "$rep" completed "$rc" "$source" "$destination" "$result"
    else
      write_progress "$task" "$rep" failed "$rc" "$source" "$destination" "$result"; FAILURES=$((FAILURES + 1))
    fi
    echo "[${INDEX}/${TOTAL_RUNS} done] task=$task rep=$rep rc=$rc"
  done
done
echo "[all-done] failures=$FAILURES run_root=$RUN_ROOT"
exit "$FAILURES"
