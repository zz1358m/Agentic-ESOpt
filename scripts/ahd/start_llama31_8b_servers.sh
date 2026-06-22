#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
MODEL="${MODEL:-}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/ahd_llama31_8b_servers_${STAMP}}"
LOGDIR="$RUN_ROOT/logs"

PORTS=(${PORTS:-11013 11014 11015 11016})
GPUS=(${GPUS:-0 1 2 3})

mkdir -p "$LOGDIR"

if [ -z "$MODEL" ]; then
  echo "Set MODEL to a local HF model path or model id." >&2
  exit 2
fi

is_port_busy() {
  local port="$1"
  ss -ltn "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q .
}

wait_server_ready() {
  local port="$1"
  local log="$2"
  local waited=0
  while true; do
    if grep -q "Running on http://127.0.0.1:${port}" "$log" 2>/dev/null; then
      return 0
    fi
    if grep -Eq "Traceback|Error|RuntimeError|ModuleNotFoundError" "$log" 2>/dev/null; then
      echo "server on port ${port} failed, see $log" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge 900 ]; then
      echo "timeout waiting for server on port ${port}" >&2
      return 1
    fi
  done
}

for i in "${!PORTS[@]}"; do
  port="${PORTS[$i]}"
  gpu="${GPUS[$i]}"
  log="$LOGDIR/server_gpu${gpu}_port${port}.log"
  if is_port_busy "$port"; then
    echo "port ${port} already in use" >&2
    exit 1
  fi
  setsid "$PY" "$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py" \
    --path "$MODEL" --d "$gpu" --port "$port" --host 127.0.0.1 \
    >"$log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$LOGDIR/server_gpu${gpu}_port${port}.pid"
done

for i in "${!PORTS[@]}"; do
  wait_server_ready "${PORTS[$i]}" "$LOGDIR/server_gpu${GPUS[$i]}_port${PORTS[$i]}.log"
done

printf 'run_root=%s\n' "$RUN_ROOT"
for i in "${!PORTS[@]}"; do
  gpu="${GPUS[$i]}"
  port="${PORTS[$i]}"
  pid="$(cat "$LOGDIR/server_gpu${gpu}_port${port}.pid")"
  printf 'gpu=%s port=%s pid=%s\n' "$gpu" "$port" "$pid"
done
