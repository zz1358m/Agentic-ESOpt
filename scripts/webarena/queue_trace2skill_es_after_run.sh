#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}
WAIT_PID=${WAIT_PID:?WAIT_PID is required}
WAIT_RUN_ID=${WAIT_RUN_ID:?WAIT_RUN_ID is required}
NEXT_RUN_ID=${NEXT_RUN_ID:?NEXT_RUN_ID is required}
NEXT_LOG=${NEXT_LOG:-$ROOT/logs/${NEXT_RUN_ID}.log}
WAIT_HISTORY="$ROOT/runs/webrl_lite_full_es/$WAIT_RUN_ID/history.json"

cd "$ROOT"
echo "[queue] waiting for pid=$WAIT_PID run=$WAIT_RUN_ID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

"$PY" - "$WAIT_HISTORY" <<'PY'
import json
import sys

history = json.load(open(sys.argv[1]))
if len(history) != 70 or history[-1].get("generation") != 69 or "eval" not in history[-1]:
    raise SystemExit(
        f"refusing to start queued run: expected 70 completed updates and final eval, got {len(history)}"
    )
PY

echo "[queue] previous run completed; restarting model services from disk"
for gpu in 0 1 2 3; do
  screen -S "webarena27b_gpu${gpu}" -X quit || true
done
for _ in $(seq 1 120); do
  if ! pgrep -f 'llama31_instruct_server.py --path /data0/zhi/meta-llama/Qwen3.5-27B --port 1201[3-6]' >/dev/null; then
    break
  fi
  sleep 1
done
if pgrep -f 'llama31_instruct_server.py --path /data0/zhi/meta-llama/Qwen3.5-27B --port 1201[3-6]' >/dev/null; then
  echo "[queue] model services did not stop cleanly" >&2
  exit 1
fi

stamp=$(date -u +%Y%m%d_%H%M%S)
for gpu in 0 1 2 3; do
  port=$((12013 + gpu))
  server_log="$ROOT/logs/qwen35_27b_server_gpu${gpu}_port${port}_nothink_${stamp}.log"
  screen -dmS "webarena27b_gpu${gpu}" bash -lc \
    "cd '$ROOT' && exec env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY '$PY' ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py --path /data0/zhi/meta-llama/Qwen3.5-27B --port '$port' --host 127.0.0.1 --dtype bfloat16 --max-repeat-prompt 8 --chat-template-enable-thinking false --d '$gpu' > '$server_log' 2>&1"
done

for _ in $(seq 1 120); do
  ready=0
  for port in 12013 12014 12015 12016; do
    body=$(curl -sS --max-time 2 -X POST -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:${port}/es/status" 2>/dev/null || true)
    if printf '%s' "$body" | grep -q '"initialized":false' \
      && printf '%s' "$body" | grep -q '"active_perturbation":null' \
      && printf '%s' "$body" | grep -q '"update_history":0'; then
      ready=$((ready + 1))
    fi
  done
  if [ "$ready" -eq 4 ]; then
    break
  fi
  sleep 5
done
if [ "${ready:-0}" -ne 4 ]; then
  echo "[queue] clean model services were not ready before timeout" >&2
  exit 1
fi

printf '%s\n' "$NEXT_RUN_ID" > /tmp/current_webarena_run_id
printf '%s\n' "$NEXT_LOG" > /tmp/current_webarena_log
echo "[queue] starting Trace2Skill+ES run=$NEXT_RUN_ID"
exec env RUN_ID="$NEXT_RUN_ID" \
  "$ROOT/scripts/webarena/launch_qwen35_trace2skill_es_70_constant_1p5e3.sh"
