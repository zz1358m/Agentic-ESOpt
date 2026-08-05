#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
SERVER="${SERVER:-$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py}"
MODEL_PATH="${MODEL_PATH:-/data0/zhi/meta-llama/Qwen3.5-4B}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
RUN_ID="${RUN_ID:-trace2skill_eval16_math_local4gpu_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/trace2skill_eval16/$RUN_ID}"
PORTS_CSV="${PORTS_CSV:-12200,12201,12202,12203}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"

cd "$ROOT"
mkdir -p logs "$OUT_DIR"

IFS=',' read -r -a PORTS <<< "$PORTS_CSV"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#PORTS[@]}" -ne "${#GPUS[@]}" ]]; then
  echo "PORTS_CSV and GPUS_CSV must have the same length." >&2
  exit 1
fi

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

base_urls_csv() {
  local joined=""
  local port
  for port in "${PORTS[@]}"; do
    [[ -n "$joined" ]] && joined+=","
    joined+="http://127.0.0.1:${port}/v1"
  done
  printf '%s' "$joined"
}

server_pids_for_ports() {
  local port
  for port in "$@"; do
    pgrep -f "llama31_instruct_server.py .*--port ${port}" || true
  done | sort -u
}

stop_servers() {
  [[ "$KEEP_SERVERS" == "1" ]] && return 0
  local pids=()
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(server_pids_for_ports "${PORTS[@]}")
  ((${#pids[@]} == 0)) && return 0
  echo "[$(timestamp)] stopping servers pids=${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
}
trap stop_servers EXIT

wait_ready() {
  local port="$1"
  "$PY" - "$port" <<'PY'
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
payload = json.dumps({
    "model": "local",
    "messages": [{"role": "user", "content": "Return only OK."}],
    "max_tokens": 4,
    "temperature": 0.0,
    "top_p": 1.0,
}).encode()
last = None
for _ in range(240):
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        if data.get("choices"):
            print(f"[ready] port={port}", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        last = repr(exc)
    time.sleep(5)
print(f"[failed] port={port} err={last}", flush=True)
raise SystemExit(1)
PY
}

stop_servers
for idx in "${!PORTS[@]}"; do
  port="${PORTS[$idx]}"
  gpu="${GPUS[$idx]}"
  log="logs/trace2skill_eval16_math_gpu${gpu}_${port}_${RUN_ID}.log"
  setsid "$PY" "$SERVER" \
    --d "$gpu" \
    --path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$port" \
    --dtype bfloat16 \
    --trust-remote-code \
    --chat-template-enable-thinking false \
    > "$log" 2>&1 < /dev/null &
  echo "[$(timestamp)] server started gpu=${gpu} port=${port} pid=$! log=${log}"
done

for port in "${PORTS[@]}"; do
  wait_ready "$port"
done

"$PY" "$ROOT/scripts/trace2skill/run_trace2skill_vllm_eval16.py" \
  --base-urls "$(base_urls_csv)" \
  --model "$MODEL_NAME" \
  --math-root "$ROOT/data/trace2skill/math_reasoning" \
  --out-dir "$OUT_DIR" \
  --datasets dapo100,aime2026 \
  --samples "${TRACE2SKILL_EVAL_SAMPLES:-16}" \
  --concurrency "${TRACE2SKILL_EVAL_CONCURRENCY:-64}" \
  --timeout "${TRACE2SKILL_EVAL_TIMEOUT:-1800}" \
  --request-retries "${TRACE2SKILL_EVAL_REQUEST_RETRIES:-3}" \
  --max-errors "${TRACE2SKILL_EVAL_MAX_ERRORS:-32}" \
  --seed "${TRACE2SKILL_EVAL_SEED:-20260629}" \
  --temperature "${TRACE2SKILL_EVAL_TEMPERATURE:-1.0}" \
  --top-p "${TRACE2SKILL_EVAL_TOP_P:-1.0}" \
  --top-k "${TRACE2SKILL_EVAL_TOP_K:-40}" \
  --min-p "${TRACE2SKILL_EVAL_MIN_P:-0.0}" \
  --presence-penalty "${TRACE2SKILL_EVAL_PRESENCE_PENALTY:-2.0}" \
  --repetition-penalty "${TRACE2SKILL_EVAL_REPETITION_PENALTY:-1.0}" \
  --math-max-tokens "${TRACE2SKILL_EVAL_MATH_MAX_TOKENS:-0}" \
  --math-max-turns "${TRACE2SKILL_EVAL_MATH_MAX_TURNS:-100}" \
  --math-python-timeout "${TRACE2SKILL_EVAL_MATH_PYTHON_TIMEOUT:-20.0}" \
  --tool-observation-limit "${TRACE2SKILL_EVAL_TOOL_OBSERVATION_LIMIT:-6000}" \
  --log-every "${TRACE2SKILL_EVAL_LOG_EVERY:-25}" \
  "$@"
