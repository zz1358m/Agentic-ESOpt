#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data0/zhi/meta-llama/Qwen3.5-4B}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
TP_SIZE="${TP_SIZE:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_ID="${RUN_ID:-trace2skill_eval16_math_vllm4gpu_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/trace2skill_eval16/$RUN_ID}"
LOG_FILE="${LOG_FILE:-$ROOT/logs/trace2skill_eval16_vllm_${RUN_ID}.log}"
KEEP_SERVER="${KEEP_SERVER:-0}"

cd "$ROOT"
mkdir -p logs "$OUT_DIR"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

server_pids() {
  pgrep -f "vllm.entrypoints.openai.api_server .*--port ${PORT}" || true
}

stop_server() {
  [[ "$KEEP_SERVER" == "1" ]] && return 0
  local pids=()
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(server_pids)
  ((${#pids[@]} == 0)) && return 0
  echo "[$(timestamp)] stopping vLLM pids=${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
}
trap stop_server EXIT

wait_ready() {
  "$PY" - "$HOST" "$PORT" "$MODEL_NAME" <<'PY'
import json
import sys
import time
import urllib.request

host = sys.argv[1]
port = int(sys.argv[2])
model = sys.argv[3]
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Return only OK."}],
    "max_tokens": 4,
    "temperature": 0.0,
    "top_p": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
last = None
for _ in range(360):
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        if data.get("choices"):
            print(f"[ready] vLLM {host}:{port}", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        last = repr(exc)
    time.sleep(5)
print(f"[failed] vLLM {host}:{port} err={last}", flush=True)
raise SystemExit(1)
PY
}

stop_server

CUDA_VISIBLE_DEVICES="$GPU_IDS" VLLM_USE_V1="${VLLM_USE_V1:-1}" \
  "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    > "$LOG_FILE" 2>&1 &
VLLM_PID=$!
echo "[$(timestamp)] vLLM started pid=${VLLM_PID} host=${HOST} port=${PORT} tp=${TP_SIZE} log=${LOG_FILE}"

wait_ready

"$PY" "$ROOT/scripts/trace2skill/run_trace2skill_vllm_eval16.py" \
  --base-url "http://${HOST}:${PORT}/v1" \
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
