#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -f "$ROOT/scripts/settings.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/settings.local.env"
fi

PY="${PY:-python}"
MODEL_PATH="${MODEL_PATH:-${MATH_MODEL_PATH:-${DOCVQA_MODEL_PATH:-}}}"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5-4B checkpoint or Hugging Face model ID.}"
MODEL_NAME="${MODEL_NAME:-${MATH_MODEL_NAME:-Qwen3.5-4B}}"
DOCVQA_ROOT="${DOCVQA_ROOT:-${ROOT}}"
DOCVQA_DATA="${DOCVQA_DATA:-${DOCVQA_ROOT}/data/trace2skill/docvqa/test.jsonl}"
DOCVQA_TOOL_PREFIX="${DOCVQA_TOOL_PREFIX:-${DOCVQA_ROOT}/.tools/tesseract/root/usr}"
RUN_ID="${RUN_ID:-qwen35_4b_react_math_docvqa_eval16_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/trace2skill_eval16/${RUN_ID}}"
PORTS_CSV="${PORTS_CSV:-18080,18081,18082,18083}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"

if [[ "$PY" != */* ]]; then
  PY="$(command -v "$PY")"
fi

cd "$ROOT"
mkdir -p logs "$OUT_DIR"
export PATH="$(dirname "$PY"):${PATH}"
export DOCVQA_TOOL_PREFIX

IFS=',' read -r -a PORTS <<< "$PORTS_CSV"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#PORTS[@]}" -ne "${#GPUS[@]}" ]]; then
  echo "PORTS_CSV and GPUS_CSV must have the same length." >&2
  exit 1
fi

PIDS=()
LOGS=()

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

stop_servers() {
  [[ "$KEEP_SERVERS" == "1" ]] && return 0
  local pid
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    fi
  done
}
trap stop_servers EXIT INT TERM

start_server() {
  local idx="$1"
  local gpu="${GPUS[$idx]}"
  local port="${PORTS[$idx]}"
  local log="${ROOT}/logs/${RUN_ID}_vllm_gpu${gpu}_port${port}.log"
  local compile_cache="${ROOT}/runs/vllm_compile_cache/qwen35_4b_gpu${gpu}"
  mkdir -p "$compile_cache"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    TRITON_CACHE_DIR="$compile_cache/triton" \
    "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --host 127.0.0.1 \
      --port "$port" \
      --tensor-parallel-size 1 \
      --dtype bfloat16 \
      --trust-remote-code \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --enforce-eager \
      --no-enable-prefix-caching \
      --gdn-prefill-backend triton \
      --language-model-only \
      >"$log" 2>&1 < /dev/null &
  PIDS+=("$!")
  LOGS+=("$log")
  echo "[$(timestamp)] vLLM started gpu=${gpu} port=${port} pid=$! log=${log}"
}

wait_ready() {
  local idx="$1"
  local port="${PORTS[$idx]}"
  local pid="${PIDS[$idx]}"
  local log="${LOGS[$idx]}"
  "$PY" - "$port" "$pid" "$log" "$MODEL_NAME" <<'PY'
import json
import os
import sys
import time
import urllib.request

port = int(sys.argv[1])
pid = int(sys.argv[2])
log = sys.argv[3]
model = sys.argv[4]
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Return only OK."}],
    "temperature": 0.0,
    "max_tokens": 8,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
last = None
for _ in range(360):
    try:
        os.kill(pid, 0)
    except OSError:
        tail = ""
        try:
            with open(log, encoding="utf-8", errors="replace") as fh:
                tail = "".join(fh.readlines()[-80:])
        except OSError:
            pass
        raise SystemExit(f"vLLM pid {pid} exited before readiness:\n{tail}")
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        if data.get("choices"):
            print(f"[ready] port={port}", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        last = repr(exc)
    time.sleep(5)
raise SystemExit(f"vLLM port {port} did not become ready: {last}")
PY
}

# Warm the shared model/compiler caches once, then start the remaining replicas.
start_server 0
wait_ready 0
for idx in "${!PORTS[@]}"; do
  ((idx == 0)) && continue
  start_server "$idx"
done
for idx in "${!PORTS[@]}"; do
  ((idx == 0)) && continue
  wait_ready "$idx"
done

SKILL_ARGS=()
if [[ -n "${MATH_SKILL_FILE:-}" ]]; then
  SKILL_ARGS+=(--math-skill-file "$MATH_SKILL_FILE")
fi
if [[ -n "${DOCVQA_SKILL_FILE:-}" ]]; then
  SKILL_ARGS+=(--docvqa-skill-file "$DOCVQA_SKILL_FILE")
fi

"$PY" "$ROOT/scripts/trace2skill/run_trace2skill_vllm_eval16.py" \
  --base-urls "$(base_urls_csv)" \
  --model "$MODEL_NAME" \
  --tokenizer-path "$MODEL_PATH" \
  --math-root "$ROOT/data/trace2skill/math_reasoning" \
  --docvqa-root "$DOCVQA_ROOT" \
  --docvqa-data "$DOCVQA_DATA" \
  --out-dir "$OUT_DIR" \
  "${SKILL_ARGS[@]}" \
  --datasets "${TRACE2SKILL_EVAL_DATASETS:-dapo100,aime2026,docvqa}" \
  --samples "${TRACE2SKILL_EVAL_SAMPLES:-4}" \
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
  --math-max-tokens "${TRACE2SKILL_EVAL_MATH_MAX_TOKENS:-4096}" \
  --math-max-turns "${TRACE2SKILL_EVAL_MATH_MAX_TURNS:-50}" \
  --math-python-timeout "${TRACE2SKILL_EVAL_MATH_PYTHON_TIMEOUT:-20.0}" \
  --docvqa-max-tokens "${TRACE2SKILL_EVAL_DOCVQA_MAX_TOKENS:-512}" \
  --docvqa-max-total-tokens "${TRACE2SKILL_EVAL_DOCVQA_MAX_TOTAL_TOKENS:-32768}" \
  --docvqa-max-turns "${TRACE2SKILL_EVAL_DOCVQA_MAX_TURNS:-50}" \
  --docvqa-limit "${TRACE2SKILL_EVAL_DOCVQA_LIMIT:-100}" \
  --docvqa-python-timeout "${TRACE2SKILL_EVAL_DOCVQA_PYTHON_TIMEOUT:-20.0}" \
  --tool-observation-limit "${TRACE2SKILL_EVAL_TOOL_OBSERVATION_LIMIT:-6000}" \
  --log-every "${TRACE2SKILL_EVAL_LOG_EVERY:-25}" \
  --resume \
  "$@"
