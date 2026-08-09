#!/usr/bin/env sh
set -eu

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python}
SERVER=${WEBARENA_ES_SERVER:-$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py}
MODEL=${MODEL:-}
if [ -z "$MODEL" ]; then
  echo "Set MODEL to the local Qwen3.5-27B checkpoint or Hugging Face model id." >&2
  exit 2
fi

MODEL_NAME=$(basename "$MODEL")
LOGDIR=${LOGDIR:-$ROOT/runs/webarena_es_servers_$MODEL_NAME}
LORA_ARGS=${LORA_ARGS:-}
mkdir -p "$LOGDIR"
unset DISPLAY XAUTHORITY WAYLAND_DISPLAY

if [ -d "$MODEL" ] && [ "$(find "$MODEL" -maxdepth 1 -name 'model-*.safetensors' | wc -l)" -lt 4 ]; then
  echo "Model safetensors are incomplete under $MODEL" >&2
  exit 1
fi

for port_gpu in 11013:0 11014:1 11015:2 11016:3; do
  port=${port_gpu%%:*}
  gpu=${port_gpu##*:}
  setsid env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY \
    "$PY" "$SERVER" \
      --path "$MODEL" \
      --port "$port" \
      --host 127.0.0.1 \
      --dtype bfloat16 \
      --max-repeat-prompt 8 \
      --d "$gpu" \
      $LORA_ARGS \
      > "$LOGDIR/server_${port}.log" 2>&1 < /dev/null &
done

echo "Started four Agentic-ESOpt model replicas on ports 11013-11016."
