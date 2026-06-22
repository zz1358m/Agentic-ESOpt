#!/usr/bin/env sh
set -eu

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python}
SERVER=$ROOT/ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py
MODEL=${MODEL:-}
if [ -z "$MODEL" ]; then
  echo "Set MODEL to a local HF model path or model id." >&2
  exit 2
fi
MODEL_NAME=$(basename "$MODEL")
LOGDIR=${LOGDIR:-$ROOT/runs/webrl_lite_llama8b_services_$MODEL_NAME}
LORA_ARGS=${LORA_ARGS:-}

mkdir -p "$LOGDIR"

unset DISPLAY XAUTHORITY WAYLAND_DISPLAY

if [ -d "$MODEL" ] && [ "$(find "$MODEL" -maxdepth 1 -name 'model-*.safetensors' | wc -l)" -lt 4 ]; then
  echo "model safetensors are incomplete under $MODEL" >&2
  exit 1
fi

ps -eo pid,args | awk '/llama31_instruct_server.py/ && /1101[3-6]/ {print $1}' | xargs -r kill
sleep 2

setsid env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY "$PY" "$SERVER" --path "$MODEL" --port 11013 --host 127.0.0.1 --dtype bfloat16 --max-repeat-prompt 8 --d 0 $LORA_ARGS \
  > "$LOGDIR/gpu0_es_rep0.log" 2>&1 < /dev/null &
setsid env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY "$PY" "$SERVER" --path "$MODEL" --port 11014 --host 127.0.0.1 --dtype bfloat16 --max-repeat-prompt 8 --d 1 $LORA_ARGS \
  > "$LOGDIR/gpu1_es_rep1.log" 2>&1 < /dev/null &
setsid env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY "$PY" "$SERVER" --path "$MODEL" --port 11015 --host 127.0.0.1 --dtype bfloat16 --max-repeat-prompt 8 --d 2 $LORA_ARGS \
  > "$LOGDIR/gpu2_es_rep2.log" 2>&1 < /dev/null &
setsid env -u DISPLAY -u XAUTHORITY -u WAYLAND_DISPLAY "$PY" "$SERVER" --path "$MODEL" --port 11016 --host 127.0.0.1 --dtype bfloat16 --max-repeat-prompt 8 --d 3 $LORA_ARGS \
  > "$LOGDIR/gpu3_es_rep3.log" 2>&1 < /dev/null &

echo "started ES model servers for $MODEL on ports 11013, 11014, 11015, 11016"
