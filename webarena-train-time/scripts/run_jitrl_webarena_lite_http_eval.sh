#!/usr/bin/env sh
set -eu

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python}
JITRL=${JITRL:-$ROOT/data/webarena/jitrl}
METHOD=${1:-Base}
RUN_ID=${2:-jitrl_lite_${METHOD}_$(date +%Y%m%d_%H%M%S)}
MAX_STEPS=${MAX_STEPS:-25}
WORKERS=${WORKERS:-1}
REPEAT=${REPEAT:-1}
TASK_TIMEOUT=${TASK_TIMEOUT:-1200}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/runs/jitrl_webarena_lite/$RUN_ID}
DISABLE_MEMORY=${DISABLE_MEMORY:-0}
SITES=${SITES:-shopping_admin,map,shopping,gitlab,reddit}
STAGGER_SECONDS=${STAGGER_SECONDS:-15}
LLM_EVAL=${LLM_EVAL:-gpt-4o-mini}

mkdir -p "$RESULT_ROOT"

if [ ! -d "$JITRL" ]; then
  echo "JitRL WebArena source not found: $JITRL" >&2
  exit 2
fi

run_site() {
  site=$1
  endpoint=$2
  log_dir="$RESULT_ROOT/${METHOD}_${site}"
  result_dir="$RESULT_ROOT/browsergym_${site}"
  if [ "$DISABLE_MEMORY" = "1" ]; then
    setsid nohup sh -c 'cd "$1" || exit 1; shift; exec "$@"' run-site "$JITRL" \
      env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY \
      "$PY" test_webarena_lite.py \
        --sites "$site" \
        --workers "$WORKERS" \
        --repeat "$REPEAT" \
        --max_steps "$MAX_STEPS" \
        --task_timeout "$TASK_TIMEOUT" \
        --model "$endpoint" \
        --llm_eval "$LLM_EVAL" \
        --log_dir "$log_dir" \
        --result_dir "$result_dir" \
        --disable_memory \
      > "$RESULT_ROOT/${METHOD}_${site}.log" 2>&1 &
  else
    setsid nohup sh -c 'cd "$1" || exit 1; shift; exec "$@"' run-site "$JITRL" \
      env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY \
      "$PY" test_webarena_lite.py \
        --sites "$site" \
        --workers "$WORKERS" \
        --repeat "$REPEAT" \
        --max_steps "$MAX_STEPS" \
        --task_timeout "$TASK_TIMEOUT" \
        --model "$endpoint" \
        --llm_eval "$LLM_EVAL" \
        --log_dir "$log_dir" \
        --result_dir "$result_dir" \
      > "$RESULT_ROOT/${METHOD}_${site}.log" 2>&1 &
  fi
  echo $! > "$RESULT_ROOT/${METHOD}_${site}.pid"
}

for site in $(printf '%s' "$SITES" | tr ',' ' '); do
  case "$site" in
    shopping_admin) endpoint=http://127.0.0.1:11013/completions ;;
    map) endpoint=http://127.0.0.1:11014/completions ;;
    shopping) endpoint=http://127.0.0.1:11015/completions ;;
    gitlab) endpoint=http://127.0.0.1:11016/completions ;;
    reddit) endpoint=http://127.0.0.1:11016/completions ;;
    *) echo "Unknown site: $site" >&2; exit 2 ;;
  esac
  run_site "$site" "$endpoint"
  if [ "$STAGGER_SECONDS" -gt 0 ]; then
    sleep "$STAGGER_SECONDS"
  fi
done

cat > "$RESULT_ROOT/README.md" <<EOF
# JitRL WebArena-Lite $METHOD

- Run ID: $RUN_ID
- Method: $METHOD
- Split: JitRL WebArena-Lite task ids 0-164 from installed WebArena package
- Max steps: $MAX_STEPS
- Repeat: $REPEAT
- Workers per site: $WORKERS
- Disable memory: $DISABLE_MEMORY
- Sites: $SITES
- Stagger seconds: $STAGGER_SECONDS
- Eval model: $LLM_EVAL
- Result root: $RESULT_ROOT

Site endpoints:
- Admin: http://127.0.0.1:11013/completions
- Map: http://127.0.0.1:11014/completions
- Shopping: http://127.0.0.1:11015/completions
- GitLab: http://127.0.0.1:11016/completions
- Reddit: http://127.0.0.1:11016/completions

Summary command:
\`\`\`sh
$PY $ROOT/webarena-train-time/scripts/summarize_jitrl_webarena_lite_results.py \\
  --result $METHOD $RESULT_ROOT/${METHOD}_shopping_admin/results.json \\
  --result $METHOD $RESULT_ROOT/${METHOD}_gitlab/results.json \\
  --result $METHOD $RESULT_ROOT/${METHOD}_map/results.json \\
  --result $METHOD $RESULT_ROOT/${METHOD}_reddit/results.json \\
  --result $METHOD $RESULT_ROOT/${METHOD}_shopping/results.json
\`\`\`
EOF

echo "$RUN_ID" > "$ROOT/runs/jitrl_webarena_lite/latest_run_id.txt"
echo "$RESULT_ROOT"
