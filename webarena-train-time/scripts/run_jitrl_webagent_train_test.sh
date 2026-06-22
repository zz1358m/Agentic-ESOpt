#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
JITRL_WEBARENA="${JITRL_WEBARENA:-$ROOT/data/webarena/jitrl}"
MODE="${MODE:-all}" # all | jitrl | train | test
MODEL="${MODEL:-${POLICY_COMPLETIONS_URL:-http://127.0.0.1:11013/completions}}"
LLM_EVAL="${LLM_EVAL:-$MODEL}"
LLM_EXTRACT="${LLM_EXTRACT:-$MODEL}"
TASKS="${TASKS:-0}"
START="${START:-}"
END="${END:-}"
WORKERS="${WORKERS:-1}"
MAX_STEPS="${MAX_STEPS:-30}"
TRAIN_REPEAT="${TRAIN_REPEAT:-3}"
TEST_REPEAT="${TEST_REPEAT:-1}"
BASELINE_REPEAT="${BASELINE_REPEAT:-1}"
TASK_TIMEOUT="${TASK_TIMEOUT:-1200}"
TASK_SIMILARITY_THRESHOLD="${TASK_SIMILARITY_THRESHOLD:-0.27}"
CHECK_SITES="${CHECK_SITES:-all}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/jitrl_webarena_${STAMP}}"
MEMORY_ROOT="${MEMORY_ROOT:-$OUT_ROOT/memory}"

if [ -f "$ROOT/apikey" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY="$(tr -d '\n\r ' < "$ROOT/apikey")"
fi

if [ -f "$JITRL_WEBARENA/env_setup.txt" ]; then
  # shellcheck disable=SC1091
  source "$JITRL_WEBARENA/env_setup.txt"
fi

if [ "$CHECK_SITES" = "all" ]; then
  required_urls=(WA_SHOPPING WA_SHOPPING_ADMIN WA_REDDIT WA_GITLAB WA_MAP WA_WIKIPEDIA)
else
  required_urls=()
  IFS=',' read -r -a check_sites_array <<< "$CHECK_SITES"
  for site in "${check_sites_array[@]}"; do
    case "$site" in
      shopping) required_urls+=(WA_SHOPPING) ;;
      shopping_admin) required_urls+=(WA_SHOPPING_ADMIN) ;;
      reddit) required_urls+=(WA_REDDIT) ;;
      gitlab) required_urls+=(WA_GITLAB) ;;
      map) required_urls+=(WA_MAP) ;;
      wikipedia) required_urls+=(WA_WIKIPEDIA) ;;
      homepage) required_urls+=(WA_HOMEPAGE) ;;
      *)
        echo "Unknown site in CHECK_SITES: $site"
        exit 2
        ;;
    esac
  done
fi
missing=()
for name in "${required_urls[@]}"; do
  if [ -z "${!name:-}" ]; then
    missing+=("$name")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing JitRL/WebArena URL vars: ${missing[*]}"
  echo "Run: scripts/setup_jitrl_webarena_env.sh <host>"
  exit 2
fi

unreachable=()
for name in "${required_urls[@]}"; do
  code="$(curl -k -L -s -o /dev/null -w '%{http_code}' --max-time 5 "${!name}" || true)"
  if [ -z "$code" ] || [ "$code" = "000" ]; then
    unreachable+=("$name=${!name}")
  fi
done
if [ "${#unreachable[@]}" -gt 0 ]; then
  echo "Unreachable WebArena services: ${unreachable[*]}"
  echo "Start the WebArena websites or point WEBARENA_HOST to a reachable deployment."
  exit 3
fi

mkdir -p "$OUT_ROOT"/{logs,results}

task_args=()
if [ -n "$TASKS" ]; then
  task_args+=(--tasks "$TASKS")
else
  task_args+=(--start "$START" --end "$END")
fi

common_args=(
  --model "$MODEL"
  --llm_eval "$LLM_EVAL"
  --llm_extract "$LLM_EXTRACT"
  --workers "$WORKERS"
  --max_steps "$MAX_STEPS"
  --task_timeout "$TASK_TIMEOUT"
  --task_similarity_threshold "$TASK_SIMILARITY_THRESHOLD"
  --logit_mode verbalized
  --no_early_stop
  "${task_args[@]}"
)

run_jitrl() {
  "$PY" test_webarena_lite.py \
    "${common_args[@]}" \
    --agent_type memory \
    --disable_memory \
    --repeat "$BASELINE_REPEAT" \
    --log_dir "$OUT_ROOT/logs/jitrl" \
    --result_dir "$OUT_ROOT/results/jitrl" \
    --output_path "$MEMORY_ROOT/jitrl" \
    --game_name webarena_jitrl
}

run_train() {
  "$PY" test_webarena_lite.py \
    "${common_args[@]}" \
    --agent_type memory \
    --repeat "$TRAIN_REPEAT" \
    --log_dir "$OUT_ROOT/logs/train" \
    --result_dir "$OUT_ROOT/results/train" \
    --output_path "$MEMORY_ROOT/jitrl" \
    --game_name webarena
}

run_test() {
  "$PY" test_webarena_lite.py \
    "${common_args[@]}" \
    --agent_type memory \
    --no-save_memory \
    --repeat "$TEST_REPEAT" \
    --log_dir "$OUT_ROOT/logs/test" \
    --result_dir "$OUT_ROOT/results/test" \
    --output_path "$MEMORY_ROOT/jitrl" \
    --game_name webarena
}

(
  cd "$JITRL_WEBARENA"
  case "$MODE" in
    jitrl) run_jitrl ;;
    train) run_train ;;
    test) run_test ;;
    all)
      run_jitrl
      run_train
      run_test
      ;;
    *)
      echo "Unknown MODE=$MODE. Use all, jitrl, train, or test."
      exit 2
      ;;
  esac
)

echo "JitRL WebAgent run complete: $OUT_ROOT"
