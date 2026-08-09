#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}
SUPERVISOR_PID=${SUPERVISOR_PID:-1669095}
SUPERVISOR_SCREEN=${SUPERVISOR_SCREEN:-t2s_eval_forever_150704}
SUPERVISOR_ID=continuous_t2s_eval_20260722_150704
CYCLE_TAG=cycle_0003_20260722_232144
CYCLE_ROOT=$ROOT/runs/trace2skill_continuous/$SUPERVISOR_ID/$CYCLE_TAG
CYCLE_SUMMARY=$ROOT/runs/webrl_lite_full_es/${SUPERVISOR_ID}_${CYCLE_TAG}_eval3/eval_summary.json
SKILL_FILE=$CYCLE_ROOT/skill/SKILL.md
CONSTANT_HISTORY=$ROOT/runs/webrl_lite_full_es/qwen35_27b_true_noskill_strict_vab_pop8_batch8_eval10_sigma1p5e3_alpha2p5e4_4gpu_20260630_045638/history.json
ENDPOINTS=${ENDPOINTS:-http://127.0.0.1:12013,http://127.0.0.1:12014,http://127.0.0.1:12015,http://127.0.0.1:12016}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
EXTRA_RUN_ID=${EXTRA_RUN_ID:-${SUPERVISOR_ID}_${CYCLE_TAG}_extra_eval3_${STAMP}}
STATUS_LOG=$CYCLE_ROOT/stop_and_extra_eval3.log
EVAL_LOG=$ROOT/logs/$EXTRA_RUN_ID.log

record() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS_LOG"
}

record "waiting for Cycle 3 original eval3"
while [[ ! -s "$CYCLE_SUMMARY" ]]; do
  if ! kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
    record "supervisor exited before Cycle 3 summary was written"
    exit 1
  fi
  sleep 2
done

original_mean=$(
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["mean"])' "$CYCLE_SUMMARY"
)
record "Cycle 3 original eval3 complete mean=$original_mean"

# Stop the sampling loop before it can retain work from a fourth cycle.
kill -TERM "$SUPERVISOR_PID" 2>/dev/null || true
for _ in $(seq 1 30); do
  if ! kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
  screen -S "$SUPERVISOR_SCREEN" -X quit 2>/dev/null || true
fi

for _ in $(seq 1 30); do
  if ! ps -eo cmd | grep -F "$SUPERVISOR_ID" | grep -v -E 'grep|queue_stop_after_cycle3' >/dev/null; then
    break
  fi
  sleep 1
done
record "continuous sampling supervisor stopped"

test -s "$SKILL_FILE"
record "Cycle 3 extra eval3 start run_id=$EXTRA_RUN_ID"
cd "$ROOT"
env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY WEBRL_LOCAL_ENABLE_THINKING=false \
  "$PY" scripts/webarena/replay_es_history_and_eval.py \
    --source-history "$CONSTANT_HISTORY" \
    --run-id "$EXTRA_RUN_ID" \
    --endpoints "$ENDPOINTS" \
    --generations 70 \
    --alpha 2.5e-4 \
    --reward-normalization zscore \
    --parameter-scope full \
    --eval-workers-per-endpoint 4 \
    --eval-repeats 3 \
    --skill-file "$SKILL_FILE" \
    --instruction-path agent/prompts/jsons/p_webrl_chat_qwen_action.json \
    --model-name Qwen3.5-27B \
    --mode chat \
    --stop-token "" \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --repetition-penalty 1.0 >"$EVAL_LOG" 2>&1

extra_summary=$ROOT/runs/webrl_lite_full_es/$EXTRA_RUN_ID/eval_summary.json
test -s "$extra_summary"
extra_mean=$(
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["mean"])' "$extra_summary"
)
record "Cycle 3 extra eval3 complete mean=$extra_mean summary=$extra_summary"
