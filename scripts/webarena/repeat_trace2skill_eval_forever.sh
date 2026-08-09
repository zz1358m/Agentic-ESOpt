#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-/home/zhi/miniconda3/envs/es4llm/bin/python}
ENDPOINTS=${ENDPOINTS:-http://127.0.0.1:12013,http://127.0.0.1:12014,http://127.0.0.1:12015,http://127.0.0.1:12016}
TRACE_SOURCE=${TRACE_SOURCE:-$ROOT/runs/trace2skill_webarena_sft/constant_last10_gpt54mini_resampled_webarena_medium_20260722_090304/step_001/update_001/trace_logs}
EMPTY_SKILL=${EMPTY_SKILL:-$ROOT/runs/trace2skill_reanalysis/webarena_629_grouped_gpt54mini_unlimited_medium_20260721/skill_backup_20260721_083359/SKILL.md}
CONSTANT_HISTORY=${CONSTANT_HISTORY:-$ROOT/runs/webrl_lite_full_es/qwen35_27b_true_noskill_strict_vab_pop8_batch8_eval10_sigma1p5e3_alpha2p5e4_4gpu_20260630_045638/history.json}
TRACE_SRC=$ROOT/webarena-train-time/methods/trace2skill/source
SUPERVISOR_ID=${SUPERVISOR_ID:-continuous_t2s_eval_$(date -u +%Y%m%d_%H%M%S)}
SUPERVISOR_ROOT=$ROOT/runs/trace2skill_continuous/$SUPERVISOR_ID
STATUS_LOG=$SUPERVISOR_ROOT/status.log

mkdir -p "$SUPERVISOR_ROOT" "$ROOT/logs"

record() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS_LOG"
}

stop_requested=0
request_stop() {
  stop_requested=1
  record "stop signal received; terminating the active child"
  if [[ -n "${active_pid:-}" ]]; then
    kill -TERM "$active_pid" 2>/dev/null || true
  fi
}
trap request_stop INT TERM HUP

run_active() {
  "$@" &
  active_pid=$!
  wait "$active_pid"
  local rc=$?
  active_pid=
  if (( stop_requested )); then
    exit 0
  fi
  return "$rc"
}

run_analysis_until_complete() {
  local kind=$1
  local output_dir=$2
  local runner system_prompt user_prompt expected
  if [[ "$kind" == error ]]; then
    runner=analysis/run_error_analysis_llm.py
    system_prompt=$ROOT/webarena-train-time/methods/trace2skill/prompts/webarena_error_system.txt
    user_prompt=$ROOT/webarena-train-time/methods/trace2skill/prompts/webarena_error_user.txt
    expected=486
  else
    runner=analysis/run_success_analysis_llm.py
    system_prompt=$ROOT/webarena-train-time/methods/trace2skill/prompts/webarena_success_system.txt
    user_prompt=$ROOT/webarena-train-time/methods/trace2skill/prompts/webarena_success_user.txt
    expected=135
  fi

  local attempt=1
  while true; do
    record "$kind analysis attempt=$attempt"
    set +e
    (
      cd "$TRACE_SRC"
      env \
        OPENAI_API_KEY="$(<"$ROOT/apikey")" \
        OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
        "TRACE2SKILL_${kind^^}_SYSTEM_PROMPT=$system_prompt" \
        "TRACE2SKILL_${kind^^}_USER_PROMPT=$user_prompt" \
        "$PY" "$runner" \
          --logs_dir "$TRACE_SOURCE" \
          --output_dir "$output_dir" \
          --model gpt-5.4-mini \
          --base_url "${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
          --max_workers 16 \
          --generation_config '{"reasoning_effort":"medium"}'
    ) >>"$output_dir.log" 2>&1 &
    active_pid=$!
    wait "$active_pid"
    local rc=$?
    active_pid=
    set -e
    if (( stop_requested )); then
      exit 0
    fi

    local parsed=0
    if [[ -s "$output_dir/parsed_${kind}_records.json" ]]; then
      parsed=$(
        "$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
          "$output_dir/parsed_${kind}_records.json"
      )
    fi
    record "$kind analysis parsed=$parsed/$expected rc=$rc"
    if (( rc == 0 && parsed == expected )); then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 10
  done
}

cycle=1
record "supervisor start id=$SUPERVISOR_ID trace_source=$TRACE_SOURCE"
while true; do
  stamp=$(date -u +%Y%m%d_%H%M%S)
  seed=$((10#$(date -u +%Y%m%d%H%M%S) + cycle))
  cycle_tag=$(printf 'cycle_%04d_%s' "$cycle" "$stamp")
  cycle_root=$SUPERVISOR_ROOT/$cycle_tag
  update_dir=$cycle_root/update
  skill_dir=$cycle_root/skill
  mkdir -p "$update_dir" "$skill_dir" "$update_dir/evolution_intermediates"
  cp "$EMPTY_SKILL" "$skill_dir/SKILL.md"

  record "cycle=$cycle start root=$cycle_root seed=$seed"
  run_analysis_until_complete error "$update_dir/error_analysis"
  run_analysis_until_complete success "$update_dir/success_analysis"

  record "cycle=$cycle MAP/merge/consolidation start"
  set +e
  (
    cd "$TRACE_SRC"
    env \
      OPENAI_API_KEY="$(<"$ROOT/apikey")" \
      OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
      TRACE2SKILL_SKILL_DOMAIN="WebArena web-agent" \
      TRACE2SKILL_RUNTIME_POLICY="Use only the available WebRL actions and visible page evidence. Before exiting, satisfy the task's full completion contract, including requested cardinality, ties, multiple entities, and persisted state when applicable. When that complete answer is visible, exit immediately with it. When the requested state-changing task is fully and visibly confirmed complete, exit immediately. Prefer the shortest valid action sequence; learned guidance must not require extra exploration or verification after these completion conditions are satisfied." \
      "$PY" -m skill_evolver.run_parallel_combined_skill_evolution \
        --error-json "$update_dir/error_analysis/parsed_error_records.json" \
        --success-json "$update_dir/success_analysis/parsed_success_records.json" \
        --skill-dir "$skill_dir" \
        --model gpt-5.4-mini \
        --base-url "${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
        --max-workers 16 \
        --batch-size 1 \
        --merge-batch-size 5 \
        --patch-pipeline json \
        --save-intermediates \
        --intermediates-dir "$update_dir/evolution_intermediates" \
        --changelog "$update_dir/change.log" \
        --seed "$seed" \
        --max-skill-lines 0 \
        --max-skill-tokens 0 \
        --max-references 0 \
        --max-verification-rounds 0 \
        --group-records-by-task \
        --generation-config '{"reasoning_effort":"medium"}' \
        --reasoning-effort medium \
        --consolidation-reasoning-effort medium
  ) >"$update_dir/evolve.log" 2>&1 &
  active_pid=$!
  wait "$active_pid"
  evolve_rc=$?
  active_pid=
  set -e
  if (( stop_requested )); then
    exit 0
  fi
  if (( evolve_rc != 0 )); then
    record "cycle=$cycle evolve failed rc=$evolve_rc; advancing to a fresh cycle"
    cycle=$((cycle + 1))
    continue
  fi

  if ! "$PY" "$ROOT/skills/skill-creator/scripts/quick_validate.py" "$skill_dir" \
    >"$update_dir/validate.log" 2>&1; then
    record "cycle=$cycle skill validation failed; advancing to a fresh cycle"
    cycle=$((cycle + 1))
    continue
  fi
  record "cycle=$cycle skill complete lines=$(wc -l <"$skill_dir/SKILL.md")"

  eval_run_id="${SUPERVISOR_ID}_${cycle_tag}_eval3"
  record "cycle=$cycle eval3 start run_id=$eval_run_id"
  set +e
  (
    cd "$ROOT"
    env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY WEBRL_LOCAL_ENABLE_THINKING=false \
      "$PY" scripts/webarena/replay_es_history_and_eval.py \
        --source-history "$CONSTANT_HISTORY" \
        --run-id "$eval_run_id" \
        --endpoints "$ENDPOINTS" \
        --generations 70 \
        --alpha 2.5e-4 \
        --reward-normalization zscore \
        --parameter-scope full \
        --eval-workers-per-endpoint 4 \
        --eval-repeats 3 \
        --skill-file "$skill_dir/SKILL.md" \
        --instruction-path agent/prompts/jsons/p_webrl_chat_qwen_action.json \
        --model-name Qwen3.5-27B \
        --mode chat \
        --stop-token "" \
        --temperature 0.7 \
        --top-p 0.8 \
        --top-k 20 \
        --min-p 0.0 \
        --presence-penalty 1.5 \
        --repetition-penalty 1.0
  ) >"$cycle_root/eval.log" 2>&1 &
  active_pid=$!
  wait "$active_pid"
  eval_rc=$?
  active_pid=
  set -e
  if (( stop_requested )); then
    exit 0
  fi
  if (( eval_rc != 0 )); then
    record "cycle=$cycle eval3 failed rc=$eval_rc; advancing to a fresh cycle"
    cycle=$((cycle + 1))
    continue
  fi

  eval_summary=$ROOT/runs/webrl_lite_full_es/$eval_run_id/eval_summary.json
  test -s "$eval_summary"
  mean=$(
    "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["mean"])' "$eval_summary"
  )
  record "cycle=$cycle eval3 complete mean=$mean summary=$eval_summary"
  cycle=$((cycle + 1))
done
