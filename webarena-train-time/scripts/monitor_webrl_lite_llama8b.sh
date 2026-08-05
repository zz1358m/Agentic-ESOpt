#!/usr/bin/env sh
set -eu

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python}
INTERVAL=${1:-60}
OUT=${2:-$ROOT/runs/webrl_lite_monitor_llama8b/monitor.log}

cd "$ROOT"
mkdir -p "$(dirname "$OUT")"

while true; do
  {
    date -Is
    echo "[gpu]"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    echo "[processes]"
    ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd \
      | rg 'llama31_instruct_server|run_webrl_lite_full_es_train.py|run_webrl_lite_distributed_es_train.py|osm_local_proxy.py|train_skillopt_webagent_epoch_eval|run_webrl_lite_vab_split_eval|run.py --' \
      | rg -v rg || true
    echo "[scores]"
    "$PY" - <<'PY'
import json
from pathlib import Path


def action_scores(paths):
    rows = []
    for path in sorted(paths):
        try:
            rows.append((path.stem, float(json.loads(path.read_text()).get("score"))))
        except Exception:
            pass
    return rows


def summarize(name, values):
    scores = [score for _, score in values]
    avg = sum(scores) / len(scores) if scores else None
    best = max(scores) if scores else None
    print(f"{name}: n={len(scores)} avg={avg} max={best} rows={values[-12:]}")


base_root = Path("runs/webrl_lite_llama8b_full/base_base_test_llama8b_20260604/actions")
summarize("base_test", action_scores(base_root.glob("*.json")))

for name, root in [
    ("base_es", Path("runs/webrl_lite_full_es/base_full_es_pop32_batch4_sigma1e-3_alpha1e-3_v2_20260604")),
    ("skillopt_es", Path("runs/webrl_lite_full_es/skillopt_full_es_pop32_batch4_sigma1e-3_alpha1e-3_v2_20260604")),
]:
    by_sample = {}
    for path in root.rglob("actions/*.json"):
        sample = next((part for part in path.parts if part.startswith("gen_000_sample_")), "unknown")
        try:
            by_sample.setdefault(sample, []).append(float(json.loads(path.read_text()).get("score")))
        except Exception:
            pass
    print(f"{name}: samples={len(by_sample)} history={(root / 'history.json').exists()}")
    for sample in sorted(by_sample)[-4:]:
        vals = by_sample[sample]
        avg = sum(vals) / len(vals) if vals else None
        print(f"  {sample}: n={len(vals)} avg={avg} vals={vals}")

skillopt_root = Path("runs/skillopt_webagent_lite/skillopt_epoch_eval_llama8b_v2_20260604")
selection = skillopt_root / "selection_eval_baseline/results.jsonl"
if selection.exists():
    rows = [json.loads(line) for line in selection.read_text().splitlines() if line.strip()]
    hard = sum(row.get("hard", 0) for row in rows) / len(rows) if rows else None
    soft = sum(row.get("soft", 0) for row in rows) / len(rows) if rows else None
    print(f"skillopt_selection: n={len(rows)} hard_avg={hard} soft_avg={soft}")
print(f"skillopt_conversations: n={len(list(skillopt_root.rglob('conversation.json')))}")
print(f"skillopt_steps: n={len(list((skillopt_root / 'steps').glob('step_*'))) if (skillopt_root / 'steps').exists() else 0}")

latest_path = Path("runs/webrl_lite_full_es/latest_run_id.txt")
if latest_path.exists():
    run_id = latest_path.read_text().strip()
    root = Path("runs/webrl_lite_full_es") / run_id
    log = root / "train.log"
    history = root / "history.json"
    print(f"latest_es_run: {run_id}")
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        print("latest_train_log_tail:")
        for line in lines[-12:]:
            print(f"  {line}")
    if history.exists():
        data = json.loads(history.read_text())
        print(f"latest_history_records: {len(data)}")
        for rec in data[-3:]:
            eval_rec = rec.get("eval", {})
            print(
                f"  gen={rec.get('generation')} kind={rec.get('kind','epoch')} "
                f"eval_count={eval_rec.get('count')} avg={eval_rec.get('average')} max={eval_rec.get('max')}"
            )
PY
    echo
  } >> "$OUT" 2>&1
  sleep "$INTERVAL"
done
