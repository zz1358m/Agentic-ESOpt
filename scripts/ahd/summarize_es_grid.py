#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_objective(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        objective = data.get("objective")
        return None if objective is None else float(objective)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    progress_path = Path(args.progress)
    rows = load_jsonl(progress_path)

    latest = {}
    for row in rows:
        key = (row["task"], row["sigma"], row["alpha"], row["rep"])
        latest[key] = row

    combos = {}
    for key, row in latest.items():
        result_path = Path(row["result_path"])
        objective = load_objective(result_path) if row["status"] == "completed" else None
        task, sigma, alpha, rep = key
        combo_key = (task, sigma, alpha)
        combos.setdefault(combo_key, []).append(
            {
                "rep": rep,
                "status": row["status"],
                "exit_code": row["exit_code"],
                "result_path": row["result_path"],
                "objective": objective,
            }
        )

    summary = {"progress": str(progress_path), "combos": []}
    for (task, sigma, alpha), reps in sorted(combos.items()):
        reps = sorted(reps, key=lambda x: x["rep"])
        objectives = [r["objective"] for r in reps if r["objective"] is not None]
        summary["combos"].append(
            {
                "task": task,
                "sigma": sigma,
                "alpha": alpha,
                "n_completed": sum(1 for r in reps if r["status"] == "completed"),
                "mean_objective": mean(objectives) if objectives else None,
                "reps": reps,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
