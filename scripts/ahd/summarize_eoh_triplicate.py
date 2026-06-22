#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--reps", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_objective(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    objective = data.get("objective")
    return None if objective is None else float(objective)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    reps = [rep.strip() for rep in args.reps.split(",") if rep.strip()]

    summary: dict[str, object] = {
        "stamp": args.stamp,
        "split": "train",
        "method": "eoh",
        "model": "Llama-3.1-8B-Instruct",
        "tasks": {},
    }

    for task in tasks:
        runs = []
        objectives = []
        for rep in reps:
            run_id = f"{task}_train_eoh_rep{rep}_{args.stamp}"
            path = root / "cache" / "active_runs" / f"{task}_train_eoh_{run_id}" / "results" / "pops_best" / "population_generation_25.json"
            objective = load_objective(path)
            runs.append(
                {
                    "rep": int(rep),
                    "run_id": run_id,
                    "objective": objective,
                    "result_path": str(path),
                }
            )
            if objective is not None:
                objectives.append(objective)

        summary["tasks"][task] = {
            "runs": runs,
            "average_objective": mean(objectives) if objectives else None,
            "completed_runs": len(objectives),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
