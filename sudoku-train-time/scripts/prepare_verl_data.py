#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT / "sudoku-train-time"))

from envs.sudoku import SudokuTask, format_board, load_tasks  # noqa: E402


def make_prompt(task: SudokuTask) -> list[dict]:
    system = (
        "You are a Sudoku agent. At each turn, output exactly one action and nothing else. "
        "The action format is: set <row> <col> <value>. Rows and columns are 1-indexed."
    )
    user = (
        f"Solve this Sudoku puzzle one cell at a time. Mask count: {task.mask_count}.\n"
        f"{format_board(task.puzzle)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def row_for_task(task: SudokuTask, split: str, index: int) -> dict:
    ground_truth = {
        "task_id": task.id,
        "puzzle": task.puzzle,
        "solution": task.solution,
        "mask_count": task.mask_count,
    }
    tools_kwargs = {"sudoku": {"create_kwargs": ground_truth}}
    return {
        "data_source": "sudoku",
        "prompt": make_prompt(task),
        "ability": "agentic_sudoku",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "split": split,
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
            **ground_truth,
        },
        "metadata": {"mask_count": task.mask_count, "task_id": task.id},
    }


def write_parquet(rows: list[dict], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("prepare_verl_data.py requires pandas and pyarrow/fastparquet for parquet output.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=str(ROOT / "data/sudoku/train.jsonl"))
    parser.add_argument("--eval-data", default=str(ROOT / "data/sudoku/eval.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/sudoku/verl"))
    parser.add_argument("--mask-count", type=int, default=int(os.environ.get("SUDOKU_TARGET_MASK_COUNT", "50")))
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()

    train_tasks = load_tasks(args.train_data, limit=args.train_limit, mask_count=args.mask_count)
    eval_tasks = load_tasks(args.eval_data, limit=args.eval_limit, mask_count=args.mask_count)
    output_dir = Path(args.output_dir)
    write_parquet([row_for_task(task, "train", idx) for idx, task in enumerate(train_tasks)], output_dir / "train.parquet")
    write_parquet([row_for_task(task, "eval", idx) for idx, task in enumerate(eval_tasks)], output_dir / "eval.parquet")
    print(f"Wrote verl Sudoku data to {output_dir} for mask_count={args.mask_count}")


if __name__ == "__main__":
    main()
