#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT / "sudoku-train-time"))

from envs.sudoku import SudokuTask, build_prompt, extract_board, load_tasks, score_board  # noqa: E402


DEFAULT_TRAIN = ROOT / "data/sudoku/train.jsonl"
DEFAULT_EVAL = ROOT / "data/sudoku/eval.jsonl"


def completion_text(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list) and item and isinstance(item[-1], dict):
        return str(item[-1].get("content", ""))
    if isinstance(item, dict):
        return str(item.get("content", item.get("text", "")))
    return str(item)


def make_rows(tasks: list[SudokuTask]) -> list[dict]:
    rows = []
    for task in tasks:
        rows.append(
            {
                "prompt": build_prompt(task),
                "task_id": task.id,
                "puzzle": task.puzzle,
                "solution": task.solution,
                "mask_count": task.mask_count,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("SUDOKU_GRPO_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL))
    parser.add_argument("--mask-count", type=int, default=int(os.environ.get("SUDOKU_TARGET_MASK_COUNT", "50")))
    parser.add_argument("--output-dir", default=os.environ.get("SUDOKU_GRPO_OUTPUT_DIR", str(ROOT / "runs/sudoku_grpo/llama31_8b")))
    parser.add_argument("--train-limit", type=int, default=int(os.environ.get("SUDOKU_GRPO_TRAIN_LIMIT", "0")))
    parser.add_argument("--eval-limit", type=int, default=int(os.environ.get("SUDOKU_GRPO_EVAL_LIMIT", "100")))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("SUDOKU_GRPO_MAX_PROMPT_LENGTH", "1024")))
    parser.add_argument("--max-completion-length", type=int, default=int(os.environ.get("SUDOKU_GRPO_MAX_COMPLETION_LENGTH", "512")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("SUDOKU_GRPO_LR", "1e-6")))
    parser.add_argument("--num-generations", type=int, default=int(os.environ.get("SUDOKU_GRPO_NUM_GENERATIONS", "8")))
    parser.add_argument("--per-device-train-batch-size", type=int, default=int(os.environ.get("SUDOKU_GRPO_BATCH", "1")))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=int(os.environ.get("SUDOKU_GRPO_GRAD_ACCUM", "8")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("SUDOKU_GRPO_MAX_STEPS", "200")))
    parser.add_argument("--beta", type=float, default=float(os.environ.get("SUDOKU_GRPO_BETA", "0.04")))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=os.environ.get("SUDOKU_GRPO_BF16", "1") == "1")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "GRPO training requires optional packages: trl, datasets, transformers, accelerate, and torch."
        ) from exc

    train_tasks = load_tasks(args.train_data, limit=args.train_limit, mask_count=args.mask_count)
    eval_tasks = load_tasks(args.eval_data, limit=args.eval_limit, mask_count=args.mask_count)
    train_dataset = Dataset.from_list(make_rows(train_tasks))
    eval_dataset = Dataset.from_list(make_rows(eval_tasks)) if eval_tasks else None

    def sudoku_reward(completions, puzzle, **kwargs):
        rewards = []
        for completion, puzzle_grid in zip(completions, puzzle):
            board = extract_board(completion_text(completion))
            rewards.append(score_board(puzzle_grid, board))
        return rewards

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=1,
        save_steps=max(25, args.max_steps // 4),
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[sudoku_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
