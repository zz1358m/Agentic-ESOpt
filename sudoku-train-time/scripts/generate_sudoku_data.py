#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SIDE = 9
BASE = 3


def shuffled(rng: random.Random, values: list[int]) -> list[int]:
    values = list(values)
    rng.shuffle(values)
    return values


def pattern(row: int, col: int) -> int:
    return (BASE * (row % BASE) + row // BASE + col) % SIDE


def generate_solution(rng: random.Random) -> list[list[int]]:
    rows = [group * BASE + row for group in shuffled(rng, list(range(BASE))) for row in shuffled(rng, list(range(BASE)))]
    cols = [group * BASE + col for group in shuffled(rng, list(range(BASE))) for col in shuffled(rng, list(range(BASE)))]
    nums = shuffled(rng, list(range(1, SIDE + 1)))
    return [[nums[pattern(row, col)] for col in cols] for row in rows]


def mask_solution(solution: list[list[int]], rng: random.Random, mask_count: int) -> list[list[int]]:
    if not 0 <= mask_count <= 81:
        raise ValueError("--mask-counts values must be between 0 and 81")
    puzzle = [row[:] for row in solution]
    positions = [(row, col) for row in range(SIDE) for col in range(SIDE)]
    for row, col in shuffled(rng, positions)[:mask_count]:
        puzzle[row][col] = 0
    return puzzle


def compact(board: list[list[int]], *, blank: str = ".") -> str:
    return "".join(blank if cell == 0 else str(cell) for row in board for cell in row)


def write_split(path: Path, *, size: int, mask_counts: list[int], rng: random.Random, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(size):
            solution = generate_solution(rng)
            mask_count = mask_counts[idx % len(mask_counts)]
            puzzle = mask_solution(solution, rng, mask_count)
            row = {
                "id": f"{split}-{idx:06d}",
                "source": "generated",
                "split": split,
                "mask_count": mask_count,
                "puzzle": puzzle,
                "solution": solution,
                "puzzle_compact": compact(puzzle),
                "solution_compact": compact(solution, blank="0"),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def parse_mask_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts:
        raise ValueError("At least one mask count is required")
    for count in counts:
        if count < 0 or count > 81:
            raise ValueError(f"Invalid mask count: {count}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sudoku")
    parser.add_argument("--train-size", type=int, default=192)
    parser.add_argument("--eval-size", type=int, default=192)
    parser.add_argument("--mask-counts", default="5,10,15,20")
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    mask_counts = parse_mask_counts(args.mask_counts)
    rng = random.Random(args.seed)
    write_split(output_dir / "train.jsonl", size=args.train_size, mask_counts=mask_counts, rng=rng, split="train")
    write_split(output_dir / "eval.jsonl", size=args.eval_size, mask_counts=mask_counts, rng=rng, split="eval")
    print(f"Wrote {args.train_size} train and {args.eval_size} eval puzzles to {output_dir}")


if __name__ == "__main__":
    main()
