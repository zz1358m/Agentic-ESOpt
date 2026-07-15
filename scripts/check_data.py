#!/usr/bin/env python3
"""Check the on-disk data contract for the five maintained tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    task: str
    component: str
    path: str
    kind: str = "exists"
    expected_count: int | None = None


CHECKS = (
    Check("sudoku", "ES / multi-turn GRPO train", "data/sudoku/train.jsonl", "jsonl"),
    Check("sudoku", "ES / multi-turn GRPO eval", "data/sudoku/eval.jsonl", "jsonl"),
    Check("math", "ES / Trace2Skill / GRPO train", "data/trace2skill/math_reasoning/dapo_evolve.jsonl", "jsonl"),
    Check("math", "held-out DAPO eval", "data/trace2skill/math_reasoning/dapo_test.jsonl", "jsonl"),
    Check("math", "AIME eval", "data/trace2skill/math_reasoning/aime_2026.jsonl", "jsonl"),
    Check("docvqa", "ES / Trace2Skill / GRPO train", "data/trace2skill/docvqa/evolve.jsonl", "jsonl"),
    Check("docvqa", "held-out eval", "data/trace2skill/docvqa/test.jsonl", "jsonl"),
    Check("docvqa", "document images", "data/trace2skill/docvqa/images", "nonempty_dir"),
    Check("webarena", "VAB checkout", "data/webarena/vab-lite", "nonempty_dir"),
    Check("webarena", "shared non-Lite train split", "data/webarena/vab_nonlite_split/train/items.json", "json"),
    Check("webarena", "shared non-Lite validation split", "data/webarena/vab_nonlite_split/val/items.json", "json"),
    Check("webarena", "WebArena-Lite test split", "data/webarena/vab_lite_split/items.json", "json", 165),
    Check("webarena", "Trace2Skill implementation", "webarena-train-time/methods/trace2skill/source/skill_evolver", "nonempty_dir"),
    Check("webarena", "WebArena rollout runtime", "webarena-train-time/third_party/skillopt/skillopt/envs/webarena_sft", "nonempty_dir"),
    Check("ahd", "TSP constructive train", "data/ahd/datasets/tsp_constructive/train50_dataset.npy"),
    Check("ahd", "KP constructive train", "data/ahd/datasets/kp_constructive/train100_dataset.npy"),
    Check("ahd", "TSP ACO train", "data/ahd/datasets/tsp_aco/train50_dataset.npy"),
    Check("ahd", "CVRP ACO train", "data/ahd/datasets/cvrp_aco/train50_dataset.npy"),
    Check("ahd", "BPP ACO train", "data/ahd/datasets/bpp_offline_aco/train500_dataset.npz"),
)


def inspect(check: Check) -> tuple[bool, str]:
    path = ROOT / check.path
    if not path.exists():
        return False, "missing"
    if check.kind == "jsonl":
        if not path.is_file():
            return False, "not a file"
        rows = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        return False, f"line {line_no} is not a JSON object"
                    rows += 1
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"invalid JSONL: {exc}"
        return rows > 0, f"{rows} rows"
    if check.kind == "json":
        if not path.is_file():
            return False, "not a file"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"invalid JSON: {exc}"
        size = len(payload) if hasattr(payload, "__len__") else 1
        if check.expected_count is not None:
            return size == check.expected_count, f"{size} entries; expected {check.expected_count}"
        return size > 0, f"{size} entries"
    if check.kind == "nonempty_dir":
        if not path.is_dir():
            return False, "not a directory"
        count = sum(1 for item in path.iterdir() if item.is_file() or item.is_dir())
        return count > 0, f"{count} entries"
    if not path.is_file():
        return False, "not a file"
    return path.stat().st_size > 0, f"{path.stat().st_size} bytes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "sudoku", "math", "docvqa", "webarena", "ahd"], default="all")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any selected item is missing or empty.")
    args = parser.parse_args()

    selected = [check for check in CHECKS if args.task == "all" or check.task == args.task]
    failures = 0
    for check in selected:
        ok, detail = inspect(check)
        failures += int(not ok)
        status = "OK" if ok else "MISSING"
        print(f"{status:7} {check.task:8} {check.component}: {check.path} ({detail})")
    print(f"\n{len(selected) - failures}/{len(selected)} checks ready")
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
