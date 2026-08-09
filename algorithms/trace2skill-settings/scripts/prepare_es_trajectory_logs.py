#!/usr/bin/env python3
"""Select saved ES training trajectories for task-level skill distillation."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DOC_PATTERN = re.compile(
    r"^docvqa_agent_gen(?P<generation>\d+)_candidate(?P<candidate>\d+)_"
    r"seed(?P<seed>\d+)_(?:docvqa_)?(?P<task>.+)_sample(?P<sample>\d+)_"
    r"(?P<outcome>FAILED|SUCCEED)\.md$"
)
MATH_PATTERN = re.compile(
    r"^math_agent_gen(?P<generation>\d+)_candidate(?P<candidate>\d+)_"
    r"seed(?P<seed>\d+)_(?P<task>dapo_[0-9a-f-]+)_sample(?P<sample>\d+)_"
    r"(?P<outcome>FAILED|SUCCEED)\.md$"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def normalized_task_id(value: object) -> str:
    return str(value).removeprefix("docvqa_")


def scan_logs(roots: list[Path], pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_keys: dict[tuple[int, int, str, int], Path] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.glob("*.md")):
            match = pattern.match(path.name)
            if not match:
                continue
            values = match.groupdict()
            row = {
                "source": str(path.resolve()),
                "generation": int(values["generation"]),
                "candidate": int(values["candidate"]),
                "seed": int(values["seed"]),
                "task_id": normalized_task_id(values["task"]),
                "sample": int(values["sample"]),
                "outcome": values["outcome"],
            }
            key = (row["generation"], row["candidate"], row["task_id"], row["sample"])
            previous = seen_keys.get(key)
            if previous is not None:
                raise ValueError(f"Duplicate trajectory key {key}: {previous} and {path}")
            seen_keys[key] = path
            rows.append(row)
    return rows


def docvqa_units(history_path: Path, checkpoint_step: int, task_count: int) -> list[tuple[int, str]]:
    generations = {
        int(record["generation"]): [normalized_task_id(item) for item in record["case_batch"]]
        for record in read_json(history_path)
        if isinstance(record, dict) and "case_batch" in record
    }
    missing = [generation for generation in range(checkpoint_step) if generation not in generations]
    if missing:
        raise ValueError(f"History is missing pre-checkpoint generations: {missing}")
    ordered = [
        (generation, task_id)
        for generation in range(checkpoint_step)
        for task_id in generations[generation]
    ]
    selected = ordered[-task_count:]
    if len(selected) != task_count:
        raise ValueError(f"Requested {task_count} DocVQA task units, found {len(selected)}")
    if len({task_id for _, task_id in selected}) != task_count:
        raise ValueError("The last DocVQA window does not contain one occurrence of every task")
    return selected


def math_units(history_path: Path, checkpoint_step: int, task_count: int) -> list[tuple[int, str]]:
    generations = {
        int(record["generation"]): [str(item) for item in record["case_batch"]]
        for record in read_json(history_path)
        if isinstance(record, dict) and "case_batch" in record
    }
    missing = [generation for generation in range(checkpoint_step) if generation not in generations]
    if missing:
        raise ValueError(f"History is missing pre-checkpoint generations: {missing}")
    ordered = [
        (generation, task_id)
        for generation in range(checkpoint_step)
        for task_id in generations[generation]
    ]
    selected = ordered[-task_count:]
    if len(selected) != task_count:
        raise ValueError(f"Requested {task_count} Math task units, found {len(selected)}")
    if len({task_id for _, task_id in selected}) != task_count:
        raise ValueError("The last Math window does not contain one occurrence of every task")
    return selected


def select_docvqa(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = scan_logs([args.trace_root], DOC_PATTERN)
    units = docvqa_units(args.history, args.checkpoint_step, args.task_count)
    selected_units = set(units)
    selected = [row for row in rows if (row["generation"], row["task_id"]) in selected_units]
    counts_by_unit = Counter((row["generation"], row["task_id"]) for row in selected)
    invalid = {unit: count for unit, count in counts_by_unit.items() if count != args.population}
    if invalid or len(counts_by_unit) != args.task_count:
        raise ValueError(
            f"DocVQA expected {args.population} trajectories for each of {args.task_count} units; "
            f"invalid={invalid}, observed_units={len(counts_by_unit)}"
        )
    selected.sort(key=lambda row: (row["generation"], row["task_id"], row["candidate"], row["sample"]))
    if args.one_per_outcome_per_task:
        by_task_outcome: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        # Only subsample the exact last-task-occurrence pool selected above.
        # Looking up the same task ID in arbitrary earlier generations leaks
        # trajectories outside the requested pre-checkpoint window.
        for row in selected:
            by_task_outcome[(row["task_id"], row["outcome"])].append(row)
        one_per_outcome: list[dict[str, Any]] = []
        missing: list[dict[str, str]] = []
        for generation, task_id in units:
            for outcome in ("FAILED", "SUCCEED"):
                candidates = by_task_outcome[(task_id, outcome)]
                if not candidates:
                    missing.append(
                        {"generation": generation, "task_id": task_id, "outcome": outcome}
                    )
                    continue
                chosen = sorted(
                    candidates,
                    key=lambda row: (row["candidate"], row["sample"]),
                )[0]
                one_per_outcome.append(chosen)
        selected = sorted(
            one_per_outcome,
            key=lambda row: (row["generation"], row["task_id"], row["outcome"]),
        )
    else:
        missing = []
    metadata = {
        "selection": (
            "one trajectory per outcome from each exact last task occurrence"
            if args.one_per_outcome_per_task
            else "last task occurrences before checkpoint step"
        ),
        "checkpoint_step": args.checkpoint_step,
        "task_units": [{"generation": generation, "task_id": task_id} for generation, task_id in units],
        "missing_task_outcomes": missing,
    }
    return selected, metadata


def select_math(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = scan_logs(args.trace_roots, MATH_PATTERN)
    if args.one_per_outcome_per_task or args.one_error_per_task:
        if args.history is None:
            raise ValueError(
                "--history is required with --one-per-outcome-per-task or --one-error-per-task"
            )
        units = math_units(args.history, args.checkpoint_step, args.task_count)
        selected_units = set(units)
        selected = [row for row in rows if (row["generation"], row["task_id"]) in selected_units]
        counts_by_unit = Counter((row["generation"], row["task_id"]) for row in selected)
        invalid = {unit: count for unit, count in counts_by_unit.items() if count != args.population}
        if invalid or len(counts_by_unit) != args.task_count:
            raise ValueError(
                f"Math expected {args.population} trajectories for each of {args.task_count} units; "
                f"invalid={invalid}, observed_units={len(counts_by_unit)}"
            )
        by_task_outcome: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_task_outcome[(row["generation"], row["task_id"], row["outcome"])].append(row)
        required_outcomes = (
            ("FAILED", "SUCCEED") if args.one_per_outcome_per_task else ("FAILED",)
        )
        chosen_rows: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for generation, task_id in units:
            for outcome in required_outcomes:
                candidates = by_task_outcome[(generation, task_id, outcome)]
                if not candidates:
                    missing.append(
                        {"generation": generation, "task_id": task_id, "outcome": outcome}
                    )
                    continue
                chosen_rows.append(
                    sorted(candidates, key=lambda row: (row["candidate"], row["sample"]))[0]
                )
        selected = sorted(
            chosen_rows,
            key=lambda row: (row["generation"], row["task_id"], row["outcome"]),
        )
        metadata = {
            "selection": (
                "one trajectory per outcome from each exact last task occurrence"
                if args.one_per_outcome_per_task
                else "one failed trajectory from each exact last task occurrence"
            ),
            "checkpoint_step": args.checkpoint_step,
            "task_units": [
                {"generation": generation, "task_id": task_id} for generation, task_id in units
            ],
            "missing_task_outcomes": missing,
        }
        return selected, metadata

    selected = [row for row in rows if args.first_generation <= row["generation"] <= args.last_generation]
    counts_by_generation = Counter(row["generation"] for row in selected)
    expected_per_generation = args.population * args.case_batch_size
    expected_generations = list(range(args.first_generation, args.last_generation + 1))
    invalid = {
        generation: counts_by_generation[generation]
        for generation in expected_generations
        if counts_by_generation[generation] != expected_per_generation
    }
    if invalid:
        raise ValueError(
            f"Math expected {expected_per_generation} trajectories per generation; invalid={invalid}"
        )
    task_ids = {row["task_id"] for row in selected}
    expected_tasks = len(expected_generations) * args.case_batch_size
    if len(task_ids) != expected_tasks:
        raise ValueError(f"Math expected {expected_tasks} distinct tasks, found {len(task_ids)}")
    selected.sort(key=lambda row: (row["generation"], row["task_id"], row["candidate"], row["sample"]))
    metadata = {
        "selection": "all trajectories from the requested generation range",
        "first_generation": args.first_generation,
        "last_generation": args.last_generation,
    }
    return selected, metadata


def materialize(
    selected: list[dict[str, Any]],
    output_dir: Path,
    *,
    analysis_outcomes: set[str],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_rows: list[dict[str, Any]] = []
    analysis_index = 0
    for raw_index, row in enumerate(selected):
        record = dict(row)
        record["raw_index"] = raw_index
        record["selected_for_analysis"] = row["outcome"] in analysis_outcomes
        if record["selected_for_analysis"]:
            analysis_id = f"traj{analysis_index:06d}"
            analysis_index += 1
            target = output_dir / f"trajectory_{analysis_id}_{row['outcome']}.md"
            os.symlink(row["source"], target)
            record["analysis_id"] = analysis_id
            record["analysis_log"] = str(target.resolve(strict=False))
        manifest_rows.append(record)
    return manifest_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="setting", required=True)

    doc = subparsers.add_parser("docvqa")
    doc.add_argument("--history", type=Path, required=True)
    doc.add_argument("--trace-root", type=Path, required=True)
    doc.add_argument("--checkpoint-step", type=int, default=40)
    doc.add_argument("--task-count", type=int, default=50)
    doc.add_argument("--population", type=int, default=16)
    doc.add_argument("--one-per-outcome-per-task", action="store_true")
    doc.add_argument("--output-dir", type=Path, required=True)

    math = subparsers.add_parser("math_reasoning")
    math.add_argument("--trace-roots", type=Path, nargs="+", required=True)
    math.add_argument("--history", type=Path)
    math.add_argument("--checkpoint-step", type=int, default=25)
    math.add_argument("--task-count", type=int, default=50)
    math.add_argument("--first-generation", type=int, default=0)
    math.add_argument("--last-generation", type=int, default=24)
    math.add_argument("--population", type=int, default=16)
    math.add_argument("--case-batch-size", type=int, default=16)
    math_selection = math.add_mutually_exclusive_group()
    math_selection.add_argument("--one-error-per-task", action="store_true")
    math_selection.add_argument("--one-per-outcome-per-task", action="store_true")
    math.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")

    if args.setting == "docvqa":
        selected, metadata = select_docvqa(args)
        analysis_outcomes = {"FAILED", "SUCCEED"}
    else:
        selected, metadata = select_math(args)
        analysis_outcomes = (
            {"FAILED", "SUCCEED"} if args.one_per_outcome_per_task else {"FAILED"}
        )

    rows = materialize(selected, args.output_dir, analysis_outcomes=analysis_outcomes)
    raw_outcomes = Counter(row["outcome"] for row in rows)
    analysis_rows = [row for row in rows if row["selected_for_analysis"]]
    analysis_outcomes_count = Counter(row["outcome"] for row in analysis_rows)
    analysis_tasks = {row["task_id"] for row in analysis_rows}
    manifest = {
        "setting": args.setting,
        **metadata,
        "raw_trajectory_count": len(rows),
        "raw_outcomes": dict(sorted(raw_outcomes.items())),
        "analysis_trajectory_count": len(analysis_rows),
        "analysis_outcomes": dict(sorted(analysis_outcomes_count.items())),
        "analysis_task_count": len(analysis_tasks),
        "rows": rows,
    }
    write_json(args.output_dir.parent / f"{args.output_dir.name}_manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
