from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ahd-test-time/results/merge_aco_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_aco_shards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(task: str, code_file: str, size: int, seed: int) -> dict[str, object]:
    return {
        "task": task,
        "size": size,
        "method": "method",
        "rep": 1,
        "code_file": code_file,
        "seed": seed,
        "setting_seed": {"tsp": 10_000, "cvrp": 20_000, "bpp": 30_000}[task] + size,
        "mean": 1.0,
        "valid_count": 2,
        "count": 2,
        "failure_reasons": {},
    }


def test_merge_validates_disjoint_complete_task_shards(tmp_path: Path) -> None:
    sizes = {"tsp": (20, 50, 100), "cvrp": (20, 50, 100), "bpp": (500, 1000)}
    for task, task_sizes in sizes.items():
        for shard_index in range(2):
            rows = [
                _row(task, f"/{task}/program_{shard_index}.py", size, 100)
                for size in task_sizes
            ]
            (tmp_path / f"aco_{task}_shard{shard_index}.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )

    merged, summary = MODULE.merge_shards(
        tmp_path,
        shard_count=2,
        expected_programs_per_task=2,
    )

    assert len(merged) == 16
    assert summary["programs_per_task"] == {"tsp": 2, "cvrp": 2, "bpp": 2}
    assert summary["rows_per_task"] == {"tsp": 6, "cvrp": 6, "bpp": 4}


def test_merge_rejects_a_program_repeated_across_shards(tmp_path: Path) -> None:
    sizes = {"tsp": (20, 50, 100), "cvrp": (20, 50, 100), "bpp": (500, 1000)}
    for task, task_sizes in sizes.items():
        for shard_index in range(2):
            code_file = f"/{task}/program.py" if task == "tsp" else f"/{task}/program_{shard_index}.py"
            rows = [_row(task, code_file, size, 100 + shard_index) for size in task_sizes]
            (tmp_path / f"aco_{task}_shard{shard_index}.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )

    try:
        MODULE.merge_shards(tmp_path, shard_count=2, expected_programs_per_task=2)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate program should be rejected")


def test_merge_rejects_missing_or_inconsistent_failure_reasons(tmp_path: Path) -> None:
    sizes = {"tsp": (20, 50, 100), "cvrp": (20, 50, 100), "bpp": (500, 1000)}
    for task, task_sizes in sizes.items():
        rows = [_row(task, f"/{task}/program.py", size, 100) for size in task_sizes]
        if task == "cvrp":
            rows[0]["valid_count"] = 1
            rows[0]["failure_reasons"] = {}
        (tmp_path / f"aco_{task}_shard0.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )

    try:
        MODULE.merge_shards(tmp_path, shard_count=1, expected_programs_per_task=1)
    except ValueError as exc:
        assert "failure reasons" in str(exc)
    else:
        raise AssertionError("failure reasons must account for every invalid instance")


def test_merge_rejects_inconsistent_common_random_seed(tmp_path: Path) -> None:
    sizes = {"tsp": (20, 50, 100), "cvrp": (20, 50, 100), "bpp": (500, 1000)}
    for task, task_sizes in sizes.items():
        for shard_index in range(2):
            rows = [
                _row(task, f"/{task}/program_{shard_index}.py", size, 100)
                for size in task_sizes
            ]
            if task == "tsp" and shard_index == 1:
                rows[0]["setting_seed"] = 999
            (tmp_path / f"aco_{task}_shard{shard_index}.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )

    try:
        MODULE.merge_shards(tmp_path, shard_count=2, expected_programs_per_task=2)
    except ValueError as exc:
        assert "common random seed" in str(exc)
    else:
        raise AssertionError("all programs in one task setting must share an RNG seed")


def test_merge_rejects_inconsistent_master_seed(tmp_path: Path) -> None:
    sizes = {"tsp": (20, 50, 100), "cvrp": (20, 50, 100), "bpp": (500, 1000)}
    for task, task_sizes in sizes.items():
        for shard_index in range(2):
            rows = [
                _row(
                    task,
                    f"/{task}/program_{shard_index}.py",
                    size,
                    100 + shard_index,
                )
                for size in task_sizes
            ]
            (tmp_path / f"aco_{task}_shard{shard_index}.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )

    try:
        MODULE.merge_shards(tmp_path, shard_count=2, expected_programs_per_task=2)
    except ValueError as exc:
        assert "master seed" in str(exc)
    else:
        raise AssertionError("all shards must use the same master seed")


def _partial_row(
    *,
    offset: int,
    stop: int,
    mean: float,
    std: float,
    valid_count: int,
    failure_reasons: dict[str, int] | None = None,
) -> dict[str, object]:
    failure_reasons = failure_reasons or {}
    count = stop - offset
    return {
        "task": "tsp",
        "size": 20,
        "objective": "min",
        "count": count,
        "valid_count": valid_count,
        "failure_count": count - valid_count,
        "mean": mean,
        "std": std,
        "min": mean - std,
        "max": mean + std,
        "failure_reasons": failure_reasons,
        "setting_seed": 20280810,
        "method": "EoH2000",
        "rep": 2,
        "code_file": "/tsp/program.py",
        "seed": 20260810,
        "dataset_count": 4,
        "instance_offset": offset,
        "instance_stop": stop,
    }


def test_merge_instance_shards_combines_population_statistics(tmp_path: Path) -> None:
    first = tmp_path / "part0.json"
    second = tmp_path / "part1.json"
    first.write_text(
        json.dumps([_partial_row(offset=0, stop=2, mean=1.5, std=0.5, valid_count=2)]),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps([_partial_row(offset=2, stop=4, mean=3.5, std=0.5, valid_count=2)]),
        encoding="utf-8",
    )

    merged = MODULE.merge_instance_shards([first, second])

    assert len(merged) == 1
    assert merged[0]["count"] == 4
    assert merged[0]["valid_count"] == 4
    assert merged[0]["mean"] == 2.5
    assert math.isclose(merged[0]["std"], math.sqrt(1.25))
    assert merged[0]["instance_shards"] == 2


def test_merge_instance_shards_rejects_a_coverage_gap(tmp_path: Path) -> None:
    first = tmp_path / "part0.json"
    second = tmp_path / "part1.json"
    first.write_text(
        json.dumps([_partial_row(offset=0, stop=1, mean=1.0, std=0.0, valid_count=1)]),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps([_partial_row(offset=2, stop=4, mean=3.5, std=0.5, valid_count=2)]),
        encoding="utf-8",
    )

    try:
        MODULE.merge_instance_shards([first, second])
    except ValueError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("instance shards must cover every original instance")
