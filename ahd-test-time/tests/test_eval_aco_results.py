from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "ahd-test-time/results/eval_aco_results.py"
SPEC = importlib.util.spec_from_file_location("eval_aco_results_for_test", EVALUATOR)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_aco_rows_record_instance_failure_reasons(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "load_module", lambda _: object())
    monkeypatch.setattr(MODULE, "get_heuristics", lambda _: lambda *_: None)
    monkeypatch.setattr(MODULE.np, "load", lambda *_, **__: MODULE.np.zeros((2, 3, 2)))

    def fail(*_args, **_kwargs):
        raise ValueError("broken heuristic")

    monkeypatch.setattr(MODULE, "solve_tsp", fail)
    args = SimpleNamespace(
        tsp_sizes="20",
        split="test",
        max_instances=0,
        tsp_iterations=100,
        tsp_ants=30,
        keep_going=True,
        seed=20260810,
    )

    rows = MODULE.eval_code_file("tsp", Path("program.py"), args)

    assert rows[0]["valid_count"] == 0
    assert rows[0]["failure_count"] == 2
    assert rows[0]["failure_reasons"] == {"ValueError: broken heuristic": 2}


def test_collect_values_uses_common_per_instance_random_streams() -> None:
    instances = [object(), object(), object()]

    first, _ = MODULE.collect_values(
        instances,
        lambda _: MODULE.np.random.random(),
        keep_going=False,
        seed=20260810,
    )
    MODULE.np.random.seed(7)
    _ = MODULE.np.random.random(100)
    second, _ = MODULE.collect_values(
        instances,
        lambda _: MODULE.np.random.random(),
        keep_going=False,
        seed=20260810,
    )

    assert first == second
    assert len(set(first)) == len(instances)


def test_collect_values_preserves_random_stream_when_instances_are_sliced() -> None:
    instances = [object(), object(), object(), object()]
    full, _ = MODULE.collect_values(
        instances,
        lambda _: MODULE.np.random.random(),
        keep_going=False,
        seed=20260810,
    )

    partial, _ = MODULE.collect_values(
        instances[2:],
        lambda _: MODULE.np.random.random(),
        keep_going=False,
        seed=20260810,
        instance_offset=2,
    )

    assert partial == full[2:]


def test_aco_cli_seed_makes_repeated_evaluations_identical(tmp_path: Path) -> None:
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--tasks",
                "bpp",
                "--bpp-sizes",
                "500",
                "--bpp-sample-count",
                "2",
                "--max-instances",
                "1",
                "--seed",
                "20260810",
                "--keep-going",
                "--output",
                str(output),
                "--csv-output",
                str(output.with_suffix(".csv")),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    first = json.loads(outputs[0].read_text(encoding="utf-8"))
    second = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert first == second
    assert first
    assert {row["seed"] for row in first} == {20260810}


def test_aco_cli_shards_partition_programs_without_overlap(tmp_path: Path) -> None:
    shard_rows = []
    for shard_index in range(2):
        output = tmp_path / f"shard_{shard_index}.json"
        subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--tasks",
                "bpp",
                "--bpp-sizes",
                "500",
                "--bpp-sample-count",
                "1",
                "--max-instances",
                "1",
                "--seed",
                str(20260810 + shard_index),
                "--keep-going",
                "--shard-count",
                "2",
                "--shard-index",
                str(shard_index),
                "--output",
                str(output),
                "--csv-output",
                str(output.with_suffix(".csv")),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        shard_rows.append(json.loads(output.read_text(encoding="utf-8")))

    code_files = [{row["code_file"] for row in rows} for rows in shard_rows]
    assert len(code_files[0]) == 12
    assert len(code_files[1]) == 12
    assert code_files[0].isdisjoint(code_files[1])
