#!/usr/bin/env python3
"""Audit and evaluate the frozen construct_tsp Stage-3 EoH run contract."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import statistics
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ENDPOINTS = tuple(f"http://127.0.0.1:{port}" for port in range(11013, 11021))
ORIGINAL_TRENDS = {
    1: (6.74067, 6.58896),
    2: (6.66389, 6.57664),
    3: (6.98264, 6.50759),
}
PAPER_RESULTS = {"N=20": 4.2107, "N=50": 6.5167}
PLAN = {
    "task": "construct_tsp",
    "method": "EoH+AgenticESOpt1000",
    "topology": {"endpoint_count": 8, "evaluation_workers": 4},
    "formal": {
        "repeats": [1, 2, 3],
        "outer_generations": 25,
        "population": 10,
        "k": 1,
        "candidate_count": 1000,
        "logical_updates": 50,
        "endpoint_updates": 400,
        "initialized_endpoints": 8,
        "operators": ["m1", "m2"],
    },
    "smoke": {"outer_generations": 1, "directions": 10, "logical_updates": 1, "endpoint_updates": 8},
    "parameters": {"seed": 2024, "sigma_start": 0.001, "sigma_end": 0.0, "sigma_schedule": "cosine", "alpha": 0.0005},
    "endpoints": [endpoint + "/completions" for endpoint in ENDPOINTS],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_set(updates: list[dict]) -> set[str]:
    return {
        str(update["endpoint"]).removesuffix("/completions").rstrip("/")
        for update in updates
        if isinstance(update, dict) and update.get("endpoint")
    }


def validate_update(record: dict, index: int, generation: int, operator: str) -> None:
    require(record.get("update_index") == index, f"ES update {index} index differs")
    require(record.get("generation") == generation, f"ES update {index} generation differs")
    require(record.get("operator") == operator, f"ES update {index} operator differs")
    seeds, rewards = record.get("seeds"), record.get("rewards")
    require(isinstance(seeds, list) and len(seeds) == 10, f"ES update {index} must have 10 directions")
    require(len(set(seeds)) == 10 and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds), f"ES update {index} seeds differ")
    require(isinstance(rewards, list) and len(rewards) == 10 and all(math.isfinite(float(value)) for value in rewards), f"ES update {index} rewards differ")
    require(record.get("update_applied") is True, f"ES update {index} was not applied")
    require(record.get("alpha") == 0.0005, f"ES update {index} alpha differs")
    require(record.get("sigma_start") == 0.001 and record.get("sigma_end") == 0.0, f"ES update {index} sigma endpoints differ")
    require(record.get("sigma_schedule") == "cosine", f"ES update {index} sigma schedule differs")
    require(record.get("engine_count") == 8, f"ES update {index} engine count differs")
    require(record.get("generation_concurrency") == 8, f"ES update {index} did not route across 8 engines")
    require(record.get("evaluation_concurrency") == 4, f"ES update {index} did not use 4 evaluator workers")
    updates = record.get("update")
    require(isinstance(updates, list) and len(updates) == 8, f"ES update {index} must synchronize 8 endpoints")
    require(all(isinstance(item, dict) and item.get("ok") is True and item.get("n") == 10 for item in updates), f"ES update {index} endpoint update failed")
    observed = endpoint_set(updates)
    require(observed == set(ENDPOINTS), f"ES update {index} endpoint set differs")


def initialized_count(log: Path) -> int:
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "- Model ES initialized:"
        if marker in line:
            value = ast.literal_eval(line.split(marker, 1)[1].strip())
            return len(value) if isinstance(value, list) else 0
    return 0


def csv_rows(path: Path, expected_columns: int) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) >= expected_columns, f"malformed CSV row in {path}: {line!r}")
        rows.append(fields)
    return rows


def validate_topology(gpu_inventory: Path, compute_inventory: Path, pid_dir: Path) -> dict[str, Any]:
    gpu_rows = csv_rows(gpu_inventory, 2)
    require(len(gpu_rows) == 8, "topology requires exactly 8 physical GPUs")
    expected_uuids: dict[int, str] = {}
    for expected_index, row in enumerate(gpu_rows):
        require(int(row[0]) == expected_index, "GPU indices must be exactly 0..7")
        expected_uuids[expected_index] = row[1]
    require(len(set(expected_uuids.values())) == 8, "physical GPU UUIDs must be unique")

    compute_by_pid = {int(row[0]): row[1] for row in csv_rows(compute_inventory, 3)}
    servers = []
    for gpu, port in enumerate(range(11013, 11021)):
        pid_file = pid_dir / f"server_gpu{gpu}_port{port}.pid"
        require(pid_file.is_file(), f"missing owned server PID file: {pid_file}")
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        require(pid in compute_by_pid, f"owned server PID {pid} is absent from nvidia-smi compute inventory")
        require(compute_by_pid[pid] == expected_uuids[gpu], f"server PID {pid} is on the wrong physical GPU")
        servers.append({"gpu_index": gpu, "gpu_uuid": expected_uuids[gpu], "port": port, "pid": pid})
    require(len({server["pid"] for server in servers}) == 8, "server PIDs must be unique")
    return {"status": "PASS", "endpoint_count": 8, "servers": servers}


def validate_runtime_audits(raw_run: Path, generations: int) -> dict[str, Any]:
    candidate_rows = read_json(raw_run / "results" / "history" / "operator_candidates.json")
    evaluator_rows = read_json(raw_run / "results" / "es" / "evaluator_processes.json")
    expected = [
        (generation, operator)
        for generation in range(generations)
        for operator in ("e1", "e2", "m1", "m2")
    ]
    candidate_by_key = {
        (row.get("generation"), row.get("operator")): row
        for row in candidate_rows
        if isinstance(row, dict) and row.get("operator") in {"e1", "e2", "m1", "m2"}
    }
    evaluator_by_key = {
        (row.get("generation"), row.get("operator")): row
        for row in evaluator_rows
        if isinstance(row, dict) and row.get("operator") in {"e1", "e2", "m1", "m2"}
    }
    audited_operators = {"e1", "e2", "m1", "m2"}
    require(len(candidate_by_key) == len([row for row in candidate_rows if isinstance(row, dict) and row.get("operator") in audited_operators]), "operator candidate audit contains duplicates")
    require(len(evaluator_by_key) == len([row for row in evaluator_rows if isinstance(row, dict) and row.get("operator") in audited_operators]), "CPU evaluator process audit contains duplicates")
    require(set(candidate_by_key) == set(expected), "operator candidate audit is incomplete or duplicated")
    require(set(evaluator_by_key) == set(expected), "CPU evaluator process audit is incomplete or duplicated")
    for key in expected:
        candidate = candidate_by_key[key]
        require(candidate.get("candidate_count") == 10, f"operator {key} did not consume 10 candidates")
        evaluator = evaluator_by_key[key]
        require(evaluator.get("configured_workers") == 4, f"operator {key} evaluator worker setting differs")
        require(evaluator.get("max_concurrent_processes") == 4, f"operator {key} did not exercise 4 evaluator processes")
        pids = evaluator.get("process_pids")
        require(isinstance(pids, list) and len(set(pids)) >= 4, f"operator {key} lacks 4 evaluator process PIDs")
    return {
        "outer_generations": generations,
        "operator_batches": len(expected),
        "candidate_count": sum(int(candidate_by_key[key]["candidate_count"]) for key in expected),
        "evaluation_workers": 4,
    }


def original_curve(repeat: int) -> list[float]:
    path = (
        ROOT
        / "ahd-test-time"
        / "results"
        / "EoH+AgenticESOpt1000"
        / "TSP_construct"
        / f"construct_tsp_k1_rep{repeat}_search.log"
    )
    require(path.is_file(), f"original search log is missing: {path}")
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Pop Objs:"):
            objectives = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", line)]
            require(objectives, f"original search log has empty population row: {path}")
            values.append(min(objectives))
    require(len(values) == 25, f"original search log must contain 25 population rows: {path}")
    return values


def validate_run(
    raw_run: Path,
    *,
    repeat: int,
    smoke: bool,
    runner_log: Path | None = None,
    final_code: Path | None = None,
) -> dict[str, Any]:
    raw_run = raw_run.resolve()
    history = read_json(raw_run / "results" / "es" / "history.json")
    expected_updates = 1 if smoke else 50
    require(isinstance(history, list) and len(history) == expected_updates, f"ES history must contain {expected_updates} logical updates")
    require(
        runner_log is not None and initialized_count(runner_log) == 8,
        "run did not initialize exactly 8 fresh Model ES clients",
    )
    runtime_audit = validate_runtime_audits(raw_run, 1 if smoke else 25)
    if smoke:
        validate_update(history[0], 0, 0, "m1")
        return {
            "status": "PASS",
            "kind": "directed-smoke",
            "initialized_endpoints": 8,
            "directions": 10,
            "logical_updates": 1,
            "endpoint_updates": 8,
            "evaluation_workers": 4,
            "runtime_audit": runtime_audit,
        }

    require(repeat in ORIGINAL_TRENDS, "formal repeat must be 1, 2, or 3")
    sigma_values = []
    for generation in range(25):
        for offset, operator in enumerate(("m1", "m2")):
            index = generation * 2 + offset
            validate_update(history[index], index, generation, operator)
            sigma_values.append(float(history[index]["sigma"]))
    require(math.isclose(sigma_values[0], 0.001, abs_tol=1e-12), "first sigma differs")
    require(math.isclose(sigma_values[-1], 0.0, abs_tol=1e-12), "last sigma differs")
    require(all(left >= right for left, right in zip(sigma_values, sigma_values[1:])), "sigma is not nonincreasing")

    best_values: list[float] = []
    final: dict[str, Any] | None = None
    for generation in range(1, 26):
        row = read_json(raw_run / "results" / "pops_best" / f"population_generation_{generation}.json")
        require(isinstance(row, dict) and math.isfinite(float(row.get("objective"))), f"generation {generation} best is invalid")
        best_values.append(float(row["objective"]))
        final = row
    require(final is not None and isinstance(final.get("code"), str) and final["code"].strip(), "final best code is missing")
    if final_code is not None:
        final_code.parent.mkdir(parents=True, exist_ok=True)
        temporary = final_code.with_suffix(final_code.suffix + ".tmp")
        temporary.write_text(final["code"].rstrip() + "\n", encoding="utf-8")
        temporary.replace(final_code)
    original_values = original_curve(repeat)
    original_first, original_last = ORIGINAL_TRENDS[repeat]
    require(math.isclose(original_values[0], original_first, abs_tol=1e-9), "original first-generation reference differs")
    require(math.isclose(original_values[-1], original_last, abs_tol=1e-9), "original final-generation reference differs")
    return {
        "status": "STRUCTURALLY_VALID",
        "kind": "formal",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repeat": repeat,
        "raw_run": str(raw_run),
        "candidate_count": runtime_audit["candidate_count"],
        "logical_updates": 50,
        "endpoint_updates": 400,
        "initialized_endpoints": 8,
        "evaluation_workers": 4,
        "runtime_audit": runtime_audit,
        "first_best": best_values[0],
        "last_best": best_values[-1],
        "observed_delta": best_values[-1] - best_values[0],
        "trend_direction": "IMPROVED" if best_values[-1] < best_values[0] else "NOT_IMPROVED",
        "original_log": {
            "first_best": original_first,
            "last_best": original_last,
            "delta": original_last - original_first,
            "best_by_generation": original_values,
        },
        "best_by_generation": best_values,
        "generation_comparison": [
            {
                "generation": index,
                "current_best": current,
                "repository_best": repository,
                "absolute_difference": current - repository,
                "relative_difference": (
                    (current - repository) / repository if repository != 0 else None
                ),
            }
            for index, (current, repository) in enumerate(
                zip(best_values, original_values),
                start=1,
            )
        ],
    }


def evaluate_final(code: Path) -> list[dict[str, Any]]:
    evaluator_path = ROOT / "ahd-test-time" / "results" / "eval_construct_results.py"
    spec = importlib.util.spec_from_file_location("stage3_construct_eval", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.MAX_INSTANCES = 0
    module = evaluator.load_module(code)
    rows = [
        {"setting": f"N={size}", **evaluator.eval_tsp(module, size)}
        for size in (20, 50, 100)
    ]
    validate_cpu_results(rows)
    return rows


def validate_cpu_results(rows: list[dict[str, Any]]) -> None:
    require([row.get("setting") for row in rows] == ["N=20", "N=50", "N=100"], "CPU final evaluator settings differ")
    for row in rows:
        count = row.get("count")
        valid_count = row.get("valid_count")
        require(isinstance(count, int) and count > 0, f"CPU final evaluator count differs at {row.get('setting')}")
        require(valid_count == count, f"CPU final evaluator found invalid instances at {row.get('setting')}")
        require(math.isfinite(float(row.get("mean"))), f"CPU final evaluator mean is invalid at {row.get('setting')}")


def compare_evaluations(current: list[dict[str, Any]], repository: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_cpu_results(current)
    validate_cpu_results(repository)
    comparisons = []
    for current_row, repository_row in zip(current, repository):
        setting = str(current_row["setting"])
        require(repository_row["setting"] == setting, "CPU comparison settings differ")
        current_mean = float(current_row["mean"])
        repository_mean = float(repository_row["mean"])
        paper_mean = PAPER_RESULTS.get(setting)
        comparisons.append(
            {
                "setting": setting,
                "current_mean": current_mean,
                "current_std": float(current_row["std"]),
                "repository_mean": repository_mean,
                "repository_std": float(repository_row["std"]),
                "repository_absolute_difference": current_mean - repository_mean,
                "repository_relative_difference": (
                    (current_mean - repository_mean) / repository_mean
                    if repository_mean != 0
                    else None
                ),
                "paper_mean": paper_mean,
                "paper_absolute_difference": current_mean - paper_mean if paper_mean is not None else None,
                "paper_relative_difference": (
                    (current_mean - paper_mean) / paper_mean
                    if paper_mean is not None and paper_mean != 0
                    else None
                ),
            }
        )
    return comparisons


def aggregate_repeat_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    require([report.get("repeat") for report in reports] == [1, 2, 3], "aggregate requires repeats 1, 2, and 3")

    def aggregate(values: list[float]) -> dict[str, Any]:
        return {"values": values, "mean": statistics.mean(values), "std": statistics.pstdev(values)}

    current_train = [float(report["trend"]["last_best"]) for report in reports]
    repository_train = [float(report["trend"]["original_log"]["last_best"]) for report in reports]
    current_train_stats = aggregate(current_train)
    repository_train_stats = aggregate(repository_train)
    train_difference = current_train_stats["mean"] - repository_train_stats["mean"]
    test_rows = []
    settings = [row["setting"] for row in reports[0]["cpu_final_eval"]["comparison"]]
    for setting in settings:
        comparisons = []
        for report in reports:
            matches = [
                row
                for row in report["cpu_final_eval"]["comparison"]
                if row.get("setting") == setting
            ]
            require(len(matches) == 1, f"repeat CPU comparison is missing {setting}")
            comparisons.append(matches[0])
        current_stats = aggregate([float(row["current_mean"]) for row in comparisons])
        repository_stats = aggregate([float(row["repository_mean"]) for row in comparisons])
        difference = current_stats["mean"] - repository_stats["mean"]
        paper_mean = comparisons[0].get("paper_mean")
        test_rows.append(
            {
                "setting": setting,
                "current": current_stats,
                "repository": repository_stats,
                "repository_absolute_difference": difference,
                "repository_relative_difference": (
                    difference / repository_stats["mean"]
                    if repository_stats["mean"] != 0
                    else None
                ),
                "paper_mean": paper_mean,
                "paper_absolute_difference": (
                    current_stats["mean"] - float(paper_mean)
                    if paper_mean is not None
                    else None
                ),
                "paper_relative_difference": (
                    (current_stats["mean"] - float(paper_mean)) / float(paper_mean)
                    if paper_mean not in (None, 0)
                    else None
                ),
            }
        )
    return {
        "status": "READY_FOR_MANUAL_ACCEPTANCE",
        "manual_acceptance_required": True,
        "train_final": {
            "current_values": current_train_stats["values"],
            "current_mean": current_train_stats["mean"],
            "current_std": current_train_stats["std"],
            "repository_values": repository_train_stats["values"],
            "repository_mean": repository_train_stats["mean"],
            "repository_std": repository_train_stats["std"],
            "repository_absolute_difference": train_difference,
            "repository_relative_difference": (
                train_difference / repository_train_stats["mean"]
                if repository_train_stats["mean"] != 0
                else None
            ),
        },
        "test": test_rows,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    topology = commands.add_parser("validate-topology")
    topology.add_argument("--gpu-inventory", type=Path, required=True)
    topology.add_argument("--compute-inventory", type=Path, required=True)
    topology.add_argument("--pid-dir", type=Path, required=True)
    topology.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--raw-run", type=Path, required=True)
    validate.add_argument("--repeat", type=int, required=True)
    validate.add_argument("--runner-log", type=Path)
    validate.add_argument("--smoke", action="store_true")
    validate.add_argument("--final-code", type=Path)
    validate.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--code", type=Path, required=True)
    evaluate.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--run-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(PLAN, indent=2))
        return 0
    if args.command == "validate-topology":
        report = validate_topology(args.gpu_inventory, args.compute_inventory, args.pid_dir)
        write_json(args.output.resolve(), report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "summarize":
        reports = [
            {
                "repeat": repeat,
                "trend": read_json(args.run_root / f"rep{repeat}" / "trend.json"),
                "cpu_final_eval": read_json(args.run_root / f"rep{repeat}" / "cpu_final_eval.json"),
            }
            for repeat in (1, 2, 3)
        ]
        report = aggregate_repeat_reports(reports)
        write_json(args.output.resolve(), report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "validate":
        report = validate_run(args.raw_run, repeat=args.repeat, smoke=args.smoke, runner_log=args.runner_log, final_code=args.final_code)
    else:
        repository_code = (
            ROOT
            / "ahd-test-time"
            / "results"
            / "EoH+AgenticESOpt1000"
            / "TSP_construct"
            / f"construct_tsp_k1_rep{args.repeat}_final_best_code.py"
        )
        require(repository_code.is_file(), f"repository comparison program is missing: {repository_code}")
        current_results = evaluate_final(args.code.resolve())
        repository_results = evaluate_final(repository_code)
        report = {
            "task": "construct_tsp",
            "split": "test",
            "repeat": args.repeat,
            "code": str(args.code.resolve()),
            "repository_code": str(repository_code),
            "paper_reference": PAPER_RESULTS,
            "current_results": current_results,
            "repository_results": repository_results,
            "comparison": compare_evaluations(current_results, repository_results),
        }
    write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
