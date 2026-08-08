#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "ahd-test-time" / "results"
PROGRESS = (
    ROOT
    / "runs"
    / "ahd_sample_extend_all6_3rep_to2000_gpu4_7_20260718_145101"
    / "progress.jsonl"
)

PREFIX_ROOTS = {
    1000: RESULTS_ROOT / "AHD-Sample1000",
    2000: RESULTS_ROOT / "AHD-Sample2000",
}

TASK_DIRS = {
    "construct_tsp": "TSP_construct",
    "construct_kp": "KP_construct",
    "construct_asp": "ASP_construct",
    "aco_tsp": "TSP_ACO",
    "aco_cvrp": "CVRP_ACO",
    "aco_bpp": "BPP_ACO",
}

SETTINGS: dict[str, list[tuple[str, tuple[Any, ...]]]] = {
    "construct_tsp": [("N=20", (20,)), ("N=50", (50,))],
    "construct_kp": [("N=50,W=12.5", (50, 12.5)), ("N=100,W=25", (100, 25.0))],
    "construct_asp": [("N=15,W=10", (15, 10)), ("N=21,W=15", (21, 15))],
    "aco_tsp": [("N=50", (50,)), ("N=100", (100,))],
    "aco_cvrp": [("N=50,C=50", (50,)), ("N=100,C=50", (100,))],
    "aco_bpp": [("N=500,C=150", (500,)), ("N=1000,C=150", (1000,))],
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def completed_runs() -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for line in PROGRESS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[(str(row["task"]), int(row["rep"]))] = row
    expected = {(task, rep) for task in TASK_DIRS for rep in (1, 2, 3)}
    if set(latest) != expected:
        raise RuntimeError(f"expected {len(expected)} task/runs, found {len(latest)}")
    incomplete = [key for key, row in latest.items() if row.get("status") != "completed"]
    if incomplete:
        raise RuntimeError(f"incomplete sample runs: {incomplete}")
    return [latest[key] for key in sorted(latest)]


def source_json(row: dict[str, Any], prefix: int) -> Path:
    if prefix == 1000:
        run_root = Path(row["source_1000"])
        generation = 50
    elif prefix == 2000:
        run_root = Path(row["destination_2000"])
        generation = 100
    else:
        raise ValueError(prefix)
    return run_root / "results" / "pops_best" / f"population_generation_{generation}.json"


def best_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"empty population in {path}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise TypeError(f"unexpected payload in {path}: {type(payload).__name__}")
    code = str(payload.get("code", "")).strip()
    if not code:
        raise ValueError(f"empty best code in {path}")
    return payload


def archive_prefix(prefix: int, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = PREFIX_ROOTS[prefix]
    out.mkdir(parents=True, exist_ok=True)
    for dirname in TASK_DIRS.values():
        (out / dirname).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for row in runs:
        task = str(row["task"])
        rep = int(row["rep"])
        source = source_json(row, prefix)
        if not source.exists():
            raise FileNotFoundError(source)
        payload = best_payload(source)
        code = str(payload["code"]).strip()
        stabilization = ""
        # Some sampled TSP heuristics exponentiate inverse distances directly.
        # Large but finite float64 values can make a float32 row sum overflow in
        # torch.distributions.Categorical. Capping only the extreme log values at
        # 60 keeps ordinary weights unchanged and bounds a 100-node float32 row
        # sum far below overflow, without triggering the evaluator's 1e-9 floor.
        unstable_exp = "return np.exp(edge_log_prob)"
        if task == "aco_tsp" and unstable_exp in code:
            stable_exp = (
                "stable_log_prob = np.minimum(edge_log_prob, 60.0)\n"
                "    return np.exp(stable_log_prob)"
            )
            code = code.replace(unstable_exp, stable_exp, 1)
            stabilization = "exp_log_upper_clip_60"
        code_path = out / TASK_DIRS[task] / f"{task}_rep{rep}_final_best_code.py"
        header = (
            f"# source: {source}\n"
            f"# method: sample, prefix={prefix}, batch_size=20\n"
            f"# task: {task}, rep: {rep}\n"
            f"# train_objective: {payload.get('objective')}\n\n"
        )
        if stabilization:
            header += f"# numerical_stabilization: {stabilization}\n\n"
        code_path.write_text(header + code + "\n", encoding="utf-8")
        manifest.append(
            {
                "prefix": prefix,
                "task": task,
                "rep": rep,
                "train_objective": payload.get("objective"),
                "code_file": str(code_path),
                "source_json": str(source),
                "numerical_stabilization": stabilization,
            }
        )

    manifest.sort(key=lambda item: (str(item["task"]), int(item["rep"])))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_csv(out / "manifest.csv", manifest)
    write_train_report(out, prefix, manifest)
    return manifest


def write_train_report(out: Path, prefix: int, manifest: list[dict[str, Any]]) -> None:
    lines = [
        f"# AHD Sample {prefix}: training best",
        "",
        "Three independent runs; these are training objectives at the requested sampling prefix.",
        "",
        "| Task | Rep1 | Rep2 | Rep3 |",
        "|---|---:|---:|---:|",
    ]
    for task in TASK_DIRS:
        values = {
            int(row["rep"]): row["train_objective"]
            for row in manifest
            if row["task"] == task
        }
        lines.append(
            f"| {task} | {values.get(1)} | {values.get(2)} | {values.get(3)} |"
        )
    (out / "TRAIN_BEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def jobs(manifests: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for prefix, manifest in manifests.items():
        for record in manifest:
            for setting, params in SETTINGS[str(record["task"])]:
                output.append(
                    {
                        **record,
                        "prefix": prefix,
                        "setting": setting,
                        "params": list(params),
                    }
                )
    return output


def evaluate_job(job: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    started = time.time()
    task = str(job["task"])
    rep = int(job["rep"])
    params = job["params"]
    try:
        if task.startswith("construct_"):
            evaluator = load_module(
                RESULTS_ROOT / "eval_construct_results.py",
                f"sample_construct_{os.getpid()}",
            )
            evaluator.MAX_INSTANCES = 0
            candidate = evaluator.load_module(Path(job["code_file"]))
            if task == "construct_tsp":
                result = evaluator.eval_tsp(candidate, int(params[0]))
            elif task == "construct_kp":
                result = evaluator.eval_kp(candidate, int(params[0]), float(params[1]))
            else:
                result = evaluator.eval_asp(candidate, int(params[0]), int(params[1]), 200000, 1234)
        else:
            evaluator = load_module(
                RESULTS_ROOT / "eval_aco_results.py",
                f"sample_aco_{os.getpid()}",
            )
            evaluator.np.random.seed(20260000 + rep * 1000 + int(params[0]))
            evaluator.torch.manual_seed(20260000 + rep * 1000 + int(params[0]))
            evaluator.torch.set_num_threads(1)
            args = SimpleNamespace(
                split="test",
                tsp_sizes=str(params[0]) if task == "aco_tsp" else "",
                cvrp_sizes=str(params[0]) if task == "aco_cvrp" else "",
                bpp_sizes=str(params[0]) if task == "aco_bpp" else "",
                tsp_iterations=100,
                tsp_ants=30,
                cvrp_iterations=100,
                cvrp_ants=30,
                cvrp_capacity=50,
                bpp_mode="sample",
                bpp_iterations=15,
                bpp_ants=20,
                bpp_sample_count=200,
                bpp_capacity=150,
                max_instances=0,
                keep_going=True,
            )
            results = evaluator.eval_code_file(task.removeprefix("aco_"), Path(job["code_file"]), args)
            if len(results) != 1:
                raise RuntimeError(f"expected one ACO result, got {len(results)}")
            result = results[0]

        raw_metric = result.get("mean")
        if raw_metric is None:
            raw_metric = result.get("asp_set_size")
        count = result.get("count")
        valid_count = result.get("valid_count")
        failure_count = result.get("failure_count")
        complete = (
            isinstance(raw_metric, (int, float))
            and int(failure_count or 0) == 0
            and (count is None or valid_count is None or int(count) == int(valid_count))
        )
        return {
            **job,
            "metric": float(raw_metric) if complete else None,
            "partial_metric": raw_metric,
            "objective": result.get("objective"),
            "count": count,
            "valid_count": valid_count,
            "failure_count": failure_count,
            "std": result.get("std"),
            "min": result.get("min"),
            "max": result.get("max"),
            "asp_set_size": result.get("asp_set_size"),
            "pre_admissible_size": result.get("pre_admissible_size"),
            "seconds": result.get("seconds"),
            "seconds_wall": round(time.time() - started, 3),
            "error": result.get("error", "") if complete else result.get("error", "incomplete_test_instances"),
        }
    except BaseException as exc:
        return {
            **job,
            "metric": None,
            "seconds_wall": round(time.time() - started, 3),
            "error": repr(exc),
        }


def result_key(row: dict[str, Any]) -> tuple[int, str, int, str]:
    return int(row["prefix"]), str(row["task"]), int(row["rep"]), str(row["setting"])


def save_prefix_rows(prefix: int, rows: list[dict[str, Any]]) -> None:
    out = PREFIX_ROOTS[prefix]
    selected = sorted(
        [row for row in rows if int(row["prefix"]) == prefix],
        key=result_key,
    )
    (out / "test_detail.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    write_csv(out / "test_detail.csv", selected)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["setting"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (task, setting), items in sorted(groups.items()):
        items.sort(key=lambda item: int(item["rep"]))
        values = [float(item["metric"]) for item in items if isinstance(item.get("metric"), (int, float))]
        summary.append(
            {
                "task": task,
                "setting": setting,
                "objective": items[0].get("objective"),
                "rep1": next((item.get("metric") for item in items if int(item["rep"]) == 1), None),
                "rep2": next((item.get("metric") for item in items if int(item["rep"]) == 2), None),
                "rep3": next((item.get("metric") for item in items if int(item["rep"]) == 3), None),
                "mean_over_reps": statistics.mean(values) if values else None,
                "std_over_reps": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
                "valid_reps": len(values),
                "total_failures": sum(int(item.get("failure_count") or 0) for item in items),
                "errors": "; ".join(str(item["error"]) for item in items if item.get("error")),
            }
        )
    return summary


def write_test_report(prefix: int, summary: list[dict[str, Any]]) -> None:
    out = PREFIX_ROOTS[prefix]
    lines = [
        f"# AHD Sample {prefix}: test results",
        "",
        "Full test split. ACO TSP/CVRP: 100 iterations and 30 ants; BPP: sample mode with 200 samples.",
        "A metric is reported only when all test instances for that run succeed.",
        "",
        "| Task | Setting | Rep1 | Rep2 | Rep3 | Mean | Std | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.8g}"

    for row in summary:
        lines.append(
            f"| {row['task']} | {row['setting']} | {fmt(row['rep1'])} | {fmt(row['rep2'])} | "
            f"{fmt(row['rep3'])} | {fmt(row['mean_over_reps'])} | {fmt(row['std_over_reps'])} | "
            f"{row['total_failures']} |"
        )
    lines.extend(
        [
            "",
            "Constructive KP/ASP: larger is better. Constructive TSP and all ACO tasks: smaller is better.",
            "",
        ]
    )
    (out / "TEST_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def finalize(all_rows: list[dict[str, Any]]) -> None:
    for prefix, out in PREFIX_ROOTS.items():
        rows = [row for row in all_rows if int(row["prefix"]) == prefix]
        save_prefix_rows(prefix, rows)
        summary = summarize(rows)
        (out / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_csv(out / "test_summary.csv", summary)
        write_test_report(prefix, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    runs = completed_runs()
    manifests = {prefix: archive_prefix(prefix, runs) for prefix in PREFIX_ROOTS}
    print("archived 18 codes for each of Sample1000 and Sample2000", flush=True)
    if args.archive_only:
        return

    existing: list[dict[str, Any]] = []
    for prefix, out in PREFIX_ROOTS.items():
        detail = out / "test_detail.json"
        if detail.exists():
            existing.extend(json.loads(detail.read_text(encoding="utf-8")))
    completed = {result_key(row) for row in existing if not row.get("error")}
    pending = [job for job in jobs(manifests) if result_key(job) not in completed]
    rows_by_key = {result_key(row): row for row in existing}
    print(f"evaluation jobs: completed={len(completed)} pending={len(pending)}", flush=True)

    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=context) as executor:
            future_jobs = {executor.submit(evaluate_job, job): job for job in pending}
            for index, future in enumerate(as_completed(future_jobs), start=1):
                row = future.result()
                rows_by_key[result_key(row)] = row
                save_prefix_rows(int(row["prefix"]), list(rows_by_key.values()))
                print(
                    f"[{index}/{len(pending)}] sample{row['prefix']} {row['task']} rep{row['rep']} "
                    f"{row['setting']} metric={row.get('metric')} error={row.get('error', '')} "
                    f"wall={row.get('seconds_wall')}s",
                    flush=True,
                )

    rows = list(rows_by_key.values())
    finalize(rows)
    errors = [row for row in rows if row.get("error")]
    print(f"all done: detail_rows={len(rows)} errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
