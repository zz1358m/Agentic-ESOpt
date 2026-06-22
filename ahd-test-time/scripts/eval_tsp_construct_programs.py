#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import concurrent.futures
import importlib.util
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROGRAM_DIR = ROOT / "ahd-test-time/results/EoH/TSP_construct"
DEFAULT_DATA_DIR = ROOT / "data/ahd/datasets/tsp_constructive"
DEFAULT_OUTPUT = ROOT / "ahd-test-time/results/EoH/TSP_construct/eval_tsp_construct_test_sizes.json"
_WORKER_MODULE = None


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "select_next_node"):
        raise AttributeError(f"{path} does not define select_next_node")
    return module


def init_worker(program_path: str) -> None:
    global _WORKER_MODULE
    _WORKER_MODULE = load_module(Path(program_path))


def eval_worker(instance: np.ndarray) -> float | None:
    if _WORKER_MODULE is None:
        raise RuntimeError("Worker module not initialized.")
    return eval_instance(_WORKER_MODULE, np.asarray(instance))


def tour_cost(instance: np.ndarray, route: np.ndarray) -> float:
    cost = 0.0
    for i in range(len(route) - 1):
        cost += float(np.linalg.norm(instance[int(route[i])] - instance[int(route[i + 1])]))
    cost += float(np.linalg.norm(instance[int(route[-1])] - instance[int(route[0])]))
    return cost


def neighborhood_matrix(instance: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(instance[:, np.newaxis] - instance, axis=2)
    return np.argsort(distances, axis=1), distances


def eval_instance(module: Any, instance: np.ndarray) -> float | None:
    problem_size = int(instance.shape[0])
    neighbor_size = min(50, problem_size)
    neighbors, distances = neighborhood_matrix(instance)
    route = np.zeros(problem_size, dtype=np.int64)
    current_node = 0
    destination_node = 0

    for i in range(1, problem_size - 1):
        near_nodes = neighbors[current_node][1:]
        unvisited_near_nodes = near_nodes[~np.isin(near_nodes, route[:i])]
        unvisited_near_nodes = unvisited_near_nodes[: min(neighbor_size, unvisited_near_nodes.size)]
        if unvisited_near_nodes.size == 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            next_node = module.select_next_node(
                current_node,
                destination_node,
                unvisited_near_nodes,
                distances,
            )
        try:
            next_node = int(next_node)
        except Exception:
            return None
        if next_node in route[:i] or next_node < 0 or next_node >= problem_size:
            return None
        current_node = next_node
        route[i] = current_node

    remaining = np.arange(problem_size)[~np.isin(np.arange(problem_size), route[: problem_size - 1])]
    if remaining.size != 1:
        return None
    route[problem_size - 1] = int(remaining[0])
    value = tour_cost(instance, route)
    if not math.isfinite(value):
        return None
    return value


def eval_dataset(module: Any, dataset: np.ndarray, *, workers: int = 1, program_path: Path | None = None) -> dict[str, Any]:
    started = time.time()
    if workers > 1:
        if program_path is None:
            raise ValueError("program_path is required when workers > 1")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(str(program_path),),
        ) as pool:
            evaluated = list(pool.map(eval_worker, list(dataset), chunksize=4))
    else:
        evaluated = [eval_instance(module, np.asarray(instance)) for instance in dataset]
    values = [value for value in evaluated if value is not None]
    failures = len(evaluated) - len(values)
    return {
        "count": int(len(dataset)),
        "valid_count": int(len(values)),
        "failure_count": int(failures),
        "avg_tour_length": float(np.mean(values)) if values else None,
        "std_tour_length": float(np.std(values)) if values else None,
        "min_tour_length": float(np.min(values)) if values else None,
        "max_tour_length": float(np.max(values)) if values else None,
        "seconds": round(time.time() - started, 3),
    }


def write_outputs(results: dict[str, Any], rows: list[dict[str, Any]], output: Path, csv_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "program",
                "size",
                "count",
                "valid_count",
                "failure_count",
                "avg_tour_length",
                "std_tour_length",
                "min_tour_length",
                "max_tour_length",
                "seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-dir", default=str(DEFAULT_PROGRAM_DIR))
    parser.add_argument("--program-glob", default="*.py")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--sizes", default="20,50,100,200")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--csv-output", default="")
    args = parser.parse_args()

    program_dir = Path(args.program_dir)
    data_dir = Path(args.data_dir)
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    programs = sorted(program_dir.glob(args.program_glob))
    if not programs:
        raise FileNotFoundError(f"No Python programs found in {program_dir}")

    datasets = {
        size: np.load(data_dir / f"test{size}_dataset.npy", allow_pickle=False)
        for size in sizes
    }
    results = {
        "program_dir": str(program_dir),
        "data_dir": str(data_dir),
        "sizes": sizes,
        "programs": [],
    }
    rows = []
    output = Path(args.output)
    csv_output = Path(args.csv_output) if args.csv_output else output.with_suffix(".csv")
    for program in programs:
        module = load_module(program)
        program_result = {"program": program.name, "results": {}}
        results["programs"].append(program_result)
        for size, dataset in datasets.items():
            rec = eval_dataset(
                module,
                dataset,
                workers=max(1, int(args.workers)),
                program_path=program,
            )
            program_result["results"][str(size)] = rec
            rows.append({"program": program.name, "size": size, **rec})
            print(
                f"{program.name} test{size}: avg={rec['avg_tour_length']} "
                f"valid={rec['valid_count']}/{rec['count']} failures={rec['failure_count']} "
                f"seconds={rec['seconds']}",
                flush=True,
            )
            write_outputs(results, rows, output, csv_output)
    print(f"wrote {output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
