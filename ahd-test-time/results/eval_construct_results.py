#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "ahd" / "datasets"


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_programs() -> dict[str, list[Path]]:
    return {
        "tsp": sorted(RESULTS_ROOT.glob("*/TSP_construct/*final_best_code.py")),
        "kp": sorted(RESULTS_ROOT.glob("*/KP_construct/*final_best_code.py")),
        "asp": sorted(RESULTS_ROOT.glob("*/ASP_construct/*final_best_code.py")),
    }


def tour_cost(instance: np.ndarray, route: np.ndarray) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        total += float(np.linalg.norm(instance[int(route[i])] - instance[int(route[i + 1])]))
    total += float(np.linalg.norm(instance[int(route[-1])] - instance[int(route[0])]))
    return total


def eval_tsp_instance(module: Any, instance: np.ndarray) -> float | None:
    if not hasattr(module, "select_next_node"):
        return None
    n = int(instance.shape[0])
    neighbor_size = min(50, n)
    distances = np.linalg.norm(instance[:, None] - instance[None, :], axis=2)
    neighbors = np.argsort(distances, axis=1)
    route = np.zeros(n, dtype=np.int64)
    current_node = 0
    destination_node = 0

    for i in range(1, n - 1):
        near_nodes = neighbors[current_node][1:]
        candidates = near_nodes[~np.isin(near_nodes, route[:i])]
        candidates = candidates[: min(neighbor_size, candidates.size)]
        if candidates.size == 0:
            return None
        try:
            next_node = int(module.select_next_node(current_node, destination_node, candidates, distances))
        except Exception:
            return None
        if next_node in route[:i] or next_node < 0 or next_node >= n:
            return None
        route[i] = next_node
        current_node = next_node

    remaining = np.arange(n)[~np.isin(np.arange(n), route[: n - 1])]
    if remaining.size != 1:
        return None
    route[n - 1] = int(remaining[0])
    value = tour_cost(instance, route)
    return value if math.isfinite(value) else None


def eval_tsp(module: Any, n: int) -> dict[str, Any]:
    data = np.load(DATA_ROOT / "tsp_constructive" / f"test{n}_dataset.npy", allow_pickle=False)
    if MAX_INSTANCES > 0:
        data = data[:MAX_INSTANCES]
    values = [eval_tsp_instance(module, np.asarray(instance)) for instance in data]
    valid = [v for v in values if v is not None]
    return summarize(valid, len(values), objective="min")


def eval_kp_instance(module: Any, instance: np.ndarray, capacity: float) -> float | None:
    if not hasattr(module, "select_next_item"):
        return None
    weights = np.asarray(instance[:, 0], dtype=float)
    values = np.asarray(instance[:, 1], dtype=float)
    remaining = float(capacity)
    remaining_indices = list(range(len(values)))
    total_value = 0.0

    for _ in range(len(values)):
        if not remaining_indices:
            break
        call_weights = weights[remaining_indices].copy()
        call_values = values[remaining_indices].copy()
        if not np.any(call_weights <= remaining):
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                item = module.select_next_item(remaining, call_weights, call_values)
        except Exception:
            return None
        if item is None:
            break
        try:
            item = int(item)
        except Exception:
            return None
        if item < 0 or item >= len(remaining_indices):
            return None
        global_item = remaining_indices[item]
        if not math.isfinite(float(weights[global_item])):
            return None
        if weights[global_item] > remaining:
            break
        remaining -= float(weights[global_item])
        total_value += float(values[global_item])
        remaining_indices.pop(item)
    return total_value if math.isfinite(total_value) else None


def eval_kp(module: Any, n: int, capacity: float) -> dict[str, Any]:
    data = np.load(DATA_ROOT / "kp_constructive" / f"test{n}_dataset.npy", allow_pickle=False)
    if MAX_INSTANCES > 0:
        data = data[:MAX_INSTANCES]
    values = [eval_kp_instance(module, np.asarray(instance), capacity) for instance in data]
    valid = [v for v in values if v is not None]
    return summarize(valid, len(values), objective="max")


TRIPLES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1), (2, 2, 2)]
INT_TO_WEIGHT = [0, 1, 1, 2, 2, 3, 3]
BAD_TRIPLES = set(
    [
        (0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5),
        (0, 6, 6), (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3), (1, 2, 4),
        (1, 3, 3), (1, 4, 4), (1, 5, 5), (1, 6, 6), (2, 2, 2), (2, 3, 3),
        (2, 4, 4), (2, 5, 5), (2, 6, 6), (3, 3, 3), (3, 3, 4), (3, 4, 4),
        (3, 4, 5), (3, 4, 6), (3, 5, 5), (3, 6, 6), (4, 4, 4), (4, 5, 5),
        (4, 6, 6), (5, 5, 5), (5, 5, 6), (5, 6, 6), (6, 6, 6),
    ]
)


def expand_admissible_set(pre_admissible_set: np.ndarray) -> list[tuple[int, ...]]:
    num_groups = len(pre_admissible_set[0])
    admissible_set = []
    for row in pre_admissible_set:
        rotations = [[] for _ in range(num_groups)]
        for i in range(num_groups):
            x, y, z = TRIPLES[int(row[i])]
            rotations[i].append((x, y, z))
            if not x == y == z:
                rotations[i].append((z, x, y))
                rotations[i].append((y, z, x))
        product = list(itertools.product(*rotations))
        admissible_set.extend(sum(xs, ()) for xs in product)
    return admissible_set


def get_surviving_children(extant_elements: np.ndarray, new_element: np.ndarray, valid_children: list[np.ndarray]) -> list[int]:
    valid_indices = []
    for index, child in enumerate(valid_children):
        if all(INT_TO_WEIGHT[int(x)] <= INT_TO_WEIGHT[int(y)] for x, y in zip(new_element, child)):
            continue
        if all(INT_TO_WEIGHT[int(x)] >= INT_TO_WEIGHT[int(y)] for x, y in zip(new_element, child)):
            continue
        is_invalid = False
        for extant_element in extant_elements:
            if all(tuple(sorted((int(x), int(y), int(z)))) in BAD_TRIPLES for x, y, z in zip(extant_element, new_element, child)):
                is_invalid = True
                break
        if is_invalid:
            continue
        valid_indices.append(index)
    return valid_indices


def solve_asp(module: Any, n: int, w: int) -> tuple[np.ndarray, np.ndarray, int]:
    num_groups = n // 3
    if 3 * num_groups != n:
        raise ValueError(f"ASP evaluator expects n divisible by 3, got n={n}")

    valid_children = []
    for child in itertools.product(range(7), repeat=num_groups):
        weight = sum(INT_TO_WEIGHT[x] for x in child)
        if weight == w:
            valid_children.append(np.array(child, dtype=np.int32))
    initial_child_count = len(valid_children)

    valid_scores = np.array(
        [module.priority(sum([TRIPLES[int(x)] for x in xs], ()), n, w) for xs in valid_children],
        dtype=float,
    )

    pre_admissible_set = np.empty((0, num_groups), dtype=np.int32)
    while valid_children:
        max_index = int(np.argmax(valid_scores))
        max_child = valid_children[max_index]
        surviving_indices = get_surviving_children(pre_admissible_set, max_child, valid_children)
        valid_children = [valid_children[i] for i in surviving_indices]
        valid_scores = valid_scores[surviving_indices]
        pre_admissible_set = np.concatenate([pre_admissible_set, max_child[None]], axis=0)

    return pre_admissible_set, np.array(expand_admissible_set(pre_admissible_set)), initial_child_count


def eval_asp(module: Any, n: int, w: int, max_candidates: int, seed: int) -> dict[str, Any]:
    del max_candidates, seed
    if not hasattr(module, "priority"):
        return summarize([], 0, objective="max", extra={"candidate_mode": "missing_priority"})
    started = time.time()
    try:
        pre_admissible_set, admissible_set, initial_child_count = solve_asp(module, n, w)
        obj = int(len(admissible_set))
        pre_size = int(len(pre_admissible_set))
        failure = 0
    except Exception as exc:
        obj = None
        pre_size = 0
        initial_child_count = 0
        failure = 1
        exc_msg = repr(exc)
    result = {
        "objective": "max",
        "count": 1,
        "valid_count": 1 - failure,
        "failure_count": failure,
        "asp_set_size": obj,
        "pre_admissible_size": pre_size,
        "initial_child_count": initial_child_count,
        "candidate_mode": "exact_reevo",
        "seconds": round(time.time() - started, 3),
    }
    if failure:
        result["error"] = exc_msg
    return result


def summarize(values: list[float], count: int, objective: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "objective": objective,
        "count": count,
        "valid_count": len(values),
        "failure_count": count - len(values),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
        "min": float(np.min(values)) if values else None,
        "max": float(np.max(values)) if values else None,
    }
    if extra:
        result.update(extra)
    return result


def row_for(method: str, task: str, program: Path, setting: str, rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "task": task,
        "program": program.name,
        "setting": setting,
        "objective": rec.get("objective"),
        "count": rec.get("count"),
        "valid_count": rec.get("valid_count"),
        "failure_count": rec.get("failure_count"),
        "mean": rec.get("mean"),
        "std": rec.get("std"),
        "min": rec.get("min"),
        "max": rec.get("max"),
        "asp_set_size": rec.get("asp_set_size"),
        "candidate_mode": rec.get("candidate_mode"),
        "seconds": rec.get("seconds"),
    }


def main() -> None:
    global MAX_INSTANCES
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsp-sizes", default="50,100")
    parser.add_argument("--kp-settings", default="100:25,200:25")
    parser.add_argument("--asp-settings", default="12:7,15:10,21:15")
    parser.add_argument("--tasks", default="tsp,kp,asp", help="Comma-separated subset from tsp,kp,asp.")
    parser.add_argument("--asp-max-candidates", type=int, default=200000, help="Compatibility only; ASP uses the exact ReEvo evaluator.")
    parser.add_argument("--asp-seed", type=int, default=1234)
    parser.add_argument("--max-instances", type=int, default=0, help="Limit TSP/KP instances for quick checks; 0 means all.")
    parser.add_argument("--output", default=str(RESULTS_ROOT / "construct_eval_results.json"))
    parser.add_argument("--csv-output", default=str(RESULTS_ROOT / "construct_eval_results.csv"))
    args = parser.parse_args()
    MAX_INSTANCES = max(0, int(args.max_instances))

    tsp_sizes = [int(x) for x in args.tsp_sizes.split(",") if x.strip()]
    kp_settings = [tuple(map(float, x.split(":"))) for x in args.kp_settings.split(",") if x.strip()]
    asp_settings = [tuple(map(int, x.split(":"))) for x in args.asp_settings.split(",") if x.strip()]
    enabled_tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}

    results: dict[str, Any] = {"results_root": str(RESULTS_ROOT), "programs": []}
    rows: list[dict[str, Any]] = []
    programs = discover_programs()

    for task, paths in programs.items():
        if task not in enabled_tasks:
            continue
        for program in paths:
            method = program.relative_to(RESULTS_ROOT).parts[0]
            module = load_module(program)
            item = {"method": method, "task": task, "program": str(program), "results": {}}
            results["programs"].append(item)

            if task == "tsp":
                for n in tsp_sizes:
                    rec = eval_tsp(module, n)
                    key = f"N={n}"
                    item["results"][key] = rec
                    rows.append(row_for(method, task, program, key, rec))
                    print(f"{method} {task} {program.name} {key}: mean={rec['mean']} valid={rec['valid_count']}/{rec['count']}", flush=True)
            elif task == "kp":
                for n_float, capacity in kp_settings:
                    n = int(n_float)
                    rec = eval_kp(module, n, capacity)
                    key = f"N={n},W={capacity:g}"
                    item["results"][key] = rec
                    rows.append(row_for(method, task, program, key, rec))
                    print(f"{method} {task} {program.name} {key}: mean={rec['mean']} valid={rec['valid_count']}/{rec['count']}", flush=True)
            elif task == "asp":
                for n, w in asp_settings:
                    rec = eval_asp(module, n, w, args.asp_max_candidates, args.asp_seed)
                    key = f"N={n},w={w}"
                    item["results"][key] = rec
                    rows.append(row_for(method, task, program, key, rec))
                    print(f"{method} {task} {program.name} {key}: set={rec['asp_set_size']} mode={rec['candidate_mode']}", flush=True)

            Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            write_csv(Path(args.csv_output), rows)

    print(f"wrote {args.output}")
    print(f"wrote {args.csv_output}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "task",
        "program",
        "setting",
        "objective",
        "count",
        "valid_count",
        "failure_count",
        "mean",
        "std",
        "min",
        "max",
        "asp_set_size",
        "candidate_mode",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    MAX_INSTANCES = 0
    main()
