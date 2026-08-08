"""Evaluate AHD ACO final code files.

This mirrors the MCTS-AHD ACO settings:
- TSP ACO: 100 iterations, 30 ants.
- CVRP ACO: 100 iterations, 30 ants, capacity 50.
- BPP offline ACO: sample mode with 200 samples by default; ACO mode uses
  15 iterations and 20 ants.

The script searches final_best_code.py files under ahd-test-time/results and
evaluates their `heuristics*` function on datasets under data/ahd/datasets.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.distributions import Categorical
except ImportError as exc:  # pragma: no cover - surfaced at runtime.
    raise SystemExit("ACO eval requires torch. Install torch or run in the project env.") from exc


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "ahd-test-time" / "results"
DATA_ROOT = ROOT / "data" / "ahd" / "datasets"

DEFAULT_TSP_SIZES = "20,50,100"
DEFAULT_CVRP_SIZES = "20,50,100"
DEFAULT_BPP_SIZES = "500,1000"


@dataclass(frozen=True)
class AcoSetting:
    task: str
    result_dir: str
    dataset_dir: str
    file_prefix: str
    objective: str = "min"


SETTINGS = {
    "tsp": AcoSetting("tsp", "TSP_ACO", "tsp_aco", "aco_tsp"),
    "cvrp": AcoSetting("cvrp", "CVRP_ACO", "cvrp_aco", "aco_cvrp"),
    "bpp": AcoSetting("bpp", "BPP_ACO", "bpp_offline_aco", "aco_bpp"),
}


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_heuristics(module: Any) -> Any:
    names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3", "heuristics_v4"]
    for name in names:
        if hasattr(module, name):
            candidate = getattr(module, name)
            if callable(candidate):
                return candidate
    for name in dir(module):
        if name.startswith("heuristics_v"):
            candidate = getattr(module, name)
            if callable(candidate):
                return candidate
    raise AttributeError("Candidate module has no heuristics function.")


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def sanitize_heuristic(heuristic: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    heuristic = np.asarray(heuristic, dtype=float)
    if heuristic.shape != shape:
        raise ValueError(f"heuristics returned shape {heuristic.shape}, expected {shape}")
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic[heuristic < 1e-9] = 1e-9
    return heuristic


class TspAco:
    def __init__(self, distances: np.ndarray, heuristic: np.ndarray, n_ants: int = 30, decay: float = 0.9):
        self.problem_size = len(distances)
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.heuristic = torch.tensor(heuristic, dtype=torch.float32)
        self.n_ants = n_ants
        self.decay = decay
        self.pheromone = torch.ones_like(self.distances)
        self.lowest_cost = float("inf")

    @torch.no_grad()
    def run(self, n_iterations: int) -> float:
        for _ in range(n_iterations):
            paths = self.gen_path()
            costs = self.gen_path_costs(paths)
            best_cost = costs.min().item()
            if best_cost < self.lowest_cost:
                self.lowest_cost = best_cost
            self.update_pheromone(paths, costs)
        return float(self.lowest_cost)

    def gen_path(self) -> torch.Tensor:
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,))
        mask = torch.ones((self.n_ants, self.problem_size), dtype=torch.float32)
        mask[torch.arange(self.n_ants), start] = 0
        paths = [start]
        prev = start
        for _ in range(self.problem_size - 1):
            dist = self.pheromone[prev] * self.heuristic[prev] * mask
            actions = Categorical(dist).sample()
            paths.append(actions)
            prev = actions
            mask[torch.arange(self.n_ants), actions] = 0
        return torch.stack(paths)

    def gen_path_costs(self, paths: torch.Tensor) -> torch.Tensor:
        u = paths.T
        v = torch.roll(u, shifts=1, dims=1)
        return torch.sum(self.distances[u, v], dim=1)

    def update_pheromone(self, paths: torch.Tensor, costs: torch.Tensor) -> None:
        self.pheromone *= self.decay
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            self.pheromone[path, torch.roll(path, shifts=1)] += 1.0 / cost
            self.pheromone[torch.roll(path, shifts=1), path] += 1.0 / cost


class CvrpAco:
    def __init__(
        self,
        distances: np.ndarray,
        demand: np.ndarray,
        heuristic: np.ndarray,
        capacity: int = 50,
        n_ants: int = 30,
        decay: float = 0.9,
    ):
        self.problem_size = len(distances)
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.demand = torch.tensor(demand, dtype=torch.float32)
        self.heuristic = torch.tensor(heuristic, dtype=torch.float32)
        self.capacity = capacity
        self.n_ants = n_ants
        self.decay = decay
        self.pheromone = torch.ones_like(self.distances)
        self.lowest_cost = float("inf")

    @torch.no_grad()
    def run(self, n_iterations: int) -> float:
        for _ in range(n_iterations):
            paths = self.gen_path()
            costs = self.gen_path_costs(paths)
            best_cost = costs.min().item()
            if best_cost < self.lowest_cost:
                self.lowest_cost = best_cost
            self.update_pheromone(paths, costs)
        return float(self.lowest_cost)

    def gen_path(self) -> torch.Tensor:
        actions = torch.zeros((self.n_ants,), dtype=torch.long)
        visit_mask = torch.ones((self.n_ants, self.problem_size), dtype=torch.float32)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros((self.n_ants,), dtype=torch.float32)
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        paths = [actions]
        while not self.check_done(visit_mask, actions):
            actions = self.pick_move(actions, visit_mask, capacity_mask)
            paths.append(actions)
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        return torch.stack(paths)

    def pick_move(self, prev: torch.Tensor, visit_mask: torch.Tensor, capacity_mask: torch.Tensor) -> torch.Tensor:
        dist = self.pheromone[prev] * self.heuristic[prev] * visit_mask * capacity_mask
        return Categorical(dist).sample()

    def update_visit_mask(self, visit_mask: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        visit_mask[torch.arange(self.n_ants), actions] = 0
        visit_mask[:, 0] = 1
        visit_mask[(actions == 0) * (visit_mask[:, 1:] != 0).any(dim=1), 0] = 0
        return visit_mask

    def update_capacity_mask(self, cur_nodes: torch.Tensor, used_capacity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        capacity_mask = torch.ones((self.n_ants, self.problem_size), dtype=torch.float32)
        used_capacity[cur_nodes == 0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        remaining = self.capacity - used_capacity
        capacity_mask[self.demand.unsqueeze(0).repeat(self.n_ants, 1) > remaining.unsqueeze(-1)] = 0
        return used_capacity, capacity_mask

    def check_done(self, visit_mask: torch.Tensor, actions: torch.Tensor) -> bool:
        return bool((visit_mask[:, 1:] == 0).all() and (actions == 0).all())

    def gen_path_costs(self, paths: torch.Tensor) -> torch.Tensor:
        u = paths.permute(1, 0)
        v = torch.roll(u, shifts=-1, dims=1)
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def update_pheromone(self, paths: torch.Tensor, costs: torch.Tensor) -> None:
        self.pheromone *= self.decay
        for i in range(self.n_ants):
            path = paths[:, i]
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0 / costs[i]
        self.pheromone[self.pheromone < 1e-10] = 1e-10


def organize_bpp_path(path: np.ndarray) -> tuple[int, np.ndarray]:
    order: dict[int, int] = {}
    result = np.zeros_like(path)
    for i, value in enumerate(path):
        key = int(value)
        if key not in order:
            order[key] = len(order)
        result[i] = order[key]
    return len(order), result


def bpp_path_fitness(vacancies: list[int], capacity: int) -> float:
    occupied = capacity - np.array(vacancies, dtype=float)
    return float(((occupied / capacity) ** 2).sum() / len(vacancies))


class BppAco:
    def __init__(
        self,
        demand: np.ndarray,
        heuristic: np.ndarray,
        capacity: int = 150,
        n_ants: int = 20,
        decay: float = 0.95,
        greedy: bool = False,
    ):
        self.problem_size = len(demand)
        self.capacity = capacity
        self.demand = np.asarray(demand, dtype=int)
        self.n_ants = n_ants
        self.decay = decay
        self.greedy_mode = greedy
        heuristic = np.asarray(heuristic, dtype=float).copy()
        heuristic[heuristic > 1e6] = 1e6
        heuristic[heuristic < 1e-6] = 1e-6
        heuristic = heuristic / heuristic.max()
        heuristic[heuristic < 1e-6] = 1e-6
        self.heuristic = heuristic
        self.pheromone = np.ones((self.problem_size, self.problem_size), dtype=float)
        self.shortest_path = np.arange(self.problem_size)
        self.best_cost = self.problem_size
        self._ordinal = np.arange(self.problem_size, dtype=int)

    def run(self, iterations: int) -> tuple[int, np.ndarray]:
        for _ in range(iterations):
            prob = self.pheromone * self.heuristic
            paths, costs, fitnesses = self.gen_paths(self.n_ants, prob)
            best_index = int(costs.argmin())
            best_cost = int(costs[best_index])
            if best_cost < self.best_cost:
                self.shortest_path = paths[best_index]
                self.best_cost = best_cost
            self.update_pheromone(paths, fitnesses)
        return organize_bpp_path(self.shortest_path)

    def sample_only(self, count: int) -> tuple[int, np.ndarray]:
        self.greedy_mode = True
        paths, costs, _ = self.gen_paths(count, self.heuristic)
        return organize_bpp_path(paths[int(costs.argmin())])

    def gen_paths(self, count: int, prob: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        paths, costs, fitnesses = [], [], []
        for _ in range(count):
            path, cost, fitness = self.sample_path(prob)
            paths.append(path)
            costs.append(cost)
            fitnesses.append(fitness)
        return paths, np.array(costs, dtype=int), np.array(fitnesses, dtype=float)

    def sample_path(self, prob: np.ndarray) -> tuple[np.ndarray, int, float]:
        path = np.ones(self.problem_size, dtype=int) * -1
        valid_items = np.ones(self.problem_size, dtype=bool)
        current_bin = item_count = 0
        vacancies: list[int] = []
        bin_vacancy = self.capacity
        bin_items = np.zeros(self.problem_size, dtype=bool)
        for _ in range(self.problem_size):
            mask = np.bitwise_and(self.demand <= bin_vacancy, valid_items)
            if not np.any(mask):
                vacancies.append(int(bin_vacancy))
                bin_vacancy, item_count = self.capacity, 0
                current_bin += 1
                bin_items[:] = False
                selected = self.random_select(valid_items)
            elif item_count == 0:
                selected = self.random_select(mask)
            else:
                item_prob = (prob[bin_items].sum(0) / item_count + 1e-5) * mask
                selected = int(item_prob.argmax()) if self.greedy_mode else self.random_sample(item_prob)
            bin_items[selected] = True
            bin_vacancy -= int(self.demand[selected])
            valid_items[selected] = False
            path[selected] = current_bin
            item_count += 1
        vacancies.append(int(bin_vacancy))
        return path, len(vacancies), bpp_path_fitness(vacancies, self.capacity)

    def update_pheromone(self, paths: list[np.ndarray], fitnesses: np.ndarray) -> None:
        delta = np.zeros_like(self.pheromone)
        for path, fitness in zip(paths, fitnesses):
            delta[path[:, None] == path[None, :]] += fitness / self.n_ants
        self.pheromone *= self.decay
        self.pheromone += delta

    def random_select(self, mask: np.ndarray) -> int:
        valid = self._ordinal[mask]
        return int(valid[np.random.randint(0, len(valid))])

    @staticmethod
    def random_sample(prob: np.ndarray) -> int:
        total = float(prob.sum())
        if total <= 0 or not math.isfinite(total):
            return int(np.argmax(prob))
        return int(np.searchsorted(np.cumsum(prob), np.random.random() * total))


def solve_tsp(heuristics: Any, node_pos: np.ndarray, n_iterations: int, n_ants: int) -> float:
    dist = pairwise_distances(node_pos)
    dist[np.diag_indices_from(dist)] = 1
    heuristic = sanitize_heuristic(heuristics(dist.copy()), dist.shape)
    return TspAco(dist, heuristic, n_ants=n_ants).run(n_iterations)


def solve_cvrp(heuristics: Any, instance: np.ndarray, n_iterations: int, n_ants: int, capacity: int) -> float:
    demand = instance[:, 0]
    node_pos = instance[:, 1:]
    dist = pairwise_distances(node_pos)
    dist[np.diag_indices_from(dist)] = 1
    argc = len(inspect.getfullargspec(heuristics).args)
    if argc == 4:
        raw = heuristics(dist.copy(), node_pos.copy(), demand.copy(), capacity)
    elif argc == 2:
        raw = heuristics(dist.copy(), demand.copy() / capacity)
    else:
        raise TypeError(f"CVRP heuristics must accept 2 or 4 args, got {argc}")
    heuristic = sanitize_heuristic(raw, dist.shape)
    return CvrpAco(dist, demand, heuristic, capacity=capacity, n_ants=n_ants).run(n_iterations)


def solve_bpp(heuristics: Any, demand: np.ndarray, mode: str, n_iterations: int, n_ants: int, sample_count: int, capacity: int) -> float:
    raw = heuristics(demand.copy(), capacity)
    heuristic = sanitize_heuristic(raw, (len(demand), len(demand)))
    aco = BppAco(demand, heuristic, capacity=capacity, n_ants=n_ants, greedy=False)
    if mode == "sample":
        obj, _ = aco.sample_only(sample_count)
    else:
        obj, _ = aco.run(n_iterations)
    return float(obj)


def summarize(values: list[float], total_count: int, objective: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {
            "objective": objective,
            "count": total_count,
            "valid_count": 0,
            "failure_count": total_count,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "objective": objective,
        "count": total_count,
        "valid_count": int(len(arr)),
        "failure_count": int(total_count - len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def discover_code_files(task: str) -> list[tuple[str, Path]]:
    setting = SETTINGS[task]
    found: list[tuple[str, Path]] = []
    for method_dir in sorted(path for path in RESULTS_ROOT.iterdir() if path.is_dir()):
        folder = method_dir / setting.result_dir
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*final_best_code.py")):
            found.append((method_dir.name, path))
    return found


def rep_from_name(path: Path) -> int | None:
    match = re.search(r"rep(\d+)", path.name)
    return int(match.group(1)) if match else None


def eval_code_file(task: str, code_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    module = load_module(code_path)
    heuristics = get_heuristics(module)
    rows: list[dict[str, Any]] = []

    if task == "tsp":
        for size in parse_ints(args.tsp_sizes):
            data = np.load(DATA_ROOT / "tsp_aco" / f"{args.split}{size}_dataset.npy", allow_pickle=False)
            if args.max_instances > 0:
                data = data[: args.max_instances]
            values = []
            for node_pos in data:
                try:
                    values.append(solve_tsp(heuristics, np.asarray(node_pos), args.tsp_iterations, args.tsp_ants))
                except Exception:
                    if not args.keep_going:
                        raise
            rows.append({"task": task, "size": size, **summarize(values, len(data), "min")})

    elif task == "cvrp":
        for size in parse_ints(args.cvrp_sizes):
            data = np.load(DATA_ROOT / "cvrp_aco" / f"{args.split}{size}_dataset.npy", allow_pickle=False)
            if args.max_instances > 0:
                data = data[: args.max_instances]
            values = []
            for instance in data:
                try:
                    values.append(solve_cvrp(heuristics, np.asarray(instance), args.cvrp_iterations, args.cvrp_ants, args.cvrp_capacity))
                except Exception:
                    if not args.keep_going:
                        raise
            rows.append({"task": task, "size": size, **summarize(values, len(data), "min")})

    elif task == "bpp":
        for size in parse_ints(args.bpp_sizes):
            data = np.load(DATA_ROOT / "bpp_offline_aco" / f"{args.split}{size}_dataset.npz", allow_pickle=False)["demands"]
            if args.max_instances > 0:
                data = data[: args.max_instances]
            values = []
            for demand in data:
                try:
                    values.append(
                        solve_bpp(
                            heuristics,
                            np.asarray(demand),
                            args.bpp_mode,
                            args.bpp_iterations,
                            args.bpp_ants,
                            args.bpp_sample_count,
                            args.bpp_capacity,
                        )
                    )
                except Exception:
                    if not args.keep_going:
                        raise
            rows.append({"task": task, "size": size, "mode": args.bpp_mode, **summarize(values, len(data), "min")})

    return rows


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="tsp,cvrp,bpp", help="Comma-separated subset from tsp,cvrp,bpp.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tsp-sizes", default=DEFAULT_TSP_SIZES)
    parser.add_argument("--cvrp-sizes", default=DEFAULT_CVRP_SIZES)
    parser.add_argument("--bpp-sizes", default=DEFAULT_BPP_SIZES)
    parser.add_argument("--tsp-iterations", type=int, default=100)
    parser.add_argument("--tsp-ants", type=int, default=30)
    parser.add_argument("--cvrp-iterations", type=int, default=100)
    parser.add_argument("--cvrp-ants", type=int, default=30)
    parser.add_argument("--cvrp-capacity", type=int, default=50)
    parser.add_argument("--bpp-mode", default="sample", choices=["sample", "aco"])
    parser.add_argument("--bpp-iterations", type=int, default=15)
    parser.add_argument("--bpp-ants", type=int, default=20)
    parser.add_argument("--bpp-sample-count", type=int, default=200)
    parser.add_argument("--bpp-capacity", type=int, default=150)
    parser.add_argument("--max-instances", type=int, default=0, help="Limit instances for quick checks; 0 means all.")
    parser.add_argument("--keep-going", action="store_true", help="Record failures instead of stopping on the first error.")
    parser.add_argument("--output", default=str(RESULTS_ROOT / "aco_eval_results.json"))
    parser.add_argument("--csv-output", default=str(RESULTS_ROOT / "aco_eval_results.csv"))
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        if task not in SETTINGS:
            raise ValueError(f"Unknown task: {task}")
        for method, code_path in discover_code_files(task):
            for row in eval_code_file(task, code_path, args):
                row.update({"method": method, "rep": rep_from_name(code_path), "code_file": str(code_path)})
                all_rows.append(row)
                print(f"{method} {task} {code_path.name} size={row['size']}: mean={row['mean']} valid={row['valid_count']}/{row['count']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
