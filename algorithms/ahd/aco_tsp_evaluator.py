"""Single numerical contract used by ACO-TSP search and final evaluation."""

from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np
import torch
from torch.distributions import Categorical


EVALUATOR_DTYPE = np.dtype(np.float32)
HEURISTIC_FLOOR = np.float32(1e-9)


class AcoNumericalError(ValueError):
    """The candidate cannot be evaluated under the frozen numeric contract."""


def select_heuristic_function(module: Any) -> tuple[str, Callable[[np.ndarray], Any]]:
    """Apply one shared function-selection order at every evaluator entry point."""
    preferred = (
        "heuristics",
        "heuristics_v1",
        "heuristics_v2",
        "heuristics_v3",
        "heuristics_v4",
    )
    for name in preferred:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return name, candidate
    for name in sorted(item for item in dir(module) if item.startswith("heuristics_v")):
        candidate = getattr(module, name)
        if callable(candidate):
            return name, candidate
    raise AttributeError("candidate module has no callable heuristics function")


def seed_aco_random_stream(seed: int) -> None:
    """Reset every RNG a generated heuristic or the ACO sampler may consume."""
    normalized = int(seed) % (2**32)
    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)


def pairwise_distances_float64(node_positions: np.ndarray) -> np.ndarray:
    """Build distances in one frozen order before the downstream float32 cast."""
    points = np.asarray(node_positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        raise AcoNumericalError("node positions must be a 2D array with at least two nodes")
    if not np.isfinite(points).all():
        raise AcoNumericalError("node positions contain non-finite values")
    differences = points[:, None, :] - points[None, :, :]
    squared_distances = np.sum(differences * differences, axis=-1, dtype=np.float64)
    distances = np.sqrt(squared_distances)
    distances[np.diag_indices_from(distances)] = 1.0
    return distances


def prepare_float32_inputs(
    distances: np.ndarray,
    raw_heuristic: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast both evaluator inputs once and reject lossy overflow fail-closed."""
    distance_array = np.asarray(distances, dtype=EVALUATOR_DTYPE)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            heuristic_array = np.asarray(raw_heuristic).astype(EVALUATOR_DTYPE, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AcoNumericalError(f"heuristic cannot be converted to float32: {exc}") from exc
    if heuristic_array.shape != distance_array.shape:
        raise AcoNumericalError(
            f"heuristic shape {heuristic_array.shape} does not match "
            f"distances {distance_array.shape}"
        )
    if not np.isfinite(distance_array).all():
        raise AcoNumericalError("distances are non-finite after float32 conversion")
    if not np.isfinite(heuristic_array).all():
        raise AcoNumericalError("heuristic is non-finite after float32 conversion")
    heuristic_array[heuristic_array < HEURISTIC_FLOOR] = HEURISTIC_FLOOR
    return distance_array, heuristic_array


class FrozenTspAco:
    """CPU ACO implementation shared verbatim by training and final scoring."""

    def __init__(
        self,
        distances: np.ndarray,
        heuristic: np.ndarray,
        n_ants: int = 30,
        decay: float = 0.9,
    ) -> None:
        self.problem_size = len(distances)
        self.distances = torch.as_tensor(distances, dtype=torch.float32, device="cpu")
        self.heuristic = torch.as_tensor(heuristic, dtype=torch.float32, device="cpu")
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.pheromone = torch.ones_like(self.distances)
        self.lowest_cost = float("inf")

    @torch.no_grad()
    def run(self, n_iterations: int) -> float:
        for _ in range(int(n_iterations)):
            paths = self.gen_path()
            costs = self.gen_path_costs(paths)
            if not torch.isfinite(costs).all() or bool((costs <= 0).any()):
                raise AcoNumericalError("ACO produced non-finite or non-positive path costs")
            best_cost = float(costs.min().item())
            if best_cost < self.lowest_cost:
                self.lowest_cost = best_cost
            self.update_pheromone(paths, costs)
        if not math.isfinite(self.lowest_cost):
            raise AcoNumericalError("ACO did not produce a finite objective")
        return self.lowest_cost

    def gen_path(self) -> torch.Tensor:
        start = torch.randint(
            low=0,
            high=self.problem_size,
            size=(self.n_ants,),
            device="cpu",
        )
        mask = torch.ones(
            (self.n_ants, self.problem_size),
            dtype=torch.float32,
            device="cpu",
        )
        mask[torch.arange(self.n_ants), start] = 0
        paths = [start]
        previous = start
        for _ in range(self.problem_size - 1):
            weights = self.pheromone[previous] * self.heuristic[previous] * mask
            self._require_valid_weights(weights)
            actions = Categorical(probs=weights).sample()
            paths.append(actions)
            previous = actions
            mask[torch.arange(self.n_ants), actions] = 0
        return torch.stack(paths)

    @staticmethod
    def _require_valid_weights(weights: torch.Tensor) -> None:
        if not torch.isfinite(weights).all():
            raise AcoNumericalError("ACO categorical weights contain non-finite values")
        if bool((weights < 0).any()):
            raise AcoNumericalError("ACO categorical weights contain negative values")
        row_mass = weights.sum(dim=-1)
        if not torch.isfinite(row_mass).all() or bool((row_mass <= 0).any()):
            raise AcoNumericalError(
                "ACO categorical weights do not define a finite positive simplex"
            )

    def gen_path_costs(self, paths: torch.Tensor) -> torch.Tensor:
        expected = (self.problem_size, self.n_ants)
        if tuple(paths.shape) != expected:
            raise AcoNumericalError(
                f"ACO path shape {tuple(paths.shape)} does not match {expected}"
            )
        starts = paths.T
        ends = torch.roll(starts, shifts=1, dims=1)
        return torch.sum(self.distances[starts, ends], dim=1)

    def update_pheromone(self, paths: torch.Tensor, costs: torch.Tensor) -> None:
        self.pheromone *= self.decay
        for ant_index in range(self.n_ants):
            path = paths[:, ant_index]
            previous = torch.roll(path, shifts=1)
            deposit = 1.0 / costs[ant_index]
            self.pheromone[path, previous] += deposit
            self.pheromone[previous, path] += deposit
        if not torch.isfinite(self.pheromone).all():
            raise AcoNumericalError("ACO pheromone became non-finite")


def solve_tsp_instance(
    heuristics: Callable[[np.ndarray], Any],
    node_positions: np.ndarray,
    *,
    n_iterations: int = 100,
    n_ants: int = 30,
) -> float:
    """Evaluate one TSP instance under the frozen downstream float32 order."""
    distances_float64 = pairwise_distances_float64(node_positions)
    raw_heuristic = heuristics(distances_float64.copy())
    distances, heuristic = prepare_float32_inputs(distances_float64, raw_heuristic)
    return FrozenTspAco(distances, heuristic, n_ants=n_ants).run(n_iterations)
