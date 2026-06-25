import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np


from . import get_prompts as _prompts


def _resolve_root():
    for parent in Path(__file__).resolve().parents:
        data_root = parent / "data" / "ahd" / "datasets" / "kp_constructive"
        if data_root.is_dir():
            return data_root
    raise RuntimeError("Could not resolve kp_constructive dataset root.")


def _resolve_instances(root: Path, split: str, problem_size: int) -> np.ndarray:
    preferred = root / f"{split}{problem_size}_dataset.npy"
    if preferred.exists():
        return np.load(preferred, allow_pickle=True)

    fallback = sorted(root.glob(f"*{split}*dataset.npy"))
    if not fallback:
        raise FileNotFoundError(f"No dataset file found in {root} for split {split!r}")
    return np.load(fallback[0], allow_pickle=True)


class KPCONST:
    def __init__(self, paras: Any = None) -> None:
        self.problem_size = 100
        self.n_instance = 64
        self.capacity = 25.0
        self.prompts = _prompts.GetPrompts()

        split = "train"
        if hasattr(paras, "data_split"):
            split = str(getattr(paras, "data_split", split))
        data_root = _resolve_root()
        if getattr(paras, "problem_data_root", None):
            explicit_root = Path(str(getattr(paras, "problem_data_root")))
            if explicit_root.exists():
                data_root = explicit_root
        data = _resolve_instances(data_root, split, self.problem_size).astype(float)

        if data.ndim != 3 or data.shape[-1] != 2:
            raise ValueError(f"Unexpected KP dataset shape: {data.shape}")

        self.instance_data = data[: self.n_instance]
        self.n_instance = min(self.n_instance, self.instance_data.shape[0])
        self.instance_data = self.instance_data[: self.n_instance]

    def greedy(self, eva):
        values = np.zeros(self.n_instance, dtype=float)
        for i, instance in enumerate(self.instance_data):
            weights = np.array(instance[:, 0], dtype=float)
            item_values = np.array(instance[:, 1], dtype=float)

            if weights.shape[0] < self.problem_size:
                # keep compatibility with smaller fallback instances
                self.problem_size = weights.shape[0]

            remaining_capacity = float(self.capacity)
            solution_value = 0.0

            while weights.size > 0 and remaining_capacity + 1e-12 >= np.min(weights):
                original_weights = weights.copy()
                original_values = item_values.copy()
                next_item = eva.select_next_item(
                    remaining_capacity=remaining_capacity,
                    weights=original_weights.copy(),
                    values=original_values.copy(),
                )

                if isinstance(next_item, np.ndarray):
                    if next_item.size != 1:
                        return None
                    next_item = next_item.item()

                next_item = int(next_item)
                if next_item < 0 or next_item >= original_weights.size:
                    return None
                if (
                    not np.isfinite(original_weights[next_item])
                    or original_weights[next_item] > remaining_capacity + 1e-12
                ):
                    return None

                solution_value += float(original_values[next_item])
                remaining_capacity -= float(original_weights[next_item])
                weights = np.delete(weights, next_item)
                item_values = np.delete(item_values, next_item)

            values[i] = solution_value

        return -float(np.mean(values))

    def evaluate(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heuristic_module = types.ModuleType("heuristic_module")
                heuristic_module.__dict__["np"] = np
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module

                fitness = self.greedy(heuristic_module)
                if fitness is None or not np.isfinite(fitness):
                    return None
                return fitness
        except Exception:
            return None
