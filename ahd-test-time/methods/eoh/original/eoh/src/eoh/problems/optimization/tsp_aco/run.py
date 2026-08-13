import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..settings_prompts import SettingsPrompts


def _resolve_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "algorithms" / "ahd" / "aco_tsp_evaluator.py").is_file():
            return parent
    raise RuntimeError("Could not resolve Agentic-ESOpt repository root.")


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ahd.aco_tsp_evaluator import (
    seed_aco_random_stream,
    select_heuristic_function,
    solve_tsp_instance,
)


def _resolve_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        data_root = parent / "data" / "ahd" / "datasets" / "tsp_aco"
        if data_root.is_dir():
            return data_root
    raise RuntimeError("Could not resolve tsp_aco dataset root.")


def _resolve_instances(root: Path, split: str, problem_size: int) -> np.ndarray:
    preferred = root / f"{split}{problem_size}_dataset.npy"
    if preferred.exists():
        return np.load(preferred, allow_pickle=True)
    fallback = sorted(root.glob(f"*{split}*dataset.npy"))
    if not fallback:
        raise FileNotFoundError(f"No dataset file found in {root} for split {split!r}")
    return np.load(fallback[0], allow_pickle=True)


def _get_heuristic_name(module) -> str:
    return select_heuristic_function(module)[0]


class TSPACO:
    def __init__(self, paras: Any | None = None) -> None:
        self.problem_size = 50
        self.prompts = SettingsPrompts("tsp_aco")
        self.n_iterations = 100
        self.n_ants = 30
        self.evaluation_seed = int(getattr(paras, "evaluation_seed", 1234))

        split = str(getattr(paras, "data_split", "train"))
        data_root = _resolve_root()
        if getattr(paras, "problem_data_root", None):
            explicit_root = Path(str(getattr(paras, "problem_data_root")))
            if explicit_root.exists():
                data_root = explicit_root
        self.node_positions = _resolve_instances(data_root, split, self.problem_size).astype(float)

    def solve(self, node_pos, heuristics):
        return solve_tsp_instance(
            heuristics,
            np.asarray(node_pos),
            n_iterations=self.n_iterations,
            n_ants=self.n_ants,
        )

    def evaluate(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heuristic_module = types.ModuleType("heuristic_module")
                heuristic_module.__dict__["np"] = np
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module
                heuristics = getattr(heuristic_module, _get_heuristic_name(heuristic_module))

                objs = []
                for instance_index, node_pos in enumerate(self.node_positions):
                    seed_aco_random_stream(
                        getattr(self, "evaluation_seed", 1234)
                        + self.problem_size * 1_000
                        + instance_index
                    )
                    objs.append(self.solve(node_pos, heuristics))
                return float(np.mean(objs))
        except Exception:
            return None
