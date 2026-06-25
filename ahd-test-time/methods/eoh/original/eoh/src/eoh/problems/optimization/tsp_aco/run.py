import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import distance_matrix

from .aco import ACO
from ..settings_prompts import SettingsPrompts


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
    for name in ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]:
        if callable(getattr(module, name, None)):
            return name
    raise AttributeError("No heuristic function found.")


class TSPACO:
    def __init__(self, paras: Any | None = None) -> None:
        self.problem_size = 50
        self.prompts = SettingsPrompts("tsp_aco")
        self.n_iterations = 100
        self.n_ants = 30

        split = str(getattr(paras, "data_split", "train"))
        data_root = _resolve_root()
        if getattr(paras, "problem_data_root", None):
            explicit_root = Path(str(getattr(paras, "problem_data_root")))
            if explicit_root.exists():
                data_root = explicit_root
        self.node_positions = _resolve_instances(data_root, split, self.problem_size).astype(float)

    def solve(self, node_pos, heuristics):
        dist_mat = distance_matrix(node_pos, node_pos)
        dist_mat[np.diag_indices_from(dist_mat)] = 1
        heu = np.asarray(heuristics(dist_mat.copy()), dtype=float) + 1e-9
        if heu.shape != dist_mat.shape:
            raise ValueError("Heuristic shape mismatch.")
        heu[heu < 1e-9] = 1e-9
        aco = ACO(dist_mat, heu, n_ants=self.n_ants)
        return float(aco.run(self.n_iterations))

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
                for node_pos in self.node_positions:
                    objs.append(self.solve(node_pos, heuristics))
                return float(np.mean(objs))
        except Exception:
            return None
