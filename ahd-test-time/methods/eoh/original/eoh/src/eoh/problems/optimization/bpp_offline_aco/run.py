import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .aco import ACO
from .gen_inst import BPPInstance, dataset_conf, load_dataset
from ..settings_prompts import SettingsPrompts


def _resolve_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        data_root = parent / "data" / "ahd" / "datasets" / "bpp_offline_aco"
        if data_root.is_dir():
            return data_root
    raise RuntimeError("Could not resolve bpp_offline_aco dataset root.")


def _resolve_instances(root: Path, split: str, problem_size: int) -> list[BPPInstance]:
    preferred = root / f"{split}{problem_size}_dataset.npz"
    if preferred.exists():
        return load_dataset(preferred)
    fallback = sorted(root.glob(f"*{split}*dataset.npz"))
    if not fallback:
        raise FileNotFoundError(f"No dataset file found in {root} for split {split!r}")
    return load_dataset(fallback[0])


def _get_heuristic_name(module) -> str:
    for name in ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]:
        if callable(getattr(module, name, None)):
            return name
    raise AttributeError("No heuristic function found.")


class BPPOFFLINEACO:
    def __init__(self, paras: Any | None = None) -> None:
        self.problem_size = dataset_conf["train"][0]
        self.prompts = SettingsPrompts("bpp_offline_aco")
        self.n_iterations = 15
        self.n_ants = 20
        self.sample_count = 200

        split = str(getattr(paras, "data_split", "train"))
        data_root = _resolve_root()
        if getattr(paras, "problem_data_root", None):
            explicit_root = Path(str(getattr(paras, "problem_data_root")))
            if explicit_root.exists():
                data_root = explicit_root
        self.dataset = _resolve_instances(data_root, split, self.problem_size)

    def solve(self, inst: BPPInstance, heuristics, mode="sample"):
        heu = np.asarray(heuristics(inst.demands.copy(), inst.capacity), dtype=float)
        assert tuple(heu.shape) == (inst.n, inst.n)
        assert 0 < heu.max() < np.inf
        aco = ACO(inst.demands, heu.astype(float), capacity=inst.capacity, n_ants=self.n_ants, greedy=False)
        if mode == "sample":
            obj, _ = aco.sample_only(self.sample_count)
        else:
            obj, _ = aco.run(self.n_iterations)
        return float(obj)

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
                for instance in self.dataset:
                    objs.append(self.solve(instance, heuristics, mode="sample"))
                return float(np.mean(objs))
        except Exception:
            return None
