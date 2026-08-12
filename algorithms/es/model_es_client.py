import json
from typing import Any, Dict, List, Optional

import requests


class ModelESClient:
    """HTTP client for seed-replay ES updates on a local LLM server."""

    def __init__(self, completions_url: str, timeout: float = 600.0):
        self.completions_url = completions_url
        self.timeout = timeout
        if completions_url.endswith("/completions"):
            self.base_url = completions_url[: -len("/completions")]
        else:
            self.base_url = completions_url.rstrip("/")
        self._initialized = False

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = requests.post(
            self.base_url + path,
            data=json.dumps(payload or {}),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("ok") is False:
            raise RuntimeError(f"Model ES server error at {path}: {data}")
        return data

    def init(self, *, parameter_scope: str = "full", target_modules=None, verbose: bool = True):
        data = self._post(
            "/es/init",
            {"parameter_scope": parameter_scope, "target_modules": target_modules, "verbose": verbose},
        )
        self._initialized = True
        return data

    def apply_perturbation(self, *, seed: int, sigma: float):
        return self._post("/es/apply", {"seed": int(seed), "sigma": float(sigma)})

    def revert_perturbation(self, *, seed: int, sigma: float):
        return self._post("/es/revert", {"seed": int(seed), "sigma": float(sigma)})

    def reset(self):
        return self._post("/es/reset", {})

    def update(
        self,
        *,
        seeds: List[int],
        rewards: List[float],
        alpha: float,
        reward_normalization: str = "zscore",
        reward_normalization_ddof: int = 0,
        reward_normalization_eps: float = 1e-8,
    ):
        result = self._post(
            "/es/update",
            {
                "seeds": [int(seed) for seed in seeds],
                "rewards": [float(reward) for reward in rewards],
                "alpha": float(alpha),
                "reward_normalization": reward_normalization,
                "reward_normalization_ddof": int(reward_normalization_ddof),
                "reward_normalization_eps": float(reward_normalization_eps),
            },
        )
        result.setdefault("endpoint", self.base_url)
        return result

    def status(self):
        return self._post("/es/status", {})
