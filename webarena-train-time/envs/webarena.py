from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REQUIRED_URL_ENVS = (
    "REDDIT",
    "SHOPPING",
    "SHOPPING_ADMIN",
    "GITLAB",
    "WIKIPEDIA",
    "MAP",
    "HOMEPAGE",
)


@dataclass(frozen=True)
class WebArenaRunResult:
    status: str
    average_score: str | None
    log_path: Path
    returncode: int


class WebArenaEnv:
    """Thin adapter around the vendored WebArena runner."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        python: str | Path | None = None,
        url_env: Mapping[str, str] | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.root = Path(root) if root else repo_root / "data" / "webarena" / "source"
        self.python = str(python) if python else sys.executable
        self.url_env = dict(url_env or {})

    @staticmethod
    def url_env_from_host(host: str) -> dict[str, str]:
        host = host.rstrip("/")
        return {
            "SHOPPING": f"http://{host}:7770",
            "SHOPPING_ADMIN": f"http://{host}:7780/admin",
            "REDDIT": f"http://{host}:9999",
            "GITLAB": f"http://{host}:8023",
            "MAP": f"http://{host}:3000",
            "WIKIPEDIA": (
                f"http://{host}:8888/"
                "wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
            ),
            "HOMEPAGE": f"http://{host}:4399",
        }

    def merged_env(self, extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.url_env)
        if extra_env:
            env.update(extra_env)
        for name in REQUIRED_URL_ENVS:
            prefixed = f"WA_{name}"
            if not env.get(name) and env.get(prefixed):
                env[name] = env[prefixed]
            if not env.get(prefixed) and env.get(name):
                env[prefixed] = env[name]
        existing = env.get("PYTHONPATH", "")
        separator = ";" if os.name == "nt" else ":"
        env["PYTHONPATH"] = str(self.root) if not existing else f"{self.root}{separator}{existing}"
        return env

    def missing_url_envs(self, env: Mapping[str, str] | None = None) -> list[str]:
        values = self.merged_env(env)
        return [name for name in REQUIRED_URL_ENVS if not values.get(name)]

    def require_ready(self, env: Mapping[str, str] | None = None) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"WebArena source tree not found: {self.root}")
        missing = self.missing_url_envs(env)
        if missing:
            raise RuntimeError(f"Missing WebArena URL env vars: {' '.join(missing)}")

    def prepare_configs(self, env: Mapping[str, str] | None = None) -> None:
        run_env = self.merged_env(env)
        self.require_ready(run_env)
        subprocess.run(
            [self.python, "agent/prompts/to_json.py"],
            cwd=self.root,
            env=run_env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [self.python, "scripts/generate_test_data.py"],
            cwd=self.root,
            env=run_env,
            check=True,
        )

    def run_official_webarena(
        self,
        *,
        result_dir: str | Path,
        log_path: str | Path,
        model_name: str,
        model_endpoint: str,
        instruction_path: str,
        test_start_idx: int,
        test_end_idx: int,
        max_steps: int = 30,
        max_tokens: int = 384,
        temperature: float = 0.1,
        top_p: float = 0.9,
        env: Mapping[str, str] | None = None,
    ) -> WebArenaRunResult:
        run_env = self.merged_env(env)
        self.require_ready(run_env)
        result_dir = Path(result_dir)
        log_path = Path(log_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python,
            "run.py",
            "--provider",
            "local",
            "--model",
            model_name,
            "--mode",
            "completion",
            "--model_endpoint",
            model_endpoint,
            "--instruction_path",
            instruction_path,
            "--temperature",
            str(temperature),
            "--top_p",
            str(top_p),
            "--max_tokens",
            str(max_tokens),
            "--max_steps",
            str(max_steps),
            "--test_start_idx",
            str(test_start_idx),
            "--test_end_idx",
            str(test_end_idx),
            "--result_dir",
            str(result_dir),
        ]
        with log_path.open("w") as log:
            completed = subprocess.run(cmd, cwd=self.root, env=run_env, stdout=log, stderr=subprocess.STDOUT)

        log_text = log_path.read_text(errors="replace")
        scores = re.findall(r"Average score:\s*([^\s]+)", log_text)
        return WebArenaRunResult(
            status="completed" if completed.returncode == 0 else "failed",
            average_score=scores[-1] if scores else None,
            log_path=log_path,
            returncode=completed.returncode,
        )
