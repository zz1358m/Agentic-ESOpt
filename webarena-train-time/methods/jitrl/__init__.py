from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util


def _load_webarena_env():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "envs" / "webarena.py"
    spec = importlib.util.spec_from_file_location("_webarena_train_time_env", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load WebArena env from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_webarena = _load_webarena_env()
WebArenaEnv = _webarena.WebArenaEnv
WebArenaRunResult = _webarena.WebArenaRunResult


class JustInRLMethod:
    """Adapter entry point for WebArena JitRL runs backed by local source snapshots."""

    SUPPORTED_WEBARENA_SETTINGS = ("jitrl", "train", "test")

    def __init__(self, *, env: WebArenaEnv | None = None) -> None:
        self.env = env or WebArenaEnv()

    def supports_setting(self, setting: str) -> bool:
        return setting in self.SUPPORTED_WEBARENA_SETTINGS

    def run_webarena(
        self,
        *,
        setting: str,
        result_dir: str | Path,
        log_path: str | Path,
        **kwargs: Any,
    ) -> WebArenaRunResult:
        if setting not in self.SUPPORTED_WEBARENA_SETTINGS:
            raise NotImplementedError(
                f"Just-in-RL setting {setting!r} is not implemented for WebArena in this repo."
            )
        return self.env.run_official_webarena(result_dir=result_dir, log_path=log_path, **kwargs)
