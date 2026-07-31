from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_COMPAT = ROOT / "verl" / "verl" / "workers" / "rollout" / "sglang_rollout" / "runtime_compat.py"
SPEC = importlib.util.spec_from_file_location("sglang_runtime_compat", RUNTIME_COMPAT)
assert SPEC is not None and SPEC.loader is not None
RUNTIME_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME_MODULE)


class VerlDocVQASamplingTests(unittest.TestCase):
    def test_async_server_honors_smaller_per_turn_budget(self) -> None:
        self.assertEqual(
            RUNTIME_MODULE.bounded_max_new_tokens(
                response_length=32768,
                max_model_len=131072,
                prompt_length=4096,
                requested_max_new_tokens=512,
            ),
            512,
        )

    def test_sglang_replicas_get_isolated_triton_cache(self) -> None:
        previous = os.environ.get("TRITON_CACHE_DIR")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = RUNTIME_MODULE.configure_triton_cache(tmpdir, "sglang_replica_2_node_0")
                self.assertTrue(path.is_dir())
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(path))
        finally:
            if previous is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous

    def test_rollout_config_accepts_required_penalties(self) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = str(ROOT / "verl")
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from verl.workers.config.rollout import RolloutConfig; "
                    "c=RolloutConfig(name='sglang', top_k=40, presence_penalty=2.0, "
                    "repetition_penalty=1.0); "
                    "print(c.top_k, c.presence_penalty, c.repetition_penalty)"
                ),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "40 2.0 1.0")


if __name__ == "__main__":
    unittest.main()
