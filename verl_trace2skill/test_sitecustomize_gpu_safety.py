from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from verl_trace2skill.sglang_dense_qwen3next_compat import (
    enable_eager_patch_for_spawn,
    remap_dense_tied_weight_name,
)


class SitecustomizeGpuSafetyTests(unittest.TestCase):
    def test_tied_lm_head_updates_shared_embedding_parameter(self) -> None:
        self.assertEqual(
            remap_dense_tied_weight_name("lm_head.weight"),
            "model.embed_tokens.weight",
        )
        self.assertEqual(
            remap_dense_tied_weight_name("model.layers.0.input_layernorm.weight"),
            "model.layers.0.input_layernorm.weight",
        )

    def test_spawn_helper_enables_patch_only_for_child_interpreters(self) -> None:
        env: dict[str, str] = {}
        enable_eager_patch_for_spawn(env)
        self.assertEqual(env["TRACE2SKILL_EAGER_PATCH_DENSE_QWEN3NEXT"], "1")

    def test_dense_patch_does_not_import_cuda_stack_at_interpreter_startup(self) -> None:
        env = os.environ.copy()
        env["TRACE2SKILL_PATCH_DENSE_QWEN3NEXT"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "verl_trace2skill"), str(ROOT), str(ROOT / "verl")]
        )
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; print(int('torch' in sys.modules), int('sglang' in sys.modules))",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "0 0")


if __name__ == "__main__":
    unittest.main()
