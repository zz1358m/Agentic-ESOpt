from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "math-train-time" / "scripts" / "run_math_es_vllm_train.py"


def load_training_script():
    spec = importlib.util.spec_from_file_location("run_math_es_vllm_train_ray_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RayInitResourcesTest(unittest.TestCase):
    def test_init_ray_limits_cpu_workers_to_engine_count(self):
        init_calls = []
        fake_ray = SimpleNamespace(
            is_initialized=lambda: False,
            init=lambda **kwargs: init_calls.append(kwargs),
        )

        with patch.dict(sys.modules, {"ray": fake_ray}):
            module = load_training_script()
            result = module.init_ray(argparse.Namespace(num_engines=4))

        self.assertIs(result, fake_ray)
        self.assertEqual(init_calls[0]["num_cpus"], 4)


if __name__ == "__main__":
    unittest.main()
