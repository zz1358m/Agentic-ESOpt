from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/math/experiment_config.py"
SPEC = importlib.util.spec_from_file_location("math_experiment_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GrpoExperimentConfigTests(unittest.TestCase):
    def test_approved_dapo400_config_has_exact_counts(self) -> None:
        report = MODULE.validate_math_experiment_config(
            physical_gpu_ids="3,4,5,6",
            train_records=400,
            val_records=100,
            aime_records=30,
            train_batch_size=20,
            ppo_mini_batch_size=20,
            rollout_n=8,
            epochs=15,
            world_size=4,
            test_freq=5,
            eval_samples=16,
            ray_num_cpus=32,
            max_user_turns=100,
            max_assistant_turns=100,
            max_response_length=8192,
            max_turn_response_length=512,
            save_freq=20,
            rollout_data_dir="/tmp/train_raw",
            validation_data_dir="/tmp/validation_raw",
            tool_config_path="/tmp/math_bash.yaml",
            parser_enabled=True,
            dense_qwen3next_patch_enabled=True,
        )

        self.assertEqual(report["steps_per_epoch"], 20)
        self.assertEqual(report["total_steps"], 300)
        self.assertEqual(report["training_trajectories"], 48000)
        self.assertEqual(report["validation_rounds"], 61)
        self.assertEqual(report["validation_trajectories"], 6100)
        self.assertEqual(report["standalone_evaluation_trajectories"], 4160)
        self.assertEqual(report["dropped_records_per_epoch"], 0)
        self.assertEqual(report["limits"]["max_response_tokens"], 8192)
        self.assertEqual(report["checkpoint_steps"], list(range(20, 301, 20)))
        self.assertEqual(report["runtime"]["ray_num_cpus"], 32)
        self.assertEqual(report["runtime"]["generate_timeout_seconds"], 600.0)
        self.assertEqual(report["runtime"]["generate_max_attempts"], 3)
        self.assertEqual(report["runtime"]["reward_timeout_seconds"], 120.0)
        self.assertEqual(report["runtime"]["reward_max_attempts"], 3)
        self.assertTrue(report["runtime"]["parser_enabled"])
        self.assertEqual(report["optimizer"], {"learning_rate": 1e-6, "use_kl_loss": True, "kl_loss_coef": 0.001})
        self.assertEqual(
            report["sampling"],
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 40,
                "presence_penalty": 2.0,
                "repetition_penalty": 1.0,
            },
        )
        self.assertEqual(report["data_order"], {"shuffle": True, "seed": 1})
        self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
