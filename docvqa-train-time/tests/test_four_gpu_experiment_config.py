from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "docvqa" / "experiment_config.py"
LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "docvqa" / "run_grpo.sh"
SPEC = importlib.util.spec_from_file_location("experiment_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FourGpuExperimentConfigTests(unittest.TestCase):
    def test_launcher_reserves_cpu_slots_for_four_workers(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"', launcher)
        self.assertIn('RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"', launcher)
        self.assertIn('AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-4}"', launcher)
        self.assertIn('MAX_TURN_RESPONSE_LENGTH="${MAX_TURN_RESPONSE_LENGTH:-512}"', launcher)
        self.assertIn('GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"', launcher)
        self.assertIn('SAVE_FREQ="${SAVE_FREQ:-1}"', launcher)
        self.assertIn('DATA_SEED="${DATA_SEED:-42}"', launcher)
        self.assertIn('VERL_PROTECTED_CHECKPOINT_STEPS="${VERL_PROTECTED_CHECKPOINT_STEPS:-60,120,180}"', launcher)

    def test_training_disables_qwen_thinking_in_chat_template(self) -> None:
        launcher = (LAUNCHER.parents[1] / "trace2skill" / "run_verl_agentic_rl.sh").read_text(encoding="utf-8")
        self.assertIn("+data.apply_chat_template_kwargs.enable_thinking=False", launcher)

    def test_training_can_keep_reference_model_separate_from_actor_weights(self) -> None:
        launcher = (LAUNCHER.parents[1] / "trace2skill" / "run_verl_agentic_rl.sh").read_text(encoding="utf-8")
        self.assertIn(
            '+actor_rollout_ref.ref.model.path="${REF_MODEL_PATH}"',
            launcher,
        )

    def test_training_log_is_appended_when_a_checkpoint_run_resumes(self) -> None:
        launcher = (LAUNCHER.parents[1] / "trace2skill" / "run_verl_agentic_rl.sh").read_text(encoding="utf-8")
        self.assertIn('| tee -a "${LOG_DIR}/train.log"', launcher)

    def test_expected_four_gpu_plan_has_180_steps(self) -> None:
        report = MODULE.validate_experiment_config(
            visible_devices="4,5,6,7",
            effective_visible_devices="GPU-four,GPU-five,GPU-six,GPU-seven",
            train_records=50,
            train_batch_size=4,
            ppo_mini_batch_size=4,
            rollout_n=8,
            epochs=15,
            world_size=4,
        )

        self.assertEqual(report["physical_gpu_ids"], [4, 5, 6, 7])
        self.assertEqual(report["steps_per_epoch"], 12)
        self.assertEqual(report["total_steps"], 180)
        self.assertEqual(report["training_trajectories"], 5760)
        self.assertEqual(report["dropped_records_per_epoch"], 2)

    def test_rejects_a_non_four_gpu_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four unique"):
            MODULE.validate_experiment_config(
                visible_devices="0,1,2",
                train_records=50,
                train_batch_size=4,
                rollout_n=8,
                epochs=15,
                world_size=4,
            )

    def test_rejects_batch_rollouts_not_divisible_by_world_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            MODULE.validate_experiment_config(
                visible_devices="0,1,2,3",
                train_records=50,
                train_batch_size=3,
                rollout_n=5,
                epochs=15,
                world_size=4,
            )


if __name__ == "__main__":
    unittest.main()
