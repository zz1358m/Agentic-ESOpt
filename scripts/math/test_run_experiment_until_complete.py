from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_experiment_until_complete.py")
SPEC = importlib.util.spec_from_file_location("math_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MathExperimentPipelineTests(unittest.TestCase):
    def test_runtime_python_falls_back_to_grpo_environment(self) -> None:
        selected = MODULE.select_runtime_python(
            requested=None,
            current="/opt/miniconda3/bin/python",
            conda_envs=["/opt/miniconda3", "/shared/conda/envs/grpo"],
            usable=lambda path: path == "/shared/conda/envs/grpo/bin/python",
        )

        self.assertEqual(selected, "/shared/conda/envs/grpo/bin/python")

    def test_eval_completion_requires_exact_unique_error_free_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            outputs.mkdir()
            for dataset, items in (("dapo100", 2), ("aime2026", 1)):
                rows = [
                    {
                        "key": f"{dataset}:q{item}:sample{sample:02d}",
                        "error": None,
                    }
                    for item in range(items)
                    for sample in range(2)
                ]
                (outputs / f"{dataset}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            status = MODULE.inspect_eval_completion(
                root,
                expected={"dapo100": 4, "aime2026": 2},
            )
            self.assertTrue(status["complete"])

            with (outputs / "dapo100.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": "dapo100:extra:sample00", "error": None}) + "\n")
            status = MODULE.inspect_eval_completion(
                root,
                expected={"dapo100": 4, "aime2026": 2},
            )
            self.assertFalse(status["complete"])

    def test_eval_command_is_fixed_and_resumable(self) -> None:
        command = MODULE.eval_command(
            python="python",
            model_path=Path("/model"),
            out_dir=Path("/eval"),
        )

        self.assertIn("--samples", command)
        self.assertEqual(command[command.index("--samples") + 1], "16")
        self.assertEqual(command[command.index("--seed") + 1], "20260629")
        self.assertEqual(command[command.index("--concurrency") + 1], "8")
        self.assertEqual(command[command.index("--context-length") + 1], str(256 * 1024))
        self.assertIn("--resume", command)

    def test_alignment_eval_command_uses_four_samples_and_50x4096_profile(self) -> None:
        command = MODULE.eval_command(
            python="python",
            model_path=Path("/model"),
            out_dir=Path("/eval"),
            profile="repo-react-v1-50x4096",
        )

        self.assertEqual(command[command.index("--samples") + 1], "4")
        self.assertEqual(command[command.index("--concurrency") + 1], "8")
        self.assertEqual(command[command.index("--profile") + 1], "repo-react-v1-50x4096")
        self.assertEqual(command[command.index("--context-length") + 1], str(256 * 1024))

    def test_data_validation_is_a_pipeline_gate(self) -> None:
        command = MODULE.data_validation_command(
            python="python",
            root=Path("/repo"),
            out=Path("/reports/data_manifest.json"),
        )

        self.assertEqual(command[0], "python")
        self.assertEqual(command[1], "/repo/scripts/math/validate_data.py")
        self.assertEqual(command[command.index("--out") + 1], "/reports/data_manifest.json")

    def test_smoke_environment_is_one_step_and_uses_isolated_outputs(self) -> None:
        env = MODULE.smoke_environment(Path("/repo"), "formal-run", tier=0)

        self.assertEqual(env["TOTAL_TRAINING_STEPS"], "1")
        self.assertEqual(env["TRAIN_BATCH_SIZE"], "20")
        self.assertEqual(env["ROLLOUT_N"], "8")
        self.assertEqual(env["MAX_USER_TURNS"], "100")
        self.assertEqual(env["MAX_RESPONSE_LENGTH"], "8192")
        self.assertIn("formal-run-smoke-step1", env["ROLLOUT_DATA_DIR"])
        self.assertEqual(env["SAVE_FREQ"], "1")
        self.assertEqual(env["TEST_FREQ"], "1")
        self.assertEqual(env["VAL_BEFORE_TRAIN"], "False")

    def test_smoke_acceptance_requires_160_tool_reward_records_update_and_hf_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke_tag = "formal-run-smoke-step1"
            train_dir = root / "runs/multiturn_grpo/trajectories" / smoke_tag / "train_raw"
            train_dir.mkdir(parents=True)
            rows = [
                {"trajectory_id": f"row-{index}", "tool_used": 1.0, "score": 0.0}
                for index in range(160)
            ]
            (train_dir / "1.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            checkpoint = (
                root
                / "runs/multiturn_grpo/checkpoints"
                / smoke_tag
                / "global_step_1/actor/huggingface"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            log_dir = root / "runs/multiturn_grpo/logs" / smoke_tag
            log_dir.mkdir(parents=True)
            (log_dir / "train.log").write_text(
                "step:1 - timing_s/update_actor:1.25 - critic/score/mean:0.0\n",
                encoding="utf-8",
            )

            status = MODULE.inspect_smoke_completion(root, "formal-run")

        self.assertTrue(status["complete"])
        self.assertEqual(status["records"], 160)
        self.assertEqual(status["bash_tool_records"], 160)
        self.assertEqual(status["reward_records"], 160)

    def test_trajectory_summary_requires_all_acceptance_phases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "records": 58260,
                        "by_phase": {
                            "train": 48000,
                            "validation": 6100,
                            "baseline": 2080,
                            "post": 2080,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(MODULE.trajectory_summary_complete(path))


if __name__ == "__main__":
    unittest.main()
