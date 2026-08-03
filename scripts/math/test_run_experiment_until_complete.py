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
