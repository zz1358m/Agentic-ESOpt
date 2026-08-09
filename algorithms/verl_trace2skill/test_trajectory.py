from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from algorithms.verl_trace2skill.trajectory import (
    build_trajectory_records,
    export_trajectory_records,
    load_raw_trajectory_records,
    normalize_evaluation_record,
    parse_react_steps,
    write_trajectory_records,
)


class TrajectoryTests(unittest.TestCase):
    def test_raw_loader_preserves_retry_attempts_with_unique_identity(self) -> None:
        record = {
            "trajectory_id": "train-step000001-dapo-a-rollout00",
            "phase": "train",
            "score": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            dump_dir = Path(tmp)
            (dump_dir / "1.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            (dump_dir / "1.attempt02.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            records = load_raw_trajectory_records(dump_dir)

        self.assertEqual(
            [(item["trajectory_id"], item["raw_attempt"]) for item in records],
            [
                ("train-step000001-dapo-a-rollout00", 1),
                ("train-step000001-dapo-a-rollout00-attempt02", 2),
            ],
        )

    def test_raw_loader_latest_only_uses_one_complete_attempt_per_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump_dir = Path(tmp)
            for step, attempt, trajectory_id in (
                (1, 1, "old-step-1"),
                (1, 2, "new-step-1"),
                (2, 1, "only-step-2"),
            ):
                suffix = "" if attempt == 1 else f".attempt{attempt:02d}"
                (dump_dir / f"{step}{suffix}.jsonl").write_text(
                    json.dumps({"trajectory_id": trajectory_id}) + "\n",
                    encoding="utf-8",
                )

            records = load_raw_trajectory_records(dump_dir, latest_only=True)

        self.assertEqual(
            [(item["trajectory_id"], item["raw_attempt"]) for item in records],
            [("new-step-1", 2), ("only-step-2", 1)],
        )

    def test_evaluation_record_normalizes_full_react_conversation(self) -> None:
        record = normalize_evaluation_record(
            {
                "dataset": "dapo100",
                "task_id": "dapo-a",
                "row_index": 2,
                "sample_index": 3,
                "question": "What is 6 * 7?",
                "target": "42",
                "completion": "Final answer: \\boxed{42}",
                "score": 1.0,
                "react_steps": [
                    {
                        "turn": 1,
                        "assistant": 'Action: {"name": "bash", "arguments": {"command": "python -c \'print(6*7)\'"}}',
                        "observation": "42\n[exit_code] 0",
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
                "error": None,
            },
            phase="baseline",
            epoch=0,
            step=0,
        )

        self.assertEqual(record["trajectory_id"], "baseline-dapo100-dapo-a-sample03")
        self.assertEqual(record["source_id"], "dapo-a")
        self.assertEqual(record["input"], "What is 6 * 7?")
        self.assertEqual(record["gts"], "42")
        self.assertEqual(record["rollout_index"], 3)
        self.assertEqual(record["tool_used"], 1.0)
        self.assertEqual(record["prompt_tokens"], 120)
        self.assertEqual(record["response_tokens"], 30)
        self.assertIn("Observation from bash:\n42", record["output"])
        self.assertTrue(record["output"].endswith("Final answer: \\boxed{42}"))

    def test_parse_react_steps_keeps_actions_and_observations(self) -> None:
        output = (
            'Action:\n{"name": "bash", "arguments": {"command": "python -c \'print(6 * 7)\'"}}\n'
            "Observation from bash:\n42\n[exit_code] 0\n"
            'Action: {"name": "bash", "arguments": {"command": "python -c \'print(42 == 42)\'"}}\n'
            "Observation from bash:\nTrue\n[exit_code] 0\n"
            "Final answer: \\boxed{42}"
        )

        self.assertEqual(
            parse_react_steps(output),
            [
                {
                    "action": {
                        "name": "bash",
                        "arguments": {"command": "python -c 'print(6 * 7)'"},
                    },
                    "observation": "42\n[exit_code] 0",
                },
                {
                    "action": {
                        "name": "bash",
                        "arguments": {"command": "python -c 'print(42 == 42)'"},
                    },
                    "observation": "True\n[exit_code] 0",
                },
            ],
        )

    def test_parse_react_steps_ignores_format_feedback_and_non_bash_actions(self) -> None:
        output = (
            'assistant\nAction: {"name":"bash","arguments":{"command":"python -c \'print(42)\'"}}\n'
            "user\nObservation from bash:\n42\n[exit_code] 0\n"
            "assistant\nmalformed call\nuser\nObservation from format_check:\n"
            "No valid action was parsed. Use exactly:\nAction:\n"
            '{"name":"bash","arguments":{"command":"<shell command>"}}\n'
            "assistant\nAction: "
            '{"name":"python","arguments":{"command":"print(1)"}}\n'
        )

        self.assertEqual(
            parse_react_steps(output),
            [
                {
                    "action": {
                        "name": "bash",
                        "arguments": {"command": "python -c 'print(42)'"},
                    },
                    "observation": "42\n[exit_code] 0",
                }
            ],
        )

    def test_parse_react_steps_drops_actions_that_were_not_executed(self) -> None:
        output = (
            'assistant\nreasoning before Action: {"name":"bash","arguments":{"command":"echo rejected"}}\n'
            "user\nObservation from format_check:\nThe first assistant turn must contain only one bash Action.\n"
            'assistant\nAction: {"name":"bash","arguments":{"command":"echo accepted"}}\n'
            "user\nObservation from bash:\naccepted\n[exit_code] 0\n"
            "assistant\nFinal answer: \\boxed{1}"
        )

        self.assertEqual(
            parse_react_steps(output),
            [
                {
                    "action": {
                        "name": "bash",
                        "arguments": {"command": "echo accepted"},
                    },
                    "observation": "accepted\n[exit_code] 0",
                }
            ],
        )

    def test_records_have_stable_source_epoch_and_rollout_identity(self) -> None:
        records = build_trajectory_records(
            [
                {
                    "input": "problem",
                    "output": "first",
                    "gts": "42",
                    "score": 0.0,
                    "extra_info": {"id": "dapo-42", "index": 7, "split": "train"},
                    "uid": "prompt-group",
                    "num_turns": 3,
                    "prompt_tokens": 100,
                    "response_tokens": 20,
                },
                {
                    "input": "problem",
                    "output": "second",
                    "gts": "42",
                    "score": 1.0,
                    "extra_info": {"id": "dapo-42", "index": 7, "split": "train"},
                    "uid": "prompt-group",
                    "num_turns": 5,
                    "prompt_tokens": 100,
                    "response_tokens": 40,
                },
            ],
            phase="train",
            step=21,
            steps_per_epoch=20,
        )

        self.assertEqual(
            [
                {
                    key: record[key]
                    for key in (
                        "trajectory_id",
                        "phase",
                        "epoch",
                        "global_step",
                        "source_id",
                        "row_index",
                        "split",
                        "rollout_index",
                        "num_turns",
                        "prompt_tokens",
                        "response_tokens",
                    )
                }
                for record in records
            ],
            [
                {
                    "trajectory_id": "train-step000021-dapo-42-rollout00",
                    "phase": "train",
                    "epoch": 2,
                    "global_step": 21,
                    "source_id": "dapo-42",
                    "row_index": 7,
                    "split": "train",
                    "rollout_index": 0,
                    "num_turns": 3,
                    "prompt_tokens": 100,
                    "response_tokens": 20,
                },
                {
                    "trajectory_id": "train-step000021-dapo-42-rollout01",
                    "phase": "train",
                    "epoch": 2,
                    "global_step": 21,
                    "source_id": "dapo-42",
                    "row_index": 7,
                    "split": "train",
                    "rollout_index": 1,
                    "num_turns": 5,
                    "prompt_tokens": 100,
                    "response_tokens": 40,
                },
            ],
        )

    def test_write_preserves_conflicting_retry_as_another_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump_dir = Path(tmp)
            first = [{"trajectory_id": "one", "score": np.float32(0.0)}]
            retry = [{"trajectory_id": "two", "score": np.float32(1.0)}]

            first_path = write_trajectory_records(first, dump_dir=dump_dir, step=21)
            same_path = write_trajectory_records(first, dump_dir=dump_dir, step=21)
            retry_path = write_trajectory_records(retry, dump_dir=dump_dir, step=21)

            self.assertEqual(first_path.name, "21.jsonl")
            self.assertEqual(same_path, first_path)
            self.assertEqual(retry_path.name, "21.attempt02.jsonl")
            self.assertEqual(json.loads(first_path.read_text())["score"], 0.0)
            self.assertEqual(json.loads(retry_path.read_text())["trajectory_id"], "two")

    def test_export_is_idempotent_and_writes_success_failure_markdown(self) -> None:
        records = [
            {
                "trajectory_id": "train-step000001-dapo-a-rollout00",
                "phase": "train",
                "epoch": 1,
                "global_step": 1,
                "source_id": "dapo-a",
                "input": "first problem",
                "output": "Observation from bash:\n0\nFinal answer: \\boxed{0}",
                "gts": "1",
                "score": 0.0,
            },
            {
                "trajectory_id": "train-step000001-dapo-b-rollout00",
                "phase": "train",
                "epoch": 1,
                "global_step": 1,
                "source_id": "dapo-b",
                "input": "second problem",
                "output": "Observation from bash:\n2\nFinal answer: \\boxed{2}",
                "gts": "2",
                "score": 1.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            first = export_trajectory_records(records, out_dir=out_dir)
            second = export_trajectory_records(records, out_dir=out_dir)
            consolidated = (out_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
            markdown = sorted(path.name for path in (out_dir / "markdown/train/epoch_01/step_000001").iterdir())

        self.assertEqual(first, {"records": 2, "succeed": 1, "failed": 1})
        self.assertEqual(second, first)
        self.assertEqual(len(consolidated), 2)
        self.assertEqual(
            markdown,
            [
                "train-step000001-dapo-a-rollout00_FAILED.md",
                "train-step000001-dapo-b-rollout00_SUCCEED.md",
            ],
        )


if __name__ == "__main__":
    unittest.main()
