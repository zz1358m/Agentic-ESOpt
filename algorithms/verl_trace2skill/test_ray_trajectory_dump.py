from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class RayTrajectoryDumpTests(unittest.TestCase):
    def test_dump_writes_replayable_training_metadata(self) -> None:
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.global_steps = 21
        trainer.train_dataloader = [None] * 20

        with tempfile.TemporaryDirectory() as tmp:
            trainer._dump_generations(
                inputs=["problem", "problem"],
                outputs=[
                    'Action: {"name":"bash","arguments":{"command":"false"}}\nObservation from bash:\nreturncode=1',
                    'Action: {"name":"bash","arguments":{"command":"python -c \'print(42)\'"}}\n'
                    'Observation from bash:\n42\nreturncode=0\nFinal answer: \\boxed{42}',
                ],
                gts=["42", "42"],
                scores=[0.0, 1.0],
                reward_extra_infos_dict={"acc": [0.0, 1.0], "tool_used": [1.0, 1.0]},
                dump_path=tmp,
                phase="train",
                metadata_fields={
                    "extra_info": [
                        {"id": "dapo-42", "index": 7, "split": "train"},
                        {"id": "dapo-42", "index": 7, "split": "train"},
                    ],
                    "uid": ["prompt-group", "prompt-group"],
                    "num_turns": [3, 5],
                    "prompt_tokens": [100, 100],
                    "response_tokens": [20, 40],
                },
            )

            records = [
                json.loads(line)
                for line in (Path(tmp) / "21.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["trajectory_id"], "train-step000021-dapo-42-rollout00")
        self.assertEqual(records[1]["trajectory_id"], "train-step000021-dapo-42-rollout01")
        self.assertEqual(records[1]["epoch"], 2)
        self.assertEqual(records[1]["acc"], 1.0)
        self.assertEqual(records[1]["tool_used"], 1.0)
        self.assertEqual(records[1]["num_turns"], 5)
        self.assertEqual(records[1]["response_tokens"], 40)
        self.assertEqual(records[1]["steps"][0]["action"]["name"], "bash")
        self.assertIn("42", records[1]["steps"][0]["observation"])


if __name__ == "__main__":
    unittest.main()
