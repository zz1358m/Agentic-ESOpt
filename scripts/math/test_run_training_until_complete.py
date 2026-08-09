from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_training_until_complete.py")
SPEC = importlib.util.spec_from_file_location("math_training_watchdog", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MathTrainingWatchdogTests(unittest.TestCase):
    def test_only_explicit_capacity_errors_trigger_fallback(self) -> None:
        self.assertEqual(MODULE.classify_failure("CUDA out of memory"), "capacity")
        self.assertEqual(
            MODULE.classify_failure("maximum sequence length exceeds model capacity"),
            "capacity",
        )
        self.assertEqual(MODULE.classify_failure("worker unexpectedly exited"), "transient")
        self.assertEqual(MODULE.classify_failure("generation is slow"), "transient")

    def test_lowest_capacity_tier_preserves_training_scope_and_trajectory_dump(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = MODULE.fixed_environment(root, "run", len(MODULE.LIMIT_TIERS) - 1)

        self.assertEqual(MODULE.LIMIT_TIERS[-1], (2, 512))
        self.assertEqual(env["MAX_USER_TURNS"], "2")
        self.assertEqual(env["MAX_ASSISTANT_TURNS"], "2")
        self.assertEqual(env["MAX_RESPONSE_LENGTH"], "512")
        self.assertEqual(env["TRAIN_BATCH_SIZE"], "20")
        self.assertEqual(env["ROLLOUT_N"], "8")
        self.assertEqual(env["TOTAL_EPOCHS"], "15")
        self.assertTrue(env["ROLLOUT_DATA_DIR"].endswith("/train_raw"))
        self.assertTrue(env["VALIDATION_DATA_DIR"].endswith("/validation_raw"))

    def test_completion_requires_exact_latest_attempt_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train_raw"
            validation = root / "validation_raw"
            checkpoints = root / "checkpoints"
            train.mkdir()
            validation.mkdir()
            for step in range(1, 301):
                (train / f"{step}.jsonl").write_text("{}\n" * 160, encoding="utf-8")
            # A retry is immutable but replaces the first attempt in the logical count.
            (train / "7.attempt02.jsonl").write_text("{}\n" * 160, encoding="utf-8")
            for step in [0, *range(5, 301, 5)]:
                (validation / f"{step}.jsonl").write_text("{}\n" * 100, encoding="utf-8")
            for step in range(20, 301, 20):
                hf = checkpoints / f"global_step_{step}" / "actor" / "huggingface"
                hf.mkdir(parents=True)
                (hf / "config.json").write_text(json.dumps({}), encoding="utf-8")

            result = MODULE.inspect_completion(train, validation, checkpoints)

        self.assertTrue(result["complete"])
        self.assertEqual(result["train_records"], 48000)
        self.assertEqual(result["validation_records"], 6100)


if __name__ == "__main__":
    unittest.main()
