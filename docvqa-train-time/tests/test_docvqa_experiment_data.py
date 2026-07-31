from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "docvqa" / "validate_experiment_data.py"
SPEC = importlib.util.spec_from_file_location("validate_experiment_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DocVQAExperimentDataTests(unittest.TestCase):
    def test_alignment_matches_ordered_identity_and_detects_no_train_overlap(self) -> None:
        train = [{"id": "docvqa_train", "answers": ["x"], "image": "images/train.png"}]
        test = [
            {"id": "docvqa_a", "answers": ["one"], "image": "images/a.png"},
            {"id": "docvqa_b", "answers": ["two"], "image": "images/b.png"},
        ]
        historical = [
            {"task_id": "docvqa_a", "row_index": 0, "sample_index": sample, "answers": ["one"], "image": "old/a.png"}
            for sample in range(4)
        ] + [
            {"task_id": "docvqa_b", "row_index": 1, "sample_index": sample, "answers": ["two"], "image": "old/b.png"}
            for sample in range(4)
        ]

        report = MODULE.validate_alignment(train, test, historical, limit=2, samples=4)

        self.assertEqual(report["train_records"], 1)
        self.assertEqual(report["test_records_checked"], 2)
        self.assertEqual(report["historical_records_checked"], 8)
        self.assertEqual(report["task_ids"], ["docvqa_a", "docvqa_b"])

    def test_alignment_rejects_answer_mismatch(self) -> None:
        test = [{"id": "docvqa_a", "answers": ["wrong"], "image": "images/a.png"}]
        historical = [
            {"task_id": "docvqa_a", "row_index": 0, "sample_index": sample, "answers": ["one"], "image": "old/a.png"}
            for sample in range(4)
        ]

        with self.assertRaisesRegex(ValueError, "answers mismatch"):
            MODULE.validate_alignment([], test, historical, limit=1, samples=4)

    def test_alignment_rejects_train_test_overlap(self) -> None:
        record = {"id": "docvqa_a", "answers": ["one"], "image": "images/a.png"}
        historical = [
            {"task_id": "docvqa_a", "row_index": 0, "sample_index": sample, "answers": ["one"], "image": "old/a.png"}
            for sample in range(4)
        ]

        with self.assertRaisesRegex(ValueError, "train/test overlap"):
            MODULE.validate_alignment([record], [record], historical, limit=1, samples=4)

    def test_alignment_skips_historical_tasks_selected_for_training(self) -> None:
        train = [{"id": "docvqa_a", "answers": ["one"], "image": "images/a.png"}]
        test = [{"id": "docvqa_b", "answers": ["two"], "image": "images/b.png"}]
        historical = [
            {"task_id": task_id, "row_index": row_index, "sample_index": sample, "answers": [answer], "image": image}
            for row_index, (task_id, answer, image) in enumerate(
                (("docvqa_a", "one", "old/a.png"), ("docvqa_b", "two", "old/b.png"))
            )
            for sample in range(4)
        ]

        report = MODULE.validate_alignment(train, test, historical, limit=1, samples=4)

        self.assertEqual(report["task_ids"], ["docvqa_b"])
        self.assertEqual(report["historical_source_row_indexes"], [1])


if __name__ == "__main__":
    unittest.main()
