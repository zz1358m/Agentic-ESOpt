from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MATH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = MATH_ROOT / "scripts"
sys.path.insert(0, str(MATH_ROOT))
sys.path.insert(0, str(SCRIPTS))

from envs.math_reasoning import trim_oldest_react_exchange  # noqa: E402
from run_math_es_train import mean_valid  # noqa: E402


def load_vllm_train_module():
    path = SCRIPTS / "run_math_es_vllm_train.py"
    spec = importlib.util.spec_from_file_location("run_math_es_vllm_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EsRolloutAccountingTest(unittest.TestCase):
    def test_failures_are_zero_in_mean(self) -> None:
        self.assertEqual(mean_valid([1.0, 0.0, -1.0]), 1.0 / 3.0)

    def test_summary_keeps_failed_rollouts_and_tasks(self) -> None:
        summarize_rows = load_vllm_train_module().summarize_rows
        rows = [
            {"task_id": "a", "row_index": 0, "sample_index": 0, "score": 1.0, "termination_reason": "final_answer", "answer_status": "answered"},
            {"task_id": "a", "row_index": 0, "sample_index": 1, "score": -1.0, "termination_reason": "request_error", "answer_status": "missing_final_answer"},
            {"task_id": "b", "row_index": 1, "sample_index": 0, "score": 0.0, "termination_reason": "max_turns", "answer_status": "missing_final_answer"},
            {"task_id": "b", "row_index": 1, "sample_index": 1, "score": 0.0, "termination_reason": "final_answer", "answer_status": "answered"},
        ]
        result = summarize_rows(rows, items=2, samples=2)
        self.assertEqual(result["valid_count"], 4)
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["mean_score"], 0.25)
        self.assertEqual(result["max@2"], 0.5)

    def test_context_trim_preserves_system_and_task(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "old action"},
            {"role": "user", "content": "old observation"},
            {"role": "assistant", "content": "new action"},
            {"role": "user", "content": "new observation"},
        ]
        self.assertTrue(trim_oldest_react_exchange(messages))
        self.assertEqual([message["content"] for message in messages], ["system", "task", "new action", "new observation"])


if __name__ == "__main__":
    unittest.main()
