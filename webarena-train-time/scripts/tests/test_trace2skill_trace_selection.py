from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_trace2skill_webarena_sft.py"
SPEC = importlib.util.spec_from_file_location("trace2skill_webarena_runner", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def result(
    iid: str,
    task_id: int,
    *,
    hard: int = 0,
    turns: int = 3,
    agent_ok: bool = True,
    fail_reason: str = "WebArena evaluator score was 0.",
    answer: str = "exit(message='answer')",
) -> dict:
    return {
        "id": iid,
        "task_id": task_id,
        "hard": hard,
        "n_turns": turns,
        "agent_ok": agent_ok,
        "fail_reason": fail_reason,
        "predicted_answer": answer,
        "wall_time_s": turns,
    }


class TraceSelectionTest(unittest.TestCase):
    def test_selects_representative_positive_and_negative(self) -> None:
        results = [
            result("1_s00", 1, hard=1, turns=9),
            result("1_s01", 1, hard=1, turns=4),
            result("1_s02", 1, turns=5),
            result("1_s03", 1, turns=12),
            result("2_s00", 2, turns=6),
            result("2_s01", 2, turns=10),
            result("3_s00", 3, hard=1, turns=8),
            result("3_s01", 3, hard=1, turns=3),
            result(
                "4_s00",
                4,
                agent_ok=False,
                fail_reason="TimeoutExpired after 900 seconds",
                answer="",
            ),
            result("5_s00", 5, turns=30),
            result("6_s00", 6, answer=""),
        ]

        selected, report = runner.select_representative_results(results, max_steps=30)

        self.assertEqual(
            {item["id"] for item in selected},
            {"1_s01", "1_s03", "2_s01", "3_s01"},
        )
        self.assertEqual(report["selected_positive"], 2)
        self.assertEqual(report["selected_negative"], 2)
        self.assertEqual(report["task_outcomes"]["mixed"], 1)
        self.assertEqual(report["task_outcomes"]["all_negative"], 1)
        self.assertEqual(report["task_outcomes"]["all_positive"], 1)
        self.assertEqual(report["task_outcomes"]["no_usable_trace"], 3)
        self.assertEqual(report["excluded_rollouts"]["infrastructure"], 1)
        self.assertEqual(report["excluded_rollouts"]["turn_limit"], 1)
        self.assertEqual(report["excluded_rollouts"]["empty_trace"], 1)


if __name__ == "__main__":
    unittest.main()
