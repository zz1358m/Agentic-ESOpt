from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_trace2skill_webarena_sft.py"
SPEC = importlib.util.spec_from_file_location("trace2skill_webarena_runner", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

ES_DISTILL_RUNNER = Path(__file__).resolve().parents[1] / "run_trace2skill_from_es_traces.py"
ES_DISTILL_SPEC = importlib.util.spec_from_file_location(
    "trace2skill_webarena_es_distill", ES_DISTILL_RUNNER
)
assert ES_DISTILL_SPEC and ES_DISTILL_SPEC.loader
es_distill = importlib.util.module_from_spec(ES_DISTILL_SPEC)
ES_DISTILL_SPEC.loader.exec_module(es_distill)


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

    def test_es_distillation_defaults_to_every_completed_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "history.json").write_text(
                json.dumps([{"generation": 0}, {"generation": 1}]),
                encoding="utf-8",
            )
            for generation, sample, tasks in (
                (0, 0, (1, 2)),
                (0, 1, (3,)),
                (1, 0, (4, 5)),
                (2, 0, (6,)),
            ):
                sample_dir = run_dir / f"gen_{generation:03d}_sample_{sample:03d}_positive"
                for task in tasks:
                    (sample_dir / f"task_{task}").mkdir(parents=True)

            selected = es_distill.collect_task_dirs(
                run_dir,
                generations=0,
                max_traces=0,
            )

        self.assertEqual(
            {path.name for path in selected},
            {"task_1", "task_2", "task_3", "task_4", "task_5"},
        )
        self.assertNotIn("task_6", {path.name for path in selected})


if __name__ == "__main__":
    unittest.main()
